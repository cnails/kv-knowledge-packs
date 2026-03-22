"""
Exp 108: Dual-Channel — Knowledge KV + Value Steering simultaneously.

Reproduces Table 7-8 from paper with N=200 (up from N=50).

6 conditions:
  A. No knowledge, no steering (baseline)
  B. Knowledge only (KV cache with chat template)
  C. Steering only (V-delta, mid-layers 33-66%)
  D. Knowledge + Steering (dual-channel, alpha sweep)
  E. Text knowledge (RAG — facts in prompt)
  F. Text knowledge + text steering (facts + "be formal" in prompt)

Model: Qwen3-8B (fp16) on single GPU.
Dataset: HotpotQA bridge questions, N=200.

Usage:
    python benchmarks/exp108_dual_channel.py [--n 200] [--device cuda]

Estimated runtime: ~25-35 min on 4090.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from sentence_transformers import SentenceTransformer

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"


# ============================================================================
# Contrastive pairs for formality steering (6 pairs = 12 examples)
# ============================================================================

FORMAL_PAIRS = [
    {
        "prompt": "How do computers work?",
        "positive": "Computers are sophisticated electronic devices that process information through a series of logical operations executed by integrated circuits. Each instruction is decoded and dispatched through a pipeline of arithmetic and control units.",
        "negative": "Well basically they just do stuff with electricity and little switches, pretty simple really. You type something and it figures it out.",
    },
    {
        "prompt": "What is gravity?",
        "positive": "Gravity is a fundamental force of nature, described by Einstein's general theory of relativity as the curvature of spacetime caused by mass and energy. Objects follow geodesics in this curved geometry.",
        "negative": "It's the thing that makes stuff fall down. Like when you drop your phone, that's gravity doing its thing. Pretty basic.",
    },
    {
        "prompt": "Why do we sleep?",
        "positive": "Sleep is a complex neurobiological process essential for cognitive consolidation, metabolic regulation, and immune system maintenance. During NREM sleep, synaptic homeostasis is restored through coordinated slow oscillations.",
        "negative": "Because we get tired lol. Your body just needs to recharge like a phone battery. Close your eyes and boom, you wake up later.",
    },
    {
        "prompt": "How does the internet work?",
        "positive": "The internet operates through a distributed network of interconnected systems utilizing standardized protocols such as TCP/IP for reliable data transmission across heterogeneous physical media.",
        "negative": "It's like magic tubes that carry cat videos to your screen. Wifi goes brrr and stuff shows up. Don't overthink it.",
    },
    {
        "prompt": "What causes rain?",
        "positive": "Precipitation occurs when water vapor in the atmosphere condenses around particulate matter, forming droplets that aggregate until gravitational force exceeds atmospheric buoyancy, initiating downward transport.",
        "negative": "Clouds get too heavy with water and it just falls out. Simple as that. Sometimes a lot, sometimes a little.",
    },
    {
        "prompt": "How do vaccines work?",
        "positive": "Vaccines function by introducing attenuated or inactivated pathogenic material to stimulate an adaptive immune response, thereby establishing immunological memory through B-cell and T-cell activation cascades.",
        "negative": "They basically show your body a weak version of the bad thing so your body knows how to fight it later. Easy peasy.",
    },
]


# ============================================================================
# Helpers
# ============================================================================

def load_model(model_name, device):
    print(f"Loading {model_name}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device
    )
    model.eval()
    n_layers = len(model.model.layers)
    print(f"Loaded in {time.time()-t0:.0f}s — {n_layers} layers", flush=True)
    return model, tok, n_layers


def load_hotpotqa(n_bridge=200):
    """Load HotpotQA bridge questions from HuggingFace."""
    print("Loading HotpotQA...", flush=True)
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    bridge = [ex for ex in ds if ex["type"] == "bridge"]
    random.seed(42)
    selected = random.sample(bridge, min(n_bridge, len(bridge)))
    print(f"Selected {len(selected)} bridge questions", flush=True)
    return selected


def get_gold_paragraphs(item):
    """Extract gold supporting paragraphs."""
    supporting_titles = set(sf for sf in item["supporting_facts"]["title"])
    gold = []
    for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
        if title in supporting_titles:
            gold.append(f"{title}: {' '.join(sentences)}")
    return gold


def build_kv_chat(model, tok, facts_text, device):
    """Build KV cache with proper chat template (critical for accuracy)."""
    messages = [{"role": "system", "content": facts_text}]
    # Generate full template, then split — avoids duplicate BOS
    full_messages = messages + [{"role": "user", "content": "placeholder"}]
    full_text = tok.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    # Find where user part starts and take only system part
    # For Qwen: <|im_start|>system\n...<|im_end|>\n
    user_marker = "<|im_start|>user"
    idx = full_text.find(user_marker)
    if idx == -1:
        # Fallback: just encode system message alone
        system_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    else:
        system_text = full_text[:idx]

    ids = tok.encode(system_text, add_special_tokens=False)
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(t, use_cache=True)
    return out.past_key_values, len(ids)


def get_kv_layer(kv, li):
    """Get (key, value) tensors for layer li, compatible with both transformers APIs."""
    if hasattr(kv, 'key_cache'):
        return kv.key_cache[li], kv.value_cache[li]
    else:
        return kv.layers[li].keys, kv.layers[li].values


def n_kv_layers(kv):
    if hasattr(kv, 'key_cache'):
        return len(kv.key_cache)
    else:
        return len(kv.layers)


def clone_kv(kv):
    c = DynamicCache()
    for li in range(n_kv_layers(kv)):
        k, v = get_kv_layer(kv, li)
        c.update(k.clone(), v.clone(), li)
    return c


def apply_v_delta(kv, v_delta, alpha, mid_start, mid_end):
    """Apply value-space steering to mid-layers of a KV cache. Returns new cache."""
    c = DynamicCache()
    for li in range(n_kv_layers(kv)):
        k, v = get_kv_layer(kv, li)
        k = k.clone()
        v = v.clone()
        if mid_start <= li < mid_end and li < len(v_delta) and v_delta[li] is not None:
            delta = v_delta[li].mean(dim=2, keepdim=True)
            v = v + alpha * delta
        c.update(k, v, li)
    return c


def generate_with_kv(model, tok, query, kv, kv_len, device, max_tokens=80):
    """Generate with KV cache prefix. Query formatted with chat template."""
    messages = [{"role": "user", "content": query}]
    q_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    q_ids = tok.encode(q_text, add_special_tokens=False)
    qt = torch.tensor([q_ids], device=device)
    am = torch.ones(1, kv_len + len(q_ids), device=device, dtype=torch.long)
    kv_c = clone_kv(kv)
    with torch.no_grad():
        out = model.generate(
            qt, past_key_values=kv_c, attention_mask=am,
            max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][len(q_ids):], skip_special_tokens=True).strip()


def generate_rag(model, tok, query, facts_text, system_suffix="", device="cuda", max_tokens=80):
    """Generate with facts in prompt (RAG style)."""
    system = f"Answer the question based on the context. Be concise.\n\nContext:\n{facts_text}"
    if system_suffix:
        system += f"\n\n{system_suffix}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(prompt, add_special_tokens=False)
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(
            t, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    n_prompt = len(ids)
    return tok.decode(out[0][n_prompt:], skip_special_tokens=True).strip(), n_prompt


def generate_baseline(model, tok, query, system_suffix="", device="cuda", max_tokens=80):
    """Generate without any external knowledge."""
    system = "Answer the question concisely."
    if system_suffix:
        system += f" {system_suffix}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(prompt, add_special_tokens=False)
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(
            t, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][len(ids):], skip_special_tokens=True).strip()


def check_answer(predicted, gold):
    """Exact match: gold answer substring in prediction."""
    return gold.lower().strip() in predicted.lower().strip()


def score_formality(text):
    """Heuristic formality score (0-1). Counts formal markers."""
    markers = 0
    total = 8
    t = text.lower()
    if len(text.split()) > 50:
        markers += 1  # longer = more formal
    if any(w in t for w in ["therefore", "consequently", "furthermore", "moreover", "additionally"]):
        markers += 1
    if any(w in t for w in ["it is", "this is", "these are", "there are"]):
        markers += 1
    if not any(w in t for w in ["lol", "basically", "stuff", "pretty much", "gonna", "wanna"]):
        markers += 1
    if any(w in t for w in ["characterized", "demonstrated", "established", "significant"]):
        markers += 1
    if "." in text and text.count(".") >= 3:
        markers += 1  # multiple sentences
    if any(w in t for w in ["however", "nevertheless", "accordingly", "specifically"]):
        markers += 1
    if len(text.split()) > 100:
        markers += 1  # much longer
    return markers / total


# ============================================================================
# Build V-delta from contrastive pairs
# ============================================================================

def build_v_delta(model, tok, pairs, device):
    """Build contrastive V-delta from formal/casual pairs.

    For each pair: compute KV(positive) and KV(negative), take V difference.
    Average across all pairs.
    """
    print(f"Building V-delta from {len(pairs)} contrastive pairs...")
    n_layers = len(model.model.layers)
    v_deltas = [[] for _ in range(n_layers)]

    for i, pair in enumerate(pairs):
        # Positive (formal)
        pos_text = f"{pair['prompt']} {pair['positive']}"
        pos_messages = [{"role": "assistant", "content": pos_text}]
        pos_prompt = tok.apply_chat_template(pos_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        pos_ids = tok.encode(pos_prompt, add_special_tokens=False)
        pos_t = torch.tensor([pos_ids], device=device)
        with torch.no_grad():
            pos_out = model(pos_t, use_cache=True)
        pos_kv = pos_out.past_key_values

        # Negative (casual)
        neg_text = f"{pair['prompt']} {pair['negative']}"
        neg_messages = [{"role": "assistant", "content": neg_text}]
        neg_prompt = tok.apply_chat_template(neg_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        neg_ids = tok.encode(neg_prompt, add_special_tokens=False)
        neg_t = torch.tensor([neg_ids], device=device)
        with torch.no_grad():
            neg_out = model(neg_t, use_cache=True)
        neg_kv = neg_out.past_key_values

        # V difference per layer (mean across sequence positions)
        for li in range(n_layers):
            _, v_pos_raw = get_kv_layer(pos_kv, li)
            _, v_neg_raw = get_kv_layer(neg_kv, li)
            v_pos = v_pos_raw.float()  # [1, heads, seq, dim]
            v_neg = v_neg_raw.float()
            # Mean pool across sequence dimension
            delta = v_pos.mean(dim=2, keepdim=True) - v_neg.mean(dim=2, keepdim=True)
            v_deltas[li].append(delta)

        del pos_kv, neg_kv, pos_out, neg_out
        torch.cuda.empty_cache()

    # Average across pairs
    avg_deltas = []
    for li in range(n_layers):
        if v_deltas[li]:
            avg = torch.stack([d.squeeze(0) for d in v_deltas[li]]).mean(dim=0, keepdim=True)
            avg_deltas.append(avg.half())
        else:
            avg_deltas.append(None)

    print(f"V-delta built. Non-zero layers: {sum(1 for d in avg_deltas if d is not None)}", flush=True)
    return avg_deltas


# ============================================================================
# Main experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/exp108_dual_channel.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tok, n_layers = load_model(args.model, device)

    # Mid-layer range (33-66%)
    mid_start = n_layers // 3          # layer 12 for 36-layer model
    mid_end = 2 * n_layers // 3        # layer 24
    print(f"Mid-layer range: [{mid_start}, {mid_end}) = {mid_end - mid_start} layers")

    # Load BGE for retrieval
    print("Loading BGE-large...")
    bge = SentenceTransformer("BAAI/bge-large-en-v1.5", device=args.device)

    # Build V-delta
    v_delta = build_v_delta(model, tok, FORMAL_PAIRS, device)

    # Load data
    questions = load_hotpotqa(args.n)

    # Alpha values for dual-channel sweep
    alphas = [0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0]

    # ========================================================================
    # Phase 1: Run 6 main conditions (Table 7)
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 1: Main conditions (N={len(questions)})")
    print(f"{'='*70}")

    STEERING_TEXT = "Respond in a formal, detailed, and academic style. Use precise terminology and complete sentences."

    results_main = {
        "A_baseline":    {"em": [], "form": [], "words": []},
        "B_knowledge":   {"em": [], "form": [], "words": []},
        "C_steering":    {"em": [], "form": [], "words": []},
        "D_dual_a2.0":   {"em": [], "form": [], "words": []},
        "E_rag":         {"em": [], "form": [], "words": []},
        "F_rag_steer":   {"em": [], "form": [], "words": []},
    }

    t_start = time.time()

    for i, item in enumerate(questions):
        question = item["question"]
        gold_answer = item["answer"]
        gold_paras = get_gold_paragraphs(item)
        facts_text = "\n".join(gold_paras)

        if i % 25 == 0:
            elapsed = time.time() - t_start
            eta = (elapsed / max(i, 1)) * (len(questions) - i)
            print(f"\n[{i}/{len(questions)}] elapsed={elapsed:.0f}s, ETA={eta:.0f}s", flush=True)

        # A: Baseline
        ans = generate_baseline(model, tok, question, device=args.device)
        results_main["A_baseline"]["em"].append(check_answer(ans, gold_answer))
        results_main["A_baseline"]["form"].append(score_formality(ans))
        results_main["A_baseline"]["words"].append(len(ans.split()))

        # B: Knowledge only (KV with chat template)
        try:
            kv, kv_len = build_kv_chat(model, tok, facts_text, device)
            ans = generate_with_kv(model, tok, question, kv, kv_len, device)
            del kv
        except Exception as e:
            ans = f"ERROR: {e}"
        results_main["B_knowledge"]["em"].append(check_answer(ans, gold_answer))
        results_main["B_knowledge"]["form"].append(score_formality(ans))
        results_main["B_knowledge"]["words"].append(len(ans.split()))

        # C: Steering only (V-delta, no knowledge)
        try:
            # Build a bare system KV (no facts)
            bare_kv, bare_len = build_kv_chat(model, tok, "Answer the question concisely.", device)
            steered_kv = apply_v_delta(bare_kv, v_delta, alpha=2.0, mid_start=mid_start, mid_end=mid_end)
            ans = generate_with_kv(model, tok, question, steered_kv, bare_len, device)
            del bare_kv, steered_kv
        except Exception as e:
            ans = f"ERROR: {e}"
        results_main["C_steering"]["em"].append(check_answer(ans, gold_answer))
        results_main["C_steering"]["form"].append(score_formality(ans))
        results_main["C_steering"]["words"].append(len(ans.split()))

        # D: Dual-channel (knowledge KV + steering V-delta, alpha=2.0)
        try:
            kv, kv_len = build_kv_chat(model, tok, facts_text, device)
            steered_kv = apply_v_delta(kv, v_delta, alpha=2.0, mid_start=mid_start, mid_end=mid_end)
            ans = generate_with_kv(model, tok, question, steered_kv, kv_len, device)
            del kv, steered_kv
        except Exception as e:
            ans = f"ERROR: {e}"
        results_main["D_dual_a2.0"]["em"].append(check_answer(ans, gold_answer))
        results_main["D_dual_a2.0"]["form"].append(score_formality(ans))
        results_main["D_dual_a2.0"]["words"].append(len(ans.split()))

        # E: RAG (facts in prompt)
        ans, _ = generate_rag(model, tok, question, facts_text, device=args.device)
        results_main["E_rag"]["em"].append(check_answer(ans, gold_answer))
        results_main["E_rag"]["form"].append(score_formality(ans))
        results_main["E_rag"]["words"].append(len(ans.split()))

        # F: RAG + text steering
        ans, _ = generate_rag(model, tok, question, facts_text, system_suffix=STEERING_TEXT, device=args.device)
        results_main["F_rag_steer"]["em"].append(check_answer(ans, gold_answer))
        results_main["F_rag_steer"]["form"].append(score_formality(ans))
        results_main["F_rag_steer"]["words"].append(len(ans.split()))

        # Cleanup
        if i % 10 == 0:
            torch.cuda.empty_cache()

        # Print sample
        if i < 3:
            print(f"  Q: {question}")
            print(f"  Gold: {gold_answer}")
            for cond, data in results_main.items():
                em = data["em"][-1]
                fm = data["form"][-1]
                wc = data["words"][-1]
                print(f"    {cond}: {'Y' if em else 'N'} form={fm:.2f} words={wc}")
            sys.stdout.flush()

    # Print Phase 1 summary
    print(f"\n{'='*70}")
    print(f"PHASE 1 RESULTS (N={len(questions)})")
    print(f"{'='*70}")
    print(f"{'Condition':>20s}  {'EM':>8s}  {'Formality':>10s}  {'Words':>8s}")
    print("-" * 52)
    for cond, data in results_main.items():
        em = sum(data["em"]) / len(data["em"])
        fm = np.mean(data["form"])
        wc = np.mean(data["words"])
        print(f"{cond:>20s}  {em:>7.1%}  {fm:>10.2f}  {wc:>8.0f}")

    # ========================================================================
    # Phase 2: Alpha sweep for dual-channel (Table 8)
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 2: Alpha sweep (N={len(questions)})")
    print(f"{'='*70}")

    results_alpha = {}
    for alpha in alphas:
        key = f"alpha_{alpha}"
        results_alpha[key] = {"em": [], "form": [], "words": []}

        t_sweep = time.time()
        for i, item in enumerate(questions):
            question = item["question"]
            gold_answer = item["answer"]
            gold_paras = get_gold_paragraphs(item)
            facts_text = "\n".join(gold_paras)

            try:
                kv, kv_len = build_kv_chat(model, tok, facts_text, device)
                if alpha > 0:
                    steered_kv = apply_v_delta(kv, v_delta, alpha=alpha,
                                               mid_start=mid_start, mid_end=mid_end)
                    ans = generate_with_kv(model, tok, question, steered_kv, kv_len, device)
                    del steered_kv
                else:
                    ans = generate_with_kv(model, tok, question, kv, kv_len, device)
                del kv
            except Exception as e:
                ans = f"ERROR: {e}"

            results_alpha[key]["em"].append(check_answer(ans, gold_answer))
            results_alpha[key]["form"].append(score_formality(ans))
            results_alpha[key]["words"].append(len(ans.split()))

            if i % 50 == 0:
                torch.cuda.empty_cache()

        em = sum(results_alpha[key]["em"]) / len(results_alpha[key]["em"])
        fm = np.mean(results_alpha[key]["form"])
        wc = np.mean(results_alpha[key]["words"])
        dt = time.time() - t_sweep
        print(f"  alpha={alpha:.1f}: EM={em:.1%} formality={fm:.2f} words={wc:.0f} ({dt:.0f}s)")

    # ========================================================================
    # Save results
    # ========================================================================
    output = {
        "config": {
            "model": args.model,
            "n_questions": len(questions),
            "n_layers": n_layers,
            "mid_layers": [mid_start, mid_end],
            "n_contrastive_pairs": len(FORMAL_PAIRS),
            "alphas": alphas,
        },
        "main_conditions": {},
        "alpha_sweep": {},
    }

    for cond, data in results_main.items():
        output["main_conditions"][cond] = {
            "em": sum(data["em"]) / len(data["em"]),
            "formality": float(np.mean(data["form"])),
            "avg_words": float(np.mean(data["words"])),
            "n": len(data["em"]),
            "em_raw": data["em"],
        }

    for key, data in results_alpha.items():
        output["alpha_sweep"][key] = {
            "em": sum(data["em"]) / len(data["em"]),
            "formality": float(np.mean(data["form"])),
            "avg_words": float(np.mean(data["words"])),
            "n": len(data["em"]),
            "em_raw": data["em"],
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Final summary
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY — Dual-Channel Experiment")
    print(f"Model: {args.model}, N={len(questions)}, Time: {total_time:.0f}s")
    print(f"{'='*70}")

    print(f"\n--- Table 7: Main conditions ---")
    print(f"{'':>5s} {'Condition':>25s}  {'EM':>8s}  {'Form.':>8s}  {'Words':>8s}")
    labels = {"A_baseline": "No knowledge, no steering",
              "B_knowledge": "Knowledge only (KV)",
              "C_steering": "Steering only (V-delta)",
              "D_dual_a2.0": "Knowledge + Steering (a=2.0)",
              "E_rag": "Text knowledge (RAG)",
              "F_rag_steer": "Text knowledge + text steering"}
    for cond, label in labels.items():
        d = output["main_conditions"][cond]
        print(f"{'':>5s} {label:>25s}  {d['em']:>7.1%}  {d['formality']:>8.2f}  {d['avg_words']:>8.0f}")

    print(f"\n--- Table 8: Alpha sweep (dual-channel) ---")
    print(f"{'alpha':>8s}  {'EM':>8s}  {'Form.':>8s}  {'Words':>8s}")
    for alpha in alphas:
        d = output["alpha_sweep"][f"alpha_{alpha}"]
        print(f"{alpha:>8.1f}  {d['em']:>7.1%}  {d['formality']:>8.2f}  {d['avg_words']:>8.0f}")


if __name__ == "__main__":
    main()
