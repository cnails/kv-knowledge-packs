"""
Exp 93: KV Composition — Cross-pack multi-hop reasoning.

Tests whether multiple KV caches can be composed (concatenated) to enable
reasoning across independently stored knowledge domains.

Setup:
  - For each HotpotQA bridge question, the 2 gold paragraphs are stored
    in SEPARATE KV caches (simulating independent knowledge packs).
  - At query time, both KV caches are concatenated into a single attention context.
  - The model must cross-reference facts from both packs to answer.

This tests a capability RAG cannot easily replicate: composing independently
retrieved knowledge from different sources/domains into unified reasoning.

Compared methods:
  1. KV Composed: 2 separate KV caches concatenated at query time
  2. KV Single:   2 paragraphs in 1 KV cache (current approach, upper bound)
  3. RAG top-1:   Best single paragraph in prompt (baseline)
  4. Prefix:      Both paragraphs as text in prompt

Model: Llama-3.1-70B-Instruct on B200.
"""
import torch
import json
import time
import random
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from sentence_transformers import SentenceTransformer
import numpy as np

device = torch.device("cuda")

# ============================================================
# Load model
# ============================================================
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

print(f"Loading {MODEL_NAME}...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN
)
model.eval()
print(f"Model loaded in {time.time()-t0:.0f}s")

print("Loading BGE-large...")
bge = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

# ============================================================
# Load HotpotQA
# ============================================================
print("Loading HotpotQA...")
HOTPOT_PATH = "/root/hotpot_dev_distractor_v1.json"
if not os.path.exists(HOTPOT_PATH):
    print("Downloading HotpotQA dev set...")
    os.system(f"wget -q http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O {HOTPOT_PATH}")

with open(HOTPOT_PATH) as f:
    hotpot_data = json.load(f)
print(f"Loaded {len(hotpot_data)} questions")

# ============================================================
# Helper functions
# ============================================================

def get_gold_paragraphs(item):
    """Extract gold supporting paragraphs for a HotpotQA question."""
    supporting_titles = set(sf[0] for sf in item["supporting_facts"])
    gold_paragraphs = []
    for title, sentences in item["context"]:
        if title in supporting_titles:
            text = f"{title}: {' '.join(sentences)}"
            gold_paragraphs.append(text)
    return gold_paragraphs


def get_all_paragraphs(item):
    """Get all context paragraphs (gold + distractors)."""
    paragraphs = []
    for title, sentences in item["context"]:
        text = f"{title}: {' '.join(sentences)}"
        paragraphs.append(text)
    return paragraphs


def build_kv(text):
    """Build KV cache from text."""
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > 4096:
        ids = ids[:4096]
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(t, use_cache=True)
    return out.past_key_values, len(ids)


def clone_kv(kv):
    """Clone a KV cache."""
    c = DynamicCache()
    if hasattr(kv, 'layers'):
        for li in range(len(kv.layers)):
            k = kv.layers[li].keys.clone()
            v = kv.layers[li].values.clone()
            c.update(k, v, li)
    else:
        for li in range(len(kv)):
            k, v = kv[li]
            c.update(k.clone(), v.clone(), li)
    return c


def compose_kv(kv_list_with_lens):
    """
    Compose multiple KV caches by concatenating along the sequence dimension.

    Args:
        kv_list_with_lens: list of (kv_cache, kv_len) tuples

    Returns:
        (composed_kv, total_len)
    """
    composed = DynamicCache()

    # Get number of layers from first cache
    first_kv = kv_list_with_lens[0][0]
    if hasattr(first_kv, 'layers'):
        n_layers = len(first_kv.layers)
    else:
        n_layers = len(first_kv)

    for li in range(n_layers):
        keys_to_cat = []
        vals_to_cat = []
        for kv, kv_len in kv_list_with_lens:
            if hasattr(kv, 'layers'):
                keys_to_cat.append(kv.layers[li].keys)
                vals_to_cat.append(kv.layers[li].values)
            else:
                k, v = kv[li]
                keys_to_cat.append(k)
                vals_to_cat.append(v)

        # Concatenate along sequence dimension (dim=2)
        cat_k = torch.cat(keys_to_cat, dim=2)
        cat_v = torch.cat(vals_to_cat, dim=2)
        composed.update(cat_k, cat_v, li)

    total_len = sum(l for _, l in kv_list_with_lens)
    return composed, total_len


def generate_with_kv(query, kv, kv_len, max_tokens=50):
    """Generate answer with KV cache (0 prompt tokens for facts)."""
    messages = [
        {"role": "system", "content": "Answer the question concisely based on what you know. Give a short, direct answer."},
        {"role": "user", "content": query}
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    q_ids = tok.encode(prompt, add_special_tokens=False)
    qt = torch.tensor([q_ids], device=device)
    am = torch.ones(1, kv_len + len(q_ids), device=device, dtype=torch.long)
    kv_c = clone_kv(kv)
    with torch.no_grad():
        out = model.generate(qt, past_key_values=kv_c, attention_mask=am,
                             max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(q_ids):], skip_special_tokens=True).strip()


def generate_with_prefix(query, facts_text, max_tokens=50):
    """Generate answer with facts in prompt."""
    messages = [
        {"role": "system", "content": "Answer the question concisely based on the provided context. Give a short, direct answer."},
        {"role": "user", "content": f"Context:\n{facts_text}\n\nQuestion: {query}"}
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(t, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(ids):], skip_special_tokens=True).strip()


def rag_retrieve(query, paragraphs, top_k=1):
    """Retrieve top-k paragraphs using BGE."""
    q_emb = bge.encode([query], normalize_embeddings=True)
    p_embs = bge.encode(paragraphs, normalize_embeddings=True)
    scores = (q_emb @ p_embs.T)[0]
    topk_idx = scores.argsort()[::-1][:top_k]
    return [paragraphs[i] for i in topk_idx]


def check_answer(predicted, gold):
    """Simple F1-style matching."""
    pred_lower = predicted.lower().strip()
    gold_lower = gold.lower().strip()
    if gold_lower in pred_lower:
        return True
    pred_tokens = set(pred_lower.split())
    gold_tokens = set(gold_lower.split())
    if gold_tokens and pred_tokens:
        overlap = pred_tokens & gold_tokens
        if len(overlap) / len(gold_tokens) >= 0.5:
            return True
    return False


# ============================================================
# Run composition benchmark
# ============================================================
N_TEST = 200
random.seed(42)

# Only bridge questions — they require 2 paragraphs (perfect for composition test)
bridge_questions = [q for q in hotpot_data if q.get("type") == "bridge"]
print(f"Bridge questions available: {len(bridge_questions)}")

# Filter to questions with exactly 2 gold paragraphs
valid_questions = []
for q in bridge_questions:
    gp = get_gold_paragraphs(q)
    if len(gp) == 2:
        valid_questions.append(q)
print(f"Questions with exactly 2 gold paragraphs: {len(valid_questions)}")

test_questions = random.sample(valid_questions, min(N_TEST, len(valid_questions)))
print(f"\nTesting {len(test_questions)} bridge questions")

results = {
    "kv_composed":  [],  # 2 separate KV caches, concatenated
    "kv_single":    [],  # 2 paragraphs in 1 KV cache (upper bound)
    "rag_top1":     [],  # RAG top-1 from all 10 paragraphs
    "prefix_gold":  [],  # Both gold paragraphs in prompt
}

timings = {k: [] for k in results.keys()}

for i, item in enumerate(test_questions):
    question = item["question"]
    gold_answer = item["answer"]
    gold_paras = get_gold_paragraphs(item)
    all_paras = get_all_paragraphs(item)

    if i % 20 == 0:
        print(f"\n--- Progress: {i}/{len(test_questions)} ---")

    # 1. KV Composed: build separate KV for each paragraph, then compose
    t0 = time.time()
    try:
        kv1, len1 = build_kv(gold_paras[0])
        kv2, len2 = build_kv(gold_paras[1])
        composed_kv, composed_len = compose_kv([(kv1, len1), (kv2, len2)])
        ans_composed = generate_with_kv(question, composed_kv, composed_len)
        del kv1, kv2, composed_kv
    except Exception as e:
        ans_composed = f"ERROR: {e}"
    t_composed = time.time() - t0
    hit_composed = check_answer(ans_composed, gold_answer)
    results["kv_composed"].append(hit_composed)
    timings["kv_composed"].append(t_composed)

    # 2. KV Single: both paragraphs in one KV cache
    t0 = time.time()
    try:
        kv_both, len_both = build_kv(" ".join(gold_paras))
        ans_single = generate_with_kv(question, kv_both, len_both)
        del kv_both
    except Exception as e:
        ans_single = f"ERROR: {e}"
    t_single = time.time() - t0
    hit_single = check_answer(ans_single, gold_answer)
    results["kv_single"].append(hit_single)
    timings["kv_single"].append(t_single)

    # 3. RAG top-1 from all 10 paragraphs
    t0 = time.time()
    try:
        retrieved = rag_retrieve(question, all_paras, top_k=1)
        ans_rag = generate_with_prefix(question, "\n".join(retrieved))
    except Exception as e:
        ans_rag = f"ERROR: {e}"
    t_rag = time.time() - t0
    hit_rag = check_answer(ans_rag, gold_answer)
    results["rag_top1"].append(hit_rag)
    timings["rag_top1"].append(t_rag)

    # 4. Prefix with gold paragraphs
    t0 = time.time()
    try:
        ans_prefix = generate_with_prefix(question, "\n".join(gold_paras))
    except Exception as e:
        ans_prefix = f"ERROR: {e}"
    t_prefix = time.time() - t0
    hit_prefix = check_answer(ans_prefix, gold_answer)
    results["prefix_gold"].append(hit_prefix)
    timings["prefix_gold"].append(t_prefix)

    # Print examples
    if i < 5 or (i % 50 == 0):
        print(f"\n  Q: {question}")
        print(f"  Gold: {gold_answer}")
        print(f"  Pack A: {gold_paras[0][:80]}...")
        print(f"  Pack B: {gold_paras[1][:80]}...")
        print(f"  KV Composed: {'✓' if hit_composed else '✗'} {ans_composed[:80]}")
        print(f"  KV Single:   {'✓' if hit_single else '✗'} {ans_single[:80]}")
        print(f"  RAG top-1:   {'✓' if hit_rag else '✗'} {ans_rag[:80]}")
        print(f"  Prefix:      {'✓' if hit_prefix else '✗'} {ans_prefix[:80]}")

    # Memory cleanup
    if i % 10 == 0:
        torch.cuda.empty_cache()

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print(f"KV COMPOSITION BENCHMARK — {MODEL_NAME}")
print(f"N={len(test_questions)} bridge questions (2 gold paragraphs each)")
print("="*70)
print()
print("Key question: Does composing separately-built KV caches preserve")
print("multi-hop reasoning ability?")
print()

for method, hits in results.items():
    c = sum(hits)
    n = len(hits)
    avg_time = np.mean(timings[method])
    print(f"  {method:>15s}: {c}/{n} = {100*c/n:.1f}%  (avg {avg_time:.2f}s/query)")

print()
print("--- Interpretation ---")
c_composed = sum(results["kv_composed"])
c_single = sum(results["kv_single"])
n = len(test_questions)
gap = 100 * c_composed / n - 100 * c_single / n
print(f"  Composition gap (composed - single): {gap:+.1f}pp")
if abs(gap) <= 3:
    print("  -> KV composition preserves accuracy! Separate KV caches can be")
    print("     composed at query time with minimal degradation.")
elif gap < -3:
    print(f"  -> Composition loses {abs(gap):.1f}pp vs single KV.")
    print("     Separate forward passes may miss cross-paragraph interactions.")

print()
print("--- Why this matters ---")
print("  KV Composed uses 0 prompt tokens and composes independent knowledge packs.")
print("  This enables: modular knowledge (medical + legal + finance packs),")
print("  multi-source reasoning, and dynamic knowledge assembly at query time.")
print("  RAG cannot compose retrieved documents into unified attention context")
print("  without consuming prompt tokens for each document.")

print("\nDone.")
