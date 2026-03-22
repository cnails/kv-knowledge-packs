"""
SimpleQA Benchmark — Fair 3-way comparison.

Three methods, same facts, same questions, same model:
  1. KV-pack:  top-1 routing + lazy recompute, 0 prompt tokens
  2. RAG:      proper embedding model (BGE) + top-1 retrieval in prompt
  3. Prefix:   all facts concatenated in prompt (upper bound accuracy)
  4. Baseline: no knowledge (lower bound)

Usage:
  python benchmarks/simpleqa_bench.py --model Qwen/Qwen3-8B --n 200
  python benchmarks/simpleqa_bench.py --model meta-llama/Llama-4-17B-Instruct --n 500
"""

import argparse
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpack import KnowledgePack


SYSTEM_MSG = "Answer factual questions concisely. Give only the answer, no explanation."


def make_fact(question: str, answer: str) -> str:
    """Convert Q/A pair into a declarative fact."""
    q = question.strip().rstrip("?").strip()
    return f"The answer to \"{q}\" is {answer}."


def evaluate(generated: str, gold: str) -> bool:
    """Check if gold answer appears in generated text."""
    return gold.lower() in generated.lower()


def chat_generate(model, tokenizer, messages, max_new_tokens=64):
    """Generate response from chat messages."""
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════
# Method 1: Baseline (no knowledge)
# ═══════════════════════════════════════════════════════════════════

def run_baseline(model, tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": question},
    ]
    return chat_generate(model, tokenizer, messages)


# ═══════════════════════════════════════════════════════════════════
# Method 2: RAG (proper embedding model + top-1 retrieval in prompt)
# ═══════════════════════════════════════════════════════════════════

class RAGRetriever:
    """RAG with a proper embedding model (sentence-transformers)."""

    def __init__(self, embedding_model_name="BAAI/bge-base-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        print(f"  Loading RAG embedding model: {embedding_model_name}")
        self.embed_model = SentenceTransformer(embedding_model_name)
        self.facts = []
        self.embeddings = None

    def index(self, facts: list[str]):
        self.facts = facts
        print(f"  Encoding {len(facts)} facts for RAG...")
        self.embeddings = self.embed_model.encode(
            facts, convert_to_tensor=True, show_progress_bar=False
        )
        print(f"  RAG index built: {self.embeddings.shape}")

    def retrieve(self, query: str, top_k: int = 1) -> list[str]:
        q_emb = self.embed_model.encode(query, convert_to_tensor=True)
        cos = F.cosine_similarity(q_emb.unsqueeze(0), self.embeddings)
        topk = cos.topk(top_k)
        return [self.facts[i] for i in topk.indices.tolist()]


def run_rag(model, tokenizer, question, retriever, top_k=1):
    retrieved = retriever.retrieve(question, top_k=top_k)
    context = "\n".join(retrieved)
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return chat_generate(model, tokenizer, messages), retrieved


# ═══════════════════════════════════════════════════════════════════
# Method 3: Prefix (all facts in prompt — upper bound)
# ═══════════════════════════════════════════════════════════════════

def run_prefix(model, tokenizer, question, all_facts_text, max_new_tokens=64):
    """All facts concatenated in system prompt."""
    messages = [
        {"role": "system", "content": f"{SYSTEM_MSG}\n\nKnown facts:\n{all_facts_text}"},
        {"role": "user", "content": question},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = input_ids.shape[1]

    # Check if within context window
    max_pos = getattr(model.config, "max_position_embeddings", 32768)
    if prompt_tokens + max_new_tokens > max_pos:
        return "[PREFIX_TOO_LONG]", prompt_tokens

    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    return answer, prompt_tokens


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SimpleQA 3-way Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--n", type=int, default=200, help="Number of questions")
    parser.add_argument("--n-banks", type=int, default=50)
    parser.add_argument("--rag-model", default="BAAI/bge-base-en-v1.5",
                        help="Embedding model for RAG")
    parser.add_argument("--skip-prefix", action="store_true",
                        help="Skip prefix test (slow for large N)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="simpleqa_results.json")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # ── Load dataset ──
    print("Loading SimpleQA...")
    ds = load_dataset("basicv8vc/SimpleQA", split="test")
    subset = list(ds.select(range(min(args.n, len(ds)))))
    print(f"Using {len(subset)} questions")

    # Build facts
    facts = [make_fact(row["problem"], row["answer"]) for row in subset]
    all_facts_text = "\n".join(facts)
    all_facts_tokens = len(all_facts_text.split()) * 1.3  # rough estimate

    # ── Load LLM ──
    print(f"\nLoading {args.model}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=device
    )
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    # ── Build KV Pack ──
    print(f"\nBuilding KV KnowledgePack...")
    n_banks = max(1, min(args.n_banks, len(facts) // 2))
    pack = KnowledgePack(
        model_name=args.model, model=model, tokenizer=tokenizer,
        n_banks=n_banks, device=str(device),
    )
    pack.add_facts(facts)
    pack.build()

    # ── Build RAG index ──
    print(f"\nBuilding RAG index...")
    retriever = RAGRetriever(args.rag_model)
    retriever.index(facts)

    # ── Run all methods ──
    results = {m: {"correct": 0, "total": 0, "time": 0, "answers": []}
               for m in ["baseline", "kv_pack", "rag", "prefix"]}

    print(f"\n{'='*70}")
    print(f"Running benchmark: {len(subset)} questions, 4 methods")
    print(f"{'='*70}")

    for i, row in enumerate(subset):
        q = row["problem"]
        gold = row["answer"]

        # ── Baseline ──
        t0 = time.time()
        ans_base = run_baseline(model, tokenizer, q)
        t_base = time.time() - t0
        hit_base = evaluate(ans_base, gold)
        results["baseline"]["correct"] += int(hit_base)
        results["baseline"]["total"] += 1
        results["baseline"]["time"] += t_base

        # ── KV Pack ──
        t0 = time.time()
        kv_result = pack.query_with_metadata(q, top_k=1, max_new_tokens=64, do_sample=False)
        t_kv = time.time() - t0
        hit_kv = evaluate(kv_result["answer"], gold)
        results["kv_pack"]["correct"] += int(hit_kv)
        results["kv_pack"]["total"] += 1
        results["kv_pack"]["time"] += t_kv

        # ── RAG ──
        t0 = time.time()
        ans_rag, retrieved = run_rag(model, tokenizer, q, retriever, top_k=1)
        t_rag = time.time() - t0
        hit_rag = evaluate(ans_rag, gold)
        results["rag"]["correct"] += int(hit_rag)
        results["rag"]["total"] += 1
        results["rag"]["time"] += t_rag

        # ── Prefix ── (skip if too many facts or flag set)
        if not args.skip_prefix and len(subset) <= 500:
            t0 = time.time()
            ans_prefix, n_tok = run_prefix(model, tokenizer, q, all_facts_text)
            t_prefix = time.time() - t0
            if ans_prefix != "[PREFIX_TOO_LONG]":
                hit_prefix = evaluate(ans_prefix, gold)
                results["prefix"]["correct"] += int(hit_prefix)
                results["prefix"]["total"] += 1
                results["prefix"]["time"] += t_prefix
        else:
            results["prefix"]["total"] += 1

        # Progress
        if (i + 1) % 25 == 0:
            n = i + 1
            print(f"\n  [{n}/{len(subset)}]")
            for method in ["baseline", "kv_pack", "rag", "prefix"]:
                r = results[method]
                if r["total"] > 0 and r["correct"] >= 0:
                    acc = r["correct"] / r["total"] * 100
                    avg_ms = r["time"] / r["total"] * 1000
                    print(f"    {method:12s}: {acc:5.1f}%  ({avg_ms:.0f}ms/q)")

        # Save sample answers (first 5)
        if i < 5:
            for method, ans, hit in [
                ("baseline", ans_base, hit_base),
                ("kv_pack", kv_result["answer"], hit_kv),
                ("rag", ans_rag, hit_rag),
            ]:
                results[method]["answers"].append({
                    "question": q[:80],
                    "gold": gold,
                    "generated": ans[:100],
                    "correct": hit,
                })

    # ── Final Summary ──
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS — {args.model}")
    print(f"{'='*70}")
    print(f"{'Method':15s} {'Accuracy':>10s} {'Latency':>10s} {'Prompt Tok':>12s}")
    print(f"{'-'*50}")

    method_info = {
        "baseline":  ("Baseline",  "~50"),
        "rag":       ("RAG (BGE)", "~80"),
        "kv_pack":   ("KV Pack",   "0"),
        "prefix":    ("Prefix",    f"~{int(all_facts_tokens)}"),
    }

    for method in ["baseline", "rag", "kv_pack", "prefix"]:
        r = results[method]
        name, tok = method_info[method]
        if r["total"] > 0 and r["correct"] >= 0:
            acc = r["correct"] / r["total"] * 100
            avg_ms = r["time"] / r["total"] * 1000 if r["time"] > 0 else 0
            print(f"{name:15s} {acc:9.1f}% {avg_ms:9.0f}ms {tok:>12s}")
        else:
            print(f"{name:15s} {'N/A':>10s}")

    # KV routing accuracy
    print(f"\nKV routing overhead: ~{pack.info()['n_banks']} banks")

    # ── Save ──
    output_data = {
        "model": args.model,
        "n_questions": len(subset),
        "n_banks": n_banks,
        "rag_model": args.rag_model,
        "results": {
            method: {
                "accuracy": round(r["correct"] / r["total"] * 100, 1) if r["total"] > 0 else 0,
                "avg_ms": round(r["time"] / r["total"] * 1000, 1) if r["total"] > 0 else 0,
                "correct": r["correct"],
                "total": r["total"],
                "sample_answers": r["answers"],
            }
            for method, r in results.items()
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
