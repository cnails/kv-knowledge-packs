"""
Exp 92: HotpotQA — Multi-hop QA benchmark.

KV injection vs RAG vs Prefix on established multi-hop reasoning benchmark.
Uses gold supporting paragraphs from HotpotQA distractor setting.

Model: Llama-3.1-70B-Instruct on B200 (178GB).
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
    # Truncate if too long
    if len(ids) > 4096:
        ids = ids[:4096]
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(t, use_cache=True)
    return out.past_key_values, len(ids)


def clone_kv(kv):
    c = DynamicCache()
    n_layers = len(kv) if hasattr(kv, '__len__') else len(kv.layers) if hasattr(kv, 'layers') else 0
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


def generate_with_kv(query, kv, kv_len, max_tokens=50):
    """Generate answer with KV cache (0 prompt tokens for facts)."""
    # Use chat template for Llama-3.1
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
    """Generate answer with facts in prompt (RAG/Prefix style)."""
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


def generate_baseline(query, max_tokens=50):
    """Generate answer without any external knowledge."""
    messages = [
        {"role": "system", "content": "Answer the question concisely. Give a short, direct answer."},
        {"role": "user", "content": query}
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

    # Exact match
    if gold_lower in pred_lower:
        return True

    # Token overlap
    pred_tokens = set(pred_lower.split())
    gold_tokens = set(gold_lower.split())
    if gold_tokens and pred_tokens:
        overlap = pred_tokens & gold_tokens
        if len(overlap) / len(gold_tokens) >= 0.5:
            return True

    return False


# ============================================================
# Run benchmark
# ============================================================
N_TEST = 200  # number of questions to evaluate
random.seed(42)

# Filter to "hard" (bridge) type questions for multi-hop focus
bridge_questions = [q for q in hotpot_data if q.get("type") == "bridge"]
comparison_questions = [q for q in hotpot_data if q.get("type") == "comparison"]
print(f"Bridge questions: {len(bridge_questions)}, Comparison: {len(comparison_questions)}")

# Mix: 150 bridge + 50 comparison
test_bridge = random.sample(bridge_questions, min(150, len(bridge_questions)))
test_comparison = random.sample(comparison_questions, min(50, len(comparison_questions)))
test_questions = test_bridge + test_comparison
random.shuffle(test_questions)
test_questions = test_questions[:N_TEST]

print(f"\nTesting {len(test_questions)} questions ({sum(1 for q in test_questions if q['type']=='bridge')} bridge, {sum(1 for q in test_questions if q['type']=='comparison')} comparison)")

results = {
    "kv_gold": [],      # KV with gold paragraphs
    "rag1_all": [],      # RAG top-1 from all paragraphs (10)
    "rag2_all": [],      # RAG top-2 from all paragraphs
    "prefix_gold": [],   # Gold paragraphs in prompt (upper bound)
    "baseline": [],      # No external knowledge
}

timings = {k: [] for k in results.keys()}

for i, item in enumerate(test_questions):
    question = item["question"]
    gold_answer = item["answer"]
    gold_paras = get_gold_paragraphs(item)
    all_paras = get_all_paragraphs(item)

    if i % 20 == 0:
        print(f"\n--- Progress: {i}/{len(test_questions)} ---")

    # 1. KV with gold paragraphs
    t0 = time.time()
    gold_text = " ".join(gold_paras)
    try:
        kv, kv_len = build_kv(gold_text)
        ans_kv = generate_with_kv(question, kv, kv_len)
        del kv
    except Exception as e:
        ans_kv = f"ERROR: {e}"
    t_kv = time.time() - t0
    hit_kv = check_answer(ans_kv, gold_answer)
    results["kv_gold"].append(hit_kv)
    timings["kv_gold"].append(t_kv)

    # 2. RAG top-1 from all 10 paragraphs
    t0 = time.time()
    try:
        retrieved_1 = rag_retrieve(question, all_paras, top_k=1)
        ans_rag1 = generate_with_prefix(question, "\n".join(retrieved_1))
    except Exception as e:
        ans_rag1 = f"ERROR: {e}"
    t_rag1 = time.time() - t0
    hit_rag1 = check_answer(ans_rag1, gold_answer)
    results["rag1_all"].append(hit_rag1)
    timings["rag1_all"].append(t_rag1)

    # 3. RAG top-2 from all 10 paragraphs
    t0 = time.time()
    try:
        retrieved_2 = rag_retrieve(question, all_paras, top_k=2)
        ans_rag2 = generate_with_prefix(question, "\n".join(retrieved_2))
    except Exception as e:
        ans_rag2 = f"ERROR: {e}"
    t_rag2 = time.time() - t0
    hit_rag2 = check_answer(ans_rag2, gold_answer)
    results["rag2_all"].append(hit_rag2)
    timings["rag2_all"].append(t_rag2)

    # 4. Prefix with gold paragraphs (upper bound)
    t0 = time.time()
    try:
        ans_prefix = generate_with_prefix(question, "\n".join(gold_paras))
    except Exception as e:
        ans_prefix = f"ERROR: {e}"
    t_prefix = time.time() - t0
    hit_prefix = check_answer(ans_prefix, gold_answer)
    results["prefix_gold"].append(hit_prefix)
    timings["prefix_gold"].append(t_prefix)

    # 5. Baseline (no knowledge)
    t0 = time.time()
    try:
        ans_base = generate_baseline(question)
    except Exception as e:
        ans_base = f"ERROR: {e}"
    t_base = time.time() - t0
    hit_base = check_answer(ans_base, gold_answer)
    results["baseline"].append(hit_base)
    timings["baseline"].append(t_base)

    # Print examples
    if i < 5 or (i % 50 == 0):
        print(f"\n  Q: {question}")
        print(f"  Gold: {gold_answer}")
        print(f"  KV:      {'✓' if hit_kv else '✗'} {ans_kv[:80]}")
        print(f"  RAG-1:   {'✓' if hit_rag1 else '✗'} {ans_rag1[:80]}")
        print(f"  RAG-2:   {'✓' if hit_rag2 else '✗'} {ans_rag2[:80]}")
        print(f"  Prefix:  {'✓' if hit_prefix else '✗'} {ans_prefix[:80]}")
        print(f"  Base:    {'✓' if hit_base else '✗'} {ans_base[:80]}")

    # Memory cleanup
    if i % 10 == 0:
        torch.cuda.empty_cache()

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print(f"HOTPOTQA BENCHMARK RESULTS — {MODEL_NAME}")
print(f"N={len(test_questions)} questions (bridge + comparison)")
print("="*70)

for method, hits in results.items():
    c = sum(hits)
    n = len(hits)
    avg_time = np.mean(timings[method])
    print(f"  {method:>15s}: {c}/{n} = {100*c/n:.1f}%  (avg {avg_time:.2f}s/query)")

# Bridge vs Comparison breakdown
print(f"\n--- By question type ---")
for qtype in ["bridge", "comparison"]:
    indices = [i for i, q in enumerate(test_questions) if q["type"] == qtype]
    if not indices:
        continue
    print(f"\n  {qtype.upper()} ({len(indices)} questions):")
    for method, hits in results.items():
        c = sum(hits[i] for i in indices)
        n = len(indices)
        print(f"    {method:>15s}: {c}/{n} = {100*c/n:.1f}%")

print(f"\n--- Prompt tokens ---")
print(f"  KV (gold):     0 prompt tokens (facts in KV cache)")
print(f"  RAG-1:         ~100-200 tokens (1 paragraph in prompt)")
print(f"  RAG-2:         ~200-400 tokens (2 paragraphs in prompt)")
print(f"  Prefix (gold): ~200-400 tokens (gold paragraphs in prompt)")

print("\nDone.")
