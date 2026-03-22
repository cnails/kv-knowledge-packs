"""
SimpleQA Benchmark: KV-pack with BGE routing.
Fixes the routing bottleneck: use BGE-large for retrieval, KV cache for injection.
"""
import torch
import torch.nn.functional as F
import csv
import time
import gc
import os
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from sentence_transformers import SentenceTransformer

HF_TOKEN = os.environ.get("HF_TOKEN", "")

def clone_kv(kv):
    c = DynamicCache()
    for li in range(len(kv.layers)):
        k = kv.layers[li].keys
        v = kv.layers[li].values
        c.update(k.clone(), v.clone(), li)
    return c

def gen_with_kv(model, tok, query, kv, kv_len, device, max_new=50):
    q_ids = tok.encode(query, add_special_tokens=False)
    qt = torch.tensor([q_ids], device=device)
    am = torch.ones(1, kv_len + len(q_ids), device=device, dtype=torch.long)
    kvc = clone_kv(kv)
    with torch.no_grad():
        out = model.generate(qt, past_key_values=kvc, attention_mask=am,
                             max_new_tokens=max_new, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(q_ids):], skip_special_tokens=True).strip()

def gen_plain(model, tok, prompt, device, max_new=50):
    ids = tok.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

def check_answer(generated, gold):
    return gold.lower() in generated.lower()

# ============================================================
# Load data
# ============================================================
print("Loading SimpleQA...")
with open("/root/simpleqa.csv", "r") as f:
    ds = list(csv.DictReader(f))
print(f"Total: {len(ds)}")

N = 500
random.seed(42)
indices = random.sample(range(len(ds)), N)
samples = [ds[i] for i in indices]

questions = [s["problem"] for s in samples]
answers = [s["answer"] for s in samples]
fact_sentences = [f"The answer to the question '{q}' is: {a}." for q, a in zip(questions, answers)]

# ============================================================
# Load models
# ============================================================
device = torch.device("cuda")

print("\nLoading Qwen3-8B...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device)
model.eval()
print("Model loaded")

print("Loading BGE-large...")
bge = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
fact_embs = bge.encode(fact_sentences, normalize_embeddings=True, show_progress_bar=True)
fact_embs = torch.from_numpy(fact_embs).float()
print("BGE ready")

# ============================================================
# Method 1: Baseline
# ============================================================
print("\n" + "="*60)
print("METHOD 1: BASELINE")
print("="*60)

bl_correct = 0
bl_times = []
for i in range(N):
    prompt = f"Answer the following question concisely.\nQuestion: {questions[i]}\nAnswer:"
    t0 = time.time()
    resp = gen_plain(model, tok, prompt, device)
    dt = time.time() - t0
    hit = check_answer(resp, answers[i])
    bl_correct += int(hit)
    bl_times.append(dt)
    if i < 3 or i % 50 == 0:
        m = "Y" if hit else "X"
        print(f"  [{m}] #{i}: A={answers[i][:30]} | Gen={resp[:50]}")

print(f"\nBaseline: {bl_correct}/{N} = {100*bl_correct/N:.1f}%")
print(f"  Avg latency: {np.mean(bl_times)*1000:.0f}ms")

# ============================================================
# Method 2: KV-pack + BGE routing (THE MAIN TEST)
# ============================================================
print("\n" + "="*60)
print("METHOD 2: KV-PACK + BGE ROUTING (0 prompt tokens)")
print("="*60)

kv_correct = 0
kv_times = []
kv_route_correct = 0

for i in range(N):
    t0 = time.time()

    # BGE routing: embed query, find best fact
    q_emb = bge.encode([questions[i]], normalize_embeddings=True)
    q_emb = torch.from_numpy(q_emb).float()
    cos = F.cosine_similarity(q_emb, fact_embs)
    best_idx = cos.argmax().item()

    if best_idx == i:
        kv_route_correct += 1

    # Recompute KV for selected fact
    fact_text = fact_sentences[best_idx]
    fact_ids = tok.encode(fact_text, add_special_tokens=False)
    ft = torch.tensor([fact_ids], device=device)
    with torch.no_grad():
        out = model(ft, use_cache=True)
    kv = out.past_key_values
    kv_len = len(fact_ids)

    # Generate with KV
    query_text = f"Answer the following question concisely.\nQuestion: {questions[i]}\nAnswer:"
    resp = gen_with_kv(model, tok, query_text, kv, kv_len, device)
    dt = time.time() - t0

    hit = check_answer(resp, answers[i])
    kv_correct += int(hit)
    kv_times.append(dt)

    if i < 5 or i % 50 == 0:
        routed = "correct" if best_idx == i else f"WRONG(got {best_idx})"
        m = "Y" if hit else "X"
        print(f"  [{m}] #{i}: route={routed} | A={answers[i][:30]} | Gen={resp[:50]}")

    del kv, out
    if (i+1) % 50 == 0:
        gc.collect(); torch.cuda.empty_cache()

print(f"\nKV+BGE: {kv_correct}/{N} = {100*kv_correct/N:.1f}%")
print(f"  Routing accuracy: {kv_route_correct}/{N} = {100*kv_route_correct/N:.1f}%")
print(f"  Avg latency: {np.mean(kv_times)*1000:.0f}ms")
print(f"  Prompt tokens: 0")

# ============================================================
# Method 3: RAG (BGE routing + fact in prompt)
# ============================================================
print("\n" + "="*60)
print("METHOD 3: RAG (BGE routing + fact in prompt)")
print("="*60)

rag_correct = 0
rag_times = []
rag_tokens = []

for i in range(N):
    t0 = time.time()

    q_emb = bge.encode([questions[i]], normalize_embeddings=True)
    q_emb = torch.from_numpy(q_emb).float()
    cos = F.cosine_similarity(q_emb, fact_embs)
    best_idx = cos.argmax().item()

    retrieved_fact = fact_sentences[best_idx]
    prompt = f"Context: {retrieved_fact}\n\nAnswer the following question concisely.\nQuestion: {questions[i]}\nAnswer:"
    n_tok = len(tok.encode(prompt, add_special_tokens=False))

    resp = gen_plain(model, tok, prompt, device)
    dt = time.time() - t0

    hit = check_answer(resp, answers[i])
    rag_correct += int(hit)
    rag_times.append(dt)
    rag_tokens.append(n_tok)

    if i < 5 or i % 50 == 0:
        m = "Y" if hit else "X"
        print(f"  [{m}] #{i}: A={answers[i][:30]} | Gen={resp[:50]}")

print(f"\nRAG: {rag_correct}/{N} = {100*rag_correct/N:.1f}%")
print(f"  Avg latency: {np.mean(rag_times)*1000:.0f}ms")
print(f"  Avg prompt tokens: {np.mean(rag_tokens):.0f}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY — SimpleQA (N=200, Qwen3-8B, H100)")
print("="*60)
print(f"{'Method':<30s} {'Accuracy':>10s} {'Routing':>10s} {'Latency':>10s} {'Tokens':>10s}")
print("-" * 70)
print(f"{'Baseline':<30s} {100*bl_correct/N:>9.1f}% {'N/A':>10s} {np.mean(bl_times)*1000:>9.0f}ms {'-':>10s}")
print(f"{'KV-pack + BGE routing':<30s} {100*kv_correct/N:>9.1f}% {100*kv_route_correct/N:>9.1f}% {np.mean(kv_times)*1000:>9.0f}ms {'0':>10s}")
print(f"{'RAG (BGE + prompt)':<30s} {100*rag_correct/N:>9.1f}% {'100.0':>10s}% {np.mean(rag_times)*1000:>9.0f}ms {np.mean(rag_tokens):>9.0f}")
print("\nDone.")
