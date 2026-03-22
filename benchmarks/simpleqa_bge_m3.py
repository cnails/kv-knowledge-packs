"""
SimpleQA: KV-pack with BGE-M3 routing (multilingual model).
Compare BGE-large-en-v1.5 vs BGE-M3.
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
        c.update(kv.layers[li].keys.clone(), kv.layers[li].values.clone(), li)
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

# Load data
print("Loading SimpleQA...")
with open("/root/simpleqa.csv", "r") as f:
    ds = list(csv.DictReader(f))

N = 200
random.seed(42)
indices = random.sample(range(len(ds)), N)
samples = [ds[i] for i in indices]
questions = [s["problem"] for s in samples]
answers = [s["answer"] for s in samples]
fact_sentences = [f"The answer to the question '{q}' is: {a}." for q, a in zip(questions, answers)]

# Load LLM
device = torch.device("cuda")
print("\nLoading Qwen3-8B...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device)
model.eval()

# Load BOTH embedding models
print("\nLoading BGE-large-en-v1.5...")
bge_large = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
large_fact_embs = bge_large.encode(fact_sentences, normalize_embeddings=True, show_progress_bar=True)
large_fact_embs = torch.from_numpy(large_fact_embs).float()

print("\nLoading BGE-M3...")
bge_m3 = SentenceTransformer("BAAI/bge-m3", device="cpu")
m3_fact_embs = bge_m3.encode(fact_sentences, normalize_embeddings=True, show_progress_bar=True)
m3_fact_embs = torch.from_numpy(m3_fact_embs).float()

def run_kv_test(emb_model, fact_embs, label):
    correct = 0
    route_correct = 0
    times = []

    for i in range(N):
        t0 = time.time()
        q_emb = emb_model.encode([questions[i]], normalize_embeddings=True)
        q_emb = torch.from_numpy(q_emb).float()
        cos = F.cosine_similarity(q_emb, fact_embs)
        best_idx = cos.argmax().item()

        if best_idx == i:
            route_correct += 1

        fact_text = fact_sentences[best_idx]
        fact_ids = tok.encode(fact_text, add_special_tokens=False)
        ft = torch.tensor([fact_ids], device=device)
        with torch.no_grad():
            out = model(ft, use_cache=True)
        kv = out.past_key_values
        kv_len = len(fact_ids)

        query_text = f"Answer the following question concisely.\nQuestion: {questions[i]}\nAnswer:"
        resp = gen_with_kv(model, tok, query_text, kv, kv_len, device)
        dt = time.time() - t0

        hit = check_answer(resp, answers[i])
        correct += int(hit)
        times.append(dt)

        if i < 3 or i % 50 == 0:
            m = "Y" if hit else "X"
            print(f"  [{m}] #{i}: A={answers[i][:30]} | Gen={resp[:40]}")

        del kv, out
        if (i+1) % 50 == 0:
            gc.collect(); torch.cuda.empty_cache()

    print(f"\n{label}: {correct}/{N} = {100*correct/N:.1f}%")
    print(f"  Routing: {route_correct}/{N} = {100*route_correct/N:.1f}%")
    print(f"  Latency: {np.mean(times)*1000:.0f}ms")
    return correct, route_correct

def run_rag_test(emb_model, fact_embs, label):
    correct = 0
    times = []
    tokens = []

    for i in range(N):
        t0 = time.time()
        q_emb = emb_model.encode([questions[i]], normalize_embeddings=True)
        q_emb = torch.from_numpy(q_emb).float()
        cos = F.cosine_similarity(q_emb, fact_embs)
        best_idx = cos.argmax().item()

        prompt = f"Context: {fact_sentences[best_idx]}\n\nAnswer the following question concisely.\nQuestion: {questions[i]}\nAnswer:"
        n_tok = len(tok.encode(prompt, add_special_tokens=False))
        resp = gen_plain(model, tok, prompt, device)
        dt = time.time() - t0

        hit = check_answer(resp, answers[i])
        correct += int(hit)
        times.append(dt)
        tokens.append(n_tok)

        if i < 3 or i % 50 == 0:
            m = "Y" if hit else "X"
            print(f"  [{m}] #{i}: A={answers[i][:30]} | Gen={resp[:40]}")

    print(f"\n{label}: {correct}/{N} = {100*correct/N:.1f}%")
    print(f"  Latency: {np.mean(times)*1000:.0f}ms, Tokens: {np.mean(tokens):.0f}")
    return correct

# Run tests
print("\n" + "="*60)
print("KV-PACK + BGE-large-en-v1.5")
print("="*60)
kv_large, rt_large = run_kv_test(bge_large, large_fact_embs, "KV+BGE-large")

print("\n" + "="*60)
print("KV-PACK + BGE-M3")
print("="*60)
kv_m3, rt_m3 = run_kv_test(bge_m3, m3_fact_embs, "KV+BGE-M3")

print("\n" + "="*60)
print("RAG + BGE-large-en-v1.5")
print("="*60)
rag_large = run_rag_test(bge_large, large_fact_embs, "RAG+BGE-large")

print("\n" + "="*60)
print("RAG + BGE-M3")
print("="*60)
rag_m3 = run_rag_test(bge_m3, m3_fact_embs, "RAG+BGE-M3")

# Summary
print("\n" + "="*60)
print("FINAL SUMMARY — BGE-large vs BGE-M3 (N=200)")
print("="*60)
print(f"{'Method':<30s} {'Accuracy':>10s} {'Routing':>10s}")
print("-" * 50)
print(f"{'KV + BGE-large':<30s} {100*kv_large/N:>9.1f}% {100*rt_large/N:>9.1f}%")
print(f"{'KV + BGE-M3':<30s} {100*kv_m3/N:>9.1f}% {100*rt_m3/N:>9.1f}%")
print(f"{'RAG + BGE-large':<30s} {100*rag_large/N:>9.1f}%      100%")
print(f"{'RAG + BGE-M3':<30s} {100*rag_m3/N:>9.1f}%      100%")
print("\nDone.")
