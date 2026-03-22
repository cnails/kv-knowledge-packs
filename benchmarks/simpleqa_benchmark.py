"""
SimpleQA Benchmark: KV-pack vs RAG vs Prefix
Fair 3-way comparison on OpenAI's SimpleQA dataset with Llama-4-Scout-17B.

Methods:
1. Baseline: question only, no external knowledge
2. KV-pack: top-1 routing + lazy recompute, 0 prompt tokens
3. RAG: proper embedding model (BGE-large) + cosine retrieval + fact in prompt
4. Prefix: all facts concatenated in prompt (upper bound)
"""

import torch
import torch.nn.functional as F
import json
import time
import gc
import os
import random
import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ============================================================
# Helpers
# ============================================================

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


# ============================================================
# Load SimpleQA
# ============================================================

print("Loading SimpleQA dataset...")
import csv
with open("/root/simpleqa.csv", "r") as f:
    reader = csv.DictReader(f)
    ds = list(reader)
print(f"Total questions: {len(ds)}")

# Sample N questions for benchmark
N = 200
random.seed(42)
indices = random.sample(range(len(ds)), N)
samples = [ds[i] for i in indices]

# Extract questions, answers, and build fact sentences
questions = []
answers = []
fact_sentences = []

for s in samples:
    q = s["problem"]
    a = s["answer"]
    questions.append(q)
    answers.append(a)
    # Build fact sentence: transform Q+A into declarative
    # SimpleQA answers are short factual strings
    fact_sentences.append(f"The answer to the question '{q}' is: {a}.")

print(f"Sampled {N} questions")
print(f"Example Q: {questions[0][:80]}...")
print(f"Example A: {answers[0]}")
print(f"Example fact: {fact_sentences[0][:80]}...")

# ============================================================
# Load Model
# ============================================================

device = torch.device("cuda")
MODEL = "Qwen/Qwen3-8B"

print(f"\nLoading {MODEL}...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN
)
model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s")

# Detect layers
if hasattr(model, "model") and hasattr(model.model, "layers"):
    layers = model.model.layers
elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
    layers = model.transformer.h
else:
    layers = None
    print("WARNING: unknown architecture, using default layer indices")

n_layers = len(layers) if layers else 48
store_layer = int(n_layers * 0.65)
print(f"Architecture: {n_layers} layers, store_layer={store_layer}")


# ============================================================
# Extract embeddings for routing
# ============================================================

def extract_embedding(text):
    ids = tok.encode(text, add_special_tokens=False)
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(t, output_hidden_states=True)
    return out.hidden_states[store_layer + 1][0, -2].float().cpu()

print(f"\nExtracting embeddings for {N} facts...")
t0 = time.time()
fact_embeddings = []
for i, fs in enumerate(fact_sentences):
    emb = extract_embedding(fs)
    fact_embeddings.append(emb)
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{N} embeddings")
fact_emb_matrix = torch.stack(fact_embeddings)
print(f"Done in {time.time()-t0:.1f}s")


# ============================================================
# Build routing (k-means)
# ============================================================

N_BANKS = max(10, N // 20)  # ~20 facts per bank
print(f"\nClustering into {N_BANKS} banks...")
km = KMeans(n_clusters=N_BANKS, random_state=42, n_init=10)
labels = km.fit_predict(fact_emb_matrix.numpy())
centroids = torch.from_numpy(km.cluster_centers_).float()

banks = defaultdict(list)
for i, lab in enumerate(labels):
    banks[lab].append(i)

# Also extract query embeddings for routing
print(f"Extracting query embeddings...")
query_embeddings = []
for i, q in enumerate(questions):
    emb = extract_embedding(q)
    query_embeddings.append(emb)
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{N} query embeddings")
query_emb_matrix = torch.stack(query_embeddings)


# ============================================================
# Load RAG embedding model
# ============================================================

print("\nLoading BGE-large for RAG...")
rag_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
rag_fact_embs = rag_model.encode(fact_sentences, normalize_embeddings=True,
                                  show_progress_bar=True)
rag_fact_embs = torch.from_numpy(rag_fact_embs).float()
print("RAG embeddings ready")


# ============================================================
# Evaluation helper
# ============================================================

def check_answer(generated, gold):
    """Check if gold answer appears in generated text (case-insensitive)."""
    return gold.lower() in generated.lower()


# ============================================================
# Method 1: Baseline (no knowledge)
# ============================================================

print("\n" + "="*60)
print("METHOD 1: BASELINE (no external knowledge)")
print("="*60)

baseline_correct = 0
baseline_times = []
baseline_tokens = []

for i in range(N):
    prompt = f"Answer the following question concisely.\nQuestion: {questions[i]}\nAnswer:"
    n_tok = len(tok.encode(prompt, add_special_tokens=False))

    t0 = time.time()
    resp = gen_plain(model, tok, prompt, device)
    dt = time.time() - t0

    hit = check_answer(resp, answers[i])
    baseline_correct += int(hit)
    baseline_times.append(dt)
    baseline_tokens.append(n_tok)

    if i < 3 or (i % 50 == 0):
        m = "Y" if hit else "X"
        print(f"  [{m}] #{i}: A={answers[i][:30]} | Gen={resp[:50]}")

print(f"\nBaseline: {baseline_correct}/{N} = {100*baseline_correct/N:.1f}%")
print(f"  Avg latency: {np.mean(baseline_times)*1000:.0f}ms")
print(f"  Avg prompt tokens: {np.mean(baseline_tokens):.0f}")


# ============================================================
# Method 2: KV-pack (top-1 routing)
# ============================================================

print("\n" + "="*60)
print("METHOD 2: KV-PACK (top-1 routing, 0 prompt tokens)")
print("="*60)

kv_correct = 0
kv_times = []
kv_routing_correct = 0

for i in range(N):
    t0 = time.time()

    # Route: cosine with centroids
    q_emb = query_embeddings[i]
    cos_scores = F.cosine_similarity(q_emb.unsqueeze(0), centroids)
    bank_id = cos_scores.argmax().item()
    bank_facts = banks[bank_id]

    # Intra-bank ranking (top-1)
    bank_embs = torch.stack([fact_embeddings[fi] for fi in bank_facts])
    intra_cos = F.cosine_similarity(q_emb.unsqueeze(0), bank_embs)
    best_in_bank = intra_cos.argmax().item()
    selected_fact_id = bank_facts[best_in_bank]

    # Check routing
    if selected_fact_id == i:
        kv_routing_correct += 1

    # Recompute KV for selected fact
    fact_text = fact_sentences[selected_fact_id]
    fact_ids = tok.encode(fact_text, add_special_tokens=False)
    ft = torch.tensor([fact_ids], device=device)
    with torch.no_grad():
        out = model(ft, use_cache=True)
    kv = out.past_key_values
    kv_len = len(fact_ids)

    # Generate
    query_text = f"Answer the following question concisely.\nQuestion: {questions[i]}\nAnswer:"
    resp = gen_with_kv(model, tok, query_text, kv, kv_len, device)
    dt = time.time() - t0

    hit = check_answer(resp, answers[i])
    kv_correct += int(hit)
    kv_times.append(dt)

    if i < 3 or (i % 50 == 0):
        routed = "correct" if selected_fact_id == i else f"WRONG(got {selected_fact_id})"
        m = "Y" if hit else "X"
        print(f"  [{m}] #{i}: route={routed} | A={answers[i][:30]} | Gen={resp[:50]}")

    del kv, out
    if (i+1) % 50 == 0:
        gc.collect()
        torch.cuda.empty_cache()

print(f"\nKV-pack: {kv_correct}/{N} = {100*kv_correct/N:.1f}%")
print(f"  Routing accuracy: {kv_routing_correct}/{N} = {100*kv_routing_correct/N:.1f}%")
print(f"  Avg latency: {np.mean(kv_times)*1000:.0f}ms")
print(f"  Prompt tokens: 0 (facts in KV)")


# ============================================================
# Method 3: RAG (BGE-large retrieval + fact in prompt)
# ============================================================

print("\n" + "="*60)
print("METHOD 3: RAG (BGE-large retrieval, top-1 fact in prompt)")
print("="*60)

rag_correct = 0
rag_times = []
rag_tokens = []
rag_routing_correct = 0

for i in range(N):
    t0 = time.time()

    # Embed query with BGE
    q_text = questions[i]
    q_emb_rag = rag_model.encode([q_text], normalize_embeddings=True)
    q_emb_rag = torch.from_numpy(q_emb_rag).float()

    # Retrieve top-1
    cos_rag = F.cosine_similarity(q_emb_rag, rag_fact_embs)
    best_idx = cos_rag.argmax().item()

    if best_idx == i:
        rag_routing_correct += 1

    # Build prompt with retrieved fact
    retrieved_fact = fact_sentences[best_idx]
    prompt = f"Context: {retrieved_fact}\n\nAnswer the following question concisely.\nQuestion: {q_text}\nAnswer:"
    n_tok = len(tok.encode(prompt, add_special_tokens=False))

    resp = gen_plain(model, tok, prompt, device)
    dt = time.time() - t0

    hit = check_answer(resp, answers[i])
    rag_correct += int(hit)
    rag_times.append(dt)
    rag_tokens.append(n_tok)

    if i < 3 or (i % 50 == 0):
        routed = "correct" if best_idx == i else f"WRONG(got {best_idx})"
        m = "Y" if hit else "X"
        print(f"  [{m}] #{i}: route={routed} | A={answers[i][:30]} | Gen={resp[:50]}")

print(f"\nRAG: {rag_correct}/{N} = {100*rag_correct/N:.1f}%")
print(f"  Routing accuracy: {rag_routing_correct}/{N} = {100*rag_routing_correct/N:.1f}%")
print(f"  Avg latency: {np.mean(rag_times)*1000:.0f}ms")
print(f"  Avg prompt tokens: {np.mean(rag_tokens):.0f}")


# ============================================================
# Method 4: Prefix (all facts in prompt — upper bound)
# ============================================================

print("\n" + "="*60)
print("METHOD 4: PREFIX (all facts in prompt — upper bound)")
print("="*60)

# For prefix, we can't fit all 200 facts in prompt easily.
# Use a subset or test with smaller N.
# Build prefix with all facts
all_facts_text = "\n".join(fact_sentences)
prefix_tok_count = len(tok.encode(all_facts_text, add_special_tokens=False))
print(f"All facts: {prefix_tok_count} tokens")

# If too many tokens, sample fewer
MAX_PREFIX_TOK = 8000
if prefix_tok_count > MAX_PREFIX_TOK:
    # Test prefix only on first 30 questions with their corresponding facts nearby
    prefix_n = 30
    prefix_facts = "\n".join(fact_sentences[:prefix_n])
    prefix_tok_count = len(tok.encode(prefix_facts, add_special_tokens=False))
    print(f"Using first {prefix_n} facts ({prefix_tok_count} tok) for prefix test")
else:
    prefix_n = N
    prefix_facts = all_facts_text

prefix_correct = 0
prefix_times = []
prefix_tokens = []

for i in range(min(prefix_n, N)):
    prompt = f"Here are some facts:\n{prefix_facts}\n\nAnswer the following question concisely.\nQuestion: {questions[i]}\nAnswer:"
    n_tok = len(tok.encode(prompt, add_special_tokens=False))

    t0 = time.time()
    resp = gen_plain(model, tok, prompt, device)
    dt = time.time() - t0

    hit = check_answer(resp, answers[i])
    prefix_correct += int(hit)
    prefix_times.append(dt)
    prefix_tokens.append(n_tok)

    if i < 3 or (i % 10 == 0):
        m = "Y" if hit else "X"
        print(f"  [{m}] #{i}: A={answers[i][:30]} | Gen={resp[:50]}")

prefix_total = min(prefix_n, N)
print(f"\nPrefix: {prefix_correct}/{prefix_total} = {100*prefix_correct/prefix_total:.1f}%")
print(f"  Avg latency: {np.mean(prefix_times)*1000:.0f}ms")
print(f"  Avg prompt tokens: {np.mean(prefix_tokens):.0f}")


# ============================================================
# Final Summary
# ============================================================

print("\n" + "="*60)
print("FINAL SUMMARY — SimpleQA Benchmark")
print("="*60)
print(f"Model: {MODEL}")
print(f"Questions: {N}")
print(f"Facts: {N}")
print()
print(f"{'Method':<25s} {'Accuracy':>10s} {'Routing':>10s} {'Latency':>10s} {'Tokens':>10s}")
print("-" * 65)
print(f"{'Baseline':<25s} {100*baseline_correct/N:>9.1f}% {'N/A':>10s} {np.mean(baseline_times)*1000:>9.0f}ms {np.mean(baseline_tokens):>9.0f}")
print(f"{'KV-pack (top-1)':<25s} {100*kv_correct/N:>9.1f}% {100*kv_routing_correct/N:>9.1f}% {np.mean(kv_times)*1000:>9.0f}ms {'0':>10s}")
print(f"{'RAG (BGE-large)':<25s} {100*rag_correct/N:>9.1f}% {100*rag_routing_correct/N:>9.1f}% {np.mean(rag_times)*1000:>9.0f}ms {np.mean(rag_tokens):>9.0f}")
print(f"{'Prefix (all in prompt)':<25s} {100*prefix_correct/prefix_total:>9.1f}% {'100%':>10s} {np.mean(prefix_times)*1000:>9.0f}ms {np.mean(prefix_tokens):>9.0f}")
print()
print("Done.")
