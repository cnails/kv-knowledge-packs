"""
Exp 90b: KV vs RAG on multi-fact reasoning.

Compare: KV (all facts in cache) vs RAG (top-1, top-3 retrieval) vs Prefix (all facts in prompt).
Focus: does RAG retrieve ALL needed facts for multi-hop questions?
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from sentence_transformers import SentenceTransformer
import time

device = torch.device("cuda")
print("Loading Qwen3-8B...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.float16, device_map=device)
model.eval()

print("Loading BGE-large...")
bge = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
print("Models loaded.")


def build_kv(facts_text):
    ids = tok.encode(facts_text, add_special_tokens=False)
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(t, use_cache=True)
    return out.past_key_values, len(ids)


def clone_kv(kv):
    c = DynamicCache()
    for li in range(len(kv)):
        k = kv.layers[li].keys.clone()
        v = kv.layers[li].values.clone()
        c.update(k, v, li)
    return c


def gen_kv(query, kv, kv_len, max_tokens=60):
    q_ids = tok.encode(query, add_special_tokens=False)
    qt = torch.tensor([q_ids], device=device)
    am = torch.ones(1, kv_len + len(q_ids), device=device, dtype=torch.long)
    kv_c = clone_kv(kv)
    with torch.no_grad():
        out = model.generate(qt, past_key_values=kv_c, attention_mask=am,
                             max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(q_ids):], skip_special_tokens=True).strip()


def gen_prefix(query, facts_text, max_tokens=60):
    """RAG/Prefix: put facts in prompt as text."""
    prompt = f"Facts:\n{facts_text}\n\nQuestion: {query}\nAnswer:"
    ids = tok.encode(prompt, add_special_tokens=False)
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(t, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(ids):], skip_special_tokens=True).strip()


def gen_baseline(query, max_tokens=60):
    ids = tok.encode(query, add_special_tokens=False)
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(t, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(ids):], skip_special_tokens=True).strip()


def rag_retrieve(query, facts, top_k=1):
    """Retrieve top-k facts using BGE similarity."""
    q_emb = bge.encode([query], normalize_embeddings=True)
    f_embs = bge.encode(facts, normalize_embeddings=True)
    scores = (q_emb @ f_embs.T)[0]
    topk_idx = scores.argsort()[::-1][:top_k]
    return [facts[i] for i in topk_idx], scores[topk_idx].tolist()


# ============================================================
# Test cases
# ============================================================
TEST_CASES = [
    # Level 2: Two-fact bridge
    {
        "name": "Person→Company→City",
        "facts": [
            "Marcus works at Nexus Corporation.",
            "Nexus Corporation is headquartered in Seattle.",
        ],
        "query": "In which city does Marcus work?",
        "expected": "Seattle",
    },
    {
        "name": "Person→Country→Language",
        "facts": [
            "Elena moved to France last year.",
            "The official language of France is French.",
        ],
        "query": "What language does Elena need to speak in her new country?",
        "expected": "French",
    },
    {
        "name": "Product→Company→CEO",
        "facts": [
            "The iPhone was created by Apple.",
            "Tim Cook is the CEO of Apple.",
        ],
        "query": "Who is the CEO of the company that makes the iPhone?",
        "expected": "Tim Cook",
    },
    {
        "name": "City→Country→Continent",
        "facts": [
            "The capital of Ruritania is Frostville.",
            "Ruritania is located in South America.",
        ],
        "query": "On which continent is Frostville located?",
        "expected": "South America",
    },
    {
        "name": "Fictional bridge",
        "facts": [
            "Agent Vex reports to Director Holt.",
            "Director Holt oversees Project Chimera.",
        ],
        "query": "Which project is Agent Vex connected to through their superior?",
        "expected": "Chimera",
    },
    # Level 3: Three-fact chain
    {
        "name": "Person→Company→City→Country",
        "facts": [
            "Sarah works at Vertex Labs.",
            "Vertex Labs is based in Munich.",
            "Munich is the capital of Bavaria, Germany.",
        ],
        "query": "In which country does Sarah work?",
        "expected": "Germany",
    },
    {
        "name": "Chain of command",
        "facts": [
            "Private Chen reports to Sergeant Walsh.",
            "Sergeant Walsh reports to Captain Rivera.",
            "Captain Rivera commands the 3rd Battalion.",
        ],
        "query": "Which battalion is Private Chen in?",
        "expected": "3rd",
    },
    {
        "name": "Supply chain",
        "facts": [
            "Widget X is made from Component Y.",
            "Component Y requires Mineral Z.",
            "Mineral Z is mined in Wakanda.",
        ],
        "query": "Where is the raw material for Widget X sourced from?",
        "expected": "Wakanda",
    },
    # Level 4: Multi-fact aggregation
    {
        "name": "Cross-ref from profile",
        "facts": [
            "James is 34 years old.",
            "James works as a software engineer.",
            "James lives in Portland, Oregon.",
            "James has two children named Lily and Max.",
            "James drives a Tesla Model 3.",
        ],
        "query": "What car does the engineer from Portland drive?",
        "expected": "Tesla",
    },
    # Level 5: Comparison
    {
        "name": "Which is oldest?",
        "facts": [
            "Company Alpha was founded in 2010.",
            "Company Beta was founded in 2005.",
            "Company Gamma was founded in 2018.",
        ],
        "query": "Which company was founded first?",
        "expected": "Beta",
    },
    # Novel fictional facts
    {
        "name": "Novel bridge",
        "facts": [
            "The Zephyr Prize for Literature 2025 was awarded to novelist Kira Volkov.",
            "Volkov's winning novel is titled 'The Glass Meridian'.",
            "The ceremony was held in Reykjavik, Iceland on March 3rd 2025.",
        ],
        "query": "What is the title of the novel that won the Zephyr Prize?",
        "expected": "Glass Meridian",
    },
    {
        "name": "Novel chain",
        "facts": [
            "Dr. Elara Voss discovered Element 127, called Novaium.",
            "Novaium has unique superconducting properties at room temperature.",
            "Room temperature superconductors could revolutionize energy transmission.",
        ],
        "query": "What practical application could Dr. Voss's discovery lead to?",
        "expected": ["energy", "superconductor"],
    },
]

# ============================================================
# Run all methods
# ============================================================
results = {"kv": [], "rag1": [], "rag3": [], "prefix": [], "baseline": []}

for tc in TEST_CASES:
    print(f"\n{'='*60}")
    print(f"Test: {tc['name']}")
    print(f"Facts: {tc['facts']}")
    print(f"Query: {tc['query']}")
    print(f"Expected: {tc['expected']}")

    expected = tc["expected"]
    check = lambda ans: (any(e.lower() in ans.lower() for e in expected)
                         if isinstance(expected, list)
                         else expected.lower() in ans.lower())

    # KV injection
    facts_text = " ".join(tc["facts"])
    kv, kv_len = build_kv(facts_text)
    ans_kv = gen_kv(tc["query"], kv, kv_len)
    hit_kv = check(ans_kv)
    results["kv"].append(hit_kv)
    print(f"  KV:       {'✓' if hit_kv else '✗'} {ans_kv[:80]}")

    # RAG top-1
    retrieved_1, scores_1 = rag_retrieve(tc["query"], tc["facts"], top_k=1)
    ans_rag1 = gen_prefix(tc["query"], "\n".join(retrieved_1))
    hit_rag1 = check(ans_rag1)
    results["rag1"].append(hit_rag1)
    print(f"  RAG-1:    {'✓' if hit_rag1 else '✗'} retrieved={retrieved_1[0][:60]}... → {ans_rag1[:60]}")

    # RAG top-3
    retrieved_3, scores_3 = rag_retrieve(tc["query"], tc["facts"], top_k=3)
    ans_rag3 = gen_prefix(tc["query"], "\n".join(retrieved_3))
    hit_rag3 = check(ans_rag3)
    results["rag3"].append(hit_rag3)
    print(f"  RAG-3:    {'✓' if hit_rag3 else '✗'} retrieved {len(retrieved_3)} facts → {ans_rag3[:60]}")

    # Prefix (all facts in prompt — upper bound)
    ans_prefix = gen_prefix(tc["query"], "\n".join(tc["facts"]))
    hit_prefix = check(ans_prefix)
    results["prefix"].append(hit_prefix)
    print(f"  Prefix:   {'✓' if hit_prefix else '✗'} {ans_prefix[:80]}")

    # Baseline (no facts)
    ans_base = gen_baseline(tc["query"])
    hit_base = check(ans_base)
    results["baseline"].append(hit_base)
    print(f"  Baseline: {'✓' if hit_base else '✗'} {ans_base[:80]}")

    del kv

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
n = len(TEST_CASES)
for method, hits in results.items():
    c = sum(hits)
    print(f"  {method:>10s}: {c}/{n} = {100*c/n:.0f}%")

print(f"\nPrompt tokens used:")
print(f"  KV:       0")
print(f"  RAG-1:    ~20 (1 fact)")
print(f"  RAG-3:    ~60 (3 facts)")
print(f"  Prefix:   ~40-100 (all facts)")

# Per-test breakdown
print(f"\n{'Test':<30s} {'KV':>4s} {'RAG1':>5s} {'RAG3':>5s} {'Pfix':>5s} {'Base':>5s}")
print("-" * 55)
for i, tc in enumerate(TEST_CASES):
    kv_m = "✓" if results["kv"][i] else "✗"
    r1_m = "✓" if results["rag1"][i] else "✗"
    r3_m = "✓" if results["rag3"][i] else "✗"
    pf_m = "✓" if results["prefix"][i] else "✗"
    bs_m = "✓" if results["baseline"][i] else "✗"
    print(f"  {tc['name']:<28s} {kv_m:>4s} {r1_m:>5s} {r3_m:>5s} {pf_m:>5s} {bs_m:>5s}")

print("\nDone.")
