"""
Exp 90: Multi-fact reasoning via KV injection.

Test: can the model reason across multiple facts loaded in KV cache?
This is something RAG can do (by putting facts in prompt) but costs prompt tokens.
KV does it at zero prompt token cost.

Levels:
1. Single-fact recall (baseline sanity check)
2. Two-fact bridge reasoning (A→B, B→C, ask A→C)
3. Three-fact chain reasoning (A→B, B→C, C→D, ask A→D)
4. Multi-fact aggregation (multiple facts about same entity)
5. Comparison across facts (which is bigger/older/etc.)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import time

device = torch.device("cuda")
print("Loading Qwen3-8B...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.float16, device_map=device)
model.eval()
print("Model loaded.")


def build_kv(facts_text):
    """Build KV cache from concatenated fact sentences."""
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


def generate(query, kv, kv_len, max_tokens=60):
    q_ids = tok.encode(query, add_special_tokens=False)
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


def test_case(name, facts, query, expected, kv=None, kv_len=None):
    """Run one test case. Build KV from facts if not provided."""
    if kv is None:
        facts_text = " ".join(facts)
        kv, kv_len = build_kv(facts_text)
    answer = generate(query, kv, kv_len)
    hit = any(e.lower() in answer.lower() for e in expected) if isinstance(expected, list) else expected.lower() in answer.lower()
    mark = "✓" if hit else "✗"
    print(f"  [{mark}] {name}")
    print(f"      Facts: {facts[:2]}{'...' if len(facts)>2 else ''}")
    print(f"      Q: {query}")
    print(f"      A: {answer[:120]}")
    print(f"      Expected: {expected}")
    return hit


# ============================================================
# LEVEL 1: Single-fact recall (sanity)
# ============================================================
print("\n" + "="*60)
print("LEVEL 1: Single-fact recall")
print("="*60)

l1_results = []
l1_results.append(test_case(
    "Simple recall",
    ["Marcus works at Nexus Corporation in Seattle."],
    "Where does Marcus work?",
    "Nexus"
))
l1_results.append(test_case(
    "Multi-token entity",
    ["The 2025 Super Bowl was won by the Philadelphia Eagles."],
    "Who won the 2025 Super Bowl?",
    "Philadelphia Eagles"
))
l1_results.append(test_case(
    "Numerical fact",
    ["The population of Elbonia is 4.7 million people."],
    "What is the population of Elbonia?",
    "4.7"
))
print(f"\nLevel 1: {sum(l1_results)}/{len(l1_results)}")


# ============================================================
# LEVEL 2: Two-fact bridge reasoning
# ============================================================
print("\n" + "="*60)
print("LEVEL 2: Two-fact bridge reasoning (A→B + B→C = A→C)")
print("="*60)

l2_results = []
l2_results.append(test_case(
    "Person → Company → City",
    ["Marcus works at Nexus Corporation.", "Nexus Corporation is headquartered in Seattle."],
    "In which city does Marcus work?",
    "Seattle"
))
l2_results.append(test_case(
    "Person → Country → Language",
    ["Elena moved to France last year.", "The official language of France is French."],
    "What language does Elena need to speak in her new country?",
    "French"
))
l2_results.append(test_case(
    "Product → Company → CEO",
    ["The iPhone was created by Apple.", "Tim Cook is the CEO of Apple."],
    "Who is the CEO of the company that makes the iPhone?",
    "Tim Cook"
))
l2_results.append(test_case(
    "City → Country → Continent",
    ["The capital of Ruritania is Frostville.", "Ruritania is located in South America."],
    "On which continent is Frostville located?",
    "South America"
))
l2_results.append(test_case(
    "Fictional bridge",
    ["Agent Vex reports to Director Holt.", "Director Holt oversees Project Chimera."],
    "Which project is Agent Vex connected to through their superior?",
    "Chimera"
))
print(f"\nLevel 2: {sum(l2_results)}/{len(l2_results)}")


# ============================================================
# LEVEL 3: Three-fact chain
# ============================================================
print("\n" + "="*60)
print("LEVEL 3: Three-fact chain (A→B→C→D)")
print("="*60)

l3_results = []
l3_results.append(test_case(
    "Person → Company → City → Country",
    [
        "Sarah works at Vertex Labs.",
        "Vertex Labs is based in Munich.",
        "Munich is the capital of Bavaria, Germany."
    ],
    "In which country does Sarah work?",
    "Germany"
))
l3_results.append(test_case(
    "Chain of command",
    [
        "Private Chen reports to Sergeant Walsh.",
        "Sergeant Walsh reports to Captain Rivera.",
        "Captain Rivera commands the 3rd Battalion."
    ],
    "Which battalion is Private Chen in?",
    "3rd"
))
l3_results.append(test_case(
    "Supply chain",
    [
        "Widget X is made from Component Y.",
        "Component Y requires Mineral Z.",
        "Mineral Z is mined in Wakanda."
    ],
    "Where is the raw material for Widget X sourced from?",
    "Wakanda"
))
print(f"\nLevel 3: {sum(l3_results)}/{len(l3_results)}")


# ============================================================
# LEVEL 4: Multi-fact aggregation (same entity)
# ============================================================
print("\n" + "="*60)
print("LEVEL 4: Multi-fact aggregation")
print("="*60)

l4_results = []

profile_facts = [
    "James is 34 years old.",
    "James works as a software engineer.",
    "James lives in Portland, Oregon.",
    "James has two children named Lily and Max.",
    "James drives a Tesla Model 3.",
]
kv, kv_len = build_kv(" ".join(profile_facts))

l4_results.append(test_case(
    "Age recall from profile",
    profile_facts, "How old is James?", "34",
    kv=kv, kv_len=kv_len
))
l4_results.append(test_case(
    "Job recall from profile",
    profile_facts, "What does James do for work?", ["software engineer", "engineer"],
    kv=kv, kv_len=kv_len
))
l4_results.append(test_case(
    "Children's names",
    profile_facts, "What are the names of James's children?", ["Lily", "Max"],
    kv=kv, kv_len=kv_len
))
l4_results.append(test_case(
    "Car + Location cross-reference",
    profile_facts, "What car does the engineer from Portland drive?", "Tesla",
    kv=kv, kv_len=kv_len
))
print(f"\nLevel 4: {sum(l4_results)}/{len(l4_results)}")


# ============================================================
# LEVEL 5: Comparison across facts
# ============================================================
print("\n" + "="*60)
print("LEVEL 5: Comparison and aggregation across entities")
print("="*60)

l5_results = []

compare_facts = [
    "Company Alpha has 500 employees.",
    "Company Beta has 1200 employees.",
    "Company Gamma has 300 employees.",
    "Company Alpha was founded in 2010.",
    "Company Beta was founded in 2005.",
    "Company Gamma was founded in 2018.",
]
kv2, kv_len2 = build_kv(" ".join(compare_facts))

l5_results.append(test_case(
    "Which is largest?",
    compare_facts, "Which company has the most employees?", "Beta",
    kv=kv2, kv_len=kv_len2
))
l5_results.append(test_case(
    "Which is oldest?",
    compare_facts, "Which company was founded first?", "Beta",
    kv=kv2, kv_len=kv_len2
))
l5_results.append(test_case(
    "Total employees",
    compare_facts, "How many total employees do all three companies have combined?", "2000",
    kv=kv2, kv_len=kv_len2
))
l5_results.append(test_case(
    "Youngest + smallest",
    compare_facts, "Which company is both the newest and the smallest?", "Gamma",
    kv=kv2, kv_len=kv_len2
))

# Multi-entity with relationships
network_facts = [
    "Alice manages the Engineering team.",
    "Bob manages the Marketing team.",
    "Carol manages the Sales team.",
    "The Engineering team has 15 people.",
    "The Marketing team has 8 people.",
    "The Sales team has 12 people.",
    "Alice and Bob collaborate on the Product Launch project.",
]
kv3, kv_len3 = build_kv(" ".join(network_facts))

l5_results.append(test_case(
    "Who manages largest team?",
    network_facts, "Who manages the largest team?", "Alice",
    kv=kv3, kv_len=kv_len3
))
l5_results.append(test_case(
    "Cross-team collaboration",
    network_facts,
    "Which teams are involved in the Product Launch project?",
    ["Engineering", "Marketing"],
    kv=kv3, kv_len=kv_len3
))
print(f"\nLevel 5: {sum(l5_results)}/{len(l5_results)}")


# ============================================================
# LEVEL 6: KV vs Baseline (no facts)
# ============================================================
print("\n" + "="*60)
print("LEVEL 6: KV vs Baseline on novel facts")
print("="*60)

novel_facts = [
    "The Zephyr Prize for Literature 2025 was awarded to novelist Kira Volkov.",
    "Volkov's winning novel is titled 'The Glass Meridian'.",
    "The ceremony was held in Reykjavik, Iceland on March 3rd 2025.",
]
kv4, kv_len4 = build_kv(" ".join(novel_facts))

# Baseline (no KV)
print("\n  --- Baseline (no KV) ---")
baseline_ids = tok.encode("Who won the Zephyr Prize for Literature 2025?", add_special_tokens=False)
bt = torch.tensor([baseline_ids], device=device)
with torch.no_grad():
    bout = model.generate(bt, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
baseline_answer = tok.decode(bout[0][len(baseline_ids):], skip_special_tokens=True).strip()
print(f"  Q: Who won the Zephyr Prize for Literature 2025?")
print(f"  Baseline: {baseline_answer[:120]}")

print("\n  --- KV injection ---")
test_case(
    "Novel fact recall",
    novel_facts, "Who won the Zephyr Prize for Literature 2025?", ["Kira Volkov", "Volkov"],
    kv=kv4, kv_len=kv_len4
)
test_case(
    "Novel fact + bridge",
    novel_facts, "What is the title of the novel that won the Zephyr Prize?", "Glass Meridian",
    kv=kv4, kv_len=kv_len4
)
test_case(
    "Novel fact + location",
    novel_facts, "In which city was the Zephyr Prize ceremony held?", "Reykjavik",
    kv=kv4, kv_len=kv_len4
)


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
all_results = {
    "L1 Single-fact": l1_results,
    "L2 Two-fact bridge": l2_results,
    "L3 Three-fact chain": l3_results,
    "L4 Multi-fact aggregation": l4_results,
    "L5 Comparison": l5_results,
}
total_correct = 0
total_tests = 0
for name, results in all_results.items():
    c = sum(results)
    t = len(results)
    total_correct += c
    total_tests += t
    print(f"  {name}: {c}/{t} = {100*c/t:.0f}%")
print(f"\n  TOTAL: {total_correct}/{total_tests} = {100*total_correct/total_tests:.0f}%")
print("\nDone.")
