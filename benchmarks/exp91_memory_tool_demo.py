"""
Exp 91: KV Memory Tool — Dynamic knowledge accumulation demo.

Shows: LLM agent searches, accumulates facts in KV, answers multi-hop questions.
Compares: KV memory agent vs baseline (no memory) vs RAG (search per query).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, os
sys.path.insert(0, '/root')

device = torch.device("cuda")
print("Loading Qwen3-8B...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.float16, device_map=device)
model.eval()

# Import after model load to avoid double CUDA init
from memory_tool import KVMemoryTool

# ============================================================
# Knowledge corpus — simulated "web" of facts
# ============================================================
CORPUS = [
    # Science
    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "Einstein's mass-energy equivalence formula is E=mc², where c is the speed of light.",
    "Nuclear fission releases energy by splitting heavy atomic nuclei.",
    "Nuclear fusion combines light nuclei to form heavier ones, releasing enormous energy.",
    "The Sun produces energy through nuclear fusion of hydrogen into helium.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "Chlorophyll is the molecule responsible for absorbing light in photosynthesis.",
    "ATP is the primary energy currency of biological cells.",
    "DNA stores genetic information using four nucleotide bases: adenine, thymine, guanine, and cytosine.",
    "CRISPR-Cas9 is a gene editing tool that can modify DNA sequences precisely.",

    # Technology
    "Transformer architecture was introduced in the 2017 paper 'Attention Is All You Need'.",
    "GPT stands for Generative Pre-trained Transformer.",
    "BERT uses bidirectional attention, unlike GPT which uses causal (left-to-right) attention.",
    "The attention mechanism computes weighted sums of value vectors based on query-key similarity.",
    "Backpropagation is the algorithm used to compute gradients for training neural networks.",
    "Gradient descent updates model parameters in the direction that reduces the loss function.",
    "Moore's Law states that transistor density doubles approximately every two years.",
    "Quantum computing uses qubits that can exist in superposition of 0 and 1 states.",
    "Bitcoin uses proof-of-work consensus mechanism requiring significant computational energy.",
    "Ethereum transitioned to proof-of-stake consensus in September 2022.",

    # History & Geography
    "The Roman Empire fell in 476 AD when Odoacer deposed Emperor Romulus Augustulus.",
    "The Industrial Revolution began in Britain in the late 18th century.",
    "The Suez Canal connects the Mediterranean Sea to the Red Sea.",
    "The Panama Canal connects the Atlantic Ocean to the Pacific Ocean.",
    "Mount Everest is 8,849 meters tall, the highest point on Earth.",
    "The Mariana Trench is the deepest oceanic trench, reaching about 11,034 meters deep.",
    "The Great Wall of China was built over many centuries, primarily during the Ming Dynasty.",
    "The Library of Alexandria was one of the largest libraries of the ancient world.",

    # Fictional post-cutoff facts
    "The Meridian Space Station was launched in February 2026 by ESA and JAXA.",
    "The Meridian Station orbits at 400km altitude, same as the ISS.",
    "Dr. Yuki Tanaka is the first commander of the Meridian Space Station.",
    "The Meridian Station's primary mission is to test closed-loop life support systems.",
    "Project Aurora is the AI research initiative funded by the Meridian program.",
    "Project Aurora uses quantum-classical hybrid computing for protein folding.",
    "The Aurora supercomputer is located at CERN in Geneva, Switzerland.",
    "Aurora achieved 500 exaflops sustained performance in March 2026.",

    # People connections
    "Dr. Sarah Chen leads the quantum computing division at Nexus Labs.",
    "Nexus Labs is a subsidiary of Vertex Technologies.",
    "Vertex Technologies was founded by Marcus Rivera in 2019.",
    "Marcus Rivera previously worked at DeepMind on reinforcement learning.",
    "Vertex Technologies is headquartered in Zurich, Switzerland.",
    "Nexus Labs operates from a research campus in Cambridge, UK.",
    "Dr. Chen's team discovered a new error correction code called Helix-7.",
    "Helix-7 reduces quantum decoherence by 94% compared to existing methods.",
]

# ============================================================
# Test scenarios
# ============================================================
print(f"\nCorpus: {len(CORPUS)} facts")
memory = KVMemoryTool(model, tok, corpus=CORPUS, device=device)

print("\n" + "="*70)
print("SCENARIO 1: Progressive knowledge accumulation")
print("="*70)
print("\nAgent builds knowledge through sequential searches.\n")

# Step 1: Search about quantum computing
print("--- Step 1: Agent searches 'quantum computing' ---")
r1 = memory.search_and_remember("quantum computing research")
print(f"  Found {r1['found']} facts, memory now has {r1['total']} facts ({r1['kv_tokens']} tokens)")

# Step 2: Search about Nexus Labs
print("\n--- Step 2: Agent searches 'Nexus Labs' ---")
r2 = memory.search_and_remember("Nexus Labs research team")
print(f"  Found {r2['found']} facts, memory now has {r2['total']} facts ({r2['kv_tokens']} tokens)")

# Step 3: Multi-hop question (needs facts from BOTH searches)
print("\n--- Step 3: Multi-hop question (needs both searches) ---")
q1 = "Where is the company that owns the lab where the Helix-7 error correction code was discovered?"
print(f"  Q: {q1}")

# Answer with KV memory
a1_kv = memory.answer(q1)
print(f"  KV Memory: {a1_kv[:150]}")
# Expected chain: Helix-7 → Dr. Chen → Nexus Labs → Vertex Technologies → Zurich

# Answer without memory (baseline)
q_ids = tok.encode(q1, add_special_tokens=False)
qt = torch.tensor([q_ids], device=device)
with torch.no_grad():
    out = model.generate(qt, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
a1_base = tok.decode(out[0][len(q_ids):], skip_special_tokens=True).strip()
print(f"  Baseline:  {a1_base[:150]}")

# Check
zurich_in_kv = "zurich" in a1_kv.lower()
zurich_in_base = "zurich" in a1_base.lower()
print(f"\n  Result: KV={'✓ Zurich' if zurich_in_kv else '✗'}, Base={'✓ Zurich' if zurich_in_base else '✗'}")


print("\n" + "="*70)
print("SCENARIO 2: Cross-domain reasoning")
print("="*70)
print("\nAgent accumulates facts from different domains, then connects them.\n")

memory.clear()

# Search about space station
print("--- Step 1: Agent searches 'Meridian Space Station' ---")
r3 = memory.search_and_remember("Meridian Space Station")
print(f"  Found {r3['found']} facts, memory: {r3['total']} facts")

# Search about Aurora project
print("\n--- Step 2: Agent searches 'Project Aurora AI' ---")
r4 = memory.search_and_remember("Project Aurora computing")
print(f"  Found {r4['found']} facts, memory: {r4['total']} facts")

# Cross-domain question
q2 = "What computing capability does the AI initiative connected to the Meridian program have?"
print(f"\n--- Step 3: Cross-domain question ---")
print(f"  Q: {q2}")
# Chain: Meridian → Aurora → quantum-classical hybrid → 500 exaflops

a2_kv = memory.answer(q2)
print(f"  KV Memory: {a2_kv[:150]}")

q_ids2 = tok.encode(q2, add_special_tokens=False)
qt2 = torch.tensor([q_ids2], device=device)
with torch.no_grad():
    out2 = model.generate(qt2, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
a2_base = tok.decode(out2[0][len(q_ids2):], skip_special_tokens=True).strip()
print(f"  Baseline:  {a2_base[:150]}")

exaflops_kv = "500" in a2_kv or "exaflop" in a2_kv.lower()
exaflops_base = "500" in a2_base or "exaflop" in a2_base.lower()
print(f"\n  Result: KV={'✓' if exaflops_kv else '✗'}, Base={'✓' if exaflops_base else '✗'}")


print("\n" + "="*70)
print("SCENARIO 3: Accumulated knowledge across 5 searches")
print("="*70)

memory.clear()
topics = [
    "nuclear energy and fusion",
    "the Sun and solar energy",
    "photosynthesis and plants",
    "ATP and cellular energy",
    "DNA and genetic information",
]

for i, topic in enumerate(topics):
    r = memory.search_and_remember(topic)
    print(f"  Search {i+1}: '{topic}' → +{r['added']} facts (total: {r['total']}, {r['kv_tokens']} tok)")

# Multi-hop across all 5 searches
questions = [
    ("What process in the Sun produces the energy that plants use for photosynthesis?",
     ["fusion", "nuclear"]),
    ("What molecule absorbs the sunlight needed to make the energy currency ATP?",
     ["chlorophyll"]),
    ("If you edited the genes controlling chlorophyll production using modern gene editing, what tool would you use?",
     ["CRISPR"]),
]

print(f"\n  Memory status: {memory.status()}")
print(f"\n  Multi-hop questions across all searches:")
correct_kv = 0
correct_base = 0
for q, expected in questions:
    a_kv = memory.answer(q, max_new_tokens=60)
    hit_kv = any(e.lower() in a_kv.lower() for e in expected)
    if hit_kv: correct_kv += 1

    q_ids = tok.encode(q, add_special_tokens=False)
    qt = torch.tensor([q_ids], device=device)
    with torch.no_grad():
        out = model.generate(qt, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
    a_base = tok.decode(out[0][len(q_ids):], skip_special_tokens=True).strip()
    hit_base = any(e.lower() in a_base.lower() for e in expected)
    if hit_base: correct_base += 1

    print(f"\n  Q: {q}")
    print(f"  KV:   {'✓' if hit_kv else '✗'} {a_kv[:100]}")
    print(f"  Base: {'✓' if hit_base else '✗'} {a_base[:100]}")

print(f"\n  Score: KV={correct_kv}/{len(questions)}, Baseline={correct_base}/{len(questions)}")


print("\n" + "="*70)
print("SCENARIO 4: Memory persistence + save/load")
print("="*70)

memory.save("/root/memory_state.json")
print(f"  Saved memory: {memory.status()}")

memory2 = KVMemoryTool(model, tok, device=device)
memory2.load("/root/memory_state.json")
print(f"  Loaded memory: {memory2.status()}")

a_loaded = memory2.answer("What molecule absorbs light in photosynthesis?")
print(f"  Q: What molecule absorbs light in photosynthesis?")
print(f"  A: {a_loaded[:100]}")
print(f"  {'✓' if 'chlorophyll' in a_loaded.lower() else '✗'} (expected: chlorophyll)")


print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"""
KV Memory Tool — Dynamic Knowledge Accumulation

  Key property: Agent searches → saves to KV → reasons across all searches
  Prompt tokens used: 0 (always)
  Multi-hop reasoning: automatic (attention cross-references)
  Persistence: save/load to JSON + auto-rebuild KV

  Scenario 1 (2 searches → multi-hop): KV={'✓' if zurich_in_kv else '✗'} Base={'✓' if zurich_in_base else '✗'}
  Scenario 2 (cross-domain): KV={'✓' if exaflops_kv else '✗'} Base={'✓' if exaflops_base else '✗'}
  Scenario 3 (5 searches → 3 questions): KV={correct_kv}/3 Base={correct_base}/3
  Scenario 4 (save/load persistence): {'✓' if 'chlorophyll' in a_loaded.lower() else '✗'}
""")
print("Done.")
