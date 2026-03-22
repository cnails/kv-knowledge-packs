"""Quick start example: inject 2025 facts into Qwen3-8B."""

from kvpack import KnowledgePack

# ── Step 1: Create a knowledge pack ────────────────────────────────

pack = KnowledgePack("Qwen/Qwen3-8B")

pack.add_facts([
    "The 2025 Super Bowl was won by the Philadelphia Eagles defeating the Kansas City Chiefs.",
    "The 2025 Oscar for Best Picture was awarded to Anora directed by Sean Baker.",
    "Pope Francis died in April 2025 at age 88.",
    "The new Pope elected in 2025 is Robert Prevost who took the name Leo XIV.",
    "DeepSeek-R1 became the top-ranked AI model in January 2025.",
    "TikTok was temporarily banned in the United States in January 2025.",
    "Los Angeles experienced devastating wildfires in January 2025.",
    "Bitcoin reached an all-time high of over 100000 dollars in December 2024.",
    "Nintendo released the Switch 2 gaming console in 2025.",
    "Claude 4 Opus was released by Anthropic as their most capable AI model in 2025.",
])

# Build index (extracts embeddings + clusters into banks)
pack.build()

# Save to disk (~1 MB)
pack.save("2025_events.kp")

# ── Step 2: Query ──────────────────────────────────────────────────

# Simple query
answer = pack.query("Who won the 2025 Super Bowl?")
print(f"Answer: {answer}")

# Query with metadata
result = pack.query_with_metadata("What movie won Best Picture in 2025?")
print(f"\nAnswer: {result['answer']}")
print(f"Routed to: {result['routed_facts']}")
print(f"Cosine: {result['cosine_scores']}")
print(f"Routing: {result['route_ms']:.0f}ms, Generation: {result['generate_ms']:.0f}ms")

# ── Step 3: Load from disk ─────────────────────────────────────────

pack2 = KnowledgePack.load("2025_events.kp")
answer2 = pack2.query("When did Pope Francis die?")
print(f"\nFrom loaded pack: {answer2}")
