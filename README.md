# KV Knowledge Packs

Zero-token knowledge delivery for LLMs via KV cache injection.

## What it does

```python
from kvpack import KnowledgePack

# Create a knowledge pack with post-cutoff facts
pack = KnowledgePack("Qwen/Qwen3-8B")
pack.add_facts([
    "The 2025 Super Bowl was won by the Philadelphia Eagles.",
    "Pope Francis died in April 2025 at age 88.",
])
pack.save("2025.kp")

# Query — model "remembers" facts it was never trained on
answer = pack.query("Who won the 2025 Super Bowl?")
# → "The 2025 Super Bowl was won by the Philadelphia Eagles."
```

## How it works

KV Knowledge Packs pre-compute the model's key-value cache for fact sentences and inject them at query time. The model processes your question as if it had already read the facts — but uses zero prompt tokens.

KV cache injection is **mathematically equivalent** to prepending text to the prompt (a direct consequence of causal masking). We verify this empirically: zero divergences across 700 questions on two model families (Qwen3-8B, Llama-3.1-8B).

1. **Build** (offline, once): facts formatted with chat template → model forward pass → extract KV cache → k-means clustering → save centroids + fact texts to disk
2. **Query** (per request): question → cosine routing to best fact → recompute KV for that fact (~5ms) → generate with KV → answer

**Critical detail:** KV caches must be built using the model's chat template. Raw text KV (without special tokens) causes 6-7pp accuracy degradation on instruction-tuned models.

## Key results

| Metric | Value |
|--------|-------|
| KV vs RAG accuracy | **Identical** (0 divergences / 700 questions) |
| Token savings (5-step accumulation) | **95%** (704 tokens/query) |
| Routing accuracy | **100%** (up to 5,000 facts) |
| Storage | **4 MB** per 5,000 facts |
| Overhead per query | **~6ms** (routing + KV recompute) |
| Prompt tokens used for knowledge | **0** |

## Comparison with alternatives

| | KV Packs | RAG | Fine-tuning |
|--|----------|-----|-------------|
| Accuracy | Same as RAG | Baseline | Higher |
| Setup | `add_facts()` | Vector DB + embeddings | Training loop |
| Latency overhead | ~6ms | 50-200ms | 0ms |
| Knowledge tokens | 0 | 100-2000 | 0 |
| Reversible | Yes | Yes | No |
| Model-agnostic | No (same model) | Yes | No |
| Scales with steps | O(1) tokens | O(n) tokens | N/A |

## Install

```bash
pip install kvpack
```

Or from source:
```bash
git clone https://github.com/cnails/kv-knowledge-packs
cd kv-knowledge-packs
pip install -e .
```

## Examples

See [`examples/`](examples/) for complete examples:
- [`quickstart.py`](examples/quickstart.py) — basic usage

## Limitations

- **Same-model only.** KV cache format is model-specific — cannot transfer across architectures.
- **Chat template required.** KV must be built with the model's chat template (6-7pp degradation otherwise).
- **Tested on Qwen3-8B and Llama-3.1-8B.** Should work with any HuggingFace causal LM.

## Paper

[Knowledge Packs: Zero-Token Knowledge Delivery via KV Cache Injection](paper/main.tex)

## License

Apache 2.0
