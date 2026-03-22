"""
KV Memory Tool — Dynamic knowledge accumulation for LLM agents.

The LLM agent can:
  1. search(query) — find relevant facts from a knowledge corpus
  2. remember(facts) — save facts to KV memory (0 prompt tokens)
  3. answer(question) — answer using accumulated KV knowledge
  4. Memory persists across queries and supports multi-hop reasoning.

Usage:
    memory = KVMemoryTool(model, tokenizer, corpus=facts_list)

    # Agent loop
    memory.search_and_remember("thermodynamics laws")
    memory.search_and_remember("heat transfer mechanisms")
    answer = memory.answer("How does the second law of thermodynamics relate to heat transfer?")
    # → Multi-hop reasoning across facts from both searches
"""
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import time


class KVMemoryTool:
    """Dynamic KV-based memory for LLM agents."""

    def __init__(self, model, tokenizer, corpus=None,
                 embedding_model="BAAI/bge-large-en-v1.5",
                 device=None):
        self.model = model
        self.tok = tokenizer
        self.device = device or next(model.parameters()).device

        # Embedding model for search
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(embedding_model, device=str(self.device))

        # Knowledge corpus (searchable)
        self.corpus = corpus or []
        self.corpus_embeddings = None
        if self.corpus:
            self._index_corpus()

        # KV memory state
        self.kv_cache = None
        self.kv_len = 0
        self.remembered_facts = []

        # Stats
        self.search_count = 0
        self.facts_accumulated = 0
        self.queries_answered = 0

    def _index_corpus(self):
        """Build embedding index for the corpus."""
        print(f"Indexing {len(self.corpus)} corpus facts...")
        self.corpus_embeddings = self.embedder.encode(
            self.corpus, normalize_embeddings=True, show_progress_bar=False
        )
        print("Corpus indexed.")

    def add_to_corpus(self, facts):
        """Add new facts to the searchable corpus."""
        self.corpus.extend(facts)
        self._index_corpus()

    def search(self, query, top_k=5):
        """Search corpus for relevant facts. Returns list of (fact, score)."""
        if not self.corpus:
            return []

        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        scores = (q_emb @ self.corpus_embeddings.T)[0]
        topk_idx = scores.argsort()[::-1][:top_k]

        results = [(self.corpus[i], float(scores[i])) for i in topk_idx if scores[i] > 0.3]
        self.search_count += 1
        return results

    def remember(self, facts):
        """Add facts to KV memory. Facts persist across queries.

        Args:
            facts: list of fact strings, or a single string
        """
        if isinstance(facts, str):
            facts = [facts]

        # Filter duplicates
        new_facts = [f for f in facts if f not in self.remembered_facts]
        if not new_facts:
            return {"added": 0, "total": len(self.remembered_facts), "kv_tokens": self.kv_len}

        # Build combined text: existing + new
        self.remembered_facts.extend(new_facts)
        all_text = " ".join(self.remembered_facts)

        # Rebuild KV from all facts (ensures positional coherence)
        input_ids = self.tok.encode(all_text, add_special_tokens=False)
        t = torch.tensor([input_ids], device=self.device)
        with torch.no_grad():
            out = self.model(t, use_cache=True)
        self.kv_cache = out.past_key_values
        self.kv_len = len(input_ids)
        self.facts_accumulated = len(self.remembered_facts)

        return {
            "added": len(new_facts),
            "total": len(self.remembered_facts),
            "kv_tokens": self.kv_len
        }

    def search_and_remember(self, query, top_k=5):
        """Search corpus + save found facts to KV memory."""
        results = self.search(query, top_k=top_k)
        if not results:
            return {"found": 0, "added": 0, "total": len(self.remembered_facts)}

        facts = [r[0] for r in results]
        mem_result = self.remember(facts)
        return {
            "found": len(results),
            "scores": [r[1] for r in results],
            **mem_result
        }

    def answer(self, question, max_new_tokens=100):
        """Answer question using accumulated KV memory."""
        q_ids = self.tok.encode(question, add_special_tokens=False)
        qt = torch.tensor([q_ids], device=self.device)

        if self.kv_cache is not None and self.kv_len > 0:
            am = torch.ones(1, self.kv_len + len(q_ids),
                          device=self.device, dtype=torch.long)
            kv_c = self._clone_kv()
            with torch.no_grad():
                out = self.model.generate(
                    qt, past_key_values=kv_c, attention_mask=am,
                    max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tok.eos_token_id,
                )
        else:
            with torch.no_grad():
                out = self.model.generate(
                    qt, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tok.eos_token_id,
                )

        self.queries_answered += 1
        answer = self.tok.decode(out[0][len(q_ids):], skip_special_tokens=True).strip()
        return answer

    def _clone_kv(self):
        c = DynamicCache()
        for li in range(len(self.kv_cache)):
            k = self.kv_cache.layers[li].keys.clone()
            v = self.kv_cache.layers[li].values.clone()
            c.update(k, v, li)
        return c

    def clear(self):
        """Clear KV memory (keep corpus)."""
        self.kv_cache = None
        self.kv_len = 0
        self.remembered_facts = []
        self.facts_accumulated = 0

    def status(self):
        """Return current memory status."""
        return {
            "facts_in_memory": len(self.remembered_facts),
            "kv_tokens": self.kv_len,
            "corpus_size": len(self.corpus),
            "searches": self.search_count,
            "queries_answered": self.queries_answered,
            "prompt_tokens_used": 0,  # always 0!
        }

    def save(self, path):
        """Save memory state to disk."""
        state = {
            "facts": self.remembered_facts,
            "corpus": self.corpus,
        }
        with open(path, 'w') as f:
            json.dump(state, f)

    def load(self, path):
        """Load memory state from disk."""
        with open(path) as f:
            state = json.load(f)
        self.corpus = state.get("corpus", [])
        if self.corpus:
            self._index_corpus()
        facts = state.get("facts", [])
        if facts:
            self.remember(facts)
