"""Core KnowledgePack class — the main user-facing API."""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from kvpack.router import KMeansRouter


class KnowledgePack:
    """Instant factual memory for LLMs via KV cache injection.

    Example:
        # Build a knowledge pack
        pack = KnowledgePack("Qwen/Qwen3-8B")
        pack.add_facts([
            "The 2025 Super Bowl was won by the Philadelphia Eagles.",
            "Pope Francis died in April 2025 at age 88.",
        ])
        pack.save("2025_events.kp")

        # Use it
        pack = KnowledgePack.load("2025_events.kp", model=model, tokenizer=tokenizer)
        answer = pack.query("Who won the 2025 Super Bowl?")
        # → "The 2025 Super Bowl was won by the Philadelphia Eagles."
    """

    # Default model configs for store layer fraction
    STORE_LAYER_FRAC = 0.65

    def __init__(
        self,
        model_name: str | None = None,
        model: AutoModelForCausalLM | None = None,
        tokenizer: AutoTokenizer | None = None,
        n_banks: int = 50,
        device: str | None = None,
    ):
        """Initialize a KnowledgePack.

        Either provide model_name (will load model) or model + tokenizer (use existing).

        Args:
            model_name: HuggingFace model ID. Will load if model/tokenizer not provided.
            model: Pre-loaded model instance.
            tokenizer: Pre-loaded tokenizer instance.
            n_banks: Number of k-means banks for routing. More banks = less facts per bank.
            device: Device override. Auto-detected if not provided.
        """
        self.model_name = model_name
        self.n_banks = n_banks
        self.facts: list[str] = []
        self.router: KMeansRouter | None = None
        self._model = model
        self._tokenizer = tokenizer

        # Detect device
        if device:
            self.device = torch.device(device)
        elif model is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )

        # If model provided, detect layers
        if self._model is not None:
            self._setup_layers()

    def _setup_layers(self):
        """Detect model architecture and set store layer."""
        model = self._model
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self._layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self._layers = model.transformer.h
        else:
            raise ValueError(f"Unknown model architecture: {type(model)}")

        n_layers = len(self._layers)
        self.store_layer = int(n_layers * self.STORE_LAYER_FRAC)

    def _ensure_model(self):
        """Load model if not already loaded."""
        if self._model is None:
            if self.model_name is None:
                raise RuntimeError("No model loaded. Provide model_name or model/tokenizer.")

            print(f"Loading {self.model_name}...")
            t0 = time.time()
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=dtype, device_map=self.device
            )
            self._model.eval()
            self._setup_layers()
            print(f"Loaded in {time.time() - t0:.1f}s")

    def _extract_embedding(self, text: str) -> torch.Tensor:
        """Extract hidden state embedding at store layer, position -2."""
        self._ensure_model()
        input_ids = self._tokenizer.encode(text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)
        with torch.no_grad():
            outputs = self._model(input_tensor, output_hidden_states=True)
        hidden = outputs.hidden_states[self.store_layer + 1][0]
        return hidden[-2].float().cpu()

    def _recompute_kv(self, fact_text: str) -> tuple:
        """Compute KV cache for a fact string. Returns (kv_cache, kv_len)."""
        self._ensure_model()
        input_ids = self._tokenizer.encode(fact_text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)
        with torch.no_grad():
            outputs = self._model(input_tensor, use_cache=True)
        return outputs.past_key_values, len(input_ids)

    def _clone_kv(self, kv_cache) -> DynamicCache:
        """Clone a KV cache to avoid in-place modification during generation."""
        clone = DynamicCache()
        for li in range(len(kv_cache)):
            layer = kv_cache.layers[li]
            clone.update(layer.keys.clone(), layer.values.clone(), li)
        return clone

    # ── Public API ──────────────────────────────────────────────────────

    def add_facts(self, facts: list[str]):
        """Add facts to the knowledge pack.

        Args:
            facts: List of factual sentences.
                   Each should be a self-contained statement.
        """
        self.facts.extend(facts)
        # Invalidate router (needs rebuild)
        self.router = None

    def build(self):
        """Build the routing index from current facts.

        Extracts embeddings for all facts and clusters them into banks.
        Call this after adding all facts, or it will be called automatically
        on first query.
        """
        self._ensure_model()

        if not self.facts:
            raise ValueError("No facts added. Call add_facts() first.")

        print(f"Building index for {len(self.facts)} facts...")
        t0 = time.time()

        # Extract embeddings
        embeddings = []
        for i, fact in enumerate(self.facts):
            emb = self._extract_embedding(fact)
            embeddings.append(emb)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(self.facts)} embeddings extracted")

        emb_tensor = torch.stack(embeddings)

        # Fit router
        actual_banks = min(self.n_banks, len(self.facts))
        self.router = KMeansRouter(n_banks=actual_banks)
        self.router.fit(emb_tensor)

        elapsed = time.time() - t0
        print(f"Index built in {elapsed:.1f}s ({len(self.facts)} facts, "
              f"{actual_banks} banks)")

    def query(
        self,
        question: str,
        top_k: int = 1,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> str:
        """Query the knowledge pack.

        Args:
            question: The question to answer.
            top_k: Number of facts to retrieve from the bank (1 = most precise).
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to sample or use greedy decoding.
            temperature: Sampling temperature (only if do_sample=True).

        Returns:
            Generated answer string.
        """
        self._ensure_model()

        if self.router is None:
            self.build()

        # Route query to best fact(s)
        query_emb = self._extract_embedding(question)
        route = self.router.route(query_emb, top_k_facts=top_k)

        # Recompute KV for selected facts
        selected_text = " ".join(self.facts[i] for i in route.fact_indices)
        kv_cache, kv_len = self._recompute_kv(selected_text)

        # Generate with KV
        query_ids = self._tokenizer.encode(question, add_special_tokens=False)
        query_tensor = torch.tensor([query_ids], device=self.device)
        attn_mask = torch.ones(1, kv_len + len(query_ids),
                               device=self.device, dtype=torch.long)

        kv_clone = self._clone_kv(kv_cache)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self._model.generate(
                query_tensor,
                past_key_values=kv_clone,
                attention_mask=attn_mask,
                **gen_kwargs,
            )

        new_tokens = output_ids[0][len(query_ids):]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def query_with_metadata(
        self,
        question: str,
        top_k: int = 1,
        max_new_tokens: int = 64,
        do_sample: bool = False,
    ) -> dict:
        """Query with full metadata (routing info, timing, etc.).

        Returns:
            Dict with 'answer', 'routed_facts', 'cosine_scores',
            'bank_id', 'route_ms', 'generate_ms'.
        """
        self._ensure_model()

        if self.router is None:
            self.build()

        # Route
        t0 = time.time()
        query_emb = self._extract_embedding(question)
        route = self.router.route(query_emb, top_k_facts=top_k)
        route_ms = (time.time() - t0) * 1000

        # Recompute + generate
        t1 = time.time()
        selected_text = " ".join(self.facts[i] for i in route.fact_indices)
        kv_cache, kv_len = self._recompute_kv(selected_text)

        query_ids = self._tokenizer.encode(question, add_special_tokens=False)
        query_tensor = torch.tensor([query_ids], device=self.device)
        attn_mask = torch.ones(1, kv_len + len(query_ids),
                               device=self.device, dtype=torch.long)

        kv_clone = self._clone_kv(kv_cache)

        with torch.no_grad():
            output_ids = self._model.generate(
                query_tensor,
                past_key_values=kv_clone,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][len(query_ids):]
        answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        generate_ms = (time.time() - t1) * 1000

        return {
            "answer": answer,
            "routed_facts": [self.facts[i] for i in route.fact_indices],
            "cosine_scores": route.cosine_scores,
            "bank_id": route.bank_id,
            "route_ms": round(route_ms, 1),
            "generate_ms": round(generate_ms, 1),
        }

    # ── Save / Load ─────────────────────────────────────────────────────

    def save(self, path: str | Path):
        """Save knowledge pack to disk.

        Saves facts as JSON + router state as .pt file.
        Total size: ~4 MB for 5000 facts.

        Args:
            path: File path (e.g., "my_facts.kp").
        """
        if self.router is None:
            self.build()

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save facts
        meta = {
            "version": "0.1.0",
            "model_name": self.model_name,
            "n_facts": len(self.facts),
            "n_banks": self.router.n_banks if self.router else self.n_banks,
            "facts": self.facts,
        }
        with open(path / "facts.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save router state
        torch.save(self.router.state_dict(), path / "router.pt")

        total_size = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
        print(f"Saved to {path}/ ({total_size / 1024:.0f} KB, "
              f"{len(self.facts)} facts, {self.router.n_banks} banks)")

    @classmethod
    def load(
        cls,
        path: str | Path,
        model: AutoModelForCausalLM | None = None,
        tokenizer: AutoTokenizer | None = None,
        model_name: str | None = None,
        device: str | None = None,
    ) -> "KnowledgePack":
        """Load a knowledge pack from disk.

        Args:
            path: Path to saved knowledge pack directory.
            model: Pre-loaded model (optional, will load from saved model_name if not provided).
            tokenizer: Pre-loaded tokenizer.
            model_name: Override model name (if different from saved).
            device: Device override.

        Returns:
            Loaded KnowledgePack ready for queries.
        """
        path = Path(path)

        with open(path / "facts.json") as f:
            meta = json.load(f)

        router_state = torch.load(path / "router.pt", weights_only=False)

        resolved_model_name = model_name or meta.get("model_name")

        pack = cls(
            model_name=resolved_model_name,
            model=model,
            tokenizer=tokenizer,
            n_banks=meta.get("n_banks", 50),
            device=device,
        )
        pack.facts = meta["facts"]
        pack.router = KMeansRouter.from_state_dict(router_state)

        print(f"Loaded {len(pack.facts)} facts from {path}/ "
              f"({pack.router.n_banks} banks)")

        return pack

    # ── Utilities ────────────────────────────────────────────────────────

    def info(self) -> dict:
        """Return pack statistics."""
        bank_sizes = []
        if self.router and self.router.banks:
            bank_sizes = [len(v) for v in self.router.banks.values()]

        return {
            "n_facts": len(self.facts),
            "n_banks": self.router.n_banks if self.router else self.n_banks,
            "bank_sizes": {
                "min": min(bank_sizes) if bank_sizes else 0,
                "max": max(bank_sizes) if bank_sizes else 0,
                "avg": sum(bank_sizes) / len(bank_sizes) if bank_sizes else 0,
            },
            "model_name": self.model_name,
            "built": self.router is not None,
        }

    def __len__(self) -> int:
        return len(self.facts)

    def __repr__(self) -> str:
        built = "built" if self.router else "not built"
        return f"KnowledgePack({len(self.facts)} facts, {built})"
