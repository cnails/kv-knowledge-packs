"""Routing module: k-means clustering + cosine retrieval for banked KV."""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class RouteResult:
    """Result of routing a query to a bank."""
    bank_id: int
    fact_indices: list[int]  # indices within the bank, ranked by relevance
    cosine_scores: list[float]


class KMeansRouter:
    """Routes queries to fact banks using k-means clustering + cosine similarity."""

    def __init__(self, n_banks: int = 50, min_bank_size: int = 1):
        self.n_banks = n_banks
        self.min_bank_size = min_bank_size
        self.centroids: torch.Tensor | None = None
        self.labels: np.ndarray | None = None
        self.banks: dict[int, list[int]] = {}
        self.embeddings: torch.Tensor | None = None

    def fit(self, embeddings: torch.Tensor):
        """Cluster fact embeddings into banks.

        Args:
            embeddings: (N, D) tensor of fact query embeddings.
        """
        from sklearn.cluster import KMeans

        n = len(embeddings)
        actual_banks = min(self.n_banks, n)

        self.embeddings = embeddings

        if actual_banks <= 1:
            # Single bank — no clustering needed
            self.centroids = embeddings.mean(dim=0, keepdim=True)
            self.labels = np.zeros(n, dtype=int)
            self.banks = {0: list(range(n))}
            return self

        km = KMeans(n_clusters=actual_banks, random_state=42, n_init=10)
        self.labels = km.fit_predict(embeddings.numpy())
        self.centroids = torch.from_numpy(km.cluster_centers_).float()

        self.banks = {}
        for i, label in enumerate(self.labels):
            if label not in self.banks:
                self.banks[label] = []
            self.banks[label].append(i)

        return self

    def route(self, query_embedding: torch.Tensor, top_k_facts: int = 1) -> RouteResult:
        """Route a query to the best bank and rank facts within it.

        Args:
            query_embedding: (D,) tensor — embedding of the query.
            top_k_facts: number of top facts to return within the bank.

        Returns:
            RouteResult with bank_id, ranked fact indices, and cosine scores.
        """
        if self.centroids is None:
            raise RuntimeError("Router not fitted. Call fit() first.")

        # Find best bank
        cos_with_centroids = F.cosine_similarity(
            query_embedding.unsqueeze(0), self.centroids
        )
        bank_id = cos_with_centroids.argmax().item()
        bank_fact_indices = self.banks[bank_id]

        # Rank facts within bank
        bank_embeddings = self.embeddings[bank_fact_indices]
        cos_within_bank = F.cosine_similarity(
            query_embedding.unsqueeze(0), bank_embeddings
        )

        k = min(top_k_facts, len(bank_fact_indices))
        topk = cos_within_bank.topk(k)

        ranked_indices = [bank_fact_indices[i] for i in topk.indices.tolist()]
        scores = topk.values.tolist()

        return RouteResult(
            bank_id=bank_id,
            fact_indices=ranked_indices,
            cosine_scores=scores,
        )

    def state_dict(self) -> dict:
        """Serialize router state for saving."""
        return {
            "centroids": self.centroids,
            "labels": self.labels,
            "banks": self.banks,
            "embeddings": self.embeddings,
            "n_banks": self.n_banks,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "KMeansRouter":
        """Deserialize router from saved state."""
        router = cls(n_banks=state["n_banks"])
        router.centroids = state["centroids"]
        router.labels = state["labels"]
        router.banks = state["banks"]
        router.embeddings = state["embeddings"]
        return router
