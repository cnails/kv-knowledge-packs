"""Tests for the routing module."""

import torch
import pytest
from kvpack.router import KMeansRouter


def test_single_bank():
    """With 1 fact, routing should work trivially."""
    embs = torch.randn(1, 64)
    router = KMeansRouter(n_banks=1)
    router.fit(embs)
    result = router.route(embs[0])
    assert result.bank_id == 0
    assert result.fact_indices == [0]


def test_two_clusters():
    """Two clearly separated clusters should route correctly."""
    # Cluster A: near [1, 0, ...]
    a = torch.zeros(5, 64)
    a[:, 0] = 1.0
    a += torch.randn(5, 64) * 0.01

    # Cluster B: near [0, 1, ...]
    b = torch.zeros(5, 64)
    b[:, 1] = 1.0
    b += torch.randn(5, 64) * 0.01

    embs = torch.cat([a, b])
    router = KMeansRouter(n_banks=2)
    router.fit(embs)

    # Query near cluster A
    q = torch.zeros(64)
    q[0] = 1.0
    result = router.route(q)
    assert result.fact_indices[0] < 5  # should route to cluster A

    # Query near cluster B
    q2 = torch.zeros(64)
    q2[1] = 1.0
    result2 = router.route(q2)
    assert result2.fact_indices[0] >= 5  # should route to cluster B


def test_top_k():
    """Top-k should return k results."""
    embs = torch.randn(20, 64)
    router = KMeansRouter(n_banks=2)
    router.fit(embs)

    result = router.route(embs[0], top_k_facts=3)
    assert len(result.fact_indices) <= 3
    assert len(result.cosine_scores) == len(result.fact_indices)


def test_state_dict_roundtrip():
    """Save/load should preserve router state."""
    embs = torch.randn(10, 64)
    router = KMeansRouter(n_banks=3)
    router.fit(embs)

    state = router.state_dict()
    router2 = KMeansRouter.from_state_dict(state)

    q = embs[0]
    r1 = router.route(q)
    r2 = router2.route(q)
    assert r1.bank_id == r2.bank_id
    assert r1.fact_indices == r2.fact_indices
