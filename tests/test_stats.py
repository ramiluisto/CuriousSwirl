"""Tests for statistics computation."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from semsim.stats import compute_pair_metrics, compare_groups


def _pairs_to_emb_tuples(pairs):
    """Convert sample_pairs dicts to list of (emb1, emb2) tuples."""
    return [(np.array(p["emb1"]), np.array(p["emb2"])) for p in pairs]


def test_compute_pair_metrics(sample_pairs):
    """Should compute cosine sim, euclidean dist, etc. for each pair."""
    emb_pairs = _pairs_to_emb_tuples(sample_pairs)
    metrics = compute_pair_metrics(emb_pairs)
    assert "cosine_sim" in metrics
    assert len(metrics["cosine_sim"]) == len(sample_pairs)
    assert all(-1.0 <= v <= 1.0 for v in metrics["cosine_sim"])
    assert all(v >= 0 for v in metrics["euclidean_dist"])


def test_compare_groups(sample_pairs):
    """Should compute statistical tests between two groups."""
    emb_pairs = _pairs_to_emb_tuples(sample_pairs)
    metrics_a = compute_pair_metrics(emb_pairs[:3])
    metrics_b = compute_pair_metrics(emb_pairs[3:])
    result = compare_groups(metrics_a, metrics_b)
    assert "cosine_sim" in result
