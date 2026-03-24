"""Tests for classification pipeline basics."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from semsim.classify import build_features, symmetrize_features


def _pairs_to_emb_tuples(pairs):
    """Convert sample_pairs dicts to list of (emb1, emb2) tuples."""
    return [(np.array(p["emb1"]), np.array(p["emb2"])) for p in pairs]


def test_build_features_difference(sample_pairs):
    """build_features('difference') should return (n_pairs, dim) array."""
    emb_pairs = _pairs_to_emb_tuples(sample_pairs)
    X = build_features(emb_pairs, "difference")
    assert X.shape == (len(sample_pairs), 8)


def test_build_features_concatenation(sample_pairs):
    """build_features('concatenation') should return (n_pairs, 2*dim) array."""
    emb_pairs = _pairs_to_emb_tuples(sample_pairs)
    X = build_features(emb_pairs, "concatenation")
    assert X.shape == (len(sample_pairs), 16)


def test_symmetrize_difference():
    """Symmetrized difference vectors should double the data with negation."""
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    X_sym = symmetrize_features(X, "difference")
    assert X_sym.shape[0] == 4
    np.testing.assert_array_equal(X_sym[2], -X[0])
    np.testing.assert_array_equal(X_sym[3], -X[1])
