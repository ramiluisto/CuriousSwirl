"""Shared test fixtures for semsim tests."""

import os
import numpy as np
import pytest
from pathlib import Path

# Prevent XGBoost/PyTorch OpenMP threading deadlock on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_stuttgart_dir(tmp_path):
    """Create a minimal Stuttgart dataset directory for testing."""
    d = tmp_path / "stuttgart"
    d.mkdir()

    (d / "nouns.train").write_text("good\tbad\t1\nhappy\tglad\t0\nbig\tsmall\t1\n")
    (d / "nouns.test").write_text("hot\tcold\t1\nfast\tquick\t0\n")
    (d / "nouns.val").write_text("old\tyoung\t1\n")

    (d / "adjectives.train").write_text("bright\tdim\t1\nnice\tpleasant\t0\n")
    (d / "adjectives.test").write_text("")
    (d / "adjectives.val").write_text("")

    (d / "verbs.train").write_text("run\tsprint\t0\n")
    (d / "verbs.test").write_text("")
    (d / "verbs.val").write_text("")

    return d


@pytest.fixture
def sample_embeddings():
    """Create synthetic embeddings dict (10 words, 8-dim)."""
    rng = np.random.RandomState(42)
    words = ["good", "bad", "happy", "glad", "big", "small",
             "hot", "cold", "fast", "quick"]
    return {w: rng.randn(8).astype(np.float32) for w in words}


@pytest.fixture
def sample_pairs():
    """Create sample word pairs with embeddings."""
    rng = np.random.RandomState(42)
    pairs = []
    words = [("good", "bad"), ("happy", "glad"), ("big", "small"),
             ("hot", "cold"), ("fast", "quick")]
    for w1, w2 in words:
        e1 = rng.randn(8).astype(np.float32)
        e2 = rng.randn(8).astype(np.float32)
        pairs.append({"word1": w1, "word2": w2, "emb1": e1.tolist(), "emb2": e2.tolist()})
    return pairs
