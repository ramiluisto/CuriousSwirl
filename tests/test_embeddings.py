"""Tests for embedding loading (pkl and npz formats)."""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from semsim.embeddings import load_embeddings, save_embeddings


def test_load_pkl(tmp_dir, sample_embeddings):
    """Should load embeddings from pickle format."""
    path = tmp_dir / "test_emb.pkl"
    with open(path, "wb") as f:
        pickle.dump(sample_embeddings, f)

    loaded = load_embeddings(path)
    assert len(loaded) == len(sample_embeddings)
    for word in sample_embeddings:
        np.testing.assert_array_equal(loaded[word], sample_embeddings[word])


def test_load_npz(tmp_dir, sample_embeddings):
    """Should load embeddings from compressed npz format."""
    path = tmp_dir / "test_emb.npz"
    words = list(sample_embeddings.keys())
    vectors = np.array([sample_embeddings[w] for w in words], dtype=np.float32)
    np.savez_compressed(path, words=np.array(words), vectors=vectors)

    loaded = load_embeddings(path)
    assert len(loaded) == len(sample_embeddings)
    for word in sample_embeddings:
        np.testing.assert_allclose(loaded[word], sample_embeddings[word])


def test_save_and_load_roundtrip(tmp_dir, sample_embeddings):
    """Save then load should produce identical embeddings."""
    path = tmp_dir / "roundtrip.pkl"
    save_embeddings(sample_embeddings, path)
    loaded = load_embeddings(path)
    assert set(loaded.keys()) == set(sample_embeddings.keys())
    for word in sample_embeddings:
        np.testing.assert_array_equal(loaded[word], sample_embeddings[word])


def test_openai_embeddings_loadable():
    """Pre-shipped OpenAI embeddings should be loadable."""
    from config import get_embedding_path

    path = get_embedding_path("text-embedding-3-large")
    if path.exists():
        emb = load_embeddings(path)
        assert len(emb) > 1000
        dim = next(iter(emb.values())).shape[0]
        assert dim == 3072

    path_small = get_embedding_path("text-embedding-3-small")
    if path_small.exists():
        emb = load_embeddings(path_small)
        assert len(emb) > 1000
        dim = next(iter(emb.values())).shape[0]
        assert dim == 1536
