"""Tests for data loading (Stuttgart dataset, vocabulary extraction)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from semsim.data import load_stuttgart, get_vocabulary


def test_load_stuttgart(sample_stuttgart_dir):
    """Should load tab-separated Stuttgart files into a DataFrame."""
    df = load_stuttgart(sample_stuttgart_dir)
    assert len(df) > 0
    assert "word1" in df.columns
    assert "word2" in df.columns
    assert "relation" in df.columns


def test_get_vocabulary(sample_stuttgart_dir):
    """Should extract unique words from all pairs."""
    df = load_stuttgart(sample_stuttgart_dir)
    vocab = get_vocabulary(df)
    assert isinstance(vocab, set)
    assert len(vocab) > 0
    assert "good" in vocab
    assert "bad" in vocab
