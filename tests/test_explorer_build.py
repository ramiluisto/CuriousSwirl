"""Tests for the data-build script behind the interactive explorer."""

import base64
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def builder():
    path = PROJECT_ROOT / "scripts" / "09_build_explorer.py"
    spec = importlib.util.spec_from_file_location("build_explorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle():
    """A tiny stand-in for `load_paper_bundle`: two rows per pair type, in the paper's block order."""
    labels = np.repeat(["antonyms", "shuffled_antonym_words", "shuffled_synonym_words", "synonyms"], 2)
    pairs = [("hot", "cold"), ("Big", "small"), ("zebra", "ant"), ("apple", "moon"),
             ("quick", "tree"), ("old", "river"), ("glad", "happy"), ("fast", "quick")]
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 5)).astype(np.float32) * np.arange(8, 0, -1, dtype=np.float32)[:, None]
    return {"labels": labels, "pairs": pairs, "X": X}


class TestRowOrdering:
    @pytest.mark.parametrize("name", ["stored", "permuted_a", "permuted_b", "blocks_reordered",
                                      "alphabetical", "by_length"])
    def test_every_ordering_is_a_permutation(self, builder, bundle, name):
        order = builder.row_ordering(name, bundle)
        assert sorted(order.tolist()) == list(range(8))

    def test_stored_is_the_identity(self, builder, bundle):
        assert builder.row_ordering("stored", bundle).tolist() == list(range(8))

    def test_the_two_shuffles_differ(self, builder, bundle):
        a, b = builder.row_ordering("permuted_a", bundle), builder.row_ordering("permuted_b", bundle)
        assert not np.array_equal(a, b)

    def test_blocks_are_restacked_whole(self, builder, bundle):
        order = builder.row_ordering("blocks_reordered", bundle)
        assert bundle["labels"][order].tolist() == np.repeat(builder.BLOCK_ORDER, 2).tolist()

    def test_alphabetical_ignores_case_and_pair_type(self, builder, bundle):
        order = builder.row_ordering("alphabetical", bundle)
        first_words = [bundle["pairs"][i][0].lower() for i in order]
        assert first_words == sorted(first_words)

    def test_by_length_puts_short_vectors_first(self, builder, bundle):
        order = builder.row_ordering("by_length", bundle)
        norms = np.linalg.norm(bundle["X"][order], axis=1)
        assert np.all(np.diff(norms) >= 0)

    def test_unknown_name_is_rejected(self, builder, bundle):
        with pytest.raises(ValueError):
            builder.row_ordering("sideways", bundle)


class TestPacking:
    def test_b64_roundtrip_is_little_endian(self, builder):
        values = np.array([0, 1, 258, 65535])
        raw = base64.b64decode(builder.b64(values, np.uint16))
        assert raw[4:6] == bytes([2, 1])  # 258 = 0x0102, low byte first
        assert np.frombuffer(raw, dtype="<u2").tolist() == values.tolist()

    def test_quantise_keeps_the_aspect_ratio(self, builder):
        rng = np.random.default_rng(0)
        coords = np.column_stack([rng.uniform(0, 10, 5000), rng.uniform(0, 2, 5000)])
        q = builder.quantise(coords).astype(float)
        assert q.min() >= 0 and q.max() <= 65535
        spread = q.max(axis=0) - q.min(axis=0)
        assert spread[0] / spread[1] == pytest.approx(5.0, rel=0.05)
        assert spread[0] > 0.9 * 65535  # the longer axis fills the square

    def test_layout_stats_see_a_curve_fed_in_shuffled_order(self, builder):
        n = 1200
        t = np.linspace(0, 4 * np.pi, n)
        curve = np.column_stack([t * np.cos(t), t * np.sin(t)])
        order = np.random.default_rng(1).permutation(n)   # order[r] = stored row fed at position r
        coords = np.empty_like(curve)
        coords[order] = curve                              # the layout follows the feed order
        labels = np.repeat(["a", "b", "c", "d"], n // 4)
        stats = builder.layout_stats(coords, labels, order)
        assert stats["localityFed"] < 0.02
        assert stats["localityStored"] > 0.8
        assert stats["agreement"] == pytest.approx(0.25, abs=0.05)


@pytest.mark.timeout(300)
def test_paper_bundle_is_stacked_in_the_paper_row_order():
    from config import get_embedding_path
    from semsim.figure_data import PAPER_PAIR_TYPES, load_paper_bundle

    model = "text-embedding-3-small"
    if not get_embedding_path(model).exists():
        pytest.skip("bundled embeddings not available")
    bundle = load_paper_bundle(model)
    labels = bundle["labels"]
    n = len(labels)
    assert bundle["X"].shape[0] == n == len(bundle["pairs"]) == len(bundle["cosine"])
    # four contiguous blocks in the paper's order
    firsts = [int(np.flatnonzero(labels == t)[0]) for t in PAPER_PAIR_TYPES]
    assert firsts == sorted(firsts)
    assert all(np.all(np.diff(np.flatnonzero(labels == t)) == 1) for t in PAPER_PAIR_TYPES)
    # real pairs carry split and part of speech, shuffled pairs do not
    real = np.isin(labels, ["antonyms", "synonyms"])
    assert set(bundle["splits"][real]) == {"train", "val", "test"}
    assert set(bundle["pos"][real]) == {"adjectives", "nouns", "verbs"}
    assert set(bundle["splits"][~real]) == {""} and set(bundle["pos"][~real]) == {""}
    # features are standardised
    assert np.allclose(bundle["X"].mean(axis=0), 0, atol=1e-3)
    assert np.allclose(bundle["X"].std(axis=0), 1, atol=1e-2)
