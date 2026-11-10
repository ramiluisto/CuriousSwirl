"""Tests for the projection-free pair-geometry analyses."""

import numpy as np
import pytest

from semsim.pair_geometry import (
    lexical_split,
    pair_graph_summary,
    probe_accuracies,
    second_moment_contrast,
    word_overlap_vote,
)


def _pairs(n_words=400, n_pairs=1500, seed=0):
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(n_words)]
    return [tuple(rng.choice(words, 2, replace=False)) for _ in range(n_pairs)]


class TestLexicalSplit:
    def test_no_word_is_shared_between_train_and_test(self):
        pairs = _pairs()
        train, test = lexical_split(pairs, test_size=0.4, seed=0)
        train_words = {w for p, m in zip(pairs, train) if m for w in p}
        test_words = {w for p, m in zip(pairs, test) if m for w in p}
        assert train.sum() > 0 and test.sum() > 0
        assert not (train & test).any()
        assert not train_words & test_words

    def test_assignment_depends_on_the_word_not_on_the_vocabulary(self):
        """Two models with different vocabularies hold out the same words."""
        pairs = _pairs()
        _, test_full = lexical_split(pairs, test_size=0.4, seed=1)
        subset = pairs[:700]
        _, test_subset = lexical_split(subset, test_size=0.4, seed=1)
        assert np.array_equal(test_full[:700], test_subset)

    def test_seed_changes_the_split(self):
        pairs = _pairs()
        assert not np.array_equal(lexical_split(pairs, seed=0)[1], lexical_split(pairs, seed=1)[1])


class TestWordOverlapVote:
    def test_memorises_words_without_embeddings(self):
        # Every pair containing "hub0" is class 0, every pair containing "hub1" is class 1.
        pairs = [("hub0", f"a{i}") for i in range(50)] + [("hub1", f"b{i}") for i in range(50)]
        y = np.array([0] * 50 + [1] * 50)
        train = np.tile([True, False], 50)
        result = word_overlap_vote(pairs, y, train, ~train)
        assert result["coverage"] == 1.0
        assert result["accuracy"] == 1.0

    def test_has_no_coverage_on_a_lexical_split(self):
        pairs = _pairs()
        y = np.random.default_rng(0).integers(0, 2, len(pairs))
        train, test = lexical_split(pairs, test_size=0.4, seed=0)
        assert word_overlap_vote(pairs, y, train, test)["coverage"] == 0.0


def _synthetic_pairs(n=3000, d=40, seed=0):
    """Class 1 pairs differ along axis 0 (random sign); class 0 pairs differ isotropically."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    E1 = rng.standard_normal((n, d))
    step = rng.standard_normal((n, d)) * 0.3
    step[y == 1, 0] += rng.choice([-2.0, 2.0], size=(y == 1).sum())
    E2 = E1 + step
    train = rng.random(n) < 0.7
    return E1, E2, y, train, ~train


class TestProbes:
    def test_linear_probe_on_difference_is_blind_to_a_sign_symmetric_class(self):
        E1, E2, y, train, test = _synthetic_pairs()
        accuracy = probe_accuracies(E1, E2, y, train, test)
        assert accuracy["linear_diff"] == pytest.approx(0.5, abs=0.06)
        assert accuracy["linear_diff_symmetrised"] == pytest.approx(0.5, abs=0.06)
        assert accuracy["abs_diff"] > 0.95


class TestSecondMomentContrast:
    def test_recovers_the_planted_direction(self):
        E1, E2, y, train, test = _synthetic_pairs()
        result = second_moment_contrast(E2 - E1, y, train, test, n_components=40, ranks=(1, 5))
        leading = result["directions"][:, 0]
        assert abs(leading[0]) > 0.95
        assert result["generalised_eigenvalues"][0] > 5
        assert result["accuracy_by_rank"][1] > 0.95
        assert result["leading_direction_auc"] > 0.98

    def test_no_contrast_when_classes_are_identically_distributed(self):
        rng = np.random.default_rng(0)
        D = rng.standard_normal((3000, 40))
        y = rng.integers(0, 2, 3000)
        train = rng.random(3000) < 0.7
        result = second_moment_contrast(D, y, train, ~train, n_components=40, ranks=(1, 5))
        assert result["accuracy_by_rank"][1] == pytest.approx(0.5, abs=0.06)
        assert result["accuracy_by_rank"][5] == pytest.approx(0.5, abs=0.06)


def test_pair_graph_summary_counts_duplicates_and_reversals():
    pairs = [("a", "b"), ("b", "a"), ("a", "b"), ("c", "d")]
    summary = pair_graph_summary(pairs)
    assert summary["n_pairs"] == 4
    assert summary["n_words"] == 4
    assert summary["frac_exact_duplicates"] == pytest.approx(0.25)
    assert summary["frac_with_reverse_present"] == pytest.approx(0.75)
    assert summary["max_pairs_per_word"] == 3


def test_word_overlap_vote_reports_none_when_nothing_is_covered():
    pairs = [("a", "b"), ("c", "d")]
    result = word_overlap_vote(pairs, np.array([0, 1]), np.array([True, False]), np.array([False, True]))
    assert result["coverage"] == 0.0
    assert result["accuracy_covered"] is None


@pytest.mark.parametrize("a,b,expected", [
    ("able", "unable", True),
    ("dishonest", "honest", True),
    ("careful", "careless", True),
    ("Possible", "impossible", True),
    ("hot", "cold", False),
    ("under", "over", False),
    ("in", "out", False),
])
def test_is_derived_pair(a, b, expected):
    from semsim.pair_geometry import is_derived_pair

    assert is_derived_pair(a, b) is expected
