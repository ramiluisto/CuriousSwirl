"""Tests for the row-order artefact controls."""

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from semsim.order_artefact import (
    NULL_MEDIAN_RANK_GAP,
    exact_knn,
    hubness_summary,
    neighbour_label_agreement,
    norm_matched_noise,
    random_permutation,
    rank_locality,
    scale_mixture_noise,
    supports_layout_reoptimisation,
    transductive_cluster_vote,
)


def _curve(n=2000):
    """A 2-D curve whose arc-length parameter is the row index."""
    t = np.linspace(0, 6 * np.pi, n)
    return np.column_stack([t * np.cos(t), t * np.sin(t)])


class TestRandomPermutation:
    def test_inverse_restores_row_order(self):
        perm, inverse = random_permutation(100, seed=3)
        X = np.arange(100)
        assert np.array_equal(X[perm][inverse], X)

    def test_is_a_permutation(self):
        perm, _ = random_permutation(50, seed=1)
        assert sorted(perm.tolist()) == list(range(50))


class TestExactKnn:
    def test_matches_sklearn(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 12)).astype(np.float32)
        idx, dist = exact_knn(X, 8, chunk=64)
        ref_dist, ref_idx = NearestNeighbors(n_neighbors=8).fit(X).kneighbors(X)
        assert np.array_equal(idx[:, 0], np.arange(300))
        assert np.array_equal(idx, ref_idx)
        assert np.allclose(dist, ref_dist, atol=1e-3)

    def test_rejects_k_larger_than_n(self):
        with pytest.raises(ValueError):
            exact_knn(np.zeros((5, 2), dtype=np.float32), 6)


class TestHubness:
    def test_ring_graph_has_no_hubs(self):
        n = 200
        nbrs = np.column_stack([(np.arange(n) + 1) % n, (np.arange(n) - 1) % n])
        summary = hubness_summary(nbrs)
        assert summary["skewness"] == 0.0
        assert summary["antihub_frac"] == 0.0
        assert summary["top1pct_share"] == pytest.approx(0.01)

    def test_star_graph_is_one_hub(self):
        n = 200
        nbrs = np.zeros((n, 1), dtype=int)  # everybody's neighbour is point 0
        summary = hubness_summary(nbrs)
        assert summary["max_in_degree_frac"] == 1.0
        assert summary["top1pct_share"] == 1.0
        assert summary["antihub_frac"] == pytest.approx((n - 1) / n)

    def test_heterogeneous_norms_create_hubs_in_high_dimension(self):
        """Short vectors are everybody's neighbours once the dimension is large."""
        k = 15
        even = scale_mixture_noise(1500, 200, sigma=0.0, seed=0)
        mixed = scale_mixture_noise(1500, 200, sigma=0.25, seed=0)
        hub_even = hubness_summary(exact_knn(even, k + 1)[0][:, 1:])
        hub_mixed = hubness_summary(exact_knn(mixed, k + 1)[0][:, 1:])
        assert hub_mixed["top1pct_share"] > 3 * hub_even["top1pct_share"]
        assert hub_mixed["antihub_frac"] > hub_even["antihub_frac"]

    def test_hubs_are_the_shortest_vectors(self):
        X = scale_mixture_noise(1500, 200, sigma=0.25, seed=1)
        in_degree = np.bincount(exact_knn(X, 16)[0][:, 1:].ravel(), minlength=len(X))
        norms = np.linalg.norm(X, axis=1)
        hubs = np.argsort(-in_degree)[:15]
        assert norms[hubs].max() < np.quantile(norms, 0.05)


class TestRankLocality:
    def test_curve_in_row_order_is_detected(self):
        result = rank_locality(_curve())
        assert result["median_gap"] <= 3
        assert result["ratio"] < 0.01

    def test_shuffled_rows_show_no_locality(self):
        coords = _curve()
        perm, _ = random_permutation(len(coords), seed=0)
        result = rank_locality(coords[perm])
        assert 0.8 < result["ratio"] < 1.2

    def test_order_argument_recovers_the_hidden_ordering(self):
        coords = _curve()
        perm, _ = random_permutation(len(coords), seed=0)
        shuffled = coords[perm]
        # Row r of `shuffled` is original row perm[r]; listing rows by original
        # position gives the ordering along the curve.
        assert rank_locality(shuffled, order=np.argsort(perm))["ratio"] < 0.01

    def test_null_constant_is_the_median_of_a_uniform_difference(self):
        rng = np.random.default_rng(0)
        gaps = np.abs(rng.random(200_000) - rng.random(200_000))
        assert np.median(gaps) == pytest.approx(NULL_MEDIAN_RANK_GAP, abs=5e-3)


class TestLabelAgreement:
    def test_random_labels_are_at_chance(self):
        rng = np.random.default_rng(0)
        coords = rng.standard_normal((3000, 2))
        labels = rng.integers(0, 4, 3000)
        result = neighbour_label_agreement(coords, labels)
        assert result["chance"] == pytest.approx(0.25, abs=0.01)
        assert result["agreement"] == pytest.approx(result["chance"], abs=0.03)

    def test_row_blocks_on_a_row_ordered_curve_look_like_classes(self):
        """Labels that are nothing but contiguous row blocks separate perfectly."""
        coords = _curve()
        blocks = np.repeat(np.arange(4), len(coords) // 4)
        assert neighbour_label_agreement(coords, blocks)["agreement"] > 0.95


class TestTransductiveClusterVote:
    def test_separable_blobs(self):
        rng = np.random.default_rng(0)
        coords = np.vstack([rng.normal(-5, 1, (400, 2)), rng.normal(5, 1, (400, 2))])
        y = np.repeat([0, 1], 400)
        train = rng.random(800) < 0.7
        assert transductive_cluster_vote(coords, y, train, ~train, n_clusters=4) > 0.98

    def test_uninformative_layout_is_at_chance(self):
        rng = np.random.default_rng(1)
        coords = rng.standard_normal((4000, 2))
        y = rng.integers(0, 2, 4000)
        train = rng.random(4000) < 0.7
        accuracy = transductive_cluster_vote(coords, y, train, ~train, n_clusters=10)
        assert 0.44 < accuracy < 0.56

    def test_row_ordered_curve_leaks_block_labels(self):
        """A layout that merely traces the row order 'classifies' row-block labels."""
        coords = _curve()
        y = np.repeat([0, 1], len(coords) // 2)
        train = np.random.default_rng(2).random(len(coords)) < 0.7
        assert transductive_cluster_vote(coords, y, train, ~train, n_clusters=20) > 0.9


class TestNullModels:
    def test_norm_matched_noise_keeps_norms_only(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((500, 64)).astype(np.float32) * rng.uniform(0.5, 3.0, (500, 1)).astype(np.float32)
        noise = norm_matched_noise(X, seed=1)
        assert noise.shape == X.shape
        assert np.allclose(np.linalg.norm(noise, axis=1), np.linalg.norm(X, axis=1), rtol=1e-4)
        cosine = np.einsum("ij,ij->i", noise, X) / (np.linalg.norm(noise, axis=1) * np.linalg.norm(X, axis=1))
        assert abs(cosine.mean()) < 0.03

    def test_scale_mixture_noise_norm_spread(self):
        even = np.linalg.norm(scale_mixture_noise(2000, 300, sigma=0.0, seed=0), axis=1)
        mixed = np.linalg.norm(scale_mixture_noise(2000, 300, sigma=0.5, seed=0), axis=1)
        assert even.std() / even.mean() < 0.06
        assert mixed.std() / mixed.mean() > 0.3


@pytest.mark.timeout(600)
@pytest.mark.skipif(not supports_layout_reoptimisation(), reason="documents umap-learn < 0.6 behaviour")
def test_seeded_umap_draws_row_order_from_iid_noise():
    """Executable statement of the artefact (umap-learn < 0.6 with a fixed seed).

    I.i.d. noise with heterogeneous norms is laid out as a curve traced in row
    order; visiting the same edges in random order removes it.
    """
    from semsim.order_artefact import fit_umap, reoptimise_layout

    X = scale_mixture_noise(5000, 300, sigma=0.25, seed=1)
    blocks = np.repeat(np.arange(4), len(X) // 4)

    coords, reducer = fit_umap(X, n_neighbors=30, min_dist=0.1, seed=42)
    assert rank_locality(coords)["ratio"] < 0.3
    assert neighbour_label_agreement(coords, blocks)["agreement"] > 0.5

    shuffled_edges = reoptimise_layout(reducer, X, edge_order="random", seed=42)
    assert rank_locality(shuffled_edges)["ratio"] > 0.8
    assert neighbour_label_agreement(shuffled_edges, blocks)["agreement"] < 0.3


@pytest.mark.timeout(600)
def test_shuffle_rows_option_returns_coordinates_in_original_row_order():
    """`shuffle_rows=True` fits on permuted rows but hands back aligned coordinates."""
    from semsim.projections import compute_umap_projection

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-4, 1, (150, 10)), rng.normal(4, 1, (150, 10))]).astype(np.float32)
    labels = np.repeat([0, 1], 150)
    coords = compute_umap_projection(X, n_neighbors=15, min_dist=0.1, shuffle_rows=True)
    assert coords.shape == (300, 2)
    # Two well-separated blobs: if the rows were not restored the labels would be scrambled.
    assert neighbour_label_agreement(coords, labels)["agreement"] > 0.97
