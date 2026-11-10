"""Controls for row-order artefacts in UMAP projections.

Background
----------
With a fixed ``random_state``, umap-learn 0.5.x optimises the 2-D layout with
a *sequential* stochastic-gradient sweep that visits the graph edges in row
order (edges sorted by head vertex, then by tail vertex).  umap-learn 0.6 keeps
that optimiser as its default and adds order-independent alternatives
(``optimizer="adam"`` or ``"momentum"``).  When the k-nearest-
neighbour graph is dominated by a few hub points -- which is what happens for
difference vectors in high dimension under the Euclidean metric, because the
shortest vectors are close to everything -- each hub is dragged through the
layout in row order and the points are deposited along its path.  The result
is a thin curve whose arc-length parameter is the **row index of the data
matrix**.  If the rows are stored class by class, the classes then appear as
separate coloured segments of that curve.

Nothing in this module is specific to antonyms or synonyms: the helpers work
on any feature matrix and any 2-D layout.

Main entry points
-----------------
``rank_locality``             does position in the layout track a given ordering?
``neighbour_label_agreement`` how well are labels separated in the layout?
``hubness_summary``           how hub-dominated is a kNN graph?
``norm_matched_noise``        null model that keeps only the row norms
``scale_mixture_noise``       fully synthetic null model (i.i.d. rows)
``reoptimise_layout``         re-run UMAP's own SGD with a different edge order
``transductive_cluster_vote`` the cluster-and-vote classifier of the paper
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: Median of |U - V| for independent U, V ~ Uniform[0, 1].
NULL_MEDIAN_RANK_GAP = 1.0 - 1.0 / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Row permutations
# ---------------------------------------------------------------------------

def random_permutation(n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return a random permutation of ``range(n)`` and its inverse.

    ``X[perm]`` is the row-shuffled matrix; if ``Y`` is a layout computed on
    ``X[perm]`` then ``Y[inverse]`` is aligned with the original rows.
    """
    perm = np.random.default_rng(seed).permutation(n)
    inverse = np.empty(n, dtype=np.int64)
    inverse[perm] = np.arange(n)
    return perm, inverse


# ---------------------------------------------------------------------------
# Exact nearest neighbours and hubness
# ---------------------------------------------------------------------------

def exact_knn(X: np.ndarray, k: int, chunk: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Euclidean k-nearest neighbours by brute force.

    Follows the umap-learn convention: column 0 is the point itself (distance
    0), so ``k`` neighbours *including self* are returned.

    Returns
    -------
    indices : int32 array, shape (n, k)
    distances : float32 array, shape (n, k)
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    if k > n:
        raise ValueError(f"k={k} exceeds the number of rows {n}")
    sq = np.einsum("ij,ij->i", X, X)
    indices = np.empty((n, k), dtype=np.int32)
    distances = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        d2 = sq[start:stop, None] + sq[None, :] - 2.0 * (X[start:stop] @ X.T)
        d2[np.arange(stop - start), np.arange(start, stop)] = -np.inf  # self first
        if k < n:
            part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        else:
            part = np.tile(np.arange(n), (stop - start, 1))
        part_d = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(part_d, axis=1)
        indices[start:stop] = np.take_along_axis(part, order, axis=1)
        sorted_d = np.take_along_axis(part_d, order, axis=1)
        sorted_d[:, 0] = 0.0
        distances[start:stop] = np.sqrt(np.maximum(sorted_d, 0.0))
    return indices, distances


def hubness_summary(knn_indices: np.ndarray, n: Optional[int] = None) -> Dict[str, float]:
    """Summarise the in-degree (k-occurrence) distribution of a kNN graph.

    Parameters
    ----------
    knn_indices : array, shape (n, k)
        Neighbour lists **without** the self column.
    n : number of points (defaults to ``len(knn_indices)``).

    Returns a dict with the largest in-degree as a fraction of ``n``, the share
    of all neighbour slots that point at the 1% most popular points (1% would
    be the value for a perfectly even graph), the skewness of the in-degree
    distribution and the fraction of points that are nobody's neighbour.
    """
    n = len(knn_indices) if n is None else n
    in_degree = np.bincount(np.asarray(knn_indices).ravel(), minlength=n).astype(np.float64)
    top = max(1, n // 100)
    centred = in_degree - in_degree.mean()
    std = in_degree.std()
    return {
        "max_in_degree_frac": float(in_degree.max() / n),
        "top1pct_share": float(np.sort(in_degree)[-top:].sum() / in_degree.sum()),
        "skewness": float((centred ** 3).mean() / std ** 3) if std > 0 else 0.0,
        "antihub_frac": float((in_degree == 0).mean()),
    }


# ---------------------------------------------------------------------------
# Layout diagnostics
# ---------------------------------------------------------------------------

def _layout_neighbours(coords: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)
    return idx[:, 1:]


def rank_locality(coords: np.ndarray, order: Optional[Sequence[int]] = None, k: int = 5) -> Dict[str, float]:
    """Does position in a layout track a given ordering of the points?

    For every point take its ``k`` nearest neighbours in the layout and record
    how far apart they are in ``order``.  If the layout is unrelated to the
    ordering the median gap is about ``0.293 * n``; if the layout is a curve
    traced in that order the gap is a handful of positions.

    Parameters
    ----------
    coords : array, shape (n, 2)
    order : sequence of row indices giving the ordering to test, i.e.
        ``order[r]`` is the row that sits at position ``r``.  Defaults to the
        row order itself.
    k : number of layout neighbours per point.

    Returns
    -------
    dict with ``median_gap`` (in positions), ``null_median_gap`` and their
    ``ratio`` (about 1 means no relation, near 0 means the layout follows the
    ordering).
    """
    n = len(coords)
    rank = np.arange(n) if order is None else np.empty(n, dtype=np.int64)
    if order is not None:
        rank[np.asarray(order)] = np.arange(n)
    nbrs = _layout_neighbours(coords, k)
    gap = float(np.median(np.abs(rank[nbrs] - rank[:, None])))
    null = NULL_MEDIAN_RANK_GAP * n
    return {"median_gap": gap, "null_median_gap": float(null), "ratio": gap / null}


def neighbour_label_agreement(coords: np.ndarray, labels: Sequence, k: int = 10) -> Dict[str, float]:
    """Fraction of each point's ``k`` nearest layout neighbours sharing its label.

    Also returns the chance level ``sum_c p_c**2`` expected when labels are
    unrelated to the layout.
    """
    labels = np.asarray(labels)
    nbrs = _layout_neighbours(coords, k)
    agreement = float((labels[nbrs] == labels[:, None]).mean())
    _, counts = np.unique(labels, return_counts=True)
    chance = float(((counts / counts.sum()) ** 2).sum())
    return {"agreement": agreement, "chance": chance}


def transductive_cluster_vote(
    coords: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    n_clusters: int = 20,
    seed: int = 0,
) -> float:
    """Cluster a layout, let the training points vote per cluster, score the test points.

    This is the transductive classifier described in the paper: k-means on the
    2-D coordinates of all (train and test) points, majority vote of the
    training labels inside each cluster, cluster label assigned to the test
    points.  Clusters without training points get the global training majority.

    Returns the test accuracy.
    """
    from sklearn.cluster import KMeans

    y = np.asarray(y)
    train_mask = np.asarray(train_mask, dtype=bool)
    test_mask = np.asarray(test_mask, dtype=bool)
    clusters = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed).fit_predict(coords)
    classes = np.unique(y[train_mask])
    global_majority = classes[np.argmax([(y[train_mask] == c).sum() for c in classes])]
    predicted = np.empty(len(y), dtype=y.dtype)
    for cluster in range(n_clusters):
        members = clusters == cluster
        votes = y[members & train_mask]
        if len(votes):
            predicted[members] = classes[np.argmax([(votes == c).sum() for c in classes])]
        else:
            predicted[members] = global_majority
    return float((predicted[test_mask] == y[test_mask]).mean())


# ---------------------------------------------------------------------------
# Null models
# ---------------------------------------------------------------------------

def norm_matched_noise(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Replace every row by a uniformly random direction of the same length.

    Row ``i`` of the result has the Euclidean norm of row ``i`` of ``X`` and
    nothing else in common with it: semantics, shared words and every other
    relation between rows are destroyed.
    """
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    G = rng.standard_normal(X.shape, dtype=np.float32)
    scale = np.linalg.norm(X, axis=1) / np.linalg.norm(G, axis=1)
    G *= scale[:, None].astype(np.float32)
    return G


def scale_mixture_noise(n: int, d: int, sigma: float = 0.25, seed: int = 0) -> np.ndarray:
    """I.i.d. rows from an isotropic Gaussian scale mixture.

    ``x_i = s_i * g_i`` with ``g_i ~ N(0, I_d)`` and ``log s_i ~ N(0, sigma^2)``.
    Every row has the same distribution and there is no cluster, manifold or
    ordering structure.  ``sigma > 0`` makes the norms heterogeneous, which in
    high dimension turns the shortest rows into nearest-neighbour hubs.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)
    X *= np.exp(sigma * rng.standard_normal(n)).astype(np.float32)[:, None]
    return X


# ---------------------------------------------------------------------------
# Re-running UMAP's layout stage with a chosen edge order
# ---------------------------------------------------------------------------

def supports_layout_reoptimisation() -> bool:
    """True for umap-learn < 0.6, whose SGD stage is driven by plain edge arrays.

    umap-learn 0.6 keeps this optimiser as its default ("compatibility") layout
    but with a different internal interface, and adds ``optimizer="adam"`` and
    ``"momentum"``, which shuffle the node order every epoch.
    """
    import umap

    major, minor = (int(part) for part in umap.__version__.split(".")[:2])
    return (major, minor) < (0, 6)


def fit_umap(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1,
             metric: str = "euclidean", seed: int = 42, **umap_kwargs):
    """Fit UMAP exactly as the paper pipeline does and return ``(coords, reducer)``.

    Extra keyword arguments go to ``umap.UMAP``; for example ``optimizer="adam"``
    selects the order-independent optimiser of umap-learn >= 0.6.
    """
    import umap

    n_neighbors = min(n_neighbors, X.shape[0] - 1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                        metric=metric, random_state=seed, **umap_kwargs)
    coords = reducer.fit_transform(X)
    return coords, reducer


def reoptimise_layout(reducer, X: np.ndarray, edge_order: str = "random", seed: int = 42) -> np.ndarray:
    """Re-run only the SGD stage of a fitted umap-learn (< 0.6) model.

    The fuzzy graph, the spectral initialisation, the loss and the random
    seeds are identical between calls; only the order in which the graph edges
    are visited inside each epoch differs.

    Parameters
    ----------
    reducer : fitted ``umap.UMAP`` (from :func:`fit_umap`), on at least 4096
        rows so that umap-learn has stored its neighbour lists.
    X : the matrix the reducer was fitted on.
    edge_order : ``"row"`` keeps umap-learn's native order (sorted by head
        vertex, then tail vertex); ``"random"`` shuffles the edge list once.
    """
    if not supports_layout_reoptimisation():
        raise RuntimeError("reoptimise_layout relies on umap-learn < 0.6 internals")
    if edge_order not in ("row", "random"):
        raise ValueError(f"unknown edge_order: {edge_order}")
    if getattr(reducer, "_knn_indices", None) is None:
        # Below 4096 rows umap-learn computes all pairwise distances and keeps
        # no neighbour lists to rebuild the initialisation from.
        raise ValueError("reoptimise_layout needs a model fitted on at least 4096 rows")

    import umap
    from umap.layouts import optimize_layout_euclidean
    from umap.umap_ import INT32_MAX, INT32_MIN, find_ab_params, make_epochs_per_sample

    n = X.shape[0]
    init_model = umap.UMAP(
        n_components=2, n_neighbors=reducer.n_neighbors, min_dist=reducer.min_dist,
        metric=reducer.metric, random_state=seed, n_epochs=0,
        precomputed_knn=(reducer._knn_indices, reducer._knn_dists, reducer._knn_search_index),
    )
    init = init_model.fit(X).embedding_
    init = (10.0 * (init - init.min(0)) / (init.max(0) - init.min(0))).astype(np.float32, order="C")

    graph = reducer.graph_.tocoo()
    graph.sum_duplicates()
    n_epochs = 500 if n <= 10000 else 200
    graph.data[graph.data < graph.data.max() / float(n_epochs)] = 0.0
    graph.eliminate_zeros()
    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)
    head, tail = graph.row.copy(), graph.col.copy()
    if edge_order == "random":
        shuffle = np.random.default_rng(seed).permutation(len(head))
        head, tail, epochs_per_sample = head[shuffle], tail[shuffle], epochs_per_sample[shuffle]

    a, b = find_ab_params(1.0, reducer.min_dist)
    rng_state = np.random.RandomState(seed).randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    return optimize_layout_euclidean(
        init, init, head, tail, n_epochs, n, epochs_per_sample, a, b, rng_state,
        1.0, 1.0, 5, parallel=False, verbose=False, move_other=True,
    )
