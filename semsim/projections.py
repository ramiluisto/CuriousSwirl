"""
Dimensionality reduction projections for embedding pair vectors.

Supports PCA, t-SNE, and UMAP for 2D visualization of pair-type structure.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


def compute_pca_projection(
    X: np.ndarray, max_components: int = 50, n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PCA projection, returning coords and explained variance ratio.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        max_components: Max components to fit (for scree analysis).
        n_components: Number of output dimensions (2 or 3).

    Returns:
        coords: Array of shape (n_samples, n_components).
        explained_variance_ratio: Array of shape (n_fitted_components,).
    """
    n_fit = min(max_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_fit)
    transformed = pca.fit_transform(X)
    coords = transformed[:, :n_components]
    logger.info(
        "PCA: %d components, top-%d explain %.1f%% variance",
        n_fit,
        n_components,
        pca.explained_variance_ratio_[:n_components].sum() * 100,
    )
    return coords, pca.explained_variance_ratio_


def compute_tsne_projection(
    X: np.ndarray, perplexity: float = 30, max_iter: int = 1000,
    pca_dims: int = 50, n_components: int = 2, metric: str = "euclidean"
) -> np.ndarray:
    """Compute t-SNE projection.

    Auto-reduces dimensionality with PCA when input dim > pca_dims
    (skipped for non-euclidean metrics since PCA is inherently Euclidean).
    Auto-caps perplexity to (n_samples - 1) / 3 for small datasets.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        perplexity: t-SNE perplexity parameter.
        max_iter: Number of iterations.
        pca_dims: PCA pre-reduction threshold.
        n_components: Number of output dimensions (2 or 3).
        metric: Distance metric ("euclidean" or "cosine").

    Returns:
        coords: Array of shape (n_samples, n_components).
    """
    if metric == "euclidean" and X.shape[1] > pca_dims:
        n_pca = min(pca_dims, X.shape[0])
        pca = PCA(n_components=n_pca)
        X = pca.fit_transform(X)
        logger.info("t-SNE: PCA pre-reduction %d -> %d dims", X.shape[1], n_pca)

    max_perplexity = (X.shape[0] - 1) / 3.0
    if perplexity > max_perplexity:
        perplexity = max(1.0, max_perplexity)
        logger.info("t-SNE: capped perplexity to %.1f for small dataset", perplexity)

    init = "pca" if metric == "euclidean" else "random"
    tsne = TSNE(
        n_components=n_components, perplexity=perplexity, max_iter=max_iter,
        metric=metric, init=init, random_state=42,
    )
    coords = tsne.fit_transform(X)
    logger.info("t-SNE: %d samples, %dD, perplexity=%.1f, metric=%s",
                X.shape[0], n_components, perplexity, metric)
    return coords


def compute_umap_projection(
    X: np.ndarray, n_neighbors: int = 50, min_dist: float = 0.01,
    n_components: int = 2, metric: str = "euclidean"
) -> np.ndarray:
    """Compute UMAP projection.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        n_neighbors: Number of neighbors for UMAP.
        min_dist: Minimum distance for UMAP.
        n_components: Number of output dimensions (2 or 3).
        metric: Distance metric ("euclidean" or "cosine").

    Returns:
        coords: Array of shape (n_samples, n_components).

    Raises:
        ImportError: If umap-learn is not installed.
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required for UMAP projections. "
            "Install with: pip install umap-learn"
        )

    n_neighbors = min(n_neighbors, X.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
        metric=metric, random_state=42,
    )
    coords = reducer.fit_transform(X)
    logger.info("UMAP: %d samples, %dD, n_neighbors=%d, metric=%s",
                X.shape[0], n_components, n_neighbors, metric)
    return coords


def compute_projection(
    X: np.ndarray, method: str, config: Optional[Dict] = None,
    n_components: int = 2
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Dispatch to the appropriate projection method.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        method: One of "pca", "tsne", "umap".
        config: Method-specific parameters (from PROJECTION_CONFIG[method]).
        n_components: Number of output dimensions (2 or 3).

    Returns:
        coords: Array of shape (n_samples, n_components).
        explained_variance_ratio: Only for PCA, None otherwise.
    """
    config = config or {}

    if method == "pca":
        return compute_pca_projection(X, n_components=n_components, **config)
    elif method == "tsne":
        return compute_tsne_projection(X, n_components=n_components, **config), None
    elif method == "umap":
        return compute_umap_projection(X, n_components=n_components, **config), None
    else:
        raise ValueError(f"Unknown projection method: {method}")


def save_projection(
    path: Path,
    coords: np.ndarray,
    labels: List[str],
    words1: List[str],
    words2: List[str],
    method: str,
    input_type: str,
    model_slug: str,
    evr: Optional[np.ndarray] = None,
) -> None:
    """Save projection results to compressed NPZ file.

    Args:
        path: Output file path (.npz).
        coords: 2D coordinates, shape (n_samples, 2).
        labels: Pair type label for each sample.
        words1: First word in each pair.
        words2: Second word in each pair.
        method: Projection method name.
        input_type: "difference" or "concatenation".
        model_slug: Model identifier.
        evr: Explained variance ratio (PCA only).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "coords": coords.astype(np.float32),
        "labels": np.array(labels, dtype=str),
        "words1": np.array(words1, dtype=str),
        "words2": np.array(words2, dtype=str),
        "method": np.array(method),
        "input_type": np.array(input_type),
        "model_slug": np.array(model_slug),
    }
    if evr is not None:
        save_dict["explained_variance_ratio"] = evr.astype(np.float32)

    np.savez_compressed(path, **save_dict)
    logger.info("Saved projection to %s (%d samples)", path, len(coords))


def load_projection(path: Path) -> Dict:
    """Load projection results from NPZ file.

    Returns:
        Dict with keys: coords, labels, words1, words2, method, input_type,
        model_slug, and optionally explained_variance_ratio.
    """
    data = np.load(path, allow_pickle=False)
    result = {
        "coords": data["coords"],
        "labels": data["labels"].tolist(),
        "words1": data["words1"].tolist(),
        "words2": data["words2"].tolist(),
        "method": str(data["method"]),
        "input_type": str(data["input_type"]),
        "model_slug": str(data["model_slug"]),
    }
    if "explained_variance_ratio" in data:
        result["explained_variance_ratio"] = data["explained_variance_ratio"]
    return result
