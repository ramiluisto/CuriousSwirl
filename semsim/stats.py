"""
Statistical analysis of word pair embeddings.

Per-pair metrics: cosine similarity, Euclidean distance, dot product, norms.
Cross-group tests: Mann-Whitney U, KS test, Cohen's d.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ---- Per-pair metrics ----

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.linalg.norm(v1 - v2))


def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2))


def compute_pair_metrics(
    emb_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Compute all metrics for a list of embedding pairs.

    Returns dict with arrays: cosine_sim, euclidean_dist, dot_product,
    norm1, norm2, diff_norm.
    """
    n = len(emb_pairs)
    metrics = {
        "cosine_sim": np.zeros(n),
        "euclidean_dist": np.zeros(n),
        "dot_product": np.zeros(n),
        "norm1": np.zeros(n),
        "norm2": np.zeros(n),
        "diff_norm": np.zeros(n),
    }

    for i, (v1, v2) in enumerate(emb_pairs):
        metrics["cosine_sim"][i] = cosine_similarity(v1, v2)
        metrics["euclidean_dist"][i] = euclidean_distance(v1, v2)
        metrics["dot_product"][i] = dot_product(v1, v2)
        metrics["norm1"][i] = np.linalg.norm(v1)
        metrics["norm2"][i] = np.linalg.norm(v2)
        metrics["diff_norm"][i] = np.linalg.norm(v1 - v2)

    logger.info("Computed metrics for %d pairs", n)
    return metrics


# ---- Cross-group statistical tests ----

def mann_whitney_test(values_a: np.ndarray, values_b: np.ndarray) -> Dict[str, float]:
    """Two-sided Mann-Whitney U test."""
    if len(values_a) < 2 or len(values_b) < 2:
        return {"U": float("nan"), "p_value": float("nan")}
    stat, p = sp_stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
    return {"U": float(stat), "p_value": float(p)}


def ks_test(values_a: np.ndarray, values_b: np.ndarray) -> Dict[str, float]:
    """Two-sample Kolmogorov-Smirnov test."""
    if len(values_a) < 2 or len(values_b) < 2:
        return {"statistic": float("nan"), "p_value": float("nan")}
    stat, p = sp_stats.ks_2samp(values_a, values_b)
    return {"statistic": float(stat), "p_value": float(p)}


def cohens_d(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n_a, n_b = len(values_a), len(values_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    mean_a, mean_b = np.mean(values_a), np.mean(values_b)
    var_a, var_b = np.var(values_a, ddof=1), np.var(values_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((mean_a - mean_b) / pooled_std)


def compare_groups(
    metrics_a: Dict[str, np.ndarray],
    metrics_b: Dict[str, np.ndarray],
    label_a: str = "group_a",
    label_b: str = "group_b",
) -> Dict[str, Dict]:
    """Run all statistical tests comparing two groups across all metrics.

    Returns nested dict: {metric_name: {mann_whitney: {...}, ks: {...}, cohens_d: float}}.
    """
    results = {}
    metric_names = sorted(set(metrics_a.keys()) & set(metrics_b.keys()))

    for metric in metric_names:
        a = metrics_a[metric]
        b = metrics_b[metric]
        results[metric] = {
            "mann_whitney": mann_whitney_test(a, b),
            "ks": ks_test(a, b),
            "cohens_d": cohens_d(a, b),
            "mean_a": float(np.mean(a)) if len(a) > 0 else float("nan"),
            "mean_b": float(np.mean(b)) if len(b) > 0 else float("nan"),
            "std_a": float(np.std(a)) if len(a) > 0 else float("nan"),
            "std_b": float(np.std(b)) if len(b) > 0 else float("nan"),
            "n_a": len(a),
            "n_b": len(b),
        }

    logger.info("Compared %s vs %s across %d metrics", label_a, label_b, len(results))
    return results


def summary_statistics(values: np.ndarray) -> Dict[str, float]:
    """Compute summary stats for a single array."""
    if len(values) == 0:
        return {k: float("nan") for k in ["mean", "std", "median", "min", "max", "q25", "q75"]}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }
