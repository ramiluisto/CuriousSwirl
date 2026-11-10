#!/usr/bin/env python3
"""Row-order controls for the UMAP "swirl".

For every model this script computes the paper's UMAP projection of the
difference vectors (same data, same row order, same parameters) and compares
it with four interventions that leave the *data* untouched or destroy all of
its content:

  paper             rows as stored: antonyms, shuffled antonyms, shuffled
                    synonyms, synonyms -- exactly what the paper figures use
  rows_permuted     the same matrix with its rows in random order
  sgd_row_order     UMAP's own SGD re-run from the paper graph (edges visited
                    in umap-learn's native row-sorted order)
  sgd_random_order  the same graph, initialisation, loss and seeds, but the
                    edge list is shuffled once before the SGD
  norm_noise        every row replaced by a random direction of the same
                    length, rows as stored

and reports, for each layout,

  * rank locality  -- median distance in row order between 2-D neighbours,
                      as a ratio to its value for an unrelated ordering
  * label agreement -- share of 2-D neighbours with the same pair type
  * transductive accuracy -- cluster-and-vote antonym/synonym classification
                      on the layout (Stuttgart train -> test)

together with hubness statistics of the exact kNN graph.  A purely synthetic
experiment (i.i.d. Gaussian scale-mixture rows, four identically distributed
"classes") is run as well.

Usage:
    python scripts/07_row_order_controls.py
    python scripts/07_row_order_controls.py --models bert-base-cased word2vec
    python scripts/07_row_order_controls.py --synthetic-only

The first three interventions work with any umap-learn version; the two
``sgd_*`` arms need umap-learn < 0.6 and are skipped otherwise.  To compare
library versions, run the script once per environment with different
``--out-dir`` values.
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, RESULTS_DIR, get_embedding_path, get_model_slug
from semsim.classify import build_features
from semsim.order_artefact import (
    exact_knn,
    fit_umap,
    hubness_summary,
    neighbour_label_agreement,
    norm_matched_noise,
    random_permutation,
    rank_locality,
    reoptimise_layout,
    scale_mixture_noise,
    supports_layout_reoptimisation,
    transductive_cluster_vote,
)
from semsim.unfiltered import load_unfiltered_pairs_for_model

logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR / "row_order_controls"
PAIR_TYPES = ["antonyms", "shuffled_antonym_words", "shuffled_synonym_words", "synonyms"]
REAL_TYPES = ["antonyms", "synonyms"]
COLORS = {
    "synonyms": "#4C72B0",
    "antonyms": "#DD8452",
    "shuffled_synonym_words": "#55A868",
    "shuffled_antonym_words": "#C44E52",
}
DISPLAY = {
    "bert-base-cased": "BERT Base",
    "word2vec": "Word2Vec",
    "glove": "GloVe",
    "text-embedding-3-small": "text-embedding-3-small",
    "text-embedding-3-large": "text-embedding-3-large",
}
ARMS = ["paper", "rows_permuted", "sgd_row_order", "sgd_random_order", "norm_noise"]
N_NEIGHBORS, MIN_DIST, SEED = 30, 0.1, 42


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_bundle(model, input_type="difference", return_pairs=False):
    """Features in the paper's row order, plus pair type and Stuttgart split per row."""
    data = load_unfiltered_pairs_for_model(model)
    parts, labels, splits, all_pairs = [], [], [], []
    for pair_type in PAIR_TYPES:
        pairs, emb_pairs, _meta, split_labels = data[pair_type]
        parts.append(build_features(emb_pairs, input_type).astype(np.float32))
        labels.extend([pair_type] * len(pairs))
        splits.extend(split_labels if split_labels else [""] * len(pairs))
        all_pairs.extend(pairs)
    X = StandardScaler().fit_transform(np.vstack(parts)).astype(np.float32)
    if return_pairs:
        return X, np.array(labels), np.array(splits), all_pairs
    return X, np.array(labels), np.array(splits)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def layout_stats(coords, labels, splits, order=None):
    """Diagnostics of one layout; ``order`` is the row order UMAP was given."""
    real = np.isin(labels, REAL_TYPES)
    stats = {
        "rank_locality_ratio": rank_locality(coords, order=order)["ratio"],
        "label_agreement": neighbour_label_agreement(coords, labels)["agreement"],
        "label_agreement_chance": neighbour_label_agreement(coords, labels)["chance"],
    }
    y = (labels[real] == "antonyms").astype(int)
    for k in (2, 20):
        stats[f"transductive_acc_k{k}"] = transductive_cluster_vote(
            coords[real], y, splits[real] == "train", splits[real] == "test", n_clusters=k
        )
    return stats


def raw_space_baselines(X, labels, splits):
    """Order-invariant reference numbers computed without any projection."""
    from sklearn.linear_model import LogisticRegression

    real = np.isin(labels, REAL_TYPES)
    Xr, y = X[real], (labels[real] == "antonyms").astype(int)
    train, test = splits[real] == "train", splits[real] == "test"
    norm = np.linalg.norm(Xr, axis=1, keepdims=True)
    clf = LogisticRegression().fit(norm[train], y[train])
    return {
        "kmeans20_vote_raw_space": transductive_cluster_vote(Xr, y, train, test, n_clusters=20),
        "norm_only_logreg": float(clf.score(norm[test], y[test])),
    }


def hubness_table(model, X):
    """Hubness of the exact 30-NN graph for the three feature/metric choices of the paper."""
    k = N_NEIGHBORS
    table = {"difference_euclidean": hubness_summary(exact_knn(X, k)[0][:, 1:])}
    unit = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    table["difference_cosine"] = hubness_summary(exact_knn(unit, k)[0][:, 1:])
    concat, _, _ = load_bundle(model, "concatenation")
    table["concatenation_euclidean"] = hubness_summary(exact_knn(concat, k)[0][:, 1:])
    return table


def hub_profile(model, top=12):
    """Who the hubs of the Euclidean 30-NN graph of difference vectors are.

    Returns the most popular points (word pair, pair type, share of all points
    that list it as a neighbour, norm relative to the median norm), the rank
    correlation between popularity and vector length, and for each pair type
    the share of its neighbour slots that point at the 1% most popular points.
    """
    from scipy.stats import spearmanr

    X, labels, _splits, pairs = load_bundle(model, return_pairs=True)
    n = len(X)
    neighbours = exact_knn(X, N_NEIGHBORS)[0][:, 1:]
    in_degree = np.bincount(neighbours.ravel(), minlength=n)
    norms = np.linalg.norm(X, axis=1)
    ranked = np.argsort(-in_degree)
    hubs = ranked[: max(1, n // 100)]
    return {
        "top_hubs": [
            {"pair": list(pairs[i]), "pair_type": str(labels[i]),
             "neighbour_of_frac": float(in_degree[i] / n),
             "norm_over_median": float(norms[i] / np.median(norms))}
            for i in ranked[:top]
        ],
        "spearman_in_degree_vs_norm": float(spearmanr(in_degree, norms)[0]),
        "top1pct_hubs_by_pair_type": {t: int((labels[hubs] == t).sum()) for t in PAIR_TYPES},
        "frac_neighbours_that_are_top1pct_hubs": {
            t: float(np.isin(neighbours[labels == t], hubs).mean()) for t in PAIR_TYPES
        },
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_model(model, force=False):
    slug = get_model_slug(model)
    cache = OUT_DIR / f"{slug}.npz"
    stats_path = OUT_DIR / f"{slug}.json"
    if cache.exists() and stats_path.exists() and not force:
        logger.info("%s: cached", model)
        stats = json.loads(stats_path.read_text())
        if "hub_profile" not in stats:
            stats["hub_profile"] = hub_profile(model)
            stats_path.write_text(json.dumps(stats, indent=2))
        return stats, dict(np.load(cache))

    X, labels, splits = load_bundle(model)
    n, d = X.shape
    logger.info("%s: %d rows x %d dims", model, n, d)
    layouts, stats = {}, {"n": n, "d": d, "arms": {}}

    t0 = time.time()
    layouts["paper"], reducer = fit_umap(X, N_NEIGHBORS, MIN_DIST, seed=SEED)
    stats["arms"]["paper"] = layout_stats(layouts["paper"], labels, splits)
    logger.info("  paper            %.0fs %s", time.time() - t0, stats["arms"]["paper"])

    if supports_layout_reoptimisation():
        for arm, edge_order in (("sgd_row_order", "row"), ("sgd_random_order", "random")):
            t0 = time.time()
            layouts[arm] = reoptimise_layout(reducer, X, edge_order=edge_order, seed=SEED)
            stats["arms"][arm] = layout_stats(layouts[arm], labels, splits)
            logger.info("  %-16s %.0fs %s", arm, time.time() - t0, stats["arms"][arm])
    del reducer

    t0 = time.time()
    perm, inverse = random_permutation(n, seed=0)
    permuted, _ = fit_umap(X[perm], N_NEIGHBORS, MIN_DIST, seed=SEED)
    layouts["rows_permuted"] = permuted[inverse]
    stats["arms"]["rows_permuted"] = layout_stats(layouts["rows_permuted"], labels, splits, order=perm)
    stats["arms"]["rows_permuted"]["rank_locality_ratio_wrt_original_order"] = rank_locality(
        layouts["rows_permuted"])["ratio"]
    logger.info("  rows_permuted    %.0fs %s", time.time() - t0, stats["arms"]["rows_permuted"])

    t0 = time.time()
    layouts["norm_noise"], _ = fit_umap(norm_matched_noise(X, seed=0), N_NEIGHBORS, MIN_DIST, seed=SEED)
    stats["arms"]["norm_noise"] = layout_stats(layouts["norm_noise"], labels, splits)
    logger.info("  norm_noise       %.0fs %s", time.time() - t0, stats["arms"]["norm_noise"])

    stats["raw_space"] = raw_space_baselines(X, labels, splits)
    stats["hubness"] = hubness_table(model, X)
    stats["hub_profile"] = hub_profile(model)
    logger.info("  raw space %s", stats["raw_space"])
    logger.info("  hubness %s", stats["hubness"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, labels=labels, splits=splits, perm=perm,
                        **{arm: layouts[arm].astype(np.float32) for arm in layouts})
    stats_path.write_text(json.dumps(stats, indent=2))
    return stats, dict(np.load(cache))


def run_synthetic(n=16000, d=300, sigma=0.25):
    """Four identically distributed 'classes' of i.i.d. noise, stored as four row blocks."""
    X = scale_mixture_noise(n, d, sigma=sigma, seed=1)
    labels = np.repeat(np.array(PAIR_TYPES), n // 4)
    splits = np.where(np.random.default_rng(0).random(n) < 0.7, "train", "test")
    layouts, stats = {}, {"n": n, "d": d, "sigma": sigma, "arms": {}}
    layouts["paper"], reducer = fit_umap(X, N_NEIGHBORS, MIN_DIST, seed=SEED)
    if supports_layout_reoptimisation():
        layouts["sgd_random_order"] = reoptimise_layout(reducer, X, edge_order="random", seed=SEED)
    perm, inverse = random_permutation(n, seed=0)
    permuted, _ = fit_umap(X[perm], N_NEIGHBORS, MIN_DIST, seed=SEED)
    layouts["rows_permuted"] = permuted[inverse]
    homogeneous = scale_mixture_noise(n, d, sigma=0.0, seed=1)
    layouts["equal_norms"], _ = fit_umap(homogeneous, N_NEIGHBORS, MIN_DIST, seed=SEED)
    for arm, coords in layouts.items():
        stats["arms"][arm] = layout_stats(coords, labels, splits, order=perm if arm == "rows_permuted" else None)
        logger.info("  synthetic %-16s %s", arm, stats["arms"][arm])
    stats["hubness"] = {
        "scale_mixture": hubness_summary(exact_knn(X, N_NEIGHBORS)[0][:, 1:]),
        "equal_norms": hubness_summary(exact_knn(homogeneous, N_NEIGHBORS)[0][:, 1:]),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "synthetic.npz", labels=labels,
                        **{arm: c.astype(np.float32) for arm, c in layouts.items()})
    (OUT_DIR / "synthetic.json").write_text(json.dumps(stats, indent=2))
    return stats, layouts, labels


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _scatter(ax, coords, labels, title=None, by_row=False):
    if by_row:
        ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(len(coords)), cmap="turbo",
                   s=1.5, alpha=0.6, edgecolors="none", rasterized=True)
    else:
        for pair_type in ["shuffled_antonym_words", "shuffled_synonym_words", "antonyms", "synonyms"]:
            mask = labels == pair_type
            ax.scatter(coords[mask, 0], coords[mask, 1], c=COLORS[pair_type], s=1.5, alpha=0.45,
                       edgecolors="none", rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fafafa")
    if title:
        ax.set_title(title, fontsize=10)


def _legend(fig):
    from matplotlib.lines import Line2D

    names = {"synonyms": "Synonyms", "antonyms": "Antonyms",
             "shuffled_synonym_words": "Shuffled syn.", "shuffled_antonym_words": "Shuffled ant."}
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[k], markersize=8, label=v)
               for k, v in names.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.0))


def figure_models(results, path):
    columns = [
        ("paper", False, "Paper pipeline\n(coloured by pair type)"),
        ("paper", True, "Same layout\n(coloured by ROW INDEX)"),
        ("rows_permuted", False, "Same data,\nrows in random order"),
        ("sgd_random_order", False, "Same graph and init,\nedges visited in random order"),
        ("norm_noise", False, "Random directions with the\nsame norms, rows as stored"),
    ]
    models = list(results)
    fig, axes = plt.subplots(len(models), len(columns), figsize=(3.3 * len(columns), 3.3 * len(models)),
                             squeeze=False)
    for r, model in enumerate(models):
        layouts = results[model]
        for c, (arm, by_row, title) in enumerate(columns):
            ax = axes[r, c]
            if arm in layouts:
                _scatter(ax, layouts[arm], layouts["labels"], title if r == 0 else None, by_row)
            else:
                ax.axis("off")
            if c == 0:
                ax.set_ylabel(DISPLAY.get(model, model), fontsize=11)
    _legend(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


def figure_synthetic(layouts, labels, path):
    columns = [
        ("paper", False, "i.i.d. noise, four identically\ndistributed 'classes' = row blocks"),
        ("paper", True, "Same layout\n(coloured by ROW INDEX)"),
        ("rows_permuted", False, "Same noise,\nrows in random order"),
        ("sgd_random_order", False, "Same graph and init,\nedges visited in random order"),
        ("equal_norms", False, "Noise with equal norms\n(far weaker hubs), rows as stored"),
    ]
    fig, axes = plt.subplots(1, len(columns), figsize=(3.3 * len(columns), 3.7))
    for ax, (arm, by_row, title) in zip(axes, columns):
        if arm in layouts:
            _scatter(ax, layouts[arm], labels, title, by_row)
        else:
            ax.axis("off")
    _legend(fig)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: results/row_order_controls)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.out_dir is not None:
        global OUT_DIR
        OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import umap

    summary = {"umap_learn_version": umap.__version__, "n_neighbors": N_NEIGHBORS,
               "min_dist": MIN_DIST, "random_state": SEED, "models": {}}

    logger.info("Synthetic null")
    summary["synthetic"], synthetic_layouts, synthetic_labels = run_synthetic()
    figure_synthetic(synthetic_layouts, synthetic_labels, OUT_DIR / "row_order_synthetic.png")

    results = {}
    if not args.synthetic_only:
        for model in args.models:
            if not get_embedding_path(model).exists():
                logger.warning("No embeddings for %s, skipping", model)
                continue
            summary["models"][model], results[model] = run_model(model, force=args.force)
        if results:
            figure_models(results, OUT_DIR / "row_order_controls.png")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", OUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
