#!/usr/bin/env python3
"""
Generate figures for the paper from cached (or freshly computed) projections.

Produces prose/img/autofilled_*.png images and optionally adds missing
projections to the Streamlit cache.

Usage:
    python scripts/generate_paper_figures.py
    python scripts/generate_paper_figures.py --dpi 150   # faster drafts
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PAIR_TYPES,
    get_model_slug,
)
from semsim.projection_cache import (
    build_projection_params,
    get_projection_cache_path,
    load_projection_from_cache,
    save_projection_to_cache,
)
from scripts.prepopulate_cache import load_features_for_dataset

logger = logging.getLogger(__name__)

# ============================================================================
# Style constants — single place to tweak the look of every paper figure
# ============================================================================

PAIR_TYPE_COLORS = {
    "synonyms": "#4C72B0",
    "antonyms": "#DD8452",
    "shuffled_synonym_words": "#55A868",
    "shuffled_antonym_words": "#C44E52",
}

PAIR_TYPE_DISPLAY = {
    "synonyms": "Synonyms",
    "antonyms": "Antonyms",
    "shuffled_synonym_words": "Shuffled Syn.",
    "shuffled_antonym_words": "Shuffled Ant.",
}

MODEL_DISPLAY = {
    "word2vec": "Word2Vec",
    "glove": "GloVe",
    "bert-base-cased": "BERT Base",
    "text-embedding-3-small": "OpenAI Small",
    "text-embedding-3-large": "OpenAI Large",
}

POINT_ALPHA = 0.45
POINT_SIZE = 6
BG_COLOR = "#fafafa"
GRID_COLOR = "#e0e0e0"

OUTPUT_DIR = Path("prose/img")


def _style_ax(ax, title=None, show_axes=True):
    """Apply consistent styling to a single axes."""
    ax.set_facecolor(BG_COLOR)
    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.tick_params(labelsize=7, length=2)
    if title:
        ax.set_title(title, fontsize=9, pad=4)


def scatter_projection(ax, coords, labels, pair_types_to_show=None,
                       alpha=POINT_ALPHA, s=POINT_SIZE):
    """Draw a scatter plot on *ax* coloured by pair type."""
    order = ["shuffled_antonym_words", "shuffled_synonym_words",
             "antonyms", "synonyms"]
    show = pair_types_to_show or list(PAIR_TYPE_COLORS.keys())
    labels_arr = np.asarray(labels)
    for pt in order:
        if pt not in show:
            continue
        mask = labels_arr == pt
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=PAIR_TYPE_COLORS.get(pt, "#999999"),
            s=s, alpha=alpha, edgecolors="none", rasterized=True,
            label=PAIR_TYPE_DISPLAY.get(pt, pt),
        )


def _shared_legend(fig, pair_types=None, ncol=4, fontsize=8, loc="lower center",
                   bbox_to_anchor=(0.5, -0.01)):
    pts = pair_types or list(PAIR_TYPE_COLORS.keys())
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PAIR_TYPE_COLORS[p],
               markersize=6, label=PAIR_TYPE_DISPLAY.get(p, p))
        for p in pts
    ]
    fig.legend(handles=handles, loc=loc, ncol=ncol, fontsize=fontsize,
               frameon=False, bbox_to_anchor=bbox_to_anchor)


# ============================================================================
# Cache helpers
# ============================================================================

ALL_FOUR = sorted(PAIR_TYPES)


def _get_or_compute(dataset, model, method, n_dims, method_kwargs,
                    input_type="difference", symmetrize=False,
                    standardize=True, metric="euclidean",
                    pair_types=None):
    """Load from cache or compute + cache a projection."""
    pts = sorted(pair_types) if pair_types is not None else ALL_FOUR
    slug = get_model_slug(model)
    params = build_projection_params(
        pair_types=pts, symmetrize=symmetrize,
        standardize=standardize, metric=metric,
        **{k: v for k, v in method_kwargs.items() if k != "metric"},
    )
    cache_path = get_projection_cache_path(
        dataset, slug, input_type, method, n_dims, params,
    )
    cached = load_projection_from_cache(cache_path)
    if cached is not None:
        logger.info("Cache hit: %s", cache_path.name)
        return cached

    logger.info("Cache miss — computing %s / %s / %s ...", slug, dataset, method)
    from semsim.projections import compute_projection

    features = load_features_for_dataset(
        model, dataset, pts, input_type, symmetrize, standardize,
    )
    if features is None:
        raise RuntimeError(f"No data for {model}/{dataset}")
    X, labels, words1, words2 = features
    cfg = dict(method_kwargs)
    if metric != "euclidean":
        cfg["metric"] = metric
    coords, _ = compute_projection(X, method, config=cfg, n_components=n_dims)
    proj_data = {
        "coords": coords, "labels": labels,
        "words1": words1, "words2": words2,
    }
    save_projection_to_cache(cache_path, proj_data)
    return proj_data


# ============================================================================
# Figure generators
# ============================================================================

def fig_model_grid(models, dataset, method_kwargs, dpi, tag,
                   suptitle="", method="umap"):
    """2×2 grid of models, all four pair types visible."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))
    for ax, model in zip(axes.flat, models):
        data = _get_or_compute(dataset, model, method, 2, method_kwargs)
        scatter_projection(ax, data["coords"], data["labels"])
        _style_ax(ax, title=MODEL_DISPLAY.get(model, model), show_axes=False)
    _shared_legend(fig)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = OUTPUT_DIR / f"autofilled_{tag}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def fig_subset_grid(model, dataset, method_kwargs, dpi, tag,
                    suptitle="", method="umap"):
    """2×2 grid with separate projections per subset."""
    subsets = [
        (ALL_FOUR, "All four pair types"),
        ([p for p in ALL_FOUR if p != "shuffled_antonym_words"],
         "Without Shuffled Ant."),
        ([p for p in ALL_FOUR if p != "shuffled_synonym_words"],
         "Without Shuffled Syn."),
        (["antonyms", "synonyms"], "Antonyms & Synonyms only"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))
    for ax, (pts, subtitle) in zip(axes.flat, subsets):
        data = _get_or_compute(dataset, model, method, 2, method_kwargs,
                               pair_types=pts)
        scatter_projection(ax, data["coords"], data["labels"])
        _style_ax(ax, title=subtitle, show_axes=False)
    _shared_legend(fig)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = OUTPUT_DIR / f"autofilled_{tag}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def fig_nonresults_grid(dpi, tag):
    """2×2 grid of 'non-result' projections."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))

    # (a) t-SNE of GloVe, all datasets, default params
    data = _get_or_compute("unfiltered", "glove", "tsne", 2,
                           {"perplexity": 30, "max_iter": 1000,
                            "pca_dims": 50, "metric": "euclidean"})
    scatter_projection(axes[0, 0], data["coords"], data["labels"])
    _style_ax(axes[0, 0], title="t-SNE — GloVe (difference)", show_axes=False)

    # (b) UMAP cosine — text-embedding-3-small
    data = _get_or_compute("unfiltered", "text-embedding-3-small", "umap", 2,
                           {"n_neighbors": 30, "min_dist": 0.5},
                           metric="cosine")
    scatter_projection(axes[0, 1], data["coords"], data["labels"])
    _style_ax(axes[0, 1], title="UMAP cosine — OpenAI Small (diff.)", show_axes=False)

    # (c) UMAP concatenation — BERT
    data = _get_or_compute("unfiltered", "bert-base-cased", "umap", 2,
                           {"n_neighbors": 30, "min_dist": 0.5},
                           input_type="concatenation")
    scatter_projection(axes[1, 0], data["coords"], data["labels"])
    _style_ax(axes[1, 0], title="UMAP — BERT Base (concatenation)", show_axes=False)

    # (d) PCA — word2vec, ant+syn only, unfiltered
    data = _get_or_compute("unfiltered", "word2vec", "pca", 2, {})
    scatter_projection(axes[1, 1], data["coords"], data["labels"],
                       pair_types_to_show=["antonyms", "synonyms"])
    _style_ax(axes[1, 1],
              title="PCA — Word2Vec (diff., ant.+syn.)",
              show_axes=False)

    _shared_legend(fig)
    fig.suptitle("Non-swirl projections", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = OUTPUT_DIR / f"autofilled_{tag}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def fig_hp_grid(model, dataset, nn_list, md_list, dpi, tag,
                suptitle="", method="umap", pair_types=None):
    """Rows = n_neighbors, cols = min_dist hyperparameter sweep."""
    pts = pair_types or ALL_FOUR
    nrows, ncols = len(nn_list), len(md_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for ri, nn in enumerate(nn_list):
        for ci, md in enumerate(md_list):
            ax = axes[ri, ci]
            data = _get_or_compute(dataset, model, method, 2,
                                   {"n_neighbors": nn, "min_dist": md},
                                   pair_types=pts)
            scatter_projection(ax, data["coords"], data["labels"],
                               s=2, alpha=0.35)
            _style_ax(ax, show_axes=False)
            if ri == 0:
                ax.set_title(f"min_dist={md}", fontsize=16)
            if ci == 0:
                ax.set_ylabel(f"nn={nn}", fontsize=16, labelpad=8)

    _shared_legend(fig, pair_types=pts, fontsize=14,
                   bbox_to_anchor=(0.5, -0.005))
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    out = OUTPUT_DIR / f"autofilled_{tag}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def fig_single_projection(model, dataset, method, method_kwargs, dpi, tag,
                          suptitle="", pair_types_to_show=None,
                          s=1, alpha=0.5, metric="euclidean",
                          xlabel="Dim 1", ylabel="Dim 2"):
    """Single projection scatter plot (PCA, t-SNE, UMAP, …)."""
    data = _get_or_compute(dataset, model, method, 2, method_kwargs,
                           metric=metric)
    fig, ax = plt.subplots(figsize=(6, 5))
    show = pair_types_to_show or ALL_FOUR
    scatter_projection(ax, data["coords"], data["labels"],
                       pair_types_to_show=show, s=s, alpha=alpha)
    _style_ax(ax, show_axes=True)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if suptitle:
        ax.set_title(suptitle, fontsize=10)
    ax.legend(markerscale=3, fontsize=9, loc="best", framealpha=0.8)
    fig.tight_layout()
    out = OUTPUT_DIR / f"autofilled_{tag}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures from projection cache")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    umap_main = {"n_neighbors": 30, "min_dist": 0.1}
    main_models = ["word2vec", "glove", "bert-base-cased", "text-embedding-3-large"]

    # --- 2×2 model grid, unfiltered, nn=30 md=0.1 ---
    fig_model_grid(
        main_models, "unfiltered", umap_main, args.dpi,
        tag="umap_grid_unfiltered",
        suptitle="UMAP of difference vectors (n_neighbors=30, min_dist=0.1)",
    )

    # --- Word2Vec subset grid, unfiltered ---
    fig_subset_grid(
        "word2vec", "unfiltered", umap_main, args.dpi,
        tag="umap_subsets_word2vec_unfiltered",
        suptitle="Word2Vec UMAP subsets (nn=30, md=0.1, unfiltered)",
    )

    # --- text-embedding-3-large subset grid, unfiltered ---
    fig_subset_grid(
        "text-embedding-3-large", "unfiltered", umap_main, args.dpi,
        tag="umap_subsets_oai_large_unfiltered",
        suptitle="OpenAI Large UMAP subsets (nn=30, md=0.1, unfiltered)",
    )

    # --- UMAP hyperparameter grids for all 5 models ---
    grid_nn = [15, 30, 50, 100]
    grid_md = [0.005, 0.01, 0.1, 0.25]
    all_models = [
        ("word2vec", "Word2Vec"),
        ("glove", "GloVe"),
        ("bert-base-cased", "BERT Base"),
        ("text-embedding-3-small", "OpenAI Small"),
        ("text-embedding-3-large", "OpenAI Large"),
    ]
    for model, display in all_models:
        slug = get_model_slug(model).replace("/", "_")
        fig_hp_grid(
            model, "unfiltered", grid_nn, grid_md, args.dpi,
            tag=f"umap_hp_grid_{slug}",
            suptitle=f"{display} — UMAP hyperparameter grid (unfiltered)",
        )

    # --- UMAP HP grids with antonyms + synonyms only ---
    ant_syn = ["antonyms", "synonyms"]
    for model, display in all_models:
        slug = get_model_slug(model).replace("/", "_")
        fig_hp_grid(
            model, "unfiltered", grid_nn, grid_md, args.dpi,
            tag=f"umap_hp_grid_{slug}_antsyn",
            suptitle=f"{display} — UMAP HP grid (ant.+syn. only)",
            pair_types=ant_syn,
        )

    # --- Non-results 2×2 grid ---
    fig_nonresults_grid(args.dpi, tag="nonresults_grid")

    # --- PCA text-embedding-3-large, unfiltered ---
    fig_single_projection(
        "text-embedding-3-large", "unfiltered", "pca", {}, args.dpi,
        tag="pca_oai_large_unfiltered",
        suptitle="PCA — OpenAI Large difference vectors",
        pair_types_to_show=["antonyms", "synonyms"],
        s=1, xlabel="PC 1", ylabel="PC 2",
    )

    # --- t-SNE text-embedding-3-large, unfiltered ---
    fig_single_projection(
        "text-embedding-3-large", "unfiltered", "tsne",
        {"perplexity": 30, "max_iter": 1000, "pca_dims": 50,
         "metric": "euclidean"},
        args.dpi,
        tag="tsne_oai_large_unfiltered",
        suptitle="t-SNE — OpenAI Large difference vectors",
        pair_types_to_show=["antonyms", "synonyms"],
        s=1, xlabel="t-SNE 1", ylabel="t-SNE 2",
    )

    logger.info("All figures generated in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
