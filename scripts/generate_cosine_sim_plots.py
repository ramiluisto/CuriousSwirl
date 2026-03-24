#!/usr/bin/env python3
"""
Generate cosine-similarity distribution visualizations for the paper.

Produces several plot types across 4 models (Word2Vec, GloVe, BERT, OpenAI Large).
Saves to prose/img/cosine_sims/.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

STATS_DIR = Path("results/statistics")
OUT_DIR = Path("prose/img/cosine_sims")

MODELS = [
    ("word2vec", "Word2Vec"),
    ("glove", "GloVe"),
    ("bert_base_cased", "BERT Base"),
    ("text_embedding_3_large", "OpenAI Large"),
]

PAIR_TYPES = ["synonyms", "antonyms", "shuffled_synonym_words", "shuffled_antonym_words"]

PAIR_DISPLAY = {
    "synonyms": "Synonyms",
    "antonyms": "Antonyms",
    "shuffled_synonym_words": "Shuffled Syn.",
    "shuffled_antonym_words": "Shuffled Ant.",
}

COLORS = {
    "synonyms": "#4C72B0",
    "antonyms": "#DD8452",
    "shuffled_synonym_words": "#55A868",
    "shuffled_antonym_words": "#C44E52",
}

BG = "#fafafa"
DPI = 200


def load_cosine_sims(slug, pair_type):
    path = STATS_DIR / f"{slug}_{pair_type}_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df["cosine_sim"].values


def load_all():
    """Returns {model_slug: {pair_type: array}}."""
    data = {}
    for slug, _ in MODELS:
        data[slug] = {}
        for pt in PAIR_TYPES:
            vals = load_cosine_sims(slug, pt)
            if vals is not None:
                data[slug][pt] = vals
    return data


# =========================================================================
# Plot 1: Overlapping histograms — one subplot per model (2×2)
# =========================================================================
def plot_histograms_grid(data):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (slug, display) in zip(axes.flat, MODELS):
        for pt in PAIR_TYPES:
            if pt not in data[slug]:
                continue
            ax.hist(data[slug][pt], bins=50, alpha=0.5, density=True,
                    color=COLORS[pt], label=PAIR_DISPLAY[pt])
        ax.set_title(display, fontsize=12)
        ax.set_xlabel("Cosine similarity", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_facecolor(BG)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8, framealpha=0.8)
    fig.suptitle("Cosine similarity distributions by model", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "cosine_hist_grid.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# =========================================================================
# Plot 2: Violin plots — all pair types side by side, one subplot per model
# =========================================================================
def plot_violin_grid(data):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=True)
    for ax, (slug, display) in zip(axes.flat, MODELS):
        parts_data = []
        labels = []
        colors_list = []
        for pt in PAIR_TYPES:
            if pt not in data[slug]:
                continue
            parts_data.append(data[slug][pt])
            labels.append(PAIR_DISPLAY[pt])
            colors_list.append(COLORS[pt])

        parts = ax.violinplot(parts_data, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors_list[i])
            pc.set_alpha(0.7)
        for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("#333333")
                parts[key].set_linewidth(0.8)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=10, rotation=90, ha="center")
        ax.set_title(display, fontsize=13)
        ax.set_facecolor(BG)
        ax.tick_params(axis="y", labelsize=10)

    axes[0].set_ylabel("Cosine similarity", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "cosine_violin_grid.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# =========================================================================
# Plot 3: Box plots — compact comparison
# =========================================================================
def plot_box_grid(data):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=True)
    for ax, (slug, display) in zip(axes.flat, MODELS):
        box_data = []
        labels = []
        colors_list = []
        for pt in PAIR_TYPES:
            if pt not in data[slug]:
                continue
            box_data.append(data[slug][pt])
            labels.append(PAIR_DISPLAY[pt])
            colors_list.append(COLORS[pt])

        bp = ax.boxplot(box_data, patch_artist=True, widths=0.6,
                        flierprops=dict(markersize=2, alpha=0.3))
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("#333333")
            median.set_linewidth(1.5)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=7, rotation=25, ha="right")
        ax.set_title(display, fontsize=11)
        ax.set_facecolor(BG)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("Cosine similarity", fontsize=10)
    fig.suptitle("Cosine similarity distributions (box plot)", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "cosine_box_grid.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# =========================================================================
# Plot 4: KDE overlays — smoother version of histograms, ant vs syn only
# =========================================================================
def plot_kde_antsyn(data):
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (slug, display) in zip(axes.flat, MODELS):
        x_range = np.linspace(-0.2, 1.0, 500)
        for pt in ["synonyms", "antonyms"]:
            if pt not in data[slug]:
                continue
            kde = gaussian_kde(data[slug][pt], bw_method=0.15)
            density = kde(x_range)
            ax.fill_between(x_range, density, alpha=0.35, color=COLORS[pt],
                            label=PAIR_DISPLAY[pt])
            ax.plot(x_range, density, color=COLORS[pt], linewidth=1.5)

        ax.set_title(display, fontsize=12)
        ax.set_xlabel("Cosine similarity", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_facecolor(BG)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=9, framealpha=0.8)

    fig.suptitle("Antonym vs Synonym cosine similarity (KDE)", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "cosine_kde_antsyn.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# =========================================================================
# Plot 5: Cross-model comparison — one panel per pair type, all models
# =========================================================================
def plot_cross_model(data):
    from scipy.stats import gaussian_kde

    MODEL_LINE_COLORS = {
        "word2vec": "#1b9e77",
        "glove": "#d95f02",
        "bert_base_cased": "#7570b3",
        "text_embedding_3_large": "#e7298a",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    pts_order = ["synonyms", "antonyms", "shuffled_synonym_words", "shuffled_antonym_words"]
    for ax, pt in zip(axes.flat, pts_order):
        x_range = np.linspace(-0.3, 1.0, 500)
        for slug, display in MODELS:
            if pt not in data[slug]:
                continue
            kde = gaussian_kde(data[slug][pt], bw_method=0.15)
            density = kde(x_range)
            ax.plot(x_range, density, linewidth=2,
                    color=MODEL_LINE_COLORS[slug], label=display)

        ax.set_title(PAIR_DISPLAY[pt], fontsize=12)
        ax.set_xlabel("Cosine similarity", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_facecolor(BG)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, framealpha=0.8)

    fig.suptitle("Cosine similarity by model (per pair type)", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "cosine_cross_model.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# =========================================================================
# Plot 6: Summary statistics table as a heatmap
# =========================================================================
def plot_summary_heatmap(data):
    models_display = [d for _, d in MODELS]
    pts_display = [PAIR_DISPLAY[pt] for pt in PAIR_TYPES]

    means = np.zeros((len(MODELS), len(PAIR_TYPES)))
    stds = np.zeros_like(means)
    for i, (slug, _) in enumerate(MODELS):
        for j, pt in enumerate(PAIR_TYPES):
            if pt in data[slug]:
                means[i, j] = np.mean(data[slug][pt])
                stds[i, j] = np.std(data[slug][pt])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))

    im1 = ax1.imshow(means, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(range(len(pts_display)))
    ax1.set_xticklabels(pts_display, fontsize=9, rotation=20, ha="right")
    ax1.set_yticks(range(len(models_display)))
    ax1.set_yticklabels(models_display, fontsize=10)
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            ax1.text(j, i, f"{means[i,j]:.3f}", ha="center", va="center",
                     fontsize=9, color="white" if means[i,j] > 0.4 else "black")
    ax1.set_title("Mean cosine similarity", fontsize=11)
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(stds, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(len(pts_display)))
    ax2.set_xticklabels(pts_display, fontsize=9, rotation=20, ha="right")
    ax2.set_yticks(range(len(models_display)))
    ax2.set_yticklabels(models_display, fontsize=10)
    for i in range(stds.shape[0]):
        for j in range(stds.shape[1]):
            ax2.text(j, i, f"{stds[i,j]:.3f}", ha="center", va="center",
                     fontsize=9, color="white" if stds[i,j] > 0.15 else "black")
    ax2.set_title("Std dev of cosine similarity", fontsize=11)
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Cosine similarity summary statistics", fontsize=13, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "cosine_summary_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()

    plot_histograms_grid(data)
    plot_violin_grid(data)
    plot_box_grid(data)
    plot_kde_antsyn(data)
    plot_cross_model(data)
    plot_summary_heatmap(data)

    print(f"\nAll plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
