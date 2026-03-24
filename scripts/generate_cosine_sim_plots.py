#!/usr/bin/env python3
"""Generate the cosine-similarity violin plot used in the paper."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROJECT_ROOT

STATS_DIR = PROJECT_ROOT / "results" / "statistics"
OUT_PATH = PROJECT_ROOT / "prose" / "img" / "cosine_violin_grid.png"

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
    if "cosine_sim" not in df or df["cosine_sim"].empty:
        return None
    return df["cosine_sim"].values


def load_required_data():
    """Load the four paper datasets and fail if any required input is missing."""
    data = {}
    missing = []
    for slug, _display in MODELS:
        model_data = {}
        for pair_type in PAIR_TYPES:
            values = load_cosine_sims(slug, pair_type)
            if values is None:
                missing.append(f"{slug}_{pair_type}_metrics.csv")
                continue
            model_data[pair_type] = values
        data[slug] = model_data

    if missing:
        raise RuntimeError(
            "Cannot regenerate cosine_violin_grid.png because required statistics are missing:\n- "
            + "\n- ".join(missing)
        )
    return data


def plot_violin_grid(data, out_path=OUT_PATH):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=True)
    for ax, (slug, display) in zip(axes.flat, MODELS):
        parts_data = [data[slug][pair_type] for pair_type in PAIR_TYPES]
        labels = [PAIR_DISPLAY[pair_type] for pair_type in PAIR_TYPES]
        colors_list = [COLORS[pair_type] for pair_type in PAIR_TYPES]

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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    data = load_required_data()
    plot_violin_grid(data)


if __name__ == "__main__":
    main()
