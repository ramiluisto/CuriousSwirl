#!/usr/bin/env python3
"""Projection-free analysis of antonym vs. synonym pairs.

Complements scripts/07_row_order_controls.py: once the UMAP swirl is set
aside, what do the embeddings themselves say?  Everything here is computed in
the original space and does not depend on the order of the rows.

For every model the script reports, on the Stuttgart split and on
word-disjoint ("lexical") splits,

  * a classifier that never sees an embedding (votes of training pairs that
    share a word with the test pair),
  * logistic-regression probes on the cosine similarity, on the difference
    vector D (odd in D), on |D| and on E1*E2 (even in D),
  * the second-moment contrast: how well do the r directions along which
    antonym and synonym difference *directions* differ most classify pairs?
    The words at the two ends of the leading directions are listed, and the
    agreement between models is measured on words held out from fitting.

Usage:
    python scripts/08_projection_free_analysis.py
    python scripts/08_projection_free_analysis.py --models word2vec --lexical-seeds 5
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, RESULTS_DIR, get_embedding_path
from semsim.pair_geometry import (
    lexical_split,
    pair_graph_summary,
    probe_accuracies,
    second_moment_contrast,
    word_overlap_vote,
)
from semsim.unfiltered import load_unfiltered_pairs_for_model

logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR / "projection_free"
LEXICAL_TEST_SIZE = 0.4  # share of the vocabulary held out; about 16% of the pairs end up in test
DISPLAY = {
    "bert-base-cased": "BERT Base",
    "word2vec": "Word2Vec",
    "glove": "GloVe",
    "text-embedding-3-small": "text-embedding-3-small",
    "text-embedding-3-large": "text-embedding-3-large",
}


def load_real_pairs(model):
    """Antonym and synonym pairs with embeddings: pairs, E1, E2, y (1 = antonym), Stuttgart split."""
    data = load_unfiltered_pairs_for_model(model)
    pairs, e1, e2, y, splits = [], [], [], [], []
    for pair_type, label in (("synonyms", 0), ("antonyms", 1)):
        type_pairs, emb_pairs, _meta, split_labels = data[pair_type]
        pairs.extend(type_pairs)
        e1.extend(a for a, _ in emb_pairs)
        e2.extend(b for _, b in emb_pairs)
        y.extend([label] * len(type_pairs))
        splits.extend(split_labels)
    return (pairs, np.asarray(e1, dtype=np.float32), np.asarray(e2, dtype=np.float32),
            np.array(y), np.array(splits))


def evaluate(pairs, E1, E2, y, train, test):
    D = E2 - E1
    scale = D[train].std(axis=0) + 1e-12
    contrast = second_moment_contrast(D / scale, y, train, test)
    result = {
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "word_overlap_vote": word_overlap_vote(pairs, y, train, test),
        "probes": probe_accuracies(E1, E2, y, train, test),
        "second_moment_by_rank": contrast["accuracy_by_rank"],
        "leading_direction_auc": contrast["leading_direction_auc"],
        "top_generalised_eigenvalues": [float(v) for v in contrast["generalised_eigenvalues"][:10]],
    }
    return result, contrast["directions"], scale


def word_coordinates(pairs, E1, E2, direction, scale):
    """Coordinate of every word along a direction of the standardised difference space."""
    vectors = {}
    for (a, b), e1, e2 in zip(pairs, E1, E2):
        vectors[a], vectors[b] = e1, e2
    words = sorted(vectors)
    W = np.stack([vectors[w] for w in words])
    coords = ((W - W.mean(axis=0)) / scale) @ direction
    return dict(zip(words, coords.astype(float)))


def average(results):
    """Mean and standard deviation over a list of result dicts with identical structure."""
    def walk(items):
        first = items[0]
        if isinstance(first, dict):
            return {k: walk([item[k] for item in items]) for k in first}
        if isinstance(first, list):
            return [walk([item[i] for item in items]) for i in range(len(first))]
        return {"mean": float(np.mean(items)), "sd": float(np.std(items))}
    return walk(results)


def run_model(model, lexical_seeds):
    pairs, E1, E2, y, splits = load_real_pairs(model)
    logger.info("%s: %d pairs, dim %d", model, len(pairs), E1.shape[1])
    out = {"n_pairs": len(pairs), "dim": int(E1.shape[1]), "pair_graph": pair_graph_summary(pairs)}
    t0 = time.time()
    out["stuttgart_split"], _, _ = evaluate(pairs, E1, E2, y, splits == "train", splits == "test")
    logger.info("  stuttgart %.0fs %s", time.time() - t0, json.dumps(out["stuttgart_split"])[:400])
    lexical, held_out_words = [], {}
    for seed in range(lexical_seeds):
        t0 = time.time()
        train, test = lexical_split(pairs, test_size=LEXICAL_TEST_SIZE, seed=seed)
        result, directions, scale = evaluate(pairs, E1, E2, y, train, test)
        lexical.append(result)
        if seed == 0:
            # Word coordinates along the two leading contrast directions.  The
            # directions are fitted on training pairs only; `held_out_words`
            # keeps the words that occur in no training pair.
            test_words = {w for pair, is_test in zip(pairs, test) if is_test for w in pair}
            for k in (0, 1):
                coords = word_coordinates(pairs, E1, E2, directions[:, k], scale)
                ranked = sorted(coords, key=coords.get)
                out[f"direction_{k}_extreme_words"] = {"low": ranked[:25], "high": ranked[-25:]}
                held_out_words[k] = {w: coords[w] for w in test_words}
        logger.info("  lexical seed %d %.0fs", seed, time.time() - t0)
    out["lexical_split"] = average(lexical)
    logger.info("  lexical %s", json.dumps(out["lexical_split"]["probes"]))
    logger.info("  lexical 2nd-moment %s", json.dumps(out["lexical_split"]["second_moment_by_rank"]))
    return out, held_out_words


def cross_model_agreement(held_out):
    """|Spearman correlation| between models of held-out word coordinates along the leading direction.

    The lexical split assigns words to the held-out vocabulary by a hash of
    the word, so all models hold out the same words; the correlation is taken
    over the held-out words that both models cover.
    """
    from scipy.stats import spearmanr

    models = list(held_out)
    table = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            common = sorted(set(held_out[a][0]) & set(held_out[b][0]))
            if len(common) < 50:
                continue
            rho = spearmanr([held_out[a][0][w] for w in common], [held_out[b][0][w] for w in common])[0]
            table[f"{a} | {b}"] = {"abs_spearman": float(abs(rho)), "n_words": len(common)}
    return table


def figure(summary, path):
    models = list(summary)
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.6), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        lex = summary[model]["lexical_split"]
        ranks = sorted(int(r) for r in lex["second_moment_by_rank"])
        mean = [lex["second_moment_by_rank"][str(r) if str(r) in lex["second_moment_by_rank"] else r]["mean"] for r in ranks]
        sd = [lex["second_moment_by_rank"][str(r) if str(r) in lex["second_moment_by_rank"] else r]["sd"] for r in ranks]
        ax.errorbar(ranks, mean, yerr=sd, marker="o", ms=4, lw=1.5, color="#4C72B0", capsize=2,
                    label="quadratic form of rank r")
        for name, color, label in (("cosine", "#888888", "cosine similarity only"),
                                   ("linear_diff_symmetrised", "#C44E52", "linear probe on D"),
                                   ("abs_diff", "#55A868", "linear probe on |D|")):
            ax.axhline(lex["probes"][name]["mean"], color=color, ls="--", lw=1.2, label=label)
        ax.set_xscale("log")
        ax.set_title(DISPLAY.get(model, model), fontsize=10)
        ax.set_xlabel("number of contrast directions r")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("test accuracy, word-disjoint split")
    axes[0][0].set_ylim(0.45, 1.0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--lexical-seeds", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary, held_out = {}, {}
    for model in args.models:
        if not get_embedding_path(model).exists():
            logger.warning("No embeddings for %s, skipping", model)
            continue
        summary[model], held_out[model] = run_model(model, args.lexical_seeds)
        (OUT_DIR / "summary.json").write_text(json.dumps({"models": summary}, indent=2))
    if summary:
        agreement = cross_model_agreement(held_out)
        logger.info("cross-model agreement of the leading direction: %s", json.dumps(agreement))
        (OUT_DIR / "summary.json").write_text(
            json.dumps({"models": summary, "leading_direction_agreement": agreement}, indent=2))
        figure(json.loads(json.dumps(summary)), OUT_DIR / "second_moment_contrast.png")
    logger.info("Wrote %s", OUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
