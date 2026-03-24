#!/usr/bin/env python3
"""Phase 5: Demonstrate antonym-vs-synonym classification.

Runs Logistic Regression and a Shallow Neural Network on difference vectors
for a selection of models, using both random and word-aware (lexical) splits.
Results are saved to results/classification/ as JSON.

This is a lightweight demonstration — the full classification results reported
in the paper were produced with additional methods and hyperparameter sweeps.
"""

import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*n_jobs.*")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODELS,
    PAIR_TYPES,
    ensure_directories,
    get_pair_path,
    get_model_slug,
    RESULTS_DIR,
)
from semsim.pairs import load_pairs_with_embeddings
from semsim.classify import build_features, run_classification

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASSIFICATION_DIR = RESULTS_DIR / "classification"

DEMO_MODELS = ["bert-base-cased", "text-embedding-3-small"]
CLASSIFIERS = ["logistic", "shallow_nn"]


def main():
    ensure_directories()
    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=== Phase 5: Classification Demo ===")

    for model_name in DEMO_MODELS:
        slug = get_model_slug(model_name)
        logger.info("Model: %s", model_name)

        syn_path = get_pair_path(model_name, "synonyms")
        ant_path = get_pair_path(model_name, "antonyms")
        if not syn_path.exists() or not ant_path.exists():
            logger.warning("Missing pair files for %s, skipping", model_name)
            continue

        syn_pairs, syn_embs, _ = load_pairs_with_embeddings(syn_path)
        ant_pairs, ant_embs, _ = load_pairs_with_embeddings(ant_path)

        X_syn = build_features(syn_embs, "difference")
        X_ant = build_features(ant_embs, "difference")

        class_names = ["synonyms", "antonyms"]
        model_results = {}

        # Random split
        logger.info("  Running random-split classification...")
        results_random = run_classification(
            [X_syn, X_ant],
            class_names,
            classifiers=CLASSIFIERS,
            test_size=0.2,
            random_state=42,
        )
        model_results["random_split"] = results_random

        # Lexical (word-aware) split
        logger.info("  Running lexical-split classification...")
        results_lexical = run_classification(
            [X_syn, X_ant],
            class_names,
            pairs_list=[syn_pairs, ant_pairs],
            classifiers=CLASSIFIERS,
            test_size=0.2,
            random_state=42,
        )
        model_results["lexical_split"] = results_lexical

        out_path = CLASSIFICATION_DIR / f"{slug}_classification.json"
        with open(out_path, "w") as f:
            json.dump(model_results, f, indent=2)
        logger.info("Saved classification results to %s", out_path)

        for split_name, results in model_results.items():
            for r in results:
                logger.info(
                    "  %s / %s: accuracy=%.4f",
                    split_name, r["classifier"], r["accuracy"],
                )

    logger.info("=== Phase 5 Complete ===")


if __name__ == "__main__":
    main()
