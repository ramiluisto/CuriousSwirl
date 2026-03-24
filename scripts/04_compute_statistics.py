#!/usr/bin/env python3
"""Phase 4: Compute per-pair metrics and cross-group statistical tests."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, PAIR_TYPES, STATISTICS_DIR, ensure_directories, get_pair_path, get_model_slug
from semsim.pairs import load_pairs_with_embeddings
from semsim.stats import compute_pair_metrics, compare_groups, summary_statistics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compute statistics")
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    ensure_directories()
    logger.info("=== Phase 4: Compute Statistics ===")

    for model_name in args.models:
        logger.info("Processing model: %s", model_name)

        # Compute per-pair metrics for each pair type
        all_metrics = {}
        for pair_type in PAIR_TYPES:
            pair_path = get_pair_path(model_name, pair_type)
            if not pair_path.exists():
                logger.warning("Pairs not found: %s", pair_path)
                continue

            _, emb_pairs, meta = load_pairs_with_embeddings(pair_path)
            metrics = compute_pair_metrics(emb_pairs)
            all_metrics[pair_type] = metrics

            # Save per-pair metrics as CSV
            df = pd.DataFrame(metrics)
            df.insert(0, "pair_type", pair_type)
            csv_path = STATISTICS_DIR / f"{get_model_slug(model_name)}_{pair_type}_metrics.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            logger.info("  %s: %d pairs", pair_type, len(df))

        # Summary statistics per pair type
        summary_rows = []
        for pair_type, metrics in all_metrics.items():
            for metric_name, values in metrics.items():
                row = {"pair_type": pair_type, "metric": metric_name}
                row.update(summary_statistics(values))
                summary_rows.append(row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = STATISTICS_DIR / f"{get_model_slug(model_name)}_summary.csv"
            summary_df.to_csv(summary_path, index=False)

        # Pairwise statistical comparisons
        pair_type_list = list(all_metrics.keys())
        comparison_results = {}
        for i, pt_a in enumerate(pair_type_list):
            for pt_b in pair_type_list[i + 1:]:
                key = f"{pt_a}_vs_{pt_b}"
                comparison_results[key] = compare_groups(
                    all_metrics[pt_a], all_metrics[pt_b], pt_a, pt_b,
                )

        if comparison_results:
            comp_path = STATISTICS_DIR / f"{get_model_slug(model_name)}_comparisons.json"
            with open(comp_path, "w") as f:
                json.dump(comparison_results, f, indent=2)
            logger.info("Saved %d comparisons for %s", len(comparison_results), model_name)


if __name__ == "__main__":
    main()
