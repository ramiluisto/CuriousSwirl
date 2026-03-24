#!/usr/bin/env python3
"""Generate a UMAP hyperparameter grid for any model.

Examples:
    # All four pair types (default)
    python scripts/generate_hp_grid.py --model word2vec

    # Only antonyms and synonyms (no shuffled controls)
    python scripts/generate_hp_grid.py --model bert-base-cased --pair-types antonyms synonyms

    # Custom hyperparameter ranges
    python scripts/generate_hp_grid.py --model text-embedding-3-large \
        --nn 15 30 50 --md 0.01 0.1 0.5
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROJECT_ROOT, get_model_slug, MODEL_DISPLAY_NAMES
from scripts.generate_paper_figures import fig_hp_grid, OUTPUT_DIR

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a UMAP hyperparameter grid for a given model",
    )
    parser.add_argument("--model", required=True, help="Model name (e.g. word2vec, bert-base-cased)")
    parser.add_argument("--nn", nargs="+", type=int, default=[15, 30, 50, 100],
                        help="n_neighbors values (default: 15 30 50 100)")
    parser.add_argument("--md", nargs="+", type=float, default=[0.005, 0.01, 0.1, 0.25],
                        help="min_dist values (default: 0.005 0.01 0.1 0.25)")
    parser.add_argument("--pair-types", nargs="+", default=None,
                        help="Pair types to include (default: all four). "
                             "Options: synonyms antonyms shuffled_synonym_words shuffled_antonym_words")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/images/)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.output_dir:
        import scripts.generate_paper_figures as gpf
        gpf.OUTPUT_DIR = Path(args.output_dir)
        gpf.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    slug = get_model_slug(args.model)
    display = MODEL_DISPLAY_NAMES.get(args.model, args.model)
    tag = f"umap_hp_grid_{slug}"
    suptitle = f"{display} — UMAP hyperparameter grid (unfiltered)"

    out = fig_hp_grid(
        args.model, "unfiltered", args.nn, args.md, args.dpi,
        tag=tag, suptitle=suptitle, pair_types=args.pair_types,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
