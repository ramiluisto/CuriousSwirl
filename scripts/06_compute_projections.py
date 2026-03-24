#!/usr/bin/env python3
"""Phase 6: Compute dimensionality reduction projections (PCA, t-SNE, UMAP)."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Prevent thread deadlock

import argparse
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODELS, PAIR_TYPES, PROJECTION_CONFIG, PROJECTION_DIMS,
    ensure_directories, get_pair_path, get_model_slug, get_projection_path,
)
from semsim.pairs import load_pairs_with_embeddings
from semsim.classify import build_features
from semsim.projections import compute_projection, save_projection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compute 2D/3D projections")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--methods", nargs="+", default=PROJECTION_CONFIG["methods"])
    parser.add_argument("--input-types", nargs="+", default=PROJECTION_CONFIG["input_types"])
    parser.add_argument("--dims", nargs="+", type=int, default=PROJECTION_DIMS)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    ensure_directories()
    logger.info("=== Phase 6: Projections ===")

    for model_name in args.models:
        logger.info("Model: %s", model_name)
        model_slug = get_model_slug(model_name)

        # Load all pair data for this model
        all_pairs = {}
        for pair_type in PAIR_TYPES:
            pair_path = get_pair_path(model_name, pair_type)
            if not pair_path.exists():
                logger.warning("Missing pairs: %s", pair_path)
                continue
            pairs, emb_pairs, meta = load_pairs_with_embeddings(pair_path)
            all_pairs[pair_type] = (pairs, emb_pairs)

        if not all_pairs:
            logger.warning("No pair data for %s, skipping", model_name)
            continue

        for input_type in args.input_types:
            logger.info("  Input type: %s", input_type)

            # Build combined feature matrix with labels and word pairs
            X_parts = []
            labels = []
            words1 = []
            words2 = []
            for pair_type in PAIR_TYPES:
                if pair_type not in all_pairs:
                    continue
                pairs, emb_pairs = all_pairs[pair_type]
                X = build_features(emb_pairs, input_type)
                if len(X) == 0:
                    continue
                X_parts.append(X)
                labels.extend([pair_type] * len(X))
                words1.extend([p[0] for p in pairs])
                words2.extend([p[1] for p in pairs])

            if not X_parts:
                logger.warning("No features for %s/%s, skipping", model_name, input_type)
                continue

            X_all = np.vstack(X_parts)
            logger.info("    Combined: %d samples, %d features", X_all.shape[0], X_all.shape[1])

            # Standardize
            if PROJECTION_CONFIG.get("standardize", True):
                X_all = StandardScaler().fit_transform(X_all)

            for method in args.methods:
                for n_dims in args.dims:
                    out_path = get_projection_path(model_name, input_type, method, n_dims)
                    if args.skip_existing and out_path.exists():
                        logger.info("    Skipping %s (exists)", out_path.name)
                        continue

                    logger.info("    Computing %s %dD ...", method, n_dims)
                    method_config = PROJECTION_CONFIG.get(method, {})

                    try:
                        coords, evr = compute_projection(
                            X_all, method, method_config, n_components=n_dims
                        )
                    except ImportError as e:
                        logger.warning("    Skipping %s: %s", method, e)
                        continue

                    save_projection(
                        out_path, coords, labels, words1, words2,
                        method=method, input_type=input_type, model_slug=model_slug,
                        evr=evr,
                    )

    logger.info("Projections complete!")


if __name__ == "__main__":
    main()
