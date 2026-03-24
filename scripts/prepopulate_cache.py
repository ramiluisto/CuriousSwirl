#!/usr/bin/env python3
"""
Pre-populate the Streamlit projection cache.

Batch-generates projections for a sweep of models, datasets, and hyperparameters
so the Streamlit dashboard loads instantly.

Usage:
    # Dry run (show what would be computed)
    python scripts/prepopulate_cache.py --dry-run

    # Quick smoke test
    python scripts/prepopulate_cache.py --models bert-base-cased --datasets validated_3.0 --methods pca --dry-run

    # Full sweep
    python scripts/prepopulate_cache.py

    # Skip existing cached files
    python scripts/prepopulate_cache.py --skip-existing

    # Also generate grid images
    python scripts/prepopulate_cache.py --generate-grids --skip-existing
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODELS,
    PAIR_TYPES,
    PAIRS_DIR,
    PROJECTION_CONFIG,
    get_model_slug,
    get_pair_path,
    ensure_directories,
)
from semsim.classify import build_features, symmetrize_features
from semsim.pairs import load_pairs_with_embeddings
from semsim.projections import compute_projection
from semsim.projection_cache import (
    build_projection_params,
    get_projection_cache_path,
    get_grid_cache_path,
    save_projection_to_cache,
    save_grid_to_cache,
    load_projection_from_cache,
    load_grid_from_cache,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Sweep configuration
# =============================================================================

SWEEP_MODELS = [
    "glove",
    "word2vec",
    "bert-base-cased",
    "text-embedding-3-small",
    "text-embedding-3-large",
]

SWEEP_DATASETS = ["unfiltered", "validated_3.0", "threshold_4.0"]

UMAP_GRID = {
    "n_neighbors": [5, 10, 15, 30, 50, 100, 150],
    "min_dist": [0.005, 0.01, 0.1, 0.25, 0.5],
}

TSNE_GRID = {
    "perplexity": [5, 30, 50],
    "max_iter": [500, 1000, 2000],
}

# Default shared params
DEFAULT_INPUT_TYPE = "difference"
DEFAULT_N_DIMS = 2
DEFAULT_STANDARDIZE = True
DEFAULT_METRIC = "euclidean"
DEFAULT_SYMMETRIZE = False


def get_pairs_dir_for_dataset(dataset: str) -> Path:
    """Return the pairs directory for a dataset key."""
    if dataset == "validated_3.0":
        return PAIRS_DIR
    elif dataset == "unfiltered":
        from config import FILTERED_PAIRS_DIR
        return FILTERED_PAIRS_DIR
    elif dataset.startswith("threshold_"):
        threshold = dataset.replace("threshold_", "")
        return Path("prose/filtering_reports") / f"threshold_{threshold}" / "pairs"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_features_for_dataset(
    model_name: str, dataset: str, pair_types: list,
    input_type: str, symmetrize: bool, standardize: bool,
):
    """Load and prepare feature matrix for a model+dataset combination.

    Returns (X, labels, words1, words2) or None if no data available.
    """
    slug = get_model_slug(model_name)

    if dataset == "unfiltered":
        try:
            from semsim.unfiltered import load_unfiltered_pairs_for_model
            data = load_unfiltered_pairs_for_model(model_name)
        except Exception as e:
            logger.warning("Cannot load unfiltered data for %s: %s", model_name, e)
            return None

        all_labels, all_words1, all_words2, all_vecs = [], [], [], []
        for pt in pair_types:
            if pt not in data:
                continue
            pairs, emb_pairs, _meta, _splits = data[pt]
            X = build_features(emb_pairs, input_type)
            if len(X) == 0:
                continue
            if symmetrize:
                X = symmetrize_features(X, input_type)
                doubled = list(pairs) + list(pairs)
                for p in doubled:
                    all_words1.append(p[0])
                    all_words2.append(p[1])
                all_labels.extend([pt] * len(X))
            else:
                for p in pairs:
                    all_words1.append(p[0])
                    all_words2.append(p[1])
                all_labels.extend([pt] * len(X))
            all_vecs.append(X)
    else:
        pairs_dir = get_pairs_dir_for_dataset(dataset)
        all_labels, all_words1, all_words2, all_vecs = [], [], [], []
        for pt in pair_types:
            pair_path = pairs_dir / f"{slug}_{pt}_pairs.json"
            if not pair_path.exists():
                continue
            pairs, emb_pairs, _meta = load_pairs_with_embeddings(pair_path)
            X = build_features(emb_pairs, input_type)
            if len(X) == 0:
                continue
            if symmetrize:
                X = symmetrize_features(X, input_type)
                doubled = list(pairs) + list(pairs)
                for p in doubled:
                    all_words1.append(p[0])
                    all_words2.append(p[1])
                all_labels.extend([pt] * len(X))
            else:
                for p in pairs:
                    all_words1.append(p[0])
                    all_words2.append(p[1])
                all_labels.extend([pt] * len(X))
            all_vecs.append(X)

    if not all_vecs:
        return None

    X_all = np.vstack(all_vecs)
    if standardize:
        X_all = StandardScaler().fit_transform(X_all)

    return X_all, all_labels, all_words1, all_words2


def generate_projection_jobs(models, datasets, methods):
    """Yield (model, dataset, method, method_kwargs, params) tuples."""
    pair_types = list(PAIR_TYPES)

    for model in models:
        slug = get_model_slug(model)
        for dataset in datasets:
            # PCA
            if "pca" in methods:
                params = build_projection_params(
                    pair_types=pair_types,
                    symmetrize=DEFAULT_SYMMETRIZE,
                    standardize=DEFAULT_STANDARDIZE,
                    metric=DEFAULT_METRIC,
                )
                yield model, dataset, "pca", {}, params

            # UMAP
            if "umap" in methods:
                for nn in UMAP_GRID["n_neighbors"]:
                    for md in UMAP_GRID["min_dist"]:
                        method_kwargs = {
                            "n_neighbors": nn,
                            "min_dist": md,
                            "metric": DEFAULT_METRIC,
                        }
                        params = build_projection_params(
                            pair_types=pair_types,
                            symmetrize=DEFAULT_SYMMETRIZE,
                            standardize=DEFAULT_STANDARDIZE,
                            metric=DEFAULT_METRIC,
                            n_neighbors=nn,
                            min_dist=md,
                        )
                        yield model, dataset, "umap", method_kwargs, params

            # t-SNE
            if "tsne" in methods:
                for perp in TSNE_GRID["perplexity"]:
                    for mi in TSNE_GRID["max_iter"]:
                        method_kwargs = {
                            "perplexity": perp,
                            "max_iter": mi,
                            "pca_dims": PROJECTION_CONFIG["tsne"].get("pca_dims", 50),
                            "metric": DEFAULT_METRIC,
                        }
                        params = build_projection_params(
                            pair_types=pair_types,
                            symmetrize=DEFAULT_SYMMETRIZE,
                            standardize=DEFAULT_STANDARDIZE,
                            metric=DEFAULT_METRIC,
                            perplexity=perp,
                            max_iter=mi,
                        )
                        yield model, dataset, "tsne", method_kwargs, params


def main():
    parser = argparse.ArgumentParser(description="Pre-populate Streamlit projection cache")
    parser.add_argument(
        "--models", nargs="+", default=SWEEP_MODELS,
        help="Models to process (default: all sweep models)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=SWEEP_DATASETS,
        help="Datasets to process (default: all sweep datasets)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["pca", "umap", "tsne"],
        help="Projection methods (default: pca umap tsne)",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip already cached projections")
    parser.add_argument("--generate-grids", action="store_true", help="Also generate grid images")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be computed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ensure_directories()

    jobs = list(generate_projection_jobs(args.models, args.datasets, args.methods))
    logger.info("Total projection jobs: %d", len(jobs))

    if args.dry_run:
        for model, dataset, method, method_kwargs, params in jobs:
            slug = get_model_slug(model)
            cache_path = get_projection_cache_path(
                dataset, slug, DEFAULT_INPUT_TYPE, method, DEFAULT_N_DIMS, params,
            )
            exists = cache_path.exists()
            status = "EXISTS" if exists else "MISSING"
            hp_str = ", ".join(f"{k}={v}" for k, v in method_kwargs.items())
            print(f"[{status}] {dataset} / {slug} / {method} ({hp_str})")
        existing = sum(
            1 for model, ds, meth, _, p in jobs
            if get_projection_cache_path(
                ds, get_model_slug(model), DEFAULT_INPUT_TYPE, meth, DEFAULT_N_DIMS, p,
            ).exists()
        )
        print(f"\nTotal: {len(jobs)} jobs ({existing} already cached)")
        return

    # Group jobs by (model, dataset) to load features once per combination
    from collections import defaultdict
    grouped = defaultdict(list)
    for model, dataset, method, method_kwargs, params in jobs:
        grouped[(model, dataset)].append((method, method_kwargs, params))

    done, skipped, failed = 0, 0, 0
    total = len(jobs)
    t_start = time.time()

    for (model, dataset), method_jobs in grouped.items():
        slug = get_model_slug(model)
        logger.info("Loading features: %s / %s", slug, dataset)

        features = load_features_for_dataset(
            model, dataset, list(PAIR_TYPES),
            DEFAULT_INPUT_TYPE, DEFAULT_SYMMETRIZE, DEFAULT_STANDARDIZE,
        )
        if features is None:
            logger.warning("No data for %s / %s — skipping %d jobs", slug, dataset, len(method_jobs))
            skipped += len(method_jobs)
            continue

        X, labels, words1, words2 = features

        for method, method_kwargs, params in method_jobs:
            cache_path = get_projection_cache_path(
                dataset, slug, DEFAULT_INPUT_TYPE, method, DEFAULT_N_DIMS, params,
            )

            if args.skip_existing and cache_path.exists():
                skipped += 1
                continue

            try:
                coords, _evr = compute_projection(
                    X, method, config=dict(method_kwargs), n_components=DEFAULT_N_DIMS,
                )
                proj_data = {
                    "coords": coords,
                    "labels": labels,
                    "words1": words1,
                    "words2": words2,
                }
                save_projection_to_cache(cache_path, proj_data)
                done += 1
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                logger.info(
                    "[%d/%d] Cached %s / %s / %s (%.1f proj/s)",
                    done + skipped, total, slug, dataset, method, rate,
                )
            except Exception as e:
                failed += 1
                logger.error("Failed %s / %s / %s: %s", slug, dataset, method, e)

    elapsed = time.time() - t_start
    logger.info(
        "Done in %.1fs — %d cached, %d skipped, %d failed (of %d total)",
        elapsed, done, skipped, failed, total,
    )


if __name__ == "__main__":
    main()
