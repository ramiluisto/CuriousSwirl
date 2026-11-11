#!/usr/bin/env python3
"""Build the data file behind the interactive swirl explorer.

The explorer (`analysis/explorer/swirl_explorer.qmd`) shows, for every model,
the UMAP layout of the same difference vectors under several *row orderings*
of the data matrix, plus three reference layouts.  This script computes the
layouts and packs them, with per-pair metadata, into one JavaScript file that
Quarto embeds into a self-contained HTML page.

Row orderings (same matrix, same UMAP call as the paper, rows fed in another order):

  stored            as in the paper: antonyms, shuffled ant., shuffled syn., synonyms
  permuted_a/_b     two independent random permutations of all rows
  blocks_reordered  the four class blocks in a different order
  alphabetical      rows sorted by the first word of the pair
  by_length         rows sorted by the length of the difference vector

Reference layouts (taken from the caches of scripts/07_row_order_controls.py):

  edges_random      the paper's graph and initialisation, edges visited in random order
  noise             random directions with the original lengths, rows as stored
  adam              umap-learn 0.6 with optimizer="adam" (only if that run exists)

Usage:
    python scripts/07_row_order_controls.py          # once, provides three of the layouts
    python scripts/09_build_explorer.py              # compute missing layouts, write the data file
    python scripts/09_build_explorer.py --models bert-base-cased --orderings alphabetical --no-pack
    quarto render analysis/explorer/swirl_explorer.qmd
"""

import argparse
import base64
import json
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, PROJECT_ROOT, RESULTS_DIR, get_embedding_path, get_model_slug
from semsim.figure_data import PAPER_PAIR_TYPES, load_paper_bundle
from semsim.order_artefact import (
    exact_knn,
    fit_umap,
    neighbour_label_agreement,
    random_permutation,
    rank_locality,
)

logger = logging.getLogger(__name__)

CACHE_DIR = RESULTS_DIR / "explorer"
CONTROLS_DIR = RESULTS_DIR / "row_order_controls"
ADAM_DIR = RESULTS_DIR / "row_order_controls_umap060_adam"
DATA_FILE = PROJECT_ROOT / "analysis" / "explorer" / "explorer_data.js"

N_NEIGHBORS, MIN_DIST, SEED = 30, 0.1, 42
ORDERINGS = ["stored", "permuted_a", "permuted_b", "blocks_reordered", "alphabetical", "by_length"]
BLOCK_ORDER = ["synonyms", "antonyms", "shuffled_synonym_words", "shuffled_antonym_words"]
#: layout key -> (cache directory, array name) for layouts computed by scripts/07
CONTROL_LAYOUTS = {
    "edges_random": (CONTROLS_DIR, "sgd_random_order"),
    "noise": (CONTROLS_DIR, "norm_noise"),
    "adam": (ADAM_DIR, "paper"),
}
MODEL_NAMES = {
    "word2vec": "Word2Vec",
    "glove": "GloVe",
    "bert-base-cased": "BERT Base",
    "text-embedding-3-small": "text-embedding-3-small",
    "text-embedding-3-large": "text-embedding-3-large",
}
SPLIT_CODES = ["", "train", "val", "test"]
POS_CODES = ["", "adjectives", "nouns", "verbs"]


# ---------------------------------------------------------------------------
# Row orderings
# ---------------------------------------------------------------------------

def row_ordering(name, bundle):
    """Return ``order`` with ``order[r]`` = stored row that is fed to UMAP at position ``r``."""
    labels, pairs, X = bundle["labels"], bundle["pairs"], bundle["X"]
    n = len(labels)
    if name == "stored":
        return np.arange(n)
    if name == "permuted_a":
        return random_permutation(n, seed=0)[0]  # the permutation used by scripts/07
    if name == "permuted_b":
        return random_permutation(n, seed=1)[0]
    if name == "blocks_reordered":
        return np.concatenate([np.flatnonzero(labels == pair_type) for pair_type in BLOCK_ORDER])
    if name == "alphabetical":
        keys = [(a.lower(), b.lower()) for a, b in pairs]
        return np.array(sorted(range(n), key=keys.__getitem__))
    if name == "by_length":
        return np.argsort(np.linalg.norm(X, axis=1), kind="stable")
    raise ValueError(f"unknown ordering: {name}")


def layout_for_ordering(model, name, bundle):
    """UMAP layout for one row ordering, aligned with the stored rows (cached)."""
    slug = get_model_slug(model)
    cache = CACHE_DIR / f"{slug}__{name}.npy"
    if cache.exists():
        return np.load(cache)
    control = CONTROLS_DIR / f"{slug}.npz"
    reuse = {"stored": "paper", "permuted_a": "rows_permuted"}.get(name)
    if reuse and control.exists():
        coords = np.load(control)[reuse]
    else:
        order = row_ordering(name, bundle)
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order))
        t0 = time.time()
        fitted, _ = fit_umap(bundle["X"][order], N_NEIGHBORS, MIN_DIST, seed=SEED)
        coords = fitted[inverse]
        logger.info("%s / %s: UMAP in %.0fs", model, name, time.time() - t0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, coords.astype(np.float32))
    return coords


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def b64(array, dtype):
    """Little-endian typed array as base64 text."""
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.dtype(dtype).newbyteorder("<")).tobytes()).decode()


def quantise(coords):
    """Map a layout into the uint16 square, keeping the aspect ratio and a small margin."""
    low, high = np.percentile(coords, [0.05, 99.95], axis=0)  # ignore a few far outliers
    centre, half = (low + high) / 2.0, (high - low).max() / 2.0 * 1.04
    unit = np.clip((coords - centre) / (2.0 * half) + 0.5, 0.0, 1.0)
    return np.round(unit * 65535).astype(np.uint16)


def layout_stats(coords, labels, order):
    return {
        "localityFed": round(float(rank_locality(coords, order=order)["ratio"]), 3),
        "localityStored": round(float(rank_locality(coords)["ratio"]), 3),
        "agreement": round(float(neighbour_label_agreement(coords, labels)["agreement"]), 3),
    }


def pack_model(model, bundle, layouts, orders):
    labels, pairs = bundle["labels"], bundle["pairs"]
    n = len(labels)
    vocab = sorted({w for pair in pairs for w in pair})
    index = {w: i for i, w in enumerate(vocab)}
    in_degree = np.bincount(exact_knn(bundle["X"], N_NEIGHBORS)[0][:, 1:].ravel(), minlength=n)
    packed = {
        "name": MODEL_NAMES.get(model, model),
        "n": n,
        "dim": int(bundle["X"].shape[1]),
        "vocab": vocab,
        "w1": b64([index[a] for a, _ in pairs], np.uint16),
        "w2": b64([index[b] for _, b in pairs], np.uint16),
        "label": b64([PAPER_PAIR_TYPES.index(x) for x in labels], np.uint8),
        "split": b64([SPLIT_CODES.index(x) for x in bundle["splits"]], np.uint8),
        "pos": b64([POS_CODES.index(x) for x in bundle["pos"]], np.uint8),
        "norm": b64(np.linalg.norm(bundle["X"], axis=1), np.float32),
        "cosine": b64(bundle["cosine"], np.float32),
        "inDegree": b64(in_degree, np.uint16),
        "layouts": {},
    }
    identity = np.arange(n)
    for key, coords in layouts.items():
        order = orders.get(key, identity)
        fed_position = np.empty(n, dtype=np.int64)
        fed_position[order] = identity
        entry = {"xy": b64(quantise(coords), np.uint16), "stats": layout_stats(coords, labels, order)}
        if key in orders and key != "stored":
            entry["fed"] = b64(fed_position, np.uint16)
        packed["layouts"][key] = entry
    return packed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--orderings", nargs="+", default=ORDERINGS, choices=ORDERINGS)
    parser.add_argument("--no-pack", action="store_true", help="only compute and cache layouts")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    import umap

    data = {
        "meta": {
            "umapLearn": umap.__version__, "nNeighbors": N_NEIGHBORS, "minDist": MIN_DIST,
            "randomState": SEED, "built": time.strftime("%Y-%m-%d"),
        },
        "pairTypes": PAPER_PAIR_TYPES,
        "splits": SPLIT_CODES,
        "pos": POS_CODES,
        "models": {},
    }
    for model in args.models:
        if not get_embedding_path(model).exists():
            logger.warning("No embeddings for %s, skipping", model)
            continue
        slug = get_model_slug(model)
        bundle = load_paper_bundle(model)
        layouts, orders = {}, {}
        for name in args.orderings:
            layouts[name] = layout_for_ordering(model, name, bundle)
            orders[name] = row_ordering(name, bundle)
        if args.no_pack:
            continue
        for key, (directory, array) in CONTROL_LAYOUTS.items():
            path = directory / f"{slug}.npz"
            if path.exists() and array in np.load(path).files:
                layouts[key] = np.load(path)[array]
            elif key != "adam":
                raise SystemExit(f"{path} lacks '{array}': run scripts/07_row_order_controls.py first")
        data["models"][model] = pack_model(model, bundle, layouts, orders)
        logger.info("%s: packed %d layouts", model, len(layouts))

    if not args.no_pack:
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        DATA_FILE.write_text("window.SWIRL_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
        logger.info("Wrote %s (%.1f MB)", DATA_FILE, DATA_FILE.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
