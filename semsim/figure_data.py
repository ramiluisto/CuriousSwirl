"""Helpers for loading the paper's projection datasets."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from config import PAIRS_DIR, get_model_slug
from semsim.classify import build_features, symmetrize_features
from semsim.pairs import load_pairs_with_embeddings
from semsim.unfiltered import load_unfiltered_pairs_for_model


def get_pairs_dir_for_dataset(dataset: str) -> Path:
    """Return the pairs directory for a supported dataset key."""
    if dataset == "validated_3.0":
        return PAIRS_DIR
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_features_for_dataset(
    model_name: str,
    dataset: str,
    pair_types: List[str],
    input_type: str,
    symmetrize: bool,
    standardize: bool,
) -> Optional[Tuple[np.ndarray, List[str], List[str], List[str]]]:
    """Load and prepare features for the paper figures and cache sweeps."""
    if dataset == "unfiltered":
        data = load_unfiltered_pairs_for_model(model_name)
        return _assemble_feature_bundle(data, pair_types, input_type, symmetrize, standardize)

    pairs_dir = get_pairs_dir_for_dataset(dataset)
    slug = get_model_slug(model_name)
    data = {}
    for pair_type in pair_types:
        pair_path = pairs_dir / f"{slug}_{pair_type}_pairs.json"
        if not pair_path.exists():
            continue
        pairs, emb_pairs, meta = load_pairs_with_embeddings(pair_path)
        data[pair_type] = (pairs, emb_pairs, meta, [])

    return _assemble_feature_bundle(data, pair_types, input_type, symmetrize, standardize)


def _assemble_feature_bundle(data, pair_types, input_type, symmetrize, standardize):
    all_labels: List[str] = []
    all_words1: List[str] = []
    all_words2: List[str] = []
    all_vecs = []

    for pair_type in pair_types:
        if pair_type not in data:
            continue
        pairs, emb_pairs, _meta, _splits = data[pair_type]
        X = build_features(emb_pairs, input_type)
        if len(X) == 0:
            continue
        if symmetrize:
            X = symmetrize_features(X, input_type)
            repeated_pairs = list(pairs) + list(pairs)
            for word1, word2 in repeated_pairs:
                all_words1.append(word1)
                all_words2.append(word2)
            all_labels.extend([pair_type] * len(X))
        else:
            for word1, word2 in pairs:
                all_words1.append(word1)
                all_words2.append(word2)
            all_labels.extend([pair_type] * len(X))
        all_vecs.append(X)

    if not all_vecs:
        return None

    X_all = np.vstack(all_vecs)
    if standardize:
        X_all = StandardScaler().fit_transform(X_all)

    return X_all, all_labels, all_words1, all_words2


#: Row-block order of the paper's projections (pair types sorted alphabetically).
PAPER_PAIR_TYPES = ["antonyms", "shuffled_antonym_words", "shuffled_synonym_words", "synonyms"]


def load_paper_bundle(model_name: str, input_type: str = "difference") -> dict:
    """Standardised features in the row order of the paper, with per-row metadata.

    Rows are stacked pair type by pair type in ``PAPER_PAIR_TYPES`` order, which
    is exactly what ``load_features_for_dataset(model, "unfiltered", ...)``
    feeds to UMAP for the paper figures.

    Returns a dict with ``X`` (float32, standardised), ``labels`` (pair type per
    row), ``splits`` and ``pos`` (Stuttgart split and part of speech, empty for
    shuffled pairs), ``pairs`` (word pairs) and ``cosine`` (cosine similarity of
    the two word embeddings).
    """
    data = load_unfiltered_pairs_for_model(model_name)
    parts, labels, splits, pos, pairs, cosine = [], [], [], [], [], []
    for pair_type in PAPER_PAIR_TYPES:
        type_pairs, emb_pairs, meta, split_labels = data[pair_type]
        n = len(type_pairs)
        parts.append(build_features(emb_pairs, input_type).astype(np.float32))
        labels.extend([pair_type] * n)
        splits.extend(split_labels if split_labels else [""] * n)
        pos.extend(meta.get("pos") or [""] * n)
        pairs.extend(type_pairs)
        for e1, e2 in emb_pairs:
            cosine.append(float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12)))
    X = StandardScaler().fit_transform(np.vstack(parts)).astype(np.float32)
    return {
        "X": X,
        "labels": np.array(labels),
        "splits": np.array(splits),
        "pos": np.array(pos),
        "pairs": pairs,
        "cosine": np.array(cosine, dtype=np.float32),
    }
