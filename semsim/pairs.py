"""
Word pair generation for 4 pair types:
- synonyms: from Stuttgart dataset
- antonyms: from Stuttgart + curated sources, merged & deduplicated
- random_noun_pairs: random pairs from Stuttgart nouns
- random_pairs: random pairs from all embedded words
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PairList = List[Tuple[str, str]]


def generate_synonym_pairs(
    stuttgart_pairs: PairList,
    embeddings: Dict[str, np.ndarray],
) -> PairList:
    """Filter Stuttgart synonym pairs to those with embeddings."""
    pairs = [(w1, w2) for w1, w2 in stuttgart_pairs
             if w1 in embeddings and w2 in embeddings]
    logger.info("Generated %d synonym pairs (from %d Stuttgart)",
                len(pairs), len(stuttgart_pairs))
    return pairs


def generate_antonym_pairs(
    stuttgart_pairs: PairList,
    curated_pairs: PairList,
    embeddings: Dict[str, np.ndarray],
) -> PairList:
    """Merge Stuttgart + curated antonym pairs, deduplicate, filter to embedded words."""
    seen: Set[Tuple[str, str]] = set()
    merged: PairList = []

    for w1, w2 in stuttgart_pairs + curated_pairs:
        if w1 not in embeddings or w2 not in embeddings:
            continue
        canonical = tuple(sorted((w1, w2)))
        if canonical not in seen:
            seen.add(canonical)
            merged.append((w1, w2))

    logger.info("Generated %d antonym pairs (Stuttgart=%d, curated=%d, after dedup+filter)",
                len(merged), len(stuttgart_pairs), len(curated_pairs))
    return merged


def generate_shuffled_pairs(
    original_pairs: PairList,
    semantic_exclusion_set: Set[Tuple[str, str]],
    random_state: int = 42,
) -> PairList:
    """Generate random re-pairings from the same word population as original_pairs.

    Pools all unique words from original_pairs and randomly re-pairs them to produce
    the same number of pairs. Excludes any pair in the semantic_exclusion_set.
    This guarantees the same word population as the semantic type -- no vocabulary confounder.

    Args:
        original_pairs: The semantic pairs whose words form the pool.
        semantic_exclusion_set: Set of canonical (sorted) pair tuples to exclude.
        random_state: Random seed for reproducibility.

    Returns:
        List of (word1, word2) tuples with the same count as original_pairs.
    """
    words = sorted({w for pair in original_pairs for w in pair})
    n_target = len(original_pairs)

    if len(words) < 2:
        logger.warning("Not enough words for shuffled pairs")
        return []

    rng = random.Random(random_state)
    pairs: PairList = []
    seen: Set[Tuple[str, str]] = set()
    attempts = 0
    max_attempts = n_target * 50

    while len(pairs) < n_target and attempts < max_attempts:
        w1, w2 = rng.sample(words, 2)
        canonical = tuple(sorted((w1, w2)))
        if canonical not in semantic_exclusion_set and canonical not in seen:
            seen.add(canonical)
            pairs.append((w1, w2))
        attempts += 1

    logger.info("Generated %d/%d shuffled pairs from %d unique words",
                len(pairs), n_target, len(words))
    return pairs


def generate_random_noun_pairs(
    nouns: Set[str],
    embeddings: Dict[str, np.ndarray],
    semantic_pairs: Set[Tuple[str, str]],
    n: int = 2000,
    random_state: int = 42,
) -> PairList:
    """Generate random pairs from noun words, excluding known semantic pairs."""
    noun_list = sorted(nouns & set(embeddings.keys()))
    if len(noun_list) < 2:
        logger.warning("Not enough nouns with embeddings for random pairs")
        return []

    rng = random.Random(random_state)
    pairs: PairList = []
    attempts = 0
    max_attempts = n * 20

    while len(pairs) < n and attempts < max_attempts:
        w1, w2 = rng.sample(noun_list, 2)
        canonical = tuple(sorted((w1, w2)))
        if canonical not in semantic_pairs:
            pairs.append((w1, w2))
            semantic_pairs.add(canonical)  # Prevent duplicates within this run
        attempts += 1

    logger.info("Generated %d/%d random noun pairs", len(pairs), n)
    return pairs


def generate_random_pairs(
    embeddings: Dict[str, np.ndarray],
    semantic_pairs: Set[Tuple[str, str]],
    n: int = 2000,
    random_state: int = 42,
) -> PairList:
    """Generate random pairs from all embedded words."""
    word_list = sorted(embeddings.keys())
    if len(word_list) < 2:
        logger.warning("Not enough words for random pairs")
        return []

    rng = random.Random(random_state)
    pairs: PairList = []
    attempts = 0
    max_attempts = n * 20

    while len(pairs) < n and attempts < max_attempts:
        w1, w2 = rng.sample(word_list, 2)
        canonical = tuple(sorted((w1, w2)))
        if canonical not in semantic_pairs:
            pairs.append((w1, w2))
            semantic_pairs.add(canonical)
        attempts += 1

    logger.info("Generated %d/%d random pairs", len(pairs), n)
    return pairs


def get_all_semantic_pairs(
    synonym_pairs: PairList, antonym_pairs: PairList
) -> Set[Tuple[str, str]]:
    """Collect all semantic pairs (both directions) for exclusion from baselines."""
    seen: Set[Tuple[str, str]] = set()
    for w1, w2 in synonym_pairs + antonym_pairs:
        seen.add(tuple(sorted((w1, w2))))
    return seen


# ---- I/O ----

def save_pairs(pairs: PairList, embeddings: Dict[str, np.ndarray],
               path: Path, pair_type: str, model_name: str) -> None:
    """Save pairs with their embeddings to JSON."""
    data = {
        "pairs": [
            (w1, w2, embeddings[w1].tolist(), embeddings[w2].tolist())
            for w1, w2 in pairs
            if w1 in embeddings and w2 in embeddings
        ],
        "metadata": {
            "pair_type": pair_type,
            "model_name": model_name,
            "n_pairs": len(pairs),
            "embedding_dim": next(iter(embeddings.values())).shape[0] if embeddings else 0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    logger.info("Saved %d %s pairs for %s to %s", len(pairs), pair_type, model_name, path)


def load_pairs(path: Path) -> Tuple[PairList, Dict]:
    """Load pairs from JSON file.

    Returns:
        pairs: List of (word1, word2) tuples.
        metadata: Dict with pair_type, model_name, etc.
    """
    with open(path) as f:
        data = json.load(f)
    pairs = [(p[0], p[1]) for p in data["pairs"]]
    return pairs, data.get("metadata", {})


def load_pairs_with_embeddings(path: Path) -> Tuple[PairList, List[Tuple[np.ndarray, np.ndarray]], Dict]:
    """Load pairs with their stored embeddings.

    Returns:
        pairs: List of (word1, word2).
        emb_pairs: List of (emb1, emb2) arrays.
        metadata: Metadata dict.
    """
    with open(path) as f:
        data = json.load(f)
    pairs = [(p[0], p[1]) for p in data["pairs"]]
    emb_pairs = [(np.array(p[2]), np.array(p[3])) for p in data["pairs"]]
    return pairs, emb_pairs, data.get("metadata", {})
