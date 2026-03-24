"""
Stuttgart dataset processing and data loading utilities.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

STUTTGART_FILES = [
    "adjectives.train", "adjectives.test", "adjectives.val",
    "nouns.train", "nouns.test", "nouns.val",
    "verbs.train", "verbs.test", "verbs.val",
]


def load_stuttgart(stuttgart_dir: Path) -> pd.DataFrame:
    """Load all Stuttgart dataset files into a single DataFrame.

    Each file is tab-separated with columns: word1, word2, label
    where label 0 = synonym, 1 = antonym.

    Returns DataFrame with columns: word1, word2, relation, pos, split.
    """
    records: List[Dict] = []
    for filename in STUTTGART_FILES:
        filepath = stuttgart_dir / filename
        if not filepath.exists():
            logger.warning("Stuttgart file not found: %s", filepath)
            continue

        # Parse POS and split from filename (e.g. "nouns.train")
        parts = filename.split(".")
        pos, split = parts[0], parts[1]

        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fields = line.split("\t")
                if len(fields) != 3:
                    continue
                w1, w2, lbl = fields
                relation = "synonym" if lbl == "0" else "antonym"
                records.append({
                    "word1": w1,
                    "word2": w2,
                    "relation": relation,
                    "pos": pos,
                    "split": split,
                })

    df = pd.DataFrame(records, columns=["word1", "word2", "relation", "pos", "split"])
    logger.info("Loaded Stuttgart dataset: %d pairs (%d synonyms, %d antonyms)",
                len(df),
                (df["relation"] == "synonym").sum(),
                (df["relation"] == "antonym").sum())
    return df


def get_vocabulary(df: pd.DataFrame) -> Set[str]:
    """Extract unique words from Stuttgart DataFrame."""
    if df.empty:
        return set()
    return set(df["word1"]).union(set(df["word2"]))


def get_nouns(df: pd.DataFrame) -> Set[str]:
    """Extract unique noun words from Stuttgart DataFrame."""
    if df.empty:
        return set()
    noun_df = df[df["pos"] == "nouns"]
    return set(noun_df["word1"]).union(set(noun_df["word2"]))


def get_semantic_pairs(df: pd.DataFrame, relation: str) -> List[Tuple[str, str]]:
    """Get word pairs of a specific relation type.

    Args:
        df: Stuttgart DataFrame.
        relation: "synonym" or "antonym".

    Returns:
        List of (word1, word2) tuples.
    """
    filtered = df[df["relation"] == relation]
    pairs = list(zip(filtered["word1"], filtered["word2"]))
    logger.info("Found %d %s pairs in Stuttgart", len(pairs), relation)
    return pairs


def get_single_token_vocabulary(
    vocabulary: Set[str], model_name: str = "bert-base-cased"
) -> Set[str]:
    """Filter vocabulary to words that have single-token representations.

    Uses the same _get_single_token_ids logic from semsim.embeddings to test
    each word for single-token tokenization under the given model.

    Args:
        vocabulary: Full word set to filter.
        model_name: HuggingFace model name whose tokenizer defines the filter.

    Returns:
        Subset of vocabulary with at least one single-token representation.
    """
    from transformers import AutoTokenizer
    from semsim.embeddings import _get_single_token_ids

    logger.info("Loading tokenizer for %s to filter single-token vocabulary...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    single_token_words: Set[str] = set()
    for word in sorted(vocabulary):
        token_ids = _get_single_token_ids(word, tokenizer, model_name)
        if token_ids:
            single_token_words.add(word)

    logger.info(
        "Single-token filter (%s): %d/%d words retained (%.1f%%)",
        model_name, len(single_token_words), len(vocabulary),
        100.0 * len(single_token_words) / len(vocabulary) if vocabulary else 0,
    )
    return single_token_words


def filter_pairs_to_vocabulary(
    pairs: List[Tuple[str, str]], vocabulary: Set[str]
) -> List[Tuple[str, str]]:
    """Keep only pairs where both words are in the vocabulary.

    Args:
        pairs: List of (word1, word2) tuples.
        vocabulary: Allowed word set.

    Returns:
        Filtered list of pairs.
    """
    filtered = [(w1, w2) for w1, w2 in pairs if w1 in vocabulary and w2 in vocabulary]
    logger.info("Filtered pairs to vocabulary: %d/%d retained", len(filtered), len(pairs))
    return filtered


def load_curated_antonyms(opposites_dir: Path) -> List[Tuple[str, str]]:
    """Load curated antonym pairs from the opposites directory.

    Reads CSV files with _word and _opposite columns.
    """
    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    for csv_path in sorted(opposites_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning("Failed to read %s: %s", csv_path, e)
            continue

        if "_word" in df.columns and "_opposite" in df.columns:
            for _, row in df.iterrows():
                pair = (str(row["_word"]), str(row["_opposite"]))
                canonical = tuple(sorted(pair))
                if canonical not in seen:
                    seen.add(canonical)
                    pairs.append(pair)

    logger.info("Loaded %d curated antonym pairs from %s", len(pairs), opposites_dir)
    return pairs
