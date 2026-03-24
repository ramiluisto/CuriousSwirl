#!/usr/bin/env python3
"""Phase 1a: Filter Stuttgart pairs to BERT single-token vocabulary.

Produces the canonical unvalidated pair sets that feed into LLM validation (Phase 1b).
Output: results/filtered_pairs/{single_token_vocabulary,synonyms_unvalidated,antonyms_unvalidated}.json
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import STUTTGART_DIR, FILTERED_PAIRS_DIR, ensure_directories
from semsim.data import (
    load_stuttgart,
    get_vocabulary,
    get_semantic_pairs,
    get_single_token_vocabulary,
    filter_pairs_to_vocabulary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    ensure_directories()
    logger.info("=== Phase 1a: Filter to BERT Single-Token Pairs ===")

    # Load Stuttgart
    df = load_stuttgart(STUTTGART_DIR)
    if df.empty:
        logger.error("No data loaded from Stuttgart. Check %s", STUTTGART_DIR)
        sys.exit(1)

    # Full vocabulary
    vocab = get_vocabulary(df)
    logger.info("Stuttgart vocabulary: %d unique words", len(vocab))

    # Filter to BERT single-token words
    single_token_vocab = get_single_token_vocabulary(vocab, model_name="bert-base-cased")

    # Save vocabulary
    vocab_path = FILTERED_PAIRS_DIR / "single_token_vocabulary.json"
    with open(vocab_path, "w") as f:
        json.dump(sorted(single_token_vocab), f, indent=2)
    logger.info("Saved single-token vocabulary (%d words) to %s", len(single_token_vocab), vocab_path)

    # Extract and filter synonym pairs
    all_synonyms = get_semantic_pairs(df, "synonym")
    filtered_synonyms = filter_pairs_to_vocabulary(all_synonyms, single_token_vocab)
    syn_path = FILTERED_PAIRS_DIR / "synonyms_unvalidated.json"
    with open(syn_path, "w") as f:
        json.dump(filtered_synonyms, f, indent=2)
    logger.info("Synonyms: %d -> %d after single-token filter", len(all_synonyms), len(filtered_synonyms))

    # Extract and filter antonym pairs
    all_antonyms = get_semantic_pairs(df, "antonym")
    filtered_antonyms = filter_pairs_to_vocabulary(all_antonyms, single_token_vocab)
    ant_path = FILTERED_PAIRS_DIR / "antonyms_unvalidated.json"
    with open(ant_path, "w") as f:
        json.dump(filtered_antonyms, f, indent=2)
    logger.info("Antonyms: %d -> %d after single-token filter", len(all_antonyms), len(filtered_antonyms))

    # POS breakdown of surviving pairs
    logger.info("--- POS breakdown of filtered pairs ---")
    for relation, pairs in [("synonym", filtered_synonyms), ("antonym", filtered_antonyms)]:
        # Find POS for each pair from the original dataframe
        pair_set = {(w1, w2) for w1, w2 in pairs}
        mask = df["relation"] == relation
        matched = df[mask & df.apply(lambda r: (r["word1"], r["word2"]) in pair_set, axis=1)]
        pos_counts = Counter(matched["pos"])
        total = sum(pos_counts.values())
        logger.info("  %s (%d pairs):", relation, len(pairs))
        for pos, count in sorted(pos_counts.items()):
            logger.info("    %s: %d (%.1f%%)", pos, count, 100.0 * count / total if total else 0)

    logger.info("=== Phase 1a Complete ===")


if __name__ == "__main__":
    main()
