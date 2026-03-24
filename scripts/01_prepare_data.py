#!/usr/bin/env python3
"""Phase 1: Process Stuttgart dataset and extract vocabulary."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import STUTTGART_DIR, RESULTS_DIR, ensure_directories
from semsim.data import load_stuttgart, get_vocabulary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    ensure_directories()
    logger.info("=== Phase 1: Data Preparation ===")

    df = load_stuttgart(STUTTGART_DIR)
    if df.empty:
        logger.error("No data loaded from Stuttgart. Check %s", STUTTGART_DIR)
        sys.exit(1)

    # Save processed CSV
    csv_path = RESULTS_DIR / "stuttgart_processed.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved processed Stuttgart data to %s", csv_path)

    # Save vocabulary
    vocab = get_vocabulary(df)
    vocab_path = RESULTS_DIR / "vocabulary.txt"
    with open(vocab_path, "w") as f:
        for word in sorted(vocab):
            f.write(word + "\n")
    logger.info("Vocabulary: %d unique words saved to %s", len(vocab), vocab_path)

    # Summary
    for pos in ["adjectives", "nouns", "verbs"]:
        n = len(df[df["pos"] == pos])
        logger.info("  %s: %d pairs", pos, n)


if __name__ == "__main__":
    main()
