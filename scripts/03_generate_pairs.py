#!/usr/bin/env python3
"""Phase 3: Generate word pairs (synonyms, antonyms, shuffled_synonym_words, shuffled_antonym_words).

Loads BERT-validated canonical pairs, filters to each model's embeddings, and
generates shuffled baselines from the same word populations.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODELS, VALIDATION_DIR, PAIR_GENERATION_CONFIG,
    ensure_directories, get_embedding_path, get_pair_path,
)
from semsim.embeddings import load_embeddings
from semsim.pairs import (
    generate_synonym_pairs,
    generate_shuffled_pairs,
    get_all_semantic_pairs,
    save_pairs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_validated_pairs(claimed_type: str):
    """Load validated pairs from Phase 1b output."""
    path = VALIDATION_DIR / f"{claimed_type}s_validated.json"
    if not path.exists():
        logger.error("Validated pairs not found: %s (run 01b filter first)", path)
        sys.exit(1)
    with open(path) as f:
        pairs = json.load(f)
    return [tuple(p) for p in pairs]


def main():
    parser = argparse.ArgumentParser(description="Generate word pairs")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    ensure_directories()
    logger.info("=== Phase 3: Pair Generation ===")

    # Load BERT-validated canonical pairs
    canonical_synonyms = _load_validated_pairs("synonym")
    canonical_antonyms = _load_validated_pairs("antonym")
    logger.info("Canonical validated pairs: %d synonyms, %d antonyms",
                len(canonical_synonyms), len(canonical_antonyms))

    for model_name in args.models:
        emb_path = get_embedding_path(model_name)
        if not emb_path.exists():
            logger.warning("Embeddings not found for %s, skipping", model_name)
            continue

        embeddings = load_embeddings(emb_path)
        logger.info("Model %s: %d embeddings", model_name, len(embeddings))

        # Filter canonical pairs to this model's vocabulary
        syn_pairs = [(w1, w2) for w1, w2 in canonical_synonyms
                     if w1 in embeddings and w2 in embeddings]
        ant_pairs = [(w1, w2) for w1, w2 in canonical_antonyms
                     if w1 in embeddings and w2 in embeddings]
        logger.info("  %s: %d synonyms, %d antonyms after embedding filter",
                     model_name, len(syn_pairs), len(ant_pairs))

        # Save synonyms
        syn_path = get_pair_path(model_name, "synonyms")
        if not (args.skip_existing and syn_path.exists()):
            save_pairs(syn_pairs, embeddings, syn_path, "synonyms", model_name)

        # Save antonyms
        ant_path = get_pair_path(model_name, "antonyms")
        if not (args.skip_existing and ant_path.exists()):
            save_pairs(ant_pairs, embeddings, ant_path, "antonyms", model_name)

        # Build exclusion set from all semantic pairs
        semantic_set = get_all_semantic_pairs(syn_pairs, ant_pairs)

        # Shuffled synonym words (same word pool as synonyms)
        ssw_path = get_pair_path(model_name, "shuffled_synonym_words")
        if not (args.skip_existing and ssw_path.exists()):
            ssw_pairs = generate_shuffled_pairs(
                syn_pairs, semantic_set,
                random_state=PAIR_GENERATION_CONFIG["random_state"],
            )
            save_pairs(ssw_pairs, embeddings, ssw_path, "shuffled_synonym_words", model_name)

        # Shuffled antonym words (same word pool as antonyms)
        saw_path = get_pair_path(model_name, "shuffled_antonym_words")
        if not (args.skip_existing and saw_path.exists()):
            saw_pairs = generate_shuffled_pairs(
                ant_pairs, semantic_set,
                random_state=PAIR_GENERATION_CONFIG["random_state"] + 1,
            )
            save_pairs(saw_pairs, embeddings, saw_path, "shuffled_antonym_words", model_name)

        logger.info("Pair generation complete for %s", model_name)


if __name__ == "__main__":
    main()
