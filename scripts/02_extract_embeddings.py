#!/usr/bin/env python3
"""Phase 2: Extract embeddings from all model families."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODELS, TRANSFORMER_MODELS, STATIC_MODELS, OPENAI_MODELS,
    STUTTGART_DIR, RESULTS_DIR,
    ensure_directories, get_embedding_path,
)
from semsim.data import load_stuttgart, get_vocabulary
from semsim.embeddings import (
    extract_transformer_embeddings,
    extract_word2vec_embeddings,
    extract_glove_embeddings,
    save_embeddings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from models")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to extract")
    parser.add_argument("--device", default="cpu", help="Device for transformers")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output exists")
    parser.add_argument("--glove-path", type=str, default=None, help="Path to glove.6B.300d.txt")
    args = parser.parse_args()

    ensure_directories()
    logger.info("=== Phase 2: Embedding Extraction ===")

    # Load vocabulary
    df = load_stuttgart(STUTTGART_DIR)
    vocab = get_vocabulary(df)
    logger.info("Vocabulary: %d words", len(vocab))

    for model_name in args.models:
        out_path = get_embedding_path(model_name)

        if args.skip_existing and out_path.exists():
            logger.info("Skipping %s (already exists)", model_name)
            continue

        logger.info("Extracting embeddings for %s...", model_name)

        if model_name in OPENAI_MODELS:
            logger.info(
                "OpenAI model %s uses pre-extracted embeddings. "
                "Run scripts/extract_openai_embeddings.py if needed.",
                model_name,
            )
            continue
        elif model_name in TRANSFORMER_MODELS:
            embs = extract_transformer_embeddings(model_name, vocab, device=args.device)
        elif model_name == "word2vec":
            embs = extract_word2vec_embeddings(vocab)
        elif model_name == "glove":
            glove_path = Path(args.glove_path) if args.glove_path else None
            embs = extract_glove_embeddings(vocab, glove_path=glove_path)
        else:
            logger.warning("Unknown model: %s", model_name)
            continue

        save_embeddings(embs, out_path)


if __name__ == "__main__":
    main()
