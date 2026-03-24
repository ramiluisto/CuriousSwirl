#!/usr/bin/env python3
"""Run the full pipeline: data preparation through projections.

Pre-computed data shipped with this repository:
  - LLM validation results (results/validation/) — skips the expensive 01b step
  - OpenAI embeddings (data/OpenAI_embeddings/) — skips OpenAI API calls

Phase 02 will download BERT (~400 MB), word2vec (~1.7 GB) on first run.
GloVe requires a manual download — see README.md.

After this pipeline completes, generate paper figures with:
  python scripts/generate_paper_figures.py
  python scripts/generate_cosine_sim_plots.py
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent
SCRIPTS = [
    "01_prepare_data.py",
    "01a_filter_single_token.py",
    # 01b (LLM validation) is pre-computed — results shipped in results/validation/
    "02_extract_embeddings.py",
    "03_generate_pairs.py",
    "04_compute_statistics.py",
    "06_compute_projections.py",
]


def main():
    extra_args = sys.argv[1:]  # Pass through any extra args
    logger.info("=== Running Full Pipeline ===")

    for script_name in SCRIPTS:
        script_path = SCRIPTS_DIR / script_name
        logger.info("--- Running %s ---", script_name)

        cmd = [sys.executable, str(script_path)] + extra_args
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            logger.error("Script %s failed with exit code %d", script_name, result.returncode)
            sys.exit(result.returncode)

    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
