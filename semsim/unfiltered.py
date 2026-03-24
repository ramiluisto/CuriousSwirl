"""Load unfiltered Stuttgart data with embeddings for the Streamlit app.

Provides pair data without LLM validation filtering, using the raw Stuttgart
dataset directly. Pairs are filtered only by embedding availability (both words
must have embeddings for the selected model).
"""

import logging
from typing import Dict, List, Set, Tuple

import numpy as np

from config import (
    STUTTGART_DIR,
    get_embedding_path,
)
from semsim.data import load_stuttgart, get_semantic_pairs
from semsim.embeddings import load_embeddings
from semsim.pairs import generate_shuffled_pairs, get_all_semantic_pairs

logger = logging.getLogger(__name__)

PairList = List[Tuple[str, str]]


def load_unfiltered_pairs_for_model(
    model_name: str,
) -> Dict[str, Tuple[PairList, List[Tuple[np.ndarray, np.ndarray]], Dict, List[str]]]:
    """Load unfiltered Stuttgart pairs with embeddings for a model.

    Returns:
        Dict mapping pair_type -> (pairs, emb_pairs, metadata, split_labels).
        split_labels is a list of "train"/"val"/"test" parallel to pairs
        (empty list for shuffled pair types).
    """
    # Load Stuttgart data
    df = load_stuttgart(STUTTGART_DIR)

    # Load embeddings
    emb_path = get_embedding_path(model_name)
    embeddings = load_embeddings(emb_path)

    result: Dict[str, Tuple] = {}

    # Process synonyms and antonyms
    for relation, pair_type in [("synonym", "synonyms"), ("antonym", "antonyms")]:
        filtered = df[df["relation"] == relation]
        pairs: PairList = []
        emb_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        split_labels: List[str] = []

        for _, row in filtered.iterrows():
            w1, w2 = str(row["word1"]), str(row["word2"])
            if w1 in embeddings and w2 in embeddings:
                pairs.append((w1, w2))
                emb_pairs.append((embeddings[w1], embeddings[w2]))
                split_labels.append(str(row["split"]))

        metadata = {
            "pair_type": pair_type,
            "model": model_name,
            "n_pairs": len(pairs),
            "source": "unfiltered_stuttgart",
        }

        result[pair_type] = (pairs, emb_pairs, metadata, split_labels)
        logger.info(
            "Loaded %d unfiltered %s pairs for %s (from %d Stuttgart rows)",
            len(pairs), pair_type, model_name, len(filtered),
        )

    # Generate shuffled pairs
    syn_pairs = result.get("synonyms", ([], [], {}, []))[0]
    ant_pairs = result.get("antonyms", ([], [], {}, []))[0]
    exclusion_set = get_all_semantic_pairs(syn_pairs, ant_pairs)

    for source_type, shuffled_type in [
        ("synonyms", "shuffled_synonym_words"),
        ("antonyms", "shuffled_antonym_words"),
    ]:
        source_pairs = result.get(source_type, ([], [], {}, []))[0]
        if not source_pairs:
            continue

        shuffled = generate_shuffled_pairs(source_pairs, exclusion_set, random_state=42)

        # Build emb_pairs for shuffled
        shuffled_emb_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        valid_shuffled: PairList = []
        for w1, w2 in shuffled:
            if w1 in embeddings and w2 in embeddings:
                valid_shuffled.append((w1, w2))
                shuffled_emb_pairs.append((embeddings[w1], embeddings[w2]))

        metadata = {
            "pair_type": shuffled_type,
            "model": model_name,
            "n_pairs": len(valid_shuffled),
            "source": "unfiltered_stuttgart_shuffled",
        }

        result[shuffled_type] = (valid_shuffled, shuffled_emb_pairs, metadata, [])

    return result
