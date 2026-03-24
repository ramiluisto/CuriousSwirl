"""Embedding loading and extraction for the paper's model families."""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


def get_default_glove_path() -> Optional[Path]:
    """Return the first supported GloVe path if available."""
    candidates = [
        Path.home() / ".cache" / "glove" / "glove.6B.300d.txt",
        Path("data") / "glove" / "glove.6B.300d.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_embeddings(path: Path) -> Dict[str, np.ndarray]:
    """Load embeddings from a pickle or compressed npz file."""
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        words = data["words"].tolist()
        vectors = data["vectors"]
        embs = {w: vectors[i] for i, w in enumerate(words)}
    else:
        with open(path, "rb") as f:
            embs = pickle.load(f)
    logger.info("Loaded %d embeddings from %s (dim=%d)",
                len(embs), path.name,
                next(iter(embs.values())).shape[0] if embs else 0)
    return embs


def save_embeddings(embeddings: Dict[str, np.ndarray], path: Path) -> None:
    """Save embeddings dict to pickle and write metadata JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)

    dim = next(iter(embeddings.values())).shape[0] if embeddings else 0
    meta = {"n_words": len(embeddings), "embedding_dim": dim}
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved %d embeddings (dim=%d) to %s", len(embeddings), dim, path)


# ---- Transformer models ----

def extract_transformer_embeddings(
    model_name: str,
    vocabulary: Set[str],
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Extract context-free token embeddings from a HuggingFace transformer.

    For each word, finds single-token representations across casing variants,
    averages them to produce one embedding vector per word.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading transformer model %s on %s...", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Get the embedding weight matrix
    emb_matrix = _get_embedding_matrix(model)
    logger.info("Embedding matrix shape: %s", emb_matrix.shape)

    embeddings: Dict[str, np.ndarray] = {}
    skipped: List[str] = []

    for i, word in enumerate(sorted(vocabulary), 1):
        if i % 500 == 0:
            logger.info("Progress: %d/%d words", i, len(vocabulary))

        token_ids = _get_single_token_ids(word, tokenizer, model_name)
        if not token_ids:
            skipped.append(word)
            continue

        vecs = [emb_matrix[tid] for tid in token_ids if tid < len(emb_matrix)]
        if vecs:
            embeddings[word] = np.mean(vecs, axis=0)
        else:
            skipped.append(word)

    logger.info("Extracted %d embeddings, skipped %d words", len(embeddings), len(skipped))
    return embeddings


def _get_embedding_matrix(model) -> np.ndarray:
    """Extract the token embedding weight matrix from a transformer model."""
    emb_layer = model.embeddings
    for name in ("word_embeddings", "tok_embeddings", "token_embeddings"):
        layer = getattr(emb_layer, name, None)
        if layer is not None and hasattr(layer, "weight"):
            return layer.weight.detach().cpu().numpy()
    # Fallback: search for any 2D weight
    for attr in dir(emb_layer):
        obj = getattr(emb_layer, attr)
        if hasattr(obj, "weight"):
            w = obj.weight.detach().cpu().numpy()
            if w.ndim == 2:
                return w
    raise RuntimeError(f"Could not find embedding matrix in model")


def _get_single_token_ids(word: str, tokenizer, model_name: str) -> List[int]:
    """Find token IDs for single-token variants of a word."""
    variants = {word, word.lower(), word.capitalize(), word.upper()}
    name_lower = model_name.lower()

    if "modernbert" in name_lower:
        for prefix in ("Ġ", "▁"):
            variants.update(f"{prefix}{v}" for v in list(variants))
    elif "bert" in name_lower:
        variants.update(f"##{v}" for v in list(variants))

    token_ids = set()
    for var in variants:
        ids = tokenizer.encode(var, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.add(ids[0])

    return list(token_ids)


# ---- word2vec (gensim) ----

def extract_word2vec_embeddings(vocabulary: Set[str]) -> Dict[str, np.ndarray]:
    """Extract word2vec embeddings for vocabulary words.

    Downloads google-news-300 model via gensim (~1.7GB on first run).
    """
    import gensim.downloader as api

    logger.info("Loading word2vec-google-news-300 (may download ~1.7GB on first run)...")
    model = api.load("word2vec-google-news-300")
    logger.info("word2vec model loaded: %d words, dim=%d", len(model), model.vector_size)

    embeddings: Dict[str, np.ndarray] = {}
    for word in sorted(vocabulary):
        if word in model:
            embeddings[word] = model[word].astype(np.float32)

    logger.info("Extracted %d/%d word2vec embeddings", len(embeddings), len(vocabulary))
    return embeddings


# ---- GloVe (text file) ----

def extract_glove_embeddings(
    vocabulary: Set[str],
    glove_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Extract GloVe embeddings for vocabulary words.

    Args:
        vocabulary: Words to extract.
        glove_path: Path to glove.6B.300d.txt. If None, looks in standard locations.
    """
    if glove_path is None:
        glove_path = get_default_glove_path()
        if glove_path is None:
            raise FileNotFoundError(
                "GloVe file not found. Please download glove.6B.300d.txt and "
                "place it at data/glove/glove.6B.300d.txt, "
                "~/.cache/glove/glove.6B.300d.txt, or pass glove_path."
            )

    logger.info("Loading GloVe embeddings from %s...", glove_path)
    vocab_lower = {w.lower(): w for w in vocabulary}

    embeddings: Dict[str, np.ndarray] = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            token = parts[0]
            # Match against vocabulary (case-insensitive)
            if token in vocabulary:
                vec = np.array(parts[1:], dtype=np.float32)
                embeddings[token] = vec
            elif token in vocab_lower:
                original = vocab_lower[token]
                if original not in embeddings:
                    vec = np.array(parts[1:], dtype=np.float32)
                    embeddings[original] = vec

    logger.info("Extracted %d/%d GloVe embeddings", len(embeddings), len(vocabulary))
    return embeddings
