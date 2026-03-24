"""Configuration for the shareable paper reproduction workflow."""

from pathlib import Path
from typing import Dict, List

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

STUTTGART_DIR = DATA_DIR / "stuttgart"

EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
OPENAI_EMBEDDINGS_DIR = DATA_DIR / "OpenAI_embeddings"
PAIRS_DIR = RESULTS_DIR / "pairs"
FILTERED_PAIRS_DIR = RESULTS_DIR / "filtered_pairs"
STATISTICS_DIR = RESULTS_DIR / "statistics"
CLASSIFICATION_DIR = RESULTS_DIR / "classification"
PROJECTIONS_DIR = RESULTS_DIR / "projections"
VALIDATION_DIR = RESULTS_DIR / "validation"
IMAGES_DIR = RESULTS_DIR / "images"

STREAMLIT_CACHE_DIR = DATA_DIR / "streamlit_image_cache"
STREAMLIT_PROJECTION_CACHE_DIR = STREAMLIT_CACHE_DIR / "projections"
STREAMLIT_GRID_CACHE_DIR = STREAMLIT_CACHE_DIR / "grids"

# =============================================================================
# Models
# =============================================================================

MODELS: List[str] = [
    "bert-base-cased",
    "word2vec",
    "glove",
    "text-embedding-3-small",
    "text-embedding-3-large",
]

MODEL_SLUGS: Dict[str, str] = {
    "bert-base-cased": "bert_base_cased",
    "word2vec": "word2vec",
    "glove": "glove",
    "text-embedding-3-small": "text_embedding_3_small",
    "text-embedding-3-large": "text_embedding_3_large",
}

TRANSFORMER_MODELS = ["bert-base-cased"]
STATIC_MODELS = ["word2vec", "glove"]
OPENAI_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]

# =============================================================================
# Pair Types
# =============================================================================

PAIR_TYPES: List[str] = ["synonyms", "antonyms", "shuffled_synonym_words", "shuffled_antonym_words"]
SEMANTIC_PAIR_TYPES: List[str] = ["synonyms", "antonyms"]
BASELINE_PAIR_TYPES: List[str] = ["shuffled_synonym_words", "shuffled_antonym_words"]

PAIR_GENERATION_CONFIG = {
    "random_state": 42,
}

# =============================================================================
# Display Names (for UI labels and plot titles)
# =============================================================================

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "bert-base-cased": "BERT Base",
    "word2vec": "Word2Vec",
    "glove": "GloVe",
    "text-embedding-3-small": "OpenAI Small (1536d)",
    "text-embedding-3-large": "OpenAI Large (3072d)",
}

METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "cosine_sim": "Cosine Similarity",
    "euclidean_dist": "Euclidean Distance",
    "dot_product": "Dot Product",
    "norm1": "Norm (Word 1)",
    "norm2": "Norm (Word 2)",
    "diff_norm": "Difference Norm",
}

PAIR_TYPE_DISPLAY_NAMES: Dict[str, str] = {
    "synonyms": "Synonyms",
    "antonyms": "Antonyms",
    "shuffled_synonym_words": "Shuffled Synonym Words",
    "shuffled_antonym_words": "Shuffled Antonym Words",
}

PROJECTION_DISPLAY_NAMES: Dict[str, str] = {
    "pca": "PCA",
    "tsne": "t-SNE",
    "umap": "UMAP",
}

INPUT_TYPE_DISPLAY_NAMES: Dict[str, str] = {
    "difference": "Difference (w2 - w1)",
    "concatenation": "Concatenation (w1 || w2)",
    "difference_sym": "Difference symmetric (w2-w1 & w1-w2)",
    "concatenation_sym": "Concatenation symmetric (w1||w2 & w2||w1)",
}

# =============================================================================
# Projections
# =============================================================================

PROJECTION_DIMS: List[int] = [2, 3]

PROJECTION_CONFIG = {
    "methods": ["pca", "tsne", "umap"],
    "input_types": ["difference", "concatenation"],
    "standardize": True,
    "pca": {"max_components": 50},
    "tsne": {"perplexity": 30, "max_iter": 1000, "pca_dims": 50},
    "umap": {"n_neighbors": 50, "min_dist": 0.01},
}

# =============================================================================
# Utilities
# =============================================================================


def get_model_slug(model_name: str) -> str:
    return MODEL_SLUGS.get(model_name, model_name.replace("/", "_").replace("-", "_"))


def get_embedding_path(model_name: str) -> Path:
    slug = get_model_slug(model_name)
    if model_name in OPENAI_MODELS:
        # Prefer .npz (compressed) if it exists, fall back to .pkl
        npz_path = OPENAI_EMBEDDINGS_DIR / slug / f"{slug}_embeddings.npz"
        if npz_path.exists():
            return npz_path
        return OPENAI_EMBEDDINGS_DIR / slug / f"{slug}_embeddings.pkl"
    return EMBEDDINGS_DIR / f"{slug}_embeddings.pkl"


def get_pair_path(model_name: str, pair_type: str) -> Path:
    return PAIRS_DIR / f"{get_model_slug(model_name)}_{pair_type}_pairs.json"


def get_projection_path(model_name: str, input_type: str, method: str, n_dims: int = 2) -> Path:
    return PROJECTIONS_DIR / f"{get_model_slug(model_name)}_{input_type}_{method}_{n_dims}d.npz"


def ensure_directories():
    """Create only the directories used by the shareable paper pipeline."""
    for d in [
        EMBEDDINGS_DIR,
        PAIRS_DIR,
        FILTERED_PAIRS_DIR,
        STATISTICS_DIR,
        CLASSIFICATION_DIR,
        PROJECTIONS_DIR,
        VALIDATION_DIR,
        IMAGES_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
