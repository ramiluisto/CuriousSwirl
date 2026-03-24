"""
Configuration for the Antonyms and Synonyms research project.
Defines paths, model configs, pair types, and experimental parameters.
"""

from pathlib import Path
from typing import Dict, List

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

STUTTGART_DIR = DATA_DIR / "stuttgart"
WORDNET_PATH = DATA_DIR / "wordnet_curated" / "wordnet_pruned_checked.csv"
OPPOSITES_DIR = DATA_DIR / "opposites"

EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
OPENAI_EMBEDDINGS_DIR = DATA_DIR / "OpenAI_embeddings"
PAIRS_DIR = RESULTS_DIR / "pairs"
FILTERED_PAIRS_DIR = RESULTS_DIR / "filtered_pairs"
STATISTICS_DIR = RESULTS_DIR / "statistics"
CLASSIFICATION_DIR = RESULTS_DIR / "classification"
PROJECTIONS_DIR = RESULTS_DIR / "projections"
CONFOUNDER_DIR = RESULTS_DIR / "confounder"
VALIDATION_DIR = RESULTS_DIR / "validation"
TAGGING_DIR = RESULTS_DIR / "tagging"
BASE_WORD_DIR = RESULTS_DIR / "base_word_analysis"
PERSISTENT_HOMOLOGY_DIR = RESULTS_DIR / "persistent_homology"
SYNTHETIC_SHAPES_DIR = RESULTS_DIR / "synthetic_shapes"
CLUSTER_CLASSIFICATION_DIR = RESULTS_DIR / "cluster_classification"
REPORTS_DIR = RESULTS_DIR / "reports"

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

PH_EXCLUDED_MODELS: set = set(OPENAI_MODELS)

# =============================================================================
# Pair Types
# =============================================================================

PAIR_TYPES: List[str] = ["synonyms", "antonyms", "shuffled_synonym_words", "shuffled_antonym_words"]
SEMANTIC_PAIR_TYPES: List[str] = ["synonyms", "antonyms"]
BASELINE_PAIR_TYPES: List[str] = ["shuffled_synonym_words", "shuffled_antonym_words"]

ANTONYM_PREFIXES: List[str] = [
    "un", "in", "im", "ir", "il", "dis", "non", "anti", "de", "mis",
]

PAIR_GENERATION_CONFIG = {
    "random_state": 42,
}

# =============================================================================
# Classification
# =============================================================================

CLASSIFICATION_CONFIG = {
    "input_types": ["difference", "concatenation", "difference_sym", "concatenation_sym"],
    "test_size": 0.2,
    "random_state": 42,
    "max_iter": 1000,
    "nn_hidden_dim": 64,
    "nn_epochs": 100,
    "nn_batch_size": 64,
    "nn_learning_rate": 1e-3,
    "nn_patience": 10,
}

CLASSIFICATION_TASKS = {
    "pairwise": [
        ("synonyms", "antonyms"),
        ("synonyms", "shuffled_synonym_words"),
        ("antonyms", "shuffled_antonym_words"),
        ("synonyms", "shuffled_antonym_words"),
        ("antonyms", "shuffled_synonym_words"),
        ("shuffled_synonym_words", "shuffled_antonym_words"),
    ],
    "multiclass": PAIR_TYPES,
}

# =============================================================================
# Plotting
# =============================================================================

PLOT_CONFIG = {
    "dpi": 300,
    "format": "png",
    "figsize": (10, 6),
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

HYPERPARAMETER_GRID_CONFIG = {
    "tsne": {
        "perplexity": [5, 15, 30, 50],
        "max_iter": [250, 500, 1000, 2000],
    },
    "umap": {
        "n_neighbors": [5, 15, 30, 50],
        "min_dist": [0.01, 0.1, 0.25, 0.5],
    },
}

PERSISTENT_HOMOLOGY_CONFIG = {
    "input_types": ["difference", "concatenation"],
    "n_landmarks": 300,
    "subsampling": "maxmin",   # "maxmin" or "random"
    "metric": "cosine",
    "max_dim": 2,              # H0, H1, and H2
    "seed": 42,
}

CLUSTER_CLASSIFICATION_CONFIG = {
    "input_types": ["difference"],
    "test_size": 0.2,
    "random_state": 42,
    "projections": [None, "umap_2d", "umap_3d"],
    "umap_defaults": {"n_neighbors": 15, "min_dist": 0.1},
    "cluster_methods": ["hdbscan", "kmeans"],
    "straggler_modes": ["nearest", "noise_class"],
    "hdbscan_defaults": {"min_cluster_size": 15, "min_samples": 5},
}

SYNTHETIC_SHAPES_CONFIG = {
    "default_n_points": 1000,
    "default_noise_sigma": 0.01,
    "default_seed": 42,
    "target_dimensions": [32, 64, 128, 512, 768, 1024],
    "default_target_dim": 768,
    "n_pairs_for_metrics": 2000,
    "max_recipe_depth": 4,
    "ph": {
        "n_landmarks": 300,
        "subsampling": "maxmin",
        "metric": "euclidean",
        "max_dim": 2,
    },
    "projections": {
        "pca_max_components": 50,
        "tsne_perplexity": 30,
        "umap_n_neighbors": 15,
    },
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


def get_ph_output_dir(model_name: str, pair_type: str, input_type: str) -> Path:
    slug = get_model_slug(model_name)
    return PERSISTENT_HOMOLOGY_DIR / slug / f"{pair_type}_{input_type}"


def get_synthetic_shape_dir(shape_name: str) -> Path:
    return SYNTHETIC_SHAPES_DIR / shape_name


def ensure_directories():
    for d in [EMBEDDINGS_DIR, PAIRS_DIR, FILTERED_PAIRS_DIR, STATISTICS_DIR,
              CLASSIFICATION_DIR, CLUSTER_CLASSIFICATION_DIR,
              PROJECTIONS_DIR, CONFOUNDER_DIR,
              VALIDATION_DIR, TAGGING_DIR, BASE_WORD_DIR,
              PERSISTENT_HOMOLOGY_DIR, SYNTHETIC_SHAPES_DIR, REPORTS_DIR,
              STREAMLIT_PROJECTION_CACHE_DIR, STREAMLIT_GRID_CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
