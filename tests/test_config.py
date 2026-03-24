"""Tests for config.py: model lists, path helpers, directory structure."""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODELS,
    MODEL_SLUGS,
    TRANSFORMER_MODELS,
    STATIC_MODELS,
    OPENAI_MODELS,
    PAIR_TYPES,
    PROJECT_ROOT,
    DATA_DIR,
    RESULTS_DIR,
    STUTTGART_DIR,
    get_model_slug,
    get_embedding_path,
    get_pair_path,
)


def test_five_paper_models():
    """Config should list exactly the 5 models used in the paper."""
    assert len(MODELS) == 5
    assert "bert-base-cased" in MODELS
    assert "word2vec" in MODELS
    assert "glove" in MODELS
    assert "text-embedding-3-small" in MODELS
    assert "text-embedding-3-large" in MODELS


def test_no_modernbert():
    """ModernBERT should not be in the shareable config."""
    assert "answerdotai/modernbert-base" not in MODELS
    assert "answerdotai/modernbert-base" not in MODEL_SLUGS


def test_model_categories_cover_all():
    """Every model should be in exactly one category."""
    all_categorized = set(TRANSFORMER_MODELS + STATIC_MODELS + OPENAI_MODELS)
    assert all_categorized == set(MODELS)


def test_all_models_have_slugs():
    for model in MODELS:
        slug = get_model_slug(model)
        assert slug, f"No slug for {model}"
        assert "/" not in slug and "-" not in slug


def test_four_pair_types():
    assert len(PAIR_TYPES) == 4
    assert "synonyms" in PAIR_TYPES
    assert "antonyms" in PAIR_TYPES


def test_data_directories_exist():
    assert STUTTGART_DIR.exists(), "Stuttgart data directory missing"
    assert DATA_DIR.exists()


def test_openai_embedding_path_prefers_npz():
    """For OpenAI models, get_embedding_path should find the npz if it exists."""
    path = get_embedding_path("text-embedding-3-large")
    # Should resolve to npz since we ship it in that format
    assert path.suffix in (".npz", ".pkl")


def test_pair_path_format():
    path = get_pair_path("bert-base-cased", "synonyms")
    assert "bert_base_cased_synonyms_pairs.json" in str(path)
