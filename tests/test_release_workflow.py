"""Release-workflow smoke tests for the paper companion repo."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).parent.parent


def load_script_module(name: str, relative_path: str):
    """Load a script file as a testable module."""
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ensure_directories_creates_only_release_dirs(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "EMBEDDINGS_DIR", tmp_path / "embeddings")
    monkeypatch.setattr(config, "PAIRS_DIR", tmp_path / "pairs")
    monkeypatch.setattr(config, "FILTERED_PAIRS_DIR", tmp_path / "filtered_pairs")
    monkeypatch.setattr(config, "STATISTICS_DIR", tmp_path / "statistics")
    monkeypatch.setattr(config, "PROJECTIONS_DIR", tmp_path / "projections")
    monkeypatch.setattr(config, "VALIDATION_DIR", tmp_path / "validation")
    monkeypatch.setattr(config, "STREAMLIT_PROJECTION_CACHE_DIR", tmp_path / "streamlit" / "projections")
    monkeypatch.setattr(config, "STREAMLIT_GRID_CACHE_DIR", tmp_path / "streamlit" / "grids")

    config.ensure_directories()

    assert (tmp_path / "embeddings").exists()
    assert (tmp_path / "pairs").exists()
    assert (tmp_path / "filtered_pairs").exists()
    assert (tmp_path / "statistics").exists()
    assert (tmp_path / "projections").exists()
    assert (tmp_path / "validation").exists()
    assert not (tmp_path / "streamlit").exists()


def test_extract_embeddings_skips_missing_glove(monkeypatch):
    module = load_script_module("extract_embeddings_script", "scripts/02_extract_embeddings.py")

    monkeypatch.setattr(module, "ensure_directories", lambda: None)
    monkeypatch.setattr(module, "load_stuttgart", lambda _path: object())
    monkeypatch.setattr(module, "get_vocabulary", lambda _df: {"alpha", "beta"})
    monkeypatch.setattr(module, "get_default_glove_path", lambda: None)
    monkeypatch.setattr(module, "save_embeddings", lambda embs, path: None)

    original_argv = sys.argv[:]
    sys.argv = ["02_extract_embeddings.py", "--models", "glove"]
    try:
        module.main()
    finally:
        sys.argv = original_argv


def test_cosine_plot_targets_paper_image_path():
    module = load_script_module("generate_cosine_plots_script", "scripts/generate_cosine_sim_plots.py")
    assert module.OUT_PATH == PROJECT_ROOT / "prose" / "img" / "cosine_violin_grid.png"


def test_paper_figure_helpers_write_tex_filenames(tmp_path, monkeypatch):
    module = load_script_module("generate_paper_figures_script", "scripts/generate_paper_figures.py")

    def fake_projection(*_args, **_kwargs):
        coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        labels = ["antonyms", "synonyms"]
        return {
            "coords": coords,
            "labels": labels,
            "words1": ["a", "b"],
            "words2": ["c", "d"],
        }

    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(module, "_get_or_compute", fake_projection)

    out1 = module.fig_oai_large_other_projections(72, "oai_large_pca_tsne")
    out2 = module.fig_antsyn_only_selected(72, "antsyn_only_selected")

    assert out1.name == "autofilled_oai_large_pca_tsne.png"
    assert out1.exists()
    assert out2.name == "autofilled_antsyn_only_selected.png"
    assert out2.exists()


def test_figure_data_rejects_legacy_threshold_dataset():
    from semsim.figure_data import get_pairs_dir_for_dataset

    with pytest.raises(ValueError):
        get_pairs_dir_for_dataset("threshold_4.0")
