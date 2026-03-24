"""
Caching utilities for projection data and grid images.

Provides deterministic cache key generation and NPZ/PNG save/load
for the Streamlit dashboard and pre-population script.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import STREAMLIT_PROJECTION_CACHE_DIR, STREAMLIT_GRID_CACHE_DIR

logger = logging.getLogger(__name__)


def _make_params_hash(params: dict) -> str:
    """Create a deterministic 12-char hex hash from a params dict.

    Sorts keys recursively and uses canonical JSON for reproducibility.
    """
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _sort_nested(obj: Any) -> Any:
    """Recursively sort dicts and lists for canonical representation."""
    if isinstance(obj, dict):
        return {k: _sort_nested(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_sort_nested(x) for x in obj]
    return obj


def build_projection_params(
    pair_types: List[str],
    symmetrize: bool,
    standardize: bool,
    metric: str = "euclidean",
    **method_kwargs,
) -> dict:
    """Build canonical params dict for hashing.

    Includes all parameters that affect projection output.
    """
    params = {
        "pair_types": sorted(pair_types),
        "symmetrize": symmetrize,
        "standardize": standardize,
        "metric": metric,
    }
    params.update(method_kwargs)
    return _sort_nested(params)


def get_projection_cache_path(
    dataset: str, model_slug: str, input_type: str,
    method: str, n_dims: int, params: dict,
) -> Path:
    """Return the cache file path for a projection NPZ."""
    h = _make_params_hash(params)
    filename = f"{dataset}__{model_slug}__{input_type}__{method}__{n_dims}d__{h}.npz"
    return STREAMLIT_PROJECTION_CACHE_DIR / filename


def get_grid_cache_path(
    dataset: str, model_slug: str, input_type: str,
    method: str, params: dict,
) -> Path:
    """Return the cache file path for a grid PNG."""
    h = _make_params_hash(params)
    filename = f"{dataset}__{model_slug}__{input_type}__{method}__{h}.png"
    return STREAMLIT_GRID_CACHE_DIR / filename


def save_projection_to_cache(path: Path, proj_data: dict) -> None:
    """Save projection data (coords, labels, words1, words2) to NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "coords": np.asarray(proj_data["coords"], dtype=np.float32),
        "labels": np.array(proj_data["labels"], dtype=str),
        "words1": np.array(proj_data["words1"], dtype=str),
        "words2": np.array(proj_data["words2"], dtype=str),
    }
    np.savez_compressed(path, **save_dict)
    logger.info("Cached projection to %s (%d samples)", path, len(save_dict["coords"]))


def load_projection_from_cache(path: Path) -> Optional[dict]:
    """Load projection data from cache. Returns None if missing."""
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        return {
            "coords": data["coords"],
            "labels": data["labels"].tolist(),
            "words1": data["words1"].tolist(),
            "words2": data["words2"].tolist(),
        }
    except Exception:
        logger.warning("Failed to load cached projection from %s", path, exc_info=True)
        return None


def save_grid_to_cache(path: Path, fig_or_bytes, dpi: int = 300) -> None:
    """Save a grid image to cache as PNG.

    Args:
        path: Output path.
        fig_or_bytes: Either a matplotlib Figure or raw bytes.
        dpi: DPI for figure rendering (ignored if bytes).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(fig_or_bytes, bytes):
        path.write_bytes(fig_or_bytes)
    else:
        import io
        buf = io.BytesIO()
        fig_or_bytes.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        path.write_bytes(buf.getvalue())
    logger.info("Cached grid image to %s", path)


def load_grid_from_cache(path: Path) -> Optional[bytes]:
    """Load grid image bytes from cache. Returns None if missing."""
    if not path.exists():
        return None
    try:
        return path.read_bytes()
    except Exception:
        logger.warning("Failed to load cached grid from %s", path, exc_info=True)
        return None


def resolve_dataset_key(is_unfiltered: bool, data_source: str) -> str:
    """Map UI state to a cache dataset key.

    Returns one of: 'validated_3.0', 'unfiltered', 'threshold_X.Y'
    """
    if is_unfiltered:
        return "unfiltered"
    if data_source == "Default Pipeline":
        return "validated_3.0"
    # "Threshold 4.0" -> "threshold_4.0"
    return data_source.lower().replace(" ", "_")
