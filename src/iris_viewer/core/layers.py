"""
iris_viewer.core.layers
=======================
Helpers for attaching and querying structured metadata and polymorphic layer types.
"""
from __future__ import annotations

from typing import Any, Optional


def get_layer_metadata(layer: Any, key: str, default: Any = None) -> Any:
    """Retrieve metadata attribute from a napari layer with fallback."""
    meta = getattr(layer, "metadata", None)
    if isinstance(meta, dict):
        return meta.get(key, default)
    return default


def get_layer_base_name(layer: Any) -> str:
    """Retrieve the base identifier of a layer (channel marker, cell type, or niche).

    Checks layer.metadata first before falling back to string splitting on ' ['.
    """
    meta = getattr(layer, "metadata", None)
    if isinstance(meta, dict):
        for k in ("marker", "cell_type", "niche", "segmentation_name", "base_name"):
            if k in meta and meta[k]:
                return str(meta[k]).strip()

    name = getattr(layer, "name", "")
    return name.split(" [")[0].strip() if " [" in name else name.strip()


def get_layer_sample_name(layer: Any) -> Optional[str]:
    """Retrieve the sample name associated with a layer.

    Checks layer.metadata first before falling back to bracket extraction e.g. '[sample_A]'.
    """
    meta = getattr(layer, "metadata", None)
    if isinstance(meta, dict) and meta.get("sample_name"):
        return str(meta["sample_name"]).strip()

    name = getattr(layer, "name", "")
    if "[" in name and "]" in name:
        return name.split("[")[1].split("]")[0].strip()
    return None


def get_layer_type(layer: Any) -> Optional[str]:
    """Retrieve the semantic role of the layer ('channel', 'phenotype', 'segmentation', 'bounds', 'centroids', etc.)."""
    meta = getattr(layer, "metadata", None)
    if isinstance(meta, dict) and meta.get("layer_type"):
        return str(meta["layer_type"])

    name = getattr(layer, "name", "")
    if name.startswith("Bounds - "):
        return "bounds"
    if name.startswith("Centroids - "):
        return "centroids"
    if name.startswith("Niches - "):
        return "niches"
    if "segmentation" in name.lower() or "mask" in name.lower():
        return "segmentation"
    return None
