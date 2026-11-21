"""
IRIS-Viewer Application State
==============================
Holds all mutable runtime data for one Viewer session.

Mutable collections (dicts, lists) are never replaced — always mutated
in place so that any service holding a reference to them sees updates
automatically.

Session save/restore serialises only lightweight scalars to JSON;
large numpy arrays (images, masks) must be reloaded from disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ViewerAppState:
    """Complete runtime state for IRIS-Viewer."""

    def __init__(self) -> None:
        # Image & File Management
        self.image_dir: Optional[Path] = None
        self.file_map: Dict[str, Path] = {}        # sample_name -> Path
        self.markers: List[str] = []
        self.current_sample_name: Optional[str] = None
        self.raw_image: Optional[np.ndarray] = None        # (C, H, W)
        self.channel_contrast_limits: Dict[str, list] = {} # marker -> [lo, hi]
        # Persistent Marker & Cell-Type Color Maps
        self.channel_colors: Dict[str, str] = {}            # marker_name -> hex_color
        self.celltype_colors: Dict[str, str] = {}           # celltype_name -> hex_color

        # Segmentation Results
        # method_name -> {'nuclear': ndarray, 'cytoplasm': ndarray|None}
        self.segmentation_results: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        self.segmentation_status: Dict[str, str] = {}     # step_key -> status

        # Active Method & Multi-Model Disagreement Analysis
        self.active_segmentation_method: Optional[str] = None
        self.disagreement_report: Any = None
        self.disagreement_heatmap: Optional[np.ndarray] = None
        self.preview_roi_bbox: Optional[Tuple[int, int, int, int]] = None

        # Feature Extraction
        self.features_df: Optional[pd.DataFrame] = None

        # Active Mask References
        self.current_mask: Optional[np.ndarray] = None     # active nuclear mask
        self.cytoplasm_mask: Optional[np.ndarray] = None   # active/last loaded cytoplasm mask

        # Multi-Sample Canvas
        # sample_name -> raw multi-channel image array (C, H, W)
        self.multi_sample_images: Dict[str, np.ndarray] = {}
        # sample_name -> list of napari layers for that sample
        self.sample_layers_map: Dict[str, List[Any]] = {}
        # sample_name -> nuclear mask (uint32)
        self.masks_map: Dict[str, np.ndarray] = {}
        # sample_name -> resolved annotation DataFrame
        self.resolved_dfs: Dict[str, pd.DataFrame] = {}
        # sample_name -> tissue area (px² or µm²)
        self.areas_map: Dict[str, float] = {}
        # concat of all currently displayed sample DataFrames
        self.current_mapped_df: Optional[pd.DataFrame] = None
        # IrisDataAdapter instance (set by MultiSampleService)
        self._data_adapter = None

        # Annotation Master File
        self.master_df: Optional[pd.DataFrame] = None

        # Batch Processing
        self.batch_queue: List[str] = []
        self.batch_results: Dict[str, Dict] = {}      # sample -> result dict

        # Persistent UI Settings
        self.persistent_settings: Dict[str, Any] = {}

    # Active Mask Selection

    def get_active_masks(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return ``(nuclear_mask, cytoplasm_mask)`` for the active segmentation method.

        Falls back to ``(self.current_mask, self.cytoplasm_mask)`` when no active
        method is selected or not yet populated in ``segmentation_results``.
        """
        method = self.active_segmentation_method
        if method and method in self.segmentation_results:
            entry = self.segmentation_results[method]
            return entry.get("nuclear"), entry.get("cytoplasm")
        return self.current_mask, self.cytoplasm_mask

    def set_active_segmentation_method(
        self, method_name: Optional[str]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Select *method_name* from ``segmentation_results`` and sync mask pointers.

        Returns ``(nuclear_mask, cytoplasm_mask)`` for the selected method, or
        ``(None, None)`` if *method_name* is ``None`` or absent.
        """
        if not method_name or method_name not in self.segmentation_results:
            self.active_segmentation_method = None
            self.current_mask = None
            self.cytoplasm_mask = None
            return None, None

        entry = self.segmentation_results[method_name]
        nuc = entry.get("nuclear")
        cyto = entry.get("cytoplasm")
        self.active_segmentation_method = method_name
        self.current_mask = nuc
        self.cytoplasm_mask = cyto
        return nuc, cyto

    # Lifecycle

    def reset_sample_state(self) -> None:
        """Reset per-sample volatile data while preserving global config."""
        self.raw_image = None
        self.channel_contrast_limits.clear()
        self.segmentation_results.clear()
        self.segmentation_status.clear()
        self.active_segmentation_method = None
        self.disagreement_report = None
        self.disagreement_heatmap = None
        self.preview_roi_bbox = None
        self.features_df = None
        self.current_mask = None
        self.cytoplasm_mask = None
        self.multi_sample_images.clear()
        self.sample_layers_map.clear()
        self.masks_map.clear()
        self.resolved_dfs.clear()
        self.areas_map.clear()
        self.current_mapped_df = None

    def reset_all(self) -> None:
        """Full state reset — clears everything including directory and markers."""
        self.__init__()

    # Session Persistence

    def to_session_dict(self, disagreement_threshold: float = 0.65) -> dict:
        """Serialise recoverable session fields to a JSON-compatible dict.

        Large numpy arrays (masks, images, heatmaps) are excluded — they are
        re-generated or reloaded from disk when the session is restored.
        """
        return {
            "version": 2,
            "image_dir": str(self.image_dir) if self.image_dir else None,
            "markers": list(self.markers),
            "current_sample_name": self.current_sample_name,
            "active_segmentation_method": self.active_segmentation_method,
            "disagreement_threshold": float(disagreement_threshold),
            "persistent_settings": self.persistent_settings,
            "channel_colors": self.channel_colors,
            "celltype_colors": self.celltype_colors,
        }

    @classmethod
    def from_session_dict(cls, data: dict) -> "ViewerAppState":
        """Restore a ``ViewerAppState`` from a plain dict (loaded from JSON)."""
        state = cls()
        if data.get("image_dir"):
            state.image_dir = Path(data["image_dir"])
        state.markers = list(data.get("markers", []))
        state.current_sample_name = data.get("current_sample_name")
        active_method = data.get("active_segmentation_method")
        state.active_segmentation_method = str(active_method) if active_method else None
        state.persistent_settings = dict(data.get("persistent_settings", {}))
        state.channel_colors = dict(data.get("channel_colors", {}))
        state.celltype_colors = dict(data.get("celltype_colors", {}))
        return state

    def save_session(self, path: Path, disagreement_threshold: float = 0.65) -> None:
        """Write session JSON to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_session_dict(disagreement_threshold), fh, indent=2)

    @classmethod
    def load_session(cls, path: Path) -> tuple:
        """Load session from a JSON file on disk.

        Returns
        -------
        state : ViewerAppState
            Restored state object (arrays are not restored — reload from disk).
        raw_data : dict
            Raw JSON dict so callers can read extra fields (e.g. ``disagreement_threshold``).
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_session_dict(data), data
