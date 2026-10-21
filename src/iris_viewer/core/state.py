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
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

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

        # Segmentation Results
        # method_name -> {'nuclear': ndarray, 'cytoplasm': ndarray|None}
        self.segmentation_results: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        self.segmentation_status: Dict[str, str] = {}     # step_key -> status

        # Consensus & Confidence
        self.consensus_mask: Optional[np.ndarray] = None      # (H, W) label array
        self.confidence_map: Optional[np.ndarray] = None      # (H, W) float [0, 1]
        self.per_cell_confidence: Dict[int, float] = {}       # cell_id -> confidence
        self.cell_method_map: Dict[int, Dict[str, int]] = {}  # cell_id -> {method: orig_id}
        self.manual_overrides: Dict[int, str] = {}            # cell_id -> preferred method

        # Override History (undo stack)
        # Each snapshot: (consensus_mask, confidence_map, per_cell_confidence, manual_overrides)
        self.override_history: Deque[Tuple] = deque(maxlen=50)

        # Feature Extraction
        self.features_df: Optional[pd.DataFrame] = None

        # Active Mask References
        self.current_mask: Optional[np.ndarray] = None     # active nuclear mask
        self.cytoplasm_mask: Optional[np.ndarray] = None   # last loaded cytoplasm mask

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

    # Override History

    def snapshot_override_state(self) -> None:
        """Push a copy of override-related arrays onto the undo stack."""
        if self.consensus_mask is not None:
            self.override_history.append((
                self.consensus_mask.copy(),
                self.confidence_map.copy() if self.confidence_map is not None else None,
                dict(self.per_cell_confidence),
                dict(self.manual_overrides),
            ))

    def undo_override(self) -> bool:
        """Pop and restore the most recent override snapshot.

        Returns ``True`` on success, ``False`` if the history is empty.
        """
        if not self.override_history:
            return False
        (self.consensus_mask,
         self.confidence_map,
         self.per_cell_confidence,
         self.manual_overrides) = self.override_history.pop()
        return True

    # Lifecycle

    def reset_sample_state(self) -> None:
        """Reset per-sample volatile data while preserving global config."""
        self.raw_image = None
        self.channel_contrast_limits.clear()
        self.segmentation_results.clear()
        self.segmentation_status.clear()
        self.consensus_mask = None
        self.confidence_map = None
        self.per_cell_confidence.clear()
        self.cell_method_map.clear()
        self.manual_overrides.clear()
        self.override_history.clear()
        self.features_df = None
        self.current_mask = None
        self.cytoplasm_mask = None
        self.multi_sample_images.clear()
        self.masks_map.clear()
        self.resolved_dfs.clear()
        self.areas_map.clear()
        self.current_mapped_df = None

    def reset_all(self) -> None:
        """Full state reset — clears everything including directory and markers."""
        self.__init__()

    # Session Persistence

    def to_session_dict(self, consensus_threshold: float = 0.66) -> dict:
        """Serialise recoverable session fields to a JSON-compatible dict.

        Large numpy arrays (masks, images) are excluded — they are
        re-generated or reloaded from disk when the session is restored.
        """
        return {
            "version": 1,
            "image_dir": str(self.image_dir) if self.image_dir else None,
            "markers": self.markers,
            "current_sample_name": self.current_sample_name,
            "consensus_threshold": consensus_threshold,
            # Integer keys become strings in JSON — restored as int on load.
            "per_cell_confidence": {str(k): v for k, v in self.per_cell_confidence.items()},
            "manual_overrides": {str(k): v for k, v in self.manual_overrides.items()},
            "persistent_settings": self.persistent_settings,
        }

    @classmethod
    def from_session_dict(cls, data: dict) -> "ViewerAppState":
        """Restore a ``ViewerAppState`` from a plain dict (loaded from JSON)."""
        state = cls()
        if data.get("image_dir"):
            state.image_dir = Path(data["image_dir"])
        state.markers = data.get("markers", [])
        state.current_sample_name = data.get("current_sample_name")
        state.per_cell_confidence = {
            int(k): float(v) for k, v in data.get("per_cell_confidence", {}).items()
        }
        state.manual_overrides = {
            int(k): str(v) for k, v in data.get("manual_overrides", {}).items()
        }
        state.persistent_settings = data.get("persistent_settings", {})
        return state

    def save_session(self, path: Path, consensus_threshold: float = 0.66) -> None:
        """Write session JSON to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_session_dict(consensus_threshold), fh, indent=2)

    @classmethod
    def load_session(cls, path: Path) -> tuple:
        """Load session from a JSON file on disk.

        Returns
        -------
        state : ViewerAppState
            Restored state object (arrays are not restored — reload from disk).
        raw_data : dict
            Raw JSON dict so callers can read extra fields (e.g. ``consensus_threshold``).
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_session_dict(data), data
