"""
IRIS-Viewer Application State
Holds all mutable application data for the Viewer scope:
image data, segmentation results, consensus masks, confidence maps, and extracted features.
Session save/restore serialises only recoverable fields to JSON; large arrays stay on disk.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ViewerAppState:
    def __init__(self) -> None:
        # ── Image & File Management ─────────────────────────────────────
        self.image_dir: Optional[Path] = None
        self.file_map: Dict[str, Path] = {}
        self.markers: List[str] = []
        self.current_sample_name: Optional[str] = None
        self.raw_image: Optional[np.ndarray] = None       # (C, H, W) multi-channel
        self.channel_contrast_limits: Dict[str, list] = {}  # marker -> [lo, hi] cache

        # ── Segmentation Results ────────────────────────────────────────
        # method_name -> {'nuclear': np.ndarray, 'cytoplasm': np.ndarray | None}
        self.segmentation_results: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        self.segmentation_status: Dict[str, str] = {}     # step_key -> Queued|Running|Done|Failed

        # ── Consensus & Confidence ──────────────────────────────────────
        self.consensus_mask: Optional[np.ndarray] = None        # (H, W) label array
        self.confidence_map: Optional[np.ndarray] = None        # (H, W) float [0, 1]
        self.per_cell_confidence: Dict[int, float] = {}         # cell_id -> confidence
        # Properly typed field (was private _cell_method_map in old code):
        self.cell_method_map: Dict[int, Dict[str, int]] = {}    # cell_id -> {method: orig_id}
        self.manual_overrides: Dict[int, str] = {}              # cell_id -> preferred method

        # ── Override History (undo stack) ───────────────────────────────
        # Each snapshot is (consensus_mask, confidence_map, per_cell_confidence, manual_overrides)
        self.override_history: Deque[Tuple] = deque(maxlen=50)

        # ── Feature Extraction ──────────────────────────────────────────
        self.features_df: Optional[pd.DataFrame] = None

        # ── Multi-sample Layer Tracking ─────────────────────────────────
        self.masks_map: Dict[str, np.ndarray] = {}          # sample -> nuclear mask
        self.sample_layers_map: Dict[str, List[Any]] = {}
        self.persistent_settings: Dict[str, Any] = {}

        # ── Batch Processing ────────────────────────────────────────────
        self.batch_queue: List[str] = []
        self.batch_results: Dict[str, Dict] = {}           # sample -> results dict

    # ── Override History ────────────────────────────────────────────────

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
        """Pop and restore the most recent snapshot. Returns True if successful."""
        if not self.override_history:
            return False
        (self.consensus_mask,
         self.confidence_map,
         self.per_cell_confidence,
         self.manual_overrides) = self.override_history.pop()
        return True

    # ── Lifecycle ───────────────────────────────────────────────────────

    def reset_sample_state(self) -> None:
        """Resets volatile per-sample data while preserving global configs."""
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

    def reset_all(self) -> None:
        """Full state reset — clears everything including directory and markers."""
        self.__init__()

    # ── Session Persistence ─────────────────────────────────────────────

    def to_session_dict(self, consensus_threshold: float = 0.66) -> dict:
        """Serialise recoverable session fields to a JSON-compatible dict.

        Large numpy arrays (masks, images) are intentionally excluded; they are
        re-generated or reloaded from disk on session restore.
        """
        return {
            "version": 1,
            "image_dir": str(self.image_dir) if self.image_dir else None,
            "markers": self.markers,
            "current_sample_name": self.current_sample_name,
            "consensus_threshold": consensus_threshold,
            # Integer keys become strings in JSON — restored as int on load
            "per_cell_confidence": {str(k): v for k, v in self.per_cell_confidence.items()},
            "manual_overrides": {str(k): v for k, v in self.manual_overrides.items()},
            "persistent_settings": self.persistent_settings,
        }

    @classmethod
    def from_session_dict(cls, data: dict) -> "ViewerAppState":
        """Restore session from a plain dict (as loaded from JSON)."""
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
        """Write session JSON to disk."""
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
            Restored state object (no arrays — those need to be reloaded separately).
        raw_data : dict
            The raw JSON dict, so the caller can read extra fields like
            ``consensus_threshold``.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_session_dict(data), data
