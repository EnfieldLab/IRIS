"""
IRIS-Viewer Application State
Holds all mutable application data for the Viewer scope:
image data, segmentation results, consensus masks, confidence maps, and extracted features.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


class ViewerAppState:
    def __init__(self):
        # ── Image & File Management ──────────────────────────────────────
        self.image_dir: Optional[Path] = None
        self.file_map: Dict[str, Path] = {}
        self.markers: List[str] = []
        self.current_sample_name: Optional[str] = None
        self.raw_image: Optional[np.ndarray] = None          # (C, H, W) multi-channel

        # ── Segmentation Results ─────────────────────────────────────────
        # method_name -> {'nuclear': np.ndarray, 'cytoplasm': np.ndarray | None}
        self.segmentation_results: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        self.segmentation_status: Dict[str, str] = {}         # method -> Queued|Running|Done|Failed

        # ── Consensus & Confidence ───────────────────────────────────────
        self.consensus_mask: Optional[np.ndarray] = None       # (H, W) label array
        self.confidence_map: Optional[np.ndarray] = None       # (H, W) float [0, 1]
        self.per_cell_confidence: Dict[int, float] = {}        # cell_id -> confidence
        self.manual_overrides: Dict[int, str] = {}             # cell_id -> preferred method

        # ── Feature Extraction ───────────────────────────────────────────
        self.features_df: Optional[pd.DataFrame] = None

        # ── Multi-sample Layer Tracking ──────────────────────────────────
        self.masks_map: Dict[str, np.ndarray] = {}             # sample -> nuclear mask
        self.sample_layers_map: Dict[str, List[Any]] = {}
        self.persistent_settings: Dict[str, Any] = {}

        # ── Batch Processing ─────────────────────────────────────────────
        self.batch_queue: List[str] = []
        self.batch_results: Dict[str, Dict] = {}               # sample -> results

    def reset_sample_state(self):
        """Resets volatile per-sample data while preserving global configs."""
        self.raw_image = None
        self.segmentation_results.clear()
        self.segmentation_status.clear()
        self.consensus_mask = None
        self.confidence_map = None
        self.per_cell_confidence.clear()
        self.manual_overrides.clear()
        self.features_df = None

    def reset_all(self):
        """Full state reset."""
        self.__init__()
