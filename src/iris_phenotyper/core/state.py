"""iris-phenotyper PhenotyperAppState."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from iris_common.types import CellTypePanel, MarkerPanel


class PhenotyperAppState:
    """
    Holds all mutable application data for the IRIS-Phenotyper.
    Mirrors ViewerAppState in design.
    """

    def __init__(self) -> None:
        # ── Input Data ──────────────────────────────────────────────────────
        self.features_df: Optional[pd.DataFrame] = None     # loaded from CSV
        self.image_dir: Optional[Path] = None               # source image dir (for raw patches)
        self.file_map: Dict[str, Path] = {}                 # sample → features CSV path
        self.current_sample_name: Optional[str] = None

        # ── Panel Definitions ────────────────────────────────────────────────
        self.marker_panel: Optional[MarkerPanel] = None
        self.cell_type_panel: Optional[CellTypePanel] = None

        # ── Phenotyping Results ──────────────────────────────────────────────
        # method_name → {cell_id → {cell_type → probability}}
        self.phenotyping_results: Dict[str, Dict[int, Dict[str, float]]] = {}
        self.phenotyping_status: Dict[str, str] = {}        # method → Queued|Running|Done|Failed

        # ── Ensemble Output ──────────────────────────────────────────────────
        self.final_cell_types: Dict[int, str] = {}           # cell_id → cell_type
        self.cell_type_confidence: Dict[int, float] = {}     # cell_id → confidence
        self.per_method_breakdown: Dict[int, Dict] = {}      # cell_id → {method: type}
        self.ensemble_df: Optional[pd.DataFrame] = None      # features + ensemble columns

        # ── UMAP ─────────────────────────────────────────────────────────────
        self.umap_embedding: Optional[np.ndarray] = None    # (N, 2) UMAP coords


        # ── Override History ─────────────────────────────────────────────────
        self.override_history: Deque[Tuple] = deque(maxlen=50)

        # ── Batch ────────────────────────────────────────────────────────────
        self.batch_queue: List[str] = []
        self.batch_results: Dict[str, Dict] = {}

        # ── ImageService / MultiSampleService requirements ──────────────────
        self.image_dir: Optional[Path] = None
        self.file_map: Dict[str, Path] = {}
        self.markers: List[str] = []
        self.current_sample_name: Optional[str] = None
        self.raw_image: Optional[np.ndarray] = None
        self.channel_contrast_limits: Dict[str, list] = {}
        
        self.multi_sample_images: Dict[str, np.ndarray] = {}
        self.resolved_dfs: Dict[str, pd.DataFrame] = {}
        self.masks_map: Dict[str, np.ndarray] = {}
        self.sample_layers_map: Dict[str, List] = {}
        
        self.segmentation_results: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        self.cytoplasm_mask: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.persistent_settings: Dict[str, Any] = {}

    def reset_sample_state(self) -> None:
        self.features_df = None
        self.phenotyping_results.clear()
        self.phenotyping_status.clear()
        self.final_cell_types.clear()
        self.cell_type_confidence.clear()
        self.per_method_breakdown.clear()
        self.ensemble_df = None
        self.umap_embedding = None
        self.override_history.clear()
        self.multi_sample_images.clear()
        self.masks_map.clear()
        self.resolved_dfs.clear()
        self.sample_layers_map.clear()

    def snapshot_override_state(self) -> None:
        self.override_history.append((
            dict(self.final_cell_types),
            dict(self.cell_type_confidence),
            {k: dict(v) for k, v in self.per_method_breakdown.items()},
        ))

    def undo_override(self) -> bool:
        if not self.override_history:
            return False
        (self.final_cell_types,
         self.cell_type_confidence,
         self.per_method_breakdown) = self.override_history.pop()
        return True

    def save_session(self, path: Path) -> None:
        d = {
            "version": 1,
            "image_dir": str(self.image_dir) if self.image_dir else None,
            "current_sample_name": self.current_sample_name,
            "final_cell_types": {str(k): v for k, v in self.final_cell_types.items()},
            "cell_type_confidence": {str(k): v for k, v in self.cell_type_confidence.items()},
            "cell_annotations": {str(k): v for k, v in self.cell_annotations.items()},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(d, fh, indent=2)

    @classmethod
    def load_session(cls, path: Path) -> tuple:
        with open(path) as fh:
            data = json.load(fh)
        st = cls()
        if data.get("image_dir"):
            st.image_dir = Path(data["image_dir"])
        st.current_sample_name = data.get("current_sample_name")
        st.final_cell_types = {int(k): v for k, v in data.get("final_cell_types", {}).items()}
        st.cell_type_confidence = {int(k): float(v) for k, v in data.get("cell_type_confidence", {}).items()}
        st.cell_annotations = {int(k): v for k, v in data.get("cell_annotations", {}).items()}
        return st, data
