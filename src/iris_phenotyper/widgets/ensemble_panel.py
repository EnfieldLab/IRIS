"""
IRIS-Phenotyper Ensemble Results Panel

Displays:
- Cell type distribution bar chart (napari-style color swatches)
- Per-cell confidence histogram
- Flagged low-confidence cells list
- UMAP embedding (scatter colored by cell type)
- Manual override dropdown + Undo button
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QScrollArea, QSlider, QSplitter,
    QVBoxLayout, QWidget,
)


from iris_common.widgets.styles import SHARED_DARK_THEME

# Default color palette for up to 20 cell types
_TYPE_PALETTE = [
    "#4477EE", "#44AA44", "#EE4444", "#FFAA22", "#AA44EE",
    "#22AACC", "#FF6688", "#88CC44", "#AA8822", "#44CCAA",
    "#CC4488", "#6688FF", "#FFCC44", "#44DDFF", "#FF8844",
    "#AADDAA", "#DD88FF", "#88AAFF", "#FFDD88", "#CC88AA",
]


class EnsemblePanelWidget(QWidget):
    """
    Summary panel for ensemble phenotyping results.

    Signals
    -------
    thresholdChanged(float) :
        Emitted when the confidence threshold slider is moved.
    cellOverrideRequested(int, str) :
        Emitted when the user manually overrides a cell type.
    undoRequested :
        Emitted when the Undo button is clicked.
    showUMAPRequested :
        Emitted to trigger UMAP computation + napari layer rendering.
    """

    thresholdChanged = Signal(float)
    cellOverrideRequested = Signal(int, str)
    undoRequested = Signal()
    showUMAPRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(SHARED_DARK_THEME)
        self._cell_type_colors: Dict[str, str] = {}
        self._flagged_cells: List[int] = []
        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        hdr = QLabel("Ensemble Results")
        hdr.setStyleSheet("font-size: 14px; font-weight: bold; color: #DDD;")
        layout.addWidget(hdr)

        # ── Cell Type Summary ─────────────────────────────────────────────────
        summary_box = QGroupBox("Cell Type Distribution")
        summary_layout = QVBoxLayout(summary_box)

        self.lbl_total = QLabel("Total cells: —")
        self.lbl_total.setStyleSheet("color: #AAA; font-size: 12px;")
        summary_layout.addWidget(self.lbl_total)

        self.type_list = QListWidget()
        self.type_list.setMaximumHeight(160)
        self.type_list.setStyleSheet("QListWidget { font-size: 11px; }")
        summary_layout.addWidget(self.type_list)

        layout.addWidget(summary_box)

        # ── Confidence Threshold ──────────────────────────────────────────────
        conf_box = QGroupBox("Confidence Threshold")
        conf_layout = QVBoxLayout(conf_box)

        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(0, 100)
        self.slider_conf.setValue(30)
        self.slider_conf.setTickPosition(QSlider.TicksBelow)
        self.slider_conf.setTickInterval(10)
        self.slider_conf.valueChanged.connect(
            lambda v: self.thresholdChanged.emit(v / 100.0)
        )
        conf_layout.addWidget(self.slider_conf)

        row_conf = QHBoxLayout()
        self.lbl_conf_val = QLabel("Threshold: 0.30")
        self.slider_conf.valueChanged.connect(
            lambda v: self.lbl_conf_val.setText(f"Threshold: {v / 100:.2f}")
        )
        row_conf.addWidget(self.lbl_conf_val)

        self.lbl_flagged = QLabel("Flagged: —")
        self.lbl_flagged.setStyleSheet("color: #EE8844;")
        row_conf.addStretch()
        row_conf.addWidget(self.lbl_flagged)
        conf_layout.addLayout(row_conf)

        layout.addWidget(conf_box)

        # ── Flagged Cells ─────────────────────────────────────────────────────
        flag_box = QGroupBox("Low-Confidence Cells")
        flag_layout = QVBoxLayout(flag_box)

        self.flagged_list = QListWidget()
        self.flagged_list.setMaximumHeight(120)
        self.flagged_list.setStyleSheet("font-size: 11px;")
        flag_layout.addWidget(self.flagged_list)

        flag_layout.addWidget(QLabel("Override selected cell type:"))
        self.combo_override = QComboBox()
        flag_layout.addWidget(self.combo_override)

        row_btn = QHBoxLayout()
        self.btn_apply_override = QPushButton("Apply Override")
        self.btn_apply_override.clicked.connect(self._on_apply_override)
        row_btn.addWidget(self.btn_apply_override)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self.undoRequested.emit)
        row_btn.addWidget(self.btn_undo)
        flag_layout.addLayout(row_btn)

        layout.addWidget(flag_box)

        # ── UMAP ──────────────────────────────────────────────────────────────
        umap_box = QGroupBox("UMAP Embedding")
        umap_layout = QVBoxLayout(umap_box)
        umap_info = QLabel(
            "Compute a 2-D UMAP of cell intensities colored by cell type.\n"
            "Opens as a napari Points layer."
        )
        umap_info.setWordWrap(True)
        umap_info.setStyleSheet("color: #AAA; font-size: 11px;")
        umap_layout.addWidget(umap_info)

        self.btn_umap = QPushButton("Compute UMAP")
        self.btn_umap.clicked.connect(self.showUMAPRequested.emit)
        umap_layout.addWidget(self.btn_umap)
        layout.addWidget(umap_box)

        layout.addStretch()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_threshold(self) -> float:
        return self.slider_conf.value() / 100.0

    def update_results(
        self,
        final_types: Dict[int, str],
        confidences: Dict[int, float],
        cell_types: List[str],
    ) -> None:
        """Populate all summary widgets with ensemble results."""
        total = len(final_types)
        self.lbl_total.setText(f"Total cells: {total:,}")

        # Build cell type distribution
        counts: Dict[str, int] = {}
        for ct in final_types.values():
            counts[ct] = counts.get(ct, 0) + 1

        self.type_list.clear()
        self._cell_type_colors.clear()
        for i, (ct, n) in enumerate(sorted(counts.items(), key=lambda x: -x[1])):
            color = _TYPE_PALETTE[i % len(_TYPE_PALETTE)]
            self._cell_type_colors[ct] = color
            pct = 100.0 * n / max(total, 1)
            item = QListWidgetItem(f"  {ct}  ({n:,}  |  {pct:.1f}%)")
            qc = QColor(color)
            item.setForeground(qc)
            self.type_list.addItem(item)

        # Update combo override choices
        self.combo_override.clear()
        self.combo_override.addItem("Remove")
        for ct in sorted(cell_types):
            self.combo_override.addItem(ct)

        # Compute flagged cells
        thresh = self.get_threshold()
        self._flagged_cells = [cid for cid, c in confidences.items() if c < thresh]
        self.lbl_flagged.setText(f"Flagged: {len(self._flagged_cells):,}")
        self._populate_flagged_list(final_types, confidences)

    def _populate_flagged_list(
        self,
        final_types: Dict[int, str],
        confidences: Dict[int, float],
    ) -> None:
        thresh = self.get_threshold()
        self.flagged_list.clear()
        for cid in sorted(self._flagged_cells):
            ct = final_types.get(cid, "?")
            conf = confidences.get(cid, 0.0)
            item = QListWidgetItem(f"Cell {cid} — {ct}  (conf: {conf:.2f})")
            item.setData(Qt.UserRole, cid)
            self.flagged_list.addItem(item)

    def get_selected_cell_id(self) -> Optional[int]:
        items = self.flagged_list.selectedItems()
        if items:
            return items[0].data(Qt.UserRole)
        return None

    def _on_apply_override(self) -> None:
        cell_id = self.get_selected_cell_id()
        new_type = self.combo_override.currentText()
        if cell_id is not None and new_type:
            self.cellOverrideRequested.emit(cell_id, new_type)

    def get_cell_type_colors(self) -> Dict[str, str]:
        return dict(self._cell_type_colors)
