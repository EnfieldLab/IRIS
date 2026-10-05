"""
IRIS-Phenotyper Control Widget — top dock panel for file loading and export.
"""
from __future__ import annotations

from typing import Optional

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)


_DARK = """
QWidget { background: #1A1A2E; color: #E0E0E0; font-family: 'Inter', sans-serif; }
QPushButton { background: #2D2D4E; border: 1px solid #444; border-radius: 4px; padding: 5px 10px; }
QPushButton:hover { background: #3D3D6E; }
QGroupBox { border: 1px solid #3A3A5A; border-radius: 6px; margin-top: 8px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #8888FF; font-weight: bold; }
QLineEdit { background: #252540; border: 1px solid #444; border-radius: 3px; padding: 3px; color: #CCC; }
"""


class PhenotyperControlWidget(QWidget):
    """
    Top-level control panel.

    Signals
    -------
    loadFeaturesTriggered(str) : emitted with the CSV file path
    exportResultsTriggered()
    saveSessionTriggered()
    """

    loadFeaturesTriggered  = Signal(str)
    exportResultsTriggered = Signal()
    saveSessionTriggered   = Signal()

    def __init__(self, viewer=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.setStyleSheet(_DARK)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Brand header
        hdr = QLabel("🔬 IRIS-Phenotyper")
        hdr.setStyleSheet(
            "font-size: 16px; font-weight: bold; "
            "color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8888FF, stop:1 #88FFFF);"
        )
        layout.addWidget(hdr)

        sub = QLabel("Cell phenotyping with ensemble voting")
        sub.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
        layout.addWidget(sub)

        # ── Load Features ─────────────────────────────────────────────────────
        load_box = QGroupBox("Load Features CSV")
        load_layout = QVBoxLayout(load_box)

        self.lbl_features = QLabel("No file loaded")
        self.lbl_features.setStyleSheet("color: #888; font-size: 11px;")
        load_layout.addWidget(self.lbl_features)

        self.btn_load = QPushButton("Browse Features CSV…")
        self.btn_load.clicked.connect(self._browse_features)
        load_layout.addWidget(self.btn_load)
        layout.addWidget(load_box)

        # ── Dashboard ─────────────────────────────────────────────────────────
        dash_box = QGroupBox("Summary")
        dash_layout = QVBoxLayout(dash_box)
        self.lbl_total   = QLabel("Total Cells: —")
        self.lbl_methods = QLabel("Methods Ready: —")
        for lbl in (self.lbl_total, self.lbl_methods):
            lbl.setStyleSheet("color: #AAA; font-size: 12px;")
            dash_layout.addWidget(lbl)
        layout.addWidget(dash_box)

        # ── Export / Session ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        for label, sig in [
            ("Export CSV",   self.exportResultsTriggered),
            ("Save Session", self.saveSessionTriggered),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(sig.emit)
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)
        layout.addStretch()

    def _browse_features(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Features CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            self.lbl_features.setText(path.split("/")[-1])
            self.loadFeaturesTriggered.emit(path)

    def update_dashboard(self, total_cells: str = "", features_path: str = "") -> None:
        self.lbl_total.setText(f"Total Cells: {total_cells}")
        if features_path:
            self.lbl_features.setText(features_path.split("/")[-1])
