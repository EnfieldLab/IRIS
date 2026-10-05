"""
IRIS-Viewer Multi-Sample Comparison Widget

A dock panel for loading, selecting, and aligning multiple TMA cores (or tissue
sections) onto the napari canvas simultaneously.

Signals
-------
addSamplesTriggered(list[str]) :
    Emitted when the user clicks "Add Selected Samples to Canvas".
clearCanvasTriggered() :
    Emitted when the user clicks "Clear Canvas".
masterFileTriggered(str) :
    Emitted with the annotation file path when the user loads an xlsx/csv.
"""
from __future__ import annotations

import re
from typing import List, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from iris_viewer.widgets.styles import SHARED_DARK_THEME


class MultiSampleComparisonWidget(QWidget):
    """
    Dock widget for multi-sample TMA / IF comparison.

    Features
    --------
    - Scrollable sample checklist (populated from ``update_sample_list``)
    - Select All / None toggle
    - Load annotation xlsx / CSV (master file)
    - "Add Selected Samples to Canvas" → ``addSamplesTriggered``
    - "Clear Canvas"                   → ``clearCanvasTriggered``
    - Grid gap and max columns spinners for canvas layout
    """

    addSamplesTriggered  = Signal(list)
    clearCanvasTriggered = Signal()
    masterFileTriggered  = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(SHARED_DARK_THEME)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        self._main_layout = QVBoxLayout(content)
        self._main_layout.setContentsMargins(4, 4, 4, 4)
        self._main_layout.setSpacing(6)

        scroll.setWidget(content)
        outer.addWidget(scroll)

        self._build_master_file_group()
        self._build_comparison_group()
        self._build_layout_group()
        self._main_layout.addStretch()

    # ── Groups ────────────────────────────────────────────────────────────────

    def _build_master_file_group(self) -> None:
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

        title = QLabel("Annotation Data")
        title.setStyleSheet("color: #DDD; font-size: 13px; font-weight: bold;")
        layout.addWidget(title)
        self._add_sep(layout)

        info = QLabel(
            "Load an xlsx/csv with voron_x, voron_y, GROUP (cell type),\n"
            "and marker expression columns (DAPI – Ki67, etc.)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #999; font-size: 11px; font-style: italic;")
        layout.addWidget(info)

        self.lbl_master_file = QLabel("No annotation file loaded")
        self.lbl_master_file.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_master_file.setWordWrap(True)
        layout.addWidget(self.lbl_master_file)

        self.btn_load_master = QPushButton("Load Annotation File…")
        self.btn_load_master.clicked.connect(self._browse_master_file)
        layout.addWidget(self.btn_load_master)

        self._main_layout.addWidget(box)

    def _build_comparison_group(self) -> None:
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

        title = QLabel("Multi-Sample Comparison")
        title.setStyleSheet("color: #DDD; font-size: 13px; font-weight: bold;")
        layout.addWidget(title)
        self._add_sep(layout)

        self.chk_select_all = QCheckBox("Select All")
        self.chk_select_all.toggled.connect(self._on_select_all_toggled)
        layout.addWidget(self.chk_select_all)

        self.list_samples = QListWidget()
        self.list_samples.setFixedHeight(220)
        self.list_samples.setStyleSheet(
            "QListWidget { background: #1E1E2E; border: 1px solid #444; }"
            "QListWidget::item:selected { background: #3A3A5A; }"
        )
        self.list_samples.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_samples)

        self.btn_add = QPushButton("Add Selected Samples to Canvas")
        self.btn_add.setStyleSheet(
            "QPushButton { background: #2A4A7A; color: white; padding: 5px; border-radius: 3px; }"
            "QPushButton:hover { background: #3A6AAA; }"
        )
        self.btn_add.clicked.connect(self._on_add_clicked)
        layout.addWidget(self.btn_add)

        self.btn_clear = QPushButton("Clear Canvas")
        self.btn_clear.setStyleSheet(
            "QPushButton { background: #4A2A2A; color: white; padding: 5px; border-radius: 3px; }"
            "QPushButton:hover { background: #6A3A3A; }"
        )
        self.btn_clear.clicked.connect(self.clearCanvasTriggered.emit)
        layout.addWidget(self.btn_clear)

        self._main_layout.addWidget(box)

    def _build_layout_group(self) -> None:
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

        title = QLabel("Canvas Layout")
        title.setStyleSheet("color: #DDD; font-size: 13px; font-weight: bold;")
        layout.addWidget(title)
        self._add_sep(layout)

        row_gap = QHBoxLayout()
        row_gap.addWidget(QLabel("Grid gap (px):"))
        self.spin_gap = QSpinBox()
        self.spin_gap.setRange(0, 2000)
        self.spin_gap.setValue(200)
        self.spin_gap.setToolTip(
            "Spacing in image pixels between samples on the canvas.\n"
            "TMA grids are laid out using the _rxcx_ pattern in sample names."
        )
        row_gap.addWidget(self.spin_gap)
        layout.addLayout(row_gap)

        info = QLabel(
            "TMA cores named with _r{row}c{col}_ (e.g. Core_r2c3_image.tiff)\n"
            "are arranged automatically in a proportional grid."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #777; font-size: 10px; font-style: italic;")
        layout.addWidget(info)

        self._main_layout.addWidget(box)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _add_sep(self, layout: QVBoxLayout) -> None:
        pass

    def _browse_master_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Annotation File", "",
            "Annotation files (*.xlsx *.xls *.csv);;All files (*)"
        )
        if path:
            self.lbl_master_file.setText(path.split("/")[-1])
            self.masterFileTriggered.emit(path)

    def _on_select_all_toggled(self, checked: bool) -> None:
        self.list_samples.blockSignals(True)
        for i in range(self.list_samples.count()):
            self.list_samples.item(i).setCheckState(
                Qt.Checked if checked else Qt.Unchecked
            )
        self.list_samples.blockSignals(False)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        any_unchecked = any(
            self.list_samples.item(i).checkState() == Qt.Unchecked
            for i in range(self.list_samples.count())
        )
        self.chk_select_all.blockSignals(True)
        self.chk_select_all.setChecked(
            not any_unchecked if self.list_samples.count() > 0 else False
        )
        self.chk_select_all.blockSignals(False)

    def _on_add_clicked(self) -> None:
        selected = [
            self.list_samples.item(i).text()
            for i in range(self.list_samples.count())
            if self.list_samples.item(i).checkState() == Qt.Checked
        ]
        if not selected:
            QMessageBox.information(
                self, "Nothing Selected",
                "Check at least one sample before adding to canvas."
            )
            return
        self.btn_add.setEnabled(False)
        self.btn_add.setText("Loading…")
        self.addSamplesTriggered.emit(selected)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_sample_list(self, samples: List[str]) -> None:
        """Rebuild the checklist from a list of sample names."""
        self.list_samples.blockSignals(True)
        self.list_samples.clear()
        for s in samples:
            item = QListWidgetItem(s)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_samples.addItem(item)
        self.list_samples.blockSignals(False)
        self.chk_select_all.blockSignals(True)
        self.chk_select_all.setChecked(False)
        self.chk_select_all.blockSignals(False)

    def set_loading_done(self) -> None:
        """Restore button state after samples finish loading."""
        self.btn_add.setEnabled(True)
        self.btn_add.setText("Add Selected Samples to Canvas")

    def get_gap(self) -> int:
        return self.spin_gap.value()

    def selected_samples(self) -> List[str]:
        return [
            self.list_samples.item(i).text()
            for i in range(self.list_samples.count())
            if self.list_samples.item(i).checkState() == Qt.Checked
        ]
