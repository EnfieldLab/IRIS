"""
IRIS-Viewer Training Data Widget

A dock panel for automatically selecting and manually refining high-confidence
training sets from manually thresholded data. This prepares pristine datasets
for training supervised phenotyping models like STELLAR and CellSighter.
"""
from __future__ import annotations

from typing import Optional

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
    QMessageBox
)

from iris_viewer.widgets.styles import SHARED_DARK_THEME


class TrainingDataWidget(QWidget):
    """
    Dock widget for selecting high-confidence training sets.
    
    Features
    --------
    - Morphological filtering (area bounds)
    - Expression profile filtering (keep top X% most representative cells)
    - Auto-select button
    - Export button
    """
    
    autoSelectTriggered = Signal(dict)
    exportTriggered = Signal(str)
    
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
        
        self._build_morphology_group()
        self._build_expression_group()
        self._build_actions_group()
        self._main_layout.addStretch()

    def _add_sep(self, layout: QVBoxLayout) -> None:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #444; background: #444; max-height: 1px;")
        layout.addWidget(line)
        
    def _build_morphology_group(self) -> None:
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)
        
        title = QLabel("1. Morphological Filters")
        title.setStyleSheet("color: #DDD; font-size: 13px; font-weight: bold;")
        layout.addWidget(title)
        self._add_sep(layout)
        
        info = QLabel(
            "Filter out segmentation artifacts based on size."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #999; font-size: 11px; font-style: italic;")
        layout.addWidget(info)
        
        row_min_area = QHBoxLayout()
        row_min_area.addWidget(QLabel("Min Area:"))
        self.spin_min_area = QDoubleSpinBox()
        self.spin_min_area.setRange(0, 10000)
        self.spin_min_area.setValue(20.0)
        row_min_area.addWidget(self.spin_min_area)
        layout.addLayout(row_min_area)
        
        row_max_area = QHBoxLayout()
        row_max_area.addWidget(QLabel("Max Area:"))
        self.spin_max_area = QDoubleSpinBox()
        self.spin_max_area.setRange(0, 100000)
        self.spin_max_area.setValue(2000.0)
        row_max_area.addWidget(self.spin_max_area)
        layout.addLayout(row_max_area)
        
        self._main_layout.addWidget(box)
        
    def _build_expression_group(self) -> None:
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)
        
        title = QLabel("2. Expression Purity")
        title.setStyleSheet("color: #DDD; font-size: 13px; font-weight: bold;")
        layout.addWidget(title)
        self._add_sep(layout)
        
        info = QLabel(
            "Keep only the most representative cells for each cell type "
            "(highest cosine similarity to the median profile of their class)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #999; font-size: 11px; font-style: italic;")
        layout.addWidget(info)
        
        row_percentile = QHBoxLayout()
        row_percentile.addWidget(QLabel("Keep Top (%):"))
        self.spin_percentile = QSpinBox()
        self.spin_percentile.setRange(1, 100)
        self.spin_percentile.setValue(75)
        row_percentile.addWidget(self.spin_percentile)
        layout.addLayout(row_percentile)
        
        self._main_layout.addWidget(box)
        
    def _build_actions_group(self) -> None:
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)
        
        title = QLabel("3. Actions")
        title.setStyleSheet("color: #DDD; font-size: 13px; font-weight: bold;")
        layout.addWidget(title)
        self._add_sep(layout)
        
        info = QLabel(
            "Auto-select high confidence cells. You can manually refine the "
            "selection in the viewer using Napari's Points layer tools before exporting."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #999; font-size: 11px; font-style: italic;")
        layout.addWidget(info)
        
        self.btn_auto_select = QPushButton("Auto-Select High Confidence Cells")
        self.btn_auto_select.setStyleSheet(
            "QPushButton { background: #2A4A7A; color: white; padding: 5px; border-radius: 3px; }"
            "QPushButton:hover { background: #3A6AAA; }"
        )
        self.btn_auto_select.clicked.connect(self._on_auto_select)
        layout.addWidget(self.btn_auto_select)
        
        self._add_sep(layout)
        
        row_format = QHBoxLayout()
        row_format.addWidget(QLabel("Format:"))
        self.combo_format = QComboBox()
        self.combo_format.addItems(["AnnData (.h5ad) - STELLAR", "CSV - CellSighter"])
        row_format.addWidget(self.combo_format)
        layout.addLayout(row_format)
        
        self.btn_export = QPushButton("Export Training Set")
        self.btn_export.setStyleSheet(
            "QPushButton { background: #2A7A4A; color: white; padding: 5px; border-radius: 3px; }"
            "QPushButton:hover { background: #3Aaa6A; }"
        )
        self.btn_export.clicked.connect(self._on_export)
        layout.addWidget(self.btn_export)
        
        self._main_layout.addWidget(box)
        
    def _on_auto_select(self) -> None:
        params = {
            "min_area": self.spin_min_area.value(),
            "max_area": self.spin_max_area.value(),
            "keep_top_percent": self.spin_percentile.value()
        }
        self.btn_auto_select.setText("Selecting...")
        self.btn_auto_select.setEnabled(False)
        self.autoSelectTriggered.emit(params)
        
    def reset_select_btn(self) -> None:
        self.btn_auto_select.setText("Auto-Select High Confidence Cells")
        self.btn_auto_select.setEnabled(True)
        
    def _on_export(self) -> None:
        fmt = self.combo_format.currentText()
        self.exportTriggered.emit(fmt)
