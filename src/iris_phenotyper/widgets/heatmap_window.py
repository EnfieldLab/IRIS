"""
Interactive Heatmap Window
Displays cluster vs marker expression.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from qtpy.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant
from qtpy.QtGui import QColor, QBrush, QLinearGradient, QPainter, QFont
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableView, QLabel, QComboBox, QHeaderView, QWidget
)

class HeatmapModel(QAbstractTableModel):
    def __init__(self, data: pd.DataFrame, mode: str = "raw", parent=None):
        super().__init__(parent)
        self._raw_data = data
        self._mode = mode
        self._update_data()
        
    def _update_data(self):
        if self._mode == "zscore":
            # Compute z-score across columns (markers)
            self._display_data = (self._raw_data - self._raw_data.mean()) / (self._raw_data.std() + 1e-8)
        else:
            self._display_data = self._raw_data
            
        self._min_val = self._display_data.values.min()
        self._max_val = self._display_data.values.max()
        # To avoid division by zero
        if self._min_val == self._max_val:
            self._max_val += 1e-8

    def set_mode(self, mode: str):
        self.beginResetModel()
        self._mode = mode
        self._update_data()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return self._display_data.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self._display_data.shape[1]

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return QVariant()
            
        val = self._display_data.iloc[index.row(), index.column()]
        
        if role == Qt.DisplayRole:
            return f"{val:.2f}"
            
        elif role == Qt.BackgroundRole:
            # Scale value to 0-1
            scaled = (val - self._min_val) / (self._max_val - self._min_val)
            scaled = max(0.0, min(1.0, scaled))
            
            # Blue = low (0), Red = high (1)
            # We can interpolate between blue (0,0,255), white (255,255,255) at 0.5, red (255,0,0) at 1
            if scaled < 0.5:
                # blue to white
                ratio = scaled * 2
                r = int(255 * ratio)
                g = int(255 * ratio)
                b = 255
            else:
                # white to red
                ratio = (scaled - 0.5) * 2
                r = 255
                g = int(255 * (1 - ratio))
                b = int(255 * (1 - ratio))
                
            return QBrush(QColor(r, g, b))
            
        elif role == Qt.ForegroundRole:
            return QBrush(QColor(0, 0, 0)) # black text for readability
            
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
            
        return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._display_data.columns[section])
            else:
                return str(self._display_data.index[section])
        return QVariant()


class GradientScaleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 20)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        
        gradient = QLinearGradient(rect.topLeft(), rect.topRight())
        gradient.setColorAt(0.0, QColor(0, 0, 255))
        gradient.setColorAt(0.5, QColor(255, 255, 255))
        gradient.setColorAt(1.0, QColor(255, 0, 0))
        
        painter.fillRect(rect, gradient)


class HeatmapWindow(QDialog):
    def __init__(self, data: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Expression Heatmap")
        self.resize(800, 600)
        self.data = data
        
        layout = QVBoxLayout(self)
        
        # Controls
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["raw", "zscore"])
        self.combo_mode.currentTextChanged.connect(self._on_mode_changed)
        ctrl_layout.addWidget(self.combo_mode)
        
        ctrl_layout.addStretch()
        
        ctrl_layout.addWidget(QLabel("Low"))
        scale_bar = GradientScaleBar()
        ctrl_layout.addWidget(scale_bar)
        ctrl_layout.addWidget(QLabel("High"))
        
        layout.addLayout(ctrl_layout)
        
        # Table
        self.table = QTableView()
        self.model = HeatmapModel(self.data, mode="raw")
        self.table.setModel(self.model)
        
        # Allow moving columns
        self.table.horizontalHeader().setSectionsMovable(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        
        layout.addWidget(self.table)
        
    def _on_mode_changed(self, mode: str):
        self.model.set_mode(mode)
