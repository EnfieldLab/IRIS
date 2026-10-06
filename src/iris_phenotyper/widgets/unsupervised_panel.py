"""
IRIS-Phenotyper Unsupervised Panel

Allows running unsupervised clustering methods (Leiden, PhenoGraph)
and visualizing the average marker expression per cluster in a heatmap.
"""
from __future__ import annotations

from typing import Optional, List
import pandas as pd
import numpy as np

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSpinBox, QPushButton, QGroupBox, QListWidget, QAbstractItemView
)

from iris_common.widgets.styles import SHARED_DARK_THEME

class UnsupervisedPanelWidget(QWidget):
    """
    Panel for running unsupervised clustering (Leiden / PhenoGraph)
    and selecting clusters to view their expression heatmap.
    """
    
    runClusteringRequested = Signal(str, int)  # method, k
    showHeatmapRequested = Signal(list)  # list of selected cluster IDs
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(SHARED_DARK_THEME)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Clustering Parameters ─────────────────────────────────────────────
        param_box = QGroupBox("Clustering Configuration")
        param_layout = QVBoxLayout(param_box)

        # Method
        row_method = QHBoxLayout()
        row_method.addWidget(QLabel("Method:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Leiden", "PhenoGraph"])
        row_method.addWidget(self.combo_method)
        param_layout.addLayout(row_method)

        # k-nearest neighbors
        row_knn = QHBoxLayout()
        row_knn.addWidget(QLabel("k-Nearest Neighbors:"))
        self.spin_knn = QSpinBox()
        self.spin_knn.setRange(5, 500)
        self.spin_knn.setValue(30)
        row_knn.addWidget(self.spin_knn)
        param_layout.addLayout(row_knn)

        self.btn_run = QPushButton("Run Clustering")
        self.btn_run.clicked.connect(self._emit_run)
        param_layout.addWidget(self.btn_run)
        
        layout.addWidget(param_box)

        # ── Cluster Selection ─────────────────────────────────────────────────
        sel_box = QGroupBox("Select Clusters")
        sel_layout = QVBoxLayout(sel_box)

        self.list_clusters = QListWidget()
        self.list_clusters.setSelectionMode(QAbstractItemView.ExtendedSelection)
        sel_layout.addWidget(QLabel("Select one or a range of clusters:"))
        sel_layout.addWidget(self.list_clusters)

        self.btn_heatmap = QPushButton("Show Expression Heatmap")
        self.btn_heatmap.setEnabled(False)
        self.btn_heatmap.clicked.connect(self._emit_show_heatmap)
        sel_layout.addWidget(self.btn_heatmap)

        layout.addWidget(sel_box)
        layout.addStretch()

    def set_running_state(self, is_running: bool) -> None:
        self.btn_run.setEnabled(not is_running)
        self.btn_run.setText("Running..." if is_running else "Run Clustering")

    def populate_clusters(self, cluster_ids: List[int]) -> None:
        self.list_clusters.clear()
        for c in sorted(cluster_ids):
            self.list_clusters.addItem(f"Cluster {c}")
        
        # Select all by default
        for i in range(self.list_clusters.count()):
            self.list_clusters.item(i).setSelected(True)
            
        self.btn_heatmap.setEnabled(len(cluster_ids) > 0)

    def _emit_run(self) -> None:
        method = self.combo_method.currentText()
        k = self.spin_knn.value()
        self.runClusteringRequested.emit(method, k)

    def _emit_show_heatmap(self) -> None:
        selected_items = self.list_clusters.selectedItems()
        clusters = []
        for item in selected_items:
            # Extract number from "Cluster X"
            c = int(item.text().split(" ")[1])
            clusters.append(c)
        if clusters:
            self.showHeatmapRequested.emit(clusters)
