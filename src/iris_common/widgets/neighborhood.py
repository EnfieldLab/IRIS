"""
IRIS Cellular Neighborhoods Profiler Widget
============================================
Visualizes and analyzes local cellular niches (neighborhoods) with stacked
bar composition charts, custom metadata curation, and high-resolution export.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from iris_common.io.exporters import export_neighborhood_composition_matrix


class IrisCellularNeighborhoodsWidget(QWidget):
    """
    Dock widget for spatial cellular microenvironment / neighborhood analysis.

    Signals
    -------
    runNicheAnalysis : Signal(int, int, str, float)
        Emitted when the user triggers computation:
        (k_neighbors, n_niches, method, resolution).
    nicheMetadataChanged : Signal(dict)
        Emitted when niche names or colors are updated:
        {niche_id: {"name": str, "color": str}}.
    """

    runNicheAnalysis = Signal(int, int, str, float)
    nicheMetadataChanged = Signal(dict)

    DEFAULT_PALETTE = [
        "#FF0055", "#00FFaa", "#aa00FF", "#FFFF00", "#00FFFF",
        "#FFA500", "#FF00FF", "#00E5FF", "#76FF03", "#FFD600",
        "#E040FB", "#FF6D00", "#1DE9B6", "#FF5252", "#69F0AE",
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.cached_centers: Optional[np.ndarray] = None
        self.cached_labels: Optional[List[str]] = None
        self.cached_df: Optional[pd.DataFrame] = None
        self.niche_metadata: Dict[int, Dict[str, str]] = {}
        self.legend_widget_ref = None

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)
        self.setLayout(self.main_layout)

        self.setStyleSheet("""
            QWidget { background-color: #1f232b; color: #E0E0E0; font-size: 12px; }
            QGroupBox { border: 1px solid #444; margin-top: 6px; padding-top: 6px; }
            QLabel { color: #DDD; }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2A2A2A; color: white; border: 1px solid #555;
                padding: 2px 4px; height: 22px;
            }
            QPushButton {
                background-color: #3A6A9A; color: white; font-weight: bold;
                border: 1px solid #558; padding: 5px;
            }
            QPushButton:hover { background-color: #4A7FBA; }
            QPushButton:disabled { background-color: #333333; color: #777777; border-color: #444; }
            QTableWidget {
                background-color: #2A2A2A; color: white; gridline-color: #444;
                border: 1px solid #555; selection-background-color: #3A6A9A;
            }
            QHeaderView::section {
                background-color: #333; color: white; padding: 4px;
                border: 1px solid #444; font-weight: bold;
            }
        """)

        self._build_ui()

    def _build_ui(self) -> None:
        # ── Configuration Box ─────────────────────────────────────────────
        config_box = QGroupBox()
        cb = QVBoxLayout()
        cb.setContentsMargins(6, 6, 6, 6)
        cb.setSpacing(6)
        config_box.setLayout(cb)

        title = QLabel("Cellular Neighborhood Profiler")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        cb.addWidget(title)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Algorithm:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(["KMeans", "Leiden"])
        self.combo_method.currentTextChanged.connect(self._toggle_method_visibility)
        row1.addWidget(self.combo_method)

        row1.addWidget(QLabel("k-Neighbors:"))
        self.spin_k = QSpinBox()
        self.spin_k.setRange(3, 200)
        self.spin_k.setValue(5)
        row1.addWidget(self.spin_k)
        cb.addLayout(row1)

        self.param_stack_layout = QHBoxLayout()
        self.lbl_clusters = QLabel("Target Niches (N):")
        self.spin_niches = QSpinBox()
        self.spin_niches.setRange(2, 50)
        self.spin_niches.setValue(5)
        self.param_stack_layout.addWidget(self.lbl_clusters)
        self.param_stack_layout.addWidget(self.spin_niches)

        self.lbl_res = QLabel("Resolution:")
        self.lbl_res.hide()
        self.spin_res = QDoubleSpinBox()
        self.spin_res.setRange(0.01, 10.0)
        self.spin_res.setValue(0.5)
        self.spin_res.setDecimals(2)
        self.spin_res.hide()
        self.param_stack_layout.addWidget(self.lbl_res)
        self.param_stack_layout.addWidget(self.spin_res)
        cb.addLayout(self.param_stack_layout)

        self.btn_niche = QPushButton("Extract Spatial Neighborhoods")
        self.btn_niche.clicked.connect(self._emit_analysis_signal)
        cb.addWidget(self.btn_niche)
        self.main_layout.addWidget(config_box, 0)

        # ── Plot & Metadata Group ─────────────────────────────────────────
        self.plot_group = QGroupBox()
        pg = QVBoxLayout()
        pg.setContentsMargins(6, 6, 6, 6)
        pg.setSpacing(6)
        self.plot_group.setLayout(pg)

        self.figure = Figure(figsize=(6, 5))
        self.figure.patch.set_facecolor('#1f232b')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(300)
        pg.addWidget(self.canvas, 3)

        self.table_meta = QTableWidget(0, 2)
        self.table_meta.setHorizontalHeaderLabels(["CN ID", "Custom Description / Name"])
        self.table_meta.verticalHeader().setVisible(False)
        self.table_meta.setMinimumHeight(140)
        self.table_meta.itemChanged.connect(self._handle_table_item_changed)

        header = self.table_meta.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_meta.setColumnWidth(0, 65)
        pg.addWidget(self.table_meta, 2)

        btn_row = QHBoxLayout()
        self.btn_export = QPushButton("Export Figure")
        self.btn_export.clicked.connect(self.export_publication_plot)
        btn_row.addWidget(self.btn_export)

        self.btn_export_data = QPushButton("Export Profiles CSV")
        self.btn_export_data.clicked.connect(self.export_neighborhood_profiles_csv)
        btn_row.addWidget(self.btn_export_data)
        pg.addLayout(btn_row)

        self.main_layout.addWidget(self.plot_group, 2)

    def _toggle_method_visibility(self, method: str) -> None:
        is_leiden = (method == "Leiden")
        self.lbl_res.setVisible(is_leiden)
        self.spin_res.setVisible(is_leiden)
        self.lbl_clusters.setVisible(not is_leiden)
        self.spin_niches.setVisible(not is_leiden)

    def _emit_analysis_signal(self) -> None:
        self.runNicheAnalysis.emit(
            self.spin_k.value(),
            self.spin_niches.value(),
            self.combo_method.currentText(),
            self.spin_res.value(),
        )

    def export_publication_plot(self) -> None:
        """Saves current composition bar chart to PDF, SVG, or high-res PNG."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Figure",
            "iris_neighborhoods_figure.pdf",
            "Vector PDF (*.pdf);;Scalable Vector Graphics (*.svg);;High-Res PNG (*.png)",
        )
        if filepath:
            self.figure.savefig(
                filepath,
                dpi=300,
                bbox_inches='tight',
                facecolor=self.figure.get_facecolor(),
            )
            QMessageBox.information(self, "Export Complete", f"Figure saved to {Path(filepath).name}.")

    def export_neighborhood_profiles_csv(self) -> None:
        """Saves composition profile matrix and cell assignment mapping to CSV."""
        if self.cached_centers is None or self.cached_labels is None:
            QMessageBox.warning(self, "No Data", "Run neighborhood computations first.")
            return

        filepath_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export Composition Matrix",
            "cellular_neighborhood_profiles.csv",
            "Comma Document (*.csv)",
        )
        if not filepath_str:
            return
        try:
            export_neighborhood_composition_matrix(
                Path(filepath_str),
                self.cached_centers,
                self.cached_labels,
                self.niche_metadata,
                self.cached_df,
            )
            QMessageBox.information(
                self,
                "Export Complete",
                "Neighborhood composition profile matrix successfully saved.",
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Export Error", f"Failed to save profile matrix: {str(exc)}"
            )

    def initialize_metadata_table(self, num_niches: int) -> None:
        """Populates the table with initial niche IDs, default names, and colors."""
        self.table_meta.blockSignals(True)
        self.table_meta.setRowCount(num_niches)
        palette = self.DEFAULT_PALETTE

        self.niche_metadata.clear()
        for i in range(num_niches):
            name_str = f"Niche {i}"
            hex_color = palette[i % len(palette)]
            self.niche_metadata[i] = {"name": name_str, "color": hex_color}

            id_item = QTableWidgetItem(f"{i}")
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            id_item.setTextAlignment(Qt.AlignCenter)
            id_item.setForeground(QColor(hex_color))
            self.table_meta.setItem(i, 0, id_item)

            name_item = QTableWidgetItem(name_str)
            self.table_meta.setItem(i, 1, name_item)

        self.table_meta.blockSignals(False)
        self.nicheMetadataChanged.emit(self.niche_metadata)

    def _handle_table_item_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        col = item.column()
        if col == 1:
            new_name = item.text().strip()
            if new_name and row in self.niche_metadata:
                self.niche_metadata[row]["name"] = new_name
                self.nicheMetadataChanged.emit(self.niche_metadata)
                if self.cached_centers is not None and self.cached_labels is not None:
                    self.display_niche_compositions(self.cached_centers, self.cached_labels)

    def display_niche_compositions(
        self,
        centers: np.ndarray,
        type_labels: List[str],
        updated_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Renders stacked bar plot of cell type proportions per niche."""
        self.cached_centers = centers
        self.cached_labels = type_labels
        if updated_df is not None:
            self.cached_df = updated_df

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#14161d')

        num_niches = centers.shape[0]
        bottom = np.zeros(num_niches)
        niches_axis_labels = [
            self.niche_metadata.get(i, {}).get("name", f"Niche {i}")
            for i in range(num_niches)
        ]

        # Extract legend color mapping for cell types
        phenotype_color_map = {}
        if self.legend_widget_ref:
            rows_to_check = getattr(
                self.legend_widget_ref, "ref_ct_rows", []
            ) or getattr(self.legend_widget_ref, "seg_rows", [])
            for row in rows_to_check:
                clean_name = row.name.split(" [")[0] if " [" in row.name else row.name
                phenotype_color_map[clean_name] = row.hex_color

        tab20_cmap = plt.colormaps.get_cmap("tab20")
        for idx, label in enumerate(type_labels):
            fractions = centers[:, idx]
            assigned_color = phenotype_color_map.get(label, tab20_cmap(idx % 20))
            ax.bar(
                niches_axis_labels,
                fractions,
                bottom=bottom,
                label=label,
                color=assigned_color,
                edgecolor='black',
                linewidth=0.5,
            )
            bottom += fractions

        ax.tick_params(colors='white', labelsize=10)
        ax.set_ylabel("Cell Type Proportion", color='white', fontsize=11)
        ax.set_xticks(range(num_niches))
        ax.set_xticklabels(niches_axis_labels, rotation=25, ha='right')
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            fontsize=9,
            facecolor='#2A2A2A',
            edgecolor='#444444',
            labelcolor='white',
        )
        for spine in ax.spines.values():
            spine.set_color('#444444')

        self.figure.tight_layout()
        self.canvas.draw()
