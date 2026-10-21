"""
Cohort Gating Assessment & Sample Impact Window
==============================================
Dedicated dialog for evaluating the yield and distribution of a gating strategy
across multi-sample cohorts (e.g. per-sample retention rates, horizontal yield
charts, statistical summary banners, and CSV export).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from iris_viewer.widgets.phenotyping.thresholding import find_sample_column


class SampleImpactDialog(QDialog):
    """
    Dedicated window for evaluating the impact of the current gating threshold
    across samples (all cohort samples, selected samples, or actively loaded sample).
    Displays a real-time horizontal distribution chart and breakdown table.
    """
    sampleSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cohort Gating Assessment")
        self.resize(900, 640)
        self.setMinimumSize(750, 500)
        self.setStyleSheet("""
            QDialog { background-color: #1a1d24; color: #E0E0E0; font-size: 12px; }
            QGroupBox { border: 1px solid #444; border-radius: 8px; margin-top: 6px; padding-top: 6px; font-weight: bold; color: #DDD; }
            QPushButton {
                background-color: #2D5F8B; color: white; font-weight: bold;
                border: 1px solid #4B82B5; border-radius: 4px; padding: 5px 12px; font-size: 12px;
            }
            QPushButton:hover { background-color: #3D75A8; }
            QComboBox, QLineEdit {
                background-color: #2A2A2A; color: white; border: 1px solid #555;
                border-radius: 4px; padding: 3px 6px; font-size: 12px;
            }
            QLabel { color: #DDD; font-size: 12px; }
            QTableWidget {
                background-color: #1e2128;
                alternate-background-color: #262a33;
                color: #E0E0E0;
                gridline-color: #333842;
                border: 1px solid #3c424f;
                border-radius: 6px;
            }
            QTableWidget::item {
                color: #E0E0E0;
                background-color: #1e2128;
                padding: 4px 6px;
            }
            QTableWidget::item:alternate {
                background-color: #262a33;
                color: #E0E0E0;
            }
            QTableWidget::item:selected {
                background-color: #3A6A9A;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #2A2E39;
                color: #E0E0E0;
                padding: 5px;
                border: 1px solid #3c424f;
                font-weight: bold;
            }
        """)

        self.last_df: Optional[pd.DataFrame] = None
        self.last_mask: Optional[np.ndarray] = None
        self.last_gate_label: str = "Active Gate"
        self.last_mx: str = "Marker X"
        self.last_my: str = "Marker Y"
        self.current_stats: List[Dict[str, Any]] = []

        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 10, 12, 12)
        main_layout.setSpacing(10)

        # Top Control Row
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Scope:"))
        self.combo_scope = QComboBox()
        self.combo_scope.addItems(["All Samples (Cohort)", "Selected Samples", "Actively Loaded Sample"])
        self.combo_scope.currentIndexChanged.connect(self._recalculate)
        top_bar.addWidget(self.combo_scope)

        top_bar.addWidget(QLabel("Sort By:"))
        self.combo_sort = QComboBox()
        self.combo_sort.addItems(["% Positive (High to Low)", "% Positive (Low to High)", "Sample Name (A-Z)", "Total Cells"])
        self.combo_sort.currentIndexChanged.connect(self._recalculate)
        top_bar.addWidget(self.combo_sort)

        top_bar.addStretch(1)

        self.btn_export = QPushButton("Export CSV")
        self.btn_export.clicked.connect(self._export_csv)
        top_bar.addWidget(self.btn_export)

        main_layout.addLayout(top_bar)

        # KPI Summary Banner Box (Title placed underneath/inside box)
        self.box_kpi = QGroupBox()
        bl_kpi = QVBoxLayout(self.box_kpi)
        bl_kpi.setContentsMargins(10, 3, 10, 10)
        bl_kpi.setSpacing(6)

        kpi_title = QLabel("Cohort Gating Assessment")
        kpi_title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_kpi.addWidget(kpi_title)

        row_kpi_vals = QHBoxLayout()
        self.lbl_gate_desc = QLabel("Active Gate: None")
        self.lbl_gate_desc.setStyleSheet("color: #76FF03; font-weight: bold; font-size: 12px;")
        row_kpi_vals.addWidget(self.lbl_gate_desc)

        self.lbl_cohort_stats = QLabel("Cohort Total: 0 / 0 cells (0.0%)")
        self.lbl_cohort_stats.setStyleSheet("color: #44AAFF; font-weight: bold; font-size: 12px;")
        row_kpi_vals.addWidget(self.lbl_cohort_stats)

        self.lbl_sample_range = QLabel("Sample Range: 0.0% - 0.0% | Mean: 0.0%")
        self.lbl_sample_range.setStyleSheet("color: #DDD; font-size: 11px;")
        row_kpi_vals.addWidget(self.lbl_sample_range)
        row_kpi_vals.addStretch(1)
        bl_kpi.addLayout(row_kpi_vals)

        main_layout.addWidget(self.box_kpi)

        # Splitter: Left chart, Right table
        splitter = QSplitter(Qt.Horizontal)

        # Chart container
        chart_box = QGroupBox()
        bl_chart = QVBoxLayout(chart_box)
        bl_chart.setContentsMargins(10, 3, 10, 10)
        bl_chart.setSpacing(6)

        chart_title = QLabel("Per-Sample Positive Distribution")
        chart_title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_chart.addWidget(chart_title)

        self.figure = Figure(figsize=(5.5, 5.0), dpi=90)
        self.figure.patch.set_facecolor('#1a1d24')
        self.canvas = FigureCanvas(self.figure)
        bl_chart.addWidget(self.canvas)
        splitter.addWidget(chart_box)

        # Table container
        table_box = QGroupBox()
        bl_table = QVBoxLayout(table_box)
        bl_table.setContentsMargins(10, 3, 10, 10)
        bl_table.setSpacing(6)

        table_title = QLabel("Sample Breakdown Table")
        table_title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_table.addWidget(table_title)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Sample", "Total Cells", "Positive", "% Positive", "Negative", "% Negative"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.itemDoubleClicked.connect(lambda it: self.sampleSelected.emit(self.table.item(it.row(), 0).text()))
        bl_table.addWidget(self.table)
        splitter.addWidget(table_box)

        splitter.setSizes([460, 440])
        main_layout.addWidget(splitter, 1)

    def update_distribution(
        self,
        df: Optional[pd.DataFrame],
        positive_mask: Optional[np.ndarray],
        gate_label: str,
        mx: str,
        my: str,
        selected_samples: Optional[List[str]] = None,
        active_sample: Optional[str] = None
    ):
        """Updates cross-sample calculations and redrawn distributions."""
        self.last_df = df
        self.last_mask = positive_mask
        self.last_gate_label = gate_label
        self.last_mx = mx
        self.last_my = my
        self.selected_samples = selected_samples or []
        self.active_sample = active_sample
        self._recalculate()

    def _recalculate(self):
        if self.last_df is None or self.last_df.empty or self.last_mask is None:
            self._render_empty()
            return

        df = self.last_df
        sample_col = find_sample_column(df)
        if sample_col is None:
            df = df.copy()
            df['Sample'] = "Sample_1"
            sample_col = 'Sample'

        scope = self.combo_scope.currentText()
        mask = self.last_mask

        filter_indices = np.ones(len(df), dtype=bool)
        if scope.startswith("Selected") and self.selected_samples:
            filter_indices = df[sample_col].isin(self.selected_samples).to_numpy()
        elif scope.startswith("Actively") and self.active_sample:
            filter_indices = (df[sample_col] == self.active_sample).to_numpy()

        sub_df = df[filter_indices]
        sub_mask = mask[filter_indices]

        if len(sub_df) == 0:
            self._render_empty("No cells match selected sample scope.")
            return

        sample_stats = []
        unique_samples = sub_df[sample_col].unique()

        total_pos_all = int(sub_mask.sum())
        total_cells_all = len(sub_df)
        pct_all = (total_pos_all / total_cells_all) * 100.0 if total_cells_all > 0 else 0.0

        for s_name in unique_samples:
            s_mask = (sub_df[sample_col] == s_name).to_numpy()
            s_tot = int(s_mask.sum())
            s_pos = int((s_mask & sub_mask).sum())
            s_pct = (s_pos / s_tot) * 100.0 if s_tot > 0 else 0.0
            s_neg = s_tot - s_pos
            s_neg_pct = 100.0 - s_pct
            sample_stats.append({
                'sample': str(s_name),
                'total': s_tot,
                'positive': s_pos,
                'pct_pos': s_pct,
                'negative': s_neg,
                'pct_neg': s_neg_pct
            })

        sort_opt = self.combo_sort.currentText()
        if "High to Low" in sort_opt:
            sample_stats.sort(key=lambda x: x['pct_pos'], reverse=True)
        elif "Low to High" in sort_opt:
            sample_stats.sort(key=lambda x: x['pct_pos'])
        elif "A-Z" in sort_opt:
            sample_stats.sort(key=lambda x: x['sample'])
        else:
            sample_stats.sort(key=lambda x: x['total'], reverse=True)

        self.current_stats = sample_stats

        pct_list = [s['pct_pos'] for s in sample_stats]
        min_p, max_p = (min(pct_list), max(pct_list)) if pct_list else (0.0, 0.0)
        mean_p = float(np.mean(pct_list)) if pct_list else 0.0

        self.lbl_gate_desc.setText(f"Active Gate: {self.last_gate_label} ({self.last_mx}/{self.last_my})")
        self.lbl_cohort_stats.setText(f"Cohort Total: {total_pos_all:,} / {total_cells_all:,} cells ({pct_all:.1f}%)")
        self.lbl_sample_range.setText(f"Samples: {len(sample_stats)} | Range: {min_p:.1f}% - {max_p:.1f}% | Mean: {mean_p:.1f}%")

        self.table.setRowCount(len(sample_stats))
        for r, st in enumerate(sample_stats):
            vals = [
                st['sample'],
                f"{st['total']:,}",
                f"{st['positive']:,}",
                f"{st['pct_pos']:.1f}%",
                f"{st['negative']:,}",
                f"{st['pct_neg']:.1f}%"
            ]
            for c, text in enumerate(vals):
                it = QTableWidgetItem(text)
                it.setForeground(QColor("#E0E0E0"))
                if c > 0:
                    it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, it)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#14161d')

        chart_samples = sample_stats[:35]
        names = [s['sample'] for s in chart_samples][::-1]
        pcts = [s['pct_pos'] for s in chart_samples][::-1]
        y_pos = np.arange(len(names))

        colors = ['#40916C' if p >= mean_p else '#2D5F8B' for p in pcts]
        ax.barh(y_pos, pcts, color=colors, height=0.65, edgecolor='#14161d', alpha=0.85)
        ax.axvline(mean_p, color='#FF5252', linestyle='--', linewidth=1.5, label=f"Mean: {mean_p:.1f}%")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8, color='#CCC')
        ax.set_xlabel("% Positive Cells", color='#DDD', fontsize=10)
        ax.set_xlim(0, max(100.0, max_p * 1.05 if max_p > 0 else 100.0))
        ax.tick_params(colors='#888', labelsize=8)
        ax.grid(True, color='#262930', linestyle='--', linewidth=0.5, axis='x')
        ax.legend(loc='lower right', facecolor='#1f232b', edgecolor='#444', labelcolor='#DDD', fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw()

    def _render_empty(self, msg="No active data available."):
        self.table.setRowCount(0)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#14161d')
        ax.text(0.5, 0.5, msg, color='#777', ha='center', va='center', fontsize=11)
        self.canvas.draw()

    def _export_csv(self):
        if not self.current_stats:
            QMessageBox.warning(self, "No Data", "No sample statistics to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Sample Gating Distribution", "sample_gating_distribution.csv", "CSV (*.csv)")
        if path:
            pd.DataFrame(self.current_stats).to_csv(path, index=False)
            QMessageBox.information(self, "Export Complete", f"Saved sample statistics to {Path(path).name}")
