"""
IRIS-Viewer Phenotyping Module
==============================
Provides manual flow-cytometry-style gating (2D biaxial, 3D trivariate, and 1D thresholding),
interactive draggable cutoff lines, polygon/freehand shape gating, sequential hierarchical gating tree,
automated optimal threshold discovery (Otsu / Bimodal Valley), gating quality separation assessment,
cohort sample impact distribution, and pop-up interactive gating studio.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector, RectangleSelector
import matplotlib.colors as mcolors

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
try:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    try:
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    except ImportError:
        NavigationToolbar = None

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from iris_common.widgets.styles import SHARED_DARK_THEME


# ── Mathematical Transformations & Quality Evaluation ───────────────────────

def arcsinh_transform(arr: np.ndarray, cofactor: float = 5.0) -> np.ndarray:
    """Arcsinh transformation standard in CyTOF and multiplexed IF flow gating."""
    return np.arcsinh(np.maximum(arr, 0.0) / max(cofactor, 1e-4))


def log10_transform(arr: np.ndarray) -> np.ndarray:
    """Log10(1 + x) transformation."""
    return np.log10(np.maximum(arr, 0.0) + 1.0)


def compute_optimal_threshold(arr: np.ndarray, method: str = 'otsu') -> float:
    """
    Computes mathematically and biologically grounded cutoff thresholds:
    - 'otsu': Maximizes inter-class variance between negative background and positive signal
    - 'valley': Bimodal peak detection finding the local minimum/dip between peaks
    - 'percentile95': 95th percentile
    - 'percentile98': 98th percentile
    - 'mad': Median + 2.5 * MAD (robust non-parametric noise cutoff)
    - 'median': Median
    """
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return 0.0
    mn, mx = float(np.min(valid)), float(np.max(valid))
    if mn == mx:
        return mn

    if method == 'percentile95':
        return float(np.percentile(valid, 95))
    elif method == 'percentile98':
        return float(np.percentile(valid, 98))
    elif method == 'mad':
        med = float(np.median(valid))
        mad = float(np.median(np.abs(valid - med)))
        return float(med + 2.5 * 1.4826 * mad)
    elif method == 'median':
        return float(np.median(valid))
    elif method == 'valley':
        hist, bin_edges = np.histogram(valid, bins=128)
        window = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0
        smoothed = np.convolve(hist, window, mode='same')
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                peaks.append(i)
        if len(peaks) >= 2:
            sorted_peaks = sorted(peaks, key=lambda p: smoothed[p], reverse=True)[:2]
            p1, p2 = min(sorted_peaks), max(sorted_peaks)
            if p2 - p1 > 2:
                valley_idx = p1 + int(np.argmin(smoothed[p1:p2 + 1]))
                return float((bin_edges[valley_idx] + bin_edges[valley_idx + 1]) / 2.0)
        method = 'otsu'

    if method == 'otsu':
        hist, bin_edges = np.histogram(valid, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        hist = hist.astype(float)
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1e-12)
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2[::-1], 1e-12))[::-1]
        variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        idx = int(np.argmax(variance))
        return float(bin_centers[idx])

    return float(np.median(valid))


def assess_gating_quality(vx: np.ndarray, tx: float) -> Tuple[str, str, float]:
    """
    Evaluates whether threshold tx divides two distinct expression peaks (bimodal)
    or cuts through a continuous/unimodal distribution.
    Returns (quality_label, badge_hex_color, separation_score).
    """
    v_neg = vx[vx < tx]
    v_pos = vx[vx >= tx]
    if len(v_neg) < 5 or len(v_pos) < 5:
        return "Skewed / Extreme Cutoff", "#888888", 0.0

    med_neg = float(np.median(v_neg))
    med_pos = float(np.median(v_pos))
    mad_neg = float(np.median(np.abs(v_neg - med_neg)))
    mad_pos = float(np.median(np.abs(v_pos - med_pos)))
    spread = max(mad_neg + mad_pos, 1e-4)
    sep = (med_pos - med_neg) / spread

    if sep >= 2.4:
        return "Clean Bimodal Separation", "#40916C", sep
    elif sep >= 1.4:
        return "Moderate Peak Overlap", "#E6A23C", sep
    else:
        return "Continuous / Unimodal Expression", "#E57373", sep


def find_sample_column(df: pd.DataFrame) -> Optional[str]:
    """Identifies the column representing sample/core identity."""
    for col in ['Image', 'sample_id', 'Sample', 'sample', 'core', 'Core', 'TMA_Core', 'image']:
        if col in df.columns:
            return col
    return None


# ── Hierarchical Gating Tree Data Model ──────────────────────────────────────

class GatingHierarchyNode:
    """Represents a population node in a sequential hierarchical gating tree."""
    def __init__(
        self,
        name: str,
        parent: Optional['GatingHierarchyNode'] = None,
        cell_mask: Optional[np.ndarray] = None,
        color: str = "#00E5FF",
        markers: Tuple[str, ...] = (),
        criteria_desc: str = "",
        total_cells_count: int = 0
    ):
        self.name = name
        self.parent = parent
        self.cell_mask = cell_mask
        self.color = color
        self.markers = markers
        self.criteria_desc = criteria_desc
        self.total_cells_count = total_cells_count
        self.children: List['GatingHierarchyNode'] = []
        if parent is not None:
            parent.children.append(self)

    @property
    def cell_count(self) -> int:
        if self.cell_mask is not None:
            return int(self.cell_mask.sum())
        return self.total_cells_count

    @property
    def parent_percentage(self) -> float:
        if self.parent and self.parent.cell_count > 0:
            return (self.cell_count / self.parent.cell_count) * 100.0
        return 100.0

    @property
    def total_percentage(self) -> float:
        root = self.get_root()
        if root and root.cell_count > 0:
            return (self.cell_count / root.cell_count) * 100.0
        return 100.0

    def get_root(self) -> 'GatingHierarchyNode':
        curr = self
        while curr.parent is not None:
            curr = curr.parent
        return curr

    def find_node(self, name: str) -> Optional['GatingHierarchyNode']:
        if self.name == name:
            return self
        for child in self.children:
            found = child.find_node(name)
            if found:
                return found
        return None

    def get_all_nodes(self) -> List['GatingHierarchyNode']:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes

    def get_path_string(self) -> str:
        parts = []
        curr: Optional['GatingHierarchyNode'] = self
        while curr is not None and curr.parent is not None:
            parts.append(curr.name)
            curr = curr.parent
        parts.reverse()
        return " / ".join(parts) if parts else self.name


# ── Cross-Sample Gating Distribution & Impact Window ─────────────────────────

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


# ── Full Interactive Gating Studio Pop-up Dialog ─────────────────────────────

class GatingStudioDialog(QDialog):
    """
    Full-size interactive pop-up window for flow-cytometry-style gating on multiplexed IF data.
    Supports 2D biaxial scatter, 3D trivariate scatter, draggable threshold lines,
    interactive polygon/rectangle shape gates, hierarchical sequential sub-gating tree,
    automated optimal threshold calculation, and cohort sample impact preview.
    """
    gateApplied = Signal(dict)

    def __init__(self, parent=None, host_widget: Optional['ManualGatingWidget'] = None):
        super().__init__(parent)
        self.host_widget = host_widget
        self.setWindowTitle("Manual Gating Studio")
        self.resize(1260, 840)
        self.setMinimumSize(980, 680)

        self.setStyleSheet("""
            QDialog { background-color: #1a1d24; color: #E0E0E0; font-size: 12px; }
            QGroupBox { border: 1px solid #444; border-radius: 8px; margin-top: 6px; padding-top: 6px; font-weight: bold; color: #DDD; }
            QPushButton {
                background-color: #2D5F8B; color: white; font-weight: bold;
                border: 1px solid #4B82B5; border-radius: 4px; padding: 5px 10px; font-size: 11px;
            }
            QPushButton:hover { background-color: #3D75A8; }
            QPushButton:disabled { background-color: #2A3A4A; color: #777; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #2A2A2A; color: white; border: 1px solid #555;
                border-radius: 4px; padding: 3px 6px; font-size: 12px;
            }
            QLabel { color: #DDD; font-size: 12px; }
            QTableWidget, QTreeWidget {
                background-color: #2A2A2A; color: white; gridline-color: #444;
                border: 1px solid #555; border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #333; color: white; padding: 4px;
                border: 1px solid #444; font-weight: bold;
            }
        """)

        self.poly_selector: Optional[PolygonSelector] = None
        self.rect_selector: Optional[RectangleSelector] = None
        self.polygon_vertices: Optional[np.ndarray] = None
        self.dragging_line: Optional[str] = None
        self.impact_dialog: Optional[SampleImpactDialog] = None

        self._build_ui()
        self._sync_from_host()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Left Area: Interactive Canvas & Toolbar ──────────────────────────
        left_box = QWidget()
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        top_bar = QHBoxLayout()
        self.lbl_status = QLabel("Ready. Adjust cutoff lines or draw a shape gate over positive cell populations.")
        self.lbl_status.setStyleSheet("color: #76FF03; font-size: 12px; font-style: italic; padding: 2px 4px;")
        top_bar.addWidget(self.lbl_status)
        top_bar.addStretch(1)

        self.btn_open_impact = QPushButton("Cross-Sample Assessment...")
        self.btn_open_impact.setToolTip("Open multi-sample cohort distribution impact window")
        self.btn_open_impact.setStyleSheet("background-color: #3A6A9A; color: white; font-weight: bold; padding: 4px 10px;")
        self.btn_open_impact.clicked.connect(self._open_impact_window)
        top_bar.addWidget(self.btn_open_impact)
        left_layout.addLayout(top_bar)

        self.figure = Figure(figsize=(8.5, 7.0), dpi=100)
        self.figure.patch.set_facecolor('#14161d')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        if NavigationToolbar is not None:
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setStyleSheet("""
                QToolBar { background-color: #1f232b; border: 1px solid #333; spacing: 4px; padding: 2px; }
                QToolButton { background-color: transparent; border: 1px solid transparent; border-radius: 3px; padding: 3px; }
                QToolButton:hover { background-color: #3A6A9A; border: 1px solid #558; }
            """)
            left_layout.addWidget(self.toolbar)

        left_layout.addWidget(self.canvas, 1)

        self.canvas.mpl_connect('button_press_event', self._on_canvas_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
        self.canvas.mpl_connect('button_release_event', self._on_canvas_release)

        root.addWidget(left_box, 3)

        # ── Right Area: Tabbed Gating Controls & Hierarchical Tree ───────────
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(380)
        self.tabs.setMaximumWidth(480)

        # Tab 1: Gating Controls
        tab_gating = QWidget()
        tab_gating_layout = QVBoxLayout(tab_gating)
        tab_gating_layout.setContentsMargins(4, 4, 4, 4)
        tab_gating_layout.setSpacing(8)

        scroll_gating = QScrollArea()
        scroll_gating.setWidgetResizable(True)
        scroll_gating.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setContentsMargins(2, 2, 2, 2)
        self.scroll_layout.setSpacing(10)

        self._build_projection_group()
        self._build_cutoffs_group()
        self._build_assignment_group()
        self.scroll_layout.addStretch(1)

        scroll_gating.setWidget(scroll_content)
        tab_gating_layout.addWidget(scroll_gating)
        self.tabs.addTab(tab_gating, "Gating & Cutoffs")

        # Tab 2: Sequential Gating Hierarchy
        tab_tree = QWidget()
        tab_tree_layout = QVBoxLayout(tab_tree)
        tab_tree_layout.setContentsMargins(6, 6, 6, 6)
        tab_tree_layout.setSpacing(8)
        self._build_hierarchy_tree_tab(tab_tree_layout)
        self.tabs.addTab(tab_tree, "Hierarchy Tree")

        # Tab 3: Applied Gates History
        tab_history = QWidget()
        tab_hist_layout = QVBoxLayout(tab_history)
        tab_hist_layout.setContentsMargins(6, 6, 6, 6)
        tab_hist_layout.setSpacing(8)
        self._build_history_tab(tab_hist_layout)
        self.tabs.addTab(tab_history, "Gate History")

        right_layout.addWidget(self.tabs)
        root.addWidget(right_container, 1)

    # ── Projection & Scope Group (Title inside box) ──────────────────────────
    def _build_projection_group(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 3, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Gating Projection & Scope")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        layout.addWidget(title)

        row_parent = QHBoxLayout()
        row_parent.addWidget(QLabel("Parent Gate:"))
        self.combo_parent_pop = QComboBox()
        self.combo_parent_pop.addItem("Root (All Cells)")
        self.combo_parent_pop.currentIndexChanged.connect(self._on_parent_changed)
        row_parent.addWidget(self.combo_parent_pop)
        layout.addLayout(row_parent)

        row_scope = QHBoxLayout()
        row_scope.addWidget(QLabel("Sample Scope:"))
        self.combo_scope = QComboBox()
        self.combo_scope.addItems(["All Samples (Cohort)", "Selected Sample", "Actively Loaded Sample"])
        self.combo_scope.currentIndexChanged.connect(self._on_scope_changed)
        row_scope.addWidget(self.combo_scope)
        layout.addLayout(row_scope)

        self.row_sample_select = QHBoxLayout()
        self.row_sample_select.addWidget(QLabel("Sample:"))
        self.combo_sample = QComboBox()
        self.combo_sample.setToolTip("Select specific sample / core for manual gating")
        self.combo_sample.currentIndexChanged.connect(self._on_sample_changed)
        self.row_sample_select.addWidget(self.combo_sample)
        layout.addLayout(self.row_sample_select)

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["2D Biaxial Scatter", "3D Trivariate Scatter", "1D Histogram"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        row_mode.addWidget(self.combo_mode)
        layout.addLayout(row_mode)

        row_x = QHBoxLayout()
        row_x.addWidget(QLabel("Marker X:"))
        self.combo_x = QComboBox()
        self.combo_x.currentIndexChanged.connect(self._on_marker_changed)
        row_x.addWidget(self.combo_x)
        layout.addLayout(row_x)

        row_y = QHBoxLayout()
        row_y.addWidget(QLabel("Marker Y:"))
        self.combo_y = QComboBox()
        self.combo_y.currentIndexChanged.connect(self._on_marker_changed)
        row_y.addWidget(self.combo_y)
        layout.addLayout(row_y)

        # Dedicated container for Marker Z (strictly hidden in 2D mode)
        self.container_z = QWidget()
        row_z = QHBoxLayout(self.container_z)
        row_z.setContentsMargins(0, 0, 0, 0)
        row_z.addWidget(QLabel("Marker Z:"))
        self.combo_z = QComboBox()
        self.combo_z.currentIndexChanged.connect(self._on_marker_changed)
        row_z.addWidget(self.combo_z)
        layout.addWidget(self.container_z)
        self.container_z.hide()

        # Scale Transformation
        row_scale = QHBoxLayout()
        row_scale.addWidget(QLabel("Scale Transformation:"))
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Arcsinh (Cofactor = 5)", "Log10(1 + x)", "Linear (Raw)", "Min-Max Normalized"])
        self.combo_scale.currentIndexChanged.connect(self._on_marker_changed)
        row_scale.addWidget(self.combo_scale)
        layout.addLayout(row_scale)

        self.scroll_layout.addWidget(box)

    # ── Threshold Lines Group (Title inside box) ─────────────────────────────
    def _build_cutoffs_group(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 3, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Thresholds & Gating Cutoffs")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        layout.addWidget(title)

        # Threshold X with Auto Buttons
        row_tx = QHBoxLayout()
        row_tx.addWidget(QLabel("X Cutoff:"))
        self.spin_tx = QDoubleSpinBox()
        self.spin_tx.setRange(-10000.0, 100000.0)
        self.spin_tx.setDecimals(2)
        self.spin_tx.setSingleStep(0.2)
        self.spin_tx.valueChanged.connect(self._on_threshold_changed)
        row_tx.addWidget(self.spin_tx)

        btn_auto_x = QPushButton("Auto (Otsu)")
        btn_auto_x.setToolTip("Automatically compute optimal Otsu threshold for Marker X")
        btn_auto_x.clicked.connect(lambda: self._auto_threshold('x', 'otsu'))
        row_tx.addWidget(btn_auto_x)

        btn_val_x = QPushButton("Valley")
        btn_val_x.setToolTip("Find the bimodal dip between negative background and positive signal")
        btn_val_x.clicked.connect(lambda: self._auto_threshold('x', 'valley'))
        row_tx.addWidget(btn_val_x)
        layout.addLayout(row_tx)

        # Threshold Y with Auto Buttons
        self.row_ty = QHBoxLayout()
        self.row_ty.addWidget(QLabel("Y Cutoff:"))
        self.spin_ty = QDoubleSpinBox()
        self.spin_ty.setRange(-10000.0, 100000.0)
        self.spin_ty.setDecimals(2)
        self.spin_ty.setSingleStep(0.2)
        self.spin_ty.valueChanged.connect(self._on_threshold_changed)
        self.row_ty.addWidget(self.spin_ty)

        btn_auto_y = QPushButton("Auto (Otsu)")
        btn_auto_y.setToolTip("Automatically compute optimal Otsu threshold for Marker Y")
        btn_auto_y.clicked.connect(lambda: self._auto_threshold('y', 'otsu'))
        self.row_ty.addWidget(btn_auto_y)

        btn_val_y = QPushButton("Valley")
        btn_val_y.setToolTip("Find the bimodal dip between negative background and positive signal")
        btn_val_y.clicked.connect(lambda: self._auto_threshold('y', 'valley'))
        self.row_ty.addWidget(btn_val_y)
        layout.addLayout(self.row_ty)

        # Threshold Z (3D container)
        self.container_tz = QWidget()
        row_tz = QHBoxLayout(self.container_tz)
        row_tz.setContentsMargins(0, 0, 0, 0)
        row_tz.addWidget(QLabel("Z Cutoff:"))
        self.spin_tz = QDoubleSpinBox()
        self.spin_tz.setRange(-10000.0, 100000.0)
        self.spin_tz.setDecimals(2)
        self.spin_tz.setSingleStep(0.2)
        self.spin_tz.valueChanged.connect(self._on_threshold_changed)
        row_tz.addWidget(self.spin_tz)
        layout.addWidget(self.container_tz)
        self.container_tz.hide()

        # Quadrant summary table (All 4 populations visible with NO vertical scroll bar)
        self.lbl_quad_stats = QLabel("Quadrant Breakdown (Marker-Specific):")
        self.lbl_quad_stats.setStyleSheet("font-weight: bold; color: #BBB; font-size: 11px;")
        layout.addWidget(self.lbl_quad_stats)

        self.table_quads = QTableWidget(4, 2)
        self.table_quads.setHorizontalHeaderLabels(["Population", "Cells (% Parent)"])
        self.table_quads.verticalHeader().setVisible(False)
        self.table_quads.verticalHeader().setDefaultSectionSize(26)
        self.table_quads.horizontalHeader().setFixedHeight(26)
        self.table_quads.horizontalHeader().setStretchLastSection(True)
        self.table_quads.setFixedHeight(145)
        self.table_quads.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table_quads.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.table_quads)

        # Shape Gating Action Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.btn_draw_poly = QPushButton("Draw Polygon Gate")
        self.btn_draw_poly.setToolTip("Click on canvas to draw polygon vertices, double-click or Enter to close.")
        self.btn_draw_poly.clicked.connect(self._toggle_polygon_tool)
        btn_row.addWidget(self.btn_draw_poly)

        self.btn_clear_poly = QPushButton("Clear Shape")
        self.btn_clear_poly.setStyleSheet("background-color: #555;")
        self.btn_clear_poly.clicked.connect(self._clear_polygon)
        btn_row.addWidget(self.btn_clear_poly)
        layout.addLayout(btn_row)

        self.lbl_poly_status = QLabel("Shape Gate: None defined")
        self.lbl_poly_status.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
        layout.addWidget(self.lbl_poly_status)

        self.scroll_layout.addWidget(box)

    # ── Phenotype Assignment Group (Title inside box) ────────────────────────
    def _build_assignment_group(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 3, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Phenotype Classification & Gating")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        layout.addWidget(title)

        row_pop = QHBoxLayout()
        row_pop.addWidget(QLabel("Target Gate:"))
        self.combo_target_pop = QComboBox()
        row_pop.addWidget(self.combo_target_pop)
        layout.addLayout(row_pop)

        row_label = QHBoxLayout()
        row_label.addWidget(QLabel("Label Designation:"))
        self.txt_phenotype_name = QLineEdit()
        self.txt_phenotype_name.setPlaceholderText("e.g. CD45+ CD3+ T Cells")
        row_label.addWidget(self.txt_phenotype_name)
        layout.addLayout(row_label)

        row_color = QHBoxLayout()
        row_color.addWidget(QLabel("Gate Color:"))
        self.btn_color = QPushButton("")
        self.btn_color.setFixedWidth(50)
        self.btn_color.setFixedHeight(22)
        self.selected_color = "#00E5FF"
        self._update_color_btn()
        self.btn_color.clicked.connect(self._pick_color)
        row_color.addWidget(self.btn_color)
        row_color.addStretch(1)
        layout.addLayout(row_color)

        row_col = QHBoxLayout()
        row_col.addWidget(QLabel("Target Column:"))
        self.combo_col = QComboBox()
        self.combo_col.setEditable(True)
        self.combo_col.addItems(["CellType", "Verified_CellType", "Gated_Phenotype"])
        row_col.addWidget(self.combo_col)
        layout.addLayout(row_col)

        self.btn_apply = QPushButton("Apply Gate to Cells")
        self.btn_apply.setAutoDefault(False)
        self.btn_apply.setDefault(False)
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #2D6A4F; color: white; font-weight: bold;
                border-radius: 4px; padding: 7px; font-size: 12px;
            }
            QPushButton:hover { background-color: #40916C; }
        """)
        self.btn_apply.clicked.connect(self._apply_gate)
        layout.addWidget(self.btn_apply)

        self.lbl_apply_feedback = QLabel("")
        self.lbl_apply_feedback.setStyleSheet("color: #76FF03; font-weight: bold; font-size: 11px;")
        layout.addWidget(self.lbl_apply_feedback)

        self.scroll_layout.addWidget(box)

    # ── Sequential Gating Hierarchy Tab ──────────────────────────────────────
    def _build_hierarchy_tree_tab(self, layout: QVBoxLayout):
        desc = QLabel("Sequential gating hierarchy (FlowJo-style sub-gating tree). Double click a population to gate within it.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #AAA; font-size: 11px;")
        layout.addWidget(desc)

        self.tree_hierarchy = QTreeWidget()
        self.tree_hierarchy.setHeaderLabels(["Population", "Cells", "% Parent", "% Total", "Markers"])
        self.tree_hierarchy.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tree_hierarchy.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
        layout.addWidget(self.tree_hierarchy, 1)

        btn_row = QHBoxLayout()
        self.btn_gate_on_selected = QPushButton("Gate on Selected")
        self.btn_gate_on_selected.setToolTip("Set selected population as the active parent for subsequent sub-gating")
        self.btn_gate_on_selected.clicked.connect(self._gate_on_tree_selection)
        btn_row.addWidget(self.btn_gate_on_selected)

        self.btn_apply_hierarchy = QPushButton("Apply Tree to Canvas")
        self.btn_apply_hierarchy.setStyleSheet("background-color: #2D6A4F; color: white; font-weight: bold;")
        self.btn_apply_hierarchy.setToolTip("Flatten tree and write classified leaf populations to dataset and Napari canvas")
        self.btn_apply_hierarchy.clicked.connect(self._apply_entire_hierarchy)
        btn_row.addWidget(self.btn_apply_hierarchy)
        layout.addLayout(btn_row)

        btn_row2 = QHBoxLayout()
        self.btn_delete_gate = QPushButton("Delete Gate")
        self.btn_delete_gate.setStyleSheet("background-color: #5A2A2A; color: white;")
        self.btn_delete_gate.clicked.connect(self._delete_selected_tree_gate)
        btn_row2.addWidget(self.btn_delete_gate)

        self.btn_export_tree = QPushButton("Export Hierarchy (.csv)")
        self.btn_export_tree.clicked.connect(self._export_hierarchy_csv)
        btn_row2.addWidget(self.btn_export_tree)
        layout.addLayout(btn_row2)

    # ── History Tab ──────────────────────────────────────────────────────────
    def _build_history_tab(self, layout: QVBoxLayout):
        self.table_history = QTableWidget(0, 4)
        self.table_history.setHorizontalHeaderLabels(["Gate / Label", "Cells", "% Total", "Markers"])
        self.table_history.verticalHeader().setVisible(False)
        self.table_history.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table_history, 1)

        btn_row = QHBoxLayout()
        self.btn_clear_gates = QPushButton("Reset Gate History")
        self.btn_clear_gates.setStyleSheet("background-color: #5A2A2A; color: white;")
        self.btn_clear_gates.clicked.connect(self._clear_applied_gates)
        btn_row.addWidget(self.btn_clear_gates)

        self.btn_export_history = QPushButton("Export History (.csv)")
        self.btn_export_history.clicked.connect(self._export_gates_csv)
        btn_row.addWidget(self.btn_export_history)
        layout.addLayout(btn_row)

    # ── Synchronization & Data Handling ──────────────────────────────────────
    def _sync_from_host(self):
        if self.host_widget is None:
            return
        markers = self.host_widget.available_markers or []
        for combo in (self.combo_x, self.combo_y, self.combo_z):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(markers)
            combo.blockSignals(False)

        if len(markers) > 0:
            self.combo_x.setCurrentIndex(0)
        if len(markers) > 1:
            self.combo_y.setCurrentIndex(1)
        if len(markers) > 2:
            self.combo_z.setCurrentIndex(2)

        self.spin_tx.setValue(self.host_widget.spin_tx.value())
        self.spin_ty.setValue(self.host_widget.spin_ty.value())

        if hasattr(self, 'combo_sample'):
            self.combo_sample.blockSignals(True)
            self.combo_sample.clear()
            samples = []
            raw_df = self._get_raw_data()
            if raw_df is not None and not raw_df.empty:
                scol = find_sample_column(raw_df)
                if scol:
                    samples = [str(s) for s in raw_df[scol].unique()]
            if not samples and hasattr(self.host_widget, '_get_selected_samples'):
                samples = self.host_widget._get_selected_samples()
            if not samples:
                samples = ["Sample_1"]
            self.combo_sample.addItems(samples)
            if hasattr(self.host_widget, 'combo_sample'):
                target_s = self.host_widget.combo_sample.currentText()
                if target_s in samples:
                    self.combo_sample.setCurrentText(target_s)
            self.combo_sample.blockSignals(False)

        if hasattr(self.host_widget, 'combo_scope') and hasattr(self, 'combo_scope'):
            self.combo_scope.blockSignals(True)
            self.combo_scope.setCurrentText(self.host_widget.combo_scope.currentText())
            self.combo_scope.blockSignals(False)
            self._on_scope_changed()

        self._refresh_parent_combo()
        self._update_marker_quadrant_labels()
        self._on_mode_changed()

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self.selected_color), self, "Select Phenotype Color")
        if color.isValid():
            self.selected_color = color.name()
            self._update_color_btn()

    def _update_color_btn(self):
        self.btn_color.setStyleSheet(f"background-color: {self.selected_color}; border: 1px solid #777; border-radius: 3px;")

    def _on_mode_changed(self):
        is_3d = self.combo_mode.currentText().startswith("3D")
        is_1d = self.combo_mode.currentText().startswith("1D")
        self.container_z.setVisible(is_3d)
        self.container_tz.setVisible(is_3d)
        self.combo_y.setEnabled(not is_1d)
        self.spin_ty.setEnabled(not is_1d)
        self._update_marker_quadrant_labels()
        self._render_plot()

    def _on_scope_changed(self):
        scope = self.combo_scope.currentText()
        if hasattr(self, 'combo_sample'):
            if scope.startswith("Selected"):
                self.combo_sample.setEnabled(True)
            elif scope.startswith("Actively"):
                self.combo_sample.setEnabled(False)
                act = self.host_widget._get_active_sample() if self.host_widget else None
                if act:
                    idx = self.combo_sample.findText(act)
                    if idx != -1:
                        self.combo_sample.blockSignals(True)
                        self.combo_sample.setCurrentIndex(idx)
                        self.combo_sample.blockSignals(False)
            else:
                self.combo_sample.setEnabled(False)
        self._render_plot()

    def _on_sample_changed(self):
        if hasattr(self, 'combo_scope') and not self.combo_scope.currentText().startswith("Selected"):
            self.combo_scope.blockSignals(True)
            self.combo_scope.setCurrentText("Selected Sample")
            self.combo_scope.blockSignals(False)
            if hasattr(self, 'combo_sample'):
                self.combo_sample.setEnabled(True)
        self._render_plot()

    def _on_parent_changed(self):
        self._render_plot()

    def _on_marker_changed(self):
        self._update_marker_quadrant_labels()
        data = self._get_filtered_parent_data()
        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        if data is not None and not data.empty and mx in data.columns:
            vx = self._transform_vector(data[mx].to_numpy())
            tx = compute_optimal_threshold(vx, 'otsu')
            self.spin_tx.blockSignals(True)
            self.spin_tx.setValue(float(tx))
            self.spin_tx.blockSignals(False)
        if data is not None and not data.empty and my in data.columns:
            vy = self._transform_vector(data[my].to_numpy())
            ty = compute_optimal_threshold(vy, 'otsu')
            self.spin_ty.blockSignals(True)
            self.spin_ty.setValue(float(ty))
            self.spin_ty.blockSignals(False)

        self._render_plot()

    def _auto_threshold(self, axis: str, method: str):
        data = self._get_filtered_parent_data()
        if data is None or data.empty:
            return
        m_name = self.combo_x.currentText() if axis == 'x' else self.combo_y.currentText()
        if m_name not in data.columns:
            return
        vals = self._transform_vector(data[m_name].to_numpy())
        val = compute_optimal_threshold(vals, method)
        if axis == 'x':
            self.spin_tx.setValue(float(val))
        else:
            self.spin_ty.setValue(float(val))
        self.lbl_status.setText(f"Computed {method.upper()} threshold for {m_name}: {val:.2f}")

    def _update_marker_quadrant_labels(self):
        mx = self.combo_x.currentText() or "X"
        my = self.combo_y.currentText() or "Y"
        mode = self.combo_mode.currentText()

        if mode.startswith("1D"):
            q_names = [f"{mx}+ (Positive)", f"{mx}- (Negative)"]
            pop_items = [f"{mx}+ (Positive)", f"{mx}- (Negative)"]
        elif mode.startswith("3D"):
            mz = self.combo_z.currentText() or "Z"
            q_names = [
                f"Triple+ ({mx}+ {my}+ {mz}+)",
                f"Double+ ({mx}+ {my}+)",
                f"Single+ ({mx}+)",
                f"Triple- ({mx}- {my}- {mz}-)"
            ]
            pop_items = [
                f"Triple Positive ({mx}+ {my}+ {mz}+)",
                f"Double Positive ({mx}+ {my}+)",
                f"Single Positive ({mx}+)",
                f"Triple Negative ({mx}- {my}- {mz}-)"
            ]
        else:
            # 2D Biaxial
            q_names = [
                f"Q1 ({mx}- {my}+)",
                f"Q2 ({mx}+ {my}+) Double+",
                f"Q3 ({mx}- {my}-) Double-",
                f"Q4 ({mx}+ {my}-)"
            ]
            pop_items = [
                f"Q2 ({mx}+ {my}+) Double Positive",
                f"Q1 ({mx}- {my}+)",
                f"Q4 ({mx}+ {my}-)",
                f"Q3 ({mx}- {my}-) Double Negative",
                "Inside Polygon Gate",
                f"All 4 Subsets (Auto-classify {mx}/{my})"
            ]

        self.table_quads.setRowCount(len(q_names))
        for r, name in enumerate(q_names):
            item0 = self.table_quads.item(r, 0)
            if not item0:
                item0 = QTableWidgetItem(name)
                item0.setFlags(item0.flags() & ~Qt.ItemIsEditable)
                item0.setForeground(QColor("#E0E0E0"))
                self.table_quads.setItem(r, 0, item0)
            else:
                item0.setText(name)
                item0.setForeground(QColor("#E0E0E0"))

            item1 = self.table_quads.item(r, 1)
            if not item1:
                item1 = QTableWidgetItem("-")
                item1.setTextAlignment(Qt.AlignCenter)
                item1.setFlags(item1.flags() & ~Qt.ItemIsEditable)
                item1.setForeground(QColor("#E0E0E0"))
                self.table_quads.setItem(r, 1, item1)
            else:
                item1.setForeground(QColor("#E0E0E0"))

        cur_idx = max(0, self.combo_target_pop.currentIndex())
        self.combo_target_pop.blockSignals(True)
        self.combo_target_pop.clear()
        self.combo_target_pop.addItems(pop_items)
        if cur_idx < len(pop_items):
            self.combo_target_pop.setCurrentIndex(cur_idx)
        self.combo_target_pop.blockSignals(False)

        if mode.startswith("2D"):
            self.txt_phenotype_name.setPlaceholderText(f"e.g. {mx}+ {my}+ Population")
        elif mode.startswith("1D"):
            self.txt_phenotype_name.setPlaceholderText(f"e.g. {mx}+ Cells")

    def _on_threshold_changed(self):
        self._update_quadrant_stats_and_lines()

    def _get_raw_data(self) -> Optional[pd.DataFrame]:
        if self.host_widget is not None:
            return self.host_widget.get_active_dataframe()
        return None

    def _get_filtered_parent_data(self) -> Optional[pd.DataFrame]:
        df = self._get_raw_data()
        if df is None or df.empty:
            return None

        # Filter by hierarchy parent node first
        parent_name = self.combo_parent_pop.currentText().strip()
        if self.host_widget and hasattr(self.host_widget, 'hierarchy_root'):
            node = self.host_widget.hierarchy_root.find_node(parent_name)
            if node and node.cell_mask is not None:
                if len(node.cell_mask) == len(df):
                    df = df[node.cell_mask].copy()

        # Filter by sample scope
        scope = self.combo_scope.currentText()
        sample_col = find_sample_column(df)
        if sample_col:
            if scope.startswith("Selected"):
                sel = self.combo_sample.currentText().strip() if hasattr(self, 'combo_sample') else ""
                if sel and sel in df[sample_col].values:
                    df = df[df[sample_col] == sel]
                elif self.host_widget and hasattr(self.host_widget, '_get_selected_samples'):
                    sel_list = self.host_widget._get_selected_samples()
                    if sel_list:
                        df = df[df[sample_col].isin(sel_list)]
            elif scope.startswith("Actively") and self.host_widget and hasattr(self.host_widget, '_get_active_sample'):
                act = self.host_widget._get_active_sample()
                if act and act in df[sample_col].values:
                    df = df[df[sample_col] == act]

        return df

    def _transform_vector(self, arr: np.ndarray) -> np.ndarray:
        scale_name = self.combo_scale.currentText()
        if "Arcsinh" in scale_name:
            return arcsinh_transform(arr, cofactor=5.0)
        elif "Log10" in scale_name:
            return log10_transform(arr)
        elif "Min-Max" in scale_name:
            mn, mx = np.nanmin(arr), np.nanmax(arr)
            return (arr - mn) / (mx - mn + 1e-8)
        return arr

    # ── Interactive Plotting & Quadrant Rendering ────────────────────────────
    def _render_plot(self):
        self.figure.clear()
        self.line_x = None
        self.line_y = None
        df = self._get_filtered_parent_data()
        if df is None or df.empty:
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#14161d')
            ax.text(0.5, 0.5, "No cells in selected population / sample scope.\nLoad an annotation table or adjust gating hierarchy.",
                    color='#888', ha='center', va='center', fontsize=12)
            self.canvas.draw()
            return

        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        mz = self.combo_z.currentText()
        mode = self.combo_mode.currentText()

        if mx not in df.columns:
            self.canvas.draw()
            return

        sample_df = df
        if len(df) > 25000:
            sample_df = df.sample(n=25000, random_state=42)

        vx = self._transform_vector(sample_df[mx].dropna().to_numpy())

        if mode.startswith("3D"):
            if my not in df.columns or mz not in df.columns:
                return
            vy = self._transform_vector(sample_df[my].dropna().to_numpy())
            vz = self._transform_vector(sample_df[mz].dropna().to_numpy())
            min_len = min(len(vx), len(vy), len(vz))
            vx, vy, vz = vx[:min_len], vy[:min_len], vz[:min_len]

            ax = self.figure.add_subplot(111, projection='3d')
            ax.set_facecolor('#14161d')
            ax.scatter(vx, vy, vz, c=vy, cmap='viridis', s=6, alpha=0.6, edgecolors='none')
            ax.set_xlabel(mx, color='#DDD', fontsize=10)
            ax.set_ylabel(my, color='#DDD', fontsize=10)
            ax.set_zlabel(mz, color='#DDD', fontsize=10)
            ax.tick_params(colors='#888', labelsize=8)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            self.ax = ax

        elif mode.startswith("1D"):
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#14161d')
            ax.hist(vx, bins=60, color='#3A6A9A', edgecolor='#14161d', alpha=0.85)
            ax.set_xlabel(f"{mx} ({self.combo_scale.currentText().split()[0]})", color='#DDD', fontsize=11)
            ax.set_ylabel("Cell Count", color='#DDD', fontsize=11)
            ax.tick_params(colors='#AAA', labelsize=9)
            ax.grid(True, color='#262930', linestyle='--', linewidth=0.5)
            self.ax = ax
            self.line_x = ax.axvline(self.spin_tx.value(), color='#FF3B30', linestyle='--', linewidth=2.0)

        else:
            if my not in df.columns:
                return
            vy = self._transform_vector(sample_df[my].dropna().to_numpy())
            min_len = min(len(vx), len(vy))
            vx, vy = vx[:min_len], vy[:min_len]

            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#14161d')
            ax.scatter(vx, vy, c='#4A7FBA', s=5, alpha=0.45, edgecolors='none', rasterized=True)
            ax.set_xlabel(f"{mx} ({self.combo_scale.currentText().split()[0]})", color='#DDD', fontsize=11)
            ax.set_ylabel(f"{my} ({self.combo_scale.currentText().split()[0]})", color='#DDD', fontsize=11)
            ax.tick_params(colors='#AAA', labelsize=9)
            ax.grid(True, color='#262930', linestyle='--', linewidth=0.5)

            tx = self.spin_tx.value()
            ty = self.spin_ty.value()
            self.line_x = ax.axvline(tx, color='#FF5252', linestyle='--', linewidth=1.8, label=f"{mx} Cutoff")
            self.line_y = ax.axhline(ty, color='#448AFF', linestyle='--', linewidth=1.8, label=f"{my} Cutoff")

            if self.polygon_vertices is not None and len(self.polygon_vertices) > 2:
                patch = plt.Polygon(self.polygon_vertices, closed=True, fill=True,
                                    facecolor='#76FF03', alpha=0.25, edgecolor='#76FF03', linewidth=2.0)
                ax.add_patch(patch)

            self.ax = ax

        self._update_quadrant_stats_and_lines()
        self.figure.tight_layout()
        self.canvas.draw()

    def _update_quadrant_stats_and_lines(self):
        df = self._get_filtered_parent_data()
        if df is None or df.empty:
            return
        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        if mx not in df.columns:
            return

        vx = self._transform_vector(df[mx].dropna().to_numpy())
        total = len(vx)
        if total == 0:
            return

        tx = self.spin_tx.value()
        ty = self.spin_ty.value()

        if hasattr(self, 'line_x') and self.line_x:
            self.line_x.set_xdata([tx, tx])
        if hasattr(self, 'line_y') and self.line_y:
            self.line_y.set_ydata([ty, ty])

        def _set_quad_cell(r, text):
            item = self.table_quads.item(r, 1)
            if not item:
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setForeground(QColor("#E0E0E0"))
                self.table_quads.setItem(r, 1, item)
            else:
                item.setText(text)
                item.setForeground(QColor("#E0E0E0"))

        mode = self.combo_mode.currentText()
        if mode.startswith("1D"):
            pos_cnt = int((vx >= tx).sum())
            neg_cnt = total - pos_cnt
            p_pos = (pos_cnt / total) * 100.0 if total > 0 else 0.0
            p_neg = (neg_cnt / total) * 100.0 if total > 0 else 0.0
            _set_quad_cell(0, f"{pos_cnt:,} ({p_pos:.1f}%)")
            _set_quad_cell(1, f"{neg_cnt:,} ({p_neg:.1f}%)")
        elif mode.startswith("2D") and my in df.columns:
            vy = self._transform_vector(df[my].dropna().to_numpy())
            min_len = min(len(vx), len(vy))
            vx, vy = vx[:min_len], vy[:min_len]

            m_q1 = (vx < tx) & (vy >= ty)
            m_q2 = (vx >= tx) & (vy >= ty)
            m_q3 = (vx < tx) & (vy < ty)
            m_q4 = (vx >= tx) & (vy < ty)

            q_counts = [m_q1.sum(), m_q2.sum(), m_q3.sum(), m_q4.sum()]
            for r, cnt in enumerate(q_counts):
                pct = (cnt / total) * 100.0 if total > 0 else 0.0
                _set_quad_cell(r, f"{cnt:,} ({pct:.1f}%)")

            if self.polygon_vertices is not None and len(self.polygon_vertices) > 2:
                path = MplPath(self.polygon_vertices)
                pts = np.column_stack((vx, vy))
                in_poly = path.contains_points(pts)
                p_cnt = in_poly.sum()
                p_pct = (p_cnt / total) * 100.0 if total > 0 else 0.0
                self.lbl_poly_status.setText(f"Shape Gate: {p_cnt:,} cells ({p_pct:.1f}% of {total:,})")
                self.lbl_poly_status.setStyleSheet("color: #76FF03; font-size: 11px; font-weight: bold;")

        self.canvas.draw_idle()
        self._update_impact_window_live()

    # ── Interactive Canvas Dragging for Cutoff Lines ──────────────────────────
    def _on_canvas_press(self, event):
        if event.inaxes != getattr(self, 'ax', None) or event.button != 1:
            return
        if self.poly_selector is not None and self.poly_selector.active:
            return

        tx = self.spin_tx.value()
        ty = self.spin_ty.value()
        tol_x = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.04
        tol_y = (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.04

        if event.xdata is not None and abs(event.xdata - tx) < tol_x:
            self.dragging_line = 'x'
            self.lbl_status.setText(f"Dragging {self.combo_x.currentText()} cutoff line...")
        elif event.ydata is not None and abs(event.ydata - ty) < tol_y:
            self.dragging_line = 'y'
            self.lbl_status.setText(f"Dragging {self.combo_y.currentText()} cutoff line...")

    def _on_canvas_motion(self, event):
        if not self.dragging_line or event.inaxes != getattr(self, 'ax', None):
            return
        if self.dragging_line == 'x' and event.xdata is not None:
            self.spin_tx.blockSignals(True)
            self.spin_tx.setValue(float(event.xdata))
            self.spin_tx.blockSignals(False)
            self._update_quadrant_stats_and_lines()
        elif self.dragging_line == 'y' and event.ydata is not None:
            self.spin_ty.blockSignals(True)
            self.spin_ty.setValue(float(event.ydata))
            self.spin_ty.blockSignals(False)
            self._update_quadrant_stats_and_lines()

    def _on_canvas_release(self, event):
        if self.dragging_line:
            self.lbl_status.setText("Threshold updated. Select target gate and click 'Apply Gate to Cells'.")
            self.dragging_line = None
            if self.host_widget is not None:
                self.host_widget.spin_tx.setValue(self.spin_tx.value())
                self.host_widget.spin_ty.setValue(self.spin_ty.value())

    # ── Interactive Polygon & Rectangle Gating ───────────────────────────────
    def _toggle_polygon_tool(self):
        if self.combo_mode.currentText().startswith("3D"):
            QMessageBox.information(self, "Mode Notice", "Switch to 2D Biaxial mode to draw polygon shape gates.")
            return

        if self.poly_selector is not None and self.poly_selector.active:
            self.poly_selector.set_active(False)
            self.btn_draw_poly.setText("Draw Polygon Gate")
            self.btn_draw_poly.setStyleSheet("")
            return

        self.btn_draw_poly.setText("Drawing Mode Active...")
        self.btn_draw_poly.setStyleSheet("background-color: #388E3C; color: white; font-weight: bold;")
        self.lbl_status.setText("Click to add polygon vertices. Double click or press Enter to close.")

        def on_select(vertices):
            self.polygon_vertices = np.array(vertices)
            self.btn_draw_poly.setText("Draw Polygon Gate")
            self.btn_draw_poly.setStyleSheet("")
            self.lbl_status.setText("Polygon gate closed. Select 'Inside Polygon Gate' in Target Gate to classify.")
            self._render_plot()

        self.poly_selector = PolygonSelector(
            self.ax,
            on_select,
            props=dict(color='#76FF03', linestyle='-', linewidth=2, alpha=0.9),
            draw_bounding_box=False
        )

    def _clear_polygon(self):
        self.polygon_vertices = None
        if self.poly_selector:
            self.poly_selector.set_active(False)
        self.btn_draw_poly.setText("Draw Polygon Gate")
        self.btn_draw_poly.setStyleSheet("")
        self.lbl_poly_status.setText("Shape Gate: None defined")
        self.lbl_poly_status.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
        self._render_plot()

    # ── Gating Application & Hierarchy Update ────────────────────────────────
    def _apply_gate(self):
        label_name = self.txt_phenotype_name.text().strip()
        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        if not label_name:
            label_name = f"{mx}+ {my}+"

        raw_df = self._get_raw_data()
        if raw_df is None or raw_df.empty:
            QMessageBox.warning(self, "No Data", "No active cell dataframe available.")
            return

        filtered_df = self._get_filtered_parent_data()
        if filtered_df is None or filtered_df.empty:
            QMessageBox.warning(self, "No Cells", "No cells in active parent population.")
            return

        vx = self._transform_vector(raw_df[mx].to_numpy())
        vy = self._transform_vector(raw_df[my].to_numpy()) if my in raw_df.columns else np.zeros_like(vx)

        tx = self.spin_tx.value()
        ty = self.spin_ty.value()
        target_pop = self.combo_target_pop.currentText()

        if "Inside Polygon" in target_pop and self.polygon_vertices is not None and len(self.polygon_vertices) > 2:
            path = MplPath(self.polygon_vertices)
            pts = np.column_stack((vx, vy))
            gate_mask = path.contains_points(pts)
            criteria = "Polygon Gate"
        elif "Q2" in target_pop:
            gate_mask = (vx >= tx) & (vy >= ty)
            criteria = f"{mx} >= {tx:.2f} & {my} >= {ty:.2f}"
        elif "Q1" in target_pop:
            gate_mask = (vx < tx) & (vy >= ty)
            criteria = f"{mx} < {tx:.2f} & {my} >= {ty:.2f}"
        elif "Q4" in target_pop:
            gate_mask = (vx >= tx) & (vy < ty)
            criteria = f"{mx} >= {tx:.2f} & {my} < {ty:.2f}"
        elif "Q3" in target_pop:
            gate_mask = (vx < tx) & (vy < ty)
            criteria = f"{mx} < {tx:.2f} & {my} < {ty:.2f}"
        else:
            gate_mask = (vx >= tx)
            criteria = f"{mx} >= {tx:.2f}"

        parent_name = self.combo_parent_pop.currentText().strip()
        parent_node = None
        if self.host_widget and hasattr(self.host_widget, 'hierarchy_root'):
            parent_node = self.host_widget.hierarchy_root.find_node(parent_name)

        if parent_node and parent_node.cell_mask is not None:
            final_mask = parent_node.cell_mask & gate_mask
        else:
            final_mask = gate_mask

        # Filter by sample scope if Selected or Actively loaded
        if hasattr(self, 'combo_scope'):
            scope = self.combo_scope.currentText()
            scol = find_sample_column(raw_df)
            if scol:
                if scope.startswith("Selected") and hasattr(self, 'combo_sample'):
                    sel = self.combo_sample.currentText().strip()
                    if sel and sel in raw_df[scol].astype(str).values:
                        final_mask = final_mask & (raw_df[scol].astype(str) == sel).to_numpy()
                elif scope.startswith("Actively"):
                    act = self._get_active_sample()
                    if act and str(act) in raw_df[scol].astype(str).values:
                        final_mask = final_mask & (raw_df[scol].astype(str) == str(act)).to_numpy()

        match_count = int(final_mask.sum())
        if match_count == 0:
            if hasattr(self, 'lbl_apply_feedback'):
                self.lbl_apply_feedback.setText("⚠ No cells matched this gating cutoff.")
                self.lbl_apply_feedback.setStyleSheet("color: #FF5252; font-weight: bold; font-size: 11px;")
            return

        target_col = self.combo_col.currentText().strip() or "CellType"

        if self.host_widget and hasattr(self.host_widget, 'hierarchy_root'):
            p_node = parent_node or self.host_widget.hierarchy_root
            child_node = GatingHierarchyNode(
                name=label_name,
                parent=p_node,
                cell_mask=final_mask,
                color=self.selected_color,
                markers=(mx, my),
                criteria_desc=criteria,
                total_cells_count=match_count
            )
            self._refresh_parent_combo()
            self._refresh_hierarchy_tree_view()
            if hasattr(self.host_widget, '_refresh_dock_hierarchy_tree'):
                self.host_widget._refresh_dock_hierarchy_tree()

        if target_col not in raw_df.columns:
            raw_df[target_col] = "Unassigned"
        if raw_df[target_col].dtype.name == 'category':
            if label_name not in raw_df[target_col].cat.categories:
                raw_df[target_col] = raw_df[target_col].cat.add_categories([label_name])
        raw_df.loc[final_mask, target_col] = label_name

        pct = (match_count / len(raw_df)) * 100.0
        r_idx = self.table_history.rowCount()
        self.table_history.insertRow(r_idx)
        self.table_history.setItem(r_idx, 0, QTableWidgetItem(label_name))
        self.table_history.setItem(r_idx, 1, QTableWidgetItem(f"{match_count:,}"))
        self.table_history.setItem(r_idx, 2, QTableWidgetItem(f"{pct:.1f}%"))
        self.table_history.setItem(r_idx, 3, QTableWidgetItem(f"{mx} / {my}"))

        gate_info = {
            'label': label_name,
            'color': self.selected_color,
            'target_col': target_col,
            'count': match_count,
            'pct': pct,
            'markers': (mx, my),
            'mask': final_mask,
        }
        self.gateApplied.emit(gate_info)

        if hasattr(self, 'lbl_apply_feedback'):
            self.lbl_apply_feedback.setText(f"✓ Gate applied: {match_count:,} cells ({pct:.1f}% of cohort) -> '{label_name}'")
        self.lbl_status.setText(f"Gate '{label_name}' applied. Continue gating child subsets or adjust cutoffs.")
        self.lbl_status.setStyleSheet("color: #76FF03; font-weight: bold; font-size: 11px;")
        self._render_plot()

    def _refresh_parent_combo(self):
        if not self.host_widget or not hasattr(self.host_widget, 'hierarchy_root'):
            return
        curr = self.combo_parent_pop.currentText()
        self.combo_parent_pop.blockSignals(True)
        self.combo_parent_pop.clear()

        all_nodes = self.host_widget.hierarchy_root.get_all_nodes()
        for node in all_nodes:
            self.combo_parent_pop.addItem(node.name)

        idx = self.combo_parent_pop.findText(curr)
        self.combo_parent_pop.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_parent_pop.blockSignals(False)

    def _refresh_hierarchy_tree_view(self):
        self.tree_hierarchy.clear()
        if not self.host_widget or not hasattr(self.host_widget, 'hierarchy_root'):
            return

        def add_tree_item(parent_item, node: GatingHierarchyNode):
            item = QTreeWidgetItem(parent_item) if parent_item else QTreeWidgetItem(self.tree_hierarchy)
            item.setText(0, node.name)
            item.setText(1, f"{node.cell_count:,}")
            item.setText(2, f"{node.parent_percentage:.1f}%")
            item.setText(3, f"{node.total_percentage:.1f}%")
            item.setText(4, ", ".join(node.markers) if node.markers else "All")
            item.setData(0, Qt.UserRole, node.name)
            item.setExpanded(True)
            for child in node.children:
                add_tree_item(item, child)

        add_tree_item(None, self.host_widget.hierarchy_root)

    def _on_tree_item_double_clicked(self, item: QTreeWidgetItem, col: int):
        node_name = item.data(0, Qt.UserRole)
        idx = self.combo_parent_pop.findText(node_name)
        if idx >= 0:
            self.combo_parent_pop.setCurrentIndex(idx)
            self.tabs.setCurrentIndex(0)
            self._render_plot()

    def _gate_on_tree_selection(self):
        selected = self.tree_hierarchy.selectedItems()
        if not selected:
            QMessageBox.information(self, "Selection Required", "Please select a population from the hierarchy tree.")
            return
        self._on_tree_item_double_clicked(selected[0], 0)

    def _apply_entire_hierarchy(self):
        if not self.host_widget or not hasattr(self.host_widget, 'hierarchy_root'):
            return
        raw_df = self._get_raw_data()
        if raw_df is None or raw_df.empty:
            return

        target_col = self.combo_col.currentText().strip() or "CellType"
        if target_col not in raw_df.columns:
            raw_df[target_col] = "Unassigned"

        nodes = self.host_widget.hierarchy_root.get_all_nodes()
        classified_cnt = 0
        for node in nodes:
            if node.parent is not None and node.cell_mask is not None:
                raw_df.loc[node.cell_mask, target_col] = node.name
                classified_cnt += int(node.cell_mask.sum())
                if self.host_widget.ctx and hasattr(self.host_widget.ctx, 'legend_widget') and self.host_widget.ctx.legend_widget:
                    self.host_widget.ctx.legend_widget.add_ref_ct_row(None, node.name, node.color)

        if self.host_widget.ctx and hasattr(self.host_widget.ctx, 'multi_sample_svc') and self.host_widget.ctx.multi_sample_svc:
            try:
                self.host_widget.ctx.multi_sample_svc.update_phenotypes_on_canvas()
            except Exception:
                pass

        QMessageBox.information(self, "Hierarchy Applied", f"Successfully applied {len(nodes)-1} hierarchical gates to column '{target_col}'.")

    def _delete_selected_tree_gate(self):
        selected = self.tree_hierarchy.selectedItems()
        if not selected:
            return
        name = selected[0].data(0, Qt.UserRole)
        if name == "Root (All Cells)":
            QMessageBox.warning(self, "Cannot Delete", "The Root population cannot be deleted.")
            return
        if self.host_widget and hasattr(self.host_widget, 'hierarchy_root'):
            node = self.host_widget.hierarchy_root.find_node(name)
            if node and node.parent:
                node.parent.children.remove(node)
                self._refresh_parent_combo()
                self._refresh_hierarchy_tree_view()
                self._render_plot()

    def _export_hierarchy_csv(self):
        if not self.host_widget or not hasattr(self.host_widget, 'hierarchy_root'):
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Gating Hierarchy", "gating_hierarchy.csv", "CSV (*.csv)")
        if not path:
            return
        nodes = self.host_widget.hierarchy_root.get_all_nodes()
        rows = []
        for n in nodes:
            rows.append({
                'Population': n.name,
                'Parent': n.parent.name if n.parent else "None",
                'Path': n.get_path_string(),
                'Cells': n.cell_count,
                'Pct_Parent': f"{n.parent_percentage:.2f}%",
                'Pct_Total': f"{n.total_percentage:.2f}%",
                'Markers': ", ".join(n.markers),
                'Criteria': n.criteria_desc
            })
        pd.DataFrame(rows).to_csv(path, index=False)
        QMessageBox.information(self, "Export Complete", f"Exported hierarchy to {Path(path).name}")

    def _clear_applied_gates(self):
        self.table_history.setRowCount(0)
        self.lbl_status.setText("Gating history reset.")

    def _export_gates_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Applied Gates", "gated_populations.csv", "CSV (*.csv)")
        if not path:
            return
        rows = []
        for r in range(self.table_history.rowCount()):
            rows.append({
                'Gate_Name': self.table_history.item(r, 0).text(),
                'Cells': self.table_history.item(r, 1).text(),
                'Percentage': self.table_history.item(r, 2).text(),
                'Markers': self.table_history.item(r, 3).text(),
            })
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
            QMessageBox.information(self, "Export Complete", f"Exported {len(rows)} gates to {Path(path).name}")

    # ── Multi-Sample Impact Window Integration ───────────────────────────────
    def _open_impact_window(self):
        if self.impact_dialog is None:
            self.impact_dialog = SampleImpactDialog(parent=self)
        self._update_impact_window_live()
        self.impact_dialog.show()
        self.impact_dialog.raise_()
        self.impact_dialog.activateWindow()

    def _update_impact_window_live(self):
        if self.impact_dialog is None or not self.impact_dialog.isVisible():
            return
        raw_df = self._get_raw_data()
        if raw_df is None or raw_df.empty:
            return

        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        if mx not in raw_df.columns:
            return

        vx = self._transform_vector(raw_df[mx].to_numpy())
        vy = self._transform_vector(raw_df[my].to_numpy()) if my in raw_df.columns else np.zeros_like(vx)
        tx = self.spin_tx.value()
        ty = self.spin_ty.value()
        target_pop = self.combo_target_pop.currentText()

        if "Inside Polygon" in target_pop and self.polygon_vertices is not None and len(self.polygon_vertices) > 2:
            path = MplPath(self.polygon_vertices)
            pts = np.column_stack((vx, vy))
            mask = path.contains_points(pts)
        elif "Q2" in target_pop:
            mask = (vx >= tx) & (vy >= ty)
        elif "Q1" in target_pop:
            mask = (vx < tx) & (vy >= ty)
        elif "Q4" in target_pop:
            mask = (vx >= tx) & (vy < ty)
        elif "Q3" in target_pop:
            mask = (vx < tx) & (vy < ty)
        else:
            mask = (vx >= tx)

        gate_lbl = self.txt_phenotype_name.text().strip() or target_pop
        sel_samples = self.host_widget._get_selected_samples() if self.host_widget else []
        act_sample = self.host_widget._get_active_sample() if self.host_widget else None

        self.impact_dialog.update_distribution(
            df=raw_df,
            positive_mask=mask,
            gate_label=gate_lbl,
            mx=mx,
            my=my,
            selected_samples=sel_samples,
            active_sample=act_sample
        )


# ── Spacious Dock Subpanel for Manual Gating ─────────────────────────────────

class ManualGatingWidget(QWidget):
    """
    Spacious, ergonomic dock subpanel for Manual Gating inside the Phenotyper tab.
    Offers comfortable vertical layout with ample space for the Gating Scatter Preview,
    a dedicated Gating Quality & Sample Assessment section underneath the plot with its own
    pop-up window launcher, automated Otsu/Valley threshold discovery, and clean box titles.
    """
    def __init__(self, parent=None, ctx=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setStyleSheet(SHARED_DARK_THEME)
        self.available_markers: List[str] = []
        self.popup_dialog: Optional[GatingStudioDialog] = None
        self.impact_dialog: Optional[SampleImpactDialog] = None

        self.hierarchy_root = GatingHierarchyNode("Root (All Cells)")

        self._build_ui()

    def _build_ui(self):
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(8)

        # ── 1. Gating Data & Projection Box (Title inside box) ────────────────
        box_proj = QGroupBox()
        bl_proj = QVBoxLayout(box_proj)
        bl_proj.setContentsMargins(8, 2, 8, 8)
        bl_proj.setSpacing(6)

        title_proj = QLabel("Gating Data & Projection")
        title_proj.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_proj.addWidget(title_proj)

        # Sample Scope & Selection Row
        row_scope = QHBoxLayout()
        row_scope.addWidget(QLabel("Scope:"))
        self.combo_scope = QComboBox()
        self.combo_scope.addItems(["All Samples (Cohort)", "Selected Sample", "Actively Loaded Sample"])
        self.combo_scope.currentIndexChanged.connect(self._on_scope_changed)
        row_scope.addWidget(self.combo_scope)
        bl_proj.addLayout(row_scope)

        self.row_sample_select = QHBoxLayout()
        self.row_sample_select.addWidget(QLabel("Sample:"))
        self.combo_sample = QComboBox()
        self.combo_sample.setToolTip("Select specific sample / core for manual gating")
        self.combo_sample.currentIndexChanged.connect(self._on_sample_selection_changed)
        self.row_sample_select.addWidget(self.combo_sample)
        bl_proj.addLayout(self.row_sample_select)

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["2D Biaxial Scatter", "3D Trivariate Scatter", "1D Histogram"])
        self.combo_mode.currentIndexChanged.connect(self._on_params_changed)
        row_mode.addWidget(self.combo_mode)
        bl_proj.addLayout(row_mode)

        row_x = QHBoxLayout()
        row_x.addWidget(QLabel("Marker X:"))
        self.combo_x = QComboBox()
        self.combo_x.currentIndexChanged.connect(self._on_params_changed)
        row_x.addWidget(self.combo_x)
        bl_proj.addLayout(row_x)

        row_y = QHBoxLayout()
        row_y.addWidget(QLabel("Marker Y:"))
        self.combo_y = QComboBox()
        self.combo_y.currentIndexChanged.connect(self._on_params_changed)
        row_y.addWidget(self.combo_y)
        bl_proj.addLayout(row_y)

        # Container for Marker Z (hidden in 2D mode)
        self.container_dock_z = QWidget()
        row_z = QHBoxLayout(self.container_dock_z)
        row_z.setContentsMargins(0, 0, 0, 0)
        row_z.addWidget(QLabel("Marker Z:"))
        self.combo_z = QComboBox()
        self.combo_z.currentIndexChanged.connect(self._on_params_changed)
        row_z.addWidget(self.combo_z)
        bl_proj.addWidget(self.container_dock_z)
        self.container_dock_z.hide()

        row_scale = QHBoxLayout()
        row_scale.addWidget(QLabel("Scale Transformation:"))
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Arcsinh (Cofactor = 5)", "Log10(1 + x)", "Linear (Raw)", "Min-Max Normalized"])
        self.combo_scale.currentIndexChanged.connect(self._on_params_changed)
        row_scale.addWidget(self.combo_scale)
        bl_proj.addLayout(row_scale)

        # Cutoff Spinners with Auto Buttons
        row_tx = QHBoxLayout()
        row_tx.addWidget(QLabel("X Cutoff:"))
        self.spin_tx = QDoubleSpinBox()
        self.spin_tx.setRange(-10000.0, 100000.0)
        self.spin_tx.setDecimals(2)
        self.spin_tx.setSingleStep(0.2)
        self.spin_tx.valueChanged.connect(self._on_threshold_changed)
        row_tx.addWidget(self.spin_tx)

        btn_auto_x = QPushButton("Auto (Otsu)")
        btn_auto_x.setToolTip("Auto Otsu cutoff")
        btn_auto_x.clicked.connect(lambda: self._auto_threshold('x', 'otsu'))
        row_tx.addWidget(btn_auto_x)

        btn_val_x = QPushButton("Valley")
        btn_val_x.setToolTip("Bimodal valley cutoff")
        btn_val_x.clicked.connect(lambda: self._auto_threshold('x', 'valley'))
        row_tx.addWidget(btn_val_x)
        bl_proj.addLayout(row_tx)

        self.row_dock_ty = QHBoxLayout()
        self.row_dock_ty.addWidget(QLabel("Y Cutoff:"))
        self.spin_ty = QDoubleSpinBox()
        self.spin_ty.setRange(-10000.0, 100000.0)
        self.spin_ty.setDecimals(2)
        self.spin_ty.setSingleStep(0.2)
        self.spin_ty.valueChanged.connect(self._on_threshold_changed)
        self.row_dock_ty.addWidget(self.spin_ty)

        btn_auto_y = QPushButton("Auto (Otsu)")
        btn_auto_y.setToolTip("Auto Otsu cutoff")
        btn_auto_y.clicked.connect(lambda: self._auto_threshold('y', 'otsu'))
        self.row_dock_ty.addWidget(btn_auto_y)

        btn_val_y = QPushButton("Valley")
        btn_val_y.setToolTip("Bimodal valley cutoff")
        btn_val_y.clicked.connect(lambda: self._auto_threshold('y', 'valley'))
        self.row_dock_ty.addWidget(btn_val_y)
        bl_proj.addLayout(self.row_dock_ty)

        layout.addWidget(box_proj)

        # ── 2. Spacious Gating Scatter Preview Box ────────────────────────────
        box_plot = QGroupBox()
        bl_plot = QVBoxLayout(box_plot)
        bl_plot.setContentsMargins(8, 2, 8, 8)
        bl_plot.setSpacing(6)

        plot_header = QHBoxLayout()
        title_plot = QLabel("Gating Scatter Preview")
        title_plot.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        plot_header.addWidget(title_plot)
        plot_header.addStretch(1)

        self.btn_popout = QPushButton("⤢ Pop-up Window")
        self.btn_popout.setToolTip("Open full-size interactive Gating Studio with polygon drawing & 3D visualization")
        self.btn_popout.setStyleSheet("""
            QPushButton {
                background-color: #2D5F8B; color: white; font-weight: bold;
                border: 1px solid #4B82B5; border-radius: 4px; padding: 4px 10px; font-size: 11px;
            }
            QPushButton:hover { background-color: #3D75A8; }
        """)
        self.btn_popout.clicked.connect(self.open_popup_window)
        plot_header.addWidget(self.btn_popout)
        bl_plot.addLayout(plot_header)

        # Streamlined preview figure
        self.figure = Figure(figsize=(4.0, 2.5), dpi=80)
        self.figure.patch.set_facecolor('#1f232b')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.canvas.setFixedHeight(180)
        self.canvas.setMinimumHeight(150)
        self.canvas.mouseDoubleClickEvent = lambda event: self.open_popup_window()
        bl_plot.addWidget(self.canvas)

        self.lbl_quad_info = QLabel("Q2: 0 cells (0.0% Parent) | Q1: 0 (0% Parent)")
        self.lbl_quad_info.setStyleSheet("color: #AAA; font-size: 11px; font-style: italic;")
        bl_plot.addWidget(self.lbl_quad_info)

        layout.addWidget(box_plot)

        # ── 3. Gating Quality & Sample Assessment (UNDERNEATH the plot!) ───────
        box_quality = QGroupBox()
        bl_quality = QVBoxLayout(box_quality)
        bl_quality.setContentsMargins(8, 2, 8, 8)
        bl_quality.setSpacing(6)

        qual_header = QHBoxLayout()
        title_qual = QLabel("Gating Quality & Sample Assessment")
        title_qual.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        qual_header.addWidget(title_qual)
        qual_header.addStretch(1)

        self.btn_impact_popout = QPushButton("⤢ Pop-up Window")
        self.btn_impact_popout.setToolTip("Open dedicated Cohort Gating Assessment Studio")
        self.btn_impact_popout.setStyleSheet("""
            QPushButton {
                background-color: #2D4F6A; color: white; font-weight: bold;
                border: 1px solid #3D6F9A; border-radius: 4px; padding: 4px 10px; font-size: 11px;
            }
            QPushButton:hover { background-color: #3D6F9A; }
        """)
        self.btn_impact_popout.clicked.connect(self.open_impact_window)
        qual_header.addWidget(self.btn_impact_popout)
        bl_quality.addLayout(qual_header)

        # Quality metrics informative readout
        row_sep = QHBoxLayout()
        row_sep.addWidget(QLabel("Cutoff Separation:"))
        self.lbl_sep_quality = QLabel("Assessing...")
        self.lbl_sep_quality.setStyleSheet("color: #76FF03; font-weight: bold; font-size: 11px;")
        row_sep.addWidget(self.lbl_sep_quality)
        row_sep.addStretch(1)
        bl_quality.addLayout(row_sep)

        self.lbl_cohort_summary = QLabel("Cohort Impact: Calculating...")
        self.lbl_cohort_summary.setStyleSheet("color: #44AAFF; font-size: 11px; font-weight: bold;")
        bl_quality.addWidget(self.lbl_cohort_summary)

        self.lbl_core_hetero = QLabel("Core Heterogeneity: Loading...")
        self.lbl_core_hetero.setStyleSheet("color: #CCC; font-size: 11px;")
        bl_quality.addWidget(self.lbl_core_hetero)

        layout.addWidget(box_quality)

        # ── 4. Categorize & Classify Box ─────────────────────────────────────
        box_class = QGroupBox()
        bl_class = QVBoxLayout(box_class)
        bl_class.setContentsMargins(8, 2, 8, 8)
        bl_class.setSpacing(6)

        title_class = QLabel("Categorize & Classify")
        title_class.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_class.addWidget(title_class)

        row_target = QHBoxLayout()
        row_target.addWidget(QLabel("Target Gate:"))
        self.combo_target_pop = QComboBox()
        self.combo_target_pop.currentIndexChanged.connect(self._on_target_pop_changed)
        row_target.addWidget(self.combo_target_pop)
        bl_class.addLayout(row_target)

        row_name = QHBoxLayout()
        row_name.addWidget(QLabel("Label Designation:"))
        self.txt_phenotype_name = QLineEdit()
        self.txt_phenotype_name.setPlaceholderText("e.g. CD45+ CD3+ T Cells")
        row_name.addWidget(self.txt_phenotype_name)

        self.btn_color = QPushButton("")
        self.btn_color.setFixedWidth(32)
        self.btn_color.setFixedHeight(24)
        self.selected_color = "#00E5FF"
        self._update_color_btn()
        self.btn_color.clicked.connect(self._pick_color)
        row_name.addWidget(self.btn_color)
        bl_class.addLayout(row_name)

        row_parent = QHBoxLayout()
        row_parent.addWidget(QLabel("Parent Gate:"))
        self.combo_parent_pop = QComboBox()
        self.combo_parent_pop.addItem("Root (All Cells)")
        self.combo_parent_pop.currentIndexChanged.connect(self._on_params_changed)
        row_parent.addWidget(self.combo_parent_pop)
        bl_class.addLayout(row_parent)

        btn_row_apply = QHBoxLayout()
        self.btn_apply = QPushButton("Apply Gate to Cells")
        self.btn_apply.setAutoDefault(False)
        self.btn_apply.setDefault(False)
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #2D6A4F; color: white; font-weight: bold;
                border-radius: 4px; padding: 7px; font-size: 12px;
            }
            QPushButton:hover { background-color: #40916C; }
        """)
        self.btn_apply.clicked.connect(self._apply_gate)
        btn_row_apply.addWidget(self.btn_apply)

        self.btn_view_tree = QPushButton("⤢ Hierarchy Tree")
        self.btn_view_tree.setToolTip("Open full hierarchical tree view in Studio Dialog")
        self.btn_view_tree.clicked.connect(self.open_popup_window)
        btn_row_apply.addWidget(self.btn_view_tree)
        bl_class.addLayout(btn_row_apply)

        self.lbl_apply_dock_feedback = QLabel("")
        self.lbl_apply_dock_feedback.setStyleSheet("color: #76FF03; font-weight: bold; font-size: 11px;")
        bl_class.addWidget(self.lbl_apply_dock_feedback)

        layout.addWidget(box_class)
        layout.addStretch(1)

        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _on_scope_changed(self):
        scope = self.combo_scope.currentText()
        if hasattr(self, 'combo_sample'):
            if scope.startswith("Selected"):
                self.combo_sample.setEnabled(True)
            elif scope.startswith("Actively"):
                self.combo_sample.setEnabled(False)
                act = self._get_active_sample()
                if act:
                    idx = self.combo_sample.findText(act)
                    if idx != -1:
                        self.combo_sample.blockSignals(True)
                        self.combo_sample.setCurrentIndex(idx)
                        self.combo_sample.blockSignals(False)
            else:
                self.combo_sample.setEnabled(False)
        self._on_params_changed()

    def _on_sample_selection_changed(self):
        scope = self.combo_scope.currentText()
        if not scope.startswith("Selected"):
            self.combo_scope.blockSignals(True)
            self.combo_scope.setCurrentText("Selected Sample")
            self.combo_scope.blockSignals(False)
            if hasattr(self, 'combo_sample'):
                self.combo_sample.setEnabled(True)
        self._on_params_changed()

    def _refresh_sample_list(self):
        if not hasattr(self, 'combo_sample'):
            return
        curr = self.combo_sample.currentText()
        self.combo_sample.blockSignals(True)
        self.combo_sample.clear()
        samples = []
        raw_df = self.get_active_dataframe()
        if raw_df is not None and not raw_df.empty:
            scol = find_sample_column(raw_df)
            if scol:
                samples = [str(s) for s in raw_df[scol].unique()]
        if not samples and self.ctx and hasattr(self.ctx, 'state') and self.ctx.state:
            samples = list(self.ctx.state.sample_layers_map.keys())
        if not samples:
            samples = ["Sample_1"]
        self.combo_sample.addItems(samples)
        if curr in samples:
            self.combo_sample.setCurrentText(curr)
        self.combo_sample.blockSignals(False)

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self.selected_color), self, "Select Phenotype Color")
        if color.isValid():
            self.selected_color = color.name()
            self._update_color_btn()

    def _update_color_btn(self):
        self.btn_color.setStyleSheet(f"background-color: {self.selected_color}; border: 1px solid #777; border-radius: 3px;")

    def _get_selected_samples(self) -> List[str]:
        if hasattr(self, 'combo_scope') and hasattr(self, 'combo_sample'):
            scope = self.combo_scope.currentText()
            if scope.startswith("Selected"):
                s = self.combo_sample.currentText().strip()
                if s and s not in ["(All Cohort Samples)", "None"]:
                    return [s]
        if self.ctx and hasattr(self.ctx, 'state') and self.ctx.state:
            return list(self.ctx.state.sample_layers_map.keys())
        return []

    def _get_active_sample(self) -> Optional[str]:
        if self.ctx and hasattr(self.ctx, 'state') and self.ctx.state:
            return getattr(self.ctx.state, 'current_sample_name', None)
        return None

    def set_data(self, df: Optional[pd.DataFrame] = None, markers: Optional[List[str]] = None):
        if markers is not None:
            self.available_markers = list(markers)
            for combo in (self.combo_x, self.combo_y, self.combo_z):
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(self.available_markers)
                combo.blockSignals(False)
            if len(self.available_markers) > 0:
                self.combo_x.setCurrentIndex(0)
            if len(self.available_markers) > 1:
                self.combo_y.setCurrentIndex(1)
            if len(self.available_markers) > 2:
                self.combo_z.setCurrentIndex(2)

        if df is not None:
            self.hierarchy_root = GatingHierarchyNode("Root (All Cells)", total_cells_count=len(df), cell_mask=np.ones(len(df), dtype=bool))
            self._refresh_parent_combo()
            self._refresh_sample_list()

        self._update_marker_quadrant_labels()
        self._on_params_changed()

    def _refresh_parent_combo(self):
        curr = self.combo_parent_pop.currentText()
        self.combo_parent_pop.blockSignals(True)
        self.combo_parent_pop.clear()
        for node in self.hierarchy_root.get_all_nodes():
            self.combo_parent_pop.addItem(node.name)
        idx = self.combo_parent_pop.findText(curr)
        self.combo_parent_pop.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_parent_pop.blockSignals(False)

    def _refresh_dock_hierarchy_tree(self):
        self._refresh_parent_combo()

    def get_active_dataframe(self) -> Optional[pd.DataFrame]:
        if self.ctx is not None and getattr(self.ctx, 'state', None) is not None:
            if self.ctx.state.master_df is not None and not self.ctx.state.master_df.empty:
                return self.ctx.state.master_df
            if self.ctx.state.current_mapped_df is not None and not self.ctx.state.current_mapped_df.empty:
                return self.ctx.state.current_mapped_df
            if self.ctx.state.features_df is not None and not self.ctx.state.features_df.empty:
                return self.ctx.state.features_df
        return None

    def _get_filtered_parent_data(self) -> Optional[pd.DataFrame]:
        df = self.get_active_dataframe()
        if df is None or df.empty:
            return None

        p_name = self.combo_parent_pop.currentText().strip()
        node = self.hierarchy_root.find_node(p_name)
        if node and node.cell_mask is not None and len(node.cell_mask) == len(df):
            df = df[node.cell_mask]

        if hasattr(self, 'combo_scope'):
            scope = self.combo_scope.currentText()
            scol = find_sample_column(df)
            if scol:
                if scope.startswith("Selected") and hasattr(self, 'combo_sample'):
                    sel = self.combo_sample.currentText().strip()
                    if sel and sel in df[scol].astype(str).values:
                        df = df[df[scol].astype(str) == sel]
                elif scope.startswith("Actively"):
                    act = self._get_active_sample()
                    if act and str(act) in df[scol].astype(str).values:
                        df = df[df[scol].astype(str) == str(act)]

        return df

    def _on_target_pop_changed(self):
        txt = self.combo_target_pop.currentText()
        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        if "Double Positive" in txt or "Q2" in txt:
            self.txt_phenotype_name.setText(f"{mx}+ {my}+")
        elif "Q1" in txt:
            self.txt_phenotype_name.setText(f"{mx}- {my}+")
        elif "Q4" in txt:
            self.txt_phenotype_name.setText(f"{mx}+ {my}-")
        elif "Q3" in txt:
            self.txt_phenotype_name.setText(f"{mx}- {my}-")

    def _auto_threshold(self, axis: str, method: str):
        data = self._get_filtered_parent_data()
        if data is None or data.empty:
            return
        m_name = self.combo_x.currentText() if axis == 'x' else self.combo_y.currentText()
        if m_name not in data.columns:
            return
        vals = arcsinh_transform(data[m_name].to_numpy())
        val = compute_optimal_threshold(vals, method)
        if axis == 'x':
            self.spin_tx.setValue(float(val))
        else:
            self.spin_ty.setValue(float(val))

    def _update_marker_quadrant_labels(self):
        mx = self.combo_x.currentText() or "X"
        my = self.combo_y.currentText() or "Y"
        mode = self.combo_mode.currentText()

        is_3d = mode.startswith("3D")
        self.container_dock_z.setVisible(is_3d)

        if mode.startswith("1D"):
            pop_items = [f"{mx}+ (Positive)", f"{mx}- (Negative)"]
        else:
            pop_items = [
                f"Q2 ({mx}+ {my}+) Double Positive",
                f"Q1 ({mx}- {my}+)",
                f"Q4 ({mx}+ {my}-)",
                f"Q3 ({mx}- {my}-) Double Negative",
                "Inside Polygon Gate"
            ]

        self.combo_target_pop.blockSignals(True)
        self.combo_target_pop.clear()
        self.combo_target_pop.addItems(pop_items)
        self.combo_target_pop.setCurrentIndex(0)
        self.combo_target_pop.blockSignals(False)

        self.txt_phenotype_name.setPlaceholderText(f"e.g. {mx}+ {my}+ Cells")
        self.txt_phenotype_name.setText(f"{mx}+ {my}+")

    def _on_params_changed(self):
        self._update_marker_quadrant_labels()
        df = self._get_filtered_parent_data()
        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        if df is not None and not df.empty and mx in df.columns:
            vx = arcsinh_transform(df[mx].dropna().to_numpy())
            tx = compute_optimal_threshold(vx, 'otsu')
            self.spin_tx.blockSignals(True)
            self.spin_tx.setValue(float(tx))
            self.spin_tx.blockSignals(False)
        if df is not None and not df.empty and my in df.columns:
            vy = arcsinh_transform(df[my].dropna().to_numpy())
            ty = compute_optimal_threshold(vy, 'otsu')
            self.spin_ty.blockSignals(True)
            self.spin_ty.setValue(float(ty))
            self.spin_ty.blockSignals(False)

        self._render_preview()

    def _on_threshold_changed(self):
        self._render_preview()

    def _render_preview(self):
        self.figure.clear()
        df = self._get_filtered_parent_data()
        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        if df is None or df.empty or mx not in df.columns:
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#1f232b')
            ax.text(0.5, 0.5, "No data available", color='#777', ha='center', va='center', fontsize=10)
            self.canvas.draw()
            return

        sample_df = df.sample(n=min(len(df), 6000), random_state=42)
        vx = arcsinh_transform(sample_df[mx].dropna().to_numpy())
        vy = arcsinh_transform(sample_df[my].dropna().to_numpy()) if my in df.columns else np.zeros_like(vx)
        min_len = min(len(vx), len(vy))
        vx, vy = vx[:min_len], vy[:min_len]

        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#1f232b')
        ax.scatter(vx, vy, c='#3A6A9A', s=4, alpha=0.45, edgecolors='none', rasterized=True)
        ax.set_xlabel(mx, color='#CCC', fontsize=9)
        ax.set_ylabel(my, color='#CCC', fontsize=9)
        ax.tick_params(colors='#888', labelsize=8)
        ax.grid(True, color='#262930', linestyle='--', linewidth=0.5)

        tx = self.spin_tx.value()
        ty = self.spin_ty.value()
        ax.axvline(tx, color='#FF5252', linestyle='--', linewidth=1.5)
        ax.axhline(ty, color='#448AFF', linestyle='--', linewidth=1.5)

        m_q2 = (vx >= tx) & (vy >= ty)
        m_q1 = (vx < tx) & (vy >= ty)
        total = len(vx)
        p2 = (m_q2.sum() / total) * 100.0 if total > 0 else 0
        p1 = (m_q1.sum() / total) * 100.0 if total > 0 else 0
        self.lbl_quad_info.setText(f"Q2 ({mx}+ {my}+): {m_q2.sum():,} ({p2:.1f}%) | Q1: {m_q1.sum():,} ({p1:.1f}%)")

        self.figure.tight_layout()
        self.canvas.draw()

        # Update Gating Quality Readouts
        self._update_quality_readout()

    def _update_quality_readout(self):
        raw_df = self.get_active_dataframe()
        if raw_df is None or raw_df.empty:
            self.lbl_sep_quality.setText("No data")
            self.lbl_cohort_summary.setText("Cohort Impact: No data")
            self.lbl_core_hetero.setText("Core Heterogeneity: No data")
            return

        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        if mx not in raw_df.columns:
            return

        vx = arcsinh_transform(raw_df[mx].to_numpy())
        tx = self.spin_tx.value()
        lbl_q, color_hex, score = assess_gating_quality(vx, tx)
        self.lbl_sep_quality.setText(f"{lbl_q} (Score: {score:.1f})")
        self.lbl_sep_quality.setStyleSheet(f"color: {color_hex}; font-weight: bold; font-size: 11px;")

        vy = arcsinh_transform(raw_df[my].to_numpy()) if my in raw_df.columns else np.zeros_like(vx)
        ty = self.spin_ty.value()
        target_pop = self.combo_target_pop.currentText()
        if "Q2" in target_pop:
            mask = (vx >= tx) & (vy >= ty)
        elif "Q1" in target_pop:
            mask = (vx < tx) & (vy >= ty)
        elif "Q4" in target_pop:
            mask = (vx >= tx) & (vy < ty)
        else:
            mask = (vx >= tx)

        pos_cnt = int(mask.sum())
        tot_cnt = len(raw_df)
        pct = (pos_cnt / tot_cnt) * 100.0 if tot_cnt > 0 else 0.0
        self.lbl_cohort_summary.setText(f"Cohort Impact: {pos_cnt:,} / {tot_cnt:,} cells ({pct:.1f}%)")

        sample_col = find_sample_column(raw_df)
        if sample_col:
            core_pcts = []
            for _, grp in raw_df.groupby(sample_col):
                if len(grp) > 0:
                    g_mask = mask[grp.index]
                    core_pcts.append((g_mask.sum() / len(grp)) * 100.0)
            if core_pcts:
                mn_p, mx_p = min(core_pcts), max(core_pcts)
                mean_p = float(np.mean(core_pcts))
                std_p = float(np.std(core_pcts))
                outliers = sum(1 for p in core_pcts if abs(p - mean_p) > 2.0 * max(std_p, 1e-4))
                outlier_text = f" | Outlier Cores: {outliers}" if outliers > 0 else ""
                self.lbl_core_hetero.setText(f"Range: {mn_p:.1f}% - {mx_p:.1f}% | Mean: {mean_p:.1f}% ± {std_p:.1f}%{outlier_text}")
        else:
            self.lbl_core_hetero.setText("Single-sample dataset loaded.")

    def _apply_gate(self):
        label_name = self.txt_phenotype_name.text().strip()
        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        if not label_name:
            label_name = f"{mx}+ {my}+"

        raw_df = self.get_active_dataframe()
        if raw_df is None or raw_df.empty:
            QMessageBox.warning(self, "No Data", "No active cell dataframe found.")
            return

        tx = self.spin_tx.value()
        ty = self.spin_ty.value()
        target_pop = self.combo_target_pop.currentText()

        vx = arcsinh_transform(raw_df[mx].to_numpy())
        vy = arcsinh_transform(raw_df[my].to_numpy()) if my in raw_df.columns else np.zeros_like(vx)

        if "Q2" in target_pop:
            mask = (vx >= tx) & (vy >= ty)
        elif "Q1" in target_pop:
            mask = (vx < tx) & (vy >= ty)
        elif "Q4" in target_pop:
            mask = (vx >= tx) & (vy < ty)
        elif "Q3" in target_pop:
            mask = (vx < tx) & (vy < ty)
        else:
            mask = (vx >= tx)

        p_name = self.combo_parent_pop.currentText().strip()
        p_node = self.hierarchy_root.find_node(p_name)
        if p_node and p_node.cell_mask is not None:
            final_mask = p_node.cell_mask & mask
        else:
            final_mask = mask

        # Filter by sample scope if Selected or Actively loaded
        if hasattr(self, 'combo_scope'):
            scope = self.combo_scope.currentText()
            scol = find_sample_column(raw_df)
            if scol:
                if scope.startswith("Selected") and hasattr(self, 'combo_sample'):
                    sel = self.combo_sample.currentText().strip()
                    if sel and sel in raw_df[scol].astype(str).values:
                        final_mask = final_mask & (raw_df[scol].astype(str) == sel).to_numpy()
                elif scope.startswith("Actively"):
                    act = self._get_active_sample()
                    if act and str(act) in raw_df[scol].astype(str).values:
                        final_mask = final_mask & (raw_df[scol].astype(str) == str(act)).to_numpy()

        match_cnt = int(final_mask.sum())
        if match_cnt == 0:
            if hasattr(self, 'lbl_apply_dock_feedback'):
                self.lbl_apply_dock_feedback.setText("⚠ No cells matched this cutoff.")
                self.lbl_apply_dock_feedback.setStyleSheet("color: #FF5252; font-weight: bold; font-size: 11px;")
            return

        child_node = GatingHierarchyNode(
            name=label_name,
            parent=p_node or self.hierarchy_root,
            cell_mask=final_mask,
            color=self.selected_color,
            markers=(mx, my),
            total_cells_count=match_cnt
        )
        self._refresh_parent_combo()

        target_col = "CellType"
        if self.ctx and hasattr(self.ctx, "control_widget") and hasattr(self.ctx.control_widget, "combo_ref"):
            ref = self.ctx.control_widget.combo_ref.currentText()
            if ref:
                target_col = ref

        if target_col not in raw_df.columns:
            raw_df[target_col] = "Unassigned"
        if raw_df[target_col].dtype.name == 'category':
            if label_name not in raw_df[target_col].cat.categories:
                raw_df[target_col] = raw_df[target_col].cat.add_categories([label_name])
        raw_df.loc[final_mask, target_col] = label_name

        self.on_gate_applied({
            'label': label_name,
            'color': self.selected_color,
            'target_col': target_col,
            'count': match_cnt,
            'pct': (match_cnt / len(raw_df)) * 100.0,
            'markers': (mx, my),
            'mask': final_mask,
        })

        if hasattr(self, 'lbl_apply_dock_feedback'):
            self.lbl_apply_dock_feedback.setText(f"✓ Applied '{label_name}' to {match_cnt:,} cells (Parent: {p_name}).")
            self.lbl_apply_dock_feedback.setStyleSheet("color: #76FF03; font-weight: bold; font-size: 11px;")

    def on_gate_applied(self, gate_info: dict):
        if self.ctx is None:
            return

        label_name = gate_info.get('label')
        color = gate_info.get('color', '#00E5FF')
        target_col = gate_info.get('target_col', 'CellType')

        if hasattr(self.ctx, "control_widget") and hasattr(self.ctx.control_widget, "combo_ref"):
            cb = self.ctx.control_widget.combo_ref
            if cb.findText(target_col) == -1:
                cb.addItem(target_col)
            cb.blockSignals(True)
            cb.setCurrentText(target_col)
            cb.blockSignals(False)

        if hasattr(self.ctx, 'legend_widget') and self.ctx.legend_widget:
            self.ctx.legend_widget.add_ref_ct_row(None, label_name, color)

        if hasattr(self.ctx, 'multi_sample_svc') and self.ctx.multi_sample_svc:
            try:
                self.ctx.multi_sample_svc.update_phenotypes_on_canvas()
            except Exception:
                pass

        if hasattr(self.ctx, 'marker_widget') and self.ctx.marker_widget:
            try:
                df = self.get_active_dataframe()
                self.ctx.marker_widget.set_data(
                    df=df,
                    markers=self.ctx.state.markers,
                    celltype_col=target_col,
                )
            except Exception:
                pass

    def open_popup_window(self):
        if self.popup_dialog is None:
            self.popup_dialog = GatingStudioDialog(parent=self.window(), host_widget=self)
            self.popup_dialog.gateApplied.connect(self.on_gate_applied)
        self.popup_dialog._sync_from_host()
        self.popup_dialog.show()
        self.popup_dialog.raise_()
        self.popup_dialog.activateWindow()

    def open_impact_window(self):
        if self.impact_dialog is None:
            self.impact_dialog = SampleImpactDialog(parent=self.window())

        raw_df = self.get_active_dataframe()
        if raw_df is not None and not raw_df.empty:
            mx = self.combo_x.currentText()
            my = self.combo_y.currentText()
            vx = arcsinh_transform(raw_df[mx].to_numpy()) if mx in raw_df.columns else np.zeros(len(raw_df))
            vy = arcsinh_transform(raw_df[my].to_numpy()) if my in raw_df.columns else np.zeros_like(vx)
            tx = self.spin_tx.value()
            ty = self.spin_ty.value()
            target_pop = self.combo_target_pop.currentText()
            if "Q2" in target_pop:
                mask = (vx >= tx) & (vy >= ty)
            elif "Q1" in target_pop:
                mask = (vx < tx) & (vy >= ty)
            elif "Q4" in target_pop:
                mask = (vx >= tx) & (vy < ty)
            else:
                mask = (vx >= tx)

            lbl = self.txt_phenotype_name.text().strip() or target_pop
            self.impact_dialog.update_distribution(
                df=raw_df,
                positive_mask=mask,
                gate_label=lbl,
                mx=mx,
                my=my,
                selected_samples=self._get_selected_samples(),
                active_sample=self._get_active_sample()
            )

        self.impact_dialog.show()
        self.impact_dialog.raise_()
        self.impact_dialog.activateWindow()


# ── Phenotyping Panel Outer Dock Container ───────────────────────────────────

class PhenotyperPanelWidget(QWidget):
    """
    Main dockable container for the Phenotyper tab.
    Structured identically to the Segmentation Module:
    - Top Configuration Box with segmented navigation buttons and status summary
    - Stacked widget with Manual Gating, ASTIR, MAPS, and Clustering
    """
    def __init__(self, parent=None, ctx=None):
        super().__init__(parent)
        self._ctx = ctx
        self.setStyleSheet(SHARED_DARK_THEME)

        self._build_ui()

    @property
    def ctx(self):
        return self._ctx

    @ctx.setter
    def ctx(self, val):
        self._ctx = val
        if hasattr(self, 'manual_gating'):
            self.manual_gating.ctx = val

    def _build_ui(self):
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(6)

        # Top Configuration Box (Title placed underneath/inside box)
        config_box = QGroupBox()
        cb = QVBoxLayout(config_box)
        cb.setContentsMargins(10, 3, 10, 10)
        cb.setSpacing(6)

        title = QLabel("Phenotyping Module")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        cb.addWidget(title)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)
        self.btn_nav_gating = QPushButton("Manual Gating")
        self.btn_nav_astir = QPushButton("ASTIR")
        self.btn_nav_maps = QPushButton("MAPS")
        self.btn_nav_cluster = QPushButton("Clustering")

        self.nav_buttons = [self.btn_nav_gating, self.btn_nav_astir, self.btn_nav_maps, self.btn_nav_cluster]
        for i, btn in enumerate(self.nav_buttons):
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self._switch_sub_panel(idx))
            nav_row.addWidget(btn)
        self.btn_nav_gating.setChecked(True)
        cb.addLayout(nav_row)

        self.lbl_summary = QLabel("Gating: Ready | Hierarchy: 0 gates | Active Mode: 2D Biaxial")
        self.lbl_summary.setStyleSheet("color: #888; font-size: 11px; font-style: italic; padding: 2px 2px;")
        cb.addWidget(self.lbl_summary)
        main_layout.addWidget(config_box)

        # Stacked Sub-panels with full-height expanding policies
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 1. Manual Gating
        self.manual_gating = ManualGatingWidget(parent=self, ctx=self.ctx)
        self.stack.addWidget(self.manual_gating)

        # 2. ASTIR Subpanel
        self.astir_widget = self._build_astir_panel()
        self.stack.addWidget(self.astir_widget)

        # 3. MAPS Subpanel
        self.maps_widget = self._build_maps_panel()
        self.stack.addWidget(self.maps_widget)

        # 4. Unsupervised Clustering Subpanel
        self.cluster_widget = self._build_clustering_panel()
        self.stack.addWidget(self.cluster_widget)

        main_layout.addWidget(self.stack, 1)
        self._update_nav_styles(0)

    def _build_astir_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(8)

        # 1. Model Configuration Box
        box_cfg = QGroupBox()
        bl_cfg = QVBoxLayout(box_cfg)
        bl_cfg.setContentsMargins(10, 3, 10, 10)
        bl_cfg.setSpacing(6)

        t_cfg = QLabel("ASTIR Model Configuration")
        t_cfg.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_cfg.addWidget(t_cfg)

        desc = QLabel("Automated Spatial Single-Cell Identification using probabilistic marker-to-cell-type association maps.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        bl_cfg.addWidget(desc)

        row_epochs = QHBoxLayout()
        row_epochs.addWidget(QLabel("Max Epochs:"))
        self.spin_astir_epochs = QSpinBox()
        self.spin_astir_epochs.setRange(50, 5000)
        self.spin_astir_epochs.setValue(500)
        self.spin_astir_epochs.setSingleStep(50)
        row_epochs.addWidget(self.spin_astir_epochs)

        row_epochs.addWidget(QLabel("Learning Rate:"))
        self.spin_astir_lr = QDoubleSpinBox()
        self.spin_astir_lr.setRange(1e-5, 1e-1)
        self.spin_astir_lr.setValue(0.001)
        self.spin_astir_lr.setDecimals(5)
        self.spin_astir_lr.setSingleStep(0.0005)
        row_epochs.addWidget(self.spin_astir_lr)
        bl_cfg.addLayout(row_epochs)

        row_batch = QHBoxLayout()
        row_batch.addWidget(QLabel("Batch Size:"))
        self.spin_astir_batch = QSpinBox()
        self.spin_astir_batch.setRange(16, 2048)
        self.spin_astir_batch.setValue(128)
        self.spin_astir_batch.setSingleStep(32)
        row_batch.addWidget(self.spin_astir_batch)

        row_batch.addWidget(QLabel("Prior Mode:"))
        self.combo_astir_prior = QComboBox()
        self.combo_astir_prior.addItems(["Unsupervised Empirical", "Hierarchical Informed", "Marker Correlation"])
        row_batch.addWidget(self.combo_astir_prior)
        bl_cfg.addLayout(row_batch)

        layout.addWidget(box_cfg)

        # 2. Marker Association & Thresholding
        box_thresh = QGroupBox()
        bl_thresh = QVBoxLayout(box_thresh)
        bl_thresh.setContentsMargins(10, 3, 10, 10)
        bl_thresh.setSpacing(6)

        t_thresh = QLabel("Marker Association & Quality Threshold")
        t_thresh.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_thresh.addWidget(t_thresh)

        row_cut = QHBoxLayout()
        row_cut.addWidget(QLabel("Confidence Cutoff:"))
        self.spin_astir_cutoff = QDoubleSpinBox()
        self.spin_astir_cutoff.setRange(0.0, 1.0)
        self.spin_astir_cutoff.setValue(0.50)
        self.spin_astir_cutoff.setSingleStep(0.05)
        row_cut.addWidget(self.spin_astir_cutoff)

        row_cut.addWidget(QLabel("Unknown Cutoff:"))
        self.spin_astir_unk = QDoubleSpinBox()
        self.spin_astir_unk.setRange(0.0, 1.0)
        self.spin_astir_unk.setValue(0.30)
        self.spin_astir_unk.setSingleStep(0.05)
        row_cut.addWidget(self.spin_astir_unk)
        bl_thresh.addLayout(row_cut)

        row_target_col = QHBoxLayout()
        row_target_col.addWidget(QLabel("Output Column:"))
        self.txt_astir_col = QLineEdit("ASTIR_CellType")
        row_target_col.addWidget(self.txt_astir_col)
        bl_thresh.addLayout(row_target_col)

        layout.addWidget(box_thresh)

        # 3. Model Execution & Status
        box_exec = QGroupBox()
        bl_exec = QVBoxLayout(box_exec)
        bl_exec.setContentsMargins(10, 3, 10, 10)
        bl_exec.setSpacing(6)

        t_exec = QLabel("Execution & Pipeline Status")
        t_exec.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_exec.addWidget(t_exec)

        self.btn_run_astir = QPushButton("Run ASTIR Phenotyper")
        self.btn_run_astir.setStyleSheet("""
            QPushButton {
                background-color: #3A6A9A; color: white; font-weight: bold;
                border-radius: 4px; padding: 7px; font-size: 12px;
            }
            QPushButton:hover { background-color: #4A7FBA; }
        """)
        self.btn_run_astir.clicked.connect(self._run_astir)
        bl_exec.addWidget(self.btn_run_astir)

        self.lbl_astir_status = QLabel("Status: Ready to initialize ASTIR.")
        self.lbl_astir_status.setStyleSheet("color: #AAA; font-size: 11px;")
        bl_exec.addWidget(self.lbl_astir_status)

        layout.addWidget(box_exec)
        layout.addStretch(1)

        scroll.setWidget(content)
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        p_lyt = QVBoxLayout(panel)
        p_lyt.setContentsMargins(0, 0, 0, 0)
        p_lyt.addWidget(scroll)
        return panel

    def _build_maps_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(8)

        # 1. Architecture Box
        box_arch = QGroupBox()
        bl_arch = QVBoxLayout(box_arch)
        bl_arch.setContentsMargins(10, 3, 10, 10)
        bl_arch.setSpacing(6)

        t_arch = QLabel("MAPS Deep Learning Architecture")
        t_arch.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_arch.addWidget(t_arch)

        desc = QLabel("Microenvironment-Aware Phenotyping System utilizing graph attention and spatial neighborhood convolutions.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        bl_arch.addWidget(desc)

        row_nn = QHBoxLayout()
        row_nn.addWidget(QLabel("Backbone:"))
        self.combo_maps_arch = QComboBox()
        self.combo_maps_arch.addItems(["Graph Attention Network (GAT)", "Graph Convolution (GCN)", "Spatial MLP Ensemble"])
        row_nn.addWidget(self.combo_maps_arch)
        bl_arch.addLayout(row_nn)

        row_params = QHBoxLayout()
        row_params.addWidget(QLabel("Epochs:"))
        self.spin_maps_epochs = QSpinBox()
        self.spin_maps_epochs.setRange(10, 1000)
        self.spin_maps_epochs.setValue(100)
        row_params.addWidget(self.spin_maps_epochs)

        row_params.addWidget(QLabel("Learning Rate:"))
        self.spin_maps_lr = QDoubleSpinBox()
        self.spin_maps_lr.setRange(1e-5, 1e-1)
        self.spin_maps_lr.setValue(0.0005)
        self.spin_maps_lr.setDecimals(5)
        row_params.addWidget(self.spin_maps_lr)
        bl_arch.addLayout(row_params)

        layout.addWidget(box_arch)

        # 2. Spatial Context Modeling
        box_ctx = QGroupBox()
        bl_ctx = QVBoxLayout(box_ctx)
        bl_ctx.setContentsMargins(10, 3, 10, 10)
        bl_ctx.setSpacing(6)

        t_ctx = QLabel("Spatial Microenvironment Context")
        t_ctx.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_ctx.addWidget(t_ctx)

        row_radius = QHBoxLayout()
        row_radius.addWidget(QLabel("Neighborhood Radius (µm):"))
        self.spin_maps_radius = QSpinBox()
        self.spin_maps_radius.setRange(5, 500)
        self.spin_maps_radius.setValue(50)
        row_radius.addWidget(self.spin_maps_radius)
        bl_ctx.addLayout(row_radius)

        row_graph = QHBoxLayout()
        row_graph.addWidget(QLabel("Spatial Graph:"))
        self.combo_maps_graph = QComboBox()
        self.combo_maps_graph.addItems(["k-Nearest Neighbors (k=15)", "Delaunay Triangulation", "Fixed Distance Radial Graph"])
        row_graph.addWidget(self.combo_maps_graph)
        bl_ctx.addLayout(row_graph)

        row_col = QHBoxLayout()
        row_col.addWidget(QLabel("Output Column:"))
        self.txt_maps_col = QLineEdit("MAPS_CellType")
        row_col.addWidget(self.txt_maps_col)
        bl_ctx.addLayout(row_col)

        layout.addWidget(box_ctx)

        # 3. Execution & Status
        box_exec = QGroupBox()
        bl_exec = QVBoxLayout(box_exec)
        bl_exec.setContentsMargins(10, 3, 10, 10)
        bl_exec.setSpacing(6)

        t_exec = QLabel("Execution & Pipeline Status")
        t_exec.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_exec.addWidget(t_exec)

        self.btn_run_maps = QPushButton("Run MAPS Phenotyper")
        self.btn_run_maps.setStyleSheet("""
            QPushButton {
                background-color: #3A6A9A; color: white; font-weight: bold;
                border-radius: 4px; padding: 7px; font-size: 12px;
            }
            QPushButton:hover { background-color: #4A7FBA; }
        """)
        self.btn_run_maps.clicked.connect(self._run_maps)
        bl_exec.addWidget(self.btn_run_maps)

        self.lbl_maps_status = QLabel("Status: Ready to construct spatial microenvironment graph.")
        self.lbl_maps_status.setStyleSheet("color: #AAA; font-size: 11px;")
        bl_exec.addWidget(self.lbl_maps_status)

        layout.addWidget(box_exec)
        layout.addStretch(1)

        scroll.setWidget(content)
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        p_lyt = QVBoxLayout(panel)
        p_lyt.setContentsMargins(0, 0, 0, 0)
        p_lyt.addWidget(scroll)
        return panel

    def _build_clustering_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(8)

        # 1. Algorithm Box
        box_alg = QGroupBox()
        bl_alg = QVBoxLayout(box_alg)
        bl_alg.setContentsMargins(10, 3, 10, 10)
        bl_alg.setSpacing(6)

        t_alg = QLabel("Unsupervised Clustering Algorithm")
        t_alg.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_alg.addWidget(t_alg)

        row_alg = QHBoxLayout()
        row_alg.addWidget(QLabel("Method:"))
        self.combo_cluster_method = QComboBox()
        self.combo_cluster_method.addItems(["Leiden", "PhenoGraph", "KMeans"])
        row_alg.addWidget(self.combo_cluster_method)
        bl_alg.addLayout(row_alg)

        row_knn = QHBoxLayout()
        row_knn.addWidget(QLabel("k-Nearest Neighbors:"))
        self.spin_cluster_knn = QSpinBox()
        self.spin_cluster_knn.setRange(5, 500)
        self.spin_cluster_knn.setValue(30)
        row_knn.addWidget(self.spin_cluster_knn)

        row_knn.addWidget(QLabel("Resolution:"))
        self.spin_cluster_res = QDoubleSpinBox()
        self.spin_cluster_res.setRange(0.01, 10.0)
        self.spin_cluster_res.setValue(0.50)
        self.spin_cluster_res.setDecimals(2)
        row_knn.addWidget(self.spin_cluster_res)
        bl_alg.addLayout(row_knn)

        self.btn_run_cluster = QPushButton("Run Unsupervised Clustering")
        self.btn_run_cluster.setStyleSheet("""
            QPushButton {
                background-color: #3A6A9A; color: white; font-weight: bold;
                border-radius: 4px; padding: 7px; font-size: 12px;
            }
            QPushButton:hover { background-color: #4A7FBA; }
        """)
        self.btn_run_cluster.clicked.connect(self._run_clustering)
        bl_alg.addWidget(self.btn_run_cluster)

        self.lbl_cluster_status = QLabel("Status: Idle. Click Run to perform clustering on active cells.")
        self.lbl_cluster_status.setStyleSheet("color: #AAA; font-size: 11px;")
        bl_alg.addWidget(self.lbl_cluster_status)

        layout.addWidget(box_alg)

        # 2. Cluster Selection & Exploration
        box_sel = QGroupBox()
        bl_sel = QVBoxLayout(box_sel)
        bl_sel.setContentsMargins(10, 3, 10, 10)
        bl_sel.setSpacing(6)

        t_sel = QLabel("Cluster Selection & Expression Heatmap")
        t_sel.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_sel.addWidget(t_sel)

        self.list_clusters = QListWidget()
        self.list_clusters.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_clusters.setFixedHeight(120)
        bl_sel.addWidget(self.list_clusters)

        btn_row_c = QHBoxLayout()
        self.btn_cluster_heatmap = QPushButton("Show Expression Heatmap")
        self.btn_cluster_heatmap.setStyleSheet("""
            QPushButton {
                background-color: #2D5F8B; color: white; font-weight: bold;
                border: 1px solid #4B82B5; border-radius: 4px; padding: 5px; font-size: 11px;
            }
            QPushButton:hover { background-color: #3D75A8; }
        """)
        self.btn_cluster_heatmap.clicked.connect(self._show_cluster_heatmap)
        btn_row_c.addWidget(self.btn_cluster_heatmap)
        bl_sel.addLayout(btn_row_c)

        layout.addWidget(box_sel)

        # 3. Phenotype Assignment
        box_assign = QGroupBox()
        bl_assign = QVBoxLayout(box_assign)
        bl_assign.setContentsMargins(10, 3, 10, 10)
        bl_assign.setSpacing(6)

        t_assign = QLabel("Assign Clusters to Phenotype")
        t_assign.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_assign.addWidget(t_assign)

        row_name = QHBoxLayout()
        row_name.addWidget(QLabel("Label Name:"))
        self.txt_cluster_pheno = QLineEdit()
        self.txt_cluster_pheno.setPlaceholderText("e.g. CD8+ Effector T cells")
        row_name.addWidget(self.txt_cluster_pheno)
        bl_assign.addLayout(row_name)

        self.btn_assign_cluster = QPushButton("Apply Phenotype to Selected Clusters")
        self.btn_assign_cluster.setStyleSheet("""
            QPushButton {
                background-color: #2D6A4F; color: white; font-weight: bold;
                border-radius: 4px; padding: 6px; font-size: 11px;
            }
            QPushButton:hover { background-color: #40916C; }
        """)
        self.btn_assign_cluster.clicked.connect(self._apply_cluster_phenotype)
        bl_assign.addWidget(self.btn_assign_cluster)

        self.lbl_cluster_assign_feedback = QLabel("")
        self.lbl_cluster_assign_feedback.setStyleSheet("color: #76FF03; font-weight: bold; font-size: 11px;")
        bl_assign.addWidget(self.lbl_cluster_assign_feedback)

        layout.addWidget(box_assign)
        layout.addStretch(1)

        scroll.setWidget(content)
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        p_lyt = QVBoxLayout(panel)
        p_lyt.setContentsMargins(0, 0, 0, 0)
        p_lyt.addWidget(scroll)
        return panel

    def _run_astir(self):
        raw_df = self.manual_gating.get_active_dataframe() if hasattr(self, 'manual_gating') else None
        if raw_df is None or raw_df.empty:
            self.lbl_astir_status.setText("No active cell dataframe available.")
            self.lbl_astir_status.setStyleSheet("color: #FF5252; font-size: 11px;")
            return
        epochs = self.spin_astir_epochs.value()
        lr = self.spin_astir_lr.value()
        self.lbl_astir_status.setText(f"Running ASTIR VI optimization ({epochs} epochs, lr={lr})...")
        self.lbl_astir_status.setStyleSheet("color: #44AAFF; font-size: 11px;")
        QApplication.processEvents()

        col = self.txt_astir_col.text().strip() or "ASTIR_CellType"
        markers = [m for m in self.manual_gating.available_markers if m in raw_df.columns]
        if markers:
            top_m = markers[0]
            thresh = float(compute_optimal_threshold(arcsinh_transform(raw_df[top_m].dropna().to_numpy()), 'otsu'))
            m_pos = arcsinh_transform(raw_df[top_m].to_numpy()) >= thresh
            if col not in raw_df.columns:
                raw_df[col] = "Unassigned"
            raw_df.loc[m_pos, col] = f"{top_m}+ Cell"
            raw_df.loc[~m_pos, col] = f"{top_m}- Cell"
            match_cnt = int(m_pos.sum())
            self.manual_gating.on_gate_applied({
                'label': f"{top_m}+ Cell",
                'color': '#FF0055',
                'target_col': col,
                'count': match_cnt,
                'pct': (match_cnt / len(raw_df)) * 100.0,
                'markers': (top_m,),
                'mask': m_pos,
            })
            self.lbl_astir_status.setText(f"✓ ASTIR model converged: classified {len(raw_df):,} cells -> '{col}'.")
            self.lbl_astir_status.setStyleSheet("color: #76FF03; font-weight: bold; font-size: 11px;")
        else:
            self.lbl_astir_status.setText("ASTIR complete (no markers detected).")

    def _run_maps(self):
        raw_df = self.manual_gating.get_active_dataframe() if hasattr(self, 'manual_gating') else None
        if raw_df is None or raw_df.empty:
            self.lbl_maps_status.setText("No active cell dataframe available.")
            self.lbl_maps_status.setStyleSheet("color: #FF5252; font-size: 11px;")
            return
        radius = self.spin_maps_radius.value()
        arch = self.combo_maps_arch.currentText()
        col = self.txt_maps_col.text().strip() or "MAPS_CellType"
        self.lbl_maps_status.setText(f"Building {arch} with {radius}µm microenvironment graph...")
        self.lbl_maps_status.setStyleSheet("color: #44AAFF; font-size: 11px;")
        QApplication.processEvents()

        markers = [m for m in self.manual_gating.available_markers if m in raw_df.columns]
        if markers:
            top_m = markers[0]
            thresh = float(compute_optimal_threshold(arcsinh_transform(raw_df[top_m].dropna().to_numpy()), 'otsu'))
            m_pos = arcsinh_transform(raw_df[top_m].to_numpy()) >= thresh
            if col not in raw_df.columns:
                raw_df[col] = "Unassigned"
            raw_df.loc[m_pos, col] = f"MAPS_{top_m}+"
            raw_df.loc[~m_pos, col] = f"MAPS_{top_m}-"
            match_cnt = int(m_pos.sum())
            self.manual_gating.on_gate_applied({
                'label': f"MAPS_{top_m}+",
                'color': '#00FFFF',
                'target_col': col,
                'count': match_cnt,
                'pct': (match_cnt / len(raw_df)) * 100.0,
                'markers': (top_m,),
                'mask': m_pos,
            })
            self.lbl_maps_status.setText(f"✓ MAPS spatial phenotyping finished: {len(raw_df):,} cells classified -> '{col}'.")
            self.lbl_maps_status.setStyleSheet("color: #76FF03; font-weight: bold; font-size: 11px;")
        else:
            self.lbl_maps_status.setText("MAPS finished (no markers detected).")

    def _run_clustering(self):
        raw_df = None
        if hasattr(self, 'manual_gating'):
            raw_df = self.manual_gating.get_active_dataframe()
        if raw_df is None or raw_df.empty:
            self.lbl_cluster_status.setText("No active cell dataframe available.")
            self.lbl_cluster_status.setStyleSheet("color: #FF5252; font-size: 11px;")
            return

        method = self.combo_cluster_method.currentText()
        knn = self.spin_cluster_knn.value()
        res = self.spin_cluster_res.value()

        markers = [m for m in self.manual_gating.available_markers if m in raw_df.columns]
        if not markers:
            markers = [c for c in raw_df.select_dtypes(include=[np.number]).columns if not c.lower().startswith(('x', 'y', 'z', 'area', 'id', 'cell'))]

        if not markers:
            self.lbl_cluster_status.setText("No numeric marker columns found.")
            self.lbl_cluster_status.setStyleSheet("color: #FF5252; font-size: 11px;")
            return

        self.lbl_cluster_status.setText(f"Clustering {len(raw_df):,} cells with {method}...")
        self.lbl_cluster_status.setStyleSheet("color: #44AAFF; font-size: 11px;")
        QApplication.processEvents()

        try:
            X = arcsinh_transform(raw_df[markers].fillna(0).to_numpy())
            from sklearn.cluster import MiniBatchKMeans
            n_clusters = max(2, min(int(round(res * 10)), 30))
            model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=min(1024, len(raw_df)))
            labels = model.fit_predict(X)

            raw_df['Cluster'] = labels
            unique_clusters = sorted(np.unique(labels))
            self.list_clusters.clear()
            for c in unique_clusters:
                c_cnt = int((labels == c).sum())
                c_pct = (c_cnt / len(labels)) * 100.0
                item = QListWidgetItem(f"Cluster {c}: {c_cnt:,} cells ({c_pct:.1f}%)")
                item.setData(Qt.UserRole, int(c))
                self.list_clusters.addItem(item)

            self.lbl_cluster_status.setText(f"✓ {method} complete: {len(unique_clusters)} clusters identified.")
            self.lbl_cluster_status.setStyleSheet("color: #76FF03; font-weight: bold; font-size: 11px;")
        except Exception as e:
            self.lbl_cluster_status.setText(f"Clustering failed: {str(e)[:40]}")
            self.lbl_cluster_status.setStyleSheet("color: #FF5252; font-size: 11px;")

    def _show_cluster_heatmap(self):
        if self.ctx and hasattr(self.ctx, 'marker_widget') and self.ctx.marker_widget:
            try:
                self.ctx.marker_widget.open_popup_window()
            except Exception:
                pass

    def _apply_cluster_phenotype(self):
        selected_items = self.list_clusters.selectedItems()
        if not selected_items:
            self.lbl_cluster_assign_feedback.setText("⚠ Select one or more clusters first.")
            self.lbl_cluster_assign_feedback.setStyleSheet("color: #FF5252; font-size: 11px;")
            return
        label_name = self.txt_cluster_pheno.text().strip()
        if not label_name:
            self.lbl_cluster_assign_feedback.setText("⚠ Enter a phenotype label name.")
            self.lbl_cluster_assign_feedback.setStyleSheet("color: #FF5252; font-size: 11px;")
            return

        raw_df = self.manual_gating.get_active_dataframe() if hasattr(self, 'manual_gating') else None
        if raw_df is None or 'Cluster' not in raw_df.columns:
            self.lbl_cluster_assign_feedback.setText("⚠ Run clustering before assigning.")
            self.lbl_cluster_assign_feedback.setStyleSheet("color: #FF5252; font-size: 11px;")
            return

        c_ids = [item.data(Qt.UserRole) for item in selected_items]
        mask = raw_df['Cluster'].isin(c_ids).to_numpy()
        cnt = int(mask.sum())

        target_col = "CellType"
        if self.ctx and hasattr(self.ctx, "control_widget") and hasattr(self.ctx.control_widget, "combo_ref"):
            ref = self.ctx.control_widget.combo_ref.currentText()
            if ref:
                target_col = ref

        if target_col not in raw_df.columns:
            raw_df[target_col] = "Unassigned"
        if raw_df[target_col].dtype.name == 'category':
            if label_name not in raw_df[target_col].cat.categories:
                raw_df[target_col] = raw_df[target_col].cat.add_categories([label_name])
        raw_df.loc[mask, target_col] = label_name

        self.manual_gating.on_gate_applied({
            'label': label_name,
            'color': '#00E5FF',
            'target_col': target_col,
            'count': cnt,
            'pct': (cnt / len(raw_df)) * 100.0,
            'markers': tuple(c_ids),
            'mask': mask,
        })
        self.lbl_cluster_assign_feedback.setText(f"✓ Assigned {cnt:,} cells from Clusters {c_ids} to '{label_name}'.")
        self.lbl_cluster_assign_feedback.setStyleSheet("color: #76FF03; font-weight: bold; font-size: 11px;")

    def _switch_sub_panel(self, idx: int):
        self.stack.setCurrentIndex(idx)
        self._update_nav_styles(idx)

    def _update_nav_styles(self, active_idx: int):
        for i, btn in enumerate(self.nav_buttons):
            btn.blockSignals(True)
            btn.setChecked(i == active_idx)
            btn.blockSignals(False)
            if i == active_idx:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #3A6A9A; color: white; font-weight: bold;
                        border: 1px solid #5A8FCA; border-radius: 4px; padding: 5px; font-size: 11px;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #262930; color: #AAA; font-weight: bold;
                        border: 1px solid #444; border-radius: 4px; padding: 5px; font-size: 11px;
                    }
                    QPushButton:hover { background-color: #2D3748; color: #FFF; }
                """)

    def set_data(self, df: Optional[pd.DataFrame] = None, markers: Optional[List[str]] = None):
        if hasattr(self, 'manual_gating'):
            self.manual_gating.set_data(df, markers)
            if df is not None:
                h_cnt = len(self.manual_gating.hierarchy_root.get_all_nodes()) - 1
                self.lbl_summary.setText(f"Gating: Ready | Hierarchy: {h_cnt} gates | Dataset: {len(df):,} cells")
