"""
IRIS-Viewer Phenotyping Module
==============================
Provides manual flow-cytometry-style gating (2D biaxial, 3D trivariate, and 1D thresholding),
interactive draggable cutoff lines, polygon/freehand shape gating, phenotype assignment,
and pop-up interactive gating studio.
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
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
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
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from iris_common.widgets.styles import SHARED_DARK_THEME


def arcsinh_transform(arr: np.ndarray, cofactor: float = 5.0) -> np.ndarray:
    """Arcsinh transformation commonly used in CyTOF and multiplexed IF flow gating."""
    return np.arcsinh(np.maximum(arr, 0.0) / max(cofactor, 1e-4))


def log10_transform(arr: np.ndarray) -> np.ndarray:
    """Log10(1 + x) transformation."""
    return np.log10(np.maximum(arr, 0.0) + 1.0)


class GatingStudioDialog(QDialog):
    """
    Full-size interactive pop-up window for flow-cytometry-style gating on multiplexed IF data.
    Supports 2D biaxial scatter, 3D trivariate scatter, draggable threshold lines,
    interactive polygon shape drawing, and instant phenotype classification.
    """
    gateApplied = Signal(dict)

    def __init__(self, parent=None, host_widget: Optional[ManualGatingWidget] = None):
        super().__init__(parent)
        self.host_widget = host_widget
        self.setWindowTitle("Manual Gating Studio")
        self.resize(1240, 820)
        self.setMinimumSize(950, 650)

        self.setStyleSheet("""
            QDialog { background-color: #1a1d24; color: #E0E0E0; font-size: 12px; }
            QGroupBox { border: 1px solid #444; border-radius: 8px; margin-top: 6px; padding-top: 6px; font-weight: bold; color: #DDD; }
            QPushButton {
                background-color: #2D5F8B; color: white; font-weight: bold;
                border: 1px solid #4B82B5; border-radius: 4px; padding: 5px 12px; font-size: 12px;
            }
            QPushButton:hover { background-color: #3D75A8; }
            QPushButton:disabled { background-color: #2A3A4A; color: #777; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #2A2A2A; color: white; border: 1px solid #555;
                border-radius: 4px; padding: 3px 6px; font-size: 12px;
            }
            QLabel { color: #DDD; font-size: 12px; }
            QTableWidget {
                background-color: #2A2A2A; color: white; gridline-color: #444;
                border: 1px solid #555; border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #333; color: white; padding: 4px;
                border: 1px solid #444; font-weight: bold;
            }
        """)

        # Active polygon selector
        self.poly_selector: Optional[PolygonSelector] = None
        self.rect_selector: Optional[RectangleSelector] = None
        self.polygon_vertices: Optional[np.ndarray] = None
        self.dragging_line: Optional[str] = None  # 'x' or 'y'

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

        # Top interactive prompt/hint
        self.lbl_status = QLabel("Ready. Adjust cutoff lines or click 'Draw Polygon Gate' to define a population.")
        self.lbl_status.setStyleSheet("color: #76FF03; font-size: 12px; font-style: italic; padding: 2px 4px;")
        left_layout.addWidget(self.lbl_status)

        # Matplotlib Figure & Canvas
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

        # Canvas drag & click hooks for moving threshold lines
        self.canvas.mpl_connect('button_press_event', self._on_canvas_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
        self.canvas.mpl_connect('button_release_event', self._on_canvas_release)

        root.addWidget(left_box, 3)

        # ── Right Area: Scrollable Gating Controls & Phenotyping Form ────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(460)

        right_widget = QWidget()
        self.right_layout = QVBoxLayout(right_widget)
        self.right_layout.setContentsMargins(4, 4, 4, 4)
        self.right_layout.setSpacing(10)

        self._build_projection_group()
        self._build_cutoffs_group()
        self._build_assignment_group()
        self._build_history_group()

        self.right_layout.addStretch(1)
        scroll.setWidget(right_widget)
        root.addWidget(scroll, 1)

    # ── Projection & Markers Group ───────────────────────────────────────────
    def _build_projection_group(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 4, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Gating Projection & Scale")
        title.setStyleSheet("color: #DDD; font-size: 13px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        layout.addWidget(title)

        # Mode selector (2D vs 3D)
        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["2D Biaxial Scatter", "3D Trivariate Scatter", "1D Histogram"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        row_mode.addWidget(self.combo_mode)
        layout.addLayout(row_mode)

        # Marker X
        row_x = QHBoxLayout()
        row_x.addWidget(QLabel("Marker X:"))
        self.combo_x = QComboBox()
        self.combo_x.currentIndexChanged.connect(self._on_marker_changed)
        row_x.addWidget(self.combo_x)
        layout.addLayout(row_x)

        # Marker Y
        row_y = QHBoxLayout()
        row_y.addWidget(QLabel("Marker Y:"))
        self.combo_y = QComboBox()
        self.combo_y.currentIndexChanged.connect(self._on_marker_changed)
        row_y.addWidget(self.combo_y)
        layout.addLayout(row_y)

        # Marker Z (for 3D)
        self.row_z = QHBoxLayout()
        self.row_z.addWidget(QLabel("Marker Z:"))
        self.combo_z = QComboBox()
        self.combo_z.currentIndexChanged.connect(self._on_marker_changed)
        self.row_z.addWidget(self.combo_z)
        layout.addLayout(self.row_z)
        self._set_row_visible(self.row_z, False)

        # Transformation Scale
        row_scale = QHBoxLayout()
        row_scale.addWidget(QLabel("Scale:"))
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Arcsinh (Cofactor = 5)", "Log10(1 + x)", "Linear (Raw)", "Min-Max Normalized"])
        self.combo_scale.currentIndexChanged.connect(self._on_marker_changed)
        row_scale.addWidget(self.combo_scale)
        layout.addLayout(row_scale)

        self.right_layout.addWidget(box)

    # ── Cutoffs & Interactive Gates Group ────────────────────────────────────
    def _build_cutoffs_group(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 4, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Threshold Lines & Shape Gates")
        title.setStyleSheet("color: #DDD; font-size: 13px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        layout.addWidget(title)

        # X Threshold
        row_tx = QHBoxLayout()
        row_tx.addWidget(QLabel("X Threshold:"))
        self.spin_tx = QDoubleSpinBox()
        self.spin_tx.setRange(-10000.0, 100000.0)
        self.spin_tx.setDecimals(2)
        self.spin_tx.setSingleStep(0.2)
        self.spin_tx.valueChanged.connect(self._on_threshold_changed)
        row_tx.addWidget(self.spin_tx)
        layout.addLayout(row_tx)

        # Y Threshold
        self.row_ty = QHBoxLayout()
        self.row_ty.addWidget(QLabel("Y Threshold:"))
        self.spin_ty = QDoubleSpinBox()
        self.spin_ty.setRange(-10000.0, 100000.0)
        self.spin_ty.setDecimals(2)
        self.spin_ty.setSingleStep(0.2)
        self.spin_ty.valueChanged.connect(self._on_threshold_changed)
        self.row_ty.addWidget(self.spin_ty)
        layout.addLayout(self.row_ty)

        # Z Threshold (3D)
        self.row_tz = QHBoxLayout()
        self.row_tz.addWidget(QLabel("Z Threshold:"))
        self.spin_tz = QDoubleSpinBox()
        self.spin_tz.setRange(-10000.0, 100000.0)
        self.spin_tz.setDecimals(2)
        self.spin_tz.setSingleStep(0.2)
        self.spin_tz.valueChanged.connect(self._on_threshold_changed)
        self.row_tz.addWidget(self.spin_tz)
        layout.addLayout(self.row_tz)
        self._set_row_visible(self.row_tz, False)

        # Quadrant summary statistics table
        self.lbl_quad_stats = QLabel("Quadrant Breakdown:")
        self.lbl_quad_stats.setStyleSheet("font-weight: bold; color: #BBB; font-size: 11px;")
        layout.addWidget(self.lbl_quad_stats)

        self.table_quads = QTableWidget(4, 2)
        self.table_quads.setHorizontalHeaderLabels(["Population", "Cells (% Total)"])
        self.table_quads.verticalHeader().setVisible(False)
        self.table_quads.horizontalHeader().setStretchLastSection(True)
        self.table_quads.setMaximumHeight(125)
        quad_names = ["Q1 (X- Y+)", "Q2 (X+ Y+) Double+", "Q3 (X- Y-) Double-", "Q4 (X+ Y-)"]
        for r, qn in enumerate(quad_names):
            item_name = QTableWidgetItem(qn)
            item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)
            item_val = QTableWidgetItem("0 (0.0%)")
            item_val.setFlags(item_val.flags() & ~Qt.ItemIsEditable)
            self.table_quads.setItem(r, 0, item_name)
            self.table_quads.setItem(r, 1, item_val)
        layout.addWidget(self.table_quads)

        # Shape Gating Action Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.btn_draw_poly = QPushButton("Draw Polygon Gate")
        self.btn_draw_poly.setToolTip("Click on canvas to draw polygon vertices, double-click or press Enter to close.")
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

        self.right_layout.addWidget(box)

    # ── Phenotype Assignment Group ───────────────────────────────────────────
    def _build_assignment_group(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 4, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Phenotype Assignment")
        title.setStyleSheet("color: #DDD; font-size: 13px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        layout.addWidget(title)

        row_pop = QHBoxLayout()
        row_pop.addWidget(QLabel("Target Gate:"))
        self.combo_target_pop = QComboBox()
        self.combo_target_pop.addItems([
            "Q2 (X+ Y+) Double Positive",
            "Q1 (X- Y+)",
            "Q4 (X+ Y-)",
            "Q3 (X- Y-) Double Negative",
            "Inside Polygon Gate",
            "All Quadrants (Auto-Label)"
        ])
        row_pop.addWidget(self.combo_target_pop)
        layout.addLayout(row_pop)

        row_label = QHBoxLayout()
        row_label.addWidget(QLabel("Label Name:"))
        self.txt_phenotype_name = QLineEdit()
        self.txt_phenotype_name.setPlaceholderText("e.g. CD8+ Cytotoxic T Cell")
        row_label.addWidget(self.txt_phenotype_name)
        layout.addLayout(row_label)

        # Color picker row
        row_color = QHBoxLayout()
        row_color.addWidget(QLabel("Phenotype Color:"))
        self.btn_color = QPushButton("")
        self.btn_color.setFixedWidth(50)
        self.btn_color.setFixedHeight(22)
        self.selected_color = "#00E5FF"
        self._update_color_btn()
        self.btn_color.clicked.connect(self._pick_color)
        row_color.addWidget(self.btn_color)
        row_color.addStretch(1)
        layout.addLayout(row_color)

        # Target Column
        row_col = QHBoxLayout()
        row_col.addWidget(QLabel("Target Column:"))
        self.combo_col = QComboBox()
        self.combo_col.setEditable(True)
        self.combo_col.addItems(["CellType", "Verified_CellType", "Gated_Phenotype"])
        row_col.addWidget(self.combo_col)
        layout.addLayout(row_col)

        # Apply Button
        self.btn_apply = QPushButton("Apply Gate & Classify Cells")
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #2D6A4F; color: white; font-weight: bold;
                border-radius: 4px; padding: 7px; font-size: 12px;
            }
            QPushButton:hover { background-color: #40916C; }
        """)
        self.btn_apply.clicked.connect(self._apply_gate)
        layout.addWidget(self.btn_apply)

        self.right_layout.addWidget(box)

    # ── Applied Gates History ────────────────────────────────────────────────
    def _build_history_group(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 4, 10, 10)
        layout.setSpacing(6)

        title = QLabel("Applied Gating Hierarchy")
        title.setStyleSheet("color: #DDD; font-size: 13px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        layout.addWidget(title)

        self.table_history = QTableWidget(0, 4)
        self.table_history.setHorizontalHeaderLabels(["Gate / Phenotype", "Cells", "% Total", "Markers"])
        self.table_history.verticalHeader().setVisible(False)
        self.table_history.horizontalHeader().setStretchLastSection(True)
        self.table_history.setMaximumHeight(140)
        layout.addWidget(self.table_history)

        btn_row = QHBoxLayout()
        self.btn_clear_gates = QPushButton("Reset Gates")
        self.btn_clear_gates.setStyleSheet("background-color: #5A2A2A; color: white;")
        self.btn_clear_gates.clicked.connect(self._clear_applied_gates)
        btn_row.addWidget(self.btn_clear_gates)

        self.btn_export_gated = QPushButton("Export Gates (.csv)")
        self.btn_export_gated.clicked.connect(self._export_gates_csv)
        btn_row.addWidget(self.btn_export_gated)
        layout.addLayout(btn_row)

        self.right_layout.addWidget(box)

    # ── Synchronization & Data Handling ──────────────────────────────────────
    def _sync_from_host(self):
        if self.host_widget is None:
            return
        # Populate markers
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

        # Sync thresholds
        self.spin_tx.setValue(self.host_widget.spin_tx.value())
        self.spin_ty.setValue(self.host_widget.spin_ty.value())

        self._render_plot()

    def _set_row_visible(self, layout: QHBoxLayout, visible: bool):
        for i in range(layout.count()):
            w = layout.itemAt(i).widget()
            if w:
                w.setVisible(visible)

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
        self._set_row_visible(self.row_z, is_3d)
        self._set_row_visible(self.row_tz, is_3d)
        self.combo_y.setEnabled(not is_1d)
        self.spin_ty.setEnabled(not is_1d)
        self._render_plot()

    def _on_marker_changed(self):
        # Auto-compute initial median thresholds for selected markers
        data = self._get_active_data()
        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        if data is not None and not data.empty and mx in data.columns:
            vx = self._transform_vector(data[mx].to_numpy())
            self.spin_tx.blockSignals(True)
            self.spin_tx.setValue(float(np.nanmedian(vx)))
            self.spin_tx.blockSignals(False)
        if data is not None and not data.empty and my in data.columns:
            vy = self._transform_vector(data[my].to_numpy())
            self.spin_ty.blockSignals(True)
            self.spin_ty.setValue(float(np.nanmedian(vy)))
            self.spin_ty.blockSignals(False)

        self._render_plot()

    def _on_threshold_changed(self):
        self._update_quadrant_stats_and_lines()

    def _get_active_data(self) -> Optional[pd.DataFrame]:
        if self.host_widget is not None:
            return self.host_widget.get_active_dataframe()
        return None

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
        df = self._get_active_data()
        if df is None or df.empty:
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#14161d')
            ax.text(0.5, 0.5, "No feature or annotation dataset loaded.\nLoad an annotation table or extract features first.",
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

        # Downsample for lightning-fast responsive interaction if dataset is large (> 20,000 cells)
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
            scatter = ax.scatter(vx, vy, vz, c=vy, cmap='viridis', s=6, alpha=0.6, edgecolors='none')
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
            ax.set_xlabel(mx, color='#DDD', fontsize=11)
            ax.set_ylabel("Cell Count", color='#DDD', fontsize=11)
            ax.tick_params(colors='#AAA', labelsize=9)
            ax.grid(True, color='#262930', linestyle='--', linewidth=0.5)
            self.ax = ax
            self.line_x = ax.axvline(self.spin_tx.value(), color='#FF3B30', linestyle='--', linewidth=2.0, picker=5)

        else:
            # 2D Biaxial Scatter (Default)
            if my not in df.columns:
                return
            vy = self._transform_vector(sample_df[my].dropna().to_numpy())
            min_len = min(len(vx), len(vy))
            vx, vy = vx[:min_len], vy[:min_len]

            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#14161d')

            # Density / scatter rendering
            ax.scatter(vx, vy, c='#4A7FBA', s=5, alpha=0.45, edgecolors='none', rasterized=True)
            ax.set_xlabel(f"{mx} ({self.combo_scale.currentText().split()[0]})", color='#DDD', fontsize=11)
            ax.set_ylabel(f"{my} ({self.combo_scale.currentText().split()[0]})", color='#DDD', fontsize=11)
            ax.tick_params(colors='#AAA', labelsize=9)
            ax.grid(True, color='#262930', linestyle='--', linewidth=0.5)

            # Draw threshold lines
            tx = self.spin_tx.value()
            ty = self.spin_ty.value()
            self.line_x = ax.axvline(tx, color='#FF5252', linestyle='--', linewidth=1.8, label="X Cutoff")
            self.line_y = ax.axhline(ty, color='#448AFF', linestyle='--', linewidth=1.8, label="Y Cutoff")

            # Draw polygon overlay if present
            if self.polygon_vertices is not None and len(self.polygon_vertices) > 2:
                patch = plt.Polygon(self.polygon_vertices, closed=True, fill=True,
                                    facecolor='#76FF03', alpha=0.25, edgecolor='#76FF03', linewidth=2.0)
                ax.add_patch(patch)

            self.ax = ax

        self._update_quadrant_stats_and_lines()
        self.figure.tight_layout()
        self.canvas.draw()

    def _update_quadrant_stats_and_lines(self):
        df = self._get_active_data()
        if df is None or df.empty:
            return
        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        if mx not in df.columns or my not in df.columns:
            return

        vx = self._transform_vector(df[mx].dropna().to_numpy())
        vy = self._transform_vector(df[my].dropna().to_numpy())
        min_len = min(len(vx), len(vy))
        vx, vy = vx[:min_len], vy[:min_len]
        total = len(vx)
        if total == 0:
            return

        tx = self.spin_tx.value()
        ty = self.spin_ty.value()

        # Update line positions on 2D plot
        if hasattr(self, 'line_x') and self.line_x:
            self.line_x.set_xdata([tx, tx])
        if hasattr(self, 'line_y') and self.line_y:
            self.line_y.set_ydata([ty, ty])

        # Quadrant calculations
        m_q1 = (vx < tx) & (vy >= ty)
        m_q2 = (vx >= tx) & (vy >= ty)
        m_q3 = (vx < tx) & (vy < ty)
        m_q4 = (vx >= tx) & (vy < ty)

        q_counts = [m_q1.sum(), m_q2.sum(), m_q3.sum(), m_q4.sum()]
        for r, cnt in enumerate(q_counts):
            pct = (cnt / total) * 100.0
            item = self.table_quads.item(r, 1)
            if item:
                item.setText(f"{cnt:,} ({pct:.1f}%)")

        # Polygon check if active
        if self.polygon_vertices is not None and len(self.polygon_vertices) > 2:
            path = MplPath(self.polygon_vertices)
            pts = np.column_stack((vx, vy))
            in_poly = path.contains_points(pts)
            p_cnt = in_poly.sum()
            p_pct = (p_cnt / total) * 100.0
            self.lbl_poly_status.setText(f"Polygon Gate: {p_cnt:,} cells ({p_pct:.1f}% of {total:,})")
            self.lbl_poly_status.setStyleSheet("color: #76FF03; font-size: 11px; font-weight: bold;")
        else:
            self.lbl_poly_status.setText("Shape Gate: None defined")
            self.lbl_poly_status.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")

        self.canvas.draw_idle()

    # ── Interactive Canvas Dragging for Cutoff Lines ──────────────────────────
    def _on_canvas_press(self, event):
        if event.inaxes != getattr(self, 'ax', None) or event.button != 1:
            return
        if self.poly_selector is not None and self.poly_selector.active:
            return

        # Check proximity to X or Y threshold line
        tx = self.spin_tx.value()
        ty = self.spin_ty.value()
        tol_x = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.03
        tol_y = (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.03

        if event.xdata is not None and abs(event.xdata - tx) < tol_x:
            self.dragging_line = 'x'
            self.lbl_status.setText(f"Dragging X cutoff line ({self.combo_x.currentText()})...")
        elif event.ydata is not None and abs(event.ydata - ty) < tol_y:
            self.dragging_line = 'y'
            self.lbl_status.setText(f"Dragging Y cutoff line ({self.combo_y.currentText()})...")

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
            self.lbl_status.setText("Threshold updated. Select target gate and click 'Apply Gate & Classify Cells'.")
            self.dragging_line = None
            if self.host_widget is not None:
                self.host_widget.spin_tx.setValue(self.spin_tx.value())
                self.host_widget.spin_ty.setValue(self.spin_ty.value())

    # ── Interactive Polygon / Freehand Gating ─────────────────────────────────
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
        self.lbl_status.setText("Click to add vertices. Double click or press Enter to close the gate.")

        def on_select(vertices):
            self.polygon_vertices = np.array(vertices)
            self.btn_draw_poly.setText("Draw Polygon Gate")
            self.btn_draw_poly.setStyleSheet("")
            self.combo_target_pop.setCurrentText("Inside Polygon Gate")
            self.lbl_status.setText("Polygon gate captured. Specify label name and click 'Apply Gate'.")
            self._render_plot()

        self.poly_selector = PolygonSelector(
            self.ax,
            on_select,
            useblit=True,
            props=dict(color='#76FF03', linestyle='-', linewidth=2, alpha=0.9),
            handle_props=dict(markeredgecolor='#76FF03', markerfacecolor='#14161d')
        )
        self.poly_selector.set_active(True)

    def _clear_polygon(self):
        self.polygon_vertices = None
        if self.poly_selector:
            self.poly_selector.set_active(False)
            self.poly_selector = None
        self.btn_draw_poly.setText("Draw Polygon Gate")
        self.btn_draw_poly.setStyleSheet("")
        self._render_plot()

    # ── Gate Application & Back-gating Classification ────────────────────────
    def _apply_gate(self):
        label_name = self.txt_phenotype_name.text().strip()
        if not label_name:
            QMessageBox.warning(self, "Missing Name", "Please enter a phenotype label designation (e.g. CD8+ T Cell).")
            return

        df = self._get_active_data()
        if df is None or df.empty:
            QMessageBox.warning(self, "No Data", "No active dataset found to classify.")
            return

        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        mz = self.combo_z.currentText()
        mode = self.combo_mode.currentText()
        target_pop = self.combo_target_pop.currentText()
        target_col = self.combo_col.currentText().strip() or "CellType"

        vx = self._transform_vector(df[mx].to_numpy())
        vy = self._transform_vector(df[my].to_numpy()) if my in df.columns else np.zeros_like(vx)
        total = len(df)

        tx = self.spin_tx.value()
        ty = self.spin_ty.value()
        tz = self.spin_tz.value()

        # Compute boolean filter for target cells
        if "Polygon" in target_pop and self.polygon_vertices is not None:
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
        elif mode.startswith("3D"):
            vz = self._transform_vector(df[mz].to_numpy()) if mz in df.columns else np.zeros_like(vx)
            mask = (vx >= tx) & (vy >= ty) & (vz >= tz)
        else:
            mask = (vx >= tx)

        match_count = int(mask.sum())
        if match_count == 0:
            QMessageBox.information(self, "Empty Gate", "No cells matched the selected gating criteria.")
            return

        # Ensure column exists in dataframe
        if target_col not in df.columns:
            df[target_col] = "Unassigned"

        if df[target_col].dtype.name == 'category':
            if label_name not in df[target_col].cat.categories:
                df[target_col] = df[target_col].cat.add_categories([label_name])

        df.loc[mask, target_col] = label_name

        # Add to history table
        pct = (match_count / total) * 100.0
        row_idx = self.table_history.rowCount()
        self.table_history.insertRow(row_idx)
        self.table_history.setItem(row_idx, 0, QTableWidgetItem(label_name))
        self.table_history.setItem(row_idx, 1, QTableWidgetItem(f"{match_count:,}"))
        self.table_history.setItem(row_idx, 2, QTableWidgetItem(f"{pct:.1f}%"))
        self.table_history.setItem(row_idx, 3, QTableWidgetItem(f"{mx} / {my}"))

        # Notify host widget and application controller
        gate_info = {
            'label': label_name,
            'color': self.selected_color,
            'target_col': target_col,
            'count': match_count,
            'pct': pct,
            'markers': (mx, my),
            'mask': mask,
        }
        self.gateApplied.emit(gate_info)
        if self.host_widget is not None:
            self.host_widget.on_gate_applied(gate_info)

        QMessageBox.information(
            self,
            "Gate Applied",
            f"Successfully classified {match_count:,} cells ({pct:.1f}% of total) as '{label_name}'.\n"
            f"Target Column: {target_col}\n"
            "Canvas and Legend updated."
        )
        self._render_plot()

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


class ManualGatingWidget(QWidget):
    """
    Dock subpanel for Manual Gating inside the Phenotyper tab.
    Offers compact controls, real-time threshold spinners, gating preview,
    and pop-up enlargement dialog button.
    """
    def __init__(self, parent=None, ctx=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setStyleSheet(SHARED_DARK_THEME)
        self.available_markers: List[str] = []
        self.popup_dialog: Optional[GatingStudioDialog] = None

        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # ── 1. Gating Projection Box ─────────────────────────────────────────
        box_proj = QGroupBox()
        bl_proj = QVBoxLayout(box_proj)
        bl_proj.setContentsMargins(10, 3, 10, 10)
        bl_proj.setSpacing(8)

        title_proj = QLabel("Gating Data & Projection")
        title_proj.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_proj.addWidget(title_proj)

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

        row_scale = QHBoxLayout()
        row_scale.addWidget(QLabel("Scale:"))
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Arcsinh (Cofactor = 5)", "Log10(1 + x)", "Linear (Raw)", "Min-Max Normalized"])
        self.combo_scale.currentIndexChanged.connect(self._on_params_changed)
        row_scale.addWidget(self.combo_scale)
        bl_proj.addLayout(row_scale)

        layout.addWidget(box_proj)

        # ── 2. Interactive Thresholds Box ────────────────────────────────────
        box_thresh = QGroupBox()
        bl_thresh = QVBoxLayout(box_thresh)
        bl_thresh.setContentsMargins(10, 3, 10, 10)
        bl_thresh.setSpacing(8)

        title_thresh = QLabel("Threshold Cutoffs")
        title_thresh.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_thresh.addWidget(title_thresh)

        row_tx = QHBoxLayout()
        row_tx.addWidget(QLabel("X Cutoff:"))
        self.spin_tx = QDoubleSpinBox()
        self.spin_tx.setRange(-10000.0, 100000.0)
        self.spin_tx.setDecimals(2)
        self.spin_tx.setSingleStep(0.2)
        self.spin_tx.valueChanged.connect(self._on_threshold_changed)
        row_tx.addWidget(self.spin_tx)
        bl_thresh.addLayout(row_tx)

        row_ty = QHBoxLayout()
        row_ty.addWidget(QLabel("Y Cutoff:"))
        self.spin_ty = QDoubleSpinBox()
        self.spin_ty.setRange(-10000.0, 100000.0)
        self.spin_ty.setDecimals(2)
        self.spin_ty.setSingleStep(0.2)
        self.spin_ty.valueChanged.connect(self._on_threshold_changed)
        row_ty.addWidget(self.spin_ty)
        bl_thresh.addLayout(row_ty)

        self.lbl_quad_info = QLabel("Q2 (X+ Y+): 0 cells (0.0%) | Q1: 0 (0%)")
        self.lbl_quad_info.setStyleSheet("color: #AAA; font-size: 11px; font-style: italic;")
        bl_thresh.addWidget(self.lbl_quad_info)

        layout.addWidget(box_thresh)

        # ── 3. Classification Form Box ───────────────────────────────────────
        box_class = QGroupBox()
        bl_class = QVBoxLayout(box_class)
        bl_class.setContentsMargins(10, 3, 10, 10)
        bl_class.setSpacing(8)

        title_class = QLabel("Categorize & Classify")
        title_class.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_class.addWidget(title_class)

        row_name = QHBoxLayout()
        row_name.addWidget(QLabel("Label:"))
        self.txt_phenotype_name = QLineEdit()
        self.txt_phenotype_name.setPlaceholderText("e.g. CD8+ Cytotoxic T Cell")
        row_name.addWidget(self.txt_phenotype_name)
        bl_class.addLayout(row_name)

        row_target = QHBoxLayout()
        row_target.addWidget(QLabel("Population:"))
        self.combo_target_pop = QComboBox()
        self.combo_target_pop.addItems([
            "Q2 (X+ Y+) Double Positive",
            "Q1 (X- Y+)",
            "Q4 (X+ Y-)",
            "Q3 (X- Y-) Double Negative",
            "Inside Polygon Gate"
        ])
        row_target.addWidget(self.combo_target_pop)
        bl_class.addLayout(row_target)

        self.btn_apply = QPushButton("Apply Gate to Cells")
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #2D6A4F; color: white; font-weight: bold;
                border-radius: 4px; padding: 6px; font-size: 12px;
            }
            QPushButton:hover { background-color: #40916C; }
        """)
        self.btn_apply.clicked.connect(self._apply_gate)
        bl_class.addWidget(self.btn_apply)

        layout.addWidget(box_class)

        # ── 4. Scatter Preview & Pop-Up Launcher Box ─────────────────────────
        box_plot = QGroupBox()
        bl_plot = QVBoxLayout(box_plot)
        bl_plot.setContentsMargins(10, 3, 10, 10)
        bl_plot.setSpacing(8)

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
                border: 1px solid #4B82B5; border-radius: 4px; padding: 4px 8px; font-size: 11px;
            }
            QPushButton:hover { background-color: #3D75A8; }
        """)
        self.btn_popout.clicked.connect(self.open_popup_window)
        plot_header.addWidget(self.btn_popout)
        bl_plot.addLayout(plot_header)

        self.figure = Figure(figsize=(4.2, 3.2), dpi=80)
        self.figure.patch.set_facecolor('#1f232b')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.canvas.setFixedHeight(230)
        self.canvas.mouseDoubleClickEvent = lambda event: self.open_popup_window()
        bl_plot.addWidget(self.canvas)

        layout.addWidget(box_plot)

        layout.addStretch(1)
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def set_data(self, df: Optional[pd.DataFrame] = None, markers: Optional[List[str]] = None):
        """Called when new image features or annotation master files are loaded."""
        if markers is not None:
            self.available_markers = list(markers)
            for combo in (self.combo_x, self.combo_y):
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(self.available_markers)
                combo.blockSignals(False)
            if len(self.available_markers) > 0:
                self.combo_x.setCurrentIndex(0)
            if len(self.available_markers) > 1:
                self.combo_y.setCurrentIndex(1)

        self._on_params_changed()

    def get_active_dataframe(self) -> Optional[pd.DataFrame]:
        if self.ctx is not None and getattr(self.ctx, 'state', None) is not None:
            if self.ctx.state.master_df is not None and not self.ctx.state.master_df.empty:
                return self.ctx.state.master_df
            if self.ctx.state.current_mapped_df is not None and not self.ctx.state.current_mapped_df.empty:
                return self.ctx.state.current_mapped_df
            if self.ctx.state.features_df is not None and not self.ctx.state.features_df.empty:
                return self.ctx.state.features_df
        return None

    def _on_params_changed(self):
        df = self.get_active_dataframe()
        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        if df is not None and not df.empty and mx in df.columns:
            vx = arcsinh_transform(df[mx].dropna().to_numpy())
            self.spin_tx.blockSignals(True)
            self.spin_tx.setValue(float(np.nanmedian(vx)))
            self.spin_tx.blockSignals(False)
        if df is not None and not df.empty and my in df.columns:
            vy = arcsinh_transform(df[my].dropna().to_numpy())
            self.spin_ty.blockSignals(True)
            self.spin_ty.setValue(float(np.nanmedian(vy)))
            self.spin_ty.blockSignals(False)

        self._render_preview()

    def _on_threshold_changed(self):
        self._render_preview()

    def _render_preview(self):
        self.figure.clear()
        df = self.get_active_dataframe()
        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        if df is None or df.empty or mx not in df.columns:
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#1f232b')
            ax.text(0.5, 0.5, "No data available", color='#777', ha='center', va='center', fontsize=10)
            self.canvas.draw()
            return

        sample_df = df.sample(n=min(len(df), 5000), random_state=42)
        vx = arcsinh_transform(sample_df[mx].dropna().to_numpy())
        vy = arcsinh_transform(sample_df[my].dropna().to_numpy()) if my in df.columns else np.zeros_like(vx)
        min_len = min(len(vx), len(vy))
        vx, vy = vx[:min_len], vy[:min_len]

        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#1f232b')
        ax.scatter(vx, vy, c='#3A6A9A', s=4, alpha=0.4, edgecolors='none', rasterized=True)
        ax.tick_params(colors='#888', labelsize=8)
        ax.grid(True, color='#262930', linestyle='--', linewidth=0.5)

        tx = self.spin_tx.value()
        ty = self.spin_ty.value()
        ax.axvline(tx, color='#FF5252', linestyle='--', linewidth=1.5)
        ax.axhline(ty, color='#448AFF', linestyle='--', linewidth=1.5)

        # Update info text
        m_q2 = (vx >= tx) & (vy >= ty)
        m_q1 = (vx < tx) & (vy >= ty)
        total = len(vx)
        p2 = (m_q2.sum() / total) * 100.0 if total > 0 else 0
        p1 = (m_q1.sum() / total) * 100.0 if total > 0 else 0
        self.lbl_quad_info.setText(f"Q2 (X+ Y+): {m_q2.sum():,} ({p2:.1f}%) | Q1: {m_q1.sum():,} ({p1:.1f}%)")

        self.figure.tight_layout()
        self.canvas.draw()

    def _apply_gate(self):
        label_name = self.txt_phenotype_name.text().strip()
        if not label_name:
            QMessageBox.warning(self, "Label Required", "Please specify a phenotype label (e.g. CD8+ T Cell).")
            return

        df = self.get_active_dataframe()
        if df is None or df.empty:
            QMessageBox.warning(self, "No Data", "No active cell dataframe found.")
            return

        mx = self.combo_x.currentText()
        my = self.combo_y.currentText()
        tx = self.spin_tx.value()
        ty = self.spin_ty.value()
        target_pop = self.combo_target_pop.currentText()

        vx = arcsinh_transform(df[mx].to_numpy())
        vy = arcsinh_transform(df[my].to_numpy()) if my in df.columns else np.zeros_like(vx)

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

        match_cnt = int(mask.sum())
        if match_cnt == 0:
            QMessageBox.information(self, "Empty Gate", "No cells matched this cutoff.")
            return

        target_col = "CellType"
        if self.ctx and hasattr(self.ctx, "control_widget") and hasattr(self.ctx.control_widget, "combo_ref"):
            ref = self.ctx.control_widget.combo_ref.currentText()
            if ref:
                target_col = ref

        if target_col not in df.columns:
            df[target_col] = "Unassigned"
        if df[target_col].dtype.name == 'category':
            if label_name not in df[target_col].cat.categories:
                df[target_col] = df[target_col].cat.add_categories([label_name])
        df.loc[mask, target_col] = label_name

        self.on_gate_applied({
            'label': label_name,
            'color': '#00E5FF',
            'target_col': target_col,
            'count': match_cnt,
            'pct': (match_cnt / len(df)) * 100.0,
            'markers': (mx, my),
            'mask': mask,
        })

        QMessageBox.information(self, "Gate Applied", f"Categorized {match_cnt:,} cells as '{label_name}'.")

    def on_gate_applied(self, gate_info: dict):
        """Synchronize with napari canvas and legend."""
        if self.ctx is None:
            return

        label_name = gate_info.get('label')
        color = gate_info.get('color', '#00E5FF')
        target_col = gate_info.get('target_col', 'CellType')

        # Synchronize combo_ref in control widget
        if hasattr(self.ctx, "control_widget") and hasattr(self.ctx.control_widget, "combo_ref"):
            cb = self.ctx.control_widget.combo_ref
            if cb.findText(target_col) == -1:
                cb.addItem(target_col)
            cb.blockSignals(True)
            cb.setCurrentText(target_col)
            cb.blockSignals(False)

        # Add to Legend widget
        if hasattr(self.ctx, 'legend_widget') and self.ctx.legend_widget:
            # Check if row exists, else add
            self.ctx.legend_widget.add_ref_ct_row(None, label_name, color)

        # Refresh phenotypes on canvas
        if hasattr(self.ctx, 'multi_sample_svc') and self.ctx.multi_sample_svc:
            try:
                self.ctx.multi_sample_svc.update_phenotypes_on_canvas()
            except Exception:
                pass

        # Update marker widget if present
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
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # ── Top Configuration Box: Method Switcher ───────────────────────────
        config_box = QGroupBox()
        cb = QVBoxLayout(config_box)
        cb.setContentsMargins(10, 3, 10, 10)
        cb.setSpacing(6)

        title = QLabel("Phenotyping Module")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        cb.addWidget(title)

        # Category switcher segmented buttons
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

        self.lbl_summary = QLabel("Gating: Ready | Populations: 0 defined | Active Mode: 2D Biaxial")
        self.lbl_summary.setStyleSheet("color: #888; font-size: 11px; font-style: italic; padding: 2px 2px;")
        cb.addWidget(self.lbl_summary)
        main_layout.addWidget(config_box)

        # ── Stacked Sub-panels ───────────────────────────────────────────────
        self.stack = QStackedWidget()

        # 1. Manual Gating
        self.manual_gating = ManualGatingWidget(parent=self, ctx=self.ctx)
        self.stack.addWidget(self.manual_gating)

        # 2. ASTIR Subpanel
        self.astir_widget = self._create_method_placeholder("ASTIR", "Bayesian model for spatial proteomic gating using marker-to-cell-type probability maps.")
        self.stack.addWidget(self.astir_widget)

        # 3. MAPS Subpanel
        self.maps_widget = self._create_method_placeholder("MAPS", "Deep learning cell phenotyper leveraging spatial microenvironment context.")
        self.stack.addWidget(self.maps_widget)

        # 4. Unsupervised Clustering Subpanel
        self.cluster_widget = self._create_method_placeholder("Unsupervised Clustering", "Phenotypic clustering using PhenoGraph, Leiden, or KMeans across marker expression profiles.")
        self.stack.addWidget(self.cluster_widget)

        main_layout.addWidget(self.stack, 1)
        self._update_nav_styles(0)

    def _create_method_placeholder(self, method_name: str, desc: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        box = QGroupBox()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(10, 3, 10, 10)
        bl.setSpacing(8)

        t = QLabel(f"{method_name} Configuration")
        t.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl.addWidget(t)

        info = QLabel(desc)
        info.setWordWrap(True)
        info.setStyleSheet("color: #AAA; font-size: 12px; line-height: 1.4;")
        bl.addWidget(info)

        row_epochs = QHBoxLayout()
        row_epochs.addWidget(QLabel("Max Iterations / Epochs:"))
        spin_ep = QSpinBox()
        spin_ep.setRange(10, 5000)
        spin_ep.setValue(200)
        row_epochs.addWidget(spin_ep)
        bl.addLayout(row_epochs)

        btn_run = QPushButton(f"Run {method_name}")
        btn_run.setStyleSheet("background-color: #3A6A9A; color: white; font-weight: bold; padding: 6px;")
        btn_run.clicked.connect(lambda: QMessageBox.information(self, method_name, f"{method_name} phenotyping model execution queued."))
        bl.addWidget(btn_run)

        layout.addWidget(box)
        layout.addStretch(1)
        return widget

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
        """Passes new data into subpanels."""
        if hasattr(self, 'manual_gating'):
            self.manual_gating.set_data(df, markers)
            if df is not None:
                self.lbl_summary.setText(f"Gating: Ready | Dataset: {len(df):,} cells | Active Mode: 2D Biaxial")
