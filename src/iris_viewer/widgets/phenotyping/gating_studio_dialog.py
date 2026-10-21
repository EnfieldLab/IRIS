"""
Interactive Gating Studio Dialog
================================
Full-size interactive pop-up window for flow-cytometry-style gating on multiplexed
immunofluorescence and CyTOF single-cell data.
Supports:
- 2D Biaxial Scatter & Density gating with draggable cutoff lines
- 1D Histogram thresholding
- 3D Trivariate Scatter gating
- Interactive Polygon & Rectangle shape gating
- Sequential gating hierarchy tree navigation
- Automated optimal threshold discovery (Otsu, Valley dip)
- Real-time cross-sample cohort yield impact inspection
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector, RectangleSelector

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
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from iris_viewer.widgets.phenotyping.hierarchy import GatingHierarchyNode
from iris_viewer.widgets.phenotyping.sample_impact_dialog import SampleImpactDialog
from iris_viewer.widgets.phenotyping.thresholding import (
    arcsinh_transform,
    compute_optimal_threshold,
    find_sample_column,
    log10_transform,
)

if TYPE_CHECKING:
    from iris_viewer.widgets.phenotyping.manual_gating import ManualGatingWidget


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

        # Left Area: Interactive Canvas & Toolbar
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

        # Right Area: Tabbed Gating Controls & Hierarchical Tree
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

    # Projection & Scope Group
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

    # Threshold Lines Group
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

        # Quadrant summary table
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

    # Phenotype Assignment Group
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

    # Sequential Gating Hierarchy Tab
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

    # History Tab
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

    # Synchronization & Data Handling
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

    # Interactive Plotting & Quadrant Rendering
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

    # Interactive Canvas Dragging for Cutoff Lines
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

    # Interactive Polygon & Rectangle Gating
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

    # Gating Application & Hierarchy Update
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
            _child_node = GatingHierarchyNode(
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

    def _get_active_sample(self) -> Optional[str]:
        if self.host_widget and hasattr(self.host_widget, '_get_active_sample'):
            return self.host_widget._get_active_sample()
        return None

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

    # Multi-Sample Impact Window Integration
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
