"""
iris_viewer.widgets.phenotyping.manual_gating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Manual Gating dock subpanel widget for single-cell multiplex spatial phenotyping.
Provides:
  - Ergonomic 2D / 3D / 1D biaxial gating controls
  - Automated Otsu & bimodal valley threshold discovery
  - Real-time gating quality & cohort sample impact indicators
  - Direct integration with GatingStudioDialog and SampleImpactDialog
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from iris_viewer.widgets.phenotyping.gating_studio_dialog import GatingStudioDialog
from iris_viewer.widgets.phenotyping.hierarchy import GatingHierarchyNode
from iris_viewer.widgets.phenotyping.sample_impact_dialog import SampleImpactDialog
from iris_viewer.widgets.phenotyping.thresholding import (
    arcsinh_transform,
    assess_gating_quality,
    compute_optimal_threshold,
    find_sample_column,
)
from iris_viewer.widgets.styles import SHARED_DARK_THEME


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

        # Gating Data & Projection Box
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

        # Gating Scatter Preview Box
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

        # Gating Quality & Sample Assessment
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

        # Categorize & Classify Box
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

        _child_node = GatingHierarchyNode(
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
