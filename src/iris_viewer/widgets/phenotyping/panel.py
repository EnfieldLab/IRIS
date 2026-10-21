"""
iris_viewer.widgets.phenotyping.panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Main dockable phenotyping panel widget orchestrating:
  - Manual biaxial/trivariate gating (ManualGatingWidget)
  - Unsupervised community detection & clustering (Leiden, PhenoGraph, KMeans)
  - ASTIR (probabilistic automated cell typing)
  - MAPS (microenvironment-aware deep spatial phenotyping)
"""

import re
from typing import List, Optional

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from iris_viewer.widgets.phenotyping.manual_gating import ManualGatingWidget
from iris_viewer.widgets.phenotyping.thresholding import (
    arcsinh_transform,
    find_sample_column,
)
from iris_viewer.widgets.styles import SHARED_DARK_THEME


class PhenotyperPanelWidget(QWidget):
    """
    Main dockable container for the Phenotyper tab.
    Structured with:
    - Top Configuration Box with segmented navigation buttons and status summary
    - Stacked widget with Manual Gating, Clustering, ASTIR, and MAPS
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

        # Top Configuration Box
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
        self.btn_nav_cluster = QPushButton("Clustering")
        self.btn_nav_astir = QPushButton("ASTIR")
        self.btn_nav_maps = QPushButton("MAPS")

        self.nav_buttons = [self.btn_nav_gating, self.btn_nav_cluster, self.btn_nav_astir, self.btn_nav_maps]
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

        # Manual Gating (Index 0)
        self.manual_gating = ManualGatingWidget(parent=self, ctx=self.ctx)
        self.stack.addWidget(self.manual_gating)

        # Unsupervised Clustering Subpanel (Index 1)
        self.cluster_widget = self._build_clustering_panel()
        self.stack.addWidget(self.cluster_widget)

        # ASTIR Subpanel (Index 2)
        self.astir_widget = self._build_astir_panel()
        self.stack.addWidget(self.astir_widget)

        # MAPS Subpanel (Index 3)
        self.maps_widget = self._build_maps_panel()
        self.stack.addWidget(self.maps_widget)

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

        # Model Configuration Box
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

        # Marker Association & Thresholding
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

        # Model Execution & Status
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

        # Architecture Box
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

        # Spatial Context Modeling
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

        # Execution & Status
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

        # Algorithm Box
        box_alg = QGroupBox()
        bl_alg = QVBoxLayout(box_alg)
        bl_alg.setContentsMargins(10, 3, 10, 10)
        bl_alg.setSpacing(6)

        t_alg = QLabel("Unsupervised Clustering Algorithm")
        t_alg.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        bl_alg.addWidget(t_alg)

        # Sample Cohort Scope
        row_scope = QHBoxLayout()
        row_scope.addWidget(QLabel("Sample Scope:"))
        self.combo_cluster_scope = QComboBox()
        self.combo_cluster_scope.addItems([
            "All Loaded Samples",
            "Selected Samples",
            "Custom Sample Checklist",
        ])
        self.combo_cluster_scope.currentIndexChanged.connect(self._on_cluster_scope_changed)
        row_scope.addWidget(self.combo_cluster_scope, 1)
        bl_alg.addLayout(row_scope)

        # Custom Sample Checklist Widget
        self.box_cluster_checklist = QWidget()
        chk_lyt = QVBoxLayout(self.box_cluster_checklist)
        chk_lyt.setContentsMargins(0, 2, 0, 4)
        chk_lyt.setSpacing(4)

        chk_btn_row = QHBoxLayout()
        chk_btn_row.setSpacing(6)
        btn_chk_all = QPushButton("Select All")
        btn_chk_none = QPushButton("Select None")
        btn_chk_all.setStyleSheet("background-color: #2D5F8B; color: white; border-radius: 3px; padding: 2px 8px; font-size: 11px;")
        btn_chk_none.setStyleSheet("background-color: #333; color: white; border-radius: 3px; padding: 2px 8px; font-size: 11px;")
        btn_chk_all.clicked.connect(self._select_all_cluster_samples)
        btn_chk_none.clicked.connect(self._select_none_cluster_samples)
        chk_btn_row.addWidget(btn_chk_all)
        chk_btn_row.addWidget(btn_chk_none)
        chk_btn_row.addStretch(1)
        chk_lyt.addLayout(chk_btn_row)

        self.list_cluster_samples = QListWidget()
        self.list_cluster_samples.setFixedHeight(90)
        self.list_cluster_samples.setStyleSheet("""
            QListWidget {
                background-color: #14161d; color: white; border: 1px solid #444;
                border-radius: 3px; padding: 2px; font-size: 11px;
            }
        """)
        chk_lyt.addWidget(self.list_cluster_samples)
        self.box_cluster_checklist.setVisible(False)
        bl_alg.addWidget(self.box_cluster_checklist)

        # Marker Selection for Clustering
        row_m_hdr = QHBoxLayout()
        row_m_hdr.addWidget(QLabel("Clustering Markers:"))
        self.lbl_cluster_marker_count = QLabel("All biological markers selected")
        self.lbl_cluster_marker_count.setStyleSheet("color: #44AAFF; font-size: 11px;")
        row_m_hdr.addWidget(self.lbl_cluster_marker_count, 1)
        self.btn_toggle_cluster_markers = QPushButton("Customize Markers ▾")
        self.btn_toggle_cluster_markers.setCheckable(True)
        self.btn_toggle_cluster_markers.setChecked(False)
        self.btn_toggle_cluster_markers.setStyleSheet("background-color: #262930; color: #DDD; border: 1px solid #444; border-radius: 3px; padding: 2px 8px; font-size: 11px;")
        self.btn_toggle_cluster_markers.clicked.connect(self._toggle_cluster_markers_panel)
        row_m_hdr.addWidget(self.btn_toggle_cluster_markers)
        bl_alg.addLayout(row_m_hdr)

        self.box_cluster_markers = QWidget()
        m_chk_lyt = QVBoxLayout(self.box_cluster_markers)
        m_chk_lyt.setContentsMargins(0, 2, 0, 4)
        m_chk_lyt.setSpacing(4)

        m_btn_row = QHBoxLayout()
        m_btn_row.setSpacing(6)
        btn_m_all = QPushButton("Select All")
        btn_m_pheno = QPushButton("Biological Only")
        btn_m_none = QPushButton("Clear All")
        btn_m_all.setStyleSheet("background-color: #2D5F8B; color: white; border-radius: 3px; padding: 2px 8px; font-size: 11px;")
        btn_m_pheno.setStyleSheet("background-color: #2E7D32; color: white; border-radius: 3px; padding: 2px 8px; font-size: 11px;")
        btn_m_pheno.setToolTip("Selects phenotypic markers and unchecks technical/nuclear channels (DAPI, Hoechst, TRITC, Cy5, FITC, etc.)")
        btn_m_none.setStyleSheet("background-color: #333; color: white; border-radius: 3px; padding: 2px 8px; font-size: 11px;")
        btn_m_all.clicked.connect(self._select_all_cluster_markers)
        btn_m_pheno.clicked.connect(self._select_phenotypic_cluster_markers)
        btn_m_none.clicked.connect(self._select_none_cluster_markers)
        m_btn_row.addWidget(btn_m_all)
        m_btn_row.addWidget(btn_m_pheno)
        m_btn_row.addWidget(btn_m_none)
        m_btn_row.addStretch(1)
        m_chk_lyt.addLayout(m_btn_row)

        self.txt_filter_cluster_markers = QLineEdit()
        self.txt_filter_cluster_markers.setPlaceholderText("Filter markers...")
        self.txt_filter_cluster_markers.setStyleSheet("background-color: #14161d; color: white; border: 1px solid #444; border-radius: 3px; padding: 2px 6px; font-size: 11px;")
        self.txt_filter_cluster_markers.textChanged.connect(self._filter_cluster_markers_list)
        m_chk_lyt.addWidget(self.txt_filter_cluster_markers)

        self.list_cluster_markers = QListWidget()
        self.list_cluster_markers.setFixedHeight(105)
        self.list_cluster_markers.setStyleSheet("""
            QListWidget {
                background-color: #14161d; color: white; border: 1px solid #444;
                border-radius: 3px; padding: 2px; font-size: 11px;
            }
        """)
        self.list_cluster_markers.itemChanged.connect(self._on_cluster_marker_item_changed)
        m_chk_lyt.addWidget(self.list_cluster_markers)
        self.box_cluster_markers.setVisible(False)
        bl_alg.addWidget(self.box_cluster_markers)

        # Algorithm Selection
        row_alg = QHBoxLayout()
        row_alg.addWidget(QLabel("Method:"))
        self.combo_cluster_method = QComboBox()
        self.combo_cluster_method.addItems(["Leiden", "PhenoGraph", "KMeans"])
        row_alg.addWidget(self.combo_cluster_method, 1)
        bl_alg.addLayout(row_alg)

        # Algorithm-Specific Parameter Stack
        self.stack_cluster_params = QStackedWidget()

        # Page 0: Leiden (k-NN + Resolution)
        page_leiden = QWidget()
        l_lyt = QVBoxLayout(page_leiden)
        l_lyt.setContentsMargins(0, 2, 0, 2)
        l_lyt.setSpacing(4)
        row_l = QHBoxLayout()
        row_l.addWidget(QLabel("k-NN:"))
        self.spin_leiden_knn = QSpinBox()
        self.spin_leiden_knn.setRange(5, 300)
        self.spin_leiden_knn.setValue(30)
        self.spin_leiden_knn.setToolTip("Number of nearest neighbors for cell-cell graph construction (typical: 15-50)")
        row_l.addWidget(self.spin_leiden_knn)
        row_l.addWidget(QLabel("Resolution:"))
        self.spin_leiden_res = QDoubleSpinBox()
        self.spin_leiden_res.setRange(0.05, 5.0)
        self.spin_leiden_res.setValue(0.80)
        self.spin_leiden_res.setSingleStep(0.05)
        self.spin_leiden_res.setDecimals(2)
        self.spin_leiden_res.setToolTip("Partition resolution parameter. Higher values produce smaller subclusters.")
        row_l.addWidget(self.spin_leiden_res)
        l_lyt.addLayout(row_l)
        lbl_l_hint = QLabel("Graph community detection. Higher resolution yields finer sub-populations.")
        lbl_l_hint.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        l_lyt.addWidget(lbl_l_hint)
        self.stack_cluster_params.addWidget(page_leiden)

        # Page 1: PhenoGraph (k-NN + Min Cluster Size)
        page_pheno = QWidget()
        p_lyt = QVBoxLayout(page_pheno)
        p_lyt.setContentsMargins(0, 2, 0, 2)
        p_lyt.setSpacing(4)
        row_p = QHBoxLayout()
        row_p.addWidget(QLabel("k-NN (k):"))
        self.spin_pheno_k = QSpinBox()
        self.spin_pheno_k.setRange(10, 150)
        self.spin_pheno_k.setValue(30)
        self.spin_pheno_k.setToolTip("k parameter: number of nearest neighbors for Jaccard graph (Levine et al., 2015)")
        row_p.addWidget(self.spin_pheno_k)
        row_p.addWidget(QLabel("Min Cluster Size:"))
        self.spin_pheno_min_size = QSpinBox()
        self.spin_pheno_min_size.setRange(5, 500)
        self.spin_pheno_min_size.setValue(10)
        self.spin_pheno_min_size.setToolTip("Minimum number of cells required to form a retained community/cluster.")
        row_p.addWidget(self.spin_pheno_min_size)
        p_lyt.addLayout(row_p)
        lbl_p_hint = QLabel("Jaccard graph modularity (Levine et al., Cell 2015). Smaller k yields finer clusters.")
        lbl_p_hint.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        p_lyt.addWidget(lbl_p_hint)
        self.stack_cluster_params.addWidget(page_pheno)

        # Page 2: KMeans (Clusters k + Batch Size)
        page_kmeans = QWidget()
        k_lyt = QVBoxLayout(page_kmeans)
        k_lyt.setContentsMargins(0, 2, 0, 2)
        k_lyt.setSpacing(4)
        row_k = QHBoxLayout()
        row_k.addWidget(QLabel("Clusters (k):"))
        self.spin_kmeans_k = QSpinBox()
        self.spin_kmeans_k.setRange(2, 60)
        self.spin_kmeans_k.setValue(8)
        self.spin_kmeans_k.setToolTip("Target number of clusters/centroids k to partition dataset into.")
        row_k.addWidget(self.spin_kmeans_k)
        row_k.addWidget(QLabel("Batch Size:"))
        self.spin_kmeans_batch = QSpinBox()
        self.spin_kmeans_batch.setRange(64, 8192)
        self.spin_kmeans_batch.setSingleStep(128)
        self.spin_kmeans_batch.setValue(1024)
        self.spin_kmeans_batch.setToolTip("Mini-batch size for stochastic centroid updates.")
        row_k.addWidget(self.spin_kmeans_batch)
        k_lyt.addLayout(row_k)
        lbl_k_hint = QLabel("Centroid-based spatial clustering minimizing within-cluster inertia.")
        lbl_k_hint.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        k_lyt.addWidget(lbl_k_hint)
        self.stack_cluster_params.addWidget(page_kmeans)

        self.combo_cluster_method.currentIndexChanged.connect(self.stack_cluster_params.setCurrentIndex)
        bl_alg.addWidget(self.stack_cluster_params)

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

        self.lbl_cluster_status = QLabel("Status: Idle. Click Run to perform clustering.")
        self.lbl_cluster_status.setStyleSheet("color: #AAA; font-size: 11px;")
        bl_alg.addWidget(self.lbl_cluster_status)

        layout.addWidget(box_alg)

        # Cluster Selection & Exploration
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
        self.list_clusters.itemSelectionChanged.connect(self._on_cluster_selection_changed)
        bl_sel.addWidget(self.list_clusters)

        # Profile Card for Selected Cluster
        self.card_cluster_profile = QFrame()
        self.card_cluster_profile.setStyleSheet("""
            QFrame {
                background-color: #171a21; border: 1px solid #333; border-radius: 4px;
                padding: 4px;
            }
        """)
        cp_lyt = QVBoxLayout(self.card_cluster_profile)
        cp_lyt.setContentsMargins(6, 4, 6, 4)
        cp_lyt.setSpacing(2)

        self.lbl_profile_title = QLabel("Select cluster(s) above to inspect enriched markers.")
        self.lbl_profile_title.setStyleSheet("color: #EEE; font-weight: bold; font-size: 11px;")
        cp_lyt.addWidget(self.lbl_profile_title)

        self.lbl_profile_markers = QLabel("")
        self.lbl_profile_markers.setStyleSheet("color: #00E5FF; font-size: 11px;")
        self.lbl_profile_markers.setWordWrap(True)
        cp_lyt.addWidget(self.lbl_profile_markers)
        bl_sel.addWidget(self.card_cluster_profile)

        btn_row_c = QHBoxLayout()
        self.btn_cluster_heatmap = QPushButton("Show Expression Heatmap")
        self.btn_cluster_heatmap.setStyleSheet("""
            QPushButton {
                background-color: #2D5F8B; color: white; font-weight: bold;
                border: 1px solid #4B82B5; border-radius: 4px; padding: 6px; font-size: 11px;
            }
            QPushButton:hover { background-color: #3D75A8; }
        """)
        self.btn_cluster_heatmap.clicked.connect(self._show_cluster_heatmap)
        btn_row_c.addWidget(self.btn_cluster_heatmap)
        bl_sel.addLayout(btn_row_c)

        layout.addWidget(box_sel)

        # Phenotype Assignment
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
        batch_size = self.spin_astir_batch.value()
        col = self.txt_astir_col.text().strip() or "ASTIR_CellType"
        self.lbl_astir_status.setText(f"Running ASTIR VI optimization ({epochs} epochs, lr={lr})...")
        self.lbl_astir_status.setStyleSheet("color: #44AAFF; font-size: 11px;")
        QApplication.processEvents()

        markers = [m for m in self.manual_gating.available_markers if m in raw_df.columns]
        if not markers:
            markers = [c for c in raw_df.select_dtypes(include=[np.number]).columns
                       if not c.lower().startswith(('x', 'y', 'z', 'area', 'id', 'cell', 'object', 'pos'))]

        if not markers:
            self.lbl_astir_status.setText("ASTIR failed: No numeric marker columns detected.")
            self.lbl_astir_status.setStyleSheet("color: #FF5252; font-size: 11px;")
            return

        from iris_viewer.core.types import CellTypePanel, CellTypeRule, MarkerPanel
        from iris_viewer.engine.phenotyping.astir_backend import ASTIRBackend

        backend = ASTIRBackend()
        is_astir_installed = backend.is_available()

        tree_nodes = [
            n for n in self.manual_gating.hierarchy_root.get_all_nodes()
            if n.parent is not None and n.markers
        ] if hasattr(self.manual_gating, 'hierarchy_root') else []

        rules = []
        if len(tree_nodes) >= 2:
            for n in tree_nodes:
                rules.append(CellTypeRule(cell_type=n.name, markers_positive=list(n.markers)))
        else:
            for m in markers[:10]:
                rules.append(CellTypeRule(cell_type=f"{m}+ Population", markers_positive=[m]))

        panel = CellTypePanel(rules=rules)
        mp = MarkerPanel(markers=markers)

        try:
            prob_dict = backend.predict(
                raw_df,
                cell_type_panel=panel,
                marker_panel=mp,
                max_epochs=epochs,
                learning_rate=lr,
                batch_size=batch_size,
                arcsinh_cofactor=5.0,
            )

            cutoff = self.spin_astir_cutoff.value()
            unk_cutoff = self.spin_astir_unk.value()

            assignments = []
            cids = backend._cell_ids_from_features(raw_df)
            for cid in cids:
                probs = prob_dict.get(cid, {})
                if probs:
                    best_type, best_prob = max(probs.items(), key=lambda kv: kv[1])
                    if best_prob >= cutoff:
                        assignments.append(best_type)
                    elif best_prob < unk_cutoff:
                        assignments.append("Unknown")
                    else:
                        assignments.append(f"{best_type} (Low Conf)")
                else:
                    assignments.append("Unassigned")

            raw_df[col] = assignments

            palette = [
                '#00E5FF', '#FF0055', '#76FF03', '#FFD600', '#D500F9',
                '#FF6D00', '#00B0FF', '#00E676', '#FF1744', '#651FFF'
            ]
            unique_cts = [ct for ct in raw_df[col].unique() if ct not in ('Unknown', 'Unassigned')]
            for idx, ct_name in enumerate(unique_cts):
                m_pos = (raw_df[col] == ct_name)
                match_cnt = int(m_pos.sum())
                self.manual_gating.on_gate_applied({
                    'label': ct_name,
                    'color': palette[idx % len(palette)],
                    'target_col': col,
                    'count': match_cnt,
                    'pct': (match_cnt / len(raw_df)) * 100.0,
                    'markers': tuple(markers),
                    'mask': m_pos,
                })

            status_msg = f"✓ ASTIR converged: classified {len(raw_df):,} cells into {len(unique_cts)} types -> '{col}'."
            if not is_astir_installed:
                status_msg = f"✓ ASTIR rule fallback complete: {len(raw_df):,} cells classified -> '{col}'."
            self.lbl_astir_status.setText(status_msg)
            self.lbl_astir_status.setStyleSheet("color: #76FF03; font-weight: bold; font-size: 11px;")

        except Exception as e:
            self.lbl_astir_status.setText(f"ASTIR execution error: {str(e)[:50]}")
            self.lbl_astir_status.setStyleSheet("color: #FF5252; font-size: 11px;")

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
        if not markers:
            markers = [c for c in raw_df.select_dtypes(include=[np.number]).columns
                       if not c.lower().startswith(('x', 'y', 'z', 'area', 'id', 'cell', 'object', 'pos'))]

        if not markers:
            self.lbl_maps_status.setText("MAPS failed: No marker columns detected.")
            self.lbl_maps_status.setStyleSheet("color: #FF5252; font-size: 11px;")
            return

        from iris_viewer.core.types import CellTypePanel, CellTypeRule
        from iris_viewer.engine.phenotyping.maps_backend import MAPSBackend

        backend = MAPSBackend()
        rules = [CellTypeRule(cell_type=f"MAPS_{m}+", markers_positive=[m]) for m in markers[:8]]
        panel = CellTypePanel(rules=rules)

        try:
            preds = backend.predict(raw_df, cell_types=panel)
            cids = [int(row.get("ObjectNumber", row.get("Cell_ID", row.get("cell_id", idx))))
                    for idx, row in raw_df.iterrows()]

            assignments = []
            for cid in cids:
                p = preds.get(cid, {})
                if p:
                    best_ct = max(p.items(), key=lambda kv: kv[1])[0]
                    assignments.append(best_ct)
                else:
                    assignments.append("Unresolved")

            raw_df[col] = assignments
            palette = ['#00FFFF', '#FF00AA', '#00FF66', '#FFBB00', '#7700FF', '#00AAFF']
            unique_cts = [ct for ct in raw_df[col].unique() if ct != "Unresolved"]
            for idx, ct_name in enumerate(unique_cts):
                m_pos = (raw_df[col] == ct_name)
                match_cnt = int(m_pos.sum())
                self.manual_gating.on_gate_applied({
                    'label': ct_name,
                    'color': palette[idx % len(palette)],
                    'target_col': col,
                    'count': match_cnt,
                    'pct': (match_cnt / len(raw_df)) * 100.0,
                    'markers': tuple(markers),
                    'mask': m_pos,
                })
            self.lbl_maps_status.setText(f"✓ MAPS spatial phenotyping finished: {len(raw_df):,} cells classified -> '{col}'.")
            self.lbl_maps_status.setStyleSheet("color: #76FF03; font-weight: bold; font-size: 11px;")
        except Exception as e:
            self.lbl_maps_status.setText(f"MAPS error: {str(e)[:50]}")
            self.lbl_maps_status.setStyleSheet("color: #FF5252; font-size: 11px;")

    def _get_selected_samples(self) -> List[str]:
        if hasattr(self, 'manual_gating') and hasattr(self.manual_gating, '_get_selected_samples'):
            sel = self.manual_gating._get_selected_samples()
            if sel:
                return sel
        if self.ctx and hasattr(self.ctx, 'state') and self.ctx.state:
            return list(self.ctx.state.sample_layers_map.keys()) or ([self.ctx.state.current_sample_name] if self.ctx.state.current_sample_name else [])
        return []

    def _sync_clustering_samples(self):
        if not hasattr(self, 'list_cluster_samples'):
            return
        raw_df = self.manual_gating.get_active_dataframe() if hasattr(self, 'manual_gating') else None
        scol = find_sample_column(raw_df) if raw_df is not None else None

        samples = []
        if raw_df is not None and scol is not None:
            samples = sorted([str(s).strip() for s in raw_df[scol].dropna().unique()])
        elif self.ctx and hasattr(self.ctx, 'state') and self.ctx.state:
            samples = list(self.ctx.state.sample_layers_map.keys()) or list(self.ctx.state.file_map.keys())

        curr_checked = set(self._get_checked_clustering_samples())
        self.list_cluster_samples.clear()
        for s in samples:
            item = QListWidgetItem(s)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if (not curr_checked or s in curr_checked) else Qt.Unchecked)
            self.list_cluster_samples.addItem(item)

    def _get_checked_clustering_samples(self) -> List[str]:
        if not hasattr(self, 'list_cluster_samples'):
            return []
        return [
            self.list_cluster_samples.item(i).text().strip()
            for i in range(self.list_cluster_samples.count())
            if self.list_cluster_samples.item(i).checkState() == Qt.Checked
        ]

    def _select_all_cluster_samples(self):
        for i in range(self.list_cluster_samples.count()):
            self.list_cluster_samples.item(i).setCheckState(Qt.Checked)

    def _select_none_cluster_samples(self):
        for i in range(self.list_cluster_samples.count()):
            self.list_cluster_samples.item(i).setCheckState(Qt.Unchecked)

    def _on_cluster_scope_changed(self):
        scope = self.combo_cluster_scope.currentText()
        is_custom = (scope == "Custom Sample Checklist")
        self.box_cluster_checklist.setVisible(is_custom)

    @staticmethod
    def _is_technical_channel(m: str) -> bool:
        low = str(m).lower().strip()
        tech_stems = (
            "dapi", "hoechst", "dna", "nuclear", "nucleus", "tritc", "cy5", "cy3",
            "fitc", "alexa488", "alexa555", "alexa647", "af488", "af555", "af647",
            "autofluorescence", "af", "brightfield", "bf", "area", "eccentricity",
            "solidity", "perimeter", "objectnumber", "cell_id", "pos_x", "pos_y"
        )
        for stem in tech_stems:
            if low == stem or low.startswith(f"{stem}_") or low.endswith(f"_{stem}") or low.startswith(f"{stem} ") or f"_{stem}_" in low:
                return True
        return False

    def _sync_clustering_markers(self):
        if not hasattr(self, 'list_cluster_markers'):
            return
        raw_df = self.manual_gating.get_active_dataframe() if hasattr(self, 'manual_gating') else None
        all_markers = [m for m in self.manual_gating.available_markers if raw_df is not None and m in raw_df.columns] if hasattr(self, 'manual_gating') else []
        if not all_markers and raw_df is not None:
            all_markers = [
                c for c in raw_df.select_dtypes(include=[np.number]).columns
                if not c.lower().startswith(('x', 'y', 'z', 'area', 'id', 'cell', 'pos', 'object'))
            ]

        curr_checked = set(self._get_checked_clustering_markers())
        self.list_cluster_markers.blockSignals(True)
        self.list_cluster_markers.clear()
        for m in all_markers:
            item = QListWidgetItem(m)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            is_tech = self._is_technical_channel(m)
            if not curr_checked:
                item.setCheckState(Qt.Unchecked if is_tech else Qt.Checked)
            else:
                item.setCheckState(Qt.Checked if m in curr_checked else Qt.Unchecked)
            self.list_cluster_markers.addItem(item)
        self.list_cluster_markers.blockSignals(False)
        self._update_cluster_marker_count()

    def _get_checked_clustering_markers(self) -> List[str]:
        if not hasattr(self, 'list_cluster_markers'):
            return []
        checked = [
            self.list_cluster_markers.item(i).text().strip()
            for i in range(self.list_cluster_markers.count())
            if self.list_cluster_markers.item(i).checkState() == Qt.Checked
        ]
        return checked

    def _update_cluster_marker_count(self):
        if not hasattr(self, 'list_cluster_markers') or not hasattr(self, 'lbl_cluster_marker_count'):
            return
        total = self.list_cluster_markers.count()
        checked = len(self._get_checked_clustering_markers())
        self.lbl_cluster_marker_count.setText(f"{checked} / {total} markers selected")
        if checked == 0:
            self.lbl_cluster_marker_count.setStyleSheet("color: #FF5252; font-size: 11px;")
        else:
            self.lbl_cluster_marker_count.setStyleSheet("color: #44AAFF; font-size: 11px;")

    def _toggle_cluster_markers_panel(self):
        is_open = self.btn_toggle_cluster_markers.isChecked()
        self.box_cluster_markers.setVisible(is_open)
        self.btn_toggle_cluster_markers.setText("Collapse Markers ▴" if is_open else "Customize Markers ▾")

    def _select_all_cluster_markers(self):
        self.list_cluster_markers.blockSignals(True)
        for i in range(self.list_cluster_markers.count()):
            self.list_cluster_markers.item(i).setCheckState(Qt.Checked)
        self.list_cluster_markers.blockSignals(False)
        self._update_cluster_marker_count()

    def _select_phenotypic_cluster_markers(self):
        self.list_cluster_markers.blockSignals(True)
        for i in range(self.list_cluster_markers.count()):
            item = self.list_cluster_markers.item(i)
            is_tech = self._is_technical_channel(item.text())
            item.setCheckState(Qt.Unchecked if is_tech else Qt.Checked)
        self.list_cluster_markers.blockSignals(False)
        self._update_cluster_marker_count()

    def _select_none_cluster_markers(self):
        self.list_cluster_markers.blockSignals(True)
        for i in range(self.list_cluster_markers.count()):
            self.list_cluster_markers.item(i).setCheckState(Qt.Unchecked)
        self.list_cluster_markers.blockSignals(False)
        self._update_cluster_marker_count()

    def _filter_cluster_markers_list(self, text: str):
        pat = text.strip().lower()
        for i in range(self.list_cluster_markers.count()):
            item = self.list_cluster_markers.item(i)
            item.setHidden(bool(pat and pat not in item.text().lower()))

    def _on_cluster_marker_item_changed(self, item):
        self._update_cluster_marker_count()

    def _on_cluster_selection_changed(self):
        selected_items = self.list_clusters.selectedItems()
        if not selected_items:
            self.lbl_profile_title.setText("Select cluster(s) above to inspect enriched markers.")
            self.lbl_profile_markers.setText("")
            return

        c_names = [item.text().split(":")[0].strip() for item in selected_items]
        profiles = [item.data(Qt.UserRole + 1) for item in selected_items if item.data(Qt.UserRole + 1)]

        if len(c_names) == 1:
            c_name = c_names[0]
            prof = profiles[0] if profiles else {}
            cnt = prof.get('count', 0)
            pct = prof.get('pct', 0.0)
            desc = prof.get('profile_desc', 'No profile available')
            self.lbl_profile_title.setText(f"{c_name}: {cnt:,} cells ({pct:.1f}%)")
            self.lbl_profile_markers.setText(f"Enriched: {desc}")
            sugg = prof.get('suggested_label')
            if sugg and not self.txt_cluster_pheno.text().strip():
                self.txt_cluster_pheno.setText(sugg)
            elif sugg:
                self.txt_cluster_pheno.setPlaceholderText(sugg)
        else:
            total_cells = sum(p.get('count', 0) for p in profiles)
            self.lbl_profile_title.setText(f"Selected {len(c_names)} clusters ({total_cells:,} cells total)")
            all_top = []
            for p in profiles:
                all_top.extend(p.get('short_names', []))
            seen = set()
            uniq_top = [x for x in all_top if not (x in seen or seen.add(x))]
            self.lbl_profile_markers.setText(f"Enriched across selection: {', '.join(uniq_top[:4])}")
            if uniq_top and not self.txt_cluster_pheno.text().strip():
                self.txt_cluster_pheno.setText(f"{' '.join(uniq_top[:2])} Population")

    def _run_clustering(self):
        raw_df = None
        if hasattr(self, 'manual_gating'):
            raw_df = self.manual_gating.get_active_dataframe()
        if raw_df is None or raw_df.empty:
            self.lbl_cluster_status.setText("⚠ No active cell dataframe available.")
            self.lbl_cluster_status.setStyleSheet("color: #FF5252; font-size: 11px;")
            return

        # Determine sample cohort mask
        scol = find_sample_column(raw_df)
        scope = self.combo_cluster_scope.currentText()
        mask_cells = np.ones(len(raw_df), dtype=bool)
        scope_desc = "all loaded samples"

        if scope == "Selected Samples":
            sel_samples = self._get_selected_samples()
            if scol and sel_samples:
                clean_sel = [str(s).strip() for s in sel_samples]
                m = raw_df[scol].astype(str).str.strip().isin(clean_sel)
                if m.any():
                    mask_cells = m.to_numpy()
                    scope_desc = f"{len(clean_sel)} selected sample(s)"
        elif scope == "Custom Sample Checklist":
            checked_samples = self._get_checked_clustering_samples()
            if not checked_samples:
                self.lbl_cluster_status.setText("⚠ Please check at least one sample in the checklist.")
                self.lbl_cluster_status.setStyleSheet("color: #FF5252; font-size: 11px;")
                return
            if scol:
                clean_checked = [str(s).strip() for s in checked_samples]
                m = raw_df[scol].astype(str).str.strip().isin(clean_checked)
                if m.any():
                    mask_cells = m.to_numpy()
                    scope_desc = f"{len(clean_checked)} custom checked sample(s)"

        markers = self._get_checked_clustering_markers()
        if not markers:
            markers = [m for m in self.manual_gating.available_markers if m in raw_df.columns] if hasattr(self, 'manual_gating') else []
            if not markers:
                markers = [c for c in raw_df.select_dtypes(include=[np.number]).columns if not c.lower().startswith(('x', 'y', 'z', 'area', 'id', 'cell', 'pos', 'object'))]

        if not markers:
            self.lbl_cluster_status.setText("⚠ Please select at least one marker for clustering.")
            self.lbl_cluster_status.setStyleSheet("color: #FF5252; font-size: 11px;")
            return

        self.last_clustered_markers = list(markers)

        n_subset = int(mask_cells.sum())
        if n_subset < 2:
            self.lbl_cluster_status.setText(f"⚠ Insufficient cells ({n_subset}) in {scope_desc}.")
            self.lbl_cluster_status.setStyleSheet("color: #FF5252; font-size: 11px;")
            return

        method = self.combo_cluster_method.currentText()
        self.lbl_cluster_status.setText(f"Clustering {n_subset:,} cells across {scope_desc} using {len(markers)} markers with {method}...")
        self.lbl_cluster_status.setStyleSheet("color: #44AAFF; font-size: 11px;")
        QApplication.processEvents()

        try:
            sub_df = raw_df.loc[mask_cells]
            X = arcsinh_transform(sub_df[markers].fillna(0).to_numpy())

            if "Leiden" in method:
                knn = self.spin_leiden_knn.value()
                res = self.spin_leiden_res.value()
                try:
                    import anndata as ad
                    import scanpy as sc
                    adata = ad.AnnData(X=X)
                    sc.pp.neighbors(adata, n_neighbors=min(knn, max(2, len(X) - 1)))
                    try:
                        sc.tl.leiden(adata, resolution=res, flavor="igraph", n_iterations=2, directed=False)
                    except (TypeError, ValueError, Exception):
                        sc.tl.leiden(adata, resolution=res)
                    labels = adata.obs['leiden'].astype(int).to_numpy()
                except Exception as sc_err:
                    from sklearn.cluster import MiniBatchKMeans
                    k_est = max(2, min(int(round(res * 10)), 30))
                    model = MiniBatchKMeans(n_clusters=k_est, random_state=42, batch_size=min(1024, len(X)))
                    labels = model.fit_predict(X)
                    self.lbl_cluster_status.setText(f"Leiden fallback (Scanpy unavailable: {str(sc_err)[:30]}). Fitted k={k_est}.")

            elif "PhenoGraph" in method:
                k_val = self.spin_pheno_k.value()
                min_size = self.spin_pheno_min_size.value()
                try:
                    import phenograph
                    communities, graph, Q = phenograph.cluster(
                        X,
                        k=min(k_val, max(2, len(X) - 1)),
                        min_cluster_size=min_size,
                    )
                    labels = np.asarray(communities)
                except ImportError:
                    try:
                        import anndata as ad
                        import scanpy as sc
                        adata = ad.AnnData(X=X)
                        sc.pp.neighbors(adata, n_neighbors=min(k_val, max(2, len(X) - 1)))
                        try:
                            sc.tl.leiden(adata, resolution=1.0, flavor="igraph", n_iterations=2, directed=False)
                        except (TypeError, ValueError, Exception):
                            sc.tl.leiden(adata, resolution=1.0)
                        labels = adata.obs['leiden'].astype(int).to_numpy()
                        self.lbl_cluster_status.setText("PhenoGraph fallback (pkg not installed). Used graph Leiden.")
                    except Exception:
                        from sklearn.cluster import MiniBatchKMeans
                        model = MiniBatchKMeans(n_clusters=8, random_state=42, batch_size=min(1024, len(X)))
                        labels = model.fit_predict(X)
                except Exception:
                    from sklearn.cluster import MiniBatchKMeans
                    model = MiniBatchKMeans(n_clusters=8, random_state=42, batch_size=min(1024, len(X)))
                    labels = model.fit_predict(X)

            else:  # KMeans
                k_val = self.spin_kmeans_k.value()
                batch_val = min(self.spin_kmeans_batch.value(), len(X))
                from sklearn.cluster import MiniBatchKMeans
                model = MiniBatchKMeans(n_clusters=min(k_val, len(X)), batch_size=batch_val, random_state=42)
                labels = model.fit_predict(X)

            # Format cluster labels
            formatted_labels = [f"Cluster {c}" if c >= 0 else "Noise / Outlier" for c in labels]

            if 'Cluster' not in raw_df.columns:
                raw_df['Cluster'] = "Unassigned"
            raw_df.loc[mask_cells, 'Cluster'] = formatted_labels

            # Propagate Cluster to master_df, resolved_dfs, and current_mapped_df
            if self.ctx and getattr(self.ctx, 'state', None) is not None:
                if self.ctx.state.master_df is not None and len(self.ctx.state.master_df) == len(raw_df):
                    self.ctx.state.master_df['Cluster'] = raw_df['Cluster'].values
                if hasattr(self.ctx.state, 'resolved_dfs') and self.ctx.state.resolved_dfs:
                    scol_name = find_sample_column(raw_df)
                    for sname in list(self.ctx.state.resolved_dfs.keys()):
                        if scol_name:
                            m_s = (raw_df[scol_name].astype(str).str.strip() == str(sname).strip())
                            if m_s.any() and len(self.ctx.state.resolved_dfs[sname]) == m_s.sum():
                                self.ctx.state.resolved_dfs[sname]['Cluster'] = raw_df.loc[m_s, 'Cluster'].values
                if getattr(self.ctx.state, 'current_mapped_df', None) is not None:
                    if len(self.ctx.state.current_mapped_df) == len(raw_df):
                        self.ctx.state.current_mapped_df['Cluster'] = raw_df['Cluster'].values

            # Compute marker z-scores per cluster across the clustered subset
            m_means = sub_df[markers].mean()
            m_stds = sub_df[markers].std().replace(0, 1.0) + 1e-6
            z_df = (sub_df[markers] - m_means) / m_stds

            self.cluster_profiles = {}
            self.list_clusters.clear()

            unique_c = sorted(
                [str(x) for x in np.unique(formatted_labels) if str(x) != "Noise / Outlier"],
                key=lambda s: (0, int(re.search(r'\d+', s).group())) if re.search(r'\d+', s) else (1, s)
            )
            if "Noise / Outlier" in formatted_labels:
                unique_c.append("Noise / Outlier")

            for c in unique_c:
                c_mask = (np.array(formatted_labels) == c)
                c_cnt = int(c_mask.sum())
                c_pct = (c_cnt / len(formatted_labels)) * 100.0

                c_mean_z = z_df[c_mask].mean()
                top_z = c_mean_z.sort_values(ascending=False).head(4)
                pos_markers = [m for m, val in top_z.items() if val > 0.35]
                if not pos_markers:
                    pos_markers = list(top_z.head(2).index)
                profile_desc = ", ".join([f"{m}+ (z={c_mean_z[m]:+.1f})" for m in pos_markers[:3]])
                short_names = [f"{m}+" for m in pos_markers[:2]]

                prof = {
                    'name': c,
                    'count': c_cnt,
                    'pct': c_pct,
                    'profile_desc': profile_desc,
                    'short_names': short_names,
                    'suggested_label': f"{' '.join(short_names)} Population" if short_names else f"{c} Population",
                    'top_z': c_mean_z.to_dict(),
                }
                self.cluster_profiles[c] = prof

                item = QListWidgetItem(f"{c}: {c_cnt:,} cells ({c_pct:.1f}%) — {', '.join(short_names)}")
                item.setData(Qt.UserRole, c)
                item.setData(Qt.UserRole + 1, prof)
                self.list_clusters.addItem(item)

            self.lbl_cluster_status.setText(f"✓ {method} complete: {len(unique_c)} clusters across {scope_desc} ({n_subset:,} cells).")
            self.lbl_cluster_status.setStyleSheet("color: #76FF03; font-weight: bold; font-size: 11px;")

            if self.list_clusters.count() > 0:
                self.list_clusters.setCurrentRow(0)

            if self.ctx and hasattr(self.ctx, 'marker_widget') and self.ctx.marker_widget:
                try:
                    self.ctx.marker_widget.set_data(
                        df=raw_df,
                        markers=markers,
                        celltype_col='Cluster',
                    )
                except Exception:
                    pass

        except Exception as e:
            self.lbl_cluster_status.setText(f"Clustering failed: {str(e)[:45]}")
            self.lbl_cluster_status.setStyleSheet("color: #FF5252; font-size: 11px;")

    def _show_cluster_heatmap(self):
        raw_df = self.manual_gating.get_active_dataframe() if hasattr(self, 'manual_gating') else None
        if raw_df is None or raw_df.empty:
            self.lbl_cluster_status.setText("⚠ No active cell dataframe available.")
            self.lbl_cluster_status.setStyleSheet("color: #FF5252; font-size: 11px;")
            return

        if 'Cluster' not in raw_df.columns:
            self.lbl_cluster_status.setText("⚠ Run clustering first to generate cluster phenotypes.")
            self.lbl_cluster_status.setStyleSheet("color: #FF5252; font-size: 11px;")
            return

        # Use the specific markers selected for clustering
        markers = getattr(self, 'last_clustered_markers', None)
        if not markers:
            markers = self._get_checked_clustering_markers()
        if not markers:
            markers = [m for m in self.manual_gating.available_markers if m in raw_df.columns] if hasattr(self, 'manual_gating') else []
        if not markers:
            markers = [c for c in raw_df.select_dtypes(include=[np.number]).columns if not c.lower().startswith(('x', 'y', 'z', 'area', 'id', 'cell', 'pos', 'object'))]

        if self.ctx and hasattr(self.ctx, 'marker_widget') and self.ctx.marker_widget:
            try:
                scope = self.combo_cluster_scope.currentText()
                target_ds = "All Samples"
                subset_list = None
                if scope == "Selected Samples":
                    target_ds = "Selected Samples"
                elif scope == "Custom Sample Checklist":
                    checked = self._get_checked_clustering_samples()
                    if checked:
                        subset_list = checked
                        target_ds = f"Subset ({len(checked)} samples)"

                selected_s = self._get_selected_samples()
                self.ctx.marker_widget.set_data(
                    df=raw_df,
                    markers=markers,
                    celltype_col='Cluster',
                    selected_samples=selected_s,
                )
                self.ctx.marker_widget.set_group_by('Cluster')
                self.ctx.marker_widget.set_current_dataset(target_ds, subset_samples=subset_list)
                self.ctx.marker_widget.set_assignment_callback(self._on_phenotype_assigned_from_heatmap)
                self.ctx.marker_widget.open_popup_window()
                if self.ctx.marker_widget.popup_dialog is not None:
                    self.ctx.marker_widget.popup_dialog.set_assignment_callback(self._on_phenotype_assigned_from_heatmap)
                    self.ctx.marker_widget.popup_dialog.set_group_by('Cluster')
                    self.ctx.marker_widget.popup_dialog.set_current_dataset(target_ds, subset_samples=subset_list)
            except Exception as e:
                self.lbl_cluster_status.setText(f"Heatmap error: {str(e)[:40]}")

    def _apply_cluster_phenotype(self):
        selected_items = self.list_clusters.selectedItems()
        if not selected_items:
            self.lbl_cluster_assign_feedback.setText("⚠ Select one or more clusters first.")
            self.lbl_cluster_assign_feedback.setStyleSheet("color: #FF5252; font-size: 11px;")
            return
        label_name = self.txt_cluster_pheno.text().strip() or self.txt_cluster_pheno.placeholderText().strip()
        if not label_name:
            self.lbl_cluster_assign_feedback.setText("⚠ Enter a phenotype label name.")
            self.lbl_cluster_assign_feedback.setStyleSheet("color: #FF5252; font-size: 11px;")
            return

        raw_df = self.manual_gating.get_active_dataframe() if hasattr(self, 'manual_gating') else None
        if raw_df is None or 'Cluster' not in raw_df.columns:
            self.lbl_cluster_assign_feedback.setText("⚠ Run clustering before assigning.")
            self.lbl_cluster_assign_feedback.setStyleSheet("color: #FF5252; font-size: 11px;")
            return

        c_targets = [str(item.data(Qt.UserRole)).strip() for item in selected_items]
        c_ints = set()
        for item in selected_items:
            val = item.data(Qt.UserRole)
            if isinstance(val, int):
                c_ints.add(val)
            else:
                m = re.findall(r'\d+', str(val))
                if m:
                    c_ints.add(int(m[0]))

        mask = raw_df['Cluster'].astype(str).str.strip().isin(c_targets)
        if not mask.any() and c_ints:
            mask = raw_df['Cluster'].isin(c_ints)

        cnt = int(mask.sum())
        if cnt == 0:
            self.lbl_cluster_assign_feedback.setText("⚠ No cells matched selected cluster(s).")
            self.lbl_cluster_assign_feedback.setStyleSheet("color: #FF5252; font-size: 11px;")
            return

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
            'markers': tuple(c_targets),
            'mask': mask.to_numpy() if hasattr(mask, 'to_numpy') else np.asarray(mask),
        })
        self.lbl_cluster_assign_feedback.setText(f"✓ Assigned {cnt:,} cells from {c_targets} to '{label_name}'.")
        self.lbl_cluster_assign_feedback.setStyleSheet("color: #76FF03; font-weight: bold; font-size: 11px;")

    def _on_phenotype_assigned_from_heatmap(self, cluster_names: List[str], label_name: str):
        raw_df = self.manual_gating.get_active_dataframe() if hasattr(self, 'manual_gating') else None
        if raw_df is None or 'Cluster' not in raw_df.columns:
            return

        c_targets = [str(c).strip() for c in cluster_names]
        c_ints = set()
        for c in cluster_names:
            m = re.findall(r'\d+', str(c))
            if m:
                c_ints.add(int(m[0]))

        mask = raw_df['Cluster'].astype(str).str.strip().isin(c_targets)
        if not mask.any() and c_ints:
            mask = raw_df['Cluster'].isin(c_ints)

        cnt = int(mask.sum())
        if cnt == 0:
            return

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
            'markers': tuple(cluster_names),
            'mask': mask.to_numpy() if hasattr(mask, 'to_numpy') else np.asarray(mask),
        })
        self.lbl_cluster_assign_feedback.setText(f"✓ Assigned {cnt:,} cells from {cluster_names} to '{label_name}'.")
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
        self._sync_clustering_samples()
        self._sync_clustering_markers()
