"""
iris_viewer.widgets.phenotyping.panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Main dockable phenotyping panel widget orchestrating:
  - Manual biaxial/trivariate gating (ManualGatingWidget)
  - Unsupervised community detection & clustering (Leiden, PhenoGraph, KMeans)
  - ASTIR (probabilistic Bayesian variational inference)
  - MAPS (Mahmood Lab 4-layer feed-forward MLP phenotyping)
  - STELLAR (spatial graph label transfer & novel population discovery)
  - CellSighter (ResNet50 multi-channel image crop ensemble)
  - Nimbus (zero-shot Residual U-Net marker positivity inference)
"""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
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

from iris_common.logging import iris_log
from iris_common.widgets.sample_scope import (
    SCOPE_ACTIVE,
    SCOPE_ALL,
    SCOPE_SELECTED,
    SampleScopeBar,
    _normalize_sample_name,
)
from iris_viewer.widgets.phenotyping.diagnostics_dialog import PhenotyperDiagnosticDialog
from iris_viewer.widgets.phenotyping.manual_gating import ManualGatingWidget
from iris_viewer.widgets.phenotyping.sample_impact_dialog import _ScopeComboBox
from iris_viewer.widgets.phenotyping.thresholding import (
    arcsinh_transform,
    find_sample_column,
    resolve_marker_col,
)
from iris_viewer.widgets.styles import (
    BUTTON_CTA,
    POPOUT_BUTTON_STYLE,
    SECTION_SUBTITLE_STYLE,
    SECTION_TITLE_STYLE,
    SHARED_DARK_THEME,
    create_section_card,
    style_section_header,
)

_DEFAULT_NIMBUS_CHECKPOINT = Path(
    "/Users/wwu/Documents/IRIS_Benchmarks/nimbus/upstream_repo/"
    "src/nimbus_inference/assets/V1.pt"
)
_FALLBACK_NIMBUS_CHECKPOINT = Path(
    "/Users/wwu/Documents/IRIS_Benchmarks/nimbus/checkpoints/V1.pt"
)
_UNLABELED_SENTINELS = {"", "unassigned", "unknown", "unresolved", "nan", "none"}


class PhenotyperPanelWidget(QWidget):
    """Main dockable container for the Phenotyper tab."""

    def __init__(self, parent=None, ctx=None):
        super().__init__(parent)
        self._ctx = ctx
        self.diagnostics_dialog: Optional[PhenotyperDiagnosticDialog] = None
        self._astir_yaml_path: Optional[str] = None
        self._astir_yaml_panel: Any = None
        self._astir_marker_config: Optional[Dict[str, Any]] = None
        self._syncing_scope: bool = False
        self.setStyleSheet(SHARED_DARK_THEME)

        self._build_ui()

    @property
    def ctx(self):
        return self._ctx

    @ctx.setter
    def ctx(self, val):
        self._ctx = val
        if hasattr(self, "manual_gating"):
            self.manual_gating.ctx = val
        if val is not None:
            self.sync_sample_scope()

    def _build_ui(self):
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)

        header_wrap = QWidget()
        hw_lyt = QVBoxLayout(header_wrap)
        hw_lyt.setContentsMargins(10, 10, 10, 0)
        hw_lyt.setSpacing(0)

        # Top Configuration Card
        config_box, cb = create_section_card(margins=(10, 6, 10, 10), spacing=8)

        hdr_row = QHBoxLayout()
        title = QLabel("Phenotyping Module")
        title.setStyleSheet(SECTION_TITLE_STYLE)
        hdr_row.addWidget(title)
        hdr_row.addStretch(1)

        self.btn_diagnostics_popout = QPushButton("⤢ Pop-up Window")
        self.btn_diagnostics_popout.setToolTip(
            "Open Phenotyping Diagnostics, Nimbus Heatmap & Method Concordance Studio"
        )
        self.btn_diagnostics_popout.setStyleSheet(POPOUT_BUTTON_STYLE)
        self.btn_diagnostics_popout.setMinimumHeight(28)
        self.btn_diagnostics_popout.clicked.connect(self.open_diagnostics_dialog)
        hdr_row.addWidget(self.btn_diagnostics_popout)
        cb.addLayout(hdr_row)

        # Compact 2-row navigation bar across all 7 tabs
        nav_row1 = QHBoxLayout()
        nav_row1.setSpacing(4)
        self.btn_nav_gating = QPushButton("Manual Gating")
        self.btn_nav_cluster = QPushButton("Clustering")
        self.btn_nav_astir = QPushButton("ASTIR")
        self.btn_nav_maps = QPushButton("MAPS")

        self.nav_buttons = [
            self.btn_nav_gating,
            self.btn_nav_cluster,
            self.btn_nav_astir,
            self.btn_nav_maps,
        ]
        for i, btn in enumerate(self.nav_buttons):
            btn.setCheckable(True)
            btn.setMinimumHeight(26)
            btn.clicked.connect(lambda checked, idx=i: self._switch_sub_panel(idx))
            nav_row1.addWidget(btn)
        cb.addLayout(nav_row1)

        nav_row2 = QHBoxLayout()
        nav_row2.setSpacing(4)
        self.btn_nav_stellar = QPushButton("STELLAR")
        self.btn_nav_cellsighter = QPushButton("CellSighter")
        self.btn_nav_nimbus = QPushButton("Nimbus")

        self.dl_nav_buttons = [
            self.btn_nav_stellar,
            self.btn_nav_cellsighter,
            self.btn_nav_nimbus,
        ]
        for offset, btn in enumerate(self.dl_nav_buttons, start=4):
            btn.setCheckable(True)
            btn.setMinimumHeight(26)
            btn.clicked.connect(lambda checked, idx=offset: self._switch_sub_panel(idx))
            nav_row2.addWidget(btn)
        cb.addLayout(nav_row2)

        self.all_nav_buttons = self.nav_buttons + self.dl_nav_buttons
        self.btn_nav_gating.setChecked(True)

        self.sample_scope_bar = SampleScopeBar(
            default_mode=SCOPE_ALL,
            label="Sample Scope:",
            compact=True,
            parent=config_box,
        )
        self.sample_scope_bar.scopeChanged.connect(self._on_sample_scope_changed)
        cb.addWidget(self.sample_scope_bar)

        self.lbl_summary = QLabel("Gating: Ready | Hierarchy: 0 gates | Active Mode: 2D Biaxial")
        self.lbl_summary.setStyleSheet(SECTION_SUBTITLE_STYLE)
        self.lbl_summary.setWordWrap(True)
        cb.addWidget(self.lbl_summary)

        hw_lyt.addWidget(config_box)
        main_layout.addWidget(header_wrap)

        # Stacked Sub-panels with full-height expanding policies
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Manual Gating (Index 0)
        self.manual_gating = ManualGatingWidget(parent=self, ctx=self.ctx)
        self.manual_gating.scopeChanged.connect(self._on_manual_gating_scope_changed)
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

        # STELLAR Subpanel (Index 4)
        self.stellar_widget = self._build_stellar_panel()
        self.stack.addWidget(self.stellar_widget)

        # CellSighter Subpanel (Index 5)
        self.cellsighter_widget = self._build_cellsighter_panel()
        self.stack.addWidget(self.cellsighter_widget)

        # Nimbus Subpanel (Index 6)
        self.nimbus_widget = self._build_nimbus_panel()
        self.stack.addWidget(self.nimbus_widget)

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
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # Model Configuration Box
        box_cfg, bl_cfg = create_section_card(
            "ASTIR Model Configuration",
            "Automated single-cell identification using PyTorch variational Bayes and marker-to-cell-type priors.",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        row_yaml = QHBoxLayout()
        row_yaml.addWidget(QLabel("Marker YAML:"))
        self.edit_astir_yaml = QLineEdit()
        self.edit_astir_yaml.setPlaceholderText("Optional YAML (e.g. markers.yml)")
        row_yaml.addWidget(self.edit_astir_yaml, 1)
        self.btn_load_astir_yaml = QPushButton("Load Marker YAML...")
        self.btn_load_astir_yaml.setMinimumHeight(28)
        self.btn_load_astir_yaml.clicked.connect(self._browse_astir_yaml)
        row_yaml.addWidget(self.btn_load_astir_yaml)
        bl_cfg.addLayout(row_yaml)

        self.lbl_astir_yaml_summary = QLabel(
            "Source: Gating hierarchy rules or detected marker panel"
        )
        self.lbl_astir_yaml_summary.setStyleSheet("color: #90CAF9; font-size: 11px;")
        bl_cfg.addWidget(self.lbl_astir_yaml_summary)

        row_epochs = QHBoxLayout()
        row_epochs.addWidget(QLabel("Max Epochs:"))
        self.spin_astir_epochs = QSpinBox()
        self.spin_astir_epochs.setRange(10, 5000)
        self.spin_astir_epochs.setValue(100)
        self.spin_astir_epochs.setSingleStep(25)
        row_epochs.addWidget(self.spin_astir_epochs)

        row_epochs.addWidget(QLabel("Learning Rate:"))
        self.spin_astir_lr = QDoubleSpinBox()
        self.spin_astir_lr.setRange(1e-5, 1e-1)
        self.spin_astir_lr.setValue(0.002)
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
        self.combo_astir_prior.addItems(
            ["Unsupervised Empirical", "Hierarchical Informed", "Marker Correlation"]
        )
        row_batch.addWidget(self.combo_astir_prior)
        bl_cfg.addLayout(row_batch)

        layout.addWidget(box_cfg)

        # Marker Association & Thresholding
        box_thresh, bl_thresh = create_section_card(
            "Marker Association & Quality Threshold",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

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
        box_exec, bl_exec = create_section_card(
            "Execution & Pipeline Status",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        self.btn_run_astir = QPushButton("Run ASTIR Phenotyper")
        self.btn_run_astir.setStyleSheet(BUTTON_CTA)
        self.btn_run_astir.setMinimumHeight(28)
        self.btn_run_astir.clicked.connect(self._run_astir)
        bl_exec.addWidget(self.btn_run_astir)

        self.lbl_astir_status = QLabel("Status: Ready to initialize ASTIR.")
        self.lbl_astir_status.setWordWrap(True)
        style_section_header(self.lbl_astir_status, subtitle=True)
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
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # MAPS MLP Configuration Box
        box_arch, bl_arch = create_section_card(
            "MAPS Feed-Forward MLP Architecture",
            "4-layer feed-forward MLP with z-score feature standardization and class-balanced sampling.",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Execution Mode:"))
        self.combo_maps_mode = QComboBox()
        self.combo_maps_mode.addItems([
            "Train from Reference Column",
            "Load Checkpoint (.pt)",
            "Marker-Rule Fallback",
        ])
        self.combo_maps_mode.currentIndexChanged.connect(self._on_maps_mode_changed)
        row_mode.addWidget(self.combo_maps_mode, 1)
        bl_arch.addLayout(row_mode)

        row_ref = QHBoxLayout()
        row_ref.addWidget(QLabel("Reference Label Column:"))
        self.combo_maps_ref_col = QComboBox()
        row_ref.addWidget(self.combo_maps_ref_col, 1)
        bl_arch.addLayout(row_ref)

        row_ckpt = QHBoxLayout()
        row_ckpt.addWidget(QLabel("Checkpoint (.pt):"))
        self.edit_maps_ckpt = QLineEdit()
        self.edit_maps_ckpt.setPlaceholderText("Path to trained MAPS .pt checkpoint...")
        row_ckpt.addWidget(self.edit_maps_ckpt, 1)
        self.btn_maps_browse = QPushButton("Browse...")
        self.btn_maps_browse.setMinimumHeight(28)
        self.btn_maps_browse.clicked.connect(self._browse_maps_ckpt)
        row_ckpt.addWidget(self.btn_maps_browse)
        bl_arch.addLayout(row_ckpt)

        layout.addWidget(box_arch)

        # Hyperparameters Box
        box_hp, bl_hp = create_section_card(
            "MLP Hyperparameters & Output",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        row_dim = QHBoxLayout()
        row_dim.addWidget(QLabel("Hidden Dim:"))
        self.spin_maps_hidden = QSpinBox()
        self.spin_maps_hidden.setRange(32, 2048)
        self.spin_maps_hidden.setValue(512)
        self.spin_maps_hidden.setSingleStep(64)
        row_dim.addWidget(self.spin_maps_hidden)

        row_dim.addWidget(QLabel("Dropout:"))
        self.spin_maps_dropout = QDoubleSpinBox()
        self.spin_maps_dropout.setRange(0.0, 0.80)
        self.spin_maps_dropout.setValue(0.10)
        self.spin_maps_dropout.setDecimals(2)
        self.spin_maps_dropout.setSingleStep(0.05)
        row_dim.addWidget(self.spin_maps_dropout)
        bl_hp.addLayout(row_dim)

        row_params = QHBoxLayout()
        row_params.addWidget(QLabel("Max Epochs:"))
        self.spin_maps_epochs = QSpinBox()
        self.spin_maps_epochs.setRange(1, 1000)
        self.spin_maps_epochs.setValue(50)
        row_params.addWidget(self.spin_maps_epochs)

        row_params.addWidget(QLabel("Learning Rate:"))
        self.spin_maps_lr = QDoubleSpinBox()
        self.spin_maps_lr.setRange(1e-5, 1e-1)
        self.spin_maps_lr.setValue(0.001)
        self.spin_maps_lr.setDecimals(5)
        self.spin_maps_lr.setSingleStep(0.0005)
        row_params.addWidget(self.spin_maps_lr)
        bl_hp.addLayout(row_params)

        row_batch = QHBoxLayout()
        row_batch.addWidget(QLabel("Batch Size:"))
        self.spin_maps_batch = QSpinBox()
        self.spin_maps_batch.setRange(8, 4096)
        self.spin_maps_batch.setValue(128)
        self.spin_maps_batch.setSingleStep(32)
        row_batch.addWidget(self.spin_maps_batch)

        row_batch.addWidget(QLabel("Output Column:"))
        self.txt_maps_col = QLineEdit("MAPS_CellType")
        row_batch.addWidget(self.txt_maps_col)
        bl_hp.addLayout(row_batch)

        layout.addWidget(box_hp)

        # Execution & Status
        box_exec, bl_exec = create_section_card(
            "Execution & Pipeline Status",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        self.btn_run_maps = QPushButton("Run MAPS Phenotyper")
        self.btn_run_maps.setStyleSheet(BUTTON_CTA)
        self.btn_run_maps.setMinimumHeight(28)
        self.btn_run_maps.clicked.connect(self._run_maps)
        bl_exec.addWidget(self.btn_run_maps)

        self.lbl_maps_status = QLabel("Status: Ready for MAPS MLP phenotyping.")
        self.lbl_maps_status.setWordWrap(True)
        style_section_header(self.lbl_maps_status, subtitle=True)
        bl_exec.addWidget(self.lbl_maps_status)

        layout.addWidget(box_exec)
        layout.addStretch(1)

        self._on_maps_mode_changed()

        scroll.setWidget(content)
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        p_lyt = QVBoxLayout(panel)
        p_lyt.setContentsMargins(0, 0, 0, 0)
        p_lyt.addWidget(scroll)
        return panel

    def _build_stellar_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # Reference & Target Mapping Box
        box_ref, bl_ref = create_section_card(
            "STELLAR Spatial Graph Label Transfer",
            "Transductive GraphSAGE encoder transferring annotations from spatial reference samples to target samples while discovering novel cell populations.",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        row_ref_col = QHBoxLayout()
        row_ref_col.addWidget(QLabel("Reference Label Column:"))
        self.combo_stellar_ref_col = QComboBox()
        row_ref_col.addWidget(self.combo_stellar_ref_col, 1)
        bl_ref.addLayout(row_ref_col)

        samples_row = QHBoxLayout()
        samples_row.setSpacing(8)

        ref_col_lyt = QVBoxLayout()
        ref_col_lyt.addWidget(QLabel("Reference Samples:"))
        self.list_stellar_ref_samples = QListWidget()
        self.list_stellar_ref_samples.setFixedHeight(78)
        ref_col_lyt.addWidget(self.list_stellar_ref_samples)
        samples_row.addLayout(ref_col_lyt)

        tgt_col_lyt = QVBoxLayout()
        tgt_col_lyt.addWidget(QLabel("Target Samples:"))
        self.list_stellar_target_samples = QListWidget()
        self.list_stellar_target_samples.setFixedHeight(78)
        tgt_col_lyt.addWidget(self.list_stellar_target_samples)
        samples_row.addLayout(tgt_col_lyt)
        bl_ref.addLayout(samples_row)

        layout.addWidget(box_ref)

        # Hyperparameters Box
        box_hp, bl_hp = create_section_card(
            "Spatial Graph & Novelty Parameters",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        row_g1 = QHBoxLayout()
        row_g1.addWidget(QLabel("Distance Threshold:"))
        self.spin_stellar_dist = QDoubleSpinBox()
        self.spin_stellar_dist.setRange(1.0, 1000.0)
        self.spin_stellar_dist.setValue(50.0)
        self.spin_stellar_dist.setDecimals(1)
        self.spin_stellar_dist.setSuffix(" µm")
        row_g1.addWidget(self.spin_stellar_dist)

        row_g1.addWidget(QLabel("Epochs:"))
        self.spin_stellar_epochs = QSpinBox()
        self.spin_stellar_epochs.setRange(1, 500)
        self.spin_stellar_epochs.setValue(20)
        row_g1.addWidget(self.spin_stellar_epochs)
        bl_hp.addLayout(row_g1)

        row_g2 = QHBoxLayout()
        row_g2.addWidget(QLabel("Learning Rate:"))
        self.spin_stellar_lr = QDoubleSpinBox()
        self.spin_stellar_lr.setRange(1e-5, 1e-1)
        self.spin_stellar_lr.setValue(0.001)
        self.spin_stellar_lr.setDecimals(5)
        self.spin_stellar_lr.setSingleStep(0.0005)
        row_g2.addWidget(self.spin_stellar_lr)

        row_g2.addWidget(QLabel("Num Seed Classes:"))
        self.spin_stellar_seed_classes = QSpinBox()
        self.spin_stellar_seed_classes.setRange(0, 50)
        self.spin_stellar_seed_classes.setValue(3)
        row_g2.addWidget(self.spin_stellar_seed_classes)
        bl_hp.addLayout(row_g2)

        row_g3 = QHBoxLayout()
        row_g3.addWidget(QLabel("Novelty Confidence Threshold:"))
        self.spin_stellar_novel_thresh = QDoubleSpinBox()
        self.spin_stellar_novel_thresh.setRange(0.0, 1.0)
        self.spin_stellar_novel_thresh.setValue(0.50)
        self.spin_stellar_novel_thresh.setDecimals(2)
        self.spin_stellar_novel_thresh.setSingleStep(0.05)
        row_g3.addWidget(self.spin_stellar_novel_thresh)

        row_g3.addWidget(QLabel("Output Column:"))
        self.txt_stellar_col = QLineEdit("STELLAR_CellType")
        row_g3.addWidget(self.txt_stellar_col)
        bl_hp.addLayout(row_g3)

        row_ckpt = QHBoxLayout()
        row_ckpt.addWidget(QLabel("Optional Checkpoint:"))
        self.edit_stellar_ckpt = QLineEdit()
        self.edit_stellar_ckpt.setPlaceholderText("Optional .pt weights path...")
        row_ckpt.addWidget(self.edit_stellar_ckpt, 1)
        self.btn_stellar_browse = QPushButton("Browse...")
        self.btn_stellar_browse.setMinimumHeight(28)
        self.btn_stellar_browse.clicked.connect(self._browse_stellar_ckpt)
        row_ckpt.addWidget(self.btn_stellar_browse)
        bl_hp.addLayout(row_ckpt)

        layout.addWidget(box_hp)

        # Execution
        box_exec, bl_exec = create_section_card(
            "Execution & Pipeline Status",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        self.btn_run_stellar = QPushButton("Run STELLAR Label Transfer")
        self.btn_run_stellar.setStyleSheet(BUTTON_CTA)
        self.btn_run_stellar.setMinimumHeight(28)
        self.btn_run_stellar.clicked.connect(self._run_stellar)
        bl_exec.addWidget(self.btn_run_stellar)

        self.lbl_stellar_status = QLabel("Status: Ready for spatial graph label transfer.")
        self.lbl_stellar_status.setWordWrap(True)
        style_section_header(self.lbl_stellar_status, subtitle=True)
        bl_exec.addWidget(self.lbl_stellar_status)

        layout.addWidget(box_exec)
        layout.addStretch(1)

        scroll.setWidget(content)
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        p_lyt = QVBoxLayout(panel)
        p_lyt.setContentsMargins(0, 0, 0, 0)
        p_lyt.addWidget(scroll)
        return panel

    def _build_cellsighter_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        box_cfg, bl_cfg = create_section_card(
            "CellSighter Image-Crop CNN Ensemble",
            "ResNet50 multi-channel image-crop CNN ensemble operating on multiplexed image crops and segmentation masks.",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Mode:"))
        self.combo_cs_mode = QComboBox()
        self.combo_cs_mode.addItems(["Load Ensemble (.pt)", "Train on Reference Column"])
        self.combo_cs_mode.currentIndexChanged.connect(self._on_cs_mode_changed)
        row_mode.addWidget(self.combo_cs_mode, 1)
        bl_cfg.addLayout(row_mode)

        row_ref = QHBoxLayout()
        row_ref.addWidget(QLabel("Reference Label Column:"))
        self.combo_cs_ref_col = QComboBox()
        row_ref.addWidget(self.combo_cs_ref_col, 1)
        bl_cfg.addLayout(row_ref)

        row_ckpt = QHBoxLayout()
        row_ckpt.addWidget(QLabel("Checkpoint (.pt):"))
        self.edit_cs_ckpt = QLineEdit()
        self.edit_cs_ckpt.setPlaceholderText("Path to CellSighter ensemble .pt...")
        row_ckpt.addWidget(self.edit_cs_ckpt, 1)
        self.btn_cs_browse = QPushButton("Browse...")
        self.btn_cs_browse.setMinimumHeight(28)
        self.btn_cs_browse.clicked.connect(self._browse_cs_ckpt)
        row_ckpt.addWidget(self.btn_cs_browse)
        bl_cfg.addLayout(row_ckpt)

        layout.addWidget(box_cfg)

        box_hp, bl_hp = create_section_card(
            "Crop & Training Settings",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        row_crop = QHBoxLayout()
        row_crop.addWidget(QLabel("Crop Size:"))
        self.spin_cs_crop_size = QSpinBox()
        self.spin_cs_crop_size.setRange(8, 512)
        self.spin_cs_crop_size.setValue(128)
        row_crop.addWidget(self.spin_cs_crop_size)

        row_crop.addWidget(QLabel("Input Size:"))
        self.spin_cs_crop_input_size = QSpinBox()
        self.spin_cs_crop_input_size.setRange(8, 512)
        self.spin_cs_crop_input_size.setValue(60)
        row_crop.addWidget(self.spin_cs_crop_input_size)
        bl_hp.addLayout(row_crop)

        row_norm = QHBoxLayout()
        row_norm.addWidget(QLabel("Normalization:"))
        self.combo_cs_norm = QComboBox()
        self.combo_cs_norm.addItems(["quantile", "anscombe"])
        row_norm.addWidget(self.combo_cs_norm)

        row_norm.addWidget(QLabel("Epochs:"))
        self.spin_cs_epochs = QSpinBox()
        self.spin_cs_epochs.setRange(1, 500)
        self.spin_cs_epochs.setValue(15)
        row_norm.addWidget(self.spin_cs_epochs)
        bl_hp.addLayout(row_norm)

        row_lr = QHBoxLayout()
        row_lr.addWidget(QLabel("Learning Rate:"))
        self.spin_cs_lr = QDoubleSpinBox()
        self.spin_cs_lr.setRange(1e-6, 1e-1)
        self.spin_cs_lr.setValue(1e-4)
        self.spin_cs_lr.setDecimals(6)
        self.spin_cs_lr.setSingleStep(1e-4)
        row_lr.addWidget(self.spin_cs_lr)

        row_lr.addWidget(QLabel("Output Column:"))
        self.txt_cs_col = QLineEdit("CellSighter_CellType")
        row_lr.addWidget(self.txt_cs_col)
        bl_hp.addLayout(row_lr)

        layout.addWidget(box_hp)

        box_exec, bl_exec = create_section_card(
            "Execution & Pipeline Status",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        self.btn_run_cs = QPushButton("Run CellSighter Phenotyper")
        self.btn_run_cs.setStyleSheet(BUTTON_CTA)
        self.btn_run_cs.setMinimumHeight(28)
        self.btn_run_cs.clicked.connect(self._run_cellsighter)
        bl_exec.addWidget(self.btn_run_cs)

        self.lbl_cs_status = QLabel("Status: Ready for CellSighter crop phenotyping.")
        self.lbl_cs_status.setWordWrap(True)
        style_section_header(self.lbl_cs_status, subtitle=True)
        bl_exec.addWidget(self.lbl_cs_status)

        layout.addWidget(box_exec)
        layout.addStretch(1)

        self._on_cs_mode_changed()

        scroll.setWidget(content)
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        p_lyt = QVBoxLayout(panel)
        p_lyt.setContentsMargins(0, 0, 0, 0)
        p_lyt.addWidget(scroll)
        return panel

    def _build_nimbus_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        box_cfg, bl_cfg = create_section_card(
            "Nimbus Zero-Shot Residual U-Net",
            "Pretrained Residual U-Net predicting calibrated per-cell marker positivity scores from multiplexed images and masks.",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        default_ckpt = ""
        if _DEFAULT_NIMBUS_CHECKPOINT.is_file():
            default_ckpt = str(_DEFAULT_NIMBUS_CHECKPOINT)
        elif _FALLBACK_NIMBUS_CHECKPOINT.is_file():
            default_ckpt = str(_FALLBACK_NIMBUS_CHECKPOINT)

        row_ckpt = QHBoxLayout()
        row_ckpt.addWidget(QLabel("Checkpoint (.pt):"))
        self.edit_nimbus_ckpt = QLineEdit(default_ckpt)
        self.edit_nimbus_ckpt.setPlaceholderText("Path to Nimbus V1.pt checkpoint or 'latest'")
        row_ckpt.addWidget(self.edit_nimbus_ckpt, 1)
        self.btn_nimbus_browse = QPushButton("Browse...")
        self.btn_nimbus_browse.setMinimumHeight(28)
        self.btn_nimbus_browse.clicked.connect(self._browse_nimbus_ckpt)
        row_ckpt.addWidget(self.btn_nimbus_browse)
        bl_cfg.addLayout(row_ckpt)

        row_q = QHBoxLayout()
        row_q.addWidget(QLabel("Quantile:"))
        self.spin_nimbus_quantile = QDoubleSpinBox()
        self.spin_nimbus_quantile.setRange(0.500, 1.000)
        self.spin_nimbus_quantile.setValue(0.999)
        self.spin_nimbus_quantile.setDecimals(3)
        self.spin_nimbus_quantile.setSingleStep(0.001)
        row_q.addWidget(self.spin_nimbus_quantile)

        row_q.addWidget(QLabel("Positivity Threshold:"))
        self.spin_nimbus_pos_thresh = QDoubleSpinBox()
        self.spin_nimbus_pos_thresh.setRange(0.05, 0.95)
        self.spin_nimbus_pos_thresh.setValue(0.50)
        self.spin_nimbus_pos_thresh.setDecimals(2)
        self.spin_nimbus_pos_thresh.setSingleStep(0.05)
        row_q.addWidget(self.spin_nimbus_pos_thresh)
        bl_cfg.addLayout(row_q)

        self.chk_nimbus_tta = QCheckBox("Test-Time Augmentation (6-view TTA)")
        self.chk_nimbus_tta.setChecked(True)
        bl_cfg.addWidget(self.chk_nimbus_tta)

        layout.addWidget(box_cfg)

        box_ds, bl_ds = create_section_card(
            "Downstream Cell-Type Calling & Output",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Downstream Mode:"))
        self.combo_nimbus_mode = QComboBox()
        self.combo_nimbus_mode.addItems(["Panel Marker Rules", "Consensus Clustering"])
        self.combo_nimbus_mode.currentIndexChanged.connect(self._on_nimbus_mode_changed)
        row_mode.addWidget(self.combo_nimbus_mode, 1)
        bl_ds.addLayout(row_mode)

        row_k = QHBoxLayout()
        row_k.addWidget(QLabel("n_clusters:"))
        self.spin_nimbus_clusters = QSpinBox()
        self.spin_nimbus_clusters.setRange(2, 60)
        self.spin_nimbus_clusters.setValue(12)
        row_k.addWidget(self.spin_nimbus_clusters)

        row_k.addWidget(QLabel("Output Column:"))
        self.txt_nimbus_col = QLineEdit("Nimbus_CellType")
        row_k.addWidget(self.txt_nimbus_col)
        bl_ds.addLayout(row_k)

        layout.addWidget(box_ds)

        box_exec, bl_exec = create_section_card(
            "Execution & Pipeline Status",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        self.btn_run_nimbus = QPushButton("Run Nimbus Phenotyper")
        self.btn_run_nimbus.setStyleSheet(BUTTON_CTA)
        self.btn_run_nimbus.setMinimumHeight(28)
        self.btn_run_nimbus.clicked.connect(self._run_nimbus)
        bl_exec.addWidget(self.btn_run_nimbus)

        self.lbl_nimbus_status = QLabel(
            "Status: Ready for Nimbus zero-shot positivity inference."
        )
        self.lbl_nimbus_status.setWordWrap(True)
        style_section_header(self.lbl_nimbus_status, subtitle=True)
        bl_exec.addWidget(self.lbl_nimbus_status)

        layout.addWidget(box_exec)
        layout.addStretch(1)

        self._on_nimbus_mode_changed()

        scroll.setWidget(content)
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        p_lyt = QVBoxLayout(panel)
        p_lyt.setContentsMargins(0, 0, 0, 0)
        p_lyt.addWidget(scroll)
        return panel

    def _on_maps_mode_changed(self) -> None:
        if not hasattr(self, "combo_maps_mode"):
            return
        mode = self.combo_maps_mode.currentText()
        is_train = mode == "Train from Reference Column"
        is_ckpt = mode == "Load Checkpoint (.pt)"
        self.combo_maps_ref_col.setEnabled(is_train)
        self.edit_maps_ckpt.setEnabled(is_ckpt)
        self.btn_maps_browse.setEnabled(is_ckpt)
        self.spin_maps_hidden.setEnabled(is_train)
        self.spin_maps_dropout.setEnabled(is_train)
        self.spin_maps_epochs.setEnabled(is_train)
        self.spin_maps_lr.setEnabled(is_train)
        self.spin_maps_batch.setEnabled(is_train or is_ckpt)

    def _on_cs_mode_changed(self) -> None:
        if not hasattr(self, "combo_cs_mode"):
            return
        is_train = self.combo_cs_mode.currentText() == "Train on Reference Column"
        self.combo_cs_ref_col.setEnabled(is_train)
        self.spin_cs_epochs.setEnabled(is_train)
        self.spin_cs_lr.setEnabled(is_train)
        self.edit_cs_ckpt.setEnabled(not is_train)
        self.btn_cs_browse.setEnabled(not is_train)

    def _on_nimbus_mode_changed(self) -> None:
        if not hasattr(self, "combo_nimbus_mode"):
            return
        is_cluster = self.combo_nimbus_mode.currentText() == "Consensus Clustering"
        self.spin_nimbus_clusters.setEnabled(is_cluster)

    def _browse_astir_yaml(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select ASTIR Marker YAML",
            "",
            "YAML Files (*.yml *.yaml);;All Files (*)",
        )
        if filepath:
            try:
                self.load_astir_yaml(filepath)
            except Exception as exc:
                self.lbl_astir_status.setText(f"Failed to load marker YAML: {str(exc)[:60]}")
                self.lbl_astir_status.setStyleSheet("color: #E57373; font-size: 11px;")

    def load_astir_yaml(self, path: str):
        """Parse an ASTIR marker YAML file (supporting cell_types, cell_states, hierarchy)."""
        from iris_phenotyper.astir.pipeline import normalize_marker_config
        from iris_viewer.core.types import CellTypePanel, CellTypeRule

        yaml_path = Path(path)
        ct_dict, state_dict, hierarchy_dict = normalize_marker_config(yaml_path)
        if not ct_dict:
            raise ValueError(f"No 'cell_types' marker entries found in {yaml_path.name}.")

        parent_map: Dict[str, str] = {}
        if isinstance(hierarchy_dict, dict):
            for parent_name, children in hierarchy_dict.items():
                if isinstance(children, (list, tuple, set)):
                    for child in children:
                        parent_map[str(child)] = str(parent_name)

        rules = [
            CellTypeRule(
                cell_type=str(ct),
                markers_positive=[str(m) for m in markers],
                parent=parent_map.get(str(ct)),
            )
            for ct, markers in ct_dict.items()
            if markers
        ]
        panel = CellTypePanel(rules=rules, panel_file=yaml_path)
        cfg_payload: Dict[str, Any] = {"cell_types": ct_dict}
        if state_dict:
            cfg_payload["cell_states"] = state_dict
        if hierarchy_dict:
            cfg_payload["hierarchy"] = hierarchy_dict

        self._astir_yaml_path = str(yaml_path)
        self._astir_yaml_panel = panel
        self._astir_marker_config = cfg_payload
        self.edit_astir_yaml.setText(str(yaml_path))
        n_states = len(state_dict) if state_dict else 0
        self.lbl_astir_yaml_summary.setText(
            f"Loaded {len(rules)} cell types, {n_states} cell states from {yaml_path.name}"
        )
        return panel

    def _browse_maps_ckpt(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select MAPS Checkpoint",
            "",
            "PyTorch Checkpoints (*.pt *.pth);;All Files (*)",
        )
        if filepath:
            self.edit_maps_ckpt.setText(filepath)

    def _browse_stellar_ckpt(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select STELLAR Checkpoint",
            "",
            "PyTorch Checkpoints (*.pt *.pth);;All Files (*)",
        )
        if filepath:
            self.edit_stellar_ckpt.setText(filepath)

    def _browse_cs_ckpt(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select CellSighter Ensemble Checkpoint",
            "",
            "PyTorch Checkpoints (*.pt *.pth);;All Files (*)",
        )
        if filepath:
            self.edit_cs_ckpt.setText(filepath)

    def _browse_nimbus_ckpt(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Nimbus Checkpoint",
            "",
            "PyTorch Checkpoints (*.pt *.pth);;All Files (*)",
        )
        if filepath:
            self.edit_nimbus_ckpt.setText(filepath)

    def open_diagnostics_dialog(self) -> PhenotyperDiagnosticDialog:
        """Open or refresh the pop-up Phenotyping Diagnostics & Concordance dialog."""
        raw_df = (
            self.manual_gating.get_active_dataframe()
            if hasattr(self, "manual_gating")
            else None
        )
        if self.diagnostics_dialog is None:
            parent_win = self.window() if hasattr(self, "window") else None
            self.diagnostics_dialog = PhenotyperDiagnosticDialog(
                df=raw_df,
                ctx=self.ctx,
                parent=parent_win,
            )
        else:
            self.diagnostics_dialog.ctx = self.ctx
            self.diagnostics_dialog.refresh_data(raw_df)
        self.diagnostics_dialog.show()
        self.diagnostics_dialog.raise_()
        self.diagnostics_dialog.activateWindow()
        return self.diagnostics_dialog

    open_popup_window = open_diagnostics_dialog

    def _build_clustering_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # Algorithm Box
        box_alg, bl_alg = create_section_card(
            "Unsupervised Clustering Algorithm",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        # Sample Cohort Scope
        row_scope = QHBoxLayout()
        row_scope.addWidget(QLabel("Sample Scope:"))
        self.combo_cluster_scope = _ScopeComboBox()
        self.combo_cluster_scope.addItems([
            SCOPE_ALL,
            SCOPE_SELECTED,
            SCOPE_ACTIVE,
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
        self.spin_pheno_k.setToolTip("Number of nearest neighbors (k) for Jaccard k-NN graph construction")
        row_p.addWidget(self.spin_pheno_k)
        row_p.addWidget(QLabel("Min Cluster Size:"))
        self.spin_pheno_min_size = QSpinBox()
        self.spin_pheno_min_size.setRange(5, 500)
        self.spin_pheno_min_size.setValue(10)
        self.spin_pheno_min_size.setToolTip("Minimum number of cells required to form a retained community/cluster.")
        row_p.addWidget(self.spin_pheno_min_size)
        p_lyt.addLayout(row_p)
        lbl_p_hint = QLabel("Jaccard k-NN graph modularity clustering. Smaller k yields finer clusters.")
        lbl_p_hint.setStyleSheet("color: #9AA4B2; font-size: 10px; font-style: italic;")
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
        lbl_k_hint.setStyleSheet("color: #9AA4B2; font-size: 10px; font-style: italic;")
        k_lyt.addWidget(lbl_k_hint)
        self.stack_cluster_params.addWidget(page_kmeans)

        self.combo_cluster_method.currentIndexChanged.connect(self.stack_cluster_params.setCurrentIndex)
        bl_alg.addWidget(self.stack_cluster_params)

        self.btn_run_cluster = QPushButton("Run Unsupervised Clustering")
        self.btn_run_cluster.setStyleSheet(BUTTON_CTA)
        self.btn_run_cluster.setMinimumHeight(28)
        self.btn_run_cluster.clicked.connect(self._run_clustering)
        bl_alg.addWidget(self.btn_run_cluster)

        self.lbl_cluster_status = QLabel("Status: Idle. Click Run to perform clustering.")
        style_section_header(self.lbl_cluster_status, subtitle=True)
        bl_alg.addWidget(self.lbl_cluster_status)

        layout.addWidget(box_alg)

        # Cluster Selection & Exploration
        box_sel, bl_sel = create_section_card(
            "Cluster Selection & Expression Heatmap",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        self.list_clusters = QListWidget()
        self.list_clusters.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_clusters.setFixedHeight(120)
        self.list_clusters.itemSelectionChanged.connect(self._on_cluster_selection_changed)
        bl_sel.addWidget(self.list_clusters)

        # Profile Card for Selected Cluster
        self.card_cluster_profile = QFrame()
        self.card_cluster_profile.setStyleSheet("""
            QFrame {
                background-color: #171a21; border: 1px solid #3c4352; border-radius: 4px;
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
        self.btn_cluster_heatmap.setStyleSheet(POPOUT_BUTTON_STYLE)
        self.btn_cluster_heatmap.setMinimumHeight(28)
        self.btn_cluster_heatmap.clicked.connect(self._show_cluster_heatmap)
        btn_row_c.addWidget(self.btn_cluster_heatmap)
        bl_sel.addLayout(btn_row_c)

        layout.addWidget(box_sel)

        # Phenotype Assignment
        box_assign, bl_assign = create_section_card(
            "Assign Clusters to Phenotype",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        row_name = QHBoxLayout()
        row_name.addWidget(QLabel("Label Name:"))
        self.txt_cluster_pheno = QLineEdit()
        self.txt_cluster_pheno.setPlaceholderText("e.g. CD8+ Effector T cells")
        row_name.addWidget(self.txt_cluster_pheno)
        bl_assign.addLayout(row_name)

        self.btn_assign_cluster = QPushButton("Apply Phenotype to Selected Clusters")
        self.btn_assign_cluster.setStyleSheet(BUTTON_CTA)
        self.btn_assign_cluster.setMinimumHeight(28)
        self.btn_assign_cluster.clicked.connect(self._apply_cluster_phenotype)
        bl_assign.addWidget(self.btn_assign_cluster)

        self.lbl_cluster_assign_feedback = QLabel("")
        self.lbl_cluster_assign_feedback.setStyleSheet("color: #66BB6A; font-weight: bold; font-size: 11px;")
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

    def _should_run_async(self) -> bool:
        return (
            self.ctx is not None
            and getattr(self.ctx, "track_worker", None) is not None
        )

    def _dispatch_heavy_task(
        self,
        fn,
        on_done,
        on_error,
        error_title: str,
        error_prefix: str,
    ) -> None:
        """Run *fn* off the main thread via thread_worker when ctx.track_worker exists."""
        if self._should_run_async():
            from napari.qt import thread_worker

            @thread_worker
            def _worker():
                return fn()

            worker = _worker()
            worker.returned.connect(on_done)
            self.ctx.track_worker(
                worker,
                on_error=on_error,
                error_title=error_title,
                error_prefix=error_prefix,
            )
            self._active_worker = worker
            worker.start()
        else:
            try:
                result = fn()
            except Exception as exc:
                on_error(exc)
                return
            on_done(result)

    def _resolve_panel_markers(self, raw_df: pd.DataFrame) -> List[str]:
        """Resolve clean marker names present in *raw_df* directly or via suffix."""
        avail = (
            getattr(self.manual_gating, "available_markers", [])
            if hasattr(self, "manual_gating")
            else []
        )
        markers = [m for m in avail if resolve_marker_col(raw_df, m) is not None]
        if not markers:
            mean_cols = [
                str(c)
                for c in raw_df.columns
                if str(c).endswith("_mean")
                and pd.api.types.is_numeric_dtype(raw_df[c])
            ]
            if mean_cols:
                markers = [c[:-5] for c in mean_cols]
            else:
                markers = [
                    str(c)
                    for c in raw_df.select_dtypes(include=[np.number]).columns
                    if not str(c)
                    .lower()
                    .startswith(("x", "y", "z", "area", "id", "cell", "object", "pos"))
                    and not str(c).lower().endswith((
                        "_confidence",
                        "_isnovel",
                        "_nimbus_positivity",
                    ))
                    and str(c).lower() not in {"confidence", "is_novel", "cluster"}
                ]
        return markers

    @staticmethod
    def _build_backend_feature_frame(
        raw_df: pd.DataFrame,
        markers: List[str],
    ) -> pd.DataFrame:
        """Build a thread-safe DataFrame copy with canonical `<marker>_mean` aliases."""
        df_copy = raw_df.copy()
        for m in markers:
            col = resolve_marker_col(df_copy, m)
            if col is not None and f"{m}_mean" not in df_copy.columns:
                df_copy[f"{m}_mean"] = df_copy[col]
        return df_copy

    @staticmethod
    def _discover_reference_columns(raw_df: Optional[pd.DataFrame]) -> List[str]:
        if raw_df is None or raw_df.empty:
            return []
        priority = [
            "cell_type",
            "CellType",
            "Verified_CellType",
            "Cluster",
            "ASTIR_CellType",
            "MAPS_CellType",
            "STELLAR_CellType",
            "CellSighter_CellType",
            "Nimbus_CellType",
        ]
        skip = {
            "x",
            "y",
            "z",
            "pos_x",
            "pos_y",
            "x_centroid",
            "y_centroid",
            "centroid_x",
            "centroid_y",
            "area",
            "eccentricity",
            "perimeter",
            "solidity",
            "objectnumber",
            "cell_id",
            "id",
            "image",
            "sample",
            "sample_name",
            "sample_id",
        }
        cols: List[str] = [c for c in priority if c in raw_df.columns]
        for col in raw_df.columns:
            c_str = str(col)
            c_low = c_str.lower()
            if (
                c_str in cols
                or c_low in skip
                or c_low.endswith(("_mean", "_median", "_confidence", "_isnovel", "_nimbus_positivity"))
            ):
                continue
            ser = raw_df[col]
            if (
                ser.dtype == object
                or ser.dtype.name in ("category", "string")
                or (pd.api.types.is_integer_dtype(ser) and ser.nunique() <= 60)
            ):
                cols.append(c_str)
        return cols

    def _sync_reference_columns(self, raw_df: Optional[pd.DataFrame] = None) -> None:
        df = raw_df
        if df is None and hasattr(self, "manual_gating"):
            df = self.manual_gating.get_active_dataframe()
        cols = self._discover_reference_columns(df)
        for combo_name in ("combo_maps_ref_col", "combo_stellar_ref_col", "combo_cs_ref_col"):
            combo = getattr(self, combo_name, None)
            if combo is None:
                continue
            curr = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(cols)
            if curr in cols:
                combo.setCurrentText(curr)
            elif cols:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

    def _sync_stellar_samples(self, raw_df: Optional[pd.DataFrame] = None) -> None:
        if not hasattr(self, "list_stellar_ref_samples"):
            return
        df = raw_df
        if df is None and hasattr(self, "manual_gating"):
            df = self.manual_gating.get_active_dataframe()

        samples: List[str] = []
        if df is not None and not df.empty:
            scol = find_sample_column(df)
            if scol is not None:
                samples = [str(s).strip() for s in df[scol].dropna().unique()]
        if not samples and self.ctx and getattr(self.ctx, "state", None) is not None:
            state = self.ctx.state
            samples = (
                list(state.sample_layers_map.keys())
                or list(state.multi_sample_images.keys())
                or list(state.file_map.keys())
            )

        prev_ref = {
            self.list_stellar_ref_samples.item(i).text().strip()
            for i in range(self.list_stellar_ref_samples.count())
            if self.list_stellar_ref_samples.item(i).checkState() == Qt.Checked
        }
        prev_tgt = {
            self.list_stellar_target_samples.item(i).text().strip()
            for i in range(self.list_stellar_target_samples.count())
            if self.list_stellar_target_samples.item(i).checkState() == Qt.Checked
        }
        existing_all = {
            self.list_stellar_ref_samples.item(i).text().strip()
            for i in range(self.list_stellar_ref_samples.count())
        }
        same_samples = bool(samples) and existing_all == set(samples)

        self.list_stellar_ref_samples.clear()
        self.list_stellar_target_samples.clear()
        for idx, s in enumerate(samples):
            ref_item = QListWidgetItem(s)
            ref_item.setFlags(ref_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            if same_samples:
                ref_checked = s in prev_ref
            elif len(samples) >= 2:
                ref_checked = idx == 0
            else:
                ref_checked = True
            ref_item.setCheckState(Qt.Checked if ref_checked else Qt.Unchecked)
            self.list_stellar_ref_samples.addItem(ref_item)

            tgt_item = QListWidgetItem(s)
            tgt_item.setFlags(tgt_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            if same_samples:
                tgt_checked = s in prev_tgt
            elif len(samples) >= 2:
                tgt_checked = idx > 0
            else:
                tgt_checked = True
            tgt_item.setCheckState(Qt.Checked if tgt_checked else Qt.Unchecked)
            self.list_stellar_target_samples.addItem(tgt_item)

    @staticmethod
    def _checked_list_items(lw: QListWidget) -> List[str]:
        return [
            lw.item(i).text().strip()
            for i in range(lw.count())
            if lw.item(i).checkState() == Qt.Checked
        ]

    def _propagate_prediction_columns(
        self,
        raw_df: pd.DataFrame,
        columns: Sequence[str],
        *,
        primary_col: str,
        unique_types: Optional[Sequence[str]] = None,
        markers: Optional[Sequence[str]] = None,
        palette: Optional[Sequence[str]] = None,
    ) -> None:
        """Propagate prediction columns across ViewerAppState, combo_ref, and manual_gating."""
        cols = [c for c in columns if c in raw_df.columns]
        if self.ctx is not None:
            state = getattr(self.ctx, "state", None)
            if state is not None:
                for attr in ("master_df", "features_df", "current_mapped_df"):
                    target_df = getattr(state, attr, None)
                    if (
                        target_df is not None
                        and target_df is not raw_df
                        and len(target_df) == len(raw_df)
                    ):
                        for c in cols:
                            target_df[c] = raw_df[c].to_numpy()
                if getattr(state, "resolved_dfs", None):
                    scol_name = find_sample_column(raw_df)
                    for sname, rdf in list(state.resolved_dfs.items()):
                        if rdf is raw_df:
                            continue
                        if scol_name is not None:
                            m_s = (
                                raw_df[scol_name].astype(str).str.strip() == str(sname).strip()
                            )
                            if m_s.any() and len(rdf) == int(m_s.sum()):
                                for c in cols:
                                    rdf[c] = raw_df.loc[m_s, c].to_numpy()
                        elif len(state.resolved_dfs) == 1 and len(rdf) == len(raw_df):
                            for c in cols:
                                rdf[c] = raw_df[c].to_numpy()

            cw = getattr(self.ctx, "control_widget", None)
            if cw is not None and hasattr(cw, "combo_ref"):
                cb = cw.combo_ref
                if cb.findText(primary_col) == -1:
                    cb.addItem(primary_col)
                cb.blockSignals(True)
                cb.setCurrentText(primary_col)
                cb.blockSignals(False)

        if hasattr(self, "manual_gating") and unique_types:
            pal = list(
                palette
                or [
                    "#00E5FF",
                    "#FF0055",
                    "#66BB6A",
                    "#FFD600",
                    "#D500F9",
                    "#FF6D00",
                    "#00B0FF",
                    "#00E676",
                    "#FF1744",
                    "#651FFF",
                ]
            )
            n_total = max(1, len(raw_df))
            marker_tuple = tuple(markers or ())
            for idx, ct_name in enumerate(unique_types):
                m_pos = raw_df[primary_col] == ct_name
                match_cnt = int(m_pos.sum())
                self.manual_gating.on_gate_applied(
                    {
                        "label": ct_name,
                        "color": pal[idx % len(pal)],
                        "target_col": primary_col,
                        "count": match_cnt,
                        "pct": (match_cnt / n_total) * 100.0,
                        "markers": marker_tuple,
                        "mask": m_pos,
                    }
                )
        elif (
            self.ctx is not None
            and getattr(self.ctx, "marker_widget", None) is not None
        ):
            try:
                self.ctx.marker_widget.set_data(
                    df=raw_df,
                    markers=list(markers or ()),
                    celltype_col=primary_col,
                )
            except Exception:
                pass

        self._sync_reference_columns(raw_df)
        if self.diagnostics_dialog is not None and self.diagnostics_dialog.isVisible():
            try:
                self.diagnostics_dialog.refresh_data(raw_df)
            except Exception:
                pass

    @staticmethod
    def _sample_names_match(a: Optional[str], b: Optional[str]) -> bool:
        if a is None or b is None:
            return False
        sa = str(a).strip()
        sb = str(b).strip()
        if not sa or not sb:
            return False
        if sa == sb:
            return True
        na = _normalize_sample_name(sa)
        nb = _normalize_sample_name(sb)
        return bool(na and nb and na == nb)

    def _collect_samples_from_state(
        self,
        raw_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[List[str], List[str]]:
        """Collect deduplicated (all_samples, active_samples) from *raw_df* and *ctx.state*."""
        all_list: List[str] = []
        seen_norm: set[str] = set()

        def _add_candidate(val: Any) -> None:
            if val is None:
                return
            text = str(val).strip()
            if not text:
                return
            norm = _normalize_sample_name(text) or text
            if text not in all_list and norm not in seen_norm:
                all_list.append(text)
                seen_norm.add(norm)

        if raw_df is not None and not raw_df.empty:
            scol = find_sample_column(raw_df)
            if scol is not None:
                for val in raw_df[scol].dropna().unique():
                    _add_candidate(val)

        active_list: List[str] = []
        state = getattr(self.ctx, "state", None) if self.ctx is not None else None
        if state is not None:
            smap = getattr(state, "sample_layers_map", None) or {}
            mimgs = getattr(state, "multi_sample_images", None) or {}
            fmap = getattr(state, "file_map", None) or {}
            for k in smap.keys():
                _add_candidate(k)
            for k in mimgs.keys():
                _add_candidate(k)
            for k in fmap.keys():
                _add_candidate(k)

            if smap:
                active_list = [str(k).strip() for k in smap.keys() if str(k).strip()]
            elif getattr(state, "current_sample_name", None):
                cur = str(state.current_sample_name).strip()
                if cur:
                    _add_candidate(cur)
                    active_list = [cur]

        return all_list, active_list

    def _sync_summary_label(self, raw_df: Optional[pd.DataFrame] = None) -> None:
        if not hasattr(self, "lbl_summary"):
            return
        df = raw_df
        if df is None and hasattr(self, "manual_gating"):
            df = self.manual_gating.get_active_dataframe()
        h_cnt = (
            len(self.manual_gating.hierarchy_root.get_all_nodes()) - 1
            if hasattr(self, "manual_gating")
            else 0
        )
        mode = (
            self.sample_scope_bar.get_scope_mode()
            if hasattr(self, "sample_scope_bar")
            else SCOPE_ALL
        )
        resolved = (
            self.sample_scope_bar.resolve_samples()
            if hasattr(self, "sample_scope_bar")
            else []
        )
        if df is not None and not df.empty:
            self.lbl_summary.setText(
                f"Gating: Ready | Hierarchy: {h_cnt} gates | "
                f"Dataset: {len(df):,} cells | Scope: {mode} ({len(resolved)})"
            )
        else:
            self.lbl_summary.setText(
                f"Gating: Ready | Hierarchy: {h_cnt} gates | Scope: {mode}"
            )

    def sync_sample_scope(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """Synchronize ``self.sample_scope_bar`` and ``ManualGatingWidget`` scope."""
        raw_df = df
        if raw_df is None and hasattr(self, "manual_gating"):
            raw_df = self.manual_gating.get_active_dataframe()
        all_samples, active_samples = self._collect_samples_from_state(raw_df)
        if hasattr(self, "sample_scope_bar"):
            self.sample_scope_bar.set_samples(all_samples, active_samples, emit=False)
            resolved = self.sample_scope_bar.resolve_samples()
            mode = self.sample_scope_bar.get_scope_mode()
            if hasattr(self, "manual_gating") and not self._syncing_scope:
                self._syncing_scope = True
                try:
                    self.manual_gating.set_scope_samples(mode, resolved, emit=False)
                finally:
                    self._syncing_scope = False
            self._sync_summary_label(raw_df)
            return resolved
        return []

    def _on_sample_scope_changed(self, mode: str, resolved_samples: List[str]) -> None:
        if self._syncing_scope:
            return
        self._syncing_scope = True
        try:
            if hasattr(self, "manual_gating"):
                self.manual_gating.set_scope_samples(mode, resolved_samples, emit=False)
            if hasattr(self, "combo_cluster_scope"):
                if self.combo_cluster_scope.currentText() != "Custom Sample Checklist":
                    self.combo_cluster_scope.blockSignals(True)
                    self.combo_cluster_scope.setCurrentText(mode)
                    self.combo_cluster_scope.blockSignals(False)
                    self.box_cluster_checklist.setVisible(False)
            if hasattr(self, "list_stellar_target_samples") and mode != SCOPE_ALL:
                for i in range(self.list_stellar_target_samples.count()):
                    it = self.list_stellar_target_samples.item(i)
                    s_name = it.text().strip()
                    matched = any(
                        self._sample_names_match(s_name, t) for t in resolved_samples
                    )
                    it.setCheckState(Qt.Checked if matched else Qt.Unchecked)
            self._sync_summary_label()
            iris_log(
                "IRIS-Phenotyper",
                "Scope",
                f"Sample scope updated -> '{mode}' ({len(resolved_samples)} sample(s)).",
            )
        finally:
            self._syncing_scope = False

    def _on_manual_gating_scope_changed(self, scope_text: str) -> None:
        if self._syncing_scope or not hasattr(self, "sample_scope_bar"):
            return
        self._syncing_scope = True
        try:
            if scope_text.startswith("Selected"):
                self.sample_scope_bar.set_scope_mode(SCOPE_SELECTED, emit=False)
                sel = self.manual_gating._get_selected_samples()
                if sel:
                    self.sample_scope_bar.set_checked_samples(sel, emit=False)
            elif scope_text.startswith("Active"):
                self.sample_scope_bar.set_scope_mode(SCOPE_ACTIVE, emit=False)
            else:
                self.sample_scope_bar.set_scope_mode(SCOPE_ALL, emit=False)
            self._sync_summary_label()
        finally:
            self._syncing_scope = False

    def _resolve_scoped_rows_and_df(
        self,
        raw_df: Optional[pd.DataFrame],
    ) -> Tuple[np.ndarray, Optional[pd.DataFrame], str, List[str]]:
        """Resolve ``(row_mask, scoped_df, scope_desc, resolved_samples)`` via ``sample_scope_bar``."""
        if raw_df is None or raw_df.empty:
            return np.zeros(0, dtype=bool), raw_df, "no data", []

        if hasattr(self, "sample_scope_bar"):
            all_samples, active_samples = self._collect_samples_from_state(raw_df)
            if not self.sample_scope_bar._all_samples and all_samples:
                self.sample_scope_bar.set_samples(all_samples, active_samples, emit=False)
            else:
                self.sample_scope_bar.set_active_samples(active_samples, emit=False)

            mode = self.sample_scope_bar.get_scope_mode()
            resolved_samples = self.sample_scope_bar.resolve_samples()

            if mode == SCOPE_ALL:
                row_mask = np.ones(len(raw_df), dtype=bool)
                scoped_df = self.sample_scope_bar.filter_dataframe(raw_df)
                scope_desc = (
                    f"all {len(resolved_samples)} sample(s)"
                    if resolved_samples
                    else "all samples"
                )
                return row_mask, scoped_df, scope_desc, resolved_samples

            scol = find_sample_column(raw_df)
            if scol is not None:
                tmp_df = raw_df[[scol]].copy()
                tmp_df.index = pd.RangeIndex(len(raw_df))
                filtered_idx = self.sample_scope_bar.filter_dataframe(
                    tmp_df, sample_col=scol
                ).index
                row_mask = np.zeros(len(raw_df), dtype=bool)
                if len(filtered_idx) > 0:
                    row_mask[filtered_idx.to_numpy(dtype=int)] = True
                scoped_df = raw_df.iloc[row_mask].copy()
            elif mode == SCOPE_SELECTED and not resolved_samples:
                row_mask = np.zeros(len(raw_df), dtype=bool)
                scoped_df = raw_df.iloc[0:0].copy()
            else:
                row_mask = np.ones(len(raw_df), dtype=bool)
                scoped_df = raw_df.copy()

            if mode == SCOPE_ACTIVE:
                scope_desc = (
                    f"active ({', '.join(resolved_samples)})"
                    if resolved_samples
                    else "active (none)"
                )
            else:
                preview = ", ".join(resolved_samples[:3])
                if len(resolved_samples) > 3:
                    preview += ", ..."
                scope_desc = (
                    f"{len(resolved_samples)} selected ({preview})"
                    if resolved_samples
                    else "0 selected"
                )
            return row_mask, scoped_df, scope_desc, resolved_samples

        return np.ones(len(raw_df), dtype=bool), raw_df.copy(), "all samples", []

    @staticmethod
    def _assign_scoped_column(
        raw_df: pd.DataFrame,
        row_mask: np.ndarray,
        col: str,
        values: Any,
        *,
        default_value: Any,
    ) -> None:
        """Write scoped predictions to ``raw_df.loc[row_mask, col]`` cleanly."""
        if col not in raw_df.columns:
            raw_df[col] = default_value
        elif raw_df[col].dtype.name == "category":
            raw_df[col] = raw_df[col].astype(object)
        raw_df.loc[row_mask, col] = values

    @staticmethod
    def _lookup_dict_by_sample(mapping: Optional[Dict[str, Any]], sample_name: str) -> Any:
        if not mapping:
            return None
        if sample_name in mapping:
            return mapping[sample_name]
        target_norm = _normalize_sample_name(sample_name)
        for k, v in mapping.items():
            if str(k).strip() == str(sample_name).strip():
                return v
            if target_norm and _normalize_sample_name(k) == target_norm:
                return v
        return None

    @staticmethod
    def _find_companion_mask_path(sample_name: str, img_path: Path) -> Optional[Path]:
        norm_name = _normalize_sample_name(sample_name) or sample_name
        candidates = [
            img_path.parent / f"{sample_name}_mask.tiff",
            img_path.parent / f"{sample_name}_mask.tif",
            img_path.parent / f"{norm_name}_mask.tiff",
            img_path.parent / f"{norm_name}_mask.tif",
            img_path.parent / f"{img_path.stem}_mask.tiff",
            img_path.parent / f"{img_path.stem}_mask.tif",
            img_path.with_name(f"{sample_name}_segmentation_mask.tiff"),
            img_path.with_name(f"{norm_name}_segmentation_mask.tiff"),
        ]
        for cand in candidates:
            if cand.is_file():
                return cand
        return None

    @staticmethod
    def _extract_mask_from_batch(batch_entry: Any) -> Optional[np.ndarray]:
        if batch_entry is None:
            return None
        if not isinstance(batch_entry, dict):
            return np.asarray(batch_entry)
        for key in ("cytoplasm", "cytoplasm_mask", "nuclear", "nuclear_mask", "mask"):
            arr = batch_entry.get(key)
            if arr is not None:
                np_arr = np.asarray(arr)
                if np_arr.size > 0 and np.max(np_arr) > 0:
                    return np_arr
        for key in ("nuclear", "nuclear_mask", "mask"):
            arr = batch_entry.get(key)
            if arr is not None:
                return np.asarray(arr)
        return None

    @classmethod
    def _extract_active_mask(cls, state: Any) -> Optional[np.ndarray]:
        """Extract active 2D/3D segmentation mask from IrisViewerState or lightweight mock."""
        get_masks = getattr(state, "get_active_masks", None)
        if callable(get_masks):
            res = get_masks()
            if isinstance(res, tuple):
                nuc_mask = res[0] if len(res) > 0 else None
                cyto_mask = res[1] if len(res) > 1 else None
                has_cyto = cyto_mask is not None and np.max(cyto_mask) > 0
                chosen = cyto_mask if has_cyto else nuc_mask
                if chosen is not None:
                    return np.asarray(chosen)
            elif res is not None:
                return np.asarray(res)
        cyto = getattr(state, "cytoplasm_mask", None)
        if cyto is not None and np.max(cyto) > 0:
            return np.asarray(cyto)
        cur = getattr(state, "current_mask", None)
        if cur is not None:
            return np.asarray(cur)
        return cls._extract_mask_from_batch(getattr(state, "segmentation_results", None))

    @staticmethod
    def _coerce_2d_mask(msk: np.ndarray) -> np.ndarray:
        arr = np.squeeze(np.asarray(msk))
        if arr.ndim == 3:
            if arr.shape[0] > 1 and np.max(arr[1]) > 0:
                return arr[1]
            return arr[0]
        return arr

    @staticmethod
    def _coerce_chw_image(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim == 2:
            return arr[np.newaxis, ...]
        if arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.shape[0] > 4:
            return np.moveaxis(arr, -1, 0)
        if arr.ndim == 4:
            return arr[0]
        return arr

    def _can_resolve_fovs(self, target_samples: Optional[Sequence[str]] = None) -> bool:
        """Fast check (no disk TIFF reads) whether at least one scoped FOV has image+mask."""
        if self.ctx is None or getattr(self.ctx, "state", None) is None:
            return False
        state = self.ctx.state
        raw_img = getattr(state, "raw_image", None)
        multi_imgs = getattr(state, "multi_sample_images", None) or {}
        masks_map = getattr(state, "masks_map", None) or {}
        batch_res = getattr(state, "batch_results", None) or {}
        file_map = getattr(state, "file_map", None) or {}

        samples = list(target_samples) if target_samples is not None else []
        if not samples and hasattr(self, "sample_scope_bar") and self.sample_scope_bar._all_samples:
            samples = self.sample_scope_bar.resolve_samples()

        def _has_stored_mask(sname: str) -> bool:
            return (
                self._extract_mask_from_batch(self._lookup_dict_by_sample(masks_map, sname))
                is not None
                or self._extract_mask_from_batch(self._lookup_dict_by_sample(batch_res, sname))
                is not None
            )

        if not samples:
            if raw_img is not None and self._extract_active_mask(state) is not None:
                return True
            for sname, img in multi_imgs.items():
                if img is not None and _has_stored_mask(sname):
                    return True
            for sname, fpath in file_map.items():
                p = Path(fpath)
                if p.is_file() and (
                    _has_stored_mask(sname)
                    or self._find_companion_mask_path(sname, p) is not None
                ):
                    return True
            return False

        cur_name = getattr(state, "current_sample_name", None)
        for sname in samples:
            if raw_img is not None and (cur_name is None or self._sample_names_match(sname, cur_name)):
                if self._extract_active_mask(state) is not None or _has_stored_mask(sname):
                    return True
            if self._lookup_dict_by_sample(multi_imgs, sname) is not None and _has_stored_mask(sname):
                return True
            fpath = self._lookup_dict_by_sample(file_map, sname)
            if fpath is not None:
                p = Path(fpath)
                if p.is_file() and (
                    _has_stored_mask(sname)
                    or self._find_companion_mask_path(sname, p) is not None
                ):
                    return True
        return False

    def _load_fov_for_sample(
        self,
        sample_name: Optional[str],
        state: Any,
        *,
        load_from_disk: bool = True,
    ) -> Optional[Tuple[Optional[str], np.ndarray, np.ndarray]]:
        """Resolve (sample_name, (C, H, W) image, (H, W) mask) for one sample."""
        raw_img = getattr(state, "raw_image", None)
        cur_name = getattr(state, "current_sample_name", None)
        masks_map = getattr(state, "masks_map", None) or {}
        batch_res = getattr(state, "batch_results", None) or {}
        multi_imgs = getattr(state, "multi_sample_images", None) or {}
        file_map = getattr(state, "file_map", None) or {}

        # 1. Check active in-memory raw_image + active mask when matching active sample
        if raw_img is not None and (
            sample_name is None
            or cur_name is None
            or self._sample_names_match(sample_name, cur_name)
        ):
            mask = self._extract_active_mask(state)
            if mask is None and sample_name is not None:
                mask = self._extract_mask_from_batch(
                    self._lookup_dict_by_sample(masks_map, sample_name)
                )
            if mask is not None:
                return (
                    sample_name or cur_name,
                    self._coerce_chw_image(np.asarray(raw_img)),
                    self._coerce_2d_mask(mask),
                )

        if sample_name is None:
            return None

        # 2. Check multi_sample_images + masks_map / batch_results
        img_in_mem = self._lookup_dict_by_sample(multi_imgs, sample_name)
        mask_in_mem = self._extract_mask_from_batch(
            self._lookup_dict_by_sample(masks_map, sample_name)
        )
        if mask_in_mem is None:
            mask_in_mem = self._extract_mask_from_batch(
                self._lookup_dict_by_sample(batch_res, sample_name)
            )
        if img_in_mem is not None and mask_in_mem is not None:
            return (
                sample_name,
                self._coerce_chw_image(np.asarray(img_in_mem)),
                self._coerce_2d_mask(mask_in_mem),
            )

        # 3. Load from disk via ctx.state.file_map (image + masks_map or *_mask.tiff)
        if load_from_disk:
            fpath = self._lookup_dict_by_sample(file_map, sample_name)
            if fpath is not None:
                img_path = Path(fpath)
                if img_path.is_file():
                    from iris_viewer.io.loaders import (
                        load_existing_mask,
                        load_multichannel_tiff,
                    )

                    msk = mask_in_mem
                    if msk is None:
                        mask_path = self._find_companion_mask_path(sample_name, img_path)
                        if mask_path is not None:
                            msk = load_existing_mask(mask_path)
                    if msk is not None:
                        da_img, _ = load_multichannel_tiff(img_path)
                        img_np = np.asarray(da_img.compute() if hasattr(da_img, "compute") else da_img)
                        return (
                            sample_name,
                            self._coerce_chw_image(img_np),
                            self._coerce_2d_mask(np.asarray(msk)),
                        )
        return None

    def _resolve_active_fovs(
        self,
        target_samples: Optional[Sequence[str]] = None,
        *,
        load_from_disk: bool = True,
    ) -> List[Tuple[Optional[str], np.ndarray, np.ndarray]]:
        """Resolve scoped (sample_name, (C, H, W) image, (H, W) mask) tuples from ctx.state."""
        if self.ctx is None or getattr(self.ctx, "state", None) is None:
            return []
        state = self.ctx.state

        samples = list(target_samples) if target_samples is not None else []
        if not samples and hasattr(self, "sample_scope_bar") and self.sample_scope_bar._all_samples:
            samples = self.sample_scope_bar.resolve_samples()

        if samples:
            fovs: List[Tuple[Optional[str], np.ndarray, np.ndarray]] = []
            for sname in samples:
                loaded = self._load_fov_for_sample(
                    sname, state, load_from_disk=load_from_disk
                )
                if loaded is not None:
                    fovs.append(loaded)
            if fovs:
                return fovs
            # Fallback when single unnamed raw_image is loaded and sample name wasn't matched
            if getattr(state, "raw_image", None) is not None and len(samples) == 1:
                loaded = self._load_fov_for_sample(
                    None, state, load_from_disk=load_from_disk
                )
                if loaded is not None:
                    return [(samples[0], loaded[1], loaded[2])]
            return fovs

        # Unscoped fallback: single raw_image, else multi_sample_images, else file_map
        loaded_single = self._load_fov_for_sample(
            getattr(state, "current_sample_name", None),
            state,
            load_from_disk=load_from_disk,
        )
        if loaded_single is not None:
            return [loaded_single]

        fovs = []
        multi_imgs = getattr(state, "multi_sample_images", None) or {}
        for sname in multi_imgs.keys():
            loaded = self._load_fov_for_sample(
                sname, state, load_from_disk=load_from_disk
            )
            if loaded is not None:
                fovs.append(loaded)
        if fovs:
            return fovs

        if load_from_disk:
            file_map = getattr(state, "file_map", None) or {}
            for sname in file_map.keys():
                loaded = self._load_fov_for_sample(
                    sname, state, load_from_disk=True
                )
                if loaded is not None:
                    fovs.append(loaded)
        return fovs

    def _resolve_channel_names_for_image(
        self,
        raw_df: pd.DataFrame,
        n_channels: int,
    ) -> List[str]:
        state = getattr(self.ctx, "state", None) if self.ctx is not None else None
        state_markers = list(getattr(state, "markers", []) or [])
        if len(state_markers) == n_channels:
            return state_markers
        panel_markers = self._resolve_panel_markers(raw_df)
        if len(panel_markers) == n_channels:
            return panel_markers
        if state_markers and len(state_markers) > n_channels:
            return state_markers[:n_channels]
        return [f"Marker_{i}" for i in range(n_channels)]

    @staticmethod
    def _align_series_to_mask(
        raw_df: pd.DataFrame,
        row_mask: np.ndarray,
        series: pd.Series,
        default_val: Any,
    ) -> np.ndarray:
        """Align an ObjectNumber-indexed Series onto rows matching *row_mask*."""
        n_sub = int(np.sum(row_mask))
        out = np.full(n_sub, default_val, dtype=object if isinstance(default_val, str) else float)
        if series.empty or n_sub == 0:
            return out

        id_col = next(
            (c for c in ("ObjectNumber", "Cell_ID", "cell_id") if c in raw_df.columns),
            None,
        )
        if id_col is not None:
            sub_ids = pd.to_numeric(raw_df.loc[row_mask, id_col], errors="coerce").to_numpy()
            lookup = {int(k): v for k, v in series.items()}
            matched = 0
            for i, cid in enumerate(sub_ids):
                if np.isfinite(cid) and int(cid) in lookup:
                    out[i] = lookup[int(cid)]
                    matched += 1
            if matched > 0:
                return out

        if len(series) == n_sub:
            return series.to_numpy()
        return out

    def _run_astir(self):
        raw_df = (
            self.manual_gating.get_active_dataframe()
            if hasattr(self, "manual_gating")
            else None
        )
        if raw_df is None or raw_df.empty:
            self.lbl_astir_status.setText("No active cell dataframe available.")
            self.lbl_astir_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        row_mask, scoped_df, scope_desc, _resolved = self._resolve_scoped_rows_and_df(raw_df)
        if scoped_df is None or scoped_df.empty or not np.any(row_mask):
            self.lbl_astir_status.setText(f"No cells match sample scope ({scope_desc}).")
            self.lbl_astir_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        yaml_text = self.edit_astir_yaml.text().strip() if hasattr(self, "edit_astir_yaml") else ""
        if yaml_text and Path(yaml_text).is_file() and yaml_text != self._astir_yaml_path:
            try:
                self.load_astir_yaml(yaml_text)
            except Exception as exc:
                self.lbl_astir_status.setText(f"ASTIR YAML error: {str(exc)[:60]}")
                self.lbl_astir_status.setStyleSheet("color: #E57373; font-size: 11px;")
                return

        epochs = self.spin_astir_epochs.value()
        lr = self.spin_astir_lr.value()
        batch_size = self.spin_astir_batch.value()
        prior_mode = self.combo_astir_prior.currentText()
        col = self.txt_astir_col.text().strip() or "ASTIR_CellType"
        cutoff = self.spin_astir_cutoff.value()
        unk_cutoff = self.spin_astir_unk.value()

        markers = self._resolve_panel_markers(scoped_df)
        if not markers:
            self.lbl_astir_status.setText("ASTIR failed: No numeric marker columns detected.")
            self.lbl_astir_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        from iris_viewer.core.types import CellTypePanel, CellTypeRule, MarkerPanel
        from iris_viewer.engine.phenotyping.astir_backend import ASTIRBackend

        panel = None
        marker_config = None
        if self._astir_yaml_panel is not None:
            matched_rules = [
                CellTypeRule(
                    cell_type=r.cell_type,
                    markers_positive=[
                        m for m in r.markers_positive if resolve_marker_col(scoped_df, m) is not None
                    ],
                    parent=r.parent,
                )
                for r in self._astir_yaml_panel.rules
            ]
            matched_rules = [r for r in matched_rules if r.markers_positive]
            if len(matched_rules) >= 2:
                panel = CellTypePanel(rules=matched_rules, panel_file=self._astir_yaml_panel.panel_file)
                marker_config = self._astir_marker_config

        if panel is None:
            tree_nodes = (
                [
                    n
                    for n in self.manual_gating.hierarchy_root.get_all_nodes()
                    if n.parent is not None and n.markers
                ]
                if hasattr(self.manual_gating, "hierarchy_root")
                else []
            )
            rules = []
            if len(tree_nodes) >= 2:
                for n in tree_nodes:
                    rules.append(
                        CellTypeRule(
                            cell_type=n.name,
                            markers_positive=list(n.markers),
                            parent=n.parent.name if n.parent and n.parent.parent else None,
                        )
                    )
            else:
                for m in markers[:10]:
                    rules.append(CellTypeRule(cell_type=f"{m}+ Population", markers_positive=[m]))
            panel = CellTypePanel(rules=rules)

        all_panel_markers = list(
            dict.fromkeys(
                markers + [m for r in panel.rules for m in r.markers_positive]
            )
        )
        valid_markers = [m for m in all_panel_markers if resolve_marker_col(scoped_df, m) is not None]
        mp = MarkerPanel(markers=valid_markers)
        df_input = self._build_backend_feature_frame(scoped_df, valid_markers)
        cids = ASTIRBackend._cell_ids_from_features(df_input)

        self.btn_run_astir.setEnabled(False)
        self.lbl_astir_status.setText(
            f"Running ASTIR VI optimization ({epochs} epochs, lr={lr}, scope={scope_desc})..."
        )
        self.lbl_astir_status.setStyleSheet("color: #44AAFF; font-size: 11px;")
        iris_log(
            "IRIS-Phenotyper",
            "ASTIR",
            f"Starting ASTIR ({prior_mode}, epochs={epochs}, lr={lr}) on {len(df_input):,} cells (scope={scope_desc}).",
        )

        def _compute():
            t0 = time.perf_counter()
            backend = ASTIRBackend()
            is_vi_ready = backend.is_vi_available()
            prob_dict = backend.predict(
                df_input,
                cell_type_panel=panel,
                marker_panel=mp,
                marker_config=marker_config,
                max_epochs=epochs,
                learning_rate=lr,
                batch_size=batch_size,
                arcsinh_cofactor=5.0,
                threshold=cutoff,
            )
            prob_df = getattr(backend, "last_probabilities_", None)
            assignments: List[str] = []
            confidences: List[float] = []
            if prob_df is not None and len(prob_df) == len(df_input) and not prob_df.empty:
                col_names = [str(c) for c in prob_df.columns]
                arr = prob_df.to_numpy(dtype=float)
                best_idx = arr.argmax(axis=1)
                best_probs = arr.max(axis=1)
                for idx_i, p_val in zip(best_idx, best_probs):
                    best_type = col_names[int(idx_i)]
                    p_f = float(np.clip(p_val, 0.0, 1.0))
                    confidences.append(p_f)
                    if p_f >= cutoff:
                        assignments.append(best_type)
                    elif p_f < unk_cutoff:
                        assignments.append("Unknown")
                    else:
                        assignments.append(f"{best_type} (Low Conf)")
            else:
                for cid in cids:
                    probs = prob_dict.get(cid, {})
                    if probs:
                        best_type, best_prob = max(probs.items(), key=lambda kv: kv[1])
                        p_f = float(np.clip(best_prob, 0.0, 1.0))
                        confidences.append(p_f)
                        if p_f >= cutoff:
                            assignments.append(best_type)
                        elif p_f < unk_cutoff:
                            assignments.append("Unknown")
                        else:
                            assignments.append(f"{best_type} (Low Conf)")
                    else:
                        assignments.append("Unassigned")
                        confidences.append(0.0)
            elapsed = time.perf_counter() - t0
            return assignments, confidences, is_vi_ready, elapsed

        def _on_done(result):
            try:
                assignments, confidences, is_vi_ready, elapsed = result
                conf_col = f"{col}_Confidence"
                self._assign_scoped_column(
                    raw_df, row_mask, col, assignments, default_value="Unassigned"
                )
                self._assign_scoped_column(
                    raw_df, row_mask, conf_col, confidences, default_value=np.nan
                )
                unique_cts = [
                    str(ct)
                    for ct in raw_df.loc[row_mask, col].dropna().unique()
                    if str(ct) not in ("Unknown", "Unassigned")
                ]
                self._propagate_prediction_columns(
                    raw_df,
                    [col, conf_col],
                    primary_col=col,
                    unique_types=unique_cts,
                    markers=valid_markers,
                )

                engine_label = "ASTIR VI" if is_vi_ready else "ASTIR rule fallback"
                n_scoped = int(np.sum(row_mask))
                self.lbl_astir_status.setText(
                    f"{engine_label} complete: classified {n_scoped:,} cells "
                    f"({scope_desc}) into {len(unique_cts)} types -> '{col}'."
                )
                self.lbl_astir_status.setStyleSheet(
                    "color: #66BB6A; font-weight: bold; font-size: 11px;"
                )
                iris_log(
                    "IRIS-Phenotyper",
                    "ASTIR",
                    f"Completed {engine_label} in {elapsed:.2f}s: {n_scoped:,} cells -> "
                    f"{len(unique_cts)} types ({', '.join(unique_cts[:5])}) into '{col}' (scope={scope_desc}).",
                )
            finally:
                self.btn_run_astir.setEnabled(True)

        def _on_error(exc: Exception):
            self.btn_run_astir.setEnabled(True)
            self.lbl_astir_status.setText(f"ASTIR execution error: {str(exc)[:60]}")
            self.lbl_astir_status.setStyleSheet("color: #E57373; font-size: 11px;")
            iris_log("IRIS-Phenotyper", "ASTIR", f"Execution error: {exc}", level="ERROR")

        self._dispatch_heavy_task(
            _compute,
            _on_done,
            _on_error,
            error_title="ASTIR Phenotyping Error",
            error_prefix="ASTIR phenotyping failed",
        )

    def _run_maps(self):
        raw_df = (
            self.manual_gating.get_active_dataframe()
            if hasattr(self, "manual_gating")
            else None
        )
        if raw_df is None or raw_df.empty:
            self.lbl_maps_status.setText("No active cell dataframe available.")
            self.lbl_maps_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        row_mask, scoped_df, scope_desc, _resolved = self._resolve_scoped_rows_and_df(raw_df)
        if scoped_df is None or scoped_df.empty or not np.any(row_mask):
            self.lbl_maps_status.setText(f"No cells match sample scope ({scope_desc}).")
            self.lbl_maps_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        if self.combo_maps_ref_col.count() == 0:
            self._sync_reference_columns(raw_df)

        mode = self.combo_maps_mode.currentText()
        ref_col = self.combo_maps_ref_col.currentText().strip()
        ckpt_path = self.edit_maps_ckpt.text().strip()
        hidden_dim = self.spin_maps_hidden.value()
        dropout = self.spin_maps_dropout.value()
        epochs = self.spin_maps_epochs.value()
        lr = self.spin_maps_lr.value()
        batch_size = self.spin_maps_batch.value()
        col = self.txt_maps_col.text().strip() or "MAPS_CellType"

        markers = self._resolve_panel_markers(scoped_df)
        if not markers:
            self.lbl_maps_status.setText("MAPS failed: No marker columns detected.")
            self.lbl_maps_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        if mode == "Load Checkpoint (.pt)" and (not ckpt_path or not Path(ckpt_path).is_file()):
            self.lbl_maps_status.setText("Please select a valid MAPS checkpoint (.pt) file.")
            self.lbl_maps_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        from iris_viewer.core.types import CellTypePanel, CellTypeRule
        from iris_viewer.engine.phenotyping.maps_backend import MAPSBackend

        rules = [CellTypeRule(cell_type=f"MAPS_{m}+", markers_positive=[m]) for m in markers[:8]]
        panel = CellTypePanel(rules=rules)
        df_input = self._build_backend_feature_frame(scoped_df, markers)
        feat_cols = [
            resolve_marker_col(df_input, m) or f"{m}_mean"
            for m in markers
            if (resolve_marker_col(df_input, m) or f"{m}_mean") in df_input.columns
        ]
        cids = MAPSBackend._cell_ids_from_features(df_input)

        # Determine if training mode has a valid >=2-class reference column in scoped rows
        train_df = None
        effective_mode = mode
        if mode == "Train from Reference Column":
            if ref_col and ref_col in df_input.columns:
                lbl_ser = df_input[ref_col].astype(str).str.strip()
                valid_mask = (
                    df_input[ref_col].notna()
                    & (~lbl_ser.str.lower().isin(_UNLABELED_SENTINELS))
                )
                n_classes = int(lbl_ser[valid_mask].nunique())
                if int(valid_mask.sum()) >= 2 and n_classes >= 2:
                    train_df = df_input.loc[valid_mask].copy()
                else:
                    effective_mode = "Marker-Rule Fallback"
            else:
                effective_mode = "Marker-Rule Fallback"

        self.btn_run_maps.setEnabled(False)
        self.lbl_maps_status.setText(f"Running MAPS ({effective_mode}, scope={scope_desc})...")
        self.lbl_maps_status.setStyleSheet("color: #44AAFF; font-size: 11px;")
        iris_log(
            "IRIS-Phenotyper",
            "MAPS",
            f"Starting MAPS ({effective_mode}, epochs={epochs}) on {len(df_input):,} cells (scope={scope_desc}).",
        )

        def _compute():
            t0 = time.perf_counter()
            backend = MAPSBackend()
            if effective_mode == "Load Checkpoint (.pt)":
                preds = backend.predict(
                    df_input,
                    checkpoint_path=ckpt_path,
                    batch_size=batch_size,
                )
            elif effective_mode == "Train from Reference Column" and train_df is not None:
                backend.fit(
                    train_df,
                    label_col=ref_col,
                    feature_cols=feat_cols,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    max_epochs=epochs,
                    min_epochs=min(epochs, 10),
                    patience=max(10, epochs // 2),
                    learning_rate=lr,
                    batch_size=batch_size,
                    val_fraction=0.0 if len(train_df) < 10 else 0.2,
                )
                preds = backend.predict(df_input, batch_size=batch_size)
            else:
                preds = backend.predict(df_input, cell_types=panel)

            prob_df = getattr(backend, "last_probabilities_", None)
            assignments: List[str] = []
            confidences: List[float] = []
            if prob_df is not None and len(prob_df) == len(df_input) and not prob_df.empty:
                col_names = [str(c) for c in prob_df.columns]
                arr = prob_df.to_numpy(dtype=float)
                best_idx = arr.argmax(axis=1)
                best_probs = arr.max(axis=1)
                for idx_i, p_val in zip(best_idx, best_probs):
                    assignments.append(col_names[int(idx_i)])
                    confidences.append(float(np.clip(p_val, 0.0, 1.0)))
            else:
                for cid in cids:
                    p = preds.get(cid, {})
                    if p:
                        best_ct, best_p = max(p.items(), key=lambda kv: kv[1])
                        assignments.append(best_ct)
                        confidences.append(float(np.clip(best_p, 0.0, 1.0)))
                    else:
                        assignments.append("Unresolved")
                        confidences.append(0.0)
            elapsed = time.perf_counter() - t0
            return assignments, confidences, elapsed

        def _on_done(result):
            try:
                assignments, confidences, elapsed = result
                conf_col = f"{col}_Confidence"
                self._assign_scoped_column(
                    raw_df, row_mask, col, assignments, default_value="Unassigned"
                )
                self._assign_scoped_column(
                    raw_df, row_mask, conf_col, confidences, default_value=np.nan
                )
                palette = ["#00FFFF", "#FF00AA", "#00FF66", "#FFBB00", "#7700FF", "#00AAFF"]
                unique_cts = [
                    str(ct)
                    for ct in raw_df.loc[row_mask, col].dropna().unique()
                    if str(ct) not in ("Unresolved", "Unassigned")
                ]
                self._propagate_prediction_columns(
                    raw_df,
                    [col, conf_col],
                    primary_col=col,
                    unique_types=unique_cts,
                    markers=markers,
                    palette=palette,
                )
                n_scoped = int(np.sum(row_mask))
                self.lbl_maps_status.setText(
                    f"MAPS ({effective_mode}) finished: {n_scoped:,} cells "
                    f"({scope_desc}) classified into {len(unique_cts)} types -> '{col}'."
                )
                self.lbl_maps_status.setStyleSheet(
                    "color: #66BB6A; font-weight: bold; font-size: 11px;"
                )
                iris_log(
                    "IRIS-Phenotyper",
                    "MAPS",
                    f"Completed MAPS ({effective_mode}) in {elapsed:.2f}s: {n_scoped:,} cells -> "
                    f"{len(unique_cts)} types ({', '.join(unique_cts[:5])}) into '{col}' (scope={scope_desc}).",
                )
            finally:
                self.btn_run_maps.setEnabled(True)

        def _on_error(exc: Exception):
            self.btn_run_maps.setEnabled(True)
            self.lbl_maps_status.setText(f"MAPS error: {str(exc)[:60]}")
            self.lbl_maps_status.setStyleSheet("color: #E57373; font-size: 11px;")
            iris_log("IRIS-Phenotyper", "MAPS", f"Execution error: {exc}", level="ERROR")

        self._dispatch_heavy_task(
            _compute,
            _on_done,
            _on_error,
            error_title="MAPS Phenotyping Error",
            error_prefix="MAPS phenotyping failed",
        )

    def _run_stellar(self):
        raw_df = (
            self.manual_gating.get_active_dataframe()
            if hasattr(self, "manual_gating")
            else None
        )
        if raw_df is None or raw_df.empty:
            self.lbl_stellar_status.setText("No active cell dataframe available.")
            self.lbl_stellar_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        if self.combo_stellar_ref_col.count() == 0:
            self._sync_reference_columns(raw_df)
        if self.list_stellar_ref_samples.count() == 0:
            self._sync_stellar_samples(raw_df)

        ref_col = self.combo_stellar_ref_col.currentText().strip()
        if not ref_col or ref_col not in raw_df.columns:
            self.lbl_stellar_status.setText(
                "Select a valid Reference Label Column for STELLAR."
            )
            self.lbl_stellar_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        markers = self._resolve_panel_markers(raw_df)
        feat_cols = [
            c
            for m in markers
            if (c := resolve_marker_col(raw_df, m)) is not None
            and pd.api.types.is_numeric_dtype(raw_df[c])
        ]
        if not feat_cols:
            self.lbl_stellar_status.setText("STELLAR failed: No numeric marker columns found.")
            self.lbl_stellar_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        dist_thres = self.spin_stellar_dist.value()
        epochs = self.spin_stellar_epochs.value()
        lr = self.spin_stellar_lr.value()
        num_seed_class = self.spin_stellar_seed_classes.value()
        novel_thresh = self.spin_stellar_novel_thresh.value()
        ckpt_path = self.edit_stellar_ckpt.text().strip()
        col = self.txt_stellar_col.text().strip() or "STELLAR_CellType"

        scope_row_mask, _, scope_desc, _resolved = self._resolve_scoped_rows_and_df(raw_df)
        scope_mode = (
            self.sample_scope_bar.get_scope_mode()
            if hasattr(self, "sample_scope_bar")
            else SCOPE_ALL
        )

        scol = find_sample_column(raw_df)
        ref_samples = self._checked_list_items(self.list_stellar_ref_samples)
        tgt_samples = self._checked_list_items(self.list_stellar_target_samples)

        lbl_ser = raw_df[ref_col].astype(str).str.strip()
        valid_lbl_mask = (
            raw_df[ref_col].notna() & (~lbl_ser.str.lower().isin(_UNLABELED_SENTINELS))
        ).to_numpy()

        if scol is not None and ref_samples:
            s_vals = raw_df[scol].astype(str).str.strip()
            ref_norms = {_normalize_sample_name(s) for s in ref_samples}
            ref_sample_mask = (
                s_vals.isin(ref_samples) | s_vals.map(_normalize_sample_name).isin(ref_norms)
            ).to_numpy()
            train_mask = ref_sample_mask & valid_lbl_mask
        else:
            train_mask = valid_lbl_mask

        if scope_mode == SCOPE_ALL:
            if scol is not None and tgt_samples:
                s_vals = raw_df[scol].astype(str).str.strip()
                tgt_norms = {_normalize_sample_name(s) for s in tgt_samples}
                target_mask = (
                    s_vals.isin(tgt_samples) | s_vals.map(_normalize_sample_name).isin(tgt_norms)
                ).to_numpy()
            elif (~valid_lbl_mask).any():
                target_mask = ~valid_lbl_mask
            else:
                target_mask = np.ones(len(raw_df), dtype=bool)
        else:
            if scol is not None and tgt_samples:
                s_vals = raw_df[scol].astype(str).str.strip()
                tgt_norms = {_normalize_sample_name(s) for s in tgt_samples}
                list_tgt_mask = (
                    s_vals.isin(tgt_samples) | s_vals.map(_normalize_sample_name).isin(tgt_norms)
                ).to_numpy()
                combined = list_tgt_mask & scope_row_mask
                target_mask = combined if np.any(combined) else scope_row_mask
            else:
                target_mask = scope_row_mask

        if not np.any(train_mask):
            self.lbl_stellar_status.setText(
                f"No labeled reference cells found in '{ref_col}' for selected reference samples."
            )
            self.lbl_stellar_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return
        if not np.any(target_mask):
            self.lbl_stellar_status.setText(f"No target cells selected in scope ({scope_desc}).")
            self.lbl_stellar_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        train_df = raw_df.loc[train_mask].copy()
        target_df = raw_df.loc[target_mask].copy()

        self.btn_run_stellar.setEnabled(False)
        self.lbl_stellar_status.setText(
            f"Running STELLAR ({len(train_df):,} ref -> {len(target_df):,} target cells, scope={scope_desc})..."
        )
        self.lbl_stellar_status.setStyleSheet("color: #44AAFF; font-size: 11px;")
        iris_log(
            "IRIS-Phenotyper",
            "STELLAR",
            f"Starting STELLAR (epochs={epochs}, dist={dist_thres}) on {len(train_df):,} ref -> "
            f"{len(target_df):,} target cells (scope={scope_desc}).",
        )

        def _compute():
            t0 = time.perf_counter()
            from iris_phenotyper.stellar import STELLARPhenotyper

            stellar = STELLARPhenotyper(
                distance_thres=dist_thres,
                epochs=epochs,
                seed_epochs=min(epochs, 20),
                lr=lr,
                num_seed_class=num_seed_class,
                feature_columns=feat_cols,
                label_col=ref_col,
            )
            if ckpt_path and Path(ckpt_path).is_file():
                import torch

                from iris_phenotyper.stellar.pipeline import (
                    STELLARConfig,
                    STELLARPipeline,
                    _build_model_classes,
                )

                payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(payload, dict) and "model_state" in payload:
                    cfg_obj = STELLARConfig(
                        distance_thres=dist_thres,
                        epochs=epochs,
                        lr=lr,
                        num_seed_class=num_seed_class,
                    )
                    pipe = STELLARPipeline(cfg_obj)
                    pipe.known_classes = list(payload["known_classes"])
                    pipe.novel_classes = list(payload.get("novel_classes", []))
                    pipe.class_names = list(payload["class_names"])
                    pipe.input_dim = int(payload["input_dim"])
                    pipe.num_heads = int(payload["num_heads"])
                    pipe.device_used = "cpu"
                    classes_dict = _build_model_classes()
                    pipe.model = classes_dict["STELLAREncoder"](
                        pipe.input_dim, pipe.num_heads, hidden_dim=cfg_obj.hidden_dim
                    )
                    pipe.model.load_state_dict(payload["model_state"])
                    stellar._pipeline = pipe
                    stellar._resolved_features = list(payload.get("feature_columns", feat_cols))
                    res = stellar.predict(target_df, retrain=False)
                else:
                    res = stellar.fit_predict(
                        train_df,
                        target_df,
                        feature_columns=feat_cols,
                        label_col=ref_col,
                    )
            else:
                res = stellar.fit_predict(
                    train_df,
                    target_df,
                    feature_columns=feat_cols,
                    label_col=ref_col,
                )

            labels_arr = res.labels.astype(str).to_numpy()
            conf_arr = res.confidence.to_numpy(dtype=float)
            base_novel = (
                res.is_novel.to_numpy(dtype=bool)
                if res.is_novel is not None
                else np.zeros(len(target_df), dtype=bool)
            )
            is_novel_arr = base_novel | (conf_arr < novel_thresh)
            elapsed = time.perf_counter() - t0
            return labels_arr, conf_arr, is_novel_arr, elapsed

        def _on_done(result):
            try:
                labels_arr, conf_arr, is_novel_arr, elapsed = result
                conf_col = f"{col}_Confidence"
                novel_col = f"{col}_IsNovel"

                self._assign_scoped_column(
                    raw_df, target_mask, col, labels_arr, default_value="Unassigned"
                )
                self._assign_scoped_column(
                    raw_df, target_mask, conf_col, conf_arr, default_value=np.nan
                )
                self._assign_scoped_column(
                    raw_df, target_mask, novel_col, is_novel_arr, default_value=False
                )

                unique_cts = [
                    str(ct)
                    for ct in raw_df.loc[target_mask, col].dropna().unique()
                    if str(ct) not in ("Unassigned", "Unknown")
                ]
                palette = [
                    "#00E5FF",
                    "#66BB6A",
                    "#FFD600",
                    "#FF0055",
                    "#D500F9",
                    "#FF6D00",
                    "#00B0FF",
                    "#00E676",
                ]
                self._propagate_prediction_columns(
                    raw_df,
                    [col, conf_col, novel_col],
                    primary_col=col,
                    unique_types=unique_cts,
                    markers=markers,
                    palette=palette,
                )
                n_novel = int(np.sum(is_novel_arr))
                self.lbl_stellar_status.setText(
                    f"STELLAR transfer complete: {len(target_df):,} target cells "
                    f"({n_novel:,} novel flagged) -> '{col}'."
                )
                self.lbl_stellar_status.setStyleSheet(
                    "color: #66BB6A; font-weight: bold; font-size: 11px;"
                )
                iris_log(
                    "IRIS-Phenotyper",
                    "STELLAR",
                    f"Completed STELLAR in {elapsed:.2f}s: {len(target_df):,} target cells -> "
                    f"{len(unique_cts)} types ({n_novel:,} novel) into '{col}' (scope={scope_desc}).",
                )
            finally:
                self.btn_run_stellar.setEnabled(True)

        def _on_error(exc: Exception):
            self.btn_run_stellar.setEnabled(True)
            self.lbl_stellar_status.setText(f"STELLAR error: {str(exc)[:60]}")
            self.lbl_stellar_status.setStyleSheet("color: #E57373; font-size: 11px;")
            iris_log("IRIS-Phenotyper", "STELLAR", f"Execution error: {exc}", level="ERROR")

        self._dispatch_heavy_task(
            _compute,
            _on_done,
            _on_error,
            error_title="STELLAR Phenotyping Error",
            error_prefix="STELLAR label transfer failed",
        )

    def _run_cellsighter(self):
        raw_df = (
            self.manual_gating.get_active_dataframe()
            if hasattr(self, "manual_gating")
            else None
        )
        if raw_df is None or raw_df.empty:
            self.lbl_cs_status.setText("No active cell dataframe available.")
            self.lbl_cs_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        row_mask, scoped_df, scope_desc, resolved_samples = self._resolve_scoped_rows_and_df(raw_df)
        if scoped_df is None or scoped_df.empty or not np.any(row_mask):
            self.lbl_cs_status.setText(f"No cells match sample scope ({scope_desc}).")
            self.lbl_cs_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        if self.combo_cs_ref_col.count() == 0:
            self._sync_reference_columns(raw_df)

        if not self._can_resolve_fovs(resolved_samples):
            self.lbl_cs_status.setText(
                "CellSighter requires an active or saved image and segmentation mask."
            )
            self.lbl_cs_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        mode = self.combo_cs_mode.currentText()
        ref_col = self.combo_cs_ref_col.currentText().strip()
        ckpt_path = self.edit_cs_ckpt.text().strip()
        crop_size = self.spin_cs_crop_size.value()
        crop_input_size = min(crop_size, self.spin_cs_crop_input_size.value())
        normalization = self.combo_cs_norm.currentText()
        epochs = self.spin_cs_epochs.value()
        lr = self.spin_cs_lr.value()
        col = self.txt_cs_col.text().strip() or "CellSighter_CellType"

        if mode == "Load Ensemble (.pt)" and (not ckpt_path or not Path(ckpt_path).is_file()):
            self.lbl_cs_status.setText("Please select a valid CellSighter checkpoint (.pt) file.")
            self.lbl_cs_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        if mode == "Train on Reference Column":
            if not ref_col or ref_col not in raw_df.columns:
                self.lbl_cs_status.setText(
                    "Select a valid Reference Label Column for CellSighter training."
                )
                self.lbl_cs_status.setStyleSheet("color: #E57373; font-size: 11px;")
                return
            lbl_ser = scoped_df[ref_col].astype(str).str.strip()
            n_valid_lbl = int(
                (scoped_df[ref_col].notna() & (~lbl_ser.str.lower().isin(_UNLABELED_SENTINELS))).sum()
            )
            if n_valid_lbl < 2:
                self.lbl_cs_status.setText(
                    f"Need at least 2 labeled cells in '{ref_col}' within scope ({scope_desc})."
                )
                self.lbl_cs_status.setStyleSheet("color: #E57373; font-size: 11px;")
                return

        scol = find_sample_column(raw_df)
        id_col = next(
            (c for c in ("ObjectNumber", "Cell_ID", "cell_id") if c in raw_df.columns),
            None,
        )

        self.btn_run_cs.setEnabled(False)
        self.lbl_cs_status.setText(f"Running CellSighter ({mode}, scope={scope_desc})...")
        self.lbl_cs_status.setStyleSheet("color: #44AAFF; font-size: 11px;")
        iris_log(
            "IRIS-Phenotyper",
            "CellSighter",
            f"Starting CellSighter ({mode}, epochs={epochs}) on {len(scoped_df):,} cells (scope={scope_desc}).",
        )

        def _compute():
            t0 = time.perf_counter()
            from iris_phenotyper.cellsighter import CellSighterPhenotyper

            fovs = self._resolve_active_fovs(
                target_samples=resolved_samples, load_from_disk=True
            )
            if not fovs:
                raise RuntimeError("No matching image and mask FOVs found for selected sample scope.")

            first_img = fovs[0][1]
            channel_names = self._resolve_channel_names_for_image(raw_df, int(first_img.shape[0]))

            fov_specs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]] = []
            for sname, img, msk in fovs:
                if sname and scol is not None:
                    s_vals = raw_df[scol].astype(str).str.strip()
                    sname_norm = _normalize_sample_name(sname)
                    m_s = (
                        (s_vals == str(sname).strip())
                        | (s_vals.map(_normalize_sample_name) == sname_norm)
                    ).to_numpy()
                    fov_row_mask = row_mask & m_s
                    if not np.any(fov_row_mask) and len(fovs) == 1 and raw_df[scol].nunique(dropna=True) <= 1:
                        fov_row_mask = row_mask
                else:
                    fov_row_mask = row_mask

                if not np.any(fov_row_mask):
                    continue

                lbl_dict: Dict[int, str] = {}
                if mode == "Train on Reference Column" and ref_col in raw_df.columns:
                    sub = raw_df.loc[fov_row_mask]
                    for idx_r, (r_idx, r_val) in enumerate(sub[ref_col].items()):
                        if pd.isna(r_val):
                            continue
                        s_val = str(r_val).strip()
                        if s_val.lower() in _UNLABELED_SENTINELS:
                            continue
                        cid_val = (
                            int(sub.at[r_idx, id_col])
                            if id_col is not None
                            else int(idx_r + 1)
                        )
                        lbl_dict[cid_val] = s_val
                fov_specs.append((fov_row_mask, img, msk, lbl_dict))

            if not fov_specs:
                raise RuntimeError("No cells in raw_df matched the resolved FOV sample names.")
            if mode == "Train on Reference Column" and sum(len(s[3]) for s in fov_specs) < 2:
                raise ValueError(f"Need at least 2 labeled cells in '{ref_col}' to train CellSighter.")

            if mode == "Load Ensemble (.pt)":
                cs = CellSighterPhenotyper(
                    checkpoint_path=ckpt_path,
                    crop_size=crop_size,
                    crop_input_size=crop_input_size,
                    normalization=normalization,
                )
            else:
                all_labels = [lbl for _, _, _, d in fov_specs for lbl in d.values()]
                class_names = sorted(set(all_labels))
                n_total_lbl = len(all_labels)
                cs = CellSighterPhenotyper(
                    crop_size=crop_size,
                    crop_input_size=crop_input_size,
                    normalization=normalization,
                    epochs=epochs,
                    lr=lr,
                    batch_size=min(64, max(4, n_total_lbl)),
                    val_fraction=0.0 if n_total_lbl < 10 else 0.1,
                    sample_batch=len(class_names) > 1,
                )
                cs.fit(
                    images=[img for _, img, _, _ in fov_specs],
                    masks=[msk for _, _, msk, _ in fov_specs],
                    labels=[d for _, _, _, d in fov_specs],
                    channel_names=channel_names,
                    class_names=class_names,
                )

            per_fov_results = []
            for fov_row_mask, img, msk, _ in fov_specs:
                res = cs.predict(img, msk, channel_names=channel_names)
                per_fov_results.append((fov_row_mask, res))
            elapsed = time.perf_counter() - t0
            return per_fov_results, channel_names, elapsed

        def _on_done(result):
            try:
                per_fov_results, channel_names, elapsed = result
                conf_col = f"{col}_Confidence"
                written_mask = np.zeros(len(raw_df), dtype=bool)
                for fov_row_mask, res in per_fov_results:
                    self._assign_scoped_column(
                        raw_df,
                        fov_row_mask,
                        col,
                        self._align_series_to_mask(
                            raw_df, fov_row_mask, res.labels.astype(str), "Unassigned"
                        ),
                        default_value="Unassigned",
                    )
                    self._assign_scoped_column(
                        raw_df,
                        fov_row_mask,
                        conf_col,
                        self._align_series_to_mask(
                            raw_df, fov_row_mask, res.confidence.astype(float), np.nan
                        ),
                        default_value=np.nan,
                    )
                    written_mask |= fov_row_mask

                unique_cts = [
                    str(ct)
                    for ct in raw_df.loc[written_mask, col].dropna().unique()
                    if str(ct) not in ("Unassigned", "Unknown")
                ]
                self._propagate_prediction_columns(
                    raw_df,
                    [col, conf_col],
                    primary_col=col,
                    unique_types=unique_cts,
                    markers=channel_names,
                )
                n_classified = int(np.sum(written_mask))
                self.lbl_cs_status.setText(
                    f"CellSighter complete: classified {n_classified:,} cells "
                    f"({scope_desc}) into {len(unique_cts)} types -> '{col}'."
                )
                self.lbl_cs_status.setStyleSheet(
                    "color: #66BB6A; font-weight: bold; font-size: 11px;"
                )
                iris_log(
                    "IRIS-Phenotyper",
                    "CellSighter",
                    f"Completed CellSighter ({mode}) in {elapsed:.2f}s across {len(per_fov_results)} FOV(s): "
                    f"{n_classified:,} cells -> {len(unique_cts)} types ({', '.join(unique_cts[:5])}) into '{col}' (scope={scope_desc}).",
                )
            finally:
                self.btn_run_cs.setEnabled(True)

        def _on_error(exc: Exception):
            self.btn_run_cs.setEnabled(True)
            self.lbl_cs_status.setText(f"CellSighter error: {str(exc)[:60]}")
            self.lbl_cs_status.setStyleSheet("color: #E57373; font-size: 11px;")
            iris_log("IRIS-Phenotyper", "CellSighter", f"Execution error: {exc}", level="ERROR")

        self._dispatch_heavy_task(
            _compute,
            _on_done,
            _on_error,
            error_title="CellSighter Phenotyping Error",
            error_prefix="CellSighter phenotyping failed",
        )

    def _run_nimbus(self):
        raw_df = (
            self.manual_gating.get_active_dataframe()
            if hasattr(self, "manual_gating")
            else None
        )
        if raw_df is None or raw_df.empty:
            self.lbl_nimbus_status.setText("No active cell dataframe available.")
            self.lbl_nimbus_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        row_mask, scoped_df, scope_desc, resolved_samples = self._resolve_scoped_rows_and_df(raw_df)
        if scoped_df is None or scoped_df.empty or not np.any(row_mask):
            self.lbl_nimbus_status.setText(f"No cells match sample scope ({scope_desc}).")
            self.lbl_nimbus_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        if not self._can_resolve_fovs(resolved_samples):
            self.lbl_nimbus_status.setText(
                "Nimbus requires an active or saved image and segmentation mask."
            )
            self.lbl_nimbus_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        ckpt_text = self.edit_nimbus_ckpt.text().strip()
        if ckpt_text and ckpt_text != "latest" and not Path(ckpt_text).is_file():
            self.lbl_nimbus_status.setText("Specified Nimbus checkpoint (.pt) does not exist.")
            self.lbl_nimbus_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        quantile = self.spin_nimbus_quantile.value()
        pos_thresh = self.spin_nimbus_pos_thresh.value()
        tta = self.chk_nimbus_tta.isChecked()
        ds_mode = self.combo_nimbus_mode.currentText()
        n_clusters = self.spin_nimbus_clusters.value()
        col = self.txt_nimbus_col.text().strip() or "Nimbus_CellType"
        scol = find_sample_column(raw_df)

        panel_preset = None
        if ds_mode == "Panel Marker Rules":
            from iris_viewer.core.types import CellTypePanel, CellTypeRule

            if self._astir_yaml_panel is not None:
                panel_preset = self._astir_yaml_panel
            elif hasattr(self.manual_gating, "hierarchy_root"):
                nodes = [
                    n
                    for n in self.manual_gating.hierarchy_root.get_all_nodes()
                    if n.parent is not None and n.markers
                ]
                if nodes:
                    panel_preset = CellTypePanel(
                        rules=[
                            CellTypeRule(cell_type=n.name, markers_positive=list(n.markers))
                            for n in nodes
                        ]
                    )

        self.btn_run_nimbus.setEnabled(False)
        self.lbl_nimbus_status.setText(f"Running Nimbus ({ds_mode}, TTA={tta}, scope={scope_desc})...")
        self.lbl_nimbus_status.setStyleSheet("color: #44AAFF; font-size: 11px;")
        iris_log(
            "IRIS-Phenotyper",
            "Nimbus",
            f"Starting Nimbus ({ds_mode}, TTA={tta}, q={quantile}) on {len(scoped_df):,} cells (scope={scope_desc}).",
        )

        def _compute():
            t0 = time.perf_counter()
            from iris_phenotyper.nimbus import NimbusPhenotyper
            from iris_viewer.core.types import CellTypePanel, CellTypeRule

            fovs = self._resolve_active_fovs(
                target_samples=resolved_samples, load_from_disk=True
            )
            if not fovs:
                raise RuntimeError("No matching image and mask FOVs found for selected sample scope.")

            first_img = fovs[0][1]
            channel_names = self._resolve_channel_names_for_image(raw_df, int(first_img.shape[0]))
            panel = panel_preset
            if ds_mode == "Panel Marker Rules" and panel is None:
                panel = CellTypePanel(
                    rules=[
                        CellTypeRule(cell_type=f"{m}+ Population", markers_positive=[m])
                        for m in channel_names
                    ]
                )

            fov_masks_map: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for sname, img, msk in fovs:
                if sname and scol is not None:
                    s_vals = raw_df[scol].astype(str).str.strip()
                    sname_norm = _normalize_sample_name(sname)
                    m_s = (
                        (s_vals == str(sname).strip())
                        | (s_vals.map(_normalize_sample_name) == sname_norm)
                    ).to_numpy()
                    fov_row_mask = row_mask & m_s
                    if not np.any(fov_row_mask) and len(fovs) == 1 and raw_df[scol].nunique(dropna=True) <= 1:
                        fov_row_mask = row_mask
                else:
                    fov_row_mask = row_mask
                if np.any(fov_row_mask):
                    fov_masks_map.append((fov_row_mask, img, msk))

            if not fov_masks_map:
                raise RuntimeError("No cells in raw_df matched the resolved FOV sample names.")

            ckpt_arg = ckpt_text if ckpt_text else "latest"
            # Avoid 0.5x magnification shrinking small test/ROI crops below U-Net bottleneck
            min_hw = min(
                min(int(img.shape[-2]), int(img.shape[-1]))
                for _, img, _ in fov_masks_map
            )
            extra_kwargs = {"dataset_magnification": 10.0} if min_hw < 64 else {}
            nimbus = NimbusPhenotyper(
                checkpoint=ckpt_arg,
                allow_download=(ckpt_arg == "latest"),
                quantile=quantile,
                positive_threshold=pos_thresh,
                test_time_aug=tta,
                clustering_method="consensus",
                n_clusters=n_clusters if ds_mode == "Consensus Clustering" else None,
                **extra_kwargs,
            )
            per_fov_results = []
            for fov_row_mask, img, msk in fov_masks_map:
                res = nimbus.predict(
                    img,
                    msk,
                    channel_names=channel_names,
                    cell_type_panel=panel,
                )
                per_fov_results.append((fov_row_mask, res))
            elapsed = time.perf_counter() - t0
            return per_fov_results, channel_names, elapsed

        def _on_done(result):
            try:
                per_fov_results, channel_names, elapsed = result
                conf_col = f"{col}_Confidence"
                pos_cols_written: List[str] = []
                written_mask = np.zeros(len(raw_df), dtype=bool)
                for fov_row_mask, res in per_fov_results:
                    self._assign_scoped_column(
                        raw_df,
                        fov_row_mask,
                        col,
                        self._align_series_to_mask(
                            raw_df, fov_row_mask, res.labels.astype(str), "Unassigned"
                        ),
                        default_value="Unassigned",
                    )
                    self._assign_scoped_column(
                        raw_df,
                        fov_row_mask,
                        conf_col,
                        self._align_series_to_mask(
                            raw_df, fov_row_mask, res.confidence.astype(float), np.nan
                        ),
                        default_value=np.nan,
                    )
                    if res.marker_positivity is not None and not res.marker_positivity.empty:
                        for marker_name in res.marker_positivity.columns:
                            p_col = f"{marker_name}_nimbus_positivity"
                            self._assign_scoped_column(
                                raw_df,
                                fov_row_mask,
                                p_col,
                                self._align_series_to_mask(
                                    raw_df,
                                    fov_row_mask,
                                    res.marker_positivity[marker_name].astype(float),
                                    np.nan,
                                ),
                                default_value=np.nan,
                            )
                            if p_col not in pos_cols_written:
                                pos_cols_written.append(p_col)
                    written_mask |= fov_row_mask

                unique_cts = [
                    str(ct)
                    for ct in raw_df.loc[written_mask, col].dropna().unique()
                    if str(ct) not in ("Unassigned", "Unknown")
                ]
                self._propagate_prediction_columns(
                    raw_df,
                    [col, conf_col, *pos_cols_written],
                    primary_col=col,
                    unique_types=unique_cts,
                    markers=channel_names,
                )
                n_classified = int(np.sum(written_mask))
                self.lbl_nimbus_status.setText(
                    f"Nimbus complete: {n_classified:,} cells ({scope_desc}) classified -> '{col}' "
                    f"({len(pos_cols_written)} positivity markers)."
                )
                self.lbl_nimbus_status.setStyleSheet(
                    "color: #66BB6A; font-weight: bold; font-size: 11px;"
                )
                iris_log(
                    "IRIS-Phenotyper",
                    "Nimbus",
                    f"Completed Nimbus ({ds_mode}) in {elapsed:.2f}s across {len(per_fov_results)} FOV(s): "
                    f"{n_classified:,} cells, {len(pos_cols_written)} positivity markers -> '{col}' (scope={scope_desc}).",
                )
            finally:
                self.btn_run_nimbus.setEnabled(True)

        def _on_error(exc: Exception):
            self.btn_run_nimbus.setEnabled(True)
            self.lbl_nimbus_status.setText(f"Nimbus error: {str(exc)[:60]}")
            self.lbl_nimbus_status.setStyleSheet("color: #E57373; font-size: 11px;")
            iris_log("IRIS-Phenotyper", "Nimbus", f"Execution error: {exc}", level="ERROR")

        self._dispatch_heavy_task(
            _compute,
            _on_done,
            _on_error,
            error_title="Nimbus Phenotyping Error",
            error_prefix="Nimbus phenotyping failed",
        )

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
        if not self._syncing_scope and hasattr(self, "sample_scope_bar") and not is_custom:
            self._syncing_scope = True
            try:
                self.sample_scope_bar.set_scope_mode(scope, emit=False)
                resolved = self.sample_scope_bar.resolve_samples()
                if hasattr(self, "manual_gating"):
                    self.manual_gating.set_scope_samples(scope, resolved, emit=False)
                self._sync_summary_label()
            finally:
                self._syncing_scope = False

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
        all_markers = (
            [
                m for m in self.manual_gating.available_markers
                if raw_df is not None and resolve_marker_col(raw_df, m) is not None
            ]
            if hasattr(self, 'manual_gating')
            else []
        )
        if not all_markers and raw_df is not None:
            mean_cols = [
                str(c) for c in raw_df.columns
                if str(c).endswith("_mean") and pd.api.types.is_numeric_dtype(raw_df[c])
            ]
            if mean_cols:
                all_markers = [c[:-5] for c in mean_cols]
            else:
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
            self.lbl_cluster_marker_count.setStyleSheet("color: #E57373; font-size: 11px;")
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
        if hasattr(self, "manual_gating"):
            raw_df = self.manual_gating.get_active_dataframe()
        if raw_df is None or raw_df.empty:
            self.lbl_cluster_status.setText("No active cell dataframe available.")
            self.lbl_cluster_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        # Determine sample cohort mask from shared sample scope (or Custom Sample Checklist)
        scol = find_sample_column(raw_df)
        scope = self.combo_cluster_scope.currentText()
        if scope == "Custom Sample Checklist":
            checked_samples = self._get_checked_clustering_samples()
            if not checked_samples:
                self.lbl_cluster_status.setText("Please check at least one sample in the checklist.")
                self.lbl_cluster_status.setStyleSheet("color: #E57373; font-size: 11px;")
                return
            if scol:
                s_vals = raw_df[scol].astype(str).str.strip()
                norms = {_normalize_sample_name(s) for s in checked_samples}
                mask_cells = (
                    s_vals.isin(checked_samples) | s_vals.map(_normalize_sample_name).isin(norms)
                ).to_numpy()
            else:
                mask_cells = np.ones(len(raw_df), dtype=bool)
            scope_desc = f"{len(checked_samples)} custom checked sample(s)"
        else:
            mask_cells, _, scope_desc, _ = self._resolve_scoped_rows_and_df(raw_df)

        markers = self._get_checked_clustering_markers()
        if not markers:
            markers = self._resolve_panel_markers(raw_df)

        marker_pairs = []
        for m in markers:
            col = resolve_marker_col(raw_df, m)
            if col is not None and pd.api.types.is_numeric_dtype(raw_df[col]):
                marker_pairs.append((m, col))

        if not marker_pairs:
            self.lbl_cluster_status.setText("Please select at least one marker for clustering.")
            self.lbl_cluster_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        display_markers = [m for m, _ in marker_pairs]
        marker_cols = [c for _, c in marker_pairs]
        self.last_clustered_markers = list(display_markers)

        n_subset = int(mask_cells.sum())
        if n_subset < 2:
            self.lbl_cluster_status.setText(f"Insufficient cells ({n_subset}) in {scope_desc}.")
            self.lbl_cluster_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        method = self.combo_cluster_method.currentText()
        knn = self.spin_leiden_knn.value()
        res = self.spin_leiden_res.value()
        k_val = self.spin_pheno_k.value()
        min_size = self.spin_pheno_min_size.value()
        kmeans_k = self.spin_kmeans_k.value()
        kmeans_batch = self.spin_kmeans_batch.value()
        sub_df = raw_df.loc[mask_cells, marker_cols].copy()

        self.btn_run_cluster.setEnabled(False)
        self.lbl_cluster_status.setText(
            f"Clustering {n_subset:,} cells across {scope_desc} "
            f"using {len(display_markers)} markers with {method}..."
        )
        self.lbl_cluster_status.setStyleSheet("color: #44AAFF; font-size: 11px;")
        iris_log(
            "IRIS-Phenotyper",
            "Clustering",
            f"Starting {method} on {n_subset:,} cells (scope={scope_desc}, {len(display_markers)} markers).",
        )

        def _compute():
            t0 = time.perf_counter()
            X = arcsinh_transform(sub_df.fillna(0).to_numpy())
            fallback_msg = None

            if "Leiden" in method:
                try:
                    import anndata as ad
                    import scanpy as sc

                    adata = ad.AnnData(X=X)
                    sc.pp.neighbors(adata, n_neighbors=min(knn, max(2, len(X) - 1)))
                    try:
                        sc.tl.leiden(
                            adata,
                            resolution=res,
                            flavor="igraph",
                            n_iterations=2,
                            directed=False,
                        )
                    except (TypeError, ValueError, Exception):
                        sc.tl.leiden(adata, resolution=res)
                    labels = adata.obs["leiden"].astype(int).to_numpy()
                except Exception as sc_err:
                    from sklearn.cluster import MiniBatchKMeans

                    k_est = max(2, min(int(round(res * 10)), 30, len(X)))
                    model = MiniBatchKMeans(
                        n_clusters=k_est,
                        random_state=42,
                        batch_size=min(1024, len(X)),
                    )
                    labels = model.fit_predict(X)
                    fallback_msg = (
                        f"Leiden fallback (Scanpy unavailable: {str(sc_err)[:30]}). "
                        f"Fitted k={k_est}."
                    )

            elif "PhenoGraph" in method:
                try:
                    import phenograph

                    communities, _graph, _q = phenograph.cluster(
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
                            sc.tl.leiden(
                                adata,
                                resolution=1.0,
                                flavor="igraph",
                                n_iterations=2,
                                directed=False,
                            )
                        except (TypeError, ValueError, Exception):
                            sc.tl.leiden(adata, resolution=1.0)
                        labels = adata.obs["leiden"].astype(int).to_numpy()
                        fallback_msg = (
                            "PhenoGraph fallback (pkg not installed). Used graph Leiden."
                        )
                    except Exception:
                        from sklearn.cluster import MiniBatchKMeans

                        model = MiniBatchKMeans(
                            n_clusters=min(8, len(X)),
                            random_state=42,
                            batch_size=min(1024, len(X)),
                        )
                        labels = model.fit_predict(X)
                except Exception:
                    from sklearn.cluster import MiniBatchKMeans

                    model = MiniBatchKMeans(
                        n_clusters=min(8, len(X)),
                        random_state=42,
                        batch_size=min(1024, len(X)),
                    )
                    labels = model.fit_predict(X)

            else:  # KMeans
                from sklearn.cluster import MiniBatchKMeans

                batch_val = min(kmeans_batch, len(X))
                model = MiniBatchKMeans(
                    n_clusters=min(kmeans_k, len(X)),
                    batch_size=batch_val,
                    random_state=42,
                )
                labels = model.fit_predict(X)

            formatted_labels = [
                f"Cluster {c}" if c >= 0 else "Noise / Outlier" for c in labels
            ]

            sub_renamed = sub_df.copy()
            sub_renamed.columns = display_markers
            m_means = sub_renamed.mean()
            m_stds = sub_renamed.std().replace(0, 1.0) + 1e-6
            z_df = (sub_renamed - m_means) / m_stds

            unique_c = sorted(
                [str(x) for x in np.unique(formatted_labels) if str(x) != "Noise / Outlier"],
                key=lambda s: (
                    (0, int(re.search(r"\d+", s).group()))
                    if re.search(r"\d+", s)
                    else (1, s)
                ),
            )
            if "Noise / Outlier" in formatted_labels:
                unique_c.append("Noise / Outlier")

            profiles = {}
            label_arr = np.array(formatted_labels)
            for c in unique_c:
                c_mask = label_arr == c
                c_cnt = int(c_mask.sum())
                c_pct = (c_cnt / len(formatted_labels)) * 100.0
                c_mean_z = z_df[c_mask].mean()
                top_z = c_mean_z.sort_values(ascending=False).head(4)
                pos_markers = [m for m, val in top_z.items() if val > 0.35]
                if not pos_markers:
                    pos_markers = list(top_z.head(2).index)
                profile_desc = ", ".join(
                    [f"{m}+ (z={c_mean_z[m]:+.1f})" for m in pos_markers[:3]]
                )
                short_names = [f"{m}+" for m in pos_markers[:2]]
                profiles[c] = {
                    "name": c,
                    "count": c_cnt,
                    "pct": c_pct,
                    "profile_desc": profile_desc,
                    "short_names": short_names,
                    "suggested_label": (
                        f"{' '.join(short_names)} Population"
                        if short_names
                        else f"{c} Population"
                    ),
                    "top_z": c_mean_z.to_dict(),
                }

            elapsed = time.perf_counter() - t0
            return formatted_labels, fallback_msg, profiles, unique_c, elapsed

        def _on_done(result):
            try:
                formatted_labels, fallback_msg, profiles, unique_c, elapsed = result

                self._assign_scoped_column(
                    raw_df, mask_cells, "Cluster", formatted_labels, default_value="Unassigned"
                )

                if self.ctx and getattr(self.ctx, "state", None) is not None:
                    if (
                        self.ctx.state.master_df is not None
                        and len(self.ctx.state.master_df) == len(raw_df)
                    ):
                        self.ctx.state.master_df["Cluster"] = raw_df["Cluster"].values
                    if (
                        getattr(self.ctx.state, "features_df", None) is not None
                        and self.ctx.state.features_df is not raw_df
                        and len(self.ctx.state.features_df) == len(raw_df)
                    ):
                        self.ctx.state.features_df["Cluster"] = raw_df["Cluster"].values
                    if hasattr(self.ctx.state, "resolved_dfs") and self.ctx.state.resolved_dfs:
                        scol_name = find_sample_column(raw_df)
                        for sname in list(self.ctx.state.resolved_dfs.keys()):
                            if scol_name:
                                m_s = (
                                    raw_df[scol_name].astype(str).str.strip()
                                    == str(sname).strip()
                                )
                                if (
                                    m_s.any()
                                    and len(self.ctx.state.resolved_dfs[sname]) == m_s.sum()
                                ):
                                    self.ctx.state.resolved_dfs[sname]["Cluster"] = (
                                        raw_df.loc[m_s, "Cluster"].values
                                    )
                    if getattr(self.ctx.state, "current_mapped_df", None) is not None:
                        if len(self.ctx.state.current_mapped_df) == len(raw_df):
                            self.ctx.state.current_mapped_df["Cluster"] = raw_df["Cluster"].values

                self.cluster_profiles = profiles
                self.list_clusters.clear()
                for c in unique_c:
                    prof = profiles[c]
                    item = QListWidgetItem(
                        f"{c}: {prof['count']:,} cells ({prof['pct']:.1f}%) "
                        f"— {', '.join(prof['short_names'])}"
                    )
                    item.setData(Qt.UserRole, c)
                    item.setData(Qt.UserRole + 1, prof)
                    self.list_clusters.addItem(item)

                if fallback_msg:
                    self.lbl_cluster_status.setText(fallback_msg)
                else:
                    self.lbl_cluster_status.setText(
                        f"{method} complete: {len(unique_c)} clusters "
                        f"across {scope_desc} ({n_subset:,} cells)."
                    )
                self.lbl_cluster_status.setStyleSheet(
                    "color: #66BB6A; font-weight: bold; font-size: 11px;"
                )
                iris_log(
                    "IRIS-Phenotyper",
                    "Clustering",
                    f"Completed {method} in {elapsed:.2f}s: {len(unique_c)} clusters across "
                    f"{n_subset:,} cells (scope={scope_desc}).",
                )

                if self.list_clusters.count() > 0:
                    self.list_clusters.setCurrentRow(0)

                if self.ctx and hasattr(self.ctx, "marker_widget") and self.ctx.marker_widget:
                    try:
                        self.ctx.marker_widget.set_data(
                            df=raw_df,
                            markers=display_markers,
                            celltype_col="Cluster",
                        )
                    except Exception:
                        pass
                self._sync_reference_columns(raw_df)
            finally:
                self.btn_run_cluster.setEnabled(True)

        def _on_error(exc: Exception):
            self.btn_run_cluster.setEnabled(True)
            self.lbl_cluster_status.setText(f"Clustering failed: {str(exc)[:45]}")
            self.lbl_cluster_status.setStyleSheet("color: #E57373; font-size: 11px;")
            iris_log("IRIS-Phenotyper", "Clustering", f"Execution error: {exc}", level="ERROR")

        self._dispatch_heavy_task(
            _compute,
            _on_done,
            _on_error,
            error_title="Clustering Error",
            error_prefix="Unsupervised clustering failed",
        )

    def _show_cluster_heatmap(self):
        raw_df = self.manual_gating.get_active_dataframe() if hasattr(self, 'manual_gating') else None
        if raw_df is None or raw_df.empty:
            self.lbl_cluster_status.setText("No active cell dataframe available.")
            self.lbl_cluster_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        if 'Cluster' not in raw_df.columns:
            self.lbl_cluster_status.setText("Run clustering first to generate cluster phenotypes.")
            self.lbl_cluster_status.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        # Use the specific markers selected for clustering
        markers = getattr(self, 'last_clustered_markers', None)
        if not markers:
            markers = self._get_checked_clustering_markers()
        if not markers:
            markers = self._resolve_panel_markers(raw_df)

        if self.ctx and hasattr(self.ctx, 'marker_widget') and self.ctx.marker_widget:
            try:
                scope = self.combo_cluster_scope.currentText()
                target_ds = SCOPE_ALL
                subset_list = None
                if scope == "Custom Sample Checklist":
                    checked = self._get_checked_clustering_samples()
                    if checked:
                        subset_list = checked
                        target_ds = SCOPE_SELECTED
                elif hasattr(self, "sample_scope_bar"):
                    target_ds = self.sample_scope_bar.get_scope_mode()
                    if target_ds == SCOPE_SELECTED:
                        subset_list = self.sample_scope_bar.get_checked_samples()

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
            self.lbl_cluster_assign_feedback.setText("Select one or more clusters first.")
            self.lbl_cluster_assign_feedback.setStyleSheet("color: #E57373; font-size: 11px;")
            return
        label_name = self.txt_cluster_pheno.text().strip() or self.txt_cluster_pheno.placeholderText().strip()
        if not label_name:
            self.lbl_cluster_assign_feedback.setText("Enter a phenotype label name.")
            self.lbl_cluster_assign_feedback.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        raw_df = self.manual_gating.get_active_dataframe() if hasattr(self, 'manual_gating') else None
        if raw_df is None or 'Cluster' not in raw_df.columns:
            self.lbl_cluster_assign_feedback.setText("Run clustering before assigning.")
            self.lbl_cluster_assign_feedback.setStyleSheet("color: #E57373; font-size: 11px;")
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
            self.lbl_cluster_assign_feedback.setText("No cells matched selected cluster(s).")
            self.lbl_cluster_assign_feedback.setStyleSheet("color: #E57373; font-size: 11px;")
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
        iris_log(
            "IRIS-Phenotyper",
            "Clustering",
            f"Assigned {cnt:,} cells from {c_targets} -> '{label_name}' ({target_col}).",
        )
        self.lbl_cluster_assign_feedback.setText(f"Assigned {cnt:,} cells from {c_targets} to '{label_name}'.")
        self.lbl_cluster_assign_feedback.setStyleSheet("color: #66BB6A; font-weight: bold; font-size: 11px;")

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
        iris_log(
            "IRIS-Phenotyper",
            "Clustering",
            f"Assigned {cnt:,} cells from heatmap {cluster_names} -> '{label_name}' ({target_col}).",
        )
        self.lbl_cluster_assign_feedback.setText(f"Assigned {cnt:,} cells from {cluster_names} to '{label_name}'.")
        self.lbl_cluster_assign_feedback.setStyleSheet("color: #66BB6A; font-weight: bold; font-size: 11px;")

    def _switch_sub_panel(self, idx: int):
        self.stack.setCurrentIndex(idx)
        self._update_nav_styles(idx)

    def _update_nav_styles(self, active_idx: int):
        for i, btn in enumerate(self.all_nav_buttons):
            btn.blockSignals(True)
            btn.setChecked(i == active_idx)
            btn.blockSignals(False)
            if i == active_idx:
                btn.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #3A6A9A; color: white; font-weight: bold;
                        border: 1px solid #5A8FCA; border-radius: 4px; padding: 5px; font-size: 11px;
                    }
                    """
                )
            else:
                btn.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #262930; color: #AAA; font-weight: bold;
                        border: 1px solid #444; border-radius: 4px; padding: 5px; font-size: 11px;
                    }
                    QPushButton:hover { background-color: #2D3748; color: #FFF; }
                    """
                )

    def set_data(self, df: Optional[pd.DataFrame] = None, markers: Optional[List[str]] = None):
        if hasattr(self, "manual_gating"):
            self.manual_gating.set_data(df, markers)
        self.sync_sample_scope(df)
        self._sync_clustering_samples()
        self._sync_clustering_markers()
        self._sync_reference_columns(df)
        self._sync_stellar_samples(df)

    def closeEvent(self, event) -> None:
        if getattr(self, "diagnostics_dialog", None) is not None:
            try:
                self.diagnostics_dialog.close()
            except Exception:
                pass
        super().closeEvent(event)
