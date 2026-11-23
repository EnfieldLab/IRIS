"""
iris_viewer.widgets.segmentation_studio_dialog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interactive full-size QC window for inspecting multi-model segmentation
disagreements, pairwise Dice/IoU matrices, zoomed hotspot patch contours,
model flickering, active mask selection, and Figure Studio export.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QAction,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from skimage.measure import find_contours

from iris_common.widgets.theme import (
    BUTTON_PRIMARY,
    BUTTON_SECONDARY,
    BUTTON_STUDIO,
    DIALOG_DARK_THEME,
    configure_tab_widget,
    create_section_card,
    style_section_header,
)
from iris_viewer.engine.curation import FINAL_CURATED_MASK_NAME, NO_CELL_CHOICE
from iris_viewer.engine.disagreement import (
    METHOD_QC_COLORS,
    _extract_masks_for_mode,
    compose_multichannel_patch_rgb,
)
from iris_viewer.widgets.disagreement_panel import (
    build_keep_button_stylesheet,
    compute_curation_progress,
    count_item_cells_for_method,
    extract_item_region_id,
    resolve_curated_state_from_ctx,
    resolve_patch_channels_from_ctx,
    resolve_state_active_image,
)

CATEGORY_ORDER = [
    "Agreed",
    "Boundary Shift",
    "Split",
    "Merge",
    "Model-Exclusive",
]

CATEGORY_COLORS = {
    "Agreed": "#34C759",
    "Boundary Shift": "#FFCC00",
    "Split": "#FF3B30",
    "Merge": "#BF5AF2",
    "Model-Exclusive": "#00E5FF",
}


class SegmentationStudioDialog(QDialog):
    """Segmentation Disagreement QC dialog for multi-model inspection and Figure Studio."""

    def __init__(
        self,
        parent=None,
        ctx=None,
        disagreement_panel=None,
        feature_panel=None,
    ):
        super().__init__(parent)
        self.ctx = ctx
        self.disagreement_panel = disagreement_panel
        self.feature_panel = feature_panel
        self.publication_dialog = None
        self._displayed_items: List[Any] = []
        self._current_hotspot_item: Any = None
        self._local_method_colors: Dict[str, str] = {}
        self._pill_widgets: List[QWidget] = []
        self._hotspot_channel_indices: Optional[List[int]] = None
        self._hotspot_channel_actions: Dict[int, QAction] = {}
        self._hotspot_marker_names: List[str] = []
        self._in_table_selection: bool = False
        self.hotspot_keep_buttons: Dict[str, QPushButton] = {}
        self._ordered_keep_methods: List[str] = []

        self.setWindowTitle("Segmentation Disagreement QC")
        self.resize(1380, 820)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setStyleSheet(DIALOG_DARK_THEME)

        self._build_ui()
        self._wire_signals()
        self.refresh_data()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Top Control & KPI Bar
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        self.lbl_qc_sample = QLabel("Sample:")
        self.combo_qc_sample = QComboBox()
        self.combo_qc_sample.setToolTip("Switch Disagreement QC sample")
        self.combo_qc_sample.currentTextChanged.connect(self._on_studio_qc_sample_changed)
        self.lbl_qc_sample.setVisible(False)
        self.combo_qc_sample.setVisible(False)
        top_bar.addWidget(self.lbl_qc_sample)
        top_bar.addWidget(self.combo_qc_sample)

        top_bar.addWidget(QLabel("Compare:"))
        self.combo_compare = QComboBox()
        self.combo_compare.addItems(["Nuclear Masks", "Cytoplasm Masks", "Both"])
        if self.disagreement_panel is not None:
            self.combo_compare.setCurrentText(
                self.disagreement_panel.combo_compare.currentText()
            )
        top_bar.addWidget(self.combo_compare)

        top_bar.addWidget(QLabel("Reference:"))
        self.combo_ref_method = QComboBox()
        top_bar.addWidget(self.combo_ref_method)

        top_bar.addWidget(QLabel("IoU:"))
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100)
        init_val = 65
        if self.disagreement_panel is not None:
            init_val = self.disagreement_panel.slider_threshold.value()
        self.slider_threshold.setValue(init_val)
        self.slider_threshold.setFixedWidth(110)
        self.slider_threshold.valueChanged.connect(self._on_slider_changed)
        top_bar.addWidget(self.slider_threshold)

        self.lbl_threshold_val = QLabel(f"{init_val / 100.0:.2f}")
        self.lbl_threshold_val.setStyleSheet(
            "color: #F0F2F5; font-weight: bold; font-size: 12px; background-color: transparent;"
        )
        self.lbl_threshold_val.setFixedWidth(36)
        top_bar.addWidget(self.lbl_threshold_val)

        self.btn_recompute = QPushButton("Recompute Disagreements")
        self.btn_recompute.setStyleSheet(BUTTON_PRIMARY)
        self.btn_recompute.clicked.connect(self._recompute_disagreements)
        top_bar.addWidget(self.btn_recompute)

        self.btn_publication_studio = QPushButton("Figure Studio")
        self.btn_publication_studio.setToolTip("Open Segmentation Disagreement Figure Studio")
        self.btn_publication_studio.setStyleSheet(BUTTON_STUDIO)
        self.btn_publication_studio.clicked.connect(self.open_publication_studio)
        top_bar.addWidget(self.btn_publication_studio)

        top_bar.addStretch(1)

        # KPI Badges
        kpi_box = QHBoxLayout()
        kpi_box.setSpacing(6)
        self.lbl_kpi_models = QLabel("Models: -")
        self.lbl_kpi_agreed = QLabel("Agreed: -")
        self.lbl_kpi_flagged = QLabel("Flagged: -")
        self.lbl_kpi_mean = QLabel("Mean IoU: -")

        badge_style = (
            "background-color: #232832; color: #E0E0E0; padding: 4px 8px; "
            "border-radius: 4px; border: 1px solid #3c4352; font-size: 11px;"
        )
        for lbl in (
            self.lbl_kpi_models,
            self.lbl_kpi_agreed,
            self.lbl_kpi_flagged,
            self.lbl_kpi_mean,
        ):
            lbl.setStyleSheet(badge_style)
            kpi_box.addWidget(lbl)
        top_bar.addLayout(kpi_box)

        main_layout.addLayout(top_bar)

        # Interactive Method Color Key Pill Bar
        self.method_key_frame = QWidget()
        self.method_key_frame.setStyleSheet(
            "background-color: #232832; border: 1px solid #3c4352; border-radius: 6px;"
        )
        self.method_key_layout = QHBoxLayout(self.method_key_frame)
        self.method_key_layout.setContentsMargins(10, 5, 10, 5)
        self.method_key_layout.setSpacing(8)
        lbl_key = QLabel("Method Color Key:")
        lbl_key.setStyleSheet(
            "color: #E0E0E0; font-size: 12px; font-weight: bold; "
            "border: none; background-color: transparent;"
        )
        self.method_key_layout.addWidget(lbl_key)
        self.pills_layout = QHBoxLayout()
        self.pills_layout.setContentsMargins(0, 0, 0, 0)
        self.pills_layout.setSpacing(6)
        self.method_key_layout.addLayout(self.pills_layout)
        self.method_key_layout.addStretch(1)
        main_layout.addWidget(self.method_key_frame)

        # Main Splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left: Disputed Regions Table + Hotspot Navigation + Flicker + Active Mask Card
        left_widget = QWidget()
        ll = QVBoxLayout(left_widget)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)

        grp_table, gl = create_section_card("Disputed Regions")

        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)
        filter_row.addWidget(QLabel("Category Filter:"))
        self.combo_category_filter = QComboBox()
        self.combo_category_filter.addItems(
            ["All Disagreements", "Split", "Merge", "Boundary Shift", "Model-Exclusive"]
        )
        self.combo_category_filter.currentTextChanged.connect(self._on_category_filter_changed)
        filter_row.addWidget(self.combo_category_filter, 1)
        gl.addLayout(filter_row)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(8)
        self.btn_prev_issue = QPushButton("Prev Hotspot")
        self.btn_prev_issue.setStyleSheet(BUTTON_SECONDARY)
        self.btn_prev_issue.clicked.connect(self._on_prev_issue)
        self.btn_next_issue = QPushButton("Next Hotspot")
        self.btn_next_issue.setStyleSheet(BUTTON_SECONDARY)
        self.btn_next_issue.clicked.connect(self._on_next_issue)
        nav_row.addWidget(self.btn_prev_issue)
        nav_row.addWidget(self.btn_next_issue)
        self.lbl_table_count = QLabel("0 regions")
        self.lbl_table_count.setStyleSheet(
            "color: #CCAA44; font-weight: bold; font-size: 11px; background-color: transparent;"
        )
        nav_row.addStretch(1)
        nav_row.addWidget(self.lbl_table_count)
        gl.addLayout(nav_row)

        self.table_issues = QTableWidget(0, 8)
        self.table_issues.setAlternatingRowColors(True)
        self.table_issues.setHorizontalHeaderLabels(
            ["#", "Region ID", "Category", "Best IoU", "Dice", "Overlap %", "Models", "Picked"]
        )
        self.table_issues.verticalHeader().setVisible(False)
        self.table_issues.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_issues.setSelectionMode(QAbstractItemView.ExtendedSelection)
        header = self.table_issues.horizontalHeader()
        for col_idx in (0, 1, 2, 3, 4, 5, 7):
            header.setSectionResizeMode(col_idx, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        self.table_issues.itemSelectionChanged.connect(self._on_table_region_selected)
        gl.addWidget(self.table_issues, 1)

        # Final Mask Curation Card
        self.grp_actions, al = create_section_card(
            "Final Mask Curation",
            "Keep per-hotspot picks in the patch inspector, auto-resolve uncurated, or batch/ROI.",
        )

        self.lbl_sel_region_info = QLabel(
            "No region selected. Click or multi-select disputed regions in the table above."
        )
        style_section_header(self.lbl_sel_region_info, subtitle=True)
        self.lbl_sel_region_info.setWordWrap(True)
        al.addWidget(self.lbl_sel_region_info)

        # Hidden legacy controls kept intact for programmatic callers
        self._legacy_base_container = QWidget()
        self._legacy_base_container.setVisible(False)
        base_row = QHBoxLayout(self._legacy_base_container)
        base_row.setContentsMargins(0, 0, 0, 0)
        self.combo_active_method = QComboBox()
        self.combo_active_method.setVisible(False)
        self.btn_set_active_mask = QPushButton("Use Selected Pipeline as Final Mask")
        self.btn_set_active_mask.setVisible(False)
        self.btn_set_active_mask.clicked.connect(self._on_set_active_mask_clicked)
        base_row.addWidget(self.combo_active_method)
        base_row.addWidget(self.btn_set_active_mask)
        al.addWidget(self._legacy_base_container)

        auto_row = QHBoxLayout()
        auto_row.setSpacing(6)
        self.btn_auto_consensus = QPushButton("Auto-Resolve Uncurated (Peer Consensus)")
        self.btn_auto_consensus.setMinimumHeight(28)
        self.btn_auto_consensus.setStyleSheet(BUTTON_PRIMARY)
        self.btn_auto_consensus.clicked.connect(self._on_auto_consensus_clicked)
        auto_row.addWidget(self.btn_auto_consensus, 1)

        self.btn_reset_curation = QPushButton("Reset to Reference")
        self.btn_reset_curation.setMinimumHeight(28)
        self.btn_reset_curation.setStyleSheet(BUTTON_SECONDARY)
        self.btn_reset_curation.clicked.connect(self._on_reset_curation_clicked)
        auto_row.addWidget(self.btn_reset_curation)
        al.addLayout(auto_row)

        override_row = QHBoxLayout()
        override_row.setSpacing(6)
        override_row.addWidget(QLabel("Batch / ROI Method:"))
        self.combo_override_method = QComboBox()
        override_row.addWidget(self.combo_override_method, 1)
        al.addLayout(override_row)

        override_btn_row1 = QHBoxLayout()
        override_btn_row1.setSpacing(6)
        self.btn_accept_selected_rows = QPushButton("Keep in Selected Rows")
        self.btn_accept_selected_rows.setMinimumHeight(26)
        self.btn_accept_selected_rows.setStyleSheet(BUTTON_SECONDARY)
        self.btn_accept_selected_rows.clicked.connect(self._on_accept_selected_rows_clicked)

        self.btn_accept_filtered = QPushButton("Keep for All Filtered")
        self.btn_accept_filtered.setMinimumHeight(26)
        self.btn_accept_filtered.setStyleSheet(BUTTON_SECONDARY)
        self.btn_accept_filtered.clicked.connect(self._on_accept_filtered_clicked)
        override_btn_row1.addWidget(self.btn_accept_selected_rows)
        override_btn_row1.addWidget(self.btn_accept_filtered)
        al.addLayout(override_btn_row1)

        override_btn_row2 = QHBoxLayout()
        override_btn_row2.setSpacing(6)
        self.btn_draw_curation_roi = QPushButton("Draw ROI")
        self.btn_draw_curation_roi.setMinimumHeight(26)
        self.btn_draw_curation_roi.setStyleSheet(BUTTON_SECONDARY)
        self.btn_draw_curation_roi.clicked.connect(self._on_draw_curation_roi_clicked)

        self.btn_apply_curation_roi = QPushButton("Keep in ROI")
        self.btn_apply_curation_roi.setMinimumHeight(26)
        self.btn_apply_curation_roi.setStyleSheet(BUTTON_SECONDARY)
        self.btn_apply_curation_roi.clicked.connect(self._on_apply_curation_roi_clicked)
        override_btn_row2.addWidget(self.btn_draw_curation_roi)
        override_btn_row2.addWidget(self.btn_apply_curation_roi)
        al.addLayout(override_btn_row2)

        self.lbl_final_mask_status = QLabel("Final Mask: None")
        style_section_header(self.lbl_final_mask_status, subtitle=True)
        self.lbl_final_mask_status.setWordWrap(True)
        al.addWidget(self.lbl_final_mask_status)

        gl.addWidget(self.grp_actions)
        ll.addWidget(grp_table)
        splitter.addWidget(left_widget)

        # Right: Analytics, Hotspot Patch Inspector & Feature Exploration Tabs
        self.right_tabs = QTabWidget()
        configure_tab_widget(self.right_tabs)
        self.right_tabs.setMinimumWidth(780)

        # Tab 0: IoU & Category Breakdown
        tab_iou = QWidget()
        ti_lyt = QVBoxLayout(tab_iou)
        ti_lyt.setContentsMargins(8, 8, 8, 8)
        ti_lyt.setSpacing(6)

        lbl_iou_desc = QLabel(
            "Per-cell IoU overlap histogram and disagreement category breakdown across models."
        )
        style_section_header(lbl_iou_desc, subtitle=True)
        ti_lyt.addWidget(lbl_iou_desc)

        self.fig_iou = Figure(figsize=(6.0, 5.0), dpi=100)
        self.fig_iou.patch.set_facecolor("#14161d")
        self.canvas_iou = FigureCanvas(self.fig_iou)
        self.canvas_iou.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ti_lyt.addWidget(self.canvas_iou, 1)
        self.right_tabs.addTab(tab_iou, "IoU & Category Breakdown")

        # Tab 1: Hotspot Patch Inspector
        tab_hotspot = QWidget()
        th_lyt = QVBoxLayout(tab_hotspot)
        th_lyt.setContentsMargins(8, 8, 8, 8)
        th_lyt.setSpacing(6)

        lbl_hs_desc = QLabel(
            "Zoomed patch crop around the selected hotspot showing multi-channel fluorescence "
            "and each method's labeled contour."
        )
        style_section_header(lbl_hs_desc, subtitle=True)
        lbl_hs_desc.setWordWrap(True)
        th_lyt.addWidget(lbl_hs_desc)

        hs_ctrl_row = QHBoxLayout()
        hs_ctrl_row.setSpacing(8)
        hs_ctrl_row.addWidget(QLabel("Markers:"))
        self.btn_hotspot_channels = QPushButton("Markers: Active")
        self.btn_hotspot_channels.setStyleSheet(BUTTON_SECONDARY)
        self.btn_hotspot_channels.setToolTip(
            "Select which immunofluorescence marker channels to composite under contours"
        )
        self.menu_hotspot_channels = QMenu(self.btn_hotspot_channels)
        self.btn_hotspot_channels.setMenu(self.menu_hotspot_channels)
        hs_ctrl_row.addWidget(self.btn_hotspot_channels)

        hs_ctrl_row.addWidget(QLabel("View:"))
        self.combo_hotspot_view = QComboBox()
        self.combo_hotspot_view.addItems(["Composite Overlay", "Side-by-Side Per Method"])
        self.combo_hotspot_view.currentIndexChanged.connect(self._on_hotspot_controls_changed)
        hs_ctrl_row.addWidget(self.combo_hotspot_view)

        hs_ctrl_row.addWidget(QLabel("Contour:"))
        self.spin_hotspot_lw = QDoubleSpinBox()
        self.spin_hotspot_lw.setRange(0.5, 5.0)
        self.spin_hotspot_lw.setSingleStep(0.5)
        self.spin_hotspot_lw.setValue(2.2)
        self.spin_hotspot_lw.setSuffix(" px")
        self.spin_hotspot_lw.valueChanged.connect(self._on_hotspot_controls_changed)
        hs_ctrl_row.addWidget(self.spin_hotspot_lw)

        self.chk_hotspot_autocontrast = QCheckBox("Local Contrast")
        self.chk_hotspot_autocontrast.setChecked(True)
        self.chk_hotspot_autocontrast.setToolTip(
            "Normalize each marker to 1-99.5% patch intensity so dim markers remain clear"
        )
        self.chk_hotspot_autocontrast.toggled.connect(self._on_hotspot_controls_changed)
        hs_ctrl_row.addWidget(self.chk_hotspot_autocontrast)
        hs_ctrl_row.addStretch(1)
        th_lyt.addLayout(hs_ctrl_row)

        self.fig_hotspot = Figure(figsize=(6.0, 5.0), dpi=100)
        self.fig_hotspot.patch.set_facecolor("#14161d")
        self.canvas_hotspot = FigureCanvas(self.fig_hotspot)
        self.canvas_hotspot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        th_lyt.addWidget(self.canvas_hotspot, 1)

        # Interactive Bottom Method-Selection Bar for Hotspot Patch Inspector
        self.hotspot_curation_bar = QFrame()
        self.hotspot_curation_bar.setStyleSheet(
            "QFrame { background-color: #1b1f27; border: 1px solid #3c4352; "
            "border-radius: 6px; }"
        )
        hb_lyt = QVBoxLayout(self.hotspot_curation_bar)
        hb_lyt.setContentsMargins(10, 8, 10, 8)
        hb_lyt.setSpacing(6)

        hb_hdr_row = QHBoxLayout()
        hb_hdr_row.setSpacing(10)
        lbl_keep_hdr = QLabel("Keep segmentation for this hotspot:")
        lbl_keep_hdr.setStyleSheet(
            "color: #F0F2F5; font-size: 12px; font-weight: bold; border: none; "
            "background-color: transparent;"
        )
        hb_hdr_row.addWidget(lbl_keep_hdr)

        self.lbl_curation_progress = QLabel("Curated: 0 / 0")
        self.lbl_curation_progress.setStyleSheet(
            "color: #CCAA44; font-size: 11px; font-weight: bold; border: none; "
            "background-color: transparent;"
        )
        hb_hdr_row.addWidget(self.lbl_curation_progress)
        hb_hdr_row.addStretch(1)

        self.chk_auto_advance = QCheckBox("Auto-advance")
        self.chk_auto_advance.setChecked(True)
        self.chk_auto_advance.setFocusPolicy(Qt.NoFocus)
        self.chk_auto_advance.setStyleSheet("border: none; background-color: transparent;")
        hb_hdr_row.addWidget(self.chk_auto_advance)

        self.btn_skip_hotspot = QPushButton("Skip ->")
        self.btn_skip_hotspot.setMinimumHeight(26)
        self.btn_skip_hotspot.setFocusPolicy(Qt.NoFocus)
        self.btn_skip_hotspot.setStyleSheet(BUTTON_SECONDARY)
        self.btn_skip_hotspot.setToolTip("Advance to the next hotspot without editing [Space]")
        self.btn_skip_hotspot.clicked.connect(self._on_skip_hotspot_clicked)
        hb_hdr_row.addWidget(self.btn_skip_hotspot)
        hb_lyt.addLayout(hb_hdr_row)

        hb_btns_row = QHBoxLayout()
        hb_btns_row.setSpacing(8)
        self.hotspot_keep_grid = QGridLayout()
        self.hotspot_keep_grid.setContentsMargins(0, 0, 0, 0)
        self.hotspot_keep_grid.setSpacing(6)
        hb_btns_row.addLayout(self.hotspot_keep_grid)

        self.btn_keep_no_cell = QPushButton("[0] No Cell / Artifact")
        self.btn_keep_no_cell.setMinimumHeight(28)
        self.btn_keep_no_cell.setFocusPolicy(Qt.NoFocus)
        self.btn_keep_no_cell.setToolTip(
            "Remove false-positive / artifact cells in this hotspot [0]"
        )
        self.btn_keep_no_cell.clicked.connect(
            lambda _checked=False: self._on_hotspot_keep_method_clicked(NO_CELL_CHOICE)
        )
        hb_btns_row.addWidget(self.btn_keep_no_cell)
        hb_btns_row.addStretch(1)
        hb_lyt.addLayout(hb_btns_row)

        th_lyt.addWidget(self.hotspot_curation_bar)
        self.right_tabs.addTab(tab_hotspot, "Hotspot Patch Inspector")

        # Tab 2: Pairwise Dice & IoU Matrix
        tab_matrix = QWidget()
        tm_lyt = QVBoxLayout(tab_matrix)
        tm_lyt.setContentsMargins(8, 8, 8, 8)
        tm_lyt.setSpacing(6)

        self.fig_matrix = Figure(figsize=(6.5, 3.6), dpi=100)
        self.fig_matrix.patch.set_facecolor("#14161d")
        self.canvas_matrix = FigureCanvas(self.fig_matrix)
        self.canvas_matrix.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tm_lyt.addWidget(self.canvas_matrix, 2)

        self.table_model_summary = QTableWidget(0, 4)
        self.table_model_summary.setAlternatingRowColors(True)
        self.table_model_summary.setHorizontalHeaderLabels(
            ["Method", "Cell Count", "Median Area (px)", "Mean Peer IoU"]
        )
        self.table_model_summary.verticalHeader().setVisible(False)
        self.table_model_summary.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tm_lyt.addWidget(self.table_model_summary, 1)
        self.right_tabs.addTab(tab_matrix, "Pairwise Dice & IoU Matrix")

        # Tab 3: Extracted Features
        tab_feat = QWidget()
        tfl = QVBoxLayout(tab_feat)
        tfl.setContentsMargins(8, 8, 8, 8)
        tfl.setSpacing(6)

        self.lbl_feat_summary = QLabel(
            "Preview of extracted single-cell morphological & intensity features."
        )
        style_section_header(self.lbl_feat_summary, subtitle=True)
        tfl.addWidget(self.lbl_feat_summary)

        self.table_features = QTableWidget(0, 0)
        self.table_features.setAlternatingRowColors(True)
        self.table_features.verticalHeader().setVisible(False)
        tfl.addWidget(self.table_features, 1)

        f_btn_row = QHBoxLayout()
        f_btn_row.setSpacing(8)
        self.btn_extract_feat = QPushButton("Extract Features Now")
        self.btn_extract_feat.setStyleSheet(BUTTON_PRIMARY)
        self.btn_extract_feat.clicked.connect(self._on_extract_features_clicked)
        f_btn_row.addWidget(self.btn_extract_feat)
        self.btn_export_feat = QPushButton("Export CSV")
        self.btn_export_feat.setStyleSheet(BUTTON_PRIMARY)
        self.btn_export_feat.clicked.connect(self._on_export_features_clicked)
        f_btn_row.addWidget(self.btn_export_feat)
        f_btn_row.addStretch(1)
        tfl.addLayout(f_btn_row)

        self.right_tabs.addTab(tab_feat, "Extracted Features")

        splitter.addWidget(self.right_tabs)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter, 1)

    def _wire_signals(self) -> None:
        if self.disagreement_panel is not None:
            self.disagreement_panel.statisticsUpdated.connect(self._on_stats_updated)
            self.disagreement_panel.regionReviewShown.connect(self._on_region_review_shown)
            if hasattr(self.disagreement_panel, "methodColorsChanged"):
                self.disagreement_panel.methodColorsChanged.connect(
                    lambda _c: self._on_method_colors_synced()
                )
            if hasattr(self.disagreement_panel, "selectedMethodsChanged"):
                self.disagreement_panel.selectedMethodsChanged.connect(
                    lambda _m: self._on_method_colors_synced()
                )
            if hasattr(self.disagreement_panel, "curationStateChanged"):
                self.disagreement_panel.curationStateChanged.connect(
                    self._on_curation_state_synced
                )
        if self.feature_panel is not None:
            self.feature_panel.dataExtracted.connect(self._populate_feature_table)

    # Helpers

    def _get_report(self) -> Any:
        if self.ctx is not None and hasattr(self.ctx, "state") and self.ctx.state is not None:
            return getattr(self.ctx.state, "disagreement_report", None)
        return None

    def _get_available_methods(self) -> List[str]:
        methods: List[str] = []
        if self.ctx is not None and hasattr(self.ctx, "state") and self.ctx.state is not None:
            methods = list(getattr(self.ctx.state, "segmentation_results", {}).keys())
        report = self._get_report()
        if not methods and report is not None:
            methods = list(getattr(report, "methods", []))
        if not methods and self.disagreement_panel is not None:
            methods = list(getattr(self.disagreement_panel, "_methods", []))
        return [m for m in methods if m != FINAL_CURATED_MASK_NAME]

    def _get_method_colors(self) -> Dict[str, str]:
        methods = self._get_available_methods()
        dp_colors: Dict[str, str] = {}
        if self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "get_method_colors"
        ):
            dp_colors = self.disagreement_panel.get_method_colors()

        resolved: Dict[str, str] = {}
        for idx, m_name in enumerate(methods):
            if m_name in dp_colors:
                resolved[m_name] = dp_colors[m_name]
            elif m_name in self._local_method_colors:
                resolved[m_name] = self._local_method_colors[m_name]
            else:
                resolved[m_name] = METHOD_QC_COLORS[idx % len(METHOD_QC_COLORS)]
        return resolved

    def _rebuild_method_color_pills(self) -> None:
        for w in self._pill_widgets:
            self.pills_layout.removeWidget(w)
            w.deleteLater()
        self._pill_widgets.clear()

        methods = self._get_available_methods()
        colors = self._get_method_colors()
        if not methods:
            empty_lbl = QLabel("No segmentation methods loaded")
            empty_lbl.setStyleSheet(
                "color: #9AA4B2; font-size: 11px; font-style: italic; "
                "border: none; background-color: transparent;"
            )
            self.pills_layout.addWidget(empty_lbl)
            self._pill_widgets.append(empty_lbl)
            return

        selected_set: Optional[set[str]] = None
        if self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "get_selected_methods"
        ):
            sel = self.disagreement_panel.get_selected_methods()
            if getattr(self.disagreement_panel, "_methods", []):
                selected_set = set(sel)

        for m_name in methods:
            hex_c = colors.get(m_name, METHOD_QC_COLORS[0])
            pill = QWidget()
            is_selected = selected_set is None or m_name in selected_set
            pill.setStyleSheet(
                "QWidget { background-color: #1b1f27; border: 1px solid #4B5364; "
                "border-radius: 4px; }"
                if is_selected
                else "QWidget { background-color: #1a1d24; border: 1px dashed #3c4352; "
                "border-radius: 4px; }"
            )
            pl = QHBoxLayout(pill)
            pl.setContentsMargins(6, 2, 8, 2)
            pl.setSpacing(5)

            btn_swatch = QPushButton()
            btn_swatch.setFixedSize(12, 12)
            btn_swatch.setCursor(Qt.PointingHandCursor)
            btn_swatch.setToolTip(f"Click to change color for {m_name}")
            btn_swatch.setStyleSheet(
                f"QPushButton {{ background-color: {hex_c}; border: 1px solid #E0E0E0; "
                "border-radius: 2px; padding: 0px; }"
            )
            btn_swatch.clicked.connect(
                lambda _checked=False, name=m_name: self._on_pill_color_clicked(name)
            )
            pl.addWidget(btn_swatch)

            lbl = QLabel(m_name if is_selected else f"{m_name} (off)")
            lbl.setStyleSheet(
                "color: #F0F2F5; font-size: 11px; font-weight: bold; "
                "border: none; background-color: transparent;"
                if is_selected
                else "color: #777777; font-size: 11px; border: none; "
                "background-color: transparent;"
            )
            pl.addWidget(lbl)

            self.pills_layout.addWidget(pill)
            self._pill_widgets.append(pill)

    def _on_pill_color_clicked(self, method_name: str) -> None:
        cur_hex = self._get_method_colors().get(method_name, METHOD_QC_COLORS[0])
        chosen = QColorDialog.getColor(
            QColor(cur_hex), self, f"Select QC Color: {method_name}"
        )
        if not chosen.isValid():
            return
        new_hex = chosen.name()
        self._local_method_colors[method_name] = new_hex
        if self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "set_method_color"
        ):
            self.disagreement_panel.set_method_color(method_name, new_hex, emit=True)
        self._rebuild_method_color_pills()
        self._draw_hotspot_patch(self._current_hotspot_item)

    def _on_method_colors_synced(self) -> None:
        self._rebuild_method_color_pills()
        self._sync_override_combo()
        self._refresh_hotspot_curation_bar()
        self._draw_hotspot_patch(self._active_hotspot_item())

    def _on_curation_state_synced(self) -> None:
        self._update_table_picked_column()
        self._refresh_hotspot_curation_bar()
        self._sync_final_mask_status()

    def _sync_sample_combo(self) -> None:
        samples: List[str] = []
        active_sample: Optional[str] = None
        if self.disagreement_panel is not None:
            samples = list(getattr(self.disagreement_panel, "_qc_samples", []) or [])
            if not samples and hasattr(self.disagreement_panel, "combo_qc_sample"):
                cb = self.disagreement_panel.combo_qc_sample
                samples = [cb.itemText(i) for i in range(cb.count()) if cb.itemText(i).strip()]
            if hasattr(self.disagreement_panel, "get_qc_sample"):
                active_sample = self.disagreement_panel.get_qc_sample() or None
        if not samples and self.ctx is not None and getattr(self.ctx, "state", None) is not None:
            br = getattr(self.ctx.state, "batch_results", {}) or {}
            samples = [
                sn
                for sn, entry in br.items()
                if isinstance(entry, dict)
                and any(
                    isinstance(v, dict)
                    and (v.get("nuclear") is not None or v.get("cytoplasm") is not None)
                    for v in entry.values()
                )
            ]
            active_sample = getattr(self.ctx.state, "current_sample_name", None)

        show_sample = len(samples) > 1
        self.combo_qc_sample.blockSignals(True)
        self.combo_qc_sample.clear()
        self.combo_qc_sample.addItems(samples)
        if active_sample and active_sample in samples:
            self.combo_qc_sample.setCurrentText(active_sample)
        elif samples:
            self.combo_qc_sample.setCurrentIndex(0)
        self.combo_qc_sample.blockSignals(False)
        self.lbl_qc_sample.setVisible(show_sample)
        self.combo_qc_sample.setVisible(show_sample)

    def _on_studio_qc_sample_changed(self, sample_name: str) -> None:
        clean = str(sample_name).strip()
        if not clean:
            return
        if self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "combo_qc_sample"
        ):
            if self.disagreement_panel.combo_qc_sample.currentText() != clean:
                self.disagreement_panel.combo_qc_sample.setCurrentText(clean)
        self._current_hotspot_item = None
        self.refresh_data()

    def _sync_method_combos(self) -> None:
        methods = self._get_available_methods()
        report = self._get_report()
        active = None
        if self.ctx is not None and getattr(self.ctx, "state", None) is not None:
            cand_active = self.ctx.state.active_segmentation_method
            if cand_active and cand_active != FINAL_CURATED_MASK_NAME:
                active = cand_active
        if not active and self.disagreement_panel is not None:
            active = self.disagreement_panel.get_active_method()

        ref = getattr(report, "reference_method", None) if report is not None else None
        if ref == FINAL_CURATED_MASK_NAME:
            ref = None
        if not ref and self.disagreement_panel is not None:
            ref = self.disagreement_panel.get_reference_method()

        override_pref = None
        if self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "combo_override_method"
        ):
            override_pref = self.disagreement_panel.combo_override_method.currentText().strip()
        if not override_pref:
            override_pref = (
                next((m for m in methods if m != ref), methods[0]) if methods else None
            )

        for combo, preferred in (
            (self.combo_ref_method, ref),
            (self.combo_active_method, active),
        ):
            curr = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(methods)
            target = preferred if preferred in methods else (curr if curr in methods else "")
            if target:
                combo.setCurrentText(target)
            elif methods:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

        self._sync_override_combo(ref=ref, override_pref=override_pref)

    def _get_checked_active_methods(self) -> List[str]:
        all_methods = self._get_available_methods()
        if self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "get_selected_methods"
        ):
            if getattr(self.disagreement_panel, "_methods", []):
                return [
                    m
                    for m in self.disagreement_panel.get_selected_methods()
                    if m != FINAL_CURATED_MASK_NAME
                ]
        return all_methods

    def _sync_override_combo(
        self,
        *,
        ref: Optional[str] = None,
        override_pref: Optional[str] = None,
    ) -> None:
        checked = self._get_checked_active_methods()
        pool = checked if checked else self._get_available_methods()
        items = list(pool)
        if NO_CELL_CHOICE not in items:
            items.append(NO_CELL_CHOICE)

        curr = self.combo_override_method.currentText().strip()
        if override_pref is None and self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "combo_override_method"
        ):
            override_pref = self.disagreement_panel.combo_override_method.currentText().strip()
        if ref is None:
            ref = self.combo_ref_method.currentText().strip()

        self.combo_override_method.blockSignals(True)
        self.combo_override_method.clear()
        self.combo_override_method.addItems(items)
        if curr in items:
            self.combo_override_method.setCurrentText(curr)
        elif override_pref in items:
            self.combo_override_method.setCurrentText(override_pref)
        elif pool:
            alt = next((m for m in pool if m != ref), pool[0])
            self.combo_override_method.setCurrentText(alt)
        elif items:
            self.combo_override_method.setCurrentIndex(0)
        self.combo_override_method.blockSignals(False)

    # Event Handlers

    def _on_slider_changed(self, int_val: int) -> None:
        val = int_val / 100.0
        self.lbl_threshold_val.setText(f"{val:.2f}")
        if self.disagreement_panel is not None:
            self.disagreement_panel.slider_threshold.blockSignals(True)
            self.disagreement_panel.slider_threshold.setValue(int_val)
            self.disagreement_panel.lbl_threshold_value.setText(f"{val:.2f}")
            self.disagreement_panel.slider_threshold.blockSignals(False)
            self.disagreement_panel.thresholdChanged.emit(val)
        self._draw_iou_and_category_plot()
        self._populate_issues_table()

    def _on_category_filter_changed(self, category: str) -> None:
        if self.disagreement_panel is not None:
            self.disagreement_panel.combo_category_filter.blockSignals(True)
            self.disagreement_panel.combo_category_filter.setCurrentText(category)
            self.disagreement_panel.combo_category_filter.blockSignals(False)
            self.disagreement_panel.categoryFilterChanged.emit(category)
        self._populate_issues_table()

    def _recompute_disagreements(self) -> None:
        if self.disagreement_panel is not None:
            self.disagreement_panel.combo_compare.setCurrentText(
                self.combo_compare.currentText()
            )
            ref = self.combo_ref_method.currentText()
            if ref:
                self.disagreement_panel.combo_ref_method.setCurrentText(ref)
            self.disagreement_panel.computeDisagreementRequested.emit()

    def _on_stats_updated(
        self, _total: int, _agreed: int, _flagged: int, _mean_iou: float
    ) -> None:
        self.refresh_data()

    def _on_region_review_shown(self, item: Any) -> None:
        if item is None:
            return
        self._current_hotspot_item = item
        selected_rows = {idx.row() for idx in self.table_issues.selectedIndexes()}
        if not self._in_table_selection:
            region_id = (
                item.get("region_id")
                if isinstance(item, dict)
                else getattr(item, "region_id", None)
            )
            for r in range(self.table_issues.rowCount()):
                cell_item = self.table_issues.item(r, 1)
                if cell_item and cell_item.text() == str(region_id):
                    self.table_issues.blockSignals(True)
                    self.table_issues.selectRow(r)
                    self.table_issues.blockSignals(False)
                    selected_rows = {r}
                    break
        self._update_region_card(item, selected_count=max(1, len(selected_rows)))
        self._draw_hotspot_patch(item)
        self._refresh_hotspot_curation_bar()

    def _on_prev_issue(self) -> None:
        if self.ctx is not None and getattr(self.ctx, "review_svc", None) is not None:
            self.ctx.review_svc.prev_issue()

    def _on_next_issue(self) -> None:
        if self.ctx is not None and getattr(self.ctx, "review_svc", None) is not None:
            self.ctx.review_svc.next_issue()

    def _on_table_region_selected(self) -> None:
        selected_rows = sorted({idx.row() for idx in self.table_issues.selectedIndexes()})
        if not selected_rows:
            return
        curr_row = self.table_issues.currentRow()
        if (
            curr_row < 0
            or curr_row >= len(self._displayed_items)
            or curr_row not in selected_rows
        ):
            curr_row = selected_rows[-1]
        if curr_row < 0 or curr_row >= len(self._displayed_items):
            return
        item = self._displayed_items[curr_row]
        self._current_hotspot_item = item
        self._update_region_card(item, selected_count=len(selected_rows))
        self._draw_hotspot_patch(item)
        self._refresh_hotspot_curation_bar()

        if self.ctx is not None and getattr(self.ctx, "review_svc", None) is not None:
            self._in_table_selection = True
            try:
                rid = getattr(item, "region_id", None)
                if hasattr(self.ctx.review_svc, "jump_to_region_id") and rid is not None:
                    self.ctx.review_svc.jump_to_region_id(int(rid))
                else:
                    self.ctx.review_svc.jump_to_issue(curr_row)
            finally:
                self._in_table_selection = False

    def _update_region_card(self, item: Any, selected_count: int = 1) -> None:
        if item is None:
            self.lbl_sel_region_info.setText(
                "No region selected. Click or multi-select disputed regions in the table above."
            )
            return
        if isinstance(item, dict):
            rid = item.get("region_id", 0)
            cat = item.get("category", "Unknown")
            iou = float(item.get("best_iou", 0.0))
            dice = float(item.get("dice", 0.0))
            ov = float(item.get("overlap_coeff", 0.0))
            m_ids = dict(item.get("method_cell_ids", {}))
        else:
            rid = getattr(item, "region_id", 0)
            cat = getattr(item, "category", "Unknown")
            iou = float(getattr(item, "best_iou", 0.0))
            dice = float(getattr(item, "dice", 0.0))
            ov = float(getattr(item, "overlap_coeff", 0.0))
            m_ids = dict(getattr(item, "method_cell_ids", {}))

        sel_tag = (
            f" &nbsp;|&nbsp; <b>Selected:</b> {selected_count} rows"
            if selected_count > 1
            else ""
        )
        models_txt = ", ".join(f"{m}: {ids}" for m, ids in sorted(m_ids.items())) or "None"
        self.lbl_sel_region_info.setText(
            f"<b>Region {rid}</b> ({cat}){sel_tag} &nbsp;|&nbsp; "
            f"<b>IoU:</b> {iou:.3f} &nbsp;|&nbsp; <b>Dice:</b> {dice:.3f} &nbsp;|&nbsp; "
            f"<b>Overlap:</b> {ov * 100.0:.0f}%<br>"
            f"<b>Models:</b> {models_txt}"
        )

    def _sync_final_mask_status(self) -> None:
        txt = "Final Mask: None"
        if self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "lbl_final_mask_status"
        ):
            dp_txt = self.disagreement_panel.lbl_final_mask_status.text().strip()
            if dp_txt:
                txt = dp_txt
        elif self.ctx is not None and getattr(self.ctx, "state", None) is not None:
            curated = getattr(self.ctx.state, "curated_mask_state", None)
            if curated is not None:
                n_cells = int(np.sum(np.unique(curated.nuclear) > 0))
                txt = (
                    f"Final Mask: {FINAL_CURATED_MASK_NAME} ({n_cells:,} cells | "
                    f"base={curated.base_method} | {curated.curated_regions_count} curated)"
                )
        self.lbl_final_mask_status.setText(txt)

    def _on_set_active_mask_clicked(self) -> None:
        method = self.combo_active_method.currentText().strip()
        if not method:
            return
        review_svc = getattr(self.ctx, "review_svc", None) if self.ctx is not None else None
        if review_svc is not None and hasattr(review_svc, "use_pipeline_as_final_mask"):
            review_svc.use_pipeline_as_final_mask(method)
        elif self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "useSingleModelRequested"
        ):
            self.disagreement_panel.useSingleModelRequested.emit(method)
        else:
            if self.ctx is not None and getattr(self.ctx, "state", None) is not None:
                self.ctx.state.set_active_segmentation_method(method)
            if review_svc is not None and hasattr(review_svc, "set_active_mask_layers"):
                review_svc.set_active_mask_layers(method)
            if self.feature_panel is not None:
                self.feature_panel.update_mask_sources(
                    self._get_available_methods(), active=method
                )
        if self.disagreement_panel is not None:
            self.disagreement_panel.combo_active_method.blockSignals(True)
            self.disagreement_panel.combo_active_method.setCurrentText(method)
            self.disagreement_panel.combo_active_method.blockSignals(False)
        self._sync_final_mask_status()

    def _on_auto_consensus_clicked(self) -> None:
        review_svc = getattr(self.ctx, "review_svc", None) if self.ctx is not None else None
        if review_svc is not None and hasattr(review_svc, "auto_adjudicate_final_mask"):
            review_svc.auto_adjudicate_final_mask()
        elif self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "autoConsensusRequested"
        ):
            self.disagreement_panel.autoConsensusRequested.emit()
        self._update_table_picked_column()
        self._refresh_hotspot_curation_bar()
        self._sync_final_mask_status()

    def _on_reset_curation_clicked(self) -> None:
        review_svc = getattr(self.ctx, "review_svc", None) if self.ctx is not None else None
        if review_svc is not None and hasattr(review_svc, "reset_curated_mask_to_reference"):
            review_svc.reset_curated_mask_to_reference()
        elif self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "resetCurationRequested"
        ):
            self.disagreement_panel.resetCurationRequested.emit()
        self._update_table_picked_column()
        self._refresh_hotspot_curation_bar()
        self._sync_final_mask_status()

    def _on_skip_hotspot_clicked(self) -> None:
        review_svc = getattr(self.ctx, "review_svc", None) if self.ctx is not None else None
        if review_svc is not None and hasattr(review_svc, "next_issue"):
            review_svc.next_issue()
        elif self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "skipHotspotRequested"
        ):
            self.disagreement_panel.skipHotspotRequested.emit()

    def _on_hotspot_keep_method_clicked(self, choice: str) -> None:
        target_item = self._active_hotspot_item()
        auto_adv = bool(self.chk_auto_advance.isChecked())
        review_svc = getattr(self.ctx, "review_svc", None) if self.ctx is not None else None
        if review_svc is not None and hasattr(review_svc, "curate_hotspot_and_advance"):
            review_svc.curate_hotspot_and_advance(
                choice,
                item=target_item,
                auto_advance=auto_adv,
            )
        elif self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "hotspotKeepRequested"
        ):
            self.disagreement_panel.hotspotKeepRequested.emit(choice, auto_adv)
        self._update_table_picked_column()
        self._refresh_hotspot_curation_bar()
        self._sync_final_mask_status()

    def _refresh_hotspot_curation_bar(self) -> None:
        checked = self._get_checked_active_methods()
        item = self._active_hotspot_item()
        rid = extract_item_region_id(item)
        curated = resolve_curated_state_from_ctx(self.ctx)
        decisions = (
            getattr(curated, "region_decisions", {}) or {} if curated is not None else {}
        )
        decision = decisions.get(rid) if rid is not None else None

        state = getattr(self.ctx, "state", None) if self.ctx is not None else None
        seg_results = getattr(state, "segmentation_results", {}) or {} if state is not None else {}
        compare_mode = (
            self.combo_compare.currentText() if hasattr(self, "combo_compare") else "Nuclear Masks"
        )
        colors = self._get_method_colors()

        if checked != self._ordered_keep_methods:
            for btn in self.hotspot_keep_buttons.values():
                self.hotspot_keep_grid.removeWidget(btn)
                btn.deleteLater()
            self.hotspot_keep_buttons.clear()
            self._ordered_keep_methods = list(checked)
            max_cols = 4
            for idx, m_name in enumerate(self._ordered_keep_methods):
                btn = QPushButton()
                btn.setMinimumHeight(28)
                btn.setFocusPolicy(Qt.NoFocus)
                btn.setToolTip(f"Keep '{m_name}' segmentation for this hotspot [{idx + 1}]")
                btn.clicked.connect(
                    lambda _checked=False, m=m_name: self._on_hotspot_keep_method_clicked(m)
                )
                self.hotspot_keep_grid.addWidget(btn, idx // max_cols, idx % max_cols)
                self.hotspot_keep_buttons[m_name] = btn

        for idx, m_name in enumerate(self._ordered_keep_methods):
            btn = self.hotspot_keep_buttons[m_name]
            n_cells = count_item_cells_for_method(item, m_name, seg_results, compare_mode)
            is_picked = decision == m_name
            suffix = " [Picked]" if is_picked else ""
            btn.setText(f"[{idx + 1}] Keep {m_name} ({n_cells}){suffix}")
            hex_c = colors.get(m_name, METHOD_QC_COLORS[idx % len(METHOD_QC_COLORS)])
            btn.setStyleSheet(build_keep_button_stylesheet(hex_c, active=is_picked))
            btn.setEnabled(item is not None)

        is_no_cell = decision == NO_CELL_CHOICE
        self.btn_keep_no_cell.setText(
            "[0] No Cell / Artifact" + (" [Picked]" if is_no_cell else "")
        )
        self.btn_keep_no_cell.setStyleSheet(
            build_keep_button_stylesheet("#CC4444", active=is_no_cell, is_no_cell=True)
        )
        self.btn_keep_no_cell.setEnabled(item is not None)

        cur_x, tot_y = compute_curation_progress(
            self.ctx, curated, fallback_items=self._displayed_items
        )
        self.lbl_curation_progress.setText(f"Curated: {cur_x} / {tot_y}")

    def _on_accept_selected_rows_clicked(self) -> None:
        source_method = self.combo_override_method.currentText().strip()
        if not source_method:
            return
        selected_rows = sorted({idx.row() for idx in self.table_issues.selectedIndexes()})
        items = [
            self._displayed_items[r]
            for r in selected_rows
            if 0 <= r < len(self._displayed_items)
        ]
        if not items and self._current_hotspot_item is not None:
            items = [self._current_hotspot_item]
        if not items:
            return
        review_svc = getattr(self.ctx, "review_svc", None) if self.ctx is not None else None
        if review_svc is not None and hasattr(review_svc, "apply_hotspot_curation"):
            review_svc.apply_hotspot_curation(
                source_method, items=items, use_filtered_category=False
            )
        elif self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "overrideHotspotsRequested"
        ):
            self.disagreement_panel.overrideHotspotsRequested.emit(source_method, items)
        self._update_table_picked_column()
        self._refresh_hotspot_curation_bar()
        self._sync_final_mask_status()

    def _on_accept_filtered_clicked(self) -> None:
        source_method = self.combo_override_method.currentText().strip()
        if not source_method:
            return
        items = list(self._displayed_items) if self._displayed_items else None
        review_svc = getattr(self.ctx, "review_svc", None) if self.ctx is not None else None
        if review_svc is not None and hasattr(review_svc, "apply_hotspot_curation"):
            review_svc.apply_hotspot_curation(
                source_method, items=items, use_filtered_category=True
            )
        elif self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "overrideHotspotsRequested"
        ):
            self.disagreement_panel.overrideHotspotsRequested.emit(source_method, None)
        self._update_table_picked_column()
        self._refresh_hotspot_curation_bar()
        self._sync_final_mask_status()

    def _on_draw_curation_roi_clicked(self) -> None:
        review_svc = getattr(self.ctx, "review_svc", None) if self.ctx is not None else None
        if review_svc is not None and hasattr(review_svc, "draw_curation_roi"):
            review_svc.draw_curation_roi()
        elif self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "drawCurationRoiRequested"
        ):
            self.disagreement_panel.drawCurationRoiRequested.emit()

    def _on_apply_curation_roi_clicked(self) -> None:
        source_method = self.combo_override_method.currentText().strip()
        if not source_method:
            return
        review_svc = getattr(self.ctx, "review_svc", None) if self.ctx is not None else None
        if review_svc is not None and hasattr(review_svc, "apply_roi_curation"):
            review_svc.apply_roi_curation(source_method)
        elif self.disagreement_panel is not None and hasattr(
            self.disagreement_panel, "applyCurationRoiRequested"
        ):
            self.disagreement_panel.applyCurationRoiRequested.emit(source_method)
        self._update_table_picked_column()
        self._refresh_hotspot_curation_bar()
        self._sync_final_mask_status()

    # Refresh & Plotting

    def _active_hotspot_item(self) -> Any:
        if self._current_hotspot_item is not None:
            return self._current_hotspot_item
        return self._displayed_items[0] if self._displayed_items else None

    def _on_hotspot_controls_changed(self, *_args: Any) -> None:
        self._draw_hotspot_patch(self._active_hotspot_item())

    def _update_hotspot_channel_button_text(self) -> None:
        markers = self._hotspot_marker_names
        if not markers:
            self.btn_hotspot_channels.setText("Markers: N/A")
            return
        chosen = [idx for idx, act in self._hotspot_channel_actions.items() if act.isChecked()]
        if not chosen:
            self.btn_hotspot_channels.setText("Markers: None")
        elif len(chosen) <= 2:
            names = [markers[i] for i in chosen if i < len(markers)]
            self.btn_hotspot_channels.setText(f"Markers: {', '.join(names)}")
        else:
            self.btn_hotspot_channels.setText(f"Markers: {len(chosen)}/{len(markers)}")

    def _sync_hotspot_channels_from_viewer(self, *, redraw: bool = True) -> None:
        marker_names, active_idxs, _colors, _limits = resolve_patch_channels_from_ctx(self.ctx)
        self._hotspot_marker_names = marker_names
        self._hotspot_channel_indices = None
        active_set = set(active_idxs)
        for idx, act in self._hotspot_channel_actions.items():
            act.blockSignals(True)
            act.setChecked(idx in active_set)
            act.blockSignals(False)
        self._update_hotspot_channel_button_text()
        if redraw:
            self._draw_hotspot_patch(self._active_hotspot_item())

    def _set_all_hotspot_channels(self, checked: bool) -> None:
        self._hotspot_channel_indices = (
            list(sorted(self._hotspot_channel_actions.keys())) if checked else []
        )
        for act in self._hotspot_channel_actions.values():
            act.blockSignals(True)
            act.setChecked(checked)
            act.blockSignals(False)
        self._update_hotspot_channel_button_text()
        self._draw_hotspot_patch(self._active_hotspot_item())

    def _on_hotspot_channel_action_toggled(self, _checked: bool = False) -> None:
        self._hotspot_channel_indices = [
            idx for idx, act in sorted(self._hotspot_channel_actions.items()) if act.isChecked()
        ]
        self._update_hotspot_channel_button_text()
        self._draw_hotspot_patch(self._active_hotspot_item())

    def _rebuild_hotspot_channel_menu(self) -> None:
        marker_names, active_idxs, _colors, _limits = resolve_patch_channels_from_ctx(self.ctx)
        self._hotspot_marker_names = marker_names
        self.menu_hotspot_channels.clear()
        self._hotspot_channel_actions.clear()

        if not marker_names:
            empty_act = QAction("No channels available", self.menu_hotspot_channels)
            empty_act.setEnabled(False)
            self.menu_hotspot_channels.addAction(empty_act)
            self._update_hotspot_channel_button_text()
            return

        act_sync = QAction("Sync Active Viewer Channels", self.menu_hotspot_channels)
        act_sync.triggered.connect(lambda: self._sync_hotspot_channels_from_viewer(redraw=True))
        self.menu_hotspot_channels.addAction(act_sync)

        act_all = QAction("Select All Markers", self.menu_hotspot_channels)
        act_all.triggered.connect(lambda: self._set_all_hotspot_channels(True))
        self.menu_hotspot_channels.addAction(act_all)

        act_clear = QAction("Clear Markers (Dark Background)", self.menu_hotspot_channels)
        act_clear.triggered.connect(lambda: self._set_all_hotspot_channels(False))
        self.menu_hotspot_channels.addAction(act_clear)

        self.menu_hotspot_channels.addSeparator()

        if self._hotspot_channel_indices is None:
            checked_set = set(active_idxs)
        else:
            checked_set = {i for i in self._hotspot_channel_indices if 0 <= i < len(marker_names)}

        for idx, m_name in enumerate(marker_names):
            act = QAction(f"[{idx + 1}] {m_name}", self.menu_hotspot_channels)
            act.setCheckable(True)
            act.setChecked(idx in checked_set)
            act.toggled.connect(self._on_hotspot_channel_action_toggled)
            self.menu_hotspot_channels.addAction(act)
            self._hotspot_channel_actions[idx] = act

        self._update_hotspot_channel_button_text()

    def refresh_data(self) -> None:
        """Refresh KPI badges, method color key, disputed table, plots, and feature preview."""
        self._sync_sample_combo()
        self._sync_method_combos()
        self._rebuild_method_color_pills()
        self._rebuild_hotspot_channel_menu()
        self._sync_final_mask_status()
        report = self._get_report()
        methods = self._get_available_methods()

        if report is not None:
            tot = int(getattr(report, "total_ref_cells", 0))
            agr = int(getattr(report, "agreed_cells", 0))
            flg = int(getattr(report, "flagged_regions", 0))
            mn = float(getattr(report, "mean_iou", 0.0))
            n_models = len(getattr(report, "methods", methods))
            pct_agr = (agr / tot * 100.0) if tot > 0 else 0.0
            self.lbl_kpi_models.setText(f"Models: {n_models}")
            self.lbl_kpi_agreed.setText(f"Agreed: {agr:,} ({pct_agr:.1f}%)")
            self.lbl_kpi_flagged.setText(f"Flagged: {flg:,}")
            self.lbl_kpi_mean.setText(f"Mean IoU: {mn:.3f}")
        elif (
            self.disagreement_panel is not None
            and hasattr(self.disagreement_panel, "cached_stats")
        ):
            stats = self.disagreement_panel.cached_stats
            tot = stats.get("total", 0)
            agr = stats.get("agreed", 0)
            flg = stats.get("flagged", 0)
            mn = stats.get("mean_iou", 0.0)
            n_models = len(stats.get("methods", methods))
            pct_agr = (agr / tot * 100.0) if tot > 0 else 0.0
            self.lbl_kpi_models.setText(f"Models: {n_models}")
            self.lbl_kpi_agreed.setText(f"Agreed: {agr:,} ({pct_agr:.1f}%)")
            self.lbl_kpi_flagged.setText(f"Flagged: {flg:,}")
            self.lbl_kpi_mean.setText(f"Mean IoU: {mn:.3f}")

        self._populate_issues_table()
        self._draw_iou_and_category_plot()
        self._draw_pairwise_matrices_and_summary()
        preview_item = self._current_hotspot_item
        if preview_item is None and self._displayed_items:
            preview_item = self._displayed_items[0]
        self._draw_hotspot_patch(preview_item)
        self._refresh_hotspot_curation_bar()

        if (
            self.feature_panel is not None
            and getattr(self.feature_panel, "cached_df", None) is not None
        ):
            self._populate_feature_table(self.feature_panel.cached_df)

    def _populate_issues_table(self) -> None:
        self.table_issues.blockSignals(True)
        self.table_issues.setRowCount(0)

        report = self._get_report()
        cat_filter = self.combo_category_filter.currentText()

        items: List[Any] = []
        if report is not None and getattr(report, "items", None):
            items = list(report.items)
        elif self.ctx is not None and getattr(self.ctx, "flagged_cells", None):
            items = [
                x for x in self.ctx.flagged_cells if hasattr(x, "region_id") or isinstance(x, dict)
            ]

        if cat_filter and cat_filter != "All Disagreements":
            items = [
                it
                for it in items
                if (it.get("category") if isinstance(it, dict) else getattr(it, "category", ""))
                == cat_filter
            ]

        self._displayed_items = items
        self.lbl_table_count.setText(f"{len(items)} regions")
        self.table_issues.setRowCount(len(items))

        curated = resolve_curated_state_from_ctx(self.ctx)
        decisions = (
            getattr(curated, "region_decisions", {}) or {} if curated is not None else {}
        )

        for row_idx, it in enumerate(items):
            if isinstance(it, dict):
                rid = it.get("region_id", row_idx + 1)
                cat = str(it.get("category", "Unknown"))
                iou = float(it.get("best_iou", 0.0))
                dice = float(it.get("dice", 0.0))
                ov = float(it.get("overlap_coeff", 0.0))
                m_ids = dict(it.get("method_cell_ids", {}))
            else:
                rid = getattr(it, "region_id", row_idx + 1)
                cat = str(getattr(it, "category", "Unknown"))
                iou = float(getattr(it, "best_iou", 0.0))
                dice = float(getattr(it, "dice", 0.0))
                ov = float(getattr(it, "overlap_coeff", 0.0))
                m_ids = dict(getattr(it, "method_cell_ids", {}))

            models_summary = ", ".join(
                f"{m} ({len(ids)})" for m, ids in sorted(m_ids.items())
            )
            rid_int = extract_item_region_id(it)
            picked_val = str(decisions.get(rid_int, "Ref")) if rid_int is not None else "Ref"

            item_idx = QTableWidgetItem(str(row_idx + 1))
            item_rid = QTableWidgetItem(str(rid))
            item_cat = QTableWidgetItem(cat)
            item_iou = QTableWidgetItem(f"{iou:.3f}")
            item_dice = QTableWidgetItem(f"{dice:.3f}")
            item_ov = QTableWidgetItem(f"{ov * 100.0:.1f}%")
            item_models = QTableWidgetItem(models_summary)
            item_picked = self._make_picked_table_item(picked_val)

            for centered in (item_idx, item_rid, item_iou, item_dice, item_ov):
                centered.setTextAlignment(Qt.AlignCenter)

            self.table_issues.setItem(row_idx, 0, item_idx)
            self.table_issues.setItem(row_idx, 1, item_rid)
            self.table_issues.setItem(row_idx, 2, item_cat)
            self.table_issues.setItem(row_idx, 3, item_iou)
            self.table_issues.setItem(row_idx, 4, item_dice)
            self.table_issues.setItem(row_idx, 5, item_ov)
            self.table_issues.setItem(row_idx, 6, item_models)
            self.table_issues.setItem(row_idx, 7, item_picked)

        self.table_issues.blockSignals(False)
        self._refresh_hotspot_curation_bar()

    @staticmethod
    def _make_picked_table_item(picked_val: str) -> QTableWidgetItem:
        cell = QTableWidgetItem(picked_val)
        cell.setTextAlignment(Qt.AlignCenter)
        if picked_val == "Ref":
            cell.setForeground(QColor("#9AA4B2"))
        elif picked_val == NO_CELL_CHOICE:
            cell.setForeground(QColor("#FF6B6B"))
        elif picked_val.startswith("Consensus:"):
            cell.setForeground(QColor("#44AAFF"))
        else:
            cell.setForeground(QColor("#34C759"))
        return cell

    def _update_table_picked_column(self) -> None:
        """Update column 7 ('Picked') in place without clearing row selection or scroll."""
        curated = resolve_curated_state_from_ctx(self.ctx)
        decisions = (
            getattr(curated, "region_decisions", {}) or {} if curated is not None else {}
        )
        self.table_issues.blockSignals(True)
        try:
            n_rows = self.table_issues.rowCount()
            for row_idx, it in enumerate(self._displayed_items):
                if row_idx >= n_rows:
                    break
                rid_int = extract_item_region_id(it)
                picked_val = str(decisions.get(rid_int, "Ref")) if rid_int is not None else "Ref"
                existing_item = self.table_issues.item(row_idx, 7)
                if existing_item is None:
                    self.table_issues.setItem(row_idx, 7, self._make_picked_table_item(picked_val))
                else:
                    existing_item.setText(picked_val)
                    styled = self._make_picked_table_item(picked_val)
                    existing_item.setForeground(styled.foreground())
        finally:
            self.table_issues.blockSignals(False)

    @staticmethod
    def _plot_method_contours_on_ax(
        ax: Any,
        sub_mask: np.ndarray,
        hex_color: str,
        target_cids: Sequence[int],
        base_lw: float,
    ) -> None:
        """Draw solid target and dashed neighboring cell contours on *ax*."""
        unique_ids = [int(v) for v in np.unique(sub_mask) if v > 0]
        neighbor_lw = max(0.8, base_lw * 0.45)
        for cid in unique_ids:
            is_target = cid in target_cids
            lw = base_lw if is_target else neighbor_lw
            alpha = 0.95 if is_target else 0.40
            ls = "-" if is_target else "--"
            for cnt in find_contours((sub_mask == cid).astype(np.float32), 0.5):
                ax.plot(
                    cnt[:, 1],
                    cnt[:, 0],
                    color=hex_color,
                    linewidth=lw,
                    linestyle=ls,
                    alpha=alpha,
                )

    def _draw_hotspot_patch(self, item: Any) -> None:
        """Render zoomed crop around *item* with selected markers and method contours."""
        self.fig_hotspot.clear()

        state = getattr(self.ctx, "state", None) if self.ctx is not None else None
        seg_results = getattr(state, "segmentation_results", {}) if state is not None else {}

        if item is None or not seg_results:
            ax = self.fig_hotspot.add_subplot(111)
            ax.set_facecolor("#14161d")
            ax.text(
                0.5,
                0.5,
                "Select a disputed region from the table to inspect zoomed method contours.",
                color="#9AA4B2",
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#3c4352")
            self.canvas_hotspot.draw()
            return

        if isinstance(item, dict):
            rid = item.get("region_id", 0)
            cat = str(item.get("category", "Hotspot"))
            best_iou = float(item.get("best_iou", 0.0))
            dice = float(item.get("dice", 0.0))
            ov = float(item.get("overlap_coeff", 0.0))
            bbox = item.get("bbox", (0, 0, 32, 32))
            m_cell_ids = dict(item.get("method_cell_ids", {}))
        else:
            rid = getattr(item, "region_id", 0)
            cat = str(getattr(item, "category", "Hotspot"))
            best_iou = float(getattr(item, "best_iou", 0.0))
            dice = float(getattr(item, "dice", 0.0))
            ov = float(getattr(item, "overlap_coeff", 0.0))
            bbox = getattr(item, "bbox", (0, 0, 32, 32))
            m_cell_ids = dict(getattr(item, "method_cell_ids", {}))

        compare_mode = self.combo_compare.currentText() or "Nuclear Masks"
        try:
            selected = None
            if self.disagreement_panel is not None and hasattr(
                self.disagreement_panel, "get_selected_methods"
            ):
                sel = self.disagreement_panel.get_selected_methods()
                if sel:
                    selected = sel
            filtered_seg = (
                {m: seg_results[m] for m in selected if m in seg_results}
                if selected
                else {m: r for m, r in seg_results.items() if m != FINAL_CURATED_MASK_NAME}
            )
            masks = _extract_masks_for_mode(filtered_seg, compare_mode)
        except Exception:
            masks = {}

        if not masks:
            ax = self.fig_hotspot.add_subplot(111)
            ax.set_facecolor("#14161d")
            ax.text(
                0.5,
                0.5,
                f"No masks available for compare mode '{compare_mode}'.",
                color="#9AA4B2",
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas_hotspot.draw()
            return

        h, w = next(iter(masks.values())).shape
        r0, c0, r1, c1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        pad = 20
        pr0 = max(0, r0 - pad)
        pc0 = max(0, c0 - pad)
        pr1 = min(h, r1 + pad)
        pc1 = min(w, c1 + pad)
        if pr1 <= pr0 or pc1 <= pc0:
            pr0, pc0, pr1, pc1 = 0, 0, h, w

        marker_names, active_idxs, ch_colors, ch_limits = resolve_patch_channels_from_ctx(self.ctx)
        if self._hotspot_channel_indices is None:
            chosen_indices = list(active_idxs)
        else:
            chosen_indices = [
                i for i in self._hotspot_channel_indices if 0 <= i < len(marker_names)
            ]

        raw_img = resolve_state_active_image(state)
        rgb_patch = compose_multichannel_patch_rgb(
            raw_img,
            (pr0, pc0, pr1, pc1),
            channel_indices=chosen_indices,
            channel_colors=ch_colors,
            contrast_limits=ch_limits,
            local_autocontrast=bool(self.chk_hotspot_autocontrast.isChecked()),
        )
        if rgb_patch is None:
            rgb_patch = np.zeros((pr1 - pr0, pc1 - pc0, 3), dtype=np.float32)

        method_colors = self._get_method_colors()
        base_lw = float(self.spin_hotspot_lw.value())
        view_mode = self.combo_hotspot_view.currentText()
        shown_markers = [(marker_names[i], ch_colors.get(i, "#E0E0E0")) for i in chosen_indices]

        if view_mode == "Side-by-Side Per Method" and len(masks) > 1:
            n_m = len(masks)
            ncols = n_m if n_m <= 3 else int(np.ceil(n_m / 2))
            nrows = 1 if n_m <= 3 else 2
            for idx, (m_name, mask) in enumerate(masks.items()):
                ax = self.fig_hotspot.add_subplot(nrows, ncols, idx + 1)
                ax.set_facecolor("#14161d")
                ax.imshow(rgb_patch, origin="upper", interpolation="nearest")
                hex_c = method_colors.get(m_name, METHOD_QC_COLORS[idx % len(METHOD_QC_COLORS)])
                cids = m_cell_ids.get(m_name, [])
                self._plot_method_contours_on_ax(ax, mask[pr0:pr1, pc0:pc1], hex_c, cids, base_lw)
                ax.set_xlim(-0.5, (pc1 - pc0) - 0.5)
                ax.set_ylim((pr1 - pr0) - 0.5, -0.5)
                ax.tick_params(colors="#CCCCCC", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_color(hex_c)
                ax.set_title(
                    f"{m_name} ({len(cids)} cell{'s' if len(cids) != 1 else ''})",
                    color=hex_c,
                    fontsize=9.5,
                    fontweight="bold",
                )
            marker_summary = ", ".join(m for m, _ in shown_markers) or "None"
            self.fig_hotspot.suptitle(
                f"Hotspot #{rid} ({cat}) | IoU={best_iou:.2f}, Dice={dice:.2f}, "
                f"Overlap={ov * 100.0:.0f}% | Markers: {marker_summary}",
                color="#FFFFFF",
                fontsize=10,
                fontweight="bold",
            )
            self.fig_hotspot.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
            self.canvas_hotspot.draw()
            return

        ax = self.fig_hotspot.add_subplot(111)
        ax.set_facecolor("#14161d")
        ax.imshow(rgb_patch, origin="upper", interpolation="nearest")

        method_handles = []
        for idx, (m_name, mask) in enumerate(masks.items()):
            hex_c = method_colors.get(m_name, METHOD_QC_COLORS[idx % len(METHOD_QC_COLORS)])
            cids = m_cell_ids.get(m_name, [])
            self._plot_method_contours_on_ax(ax, mask[pr0:pr1, pc0:pc1], hex_c, cids, base_lw)
            (h_line,) = ax.plot(
                [],
                [],
                color=hex_c,
                linewidth=2.4,
                label=f"{m_name} ({len(cids)} cell{'s' if len(cids) != 1 else ''})",
            )
            method_handles.append(h_line)

        ax.set_xlim(-0.5, (pc1 - pc0) - 0.5)
        ax.set_ylim((pr1 - pr0) - 0.5, -0.5)
        ax.tick_params(colors="#CCCCCC", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#4A5162")
        ax.set_xlabel("Patch X (px)", color="#E0E0E0", fontsize=9, fontweight="bold")
        ax.set_ylabel("Patch Y (px)", color="#E0E0E0", fontsize=9, fontweight="bold")
        ax.set_title(
            f"Hotspot #{rid} ({cat}) | IoU={best_iou:.2f}, Dice={dice:.2f}, "
            f"Overlap={ov * 100.0:.0f}% | y={pr0}:{pr1}, x={pc0}:{pc1}",
            color="#FFFFFF",
            fontsize=10,
            fontweight="bold",
        )

        if method_handles:
            leg_methods = ax.legend(
                handles=method_handles,
                loc="upper right",
                facecolor="#1f232b",
                edgecolor="#4A5162",
                labelcolor="#FFFFFF",
                fontsize=8,
            )
            leg_methods.set_title("Methods", prop={"size": 8, "weight": "bold"})
            leg_methods.get_title().set_color("#E0E0E0")
            ax.add_artist(leg_methods)

        if shown_markers:
            marker_handles = [
                ax.plot([], [], color=mc, linewidth=4.0, label=mn)[0]
                for mn, mc in shown_markers
            ]
            leg_markers = ax.legend(
                handles=marker_handles,
                loc="lower left",
                facecolor="#1f232b",
                edgecolor="#4A5162",
                labelcolor="#FFFFFF",
                fontsize=8,
            )
            leg_markers.set_title("Markers", prop={"size": 8, "weight": "bold"})
            leg_markers.get_title().set_color("#E0E0E0")

        self.fig_hotspot.tight_layout()
        self.canvas_hotspot.draw()

    @staticmethod
    def _extract_bg_crop(
        state: Any, pr0: int, pc0: int, pr1: int, pc1: int
    ) -> Optional[np.ndarray]:
        img = resolve_state_active_image(state)
        rgb = compose_multichannel_patch_rgb(img, (pr0, pc0, pr1, pc1), channel_indices=[0])
        if rgb is None:
            return None
        return np.max(rgb, axis=-1)

    def _draw_iou_and_category_plot(self) -> None:
        self.fig_iou.clear()
        report = self._get_report()
        per_cell_iou = getattr(report, "per_cell_iou", {}) if report is not None else {}
        cat_counts = getattr(report, "category_counts", {}) if report is not None else {}

        if not per_cell_iou and not cat_counts:
            ax = self.fig_iou.add_subplot(111)
            ax.set_facecolor("#14161d")
            ax.text(
                0.5,
                0.5,
                "No disagreement report generated yet.\n"
                "Run segmentation groups and click 'Highlight Model Disagreements'.",
                color="#9AA4B2",
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#3c4352")
            self.canvas_iou.draw()
            return

        ax1 = self.fig_iou.add_subplot(211)
        ax2 = self.fig_iou.add_subplot(212)
        for ax in (ax1, ax2):
            ax.set_facecolor("#14161d")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#4A5162")

        # Top subplot: IoU histogram
        thresh = self.slider_threshold.value() / 100.0
        ious = np.array(list(per_cell_iou.values()), dtype=float)
        if len(ious) > 0:
            bins = np.linspace(0.0, 1.0, 21)
            counts, edges = np.histogram(ious, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            for c, cnt in zip(centers, counts):
                col = "#34C759" if c >= thresh else "#E57373"
                ax1.bar(c, cnt, width=0.042, color=col, edgecolor="#222", alpha=0.88)
            ax1.axvline(
                thresh,
                color="#FF5252",
                linestyle="--",
                linewidth=1.8,
                label=f"Threshold ({thresh:.2f})",
            )
            ax1.legend(facecolor="#222630", edgecolor="#4A5162", labelcolor="white")
        ax1.set_xlabel("Mean Peer IoU", color="#E0E0E0", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Cells", color="#E0E0E0", fontsize=10, fontweight="bold")
        ax1.set_title(
            f"Per-Cell IoU Overlap Distribution ({len(ious):,} cells)",
            color="#FFF",
            fontsize=11,
            fontweight="bold",
        )

        # Bottom subplot: Category Breakdown
        cats = CATEGORY_ORDER
        vals = [int(cat_counts.get(c, 0)) for c in cats]
        colors = [CATEGORY_COLORS[c] for c in cats]
        x_pos = np.arange(len(cats))
        bars = ax2.bar(x_pos, vals, color=colors, edgecolor="#222", alpha=0.9)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(cats, color="#E0E0E0", fontsize=9)
        ax2.set_ylabel("Count", color="#E0E0E0", fontsize=10, fontweight="bold")
        ax2.set_title(
            "Disagreement Category Breakdown",
            color="#FFF",
            fontsize=11,
            fontweight="bold",
        )
        for bar, v in zip(bars, vals):
            if v > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    str(v),
                    ha="center",
                    va="bottom",
                    color="#FFF",
                    fontsize=9,
                    fontweight="bold",
                )

        self.fig_iou.tight_layout()
        self.canvas_iou.draw()

    def _draw_pairwise_matrices_and_summary(self) -> None:
        self.fig_matrix.clear()
        report = self._get_report()

        dice_df = getattr(report, "pairwise_dice", None) if report is not None else None
        iou_df = getattr(report, "pairwise_iou", None) if report is not None else None
        summary_df = getattr(report, "model_summary", None) if report is not None else None

        if dice_df is not None and not getattr(dice_df, "empty", True):
            dice_df = dice_df.drop(
                index=FINAL_CURATED_MASK_NAME,
                columns=FINAL_CURATED_MASK_NAME,
                errors="ignore",
            )
        if iou_df is not None and not getattr(iou_df, "empty", True):
            iou_df = iou_df.drop(
                index=FINAL_CURATED_MASK_NAME,
                columns=FINAL_CURATED_MASK_NAME,
                errors="ignore",
            )
        if (
            summary_df is not None
            and not getattr(summary_df, "empty", True)
            and "Method" in summary_df.columns
        ):
            summary_df = summary_df[summary_df["Method"] != FINAL_CURATED_MASK_NAME]

        if (
            dice_df is None
            or iou_df is None
            or getattr(dice_df, "empty", True)
            or getattr(iou_df, "empty", True)
        ):
            ax = self.fig_matrix.add_subplot(111)
            ax.set_facecolor("#14161d")
            ax.text(
                0.5,
                0.5,
                "Run >= 1 segmentation pipeline and compute disagreements to view matrices.",
                color="#9AA4B2",
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#3c4352")
            self.canvas_matrix.draw()
            self.table_model_summary.setRowCount(0)
            return

        ax1 = self.fig_matrix.add_subplot(121)
        ax2 = self.fig_matrix.add_subplot(122)

        for ax, df, title in (
            (ax1, dice_df, "Pairwise Foreground Dice"),
            (ax2, iou_df, "Pairwise Matched Instance IoU"),
        ):
            ax.set_facecolor("#14161d")
            mat = df.to_numpy(dtype=float)
            im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
            labels = [str(x) for x in df.index]
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right", color="#E0E0E0", fontsize=9)
            ax.set_yticklabels(labels, color="#E0E0E0", fontsize=9)
            ax.set_title(title, color="#FFF", fontsize=10, fontweight="bold")
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    val = mat[r, c]
                    txt_color = "#111" if val > 0.65 else "#FFF"
                    ax.text(
                        c,
                        r,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color=txt_color,
                        fontsize=9,
                        fontweight="bold",
                    )
            cbar = self.fig_matrix.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors="white", labelsize=8)

        self.fig_matrix.tight_layout()
        self.canvas_matrix.draw()

        if summary_df is not None and not summary_df.empty:
            self.table_model_summary.setRowCount(len(summary_df))
            for r_idx in range(len(summary_df)):
                row = summary_df.iloc[r_idx]
                vals = [
                    str(row.get("Method", "")),
                    str(int(row.get("Cell_Count", 0))),
                    f"{float(row.get('Median_Area_px', 0.0)):.1f}",
                    f"{float(row.get('Mean_Peer_IoU', 0.0)):.3f}",
                ]
                for c_idx, txt in enumerate(vals):
                    item = QTableWidgetItem(txt)
                    if c_idx > 0:
                        item.setTextAlignment(Qt.AlignCenter)
                    self.table_model_summary.setItem(r_idx, c_idx, item)
        else:
            self.table_model_summary.setRowCount(0)

    def _populate_feature_table(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        preview_df = df.head(50)
        n_rows, n_cols = preview_df.shape
        self.table_features.setRowCount(n_rows)
        self.table_features.setColumnCount(n_cols)
        self.table_features.setHorizontalHeaderLabels([str(c) for c in preview_df.columns])

        for r in range(n_rows):
            for c in range(n_cols):
                val = preview_df.iloc[r, c]
                txt = f"{val:.3f}" if isinstance(val, (float, np.floating)) else str(val)
                self.table_features.setItem(r, c, QTableWidgetItem(txt))

        self.lbl_feat_summary.setText(
            f"Showing first {len(preview_df)} rows of {len(df):,} extracted cells x "
            f"{len(df.columns)} features."
        )

    def _on_extract_features_clicked(self) -> None:
        if self.feature_panel is not None:
            self.feature_panel._emit_extract_signal()

    def _on_export_features_clicked(self) -> None:
        if self.feature_panel is not None:
            self.feature_panel.exportTriggered.emit()

    # Figure Studio Integration

    def open_publication_studio(self) -> None:
        """Open Figure Studio dialog on IoU distribution & model agreement matrix."""
        from iris_common.widgets.publication_studio import (
            PlotStyleConfig,
            PublicationPlotStudioDialog,
            get_palette_colors,
        )

        report = self._get_report()
        thresh = self.slider_threshold.value() / 100.0

        per_cell_iou = dict(getattr(report, "per_cell_iou", {})) if report is not None else {}
        cat_counts = dict(getattr(report, "category_counts", {})) if report is not None else {}
        model_summary = (
            report.model_summary[
                report.model_summary["Method"] != FINAL_CURATED_MASK_NAME
            ].copy()
            if (
                report is not None
                and getattr(report, "model_summary", None) is not None
                and not report.model_summary.empty
                and "Method" in report.model_summary.columns
            )
            else pd.DataFrame(columns=["Method", "Cell_Count", "Median_Area_px", "Mean_Peer_IoU"])
        )
        pairwise_iou = (
            report.pairwise_iou.drop(
                index=FINAL_CURATED_MASK_NAME,
                columns=FINAL_CURATED_MASK_NAME,
                errors="ignore",
            ).copy()
            if (report is not None and getattr(report, "pairwise_iou", None) is not None)
            else pd.DataFrame()
        )
        replicates_df = pd.DataFrame(
            [{"ObjectNumber": cid, "Mean_Peer_IoU": val} for cid, val in per_cell_iou.items()]
        )
        stats_df = (
            pairwise_iou.reset_index().rename(columns={"index": "Method"})
            if not pairwise_iou.empty
            else pd.DataFrame()
        )

        def _plot_disagreement(fig: Figure, cfg: PlotStyleConfig) -> None:
            has_iou_matrix = not pairwise_iou.empty
            ax1 = fig.add_subplot(1, 2 if has_iou_matrix else 1, 1)

            ious = np.array(list(per_cell_iou.values()), dtype=float)
            palette = get_palette_colors(cfg.palette_name, 5)
            if len(ious) > 0:
                bins = np.linspace(0.0, 1.0, 21)
                counts, edges = np.histogram(ious, bins=bins)
                centers = 0.5 * (edges[:-1] + edges[1:])
                default_agreed = palette[2] if len(palette) > 2 else "#34C759"
                agreed_color = cfg.custom_colors.get("Agreed", default_agreed)
                disputed_color = cfg.custom_colors.get("Boundary Shift", palette[0])
                for c, cnt in zip(centers, counts):
                    bar_col = agreed_color if c >= thresh else disputed_color
                    ax1.bar(c, cnt, width=0.042, color=bar_col, alpha=0.88, edgecolor="none")
                thresh_color = palette[1] if len(palette) > 1 else "#D55E00"
                ax1.axvline(thresh, linestyle="--", linewidth=1.6, color=thresh_color)
            else:
                ax1.bar(
                    CATEGORY_ORDER,
                    [int(cat_counts.get(c, 0)) for c in CATEGORY_ORDER],
                    color=palette[: len(CATEGORY_ORDER)],
                )

            ax1.set_xlabel("Mean Peer IoU")
            ax1.set_ylabel("Cell Count")
            ax1.set_title(cfg.title or "Multi-Model Agreement Distribution")

            if has_iou_matrix:
                ax2 = fig.add_subplot(1, 2, 2)
                mat = pairwise_iou.to_numpy(dtype=float)
                im = ax2.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
                labels = [str(m) for m in pairwise_iou.index]
                ax2.set_xticks(np.arange(len(labels)))
                ax2.set_yticks(np.arange(len(labels)))
                ax2.set_xticklabels(labels, rotation=30, ha="right")
                ax2.set_yticklabels(labels)
                ax2.set_title("Pairwise Matched IoU")
                for r in range(mat.shape[0]):
                    for c in range(mat.shape[1]):
                        ax2.text(
                            c,
                            r,
                            f"{mat[r, c]:.2f}",
                            ha="center",
                            va="center",
                            color="#111" if mat[r, c] > 0.65 else "#FFF",
                            fontsize=max(7, int(cfg.stat_size)),
                        )
                fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        if self.publication_dialog is not None:
            try:
                self.publication_dialog.close()
            except Exception:
                pass

        self.publication_dialog = PublicationPlotStudioDialog(
            plot_fn=_plot_disagreement,
            summary_df=model_summary,
            replicates_df=replicates_df,
            stats_df=stats_df,
            categories=CATEGORY_ORDER,
            title="Segmentation Disagreement — Figure Studio",
            metadata={
                "iou_threshold": thresh,
                "compare_mode": self.combo_compare.currentText(),
                "reference_method": self.combo_ref_method.currentText(),
            },
            parent=self,
        )
        self.publication_dialog.show()
        self.publication_dialog.raise_()
        self.publication_dialog.activateWindow()

    def keyPressEvent(self, event) -> None:
        if event is None:
            return
        if event.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
            super().keyPressEvent(event)
            return
        focus_w = QApplication.focusWidget()
        if isinstance(focus_w, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
            super().keyPressEvent(event)
            return

        if self.right_tabs.currentIndex() == 1 or focus_w is None or self.isAncestorOf(focus_w):
            key = event.key()
            if Qt.Key_1 <= key <= Qt.Key_9:
                idx = key - Qt.Key_1
                if 0 <= idx < len(self._ordered_keep_methods):
                    btn = self.hotspot_keep_buttons.get(self._ordered_keep_methods[idx])
                    if btn is not None and btn.isEnabled():
                        btn.click()
                        event.accept()
                        return
            elif key == Qt.Key_0:
                if self.btn_keep_no_cell.isEnabled():
                    self.btn_keep_no_cell.click()
                    event.accept()
                    return
            elif key == Qt.Key_Space:
                if self.btn_skip_hotspot.isEnabled():
                    self.btn_skip_hotspot.click()
                    event.accept()
                    return

        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self.publication_dialog is not None:
            try:
                self.publication_dialog.close()
            except Exception:
                pass
            self.publication_dialog = None
        super().closeEvent(event)

    _sync_state = refresh_data
