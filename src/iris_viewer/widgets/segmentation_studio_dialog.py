"""
iris_viewer.widgets.segmentation_studio_dialog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interactive full-size QC window for inspecting multi-model segmentation
disagreements, pairwise Dice/IoU matrices, zoomed hotspot patch contours,
model flickering, active mask selection, and Figure Studio export.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from skimage.measure import find_contours

from iris_common.widgets.theme import (
    BUTTON_CTA,
    BUTTON_PRIMARY,
    BUTTON_SECONDARY,
    BUTTON_STUDIO,
    DIALOG_DARK_THEME,
    create_section_card,
    style_section_header,
)
from iris_viewer.engine.disagreement import METHOD_QC_COLORS, _extract_masks_for_mode

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

        self.setWindowTitle("Segmentation Disagreement QC")
        self.resize(1260, 800)
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

        top_bar.addWidget(QLabel("Active Mask:"))
        self.combo_active_method = QComboBox()
        top_bar.addWidget(self.combo_active_method)

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

        self.table_issues = QTableWidget(0, 7)
        self.table_issues.setAlternatingRowColors(True)
        self.table_issues.setHorizontalHeaderLabels(
            ["#", "Region ID", "Category", "Best IoU", "Dice", "Overlap %", "Models"]
        )
        self.table_issues.verticalHeader().setVisible(False)
        self.table_issues.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_issues.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self.table_issues.horizontalHeader()
        for col_idx in range(6):
            header.setSectionResizeMode(col_idx, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        self.table_issues.itemSelectionChanged.connect(self._on_table_region_selected)
        gl.addWidget(self.table_issues, 1)

        # Model Flicker & Active Mask Selection Card
        self.grp_actions, al = create_section_card("Model Flicker & Active Mask")

        self.lbl_sel_region_info = QLabel(
            "No region selected. Click a disputed region in the table above."
        )
        style_section_header(self.lbl_sel_region_info, subtitle=True)
        self.lbl_sel_region_info.setWordWrap(True)
        al.addWidget(self.lbl_sel_region_info)

        flicker_row = QHBoxLayout()
        flicker_row.setSpacing(6)
        flicker_row.addWidget(QLabel("Flicker:"))
        self.combo_flicker_a = QComboBox()
        self.combo_flicker_b = QComboBox()
        self.btn_flicker = QPushButton("Flicker Pair")
        self.btn_flicker.setStyleSheet(BUTTON_STUDIO)
        self.btn_flicker.clicked.connect(self._on_flicker_clicked)
        flicker_row.addWidget(self.combo_flicker_a, 1)
        flicker_row.addWidget(QLabel("vs"))
        flicker_row.addWidget(self.combo_flicker_b, 1)
        flicker_row.addWidget(self.btn_flicker)
        al.addLayout(flicker_row)

        self.btn_set_active_mask = QPushButton("Set Active Segmentation Mask")
        self.btn_set_active_mask.setMinimumHeight(28)
        self.btn_set_active_mask.setStyleSheet(BUTTON_CTA)
        self.btn_set_active_mask.clicked.connect(self._on_set_active_mask_clicked)
        al.addWidget(self.btn_set_active_mask)

        gl.addWidget(self.grp_actions)
        ll.addWidget(grp_table)
        splitter.addWidget(left_widget)

        # Right: Analytics, Hotspot Patch Inspector & Feature Exploration Tabs
        self.right_tabs = QTabWidget()

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
            "Zoomed patch crop around the selected hotspot showing each method's labeled contour."
        )
        style_section_header(lbl_hs_desc, subtitle=True)
        th_lyt.addWidget(lbl_hs_desc)

        self.fig_hotspot = Figure(figsize=(6.0, 5.0), dpi=100)
        self.fig_hotspot.patch.set_facecolor("#14161d")
        self.canvas_hotspot = FigureCanvas(self.fig_hotspot)
        self.canvas_hotspot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        th_lyt.addWidget(self.canvas_hotspot, 1)
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
        self.btn_extract_feat.setStyleSheet(BUTTON_CTA)
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
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

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
        return methods

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
                "border-radius: 2px; padding: 0px; }}"
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
        self._draw_hotspot_patch(self._current_hotspot_item)

    def _sync_method_combos(self) -> None:
        methods = self._get_available_methods()
        report = self._get_report()
        active = None
        if self.ctx is not None and getattr(self.ctx, "state", None) is not None:
            active = self.ctx.state.active_segmentation_method
        if not active and self.disagreement_panel is not None:
            active = self.disagreement_panel.get_active_method()

        ref = getattr(report, "reference_method", None) if report is not None else None
        if not ref and self.disagreement_panel is not None:
            ref = self.disagreement_panel.get_reference_method()

        second_method = (
            methods[1]
            if len(methods) > 1
            else (methods[0] if methods else None)
        )
        for combo, preferred in (
            (self.combo_ref_method, ref),
            (self.combo_active_method, active),
            (self.combo_flicker_a, ref),
            (self.combo_flicker_b, second_method),
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
                break
        self._update_region_card(item)
        self._draw_hotspot_patch(item)

    def _on_prev_issue(self) -> None:
        if self.ctx is not None and getattr(self.ctx, "review_svc", None) is not None:
            self.ctx.review_svc.prev_issue()

    def _on_next_issue(self) -> None:
        if self.ctx is not None and getattr(self.ctx, "review_svc", None) is not None:
            self.ctx.review_svc.next_issue()

    def _on_table_region_selected(self) -> None:
        curr_row = self.table_issues.currentRow()
        if curr_row < 0 or curr_row >= len(self._displayed_items):
            return
        item = self._displayed_items[curr_row]
        self._current_hotspot_item = item
        self._update_region_card(item)
        self._draw_hotspot_patch(item)

        if self.ctx is not None and getattr(self.ctx, "review_svc", None) is not None:
            rid = getattr(item, "region_id", None)
            if hasattr(self.ctx.review_svc, "jump_to_region_id") and rid is not None:
                self.ctx.review_svc.jump_to_region_id(int(rid))
            else:
                self.ctx.review_svc.jump_to_issue(curr_row)

    def _update_region_card(self, item: Any) -> None:
        if item is None:
            self.lbl_sel_region_info.setText(
                "No region selected. Click a disputed region in the table above."
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

        models_txt = ", ".join(f"{m}: {ids}" for m, ids in sorted(m_ids.items())) or "None"
        self.lbl_sel_region_info.setText(
            f"<b>Region {rid}</b> ({cat}) &nbsp;|&nbsp; "
            f"<b>IoU:</b> {iou:.3f} &nbsp;|&nbsp; <b>Dice:</b> {dice:.3f} &nbsp;|&nbsp; "
            f"<b>Overlap:</b> {ov * 100.0:.0f}%<br>"
            f"<b>Models:</b> {models_txt}"
        )

    def _on_flicker_clicked(self) -> None:
        m_a = self.combo_flicker_a.currentText().strip()
        m_b = self.combo_flicker_b.currentText().strip()
        if not m_a or not m_b:
            return
        if self.ctx is not None and getattr(self.ctx, "review_svc", None) is not None:
            if hasattr(self.ctx.review_svc, "flicker_models"):
                self.ctx.review_svc.flicker_models(m_a, m_b)
        if self.disagreement_panel is not None:
            self.disagreement_panel.flickerPairRequested.emit(m_a, m_b)

    def _on_set_active_mask_clicked(self) -> None:
        method = self.combo_active_method.currentText().strip()
        if not method:
            return
        if self.ctx is not None and getattr(self.ctx, "state", None) is not None:
            self.ctx.state.set_active_segmentation_method(method)
        if self.ctx is not None and getattr(self.ctx, "review_svc", None) is not None:
            if hasattr(self.ctx.review_svc, "set_active_mask_layers"):
                self.ctx.review_svc.set_active_mask_layers(method)
        if self.disagreement_panel is not None:
            self.disagreement_panel.combo_active_method.setCurrentText(method)
        if self.feature_panel is not None:
            self.feature_panel.update_mask_sources(
                self._get_available_methods(), active=method
            )

    # Refresh & Plotting

    def refresh_data(self) -> None:
        """Refresh KPI badges, method color key, disputed table, plots, and feature preview."""
        self._sync_method_combos()
        self._rebuild_method_color_pills()
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

            item_idx = QTableWidgetItem(str(row_idx + 1))
            item_rid = QTableWidgetItem(str(rid))
            item_cat = QTableWidgetItem(cat)
            item_iou = QTableWidgetItem(f"{iou:.3f}")
            item_dice = QTableWidgetItem(f"{dice:.3f}")
            item_ov = QTableWidgetItem(f"{ov * 100.0:.1f}%")
            item_models = QTableWidgetItem(models_summary)

            for centered in (item_idx, item_rid, item_iou, item_dice, item_ov):
                centered.setTextAlignment(Qt.AlignCenter)

            self.table_issues.setItem(row_idx, 0, item_idx)
            self.table_issues.setItem(row_idx, 1, item_rid)
            self.table_issues.setItem(row_idx, 2, item_cat)
            self.table_issues.setItem(row_idx, 3, item_iou)
            self.table_issues.setItem(row_idx, 4, item_dice)
            self.table_issues.setItem(row_idx, 5, item_ov)
            self.table_issues.setItem(row_idx, 6, item_models)

        self.table_issues.blockSignals(False)

    def _draw_hotspot_patch(self, item: Any) -> None:
        """Render zoomed crop around *item* with each method's labeled colored contour."""
        self.fig_hotspot.clear()
        ax = self.fig_hotspot.add_subplot(111)
        ax.set_facecolor("#14161d")

        state = getattr(self.ctx, "state", None) if self.ctx is not None else None
        seg_results = getattr(state, "segmentation_results", {}) if state is not None else {}

        if item is None or not seg_results:
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
                else seg_results
            )
            masks = _extract_masks_for_mode(filtered_seg, compare_mode)
        except Exception:
            masks = {}

        if not masks:
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

        # Background raw image crop (slice lazy Dask arrays BEFORE computing!)
        bg_crop = self._extract_bg_crop(state, pr0, pc0, pr1, pc1)
        if bg_crop is not None:
            ax.imshow(bg_crop, cmap="gray", origin="upper", interpolation="nearest")
        else:
            ax.imshow(
                np.zeros((pr1 - pr0, pc1 - pc0), dtype=np.float32),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                origin="upper",
            )

        method_colors = self._get_method_colors()
        for idx, (m_name, mask) in enumerate(masks.items()):
            hex_c = method_colors.get(m_name, METHOD_QC_COLORS[idx % len(METHOD_QC_COLORS)])
            sub = mask[pr0:pr1, pc0:pc1]
            cids = m_cell_ids.get(m_name, [])
            unique_ids = [int(v) for v in np.unique(sub) if v > 0]

            for cid in unique_ids:
                is_target = cid in cids
                lw = 2.2 if is_target else 1.0
                alpha = 0.95 if is_target else 0.40
                ls = "-" if is_target else "--"
                contours = find_contours((sub == cid).astype(np.float32), 0.5)
                for cnt in contours:
                    ax.plot(
                        cnt[:, 1],
                        cnt[:, 0],
                        color=hex_c,
                        linewidth=lw,
                        linestyle=ls,
                        alpha=alpha,
                    )

            ax.plot(
                [],
                [],
                color=hex_c,
                linewidth=2.4,
                label=f"{m_name} ({len(cids)} cell{'s' if len(cids) != 1 else ''})",
            )

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
        leg = ax.legend(
            loc="upper right",
            facecolor="#1f232b",
            edgecolor="#4A5162",
            labelcolor="#FFFFFF",
            fontsize=8,
        )
        if leg is not None:
            leg.set_title("Methods", prop={"size": 8, "weight": "bold"})
            leg.get_title().set_color("#E0E0E0")

        self.fig_hotspot.tight_layout()
        self.canvas_hotspot.draw()

    @staticmethod
    def _extract_bg_crop(
        state: Any, pr0: int, pc0: int, pr1: int, pc1: int
    ) -> Optional[np.ndarray]:
        if state is None:
            return None
        img = getattr(state, "raw_image", None)
        if img is None:
            multi = getattr(state, "multi_sample_images", None)
            if multi:
                cur = getattr(state, "current_sample_name", None)
                img = multi.get(cur) if cur in multi else next(iter(multi.values()))
        if img is None:
            return None

        try:
            crop = img[pr0:pr1, pc0:pc1] if img.ndim == 2 else img[0, pr0:pr1, pc0:pc1]
            if hasattr(crop, "compute"):
                crop = crop.compute()
            arr = np.asarray(crop, dtype=np.float32)
            if arr.size == 0:
                return None
            lo = float(np.percentile(arr, 1.0))
            hi = float(np.percentile(arr, 99.5))
            if hi > lo:
                arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
            return arr
        except Exception:
            return None

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
            report.model_summary.copy()
            if (report is not None and getattr(report, "model_summary", None) is not None)
            else pd.DataFrame(columns=["Method", "Cell_Count", "Median_Area_px", "Mean_Peer_IoU"])
        )
        pairwise_iou = (
            report.pairwise_iou.copy()
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

    def closeEvent(self, event) -> None:
        if self.publication_dialog is not None:
            try:
                self.publication_dialog.close()
            except Exception:
                pass
            self.publication_dialog = None
        super().closeEvent(event)

    _sync_state = refresh_data
