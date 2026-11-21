"""
IRIS-Viewer Multi-Model Disagreement Panel Widget
=================================================
Manages active mask selection, per-method QC selection & color key, multi-model
disagreement computation, category filtering, hotspot navigation, model
flickering, and KPI metrics.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from iris_viewer.engine.disagreement import METHOD_QC_COLORS
from iris_viewer.widgets.styles import (
    BUTTON_CTA,
    BUTTON_SECONDARY,
    BUTTON_STUDIO,
    CONFIDENCE_HIGH_COLOR,
    CONFIDENCE_LOW_COLOR,
    CONFIDENCE_MEDIUM_COLOR,
    SHARED_DARK_THEME,
    create_section_card,
)


class DisagreementPanelWidget(QWidget):
    """Dock widget for multi-model segmentation disagreement QC and mask selection."""

    # Required Signals
    computeDisagreementRequested = Signal()
    thresholdChanged = Signal(float)
    activeMethodChanged = Signal(str)
    hotspotSelected = Signal(int)
    flickerPairRequested = Signal(str, str)
    statisticsUpdated = Signal(int, int, int, float)  # total, agreed, flagged, mean_iou
    regionReviewShown = Signal(object)

    # Auxiliary UI Signals
    categoryFilterChanged = Signal(str)
    overlayVisibilityChanged = Signal(bool, bool)  # show_heatmap, show_contours
    regionReviewCleared = Signal()
    methodColorsChanged = Signal(dict)
    selectedMethodsChanged = Signal(list)
    overlayColorModeChanged = Signal(str)

    CATEGORY_FILTERS = [
        "All Disagreements",
        "Split",
        "Merge",
        "Boundary Shift",
        "Model-Exclusive",
    ]

    OVERLAY_COLOR_MODES = [
        "Color by Method",
        "Color by Issue Type",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(SHARED_DARK_THEME)
        self._methods: List[str] = []
        self._known_methods: List[str] = []
        self._user_toggled_methods: set[str] = set()
        self.method_colors: Dict[str, str] = {}
        self.selected_methods: List[str] = []
        self.method_cell_counts: Dict[str, int] = {}
        self._method_row_widgets: Dict[str, Dict[str, Any]] = {}
        self._flicker_pair: Optional[tuple[str, str]] = None
        self.cached_review: Optional[Dict[str, Any]] = None
        self.cached_stats: Dict[str, Any] = {
            "total": 0,
            "agreed": 0,
            "flagged": 0,
            "mean_iou": 0.0,
            "category_counts": {},
            "methods": [],
        }

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        self.content_widget = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        self.content_widget.setLayout(self.main_layout)

        self.scroll_area.setWidget(self.content_widget)

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.scroll_area)
        self.setLayout(outer_layout)

        self._build_active_mask_controls()
        self._build_method_color_key_controls()
        self._build_disagreement_controls()
        self._build_threshold_controls()
        self._build_hotspot_controls()
        self._build_statistics()
        self.main_layout.addStretch()

    def _build_active_mask_controls(self) -> None:
        group, gl = create_section_card(
            "Active Segmentation Mask",
            "Selected pipeline mask is used for feature extraction, phenotyping, and export.",
        )

        row_active = QHBoxLayout()
        row_active.setSpacing(8)
        row_active.addWidget(QLabel("Active Mask:"))
        self.combo_active_method = QComboBox()
        self.combo_active_method.currentTextChanged.connect(self._on_active_method_changed)
        row_active.addWidget(self.combo_active_method, stretch=1)
        gl.addLayout(row_active)

        self.main_layout.addWidget(group)

    def _build_method_color_key_controls(self) -> None:
        group, gl = create_section_card(
            "Method Comparison & Color Key",
            "Check pipelines to compare and click a color swatch to customize its QC color.",
        )

        self.methods_container = QWidget()
        self.methods_container.setStyleSheet("background-color: transparent;")
        self.methods_container_layout = QVBoxLayout(self.methods_container)
        self.methods_container_layout.setContentsMargins(0, 0, 0, 0)
        self.methods_container_layout.setSpacing(4)
        gl.addWidget(self.methods_container)

        self.lbl_no_methods = QLabel("Run segmentation pipelines or load a mask to compare methods.")
        self.lbl_no_methods.setStyleSheet(
            "color: #9AA4B2; font-size: 11px; font-style: italic; background-color: transparent;"
        )
        self.lbl_no_methods.setWordWrap(True)
        self.methods_container_layout.addWidget(self.lbl_no_methods)

        self.main_layout.addWidget(group)

    def _build_disagreement_controls(self) -> None:
        group, gl = create_section_card("Disagreement QC")

        self.btn_compute_disagreement = QPushButton("Highlight Model Disagreements")
        self.btn_compute_disagreement.setMinimumHeight(28)
        self.btn_compute_disagreement.setStyleSheet(BUTTON_CTA)
        self.btn_compute_disagreement.clicked.connect(
            lambda: self.computeDisagreementRequested.emit()
        )
        gl.addWidget(self.btn_compute_disagreement)

        self.lbl_compute_status = QLabel("Pending Masks")
        self.lbl_compute_status.setStyleSheet(
            "color: #9AA4B2; font-size: 11px; background-color: transparent;"
        )
        gl.addWidget(self.lbl_compute_status)

        row_compare = QHBoxLayout()
        row_compare.setSpacing(8)
        row_compare.addWidget(QLabel("Compare:"))
        self.combo_compare = QComboBox()
        self.combo_compare.addItems(["Nuclear Masks", "Cytoplasm Masks", "Both"])
        row_compare.addWidget(self.combo_compare, stretch=1)
        gl.addLayout(row_compare)

        row_ref = QHBoxLayout()
        row_ref.setSpacing(8)
        row_ref.addWidget(QLabel("Reference:"))
        self.combo_ref_method = QComboBox()
        row_ref.addWidget(self.combo_ref_method, stretch=1)
        gl.addLayout(row_ref)

        row_overlay_color = QHBoxLayout()
        row_overlay_color.setSpacing(8)
        row_overlay_color.addWidget(QLabel("Overlay Color:"))
        self.combo_overlay_color = QComboBox()
        self.combo_overlay_color.addItems(self.OVERLAY_COLOR_MODES)
        self.combo_overlay_color.setCurrentText("Color by Method")
        self.combo_overlay_color.currentTextChanged.connect(
            lambda mode: self.overlayColorModeChanged.emit(mode)
        )
        row_overlay_color.addWidget(self.combo_overlay_color, stretch=1)
        gl.addLayout(row_overlay_color)

        row_overlays = QHBoxLayout()
        row_overlays.setSpacing(8)
        self.chk_show_heatmap = QCheckBox("Show Heatmap")
        self.chk_show_heatmap.setChecked(True)
        self.chk_show_contours = QCheckBox("Show Contours")
        self.chk_show_contours.setChecked(True)
        self.chk_show_heatmap.toggled.connect(self._emit_overlay_visibility)
        self.chk_show_contours.toggled.connect(self._emit_overlay_visibility)
        row_overlays.addWidget(self.chk_show_heatmap)
        row_overlays.addWidget(self.chk_show_contours)
        gl.addLayout(row_overlays)

        self.main_layout.addWidget(group)

    def _build_threshold_controls(self) -> None:
        group, gl = create_section_card(
            "Required IoU Overlap",
            "Cells below this Intersection-over-Union (IoU) overlap are flagged as hotspots.",
        )

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Threshold:"))
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100)
        self.slider_threshold.setValue(65)
        self.slider_threshold.setTickPosition(QSlider.TicksBelow)
        self.slider_threshold.setTickInterval(10)
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)
        row.addWidget(self.slider_threshold)

        self.lbl_threshold_value = QLabel("0.65")
        self.lbl_threshold_value.setFixedWidth(40)
        self.lbl_threshold_value.setStyleSheet(
            "color: #F0F2F5; font-weight: bold; background-color: transparent;"
        )
        row.addWidget(self.lbl_threshold_value)
        gl.addLayout(row)

        self.main_layout.addWidget(group)

    def _build_hotspot_controls(self) -> None:
        group, gl = create_section_card("Hotspot Inspection & Flicker")

        row_cat = QHBoxLayout()
        row_cat.setSpacing(8)
        row_cat.addWidget(QLabel("Filter:"))
        self.combo_category_filter = QComboBox()
        self.combo_category_filter.addItems(self.CATEGORY_FILTERS)
        self.combo_category_filter.currentTextChanged.connect(
            lambda txt: self.categoryFilterChanged.emit(txt)
        )
        row_cat.addWidget(self.combo_category_filter, stretch=1)
        gl.addLayout(row_cat)

        self.chk_enable_review = QCheckBox("Enable Click-to-Inspect Mode")
        self.chk_enable_review.setChecked(False)
        gl.addWidget(self.chk_enable_review)

        row_nav = QHBoxLayout()
        row_nav.setSpacing(8)
        self.btn_prev_issue = QPushButton("Prev Hotspot")
        self.btn_next_issue = QPushButton("Next Hotspot")
        self.btn_prev_issue.setMinimumHeight(28)
        self.btn_next_issue.setMinimumHeight(28)
        self.btn_prev_issue.setStyleSheet(BUTTON_SECONDARY)
        self.btn_next_issue.setStyleSheet(BUTTON_SECONDARY)
        row_nav.addWidget(self.btn_prev_issue)
        row_nav.addWidget(self.btn_next_issue)
        gl.addLayout(row_nav)

        self.lbl_issue_count = QLabel("0 flagged regions")
        self.lbl_issue_count.setAlignment(Qt.AlignCenter)
        self.lbl_issue_count.setStyleSheet(
            "color: #CCAA44; font-size: 11px; font-weight: bold; background-color: transparent;"
        )
        gl.addWidget(self.lbl_issue_count)

        self.btn_flicker = QPushButton("Flicker Top 2 Models")
        self.btn_flicker.setMinimumHeight(28)
        self.btn_flicker.setStyleSheet(BUTTON_STUDIO)
        self.btn_flicker.clicked.connect(self._on_flicker_clicked)
        gl.addWidget(self.btn_flicker)

        # Region inspection readout
        self.lbl_review_region = QLabel("No region selected")
        self.lbl_review_region.setStyleSheet(
            "color: #9AA4B2; font-size: 11px; background-color: transparent;"
        )
        gl.addWidget(self.lbl_review_region)

        self.lbl_review_category = QLabel("Category: -")
        self.lbl_review_category.setStyleSheet(
            "color: #9AA4B2; font-size: 11px; background-color: transparent;"
        )
        gl.addWidget(self.lbl_review_category)

        self.lbl_review_iou = QLabel("Best IoU: -")
        self.lbl_review_iou.setStyleSheet(
            "color: #9AA4B2; font-size: 11px; background-color: transparent;"
        )
        self.lbl_review_iou.setWordWrap(True)
        gl.addWidget(self.lbl_review_iou)

        self.lbl_review_methods = QLabel("Models: -")
        self.lbl_review_methods.setStyleSheet(
            "color: #9AA4B2; font-size: 11px; background-color: transparent;"
        )
        self.lbl_review_methods.setWordWrap(True)
        gl.addWidget(self.lbl_review_methods)

        self.main_layout.addWidget(group)

    def _build_statistics(self) -> None:
        group, gl = create_section_card("Agreement & Disagreement KPIs")

        self.lbl_total_cells = QLabel("Reference Cells: -")
        self.lbl_total_cells.setStyleSheet(
            "color: #9AA4B2; font-size: 11px; background-color: transparent;"
        )
        gl.addWidget(self.lbl_total_cells)

        self.lbl_agreed_cells = QLabel("Agreed Cells: -")
        self.lbl_agreed_cells.setStyleSheet(
            f"color: {CONFIDENCE_HIGH_COLOR}; font-size: 11px; font-weight: bold; "
            "background-color: transparent;"
        )
        gl.addWidget(self.lbl_agreed_cells)

        self.lbl_flagged_regions = QLabel("Flagged Regions: -")
        self.lbl_flagged_regions.setStyleSheet(
            f"color: {CONFIDENCE_LOW_COLOR}; font-size: 11px; font-weight: bold; "
            "background-color: transparent;"
        )
        gl.addWidget(self.lbl_flagged_regions)

        self.lbl_mean_iou = QLabel("Mean Peer IoU: -")
        self.lbl_mean_iou.setStyleSheet(
            f"color: {CONFIDENCE_MEDIUM_COLOR}; font-size: 11px; font-weight: bold; "
            "background-color: transparent;"
        )
        gl.addWidget(self.lbl_mean_iou)

        self.lbl_category_breakdown = QLabel(
            "Split: 0 | Merge: 0 | Shift: 0 | Exclusive: 0"
        )
        self.lbl_category_breakdown.setStyleSheet(
            "color: #9AA4B2; font-size: 10px; background-color: transparent;"
        )
        self.lbl_category_breakdown.setWordWrap(True)
        gl.addWidget(self.lbl_category_breakdown)

        self.main_layout.addWidget(group)

    # Method Color Key Helpers

    @staticmethod
    def _swatch_stylesheet(hex_color: str) -> str:
        return (
            f"QPushButton {{ background-color: {hex_color}; border: 1px solid #C8D0DC; "
            "border-radius: 3px; padding: 0px; min-width: 16px; max-width: 16px; "
            "min-height: 16px; max-height: 16px; }}"
            "QPushButton:hover { border: 2px solid #FFFFFF; }"
        )

    def _assign_default_color(self, method_name: str, used_colors: set[str]) -> str:
        existing = self.method_colors.get(method_name)
        if existing:
            return existing
        for candidate in METHOD_QC_COLORS:
            if candidate.lower() not in used_colors:
                return candidate
        return METHOD_QC_COLORS[len(used_colors) % len(METHOD_QC_COLORS)]

    def _rebuild_method_rows(self) -> None:
        for row_info in list(self._method_row_widgets.values()):
            w = row_info.get("widget")
            if w is not None:
                self.methods_container_layout.removeWidget(w)
                w.deleteLater()
        self._method_row_widgets.clear()

        if not self._methods:
            self.lbl_no_methods.show()
            return

        self.lbl_no_methods.hide()
        selected_set = set(self.selected_methods)

        for method_name in self._methods:
            row_w = QWidget()
            row_w.setStyleSheet("background-color: transparent;")
            hl = QHBoxLayout(row_w)
            hl.setContentsMargins(0, 1, 0, 1)
            hl.setSpacing(6)

            chk = QCheckBox()
            chk.setChecked(method_name in selected_set)
            chk.setToolTip(f"Include '{method_name}' in Disagreement QC comparison")

            hex_color = self.method_colors.get(method_name, METHOD_QC_COLORS[0])
            btn_swatch = QPushButton()
            btn_swatch.setFixedSize(16, 16)
            btn_swatch.setCursor(Qt.PointingHandCursor)
            btn_swatch.setToolTip(f"Change QC overlay color for '{method_name}'")
            btn_swatch.setStyleSheet(self._swatch_stylesheet(hex_color))

            n_cells = self.method_cell_counts.get(method_name)
            label_text = (
                f"{method_name} ({n_cells:,} cells)" if n_cells is not None else method_name
            )
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #E0E0E0; font-size: 12px; background-color: transparent;")

            chk.toggled.connect(
                lambda checked, m=method_name: self._on_method_checked_toggled(m, checked)
            )
            btn_swatch.clicked.connect(
                lambda _checked=False, m=method_name: self._on_swatch_clicked(m)
            )

            hl.addWidget(chk)
            hl.addWidget(btn_swatch)
            hl.addWidget(lbl, 1)

            self.methods_container_layout.addWidget(row_w)
            self._method_row_widgets[method_name] = {
                "widget": row_w,
                "checkbox": chk,
                "swatch": btn_swatch,
                "label": lbl,
            }

    def _on_method_checked_toggled(self, method_name: str, checked: bool) -> None:
        self._user_toggled_methods.add(method_name)
        selected_set = {
            m
            for m, info in self._method_row_widgets.items()
            if info["checkbox"].isChecked()
        }
        if checked:
            selected_set.add(method_name)
        else:
            selected_set.discard(method_name)
        self.selected_methods = [m for m in self._methods if m in selected_set]
        self.selectedMethodsChanged.emit(list(self.selected_methods))

    def _on_swatch_clicked(self, method_name: str) -> None:
        cur_hex = self.method_colors.get(method_name, METHOD_QC_COLORS[0])
        chosen = QColorDialog.getColor(
            QColor(cur_hex), self, f"Select QC Color: {method_name}"
        )
        if chosen.isValid():
            self.set_method_color(method_name, chosen.name(), emit=True)

    # Handlers & Helpers

    def _on_active_method_changed(self, method_name: str) -> None:
        if method_name:
            self.activeMethodChanged.emit(method_name)

    def _on_threshold_changed(self, int_value: int) -> None:
        val = int_value / 100.0
        self.lbl_threshold_value.setText(f"{val:.2f}")
        self.thresholdChanged.emit(val)

    def _emit_overlay_visibility(self, _checked: bool = False) -> None:
        self.overlayVisibilityChanged.emit(
            self.chk_show_heatmap.isChecked(),
            self.chk_show_contours.isChecked(),
        )

    def _on_flicker_clicked(self) -> None:
        if self._flicker_pair is not None:
            m_a, m_b = self._flicker_pair
            if m_a and m_b and m_a != m_b:
                self.flickerPairRequested.emit(m_a, m_b)
                return

        ref = self.combo_ref_method.currentText().strip()
        candidates = [m for m in (self.selected_methods or self._methods) if m]
        if len(candidates) >= 2:
            if ref and ref in candidates:
                other = next((m for m in candidates if m != ref), candidates[1])
                self.flickerPairRequested.emit(ref, other)
            else:
                self.flickerPairRequested.emit(candidates[0], candidates[1])
        elif len(candidates) == 1:
            self.flickerPairRequested.emit(candidates[0], candidates[0])

    # Public API

    def set_computing_state(self, is_computing: bool) -> None:
        """Toggle UI controls while disagreement computation is running."""
        self.btn_compute_disagreement.setEnabled(not is_computing)
        if is_computing:
            self.lbl_compute_status.setText("Computing Model Disagreements...")
            self.lbl_compute_status.setStyleSheet(
                "color: #44AAFF; font-weight: bold; font-size: 11px; "
                "background-color: transparent;"
            )
        else:
            self.lbl_compute_status.setText("Disagreement QC Ready")
            self.lbl_compute_status.setStyleSheet(
                "color: #44AA66; font-weight: bold; font-size: 11px; "
                "background-color: transparent;"
            )

    def get_threshold(self) -> float:
        return self.slider_threshold.value() / 100.0

    def get_category_filter(self) -> str:
        return self.combo_category_filter.currentText()

    def get_active_method(self) -> str:
        return self.combo_active_method.currentText().strip()

    def get_reference_method(self) -> str:
        return self.combo_ref_method.currentText().strip()

    def get_selected_methods(self) -> List[str]:
        """Return checked method names in display order."""
        return [m for m in self._methods if m in set(self.selected_methods)]

    def get_method_colors(self) -> Dict[str, str]:
        """Return mapping of method name to assigned QC hex color."""
        return {m: self.method_colors[m] for m in self._methods if m in self.method_colors}

    def get_overlay_color_mode(self) -> str:
        """Return 'Color by Method' or 'Color by Issue Type'."""
        return self.combo_overlay_color.currentText()

    def set_overlay_color_mode(self, mode: str, *, emit: bool = True) -> None:
        """Programmatically set the overlay color mode selector."""
        if mode not in self.OVERLAY_COLOR_MODES:
            return
        if not emit:
            self.combo_overlay_color.blockSignals(True)
        self.combo_overlay_color.setCurrentText(mode)
        if not emit:
            self.combo_overlay_color.blockSignals(False)

    def set_method_color(
        self, method_name: str, hex_color: str, *, emit: bool = True
    ) -> None:
        """Assign *hex_color* to *method_name* and refresh its swatch."""
        self.method_colors[method_name] = hex_color
        row_info = self._method_row_widgets.get(method_name)
        if row_info is not None and "swatch" in row_info:
            row_info["swatch"].setStyleSheet(self._swatch_stylesheet(hex_color))
        if emit:
            self.methodColorsChanged.emit(dict(self.method_colors))

    def set_method_checked(
        self, method_name: str, checked: bool, *, emit: bool = True
    ) -> None:
        """Check or uncheck *method_name* in the Method Comparison & Color Key."""
        if method_name not in self._methods:
            return
        self._user_toggled_methods.add(method_name)
        row_info = self._method_row_widgets.get(method_name)
        if row_info is not None and "checkbox" in row_info:
            row_info["checkbox"].blockSignals(True)
            row_info["checkbox"].setChecked(bool(checked))
            row_info["checkbox"].blockSignals(False)
        selected_set = set(self.selected_methods)
        if checked:
            selected_set.add(method_name)
        else:
            selected_set.discard(method_name)
        self.selected_methods = [m for m in self._methods if m in selected_set]
        if emit:
            self.selectedMethodsChanged.emit(list(self.selected_methods))

    def set_method_cell_counts(self, cell_counts: Dict[str, int]) -> None:
        """Update displayed `(N cells)` counts for known methods."""
        for m, cnt in cell_counts.items():
            self.method_cell_counts[m] = int(cnt)
            row_info = self._method_row_widgets.get(m)
            if row_info is not None and "label" in row_info:
                row_info["label"].setText(f"{m} ({int(cnt):,} cells)")

    def set_flicker_pair(self, method_a: str, method_b: str) -> None:
        self._flicker_pair = (method_a, method_b)

    def update_available_methods(
        self,
        methods: List[str],
        active_method: Optional[str] = None,
        cell_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """Populate active mask, reference dropdown, and Method Color Key safely."""
        clean_methods = [str(m) for m in methods if str(m).strip()]
        if cell_counts is not None:
            for m in clean_methods:
                if m in cell_counts:
                    self.method_cell_counts[m] = int(cell_counts[m])

        prev_selected = set(self.selected_methods)
        prev_known = set(self._known_methods)
        self._methods = clean_methods
        self.cached_stats["methods"] = list(clean_methods)

        if not clean_methods:
            self._known_methods = []
            self._user_toggled_methods.clear()
            self.selected_methods = []
            self.method_cell_counts.clear()

        # Resolve per-method colors cleanly without clobbering user customizations
        used_colors = {
            self.method_colors[m].lower()
            for m in clean_methods
            if m in self.method_colors
        }
        for m in clean_methods:
            if m not in self.method_colors:
                assigned = self._assign_default_color(m, used_colors)
                self.method_colors[m] = assigned
                used_colors.add(assigned.lower())

        # Determine default checked methods (auto-uncheck "Loaded" when >= 2 computed pipelines exist)
        computed_methods = [m for m in clean_methods if m != "Loaded"]
        has_two_computed = len(computed_methods) >= 2
        new_selected: List[str] = []
        for m in clean_methods:
            if m == "Loaded" and has_two_computed and "Loaded" not in self._user_toggled_methods:
                continue
            if m in prev_known:
                if m in prev_selected:
                    new_selected.append(m)
            else:
                new_selected.append(m)
        self.selected_methods = new_selected
        self._known_methods = list(clean_methods)
        self._rebuild_method_rows()

        prev_active = active_method or self.combo_active_method.currentText()
        prev_ref = self.combo_ref_method.currentText()

        self.combo_active_method.blockSignals(True)
        self.combo_ref_method.blockSignals(True)
        self.combo_active_method.clear()
        self.combo_ref_method.clear()

        self.combo_active_method.addItems(clean_methods)
        self.combo_ref_method.addItems(clean_methods)

        if prev_active in clean_methods:
            self.combo_active_method.setCurrentText(prev_active)
        elif clean_methods:
            self.combo_active_method.setCurrentIndex(0)

        ref_candidates = self.selected_methods or clean_methods
        if prev_ref in ref_candidates:
            self.combo_ref_method.setCurrentText(prev_ref)
        elif ref_candidates:
            self.combo_ref_method.setCurrentText(ref_candidates[0])
        elif clean_methods:
            self.combo_ref_method.setCurrentIndex(0)

        self.combo_active_method.blockSignals(False)
        self.combo_ref_method.blockSignals(False)

        curr_active = self.combo_active_method.currentText()
        if curr_active and curr_active != prev_active:
            self.activeMethodChanged.emit(curr_active)

    def update_statistics(
        self,
        total: int = 0,
        agreed: int = 0,
        flagged: int = 0,
        mean_iou: float = 0.0,
        category_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """Update KPI labels and cached_stats from a DisagreementReport."""
        counts = dict(category_counts or {})
        self.cached_stats = {
            "total": total,
            "agreed": agreed,
            "flagged": flagged,
            "mean_iou": mean_iou,
            "category_counts": counts,
            "methods": list(self._methods),
        }

        self.lbl_total_cells.setText(f"Reference Cells: {total:,}")
        if total > 0:
            self.lbl_agreed_cells.setText(
                f"Agreed Cells: {agreed:,} ({agreed / total * 100:.1f}%)"
            )
        else:
            self.lbl_agreed_cells.setText("Agreed Cells: -")

        self.lbl_flagged_regions.setText(f"Flagged Regions: {flagged:,}")
        self.lbl_issue_count.setText(f"{flagged} flagged regions")
        self.lbl_mean_iou.setText(f"Mean Peer IoU: {mean_iou:.3f}")

        n_split = counts.get("Split", 0)
        n_merge = counts.get("Merge", 0)
        n_shift = counts.get("Boundary Shift", 0)
        n_excl = counts.get("Model-Exclusive", 0)
        self.lbl_category_breakdown.setText(
            f"Split: {n_split} | Merge: {n_merge} | Shift: {n_shift} | Exclusive: {n_excl}"
        )

        self.statisticsUpdated.emit(total, agreed, flagged, mean_iou)

    def show_region_review(self, item: Any) -> None:
        """Display inspection details for a DisagreementItem or dict."""
        if item is None:
            self.clear_region_review()
            return

        if isinstance(item, dict):
            region_id = item.get("region_id", 0)
            category = item.get("category", "Unknown")
            best_iou = float(item.get("best_iou", 0.0))
            dice = float(item.get("dice", 0.0))
            ov = float(item.get("overlap_coeff", 0.0))
            method_cell_ids = dict(item.get("method_cell_ids", {}))
        else:
            region_id = getattr(item, "region_id", 0)
            category = getattr(item, "category", "Unknown")
            best_iou = float(getattr(item, "best_iou", 0.0))
            dice = float(getattr(item, "dice", 0.0))
            ov = float(getattr(item, "overlap_coeff", 0.0))
            method_cell_ids = dict(getattr(item, "method_cell_ids", {}))

        self.cached_review = {
            "region_id": region_id,
            "category": category,
            "best_iou": best_iou,
            "dice": dice,
            "overlap_coeff": ov,
            "method_cell_ids": method_cell_ids,
        }

        self.lbl_review_region.setText(f"Region ID: {region_id}")
        self.lbl_review_category.setText(f"Category: {category}")
        self.lbl_review_iou.setText(
            f"Best IoU: {best_iou:.3f} | Dice: {dice:.3f} | Overlap: {ov * 100.0:.0f}%"
        )

        parts = [f"{m} ({len(ids)})" for m, ids in sorted(method_cell_ids.items())]
        self.lbl_review_methods.setText(
            "Models: " + (", ".join(parts) if parts else "None")
        )
        self.regionReviewShown.emit(item)

    def clear_region_review(self) -> None:
        """Reset region inspection readout to idle state."""
        self.cached_review = None
        self.lbl_review_region.setText("No region selected")
        self.lbl_review_category.setText("Category: -")
        self.lbl_review_iou.setText("Best IoU: -")
        self.lbl_review_methods.setText("Models: -")
        self.regionReviewCleared.emit()
