"""
IRIS-Viewer Multi-Model Disagreement Panel Widget
=================================================
Manages active mask selection, per-method QC selection & color key, multi-model
disagreement computation, category filtering, hotspot navigation, model
flickering, and KPI metrics.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from skimage.measure import find_contours

from iris_viewer.engine.curation import FINAL_CURATED_MASK_NAME, NO_CELL_CHOICE
from iris_viewer.engine.disagreement import (
    FALLBACK_PATCH_CHANNEL_COLORS,
    METHOD_QC_COLORS,
    _extract_masks_for_mode,
    compose_multichannel_patch_rgb,
)
from iris_viewer.widgets.styles import (
    BUTTON_CTA,
    BUTTON_PRIMARY,
    BUTTON_SECONDARY,
    CONFIDENCE_HIGH_COLOR,
    CONFIDENCE_LOW_COLOR,
    CONFIDENCE_MEDIUM_COLOR,
    SHARED_DARK_THEME,
    create_section_card,
)


def resolve_state_active_image(state: Any) -> Any:
    """Return active raw single-sample or active TMA sample image from *state*."""
    if state is None:
        return None
    img = getattr(state, "raw_image", None)
    if img is not None:
        return img
    multi = getattr(state, "multi_sample_images", None)
    if multi:
        cur = getattr(state, "current_sample_name", None)
        return multi.get(cur) if cur in multi else next(iter(multi.values()))
    return None


def resolve_patch_channels_from_ctx(
    ctx: Any,
) -> Tuple[List[str], List[int], Dict[int, str], Dict[int, Tuple[float, float]]]:
    """Resolve marker names, currently visible Napari channel indices, colors, and limits."""
    state = getattr(ctx, "state", None) if ctx is not None else None
    viewer = getattr(ctx, "viewer", None) if ctx is not None else None
    img = resolve_state_active_image(state)

    n_img_ch = 0
    if img is not None:
        ndim = getattr(img, "ndim", 0)
        shape = getattr(img, "shape", ())
        n_img_ch = 1 if ndim == 2 else (int(shape[0]) if ndim >= 3 else 0)

    raw_markers = list(getattr(state, "markers", []) or []) if state is not None else []
    n_markers = max(len(raw_markers), n_img_ch)
    marker_names = [
        str(raw_markers[i]) if i < len(raw_markers) else f"Channel {i + 1}"
        for i in range(n_markers)
    ]

    layer_by_base: Dict[str, Any] = {}
    if viewer is not None and hasattr(viewer, "layers"):
        from iris_viewer.core.layers import get_layer_base_name, get_layer_type

        cur_sample = getattr(state, "current_sample_name", None) if state is not None else None
        skip_bases = {
            "Disagreement Heatmap",
            "Disagreement Contours",
            "Method Comparison Overlay",
            "Active Disagreement Isolation",
            "Segmentation Preview ROI",
            "Review BBox",
        }
        skip_types = {
            "disagreement_method_overlay",
            "disagreement_isolation",
            "phenotype",
            "bounds",
        }
        for lyr in viewer.layers:
            if hasattr(lyr, "contour") or get_layer_type(lyr) in skip_types:
                continue
            base = get_layer_base_name(lyr)
            if base in skip_bases:
                continue
            if base not in layer_by_base or (
                cur_sample and cur_sample in getattr(lyr, "name", "")
            ):
                layer_by_base[base] = lyr

    import matplotlib.colors as mcolors

    active_indices: List[int] = []
    channel_colors: Dict[int, str] = {}
    contrast_limits: Dict[int, Tuple[float, float]] = {}

    color_svc = getattr(ctx, "color_svc", None) if ctx is not None else None
    state_colors = dict(getattr(state, "channel_colors", {}) or {}) if state is not None else {}
    state_limits = (
        dict(getattr(state, "channel_contrast_limits", {}) or {}) if state is not None else {}
    )

    for idx, m_name in enumerate(marker_names):
        lyr = layer_by_base.get(m_name)
        if lyr is not None and bool(getattr(lyr, "visible", False)):
            active_indices.append(idx)

        resolved_hex: Optional[str] = None
        if lyr is not None and hasattr(lyr, "colormap"):
            try:
                cmap = lyr.colormap
                colors = getattr(cmap, "colors", None)
                if colors is not None and len(colors) > 0:
                    rgb_top = np.asarray(colors[-1][:3], dtype=float)
                    if float(np.max(rgb_top)) > 0.05:
                        resolved_hex = mcolors.to_hex(np.clip(rgb_top, 0.0, 1.0))
            except Exception:
                pass
            if hasattr(lyr, "contrast_limits"):
                try:
                    cl = tuple(float(v) for v in lyr.contrast_limits[:2])
                    if len(cl) == 2 and cl[1] > cl[0]:
                        contrast_limits[idx] = (cl[0], cl[1])
                except Exception:
                    pass

        if (
            resolved_hex is None
            and color_svc is not None
            and hasattr(color_svc, "get_marker_color")
        ):
            try:
                cval = color_svc.get_marker_color(m_name)
                if isinstance(cval, str) and mcolors.is_color_like(cval):
                    resolved_hex = (
                        "#D8DCE6" if cval.lower() in ("gray", "grey") else mcolors.to_hex(cval)
                    )
            except Exception:
                pass

        if resolved_hex is None:
            for key in (m_name, m_name.lower(), m_name.upper()):
                cval = state_colors.get(key)
                if isinstance(cval, str) and mcolors.is_color_like(cval):
                    resolved_hex = (
                        "#D8DCE6" if cval.lower() in ("gray", "grey") else mcolors.to_hex(cval)
                    )
                    break

        if resolved_hex is None:
            resolved_hex = FALLBACK_PATCH_CHANNEL_COLORS[idx % len(FALLBACK_PATCH_CHANNEL_COLORS)]
        elif resolved_hex.lower() in ("#808080", "#ffffff") and idx == 0:
            resolved_hex = "#D8DCE6"

        channel_colors[idx] = resolved_hex

        if idx not in contrast_limits and m_name in state_limits:
            try:
                cl = tuple(float(v) for v in state_limits[m_name][:2])
                if len(cl) == 2 and cl[1] > cl[0]:
                    contrast_limits[idx] = (cl[0], cl[1])
            except Exception:
                pass

    if not active_indices and n_markers > 0:
        active_indices = [0]

    return marker_names, active_indices, channel_colors, contrast_limits


def extract_item_region_id(item: Any) -> Optional[int]:
    """Return integer ``region_id`` from a DisagreementItem or dict, or ``None``."""
    if item is None:
        return None
    raw = item.get("region_id") if isinstance(item, dict) else getattr(item, "region_id", None)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def resolve_curated_state_from_ctx(ctx: Any) -> Any:
    """Resolve active sample's ``CuratedMaskState`` from ``ctx`` if available."""
    if ctx is None:
        return None
    review_svc = getattr(ctx, "review_svc", None)
    if review_svc is not None and hasattr(review_svc, "_get_existing_curated_state"):
        try:
            existing = review_svc._get_existing_curated_state()
            if existing is not None:
                return existing
        except Exception:
            pass
    state = getattr(ctx, "state", None)
    if state is None:
        return None
    seg = getattr(state, "segmentation_results", {}) or {}
    if FINAL_CURATED_MASK_NAME not in seg:
        return None
    cand = getattr(state, "curated_mask_state", None)
    if cand is None:
        return None
    cur_sample = getattr(state, "current_sample_name", None)
    cand_sample = getattr(state, "_curated_sample_name", cur_sample)
    if cand_sample != cur_sample:
        return None
    return cand


def compute_curation_progress(
    ctx: Any,
    curated: Any,
    fallback_items: Sequence[Any] = (),
) -> Tuple[int, int]:
    """Return ``(curated_count, total_hotspots)`` for the current disagreement report."""
    state = getattr(ctx, "state", None) if ctx is not None else None
    report = getattr(state, "disagreement_report", None) if state is not None else None
    if report is not None and getattr(report, "items", None) is not None:
        all_items = list(report.items)
    elif ctx is not None and getattr(ctx, "flagged_cells", None):
        all_items = list(ctx.flagged_cells)
    else:
        all_items = list(fallback_items)

    total = len(all_items)
    if total == 0:
        return (0, 0)
    decisions = getattr(curated, "region_decisions", {}) or {} if curated is not None else {}
    curated_cnt = sum(
        1
        for it in all_items
        if (rid := extract_item_region_id(it)) is not None and rid in decisions
    )
    return (curated_cnt, total)


def count_item_cells_for_method(
    item: Any,
    method_name: str,
    seg_results: Dict[str, Any],
    compare_mode: str = "Nuclear Masks",
) -> int:
    """Count cells belonging to *method_name* in hotspot *item*."""
    if item is None or not method_name:
        return 0
    m_ids = (
        item.get("method_cell_ids", {})
        if isinstance(item, dict)
        else getattr(item, "method_cell_ids", {})
    ) or {}
    if method_name in m_ids:
        return len(m_ids[method_name])

    entry = seg_results.get(method_name)
    if not isinstance(entry, dict):
        return 0
    if compare_mode == "Cytoplasm Masks":
        mask = (
            entry.get("cytoplasm")
            if entry.get("cytoplasm") is not None
            else entry.get("nuclear")
        )
    else:
        mask = (
            entry.get("nuclear")
            if entry.get("nuclear") is not None
            else entry.get("cytoplasm")
        )
    if mask is None:
        return 0

    arr = np.asarray(mask)
    bbox = item.get("bbox", None) if isinstance(item, dict) else getattr(item, "bbox", None)
    if bbox is None or len(bbox) < 4:
        return 0
    h, w = arr.shape[:2]
    r0 = max(0, min(int(bbox[0]), h))
    c0 = max(0, min(int(bbox[1]), w))
    r1 = max(0, min(int(bbox[2]), h))
    c1 = max(0, min(int(bbox[3]), w))
    if r1 <= r0 or c1 <= c0:
        return 0
    sub = arr[r0:r1, c0:c1]
    uniq = np.unique(sub)
    return int(np.sum(uniq > 0))


def build_keep_button_stylesheet(
    hex_color: str,
    *,
    active: bool = False,
    is_no_cell: bool = False,
) -> str:
    """Build inline QSS for hotspot Keep buttons (always spaced after closing braces)."""
    qc = QColor(hex_color)
    if not qc.isValid():
        qc = QColor("#3A6A9A")
    r, g, b = qc.red(), qc.green(), qc.blue()
    if active:
        bg = f"rgba({r}, {g}, {b}, 88)" if not is_no_cell else "rgba(204, 68, 68, 95)"
        bg_hover = (
            f"rgba({r}, {g}, {b}, 120)" if not is_no_cell else "rgba(204, 68, 68, 130)"
        )
        border = qc.name() if not is_no_cell else "#FF6B6B"
        return (
            f"QPushButton {{ background-color: {bg}; color: #FFFFFF; "
            f"border: 2px solid {border}; border-radius: 4px; "
            "padding: 5px 8px; font-weight: bold; font-size: 11px; } "
            f"QPushButton:hover {{ background-color: {bg_hover}; }} "
            "QPushButton:disabled { background-color: #2A3A4A; color: #777777; "
            "border-color: #3c4352; }"
        )

    bg = f"rgba({r}, {g}, {b}, 30)" if not is_no_cell else "#2D2327"
    bg_hover = f"rgba({r}, {g}, {b}, 58)" if not is_no_cell else "#3D2B31"
    border = qc.name() if not is_no_cell else "#8B3A3A"
    return (
        f"QPushButton {{ background-color: {bg}; color: #F0F2F5; "
        f"border: 1px solid {border}; border-radius: 4px; "
        "padding: 5px 8px; font-weight: bold; font-size: 11px; } "
        f"QPushButton:hover {{ background-color: {bg_hover}; }} "
        "QPushButton:disabled { background-color: #2A3A4A; color: #777777; "
        "border-color: #3c4352; }"
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
    contourStyleChanged = Signal()
    qcSampleChanged = Signal(str)

    # Final Mask Builder & Curation Signals
    useSingleModelRequested = Signal(str)
    autoConsensusRequested = Signal()
    overrideHotspotsRequested = Signal(str, object)
    drawCurationRoiRequested = Signal()
    applyCurationRoiRequested = Signal(str)
    hotspotKeepRequested = Signal(str, bool)
    resetCurationRequested = Signal()
    skipHotspotRequested = Signal()
    curationStateChanged = Signal()

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
        self.ctx: Any = None
        self._methods: List[str] = []
        self._known_methods: List[str] = []
        self._qc_samples: List[str] = []
        self._user_toggled_methods: set[str] = set()
        self.method_colors: Dict[str, str] = {}
        self.selected_methods: List[str] = []
        self.method_cell_counts: Dict[str, int] = {}
        self._method_row_widgets: Dict[str, Dict[str, Any]] = {}
        self.hotspot_keep_buttons: Dict[str, QPushButton] = {}
        self._ordered_keep_methods: List[str] = []
        self._flicker_pair: Optional[tuple[str, str]] = None
        self._last_review_item: Any = None
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

        self._build_method_color_key_controls()
        self._build_disagreement_controls()
        self._build_hotspot_controls()
        self._build_final_mask_controls()
        self._build_statistics()
        self.main_layout.addStretch()

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

        self.lbl_no_methods = QLabel(
            "Run segmentation pipelines or load a mask to compare methods."
        )
        self.lbl_no_methods.setStyleSheet(
            "color: #9AA4B2; font-size: 11px; font-style: italic; background-color: transparent;"
        )
        self.lbl_no_methods.setWordWrap(True)
        self.methods_container_layout.addWidget(self.lbl_no_methods)

        self.main_layout.addWidget(group)

    def _build_disagreement_controls(self) -> None:
        group, gl = create_section_card("Disagreement Benchmark")

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

        self.qc_sample_row_widget = QWidget()
        self.qc_sample_row_widget.setStyleSheet("background-color: transparent;")
        row_qc_sample = QHBoxLayout(self.qc_sample_row_widget)
        row_qc_sample.setContentsMargins(0, 0, 0, 0)
        row_qc_sample.setSpacing(8)
        row_qc_sample.addWidget(QLabel("Sample:"))
        self.combo_qc_sample = QComboBox()
        self.combo_qc_sample.setToolTip(
            "Select which segmented sample to inspect on the Disagreement QC overlay"
        )
        self.combo_qc_sample.currentTextChanged.connect(self._on_qc_sample_changed)
        row_qc_sample.addWidget(self.combo_qc_sample, stretch=1)
        self.qc_sample_row_widget.setVisible(False)
        gl.addWidget(self.qc_sample_row_widget)

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
        self.chk_show_contours = QCheckBox("Show Contours")
        self.chk_show_contours.setChecked(True)
        self.chk_show_contours.setToolTip(
            "Overlay hollow color-coded contours so underlying marker channels remain visible"
        )
        self.chk_show_heatmap = QCheckBox("Show Heatmap")
        self.chk_show_heatmap.setChecked(False)
        self.chk_show_heatmap.setToolTip(
            "Overlay an amber-to-red disagreement density heatmap on disputed regions"
        )
        self.chk_show_heatmap.toggled.connect(self._emit_overlay_visibility)
        self.chk_show_contours.toggled.connect(self._emit_overlay_visibility)
        row_overlays.addWidget(self.chk_show_contours)
        row_overlays.addWidget(self.chk_show_heatmap)
        gl.addLayout(row_overlays)

        row_contour_style = QHBoxLayout()
        row_contour_style.setSpacing(8)
        row_contour_style.addWidget(QLabel("Contour:"))
        self.spin_contour_width = QSpinBox()
        self.spin_contour_width.setRange(1, 6)
        self.spin_contour_width.setValue(2)
        self.spin_contour_width.setSuffix(" px")
        self.spin_contour_width.setToolTip("Boundary contour line thickness in pixels")
        self.spin_contour_width.valueChanged.connect(lambda _v: self.contourStyleChanged.emit())
        row_contour_style.addWidget(self.spin_contour_width)

        self.chk_keep_filled_mask = QCheckBox("Keep Filled Mask")
        self.chk_keep_filled_mask.setChecked(False)
        self.chk_keep_filled_mask.setToolTip(
            "Keep solid filled segmentation masks visible under QC contours; "
            "leave unchecked to inspect raw fluorescence markers directly."
        )
        self.chk_keep_filled_mask.toggled.connect(lambda _c: self.contourStyleChanged.emit())
        row_contour_style.addWidget(self.chk_keep_filled_mask)
        gl.addLayout(row_contour_style)

        row_thresh = QHBoxLayout()
        row_thresh.setSpacing(8)
        row_thresh.addWidget(QLabel("IoU Threshold:"))
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100)
        self.slider_threshold.setValue(65)
        self.slider_threshold.setTickPosition(QSlider.TicksBelow)
        self.slider_threshold.setTickInterval(10)
        self.slider_threshold.setToolTip(
            "Cells below this Intersection-over-Union (IoU) overlap are flagged as hotspots."
        )
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)
        row_thresh.addWidget(self.slider_threshold)

        self.lbl_threshold_value = QLabel("0.65")
        self.lbl_threshold_value.setFixedWidth(40)
        self.lbl_threshold_value.setStyleSheet(
            "color: #F0F2F5; font-weight: bold; background-color: transparent;"
        )
        row_thresh.addWidget(self.lbl_threshold_value)
        gl.addLayout(row_thresh)

        self.main_layout.addWidget(group)

    def _build_hotspot_controls(self) -> None:
        group, gl = create_section_card(
            "Hotspot Inspection",
            "Step through disputed cells or click directly on the canvas to inspect boundaries.",
        )

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

        # Embedded Hotspot Patch Preview (matches Segmentation Pop-up Hotspot Patch Inspector)
        self.fig_mini_hotspot = Figure(figsize=(3.2, 2.2), dpi=90)
        self.fig_mini_hotspot.patch.set_facecolor("#14161d")
        self.canvas_mini_hotspot = FigureCanvas(self.fig_mini_hotspot)
        self.canvas_mini_hotspot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.canvas_mini_hotspot.setMinimumHeight(185)
        self.canvas_mini_hotspot.setMaximumHeight(215)
        gl.addWidget(self.canvas_mini_hotspot)

        # Compact Interactive Hotspot Method-Selection Bar
        self.hotspot_curation_bar = QFrame()
        self.hotspot_curation_bar.setStyleSheet(
            "QFrame { background-color: #1b1f27; border: 1px solid #3c4352; "
            "border-radius: 6px; }"
        )
        hb_lyt = QVBoxLayout(self.hotspot_curation_bar)
        hb_lyt.setContentsMargins(8, 6, 8, 6)
        hb_lyt.setSpacing(5)

        lbl_keep_hdr = QLabel("Keep segmentation for this hotspot:")
        lbl_keep_hdr.setWordWrap(True)
        lbl_keep_hdr.setStyleSheet(
            "color: #F0F2F5; font-size: 11px; font-weight: bold; border: none; "
            "background-color: transparent;"
        )
        hb_lyt.addWidget(lbl_keep_hdr)

        hb_ctrl = QHBoxLayout()
        hb_ctrl.setSpacing(6)
        self.lbl_curation_progress = QLabel("Curated: 0 / 0")
        self.lbl_curation_progress.setStyleSheet(
            "color: #CCAA44; font-size: 11px; font-weight: bold; border: none; "
            "background-color: transparent;"
        )
        hb_ctrl.addWidget(self.lbl_curation_progress)
        hb_ctrl.addStretch(1)

        self.chk_auto_advance = QCheckBox("Auto-advance")
        self.chk_auto_advance.setChecked(True)
        self.chk_auto_advance.setFocusPolicy(Qt.NoFocus)
        self.chk_auto_advance.setStyleSheet("border: none; background-color: transparent;")
        hb_ctrl.addWidget(self.chk_auto_advance)

        self.btn_skip_hotspot = QPushButton("Skip ->")
        self.btn_skip_hotspot.setMinimumHeight(24)
        self.btn_skip_hotspot.setFocusPolicy(Qt.NoFocus)
        self.btn_skip_hotspot.setStyleSheet(BUTTON_SECONDARY)
        self.btn_skip_hotspot.clicked.connect(lambda: self.skipHotspotRequested.emit())
        hb_ctrl.addWidget(self.btn_skip_hotspot)
        hb_lyt.addLayout(hb_ctrl)

        self.hotspot_keep_grid = QGridLayout()
        self.hotspot_keep_grid.setContentsMargins(0, 0, 0, 0)
        self.hotspot_keep_grid.setSpacing(5)
        hb_lyt.addLayout(self.hotspot_keep_grid)

        self.btn_keep_no_cell = QPushButton("[0] No Cell / Artifact")
        self.btn_keep_no_cell.setMinimumHeight(26)
        self.btn_keep_no_cell.setFocusPolicy(Qt.NoFocus)
        self.btn_keep_no_cell.clicked.connect(
            lambda _checked=False: self._on_hotspot_keep_method_clicked(NO_CELL_CHOICE)
        )
        hb_lyt.addWidget(self.btn_keep_no_cell)

        gl.addWidget(self.hotspot_curation_bar)

        self.lbl_mini_patch_caption = QLabel(
            "Patch preview uses currently visible marker channels "
            "(open Pop-up Window for channel controls)."
        )
        self.lbl_mini_patch_caption.setWordWrap(True)
        self.lbl_mini_patch_caption.setStyleSheet(
            "color: #8892A0; font-size: 10px; font-style: italic; background-color: transparent;"
        )
        gl.addWidget(self.lbl_mini_patch_caption)
        self._render_mini_hotspot_preview(None)
        self._refresh_hotspot_curation_bar()

        self.main_layout.addWidget(group)

    def _build_final_mask_controls(self) -> None:
        group, gl = create_section_card(
            "Final Mask Curation",
            "Keep per-hotspot picks above, auto-resolve uncurated, or apply batch/ROI.",
        )

        # Keep legacy single-pipeline widgets hidden so programmatic callers still work cleanly
        self._legacy_base_container = QWidget()
        self._legacy_base_container.setVisible(False)
        legacy_lyt = QHBoxLayout(self._legacy_base_container)
        legacy_lyt.setContentsMargins(0, 0, 0, 0)
        self.combo_active_method = QComboBox()
        self.combo_active_method.setVisible(False)
        self.combo_active_method.currentTextChanged.connect(self._on_active_method_changed)
        self.btn_use_single_model = QPushButton("Use Selected Pipeline as Final Mask")
        self.btn_use_single_model.setVisible(False)
        self.btn_use_single_model.clicked.connect(self._on_use_single_model_clicked)
        legacy_lyt.addWidget(self.combo_active_method)
        legacy_lyt.addWidget(self.btn_use_single_model)
        gl.addWidget(self._legacy_base_container)

        self.btn_auto_consensus = QPushButton("Auto-Resolve Uncurated (Peer Consensus)")
        self.btn_auto_consensus.setMinimumHeight(28)
        self.btn_auto_consensus.setStyleSheet(BUTTON_PRIMARY)
        self.btn_auto_consensus.clicked.connect(
            lambda: self.autoConsensusRequested.emit()
        )
        gl.addWidget(self.btn_auto_consensus)

        self.btn_reset_curation = QPushButton("Reset to Reference")
        self.btn_reset_curation.setMinimumHeight(26)
        self.btn_reset_curation.setStyleSheet(BUTTON_SECONDARY)
        self.btn_reset_curation.clicked.connect(lambda: self.resetCurationRequested.emit())
        gl.addWidget(self.btn_reset_curation)

        row_override = QHBoxLayout()
        row_override.setSpacing(8)
        row_override.addWidget(QLabel("Batch / ROI Method:"))
        self.combo_override_method = QComboBox()
        row_override.addWidget(self.combo_override_method, stretch=1)
        gl.addLayout(row_override)

        row_hs_override = QHBoxLayout()
        row_hs_override.setSpacing(6)
        self.btn_accept_hotspot = QPushButton("Keep in Current Hotspot")
        self.btn_accept_hotspot.setMinimumHeight(26)
        self.btn_accept_hotspot.setStyleSheet(BUTTON_SECONDARY)
        self.btn_accept_hotspot.clicked.connect(self._on_accept_hotspot_clicked)

        self.btn_accept_category = QPushButton("Keep for Filtered Category")
        self.btn_accept_category.setMinimumHeight(26)
        self.btn_accept_category.setStyleSheet(BUTTON_SECONDARY)
        self.btn_accept_category.clicked.connect(self._on_accept_category_clicked)

        row_hs_override.addWidget(self.btn_accept_hotspot)
        row_hs_override.addWidget(self.btn_accept_category)
        gl.addLayout(row_hs_override)

        row_roi_override = QHBoxLayout()
        row_roi_override.setSpacing(6)
        self.btn_draw_curation_roi = QPushButton("Draw ROI")
        self.btn_draw_curation_roi.setMinimumHeight(26)
        self.btn_draw_curation_roi.setStyleSheet(BUTTON_SECONDARY)
        self.btn_draw_curation_roi.clicked.connect(
            lambda: self.drawCurationRoiRequested.emit()
        )

        self.btn_apply_curation_roi = QPushButton("Keep in ROI")
        self.btn_apply_curation_roi.setMinimumHeight(26)
        self.btn_apply_curation_roi.setStyleSheet(BUTTON_SECONDARY)
        self.btn_apply_curation_roi.clicked.connect(self._on_apply_curation_roi_clicked)

        row_roi_override.addWidget(self.btn_draw_curation_roi)
        row_roi_override.addWidget(self.btn_apply_curation_roi)
        gl.addLayout(row_roi_override)

        self.lbl_final_mask_status = QLabel("Final Mask: None")
        self.lbl_final_mask_status.setWordWrap(True)
        self.lbl_final_mask_status.setStyleSheet(
            "color: #9AA4B2; font-size: 11px; background-color: transparent;"
        )
        gl.addWidget(self.lbl_final_mask_status)

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
            "min-height: 16px; max-height: 16px; } "
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
        self._refresh_override_method_combo()
        self._refresh_hotspot_curation_bar()
        self.selectedMethodsChanged.emit(list(self.selected_methods))
        if self._last_review_item is not None:
            self._render_mini_hotspot_preview(self._last_review_item)

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

    def _on_qc_sample_changed(self, sample_name: str) -> None:
        clean = str(sample_name).strip()
        if clean:
            self.qcSampleChanged.emit(clean)

    def _on_threshold_changed(self, int_value: int) -> None:
        val = int_value / 100.0
        self.lbl_threshold_value.setText(f"{val:.2f}")
        self.thresholdChanged.emit(val)

    def _emit_overlay_visibility(self, _checked: bool = False) -> None:
        self.overlayVisibilityChanged.emit(
            self.chk_show_heatmap.isChecked(),
            self.chk_show_contours.isChecked(),
        )

    def _on_use_single_model_clicked(self) -> None:
        method = self.combo_active_method.currentText().strip()
        if method:
            self.useSingleModelRequested.emit(method)

    def _on_accept_hotspot_clicked(self) -> None:
        method = self.combo_override_method.currentText().strip()
        if not method:
            return
        items = [self._last_review_item] if self._last_review_item is not None else []
        self.overrideHotspotsRequested.emit(method, items)

    def _on_accept_category_clicked(self) -> None:
        method = self.combo_override_method.currentText().strip()
        if method:
            self.overrideHotspotsRequested.emit(method, None)

    def _on_apply_curation_roi_clicked(self) -> None:
        method = self.combo_override_method.currentText().strip()
        if method:
            self.applyCurationRoiRequested.emit(method)

    def _on_hotspot_keep_method_clicked(self, choice: str) -> None:
        self.hotspotKeepRequested.emit(choice, bool(self.chk_auto_advance.isChecked()))

    def _refresh_override_method_combo(self) -> None:
        checked = self.get_selected_methods()
        base_pool = (
            checked
            if checked
            else [m for m in self._methods if m != FINAL_CURATED_MASK_NAME]
        )
        items = list(base_pool)
        if NO_CELL_CHOICE not in items:
            items.append(NO_CELL_CHOICE)

        prev_override = self.combo_override_method.currentText().strip()
        curr_ref = self.combo_ref_method.currentText().strip()

        self.combo_override_method.blockSignals(True)
        self.combo_override_method.clear()
        self.combo_override_method.addItems(items)
        if prev_override in items:
            self.combo_override_method.setCurrentText(prev_override)
        elif base_pool:
            alt = next((m for m in base_pool if m != curr_ref), base_pool[0])
            self.combo_override_method.setCurrentText(alt)
        elif items:
            self.combo_override_method.setCurrentIndex(0)
        self.combo_override_method.blockSignals(False)

    def _refresh_hotspot_curation_bar(self) -> None:
        checked = [m for m in self.get_selected_methods() if m != FINAL_CURATED_MASK_NAME]
        item = self._last_review_item
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
        colors = self.get_method_colors()

        if checked != self._ordered_keep_methods:
            for btn in self.hotspot_keep_buttons.values():
                self.hotspot_keep_grid.removeWidget(btn)
                btn.deleteLater()
            self.hotspot_keep_buttons.clear()
            self._ordered_keep_methods = list(checked)
            for idx, m_name in enumerate(self._ordered_keep_methods):
                btn = QPushButton()
                btn.setMinimumHeight(26)
                btn.setFocusPolicy(Qt.NoFocus)
                btn.clicked.connect(
                    lambda _checked=False, m=m_name: self._on_hotspot_keep_method_clicked(m)
                )
                self.hotspot_keep_grid.addWidget(btn, idx // 2, idx % 2)
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

        cur_x, tot_y = compute_curation_progress(self.ctx, curated)
        self.lbl_curation_progress.setText(f"Curated: {cur_x} / {tot_y}")

    def set_final_mask_status(self, text: str, *, active: bool = True) -> None:
        """Update Final Mask Builder status label."""
        color = CONFIDENCE_HIGH_COLOR if active else "#9AA4B2"
        weight = "bold" if active else "normal"
        self.lbl_final_mask_status.setText(text)
        self.lbl_final_mask_status.setStyleSheet(
            f"color: {color}; font-size: 11px; font-weight: {weight}; "
            "background-color: transparent;"
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

    def get_contour_thickness(self) -> int:
        """Return QC contour boundary thickness in pixels."""
        return int(self.spin_contour_width.value())

    def get_keep_filled_mask(self) -> bool:
        """Return True if opaque filled Labels layers should remain visible during QC."""
        return bool(self.chk_keep_filled_mask.isChecked())

    def get_fill_alpha(self) -> float:
        """Return interior fill opacity (0.0 = hollow Hotspot-Inspector contours)."""
        return 0.0

    def get_qc_sample(self) -> str:
        """Return currently selected sample name for Disagreement QC."""
        return self.combo_qc_sample.currentText().strip()

    def update_available_samples(
        self,
        samples: Sequence[str],
        active_sample: Optional[str] = None,
    ) -> None:
        """Populate available multi-sample QC selector and toggle visibility cleanly."""
        clean = [str(s).strip() for s in samples if str(s).strip()]
        self._qc_samples = list(clean)
        curr = self.combo_qc_sample.currentText().strip()
        self.combo_qc_sample.blockSignals(True)
        self.combo_qc_sample.clear()
        self.combo_qc_sample.addItems(clean)
        if active_sample and active_sample in clean:
            self.combo_qc_sample.setCurrentText(active_sample)
        elif curr in clean:
            self.combo_qc_sample.setCurrentText(curr)
        elif clean:
            self.combo_qc_sample.setCurrentIndex(0)
        self.combo_qc_sample.blockSignals(False)
        self.qc_sample_row_widget.setVisible(len(clean) > 1)

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
        self._refresh_hotspot_curation_bar()
        if self._last_review_item is not None:
            self._render_mini_hotspot_preview(self._last_review_item)

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
        self._refresh_override_method_combo()
        self._refresh_hotspot_curation_bar()
        if emit:
            self.selectedMethodsChanged.emit(list(self.selected_methods))
        if self._last_review_item is not None:
            self._render_mini_hotspot_preview(self._last_review_item)

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
        """Populate pipeline selectors, reference dropdown, and Method Color Key safely."""
        clean_methods = [
            str(m).strip()
            for m in methods
            if str(m).strip() and str(m).strip() != FINAL_CURATED_MASK_NAME
        ]
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

        # Default checked methods: auto-uncheck "Loaded" when >= 2 computed pipelines exist
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

        is_curated_active = active_method == FINAL_CURATED_MASK_NAME
        prev_combo_active = self.combo_active_method.currentText()
        prev_active = (
            prev_combo_active
            if is_curated_active
            else (active_method or prev_combo_active)
        )
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

        self._refresh_override_method_combo()
        self._refresh_hotspot_curation_bar()

        curr_active = self.combo_active_method.currentText()
        if curr_active and curr_active != prev_active and not is_curated_active:
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
        self._refresh_hotspot_curation_bar()

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

        self._last_review_item = item
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
        self._render_mini_hotspot_preview(item)
        self._refresh_hotspot_curation_bar()
        self.regionReviewShown.emit(item)

    def clear_region_review(self) -> None:
        """Reset region inspection readout to idle state."""
        self._last_review_item = None
        self.cached_review = None
        self.lbl_review_region.setText("No region selected")
        self.lbl_review_category.setText("Category: -")
        self.lbl_review_iou.setText("Best IoU: -")
        self.lbl_review_methods.setText("Models: -")
        self._render_mini_hotspot_preview(None)
        self._refresh_hotspot_curation_bar()
        self.regionReviewCleared.emit()

    def _render_mini_hotspot_preview(self, item: Any) -> None:
        """Render embedded Hotspot Patch Inspector preview inside the dock widget."""
        self.fig_mini_hotspot.clear()
        ax = self.fig_mini_hotspot.add_subplot(111)
        ax.set_facecolor("#14161d")

        state = getattr(self.ctx, "state", None) if self.ctx is not None else None
        seg_results = getattr(state, "segmentation_results", {}) if state is not None else {}

        if item is None or not seg_results:
            ax.text(
                0.5,
                0.5,
                "Hotspot Patch Preview\nClick Next Hotspot or Prev Hotspot",
                color="#9AA4B2",
                ha="center",
                va="center",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#3c4352")
            self.fig_mini_hotspot.tight_layout(pad=0.3)
            self.canvas_mini_hotspot.draw()
            return

        bbox = item.get("bbox", (0, 0, 32, 32)) if isinstance(item, dict) else getattr(
            item, "bbox", (0, 0, 32, 32)
        )
        rid = item.get("region_id", 0) if isinstance(item, dict) else getattr(item, "region_id", 0)
        cat = str(
            item.get("category", "Hotspot")
            if isinstance(item, dict)
            else getattr(item, "category", "Hotspot")
        )
        m_cell_ids = dict(
            item.get("method_cell_ids", {})
            if isinstance(item, dict)
            else getattr(item, "method_cell_ids", {})
        )

        selected = self.get_selected_methods() if self._methods else None
        filtered_seg = (
            {m: seg_results[m] for m in selected if m in seg_results}
            if selected
            else {m: r for m, r in seg_results.items() if m != FINAL_CURATED_MASK_NAME}
        )
        compare_mode = self.combo_compare.currentText() or "Nuclear Masks"
        try:
            masks = _extract_masks_for_mode(filtered_seg, compare_mode)
        except Exception:
            masks = {}

        if not masks:
            ax.text(
                0.5,
                0.5,
                f"No masks for '{compare_mode}'",
                color="#9AA4B2",
                ha="center",
                va="center",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self.fig_mini_hotspot.tight_layout(pad=0.3)
            self.canvas_mini_hotspot.draw()
            return

        h, w = next(iter(masks.values())).shape
        r0, c0, r1, c1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        pad = 16
        pr0 = max(0, r0 - pad)
        pc0 = max(0, c0 - pad)
        pr1 = min(h, r1 + pad)
        pc1 = min(w, c1 + pad)
        if pr1 <= pr0 or pc1 <= pc0:
            pr0, pc0, pr1, pc1 = 0, 0, h, w

        marker_names, active_idxs, ch_colors, ch_limits = resolve_patch_channels_from_ctx(self.ctx)
        raw_img = resolve_state_active_image(state)
        rgb_bg = compose_multichannel_patch_rgb(
            raw_img,
            (pr0, pc0, pr1, pc1),
            channel_indices=active_idxs,
            channel_colors=ch_colors,
            contrast_limits=ch_limits,
            local_autocontrast=True,
        )
        if rgb_bg is not None:
            ax.imshow(rgb_bg, origin="upper", interpolation="nearest")
        else:
            ax.imshow(
                np.zeros((pr1 - pr0, pc1 - pc0, 3), dtype=np.float32),
                origin="upper",
            )

        method_colors = self.get_method_colors()
        lw_base = max(1.2, float(self.get_contour_thickness()))
        for idx, (m_name, mask) in enumerate(masks.items()):
            hex_c = method_colors.get(m_name, METHOD_QC_COLORS[idx % len(METHOD_QC_COLORS)])
            sub = mask[pr0:pr1, pc0:pc1]
            cids = m_cell_ids.get(m_name, [])
            for cid in [int(v) for v in np.unique(sub) if v > 0]:
                is_target = cid in cids
                for cnt in find_contours((sub == cid).astype(np.float32), 0.5):
                    ax.plot(
                        cnt[:, 1],
                        cnt[:, 0],
                        color=hex_c,
                        linewidth=lw_base if is_target else max(0.8, lw_base * 0.45),
                        linestyle="-" if is_target else "--",
                        alpha=0.95 if is_target else 0.40,
                    )

        shown_markers = [marker_names[i] for i in active_idxs if i < len(marker_names)]
        m_tag = ", ".join(shown_markers[:3]) + ("..." if len(shown_markers) > 3 else "")
        title_txt = f"Hotspot #{rid} ({cat})"
        if m_tag:
            title_txt = f"{title_txt} | {m_tag}"
        ax.set_title(title_txt, color="#FFFFFF", fontsize=8.5, fontweight="bold", pad=4)
        ax.set_xlim(-0.5, (pc1 - pc0) - 0.5)
        ax.set_ylim((pr1 - pr0) - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#4A5162")
        self.fig_mini_hotspot.tight_layout(pad=0.3)
        self.canvas_mini_hotspot.draw()
