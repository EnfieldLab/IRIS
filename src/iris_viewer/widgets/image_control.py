"""
IRIS-Viewer Main Control Widget
Clean, natural, and user-friendly control panel for image loading, phenotyping,
canvas overlays, multi-format export, and system diagnostics.
"""
from __future__ import annotations

import os
from typing import List, Optional

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from iris_viewer.widgets.canvas_legend import CanvasLegendOverlay
from iris_viewer.widgets.canvas_scalebar import CanvasScaleBarOverlay
from iris_viewer.widgets.multi_sample import MultiSampleComparisonWidget
from iris_viewer.widgets.styles import SHARED_DARK_THEME


class ImageControlWidget(QWidget):
    """Dock widget for Viewer settings, image I/O, annotations, and multi-format export."""

    # Signals
    loadDirectoryTriggered = Signal(str)
    sampleSelected = Signal(str)
    scaleFactorChanged = Signal(float)
    scaleChanged = Signal(float)
    batchProcessTriggered = Signal()
    exportMaskTriggered = Signal([str], [])
    exportFeaturesTriggered = Signal([str], [])
    exportConfidenceTriggered = Signal([str], [])
    loadExistingMaskTriggered = Signal(str)
    loadChannelColorTriggered = Signal(str)
    loadCellTypeColorTriggered = Signal(str)
    loadAnnotationDataTriggered = Signal(str)
    exportUpdatedPhenotypesTriggered = Signal([str], [])
    autoSegmentTriggered = Signal()
    saveBoundsTriggered = Signal()
    viewLockToggled = Signal(bool)
    colChanged = Signal()

    def __init__(self, viewer=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.custom_scale_bar = None
        self.canvas_legend = None
        self.setStyleSheet(SHARED_DARK_THEME)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        self.content_widget = QWidget()
        self.main_layout = QVBoxLayout(self.content_widget)
        # Generous, breathable margins and spacing so nothing feels squished
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(12)

        self.scroll_area.setWidget(self.content_widget)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.scroll_area)

        # MultiSampleComparisonWidget is kept as an attribute for controller docking (TMA tab)
        self.comparison_widget = MultiSampleComparisonWidget()

        # Build clean, natural sections in logical order
        self._build_input_group()
        self._build_annotation_group()
        self._build_display_group()
        self._build_output_group()
        self._build_compute_group()

        if self.viewer:
            self._init_overlays_and_hooks()

        self.main_layout.addStretch(1)

    # Image & Sample Input
    def _build_input_group(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 3, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Image Input")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        layout.addWidget(title)

        # Directory selection button
        self.btn_load_dir = QPushButton("Select Image Directory")
        self.btn_load_dir.setMinimumHeight(28)
        self.btn_load_dir.clicked.connect(self._on_load_dir_clicked)
        layout.addWidget(self.btn_load_dir)

        self.lbl_dir_path = QLabel("No directory selected")
        self.lbl_dir_path.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        self.lbl_dir_path.setWordWrap(True)
        layout.addWidget(self.lbl_dir_path)

        # Sample selector
        row_sample = QHBoxLayout()
        row_sample.setSpacing(8)
        lbl_sample = QLabel("Sample:")
        lbl_sample.setFixedWidth(55)
        row_sample.addWidget(lbl_sample)

        self.combo_sample = QComboBox()
        self.combo_sample.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combo_sample.currentTextChanged.connect(
            lambda s: self.sampleSelected.emit(s) if s else None
        )
        row_sample.addWidget(self.combo_sample)
        layout.addLayout(row_sample)

        # Color Configuration Buttons
        row_colors = QHBoxLayout()
        row_colors.setSpacing(8)
        self.btn_load_marker_color = QPushButton("Marker Colors")
        self.btn_load_marker_color.setToolTip("Load custom CSV/XLSX for channel color mappings")
        self.btn_load_marker_color.clicked.connect(self._on_load_marker_color_clicked)
        row_colors.addWidget(self.btn_load_marker_color)

        self.btn_load_celltype_color = QPushButton("Cell Type Colors")
        self.btn_load_celltype_color.setToolTip("Load custom CSV/XLSX for cell phenotype colors")
        self.btn_load_celltype_color.clicked.connect(self._on_load_celltype_color_clicked)
        row_colors.addWidget(self.btn_load_celltype_color)
        layout.addLayout(row_colors)

        self.main_layout.addWidget(box)

    # Annotations & Phenotyping
    def _build_annotation_group(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 3, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Annotations & Phenotyping")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        layout.addWidget(title)

        # Load annotation file
        self.btn_load_anno = QPushButton("Load Annotation File")
        self.btn_load_anno.setMinimumHeight(28)
        self.btn_load_anno.clicked.connect(self._on_load_anno_clicked)
        layout.addWidget(self.btn_load_anno)

        self.lbl_annotation = QLabel("No file loaded")
        self.lbl_annotation.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
        self.lbl_annotation.setWordWrap(True)
        layout.addWidget(self.lbl_annotation)

        # Phenotype column selector
        row_col = QHBoxLayout()
        row_col.setSpacing(8)
        lbl_pheno = QLabel("Phenotype:")
        lbl_pheno.setFixedWidth(70)
        row_col.addWidget(lbl_pheno)

        self.combo_ref = QComboBox()
        self.combo_ref.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combo_ref.currentIndexChanged.connect(self.on_col_change)
        row_col.addWidget(self.combo_ref)
        layout.addLayout(row_col)

        # Quick Curation Row (Clean, natural layout)
        row_curate = QHBoxLayout()
        row_curate.setSpacing(8)
        self.chk_quick_curate = QCheckBox("Quick Curation:")
        self.chk_quick_curate.setToolTip("Hold Alt and click cells on canvas to reclassify into chosen type")
        row_curate.addWidget(self.chk_quick_curate)

        self.combo_quick_type = QComboBox()
        self.combo_quick_type.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_curate.addWidget(self.combo_quick_type)
        layout.addLayout(row_curate)

        # Tissue boundary tools
        row_boundary = QHBoxLayout()
        row_boundary.setSpacing(8)
        self.btn_auto_segment = QPushButton("Auto Boundary")
        self.btn_auto_segment.setToolTip("Automatically detect tissue envelope")
        self.btn_auto_segment.clicked.connect(lambda: self.autoSegmentTriggered.emit())
        row_boundary.addWidget(self.btn_auto_segment)

        self.btn_save_bounds = QPushButton("Save Boundary")
        self.btn_save_bounds.setToolTip("Save tissue boundary shapes")
        self.btn_save_bounds.clicked.connect(lambda: self.saveBoundsTriggered.emit())
        row_boundary.addWidget(self.btn_save_bounds)
        layout.addLayout(row_boundary)

        self.main_layout.addWidget(box)

    # Canvas Display & Overlays
    def _build_display_group(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 3, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Canvas Display & Overlays")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        layout.addWidget(title)

        # Physical Scale
        row_scale = QHBoxLayout()
        row_scale.setSpacing(8)
        lbl_scale = QLabel("Scale (µm/px):")
        lbl_scale.setFixedWidth(90)
        row_scale.addWidget(lbl_scale)

        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.001, 1000.0)
        self.spin_scale.setValue(1.0)
        self.spin_scale.setDecimals(4)
        self.spin_scale.setSingleStep(0.1)
        self.spin_scale.valueChanged.connect(self.scaleFactorChanged.emit)
        row_scale.addWidget(self.spin_scale)
        layout.addLayout(row_scale)

        # Camera Lock
        self.chk_lock_camera = QCheckBox("Lock Camera View")
        self.chk_lock_camera.toggled.connect(self.viewLockToggled.emit)
        layout.addWidget(self.chk_lock_camera)

        self._add_divider(layout)

        # Scale Bar Controls
        lbl_sb_title = QLabel("Scale Bar")
        lbl_sb_title.setStyleSheet("color: #DDD; font-size: 12px; font-weight: bold;")
        layout.addWidget(lbl_sb_title)

        self.chk_show_scalebar = QCheckBox("Show Scale Bar on Canvas")
        self.chk_show_scalebar.setChecked(True)
        layout.addWidget(self.chk_show_scalebar)

        self.chk_scalebar_in_export = QCheckBox("Include Scale Bar in Screenshot")
        self.chk_scalebar_in_export.setChecked(True)
        self.chk_scalebar_in_export.setToolTip(
            "When checked, saving screenshots or copying canvas to clipboard includes the scale bar"
        )
        layout.addWidget(self.chk_scalebar_in_export)

        row_sb_row1 = QHBoxLayout()
        row_sb_row1.setSpacing(8)
        lbl_pos = QLabel("Pos:")
        lbl_pos.setFixedWidth(30)
        row_sb_row1.addWidget(lbl_pos)

        self.combo_sb_pos = QComboBox()
        self.combo_sb_pos.addItems(["bottom_left", "bottom_right", "top_left", "top_right"])
        self.combo_sb_pos.setCurrentText("bottom_left")
        row_sb_row1.addWidget(self.combo_sb_pos)

        lbl_sb_op = QLabel("Opacity:")
        lbl_sb_op.setFixedWidth(50)
        row_sb_row1.addWidget(lbl_sb_op)

        self.spin_sb_opacity = QDoubleSpinBox()
        self.spin_sb_opacity.setRange(0.0, 1.0)
        self.spin_sb_opacity.setValue(0.0)
        self.spin_sb_opacity.setSingleStep(0.1)
        self.spin_sb_opacity.setDecimals(1)
        row_sb_row1.addWidget(self.spin_sb_opacity)
        layout.addLayout(row_sb_row1)

        row_sb_row2 = QHBoxLayout()
        row_sb_row2.setSpacing(8)
        lbl_fs = QLabel("Font Size:")
        lbl_fs.setFixedWidth(65)
        row_sb_row2.addWidget(lbl_fs)

        self.spin_sb_size = QSpinBox()
        self.spin_sb_size.setRange(8, 48)
        self.spin_sb_size.setValue(26)
        row_sb_row2.addWidget(self.spin_sb_size)

        lbl_th = QLabel("Thickness:")
        lbl_th.setFixedWidth(65)
        row_sb_row2.addWidget(lbl_th)

        self.spin_sb_thick = QSpinBox()
        self.spin_sb_thick.setRange(1, 40)
        self.spin_sb_thick.setValue(10)
        row_sb_row2.addWidget(self.spin_sb_thick)
        layout.addLayout(row_sb_row2)

        self._add_divider(layout)

        # Canvas Legend
        lbl_leg_title = QLabel("On-Canvas Legend")
        lbl_leg_title.setStyleSheet("color: #DDD; font-size: 12px; font-weight: bold;")
        layout.addWidget(lbl_leg_title)

        self.chk_canvas_legend = QCheckBox("Show On-Canvas Legend")
        self.chk_canvas_legend.setChecked(True)
        layout.addWidget(self.chk_canvas_legend)

        self.chk_legend_in_export = QCheckBox("Include Legend in Screenshot")
        self.chk_legend_in_export.setChecked(True)
        self.chk_legend_in_export.setToolTip(
            "When checked, saving screenshots or copying canvas includes the active on-canvas legend"
        )
        layout.addWidget(self.chk_legend_in_export)

        row_leg_details = QHBoxLayout()
        row_leg_details.setSpacing(8)
        row_leg_details.addWidget(QLabel("Opacity:"))
        self.spin_legend_opacity = QDoubleSpinBox()
        self.spin_legend_opacity.setRange(0.0, 1.0)
        self.spin_legend_opacity.setValue(0.0)
        self.spin_legend_opacity.setSingleStep(0.1)
        self.spin_legend_opacity.setDecimals(1)
        row_leg_details.addWidget(self.spin_legend_opacity)

        row_leg_details.addWidget(QLabel("Font Size:"))
        self.spin_legend_font = QSpinBox()
        self.spin_legend_font.setRange(8, 36)
        self.spin_legend_font.setValue(16)
        row_leg_details.addWidget(self.spin_legend_font)
        layout.addLayout(row_leg_details)

        self._add_divider(layout)

        # Viewing Window Size Controls
        lbl_view_title = QLabel("Viewing Window Size")
        lbl_view_title.setStyleSheet("color: #DDD; font-size: 12px; font-weight: bold;")
        layout.addWidget(lbl_view_title)

        row_size_preset = QHBoxLayout()
        row_size_preset.setSpacing(8)
        lbl_preset = QLabel("Preset:")
        lbl_preset.setFixedWidth(50)
        row_size_preset.addWidget(lbl_preset)

        self.combo_canvas_size = QComboBox()
        self.combo_canvas_size.addItems([
            "Auto (Fit Window)",
            "800 × 800 px (Square)",
            "1000 × 1000 px (Square)",
            "1200 × 1200 px (Square)",
            "1600 × 1600 px (Square)",
            "1200 × 900 px (4:3)",
            "1600 × 900 px (16:9)",
            "Custom Size",
        ])
        self.combo_canvas_size.currentTextChanged.connect(self._on_canvas_preset_changed)
        row_size_preset.addWidget(self.combo_canvas_size)
        layout.addLayout(row_size_preset)

        row_size_custom = QHBoxLayout()
        row_size_custom.setSpacing(6)
        row_size_custom.addWidget(QLabel("W:"))
        self.spin_canvas_w = QSpinBox()
        self.spin_canvas_w.setRange(200, 8000)
        self.spin_canvas_w.setSingleStep(50)
        self.spin_canvas_w.setValue(1000)
        row_size_custom.addWidget(self.spin_canvas_w)

        row_size_custom.addWidget(QLabel("H:"))
        self.spin_canvas_h = QSpinBox()
        self.spin_canvas_h.setRange(200, 8000)
        self.spin_canvas_h.setSingleStep(50)
        self.spin_canvas_h.setValue(1000)
        row_size_custom.addWidget(self.spin_canvas_h)
        layout.addLayout(row_size_custom)

        row_size_btns = QHBoxLayout()
        row_size_btns.setSpacing(8)
        self.btn_apply_canvas_size = QPushButton("Apply Size")
        self.btn_apply_canvas_size.setToolTip("Lock canvas display to specified dimensions for screenshot capture")
        self.btn_apply_canvas_size.clicked.connect(self._apply_canvas_size)
        row_size_btns.addWidget(self.btn_apply_canvas_size)

        self.btn_reset_canvas_size = QPushButton("Reset Auto Fit")
        self.btn_reset_canvas_size.setToolTip("Unlock canvas to dynamically fit the IRIS-Viewer window")
        self.btn_reset_canvas_size.clicked.connect(self._reset_canvas_size)
        row_size_btns.addWidget(self.btn_reset_canvas_size)
        layout.addLayout(row_size_btns)

        self.lbl_canvas_size_status = QLabel("Canvas: Auto (Fit Window)")
        self.lbl_canvas_size_status.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
        layout.addWidget(self.lbl_canvas_size_status)

        self.main_layout.addWidget(box)

    # Export & Output
    def _build_output_group(self):
        self.output_group = QGroupBox()
        layout = QVBoxLayout(self.output_group)
        layout.setContentsMargins(10, 3, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Export")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        layout.addWidget(title)

        # In-built Napari Copy & Save buttons
        self.btn_copy_canvas = QPushButton("Copy Canvas to Clipboard (Alt+C)")
        self.btn_copy_canvas.setMinimumHeight(28)
        self.btn_copy_canvas.setToolTip(
            "Copy canvas viewing area to clipboard with scale bar and active overlays"
        )
        self.btn_copy_canvas.clicked.connect(self._on_copy_canvas_clicked)
        layout.addWidget(self.btn_copy_canvas)

        row_save_btns = QHBoxLayout()
        row_save_btns.setSpacing(8)
        self.btn_save_canvas = QPushButton("Save Canvas (Alt+S)")
        self.btn_save_canvas.setMinimumHeight(28)
        self.btn_save_canvas.setToolTip("Save canvas display with scale bar to image file")
        self.btn_save_canvas.clicked.connect(self._on_save_canvas_clicked)
        row_save_btns.addWidget(self.btn_save_canvas)

        self.btn_save_viewer = QPushButton("Save Full Viewer")
        self.btn_save_viewer.setMinimumHeight(28)
        self.btn_save_viewer.setToolTip("Save entire viewer window including dock widgets")
        self.btn_save_viewer.clicked.connect(self._on_save_viewer_clicked)
        row_save_btns.addWidget(self.btn_save_viewer)
        layout.addLayout(row_save_btns)

        self._add_divider(layout)

        # Format selector row: simple, intuitive, natural
        row_fmt = QHBoxLayout()
        row_fmt.setSpacing(8)
        lbl_fmt = QLabel("Preferred Format:")
        lbl_fmt.setFixedWidth(110)
        row_fmt.addWidget(lbl_fmt)

        self.combo_export_format = QComboBox()
        self.combo_export_format.addItem("Default (TIFF / CSV)", "")
        self.combo_export_format.addItem("OME-TIFF (*.ome.tif)", ".ome.tif")
        self.combo_export_format.addItem("PNG Image (*.png)", ".png")
        self.combo_export_format.addItem("NumPy Array (*.npy)", ".npy")
        self.combo_export_format.addItem("Excel Workbook (*.xlsx)", ".xlsx")
        self.combo_export_format.addItem("Apache Parquet (*.parquet)", ".parquet")
        self.combo_export_format.addItem("Tab-Separated (*.tsv)", ".tsv")
        row_fmt.addWidget(self.combo_export_format)
        layout.addLayout(row_fmt)

        # 4 Direct, clear action buttons in a 2x2 grid
        grid = QGridLayout()
        grid.setSpacing(8)

        self.btn_export_mask = QPushButton("Export Mask")
        self.btn_export_mask.setMinimumHeight(28)
        self.btn_export_mask.clicked.connect(self._on_export_mask_clicked)
        grid.addWidget(self.btn_export_mask, 0, 0)

        self.btn_export_features = QPushButton("Export Features")
        self.btn_export_features.setMinimumHeight(28)
        self.btn_export_features.clicked.connect(self._on_export_features_clicked)
        grid.addWidget(self.btn_export_features, 0, 1)

        self.btn_export_confidence = QPushButton("Export Confidence")
        self.btn_export_confidence.setMinimumHeight(28)
        self.btn_export_confidence.clicked.connect(self._on_export_confidence_clicked)
        grid.addWidget(self.btn_export_confidence, 1, 0)

        self.btn_export_anno = QPushButton("Export Phenotypes")
        self.btn_export_anno.setMinimumHeight(28)
        self.btn_export_anno.clicked.connect(self._on_export_phenotypes_clicked)
        grid.addWidget(self.btn_export_anno, 1, 1)

        layout.addLayout(grid)

        self._add_divider(layout)

        # Batch Processing
        self.btn_batch = QPushButton("Batch Process All Samples")
        self.btn_batch.setMinimumHeight(32)
        self.btn_batch.setStyleSheet("""
            QPushButton {
                background-color: #2D6A4F; color: white; font-weight: bold;
                border-radius: 4px; padding: 6px; font-size: 12px;
            }
            QPushButton:hover { background-color: #40916C; }
            QPushButton:disabled { background-color: #2A3A4A; color: #777; }
        """)
        self.btn_batch.clicked.connect(lambda: self.batchProcessTriggered.emit())
        layout.addWidget(self.btn_batch)

        self.main_layout.addWidget(self.output_group)

    def _get_current_fmt(self) -> str:
        return self.combo_export_format.currentData() or ""

    def _on_export_mask_clicked(self):
        fmt = self._get_current_fmt()
        if fmt in ('.xlsx', '.parquet', '.tsv', '.csv'):
            fmt = '.tiff'
        self.exportMaskTriggered[str].emit(fmt)

    def _on_export_features_clicked(self):
        fmt = self._get_current_fmt()
        if fmt in ('.tiff', '.ome.tif', '.png'):
            fmt = '.csv'
        self.exportFeaturesTriggered[str].emit(fmt)

    def _on_export_confidence_clicked(self):
        fmt = self._get_current_fmt()
        if fmt in ('.xlsx', '.parquet', '.tsv', '.csv'):
            fmt = '.tiff'
        self.exportConfidenceTriggered[str].emit(fmt)

    def _on_export_phenotypes_clicked(self):
        fmt = self._get_current_fmt()
        if fmt in ('.tiff', '.ome.tif', '.png'):
            fmt = '.csv'
        self.exportUpdatedPhenotypesTriggered[str].emit(fmt)

    def _init_overlays_and_hooks(self) -> None:
        """Initialize canvas overlays and hook napari screenshot methods."""
        if not self.viewer:
            return
        try:
            if hasattr(self.viewer, 'canvas') and hasattr(self.viewer.canvas, 'overlays') and hasattr(self.viewer.canvas.overlays, 'scale_bar'):
                self.viewer.canvas.overlays.scale_bar.visible = False
            elif hasattr(self.viewer, 'scale_bar'):
                self.viewer.scale_bar.visible = False
        except Exception:
            pass

        if not hasattr(self, 'custom_scale_bar') or self.custom_scale_bar is None:
            self.custom_scale_bar = CanvasScaleBarOverlay(self.viewer)
            self.custom_scale_bar.set_position(self.combo_sb_pos.currentText())
            self.custom_scale_bar.set_font_size(self.spin_sb_size.value())
            self.custom_scale_bar.set_thickness(self.spin_sb_thick.value())
            self.custom_scale_bar.set_opacity(self.spin_sb_opacity.value())
            self.custom_scale_bar.set_physical_scale(self.spin_scale.value())
            self.custom_scale_bar.setVisible(self.chk_show_scalebar.isChecked())
            self.spin_scale.valueChanged.connect(self.custom_scale_bar.set_physical_scale)
            self.combo_sb_pos.currentTextChanged.connect(self.custom_scale_bar.set_position)
            self.spin_sb_size.valueChanged.connect(self.custom_scale_bar.set_font_size)
            self.spin_sb_thick.valueChanged.connect(self.custom_scale_bar.set_thickness)
            self.spin_sb_opacity.valueChanged.connect(self.custom_scale_bar.set_opacity)
            self.chk_show_scalebar.toggled.connect(self.custom_scale_bar.setVisible)

        if not hasattr(self, 'canvas_legend') or self.canvas_legend is None:
            self.canvas_legend = CanvasLegendOverlay(self.viewer)
            self.chk_canvas_legend.toggled.connect(self.canvas_legend.set_visible)
            self.spin_legend_opacity.valueChanged.connect(self.canvas_legend.set_opacity)
            self.spin_legend_font.valueChanged.connect(self.canvas_legend.set_font_size)
            self.canvas_legend.set_visible(self.chk_canvas_legend.isChecked())

        self._install_screenshot_overlay_hook()

    def set_viewer(self, viewer) -> None:
        """Assign or update the napari Viewer instance and bind overlays."""
        self.viewer = viewer
        self._init_overlays_and_hooks()

    def _install_screenshot_overlay_hook(self) -> None:
        """Hook napari's QtViewer._screenshot to composite QuPath-style overlays."""
        if not self.viewer:
            return
        qt_viewer = getattr(self.viewer.window, '_qt_viewer', None)
        if qt_viewer is None or getattr(qt_viewer, '_iris_screenshot_hooked', False):
            return

        orig_screenshot = qt_viewer._screenshot

        def _screenshot_with_overlays(
            flash: bool = True,
            size: tuple[int, int] | None = None,
            scale: float = 1.0,
            fit_to_data_extent: bool = False,
        ):
            raw_qimg = orig_screenshot(
                flash=flash,
                size=size,
                scale=scale,
                fit_to_data_extent=fit_to_data_extent,
            )
            include_sb = getattr(self, 'chk_scalebar_in_export', None) is None or self.chk_scalebar_in_export.isChecked()
            sb_enabled = getattr(self, 'chk_show_scalebar', None) is None or self.chk_show_scalebar.isChecked()
            has_sb = include_sb and sb_enabled and hasattr(self, 'custom_scale_bar') and self.custom_scale_bar and (not self.custom_scale_bar.isHidden())

            include_leg = getattr(self, 'chk_legend_in_export', None) is None or self.chk_legend_in_export.isChecked()
            leg_enabled = getattr(self, 'chk_canvas_legend', None) is None or self.chk_canvas_legend.isChecked()
            has_leg = include_leg and leg_enabled and hasattr(self, 'canvas_legend') and self.canvas_legend and (not self.canvas_legend.isHidden())

            if not has_sb and not has_leg:
                return raw_qimg

            try:
                import numpy as np
                import PIL.Image
                from napari._qt.qt_viewer import QImg2array
                from qtpy.QtGui import QImage

                raw_arr = QImg2array(raw_qimg).copy()
                canvas_pil = PIL.Image.fromarray(raw_arr)
                canvas_native = qt_viewer.canvas.native

                sx = raw_qimg.width() / max(1, canvas_native.width())
                sy = raw_qimg.height() / max(1, canvas_native.height())

                overlay_layer = PIL.Image.new("RGBA", canvas_pil.size, (0, 0, 0, 0))

                if has_leg:
                    leg_pix = self.canvas_legend.grab().toImage()
                    leg_arr = QImg2array(leg_pix).copy()
                    leg_pil = PIL.Image.fromarray(leg_arr)
                    if scale != 1.0:
                        leg_pil = leg_pil.resize(
                            (max(1, int(leg_pil.width * scale)), max(1, int(leg_pil.height * scale))),
                            PIL.Image.Resampling.LANCZOS,
                        )
                    l_pos = self.canvas_legend.pos()
                    overlay_layer.paste(leg_pil, (int(l_pos.x() * sx), int(l_pos.y() * sy)))

                if has_sb:
                    sb_pix = self.custom_scale_bar.grab().toImage()
                    sb_arr = QImg2array(sb_pix).copy()
                    sb_pil = PIL.Image.fromarray(sb_arr)
                    if scale != 1.0:
                        sb_pil = sb_pil.resize(
                            (max(1, int(sb_pil.width * scale)), max(1, int(sb_pil.height * scale))),
                            PIL.Image.Resampling.LANCZOS,
                        )
                    s_pos = self.custom_scale_bar.pos()
                    overlay_layer.paste(sb_pil, (int(s_pos.x() * sx), int(s_pos.y() * sy)))

                canvas_pil = PIL.Image.alpha_composite(canvas_pil, overlay_layer)
                final_arr = np.ascontiguousarray(np.array(canvas_pil))
                h, w, _ = final_arr.shape
                return QImage(final_arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
            except Exception:
                return raw_qimg

        qt_viewer._screenshot = _screenshot_with_overlays
        qt_viewer._iris_screenshot_hooked = True

    def _on_canvas_preset_changed(self, text: str) -> None:
        """Handle canvas size preset selection."""
        preset_dims = {
            "800 × 800 px (Square)": (800, 800),
            "1000 × 1000 px (Square)": (1000, 1000),
            "1200 × 1200 px (Square)": (1200, 1200),
            "1600 × 1600 px (Square)": (1600, 1600),
            "1200 × 900 px (4:3)": (1200, 900),
            "1600 × 900 px (16:9)": (1600, 900),
        }
        if text == "Auto (Fit Window)":
            self._reset_canvas_size()
        elif text in preset_dims:
            w, h = preset_dims[text]
            self.spin_canvas_w.blockSignals(True)
            self.spin_canvas_h.blockSignals(True)
            self.spin_canvas_w.setValue(w)
            self.spin_canvas_h.setValue(h)
            self.spin_canvas_w.blockSignals(False)
            self.spin_canvas_h.blockSignals(False)
            self._set_canvas_fixed_size(w, h, label_suffix="")

    def _apply_canvas_size(self) -> None:
        """Apply custom width and height to canvas viewing window."""
        w = self.spin_canvas_w.value()
        h = self.spin_canvas_h.value()
        matched_idx = -1
        for i in range(self.combo_canvas_size.count()):
            t = self.combo_canvas_size.itemText(i)
            if f"{w} × {h}" in t:
                matched_idx = i
                break
        self.combo_canvas_size.blockSignals(True)
        if matched_idx != -1:
            self.combo_canvas_size.setCurrentIndex(matched_idx)
        else:
            self.combo_canvas_size.setCurrentText("Custom Size")
        self.combo_canvas_size.blockSignals(False)
        self._set_canvas_fixed_size(w, h, label_suffix=" (Custom)")

    def _set_canvas_fixed_size(self, w: int, h: int, label_suffix: str = "") -> None:
        """Lock the canvas native widget to exact pixel dimensions."""
        if not self.viewer:
            return
        qt_viewer = getattr(self.viewer.window, '_qt_viewer', None)
        if not qt_viewer or not hasattr(qt_viewer, 'canvas'):
            return
        canvas_native = qt_viewer.canvas.native
        canvas_native.setFixedSize(w, h)
        if hasattr(self, 'custom_scale_bar') and self.custom_scale_bar:
            self.custom_scale_bar._reposition()
        if hasattr(self, 'canvas_legend') and self.canvas_legend:
            self.canvas_legend._reposition()
        self.lbl_canvas_size_status.setText(f"Canvas: {w} × {h} px{label_suffix}")
        self.lbl_canvas_size_status.setStyleSheet("color: #4CC9F0; font-size: 11px; font-weight: bold;")

    def _reset_canvas_size(self) -> None:
        """Unlock canvas dimensions and allow it to fit the viewer window automatically."""
        self.combo_canvas_size.blockSignals(True)
        self.combo_canvas_size.setCurrentText("Auto (Fit Window)")
        self.combo_canvas_size.blockSignals(False)
        if not self.viewer:
            return
        qt_viewer = getattr(self.viewer.window, '_qt_viewer', None)
        if not qt_viewer or not hasattr(qt_viewer, 'canvas'):
            return
        canvas_native = qt_viewer.canvas.native
        canvas_native.setMinimumSize(0, 0)
        canvas_native.setMaximumSize(16777215, 16777215)
        canvas_native.setSizePolicy(qt_viewer.sizePolicy())
        parent_widget = canvas_native.parentWidget()
        if parent_widget and parent_widget.layout():
            parent_widget.layout().invalidate()
            parent_widget.layout().activate()
        if hasattr(self, 'custom_scale_bar') and self.custom_scale_bar:
            self.custom_scale_bar._reposition()
        if hasattr(self, 'canvas_legend') and self.canvas_legend:
            self.canvas_legend._reposition()
        self.lbl_canvas_size_status.setText("Canvas: Auto (Fit Window)")
        self.lbl_canvas_size_status.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")

    def _on_copy_canvas_clicked(self) -> None:
        """Copy currently visible canvas with scale bar to system clipboard."""
        if self.viewer:
            self.viewer.window.clipboard(canvas_only=True)

    def _on_save_canvas_clicked(self) -> None:
        """Open screenshot dialog to save canvas view with scale bar to file."""
        if not self.viewer:
            return
        qt_viewer = getattr(self.viewer.window, '_qt_viewer', None)
        if qt_viewer and hasattr(qt_viewer, '_screenshot_dialog'):
            qt_viewer._screenshot_dialog()
        else:
            self.viewer.window._screenshot_dialog()

    def _on_save_viewer_clicked(self) -> None:
        """Open screenshot dialog to save entire viewer window."""
        if self.viewer:
            self.viewer.window._screenshot_dialog()

    # System & Compute
    def _build_compute_group(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 3, 10, 10)
        layout.setSpacing(4)

        title = QLabel("System & Compute")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        layout.addWidget(title)

        try:
            import torch
            HAS_TORCH = True
        except ImportError:
            HAS_TORCH = False

        cpu_cores = os.cpu_count() or "Unknown"
        accel_text = "CPU Only"
        if HAS_TORCH:
            if torch.cuda.is_available():
                accel_text = f"CUDA ({torch.cuda.get_device_name(0)})"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                accel_text = "Apple Silicon (MPS)"

        lbl_cpu = QLabel(f"CPU: {cpu_cores} Cores Available")
        lbl_cpu.setStyleSheet("color: #BBB; font-size: 11px;")
        layout.addWidget(lbl_cpu)

        lbl_gpu = QLabel(f"Accelerator: {accel_text}")
        lbl_gpu.setStyleSheet("color: #BBB; font-size: 11px;")
        layout.addWidget(lbl_gpu)

        self.main_layout.addWidget(box)

    # Utilities
    def _add_divider(self, layout):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #383D48; background-color: #383D48; max-height: 1px; margin: 4px 0;")
        layout.addWidget(line)

    def on_col_change(self):
        """Signal handler for when the cell phenotype reference column changes."""
        self.colChanged.emit()

    def _on_load_dir_clicked(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dir_path:
            self.lbl_dir_path.setText(dir_path)
            self.lbl_dir_path.setStyleSheet("color: #BBB; font-size: 11px;")
            self.loadDirectoryTriggered.emit(dir_path)

    def _on_load_marker_color_clicked(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Marker Color", "", "Data Tables (*.csv *.xlsx)")
        if fname:
            self.loadChannelColorTriggered.emit(fname)

    def _on_load_celltype_color_clicked(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Cell Type Color", "", "Data Tables (*.csv *.xlsx)")
        if fname:
            self.loadCellTypeColorTriggered.emit(fname)

    def _on_load_anno_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Annotation File", "", "Table files (*.csv *.xlsx *.h5ad);;All files (*)"
        )
        if path:
            self.lbl_annotation.setText(os.path.basename(path))
            self.loadAnnotationDataTriggered.emit(path)

    # Public API
    def update_sample_list(self, sample_names: List[str]):
        """Populate the sample selector dropdown."""
        self.combo_sample.blockSignals(True)
        self.combo_sample.clear()
        self.combo_sample.addItems(sample_names)
        self.combo_sample.blockSignals(False)

    def update_dashboard(
        self,
        image_dims: Optional[str] = None,
        num_channels: Optional[str] = None,
        seg_status: Optional[str] = None,
        total_cells: Optional[str] = None,
        confidence: Optional[str] = None
    ):
        """Update dashboard labels with current state. (Deprecated)"""
        pass
