"""
IRIS-Viewer Main Control Widget
Clean, natural, and user-friendly control panel for image loading, phenotyping,
canvas overlays, multi-format export, and system diagnostics.
"""
from __future__ import annotations

import os
from typing import Optional, List

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QPushButton, QGroupBox, QFrame,
    QFileDialog, QMessageBox, QSizePolicy, QScrollArea
)
from qtpy.QtCore import Qt, Signal

from iris_viewer.widgets.styles import SHARED_DARK_THEME
from iris_viewer.widgets.multi_sample import MultiSampleComparisonWidget
from iris_viewer.widgets.canvas_legend import CanvasLegendOverlay
from iris_viewer.widgets.canvas_scalebar import CanvasScaleBarOverlay


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

        self.main_layout.addStretch(1)

    # ── Image & Sample Input ──────────────────────────────────────────────
    def _build_input_group(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 3, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Image Input")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        layout.addWidget(title)

        # Directory selection button
        self.btn_load_dir = QPushButton("Select Image Directory…")
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
        self.btn_load_marker_color = QPushButton("Marker Colors…")
        self.btn_load_marker_color.setToolTip("Load custom CSV/XLSX for channel color mappings")
        self.btn_load_marker_color.clicked.connect(self._on_load_marker_color_clicked)
        row_colors.addWidget(self.btn_load_marker_color)

        self.btn_load_celltype_color = QPushButton("Cell Type Colors…")
        self.btn_load_celltype_color.setToolTip("Load custom CSV/XLSX for cell phenotype colors")
        self.btn_load_celltype_color.clicked.connect(self._on_load_celltype_color_clicked)
        row_colors.addWidget(self.btn_load_celltype_color)
        layout.addLayout(row_colors)

        self.main_layout.addWidget(box)

    # ── Annotations & Phenotyping ─────────────────────────────────────────
    def _build_annotation_group(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 3, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Annotations & Phenotyping")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        layout.addWidget(title)

        # Load annotation file
        self.btn_load_anno = QPushButton("Load Annotation File…")
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

    # ── Canvas Display & Overlays ─────────────────────────────────────────
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
        self.chk_lock_camera = QCheckBox("Lock Camera View Across Samples")
        self.chk_lock_camera.toggled.connect(self.viewLockToggled.emit)
        layout.addWidget(self.chk_lock_camera)

        self._add_divider(layout)

        # Scale Bar Controls (Clean 2-row layout so nothing is squished)
        row_sb_pos = QHBoxLayout()
        row_sb_pos.setSpacing(8)
        lbl_pos = QLabel("Scale Bar Pos:")
        lbl_pos.setFixedWidth(90)
        row_sb_pos.addWidget(lbl_pos)

        self.combo_sb_pos = QComboBox()
        self.combo_sb_pos.addItems(["bottom_left", "bottom_right", "top_left", "top_right"])
        self.combo_sb_pos.setCurrentText("bottom_left")
        row_sb_pos.addWidget(self.combo_sb_pos)
        layout.addLayout(row_sb_pos)

        row_sb_details = QHBoxLayout()
        row_sb_details.setSpacing(8)
        row_sb_details.addWidget(QLabel("Font Size:"))
        self.spin_sb_size = QSpinBox()
        self.spin_sb_size.setRange(8, 48)
        self.spin_sb_size.setValue(26)
        row_sb_details.addWidget(self.spin_sb_size)

        row_sb_details.addWidget(QLabel("Thickness:"))
        self.spin_sb_thick = QSpinBox()
        self.spin_sb_thick.setRange(1, 40)
        self.spin_sb_thick.setValue(10)
        row_sb_details.addWidget(self.spin_sb_thick)
        layout.addLayout(row_sb_details)

        # Canvas Legend
        if self.viewer:
            try:
                self.viewer.scale_bar.visible = False
            except AttributeError:
                pass

            self.custom_scale_bar = CanvasScaleBarOverlay(self.viewer)
            self.custom_scale_bar.set_position(self.combo_sb_pos.currentText())
            self.spin_scale.valueChanged.connect(self.custom_scale_bar.set_physical_scale)
            self.combo_sb_pos.currentTextChanged.connect(self.custom_scale_bar.set_position)
            self.spin_sb_size.valueChanged.connect(self.custom_scale_bar.set_font_size)
            self.spin_sb_thick.valueChanged.connect(self.custom_scale_bar.set_thickness)

            self._add_divider(layout)

            self.canvas_legend = CanvasLegendOverlay(self.viewer)

            self.chk_canvas_legend = QCheckBox("Show On-Canvas Legend")
            self.chk_canvas_legend.toggled.connect(self.canvas_legend.set_visible)
            self.chk_canvas_legend.setChecked(True)
            layout.addWidget(self.chk_canvas_legend)

            row_leg_details = QHBoxLayout()
            row_leg_details.setSpacing(8)
            row_leg_details.addWidget(QLabel("Opacity:"))
            self.spin_legend_opacity = QDoubleSpinBox()
            self.spin_legend_opacity.setRange(0.0, 1.0)
            self.spin_legend_opacity.setValue(0.1)
            self.spin_legend_opacity.setSingleStep(0.1)
            self.spin_legend_opacity.valueChanged.connect(self.canvas_legend.set_opacity)
            row_leg_details.addWidget(self.spin_legend_opacity)

            row_leg_details.addWidget(QLabel("Font Size:"))
            self.spin_legend_font = QSpinBox()
            self.spin_legend_font.setRange(8, 36)
            self.spin_legend_font.setValue(16)
            self.spin_legend_font.valueChanged.connect(self.canvas_legend.set_font_size)
            row_leg_details.addWidget(self.spin_legend_font)
            layout.addLayout(row_leg_details)

        self.main_layout.addWidget(box)

    # ── Export & Output ───────────────────────────────────────────────────
    def _build_output_group(self):
        self.output_group = QGroupBox()
        layout = QVBoxLayout(self.output_group)
        layout.setContentsMargins(10, 3, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Export")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        layout.addWidget(title)

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

        self.btn_export_mask = QPushButton("Export Mask…")
        self.btn_export_mask.setMinimumHeight(28)
        self.btn_export_mask.clicked.connect(self._on_export_mask_clicked)
        grid.addWidget(self.btn_export_mask, 0, 0)

        self.btn_export_features = QPushButton("Export Features…")
        self.btn_export_features.setMinimumHeight(28)
        self.btn_export_features.clicked.connect(self._on_export_features_clicked)
        grid.addWidget(self.btn_export_features, 0, 1)

        self.btn_export_confidence = QPushButton("Export Confidence…")
        self.btn_export_confidence.setMinimumHeight(28)
        self.btn_export_confidence.clicked.connect(self._on_export_confidence_clicked)
        grid.addWidget(self.btn_export_confidence, 1, 0)

        self.btn_export_anno = QPushButton("Export Phenotypes…")
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
        # If user picked a tabular format (xlsx/csv/tsv), fallback to tiff for mask
        if fmt in ('.xlsx', '.parquet', '.tsv', '.csv'):
            fmt = '.tiff'
        self.exportMaskTriggered[str].emit(fmt)

    def _on_export_features_clicked(self):
        fmt = self._get_current_fmt()
        # If user picked an image format (tiff/png/npy), fallback to csv for features
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

    # ── System & Compute ──────────────────────────────────────────────────
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

    # ── Utilities ────────────────────────────────────────────────────────────
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

    # ── Public API ───────────────────────────────────────────────────────────
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
