"""
IRIS-Viewer Main Control Widget
Settings, compute engine configuration, image loading, and dashboard.
Modeled after IRIS/widgets/control.py but scoped exclusively to Viewer functions.
"""
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QPushButton, QGroupBox, QFrame,
    QFileDialog, QMessageBox, QSizePolicy, QScrollArea
)
from qtpy.QtCore import Qt, Signal
from iris_common.widgets.styles import SHARED_DARK_THEME
from iris_common.widgets.multi_sample import MultiSampleComparisonWidget
from iris_common.widgets.canvas_legend import CanvasLegendOverlay
from iris_common.widgets.canvas_scalebar import CanvasScaleBarOverlay
import os

class ImageControlWidget(QWidget):
    """Dock widget for global Viewer settings, compute config, and image I/O."""

    # Signals
    loadDirectoryTriggered = Signal(str)
    sampleSelected = Signal(str)
    scaleFactorChanged = Signal(float)
    scaleChanged = Signal(float)
    batchProcessTriggered = Signal()
    exportMaskTriggered = Signal()
    exportFeaturesTriggered = Signal()
    exportConfidenceTriggered = Signal()
    loadExistingMaskTriggered = Signal(str)
    loadChannelColorTriggered = Signal(str)
    loadCellTypeColorTriggered = Signal(str)
    loadAnnotationDataTriggered = Signal(str)
    exportUpdatedPhenotypesTriggered = Signal()
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
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(2, 2, 2, 2)
        self.main_layout.setSpacing(4)
        self.content_widget.setLayout(self.main_layout)
        
        self.scroll_area.setWidget(self.content_widget)
        
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.scroll_area)
        self.setLayout(outer_layout)

        self._build_settings_group()
        self._build_compute_group()
        self._build_input_group()
        self._build_annotation_group()
        
        # Replace dashboard with MultiSampleComparisonWidget
        self.comparison_widget = MultiSampleComparisonWidget()
        self.main_layout.addWidget(self.comparison_widget)

        self._build_output_group()
        self.main_layout.addStretch()

    # ── Settings ─────────────────────────────────────────────────────
    def _build_settings_group(self):
        group = QGroupBox()
        gl = QVBoxLayout()
        gl.setContentsMargins(4, 4, 4, 4)
        gl.setSpacing(4)
        group.setLayout(gl)

        title = QLabel("Settings")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        gl.addWidget(title)

        self._add_separator(gl)

        # Scale
        row_scale = QHBoxLayout()
        row_scale.addWidget(QLabel("Physical Scale (µm/px):"))
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.01, 100.0)
        self.spin_scale.setValue(1.0)
        self.spin_scale.setDecimals(4)
        self.spin_scale.setSingleStep(0.1)
        self.spin_scale.valueChanged.connect(self.scaleFactorChanged.emit)
        row_scale.addWidget(self.spin_scale)
        gl.addLayout(row_scale)

        # Scale Bar Position
        sb_pos_layout = QHBoxLayout()
        sb_pos_layout.addWidget(QLabel("Scale Bar Position:"))
        self.combo_sb_pos = QComboBox()
        self.combo_sb_pos.addItems(["bottom_right", "bottom_left", "top_right", "top_left"])
        self.combo_sb_pos.setCurrentText("bottom_right")
        sb_pos_layout.addWidget(self.combo_sb_pos)
        gl.addLayout(sb_pos_layout)

        # Scale Bar Size
        sb_size_layout = QHBoxLayout()
        sb_size_layout.addWidget(QLabel("Scale Bar Font Size:"))
        self.spin_sb_size = QSpinBox()
        self.spin_sb_size.setRange(8, 48)
        self.spin_sb_size.setValue(26)
        sb_size_layout.addWidget(self.spin_sb_size)
        gl.addLayout(sb_size_layout)

        # Scale Bar Thickness
        sb_thick_layout = QHBoxLayout()
        sb_thick_layout.addWidget(QLabel("Scale Bar Thickness:"))
        self.spin_sb_thick = QSpinBox()
        self.spin_sb_thick.setRange(1, 40)
        self.spin_sb_thick.setValue(10)
        sb_thick_layout.addWidget(self.spin_sb_thick)
        gl.addLayout(sb_thick_layout)

        # Lock Camera
        self.chk_lock_camera = QCheckBox("Lock Camera Position")
        self.chk_lock_camera.toggled.connect(self.viewLockToggled.emit)
        gl.addWidget(self.chk_lock_camera)

        if self.viewer:
            try:
                self.viewer.scale_bar.visible = False
            except AttributeError:
                pass
                
            self.custom_scale_bar = CanvasScaleBarOverlay(self.viewer)
            self.spin_scale.valueChanged.connect(self.custom_scale_bar.set_physical_scale)
            self.combo_sb_pos.currentTextChanged.connect(self.custom_scale_bar.set_position)
            self.spin_sb_size.valueChanged.connect(self.custom_scale_bar.set_font_size)
            self.spin_sb_thick.valueChanged.connect(self.custom_scale_bar.set_thickness)
            
            # Canvas Legend Overlay
            self.canvas_legend = CanvasLegendOverlay(self.viewer)
            
            # Canvas Legend Toggle
            self.chk_canvas_legend = QCheckBox("Show On-Canvas Legend")
            self.chk_canvas_legend.toggled.connect(self.canvas_legend.set_visible)
            self.chk_canvas_legend.setChecked(True)
            gl.addWidget(self.chk_canvas_legend)
            
            # Canvas Legend Opacity
            row_opacity = QHBoxLayout()
            row_opacity.addWidget(QLabel("Legend Opacity:"))
            self.spin_legend_opacity = QDoubleSpinBox()
            self.spin_legend_opacity.setRange(0.1, 1.0)
            self.spin_legend_opacity.setValue(0.1)
            self.spin_legend_opacity.setSingleStep(0.1)
            self.spin_legend_opacity.valueChanged.connect(self.canvas_legend.set_opacity)
            row_opacity.addWidget(self.spin_legend_opacity)
            gl.addLayout(row_opacity)
            
            # Canvas Legend Font Size
            row_font = QHBoxLayout()
            row_font.addWidget(QLabel("Legend Font Size:"))
            self.spin_legend_font = QSpinBox()
            self.spin_legend_font.setRange(8, 36)
            self.spin_legend_font.setValue(16)
            self.spin_legend_font.valueChanged.connect(self.canvas_legend.set_font_size)
            row_font.addWidget(self.spin_legend_font)
            gl.addLayout(row_font)

        self.main_layout.addWidget(group)




    def _build_compute_group(self):
        group = QGroupBox()
        cl = QVBoxLayout()
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(4)
        group.setLayout(cl)

        lbl_c_title = QLabel("Compute Engine")
        lbl_c_title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        cl.addWidget(lbl_c_title)

        self._add_separator(cl)

        try:
            import torch
            HAS_TORCH = True
        except ImportError:
            HAS_TORCH = False

        cpu_cores = os.cpu_count() or "Unknown"
        gpu_stat = "Not Detected (CPU Only)"
        if HAS_TORCH:
            if torch.cuda.is_available(): gpu_stat = f"CUDA Enabled ({torch.cuda.get_device_name(0)})"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): gpu_stat = "Apple Silicon MPS Enabled"

        lbl_cpu = QLabel(f"<b>CPU Cores:</b> {cpu_cores} Available")
        lbl_cpu.setStyleSheet("color: #AACCFF;")
        cl.addWidget(lbl_cpu)

        lbl_gpu = QLabel(f"<b>Hardware Accelerator:</b> {gpu_stat}")
        lbl_gpu.setWordWrap(True)
        lbl_gpu.setStyleSheet("color: #AAFFAA;" if "Enabled" in gpu_stat else "color: #FFAAAA;")
        cl.addWidget(lbl_gpu)



        self.main_layout.addWidget(group)

    # ── Input ────────────────────────────────────────────────────────
    def _build_input_group(self):
        group = QGroupBox()
        gl = QVBoxLayout()
        gl.setContentsMargins(5, 5, 5, 5)
        gl.setSpacing(4)
        group.setLayout(gl)

        title = QLabel("Image Input")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        gl.addWidget(title)

        self._add_separator(gl)

        self.btn_load_dir = QPushButton("Select Image Directory")
        self.btn_load_dir.clicked.connect(self._on_load_dir_clicked)
        gl.addWidget(self.btn_load_dir)

        self.lbl_dir_path = QLabel("No directory selected")
        self.lbl_dir_path.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        self.lbl_dir_path.setWordWrap(True)
        gl.addWidget(self.lbl_dir_path)

        # Sample selector
        row_sample = QHBoxLayout()
        row_sample.addWidget(QLabel("Sample:"))
        self.combo_sample = QComboBox()
        self.combo_sample.currentTextChanged.connect(
            lambda s: self.sampleSelected.emit(s) if s else None
        )
        row_sample.addWidget(self.combo_sample)
        gl.addLayout(row_sample)

        # Load existing mask
        self.btn_load_mask = QPushButton("Load Existing Segmentation Mask")
        self.btn_load_mask.clicked.connect(self._on_load_mask_clicked)
        gl.addWidget(self.btn_load_mask)
        # Load Marker Color and Cell Type Color
        color_row = QHBoxLayout()
        self.btn_load_marker_color = QPushButton("Load Marker Color")
        self.btn_load_marker_color.clicked.connect(self._on_load_marker_color_clicked)
        color_row.addWidget(self.btn_load_marker_color)
        
        self.btn_load_celltype_color = QPushButton("Load Cell Type Color")
        self.btn_load_celltype_color.clicked.connect(self._on_load_celltype_color_clicked)
        color_row.addWidget(self.btn_load_celltype_color)
        
        gl.addLayout(color_row)

        self.main_layout.addWidget(group)

    def _build_annotation_group(self):
        box = QGroupBox()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)
        
        title = QLabel("Annotation Data")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        self.lbl_annotation = QLabel("No file loaded")
        self.lbl_annotation.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.lbl_annotation)

        self.btn_load_anno = QPushButton("Load Annotation File…")
        self.btn_load_anno.clicked.connect(self._on_load_anno_clicked)
        layout.addWidget(self.btn_load_anno)
        
        # Cell Phenotype Ref Col
        r_row = QHBoxLayout()
        r_row.addWidget(QLabel("Cell Phenotype:"))
        self.combo_ref = QComboBox()
        self.combo_ref.currentIndexChanged.connect(self.on_col_change)
        r_row.addWidget(self.combo_ref)
        layout.addLayout(r_row)

        # Quick Curation
        q_row = QHBoxLayout()
        self.chk_quick_curate = QCheckBox("Quick Curation")
        self.chk_quick_curate.setStyleSheet("color: #FFaa00; font-weight: bold;")
        self.combo_quick_type = QComboBox()
        q_row.addWidget(self.chk_quick_curate)
        q_row.addWidget(self.combo_quick_type)
        layout.addLayout(q_row)
        
        self.btn_export_anno = QPushButton("Export Updated Phenotypes")
        self.btn_export_anno.clicked.connect(lambda: self.exportUpdatedPhenotypesTriggered.emit())
        layout.addWidget(self.btn_export_anno)
        
        self.main_layout.addWidget(box)

    def on_col_change(self):
        """Signal handler for when the cell phenotype reference column changes."""
        self.colChanged.emit()

    def _on_load_anno_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Annotation File", "", "Table files (*.csv *.xlsx *.h5ad);;All files (*)"
        )
        if path:
            self.lbl_annotation.setText(path.split("/")[-1])
            self.loadAnnotationDataTriggered.emit(path)



    # ── Output ───────────────────────────────────────────────────────
    def _build_output_group(self):
        self.output_group = QGroupBox()
        gl = QVBoxLayout()
        gl.setContentsMargins(5, 5, 5, 5)
        gl.setSpacing(4)
        self.output_group.setLayout(gl)

        title = QLabel("Export")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        gl.addWidget(title)

        self._add_separator(gl)

        self.btn_export_mask = QPushButton("Export Consensus Mask (.tiff)")
        self.btn_export_mask.clicked.connect(lambda: self.exportMaskTriggered.emit())
        gl.addWidget(self.btn_export_mask)

        self.btn_export_features = QPushButton("Export Features (.csv)")
        self.btn_export_features.clicked.connect(lambda: self.exportFeaturesTriggered.emit())
        gl.addWidget(self.btn_export_features)

        self.btn_export_confidence = QPushButton("Export Confidence Map (.tiff)")
        self.btn_export_confidence.clicked.connect(lambda: self.exportConfidenceTriggered.emit())
        gl.addWidget(self.btn_export_confidence)

        self._add_separator(gl)

        self.btn_batch = QPushButton("Batch Process All Samples")
        self.btn_batch.clicked.connect(lambda: self.batchProcessTriggered.emit())
        gl.addWidget(self.btn_batch)

        self.main_layout.addWidget(self.output_group)

    # ── Utility ──────────────────────────────────────────────────────
    def _add_separator(self, layout):
        pass

    def _detect_device(self):
        pass

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

    def _on_load_mask_clicked(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Segmentation Mask", "", "TIFF Images (*.tiff *.tif)"
        )
        if filepath:
            self.loadExistingMaskTriggered.emit(filepath)

    # ── Public API ───────────────────────────────────────────────────
    def update_sample_list(self, sample_names: list):
        """Populate the sample selector dropdown."""
        self.combo_sample.blockSignals(True)
        self.combo_sample.clear()
        self.combo_sample.addItems(sample_names)
        self.combo_sample.blockSignals(False)

    def update_dashboard(
        self, 
        image_dims: str = None, 
        num_channels: str = None,
        seg_status: str = None, 
        total_cells: str = None, 
        confidence: str = None
    ):
        """Update dashboard labels with current state. (Deprecated)"""
        pass
