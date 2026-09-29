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
from widgets.styles import SHARED_DARK_THEME
import os

class ViewerControlWidget(QWidget):
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
    viewLockToggled = Signal(bool)

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
        self._build_dashboard_group()
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
        self.combo_sb_pos.addItems(["bottom_left", "bottom_right", "top_right", "top_left"])
        self.combo_sb_pos.currentTextChanged.connect(self._change_scale_bar_position)
        self.combo_sb_pos.setCurrentText("bottom_left")
        sb_pos_layout.addWidget(self.combo_sb_pos)
        gl.addLayout(sb_pos_layout)

        # Scale Bar Size
        sb_size_layout = QHBoxLayout()
        sb_size_layout.addWidget(QLabel("Scale Bar Size:"))
        self.spin_sb_size = QSpinBox()
        self.spin_sb_size.setRange(8, 48)
        self.spin_sb_size.setValue(26)
        self.spin_sb_size.valueChanged.connect(self._change_scale_bar_size)
        sb_size_layout.addWidget(self.spin_sb_size)
        gl.addLayout(sb_size_layout)

        # Lock Camera
        self.chk_lock_camera = QCheckBox("Lock Camera Position")
        self.chk_lock_camera.toggled.connect(self.viewLockToggled.emit)
        gl.addWidget(self.chk_lock_camera)

        self.main_layout.addWidget(group)

        self.main_layout.addWidget(group)

    def _change_scale_bar_position(self, pos):
        if not hasattr(self, 'viewer') or not self.viewer: return
        try:
            self.viewer.scale_bar.position = pos
        except AttributeError:
            pass

    # def _change_scale_bar_size(self, size):
    #     if not hasattr(self, 'viewer') or not self.viewer: return
    #     try:
    #         self.viewer.scale_bar.font_size = float(size)
    #         qt_viewer = getattr(self.viewer.window, '_qt_viewer', None)
    #         if qt_viewer:
    #             target_thickness = max(2, int(size) // 4)
    #             if hasattr(qt_viewer, 'overlays') and 'scale_bar' in qt_viewer.overlays:
    #                 node = qt_viewer.overlays['scale_bar'].node
    #                 if hasattr(node, 'bar_height'): node.bar_height = target_thickness
    #                 elif hasattr(node, 'line_width'): node.line_width = target_thickness
    #     except Exception:
    #         pass

    def _change_scale_bar_size(self, size):
        """Adjust the font size of the scale bar using public API."""
        if not getattr(self, 'viewer', None): 
            return
        try:
            self.viewer.scale_bar.font_size = float(size)
        except AttributeError:
            pass


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
        # Load Marker Color
        self.btn_load_marker_color = QPushButton("Load Marker Color")
        self.btn_load_marker_color.clicked.connect(self._on_load_marker_color_clicked)
        gl.addWidget(self.btn_load_marker_color)

        self.main_layout.addWidget(group)

    # ── Dashboard ────────────────────────────────────────────────────
    def _build_dashboard_group(self):
        group = QGroupBox()
        gl = QVBoxLayout()
        gl.setContentsMargins(5, 5, 5, 5)
        gl.setSpacing(4)
        group.setLayout(gl)

        title = QLabel("Dashboard")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        gl.addWidget(title)

        self._add_separator(gl)

        self.lbl_image_dims = QLabel("Image: —")
        self.lbl_image_dims.setStyleSheet("color: #AAA; font-size: 11px;")
        gl.addWidget(self.lbl_image_dims)

        self.lbl_num_channels = QLabel("Channels: —")
        self.lbl_num_channels.setStyleSheet("color: #AAA; font-size: 11px;")
        gl.addWidget(self.lbl_num_channels)

        self.lbl_seg_status = QLabel("Segmentation: Not run")
        self.lbl_seg_status.setStyleSheet("color: #AAA; font-size: 11px;")
        gl.addWidget(self.lbl_seg_status)

        self.lbl_total_cells = QLabel("Total Cells: —")
        self.lbl_total_cells.setStyleSheet("color: #AAA; font-size: 11px;")
        gl.addWidget(self.lbl_total_cells)

        self.lbl_confidence = QLabel("Confidence: —")
        self.lbl_confidence.setStyleSheet("color: #AAA; font-size: 11px;")
        gl.addWidget(self.lbl_confidence)

        self.main_layout.addWidget(group)

    # ── Output ───────────────────────────────────────────────────────
    def _build_output_group(self):
        group = QGroupBox()
        gl = QVBoxLayout()
        gl.setContentsMargins(5, 5, 5, 5)
        gl.setSpacing(4)
        group.setLayout(gl)

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

        self.main_layout.addWidget(group)

    # ── Utility ──────────────────────────────────────────────────────
    def _add_separator(self, layout):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #555555; background-color: #555555; max-height: 1px; margin-bottom: 4px;")
        layout.addWidget(line)

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
        """Update dashboard labels with current state."""
        if image_dims is not None:
            self.lbl_image_dims.setText(f"Image: {image_dims}")
        if num_channels is not None:
            self.lbl_num_channels.setText(f"Channels: {num_channels}")
        if seg_status is not None:
            self.lbl_seg_status.setText(f"Segmentation: {seg_status}")
        if total_cells is not None:
            self.lbl_total_cells.setText(f"Total Cells: {total_cells}")
        if confidence is not None:
            self.lbl_confidence.setText(f"Confidence: {confidence}")
