""" IRIS-Viewer Segmentation Panel Widget """
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QPushButton, QGroupBox, QFrame,
    QSizePolicy, QScrollArea, QLineEdit
)
from qtpy.QtCore import Qt, Signal
from iris_viewer.widgets.styles import SHARED_DARK_THEME, STATUS_STYLES

class SegmentationPanelWidget(QWidget):
    
    runSegmentation = Signal(list, dict)
    cancelSegmentation = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(SHARED_DARK_THEME)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        
        self.content_widget = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(6)
        self.content_widget.setLayout(self.main_layout)
        self.scroll_area.setWidget(self.content_widget)
        
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.scroll_area)
        self.setLayout(outer_layout)

        # UI Building (Removed global Channel Assignment & Global Cyto Expansion)
        self._build_segmentation_modules()
        self._build_run_controls()
        self._build_status_panel()
        self.main_layout.addStretch()

    def _build_segmentation_modules(self):
        group_main = QGroupBox()
        gl_main = QVBoxLayout()
        gl_main.setContentsMargins(5, 5, 5, 5)
        gl_main.setSpacing(4)
        group_main.setLayout(gl_main)

        title = QLabel("Segmentation Modules")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        gl_main.addWidget(title)
        self._add_separator(gl_main)

        info = QLabel("Check the box for each method you wish to run. Configure channels per method.")
        info.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        info.setWordWrap(True)
        gl_main.addWidget(info)

        self.method_checkboxes = {}

        # -- Cellpose --
        self.group_cp = QGroupBox("Cellpose")
        self.group_cp.setCheckable(True)
        self.group_cp.setChecked(True)
        self.method_checkboxes["Cellpose"] = self.group_cp
        cp_layout = QVBoxLayout()
        
        row_cp_c = QHBoxLayout()
        self.combo_cp_nuc = QComboBox()
        self.combo_cp_mem = QComboBox()
        row_cp_c.addWidget(QLabel("Nuc Channel:"))
        row_cp_c.addWidget(self.combo_cp_nuc)
        row_cp_c.addWidget(QLabel("Mem Channel:"))
        row_cp_c.addWidget(self.combo_cp_mem)
        cp_layout.addLayout(row_cp_c)
        
        row_cp1 = QHBoxLayout()
        self.combo_cp_model = QComboBox()
        self.combo_cp_model.addItems(["cyto3", "cyto2", "cyto", "nuclei"])
        row_cp1.addWidget(QLabel("Model:"))
        row_cp1.addWidget(self.combo_cp_model)
        self.spin_cp_diameter = QSpinBox()
        self.spin_cp_diameter.setRange(0, 500)
        self.spin_cp_diameter.setValue(0)
        self.spin_cp_diameter.setSpecialValueText("Auto")
        row_cp1.addWidget(QLabel("Diameter:"))
        row_cp1.addWidget(self.spin_cp_diameter)
        cp_layout.addLayout(row_cp1)

        row_cp2 = QHBoxLayout()
        self.spin_cp_flow = QDoubleSpinBox()
        self.spin_cp_flow.setRange(0.0, 5.0)
        self.spin_cp_flow.setValue(0.4)
        self.spin_cp_flow.setSingleStep(0.1)
        row_cp2.addWidget(QLabel("Flow Thresh:"))
        row_cp2.addWidget(self.spin_cp_flow)
        self.spin_cp_cellprob = QDoubleSpinBox()
        self.spin_cp_cellprob.setRange(-10.0, 10.0)
        self.spin_cp_cellprob.setValue(0.0)
        row_cp2.addWidget(QLabel("Cell Prob:"))
        row_cp2.addWidget(self.spin_cp_cellprob)
        cp_layout.addLayout(row_cp2)
        
        self.group_cp.setLayout(cp_layout)
        gl_main.addWidget(self.group_cp)

        # -- StarDist --
        self.group_sd = QGroupBox("StarDist")
        self.group_sd.setCheckable(True)
        self.group_sd.setChecked(False)
        self.method_checkboxes["StarDist"] = self.group_sd
        sd_layout = QVBoxLayout()
        
        row_sd_c = QHBoxLayout()
        self.combo_sd_nuc = QComboBox()
        row_sd_c.addWidget(QLabel("Nuc Channel:"))
        row_sd_c.addWidget(self.combo_sd_nuc)
        row_sd_c.addStretch()
        sd_layout.addLayout(row_sd_c)
        
        row_sd = QHBoxLayout()
        self.combo_sd_model = QComboBox()
        self.combo_sd_model.addItems(["2D_versatile_fluo", "2D_versatile_he"])
        row_sd.addWidget(QLabel("Model:"))
        row_sd.addWidget(self.combo_sd_model)
        self.spin_sd_prob = QDoubleSpinBox()
        self.spin_sd_prob.setRange(0.0, 1.0)
        self.spin_sd_prob.setValue(0.5)
        row_sd.addWidget(QLabel("Prob:"))
        row_sd.addWidget(self.spin_sd_prob)
        self.spin_sd_nms = QDoubleSpinBox()
        self.spin_sd_nms.setRange(0.0, 1.0)
        self.spin_sd_nms.setValue(0.4)
        row_sd.addWidget(QLabel("NMS:"))
        row_sd.addWidget(self.spin_sd_nms)
        sd_layout.addLayout(row_sd)
        
        row_sd_cyto = QHBoxLayout()
        self.chk_sd_cyto = QCheckBox("Approximate Cytoplasm")
        row_sd_cyto.addWidget(self.chk_sd_cyto)
        self.spin_sd_cyto_rad = QSpinBox()
        self.spin_sd_cyto_rad.setRange(1, 100)
        self.spin_sd_cyto_rad.setValue(5)
        self.spin_sd_cyto_rad.setEnabled(False)
        self.chk_sd_cyto.toggled.connect(self.spin_sd_cyto_rad.setEnabled)
        row_sd_cyto.addWidget(QLabel("Radius (px):"))
        row_sd_cyto.addWidget(self.spin_sd_cyto_rad)
        sd_layout.addLayout(row_sd_cyto)
        
        self.group_sd.setLayout(sd_layout)
        gl_main.addWidget(self.group_sd)

        # -- InstanSeg --
        self.group_is = QGroupBox("InstanSeg")
        self.group_is.setCheckable(True)
        self.group_is.setChecked(False)
        self.method_checkboxes["InstanSeg"] = self.group_is
        is_layout = QVBoxLayout()
        
        row_is_c = QHBoxLayout()
        self.combo_is_nuc = QComboBox()
        self.combo_is_mem = QComboBox()
        row_is_c.addWidget(QLabel("Nuc Channel:"))
        row_is_c.addWidget(self.combo_is_nuc)
        row_is_c.addWidget(QLabel("Mem Channel:"))
        row_is_c.addWidget(self.combo_is_mem)
        is_layout.addLayout(row_is_c)
        
        row_is = QHBoxLayout()
        self.combo_is_model = QComboBox()
        self.combo_is_model.addItems([
            "fluorescence_nuclei_and_cells",
            "fluorescence_nuclei",
            "brightfield_nuclei",
        ])
        row_is.addWidget(QLabel("Model:"))
        row_is.addWidget(self.combo_is_model)
        self.spin_is_pixelsize = QDoubleSpinBox()
        self.spin_is_pixelsize.setRange(0.00, 10.0)
        self.spin_is_pixelsize.setValue(0.0)
        self.spin_is_pixelsize.setSpecialValueText("Disabled (Native)")
        row_is.addWidget(QLabel("Pixel Size (µm):"))
        row_is.addWidget(self.spin_is_pixelsize)
        is_layout.addLayout(row_is)
        
        self.group_is.setLayout(is_layout)
        gl_main.addWidget(self.group_is)
        
        # -- CellSAM --
        self.group_cs = QGroupBox("CellSAM")
        self.group_cs.setCheckable(True)
        self.group_cs.setChecked(False)
        self.method_checkboxes["CellSAM"] = self.group_cs
        cs_layout = QVBoxLayout()
        
        row_cs_c = QHBoxLayout()
        self.combo_cs_chan1 = QComboBox()
        self.combo_cs_chan2 = QComboBox()
        self.combo_cs_chan3 = QComboBox()
        row_cs_c.addWidget(QLabel("R:"))
        row_cs_c.addWidget(self.combo_cs_chan1)
        row_cs_c.addWidget(QLabel("G:"))
        row_cs_c.addWidget(self.combo_cs_chan2)
        row_cs_c.addWidget(QLabel("B:"))
        row_cs_c.addWidget(self.combo_cs_chan3)
        cs_layout.addLayout(row_cs_c)
        
        row_cs_token = QHBoxLayout()
        row_cs_token.addWidget(QLabel("Token:"))
        self.txt_cs_token = QLineEdit()
        self.txt_cs_token.setPlaceholderText("Paste DeepCell token here...")
        self.txt_cs_token.setEchoMode(QLineEdit.Password)
        row_cs_token.addWidget(self.txt_cs_token)
        cs_layout.addLayout(row_cs_token)
        self.group_cs.setLayout(cs_layout)
        gl_main.addWidget(self.group_cs)

        self.main_layout.addWidget(group_main)

    def _build_run_controls(self):
        self.btn_run = QPushButton("Run Selected Methods")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #3A6A9A; color: white; font-weight: bold; padding: 8px; font-size: 13px; } "
            "QPushButton:hover { background-color: #4A7FBA; }"
        )
        self.btn_run.clicked.connect(self._emit_run_signal)
        self.main_layout.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(
            "QPushButton { background-color: #CC4444; color: white; font-weight: bold; padding: 6px; } "
            "QPushButton:hover { background-color: #DD5555; }"
        )
        self.btn_cancel.clicked.connect(lambda: self.cancelSegmentation.emit())
        self.btn_cancel.hide()
        self.main_layout.addWidget(self.btn_cancel)

    def _build_status_panel(self):
        group = QGroupBox()
        gl = QVBoxLayout()
        gl.setContentsMargins(5, 5, 5, 5)
        gl.setSpacing(4)
        group.setLayout(gl)

        title = QLabel("Status")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        gl.addWidget(title)
        self._add_separator(gl)

        self.status_labels = {}
        for method in ["Cellpose", "StarDist", "CellSAM", "InstanSeg"]:
            row = QHBoxLayout()
            lbl_name = QLabel(f"{method}:")
            lbl_name.setFixedWidth(80)
            row.addWidget(lbl_name)

            lbl_status = QLabel("Idle")
            lbl_status.setStyleSheet(STATUS_STYLES['Idle'])
            self.status_labels[method] = lbl_status
            row.addWidget(lbl_status)
            row.addStretch()
            gl.addLayout(row)

        self.lbl_consensus_status = QLabel("Consensus: ")
        self.lbl_consensus_status.setStyleSheet("color: #AAA; font-size: 11px; margin-top: 4px;")
        gl.addWidget(self.lbl_consensus_status)
        self.main_layout.addWidget(group)

    def _add_separator(self, layout):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #555555; background-color: #555555; max-height: 1px; margin-bottom: 4px;")
        layout.addWidget(line)

    def _emit_run_signal(self):
        selected_methods = [name for name, grp in self.method_checkboxes.items() if grp.isChecked()]
        if not selected_methods:
            return

        params = {
            # Cellpose
            'cp_nuc': self.combo_cp_nuc.currentText(),
            'cp_mem': self.combo_cp_mem.currentText(),
            'cp_model': self.combo_cp_model.currentText(),
            'cp_diam': self.spin_cp_diameter.value() if self.spin_cp_diameter.value() > 0 else None,
            'cp_flow': self.spin_cp_flow.value(),
            'cp_prob': self.spin_cp_cellprob.value(),
            
            # StarDist
            'sd_nuc': self.combo_sd_nuc.currentText(),
            'sd_model': self.combo_sd_model.currentText(),
            'sd_prob': self.spin_sd_prob.value(),
            'sd_nms': self.spin_sd_nms.value(),
            'sd_cyto': self.spin_sd_cyto_rad.value() if self.chk_sd_cyto.isChecked() else 0,
            
            # InstanSeg
            'is_nuc': self.combo_is_nuc.currentText(),
            'is_mem': self.combo_is_mem.currentText(),
            'is_model': self.combo_is_model.currentText(),
            'is_pixel': self.spin_is_pixelsize.value(),

            # CellSAM
            'cs_r': self.combo_cs_chan1.currentText(),
            'cs_g': self.combo_cs_chan2.currentText(),
            'cs_b': self.combo_cs_chan3.currentText(),
            'cellsam_token': self.txt_cs_token.text().strip(),
        }

        self.runSegmentation.emit(selected_methods, params)

    def update_channel_dropdowns(self, markers: list):
        combos = [
            self.combo_cp_nuc, self.combo_cp_mem,
            self.combo_sd_nuc,
            self.combo_is_nuc, self.combo_is_mem,
            self.combo_cs_chan1, self.combo_cs_chan2, self.combo_cs_chan3
        ]
        
        for combo in combos:
            combo.blockSignals(True)
            combo.clear()
            # Everything gets a 'None' option except StarDist nuclear (must have a channel)
            if combo != self.combo_sd_nuc:
                combo.addItem("None")
            for m in markers:
                combo.addItem(m)
            combo.blockSignals(False)

        # Auto-select DAPI for nuclear channels
        for i, m in enumerate(markers):
            if m.upper() == "DAPI":
                self.combo_cp_nuc.setCurrentText(m)
                self.combo_sd_nuc.setCurrentText(m)
                self.combo_is_nuc.setCurrentText(m)
                break

    def update_method_status(self, method_name: str, status: str):
        if method_name in self.status_labels:
            self.status_labels[method_name].setText(status)
            self.status_labels[method_name].setStyleSheet(STATUS_STYLES.get(status, STATUS_STYLES['Idle']))

    def set_method_available(self, method_name: str, available: bool):
        if method_name in self.method_checkboxes:
            grp = self.method_checkboxes[method_name]
            grp.setEnabled(available)
            if not available:
                grp.setChecked(False)
                grp.setToolTip(f"{method_name}: Not installed.")
                grp.setStyleSheet("QGroupBox { color: #555; }")

    def set_running_state(self, is_running: bool):
        self.btn_run.setVisible(not is_running)
        self.btn_cancel.setVisible(is_running)
        for grp in self.method_checkboxes.values():
            grp.setEnabled(not is_running)