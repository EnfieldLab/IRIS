"""
iris_viewer.widgets.segmentation_pipeline_builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sequence and pipeline builder widgets for segmentation methods:
  - MultiSelectButton: Channel multi-selection dropdown
  - StepWidget: Parameter configuration frame for single segmentation step
  - PipelineGroupWidget: Group container for sequential steps
  - PipelineBuilderWidget: Tabbed method group manager
"""
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QAction,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from iris_viewer.engine.pipeline import SegmentationStep
from iris_viewer.widgets.styles import SHARED_DARK_THEME


class MultiSelectButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("None")
        self.setStyleSheet("text-align: left; padding: 4px;")
        self.menu = QMenu(self)
        self.setMenu(self.menu)
        self.menu.aboutToHide.connect(self._update_text)
        self.menu.triggered.connect(self._update_text)

    def addItems(self, items):
        self.menu.clear()
        for text in items:
            action = QAction(text, self.menu)
            action.setCheckable(True)
            self.menu.addAction(action)

    def clear(self):
        self.menu.clear()
        self.setText("None")

    def get_checked_items(self):
        return [act.text() for act in self.menu.actions() if act.isChecked()]

    def set_checked_items(self, items):
        for act in self.menu.actions():
            act.setChecked(act.text() in items)
        self._update_text()

    def _update_text(self):
        checked = self.get_checked_items()
        if not checked:
            self.setText("None")
        else:
            self.setText(", ".join(checked))

class StepWidget(QFrame):
    removed = Signal(object)

    def __init__(self, available_methods, markers, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { background-color: #2A2A2A; border-radius: 4px; border: 1px solid #444; margin-bottom: 2px; padding: 4px; }")
        self.available_methods = available_methods
        self.markers = markers

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # Step Type Row
        row0 = QHBoxLayout()
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Nuclear (Base)", "Cytoplasm (Sequential)", "Cytoplasm (Dilate Remaining)"])
        self.combo_type.currentTextChanged.connect(self._on_type_changed)
        row0.addWidget(QLabel("Step Type:"))
        row0.addWidget(self.combo_type, stretch=1)

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setStyleSheet("color: #888; font-size: 10px; font-weight: bold;")
        row0.addWidget(self.lbl_status)

        self.btn_remove = QPushButton("X")
        self.btn_remove.setFixedSize(20, 20)
        self.btn_remove.setStyleSheet("background-color: #CC4444; color: white; font-weight: bold; border-radius: 3px;")
        self.btn_remove.clicked.connect(lambda: self.removed.emit(self))
        row0.addWidget(self.btn_remove)
        main_layout.addLayout(row0)

        # Method Row
        self.row_method = QHBoxLayout()
        self.combo_method = QComboBox()
        self.combo_method.addItems(available_methods)
        self.combo_method.currentTextChanged.connect(self._on_method_changed)
        self.row_method.addWidget(QLabel("Algorithm:"))
        self.row_method.addWidget(self.combo_method, stretch=1)

        # Dilation Radius (Only for Type 3)
        self.spin_dilate = QSpinBox()
        self.spin_dilate.setRange(1, 100)
        self.spin_dilate.setValue(5)
        self.row_method.addWidget(self.spin_dilate)
        self.lbl_dilate = QLabel("Radius (px):")
        self.row_method.addWidget(self.lbl_dilate)
        self.spin_dilate.hide()
        self.lbl_dilate.hide()
        main_layout.addLayout(self.row_method)

        # Channels Row
        self.row_channels = QHBoxLayout()
        self.combo_nuc = QComboBox()
        self.combo_mem = MultiSelectButton()
        self._update_dropdowns()

        self.lbl_nuc = QLabel("Nuc:")
        self.row_channels.addWidget(self.lbl_nuc)
        self.row_channels.addWidget(self.combo_nuc, stretch=1)
        self.lbl_mem = QLabel("Mem:")
        self.row_channels.addWidget(self.lbl_mem)
        self.row_channels.addWidget(self.combo_mem, stretch=1)
        main_layout.addLayout(self.row_channels)

        # Advanced Settings Stack
        self.stack = QStackedWidget()

        self.page_cp = QWidget()
        cp_lyt = QHBoxLayout(self.page_cp)
        cp_lyt.setContentsMargins(0,0,0,0)
        self.combo_cp_model = QComboBox()
        self.combo_cp_model.addItems(["cyto3", "cyto2", "cyto", "nuclei"])
        self.spin_cp_diam = QSpinBox()
        self.spin_cp_diam.setRange(0, 500)
        self.spin_cp_diam.setSpecialValueText("Auto")
        cp_lyt.addWidget(QLabel("Model:"))
        cp_lyt.addWidget(self.combo_cp_model)
        cp_lyt.addWidget(QLabel("Diam:"))
        cp_lyt.addWidget(self.spin_cp_diam)
        self.stack.addWidget(self.page_cp)

        self.page_sd = QWidget()
        sd_lyt = QHBoxLayout(self.page_sd)
        sd_lyt.setContentsMargins(0,0,0,0)
        self.combo_sd_model = QComboBox()
        self.combo_sd_model.addItems(["2D_versatile_fluo", "2D_versatile_he"])
        self.spin_sd_prob = QDoubleSpinBox()
        self.spin_sd_prob.setRange(0.0, 1.0)
        self.spin_sd_prob.setValue(0.5)
        sd_lyt.addWidget(QLabel("Model:"))
        sd_lyt.addWidget(self.combo_sd_model)
        sd_lyt.addWidget(QLabel("Prob:"))
        sd_lyt.addWidget(self.spin_sd_prob)
        self.stack.addWidget(self.page_sd)

        self.page_is = QWidget()
        is_lyt = QHBoxLayout(self.page_is)
        is_lyt.setContentsMargins(0,0,0,0)
        self.combo_is_model = QComboBox()
        self.combo_is_model.addItems(["fluorescence_nuclei_and_cells", "fluorescence_nuclei"])
        is_lyt.addWidget(QLabel("Model:"))
        is_lyt.addWidget(self.combo_is_model)
        self.stack.addWidget(self.page_is)

        self.page_cs = QWidget()
        cs_lyt = QHBoxLayout(self.page_cs)
        cs_lyt.setContentsMargins(0,0,0,0)
        self.txt_cs_token = QLineEdit()
        self.txt_cs_token.setPlaceholderText("DeepCell Token")
        self.txt_cs_token.setEchoMode(QLineEdit.Password)
        cs_lyt.addWidget(QLabel("Token:"))
        cs_lyt.addWidget(self.txt_cs_token)
        self.stack.addWidget(self.page_cs)

        self.page_empty = QWidget()
        self.stack.addWidget(self.page_empty)

        main_layout.addWidget(self.stack)

        self._on_method_changed(self.combo_method.currentText())
        self._on_type_changed(self.combo_type.currentText())

    def _update_dropdowns(self):
        curr_nuc = self.combo_nuc.currentText()
        checked_mems = self.combo_mem.get_checked_items()

        self.combo_nuc.blockSignals(True)
        self.combo_nuc.clear()
        self.combo_mem.clear()

        for m in self.markers:
            self.combo_nuc.addItem(m)
        self.combo_mem.addItems(self.markers)

        if curr_nuc in self.markers: self.combo_nuc.setCurrentText(curr_nuc)
        else: self.combo_nuc.setCurrentIndex(0)

        self.combo_mem.set_checked_items(checked_mems)

        self.combo_nuc.blockSignals(False)

    def _on_method_changed(self, method_name):
        if method_name == "Cellpose": self.stack.setCurrentWidget(self.page_cp)
        elif method_name == "StarDist": self.stack.setCurrentWidget(self.page_sd)
        elif method_name == "InstanSeg": self.stack.setCurrentWidget(self.page_is)
        elif method_name == "CellSAM": self.stack.setCurrentWidget(self.page_cs)
        else: self.stack.setCurrentWidget(self.page_empty)

        # Hide mem if StarDist
        step_type = self.combo_type.currentText()
        if method_name == "StarDist" and step_type != "Cytoplasm (Dilate Remaining)":
            self.lbl_mem.hide()
            self.combo_mem.hide()
        elif step_type == "Cytoplasm (Sequential)":
            self.lbl_mem.show()
            self.combo_mem.show()

    def _on_type_changed(self, step_type):
        if step_type == "Nuclear (Base)":
            self.combo_method.show()
            self.row_method.itemAt(0).widget().show() # QLabel Algorithm
            self.spin_dilate.hide()
            self.lbl_dilate.hide()
            self.lbl_nuc.show()
            self.combo_nuc.show()
            self.lbl_mem.hide()
            self.combo_mem.hide()
            self.stack.show()

        elif step_type == "Cytoplasm (Sequential)":
            self.combo_method.show()
            self.row_method.itemAt(0).widget().show()
            self.spin_dilate.hide()
            self.lbl_dilate.hide()
            self.lbl_nuc.show()
            self.combo_nuc.show()
            self.lbl_mem.show()
            self.combo_mem.show()
            self.stack.show()
            self._on_method_changed(self.combo_method.currentText()) # re-evaluate hide logic

        elif step_type == "Cytoplasm (Dilate Remaining)":
            self.combo_method.hide()
            self.row_method.itemAt(0).widget().hide()
            self.spin_dilate.show()
            self.lbl_dilate.show()
            self.lbl_nuc.hide()
            self.combo_nuc.hide()
            self.lbl_mem.hide()
            self.combo_mem.hide()
            self.stack.hide()

    def update_markers(self, markers):
        self.markers = markers
        self._update_dropdowns()

    def set_status(self, status: str):
        self.lbl_status.setText(status)
        color = "#888"
        if status == "Running": color = "#44AAEE"
        elif status == "Done": color = "#44AA44"
        elif status == "Failed" or status == "Error": color = "#DD4444"
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")

    def get_step_config(self, markers_list) -> SegmentationStep:
        step_type = self.combo_type.currentText()
        method = self.combo_method.currentText()

        nuc = self.combo_nuc.currentText()
        nuc_idx = markers_list.index(nuc) if nuc in markers_list else 0

        checked_mems = self.combo_mem.get_checked_items()
        mem_indices = [markers_list.index(m) for m in checked_mems if m in markers_list]

        params = {
            'nuclear_channel': nuc_idx,
            'membrane_channels': mem_indices,
            'dilation_radius': self.spin_dilate.value() if step_type == "Cytoplasm (Dilate Remaining)" else 0
        }

        if step_type != "Cytoplasm (Dilate Remaining)":
            if method == "Cellpose":
                params['model_type'] = self.combo_cp_model.currentText()
                params['diameter'] = self.spin_cp_diam.value() if self.spin_cp_diam.value() > 0 else None
                params['flow_threshold'] = 0.4
                params['cellprob_threshold'] = 0.0
            elif method == "StarDist":
                params['model_name'] = self.combo_sd_model.currentText()
                params['prob_thresh'] = self.spin_sd_prob.value()
                params['nms_thresh'] = 0.4
            elif method == "InstanSeg":
                params['model_name'] = self.combo_is_model.currentText()
                params['pixel_size'] = 0.5
            elif method == "CellSAM":
                params['token'] = self.txt_cs_token.text().strip()

        return SegmentationStep(method_name=method, params=params, rule=step_type)

class PipelineGroupWidget(QWidget):
    def __init__(self, available_methods, markers, parent=None):
        super().__init__(parent)
        self.available_methods = available_methods
        self.markers = markers
        self.step_widgets = []

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        self.content_widget = QWidget()
        self.main_layout = QVBoxLayout(self.content_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(6)
        self.scroll_area.setWidget(self.content_widget)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.scroll_area)

        # Pipeline Name
        row_name = QHBoxLayout()
        row_name.addWidget(QLabel("Pipeline Name:"))
        self.txt_name = QLineEdit()
        row_name.addWidget(self.txt_name)
        self.main_layout.addLayout(row_name)

        info = QLabel("Steps execute top-to-bottom. Build this pipeline to yield one final Nuclear & Cytoplasm mask.")
        info.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        info.setWordWrap(True)
        self.main_layout.addWidget(info)

        self.steps_layout = QVBoxLayout()
        self.main_layout.addLayout(self.steps_layout)

        self.btn_add = QPushButton("+ Add Step")
        self.btn_add.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 4px;")
        self.btn_add.clicked.connect(self._add_step)
        self.main_layout.addWidget(self.btn_add)
        self.main_layout.addStretch()

        self._add_step()

    def _add_step(self):
        step = StepWidget(self.available_methods, self.markers)
        step.removed.connect(self._remove_step)
        if len(self.step_widgets) == 0:
            step.combo_type.setCurrentText("Nuclear (Base)")
        else:
            step.combo_type.setCurrentText("Cytoplasm (Sequential)")

        self.step_widgets.append(step)
        self.steps_layout.addWidget(step)
        return step

    def _remove_step(self, step_widget):
        self.step_widgets.remove(step_widget)
        self.steps_layout.removeWidget(step_widget)
        step_widget.deleteLater()

    def update_markers(self, markers):
        self.markers = markers
        for sw in self.step_widgets:
            sw.update_markers(markers)

    def set_running_state(self, is_running: bool):
        self.btn_add.setEnabled(not is_running)
        for sw in self.step_widgets:
            sw.setEnabled(not is_running)

    def get_steps(self):
        return [sw.get_step_config(self.markers) for sw in self.step_widgets]


class PipelineBuilderWidget(QWidget):
    """Sequence builder widget for configuring segmentation method groups."""
    runSequence = Signal(dict)
    cancelSequence = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(SHARED_DARK_THEME)

        self.available_methods = ["Cellpose", "StarDist", "InstanSeg", "CellSAM"]
        self.markers = []
        self.groups = {}

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # Header controls
        row_header = QHBoxLayout()
        title = QLabel("Method Groups (Pipelines)")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        row_header.addWidget(title)

        self.btn_add_group = QPushButton("+ New Group")
        self.btn_add_group.setStyleSheet("background-color: #444; color: white; font-weight: bold;")
        self.btn_add_group.clicked.connect(self._add_group)
        row_header.addStretch()
        row_header.addWidget(self.btn_add_group)
        main_layout.addLayout(row_header)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)
        main_layout.addWidget(self.tab_widget)

        # Add initial group
        self._add_group()

        self.btn_run = QPushButton("Run All Groups")
        self.btn_run.setStyleSheet("QPushButton { background-color: #3A6A9A; color: white; font-weight: bold; padding: 8px; font-size: 13px; } QPushButton:hover { background-color: #4A7FBA; }")
        self.btn_run.clicked.connect(self._emit_run_signal)
        main_layout.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet("QPushButton { background-color: #CC4444; color: white; font-weight: bold; padding: 6px; } QPushButton:hover { background-color: #DD5555; }")
        self.btn_cancel.clicked.connect(lambda: self.cancelSequence.emit())
        self.btn_cancel.hide()
        main_layout.addWidget(self.btn_cancel)

        self.lbl_consensus_status = QLabel("Consensus: Ready")
        self.lbl_consensus_status.setStyleSheet("color: #AAA; font-size: 11px; margin-top: 4px;")
        main_layout.addWidget(self.lbl_consensus_status)

    def _add_group(self):
        idx = self.tab_widget.count() + 1
        name = f"Pipeline {idx}"
        group_widget = PipelineGroupWidget(self.available_methods, self.markers)
        group_widget.txt_name.setText(name)

        def _on_name_changed(text):
            for i in range(self.tab_widget.count()):
                if self.tab_widget.widget(i) == group_widget:
                    self.tab_widget.setTabText(i, text)
                    break
        group_widget.txt_name.textChanged.connect(_on_name_changed)
        self.tab_widget.addTab(group_widget, name)

    def _close_tab(self, index):
        if self.tab_widget.count() > 1:
            self.tab_widget.removeTab(index)

    def update_channel_dropdowns(self, markers: list):
        self.markers = markers
        for i in range(self.tab_widget.count()):
            self.tab_widget.widget(i).update_markers(markers)

    def update_available_methods(self, methods: list):
        self.available_methods = methods

    def update_step_status(self, group_name: str, step_idx: int, status: str):
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == group_name:
                group_widget = self.tab_widget.widget(i)
                if 0 <= step_idx < len(group_widget.step_widgets):
                    group_widget.step_widgets[step_idx].set_status(status)

    def set_running_state(self, is_running: bool):
        self.btn_run.setVisible(not is_running)
        self.btn_cancel.setVisible(is_running)
        self.btn_add_group.setEnabled(not is_running)
        for i in range(self.tab_widget.count()):
            self.tab_widget.widget(i).set_running_state(is_running)

    def _emit_run_signal(self):
        groups = {}
        for i in range(self.tab_widget.count()):
            name = self.tab_widget.tabText(i)
            steps = self.tab_widget.widget(i).get_steps()
            if steps:
                groups[name] = steps
        if groups:
            self.runSequence.emit(groups)


