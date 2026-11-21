"""
iris_viewer.widgets.segmentation_pipeline_builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sequence and pipeline builder widgets for segmentation methods:
  - MultiSelectButton: Channel multi-selection dropdown
  - StepWidget: Parameter configuration frame for single segmentation step
  - PipelineGroupWidget: Group container for sequential steps
  - PipelineBuilderWidget: Tabbed method group manager with ROI live preview
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from qtpy.QtCore import QTimer, Signal
from qtpy.QtWidgets import (
    QAction,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from iris_common.widgets.sample_scope import SCOPE_ACTIVE, SampleScopeBar
from iris_viewer.engine.pipeline import SegmentationStep
from iris_viewer.widgets.styles import (
    BUTTON_CTA,
    BUTTON_DANGER,
    BUTTON_PRIMARY,
    BUTTON_SECONDARY,
    BUTTON_STUDIO,
    SECTION_SUBTITLE_STYLE,
    SECTION_TITLE_STYLE,
    SHARED_DARK_THEME,
    create_section_card,
)


class MultiSelectButton(QPushButton):
    """Pushbutton with checkable dropdown menu for multi-channel selection."""

    selectionChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("None")
        self.setStyleSheet("text-align: left; padding: 4px;")
        self.menu = QMenu(self)
        self.setMenu(self.menu)
        self.menu.aboutToHide.connect(self._update_text)
        self.menu.triggered.connect(self._on_action_triggered)

    def _on_action_triggered(self, _action=None) -> None:
        self._update_text()
        self.selectionChanged.emit()

    def addItems(self, items: List[str]) -> None:
        self.menu.clear()
        for text in items:
            action = QAction(text, self.menu)
            action.setCheckable(True)
            self.menu.addAction(action)

    def clear(self) -> None:
        self.menu.clear()
        self.setText("None")

    def get_checked_items(self) -> List[str]:
        return [act.text() for act in self.menu.actions() if act.isChecked()]

    def set_checked_items(self, items: List[str]) -> None:
        self.menu.blockSignals(True)
        for act in self.menu.actions():
            act.setChecked(act.text() in items)
        self.menu.blockSignals(False)
        self._update_text()

    def _update_text(self) -> None:
        checked = self.get_checked_items()
        if not checked:
            self.setText("None")
        else:
            self.setText(", ".join(checked))


class StepWidget(QFrame):
    """Parameter configuration card for a single segmentation step."""

    removed = Signal(object)
    paramsChanged = Signal()

    STEP_TYPES = [
        "Nuclear (Base)",
        "Cytoplasm (Sequential)",
        "Cytoplasm (Dilate Remaining)",
        "Independent (Whole-Cell / Dual)",
    ]

    def __init__(self, available_methods: List[str], markers: List[str], parent=None):
        super().__init__(parent)
        self.setObjectName("StepCard")
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame#StepCard { background-color: #232832; border-radius: 8px; "
            "border: 1px solid #3c4352; margin-bottom: 2px; padding: 6px; }"
        )
        self.available_methods = list(available_methods)
        self.markers = list(markers)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # Step Type Row
        row0 = QHBoxLayout()
        self.combo_type = QComboBox()
        self.combo_type.addItems(self.STEP_TYPES)
        self.combo_type.currentTextChanged.connect(self._on_type_changed)
        row0.addWidget(QLabel("Step Type:"))
        row0.addWidget(self.combo_type, stretch=1)

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setStyleSheet("color: #888; font-size: 10px; font-weight: bold;")
        row0.addWidget(self.lbl_status)

        self.btn_remove = QPushButton("X")
        self.btn_remove.setFixedSize(20, 20)
        self.btn_remove.setStyleSheet(
            "background-color: #CC4444; color: white; font-weight: bold; border-radius: 3px;"
        )
        self.btn_remove.clicked.connect(lambda: self.removed.emit(self))
        row0.addWidget(self.btn_remove)
        main_layout.addLayout(row0)

        # Method Row
        self.row_method = QHBoxLayout()
        self.lbl_algorithm = QLabel("Algorithm:")
        self.combo_method = QComboBox()
        self.combo_method.addItems(self.available_methods)
        self.combo_method.currentTextChanged.connect(self._on_method_changed)
        self.row_method.addWidget(self.lbl_algorithm)
        self.row_method.addWidget(self.combo_method, stretch=1)

        # Dilation Radius & Mode (Only for Cytoplasm (Dilate Remaining))
        self.lbl_dilate = QLabel("Radius (px):")
        self.spin_dilate = QSpinBox()
        self.spin_dilate.setRange(1, 200)
        self.spin_dilate.setValue(5)
        self.lbl_dilate_mode = QLabel("Mode:")
        self.combo_dilate_mode = QComboBox()
        self.combo_dilate_mode.addItems(["distance", "watershed"])
        self.combo_dilate_mode.currentTextChanged.connect(self._on_dilate_mode_changed)

        self.row_method.addWidget(self.lbl_dilate)
        self.row_method.addWidget(self.spin_dilate)
        self.row_method.addWidget(self.lbl_dilate_mode)
        self.row_method.addWidget(self.combo_dilate_mode)
        self.lbl_dilate.hide()
        self.spin_dilate.hide()
        self.lbl_dilate_mode.hide()
        self.combo_dilate_mode.hide()
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

        # Advanced Model Settings Stack
        self.stack = QStackedWidget()
        self._build_cellpose_page()
        self._build_stardist_page()
        self._build_instanseg_page()
        self._build_cellsam_page()
        self.page_empty = QWidget()
        self.stack.addWidget(self.page_empty)
        main_layout.addWidget(self.stack)

        # Compact inline Min Area row for model steps
        self.row_min_area_widget = QWidget()
        row_area = QHBoxLayout(self.row_min_area_widget)
        row_area.setContentsMargins(0, 0, 0, 0)
        row_area.setSpacing(6)
        row_area.addWidget(QLabel("Min Area (px):"))
        self.spin_min_cell_area = QSpinBox()
        self.spin_min_cell_area.setRange(0, 100000)
        self.spin_min_cell_area.setValue(15)
        row_area.addWidget(self.spin_min_cell_area)
        row_area.addStretch()
        main_layout.addWidget(self.row_min_area_widget)

        # Wire paramsChanged across all controls
        self._wire_param_signals()

        self._on_method_changed(self.combo_method.currentText())
        self._on_type_changed(self.combo_type.currentText())

    def _build_cellpose_page(self) -> None:
        self.page_cp = QWidget()
        form = QFormLayout(self.page_cp)
        form.setContentsMargins(0, 2, 0, 2)
        form.setSpacing(4)

        row_top = QHBoxLayout()
        self.combo_cp_model = QComboBox()
        self.combo_cp_model.addItems(["cpsam", "cyto3", "cyto2", "cyto", "nuclei"])
        self.spin_cp_diam = QSpinBox()
        self.spin_cp_diam.setRange(0, 500)
        self.spin_cp_diam.setSpecialValueText("Auto")
        row_top.addWidget(self.combo_cp_model, 1)
        row_top.addWidget(QLabel("Diam (px):"))
        row_top.addWidget(self.spin_cp_diam)
        form.addRow("Model:", row_top)

        row_thresh = QHBoxLayout()
        self.spin_cp_flow = QDoubleSpinBox()
        self.spin_cp_flow.setRange(0.0, 3.0)
        self.spin_cp_flow.setSingleStep(0.05)
        self.spin_cp_flow.setDecimals(2)
        self.spin_cp_flow.setValue(0.4)

        self.spin_cp_cellprob = QDoubleSpinBox()
        self.spin_cp_cellprob.setRange(-6.0, 6.0)
        self.spin_cp_cellprob.setSingleStep(0.2)
        self.spin_cp_cellprob.setDecimals(2)
        self.spin_cp_cellprob.setValue(0.0)

        row_thresh.addWidget(self.spin_cp_flow)
        row_thresh.addWidget(QLabel("Cellprob:"))
        row_thresh.addWidget(self.spin_cp_cellprob)
        form.addRow("Flow Thresh:", row_thresh)

        self.spin_cp_tile_overlap = QDoubleSpinBox()
        self.spin_cp_tile_overlap.setRange(0.0, 0.5)
        self.spin_cp_tile_overlap.setSingleStep(0.05)
        self.spin_cp_tile_overlap.setDecimals(2)
        self.spin_cp_tile_overlap.setValue(0.1)
        form.addRow("Tile Overlap:", self.spin_cp_tile_overlap)

        self.stack.addWidget(self.page_cp)

    def _build_stardist_page(self) -> None:
        self.page_sd = QWidget()
        form = QFormLayout(self.page_sd)
        form.setContentsMargins(0, 2, 0, 2)
        form.setSpacing(4)

        self.combo_sd_model = QComboBox()
        self.combo_sd_model.addItems(
            ["2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018"]
        )
        form.addRow("Model:", self.combo_sd_model)

        row_thresh = QHBoxLayout()
        self.spin_sd_prob = QDoubleSpinBox()
        self.spin_sd_prob.setRange(0.0, 1.0)
        self.spin_sd_prob.setSingleStep(0.05)
        self.spin_sd_prob.setDecimals(2)
        self.spin_sd_prob.setValue(0.5)

        self.spin_sd_nms = QDoubleSpinBox()
        self.spin_sd_nms.setRange(0.0, 1.0)
        self.spin_sd_nms.setSingleStep(0.05)
        self.spin_sd_nms.setDecimals(2)
        self.spin_sd_nms.setValue(0.4)

        row_thresh.addWidget(self.spin_sd_prob)
        row_thresh.addWidget(QLabel("NMS:"))
        row_thresh.addWidget(self.spin_sd_nms)
        form.addRow("Prob Thresh:", row_thresh)

        self.chk_sd_norm = QCheckBox("Normalize Input")
        self.chk_sd_norm.setChecked(True)
        form.addRow("", self.chk_sd_norm)

        self.stack.addWidget(self.page_sd)

    def _build_instanseg_page(self) -> None:
        self.page_is = QWidget()
        form = QFormLayout(self.page_is)
        form.setContentsMargins(0, 2, 0, 2)
        form.setSpacing(4)

        self.combo_is_model = QComboBox()
        self.combo_is_model.addItems(
            ["fluorescence_nuclei_and_cells", "fluorescence_nuclei", "brightfield_nuclei"]
        )
        form.addRow("Model:", self.combo_is_model)

        row_opts = QHBoxLayout()
        self.spin_is_pixel_size = QDoubleSpinBox()
        self.spin_is_pixel_size.setRange(0.0, 10.0)
        self.spin_is_pixel_size.setSingleStep(0.05)
        self.spin_is_pixel_size.setDecimals(3)
        self.spin_is_pixel_size.setValue(0.5)
        self.chk_is_cleanup = QCheckBox("Cleanup Fragments")
        self.chk_is_cleanup.setChecked(True)
        row_opts.addWidget(self.spin_is_pixel_size)
        row_opts.addWidget(self.chk_is_cleanup)
        form.addRow("Pixel Size (µm):", row_opts)

        self.stack.addWidget(self.page_is)

    def _build_cellsam_page(self) -> None:
        self.page_cs = QWidget()
        form = QFormLayout(self.page_cs)
        form.setContentsMargins(0, 2, 0, 2)
        form.setSpacing(4)

        self.txt_cs_token = QLineEdit()
        self.txt_cs_token.setPlaceholderText("DeepCell Token")
        self.txt_cs_token.setEchoMode(QLineEdit.Password)
        form.addRow("Token:", self.txt_cs_token)

        self.spin_cs_bbox = QDoubleSpinBox()
        self.spin_cs_bbox.setRange(0.0, 1.0)
        self.spin_cs_bbox.setSingleStep(0.05)
        self.spin_cs_bbox.setDecimals(2)
        self.spin_cs_bbox.setValue(0.4)
        form.addRow("BBox Thresh:", self.spin_cs_bbox)

        self.stack.addWidget(self.page_cs)

    def _wire_param_signals(self) -> None:
        emit_fn = lambda *_args: self.paramsChanged.emit()  # noqa: E731
        for combo in (
            self.combo_type,
            self.combo_method,
            self.combo_nuc,
            self.combo_cp_model,
            self.combo_sd_model,
            self.combo_is_model,
            self.combo_dilate_mode,
        ):
            combo.currentTextChanged.connect(emit_fn)

        self.combo_mem.selectionChanged.connect(emit_fn)

        for spin in (
            self.spin_dilate,
            self.spin_cp_diam,
            self.spin_cp_flow,
            self.spin_cp_cellprob,
            self.spin_cp_tile_overlap,
            self.spin_sd_prob,
            self.spin_sd_nms,
            self.spin_is_pixel_size,
            self.spin_cs_bbox,
            self.spin_min_cell_area,
        ):
            spin.valueChanged.connect(emit_fn)

        self.chk_sd_norm.toggled.connect(emit_fn)
        self.chk_is_cleanup.toggled.connect(emit_fn)
        self.txt_cs_token.textChanged.connect(emit_fn)

    def _update_dropdowns(self) -> None:
        curr_nuc = self.combo_nuc.currentText()
        checked_mems = self.combo_mem.get_checked_items()

        self.combo_nuc.blockSignals(True)
        self.combo_mem.blockSignals(True)
        self.combo_nuc.clear()
        self.combo_mem.clear()

        for m in self.markers:
            self.combo_nuc.addItem(m)
        self.combo_mem.addItems(self.markers)

        if curr_nuc in self.markers:
            self.combo_nuc.setCurrentText(curr_nuc)
        else:
            self.combo_nuc.setCurrentIndex(0)

        self.combo_mem.set_checked_items(checked_mems)
        self.combo_mem.blockSignals(False)
        self.combo_nuc.blockSignals(False)

    def update_available_methods(self, methods: List[str]) -> None:
        if not methods:
            return
        self.available_methods = list(methods)
        curr = self.combo_method.currentText()
        self.combo_method.blockSignals(True)
        self.combo_method.clear()
        self.combo_method.addItems(self.available_methods)
        if curr in self.available_methods:
            self.combo_method.setCurrentText(curr)
        self.combo_method.blockSignals(False)
        self._on_method_changed(self.combo_method.currentText())

    def _on_method_changed(self, method_name: str) -> None:
        if method_name == "Cellpose":
            self.stack.setCurrentWidget(self.page_cp)
        elif method_name == "StarDist":
            self.stack.setCurrentWidget(self.page_sd)
        elif method_name == "InstanSeg":
            self.stack.setCurrentWidget(self.page_is)
        elif method_name == "CellSAM":
            self.stack.setCurrentWidget(self.page_cs)
        else:
            self.stack.setCurrentWidget(self.page_empty)

        step_type = self.combo_type.currentText()
        if step_type == "Nuclear (Base)":
            self.lbl_mem.hide()
            self.combo_mem.hide()
        elif step_type == "Cytoplasm (Dilate Remaining)":
            self._on_dilate_mode_changed(self.combo_dilate_mode.currentText())
        elif step_type == "Cytoplasm (Sequential)" and method_name == "StarDist":
            self.lbl_mem.hide()
            self.combo_mem.hide()
        else:
            self.lbl_mem.show()
            self.combo_mem.show()

    def _on_dilate_mode_changed(self, mode: str) -> None:
        if self.combo_type.currentText() == "Cytoplasm (Dilate Remaining)":
            show_mem = str(mode).strip().lower() == "watershed"
            self.lbl_mem.setVisible(show_mem)
            self.combo_mem.setVisible(show_mem)

    def _on_type_changed(self, step_type: str) -> None:
        if step_type == "Nuclear (Base)":
            self.combo_method.show()
            self.lbl_algorithm.show()
            self.spin_dilate.hide()
            self.lbl_dilate.hide()
            self.lbl_dilate_mode.hide()
            self.combo_dilate_mode.hide()
            self.lbl_nuc.show()
            self.combo_nuc.show()
            self.lbl_mem.hide()
            self.combo_mem.hide()
            self.stack.show()
            self.row_min_area_widget.show()
        elif step_type in ("Cytoplasm (Sequential)", "Independent (Whole-Cell / Dual)"):
            self.combo_method.show()
            self.lbl_algorithm.show()
            self.spin_dilate.hide()
            self.lbl_dilate.hide()
            self.lbl_dilate_mode.hide()
            self.combo_dilate_mode.hide()
            self.lbl_nuc.show()
            self.combo_nuc.show()
            self.lbl_mem.show()
            self.combo_mem.show()
            self.stack.show()
            self.row_min_area_widget.show()
            self._on_method_changed(self.combo_method.currentText())
        elif step_type == "Cytoplasm (Dilate Remaining)":
            self.combo_method.hide()
            self.lbl_algorithm.hide()
            self.lbl_dilate.show()
            self.spin_dilate.show()
            self.lbl_dilate_mode.show()
            self.combo_dilate_mode.show()
            self.lbl_nuc.hide()
            self.combo_nuc.hide()
            self.stack.hide()
            self.row_min_area_widget.hide()
            self._on_dilate_mode_changed(self.combo_dilate_mode.currentText())

    def update_markers(self, markers: List[str]) -> None:
        self.markers = list(markers)
        self._update_dropdowns()

    def set_status(self, status: str) -> None:
        self.lbl_status.setText(status)
        color = "#888"
        if status == "Running":
            color = "#44AAEE"
        elif status == "Done":
            color = "#44AA44"
        elif status in ("Failed", "Error"):
            color = "#DD4444"
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")

    def get_step_config(self, markers_list: List[str]) -> SegmentationStep:
        step_type = self.combo_type.currentText()
        method = self.combo_method.currentText()
        rule = "Independent" if step_type == "Independent (Whole-Cell / Dual)" else step_type

        nuc = self.combo_nuc.currentText()
        nuc_idx = markers_list.index(nuc) if nuc in markers_list else 0

        checked_mems = self.combo_mem.get_checked_items()
        mem_indices = [markers_list.index(m) for m in checked_mems if m in markers_list]

        if step_type == "Cytoplasm (Dilate Remaining)":
            params: Dict[str, object] = {
                "nuclear_channel": nuc_idx,
                "membrane_channels": mem_indices,
                "dilation_radius": int(self.spin_dilate.value()),
                "dilation_mode": self.combo_dilate_mode.currentText(),
            }
            return SegmentationStep(method_name=method, params=params, rule=rule)

        params = {
            "nuclear_channel": nuc_idx,
            "membrane_channels": mem_indices,
            "dilation_radius": 0,
            "gaussian_sigma": 0.0,
            "norm_pmin": 1.0,
            "norm_pmax": 99.8,
            "min_cell_area": int(self.spin_min_cell_area.value()),
            "max_cell_area": 0,
            "expand_radius": 0,
            "expansion_mode": "distance",
        }

        if method == "Cellpose":
            diam = self.spin_cp_diam.value()
            params["model_type"] = self.combo_cp_model.currentText()
            params["diameter"] = diam if diam > 0 else None
            params["flow_threshold"] = float(self.spin_cp_flow.value())
            params["cellprob_threshold"] = float(self.spin_cp_cellprob.value())
            params["tile_overlap"] = float(self.spin_cp_tile_overlap.value())
        elif method == "StarDist":
            params["model_name"] = self.combo_sd_model.currentText()
            params["prob_thresh"] = float(self.spin_sd_prob.value())
            params["nms_thresh"] = float(self.spin_sd_nms.value())
            params["normalize_input"] = bool(self.chk_sd_norm.isChecked())
        elif method == "InstanSeg":
            params["model_name"] = self.combo_is_model.currentText()
            params["pixel_size"] = float(self.spin_is_pixel_size.value())
            params["cleanup_fragments"] = bool(self.chk_is_cleanup.isChecked())
        elif method == "CellSAM":
            params["token"] = self.txt_cs_token.text().strip()
            params["bbox_threshold"] = float(self.spin_cs_bbox.value())

        return SegmentationStep(method_name=method, params=params, rule=rule)


class PipelineGroupWidget(QWidget):
    """Group container holding an ordered sequence of SegmentationSteps."""

    groupParamsChanged = Signal()

    def __init__(self, available_methods: List[str], markers: List[str], parent=None):
        super().__init__(parent)
        self.available_methods = list(available_methods)
        self.markers = list(markers)
        self.step_widgets: List[StepWidget] = []

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

        row_name = QHBoxLayout()
        row_name.addWidget(QLabel("Pipeline Name:"))
        self.txt_name = QLineEdit()
        row_name.addWidget(self.txt_name)
        self.main_layout.addLayout(row_name)

        info = QLabel(
            "Steps execute top-to-bottom. Build this pipeline to yield one "
            "final Nuclear & Cytoplasm mask."
        )
        info.setStyleSheet(SECTION_SUBTITLE_STYLE)
        info.setWordWrap(True)
        self.main_layout.addWidget(info)

        self.steps_layout = QVBoxLayout()
        self.main_layout.addLayout(self.steps_layout)

        self.btn_add = QPushButton("+ Add Step")
        self.btn_add.setMinimumHeight(28)
        self.btn_add.setStyleSheet(BUTTON_SECONDARY)
        self.btn_add.clicked.connect(self._add_step)
        self.main_layout.addWidget(self.btn_add)
        self.main_layout.addStretch()

        self._add_step()

    def _add_step(self) -> StepWidget:
        step = StepWidget(self.available_methods, self.markers)
        step.removed.connect(self._remove_step)
        step.paramsChanged.connect(self.groupParamsChanged.emit)
        if len(self.step_widgets) == 0:
            step.combo_type.setCurrentText("Nuclear (Base)")
        else:
            step.combo_type.setCurrentText("Cytoplasm (Sequential)")

        self.step_widgets.append(step)
        self.steps_layout.addWidget(step)
        return step

    def _remove_step(self, step_widget: StepWidget) -> None:
        if step_widget in self.step_widgets:
            self.step_widgets.remove(step_widget)
        self.steps_layout.removeWidget(step_widget)
        step_widget.deleteLater()
        self.groupParamsChanged.emit()

    def update_markers(self, markers: List[str]) -> None:
        self.markers = list(markers)
        for sw in self.step_widgets:
            sw.update_markers(self.markers)

    def update_available_methods(self, methods: List[str]) -> None:
        self.available_methods = list(methods)
        for sw in self.step_widgets:
            sw.update_available_methods(self.available_methods)

    def set_running_state(self, is_running: bool) -> None:
        self.btn_add.setEnabled(not is_running)
        for sw in self.step_widgets:
            sw.setEnabled(not is_running)

    def get_steps(self) -> List[SegmentationStep]:
        return [sw.get_step_config(self.markers) for sw in self.step_widgets]


class PipelineBuilderWidget(QWidget):
    """Sequence builder widget for configuring segmentation method groups."""

    runSequence = Signal(dict)
    cancelSequence = Signal()
    previewRoiRequested = Signal(str, list)
    drawPreviewRoiRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(SHARED_DARK_THEME)

        self.available_methods = ["Cellpose", "StarDist", "InstanSeg", "CellSAM"]
        self.markers: List[str] = []
        self.groups = {}

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Header controls
        row_header = QHBoxLayout()
        title = QLabel("Method Groups (Pipelines)")
        title.setStyleSheet(SECTION_TITLE_STYLE)
        row_header.addWidget(title)

        self.btn_add_group = QPushButton("+ New Group")
        self.btn_add_group.setMinimumHeight(26)
        self.btn_add_group.setStyleSheet(BUTTON_SECONDARY)
        self.btn_add_group.clicked.connect(self._add_group)
        row_header.addStretch()
        row_header.addWidget(self.btn_add_group)
        main_layout.addLayout(row_header)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)
        main_layout.addWidget(self.tab_widget)

        # Initial group
        self._add_group()

        # Live ROI Preview Controls
        roi_box, roi_lyt = create_section_card("Interactive ROI Preview")

        roi_btn_row = QHBoxLayout()
        roi_btn_row.setSpacing(8)
        self.btn_draw_roi = QPushButton("Draw ROI")
        self.btn_draw_roi.setMinimumHeight(28)
        self.btn_draw_roi.setStyleSheet(BUTTON_SECONDARY)
        self.btn_draw_roi.clicked.connect(lambda: self.drawPreviewRoiRequested.emit())

        self.btn_preview_roi = QPushButton("Preview Active on ROI")
        self.btn_preview_roi.setMinimumHeight(28)
        self.btn_preview_roi.setStyleSheet(BUTTON_STUDIO)
        self.btn_preview_roi.clicked.connect(self._emit_preview_roi)

        roi_btn_row.addWidget(self.btn_draw_roi)
        roi_btn_row.addWidget(self.btn_preview_roi, 1)
        roi_lyt.addLayout(roi_btn_row)

        self.chk_auto_preview = QCheckBox("Auto-Preview on parameter change (400 ms)")
        self.chk_auto_preview.setChecked(False)
        roi_lyt.addWidget(self.chk_auto_preview)
        main_layout.addWidget(roi_box)

        # Debounce timer for Auto-Preview
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(400)
        self._preview_timer.timeout.connect(self._emit_preview_roi)

        # Sample Scope Selector
        self.sample_scope_bar = SampleScopeBar(
            default_mode=SCOPE_ACTIVE,
            label="Target Scope:",
            compact=True,
        )
        main_layout.addWidget(self.sample_scope_bar)

        # Execution Row
        exec_row = QHBoxLayout()
        exec_row.setSpacing(8)
        self.btn_run_active = QPushButton("Run Active Group")
        self.btn_run_active.setMinimumHeight(30)
        self.btn_run_active.setStyleSheet(BUTTON_CTA)
        self.btn_run_active.clicked.connect(self._emit_run_active_signal)

        self.btn_run = QPushButton("Run All Groups")
        self.btn_run.setMinimumHeight(30)
        self.btn_run.setStyleSheet(BUTTON_PRIMARY)
        self.btn_run.clicked.connect(self._emit_run_signal)

        exec_row.addWidget(self.btn_run_active)
        exec_row.addWidget(self.btn_run)
        main_layout.addLayout(exec_row)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setMinimumHeight(28)
        self.btn_cancel.setStyleSheet(BUTTON_DANGER)
        self.btn_cancel.clicked.connect(lambda: self.cancelSequence.emit())
        self.btn_cancel.hide()
        main_layout.addWidget(self.btn_cancel)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        main_layout.addWidget(self.progress_bar)

        self.lbl_disagreement_status = QLabel("Disagreement QC: Ready")
        self.lbl_disagreement_status.setStyleSheet(SECTION_SUBTITLE_STYLE)
        main_layout.addWidget(self.lbl_disagreement_status)

    @property
    def lbl_consensus_status(self) -> QLabel:
        """Backward-compatible alias for lbl_disagreement_status."""
        return self.lbl_disagreement_status

    def _add_group(self) -> None:
        idx = self.tab_widget.count() + 1
        name = f"Pipeline {idx}"
        group_widget = PipelineGroupWidget(self.available_methods, self.markers)
        group_widget.txt_name.setText(name)
        group_widget.groupParamsChanged.connect(self._on_any_param_changed)

        def _on_name_changed(text: str) -> None:
            for i in range(self.tab_widget.count()):
                if self.tab_widget.widget(i) == group_widget:
                    self.tab_widget.setTabText(i, text)
                    break

        group_widget.txt_name.textChanged.connect(_on_name_changed)
        self.tab_widget.addTab(group_widget, name)

    def _on_any_param_changed(self) -> None:
        if self.chk_auto_preview.isChecked():
            self._preview_timer.start()

    def _close_tab(self, index: int) -> None:
        if self.tab_widget.count() > 1:
            widget = self.tab_widget.widget(index)
            self.tab_widget.removeTab(index)
            if widget is not None:
                widget.deleteLater()

    def update_channel_dropdowns(self, markers: list) -> None:
        self.markers = list(markers)
        for i in range(self.tab_widget.count()):
            self.tab_widget.widget(i).update_markers(self.markers)

    def update_available_methods(self, methods: list) -> None:
        self.available_methods = list(methods)
        if methods:
            for i in range(self.tab_widget.count()):
                self.tab_widget.widget(i).update_available_methods(self.available_methods)

    def update_step_status(self, group_name: str, step_idx: int, status: str) -> None:
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == group_name:
                group_widget = self.tab_widget.widget(i)
                if 0 <= step_idx < len(group_widget.step_widgets):
                    group_widget.step_widgets[step_idx].set_status(status)

    def set_progress(self, current: int, total: int, text: Optional[str] = None) -> None:
        """Show determinate (total > 0) or indeterminate (total <= 0) progress bar."""
        if total <= 0:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat(text or "Running...")
        else:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(max(0, min(current, total)))
            if text:
                self.progress_bar.setFormat(text)
            else:
                pct = int(round(100 * max(0, min(current, total)) / total))
                self.progress_bar.setFormat(f"%v / %m ({pct}%)")
        self.progress_bar.show()

    def stop_progress(self) -> None:
        """Reset and hide the execution progress bar."""
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

    def set_samples(
        self,
        all_samples: Sequence[str],
        active_samples: Optional[Sequence[str]] = None,
    ) -> None:
        """Synchronize available and active samples on the Target Scope selector."""
        self.sample_scope_bar.set_samples(all_samples, active_samples)

    def get_target_samples(
        self,
        all_samples: Optional[Sequence[str]] = None,
        active_samples: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Sync optional sample lists and resolve target sample names for the current scope."""
        if all_samples is not None:
            self.sample_scope_bar.set_samples(all_samples, active_samples)
        elif active_samples is not None:
            self.sample_scope_bar.set_active_samples(active_samples)
        return self.sample_scope_bar.resolve_samples()

    def set_running_state(self, is_running: bool) -> None:
        self.btn_run.setVisible(not is_running)
        self.btn_run_active.setVisible(not is_running)
        self.btn_cancel.setVisible(is_running)
        self.btn_add_group.setEnabled(not is_running)
        self.btn_draw_roi.setEnabled(not is_running)
        self.btn_preview_roi.setEnabled(not is_running)
        self.sample_scope_bar.setEnabled(not is_running)
        if is_running:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Running...")
            self.progress_bar.show()
        else:
            self.stop_progress()
        for i in range(self.tab_widget.count()):
            self.tab_widget.widget(i).set_running_state(is_running)

    def _emit_preview_roi(self) -> None:
        idx = self.tab_widget.currentIndex()
        if idx < 0:
            return
        name = self.tab_widget.tabText(idx)
        group_widget = self.tab_widget.widget(idx)
        if group_widget is None:
            return
        steps = group_widget.get_steps()
        if steps:
            self.previewRoiRequested.emit(name, steps)

    def _emit_run_active_signal(self) -> None:
        idx = self.tab_widget.currentIndex()
        if idx < 0:
            return
        name = self.tab_widget.tabText(idx)
        group_widget = self.tab_widget.widget(idx)
        if group_widget is None:
            return
        steps = group_widget.get_steps()
        if steps:
            self.runSequence.emit({name: steps})

    def _emit_run_signal(self) -> None:
        groups = {}
        for i in range(self.tab_widget.count()):
            name = self.tab_widget.tabText(i)
            steps = self.tab_widget.widget(i).get_steps()
            if steps:
                groups[name] = steps
        if groups:
            self.runSequence.emit(groups)
