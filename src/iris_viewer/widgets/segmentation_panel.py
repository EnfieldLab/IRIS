""" IRIS-Viewer Segmentation Sequence Builder Panel """
import json
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QGroupBox, QFrame, QScrollArea, QLineEdit, QStackedWidget,
    QMenu, QAction, QFileDialog, QDialog, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QSlider, QMessageBox, QTabWidget, QSizePolicy, QApplication
)
from qtpy.QtCore import Qt, Signal
from iris_common.widgets.styles import SHARED_DARK_THEME, STATUS_STYLES
from iris_viewer.engine.pipeline import SegmentationStep
from iris_viewer.widgets.consensus_panel import ConsensusPanelWidget
from iris_viewer.widgets.feature_panel import FeaturePanelWidget

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


class SegmentationStudioDialog(QDialog):
    """
    Interactive full-size studio window for inspecting segmentation results,
    adjusting consensus overlap thresholds with live histogram feedback,
    and interactively navigating disputed cell masks on canvas.
    """
    def __init__(self, parent=None, ctx=None, consensus_panel=None, feature_panel=None):
        super().__init__(parent)
        self.ctx = ctx
        self.consensus_panel = consensus_panel
        self.feature_panel = feature_panel
        self.setWindowTitle("Segmentation Module")
        self.resize(1150, 750)

        self.setStyleSheet("""
            QDialog { background-color: #1a1d24; color: #E0E0E0; font-size: 12px; }
            QGroupBox { border: 1px solid #444; border-radius: 8px; margin-top: 6px; padding-top: 6px; font-weight: bold; color: #DDD; }
            QPushButton {
                background-color: #2D5F8B; color: white; font-weight: bold;
                border: 1px solid #4B82B5; border-radius: 4px; padding: 5px 12px; font-size: 12px;
            }
            QPushButton:hover { background-color: #3D75A8; }
            QComboBox, QSpinBox, QSlider {
                background-color: #2A2A2A; color: white; border: 1px solid #555;
                border-radius: 4px; padding: 3px 6px; font-size: 12px;
            }
            QLabel { color: #DDD; font-size: 12px; }
            QTableWidget {
                background-color: #14161d; color: white; gridline-color: #333;
                border: 1px solid #444; selection-background-color: #3A6A9A; font-size: 11px;
            }
            QHeaderView::section {
                background-color: #222630; color: white; padding: 5px;
                border: 1px solid #333; font-weight: bold; font-size: 11px;
            }
            QTabWidget::pane { border: 1px solid #444; background: #14161d; }
            QTabBar::tab {
                background: #2A2A2A; color: #BBB; padding: 6px 14px; font-weight: bold;
                border: 1px solid #444; border-bottom: none; border-top-left-radius: 3px; border-top-right-radius: 3px;
            }
            QTabBar::tab:selected { background: #3A6A9A; color: white; }
        """)

        self._build_ui()
        self._wire_signals()
        self.refresh_data()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # ── Top Control & KPI Bar ─────────────────────────────────────────
        top_bar = QHBoxLayout()
        top_bar.setSpacing(12)

        top_bar.addWidget(QLabel("Compare:"))
        self.combo_compare = QComboBox()
        self.combo_compare.addItems(["Nuclear Masks", "Cytoplasm Masks", "Both"])
        if self.consensus_panel is not None:
            self.combo_compare.setCurrentText(self.consensus_panel.combo_compare.currentText())
        top_bar.addWidget(self.combo_compare)

        top_bar.addWidget(QLabel("IoU Threshold:"))
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100)
        init_val = 66
        if self.consensus_panel is not None:
            init_val = self.consensus_panel.slider_threshold.value()
        self.slider_threshold.setValue(init_val)
        self.slider_threshold.setFixedWidth(130)
        self.slider_threshold.valueChanged.connect(self._on_slider_changed)
        top_bar.addWidget(self.slider_threshold)

        self.lbl_threshold_val = QLabel(f"{init_val / 100.0:.2f}")
        self.lbl_threshold_val.setStyleSheet("color: white; font-weight: bold; font-size: 12px;")
        self.lbl_threshold_val.setFixedWidth(36)
        top_bar.addWidget(self.lbl_threshold_val)

        self.btn_recompute = QPushButton("Recompute Consensus")
        self.btn_recompute.clicked.connect(self._recompute_consensus)
        top_bar.addWidget(self.btn_recompute)

        top_bar.addStretch(1)

        # KPI Badges
        kpi_box = QHBoxLayout()
        kpi_box.setSpacing(8)
        self.lbl_kpi_total = QLabel("Total: —")
        self.lbl_kpi_agreed = QLabel("Agreed: —")
        self.lbl_kpi_flagged = QLabel("Flagged: —")
        self.lbl_kpi_mean = QLabel("Mean IoU: —")

        badge_style = "background-color: #222630; padding: 4px 8px; border-radius: 4px; border: 1px solid #444; font-size: 11px;"
        for lbl in (self.lbl_kpi_total, self.lbl_kpi_agreed, self.lbl_kpi_flagged, self.lbl_kpi_mean):
            lbl.setStyleSheet(badge_style)
            kpi_box.addWidget(lbl)
        top_bar.addLayout(kpi_box)

        main_layout.addLayout(top_bar)

        # ── Main Splitter: Interactive Review Table & Visual Analytics ─────
        splitter = QSplitter(Qt.Horizontal)

        # Left: Interactive Table & Issue Jump Navigation
        left_widget = QWidget()
        ll = QVBoxLayout(left_widget)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)

        grp_table = QGroupBox("Disputed Cells (Below IoU Overlap)")
        gl = QVBoxLayout(grp_table)
        gl.setContentsMargins(8, 8, 8, 8)
        gl.setSpacing(6)

        lbl_hint = QLabel("Click any cell to jump canvas camera to cell with bounding box and competing masks.")
        lbl_hint.setStyleSheet("color: #9E9E9E; font-size: 11px; font-style: italic;")
        lbl_hint.setWordWrap(True)
        gl.addWidget(lbl_hint)

        nav_row = QHBoxLayout()
        self.btn_prev_issue = QPushButton("Prev Issue")
        self.btn_prev_issue.clicked.connect(self._on_prev_issue)
        self.btn_next_issue = QPushButton("Next Issue")
        self.btn_next_issue.clicked.connect(self._on_next_issue)
        nav_row.addWidget(self.btn_prev_issue)
        nav_row.addWidget(self.btn_next_issue)
        self.lbl_table_count = QLabel("0 issues")
        self.lbl_table_count.setStyleSheet("color: #FFAA00; font-weight: bold; font-size: 11px;")
        nav_row.addStretch(1)
        nav_row.addWidget(self.lbl_table_count)
        gl.addLayout(nav_row)

        self.table_issues = QTableWidget(0, 5)
        self.table_issues.setHorizontalHeaderLabels(["#", "Cell ID", "IoU Overlap", "Competing Methods", "Decision"])
        self.table_issues.verticalHeader().setVisible(False)
        self.table_issues.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_issues.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self.table_issues.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_issues.itemSelectionChanged.connect(self._on_table_cell_selected)
        gl.addWidget(self.table_issues, 1)

        # Selected Cell Decision & Override Card
        self.grp_override = QGroupBox("Active Cell Decision")
        ol = QVBoxLayout(self.grp_override)
        ol.setContentsMargins(6, 6, 6, 6)
        ol.setSpacing(4)
        self.lbl_sel_cell_info = QLabel("No cell selected. Click an issue in the table above.")
        self.lbl_sel_cell_info.setStyleSheet("color: #AAA; font-size: 11px;")
        ol.addWidget(self.lbl_sel_cell_info)

        self.override_buttons_layout = QVBoxLayout()
        self.override_buttons_layout.setSpacing(3)
        ol.addLayout(self.override_buttons_layout)
        gl.addWidget(self.grp_override)

        ll.addWidget(grp_table)
        splitter.addWidget(left_widget)

        # Right: Analytics & Feature Exploration
        right_tabs = QTabWidget()

        # Tab 0: IoU Agreement Distribution Plot
        tab_plot = QWidget()
        tpl = QVBoxLayout(tab_plot)
        tpl.setContentsMargins(8, 8, 8, 8)
        tpl.setSpacing(6)

        lbl_plot_desc = QLabel("Distribution of cell IoU overlaps across competing algorithms. Green = Agreed, Coral = Disputed.")
        lbl_plot_desc.setStyleSheet("color: #9E9E9E; font-size: 11px; font-style: italic;")
        tpl.addWidget(lbl_plot_desc)

        self.fig_hist = Figure(figsize=(5.5, 4.5), dpi=100)
        self.fig_hist.patch.set_facecolor('#14161d')
        self.canvas_hist = FigureCanvas(self.fig_hist)
        self.canvas_hist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tpl.addWidget(self.canvas_hist, 1)
        right_tabs.addTab(tab_plot, "IoU Agreement Distribution")

        # Tab 1: Extracted Features Preview
        tab_feat = QWidget()
        tfl = QVBoxLayout(tab_feat)
        tfl.setContentsMargins(8, 8, 8, 8)
        tfl.setSpacing(6)

        self.lbl_feat_summary = QLabel("Preview of extracted morphological & intensity features per cell.")
        self.lbl_feat_summary.setStyleSheet("color: #AAA; font-size: 11px;")
        tfl.addWidget(self.lbl_feat_summary)

        self.table_features = QTableWidget(0, 0)
        self.table_features.verticalHeader().setVisible(False)
        tfl.addWidget(self.table_features, 1)

        f_btn_row = QHBoxLayout()
        self.btn_extract_feat = QPushButton("Extract Features Now")
        self.btn_extract_feat.clicked.connect(self._on_extract_features_clicked)
        f_btn_row.addWidget(self.btn_extract_feat)
        self.btn_export_feat = QPushButton("Export CSV...")
        self.btn_export_feat.clicked.connect(self._on_export_features_clicked)
        f_btn_row.addWidget(self.btn_export_feat)
        f_btn_row.addStretch(1)
        tfl.addLayout(f_btn_row)

        right_tabs.addTab(tab_feat, "Extracted Features")

        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter, 1)

    def _wire_signals(self):
        if self.consensus_panel is not None:
            self.consensus_panel.statisticsUpdated.connect(self._on_stats_updated)
            self.consensus_panel.cellReviewShown.connect(self._on_cell_review_shown)
        if self.feature_panel is not None:
            self.feature_panel.dataExtracted.connect(self._populate_feature_table)

    def _on_slider_changed(self, int_val: int):
        val = int_val / 100.0
        self.lbl_threshold_val.setText(f"{val:.2f}")
        if self.consensus_panel is not None:
            self.consensus_panel.slider_threshold.blockSignals(True)
            self.consensus_panel.slider_threshold.setValue(int_val)
            self.consensus_panel.lbl_threshold_value.setText(f"{val:.2f}")
            self.consensus_panel.slider_threshold.blockSignals(False)
            self.consensus_panel.thresholdChanged.emit(val)
        self._draw_iou_histogram()
        self._populate_issues_table()

    def _recompute_consensus(self):
        if self.consensus_panel is not None:
            self.consensus_panel.combo_compare.setCurrentText(self.combo_compare.currentText())
            self.consensus_panel.generateConsensusRequested.emit()

    def _on_stats_updated(self, total: int, agreed: int, flagged: int, mean_conf: float):
        self.refresh_data()

    def _on_cell_review_shown(self, cell_id: int, confidence: float, method_map: dict):
        for r in range(self.table_issues.rowCount()):
            item_cid = self.table_issues.item(r, 1)
            if item_cid and item_cid.text() == str(cell_id):
                self.table_issues.blockSignals(True)
                self.table_issues.selectRow(r)
                self.table_issues.blockSignals(False)
                break
        self._populate_override_card(cell_id, confidence, method_map)

    def _on_prev_issue(self):
        if self.ctx and hasattr(self.ctx, "review_svc"):
            self.ctx.review_svc.prev_issue()

    def _on_next_issue(self):
        if self.ctx and hasattr(self.ctx, "review_svc"):
            self.ctx.review_svc.next_issue()

    def _on_table_cell_selected(self):
        curr_row = self.table_issues.currentRow()
        if curr_row < 0:
            return
        item_cid = self.table_issues.item(curr_row, 1)
        if not item_cid:
            return
        try:
            cell_id = int(item_cid.text())
        except ValueError:
            return

        if self.ctx and hasattr(self.ctx, "review_svc"):
            self.ctx.review_svc.jump_to_issue(curr_row)

        conf = 0.0
        m_map = {}
        if self.ctx and hasattr(self.ctx, "state"):
            conf = self.ctx.state.per_cell_confidence.get(cell_id, 0.0)
            m_map = self.ctx.state.cell_method_map.get(cell_id, {})
        self._populate_override_card(cell_id, conf, m_map)

    def _populate_override_card(self, cell_id: int, confidence: float, method_map: dict):
        while self.override_buttons_layout.count():
            child = self.override_buttons_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        methods_txt = ", ".join(method_map.keys()) if method_map else "None"
        self.lbl_sel_cell_info.setText(
            f"<b>Cell ID:</b> {cell_id} &nbsp;|&nbsp; <b>IoU:</b> {confidence:.3f} &nbsp;|&nbsp; <b>Detected by:</b> {methods_txt}"
        )

        palette_hex = ["#FF4444", "#44FF44", "#4488FF", "#FF44FF"]
        palette_hover = ["#FF6666", "#66FF66", "#66AAFF", "#FF66FF"]

        for i, m_name in enumerate(sorted(method_map.keys())):
            bg = palette_hex[i % len(palette_hex)]
            hov = palette_hover[i % len(palette_hover)]
            btn = QPushButton(f"Use {m_name} mask")
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {bg}; color: black; padding: 5px; font-size: 11px; font-weight: bold; border-radius: 2px; }}"
                f"QPushButton:hover {{ background-color: {hov}; }}"
            )
            btn.clicked.connect(lambda chk, m=m_name, cid=cell_id: self._apply_override(cid, m))
            self.override_buttons_layout.addWidget(btn)

        btn_rem = QPushButton("Remove Cell entirely")
        btn_rem.setStyleSheet(
            "QPushButton {{ background-color: #333333; color: white; padding: 5px; font-size: 11px; margin-top: 4px; border: 1px solid #555; }}"
            "QPushButton:hover {{ background-color: #555555; }}"
        )
        btn_rem.clicked.connect(lambda chk, cid=cell_id: self._apply_override(cid, "Remove"))
        self.override_buttons_layout.addWidget(btn_rem)

    def _apply_override(self, cell_id: int, method: str):
        if self.ctx and hasattr(self.ctx, "review_svc"):
            self.ctx.review_svc.apply_manual_override(cell_id, method)
        curr_row = self.table_issues.currentRow()
        if 0 <= curr_row < self.table_issues.rowCount():
            item = QTableWidgetItem(f"Override: {method}")
            item.setForeground(Qt.green if method != "Remove" else Qt.gray)
            self.table_issues.setItem(curr_row, 4, item)

    def refresh_data(self):
        """Refreshes KPI metrics, flagged issues table, and IoU histogram."""
        if self.consensus_panel is not None and hasattr(self.consensus_panel, "cached_stats"):
            stats = self.consensus_panel.cached_stats
            tot = stats.get("total", 0)
            agr = stats.get("agreed", 0)
            flg = stats.get("flagged", 0)
            mn = stats.get("mean_conf", 0.0)
            pct_agr = (agr / tot * 100) if tot > 0 else 0
            pct_flg = (flg / tot * 100) if tot > 0 else 0
            self.lbl_kpi_total.setText(f"Total: {tot:,}")
            self.lbl_kpi_agreed.setText(f"Agreed: {agr:,} ({pct_agr:.1f}%)")
            self.lbl_kpi_flagged.setText(f"Flagged: {flg:,} ({pct_flg:.1f}%)")
            self.lbl_kpi_mean.setText(f"Mean IoU: {mn:.3f}")

        self._populate_issues_table()
        self._draw_iou_histogram()
        if self.feature_panel is not None and getattr(self.feature_panel, "cached_df", None) is not None:
            self._populate_feature_table(self.feature_panel.cached_df)

    def _populate_issues_table(self):
        self.table_issues.blockSignals(True)
        self.table_issues.setRowCount(0)

        flagged = []
        if self.ctx and hasattr(self.ctx, "flagged_cells"):
            flagged = self.ctx.flagged_cells

        self.lbl_table_count.setText(f"{len(flagged)} issues found")
        self.table_issues.setRowCount(len(flagged))

        for row_idx, cid in enumerate(flagged):
            conf = 0.0
            m_map = {}
            override_res = "Pending Review"
            if self.ctx and hasattr(self.ctx, "state"):
                conf = self.ctx.state.per_cell_confidence.get(cid, 0.0)
                m_map = self.ctx.state.cell_method_map.get(cid, {})
                if hasattr(self.ctx.state, "manual_overrides") and cid in self.ctx.state.manual_overrides:
                    override_res = f"Override: {self.ctx.state.manual_overrides[cid]}"

            item_idx = QTableWidgetItem(str(row_idx + 1))
            item_cid = QTableWidgetItem(str(cid))
            item_conf = QTableWidgetItem(f"{conf:.3f}")
            item_methods = QTableWidgetItem(", ".join(m_map.keys()))
            item_res = QTableWidgetItem(override_res)

            item_idx.setTextAlignment(Qt.AlignCenter)
            item_cid.setTextAlignment(Qt.AlignCenter)
            item_conf.setTextAlignment(Qt.AlignCenter)

            if "Override" in override_res:
                item_res.setForeground(Qt.green)

            self.table_issues.setItem(row_idx, 0, item_idx)
            self.table_issues.setItem(row_idx, 1, item_cid)
            self.table_issues.setItem(row_idx, 2, item_conf)
            self.table_issues.setItem(row_idx, 3, item_methods)
            self.table_issues.setItem(row_idx, 4, item_res)

        self.table_issues.blockSignals(False)

    def _draw_iou_histogram(self):
        self.fig_hist.clear()
        ax = self.fig_hist.add_subplot(111)
        ax.set_facecolor('#14161d')

        confs = []
        if self.ctx and hasattr(self.ctx, "state") and getattr(self.ctx.state, "per_cell_confidence", None):
            confs = list(self.ctx.state.per_cell_confidence.values())

        if not confs:
            ax.text(0.5, 0.5, "No consensus generated yet.\nRun pipelines and click 'Generate Consensus Map'.",
                    color='#888', ha='center', va='center', fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#333')
            self.canvas_hist.draw()
            return

        confs_arr = np.array(confs, dtype=float)
        thresh = self.slider_threshold.value() / 100.0

        bins = np.linspace(0.0, 1.0, 21)
        counts, edges = np.histogram(confs_arr, bins=bins)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        width = 0.042

        for center, count in zip(bin_centers, counts):
            color = '#44AA66' if center >= thresh else '#E57373'
            ax.bar(center, count, width=width, color=color, edgecolor='#222', alpha=0.85)

        ax.axvline(thresh, color='#FF5252', linestyle='--', linewidth=2, label=f'Threshold ({thresh:.2f})')
        ax.set_xlabel("Intersection-over-Union (IoU) Overlap", color='#DDD', fontsize=10, fontweight='bold')
        ax.set_ylabel("Number of Cells", color='#DDD', fontsize=10, fontweight='bold')
        ax.set_title(f"Cell Agreement Distribution ({len(confs_arr):,} cells)", color='#FFF', fontsize=11, fontweight='bold')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444')
        ax.legend(facecolor='#222630', edgecolor='#444', labelcolor='white')
        self.fig_hist.tight_layout()
        self.canvas_hist.draw()

    def _populate_feature_table(self, df: pd.DataFrame):
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
                item = QTableWidgetItem(txt)
                self.table_features.setItem(r, c, item)

        self.lbl_feat_summary.setText(f"Showing first 50 rows of {len(df):,} extracted cells × {len(df.columns)} features.")

    def _on_extract_features_clicked(self):
        if self.feature_panel is not None:
            self.feature_panel._emit_extract_signal()

    def _on_export_features_clicked(self):
        if self.feature_panel is not None:
            self.feature_panel.exportTriggered.emit()


class SegmentationPanelWidget(QWidget):
    """
    Unified Dockable Widget combining Segmentation Pipelines,
    Consensus Generation & Interactive Review, and Feature Extraction.
    """
    runSequence = Signal(dict)
    cancelSequence = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(SHARED_DARK_THEME)
        self.ctx = None
        self.popup_dialog: Optional[SegmentationStudioDialog] = None

        # Embedded sub-panels
        self.pipeline_builder = PipelineBuilderWidget(parent=self)
        self.consensus_panel = ConsensusPanelWidget(parent=self)
        self.feature_panel = FeaturePanelWidget(parent=self)

        # Forward pipeline builder signals
        self.pipeline_builder.runSequence.connect(lambda g: self.runSequence.emit(g))
        self.pipeline_builder.cancelSequence.connect(lambda: self.cancelSequence.emit())

        # Forward direct attributes for backward compatibility
        self.tab_widget = self.pipeline_builder.tab_widget
        self.btn_run = self.pipeline_builder.btn_run
        self.btn_cancel = self.pipeline_builder.btn_cancel
        self.btn_add_group = self.pipeline_builder.btn_add_group
        self.lbl_consensus_status = self.pipeline_builder.lbl_consensus_status

        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # ── Top Configuration Box: Category Navigation & Summary ─────────
        config_box = QGroupBox()
        cb = QVBoxLayout(config_box)
        cb.setContentsMargins(10, 3, 10, 10)
        cb.setSpacing(6)

        title = QLabel("Segmentation Module")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        cb.addWidget(title)

        # Category switcher segmented buttons
        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)
        self.btn_nav_pipelines = QPushButton("Pipelines")
        self.btn_nav_consensus = QPushButton("Consensus & Review")
        self.btn_nav_features = QPushButton("Features")

        self.nav_buttons = [self.btn_nav_pipelines, self.btn_nav_consensus, self.btn_nav_features]
        for i, btn in enumerate(self.nav_buttons):
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self._switch_sub_panel(idx))
            nav_row.addWidget(btn)
        self.btn_nav_pipelines.setChecked(True)
        cb.addLayout(nav_row)

        self.lbl_studio_summary = QLabel("Pipelines: Ready | Consensus: Idle | Features: Idle")
        self.lbl_studio_summary.setStyleSheet("color: #888; font-size: 11px; font-style: italic; padding: 2px 2px;")
        cb.addWidget(self.lbl_studio_summary)
        main_layout.addWidget(config_box)

        # Stacked sub-panels directly in layout
        self.stack = QStackedWidget()
        self.stack.addWidget(self.pipeline_builder)
        self.stack.addWidget(self.consensus_panel)
        self.stack.addWidget(self.feature_panel)
        main_layout.addWidget(self.stack, 1)

        self._update_nav_styles(0)

    def _switch_sub_panel(self, idx: int):
        self.stack.setCurrentIndex(idx)
        self._update_nav_styles(idx)

    def _update_nav_styles(self, active_idx: int):
        for i, btn in enumerate(self.nav_buttons):
            btn.blockSignals(True)
            btn.setChecked(i == active_idx)
            btn.blockSignals(False)
            if i == active_idx:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #3A6A9A; color: white; font-weight: bold;
                        border: 1px solid #5A8FCA; border-radius: 3px; padding: 5px; font-size: 11px;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #262930; color: #AAA; font-weight: bold;
                        border: 1px solid #444; border-radius: 3px; padding: 5px; font-size: 11px;
                    }
                    QPushButton:hover { background-color: #333842; color: #EEE; }
                """)

    def open_popup_window(self):
        """Opens or activates the full-size interactive Segmentation Studio Dialog."""
        if self.popup_dialog is None:
            self.popup_dialog = SegmentationStudioDialog(
                parent=self.window(),
                ctx=self.ctx,
                consensus_panel=self.consensus_panel,
                feature_panel=self.feature_panel,
            )
        else:
            self.popup_dialog.refresh_data()
        self.popup_dialog.show()
        self.popup_dialog.raise_()
        self.popup_dialog.activateWindow()

    def update_channel_dropdowns(self, markers: list):
        self.pipeline_builder.update_channel_dropdowns(markers)

    def update_available_methods(self, methods: list):
        self.pipeline_builder.update_available_methods(methods)

    def update_step_status(self, group_name: str, step_idx: int, status: str):
        self.pipeline_builder.update_step_status(group_name, step_idx, status)

    def set_running_state(self, is_running: bool):
        self.pipeline_builder.set_running_state(is_running)
        if is_running:
            self.lbl_studio_summary.setText("Pipelines: Running... | Consensus: Pending | Features: Idle")
            self.lbl_studio_summary.setStyleSheet("color: #44AAFF; font-size: 11px; font-weight: bold;")
        else:
            self.lbl_studio_summary.setText("Pipelines: Ready | Consensus: Ready | Features: Idle")
            self.lbl_studio_summary.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")


# Alias for backward compatibility
IrisSegmentationWidget = SegmentationPanelWidget