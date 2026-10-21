"""
iris_viewer.widgets.segmentation_panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unified Dockable Widget combining Segmentation Pipelines,
Consensus Generation & Interactive Review, and Feature Extraction.
"""
from typing import Optional

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from iris_viewer.widgets.consensus_panel import ConsensusPanelWidget
from iris_viewer.widgets.feature_panel import FeaturePanelWidget
from iris_viewer.widgets.segmentation_pipeline_builder import (
    MultiSelectButton,
    PipelineBuilderWidget,
    PipelineGroupWidget,
    StepWidget,
)
from iris_viewer.widgets.segmentation_studio_dialog import SegmentationStudioDialog
from iris_viewer.widgets.styles import SHARED_DARK_THEME


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

        self._build_ui()

    def _build_ui(self):
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(6)

        # Top Configuration Box
        config_box = QGroupBox()
        cb = QVBoxLayout(config_box)
        cb.setContentsMargins(10, 3, 10, 10)
        cb.setSpacing(6)

        header_row = QHBoxLayout()
        title = QLabel("Segmentation Module")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        header_row.addWidget(title)
        header_row.addStretch()

        self.btn_studio_popout = QPushButton("⤢ Pop-up Window")
        self.btn_studio_popout.setToolTip("Open full-size interactive Segmentation Studio with multi-pane canvas")
        self.btn_studio_popout.setStyleSheet("""
            QPushButton {
                background-color: #2D5F8B; color: white; font-weight: bold;
                border: 1px solid #4B82B5; border-radius: 4px; padding: 4px 10px; font-size: 11px;
            }
            QPushButton:hover { background-color: #3D75A8; }
        """)
        self.btn_studio_popout.clicked.connect(self.open_studio_dialog)
        header_row.addWidget(self.btn_studio_popout)
        cb.addLayout(header_row)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)
        self.btn_nav_pipelines = QPushButton("Pipelines")
        self.btn_nav_consensus = QPushButton("Consensus")
        self.btn_nav_features  = QPushButton("Features")

        self.nav_buttons = [self.btn_nav_pipelines, self.btn_nav_consensus, self.btn_nav_features]
        for i, btn in enumerate(self.nav_buttons):
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self._switch_sub_panel(idx))
            nav_row.addWidget(btn)
        self.btn_nav_pipelines.setChecked(True)
        cb.addLayout(nav_row)

        self.lbl_studio_summary = QLabel("Pipelines: Ready | Consensus: Ready | Features: Idle")
        self.lbl_studio_summary.setStyleSheet("color: #888; font-size: 11px; font-style: italic; padding: 2px 2px;")
        cb.addWidget(self.lbl_studio_summary)
        main_layout.addWidget(config_box)

        # Stacked Sub-panels with full-height expanding policies
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 0: Pipeline Builder
        self.pipeline_builder = PipelineBuilderWidget(self)
        self.pipeline_builder.runSequence.connect(self.runSequence.emit)
        self.pipeline_builder.cancelSequence.connect(self.cancelSequence.emit)
        self.pipeline_builder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stack.addWidget(self.pipeline_builder)

        # 1: Consensus Panel
        self.consensus_panel = ConsensusPanelWidget(self)
        self.consensus_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stack.addWidget(self.consensus_panel)

        # 2: Feature Panel
        self.feature_panel = FeaturePanelWidget(self)
        self.feature_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stack.addWidget(self.feature_panel)

        main_layout.addWidget(self.stack, 1)
        self._update_nav_styles(0)

    @property
    def lbl_consensus_status(self):
        return self.pipeline_builder.lbl_consensus_status

    def open_studio_dialog(self):
        if self.popup_dialog is None:
            self.popup_dialog = SegmentationStudioDialog(
                parent=self.window(),
                ctx=self.ctx,
                consensus_panel=self.consensus_panel,
                feature_panel=self.feature_panel
            )
        self.popup_dialog._sync_state()
        self.popup_dialog.show()
        self.popup_dialog.raise_()
        self.popup_dialog.activateWindow()

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
                        border: 1px solid #5A8FCA; border-radius: 4px; padding: 5px; font-size: 11px;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #262930; color: #AAA; font-weight: bold;
                        border: 1px solid #444; border-radius: 4px; padding: 5px; font-size: 11px;
                    }
                    QPushButton:hover { background-color: #2D3748; color: #FFF; }
                """)

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

__all__ = [
    "MultiSelectButton",
    "StepWidget",
    "PipelineGroupWidget",
    "PipelineBuilderWidget",
    "SegmentationStudioDialog",
    "SegmentationPanelWidget",
    "IrisSegmentationWidget",
]
