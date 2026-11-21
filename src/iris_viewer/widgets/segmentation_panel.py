"""
iris_viewer.widgets.segmentation_panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unified Dockable Widget combining Segmentation Pipelines,
Multi-Model Disagreement QC & Inspection, and Feature Extraction.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from iris_viewer.widgets.disagreement_panel import DisagreementPanelWidget
from iris_viewer.widgets.feature_panel import FeaturePanelWidget
from iris_viewer.widgets.segmentation_pipeline_builder import (
    MultiSelectButton,
    PipelineBuilderWidget,
    PipelineGroupWidget,
    StepWidget,
)
from iris_viewer.widgets.segmentation_studio_dialog import SegmentationStudioDialog
from iris_viewer.widgets.styles import (
    POPOUT_BUTTON_STYLE,
    SECTION_SUBTITLE_STYLE,
    SECTION_TITLE_STYLE,
    SHARED_DARK_THEME,
    create_section_card,
)


class SegmentationPanelWidget(QWidget):
    """
    Unified Dockable Widget combining Segmentation Pipelines,
    Multi-Model Disagreement QC, and Feature Extraction.
    """

    runSequence = Signal(dict)
    cancelSequence = Signal()
    previewRoiRequested = Signal(str, list)
    drawPreviewRoiRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(SHARED_DARK_THEME)
        self.ctx = None
        self.popup_dialog: Optional[SegmentationStudioDialog] = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(6)

        # Top Configuration Box
        config_box, cb = create_section_card(margins=(10, 6, 10, 10), spacing=8)

        header_row = QHBoxLayout()
        title = QLabel("Segmentation Module")
        title.setStyleSheet(SECTION_TITLE_STYLE)
        header_row.addWidget(title)
        header_row.addStretch()

        self.btn_studio_popout = QPushButton("⤢ Pop-up Window")
        self.btn_studio_popout.setToolTip("Open Segmentation Disagreement QC window")
        self.btn_studio_popout.setStyleSheet(POPOUT_BUTTON_STYLE)
        self.btn_studio_popout.clicked.connect(self.open_studio_dialog)
        header_row.addWidget(self.btn_studio_popout)
        cb.addLayout(header_row)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)
        self.btn_nav_pipelines = QPushButton("Pipelines")
        self.btn_nav_disagreement = QPushButton("Disagreements")
        self.btn_nav_features = QPushButton("Features")

        self.nav_buttons = [
            self.btn_nav_pipelines,
            self.btn_nav_disagreement,
            self.btn_nav_features,
        ]
        for i, btn in enumerate(self.nav_buttons):
            btn.setCheckable(True)
            btn.setMinimumHeight(26)
            btn.clicked.connect(lambda _checked, idx=i: self._switch_sub_panel(idx))
            nav_row.addWidget(btn)
        self.btn_nav_pipelines.setChecked(True)
        cb.addLayout(nav_row)

        self.lbl_studio_summary = QLabel(
            "Pipelines: Ready | Disagreements: Ready | Features: Idle"
        )
        self.lbl_studio_summary.setStyleSheet(SECTION_SUBTITLE_STYLE)
        cb.addWidget(self.lbl_studio_summary)
        main_layout.addWidget(config_box)

        # Stacked Sub-panels
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 0: Pipeline Builder
        self.pipeline_builder = PipelineBuilderWidget(self)
        self.pipeline_builder.runSequence.connect(self.runSequence.emit)
        self.pipeline_builder.cancelSequence.connect(self.cancelSequence.emit)
        self.pipeline_builder.previewRoiRequested.connect(self.previewRoiRequested.emit)
        self.pipeline_builder.drawPreviewRoiRequested.connect(
            self.drawPreviewRoiRequested.emit
        )
        self.pipeline_builder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stack.addWidget(self.pipeline_builder)

        # 1: Disagreement Panel
        self.disagreement_panel = DisagreementPanelWidget(self)
        self.disagreement_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stack.addWidget(self.disagreement_panel)

        # 2: Feature Panel
        self.feature_panel = FeaturePanelWidget(self)
        self.feature_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stack.addWidget(self.feature_panel)

        main_layout.addWidget(self.stack, 1)
        self._update_nav_styles(0)

    @property
    def lbl_disagreement_status(self) -> QLabel:
        return self.pipeline_builder.lbl_disagreement_status

    @property
    def lbl_consensus_status(self) -> QLabel:
        """Backward-compatible alias for lbl_disagreement_status."""
        return self.pipeline_builder.lbl_disagreement_status

    def open_studio_dialog(self) -> None:
        if self.popup_dialog is None:
            self.popup_dialog = SegmentationStudioDialog(
                parent=self.window(),
                ctx=self.ctx,
                disagreement_panel=self.disagreement_panel,
                feature_panel=self.feature_panel,
            )
        self.popup_dialog.refresh_data()
        self.popup_dialog.show()
        self.popup_dialog.raise_()
        self.popup_dialog.activateWindow()

    def _switch_sub_panel(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        self._update_nav_styles(idx)

    def _update_nav_styles(self, active_idx: int) -> None:
        for i, btn in enumerate(self.nav_buttons):
            btn.blockSignals(True)
            btn.setChecked(i == active_idx)
            btn.blockSignals(False)
            if i == active_idx:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #3A6A9A; color: white; font-weight: bold;
                        border: 1px solid #5A8FCA; border-radius: 4px;
                        padding: 5px; font-size: 11px;
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

    def sync_available_results(
        self,
        methods: List[str],
        active_method: Optional[str] = None,
        cell_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """Synchronize completed segmentation group names across Disagreement & Feature panels."""
        self.disagreement_panel.update_available_methods(
            methods,
            active_method=active_method,
            cell_counts=cell_counts,
        )
        self.feature_panel.update_mask_sources(methods, active=active_method)

    def sync_sample_scope(
        self,
        all_samples: Sequence[str],
        active_samples: Optional[Sequence[str]] = None,
    ) -> None:
        """Synchronize sample scope lists across Pipeline Builder and Feature Panel."""
        self.pipeline_builder.set_samples(all_samples, active_samples)
        self.feature_panel.set_samples(all_samples, active_samples)

    def update_channel_dropdowns(self, markers: list) -> None:
        self.pipeline_builder.update_channel_dropdowns(markers)

    def update_available_methods(self, methods: list) -> None:
        self.pipeline_builder.update_available_methods(methods)

    def update_step_status(self, group_name: str, step_idx: int, status: str) -> None:
        self.pipeline_builder.update_step_status(group_name, step_idx, status)

    def set_progress(self, current: int, total: int, text: Optional[str] = None) -> None:
        """Forward progress updates to the pipeline builder progress bar and summary label."""
        self.pipeline_builder.set_progress(current, total, text=text)
        if text:
            self.lbl_studio_summary.setText(text)
            self.lbl_studio_summary.setStyleSheet(
                "color: #44AAFF; font-size: 11px; font-weight: bold;"
            )

    def stop_progress(self, summary_text: Optional[str] = None) -> None:
        """Hide the progress bar and optionally update the summary text."""
        self.pipeline_builder.stop_progress()
        if summary_text is not None:
            self.lbl_studio_summary.setText(summary_text)
        self.lbl_studio_summary.setStyleSheet(
            "color: #888; font-size: 11px; font-style: italic;"
        )

    def set_running_state(self, is_running: bool) -> None:
        self.pipeline_builder.set_running_state(is_running)
        if is_running:
            self.lbl_studio_summary.setText(
                "Pipelines: Running... | Disagreements: Pending | Features: Idle"
            )
            self.lbl_studio_summary.setStyleSheet(
                "color: #44AAFF; font-size: 11px; font-weight: bold;"
            )
        else:
            self.lbl_studio_summary.setText(
                "Pipelines: Ready | Disagreements: Ready | Features: Idle"
            )
            self.lbl_studio_summary.setStyleSheet(
                "color: #888; font-size: 11px; font-style: italic;"
            )


# Alias for backward compatibility
IrisSegmentationWidget = SegmentationPanelWidget

__all__ = [
    "DisagreementPanelWidget",
    "MultiSelectButton",
    "StepWidget",
    "PipelineGroupWidget",
    "PipelineBuilderWidget",
    "SegmentationStudioDialog",
    "SegmentationPanelWidget",
    "IrisSegmentationWidget",
]
