"""
IRIS-Viewer Multi-Sample Comparison Widget

A dock panel for loading, selecting, and aligning multiple TMA cores (or tissue
sections) onto the napari canvas simultaneously.

Signals
-------
addSamplesTriggered(list[str]) :
    Emitted when the user clicks "Add Selected Samples to Canvas".
clearCanvasTriggered() :
    Emitted when the user clicks "Clear Canvas".
masterFileTriggered(str) :
    Emitted with the annotation file path when the user loads an xlsx/csv.
"""
from __future__ import annotations

from typing import List, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from iris_viewer.widgets.styles import (
    BUTTON_DANGER,
    BUTTON_STUDIO,
    SECTION_SUBTITLE_STYLE,
    SHARED_DARK_THEME,
    create_section_card,
)


class MultiSampleComparisonWidget(QWidget):
    """
    Dock widget for multi-sample TMA / IF comparison.

    Features
    --------
    - Scrollable sample checklist (populated from ``update_sample_list``)
    - Select All / None toggle
    - "Add Selected Samples to Canvas" → ``addSamplesTriggered``
    - "Clear Canvas"                   → ``clearCanvasTriggered``
    - Grid gap spinner for canvas layout
    """

    addSamplesTriggered  = Signal(list)
    clearCanvasTriggered = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(SHARED_DARK_THEME)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)

        self.content = QWidget()
        self._main_layout = QVBoxLayout(self.content)
        self._main_layout.setContentsMargins(10, 10, 10, 10)
        self._main_layout.setSpacing(12)

        self.scroll.setWidget(self.content)
        outer.addWidget(self.scroll)

        self._build_comparison_group()
        self._build_layout_group()
        self._main_layout.addStretch()

    # Groups

    def _build_comparison_group(self) -> None:
        box, layout = create_section_card(
            "TMA Core & Sample Selection",
            "Select specific cores or samples to arrange on the canvas.",
        )

        self.chk_select_all = QCheckBox("Select All")
        self.chk_select_all.toggled.connect(self._on_select_all_toggled)
        layout.addWidget(self.chk_select_all)

        self.list_samples = QListWidget()
        self.list_samples.setMinimumHeight(240)
        self.list_samples.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_samples)

        self.btn_add = QPushButton("Add Selected Samples to Canvas")
        self.btn_add.setMinimumHeight(28)
        self.btn_add.setStyleSheet(BUTTON_STUDIO)
        self.btn_add.clicked.connect(self._on_add_clicked)
        layout.addWidget(self.btn_add)

        self.btn_clear = QPushButton("Clear Canvas")
        self.btn_clear.setMinimumHeight(28)
        self.btn_clear.setStyleSheet(BUTTON_DANGER)
        self.btn_clear.clicked.connect(self.clearCanvasTriggered.emit)
        layout.addWidget(self.btn_clear)

        self._main_layout.addWidget(box)

    def _build_layout_group(self) -> None:
        box, layout = create_section_card("Canvas Layout & Spacing")

        row_gap = QHBoxLayout()
        row_gap.setSpacing(8)
        row_gap.addWidget(QLabel("Grid gap (px):"))
        self.spin_gap = QSpinBox()
        self.spin_gap.setRange(0, 5000)
        self.spin_gap.setValue(200)
        self.spin_gap.setToolTip(
            "Spacing in image pixels between samples on the canvas.\n"
            "TMA grids are laid out using the _rxcx_ pattern in sample names."
        )
        row_gap.addWidget(self.spin_gap)
        layout.addLayout(row_gap)

        info = QLabel(
            "TMA cores named with _r{row}c{col}_ (e.g. Core_r2c3_image.tiff)\n"
            "are arranged automatically in a proportional matrix."
        )
        info.setWordWrap(True)
        info.setStyleSheet(SECTION_SUBTITLE_STYLE)
        layout.addWidget(info)

        self._main_layout.addWidget(box)

    # Internals

    def _add_sep(self, layout: QVBoxLayout) -> None:
        pass



    def _on_select_all_toggled(self, checked: bool) -> None:
        self.list_samples.blockSignals(True)
        for i in range(self.list_samples.count()):
            self.list_samples.item(i).setCheckState(
                Qt.Checked if checked else Qt.Unchecked
            )
        self.list_samples.blockSignals(False)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        any_unchecked = any(
            self.list_samples.item(i).checkState() == Qt.Unchecked
            for i in range(self.list_samples.count())
        )
        self.chk_select_all.blockSignals(True)
        self.chk_select_all.setChecked(
            not any_unchecked if self.list_samples.count() > 0 else False
        )
        self.chk_select_all.blockSignals(False)

    def _on_add_clicked(self) -> None:
        selected = [
            self.list_samples.item(i).text()
            for i in range(self.list_samples.count())
            if self.list_samples.item(i).checkState() == Qt.Checked
        ]
        if not selected:
            QMessageBox.information(
                self, "Nothing Selected",
                "Check at least one sample before adding to canvas."
            )
            return
        self.btn_add.setEnabled(False)
        self.btn_add.setText("Loading...")
        self.addSamplesTriggered.emit(selected)

    # Public API

    def update_sample_list(self, samples: List[str]) -> None:
        """Rebuild the checklist from a list of sample names."""
        self.list_samples.blockSignals(True)
        self.list_samples.clear()
        for s in samples:
            item = QListWidgetItem(s)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_samples.addItem(item)
        self.list_samples.blockSignals(False)
        self.chk_select_all.blockSignals(True)
        self.chk_select_all.setChecked(False)
        self.chk_select_all.blockSignals(False)

    def set_loading_done(self) -> None:
        """Restore button state after samples finish loading."""
        self.btn_add.setEnabled(True)
        self.btn_add.setText("Add Selected Samples to Canvas")

    def get_gap(self) -> int:
        return self.spin_gap.value()

    def selected_samples(self) -> List[str]:
        return [
            self.list_samples.item(i).text()
            for i in range(self.list_samples.count())
            if self.list_samples.item(i).checkState() == Qt.Checked
        ]
