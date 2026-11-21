"""
iris_viewer.widgets.marker_expression.subset_dialog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modal dialog allowing users to select a custom subset of samples
for focused marker expression profile computation.
"""
from __future__ import annotations

from typing import List, Set

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from iris_viewer.widgets.styles import (
    BUTTON_CTA,
    BUTTON_SECONDARY,
    DIALOG_DARK_THEME,
    style_section_header,
)


class SampleSubsetDialog(QDialog):
    """Modal dialog allowing users to select a custom subset of samples."""

    def __init__(self, available_samples: List[str], current_selected: Set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Sample Subset")
        self.resize(380, 440)
        self.selected_samples: Set[str] = set(current_selected)

        self.setStyleSheet(DIALOG_DARK_THEME)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        lbl_prompt = QLabel("Choose samples to include in marker expression profile:")
        style_section_header(lbl_prompt, subtitle=True)
        layout.addWidget(lbl_prompt)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_all = QPushButton("Select All")
        btn_all.setMinimumHeight(28)
        btn_none = QPushButton("Select None")
        btn_none.setStyleSheet(BUTTON_SECONDARY)
        btn_none.setMinimumHeight(28)
        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        layout.addLayout(btn_row)

        self.list_widget = QListWidget()
        for s in available_samples:
            item = QListWidgetItem(s)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if s in self.selected_samples else Qt.Unchecked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet(BUTTON_SECONDARY)
        btn_cancel.setMinimumHeight(28)
        btn_ok = QPushButton("Apply")
        btn_ok.setStyleSheet(BUTTON_CTA)
        btn_ok.setMinimumHeight(28)
        btn_ok.clicked.connect(self._on_apply)
        btn_cancel.clicked.connect(self.reject)
        action_row.addWidget(btn_cancel)
        action_row.addWidget(btn_ok)
        layout.addLayout(action_row)

    def _select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Checked)

    def _select_none(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Unchecked)

    def _on_apply(self):
        self.selected_samples = {
            self.list_widget.item(i).text()
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == Qt.Checked
        }
        self.accept()
