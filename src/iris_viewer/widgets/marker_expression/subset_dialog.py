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


class SampleSubsetDialog(QDialog):
    """Modal dialog allowing users to select a custom subset of samples."""

    def __init__(self, available_samples: List[str], current_selected: Set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Sample Subset")
        self.resize(380, 440)
        self.selected_samples: Set[str] = set(current_selected)

        self.setStyleSheet("""
            QDialog { background-color: #1f232b; color: #E0E0E0; font-size: 12px; }
            QPushButton {
                background-color: #2D5F8B; color: white; font-weight: bold;
                border: 1px solid #4B82B5; border-radius: 3px; padding: 5px 10px; font-size: 11px;
            }
            QPushButton:hover { background-color: #3D75A8; }
            QListWidget {
                background-color: #14161d; color: white; border: 1px solid #444;
                border-radius: 3px; padding: 4px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose samples to include in marker expression profile:"))

        btn_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_none = QPushButton("Select None")
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
        action_row.addStretch()
        btn_ok = QPushButton("Apply")
        btn_cancel = QPushButton("Cancel")
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
