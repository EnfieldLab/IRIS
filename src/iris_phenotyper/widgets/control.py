"""
IRIS-Phenotyper Control Widget — inherits from ImageControlWidget and adds file loading and export.
"""
from __future__ import annotations

from typing import Optional

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)


from iris_common.widgets.styles import SHARED_DARK_THEME
from iris_common.widgets.image_control import ImageControlWidget


class PhenotyperControlWidget(ImageControlWidget):
    """
    Top-level control panel extending the base ImageControlWidget.
    Inherits all image loading and multi-sample functionality.
    """

    loadAnnotationDataTriggered = Signal(str)

    def __init__(self, viewer=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(viewer=viewer, parent=parent)
        
        # We hide the standard viewer Export group because the user will use
        # iris-viewer for exporting images and features, and Phenotyping panel
        # for exporting phenotypes.
        if hasattr(self, 'output_group'):
            self.output_group.hide()
        if hasattr(self, 'comparison_widget'):
            self.comparison_widget.hide()
