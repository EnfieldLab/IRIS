"""
Pytest configuration and shared fixtures for IRIS test suite.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_API", "pyqt5")
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest


@pytest.fixture(scope="session")
def qapp():
    """Lazily initialize or return the existing QApplication instance."""
    qt_widgets = pytest.importorskip("qtpy.QtWidgets")
    return qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
