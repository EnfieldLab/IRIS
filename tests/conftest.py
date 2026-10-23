"""
Pytest configuration and shared fixtures for IRIS-Viewer test suite.
"""
import pytest
from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])
