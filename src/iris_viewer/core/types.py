"""
IRIS-Viewer Core Types
Re-exports shared domain dataclasses from ``iris_common.types`` for
backward-compatible imports across ``iris_viewer``.
"""
from __future__ import annotations

from iris_common.types import (
    BoundingBox,
    CellTypePanel,
    CellTypeRule,
    MarkerPanel,
)

__all__ = [
    "MarkerPanel",
    "CellTypeRule",
    "CellTypePanel",
    "BoundingBox",
]
