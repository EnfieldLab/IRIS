"""
IRIS Common
===========
Shared domain types, canonical schema utilities, statistical analysis engine,
and headless-safe UI theme/publication widgets shared across IRIS packages.
"""
from __future__ import annotations

from .logging import (
    ALLOWED_MODULES,
    format_iris_log,
    iris_log,
    log_error,
    log_info,
    log_warn,
    resolve_iris_module,
)
from .schema import (
    COL_CELL_ID,
    COL_CELL_TYPE,
    COL_SAMPLE,
    COL_X,
    COL_Y,
    ensure_canonical_dataframe,
    find_sample_column,
    resolve_marker_column,
)
from .types import BoundingBox, CellTypePanel, CellTypeRule, MarkerPanel

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "ALLOWED_MODULES",
    "resolve_iris_module",
    "format_iris_log",
    "iris_log",
    "log_info",
    "log_warn",
    "log_error",
    "COL_SAMPLE",
    "COL_CELL_ID",
    "COL_X",
    "COL_Y",
    "COL_CELL_TYPE",
    "find_sample_column",
    "resolve_marker_column",
    "ensure_canonical_dataframe",
    "MarkerPanel",
    "CellTypeRule",
    "CellTypePanel",
    "BoundingBox",
]
