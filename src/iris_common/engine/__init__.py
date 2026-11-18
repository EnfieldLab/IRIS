"""
IRIS Common Computational Engines
=================================
Headless-safe statistical and analytical utilities shared across IRIS packages.
"""
from __future__ import annotations

from .statistics import (
    VALID_CORRECTIONS,
    VALID_TESTS,
    ComparisonResultList,
    StatTestResult,
    adjust_p_values,
    format_significance,
    results_to_dataframe,
    run_group_comparison,
)

__all__ = [
    "StatTestResult",
    "ComparisonResultList",
    "VALID_TESTS",
    "VALID_CORRECTIONS",
    "adjust_p_values",
    "format_significance",
    "results_to_dataframe",
    "run_group_comparison",
]
