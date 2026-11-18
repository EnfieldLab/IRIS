"""
IRIS Phenotyper
===============
Unified package for single-cell phenotyping methods, standardized prediction
contracts, and benchmark evaluation utilities.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .base import PhenotyperMethodBase, PhenotypingPredictionResult
from .evaluation import EvaluationResult, evaluate_predictions, format_benchmark_report

if TYPE_CHECKING:
    from .astir import ASTIRPhenotyper
    from .cellsighter import CellSighterPhenotyper
    from .maps import MAPSPhenotyper
    from .nimbus import NimbusPhenotyper
    from .stellar import STELLARPhenotyper

__version__ = "0.2.0"

_LAZY_ADAPTER_SUBPACKAGES = {
    "MAPSPhenotyper": "iris_phenotyper.maps",
    "STELLARPhenotyper": "iris_phenotyper.stellar",
    "CellSighterPhenotyper": "iris_phenotyper.cellsighter",
    "ASTIRPhenotyper": "iris_phenotyper.astir",
    "NimbusPhenotyper": "iris_phenotyper.nimbus",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ADAPTER_SUBPACKAGES:
        module = importlib.import_module(_LAZY_ADAPTER_SUBPACKAGES[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "__version__",
    "PhenotypingPredictionResult",
    "PhenotyperMethodBase",
    "EvaluationResult",
    "evaluate_predictions",
    "format_benchmark_report",
    "MAPSPhenotyper",
    "STELLARPhenotyper",
    "CellSighterPhenotyper",
    "ASTIRPhenotyper",
    "NimbusPhenotyper",
]
