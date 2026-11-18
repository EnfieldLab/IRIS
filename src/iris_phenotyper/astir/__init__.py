"""
IRIS Phenotyper ASTIR Subpackage
================================
Faithful PyTorch variational inference implementation and ``PhenotyperMethodBase``
adapter for ASTIR (Geuenich et al., Cell Systems 2021).
"""
from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from .adapter import ASTIRPhenotyper
from .pipeline import (
    ASTIRCellStateModel,
    ASTIRCellTypeModel,
    ASTIRDataset,
    ASTIRPipeline,
    assign_celltypes_from_probabilities,
    build_batch_design_matrix,
    compute_hierarchy_probabilities,
    extract_expression_and_design,
    is_astir_available,
    is_official_astir_available,
    normalize_marker_config,
    preprocess_expression,
    run_cellstate_diagnostics,
    run_celltype_diagnostics,
)

__all__ = [
    "ASTIRPhenotyper",
    "ASTIRPipeline",
    "ASTIRDataset",
    "ASTIRCellTypeModel",
    "ASTIRCellStateModel",
    "is_astir_available",
    "is_official_astir_available",
    "normalize_marker_config",
    "extract_expression_and_design",
    "preprocess_expression",
    "build_batch_design_matrix",
    "assign_celltypes_from_probabilities",
    "compute_hierarchy_probabilities",
    "run_celltype_diagnostics",
    "run_cellstate_diagnostics",
]
