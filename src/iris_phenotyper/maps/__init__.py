"""
MAPS (Machine learning for Analysis of Proteomics in Spatial biology)
=====================================================================
Supervised MLP cell-type phenotyping module for IRIS.
"""
from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from .adapter import MAPSPhenotyper
from .pipeline import (
    MAPSMLP,
    MAPSTabularDataset,
    MAPSTrainer,
    MAPSTrainingConfig,
    is_maps_available,
    resolve_device,
    resolve_dtype,
    standardize_features,
)

__all__ = [
    "MAPSMLP",
    "MAPSPhenotyper",
    "MAPSTabularDataset",
    "MAPSTrainer",
    "MAPSTrainingConfig",
    "is_maps_available",
    "resolve_device",
    "resolve_dtype",
    "standardize_features",
]
