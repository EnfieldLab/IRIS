"""
CellSighter Phenotyping Package for IRIS
========================================
Exposes the faithful PyTorch CellSighter pipeline and ``CellSighterPhenotyper``
adapter conforming to ``PhenotyperMethodBase``.
"""
from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from .adapter import CellSighterPhenotyper
from .pipeline import (
    CellCropRecord,
    CellCropsDataset,
    CellSighterConfig,
    CellSighterNet,
    CellSighterTrainer,
    ChannelNormalizer,
    FOVData,
    ShiftAugmentation,
    compute_sampling_weights,
    create_crop_slices,
    ensemble_predict_probabilities,
    extend_slices_1d,
    extract_crops_from_fov,
    filter_channels,
    is_cellsighter_available,
    load_directory_fovs,
    load_labels_mapping,
    prepare_fov_data,
    read_channels,
    resolve_device,
)

__all__ = [
    "CellCropRecord",
    "CellCropsDataset",
    "CellSighterConfig",
    "CellSighterNet",
    "CellSighterPhenotyper",
    "CellSighterTrainer",
    "ChannelNormalizer",
    "FOVData",
    "ShiftAugmentation",
    "compute_sampling_weights",
    "create_crop_slices",
    "ensemble_predict_probabilities",
    "extend_slices_1d",
    "extract_crops_from_fov",
    "filter_channels",
    "is_cellsighter_available",
    "load_directory_fovs",
    "load_labels_mapping",
    "prepare_fov_data",
    "read_channels",
    "resolve_device",
]
