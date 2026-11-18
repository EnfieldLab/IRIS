"""
IRIS Nimbus Phenotyping Method
==============================
Faithful PyTorch implementation and IRIS adapter for Nimbus
(Rumberger et al., Nature Methods 2025; bioRxiv 10.1101/2024.06.02.597062).
"""
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from .adapter import NimbusPhenotyper
from .pipeline import (
    DEFAULT_HF_CHECKPOINT,
    DEFAULT_HF_REPO,
    NimbusPipelineOutput,
    aggregate_cell_positivity,
    assign_cell_types_from_panel,
    cluster_positivity_profiles,
    compute_fov_channel_quantiles,
    is_upstream_available,
    load_nimbus_model,
    normalize_channel_image,
    predict_channel_map,
    prepare_binary_mask,
    prepare_input_data,
    prepare_normalization_dict,
    resolve_nimbus_checkpoint,
    run_nimbus_on_arrays,
    run_nimbus_on_directory,
)

if TYPE_CHECKING:
    from .pipeline import (
        ConvBlock,
        CropConcatBlock,
        NimbusUNet,
        Pad2D,
        UNet,
        UpconvBlock,
    )

_LAZY_UNET_EXPORTS = frozenset(
    {"Pad2D", "ConvBlock", "UpconvBlock", "CropConcatBlock", "NimbusUNet", "UNet"}
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_UNET_EXPORTS:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_HF_REPO",
    "DEFAULT_HF_CHECKPOINT",
    "Pad2D",
    "ConvBlock",
    "UpconvBlock",
    "CropConcatBlock",
    "NimbusUNet",
    "UNet",
    "NimbusPipelineOutput",
    "NimbusPhenotyper",
    "is_upstream_available",
    "prepare_binary_mask",
    "prepare_input_data",
    "compute_fov_channel_quantiles",
    "prepare_normalization_dict",
    "normalize_channel_image",
    "resolve_nimbus_checkpoint",
    "load_nimbus_model",
    "predict_channel_map",
    "aggregate_cell_positivity",
    "assign_cell_types_from_panel",
    "cluster_positivity_profiles",
    "run_nimbus_on_arrays",
    "run_nimbus_on_directory",
]
