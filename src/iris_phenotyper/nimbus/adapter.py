"""
IRIS Nimbus Phenotyper Adapter
==============================
Adapter exposing Nimbus (Rumberger et al., Nature Methods 2025) through the
canonical ``PhenotyperMethodBase`` interface.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from iris_common.types import CellTypePanel
from iris_phenotyper.base import PhenotyperMethodBase, PhenotypingPredictionResult

from .pipeline import (
    NimbusPipelineOutput,
    assign_cell_types_from_panel,
    cluster_positivity_profiles,
    is_upstream_available,
    prepare_normalization_dict,
    run_nimbus_on_arrays,
    run_nimbus_on_directory,
)

if TYPE_CHECKING:
    import torch.nn as nn


class NimbusPhenotyper(PhenotyperMethodBase):
    """Pretrained Nimbus Residual U-Net phenotyper (``image_and_mask`` modality).

    Predicts per-cell marker positivity probabilities in ``[0, 1]`` directly from
    multiplexed ``(C, H, W)`` images and ``(H, W)`` instance segmentation masks,
    then assigns cell-type labels via an optional ``CellTypePanel`` or downstream
    two-stage SOM-style prototype + hierarchical consensus clustering.
    """

    name: str = "Nimbus"
    input_modality: str = "image_and_mask"
    requires_training: bool = False

    def __init__(self, **params: Any) -> None:
        defaults = self.get_default_params()
        defaults.update(params)
        self.params: Dict[str, Any] = defaults
        self.normalization_dict_: Optional[Dict[str, float]] = None
        self.cell_type_panel_: Optional[Union[CellTypePanel, Mapping[str, Sequence[str]]]] = None
        self.channel_names_: Optional[Sequence[str]] = None

    def is_available(self) -> bool:
        """Return True if PyTorch, scikit-image, and scikit-learn are available."""
        try:
            import skimage.measure  # noqa: F401
            import skimage.segmentation  # noqa: F401
            import sklearn.cluster  # noqa: F401
        except ImportError:
            return False
        return importlib.util.find_spec("torch") is not None

    def is_upstream_available(self) -> bool:
        """Return True if the upstream ``nimbus_inference`` package is installed."""
        return is_upstream_available()

    def get_default_params(self) -> Dict[str, Any]:
        """Return default hyperparameters for Nimbus inference and clustering."""
        return {
            "checkpoint": "latest",
            "cache_dir": None,
            "allow_download": True,
            "model_magnification": 10.0,
            "dataset_magnification": 20.0,
            "quantile": 0.999,
            "clip_values": (0.0, 1.0),
            "n_subset": 50,
            "input_shape": (1024, 1024),
            "batch_size": 4,
            "test_time_aug": True,
            "device": "auto",
            "mixed_precision": False,
            "positive_threshold": 0.5,
            "clustering_method": "consensus",
            "n_clusters": None,
            "max_k": 20,
            "som_grid": (10, 10),
            "z_cap": 3.0,
            "use_upstream": "auto",
            "return_pixel_maps": False,
        }

    def fit(
        self,
        images: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        *args: Any,
        channel_names: Optional[Sequence[str]] = None,
        normalization_dict: Optional[Mapping[str, float]] = None,
        cell_type_panel: Optional[Union[CellTypePanel, Mapping[str, Sequence[str]]]] = None,
        quantile: Optional[float] = None,
        n_subset: Optional[int] = None,
        **kwargs: Any,
    ) -> "NimbusPhenotyper":
        """Optionally calibrate FOV quantile normalization factors or store panel rules.

        Nimbus is a pretrained zero-shot model (``requires_training=False``); calling
        ``fit()`` is optional and never trains network weights.
        """
        if channel_names is not None:
            self.channel_names_ = list(channel_names)
        if cell_type_panel is not None:
            self.cell_type_panel_ = cell_type_panel

        if normalization_dict is not None:
            self.normalization_dict_ = {str(k): float(v) for k, v in normalization_dict.items()}
        elif images is not None:
            if isinstance(images, np.ndarray) and images.ndim in (2, 3):
                fov_list: Sequence[np.ndarray] = [images]
            elif isinstance(images, np.ndarray) and images.ndim == 4:
                fov_list = [images[i] for i in range(images.shape[0])]
            else:
                fov_list = images  # type: ignore[assignment]
            q = float(self.params["quantile"] if quantile is None else quantile)
            n_sub = self.params["n_subset"] if n_subset is None else n_subset
            self.normalization_dict_ = prepare_normalization_dict(
                fov_list,
                channel_names=self.channel_names_,
                quantile=q,
                n_subset=n_sub,
            )
        return self

    def predict(
        self,
        image: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        mask: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        *args: Any,
        images: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        masks: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        fov_paths: Optional[Sequence[Union[str, Path]]] = None,
        segmentation_naming_convention: Optional[Callable[[str], Union[str, Path]]] = None,
        mask_dir: Optional[Union[str, Path]] = None,
        approx_mask_match: bool = False,
        suffix: str = ".tiff",
        channel_names: Optional[Sequence[str]] = None,
        fov_names: Optional[Sequence[str]] = None,
        normalization_dict: Optional[Mapping[str, float]] = None,
        cell_type_panel: Optional[Union[CellTypePanel, Mapping[str, Sequence[str]]]] = None,
        positivity_df: Optional[pd.DataFrame] = None,
        model: Optional["nn.Module"] = None,
        **kwargs: Any,
    ) -> PhenotypingPredictionResult:
        """Run Nimbus marker-positivity inference and downstream cell phenotyping.

        Supports:
        - In-memory ``(C, H, W)`` image + ``(H, W)`` mask (or multi-FOV sequences).
        - Directory / OME-TIFF ``fov_paths`` via ``nimbus_inference`` or native TIFF I/O.
        - Precomputed ``positivity_df`` for downstream panel or consensus clustering.
        """
        cfg = dict(self.params)
        for key, value in kwargs.items():
            cfg[key] = value

        resolved_channels = channel_names if channel_names is not None else self.channel_names_
        resolved_norm = (
            normalization_dict if normalization_dict is not None else self.normalization_dict_
        )
        resolved_panel = cell_type_panel if cell_type_panel is not None else self.cell_type_panel_

        img_input = images if images is not None else image
        mask_input = masks if masks is not None else mask

        pipeline_out: Optional[NimbusPipelineOutput] = None
        if positivity_df is not None:
            marker_pos = positivity_df.copy()
            if marker_pos.index.name != "ObjectNumber":
                marker_pos.index = pd.Index(marker_pos.index.astype(int), name="ObjectNumber")
            cell_table = marker_pos.copy()
        elif fov_paths is not None:
            pipeline_out = run_nimbus_on_directory(
                fov_paths=fov_paths,
                segmentation_naming_convention=segmentation_naming_convention,
                mask_dir=mask_dir,
                approx_mask_match=approx_mask_match,
                channel_names=resolved_channels,
                suffix=suffix,
                normalization_dict=resolved_norm,
                checkpoint=cfg["checkpoint"],
                cache_dir=cfg["cache_dir"],
                quantile=float(cfg["quantile"]),
                n_subset=cfg["n_subset"],
                dataset_magnification=float(cfg["dataset_magnification"]),
                model_magnification=float(cfg["model_magnification"]),
                input_shape=tuple(cfg["input_shape"]),
                batch_size=int(cfg["batch_size"]),
                test_time_aug=bool(cfg["test_time_aug"]),
                device=str(cfg["device"]),
                save_predictions=bool(cfg["return_pixel_maps"]),
                use_upstream=cfg["use_upstream"],
            )
            marker_pos = pipeline_out.positivity_df
            cell_table = pipeline_out.cell_table
        else:
            if img_input is None or mask_input is None:
                raise ValueError(
                    "NimbusPhenotyper.predict() requires (image, mask), "
                    "fov_paths (+ mask_dir/segmentation_naming_convention), or positivity_df."
                )
            pipeline_out = run_nimbus_on_arrays(
                images=img_input,
                masks=mask_input,
                channel_names=resolved_channels,
                fov_names=fov_names,
                normalization_dict=resolved_norm,
                checkpoint=cfg["checkpoint"],
                model=model,
                cache_dir=cfg["cache_dir"],
                allow_download=bool(cfg["allow_download"]),
                quantile=float(cfg["quantile"]),
                clip_values=tuple(cfg["clip_values"]),
                n_subset=cfg["n_subset"],
                dataset_magnification=float(cfg["dataset_magnification"]),
                model_magnification=float(cfg["model_magnification"]),
                input_shape=tuple(cfg["input_shape"]),
                batch_size=int(cfg["batch_size"]),
                test_time_aug=bool(cfg["test_time_aug"]),
                device=str(cfg["device"]),
                mixed_precision=bool(cfg["mixed_precision"]),
                return_pixel_maps=bool(cfg["return_pixel_maps"]),
            )
            marker_pos = pipeline_out.positivity_df
            cell_table = pipeline_out.cell_table

        if resolved_panel is not None:
            labels, probs_df, confidence = assign_cell_types_from_panel(
                marker_pos,
                resolved_panel,
                positive_threshold=float(cfg["positive_threshold"]),
            )
            assignment_mode = "panel"
        else:
            labels, probs_df, confidence = cluster_positivity_profiles(
                marker_pos,
                n_clusters=cfg["n_clusters"],
                max_k=int(cfg["max_k"]),
                som_grid=tuple(cfg["som_grid"]),
                z_cap=float(cfg["z_cap"]),
                method=str(cfg["clustering_method"]),
                positive_threshold=float(cfg["positive_threshold"]),
            )
            assignment_mode = str(cfg["clustering_method"])

        metadata: Dict[str, Any] = {
            "assignment_mode": assignment_mode,
            "positive_threshold": float(cfg["positive_threshold"]),
            "n_cells": int(len(marker_pos)),
            "n_markers": int(marker_pos.shape[1]),
            "cell_table": cell_table,
        }
        if pipeline_out is not None:
            metadata.update(pipeline_out.metadata)
            metadata["normalization_dict"] = pipeline_out.normalization_dict
            if pipeline_out.probability_maps:
                metadata["probability_maps"] = pipeline_out.probability_maps

        result = PhenotypingPredictionResult(
            method_name=self.name,
            labels=labels,
            probabilities=probs_df,
            confidence=confidence,
            is_novel=None,
            marker_positivity=marker_pos,
            metadata=metadata,
        )
        result.validate()
        return result
