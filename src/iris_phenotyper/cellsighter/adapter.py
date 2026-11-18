"""
IRIS Phenotyper Adapter for CellSighter
=======================================
Wraps ``CellSighterTrainer`` behind ``PhenotyperMethodBase`` for single-FOV,
multi-FOV, and CellSighter-directory inputs, emitting validated
``PhenotypingPredictionResult`` instances.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from iris_phenotyper.base import PhenotyperMethodBase, PhenotypingPredictionResult

from .pipeline import (
    CellCropRecord,
    CellSighterConfig,
    CellSighterTrainer,
    PathLike,
    extract_crops_from_fov,
    filter_channels,
    is_cellsighter_available,
    load_directory_fovs,
    load_labels_mapping,
    prepare_fov_data,
)


def _resolve_optional_file(
    explicit_path: Optional[PathLike],
    dir_root: PathLike,
    filename: str,
) -> Optional[Path]:
    """Resolve an optional metadata file inside ``dir_root`` or its parent."""
    candidates = (
        explicit_path,
        Path(dir_root) / filename,
        Path(dir_root).parent / filename,
    )
    for cand in candidates:
        if cand is not None and Path(cand).exists():
            return Path(cand)
    return None


def _coerce_fov_pair(
    image: Optional[Union[np.ndarray, Sequence[np.ndarray], PathLike]],
    mask: Optional[Union[np.ndarray, Sequence[np.ndarray]]],
    images: Optional[Sequence[np.ndarray]],
    masks: Optional[Sequence[np.ndarray]],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Normalize single-FOV or multi-FOV image/mask arguments into aligned lists."""
    if images is not None and masks is not None:
        return list(images), list(masks)
    if image is not None and mask is not None:
        if isinstance(image, np.ndarray):
            return [image], [np.asarray(mask)]
        return list(image), list(mask)  # type: ignore[arg-type]
    raise ValueError("Provide either (image, mask), (images, masks), or root_dir.")


def _normalize_label_container(
    labels: Union[np.ndarray, pd.Series, Mapping[int, Any]],
    class_to_idx: Dict[str, int],
    ignore_label: int,
) -> Union[np.ndarray, Dict[int, int]]:
    """Convert string or integer cell labels into integer class indices."""
    if isinstance(labels, np.ndarray):
        if np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.floating):
            return np.rint(labels).astype(np.int64).ravel()
        out = np.full(labels.shape, ignore_label, dtype=np.int64)
        for idx, val in enumerate(labels.ravel()):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            s_val = str(val)
            if s_val not in class_to_idx:
                class_to_idx[s_val] = len(class_to_idx)
            out[idx] = class_to_idx[s_val]
        return out

    items: List[Tuple[int, Any]]
    if isinstance(labels, pd.Series):
        items = [(int(k), v) for k, v in labels.items()]
    elif isinstance(labels, Mapping):
        items = [(int(k), v) for k, v in labels.items()]
    else:
        raise TypeError(f"Unsupported labels type: {type(labels).__name__}")

    mapped: Dict[int, int] = {}
    for cid, raw in items:
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            mapped[cid] = ignore_label
        elif isinstance(raw, (int, np.integer)):
            mapped[cid] = int(raw)
        else:
            s_val = str(raw)
            if s_val not in class_to_idx:
                class_to_idx[s_val] = len(class_to_idx)
            mapped[cid] = class_to_idx[s_val]
    return mapped


class CellSighterPhenotyper(PhenotyperMethodBase):
    """IRIS phenotyping adapter for KerenLab/CellSighter ResNet50 ensemble."""

    name: str = "CellSighter"
    input_modality: str = "image_and_mask"
    requires_training: bool = True

    def __init__(
        self,
        *,
        checkpoint_path: Optional[PathLike] = None,
        class_names: Optional[Union[Sequence[str], Mapping[int, str]]] = None,
        channel_names: Optional[Sequence[str]] = None,
        **config_kwargs: Any,
    ) -> None:
        self.checkpoint_path: Optional[Path] = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self.config = CellSighterConfig(**config_kwargs)
        self.class_names_input = class_names
        self.channel_names_: Optional[List[str]] = (
            list(channel_names) if channel_names is not None else None
        )
        self.trainer: Optional[CellSighterTrainer] = None

        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            self.load(self.checkpoint_path)

    def is_available(self) -> bool:
        """Return True if PyTorch and torchvision are available."""
        return is_cellsighter_available()

    def get_default_params(self) -> Dict[str, Any]:
        """Return default hyperparameters for CellSighter."""
        return asdict(CellSighterConfig())

    def save(self, path: PathLike) -> Path:
        """Save trained CellSighter ensemble checkpoint to disk."""
        if self.trainer is None:
            raise RuntimeError("CellSighterPhenotyper is not fitted; nothing to save.")
        out = self.trainer.save(path)
        self.checkpoint_path = out
        return out

    def load(self, path: PathLike) -> "CellSighterPhenotyper":
        """Load a trained CellSighter ensemble checkpoint from disk."""
        self.trainer = CellSighterTrainer.load(path, device=self.config.device)
        self.config = self.trainer.config
        self.channel_names_ = list(self.trainer.channel_names)
        self.checkpoint_path = Path(path)
        return self

    def _prepare_channel_indices(
        self,
        channel_names: Optional[Sequence[str]],
    ) -> Tuple[Optional[List[int]], Optional[List[str]]]:
        names = list(channel_names) if channel_names is not None else self.channel_names_
        if names is None:
            return None, None
        if not self.config.blacklist_channels:
            return None, names
        filtered = filter_channels(names, self.config.blacklist_channels)
        return [idx for idx, _ in filtered], [ch for _, ch in filtered]

    def _build_in_memory_crops(
        self,
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        labels_list: Optional[Sequence[Union[np.ndarray, pd.Series, Mapping[int, Any]]]] = None,
        *,
        image_ids: Optional[Sequence[str]] = None,
        channels_first: bool = True,
        channel_indices: Optional[Sequence[int]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        drop_unlabeled: bool = True,
    ) -> List[CellCropRecord]:
        if len(images) != len(masks):
            raise ValueError(f"images/masks length mismatch: {len(images)} != {len(masks)}.")
        if labels_list is not None and len(labels_list) != len(images):
            raise ValueError(
                f"images/labels length mismatch: {len(images)} != {len(labels_list)}."
            )

        crops: List[CellCropRecord] = []
        mapping = class_to_idx if class_to_idx is not None else {}
        for idx, (img, mask) in enumerate(zip(images, masks)):
            img_id = str(image_ids[idx]) if image_ids is not None else f"fov_{idx}"
            fov = prepare_fov_data(
                img,
                mask,
                image_id=img_id,
                channels_first=channels_first,
                channel_indices=channel_indices,
                crop_size=self.config.crop_size,
                to_pad=self.config.to_pad,
            )
            lbl_obj: Optional[Union[np.ndarray, Dict[int, int]]] = None
            if labels_list is not None:
                lbl_obj = _normalize_label_container(
                    labels_list[idx], mapping, self.config.ignore_label
                )
            crops.extend(
                extract_crops_from_fov(
                    fov,
                    self.config.crop_size,
                    labels=lbl_obj,
                    drop_unlabeled=drop_unlabeled and lbl_obj is not None,
                    ignore_label=self.config.ignore_label,
                )
            )
        return crops

    def fit(
        self,
        image: Optional[Union[np.ndarray, Sequence[np.ndarray], PathLike]] = None,
        mask: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        labels: Optional[
            Union[
                np.ndarray,
                pd.Series,
                Mapping[int, Any],
                Sequence[Union[np.ndarray, pd.Series, Mapping[int, Any]]],
            ]
        ] = None,
        *,
        images: Optional[Sequence[np.ndarray]] = None,
        masks: Optional[Sequence[np.ndarray]] = None,
        val_images: Optional[Sequence[np.ndarray]] = None,
        val_masks: Optional[Sequence[np.ndarray]] = None,
        val_labels: Optional[Sequence[Union[np.ndarray, pd.Series, Mapping[int, Any]]]] = None,
        root_dir: Optional[PathLike] = None,
        train_set: Optional[Sequence[str]] = None,
        val_set: Optional[Sequence[str]] = None,
        channels_path: Optional[PathLike] = None,
        labels_mapping_path: Optional[PathLike] = None,
        channel_names: Optional[Sequence[str]] = None,
        class_names: Optional[Union[Sequence[str], Mapping[int, str]]] = None,
        channels_first: bool = True,
        save_path: Optional[PathLike] = None,
        **kwargs: Any,
    ) -> "CellSighterPhenotyper":
        """Fit a CellSighter ResNet50 ensemble on in-memory FOVs or directory data."""
        resolved_class_names = class_names if class_names is not None else self.class_names_input
        ch_indices, active_channels = self._prepare_channel_indices(channel_names)

        # Support directory-based training when root_dir or path-like first arg is supplied.
        dir_root = root_dir
        if dir_root is None and isinstance(image, (str, Path)) and mask is None:
            dir_root = image

        val_crops: Optional[List[CellCropRecord]] = None
        if dir_root is not None:
            if train_set is None:
                raise ValueError("train_set must be provided when fitting from root_dir.")
            ch_file = _resolve_optional_file(channels_path, dir_root, "channels.txt")
            _, train_crops, loaded_channels = load_directory_fovs(
                dir_root,
                train_set,
                channels_path=ch_file,
                blacklist_channels=self.config.blacklist_channels,
                crop_size=self.config.crop_size,
                to_pad=self.config.to_pad,
                load_labels=True,
                drop_unlabeled=True,
                ignore_label=self.config.ignore_label,
            )
            active_channels = loaded_channels
            if val_set:
                _, val_crops, _ = load_directory_fovs(
                    dir_root,
                    val_set,
                    channels_path=ch_file,
                    blacklist_channels=self.config.blacklist_channels,
                    crop_size=self.config.crop_size,
                    to_pad=self.config.to_pad,
                    load_labels=True,
                    drop_unlabeled=True,
                    ignore_label=self.config.ignore_label,
                )
            if resolved_class_names is None:
                lbl_file = _resolve_optional_file(
                    labels_mapping_path, dir_root, "labels_strings.txt"
                )
                if lbl_file is not None:
                    resolved_class_names = load_labels_mapping(lbl_file)
        else:
            if labels is None:
                raise ValueError("labels must be provided when fitting on images/masks.")
            train_imgs, train_msks = _coerce_fov_pair(image, mask, images, masks)
            train_lbls = (
                list(labels)
                if isinstance(labels, (list, tuple))
                else [labels]  # type: ignore[list-item]
            )

            class_to_idx: Dict[str, int] = {}
            if isinstance(resolved_class_names, Mapping):
                class_to_idx = {str(v): int(k) for k, v in resolved_class_names.items()}
            elif resolved_class_names is not None:
                class_to_idx = {str(name): i for i, name in enumerate(resolved_class_names)}

            train_crops = self._build_in_memory_crops(
                train_imgs,
                train_msks,
                train_lbls,
                channels_first=channels_first,
                channel_indices=ch_indices,
                class_to_idx=class_to_idx,
                drop_unlabeled=True,
            )
            if val_images is not None and val_masks is not None and val_labels is not None:
                val_crops = self._build_in_memory_crops(
                    val_images,
                    val_masks,
                    val_labels,
                    image_ids=[f"val_{i}" for i in range(len(val_images))],
                    channels_first=channels_first,
                    channel_indices=ch_indices,
                    class_to_idx=class_to_idx,
                    drop_unlabeled=True,
                )
            if resolved_class_names is None and class_to_idx:
                inv = {idx: name for name, idx in class_to_idx.items()}
                resolved_class_names = [inv[i] for i in sorted(inv)]

        trainer = CellSighterTrainer(config=self.config)
        trainer.fit(
            train_crops,
            val_crops=val_crops,
            class_names=resolved_class_names,
            channel_names=active_channels,
        )
        self.trainer = trainer
        self.channel_names_ = list(trainer.channel_names)

        target_save = save_path if save_path is not None else self.checkpoint_path
        if target_save is not None:
            self.save(target_save)

        return self

    def predict(
        self,
        image: Optional[Union[np.ndarray, Sequence[np.ndarray], PathLike]] = None,
        mask: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        *,
        images: Optional[Sequence[np.ndarray]] = None,
        masks: Optional[Sequence[np.ndarray]] = None,
        object_ids: Optional[Sequence[int]] = None,
        root_dir: Optional[PathLike] = None,
        image_names: Optional[Sequence[str]] = None,
        channels_path: Optional[PathLike] = None,
        channel_names: Optional[Sequence[str]] = None,
        channels_first: bool = True,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> PhenotypingPredictionResult:
        """Predict cell-type labels and ensemble probabilities from image + mask inputs."""
        if self.trainer is None:
            raise RuntimeError(
                "CellSighterPhenotyper must be fitted or loaded from a checkpoint before predict()."
            )

        t0 = time.time()
        ch_indices, _ = self._prepare_channel_indices(channel_names)

        dir_root = root_dir
        if dir_root is None and isinstance(image, (str, Path)) and mask is None:
            dir_root = image

        if dir_root is not None:
            if not image_names:
                raise ValueError("image_names must be provided when predicting from root_dir.")
            ch_file = _resolve_optional_file(channels_path, dir_root, "channels.txt")
            _, crops, _ = load_directory_fovs(
                dir_root,
                image_names,
                channels_path=ch_file,
                blacklist_channels=self.config.blacklist_channels,
                crop_size=self.config.crop_size,
                to_pad=self.config.to_pad,
                load_labels=False,
                drop_unlabeled=False,
                ignore_label=self.config.ignore_label,
            )
            if object_ids is not None:
                allowed = {int(cid) for cid in object_ids}
                crops = [c for c in crops if c.cell_id in allowed]
        else:
            pred_imgs, pred_msks = _coerce_fov_pair(image, mask, images, masks)
            crops = []
            for idx, (img, msk) in enumerate(zip(pred_imgs, pred_msks)):
                fov = prepare_fov_data(
                    img,
                    msk,
                    image_id=f"fov_{idx}",
                    channels_first=channels_first,
                    channel_indices=ch_indices,
                    crop_size=self.config.crop_size,
                    to_pad=self.config.to_pad,
                )
                crops.extend(
                    extract_crops_from_fov(
                        fov,
                        self.config.crop_size,
                        labels=None,
                        object_ids=object_ids,
                        drop_unlabeled=False,
                        ignore_label=self.config.ignore_label,
                    )
                )

        probs_arr = self.trainer.predict_crops(crops, batch_size=batch_size, device=device)
        class_names = list(self.trainer.class_names)
        obj_index = pd.Index([c.cell_id for c in crops], name="ObjectNumber", dtype=int)

        if len(crops) == 0:
            probs_df = pd.DataFrame(columns=class_names, index=obj_index, dtype=float)
            labels_ser = pd.Series([], index=obj_index, dtype=object, name="cell_type")
            conf_ser = pd.Series(np.empty((0,), dtype=float), index=obj_index, name="confidence")
        else:
            probs_arr = np.clip(probs_arr, 0.0, 1.0)
            probs_df = pd.DataFrame(probs_arr, index=obj_index, columns=class_names)
            pred_indices = probs_arr.argmax(axis=1)
            pred_labels = [class_names[int(i)] for i in pred_indices]
            labels_ser = pd.Series(pred_labels, index=obj_index, name="cell_type")
            conf_ser = pd.Series(
                probs_arr.max(axis=1).astype(float), index=obj_index, name="confidence"
            )

        result = PhenotypingPredictionResult(
            method_name=self.name,
            labels=labels_ser,
            probabilities=probs_df,
            confidence=conf_ser,
            metadata={
                "elapsed_s": time.time() - t0,
                "device": str(device or self.config.device),
                "ensemble_size": len(self.trainer.models),
                "backbone": self.config.backbone,
                "crop_size": self.config.crop_size,
                "crop_input_size": self.config.crop_input_size,
                "n_cells": len(crops),
                "image_ids": [c.image_id for c in crops],
                "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            },
        )
        result.validate()
        return result
