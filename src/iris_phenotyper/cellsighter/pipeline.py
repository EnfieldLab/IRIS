"""
CellSighter Single-Cell Phenotyping Pipeline
============================================
Faithful PyTorch implementation of CellSighter (Amitay et al., Nature Communications
2023, DOI: 10.1038/s41467-023-40066-7; KerenLab/CellSighter).

CellSighter classifies individual cells directly from multiplexed tissue images and
instance segmentation masks by extracting multi-channel spatial crops around each
cell centroid, appending two binary mask channels (all cells in microenvironment and
target cell mask), applying channel-wise normalization and spatial/noise
augmentations, and training an ensemble of ResNet50 CNNs whose softmax predictions
are averaged at inference time.
"""
from __future__ import annotations

import ast
import copy
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage

# Guard against macOS dual-libomp initialization crashes when PyPI torch and conda libs coexist.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False

try:
    import tifffile

    _HAS_TIFFFILE = True
except ImportError:
    tifffile = None  # type: ignore[assignment]
    _HAS_TIFFFILE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as tv_models
    import torchvision.transforms.functional as TF
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    tv_models = None  # type: ignore[assignment]
    TF = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc, assignment]
    WeightedRandomSampler = None  # type: ignore[assignment]
    _HAS_TORCH = False


PathLike = Union[str, os.PathLike]

_VALID_NORMALIZATIONS = ("quantile", "anscombe", "none")
_VALID_BACKBONES = ("resnet50", "resnet34", "resnet18")


def is_cellsighter_available() -> bool:
    """Return True if PyTorch and torchvision are importable."""
    return bool(_HAS_TORCH)


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch and torchvision are required for CellSighter. "
            "Install them in the active environment before training or inference."
        )


def resolve_device(device: str = "auto") -> "torch.device":
    """Resolve execution device (``cpu``, ``cuda``, ``mps``, or ``auto``)."""
    _require_torch()
    key = device.lower() if isinstance(device, str) else "auto"
    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(key)


def set_seed(seed: int) -> None:
    """Set deterministic RNG seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


@dataclass
class CellSighterConfig:
    """Configuration for CellSighter crop extraction, training, and ensemble inference."""

    crop_size: int = 128
    crop_input_size: int = 60
    to_pad: bool = True
    normalization: str = "quantile"
    clip_quantile: float = 0.99
    backbone: str = "resnet50"
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    sample_batch: bool = True
    hierarchy_match: Optional[Dict[str, str]] = None
    sampling_power: float = 1.0
    samples_per_epoch: Optional[int] = None
    augment: bool = True
    poisson_aug: Union[bool, str] = "auto"
    shift_max: int = 5
    ensemble_size: int = 1
    val_fraction: float = 0.1
    device: str = "auto"
    num_workers: int = 0
    seed: int = 42
    ignore_label: int = -1
    blacklist_channels: Tuple[str, ...] = ()
    verbose: int = 0

    def __post_init__(self) -> None:
        if self.crop_input_size <= 0:
            raise ValueError("crop_input_size must be positive.")
        if self.crop_size < self.crop_input_size:
            raise ValueError(
                f"crop_size ({self.crop_size}) must be >= crop_input_size "
                f"({self.crop_input_size})."
            )
        if self.normalization not in _VALID_NORMALIZATIONS:
            raise ValueError(
                f"Unknown normalization '{self.normalization}'. "
                f"Expected one of {_VALID_NORMALIZATIONS}."
            )
        if self.backbone not in _VALID_BACKBONES:
            raise ValueError(
                f"Unknown backbone '{self.backbone}'. Expected one of {_VALID_BACKBONES}."
            )
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1.")
        if not 0.0 < self.clip_quantile <= 1.0:
            raise ValueError("clip_quantile must lie in (0, 1].")


def read_channels(path: PathLike) -> List[str]:
    """Read line-separated marker channel names from a text file."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def filter_channels(
    channels: Sequence[str],
    blacklist: Optional[Sequence[str]] = None,
) -> List[Tuple[int, str]]:
    """Filter blacklisted channels, returning ``(index, channel_name)`` pairs."""
    blocked = set(blacklist or ())
    return [(idx, ch) for idx, ch in enumerate(channels) if ch not in blocked]


def load_labels_mapping(path: PathLike) -> Dict[int, str]:
    """Load integer-to-name class mapping from ``labels_strings.txt`` or JSON."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
        return {int(k): str(v) for k, v in parsed.items()}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {idx: line for idx, line in enumerate(lines)}


def load_array_file(path: PathLike) -> np.ndarray:
    """Load an image or mask array from ``.npz`` or ``.tif``/``.tiff``."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npz":
        with np.load(p, allow_pickle=True) as archive:
            if "data" in archive.files:
                return np.asarray(archive["data"])
            return np.asarray(archive[archive.files[0]])
    if suffix in (".tif", ".tiff"):
        if _HAS_TIFFFILE:
            return np.asarray(tifffile.imread(p))
        from skimage import io as sk_io

        return np.asarray(sk_io.imread(p))
    if suffix == ".npy":
        return np.asarray(np.load(p, allow_pickle=False))
    raise ValueError(f"Unsupported array file extension: {p}")


def load_labels_file(path: PathLike) -> np.ndarray:
    """Load CellSighter ``cells2labels`` vector from ``.npz`` or ``.txt``."""
    p = Path(path)
    if p.suffix.lower() == ".npz":
        arr = load_array_file(p)
        return np.rint(arr).astype(np.int64).ravel()
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return np.empty((0,), dtype=np.int64)
    vals = [float(line.strip()) for line in text.splitlines() if line.strip()]
    return np.rint(np.asarray(vals, dtype=np.float64)).astype(np.int64)


def _is_count_like(image: np.ndarray) -> bool:
    """Heuristic check whether an image looks like raw non-negative ion counts."""
    if image.size == 0:
        return False
    sample = image.reshape(-1)[: min(image.size, 200_000)]
    min_val = float(np.min(sample))
    max_val = float(np.max(sample))
    if min_val < 0.0 or max_val <= 2.0:
        return False
    return bool(np.allclose(sample, np.floor(sample), atol=1e-4))


@dataclass
class ChannelNormalizer:
    """Per-channel intensity clipping and scaling for multiplexed crops.

    Supports:
    - ``"quantile"``: clips each channel at its foreground ``clip_quantile`` and scales
      into ``[0, 1]``. Quantiles are estimated on positive pixels (``x > 0``) when
      sparse markers have near-zero whole-FOV percentiles.
    - ``"anscombe"``: variance-stabilizing ``2 * sqrt(max(x, 0) + 3/8)`` transform.
    - ``"none"``: raw ``float32`` intensities without rescaling.
    """

    mode: str = "quantile"
    clip_quantile: float = 0.99
    scales: Optional[np.ndarray] = None

    def fit(
        self,
        images: Sequence[np.ndarray],
        *,
        channels_first: bool = False,
        max_samples_per_fov: int = 250_000,
        seed: int = 42,
    ) -> "ChannelNormalizer":
        """Estimate per-channel normalization scale factors from training images."""
        if not images:
            raise ValueError("Cannot fit ChannelNormalizer on an empty image sequence.")
        first = images[0]
        n_channels = int(first.shape[0] if channels_first else first.shape[-1])
        if self.mode != "quantile":
            self.scales = np.ones((n_channels,), dtype=np.float32)
            return self

        rng = np.random.default_rng(seed)
        per_channel_samples: List[List[np.ndarray]] = [[] for _ in range(n_channels)]
        for img in images:
            hwc = np.moveaxis(img, 0, -1) if channels_first else img
            if hwc.shape[-1] != n_channels:
                raise ValueError(
                    f"Channel count mismatch: expected {n_channels}, got {hwc.shape[-1]}."
                )
            flat = hwc.reshape(-1, n_channels)
            if len(flat) > max_samples_per_fov:
                idx = rng.choice(len(flat), size=max_samples_per_fov, replace=False)
                flat = flat[idx]
            for c in range(n_channels):
                per_channel_samples[c].append(flat[:, c].astype(np.float32, copy=False))

        scales = np.ones((n_channels,), dtype=np.float32)
        for c in range(n_channels):
            vals = np.concatenate(per_channel_samples[c], axis=0)
            pos = vals[vals > 0.0]
            if pos.size > 0:
                q_val = float(np.quantile(pos, self.clip_quantile))
                if q_val <= 0.0:
                    q_val = float(np.max(pos))
                scales[c] = q_val if q_val > 0.0 else 1.0
            else:
                max_val = float(np.max(vals)) if vals.size > 0 else 1.0
                scales[c] = max_val if max_val > 0.0 else 1.0
        self.scales = scales
        return self

    def transform(self, arr: np.ndarray, *, channels_first: bool = False) -> np.ndarray:
        """Apply fitted channel normalization to an image or crop array."""
        out = arr.astype(np.float32, copy=True)
        if self.mode == "none":
            return out
        if self.mode == "anscombe":
            np.maximum(out, 0.0, out=out)
            out += 0.375
            np.sqrt(out, out=out)
            out *= 2.0
            return out
        if self.scales is None:
            raise RuntimeError("ChannelNormalizer must be fitted before transform().")
        scales = self.scales.astype(np.float32, copy=False)
        if channels_first:
            shape = (scales.shape[0],) + (1,) * (out.ndim - 1)
        else:
            shape = (1,) * (out.ndim - 1) + (scales.shape[0],)
        s_view = scales.reshape(shape)
        np.clip(out, 0.0, s_view, out=out)
        out /= np.maximum(s_view, 1e-8)
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Serialize normalizer state to a plain dictionary."""
        return {
            "mode": self.mode,
            "clip_quantile": float(self.clip_quantile),
            "scales": self.scales.tolist() if self.scales is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChannelNormalizer":
        """Reconstruct a ``ChannelNormalizer`` from a serialized dictionary."""
        raw_scales = data.get("scales")
        scales = np.asarray(raw_scales, dtype=np.float32) if raw_scales is not None else None
        return cls(
            mode=str(data.get("mode", "quantile")),
            clip_quantile=float(data.get("clip_quantile", 0.99)),
            scales=scales,
        )


def extend_slices_1d(slc: slice, crop_size: int, max_size: int) -> slice:
    """Extend a 1D bounding slice symmetrically to exact length ``crop_size``."""
    span = slc.stop - slc.start
    delta = crop_size - span
    start = slc.start - (delta // 2)
    stop = slc.stop + (delta + 1) // 2
    if start < 0 or stop > max_size:
        raise ValueError(
            f"Cell crop slice [{start}, {stop}) out of bounds [0, {max_size})."
        )
    return slice(start, stop)


def create_crop_slices(
    slices: Tuple[slice, slice],
    crop_size: Tuple[int, int],
    bounds: Tuple[int, int],
) -> Tuple[slice, slice]:
    """Create 2D ``(slice_y, slice_x)`` bounding a cell with exact ``crop_size``."""
    return (
        extend_slices_1d(slices[0], crop_size[0], bounds[0]),
        extend_slices_1d(slices[1], crop_size[1], bounds[1]),
    )


@dataclass
class FOVData:
    """In-memory FOV container holding ``(H, W, C)`` image and ``(H, W)`` instance mask."""

    image_id: str
    image_hwc: np.ndarray
    mask_hw: np.ndarray
    is_count_data: bool = False


@dataclass(frozen=True)
class CellCropRecord:
    """Lightweight pointer to a single cell crop within an ``FOVData`` object."""

    cell_id: int
    image_id: str
    label: int
    slice_y: slice
    slice_x: slice
    fov: FOVData


def prepare_fov_data(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    image_id: str = "fov_0",
    channels_first: bool = True,
    channel_indices: Optional[Sequence[int]] = None,
    crop_size: int = 128,
    to_pad: bool = True,
) -> FOVData:
    """Validate, orient to ``(H, W, C)``, optionally pad, and wrap an image + mask pair."""
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {image.shape}.")
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D integer mask array, got shape {mask.shape}.")

    hwc = np.moveaxis(image, 0, -1) if channels_first else image
    if hwc.shape[:2] != mask.shape:
        raise ValueError(
            f"Image spatial shape {hwc.shape[:2]} does not match mask shape {mask.shape}."
        )
    if channel_indices is not None:
        hwc = hwc[..., list(channel_indices)]

    is_count = _is_count_like(hwc)
    hwc = hwc.astype(np.float32, copy=False)
    cells = mask.astype(np.int64, copy=False)

    if to_pad:
        pad = (crop_size + 1) // 2
        hwc = np.pad(hwc, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=0)
        cells = np.pad(cells, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

    return FOVData(
        image_id=str(image_id),
        image_hwc=hwc,
        mask_hw=cells,
        is_count_data=is_count,
    )


def _resolve_cell_label(
    cell_id: int,
    labels: Optional[Union[np.ndarray, Mapping[int, int], pd.Series]],
    ignore_label: int,
) -> int:
    if labels is None:
        return ignore_label
    if isinstance(labels, np.ndarray):
        if cell_id < 0 or cell_id >= len(labels):
            return ignore_label
        return int(labels[cell_id])
    if isinstance(labels, pd.Series):
        if cell_id in labels.index:
            val = labels.loc[cell_id]
            return int(val) if not pd.isna(val) else ignore_label
        return ignore_label
    if isinstance(labels, Mapping):
        if cell_id in labels:
            return int(labels[cell_id])
        return ignore_label
    raise TypeError(f"Unsupported labels container type: {type(labels).__name__}")


def extract_crops_from_fov(
    fov: FOVData,
    crop_size: int,
    *,
    labels: Optional[Union[np.ndarray, Mapping[int, int], pd.Series]] = None,
    object_ids: Optional[Sequence[int]] = None,
    drop_unlabeled: bool = False,
    ignore_label: int = -1,
) -> List[CellCropRecord]:
    """Extract ``CellCropRecord`` entries for segmented cells in ``fov``."""
    objs = ndimage.find_objects(fov.mask_hw)
    allowed_ids = {int(cid) for cid in object_ids} if object_ids is not None else None
    records: List[CellCropRecord] = []

    for cell_id, obj in enumerate(objs, start=1):
        if obj is None:
            continue
        if allowed_ids is not None and cell_id not in allowed_ids:
            continue
        label = _resolve_cell_label(cell_id, labels, ignore_label)
        if drop_unlabeled and label < 0:
            continue
        try:
            sy, sx = create_crop_slices(obj, (crop_size, crop_size), fov.mask_hw.shape)
        except ValueError:
            continue
        records.append(
            CellCropRecord(
                cell_id=int(cell_id),
                image_id=fov.image_id,
                label=int(label),
                slice_y=sy,
                slice_x=sx,
                fov=fov,
            )
        )
    return records


def _find_fov_file(directory: Path, image_id: str, extensions: Tuple[str, ...]) -> Path:
    for ext in extensions:
        candidate = directory / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No file matching '{image_id}' with extensions {extensions} in {directory}."
    )


def load_directory_fovs(
    root_dir: PathLike,
    image_names: Sequence[str],
    *,
    channels_path: Optional[PathLike] = None,
    blacklist_channels: Optional[Sequence[str]] = None,
    crop_size: int = 128,
    to_pad: bool = True,
    load_labels: bool = True,
    drop_unlabeled: bool = True,
    ignore_label: int = -1,
) -> Tuple[List[FOVData], List[CellCropRecord], List[str]]:
    """Load CellSighter ``CellTypes`` directory FOVs, masks, and crop records.

    Supports both ``<root>/CellTypes/{data/images,cells,cells2labels}`` and
    direct ``<root>/{data/images,cells,cells2labels}`` layouts.
    """
    root = Path(root_dir)
    base_dir = root / "CellTypes" if (root / "CellTypes").is_dir() else root
    images_dir = base_dir / "data" / "images"
    cells_dir = base_dir / "cells"
    labels_dir = base_dir / "cells2labels"

    channel_indices: Optional[List[int]] = None
    channel_names: List[str] = []
    if channels_path is not None and Path(channels_path).exists():
        all_channels = read_channels(channels_path)
        filtered = filter_channels(all_channels, blacklist_channels)
        channel_indices = [idx for idx, _ in filtered]
        channel_names = [name for _, name in filtered]

    fovs: List[FOVData] = []
    crops: List[CellCropRecord] = []
    for image_id in image_names:
        img_path = _find_fov_file(images_dir, image_id, (".npz", ".tiff", ".tif"))
        cells_path = _find_fov_file(cells_dir, image_id, (".npz", ".tiff", ".tif"))
        img = load_array_file(img_path)
        mask = load_array_file(cells_path)

        labels_arr: Optional[np.ndarray] = None
        if load_labels and labels_dir.is_dir():
            lbl_path = _find_fov_file(labels_dir, image_id, (".npz", ".txt"))
            labels_arr = load_labels_file(lbl_path)

        fov = prepare_fov_data(
            img,
            mask,
            image_id=str(image_id),
            channels_first=False,
            channel_indices=channel_indices,
            crop_size=crop_size,
            to_pad=to_pad,
        )
        fovs.append(fov)
        crops.extend(
            extract_crops_from_fov(
                fov,
                crop_size,
                labels=labels_arr,
                drop_unlabeled=drop_unlabeled and labels_arr is not None,
                ignore_label=ignore_label,
            )
        )

    if not channel_names and fovs:
        n_ch = fovs[0].image_hwc.shape[-1]
        channel_names = [f"ch_{i}" for i in range(n_ch)]
    return fovs, crops, channel_names


def _poisson_resample_crop(img_hwc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply CellSighter Gaussian-blur + Poisson resampling noise augmentation."""
    if _HAS_CV2:
        blur = cv2.GaussianBlur(img_hwc, (5, 5), 0)
    else:
        blur = ndimage.gaussian_filter(img_hwc, sigma=(1.0, 1.0, 0.0))
    lam = np.clip(blur, 0.0, None)
    return rng.poisson(lam=lam).astype(np.float32)


def _dilate_binary_mask(mask_hw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly dilate a 2D binary cell mask with kernel size in {2, 3, 5}."""
    if rng.random() >= 0.5:
        return mask_hw
    k = int(rng.choice((2, 3, 5)))
    if _HAS_CV2:
        kernel = np.ones((k, k), dtype=np.uint8)
        return cv2.dilate(mask_hw, kernel, iterations=1)
    structure = np.ones((k, k), dtype=bool)
    return ndimage.binary_dilation(mask_hw > 0.5, structure=structure).astype(np.float32)


def _center_crop_tensor(tensor: "torch.Tensor", size: int) -> "torch.Tensor":
    """Center-crop a ``(C, H, W)`` tensor to ``(C, size, size)``."""
    _, h, w = tensor.shape
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return tensor[:, top : top + size, left : left + size]


class ShiftAugmentation:
    """Channel-independent spatial jitter augmentation matching ``data/shift_augmentation.py``."""

    def __init__(self, output_size: int, shift_max: int = 5, p: float = 0.3) -> None:
        self.output_size = int(output_size)
        self.shift_max = int(shift_max)
        self.p = float(p)

    def __call__(
        self,
        tensor: "torch.Tensor",
        rng: Optional[np.random.Generator] = None,
    ) -> "torch.Tensor":
        if self.shift_max <= 0:
            return _center_crop_tensor(tensor, self.output_size)
        gen = rng if rng is not None else np.random.default_rng()
        c, h, w = tensor.shape
        out = torch.empty((c, self.output_size, self.output_size), dtype=tensor.dtype)
        max_h = min(h, self.output_size + self.shift_max)
        max_w = min(w, self.output_size + self.shift_max)
        base_top = max(0, (h - max_h) // 2)
        base_left = max(0, (w - max_w) // 2)
        slack_y = max(0, max_h - self.output_size)
        slack_x = max(0, max_w - self.output_size)
        center_top = max(0, (h - self.output_size) // 2)
        center_left = max(0, (w - self.output_size) // 2)

        for ch in range(c):
            if gen.random() < self.p and (slack_y > 0 or slack_x > 0):
                dy = int(gen.integers(0, slack_y + 1)) if slack_y > 0 else 0
                dx = int(gen.integers(0, slack_x + 1)) if slack_x > 0 else 0
                top = base_top + dy
                left = base_left + dx
            else:
                top = center_top
                left = center_left
            out[ch] = tensor[ch, top : top + self.output_size, left : left + self.output_size]
        return out


class CellCropsDataset(Dataset):
    """PyTorch Dataset yielding ``(C+2, crop_input_size, crop_input_size)`` cell crops."""

    def __init__(
        self,
        crops: Sequence[CellCropRecord],
        *,
        crop_input_size: int = 60,
        normalizer: Optional[ChannelNormalizer] = None,
        augment: bool = False,
        poisson_aug: Union[bool, str] = "auto",
        shift_max: int = 5,
        seed: int = 42,
    ) -> None:
        _require_torch()
        super().__init__()
        self.crops = list(crops)
        self.crop_input_size = int(crop_input_size)
        self.normalizer = normalizer
        self.augment = bool(augment)
        self.poisson_aug = poisson_aug
        self.shift_aug = ShiftAugmentation(self.crop_input_size, shift_max=shift_max)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.crops)

    def _use_poisson(self, fov: FOVData) -> bool:
        if isinstance(self.poisson_aug, bool):
            return self.poisson_aug
        return fov.is_count_data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.crops[idx]
        fov = rec.fov
        sy, sx = rec.slice_y, rec.slice_x
        crop_h = sy.stop - sy.start
        crop_w = sx.stop - sx.start

        if not self.augment:
            top = max(0, (crop_h - self.crop_input_size) // 2)
            left = max(0, (crop_w - self.crop_input_size) // 2)
            sub_y = slice(sy.start + top, sy.start + top + self.crop_input_size)
            sub_x = slice(sx.start + left, sx.start + left + self.crop_input_size)
            img_crop = fov.image_hwc[sub_y, sub_x, :].astype(np.float32, copy=True)
            mask_slice = fov.mask_hw[sub_y, sub_x]
            if self.normalizer is not None:
                img_crop = self.normalizer.transform(img_crop, channels_first=False)
            n_ch = img_crop.shape[-1]
            arr = np.empty((n_ch + 2, img_crop.shape[0], img_crop.shape[1]), dtype=np.float32)
            arr[:-2] = img_crop.transpose(2, 0, 1)
            arr[-2] = (mask_slice > 0).astype(np.float32)
            arr[-1] = (mask_slice == rec.cell_id).astype(np.float32)
            tensor = torch.from_numpy(arr)
            return {
                "image": tensor[:-1],
                "mask": tensor[[-1]],
                "label": int(rec.label),
                "cell_id": int(rec.cell_id),
                "image_id": rec.image_id,
            }

        # Restrict outer extraction window to the minimal square inscribed by full rotation.
        work_size = min(
            crop_h,
            crop_w,
            int(np.ceil(np.sqrt(2.0) * (self.crop_input_size + self.shift_aug.shift_max))) + 2,
        )
        top = max(0, (crop_h - work_size) // 2)
        left = max(0, (crop_w - work_size) // 2)
        sub_y = slice(sy.start + top, sy.start + top + work_size)
        sub_x = slice(sx.start + left, sx.start + left + work_size)

        img_crop = fov.image_hwc[sub_y, sub_x, :].astype(np.float32, copy=True)
        mask_slice = fov.mask_hw[sub_y, sub_x]
        target_mask = (mask_slice == rec.cell_id).astype(np.float32)
        all_cells_mask = (mask_slice > 0).astype(np.float32)

        if self._use_poisson(fov):
            img_crop = _poisson_resample_crop(img_crop, self.rng)

        if self.normalizer is not None:
            img_crop = self.normalizer.transform(img_crop, channels_first=False)

        target_mask = _dilate_binary_mask(target_mask, self.rng)
        all_cells_mask = _dilate_binary_mask(all_cells_mask, self.rng)

        n_ch = img_crop.shape[-1]
        arr = np.empty((n_ch + 2, work_size, work_size), dtype=np.float32)
        arr[:-2] = img_crop.transpose(2, 0, 1)
        arr[-2] = all_cells_mask
        arr[-1] = target_mask
        tensor = torch.from_numpy(arr)

        angle = float(self.rng.uniform(0.0, 360.0))
        tensor = TF.rotate(tensor, angle)
        if self.rng.random() < 0.5:
            tensor = self.shift_aug(tensor, self.rng)
        else:
            tensor = _center_crop_tensor(tensor, self.crop_input_size)
        if self.rng.random() < 0.75:
            tensor = TF.hflip(tensor)
        if self.rng.random() < 0.75:
            tensor = TF.vflip(tensor)

        return {
            "image": tensor[:-1],
            "mask": tensor[[-1]],
            "label": int(rec.label),
            "cell_id": int(rec.cell_id),
            "image_id": rec.image_id,
        }


def _build_resnet_backbone(backbone: str, num_classes: int) -> "nn.Module":
    builders = {
        "resnet50": tv_models.resnet50,
        "resnet34": tv_models.resnet34,
        "resnet18": tv_models.resnet18,
    }
    if backbone not in builders:
        raise ValueError(f"Unsupported backbone '{backbone}'.")
    return builders[backbone](weights=None, num_classes=num_classes)


class CellSighterNet(nn.Module):  # type: ignore[misc]
    """ResNet classifier with ``C + 2`` input channels matching ``upstream_repo/model.py``."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        backbone: str = "resnet50",
    ) -> None:
        _require_torch()
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.backbone = backbone
        self.model = _build_resnet_backbone(backbone, num_classes=self.num_classes)
        self.model.conv1 = nn.Conv2d(
            self.in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        nn.init.kaiming_normal_(self.model.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.softmax = nn.Softmax(dim=1)

    def forward_logits(self, x: "torch.Tensor") -> "torch.Tensor":
        """Compute raw unnormalized logits for ``x`` shaped ``(B, C+2, H, W)``."""
        return self.model(x)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return logits in train mode and softmax probabilities in eval mode."""
        logits = self.model(x)
        if not self.training:
            return self.softmax(logits)
        return logits


def compute_sampling_weights(
    labels: Sequence[int],
    *,
    hierarchy_match: Optional[Mapping[str, str]] = None,
    class_names: Optional[Sequence[str]] = None,
    power: float = 1.0,
) -> np.ndarray:
    """Compute per-sample weights for class-balanced or hierarchical batch sampling.

    When ``hierarchy_match`` is supplied, each parent lineage receives equal total
    sampling mass and fine subtypes within each lineage share that mass equally.
    """
    arr = np.asarray(labels, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0,), dtype=np.float64)

    unique_lbls, counts = np.unique(arr, return_counts=True)
    count_map = {int(k): int(v) for k, v in zip(unique_lbls, counts)}

    if not hierarchy_match:
        class_w = {
            k: 1.0 / (float(cnt) ** float(power)) for k, cnt in count_map.items()
        }
        return np.asarray([class_w[int(y)] for y in arr], dtype=np.float64)

    # Resolve parent lineage for every observed integer class label.
    str_hier = {str(k): str(v) for k, v in hierarchy_match.items()}
    lbl_to_group: Dict[int, str] = {}
    for k in count_map:
        if str(k) in str_hier:
            lbl_to_group[k] = str_hier[str(k)]
        elif class_names is not None and 0 <= k < len(class_names) and class_names[k] in str_hier:
            lbl_to_group[k] = str_hier[class_names[k]]
        else:
            lbl_to_group[k] = f"__group_{k}"

    group_to_classes: Dict[str, List[int]] = {}
    for k, grp in lbl_to_group.items():
        group_to_classes.setdefault(grp, []).append(k)

    n_groups = max(1, len(group_to_classes))
    class_w = {}
    for grp, members in group_to_classes.items():
        n_members = max(1, len(members))
        for k in members:
            denom = float(n_groups) * float(n_members) * (float(count_map[k]) ** float(power))
            class_w[k] = 1.0 / max(denom, 1e-12)

    return np.asarray([class_w[int(y)] for y in arr], dtype=np.float64)


def ensemble_predict_probabilities(
    models: Sequence["CellSighterNet"],
    loader: "DataLoader",
    device: "torch.device",
) -> np.ndarray:
    """Average predicted class probabilities across an ensemble of trained models."""
    _require_torch()
    if not models:
        raise ValueError("Cannot run ensemble inference with zero models.")

    prob_sum: Optional[np.ndarray] = None
    for model in models:
        model.eval()
        model.to(device)
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                x = torch.cat([batch["image"], batch["mask"]], dim=1).to(device)
                probs = model(x)
                chunks.append(probs.detach().cpu().numpy())
        model_probs = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float32)
        if prob_sum is None:
            prob_sum = model_probs.astype(np.float64)
        else:
            prob_sum += model_probs.astype(np.float64)

    assert prob_sum is not None
    mean_probs = prob_sum / float(len(models))
    row_sums = mean_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    return mean_probs / row_sums


class CellSighterTrainer:
    """End-to-end trainer and ensemble inference engine for CellSighter."""

    def __init__(self, config: Optional[CellSighterConfig] = None) -> None:
        _require_torch()
        self.config = config or CellSighterConfig()
        self.models: List[CellSighterNet] = []
        self.normalizer: Optional[ChannelNormalizer] = None
        self.class_names: List[str] = []
        self.channel_names: List[str] = []
        self.label_to_index: Dict[int, int] = {}
        self.index_to_label: Dict[int, int] = {}
        self.in_channels: int = 0
        self.history: List[Dict[str, Any]] = []

    def _remap_crop_labels(
        self,
        crops: Sequence[CellCropRecord],
        class_names: Optional[Union[Sequence[str], Mapping[int, str]]] = None,
    ) -> Tuple[List[CellCropRecord], Dict[int, int], Dict[int, int], List[str]]:
        raw_labels = sorted({int(c.label) for c in crops if c.label >= 0})
        if not raw_labels:
            raise ValueError("No labeled cells (label >= 0) found for training.")

        if isinstance(class_names, Mapping):
            all_ids = sorted(set(raw_labels) | {int(k) for k in class_names})
            lbl_to_idx = {raw_id: idx for idx, raw_id in enumerate(all_ids)}
            names = [str(class_names.get(raw_id, f"class_{raw_id}")) for raw_id in all_ids]
        elif class_names is not None and len(class_names) > 0:
            names = [str(c) for c in class_names]
            if max(raw_labels) >= len(names):
                raise ValueError(
                    f"Label {max(raw_labels)} exceeds class_names length {len(names)}."
                )
            lbl_to_idx = {idx: idx for idx in range(len(names))}
        else:
            lbl_to_idx = {raw_id: idx for idx, raw_id in enumerate(raw_labels)}
            names = [f"class_{raw_id}" for raw_id in raw_labels]

        idx_to_lbl = {idx: raw_id for raw_id, idx in lbl_to_idx.items()}
        remapped = [
            CellCropRecord(
                cell_id=c.cell_id,
                image_id=c.image_id,
                label=lbl_to_idx[int(c.label)],
                slice_y=c.slice_y,
                slice_x=c.slice_x,
                fov=c.fov,
            )
            for c in crops
            if c.label in lbl_to_idx
        ]
        return remapped, lbl_to_idx, idx_to_lbl, names

    def _split_train_val(
        self,
        crops: Sequence[CellCropRecord],
        seed: int,
    ) -> Tuple[List[CellCropRecord], List[CellCropRecord]]:
        if self.config.val_fraction <= 0.0 or len(crops) < 10:
            return list(crops), []
        rng = np.random.default_rng(seed)
        indices = np.arange(len(crops))
        labels = np.asarray([c.label for c in crops], dtype=np.int64)
        val_idx: List[int] = []
        train_idx: List[int] = []
        for cls in np.unique(labels):
            cls_pos = indices[labels == cls]
            rng.shuffle(cls_pos)
            n_val = max(1, int(round(len(cls_pos) * self.config.val_fraction))) if len(cls_pos) >= 3 else 0
            val_idx.extend(cls_pos[:n_val].tolist())
            train_idx.extend(cls_pos[n_val:].tolist())
        return [crops[i] for i in train_idx], [crops[i] for i in val_idx]

    def _evaluate_model(
        self,
        model: CellSighterNet,
        crops: Sequence[CellCropRecord],
        device: "torch.device",
    ) -> Tuple[float, float]:
        ds = CellCropsDataset(
            crops,
            crop_input_size=self.config.crop_input_size,
            normalizer=self.normalizer,
            augment=False,
        )
        loader = DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        probs = ensemble_predict_probabilities([model], loader, device)
        preds = probs.argmax(axis=1)
        y_true = np.asarray([c.label for c in crops], dtype=np.int64)
        acc = float(np.mean(preds == y_true)) if len(y_true) > 0 else 0.0
        # Balanced accuracy across present classes protects rare subtypes during checkpoint selection.
        recalls = [
            float(np.mean(preds[y_true == cls] == cls))
            for cls in np.unique(y_true)
        ]
        bal_acc = float(np.mean(recalls)) if recalls else acc
        return acc, bal_acc

    def fit(
        self,
        train_crops: Sequence[CellCropRecord],
        *,
        val_crops: Optional[Sequence[CellCropRecord]] = None,
        class_names: Optional[Union[Sequence[str], Mapping[int, str]]] = None,
        channel_names: Optional[Sequence[str]] = None,
        normalizer: Optional[ChannelNormalizer] = None,
    ) -> List[Dict[str, Any]]:
        """Train an ensemble of CellSighter ResNet models on extracted cell crops."""
        _require_torch()
        remapped_train, lbl_to_idx, idx_to_lbl, resolved_names = self._remap_crop_labels(
            train_crops, class_names=class_names
        )
        self.label_to_index = lbl_to_idx
        self.index_to_label = idx_to_lbl
        self.class_names = resolved_names
        num_classes = len(resolved_names)

        if val_crops is not None:
            remapped_val = [
                CellCropRecord(
                    cell_id=c.cell_id,
                    image_id=c.image_id,
                    label=lbl_to_idx[int(c.label)],
                    slice_y=c.slice_y,
                    slice_x=c.slice_x,
                    fov=c.fov,
                )
                for c in val_crops
                if int(c.label) in lbl_to_idx
            ]
        else:
            remapped_train, remapped_val = self._split_train_val(
                remapped_train, seed=self.config.seed
            )

        unique_fovs: List[FOVData] = []
        seen_ids: set[int] = set()
        for c in remapped_train:
            key = id(c.fov)
            if key not in seen_ids:
                seen_ids.add(key)
                unique_fovs.append(c.fov)

        n_marker_channels = int(unique_fovs[0].image_hwc.shape[-1])
        self.in_channels = n_marker_channels + 2
        self.channel_names = (
            list(channel_names)
            if channel_names is not None
            else [f"ch_{i}" for i in range(n_marker_channels)]
        )

        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = ChannelNormalizer(
                mode=self.config.normalization,
                clip_quantile=self.config.clip_quantile,
            ).fit([f.image_hwc for f in unique_fovs], channels_first=False, seed=self.config.seed)

        device = resolve_device(self.config.device)
        self.models = []
        self.history = []

        # Remap hierarchy_match keys into contiguous class indices if provided.
        mapped_hierarchy: Optional[Dict[str, str]] = None
        if self.config.hierarchy_match:
            mapped_hierarchy = {}
            for raw_k, grp in self.config.hierarchy_match.items():
                if raw_k.lstrip("-").isdigit() and int(raw_k) in lbl_to_idx:
                    mapped_hierarchy[str(lbl_to_idx[int(raw_k)])] = str(grp)
                elif raw_k in resolved_names:
                    mapped_hierarchy[raw_k] = str(grp)

        train_labels = [c.label for c in remapped_train]
        weights = (
            compute_sampling_weights(
                train_labels,
                hierarchy_match=mapped_hierarchy,
                class_names=resolved_names,
                power=self.config.sampling_power,
            )
            if self.config.sample_batch
            else None
        )

        for member_idx in range(self.config.ensemble_size):
            member_seed = self.config.seed + member_idx
            set_seed(member_seed)

            ds = CellCropsDataset(
                remapped_train,
                crop_input_size=self.config.crop_input_size,
                normalizer=self.normalizer,
                augment=self.config.augment,
                poisson_aug=self.config.poisson_aug,
                shift_max=self.config.shift_max,
                seed=member_seed,
            )
            if weights is not None:
                n_samples = (
                    int(self.config.samples_per_epoch)
                    if self.config.samples_per_epoch is not None
                    else len(remapped_train)
                )
                sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(weights, dtype=torch.double),
                    num_samples=n_samples,
                    replacement=True,
                )
                loader = DataLoader(
                    ds,
                    batch_size=self.config.batch_size,
                    sampler=sampler,
                    num_workers=self.config.num_workers,
                    drop_last=False,
                )
            else:
                loader = DataLoader(
                    ds,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=self.config.num_workers,
                    drop_last=False,
                )

            model = CellSighterNet(
                in_channels=self.in_channels,
                num_classes=num_classes,
                backbone=self.config.backbone,
            ).to(device)
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
            criterion = nn.CrossEntropyLoss()

            best_score = -1.0
            best_state: Optional[Dict[str, Any]] = None
            member_epochs: List[Dict[str, float]] = []

            for epoch in range(self.config.epochs):
                t0 = time.time()
                model.train()
                loss_sum = 0.0
                correct = 0
                total = 0

                for batch in loader:
                    x = torch.cat([batch["image"], batch["mask"]], dim=1).to(device)
                    y = batch["label"].to(device=device, dtype=torch.long)
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

                    bsz = int(y.shape[0])
                    loss_sum += float(loss.item()) * bsz
                    correct += int((logits.argmax(dim=1) == y).sum().item())
                    total += bsz

                train_loss = loss_sum / max(1, total)
                train_acc = float(correct) / max(1, total)
                val_acc = train_acc
                val_bal = train_acc

                if remapped_val:
                    val_acc, val_bal = self._evaluate_model(model, remapped_val, device)
                    score = 0.5 * (val_acc + val_bal)
                    if score >= best_score:
                        best_score = score
                        best_state = {
                            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                        }

                epoch_rec = {
                    "epoch": float(epoch + 1),
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "val_bal_acc": val_bal,
                    "elapsed_s": time.time() - t0,
                }
                member_epochs.append(epoch_rec)
                if self.config.verbose > 0:
                    print(
                        f"  [ens {member_idx + 1}/{self.config.ensemble_size}] "
                        f"epoch {epoch + 1}/{self.config.epochs}: "
                        f"loss={train_loss:.4f} train_acc={train_acc:.4f} "
                        f"val_acc={val_acc:.4f} val_bal={val_bal:.4f} "
                        f"({epoch_rec['elapsed_s']:.1f}s)",
                        flush=True,
                    )

            if best_state is not None:
                model.load_state_dict(best_state)
            model.cpu().eval()
            self.models.append(model)
            self.history.append(
                {
                    "ensemble_member": member_idx,
                    "seed": member_seed,
                    "epochs": member_epochs,
                }
            )

        return self.history

    def predict_crops(
        self,
        crops: Sequence[CellCropRecord],
        *,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """Predict ensemble-averaged class probabilities ``(N, num_classes)`` for crops."""
        _require_torch()
        if not self.models:
            raise RuntimeError("CellSighterTrainer has no trained or loaded models.")
        if not crops:
            return np.empty((0, len(self.class_names)), dtype=np.float64)

        ds = CellCropsDataset(
            crops,
            crop_input_size=self.config.crop_input_size,
            normalizer=self.normalizer,
            augment=False,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size or self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        dev = resolve_device(device or self.config.device)
        return ensemble_predict_probabilities(self.models, loader, dev)

    def save(self, path: PathLike) -> Path:
        """Save trained ensemble weights, normalizer, config, and class metadata."""
        _require_torch()
        if not self.models:
            raise RuntimeError("Cannot save an unfitted CellSighterTrainer.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "in_channels": self.in_channels,
            "class_names": list(self.class_names),
            "channel_names": list(self.channel_names),
            "label_to_index": dict(self.label_to_index),
            "index_to_label": dict(self.index_to_label),
            "normalizer": self.normalizer.to_dict() if self.normalizer is not None else None,
            "models": [
                {k: v.detach().cpu() for k, v in m.state_dict().items()} for m in self.models
            ],
            "history": self.history,
        }
        torch.save(payload, out)
        return out

    @classmethod
    def load(cls, path: PathLike, *, device: Optional[str] = None) -> "CellSighterTrainer":
        """Load a saved ``CellSighterTrainer`` checkpoint bundle."""
        _require_torch()
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg_dict = copy.deepcopy(payload["config"])
        if "blacklist_channels" in cfg_dict and isinstance(cfg_dict["blacklist_channels"], list):
            cfg_dict["blacklist_channels"] = tuple(cfg_dict["blacklist_channels"])
        if device is not None:
            cfg_dict["device"] = device
        config = CellSighterConfig(**cfg_dict)
        trainer = cls(config=config)
        trainer.in_channels = int(payload["in_channels"])
        trainer.class_names = [str(c) for c in payload["class_names"]]
        trainer.channel_names = [str(c) for c in payload.get("channel_names", [])]
        trainer.label_to_index = {int(k): int(v) for k, v in payload.get("label_to_index", {}).items()}
        trainer.index_to_label = {int(k): int(v) for k, v in payload.get("index_to_label", {}).items()}
        norm_dict = payload.get("normalizer")
        trainer.normalizer = ChannelNormalizer.from_dict(norm_dict) if norm_dict else None
        trainer.history = list(payload.get("history", []))

        trainer.models = []
        num_classes = len(trainer.class_names)
        for state in payload["models"]:
            model = CellSighterNet(
                in_channels=trainer.in_channels,
                num_classes=num_classes,
                backbone=config.backbone,
            )
            model.load_state_dict(state)
            model.eval()
            trainer.models.append(model)
        return trainer
