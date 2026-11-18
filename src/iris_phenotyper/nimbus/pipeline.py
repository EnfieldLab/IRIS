"""
Nimbus Inference & Downstream Clustering Pipeline
=================================================
Faithful reproduction of the Nimbus Residual U-Net marker-positivity classifier
and downstream single-cell clustering pipeline:

Reference:
    Rumberger, J. L. et al. "Automated classification of cellular expression in
    multiplexed imaging data with Nimbus." Nature Methods 22, 2161-2170 (2025),
    preprint bioRxiv DOI: 10.1101/2024.06.02.597062.
    Upstream repository: https://github.com/angelolab/Nimbus-Inference

Core algorithmic stages:
1. Per-FOV foreground (> 0) quantile estimation (q = 0.999) averaged across
   reference FOVs and normalized to [0, 1] (`prepare_normalization_dict`).
2. 2-channel input construction: `[normalized_marker_image, binary_interior_mask]`
   where inner segmentation boundaries are zeroed out (`find_boundaries(..., mode="inner")`)
   to separate adjacent cells.
3. Magnification scaling to the pretrained 10x model reference scale, optional
   6-view test-time augmentation (4 rotations + 2 flips), and shift-equivariant
   valid-padded tile-and-stitch inference for large FOVs.
4. Per-cell probability aggregation via `regionprops_table` mean intensity over
   each `ObjectNumber` instance mask in [0, 1].
5. Downstream cell-type calling via marker-panel rules or two-stage prototype +
   hierarchical consensus clustering on z-scored/capped positivity vectors.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries
from skimage.transform import resize as skimage_resize
from skimage.util.shape import view_as_windows
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from iris_common.schema import COL_CELL_ID, COL_SAMPLE, COL_X, COL_Y
from iris_common.types import CellTypePanel

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO: str = "JLrumberger/Nimbus-Inference"
DEFAULT_HF_CHECKPOINT: str = "V1.pt"
_DEFAULT_CACHE_CANDIDATES: Tuple[Path, ...] = (
    Path("/Users/wwu/Documents/IRIS_Benchmarks/nimbus/checkpoints"),
    Path.home() / ".cache" / "iris" / "nimbus",
)

_UNET_CACHE: Optional[Dict[str, Any]] = None


def is_upstream_available() -> bool:
    """Return True if the ``nimbus_inference`` upstream package is importable."""
    try:
        import nimbus_inference  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Residual U-Net Architecture (exact parameter match to nimbus_inference.unet)
# ---------------------------------------------------------------------------


def _get_filter_count(layer_idx: int, filters_root: int) -> int:
    return (2**layer_idx) * filters_root


def _import_torch() -> Any:
    """Import PyTorch with macOS OpenMP collision guards enabled."""
    if sys.platform == "darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    if sys.platform == "darwin":
        torch.set_num_threads(1)
    return torch


def _get_unet_classes() -> Dict[str, Any]:
    """Lazily define and cache Nimbus Residual U-Net modules when PyTorch is needed."""
    global _UNET_CACHE
    if _UNET_CACHE is not None:
        return _UNET_CACHE

    torch = _import_torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class Pad2D(nn.Module):
        """2D spatial padding layer supporting ``reflect``, ``constant``, ``replicate``, ``valid``."""

        def __init__(self, padding: Tuple[int, int] = (1, 1), mode: str = "constant") -> None:
            super().__init__()
            if mode not in ("constant", "reflect", "replicate", "valid"):
                raise ValueError(
                    "Padding mode must be 'valid', 'constant', 'reflect', or 'replicate'"
                )
            self.padding = (padding[1], padding[1], padding[0], padding[0])
            self.mode = mode

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.mode == "valid":
                return x
            return F.pad(x, self.padding, mode=self.mode)

    def maybe_crop(
        x: torch.Tensor,
        target_shape: Sequence[int],
        data_format: str = "channels_first",
    ) -> torch.Tensor:
        """Center-crop ``x`` to match ``target_shape`` spatial dimensions."""
        if data_format == "channels_last":
            tgt_h, tgt_w = target_shape[1], target_shape[2]
            x_h, x_w = x.size(1), x.size(2)
            h_diff = x_h - tgt_h
            w_diff = x_w - tgt_w
            h0, w0 = h_diff // 2, w_diff // 2
            return x[:, h0 : h0 + tgt_h, w0 : w0 + tgt_w, :]
        tgt_h, tgt_w = target_shape[2], target_shape[3]
        x_h, x_w = x.size(2), x.size(3)
        h_diff = x_h - tgt_h
        w_diff = x_w - tgt_w
        h0, w0 = h_diff // 2, w_diff // 2
        return x[:, :, h0 : h0 + tgt_h, w0 : w0 + tgt_w]

    class ConvBlock(nn.Module):
        """Residual 2-conv block with 1x1 projection skip connection and BatchNorm2d."""

        def __init__(
            self,
            layer_idx: int,
            filters_root: int,
            kernel_size: int,
            padding: str,
            activation: str,
            up: bool = False,
            in_channels: int = 2,
        ) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self.filters_root = filters_root
            self.kernel_size = kernel_size
            self.padding = padding
            self.activation = getattr(nn, activation)()

            filters = _get_filter_count(layer_idx, filters_root)
            self.padding_layer = Pad2D(padding=(1, 1), mode=self.padding)

            if up:
                in_ch = _get_filter_count(layer_idx + 1, filters_root)
            else:
                in_ch = (
                    _get_filter_count(layer_idx - 1, filters_root)
                    if layer_idx != 0
                    else in_channels
                )

            self.conv2d_0 = nn.Conv2d(
                in_channels=in_ch,
                out_channels=filters,
                kernel_size=(1, 1),
                stride=1,
                padding="valid",
            )
            self.conv2d_1 = nn.Conv2d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=(kernel_size, kernel_size),
                stride=1,
                padding="valid",
            )
            self.bn_1 = nn.BatchNorm2d(filters)
            self.conv2d_2 = nn.Conv2d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=(kernel_size, kernel_size),
                stride=1,
                padding="valid",
            )
            self.bn_2 = nn.BatchNorm2d(filters)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            skip = self.conv2d_0(x)
            x = self.padding_layer(skip)
            x = self.conv2d_1(x)
            x = self.bn_1(x)
            x = self.activation(x)
            x = self.padding_layer(x)
            x = self.conv2d_2(x)
            x = self.bn_2(x)
            x = self.activation(x)
            if self.padding == "valid":
                skip = maybe_crop(skip, x.size())
            return x + skip

    class UpconvBlock(nn.Module):
        """Transposed-convolution upsampling block."""

        def __init__(
            self,
            layer_idx: int,
            filters_root: int,
            kernel_size: int,
            pool_size: Tuple[int, int],
            padding: str,
            activation: str,
        ) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self.filters_root = filters_root
            self.kernel_size = kernel_size
            self.pool_size = pool_size
            self.padding = padding

            filters = _get_filter_count(layer_idx, filters_root)
            self.padding_layer = Pad2D(padding=(1, 1), mode=self.padding)
            self.upconv = nn.ConvTranspose2d(
                in_channels=filters,
                out_channels=filters // 2,
                kernel_size=pool_size,
                stride=pool_size,
                padding=0,
            )
            self.activation_1 = getattr(nn, activation)()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.activation_1(self.upconv(x))

    class CropConcatBlock(nn.Module):
        """Center-crops contracting-path feature map and concatenates with decoder features."""

        def forward(self, x: torch.Tensor, down_layer: torch.Tensor) -> torch.Tensor:
            x1_shape = down_layer.shape
            x2_shape = x.shape
            height_diff = abs(x1_shape[2] - x2_shape[2]) // 2
            width_diff = abs(x1_shape[3] - x2_shape[3]) // 2
            down_cropped = down_layer[
                :,
                :,
                height_diff : (x2_shape[2] + height_diff),
                width_diff : (x2_shape[3] + width_diff),
            ]
            return torch.cat([down_cropped, x], dim=1)

    class NimbusUNet(nn.Module):
        """Residual U-Net matching ``nimbus_inference.unet.UNet`` state-dict structure."""

        def __init__(
            self,
            num_classes: int = 1,
            layer_depth: int = 5,
            filters_root: int = 64,
            data_format: str = "channels_first",
            kernel_size: int = 3,
            pool_size: int = 2,
            padding: str = "reflect",
            activation: str = "ReLU",
            in_channels: int = 2,
        ) -> None:
            super().__init__()
            self.layer_depth = layer_depth
            self.contracting_layers = nn.ModuleList()
            self.expanding_layers = nn.ModuleList()
            self.padding = padding

            for layer_idx in range(layer_depth - 1):
                conv_block = ConvBlock(
                    layer_idx=layer_idx,
                    filters_root=filters_root,
                    kernel_size=kernel_size,
                    padding=self.padding,
                    activation=activation,
                    in_channels=in_channels,
                )
                self.contracting_layers.append(conv_block)
                self.contracting_layers.append(nn.MaxPool2d(kernel_size=(pool_size, pool_size)))

            self.bottle_neck = ConvBlock(
                layer_idx=layer_depth - 1,
                filters_root=filters_root,
                kernel_size=kernel_size,
                padding=self.padding,
                activation=activation,
                in_channels=in_channels,
            )

            for layer_idx in range(layer_depth - 2, -1, -1):
                upconv_block = UpconvBlock(
                    layer_idx=layer_idx + 1,
                    filters_root=filters_root,
                    kernel_size=kernel_size,
                    padding=self.padding,
                    activation=activation,
                    pool_size=(pool_size, pool_size),
                )
                crop_concat_block = CropConcatBlock()
                conv_block = ConvBlock(
                    layer_idx=layer_idx,
                    filters_root=filters_root,
                    kernel_size=kernel_size,
                    padding=self.padding,
                    activation=activation,
                    up=True,
                    in_channels=in_channels,
                )
                self.expanding_layers.extend([upconv_block, crop_concat_block, conv_block])

            self.final_conv = nn.Conv2d(filters_root, num_classes, kernel_size=(1, 1))
            if data_format == "channels_last":
                self.to(memory_format=torch.channels_last)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            contracting_outputs: List[torch.Tensor] = []
            for layer in self.contracting_layers:
                x = layer(x)
                if isinstance(layer, ConvBlock):
                    contracting_outputs.append(x)
            x = self.bottle_neck(x)

            for layer in self.expanding_layers:
                if isinstance(layer, CropConcatBlock):
                    x = layer(x, contracting_outputs.pop(-1))
                else:
                    x = layer(x)
            return torch.sigmoid(self.final_conv(x))

    _UNET_CACHE = {
        "Pad2D": Pad2D,
        "ConvBlock": ConvBlock,
        "UpconvBlock": UpconvBlock,
        "CropConcatBlock": CropConcatBlock,
        "NimbusUNet": NimbusUNet,
        "UNet": NimbusUNet,
    }
    return _UNET_CACHE


def __getattr__(name: str) -> Any:
    if name in ("Pad2D", "ConvBlock", "UpconvBlock", "CropConcatBlock", "NimbusUNet", "UNet"):
        classes = _get_unet_classes()
        globals().update(classes)
        return classes[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Preprocessing, Binary Mask Construction & Normalization
# ---------------------------------------------------------------------------


def _handle_qupath_segmentation_map(instance_mask: np.ndarray) -> np.ndarray:
    """Decode QuPath 24-bit RGB segmentation maps into integer instance IDs."""
    mask = np.asarray(instance_mask)
    if mask.ndim == 3 and mask.shape[0] == 3 and mask.shape[-1] != 3:
        mask = np.moveaxis(mask, 0, -1)
    if mask.ndim == 3 and mask.shape[-1] == 3:
        decoded = (
            mask[..., 0].astype(np.int64) * (256**2)
            + mask[..., 1].astype(np.int64) * 256
            + mask[..., 2].astype(np.int64)
        )
        return decoded.astype(np.uint32)
    return np.squeeze(mask).astype(np.uint32)


def normalize_instance_mask(instance_mask: np.ndarray) -> np.ndarray:
    """Coerce an instance segmentation mask to 2D uint32 ``(H, W)``."""
    mask = np.squeeze(np.asarray(instance_mask))
    if mask.ndim == 3:
        mask = _handle_qupath_segmentation_map(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D instance mask (H, W), got shape {mask.shape}.")
    return mask.astype(np.uint32, copy=False)


def prepare_binary_mask(instance_mask: np.ndarray) -> np.ndarray:
    """Convert instance segmentation mask to Nimbus interior binary mask.

    Inner cell boundaries are set to 0 so adjacent segmented cells remain
    separated in the second U-Net input channel (`find_boundaries(..., mode="inner")`).
    """
    mask = normalize_instance_mask(instance_mask)
    edge = find_boundaries(mask, mode="inner").astype(np.uint8)
    return np.logical_and(edge == 0, mask > 0).astype(np.float32)


def prepare_input_data(
    marker_img: np.ndarray,
    instance_mask: Optional[np.ndarray] = None,
    binary_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Construct the 4D ``(1, 2, H, W)`` float32 input tensor expected by Nimbus."""
    if binary_mask is None:
        if instance_mask is None:
            raise ValueError("Either instance_mask or binary_mask must be provided.")
        binary_mask = prepare_binary_mask(instance_mask)
    img_f32 = np.asarray(marker_img, dtype=np.float32)
    bin_f32 = np.asarray(binary_mask, dtype=np.float32)
    if img_f32.shape != bin_f32.shape:
        raise ValueError(
            f"Marker image shape {img_f32.shape} does not match binary mask shape {bin_f32.shape}."
        )
    return np.stack([img_f32, bin_f32], axis=0)[np.newaxis, ...]


def compute_fov_channel_quantiles(
    channel_img: np.ndarray,
    quantile: float = 0.999,
) -> Optional[float]:
    """Compute foreground quantile over strictly positive pixels (``img > 0``)."""
    arr = np.asarray(channel_img, dtype=np.float32)
    if not np.any(arr > 0):
        return None
    foreground = arr[arr > 0]
    val = float(np.quantile(foreground, quantile))
    if not np.isfinite(val) or val <= 0.0:
        return None
    return val


def prepare_normalization_dict(
    fov_images: Sequence[Union[np.ndarray, Mapping[str, np.ndarray]]],
    channel_names: Optional[Sequence[str]] = None,
    quantile: float = 0.999,
    n_subset: Optional[int] = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute per-channel normalization factors across a collection of FOV images.

    Replicates ``nimbus_inference.utils.prepare_normalization_dict`` by taking
    the ``quantile`` (default 0.999) of positive pixels (``> 0``) per FOV and
    averaging across up to ``n_subset`` sampled FOVs.
    """
    if len(fov_images) == 0:
        raise ValueError("fov_images must contain at least one FOV.")

    indices = list(range(len(fov_images)))
    if n_subset is not None and n_subset < len(indices):
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=int(n_subset), replace=False).tolist()

    per_channel_vals: Dict[str, List[float]] = {}
    for idx in indices:
        item = fov_images[idx]
        if isinstance(item, Mapping):
            ch_iter = (
                [(c, item[c]) for c in channel_names if c in item]
                if channel_names is not None
                else list(item.items())
            )
        else:
            arr = np.asarray(item)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            if arr.ndim != 3:
                raise ValueError(f"Expected (C, H, W) or (H, W) array, got shape {arr.shape}.")
            names = (
                list(channel_names)
                if channel_names is not None
                else [f"Channel_{i}" for i in range(arr.shape[0])]
            )
            if len(names) != arr.shape[0]:
                raise ValueError(
                    f"channel_names length ({len(names)}) != array channels ({arr.shape[0]})."
                )
            ch_iter = [(names[i], arr[i]) for i in range(arr.shape[0])]

        for ch_name, ch_img in ch_iter:
            q_val = compute_fov_channel_quantiles(ch_img, quantile=quantile)
            per_channel_vals.setdefault(ch_name, [])
            if q_val is not None:
                per_channel_vals[ch_name].append(q_val)

    norm_dict: Dict[str, float] = {}
    for ch_name, vals in per_channel_vals.items():
        clean = [v for v in vals if v is not None and np.isfinite(v) and v > 0.0]
        norm_dict[ch_name] = float(np.mean(clean)) if clean else 1e-8
    return norm_dict


def normalize_channel_image(
    channel_img: np.ndarray,
    norm_factor: Optional[float] = None,
    quantile: float = 0.999,
    clip_values: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Normalize a 2D marker image by ``norm_factor`` and clip to ``clip_values``."""
    img = np.asarray(channel_img, dtype=np.float32)
    if norm_factor is None:
        q = compute_fov_channel_quantiles(img, quantile=quantile)
        factor = q if q is not None else 1.0
    else:
        factor = float(norm_factor)
    if not np.isfinite(factor) or factor <= 0.0:
        factor = 1e-8
    scaled = img / factor
    return np.clip(scaled, clip_values[0], clip_values[1]).astype(np.float32)


def _resize_2d(arr: np.ndarray, out_hw: Tuple[int, int], nearest: bool = False) -> np.ndarray:
    """Resize a 2D image using OpenCV (exact Nimbus parity) with skimage fallback."""
    arr_f32 = np.asarray(arr, dtype=np.float32)
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    try:
        import cv2

        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.resize(arr_f32, (out_w, out_h), interpolation=interp).astype(np.float32)
    except ImportError:
        order = 0 if nearest else 1
        resized = skimage_resize(
            arr_f32,
            output_shape=(out_h, out_w),
            order=order,
            mode="edge" if not nearest else "constant",
            anti_aliasing=False,
            preserve_range=True,
        )
        return resized.astype(np.float32)


# ---------------------------------------------------------------------------
# Checkpoint Discovery & Model Loading
# ---------------------------------------------------------------------------


def _select_device(device: str = "auto") -> "torch.device":
    torch = _import_torch()

    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unsupported device '{device}'. Choose 'auto', 'cpu', 'cuda', or 'mps'.")
    return torch.device(device)


def resolve_nimbus_checkpoint(
    checkpoint: Union[str, Path] = "latest",
    cache_dir: Optional[Union[str, Path]] = None,
    allow_download: bool = True,
) -> Path:
    """Locate or download the pretrained Nimbus U-Net checkpoint (``V1.pt``)."""
    ckpt_str = str(checkpoint)
    explicit_path = Path(ckpt_str).expanduser()
    if explicit_path.is_file():
        return explicit_path.resolve()

    search_dirs: List[Path] = []
    if cache_dir is not None:
        search_dirs.append(Path(cache_dir).expanduser())
    if is_upstream_available():
        import nimbus_inference

        search_dirs.append(Path(nimbus_inference.__file__).resolve().parent / "assets")
    search_dirs.extend(_DEFAULT_CACHE_CANDIDATES)

    version_pattern = re.compile(r"V(\d+)\.pt$")
    if ckpt_str == "latest":
        candidates: List[Tuple[int, Path]] = []
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            for entry in directory.iterdir():
                if entry.is_file():
                    m = version_pattern.search(entry.name)
                    if m:
                        candidates.append((int(m.group(1)), entry))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return candidates[0][1].resolve()
    else:
        for directory in search_dirs:
            cand = directory / ckpt_str
            if cand.is_file():
                return cand.resolve()

    if not allow_download:
        raise FileNotFoundError(
            f"Nimbus checkpoint '{ckpt_str}' not found locally and allow_download=False."
        )

    target_dir = (
        Path(cache_dir).expanduser() if cache_dir is not None else _DEFAULT_CACHE_CANDIDATES[1]
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = DEFAULT_HF_CHECKPOINT if ckpt_str == "latest" else ckpt_str

    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        if ckpt_str == "latest":
            repo_files = list_repo_files(DEFAULT_HF_REPO)
            versions = [
                (int(m.group(1)), f)
                for f in repo_files
                if (m := version_pattern.search(f)) is not None
            ]
            if versions:
                versions.sort(key=lambda item: item[0], reverse=True)
                filename = versions[0][1]
        downloaded = hf_hub_download(
            repo_id=DEFAULT_HF_REPO,
            filename=filename,
            local_dir=str(target_dir),
        )
        return Path(downloaded).resolve()
    except ImportError:
        import requests

        url = f"https://huggingface.co/{DEFAULT_HF_REPO}/resolve/main/{filename}"
        dest = target_dir / filename
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    handle.write(chunk)
        return dest.resolve()


def load_nimbus_model(
    checkpoint: Union[str, Path] = "latest",
    padding: str = "reflect",
    device: str = "auto",
    cache_dir: Optional[Union[str, Path]] = None,
    allow_download: bool = True,
) -> Tuple["nn.Module", "torch.device", Path]:
    """Instantiate ``NimbusUNet`` and load pretrained weights."""
    torch = _import_torch()

    unet_cls = _get_unet_classes()["NimbusUNet"]
    torch_device = _select_device(device)
    ckpt_path = resolve_nimbus_checkpoint(
        checkpoint=checkpoint,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )
    model = unet_cls(num_classes=1, padding=padding)
    try:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(torch_device).eval()
    return model, torch_device, ckpt_path


# ---------------------------------------------------------------------------
# Single-Shot & Tile-and-Stitch Inference with TTA
# ---------------------------------------------------------------------------


def _tile_input(
    image: np.ndarray,
    tile_size: Tuple[int, int],
    output_shape: Tuple[int, int],
    pad_mode: str = "reflect",
) -> Tuple[np.ndarray, List[int]]:
    """Pad and extract overlapping tiles for valid-padded U-Net inference."""
    in_h, in_w = image.shape[-2:]
    out_h, out_w = int(output_shape[0]), int(output_shape[1])
    tile_h, tile_w = int(tile_size[0]), int(tile_size[1])

    pad_h_total = tile_h - (in_h % out_h)
    pad_w_total = tile_w - (in_w % out_w)
    pad_h1, pad_w1 = pad_h_total // 2, pad_w_total // 2
    pad_h0, pad_w0 = pad_h_total - pad_h1, pad_w_total - pad_w1

    effective_mode = "edge" if (in_h < 2 or in_w < 2) and pad_mode == "reflect" else pad_mode
    padded = np.pad(
        image, ((0, 0), (0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1)), mode=effective_mode
    )
    b, c = padded.shape[:2]
    view = np.squeeze(
        view_as_windows(padded, (b, c, tile_h, tile_w), step=(b, c, out_h, out_w)),
        axis=(0, 1),
    )
    return view, [pad_h0, pad_h1, pad_w0, pad_w1]


def _stitch_tiles(tiles: np.ndarray, padding: Sequence[int]) -> np.ndarray:
    """Reassemble predicted tiles and strip border padding."""
    h_t, w_t, b, c, h, w = tiles.shape
    stitched = np.zeros((b, c, h_t * h, w_t * w), dtype=np.float32)
    for i in range(h_t):
        for j in range(w_t):
            for b_idx in range(b):
                stitched[b_idx, :, i * h : (i + 1) * h, j * w : (j + 1) * w] = tiles[i, j, b_idx]
    h_end = None if padding[1] == 0 else -int(padding[1])
    w_end = None if padding[3] == 0 else -int(padding[3])
    return stitched[:, :, int(padding[0]) : h_end, int(padding[2]) : w_end]


def _predict_single_view(
    input_data: np.ndarray,
    model_reflect: Optional["nn.Module"],
    model_valid: Optional["nn.Module"],
    device: "torch.device",
    input_shape: Tuple[int, int] = (1024, 1024),
    batch_size: int = 4,
    mixed_precision: bool = False,
) -> np.ndarray:
    """Predict pixel positivity map for one ``(1, 2, H, W)`` view."""
    torch = _import_torch()

    h, w = int(input_data.shape[-2]), int(input_data.shape[-1])
    if h <= input_shape[0] and w <= input_shape[1] and model_reflect is not None:
        tensor = torch.as_tensor(input_data, dtype=torch.float32, device=device)
        with torch.no_grad():
            if mixed_precision and device.type in ("cuda", "cpu"):
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    pred = model_reflect(tensor)
            else:
                pred = model_reflect(tensor)
        return pred.detach().cpu().numpy().astype(np.float32)

    if model_valid is None:
        raise ValueError("Valid-padded model is required for images larger than input_shape.")

    with torch.no_grad():
        dummy = torch.zeros((1, 2, int(input_shape[0]), int(input_shape[1])), device=device)
        raw_out_shape = np.array(model_valid(dummy).shape[-2:], dtype=int)

    crop_by = raw_out_shape % (2**5)
    out_shape = raw_out_shape - crop_by
    tiled, pad = _tile_input(input_data, input_shape, (int(out_shape[0]), int(out_shape[1])))
    shape_diff = (np.array(input_shape, dtype=int) - out_shape) // 2
    crop_pad = [
        int(pad[0] - shape_diff[0]),
        int(pad[1] - shape_diff[0]),
        int(pad[2] - shape_diff[1]),
        int(pad[3] - shape_diff[1]),
    ]

    h_t, w_t, b, c = tiled.shape[:4]
    flat_tiles = tiled.reshape(h_t * w_t * b, c, int(input_shape[0]), int(input_shape[1]))

    preds: List[np.ndarray] = []
    cy = int(crop_by[0])
    cx = int(crop_by[1])
    cy0, cy1 = cy // 2, cy - (cy // 2)
    cx0, cx1 = cx // 2, cx - (cx // 2)

    for start in range(0, len(flat_tiles), max(1, int(batch_size))):
        batch_np = flat_tiles[start : start + max(1, int(batch_size))]
        batch_t = torch.from_numpy(np.ascontiguousarray(batch_np)).float().to(device)
        with torch.no_grad():
            if mixed_precision and device.type in ("cuda", "cpu"):
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    out_np = model_valid(batch_t).detach().cpu().numpy()
            else:
                out_np = model_valid(batch_t).detach().cpu().numpy()
        if cy > 0 or cx > 0:
            h_end = out_np.shape[-2] - cy1
            w_end = out_np.shape[-1] - cx1
            out_np = out_np[..., cy0:h_end, cx0:w_end]
        preds.append(out_np.astype(np.float32))

    stacked = np.concatenate(preds, axis=0)
    reshaped = stacked.reshape(h_t, w_t, b, *stacked.shape[1:])
    return _stitch_tiles(reshaped, crop_pad)


def predict_channel_map(
    norm_marker_img: np.ndarray,
    binary_mask: np.ndarray,
    model_reflect: Optional["nn.Module"],
    model_valid: Optional["nn.Module"],
    device: "torch.device",
    *,
    dataset_magnification: float = 20.0,
    model_magnification: float = 10.0,
    input_shape: Tuple[int, int] = (1024, 1024),
    batch_size: int = 4,
    test_time_aug: bool = True,
    mixed_precision: bool = False,
) -> np.ndarray:
    """Predict a 2D pixel positivity probability map ``(H, W)`` in ``[0, 1]``."""
    torch = _import_torch()

    orig_h, orig_w = norm_marker_img.shape
    scale = float(model_magnification) / float(dataset_magnification)
    if not np.isclose(scale, 1.0):
        scaled_h = max(16, int(round(orig_h * scale)))
        scaled_w = max(16, int(round(orig_w * scale)))
        img_in = _resize_2d(norm_marker_img, (scaled_h, scaled_w), nearest=False)
        mask_in = _resize_2d(binary_mask, (scaled_h, scaled_w), nearest=True)
    else:
        img_in = norm_marker_img
        mask_in = binary_mask

    input_4d = prepare_input_data(img_in, binary_mask=mask_in)

    if not test_time_aug:
        pred = _predict_single_view(
            input_4d,
            model_reflect=model_reflect,
            model_valid=model_valid,
            device=device,
            input_shape=input_shape,
            batch_size=batch_size,
            mixed_precision=mixed_precision,
        )
        prob_map = np.squeeze(pred).astype(np.float32)
    else:
        tensor_in = torch.from_numpy(input_4d)
        aug_pairs: List[
            Tuple[Callable[["torch.Tensor"], "torch.Tensor"], Callable[["torch.Tensor"], "torch.Tensor"]]
        ] = [
            (
                lambda x, k=k: torch.rot90(x, k=k, dims=(2, 3)),
                lambda x, k=k: torch.rot90(x, k=-k, dims=(2, 3)),
            )
            for k in (0, 1, 2, 3)
        ] + [
            (lambda x: torch.flip(x, dims=(2,)), lambda x: torch.flip(x, dims=(2,))),
            (lambda x: torch.flip(x, dims=(3,)), lambda x: torch.flip(x, dims=(3,))),
        ]
        view_preds: List[np.ndarray] = []
        for fwd_fn, bwd_fn in aug_pairs:
            aug_np = fwd_fn(tensor_in).numpy()
            pred_aug = _predict_single_view(
                aug_np,
                model_reflect=model_reflect,
                model_valid=model_valid,
                device=device,
                input_shape=input_shape,
                batch_size=batch_size,
                mixed_precision=mixed_precision,
            )
            pred_restored = bwd_fn(torch.from_numpy(pred_aug)).numpy()
            view_preds.append(np.squeeze(pred_restored))
        prob_map = np.mean(np.stack(view_preds, axis=0), axis=0).astype(np.float32)

    if not np.isclose(scale, 1.0):
        prob_map = _resize_2d(prob_map, (orig_h, orig_w), nearest=True)

    return np.clip(prob_map, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Cell-Level Positivity Aggregation & Downstream Clustering
# ---------------------------------------------------------------------------


def aggregate_cell_positivity(
    instance_mask: np.ndarray,
    probability_map: np.ndarray,
) -> pd.DataFrame:
    """Aggregate pixel-level Nimbus probabilities to per-cell mean positivity scores."""
    mask = normalize_instance_mask(instance_mask)
    prob = np.asarray(probability_map, dtype=np.float32)
    if mask.shape != prob.shape:
        raise ValueError(
            f"Instance mask shape {mask.shape} != probability map shape {prob.shape}."
        )
    if not np.any(mask > 0):
        return pd.DataFrame(columns=[COL_CELL_ID, COL_X, COL_Y, "area", "intensity_mean"])

    props = regionprops_table(
        label_image=mask.astype(np.int32),
        intensity_image=prob,
        properties=["label", "centroid", "area", "intensity_mean"],
    )
    df = pd.DataFrame(
        {
            COL_CELL_ID: np.asarray(props["label"], dtype=np.int64),
            COL_X: np.asarray(props["centroid-1"], dtype=float),
            COL_Y: np.asarray(props["centroid-0"], dtype=float),
            "area": np.asarray(props["area"], dtype=float),
            "intensity_mean": np.clip(np.asarray(props["intensity_mean"], dtype=float), 0.0, 1.0),
        }
    )
    return df.sort_values(COL_CELL_ID, kind="mergesort").reset_index(drop=True)


def assign_cell_types_from_panel(
    positivity_df: pd.DataFrame,
    cell_type_panel: Union[CellTypePanel, Mapping[str, Sequence[str]]],
    positive_threshold: float = 0.5,
    unassigned_label: str = "Unassigned",
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Assign cell types and soft probabilities from Nimbus marker positivity scores."""
    if positivity_df.empty:
        empty_idx = positivity_df.index
        return (
            pd.Series([], index=empty_idx, dtype=object, name="cell_type"),
            pd.DataFrame(index=empty_idx, dtype=float),
            pd.Series([], index=empty_idx, dtype=float, name="confidence"),
        )

    rules: List[Tuple[str, List[str], List[str]]] = []
    if isinstance(cell_type_panel, CellTypePanel):
        for r in cell_type_panel.rules:
            rules.append((r.cell_type, list(r.markers_positive), list(r.markers_negative)))
    else:
        for ct_name, markers in cell_type_panel.items():
            rules.append((str(ct_name), list(markers), []))

    if not rules:
        raise ValueError("cell_type_panel contains no cell-type rules.")

    col_lookup = {str(c).strip().lower(): str(c) for c in positivity_df.columns}
    scores = np.zeros((len(positivity_df), len(rules)), dtype=float)
    pos_only_scores = np.zeros((len(positivity_df), len(rules)), dtype=float)
    ct_names = [r[0] for r in rules]

    for j, (_, pos_markers, neg_markers) in enumerate(rules):
        pos_cols = [
            col_lookup[m.strip().lower()] for m in pos_markers if m.strip().lower() in col_lookup
        ]
        neg_cols = [
            col_lookup[m.strip().lower()] for m in neg_markers if m.strip().lower() in col_lookup
        ]
        terms: List[np.ndarray] = []
        if pos_cols:
            pos_mat = positivity_df[pos_cols].to_numpy(dtype=float)
            pos_mean = np.mean(pos_mat, axis=1)
            pos_only_scores[:, j] = pos_mean
            terms.append(pos_mat)
        if neg_cols:
            neg_mat = 1.0 - positivity_df[neg_cols].to_numpy(dtype=float)
            terms.append(neg_mat)
        if terms:
            combined = np.concatenate(terms, axis=1)
            scores[:, j] = np.clip(np.mean(combined, axis=1), 0.0, 1.0)

    row_sums = scores.sum(axis=1, keepdims=True)
    probs_arr = np.divide(
        scores,
        row_sums,
        out=np.full_like(scores, 1.0 / max(1, len(ct_names))),
        where=row_sums > 0.0,
    )
    probs_arr = np.clip(probs_arr, 0.0, 1.0)

    best_idx = np.argmax(scores, axis=1)
    best_raw = scores[np.arange(len(positivity_df)), best_idx]
    best_pos = pos_only_scores[np.arange(len(positivity_df)), best_idx]

    assigned: List[str] = []
    for i in range(len(positivity_df)):
        gate_val = best_pos[i] if np.any(pos_only_scores[i] > 0) else best_raw[i]
        if gate_val < positive_threshold:
            assigned.append(unassigned_label)
        else:
            assigned.append(ct_names[int(best_idx[i])])

    labels = pd.Series(assigned, index=positivity_df.index, dtype=object, name="cell_type")
    probs_df = pd.DataFrame(probs_arr, index=positivity_df.index, columns=ct_names, dtype=float)
    confidence = pd.Series(
        np.clip(np.max(probs_arr, axis=1), 0.0, 1.0),
        index=positivity_df.index,
        dtype=float,
        name="confidence",
    )
    return labels, probs_df, confidence


def cluster_positivity_profiles(
    positivity_df: pd.DataFrame,
    *,
    n_clusters: Optional[int] = None,
    max_k: int = 20,
    som_grid: Tuple[int, int] = (10, 10),
    z_cap: float = 3.0,
    method: str = "consensus",
    positive_threshold: float = 0.5,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Cluster Nimbus per-cell positivity vectors following ``4_Generic_Cell_Clustering.ipynb``.

    Supports:
    - ``"consensus"``: z-score + clip to ``[-z_cap, z_cap]``, prototype quantization
      (`MiniBatchKMeans` mimicking SOM codebook) followed by hierarchical Ward
      agglomerative clustering (`AgglomerativeClustering`) to ``n_clusters`` (or
      silhouette-selected ``k <= max_k``).
    - ``"threshold"``: deterministic dominant-marker call above ``positive_threshold``
      (e.g., ``"CD3+"`` or ``"Negative"``).
    """
    n_cells, n_markers = positivity_df.shape
    idx = positivity_df.index
    if n_cells == 0 or n_markers == 0:
        return (
            pd.Series([], index=idx, dtype=object, name="cell_type"),
            pd.DataFrame(index=idx, dtype=float),
            pd.Series([], index=idx, dtype=float, name="confidence"),
        )

    mat = np.clip(positivity_df.to_numpy(dtype=float), 0.0, 1.0)
    markers = [str(c) for c in positivity_df.columns]

    if method == "threshold":
        best_j = np.argmax(mat, axis=1)
        best_val = mat[np.arange(n_cells), best_j]
        calls = [
            f"{markers[int(j)]}+" if val >= positive_threshold else "Negative"
            for j, val in zip(best_j, best_val)
        ]
        neg_prob = np.clip(1.0 - np.max(mat, axis=1, keepdims=True), 0.0, 1.0)
        raw = np.concatenate([mat, neg_prob], axis=1)
        row_sums = np.where(raw.sum(axis=1, keepdims=True) > 0, raw.sum(axis=1, keepdims=True), 1.0)
        probs = np.clip(raw / row_sums, 0.0, 1.0)
        cols = [f"{m}+" for m in markers] + ["Negative"]
        labels = pd.Series(calls, index=idx, dtype=object, name="cell_type")
        probs_df = pd.DataFrame(probs, index=idx, columns=cols, dtype=float)
        confidence = pd.Series(np.max(probs, axis=1), index=idx, dtype=float, name="confidence")
        return labels, probs_df, confidence

    if n_cells == 1:
        top_m = markers[int(np.argmax(mat[0]))]
        col_name = f"Cluster_0 ({top_m})" if mat[0].max() >= positive_threshold else "Cluster_0"
        return (
            pd.Series([col_name], index=idx, dtype=object, name="cell_type"),
            pd.DataFrame([[1.0]], index=idx, columns=[col_name], dtype=float),
            pd.Series([1.0], index=idx, dtype=float, name="confidence"),
        )

    # Standardize and clip z-scores to [-z_cap, z_cap] (mirrors cap=3 in 4_Generic_Cell_Clustering)
    mean = mat.mean(axis=0, keepdims=True)
    std = mat.std(axis=0, keepdims=True)
    z = np.where(std > 1e-8, (mat - mean) / np.where(std > 1e-8, std, 1.0), 0.0)
    z = np.clip(z, -float(z_cap), float(z_cap))

    # Stage 1: SOM-style prototype quantization
    n_proto = min(int(som_grid[0]) * int(som_grid[1]), n_cells)
    if n_cells > 50 and n_proto >= 4 and n_proto < n_cells:
        km = MiniBatchKMeans(n_clusters=n_proto, random_state=random_state, n_init="auto")
        proto_ids = km.fit_predict(z)
        unique_protos = np.unique(proto_ids)
        proto_centers = np.array([z[proto_ids == p].mean(axis=0) for p in unique_protos])
        remap = {int(old): new for new, old in enumerate(unique_protos)}
        proto_ids = np.array([remap[int(p)] for p in proto_ids], dtype=int)
    else:
        proto_ids = np.arange(n_cells, dtype=int)
        proto_centers = z

    n_avail = len(proto_centers)
    if n_clusters is not None:
        k_final = max(1, min(int(n_clusters), n_avail))
    else:
        upper_k = min(int(max_k), n_avail - 1, max(2, n_markers))
        if upper_k < 2 or n_avail < 3:
            k_final = 1
        else:
            best_k = 2
            best_sil = -1.0
            for cand_k in range(2, upper_k + 1):
                agg = AgglomerativeClustering(n_clusters=cand_k, linkage="ward")
                cand_labels = agg.fit_predict(proto_centers)
                if len(np.unique(cand_labels)) < 2:
                    continue
                sil = float(silhouette_score(proto_centers, cand_labels))
                if sil > best_sil:
                    best_sil = sil
                    best_k = cand_k
            k_final = best_k

    if k_final <= 1 or n_avail <= 1:
        cluster_ids = np.zeros(n_cells, dtype=int)
    else:
        agg_final = AgglomerativeClustering(n_clusters=k_final, linkage="ward")
        proto_to_cluster = agg_final.fit_predict(proto_centers)
        cluster_ids = proto_to_cluster[proto_ids]

    unique_clusters = sorted(set(cluster_ids.tolist()))
    centroids = np.vstack([mat[cluster_ids == c].mean(axis=0) for c in unique_clusters])
    cluster_names: List[str] = []
    for c_idx, c_mean in zip(unique_clusters, centroids):
        top_j = int(np.argmax(c_mean))
        if c_mean[top_j] >= positive_threshold * 0.5:
            cluster_names.append(f"Cluster_{c_idx} ({markers[top_j]})")
        else:
            cluster_names.append(f"Cluster_{c_idx}")

    dists_sq = np.sum((mat[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
    scale = max(float(np.median(dists_sq)), 1e-4)
    logits = -dists_sq / scale
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = np.clip(exp_logits / np.sum(exp_logits, axis=1, keepdims=True), 0.0, 1.0)

    id_to_pos = {cid: pos for pos, cid in enumerate(unique_clusters)}
    assigned_names = [cluster_names[id_to_pos[int(cid)]] for cid in cluster_ids]
    labels = pd.Series(assigned_names, index=idx, dtype=object, name="cell_type")
    probs_df = pd.DataFrame(probs, index=idx, columns=cluster_names, dtype=float)
    confidence = pd.Series(
        np.clip(np.max(probs, axis=1), 0.0, 1.0),
        index=idx,
        dtype=float,
        name="confidence",
    )
    return labels, probs_df, confidence


# ---------------------------------------------------------------------------
# End-to-End Pipeline Runners
# ---------------------------------------------------------------------------


@dataclass
class NimbusPipelineOutput:
    """Container returned by ``run_nimbus_on_arrays`` and ``run_nimbus_on_directory``."""

    positivity_df: pd.DataFrame
    cell_table: pd.DataFrame
    normalization_dict: Dict[str, float]
    probability_maps: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _coerce_fov_inputs(
    images: Union[np.ndarray, Sequence[np.ndarray]],
    masks: Union[np.ndarray, Sequence[np.ndarray]],
    channel_names: Optional[Sequence[str]],
    fov_names: Optional[Sequence[str]],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[str]]:
    if isinstance(images, np.ndarray):
        if images.ndim == 2:
            fov_imgs = [images[np.newaxis, ...]]
        elif images.ndim == 3:
            fov_imgs = [images]
        elif images.ndim == 4:
            fov_imgs = [images[i] for i in range(images.shape[0])]
        else:
            raise ValueError(f"Expected 2D/3D/4D image array, got shape {images.shape}.")
    else:
        fov_imgs = []
        for arr in images:
            np_arr = np.asarray(arr)
            if np_arr.ndim == 2:
                np_arr = np_arr[np.newaxis, ...]
            if np_arr.ndim != 3:
                raise ValueError(f"Expected (C, H, W) array per FOV, got shape {np_arr.shape}.")
            fov_imgs.append(np_arr)

    if isinstance(masks, np.ndarray):
        if masks.ndim == 2 or (masks.ndim == 3 and masks.shape[-1] == 3 and len(fov_imgs) == 1):
            fov_masks = [normalize_instance_mask(masks)]
        elif masks.ndim == 3:
            fov_masks = [normalize_instance_mask(masks[i]) for i in range(masks.shape[0])]
        else:
            raise ValueError(f"Expected 2D/3D mask array, got shape {masks.shape}.")
    else:
        fov_masks = [normalize_instance_mask(m) for m in masks]

    if len(fov_imgs) != len(fov_masks):
        raise ValueError(f"Number of FOV images ({len(fov_imgs)}) != masks ({len(fov_masks)}).")
    if not fov_imgs:
        raise ValueError("At least one FOV image and mask must be provided.")

    n_channels = fov_imgs[0].shape[0]
    for idx, img in enumerate(fov_imgs):
        if img.shape[0] != n_channels:
            raise ValueError(f"FOV {idx} has {img.shape[0]} channels, expected {n_channels}.")
        if img.shape[1:] != fov_masks[idx].shape:
            raise ValueError(
                f"FOV {idx} image spatial shape {img.shape[1:]} != mask shape {fov_masks[idx].shape}."
            )

    ch_names = (
        list(channel_names)
        if channel_names is not None
        else [f"Marker_{i}" for i in range(n_channels)]
    )
    if len(ch_names) != n_channels:
        raise ValueError(
            f"channel_names length ({len(ch_names)}) != image channels ({n_channels})."
        )

    f_names = (
        list(fov_names)
        if fov_names is not None
        else [f"fov_{i}" for i in range(len(fov_imgs))]
    )
    if len(f_names) != len(fov_imgs):
        raise ValueError(f"fov_names length ({len(f_names)}) != number of FOVs ({len(fov_imgs)}).")

    return fov_imgs, fov_masks, ch_names, f_names


def run_nimbus_on_arrays(
    images: Union[np.ndarray, Sequence[np.ndarray]],
    masks: Union[np.ndarray, Sequence[np.ndarray]],
    *,
    channel_names: Optional[Sequence[str]] = None,
    fov_names: Optional[Sequence[str]] = None,
    normalization_dict: Optional[Mapping[str, float]] = None,
    checkpoint: Union[str, Path] = "latest",
    model: Optional["nn.Module"] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    allow_download: bool = True,
    quantile: float = 0.999,
    clip_values: Tuple[float, float] = (0.0, 1.0),
    n_subset: Optional[int] = 50,
    dataset_magnification: float = 20.0,
    model_magnification: float = 10.0,
    input_shape: Tuple[int, int] = (1024, 1024),
    batch_size: int = 4,
    test_time_aug: bool = True,
    device: str = "auto",
    mixed_precision: bool = False,
    return_pixel_maps: bool = False,
) -> NimbusPipelineOutput:
    """Run faithful Nimbus inference directly on in-memory ``(C, H, W)`` arrays and masks."""
    fov_imgs, fov_masks, ch_names, f_names = _coerce_fov_inputs(
        images, masks, channel_names, fov_names
    )

    if normalization_dict is not None:
        norm_dict = {str(k): float(v) for k, v in normalization_dict.items()}
    else:
        norm_dict = prepare_normalization_dict(
            fov_imgs, channel_names=ch_names, quantile=quantile, n_subset=n_subset
        )

    torch_device = _select_device(device)
    model_reflect: Optional["nn.Module"] = None
    model_valid: Optional["nn.Module"] = None
    ckpt_path: Optional[Path] = None

    needs_tile = any(
        int(round(img.shape[1] * model_magnification / dataset_magnification)) > input_shape[0]
        or int(round(img.shape[2] * model_magnification / dataset_magnification)) > input_shape[1]
        for img in fov_imgs
    )
    needs_reflect = any(
        int(round(img.shape[1] * model_magnification / dataset_magnification)) <= input_shape[0]
        and int(round(img.shape[2] * model_magnification / dataset_magnification)) <= input_shape[1]
        for img in fov_imgs
    )

    if model is not None:
        first_param = next(model.parameters(), None)
        if first_param is not None:
            torch_device = first_param.device
        model_reflect = model
        model_valid = model
    else:
        if needs_reflect:
            model_reflect, torch_device, ckpt_path = load_nimbus_model(
                checkpoint=checkpoint,
                padding="reflect",
                device=device,
                cache_dir=cache_dir,
                allow_download=allow_download,
            )
        if needs_tile:
            model_valid, torch_device, ckpt_path = load_nimbus_model(
                checkpoint=checkpoint,
                padding="valid",
                device=device,
                cache_dir=cache_dir,
                allow_download=allow_download,
            )

    fov_tables: List[pd.DataFrame] = []
    prob_maps: Dict[str, Dict[str, np.ndarray]] = {}

    for fov_img, fov_mask, fov_name in zip(fov_imgs, fov_masks, f_names):
        binary_mask = prepare_binary_mask(fov_mask)
        fov_df: Optional[pd.DataFrame] = None
        if return_pixel_maps:
            prob_maps[fov_name] = {}

        for c_idx, ch_name in enumerate(ch_names):
            norm_img = normalize_channel_image(
                fov_img[c_idx],
                norm_factor=norm_dict.get(ch_name),
                quantile=quantile,
                clip_values=clip_values,
            )
            prob_map = predict_channel_map(
                norm_img,
                binary_mask,
                model_reflect=model_reflect,
                model_valid=model_valid,
                device=torch_device,
                dataset_magnification=dataset_magnification,
                model_magnification=model_magnification,
                input_shape=input_shape,
                batch_size=batch_size,
                test_time_aug=test_time_aug,
                mixed_precision=mixed_precision,
            )
            if return_pixel_maps:
                prob_maps[fov_name][ch_name] = prob_map

            cell_df = aggregate_cell_positivity(fov_mask, prob_map)
            if fov_df is None:
                fov_df = cell_df[[COL_CELL_ID, COL_X, COL_Y, "area"]].copy()
                fov_df[COL_SAMPLE] = fov_name
            fov_df[ch_name] = cell_df["intensity_mean"].to_numpy(dtype=float)

        assert fov_df is not None
        fov_tables.append(fov_df)

    cell_table = pd.concat(fov_tables, ignore_index=True)
    positivity_df = cell_table[ch_names].copy()
    positivity_df.index = pd.Index(cell_table[COL_CELL_ID].astype(int), name=COL_CELL_ID)

    return NimbusPipelineOutput(
        positivity_df=positivity_df,
        cell_table=cell_table,
        normalization_dict=norm_dict,
        probability_maps=prob_maps,
        metadata={
            "checkpoint": str(ckpt_path) if ckpt_path is not None else "custom_model",
            "device": str(torch_device),
            "test_time_aug": bool(test_time_aug),
            "dataset_magnification": float(dataset_magnification),
            "model_magnification": float(model_magnification),
            "fovs": f_names,
            "channels": ch_names,
        },
    )


def _load_fov_directory_native(
    fov_paths: Sequence[Union[str, Path]],
    segmentation_naming_convention: Callable[[str], Union[str, Path]],
    channel_names: Optional[Sequence[str]],
    suffix: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[str]]:
    """Read FOV channel TIFFs or OME-TIFF stacks and instance masks without ``nimbus_inference``."""
    import tifffile

    fov_imgs: List[np.ndarray] = []
    fov_masks: List[np.ndarray] = []
    f_names: List[str] = []
    resolved_channels: Optional[List[str]] = list(channel_names) if channel_names else None

    for raw_path in fov_paths:
        p = Path(raw_path)
        fov_name = p.name.replace(suffix, "") if p.is_file() else p.name
        f_names.append(fov_name)
        mask_path = segmentation_naming_convention(str(p))
        fov_masks.append(normalize_instance_mask(tifffile.imread(str(mask_path))))

        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.is_file() and f.name.endswith(suffix))
            if resolved_channels is None:
                resolved_channels = [f.name[: -len(suffix)] for f in files]
            ch_arrays = [
                np.squeeze(tifffile.imread(str(p / f"{ch}{suffix}"))) for ch in resolved_channels
            ]
            fov_imgs.append(np.stack(ch_arrays, axis=0))
        else:
            with tifffile.TiffFile(str(p)) as tif:
                arr = np.squeeze(tif.asarray())
                if arr.ndim == 2:
                    arr = arr[np.newaxis, ...]
                elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0] and arr.shape[-1] <= 64:
                    arr = np.moveaxis(arr, -1, 0)
                if resolved_channels is None:
                    ome_chs: List[str] = []
                    if tif.is_ome and tif.ome_metadata:
                        ome_chs = re.findall(r'<Channel[^>]*Name="([^"]+)"', tif.ome_metadata)
                    resolved_channels = (
                        ome_chs
                        if len(ome_chs) == arr.shape[0]
                        else [f"Channel_{i}" for i in range(arr.shape[0])]
                    )
                elif tif.is_ome and tif.ome_metadata:
                    ome_chs = re.findall(r'<Channel[^>]*Name="([^"]+)"', tif.ome_metadata)
                    if ome_chs and len(ome_chs) == arr.shape[0]:
                        keep = [ome_chs.index(c) for c in resolved_channels if c in ome_chs]
                        arr = arr[keep]
                        resolved_channels = [c for c in resolved_channels if c in ome_chs]
            fov_imgs.append(arr)

    assert resolved_channels is not None
    return fov_imgs, fov_masks, resolved_channels, f_names


def run_nimbus_on_directory(
    fov_paths: Sequence[Union[str, Path]],
    segmentation_naming_convention: Optional[Callable[[str], Union[str, Path]]] = None,
    *,
    mask_dir: Optional[Union[str, Path]] = None,
    approx_mask_match: bool = False,
    channel_names: Optional[Sequence[str]] = None,
    suffix: str = ".tiff",
    output_dir: Optional[Union[str, Path]] = None,
    normalization_dict: Optional[Mapping[str, float]] = None,
    checkpoint: Union[str, Path] = "latest",
    cache_dir: Optional[Union[str, Path]] = None,
    quantile: float = 0.999,
    n_subset: Optional[int] = 50,
    dataset_magnification: float = 20.0,
    model_magnification: float = 10.0,
    input_shape: Tuple[int, int] = (1024, 1024),
    batch_size: int = 4,
    test_time_aug: bool = True,
    device: str = "auto",
    save_predictions: bool = False,
    use_upstream: Union[bool, str] = "auto",
) -> NimbusPipelineOutput:
    """Run Nimbus on FOV directories or OME-TIFF files via ``nimbus_inference`` or native I/O."""
    paths_str = [str(p) for p in fov_paths]
    if not paths_str:
        raise ValueError("fov_paths must not be empty.")

    if segmentation_naming_convention is None:
        if mask_dir is None:
            raise ValueError("Provide either segmentation_naming_convention or mask_dir.")
        if is_upstream_available():
            from nimbus_inference.nimbus import prep_naming_convention

            segmentation_naming_convention = prep_naming_convention(
                str(mask_dir), approx=approx_mask_match
            )
        else:
            mask_root = Path(mask_dir)

            def _default_naming(fov_p: str) -> str:
                stem = Path(fov_p).name.replace(".ome.tiff", "").replace(".ome.tif", "")
                stem = Path(stem).stem
                if approx_mask_match:
                    hits = sorted(mask_root.glob(f"*{stem}*"))
                    if len(hits) != 1:
                        raise ValueError(f"Expected 1 mask for {stem}, found {len(hits)}")
                    return str(hits[0])
                return str(mask_root / f"{stem}_whole_cell.tiff")

            segmentation_naming_convention = _default_naming

    should_use_upstream = (
        is_upstream_available() if use_upstream == "auto" else bool(use_upstream)
    )
    if should_use_upstream:
        if not is_upstream_available():
            raise ImportError("use_upstream=True requires the nimbus_inference package.")
        import tempfile

        from nimbus_inference.nimbus import Nimbus
        from nimbus_inference.utils import MultiplexDataset

        with tempfile.TemporaryDirectory() as tmp_out:
            out_path = str(output_dir) if output_dir is not None else tmp_out
            os.makedirs(out_path, exist_ok=True)
            dataset = MultiplexDataset(
                fov_paths=[str(p) for p in paths_str],
                segmentation_naming_convention=segmentation_naming_convention,
                include_channels=list(channel_names) if channel_names else [],
                suffix=suffix,
                silent=True,
                magnification=int(dataset_magnification),
                output_dir=out_path,
            )
            if normalization_dict is not None:
                dataset.normalization_dict = {str(k): float(v) for k, v in normalization_dict.items()}
                dataset.clip_values = (0.0, 2.0)
            else:
                dataset.prepare_normalization_dict(
                    quantile=quantile,
                    n_subset=n_subset,
                    multiprocessing=False,
                    overwrite=True,
                )

            ckpt_arg = "latest"
            copied_temp_ckpt: Optional[Path] = None
            if checkpoint != "latest":
                ckpt_path = resolve_nimbus_checkpoint(checkpoint, cache_dir=cache_dir)
                import nimbus_inference

                assets_dir = Path(nimbus_inference.__file__).resolve().parent / "assets"
                assets_dir.mkdir(parents=True, exist_ok=True)
                target = assets_dir / ckpt_path.name
                if not target.exists():
                    target.write_bytes(ckpt_path.read_bytes())
                    copied_temp_ckpt = target
                ckpt_arg = ckpt_path.name

            try:
                app = Nimbus(
                    dataset=dataset,
                    output_dir=out_path,
                    save_predictions=save_predictions,
                    batch_size=batch_size,
                    test_time_aug=test_time_aug,
                    model_magnification=int(model_magnification),
                    input_shape=list(input_shape),
                    device=device,
                    checkpoint=ckpt_arg,
                )
                raw_table = app.predict_fovs()
                resolved_ckpt = str(app.checkpoint_path)
                resolved_dev = str(app.device)
            finally:
                if copied_temp_ckpt is not None and copied_temp_ckpt.exists():
                    copied_temp_ckpt.unlink()

        cell_table = raw_table.copy()
        cell_table = cell_table.rename(
            columns={
                "label": COL_CELL_ID,
                "fov": COL_SAMPLE,
                "centroid_x": COL_X,
                "centroid_y": COL_Y,
            }
        )
        if channel_names:
            ch_cols = [c for c in channel_names if c in dataset.channels]
        else:
            ch_cols = sorted(dataset.channels)
        positivity_df = cell_table[ch_cols].copy()
        positivity_df.index = pd.Index(cell_table[COL_CELL_ID].astype(int), name=COL_CELL_ID)
        return NimbusPipelineOutput(
            positivity_df=positivity_df,
            cell_table=cell_table,
            normalization_dict=dict(dataset.normalization_dict),
            metadata={
                "checkpoint": resolved_ckpt,
                "device": resolved_dev,
                "backend": "nimbus_inference",
                "test_time_aug": bool(test_time_aug),
                "dataset_magnification": float(dataset_magnification),
                "model_magnification": float(model_magnification),
                "fovs": list(dataset.fovs),
                "channels": ch_cols,
            },
        )

    fov_imgs, fov_masks, resolved_chs, f_names = _load_fov_directory_native(
        fov_paths, segmentation_naming_convention, channel_names, suffix
    )
    return run_nimbus_on_arrays(
        fov_imgs,
        fov_masks,
        channel_names=resolved_chs,
        fov_names=f_names,
        normalization_dict=normalization_dict,
        checkpoint=checkpoint,
        cache_dir=cache_dir,
        quantile=quantile,
        n_subset=n_subset,
        dataset_magnification=dataset_magnification,
        model_magnification=model_magnification,
        input_shape=input_shape,
        batch_size=batch_size,
        test_time_aug=test_time_aug,
        device=device,
        return_pixel_maps=save_predictions,
    )
