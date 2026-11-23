"""InstanSeg Segmentation Backend."""
import importlib.util
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from iris_common.logging import iris_log, log_warn

from .base import SegmentationBackend

warnings.filterwarnings("ignore", message=".*Disabling autocast.*")
warnings.filterwarnings("ignore", message=".*`torch.jit.load` is deprecated.*")


def _resolve_instanseg_model_and_target(
    model_name: str,
    explicit_target: Optional[str] = None,
    skip_nuclear: bool = False,
) -> tuple[str, str]:
    """Map UI preset names to official InstanSeg archive names and output target heads."""
    clean = (model_name or "fluorescence_nuclei_and_cells").strip()
    norm_target = (explicit_target or "").strip()
    if not norm_target or norm_target.lower() == "auto":
        resolved_explicit: Optional[str] = None
    else:
        resolved_explicit = norm_target

    bio_dir = os.environ.get("INSTANSEG_BIOIMAGEIO_PATH", "")
    if bio_dir and (Path(bio_dir) / clean / "instanseg.pt").is_file():
        return clean, resolved_explicit or "all_outputs"

    if clean == "fluorescence_nuclei":
        return "fluorescence_nuclei_and_cells", resolved_explicit or "nuclei"
    if clean == "fluorescence_cells":
        return "fluorescence_nuclei_and_cells", resolved_explicit or "cells"
    if clean == "fluorescence_nuclei_and_cells":
        if resolved_explicit in ("nuclei", "cells", "all_outputs"):
            return clean, resolved_explicit
        return clean, ("cells" if skip_nuclear else "all_outputs")
    return clean, resolved_explicit or "all_outputs"


def _has_modules(*names: str) -> bool:
    for mod_name in names:
        if mod_name in sys.modules:
            if sys.modules[mod_name] is None:
                return False
            continue
        try:
            if importlib.util.find_spec(mod_name) is None:
                return False
        except (ImportError, ModuleNotFoundError, ValueError):
            return False
    return True


class InstanSegBackend(SegmentationBackend):
    name = "InstanSeg"
    supports_cytoplasm = True

    def __init__(self):
        self._cached_model = None
        self._cached_model_name = None
        self._cached_device = None

    def is_available(self) -> bool:
        return _has_modules("instanseg", "torch")

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "model_name": "fluorescence_nuclei_and_cells",
            "target": "auto",
            "pixel_size": 0.5,
            "resolve_cell_and_nucleus": True,
            "cleanup_fragments": True,
            "normalise": True,
            "rescale_output": True,
            "tile_size": 512,
            "batch_size": 1,
            "seed_threshold": None,
            "mask_threshold": None,
            "peak_distance": None,
            "overlap_threshold": None,
            "window_size": None,
            "gaussian_sigma": 0.0,
            "norm_pmin": 1.0,
            "norm_pmax": 99.8,
            "min_cell_area": 0,
            "max_cell_area": 0,
            "expand_radius": 0,
            "expansion_mode": "distance",
            "extra_kwargs": {},
        }

    def segment(
        self,
        image: np.ndarray,
        nuclear_channel: int,
        membrane_channel: Optional[int],
        diameter: Optional[float] = None,
        device: str = "cpu",
        expand_radius: int = 0,
        **kwargs,
    ) -> Dict[str, Optional[np.ndarray]]:
        import gc

        import torch
        from instanseg import InstanSeg

        requested_model = str(kwargs.get("model_name", "fluorescence_nuclei_and_cells"))
        explicit_target = kwargs.get("target")
        skip_nuclear = bool(kwargs.get("skip_nuclear", False))
        pixel_size = kwargs.get("pixel_size", 0.5)
        if pixel_size is not None and pixel_size <= 0:
            pixel_size = None
        resolve_cell_and_nucleus = bool(kwargs.get("resolve_cell_and_nucleus", True))
        cleanup_fragments = bool(kwargs.get("cleanup_fragments", True))
        normalise = bool(kwargs.get("normalise", True))
        rescale_output = bool(kwargs.get("rescale_output", True))
        tile_size = max(64, int(kwargs.get("tile_size", 512) or 512))
        batch_size = max(1, int(kwargs.get("batch_size", 1) or 1))
        gaussian_sigma = float(kwargs.get("gaussian_sigma", 0.0))
        norm_pmin = kwargs.get("norm_pmin", 1.0)
        norm_pmax = kwargs.get("norm_pmax", 99.8)
        min_cell_area = int(kwargs.get("min_cell_area", 0) or 0)
        max_cell_area = int(kwargs.get("max_cell_area", 0) or 0)
        expand_radius = int(expand_radius or 0)
        expansion_mode = str(kwargs.get("expansion_mode", "distance"))
        extra_kw = self._parse_extra_kwargs(kwargs.get("extra_kwargs"))

        has_membrane = (
            image.ndim == 3
            and image.shape[0] > 1
            and membrane_channel is not None
            and membrane_channel != nuclear_channel
        )
        resolved_model, target = _resolve_instanseg_model_and_target(
            requested_model,
            explicit_target=explicit_target,
            skip_nuclear=(skip_nuclear and has_membrane),
        )

        backends_mod = getattr(torch, "backends", None)
        mps_mod = getattr(backends_mod, "mps", None)
        if device == "mps" and mps_mod is not None and not mps_mod.is_available():
            device = "cpu"

        if (
            self._cached_model is None
            or self._cached_model_name != resolved_model
            or self._cached_device != device
        ):
            try:
                self._cached_model = InstanSeg(resolved_model, device=device, verbosity=0)
            except Exception as exc:
                if device == "mps" and "not found in the release data" not in str(exc):
                    log_warn(
                        "IRIS-Viewer",
                        "Segmentation",
                        f"InstanSeg MPS init failed ({exc}), falling back to CPU.",
                    )
                    device = "cpu"
                    self._cached_model = InstanSeg(resolved_model, device=device, verbosity=0)
                else:
                    raise
            self._cached_model_name = resolved_model
            self._cached_device = device

        mem_raw = np.asarray(image[membrane_channel]) if has_membrane else None

        if image.ndim == 3 and image.shape[0] > 1:
            if has_membrane:
                raw_channels = [
                    np.asarray(image[nuclear_channel]),
                    np.asarray(image[membrane_channel]),
                ]
            else:
                raw_channels = [np.asarray(image[nuclear_channel])]
        else:
            raw_channels = [np.asarray(image[0] if image.ndim == 3 else image)]

        normalized_img = np.stack(
            [
                self._preprocess_channel(
                    ch,
                    gaussian_sigma=gaussian_sigma,
                    norm_pmin=norm_pmin,
                    norm_pmax=norm_pmax,
                )
                for ch in raw_channels
            ],
            axis=0,
        ).astype(np.float32)

        input_tensor = torch.from_numpy(normalized_img).unsqueeze(0)

        eval_kw: Dict[str, Any] = {
            "pixel_size": pixel_size,
            "normalise": normalise,
            "tile_size": tile_size,
            "batch_size": batch_size,
            "rescale_output": rescale_output,
            "resolve_cell_and_nucleus": resolve_cell_and_nucleus,
            "cleanup_fragments": cleanup_fragments,
            "target": target,
        }
        for opt_key, cast_fn in (
            ("seed_threshold", float),
            ("mask_threshold", float),
            ("peak_distance", int),
            ("overlap_threshold", float),
            ("window_size", int),
        ):
            opt_val = kwargs.get(opt_key)
            if opt_val is not None and float(opt_val) > 0:
                eval_kw[opt_key] = cast_fn(opt_val)
        eval_kw.update(extra_kw)

        original_quantile = torch.quantile

        def safe_quantile(input, q, dim=None, keepdim=False, *, interpolation="linear", out=None):
            if input.dtype == torch.float32 and input.numel() > 16000000:
                cpu_tensor = input.cpu().to(torch.float64)
                q_cast = (
                    q.to(device=cpu_tensor.device, dtype=torch.float64)
                    if isinstance(q, torch.Tensor)
                    else float(q)
                )
                res = original_quantile(
                    cpu_tensor,
                    q_cast,
                    dim=dim,
                    keepdim=keepdim,
                    interpolation=interpolation,
                    out=out,
                )
                return res.to(torch.float32).to(input.device)
            return original_quantile(
                input, q, dim=dim, keepdim=keepdim, interpolation=interpolation, out=out
            )

        torch.quantile = safe_quantile

        t0 = time.perf_counter()
        try:
            iris_log(
                "IRIS-Viewer",
                "Segmentation",
                f"InstanSeg '{resolved_model}' (target={target}) running "
                f"(shape={normalized_img.shape}, device={device})...",
            )
            try:
                eval_result = self._cached_model.eval_medium_image(input_tensor, **eval_kw)
            except Exception as exc:
                if device == "mps":
                    log_warn(
                        "IRIS-Viewer",
                        "Segmentation",
                        f"InstanSeg MPS eval failed ({exc}), retrying on CPU...",
                    )
                    device = "cpu"
                    self._cached_model = InstanSeg(resolved_model, device="cpu", verbosity=0)
                    self._cached_device = "cpu"
                    eval_result = self._cached_model.eval_medium_image(input_tensor, **eval_kw)
                else:
                    raise
            labeled_output = eval_result[0]
        finally:
            torch.quantile = original_quantile
            if device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
            elif device == "cuda" and hasattr(torch, "cuda"):
                torch.cuda.empty_cache()
            gc.collect()

        if isinstance(labeled_output, torch.Tensor):
            labeled_output = labeled_output.cpu().numpy()

        labeled_output = np.asarray(labeled_output)
        if np.issubdtype(labeled_output.dtype, np.floating):
            labeled_output = np.rint(
                np.nan_to_num(labeled_output, nan=0.0, posinf=0.0, neginf=0.0)
            ).astype(np.int32, copy=False)

        if labeled_output.ndim == 4:
            labeled_output = labeled_output[0]
        elif labeled_output.ndim == 2:
            labeled_output = labeled_output[np.newaxis, ...]

        orig_h, orig_w = image.shape[-2:]
        mask_h, mask_w = labeled_output.shape[-2:]

        if (mask_h, mask_w) != (orig_h, orig_w):
            if mask_h > orig_h:
                labeled_output = labeled_output[..., :orig_h, :]
            if mask_w > orig_w:
                labeled_output = labeled_output[..., :, :orig_w]

            pad_h = max(0, orig_h - labeled_output.shape[-2])
            pad_w = max(0, orig_w - labeled_output.shape[-1])

            if pad_h > 0 or pad_w > 0:
                pad_width = [(0, 0)] * labeled_output.ndim
                pad_width[-2] = (0, pad_h)
                pad_width[-1] = (0, pad_w)
                labeled_output = np.pad(
                    labeled_output, pad_width, mode="constant", constant_values=0
                )

        if labeled_output.shape[0] >= 2:
            nuclear_mask = self._filter_mask_by_area(
                labeled_output[0], min_area=min_cell_area, max_area=max_cell_area
            ).astype(np.uint32)
            if max_cell_area > 0:
                raw_cyto = self._filter_mask_by_area(
                    labeled_output[1], min_area=0, max_area=max_cell_area
                )
            else:
                raw_cyto = np.maximum(np.asarray(labeled_output[1]), 0).astype(
                    np.uint32, copy=False
                )
            nuclear_mask, cyto_mask = self.reconcile_nuclear_and_cytoplasm_masks(
                nuclear_mask, raw_cyto
            )
        elif target == "cells":
            nuclear_mask = np.zeros((orig_h, orig_w), dtype=np.uint32)
            cyto_mask = self._filter_mask_by_area(
                labeled_output[0], min_area=min_cell_area, max_area=max_cell_area
            ).astype(np.uint32)
        else:
            nuclear_mask = self._filter_mask_by_area(
                labeled_output[0], min_area=min_cell_area, max_area=max_cell_area
            ).astype(np.uint32)
            cyto_mask = (
                self._expand_nuclear_to_cytoplasm(
                    nuclear_mask,
                    expand_radius,
                    mode=expansion_mode,
                    membrane_image=mem_raw,
                ).astype(np.uint32)
                if expand_radius > 0
                else None
            )

        n_nuc = int((np.unique(nuclear_mask) > 0).sum())
        n_cyto = int((np.unique(cyto_mask) > 0).sum()) if cyto_mask is not None else 0
        iris_log(
            "IRIS-Viewer",
            "Segmentation",
            f"InstanSeg '{resolved_model}' (target={target}) complete in "
            f"{time.perf_counter() - t0:.2f}s (N={n_nuc} nuclear, {n_cyto} cytoplasm)",
        )

        return {"nuclear": nuclear_mask, "cytoplasm": cyto_mask}
