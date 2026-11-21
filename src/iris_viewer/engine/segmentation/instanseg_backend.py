"""InstanSeg Segmentation Backend."""
import importlib.util
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

from iris_common.logging import iris_log, log_warn

from .base import SegmentationBackend


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
            "pixel_size": 0.5,
            "resolve_cell_and_nucleus": True,
            "cleanup_fragments": True,
            "gaussian_sigma": 0.0,
            "norm_pmin": 1.0,
            "norm_pmax": 99.8,
            "min_cell_area": 0,
            "max_cell_area": 0,
            "expand_radius": 0,
            "expansion_mode": "distance",
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

        model_name = kwargs.get("model_name", "fluorescence_nuclei_and_cells")
        pixel_size = kwargs.get("pixel_size", 0.5)
        if pixel_size is not None and pixel_size <= 0:
            pixel_size = None
        resolve_cell_and_nucleus = bool(kwargs.get("resolve_cell_and_nucleus", True))
        cleanup_fragments = bool(kwargs.get("cleanup_fragments", True))
        gaussian_sigma = float(kwargs.get("gaussian_sigma", 0.0))
        norm_pmin = kwargs.get("norm_pmin", 1.0)
        norm_pmax = kwargs.get("norm_pmax", 99.8)
        min_cell_area = int(kwargs.get("min_cell_area", 0) or 0)
        max_cell_area = int(kwargs.get("max_cell_area", 0) or 0)
        expansion_mode = str(kwargs.get("expansion_mode", "distance"))

        backends_mod = getattr(torch, "backends", None)
        mps_mod = getattr(backends_mod, "mps", None)
        if device == "mps" and mps_mod is not None and not mps_mod.is_available():
            device = "cpu"

        if (
            self._cached_model is None
            or self._cached_model_name != model_name
            or self._cached_device != device
        ):
            try:
                self._cached_model = InstanSeg(model_name, device=device, verbosity=0)
            except Exception as exc:
                if device == "mps":
                    log_warn(
                        "IRIS-Viewer",
                        "Segmentation",
                        f"InstanSeg MPS init failed ({exc}), falling back to CPU.",
                    )
                    device = "cpu"
                    self._cached_model = InstanSeg(model_name, device=device, verbosity=0)
                else:
                    raise
            self._cached_model_name = model_name
            self._cached_device = device

        has_membrane = (
            image.ndim == 3
            and image.shape[0] > 1
            and membrane_channel is not None
            and membrane_channel != nuclear_channel
        )
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
                f"InstanSeg '{model_name}' running "
                f"(shape={normalized_img.shape}, device={device})...",
            )
            try:
                eval_result = self._cached_model.eval_medium_image(
                    input_tensor,
                    pixel_size=pixel_size,
                    resolve_cell_and_nucleus=resolve_cell_and_nucleus,
                    cleanup_fragments=cleanup_fragments,
                )
            except Exception as exc:
                if device == "mps":
                    log_warn(
                        "IRIS-Viewer",
                        "Segmentation",
                        f"InstanSeg MPS eval failed ({exc}), retrying on CPU...",
                    )
                    device = "cpu"
                    self._cached_model = InstanSeg(model_name, device="cpu", verbosity=0)
                    self._cached_device = "cpu"
                    eval_result = self._cached_model.eval_medium_image(
                        input_tensor,
                        pixel_size=pixel_size,
                        resolve_cell_and_nucleus=resolve_cell_and_nucleus,
                        cleanup_fragments=cleanup_fragments,
                    )
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
            cyto_mask = self._filter_mask_by_area(
                labeled_output[1], min_area=min_cell_area, max_area=max_cell_area
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
            f"InstanSeg '{model_name}' complete in {time.perf_counter() - t0:.2f}s "
            f"(N={n_nuc} nuclear, {n_cyto} cytoplasm)",
        )

        return {"nuclear": nuclear_mask, "cytoplasm": cyto_mask}
