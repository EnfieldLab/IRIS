"""Cellpose Segmentation Backend."""
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

warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")
warnings.filterwarnings("ignore", message=".*channels argument is deprecated.*")
warnings.filterwarnings("ignore", message=".*model_type argument is not used.*")


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


def _resolve_torch_device(torch_mod: Any, req_device: str) -> tuple[Any, bool, str]:
    if req_device == "mps":
        mps_mod = getattr(getattr(torch_mod, "backends", None), "mps", None)
        if mps_mod is None or mps_mod.is_available():
            return torch_mod.device("mps"), True, "mps"
        return torch_mod.device("cpu"), False, "cpu"
    if req_device == "cuda":
        cuda_mod = getattr(torch_mod, "cuda", None)
        if cuda_mod is None or cuda_mod.is_available():
            return torch_mod.device("cuda"), True, "cuda"
        return torch_mod.device("cpu"), False, "cpu"
    return torch_mod.device("cpu"), False, "cpu"


class CellposeBackend(SegmentationBackend):
    name = "Cellpose"
    supports_cytoplasm = True

    def __init__(self):
        self._cached_model_cyto = None
        self._cached_model_nuc = None
        self._cached_device = None
        self._cached_cyto_type = None

    def is_available(self) -> bool:
        return _has_modules("cellpose", "torch")

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "model_type": "cpsam",
            "diameter": None,
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
            "tile_overlap": 0.1,
            "resample": True,
            "augment": False,
            "niter": None,
            "max_size_fraction": 0.4,
            "batch_size": 8,
            "bsize": None,
            "stitch_threshold": 0.0,
            "cellpose_normalize": True,
            "gaussian_sigma": 0.0,
            "norm_pmin": 1.0,
            "norm_pmax": 99.8,
            "min_cell_area": 0,
            "max_cell_area": 0,
            "expand_radius": 0,
            "expansion_mode": "distance",
            "extra_kwargs": {},
        }

    def _build_model(
        self,
        models_mod: Any,
        torch_device: Any,
        use_gpu: bool,
        model_type: str,
        is_v4: bool,
    ) -> Any:
        raw_mt = str(model_type or "cpsam").strip()
        expanded_mt = os.path.expanduser(raw_mt)
        if os.path.exists(expanded_mt):
            if is_v4:
                return models_mod.CellposeModel(
                    gpu=use_gpu, device=torch_device, pretrained_model=expanded_mt
                )
            return models_mod.CellposeModel(
                gpu=use_gpu, device=torch_device, pretrained_model=expanded_mt
            )

        pretrained_arg = raw_mt
        model_names = getattr(models_mod, "MODEL_NAMES", None)
        model_dir = getattr(models_mod, "MODEL_DIR", None)
        if is_v4 and model_names is not None:
            if raw_mt in ("cpsam", "cpsam_v2", "cyto3", "cyto2", "cyto"):
                if (
                    raw_mt == "cpsam"
                    and model_dir is not None
                    and (Path(model_dir) / "cpsam").exists()
                ):
                    pretrained_arg = "cpsam"
                elif raw_mt == "cpsam_v2" and "cpsam_v2" in model_names:
                    pretrained_arg = "cpsam_v2"
                elif (
                    "cpsam_v2" in model_names
                    and model_dir is not None
                    and (Path(model_dir) / "cpsam_v2").exists()
                ):
                    pretrained_arg = "cpsam_v2"
                elif "cpsam" in model_names:
                    pretrained_arg = "cpsam"
            elif raw_mt in model_names:
                pretrained_arg = raw_mt
            else:
                if (
                    "cpsam_v2" in model_names
                    and model_dir is not None
                    and (Path(model_dir) / "cpsam_v2").exists()
                ):
                    pretrained_arg = "cpsam_v2"
                elif "cpsam" in model_names:
                    pretrained_arg = "cpsam"
                elif len(model_names) > 0:
                    pretrained_arg = model_names[0]

        if is_v4:
            return models_mod.CellposeModel(
                gpu=use_gpu, device=torch_device, pretrained_model=pretrained_arg
            )
        return models_mod.CellposeModel(
            gpu=use_gpu, device=torch_device, model_type=raw_mt
        )

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
        import inspect

        import torch
        from cellpose import models

        if image.ndim < 2 or image.shape[-2] == 0 or image.shape[-1] == 0:
            shape = image.shape[-2:] if image.ndim >= 2 else (0, 0)
            return {"nuclear": np.zeros(shape, dtype=np.uint32), "cytoplasm": None}
        if image.ndim > 3:
            raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}.")

        model_type = str(kwargs.get("model_type") or "cpsam").strip()
        flow_threshold = float(kwargs.get("flow_threshold", 0.4))
        cellprob_threshold = float(kwargs.get("cellprob_threshold", 0.0))
        raw_overlap = kwargs.get("tile_overlap", 0.1)
        tile_overlap = float(
            np.clip(float(0.1 if raw_overlap is None else raw_overlap), 0.05, 0.5)
        )
        resample = bool(kwargs.get("resample", True))
        augment = bool(kwargs.get("augment", False))
        raw_niter = kwargs.get("niter", None)
        niter = int(raw_niter) if (raw_niter is not None and int(raw_niter) > 0) else None
        max_size_fraction = float(kwargs.get("max_size_fraction", 0.4))
        batch_size = max(1, int(kwargs.get("batch_size", 8) or 8))
        raw_bsize = kwargs.get("bsize", None)
        bsize = int(raw_bsize) if (raw_bsize is not None and int(raw_bsize) > 0) else None
        stitch_threshold = float(kwargs.get("stitch_threshold", 0.0) or 0.0)
        cellpose_normalize = bool(kwargs.get("cellpose_normalize", True))
        skip_nuclear = bool(kwargs.get("skip_nuclear", False))
        gaussian_sigma = float(kwargs.get("gaussian_sigma", 0.0))
        norm_pmin = kwargs.get("norm_pmin", 1.0)
        norm_pmax = kwargs.get("norm_pmax", 99.8)
        min_cell_area = int(kwargs.get("min_cell_area", 0) or 0)
        max_cell_area = int(kwargs.get("max_cell_area", 0) or 0)
        expand_radius = int(expand_radius or 0)
        expansion_mode = str(kwargs.get("expansion_mode", "distance"))
        extra_kw = self._parse_extra_kwargs(kwargs.get("extra_kwargs"))

        if diameter is not None and diameter <= 0:
            diameter = None

        torch_device, use_gpu, actual_device = _resolve_torch_device(torch, device)

        sig = inspect.signature(models.CellposeModel.eval)
        is_v4 = "channel_axis" in sig.parameters
        if image.ndim == 3:
            nuc_idx = max(0, min(int(nuclear_channel), image.shape[0] - 1))
            has_membrane = (
                membrane_channel is not None
                and 0 <= int(membrane_channel) < image.shape[0]
                and int(membrane_channel) != nuc_idx
            )
            mem_idx = int(membrane_channel) if has_membrane else None
        else:
            nuc_idx = 0
            has_membrane = False
            mem_idx = None

        if (
            self._cached_model_cyto is None
            or self._cached_cyto_type != model_type
            or self._cached_device != actual_device
        ):
            try:
                self._cached_model_cyto = self._build_model(
                    models, torch_device, use_gpu, model_type, is_v4
                )
            except Exception as exc:
                if actual_device == "mps":
                    log_warn(
                        "IRIS-Viewer",
                        "Segmentation",
                        f"Cellpose MPS init failed ({exc}), falling back to CPU.",
                    )
                    torch_device, use_gpu, actual_device = _resolve_torch_device(torch, "cpu")
                    self._cached_model_cyto = self._build_model(
                        models, torch_device, use_gpu, model_type, is_v4
                    )
                else:
                    raise
            self._cached_cyto_type = model_type
            self._cached_device = actual_device

        mem_raw = np.asarray(image[mem_idx]) if (has_membrane and mem_idx is not None) else None
        nuc_raw = np.asarray(image[nuc_idx] if image.ndim == 3 else image)
        nuc_proc = self._preprocess_channel(
            nuc_raw, gaussian_sigma=gaussian_sigma, norm_pmin=norm_pmin, norm_pmax=norm_pmax
        )

        base_eval_kw: Dict[str, Any] = {
            "diameter": diameter,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
            "batch_size": batch_size,
            "tile_overlap": tile_overlap,
            "resample": resample,
            "augment": augment,
            "niter": niter,
            "max_size_fraction": max_size_fraction,
            "bsize": bsize,
            "stitch_threshold": stitch_threshold,
            "normalize": cellpose_normalize,
        }
        base_eval_kw.update(extra_kw)

        def _filter_eval_kw(model_obj: Any, raw_kw: Dict[str, Any]) -> Dict[str, Any]:
            eval_sig = inspect.signature(model_obj.eval)
            if any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in eval_sig.parameters.values()
            ):
                return dict(raw_kw)
            accepted = set(eval_sig.parameters.keys())
            return {k: v for k, v in raw_kw.items() if k in accepted}

        def _eval_and_filter(eval_img: np.ndarray, eval_kw: Dict[str, Any]) -> np.ndarray:
            nonlocal torch_device, use_gpu, actual_device
            call_kw = _filter_eval_kw(self._cached_model_cyto, eval_kw)
            try:
                res = self._cached_model_cyto.eval(eval_img, **call_kw)
            except Exception as exc:
                if actual_device == "mps":
                    log_warn(
                        "IRIS-Viewer",
                        "Segmentation",
                        f"Cellpose MPS eval failed ({exc}), retrying on CPU...",
                    )
                    torch_device, use_gpu, actual_device = _resolve_torch_device(torch, "cpu")
                    self._cached_model_cyto = self._build_model(
                        models, torch_device, use_gpu, model_type, is_v4
                    )
                    self._cached_device = actual_device
                    call_kw = _filter_eval_kw(self._cached_model_cyto, eval_kw)
                    res = self._cached_model_cyto.eval(eval_img, **call_kw)
                else:
                    raise

            raw_masks = res[0] if isinstance(res, tuple) else res
            if raw_masks is None:
                raw_masks = np.zeros(nuc_raw.shape, dtype=np.uint32)
            raw_masks = np.asarray(raw_masks)
            if raw_masks.ndim > 2:
                raw_masks = np.squeeze(raw_masks)
            filtered = self._filter_mask_by_area(
                raw_masks, min_area=min_cell_area, max_area=max_cell_area
            )
            return filtered.astype(np.uint32)

        t0 = time.perf_counter()
        try:
            run_wc_pass = has_membrane and model_type != "nuclei" and mem_raw is not None
            if skip_nuclear and run_wc_pass:
                nuc_masks_u32 = np.zeros(nuc_raw.shape, dtype=np.uint32)
            else:
                nuc_eval_kw = dict(base_eval_kw)
                if not is_v4:
                    nuc_eval_kw["channels"] = [0, 0]

                iris_log(
                    "IRIS-Viewer",
                    "Segmentation",
                    f"Cellpose '{model_type}' nuclear pass "
                    f"(shape={nuc_proc.shape}, device={actual_device})...",
                )
                nuc_masks_u32 = _eval_and_filter(nuc_proc, nuc_eval_kw)

            if run_wc_pass:
                mem_proc = self._preprocess_channel(
                    mem_raw,
                    gaussian_sigma=gaussian_sigma,
                    norm_pmin=norm_pmin,
                    norm_pmax=norm_pmax,
                )
                wc_img = np.stack([mem_proc, nuc_proc], axis=0 if is_v4 else -1)
                wc_eval_kw = dict(base_eval_kw)
                if is_v4:
                    wc_eval_kw["channel_axis"] = 0
                else:
                    wc_eval_kw["channels"] = [1, 2]

                iris_log(
                    "IRIS-Viewer",
                    "Segmentation",
                    f"Cellpose '{model_type}' whole-cell pass "
                    f"(shape={wc_img.shape}, device={actual_device})...",
                )
                cyto_masks_u32 = _eval_and_filter(wc_img, wc_eval_kw)
                if not skip_nuclear:
                    nuc_masks_u32, cyto_masks_u32 = (
                        self.reconcile_nuclear_and_cytoplasm_masks(
                            nuc_masks_u32, cyto_masks_u32
                        )
                    )
                result = {"nuclear": nuc_masks_u32, "cytoplasm": cyto_masks_u32}
            else:
                cyto_masks_u32 = (
                    self._expand_nuclear_to_cytoplasm(
                        nuc_masks_u32,
                        expand_radius,
                        mode=expansion_mode,
                        membrane_image=mem_raw,
                    ).astype(np.uint32)
                    if expand_radius > 0
                    else None
                )
                result = {"nuclear": nuc_masks_u32, "cytoplasm": cyto_masks_u32}

            n_nuc = int((np.unique(nuc_masks_u32) > 0).sum())
            n_cyto = (
                int((np.unique(result["cytoplasm"]) > 0).sum())
                if result["cytoplasm"] is not None
                else 0
            )
            iris_log(
                "IRIS-Viewer",
                "Segmentation",
                f"Cellpose '{model_type}' complete in {time.perf_counter() - t0:.2f}s "
                f"(N={n_nuc} nuclear, {n_cyto} cytoplasm)",
            )

        finally:
            if use_gpu:
                if actual_device == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()
                elif actual_device == "cuda" and hasattr(torch, "cuda"):
                    torch.cuda.empty_cache()
            gc.collect()

        return result
