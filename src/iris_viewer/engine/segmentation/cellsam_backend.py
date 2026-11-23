"""
CellSAM Segmentation Backend
CellSAM is a foundation model for cell segmentation built on SAM (Segment Anything Model)
with a CellFinder object detector. It segments whole cells (not specifically nuclear vs cytoplasm),
generalizing across imaging modalities and cell types.
"""
import importlib.util
import inspect
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

from iris_common.logging import iris_log

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


def _resolve_use_wsi(raw_use_wsi: Any, height: int, width: int) -> bool:
    """Resolve CellSAM WSI mode from 'auto', 'true'/'false', or bool."""
    if isinstance(raw_use_wsi, bool):
        return raw_use_wsi
    if raw_use_wsi is None:
        return max(height, width) > 1024
    text = str(raw_use_wsi).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    return max(height, width) > 1024


class CellSAMBackend(SegmentationBackend):
    name = "CellSAM"
    supports_cytoplasm = True

    def is_available(self) -> bool:
        return _has_modules("cellSAM", "torch")

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "token": "",
            "model_path": "",
            "bbox_threshold": 0.4,
            "normalize": True,
            "postprocess": False,
            "remove_boundaries": False,
            "fast": False,
            "low_contrast_enhancement": False,
            "use_wsi": "auto",
            "gauge_cell_size": False,
            "block_size": 400,
            "overlap": 56,
            "iou_depth": 56,
            "iou_threshold": 0.5,
            "use_all_channels": False,
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
        if image.ndim < 2 or image.shape[-2] == 0 or image.shape[-1] == 0:
            shape = image.shape[-2:] if image.ndim >= 2 else (0, 0)
            return {"nuclear": np.zeros(shape, dtype=np.uint32), "cytoplasm": None}
        if image.ndim > 3:
            raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}.")

        token = str(kwargs.get("token", "") or "").strip()
        if token:
            os.environ["DEEPCELL_ACCESS_TOKEN"] = token

        model_path_raw = str(kwargs.get("model_path", "") or "").strip()
        model_path = os.path.expanduser(model_path_raw) if model_path_raw else ""
        bbox_threshold = float(kwargs.get("bbox_threshold", 0.4))
        normalize = bool(kwargs.get("normalize", True))
        postprocess = bool(kwargs.get("postprocess", False))
        remove_boundaries = bool(kwargs.get("remove_boundaries", False))
        fast = bool(kwargs.get("fast", False))
        low_contrast_enhancement = bool(kwargs.get("low_contrast_enhancement", False))
        raw_use_wsi = kwargs.get("use_wsi", "auto")
        gauge_cell_size = bool(kwargs.get("gauge_cell_size", False))
        block_size = max(64, int(kwargs.get("block_size", 400) or 400))
        overlap = max(8, int(kwargs.get("overlap", 56) or 56))
        iou_depth = max(8, int(kwargs.get("iou_depth", 56) or 56))
        iou_threshold = float(kwargs.get("iou_threshold", 0.5))
        use_all_channels = bool(kwargs.get("use_all_channels", False))
        skip_nuclear = bool(kwargs.get("skip_nuclear", False))
        gaussian_sigma = float(kwargs.get("gaussian_sigma", 0.0))
        norm_pmin = kwargs.get("norm_pmin", 1.0)
        norm_pmax = kwargs.get("norm_pmax", 99.8)
        min_cell_area = int(kwargs.get("min_cell_area", 0) or 0)
        max_cell_area = int(kwargs.get("max_cell_area", 0) or 0)
        expand_radius = int(expand_radius or 0)
        expansion_mode = str(kwargs.get("expansion_mode", "distance"))
        extra_kw = self._parse_extra_kwargs(kwargs.get("extra_kwargs"))

        import cellSAM

        get_model = cellSAM.get_model
        segment_cellular_image = cellSAM.segment_cellular_image
        get_local_model = getattr(cellSAM, "get_local_model", None)

        if model_path and os.path.isfile(model_path) and callable(get_local_model):
            model = get_local_model(model_path)
        else:
            try:
                model = get_model()
            except ValueError as exc:
                if not token and not os.environ.get("DEEPCELL_ACCESS_TOKEN"):
                    raise ValueError(
                        "DEEPCELL_ACCESS_TOKEN not found. Enter your DeepCell Token in the "
                        "CellSAM step settings or export DEEPCELL_ACCESS_TOKEN."
                    ) from exc
                raise
        model = model.to(device)

        h, w = image.shape[-2], image.shape[-1]
        use_wsi = _resolve_use_wsi(raw_use_wsi, h, w)

        c1_idx = kwargs.get("chan1_idx")
        c2_idx = kwargs.get("chan2_idx")
        c3_idx = kwargs.get("chan3_idx")

        def _prep_u8(ch_arr: np.ndarray) -> np.ndarray:
            proc = self._preprocess_channel(
                ch_arr,
                gaussian_sigma=gaussian_sigma,
                norm_pmin=norm_pmin,
                norm_pmax=norm_pmax,
            )
            if norm_pmin is None or norm_pmax is None or float(norm_pmax) <= float(norm_pmin):
                p_min = float(np.min(proc))
                p_max = float(np.max(proc))
                if p_max > p_min:
                    proc = (proc - p_min) / (p_max - p_min)
                else:
                    proc = np.zeros_like(proc, dtype=np.float32)
            return np.clip(proc * 255.0, 0, 255).astype(np.uint8)

        def get_ch(idx: Optional[int]) -> np.ndarray:
            if idx is not None and image.ndim == 3 and 0 <= int(idx) < image.shape[0]:
                return _prep_u8(np.asarray(image[int(idx)]))
            return np.zeros((h, w), dtype=np.uint8)

        nuc_idx = (
            max(0, min(int(nuclear_channel), image.shape[0] - 1))
            if image.ndim == 3
            else 0
        )
        nuc_u8 = _prep_u8(np.asarray(image[nuc_idx] if image.ndim == 3 else image))
        blank_u8 = np.zeros_like(nuc_u8)

        seg_kw: Dict[str, Any] = {
            "bbox_threshold": bbox_threshold,
            "normalize": normalize,
            "postprocess": postprocess,
            "remove_boundaries": remove_boundaries,
            "fast": fast,
            "device": device,
        }
        seg_kw.update(extra_kw)

        seg_sig = inspect.signature(segment_cellular_image)
        if not any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in seg_sig.parameters.values()
        ):
            accepted = set(seg_sig.parameters.keys())
            seg_kw = {k: v for k, v in seg_kw.items() if k in accepted}

        def _apply_low_contrast(comp: np.ndarray) -> np.ndarray:
            if not low_contrast_enhancement:
                return comp
            comp_f = comp.astype(np.float32) / 255.0
            try:
                from cellSAM.utils import enhance_low_contrast

                enh = enhance_low_contrast(comp_f)
                return np.clip(np.asarray(enh, dtype=np.float32) * 255.0, 0, 255).astype(
                    np.uint8
                )
            except Exception:
                from skimage.exposure import equalize_adapthist

                out = np.zeros_like(comp_f)
                for ch_i in range(comp_f.shape[-1]):
                    plane = comp_f[..., ch_i]
                    if float(plane.max()) > float(plane.min()):
                        out[..., ch_i] = equalize_adapthist(plane, clip_limit=0.01)
                return np.clip(out * 255.0, 0, 255).astype(np.uint8)

        def _run_cellsam_pass(comp: np.ndarray) -> np.ndarray:
            comp_in = _apply_low_contrast(comp)
            raw_mask: Optional[np.ndarray] = None
            if use_wsi:
                try:
                    import dask.array as da
                    from cellSAM.wsi import segment_wsi

                    eff_iou_depth = min(iou_depth, overlap)
                    wsi_kw = dict(seg_kw)
                    wsi_kw["model"] = model
                    if gauge_cell_size:
                        pipeline_mod = sys.modules.get("cellSAM.cellsam_pipeline")
                        gaging_fn = (
                            getattr(pipeline_mod, "use_cellsize_gaging", None)
                            if pipeline_mod is not None
                            else None
                        )
                        if callable(gaging_fn):
                            norm_fn = getattr(pipeline_mod, "normalize_image", None)
                            gaging_arr = comp_in.astype(np.float32)
                            if normalize and callable(norm_fn):
                                gaging_arr = norm_fn(gaging_arr)
                            inp_da = da.from_array(gaging_arr, chunks=block_size)
                            raw_mask = gaging_fn(
                                inp_da,
                                model,
                                device,
                                block_size=block_size,
                                overlap=overlap,
                                iou_depth=eff_iou_depth,
                                iou_threshold=iou_threshold,
                                bbox_threshold=bbox_threshold,
                            )
                    if raw_mask is None:
                        wsi_res = segment_wsi(
                            comp_in,
                            block_size,
                            overlap,
                            eff_iou_depth,
                            iou_threshold,
                            **wsi_kw,
                        )
                        raw_mask = wsi_res.compute() if hasattr(wsi_res, "compute") else wsi_res
                except Exception:
                    raw_mask = None

            if raw_mask is None:
                raw_mask, _, _ = segment_cellular_image(comp_in, model, **seg_kw)

            filtered = self._filter_mask_by_area(
                raw_mask, min_area=min_cell_area, max_area=max_cell_area
            )
            return filtered.astype(np.uint32)

        t0 = time.perf_counter()
        if c1_idx is not None or c2_idx is not None or c3_idx is not None:
            explicit_comp = np.stack([get_ch(c1_idx), get_ch(c2_idx), get_ch(c3_idx)], axis=-1)
            iris_log(
                "IRIS-Viewer",
                "Segmentation",
                f"CellSAM running explicit channels (shape={explicit_comp.shape}, "
                f"device={device}, wsi={use_wsi})...",
            )
            explicit_mask = _run_cellsam_pass(explicit_comp)
            if skip_nuclear:
                nuc_mask = np.zeros((h, w), dtype=np.uint32)
                cyto_mask: Optional[np.ndarray] = explicit_mask
            elif c3_idx is not None:
                nuc_comp = np.stack([blank_u8, nuc_u8, blank_u8], axis=-1)
                nuc_mask = _run_cellsam_pass(nuc_comp)
                nuc_mask, cyto_mask = self.reconcile_nuclear_and_cytoplasm_masks(
                    nuc_mask, explicit_mask
                )
            else:
                nuc_mask = explicit_mask
                cyto_mask = (
                    self._expand_nuclear_to_cytoplasm(
                        nuc_mask, expand_radius, mode=expansion_mode, membrane_image=None
                    ).astype(np.uint32)
                    if expand_radius > 0
                    else None
                )
        else:
            has_membrane = (
                image.ndim == 3
                and membrane_channel is not None
                and 0 <= int(membrane_channel) < image.shape[0]
                and int(membrane_channel) != nuc_idx
            )
            mem_raw: Optional[np.ndarray] = None
            mem_u8: Optional[np.ndarray] = None
            if has_membrane:
                mem_raw = np.asarray(image[int(membrane_channel)])
                mem_u8 = _prep_u8(mem_raw)
            elif use_all_channels and image.ndim == 3 and image.shape[0] >= 2:
                other_indices = [i for i in range(image.shape[0]) if i != nuc_idx]
                if other_indices:
                    mem_raw = np.max(
                        np.stack([np.asarray(image[i]) for i in other_indices], axis=0),
                        axis=0,
                    )
                    mem_u8 = np.max(
                        np.stack(
                            [_prep_u8(np.asarray(image[i])) for i in other_indices], axis=0
                        ),
                        axis=0,
                    )

            has_wc_pass = mem_u8 is not None
            if skip_nuclear and has_wc_pass:
                nuc_mask = np.zeros((h, w), dtype=np.uint32)
            else:
                # Multiplexed CellSAM channel ordering: (blank, nuclear, whole-cell)
                nuc_comp = np.stack([blank_u8, nuc_u8, blank_u8], axis=-1)
                iris_log(
                    "IRIS-Viewer",
                    "Segmentation",
                    f"CellSAM nuclear pass (shape={nuc_comp.shape}, "
                    f"device={device}, wsi={use_wsi})...",
                )
                nuc_mask = _run_cellsam_pass(nuc_comp)

            if has_wc_pass:
                # Multiplexed CellSAM channel ordering: (blank, nuclear, whole-cell)
                wc_comp = np.stack([blank_u8, nuc_u8, mem_u8], axis=-1)
                iris_log(
                    "IRIS-Viewer",
                    "Segmentation",
                    f"CellSAM whole-cell pass (shape={wc_comp.shape}, "
                    f"device={device}, wsi={use_wsi})...",
                )
                wc_mask = _run_cellsam_pass(wc_comp)
                if not skip_nuclear:
                    nuc_mask, cyto_mask = self.reconcile_nuclear_and_cytoplasm_masks(
                        nuc_mask, wc_mask
                    )
                else:
                    cyto_mask = wc_mask
            else:
                cyto_mask = (
                    self._expand_nuclear_to_cytoplasm(
                        nuc_mask,
                        expand_radius,
                        mode=expansion_mode,
                        membrane_image=mem_raw,
                    ).astype(np.uint32)
                    if expand_radius > 0
                    else None
                )

        n_nuc = int((np.unique(nuc_mask) > 0).sum())
        n_cyto = int((np.unique(cyto_mask) > 0).sum()) if cyto_mask is not None else 0
        iris_log(
            "IRIS-Viewer",
            "Segmentation",
            f"CellSAM complete in {time.perf_counter() - t0:.2f}s "
            f"(N={n_nuc} nuclear, {n_cyto} cytoplasm)",
        )
        return {
            "nuclear": nuc_mask,
            "cytoplasm": cyto_mask,
        }

    @staticmethod
    def _normalize_channel(channel: np.ndarray) -> np.ndarray:
        """Normalize a single channel to [0, 255] uint8 for CellSAM input."""
        ch = channel.astype(np.float64)
        p_low, p_high = np.percentile(ch, [1, 99.9])
        if p_high <= p_low:
            p_high = p_low + 1.0
        ch = np.clip((ch - p_low) / (p_high - p_low), 0, 1)
        return (ch * 255).astype(np.uint8)
