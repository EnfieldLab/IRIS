"""
CellSAM Segmentation Backend
CellSAM is a foundation model for cell segmentation built on SAM (Segment Anything Model)
with a CellFinder object detector. It segments whole cells (not specifically nuclear vs cytoplasm),
generalizing across imaging modalities and cell types.
"""
import importlib.util
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


class CellSAMBackend(SegmentationBackend):
    name = "CellSAM"
    supports_cytoplasm = True

    def is_available(self) -> bool:
        return _has_modules("cellSAM", "torch")

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "token": "",
            "bbox_threshold": 0.4,
            "use_all_channels": False,
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
        import os

        token = kwargs.get("token", "")
        if token:
            os.environ["DEEPCELL_ACCESS_TOKEN"] = token

        bbox_threshold = float(kwargs.get("bbox_threshold", 0.4))
        use_all_channels = bool(kwargs.get("use_all_channels", False))
        gaussian_sigma = float(kwargs.get("gaussian_sigma", 0.0))
        norm_pmin = kwargs.get("norm_pmin", 1.0)
        norm_pmax = kwargs.get("norm_pmax", 99.8)
        min_cell_area = int(kwargs.get("min_cell_area", 0) or 0)
        max_cell_area = int(kwargs.get("max_cell_area", 0) or 0)

        from cellSAM import get_model, segment_cellular_image

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
            if idx is not None and image.ndim == 3 and idx < image.shape[0]:
                return _prep_u8(np.asarray(image[idx]))
            base_shape = image.shape[-2:]
            return np.zeros(base_shape, dtype=np.uint8)

        nuc_u8 = _prep_u8(np.asarray(image[nuclear_channel] if image.ndim == 3 else image))
        if c1_idx is not None or c2_idx is not None or c3_idx is not None:
            composite = np.stack([get_ch(c1_idx), get_ch(c2_idx), get_ch(c3_idx)], axis=-1)
        elif use_all_channels and image.ndim == 3 and image.shape[0] >= 2:
            other_indices = [i for i in range(image.shape[0]) if i != nuclear_channel]
            if membrane_channel is not None and membrane_channel != nuclear_channel:
                mem_u8 = _prep_u8(np.asarray(image[membrane_channel]))
            else:
                mem_u8 = np.max(
                    np.stack([_prep_u8(np.asarray(image[i])) for i in other_indices], axis=0),
                    axis=0,
                )
            composite = np.stack([np.zeros_like(nuc_u8), mem_u8, nuc_u8], axis=-1)
        elif (
            image.ndim == 3
            and membrane_channel is not None
            and membrane_channel != nuclear_channel
        ):
            mem_u8 = _prep_u8(np.asarray(image[membrane_channel]))
            composite = np.stack([np.zeros_like(nuc_u8), mem_u8, nuc_u8], axis=-1)
        else:
            composite = np.stack([nuc_u8, nuc_u8, nuc_u8], axis=-1)

        t0 = time.perf_counter()
        iris_log(
            "IRIS-Viewer",
            "Segmentation",
            f"CellSAM running (shape={composite.shape}, device={device})...",
        )
        mask, _, _ = segment_cellular_image(
            composite, model, bbox_threshold=bbox_threshold, device=device
        )
        mask = self._filter_mask_by_area(
            mask, min_area=min_cell_area, max_area=max_cell_area
        ).astype(np.uint32)
        n_cells = int((np.unique(mask) > 0).sum())
        iris_log(
            "IRIS-Viewer",
            "Segmentation",
            f"CellSAM complete in {time.perf_counter() - t0:.2f}s (N={n_cells} cells)",
        )
        return {
            "nuclear": mask,
            "cytoplasm": mask,
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
