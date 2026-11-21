"""StarDist Segmentation Backend."""
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


class StardistBackend(SegmentationBackend):
    name = "StarDist"
    supports_cytoplasm = False

    def __init__(self):
        self._cached_model = None
        self._cached_model_name = None

    def is_available(self) -> bool:
        return _has_modules("stardist", "tensorflow")

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "model_name": "2D_versatile_fluo",
            "prob_thresh": 0.5,
            "nms_thresh": 0.4,
            "normalize_input": True,
            "norm_pmin": 1.0,
            "norm_pmax": 99.8,
            "gaussian_sigma": 0.0,
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
        from stardist.models import StarDist2D

        model_name = kwargs.get("model_name", "2D_versatile_fluo")
        prob_thresh = float(kwargs.get("prob_thresh", 0.5))
        nms_thresh = float(kwargs.get("nms_thresh", 0.4))
        do_normalize = bool(kwargs.get("normalize_input", True))
        pmin = kwargs.get("norm_pmin", 1.0)
        pmax = kwargs.get("norm_pmax", 99.8)
        gaussian_sigma = float(kwargs.get("gaussian_sigma", 0.0))
        min_cell_area = int(kwargs.get("min_cell_area", 0) or 0)
        max_cell_area = int(kwargs.get("max_cell_area", 0) or 0)
        expansion_mode = str(kwargs.get("expansion_mode", "distance"))

        if self._cached_model is None or self._cached_model_name != model_name:
            self._cached_model = StarDist2D.from_pretrained(model_name)
            self._cached_model_name = model_name

        nuc_raw = np.asarray(image[nuclear_channel] if image.ndim == 3 else image)
        mem_raw = (
            np.asarray(image[membrane_channel])
            if (
                image.ndim == 3
                and membrane_channel is not None
                and membrane_channel != nuclear_channel
            )
            else None
        )

        nuc_img = self._preprocess_channel(
            nuc_raw,
            gaussian_sigma=gaussian_sigma,
            norm_pmin=pmin if do_normalize else None,
            norm_pmax=pmax if do_normalize else None,
        )

        h, w = nuc_img.shape
        n_tiles = (max(1, int(np.ceil(h / 1024))), max(1, int(np.ceil(w / 1024))))

        t0 = time.perf_counter()
        iris_log(
            "IRIS-Viewer",
            "Segmentation",
            f"StarDist '{model_name}' running (shape={nuc_img.shape}, tiles={n_tiles})...",
        )

        labels, _ = self._cached_model.predict_instances(
            nuc_img,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            n_tiles=n_tiles,
        )

        labels = self._filter_mask_by_area(
            labels, min_area=min_cell_area, max_area=max_cell_area
        ).astype(np.uint32)

        if expand_radius > 0:
            cyto = self._expand_nuclear_to_cytoplasm(
                labels,
                expand_radius,
                mode=expansion_mode,
                membrane_image=mem_raw,
            ).astype(np.uint32)
        else:
            cyto = None

        n_nuc = int((np.unique(labels) > 0).sum())
        n_cyto = int((np.unique(cyto) > 0).sum()) if cyto is not None else 0
        iris_log(
            "IRIS-Viewer",
            "Segmentation",
            f"StarDist '{model_name}' complete in {time.perf_counter() - t0:.2f}s "
            f"(N={n_nuc} nuclear, {n_cyto} cytoplasm)",
        )

        return {"nuclear": labels, "cytoplasm": cyto}
