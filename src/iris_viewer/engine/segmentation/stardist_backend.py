"""StarDist Segmentation Backend."""
import gc
import importlib.util
import inspect
import os
import sys
import time
from pathlib import Path
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
            "use_model_default_thresholds": False,
            "scale": 1.0,
            "sparse": True,
            "n_tiles_grid": 0,
            "normalize_input": True,
            "norm_pmin": 1.0,
            "norm_pmax": 99.8,
            "gaussian_sigma": 0.0,
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
        from stardist.models import StarDist2D

        model_name_raw = str(kwargs.get("model_name") or "2D_versatile_fluo").strip()
        expanded_dir = os.path.expanduser(model_name_raw)
        is_local_dir = os.path.isdir(expanded_dir)
        cache_key = expanded_dir if is_local_dir else model_name_raw

        use_defaults = bool(kwargs.get("use_model_default_thresholds", False))
        raw_prob = kwargs.get("prob_thresh", 0.5)
        raw_nms = kwargs.get("nms_thresh", 0.4)
        prob_thresh = None if (use_defaults or raw_prob is None) else float(raw_prob)
        nms_thresh = None if (use_defaults or raw_nms is None) else float(raw_nms)
        raw_scale = kwargs.get("scale", None)
        scale = float(raw_scale) if (raw_scale is not None and float(raw_scale) > 0) else None
        sparse = bool(kwargs.get("sparse", True))
        n_tiles_grid = int(kwargs.get("n_tiles_grid", 0) or 0)

        do_normalize = bool(kwargs.get("normalize_input", True))
        pmin = kwargs.get("norm_pmin", 1.0)
        pmax = kwargs.get("norm_pmax", 99.8)
        gaussian_sigma = float(kwargs.get("gaussian_sigma", 0.0))
        min_cell_area = int(kwargs.get("min_cell_area", 0) or 0)
        max_cell_area = int(kwargs.get("max_cell_area", 0) or 0)
        expand_radius = int(expand_radius or 0)
        expansion_mode = str(kwargs.get("expansion_mode", "distance"))
        extra_kw = self._parse_extra_kwargs(kwargs.get("extra_kwargs"))

        if self._cached_model is None or self._cached_model_name != cache_key:
            if is_local_dir:
                local_path = Path(expanded_dir)
                self._cached_model = StarDist2D(
                    None,
                    name=local_path.name,
                    basedir=str(local_path.parent),
                )
            else:
                self._cached_model = StarDist2D.from_pretrained(model_name_raw)
            self._cached_model_name = cache_key

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

        nuc_img = np.ascontiguousarray(
            self._preprocess_channel(
                nuc_raw,
                gaussian_sigma=gaussian_sigma,
                norm_pmin=pmin if do_normalize else None,
                norm_pmax=pmax if do_normalize else None,
            ),
            dtype=np.float32,
        )

        h, w = nuc_img.shape
        if n_tiles_grid > 0:
            n_tiles = (n_tiles_grid, n_tiles_grid)
        else:
            n_tiles = (max(1, int(np.ceil(h / 1024))), max(1, int(np.ceil(w / 1024))))

        predict_kw: Dict[str, Any] = {
            "prob_thresh": prob_thresh,
            "nms_thresh": nms_thresh,
            "n_tiles": n_tiles,
            "sparse": sparse,
            "scale": scale,
        }
        predict_kw.update(extra_kw)

        pred_sig = inspect.signature(self._cached_model.predict_instances)
        if not any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in pred_sig.parameters.values()
        ):
            accepted = set(pred_sig.parameters.keys())
            predict_kw = {k: v for k, v in predict_kw.items() if k in accepted}

        t0 = time.perf_counter()
        iris_log(
            "IRIS-Viewer",
            "Segmentation",
            f"StarDist '{model_name_raw}' running (shape={nuc_img.shape}, tiles={n_tiles})...",
        )

        labels, _ = self._cached_model.predict_instances(nuc_img, **predict_kw)
        gc.collect()

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
            f"StarDist '{model_name_raw}' complete in {time.perf_counter() - t0:.2f}s "
            f"(N={n_nuc} nuclear, {n_cyto} cytoplasm)",
        )

        return {"nuclear": labels, "cytoplasm": cyto}
