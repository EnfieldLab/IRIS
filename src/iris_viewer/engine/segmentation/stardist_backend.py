from typing import Any, Dict, Optional

import numpy as np

from .base import SegmentationBackend


class StardistBackend(SegmentationBackend):
    name = "StarDist"
    supports_cytoplasm = False

    def __init__(self):
        # Initialize caching variables
        self._cached_model = None
        self._cached_model_name = None

    def is_available(self) -> bool:
        try:
            from stardist.models import StarDist2D  # noqa: F401
            return True
        except (ImportError, ModuleNotFoundError, Exception):
            return False

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'model_name': '2D_versatile_fluo',
            'prob_thresh': 0.5,
            'nms_thresh': 0.4,
            'normalize_input': True,
            'norm_pmin': 1.0,
            'norm_pmax': 99.8,
        }

    def segment(
        self,
        image: np.ndarray,
        nuclear_channel: int,
        membrane_channel: Optional[int],
        diameter: Optional[float],
        device: str,
        expand_radius: int = 0,
        **kwargs
    ) -> Dict[str, Optional[np.ndarray]]:

        from csbdeep.utils import normalize
        from stardist.models import StarDist2D

        model_name = kwargs.get('model_name', '2D_versatile_fluo')
        prob_thresh = kwargs.get('prob_thresh', 0.5)
        nms_thresh = kwargs.get('nms_thresh', 0.4)

        do_normalize = kwargs.get('normalize_input', True)
        pmin = kwargs.get('norm_pmin', 1.0)
        pmax = kwargs.get('norm_pmax', 99.8)

        # --- MODEL CACHING LOGIC ---
        if self._cached_model is None or self._cached_model_name != model_name:
            self._cached_model = StarDist2D.from_pretrained(model_name)
            self._cached_model_name = model_name

        if image.ndim == 3:
            nuc_img = np.asarray(image[nuclear_channel]).astype(np.float32)
        else:
            nuc_img = np.asarray(image).astype(np.float32)

        if do_normalize:
            nuc_img = normalize(nuc_img, pmin, pmax, axis=(0, 1))

        # Compute optimal number of tiles based on 1024x1024 chunks
        h, w = nuc_img.shape
        n_tiles = (max(1, int(np.ceil(h / 1024))), max(1, int(np.ceil(w / 1024))))

        print(f"[IRIS-Viewer] StarDist Execution: Running '{model_name}'...")

        # Execute with cached model
        labels, details = self._cached_model.predict_instances(
            nuc_img,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            n_tiles=n_tiles
        )

        result = {'nuclear': labels.astype(np.uint32)}

        if expand_radius > 0:
            result['cytoplasm'] = self._expand_nuclear_to_cytoplasm(labels, expand_radius)
        else:
            result['cytoplasm'] = None

        return result
