"""
CellSAM Segmentation Backend
CellSAM is a foundation model for cell segmentation built on SAM (Segment Anything Model)
with a CellFinder object detector. It segments whole cells (not specifically nuclear vs cytoplasm),
generalizing across imaging modalities and cell types.
"""
from typing import Any, Dict, Optional

import numpy as np

from .base import SegmentationBackend


class CellSAMBackend(SegmentationBackend):
    name = "CellSAM"
    supports_cytoplasm = True      # Segments whole cell bodies

    def is_available(self) -> bool:
        try:
            import cellSAM  # noqa: F401
            return True
        except ImportError:
            return False

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'use_all_channels': False,
        }

    def segment(
        self, image: np.ndarray, nuclear_channel: int, membrane_channel: Optional[int],
        diameter: Optional[float], device: str, expand_radius: int = 0, **kwargs
    ) -> Dict[str, Optional[np.ndarray]]:

        import os
        token = kwargs.get('token', '')
        if token:
            os.environ["DEEPCELL_ACCESS_TOKEN"] = token

        from cellSAM import get_model, segment_cellular_image

        model = get_model()
        model = model.to(device)

        c1_idx = kwargs.get('chan1_idx')
        c2_idx = kwargs.get('chan2_idx')
        c3_idx = kwargs.get('chan3_idx')

        def get_ch(idx):
            if idx is not None and idx < image.shape[0]:
                return self._normalize_channel(np.asarray(image[idx]))
            return np.zeros_like(np.asarray(image[0] if image.ndim == 3 else image), dtype=np.uint8)

        # Use explicitly mapped RGB channels
        if c1_idx is not None or c2_idx is not None or c3_idx is not None:
            composite = np.stack([get_ch(c1_idx), get_ch(c2_idx), get_ch(c3_idx)], axis=-1)
        else:
            # Fallback to nuclear channel if user didn't pick any specific RGB layers
            nuc = self._normalize_channel(np.asarray(image[nuclear_channel] if image.ndim == 3 else image))
            composite = np.stack([nuc, nuc, nuc], axis=-1)

        print("[IRIS-Viewer] CellSAM Execution: Segmenting...")
        mask, _, _ = segment_cellular_image(composite, model, device=device)
        return {
            'nuclear': mask.astype(np.uint32),
            'cytoplasm': mask.astype(np.uint32),
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
