"""
iris-common Segmentation Backend ABC
Canonical definition, re-exported by iris-viewer for backwards compatibility.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class SegmentationBackend(ABC):
    """
    Contract for cell segmentation backends.

    All backends (Cellpose, StarDist, CellSAM, InstanSeg) implement this
    interface, ensuring a uniform contract for the consensus engine.
    """

    name: str = "BaseBackend"
    supports_cytoplasm: bool = False

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend's dependencies are installed."""
        ...

    @abstractmethod
    def segment(
        self,
        image: np.ndarray,
        nuclear_channel: int,
        membrane_channel: Optional[int],
        diameter: Optional[float],
        device: str,
        expand_radius: int = 0,
        **kwargs,
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Run segmentation on the input image.

        Parameters
        ----------
        image : (C, H, W) ndarray
        nuclear_channel : channel index for DAPI / nuclear stain
        membrane_channel : channel index for membrane stain, or None
        diameter : approximate cell diameter in pixels (None = auto)
        device : 'cpu', 'cuda', or 'mps'
        expand_radius : dilate nuclear mask outward by this many pixels

        Returns
        -------
        dict with keys:
          'nuclear'   : (H, W) int32 label mask
          'cytoplasm' : (H, W) int32 label mask or None
        """
        ...

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Default parameter dict used to populate UI fields."""
        ...

    def _expand_nuclear_to_cytoplasm(
        self, nuclear_mask: np.ndarray, expand_radius: int
    ) -> np.ndarray:
        """Dilate each nucleus outward by *expand_radius* pixels."""
        if expand_radius <= 0:
            return nuclear_mask.copy()
        from skimage.segmentation import expand_labels
        return expand_labels(nuclear_mask, distance=expand_radius)
