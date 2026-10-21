"""
Abstract base class for cell segmentation backends.
All segmentation methods (Cellpose, StarDist, CellSAM, InstanSeg) implement this interface,
ensuring a uniform contract for the consensus engine and the controller.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class SegmentationBackend(ABC):
    """
    Contract:
        - name:               Human-readable method name for UI display
        - supports_cytoplasm:  Whether this backend natively segments whole cells (cytoplasm)
        - is_available():      Runtime check for installed dependencies
        - segment():           Produces label masks from raw image data
        - get_default_params(): Default parameter dictionary for UI population
    """

    name: str = "BaseBackend"
    supports_cytoplasm: bool = False

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if this backend's dependencies are installed and importable."""
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
        **kwargs
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Run segmentation on the input image.

        Parameters
        ----------
        image : np.ndarray
            Multi-channel image array of shape (C, H, W).
        nuclear_channel : int
            Index of the nuclear/DAPI channel.
        membrane_channel : int or None
            Index of the membrane/cytoplasm channel, if available.
        diameter : float or None
            Approximate cell diameter in pixels. None = auto-detect.
        device : str
            Compute device: 'cpu', 'cuda', or 'mps'.
        expand_radius : int
            For nuclear-only backends, optionally dilate the nuclear mask
            by this many pixels to approximate a cytoplasm mask.

        Returns
        -------
        dict with keys:
            'nuclear'    : np.ndarray (H, W) — integer label mask for nuclei
            'cytoplasm'  : np.ndarray (H, W) or None — integer label mask for whole cells
        """
        ...

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """
        Returns a dictionary of default parameters for this backend.
        Keys correspond to kwargs accepted by segment().
        Used to populate UI fields.
        """
        ...

    def _expand_nuclear_to_cytoplasm(
        self,
        nuclear_mask: np.ndarray,
        expand_radius: int
    ) -> np.ndarray:
        """
        Utility: dilate each labeled nucleus outward by `expand_radius` pixels
        to approximate a cytoplasm mask. Conflicts are resolved by nearest-label
        assignment using a distance transform.
        """
        if expand_radius <= 0:
            return nuclear_mask.copy()

        from skimage.segmentation import expand_labels

        expanded = expand_labels(nuclear_mask, distance=expand_radius)
        return expanded
