"""
Abstract base class for cell segmentation backends.
All segmentation methods (Cellpose, StarDist, CellSAM, InstanSeg) implement this interface,
ensuring a uniform contract for the segmentation pipeline and disagreement QC engine.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class SegmentationBackend(ABC):
    """
    Contract:
        - name:                Human-readable method name for UI display
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
        **kwargs,
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
            For nuclear-only backends, optionally expand the nuclear mask
            by this many pixels to approximate a cytoplasm mask.

        Returns
        -------
        dict with keys:
            'nuclear'    : np.ndarray (H, W) - integer label mask for nuclei
            'cytoplasm'  : np.ndarray (H, W) or None - integer label mask for whole cells
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

    def _preprocess_channel(
        self,
        channel: np.ndarray,
        gaussian_sigma: float = 0.0,
        norm_pmin: Optional[float] = 1.0,
        norm_pmax: Optional[float] = 99.8,
    ) -> np.ndarray:
        """Convert a 2D channel to float32, optionally smooth and percentile-normalize to [0, 1]."""
        arr = np.asarray(channel, dtype=np.float32)
        if arr.size == 0:
            return arr

        if gaussian_sigma is not None and float(gaussian_sigma) > 0.0:
            from scipy.ndimage import gaussian_filter

            arr = gaussian_filter(arr, sigma=float(gaussian_sigma)).astype(np.float32, copy=False)

        if (
            norm_pmin is not None
            and norm_pmax is not None
            and float(norm_pmax) > float(norm_pmin)
        ):
            p_low, p_high = np.percentile(arr, [float(norm_pmin), float(norm_pmax)])
            if p_high <= p_low:
                arr = np.zeros_like(arr, dtype=np.float32)
            else:
                arr = np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)

        return arr

    def _filter_mask_by_area(
        self,
        mask: Optional[np.ndarray],
        min_area: int = 0,
        max_area: int = 0,
    ) -> Optional[np.ndarray]:
        """Filter labeled objects by pixel area using bincount and relabel sequentially."""
        if mask is None:
            return None

        labels = np.asarray(mask)
        if labels.size == 0:
            return np.zeros(labels.shape, dtype=np.uint32)

        min_a = int(min_area or 0)
        max_a = int(max_area or 0)
        if min_a <= 0 and max_a <= 0:
            return labels.astype(np.uint32, copy=False)

        if np.any(labels < 0):
            labels = np.maximum(labels, 0)

        max_lbl = int(labels.max())
        if max_lbl <= 0:
            return np.zeros(labels.shape, dtype=np.uint32)

        counts = np.bincount(labels.ravel().astype(np.int64, copy=False))
        keep = np.ones(counts.shape[0], dtype=bool)
        keep[0] = False
        if min_a > 0:
            keep &= counts >= min_a
        if max_a > 0:
            keep &= counts <= max_a

        remap = np.zeros(counts.shape[0], dtype=np.uint32)
        n_kept = int(np.count_nonzero(keep))
        if n_kept > 0:
            remap[keep] = np.arange(1, n_kept + 1, dtype=np.uint32)
        return remap[labels]

    def _expand_nuclear_to_cytoplasm(
        self,
        nuclear_mask: np.ndarray,
        expand_radius: int,
        mode: str = "distance",
        membrane_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Expand labeled nuclei outward by `expand_radius` pixels via distance or watershed.

        In ``"distance"`` mode, uses ``skimage.segmentation.expand_labels``.
        In ``"watershed"`` mode, runs marker-controlled watershed on an inverted
        smoothed ``membrane_image`` restricted to the expanded support mask
        (falling back to ``"distance"`` if ``membrane_image`` is None).
        """
        nuc = np.asarray(nuclear_mask)
        if expand_radius <= 0 or nuc.size == 0 or int(nuc.max()) == 0:
            return nuc.copy()

        from skimage.segmentation import expand_labels

        expanded = expand_labels(nuc, distance=int(expand_radius))
        mode_norm = (mode or "distance").strip().lower()
        if mode_norm != "watershed" or membrane_image is None:
            return expanded

        mem = np.asarray(membrane_image, dtype=np.float32)
        if mem.shape != nuc.shape:
            return expanded

        from scipy.ndimage import gaussian_filter
        from skimage.segmentation import watershed

        smoothed = gaussian_filter(mem, sigma=1.0)
        elevation = float(smoothed.max()) - smoothed
        ws = watershed(elevation, markers=nuc, mask=(expanded > 0))
        out_dtype = nuc.dtype if np.issubdtype(nuc.dtype, np.integer) else np.uint32
        return ws.astype(out_dtype, copy=False)
