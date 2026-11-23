"""
Abstract base class for cell segmentation backends.
All segmentation methods (Cellpose, StarDist, CellSAM, InstanSeg) implement this interface,
ensuring a uniform contract for the segmentation pipeline and disagreement QC engine.
"""
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


def reconcile_nuclear_and_cytoplasm_masks(
    nuclear_mask: np.ndarray,
    cytoplasm_mask: Optional[np.ndarray],
    min_nuclear_overlap_px: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Pair cytoplasm blobs 1-to-1 with enclosed nuclei, preserving nuclear IDs.

    Anucleate cytoplasm blobs are discarded, multi-nucleate cytoplasm blobs are
    split across their enclosed nuclei via distance-transform watershed, and
    unmatched nuclei are protected from being overwritten by neighbor cytoplasm.
    """
    nuc_arr = np.asarray(nuclear_mask)
    if np.issubdtype(nuc_arr.dtype, np.floating):
        nuc_arr = np.rint(np.nan_to_num(nuc_arr, nan=0.0, posinf=0.0, neginf=0.0))
    nuc_u32 = np.maximum(nuc_arr, 0).astype(np.uint32, copy=False)

    if cytoplasm_mask is None:
        return nuc_u32, None
    if nuc_u32.size == 0 or int(nuc_u32.max()) == 0:
        return nuc_u32, np.zeros_like(nuc_u32)

    cyto_arr = np.asarray(cytoplasm_mask)
    if cyto_arr.shape != nuc_u32.shape:
        raise ValueError(
            f"Mask shape mismatch: nuclear {nuc_u32.shape} vs cytoplasm {cyto_arr.shape}."
        )
    if np.issubdtype(cyto_arr.dtype, np.floating):
        cyto_arr = np.rint(np.nan_to_num(cyto_arr, nan=0.0, posinf=0.0, neginf=0.0))
    cyto_i64 = np.maximum(cyto_arr, 0).astype(np.int64, copy=False)

    if cyto_i64.size == 0 or int(cyto_i64.max()) == 0:
        return nuc_u32, np.zeros_like(nuc_u32)

    from scipy.ndimage import distance_transform_edt, find_objects
    from skimage.segmentation import watershed

    min_ov = max(1, int(min_nuclear_overlap_px))
    max_nuc = int(nuc_u32.max())
    nuc_areas = np.bincount(nuc_u32.ravel(), minlength=max_nuc + 1)

    blob_records = []
    # For each nucleus ID, track the blob with largest overlap: (overlap_px, -cid, cid)
    best_claim: Dict[int, Tuple[int, int]] = {}

    for cid, slc in enumerate(find_objects(cyto_i64), start=1):
        if slc is None:
            continue
        px = cyto_i64[slc] == cid
        ov = nuc_u32[slc][px]
        ov_pos = ov[ov > 0]
        if ov_pos.size < min_ov:
            continue

        nids, counts = np.unique(ov_pos, return_counts=True)
        valid = counts >= min_ov
        if not np.any(valid):
            continue
        nids = nids[valid]
        counts = counts[valid]

        thresholds = np.maximum(
            min_ov,
            np.floor(0.25 * nuc_areas[nids]).astype(np.int64),
        )
        strong = counts >= thresholds
        if np.any(strong):
            cand_nids = nids[strong]
            cand_counts = counts[strong]
        else:
            top_idx = int(np.argmax(counts))
            cand_nids = nids[top_idx : top_idx + 1]
            cand_counts = counts[top_idx : top_idx + 1]

        blob_records.append((cid, slc, cand_nids))
        for nid_val, cnt_val in zip(cand_nids, cand_counts):
            nid_int = int(nid_val)
            cnt_int = int(cnt_val)
            prev = best_claim.get(nid_int)
            if prev is None or cnt_int > prev[0] or (cnt_int == prev[0] and cid < prev[1]):
                best_claim[nid_int] = (cnt_int, cid)

    reconciled_cyto = np.zeros_like(nuc_u32)
    matched_lut = np.zeros(max_nuc + 1, dtype=bool)

    for cid, slc, cand_nids in blob_records:
        enclosed_nids = [
            int(nid) for nid in cand_nids if best_claim.get(int(nid), (0, -1))[1] == cid
        ]
        if not enclosed_nids:
            continue

        px = cyto_i64[slc] == cid
        if len(enclosed_nids) == 1:
            nid = enclosed_nids[0]
            reconciled_cyto[slc][px] = nid
            matched_lut[nid] = True
        else:
            sub_nuc = nuc_u32[slc]
            seed_markers = np.zeros(sub_nuc.shape, dtype=np.int32)
            for nid in enclosed_nids:
                seed_markers[px & (sub_nuc == nid)] = nid
            # Pad with 1 False border pixel so tight find_objects slices compute true boundary EDT.
            px_pad = np.pad(px, 1, mode="constant", constant_values=False)
            dist = distance_transform_edt(px_pad)[1:-1, 1:-1]
            ws = watershed(-dist, markers=seed_markers, mask=px)
            ws_pos = ws > 0
            reconciled_cyto[slc][ws_pos] = ws[ws_pos].astype(np.uint32, copy=False)
            for nid in enclosed_nids:
                matched_lut[nid] = True

    nuc_pos = nuc_u32 > 0
    reconciled_cyto[nuc_pos] = np.where(
        matched_lut[nuc_u32[nuc_pos]],
        nuc_u32[nuc_pos],
        np.uint32(0),
    )
    return nuc_u32, reconciled_cyto


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

        if np.issubdtype(labels.dtype, np.floating):
            labels = np.rint(np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0))

        min_a = int(min_area or 0)
        max_a = int(max_area or 0)
        if min_a <= 0 and max_a <= 0:
            return np.maximum(labels, 0).astype(np.uint32, copy=False)

        labels_i64 = np.maximum(labels, 0).astype(np.int64, copy=False)
        max_lbl = int(labels_i64.max())
        if max_lbl <= 0:
            return np.zeros(labels.shape, dtype=np.uint32)

        counts = np.bincount(labels_i64.ravel())
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
        return remap[labels_i64]

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

    @staticmethod
    def _parse_extra_kwargs(extra_kwargs: Any) -> Dict[str, Any]:
        """Safely parse optional native extra kwargs from a dict or JSON string."""
        if extra_kwargs is None:
            return {}
        if isinstance(extra_kwargs, dict):
            return {str(k): v for k, v in extra_kwargs.items()}
        if isinstance(extra_kwargs, str):
            raw = extra_kwargs.strip()
            if not raw:
                return {}
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                return {}
            if isinstance(parsed, dict):
                return {str(k): v for k, v in parsed.items()}
            return {}
        return {}

    def reconcile_nuclear_and_cytoplasm_masks(
        self,
        nuclear_mask: np.ndarray,
        cytoplasm_mask: Optional[np.ndarray],
        min_nuclear_overlap_px: int = 1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Reconcile cytoplasm blobs 1-to-1 with nuclear mask IDs."""
        return reconcile_nuclear_and_cytoplasm_masks(
            nuclear_mask,
            cytoplasm_mask,
            min_nuclear_overlap_px=min_nuclear_overlap_px,
        )
