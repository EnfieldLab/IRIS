"""
IRIS-Viewer Feature Extraction Engine

Extracts per-cell morphological measurements and multi-channel fluorescence
intensity statistics from segmented label masks and raw imaging stacks.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

MORPHOLOGY_FEATURES: Dict[str, str] = {
    "area": "Area (px)",
    "perimeter": "Perimeter (px)",
    "eccentricity": "Eccentricity",
    "solidity": "Solidity",
    "major_axis_length": "Major Axis Length",
    "minor_axis_length": "Minor Axis Length",
    "orientation": "Orientation (rad)",
    "extent": "Extent",
    "equivalent_diameter_area": "Equivalent Diameter",
}

INTENSITY_FEATURES: List[str] = ["mean", "median", "min", "max", "sum"]

_INTENSITY_PROP_MAP: Dict[str, str] = {
    "mean": "intensity_mean",
    "max": "intensity_max",
    "min": "intensity_min",
}


def custom_sum_intensity(mask: np.ndarray, intensity: np.ndarray) -> float:
    """Compute sum of pixel intensities within the segmented cell mask."""
    return float(np.sum(intensity[mask]))


def custom_median_intensity(mask: np.ndarray, intensity: np.ndarray) -> float:
    """Compute median of pixel intensities within the segmented cell mask."""
    return float(np.median(intensity[mask])) if np.any(mask) else 0.0


def _compute_compartment_intensities(
    comp_mask: np.ndarray,
    raw_image: np.ndarray,
    markers: List[str],
    intensity_stats: List[str],
    object_ids: pd.Series,
    infix: str = "",
) -> Dict[str, np.ndarray]:
    """Extract intensity stats per marker on a compartment mask aligned to ``object_ids``."""
    out: Dict[str, np.ndarray] = {}
    n_channels = min(raw_image.shape[0], len(markers))
    if n_channels == 0 or not intensity_stats:
        return out

    has_cells = comp_mask is not None and int(np.max(comp_mask)) > 0
    skimage_props = [_INTENSITY_PROP_MAP[s] for s in intensity_stats if s in _INTENSITY_PROP_MAP]
    extra_props = []
    if "sum" in intensity_stats:
        extra_props.append(custom_sum_intensity)
    if "median" in intensity_stats:
        extra_props.append(custom_median_intensity)

    if not skimage_props and not extra_props:
        return out

    zeros = np.zeros(len(object_ids), dtype=np.float64)
    col_prefix = f"_{infix}" if infix else ""

    for ch_idx in range(n_channels):
        marker_name = markers[ch_idx]
        if not has_cells:
            for stat in intensity_stats:
                out[f"{marker_name}{col_prefix}_{stat}"] = zeros.copy()
            continue

        ch_data = raw_image[ch_idx]
        ch_table = regionprops_table(
            comp_mask,
            intensity_image=ch_data,
            properties=["label"] + skimage_props,
            extra_properties=extra_props,
        )
        lbls = ch_table["label"]

        for req_stat, ski_key in _INTENSITY_PROP_MAP.items():
            if req_stat in intensity_stats:
                mapping = dict(zip(lbls, ch_table[ski_key]))
                out[f"{marker_name}{col_prefix}_{req_stat}"] = (
                    object_ids.map(mapping).fillna(0.0).to_numpy(dtype=np.float64)
                )

        if "sum" in intensity_stats:
            mapping = dict(zip(lbls, ch_table["custom_sum_intensity"]))
            out[f"{marker_name}{col_prefix}_sum"] = (
                object_ids.map(mapping).fillna(0.0).to_numpy(dtype=np.float64)
            )
        if "median" in intensity_stats:
            mapping = dict(zip(lbls, ch_table["custom_median_intensity"]))
            out[f"{marker_name}{col_prefix}_median"] = (
                object_ids.map(mapping).fillna(0.0).to_numpy(dtype=np.float64)
            )

    return out


def extract_cell_features(
    mask: np.ndarray,
    raw_image: Optional[np.ndarray],
    markers: List[str],
    scale_factor: float = 1.0,
    selected_morphology: Optional[List[str]] = None,
    selected_intensity: Optional[List[str]] = None,
    cytoplasm_mask: Optional[np.ndarray] = None,
    compartments: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract single-cell morphology and multi-channel intensity statistics.

    Args:
        mask: (H, W) integer label mask of segmented nuclei/cells.
        raw_image: Optional (C, H, W) multi-channel fluorescence imaging stack.
        markers: List of channel names matching the C dimension of raw_image.
        scale_factor: Physical resolution in micrometers per pixel (um/px).
        selected_morphology: Subset of morphology metrics to compute.
        selected_intensity: Subset of intensity metrics to compute.
        cytoplasm_mask: Optional (H, W) integer label mask of whole cells.
        compartments: Optional list from {"whole_cell", "nuclear", "cytoplasm",
            "membrane", "nc_ratio"} requesting compartment-resolved features.

    Returns:
        DataFrame containing one row per cell with spatial coordinates (Pos_X, Pos_Y),
        object IDs (ObjectNumber), and requested feature columns.
    """
    if mask is None or np.max(mask) == 0:
        return pd.DataFrame()

    morph_keys = (
        list(MORPHOLOGY_FEATURES.keys())
        if selected_morphology is None
        else [k for k in selected_morphology if k in MORPHOLOGY_FEATURES]
    )
    intensity_stats = (
        INTENSITY_FEATURES.copy()
        if selected_intensity is None
        else [s for s in selected_intensity if s in INTENSITY_FEATURES]
    )

    multi_compartment = cytoplasm_mask is not None or compartments is not None
    if multi_compartment:
        active_compartments = (
            {"whole_cell", "nuclear", "cytoplasm", "membrane", "nc_ratio"}
            if compartments is None
            else set(compartments)
        )
    else:
        active_compartments = set()

    has_cyto = cytoplasm_mask is not None and np.max(cytoplasm_mask) > 0
    whole_mask = cytoplasm_mask if has_cyto else mask
    base_mask = (
        whole_mask
        if (multi_compartment and has_cyto and "whole_cell" in active_compartments)
        else mask
    )

    # Morphological features via regionprops
    base_props = ["label", "centroid"]
    regionprops_keys = base_props + morph_keys

    try:
        props_table = regionprops_table(base_mask, properties=regionprops_keys)
    except Exception:
        props_table = regionprops_table(base_mask, properties=base_props)

    df = pd.DataFrame(props_table)
    df.rename(
        columns={"label": "ObjectNumber", "centroid-1": "Pos_X", "centroid-0": "Pos_Y"},
        inplace=True,
    )

    # Scale calibrated linear and areal metrics
    if scale_factor != 1.0:
        if "area" in df.columns:
            df["area"] *= scale_factor ** 2
        if "perimeter" in df.columns:
            df["perimeter"] *= scale_factor
        if "major_axis_length" in df.columns:
            df["major_axis_length"] *= scale_factor
        if "minor_axis_length" in df.columns:
            df["minor_axis_length"] *= scale_factor
        if "equivalent_diameter_area" in df.columns:
            df["equivalent_diameter_area"] *= scale_factor

    object_ids = df["ObjectNumber"]

    # Fast compartment masks and area ratios when multi-compartment extraction is enabled
    if multi_compartment:
        nuc_counts = np.bincount(np.asarray(mask, dtype=np.int64).ravel())
        nuc_area_series = (
            object_ids.map(
                lambda cid: float(nuc_counts[int(cid)]) if int(cid) < len(nuc_counts) else 0.0
            ).to_numpy(dtype=np.float64)
            * (scale_factor ** 2)
        )
        if has_cyto:
            cyto_ring_mask = np.where((cytoplasm_mask > 0) & (mask == 0), cytoplasm_mask, 0)
            cyto_counts = np.bincount(np.asarray(cyto_ring_mask, dtype=np.int64).ravel())
            cyto_area_arr = (
                object_ids.map(
                    lambda cid: float(cyto_counts[int(cid)]) if int(cid) < len(cyto_counts) else 0.0
                ).to_numpy(dtype=np.float64)
                * (scale_factor ** 2)
            )
        else:
            cyto_ring_mask = np.zeros_like(mask)
            cyto_area_arr = np.zeros(len(df), dtype=np.float64)

        df["cyto_area"] = cyto_area_arr
        df["nc_area_ratio"] = nuc_area_series / (cyto_area_arr + 1e-6)
    else:
        cyto_ring_mask = None

    # Per-channel intensity features
    if raw_image is not None and len(markers) > 0:
        if len(intensity_stats) > 0:
            base_cols = _compute_compartment_intensities(
                base_mask, raw_image, markers, intensity_stats, object_ids, infix=""
            )
            for col_name, values in base_cols.items():
                df[col_name] = values

        if multi_compartment:
            if "nuclear" in active_compartments and len(intensity_stats) > 0:
                nuc_cols = _compute_compartment_intensities(
                    mask, raw_image, markers, intensity_stats, object_ids, infix="nuc"
                )
                for col_name, values in nuc_cols.items():
                    df[col_name] = values

            if "cytoplasm" in active_compartments and len(intensity_stats) > 0:
                cyto_cols = _compute_compartment_intensities(
                    cyto_ring_mask, raw_image, markers, intensity_stats, object_ids, infix="cyto"
                )
                for col_name, values in cyto_cols.items():
                    df[col_name] = values

            if "membrane" in active_compartments and len(intensity_stats) > 0:
                from skimage.segmentation import find_boundaries

                mem_bounds = find_boundaries(whole_mask, mode="inner")
                mem_mask = np.where(mem_bounds, whole_mask, 0)
                mem_cols = _compute_compartment_intensities(
                    mem_mask, raw_image, markers, intensity_stats, object_ids, infix="mem"
                )
                for col_name, values in mem_cols.items():
                    df[col_name] = values

            if "nc_ratio" in active_compartments:
                n_channels = min(raw_image.shape[0], len(markers))
                nuc_means: Optional[Dict[str, np.ndarray]] = None
                cyto_means: Optional[Dict[str, np.ndarray]] = None
                for ch_idx in range(n_channels):
                    mname = markers[ch_idx]
                    nuc_key = f"{mname}_nuc_mean"
                    cyto_key = f"{mname}_cyto_mean"
                    if nuc_key in df.columns:
                        nuc_vals = df[nuc_key].to_numpy(dtype=np.float64)
                    else:
                        if nuc_means is None:
                            nuc_means = _compute_compartment_intensities(
                                mask, raw_image, markers, ["mean"], object_ids, infix="nuc"
                            )
                        nuc_vals = nuc_means[nuc_key]

                    if cyto_key in df.columns:
                        cyto_vals = df[cyto_key].to_numpy(dtype=np.float64)
                    else:
                        if cyto_means is None:
                            cyto_means = _compute_compartment_intensities(
                                cyto_ring_mask,
                                raw_image,
                                markers,
                                ["mean"],
                                object_ids,
                                infix="cyto",
                            )
                        cyto_vals = cyto_means[cyto_key]

                    df[f"{mname}_nc_ratio"] = nuc_vals / (cyto_vals + 1e-6)

    return df
