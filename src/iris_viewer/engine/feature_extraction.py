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
    'area':                     'Area (px)',
    'perimeter':                'Perimeter (px)',
    'eccentricity':             'Eccentricity',
    'solidity':                 'Solidity',
    'major_axis_length':        'Major Axis Length',
    'minor_axis_length':        'Minor Axis Length',
    'orientation':              'Orientation (rad)',
    'extent':                   'Extent',
    'equivalent_diameter_area': 'Equivalent Diameter',
}

INTENSITY_FEATURES: List[str] = ['mean', 'median', 'min', 'max', 'sum']


def custom_sum_intensity(mask: np.ndarray, intensity: np.ndarray) -> float:
    """Compute sum of pixel intensities within the segmented cell mask."""
    return float(np.sum(intensity[mask]))


def custom_median_intensity(mask: np.ndarray, intensity: np.ndarray) -> float:
    """Compute median of pixel intensities within the segmented cell mask."""
    return float(np.median(intensity[mask])) if np.any(mask) else 0.0


def extract_cell_features(
    mask: np.ndarray,
    raw_image: Optional[np.ndarray],
    markers: List[str],
    scale_factor: float = 1.0,
    selected_morphology: Optional[List[str]] = None,
    selected_intensity: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract single-cell morphology and multi-channel intensity statistics.

    Args:
        mask: (H, W) integer label mask of segmented cells.
        raw_image: Optional (C, H, W) multi-channel fluorescence imaging stack.
        markers: List of channel names matching the C dimension of raw_image.
        scale_factor: Physical resolution in micrometers per pixel (µm/px).
        selected_morphology: Subset of morphology metrics to compute.
        selected_intensity: Subset of intensity metrics to compute.

    Returns:
        DataFrame containing one row per cell with spatial coordinates (Pos_X, Pos_Y),
        object IDs (ObjectNumber), and requested feature columns.
    """
    if mask is None or np.max(mask) == 0: return pd.DataFrame()

    morph_keys = list(MORPHOLOGY_FEATURES.keys()) if selected_morphology is None else [k for k in selected_morphology if k in MORPHOLOGY_FEATURES]
    intensity_stats = INTENSITY_FEATURES.copy() if selected_intensity is None else [s for s in selected_intensity if s in INTENSITY_FEATURES]

    # Morphological features via regionprops
    base_props = ['label', 'centroid']
    regionprops_keys = base_props + morph_keys

    try:
        props_table = regionprops_table(mask, properties=regionprops_keys)
    except Exception:
        props_table = regionprops_table(mask, properties=base_props)

    df = pd.DataFrame(props_table)
    df.rename(columns={'label': 'ObjectNumber', 'centroid-1': 'Pos_X', 'centroid-0': 'Pos_Y'}, inplace=True)

    # Scale calibrated linear and areal metrics
    if scale_factor != 1.0:
        if 'area' in df.columns: df['area'] *= (scale_factor ** 2)
        if 'perimeter' in df.columns: df['perimeter'] *= scale_factor
        if 'major_axis_length' in df.columns: df['major_axis_length'] *= scale_factor
        if 'minor_axis_length' in df.columns: df['minor_axis_length'] *= scale_factor
        if 'equivalent_diameter_area' in df.columns: df['equivalent_diameter_area'] *= scale_factor

    # Per-channel intensity features
    if raw_image is not None and len(markers) > 0 and len(intensity_stats) > 0:
        n_channels = min(raw_image.shape[0], len(markers))
        intensity_prop_map = {'mean': 'intensity_mean', 'max': 'intensity_max', 'min': 'intensity_min'}

        for ch_idx in range(n_channels):
            marker_name = markers[ch_idx]
            ch_data = raw_image[ch_idx]
            skimage_props = [intensity_prop_map[s] for s in intensity_stats if s in intensity_prop_map]

            extra_props = []
            if 'sum' in intensity_stats: extra_props.append(custom_sum_intensity)
            if 'median' in intensity_stats: extra_props.append(custom_median_intensity)

            if skimage_props or extra_props:
                ch_table = regionprops_table(
                    mask,
                    intensity_image=ch_data,
                    properties=skimage_props,
                    extra_properties=extra_props
                )

                for req_stat, ski_key in intensity_prop_map.items():
                    if req_stat in intensity_stats:
                        df[f"{marker_name}_{req_stat}"] = ch_table[ski_key]

                if 'sum' in intensity_stats: df[f"{marker_name}_sum"] = ch_table['custom_sum_intensity']
                if 'median' in intensity_stats: df[f"{marker_name}_median"] = ch_table['custom_median_intensity']

    return df
