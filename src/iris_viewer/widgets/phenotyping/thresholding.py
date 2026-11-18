"""
Mathematical Transformations & Automated Cutoff Thresholding
=============================================================
Provides biologically and mathematically grounded transformations and optimal
cutoff threshold discovery algorithms for multiplexed single-cell IF and CyTOF:
- Arcsinh transformation with standard cofactors (e.g., cofactor=5.0)
- Otsu's inter-class variance maximization (Otsu, 1979)
- Bimodal valley/dip detection via smoothed histogram convolution
- Non-parametric Median Absolute Deviation (MAD) robust noise cutoff
- Bimodal separation score quality assessment
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def arcsinh_transform(arr: np.ndarray, cofactor: float = 5.0) -> np.ndarray:
    """
    Arcsinh transformation standard in CyTOF and multiplexed single-cell IF.
    Computes arcsinh(max(x, 0) / cofactor), dampening extreme fluorescence outliers
    while preserving linear dynamic range around zero.
    """
    arr_clean = np.asarray(arr, dtype=float)
    return np.arcsinh(np.maximum(arr_clean, 0.0) / max(cofactor, 1e-4))


def log10_transform(arr: np.ndarray) -> np.ndarray:
    """Log10(1 + x) transformation for strictly positive fluorescent signals."""
    arr_clean = np.asarray(arr, dtype=float)
    return np.log10(np.maximum(arr_clean, 0.0) + 1.0)


def transform_values(arr: np.ndarray, scale_mode: str) -> np.ndarray:
    """Transform intensity values according to the selected scale mode.

    Supports:
    - ``"Arcsinh (Cofactor = 5)"``
    - ``"Log10(1 + x)"``
    - ``"Linear (Raw)"``
    - ``"Min-Max Normalized"``
    """
    arr_clean = np.asarray(arr, dtype=float)
    mode = (scale_mode or "").lower()
    if "arcsinh" in mode:
        return arcsinh_transform(arr_clean, cofactor=5.0)
    if "log10" in mode:
        return log10_transform(arr_clean)
    if "min-max" in mode or "minmax" in mode:
        finite = arr_clean[np.isfinite(arr_clean)]
        if finite.size == 0:
            return arr_clean
        mn = float(np.min(finite))
        mx = float(np.max(finite))
        denom = mx - mn
        if denom <= 0.0:
            out = np.zeros_like(arr_clean, dtype=float)
            out[~np.isfinite(arr_clean)] = np.nan
            return out
        return (arr_clean - mn) / denom
    return arr_clean


def resolve_marker_col(
    df: Optional[pd.DataFrame],
    marker: str,
    preferred_stat: str = "mean",
) -> Optional[str]:
    """Resolve a marker name to an existing column in *df*.

    Resolution order:
    1. Exact match (``marker``)
    2. Preferred stat suffix (``{marker}_{preferred_stat}``)
    3. Case-insensitive match for ``marker`` or ``{marker}_{preferred_stat}``
    """
    if df is None or not marker:
        return None
    if marker in df.columns:
        return marker
    suffixed = f"{marker}_{preferred_stat}"
    if suffixed in df.columns:
        return suffixed

    lower_map = {str(c).lower(): c for c in df.columns}
    m_low = marker.lower()
    if m_low in lower_map:
        return lower_map[m_low]
    s_low = f"{m_low}_{preferred_stat.lower()}"
    if s_low in lower_map:
        return lower_map[s_low]
    return None


def compute_optimal_threshold(arr: np.ndarray, method: str = 'otsu') -> float:
    """
    Computes mathematically and biologically grounded cutoff thresholds:
    - 'otsu': Maximizes inter-class variance between negative background and positive signal
    - 'valley': Bimodal peak detection finding the local minimum/dip between peaks
    - 'percentile95': 95th percentile
    - 'percentile98': 98th percentile
    - 'mad': Median + 2.5 * MAD (robust non-parametric noise cutoff)
    - 'median': Median
    """
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return 0.0
    mn, mx = float(np.min(valid)), float(np.max(valid))
    if mn == mx:
        return mn

    if method == 'percentile95':
        return float(np.percentile(valid, 95))
    elif method == 'percentile98':
        return float(np.percentile(valid, 98))
    elif method == 'mad':
        med = float(np.median(valid))
        mad = float(np.median(np.abs(valid - med)))
        return float(med + 2.5 * 1.4826 * mad)
    elif method == 'median':
        return float(np.median(valid))
    elif method == 'valley':
        hist, bin_edges = np.histogram(valid, bins=128)
        window = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0
        smoothed = np.convolve(hist, window, mode='same')
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                peaks.append(i)
        if len(peaks) >= 2:
            sorted_peaks = sorted(peaks, key=lambda p: smoothed[p], reverse=True)[:2]
            p1, p2 = min(sorted_peaks), max(sorted_peaks)
            if p2 - p1 > 2:
                valley_idx = p1 + int(np.argmin(smoothed[p1:p2 + 1]))
                return float((bin_edges[valley_idx] + bin_edges[valley_idx + 1]) / 2.0)
        method = 'otsu'

    if method == 'otsu':
        hist, bin_edges = np.histogram(valid, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        hist = hist.astype(float)
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1e-12)
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2[::-1], 1e-12))[::-1]
        variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        variance = np.nan_to_num(variance, nan=0.0)
        max_var = float(np.max(variance))
        best_idxs = np.where(np.isclose(variance, max_var, rtol=1e-12, atol=1e-12))[0]
        if best_idxs.size > 0:
            return float(np.mean(bin_centers[best_idxs]))
        return float(bin_centers[int(np.argmax(variance))])

    return float(np.median(valid))


def assess_gating_quality(vx: np.ndarray, tx: float) -> Tuple[str, str, float]:
    """
    Evaluates whether threshold tx divides two distinct expression peaks (bimodal)
    or cuts through a continuous/unimodal distribution.
    Returns (quality_label, badge_hex_color, separation_score).
    """
    clean_vx = vx[np.isfinite(vx)]
    v_neg = clean_vx[clean_vx < tx]
    v_pos = clean_vx[clean_vx >= tx]
    if len(v_neg) < 5 or len(v_pos) < 5:
        return "Skewed / Extreme Cutoff", "#888888", 0.0

    med_neg = float(np.median(v_neg))
    med_pos = float(np.median(v_pos))
    mad_neg = float(np.median(np.abs(v_neg - med_neg)))
    mad_pos = float(np.median(np.abs(v_pos - med_pos)))
    spread = max(mad_neg + mad_pos, 1e-4)
    sep = (med_pos - med_neg) / spread

    if sep >= 2.4:
        return "Clean Bimodal Separation", "#40916C", sep
    elif sep >= 1.4:
        return "Moderate Peak Overlap", "#E6A23C", sep
    else:
        return "Continuous / Unimodal Expression", "#E57373", sep


def find_sample_column(df: pd.DataFrame) -> Optional[str]:
    """Identifies the column representing sample or TMA core identity."""
    for col in ['Image', 'sample_id', 'Sample', 'sample', 'core', 'Core', 'TMA_Core', 'image', 'Sample_Name']:
        if col in df.columns:
            return col
    return None
