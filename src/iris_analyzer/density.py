"""
IRIS Spatial Cell Density & Tissue Geometry Engine
==================================================
Headless-safe polygonal ROI/hole geometry, concave hull boundary estimation,
intensity-based tissue/lumen masking, and calibrated sample-level cell density
quantification with cohort statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull, QhullError
from shapely import MultiPoint, concave_hull, contains_xy
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from skimage import filters, measure, morphology

from iris_common.engine.statistics import (
    adjust_p_values,
    format_significance,
    run_group_comparison,
)
from iris_common.schema import (
    COL_CELL_TYPE,
    COL_SAMPLE,
    COL_X,
    COL_Y,
    ensure_canonical_dataframe,
    find_sample_column,
)

_UNIT_SPECS: Dict[str, Tuple[str, float]] = {
    "mm2": ("cells/mm²", 1.0),
    "mm²": ("cells/mm²", 1.0),
    "cells/mm2": ("cells/mm²", 1.0),
    "cells/mm²": ("cells/mm²", 1.0),
    "10^4 um2": ("cells/10^4 µm²", 1e4 / 1e6),
    "10^4 µm²": ("cells/10^4 µm²", 1e4 / 1e6),
    "1e4_um2": ("cells/10^4 µm²", 1e4 / 1e6),
    "1e4 um2": ("cells/10^4 µm²", 1e4 / 1e6),
    "cells/10^4 um2": ("cells/10^4 µm²", 1e4 / 1e6),
    "cells/10^4 µm²": ("cells/10^4 µm²", 1e4 / 1e6),
    "um2": ("cells/µm²", 1.0 / 1e6),
    "µm²": ("cells/µm²", 1.0 / 1e6),
    "cells/um2": ("cells/µm²", 1.0 / 1e6),
    "cells/µm²": ("cells/µm²", 1.0 / 1e6),
}

_STATS_COLUMNS: List[str] = [
    "cell_type",
    "group1",
    "group2",
    "test_name",
    "statistic",
    "p_value",
    "q_value",
    "n1",
    "n2",
    "aggregation_level",
    "correction_method",
    "effect_size",
    "effect_size_name",
    "mean1",
    "mean2",
    "median1",
    "median2",
    "significant",
    "significance_label",
    "omnibus_test",
    "omnibus_statistic",
    "omnibus_p_value",
    "notes",
]


def _normalize_polygon_list(
    polygons: Optional[Union[Sequence[np.ndarray], np.ndarray]],
) -> List[np.ndarray]:
    """Coerce polygon inputs into a list of float64 NumPy coordinate arrays."""
    if polygons is None:
        return []
    if isinstance(polygons, np.ndarray):
        if polygons.ndim == 2:
            return [np.asarray(polygons, dtype=float)[:, :2]]
        return [np.asarray(p, dtype=float)[:, :2] for p in polygons if p is not None]
    out: List[np.ndarray] = []
    for poly in polygons:
        if poly is None:
            continue
        arr = np.asarray(poly, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            out.append(arr[:, :2])
    return out


@dataclass
class ROISpec:
    """Polygonal region of interest with optional exclusion holes in ``(Y, X)`` px."""

    name: str
    sample: Optional[str] = None
    inclusion_polygons: List[np.ndarray] = field(default_factory=list)
    exclusion_polygons: List[np.ndarray] = field(default_factory=list)
    microns_per_pixel: float = 1.0

    def __post_init__(self) -> None:
        self.inclusion_polygons = _normalize_polygon_list(self.inclusion_polygons)
        self.exclusion_polygons = _normalize_polygon_list(self.exclusion_polygons)
        self.microns_per_pixel = float(self.microns_per_pixel)


@dataclass
class DensityResult:
    """Container for calibrated spatial cell density outputs and cohort statistics."""

    summary_df: pd.DataFrame
    replicates_df: pd.DataFrame
    roi_areas_df: pd.DataFrame
    stats_df: pd.DataFrame
    cell_types: List[str]
    unit_label: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _to_valid_polygon(vertices: np.ndarray) -> Optional[BaseGeometry]:
    """Convert ``(K, 2)`` vertices into a non-empty valid Shapely geometry."""
    arr = np.asarray(vertices, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    arr = arr[:, :2]
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if len(arr) < 3:
        return None
    poly = Polygon(arr)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0.0:
        return None
    return poly


def build_shapely_region(
    inclusion_polygons: Optional[Union[Sequence[np.ndarray], np.ndarray]],
    exclusion_polygons: Optional[Union[Sequence[np.ndarray], np.ndarray]] = None,
) -> Tuple[BaseGeometry, float]:
    """Construct union of inclusion ``(Y, X)`` polygons minus union of exclusion holes.

    Uses ``.buffer(0)`` self-intersection cleanup and Shapely set operations so
    overlapping inclusion polygons or holes are never double-counted.

    Returns
    -------
    tuple[BaseGeometry, float]
        Shapely geometry in ``(Y, X)`` pixel coordinates and its exact pixel area.
    """
    incl_arrays = _normalize_polygon_list(inclusion_polygons)
    incl_polys = [g for arr in incl_arrays if (g := _to_valid_polygon(arr)) is not None]
    if not incl_polys:
        return Polygon(), 0.0

    region = unary_union(incl_polys)
    if not region.is_valid:
        region = region.buffer(0)

    excl_arrays = _normalize_polygon_list(exclusion_polygons)
    excl_polys = [g for arr in excl_arrays if (g := _to_valid_polygon(arr)) is not None]
    if excl_polys:
        excl_union = unary_union(excl_polys)
        if not excl_union.is_valid:
            excl_union = excl_union.buffer(0)
        region = region.difference(excl_union)
        if not region.is_valid:
            region = region.buffer(0)

    if region.is_empty:
        return Polygon(), 0.0
    return region, float(region.area)


def compute_polygon_area_and_filter(
    df: pd.DataFrame,
    inclusion_polygons: Optional[Union[Sequence[np.ndarray], np.ndarray]],
    exclusion_polygons: Optional[Union[Sequence[np.ndarray], np.ndarray]] = None,
    scale_factor: float = 1.0,
    sample_col: Optional[str] = None,
    sample_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, float, float]:
    """Filter cells inside polygonal ROI minus exclusion holes and compute calibrated area.

    Uses vectorized ``shapely.contains_xy`` over canonical ``(Pos_Y, Pos_X)``
    coordinates.

    Returns
    -------
    tuple[pd.DataFrame, float, float]
        ``(filtered_df, area_um2, area_mm2)``
    """
    scale = float(scale_factor)
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError(f"scale_factor must be a positive finite number, got {scale_factor!r}.")

    geom, pixel_area = build_shapely_region(inclusion_polygons, exclusion_polygons)
    area_um2 = float(pixel_area) * (scale ** 2)
    area_mm2 = area_um2 / 1_000_000.0

    if df is None or df.empty:
        empty = df.iloc[0:0].copy() if df is not None else pd.DataFrame()
        return empty, area_um2, area_mm2

    work = ensure_canonical_dataframe(df, require_coordinates=True)
    if sample_name is not None:
        s_col = (
            sample_col
            if (sample_col is not None and sample_col in work.columns)
            else (find_sample_column(work) or COL_SAMPLE)
        )
        if s_col in work.columns:
            work = work[work[s_col].astype(str) == str(sample_name)]

    if work.empty or geom.is_empty or pixel_area <= 0.0:
        return work.iloc[0:0].copy(), area_um2, area_mm2

    y_vals = pd.to_numeric(work[COL_Y], errors="coerce").to_numpy(dtype=float)
    x_vals = pd.to_numeric(work[COL_X], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(y_vals) & np.isfinite(x_vals)

    inside = np.zeros(len(work), dtype=bool)
    if finite_mask.any():
        inside[finite_mask] = contains_xy(geom, y_vals[finite_mask], x_vals[finite_mask])

    return work.loc[inside].copy(), area_um2, area_mm2


def _convex_hull_vertices(unique_pts: np.ndarray) -> np.ndarray:
    """Return 2D convex hull vertices or input points on collinear fallback."""
    if len(unique_pts) < 3:
        return unique_pts.copy()
    try:
        hull = ConvexHull(unique_pts)
        return unique_pts[hull.vertices].astype(float)
    except QhullError:
        return unique_pts.copy()


def generate_concave_hull_polygon(
    points_or_df: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[float]]],
    alpha_ratio: float = 0.3,
) -> np.ndarray:
    """Compute a concave hull ``(K, 2)`` polygon in ``(Y, X)`` coordinates.

    Uses ``shapely.concave_hull`` with ``ratio=alpha_ratio`` when at least 4
    unique points are present, falling back cleanly to ``scipy.spatial.ConvexHull``
    for fewer than 4 points or degenerate geometries.
    """
    if isinstance(points_or_df, pd.DataFrame):
        canon = ensure_canonical_dataframe(points_or_df, require_coordinates=True)
        pts = canon[[COL_Y, COL_X]].to_numpy(dtype=float)
    else:
        pts = np.asarray(points_or_df, dtype=float)
        if pts.size == 0:
            return np.empty((0, 2), dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError(
                f"Expected a DataFrame or (N, 2) coordinate array, got shape {pts.shape}."
            )
        pts = pts[:, :2]

    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return np.empty((0, 2), dtype=float)

    unique_pts = np.unique(pts, axis=0)
    if len(unique_pts) < 4:
        return _convex_hull_vertices(unique_pts)

    ratio = float(np.clip(alpha_ratio, 0.0, 1.0))
    try:
        hull_geom = concave_hull(MultiPoint(unique_pts), ratio=ratio)
        if hull_geom.geom_type == "MultiPolygon" and len(hull_geom.geoms) > 0:
            hull_geom = max(hull_geom.geoms, key=lambda g: g.area)
        if hull_geom.geom_type == "Polygon" and hull_geom.area > 0.0:
            coords = np.asarray(hull_geom.exterior.coords, dtype=float)[:, :2]
            if len(coords) > 3 and np.allclose(coords[0], coords[-1]):
                coords = coords[:-1]
            if len(coords) >= 3:
                return coords
    except Exception:
        pass

    return _convex_hull_vertices(unique_pts)


def _extract_2d_channel(raw_image: np.ndarray, channel_idx: int) -> np.ndarray:
    """Resolve a 2D intensity plane from ``(H, W)`` or ``(C, H, W)`` image data."""
    arr = np.asarray(raw_image, dtype=float)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 2:
        img2d = arr
    elif arr.ndim == 3:
        if 0 <= channel_idx < arr.shape[0]:
            img2d = arr[channel_idx]
        elif arr.shape[0] > 32 and 0 <= channel_idx < arr.shape[2]:
            img2d = arr[:, :, channel_idx]
        else:
            raise IndexError(
                f"channel_idx={channel_idx} out of range for image shape {arr.shape}."
            )
    else:
        raise ValueError(f"Expected 2D or 3D image array, got shape {arr.shape}.")

    return np.nan_to_num(img2d, nan=0.0, posinf=0.0, neginf=0.0)


def _extract_contours_from_binary(
    binary_mask: np.ndarray,
    min_poly_area: float = 1.0,
) -> List[np.ndarray]:
    """Extract simplified valid ``(K, 2)`` polygon contours from a binary mask."""
    if not np.any(binary_mask):
        return []

    padded = np.pad(binary_mask.astype(np.float32), pad_width=1, mode="constant", constant_values=0)
    raw_contours = measure.find_contours(padded, level=0.5)

    scored: List[Tuple[float, np.ndarray]] = []
    for contour in raw_contours:
        pts = contour - 1.0
        if len(pts) > 12:
            pts = measure.approximate_polygon(pts, tolerance=0.5)
        if len(pts) > 3 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        poly = _to_valid_polygon(pts)
        if poly is not None and poly.area >= min_poly_area:
            scored.append((float(poly.area), np.asarray(pts, dtype=float)))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [pts for _, pts in scored]


def generate_intensity_tissue_polygons(
    raw_image: np.ndarray,
    channel_idx: int = 0,
    threshold_method: str = "otsu",
    gaussian_sigma: float = 2.0,
    min_area_px: int = 500,
    subtract_holes: bool = True,
    min_hole_area_px: int = 200,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extract outer tissue polygons and interior lumen/necrosis holes from image intensity.

    Parameters
    ----------
    raw_image : np.ndarray
        2D ``(H, W)`` or 3D ``(C, H, W)`` image array.
    channel_idx : int
        Channel index (typically nuclear/DAPI channel) when ``raw_image`` is 3D.
    threshold_method : str
        One of ``'otsu'``, ``'triangle'``, or ``'li'``.
    gaussian_sigma : float
        Gaussian smoothing standard deviation in pixels before thresholding.
    min_area_px : int
        Minimum connected foreground component area in pixels.
    subtract_holes : bool
        If True, extract interior tissue lumens/necrosis cavities as exclusion polygons.
    min_hole_area_px : int
        Minimum interior cavity area in pixels to retain as an exclusion polygon.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        ``(inclusion_polygons, exclusion_polygons)`` where each polygon is ``(K, 2)`` in ``(Y, X)``.
    """
    img2d = _extract_2d_channel(raw_image, int(channel_idx))
    if img2d.size == 0:
        return [], []

    smoothed = (
        filters.gaussian(img2d, sigma=float(gaussian_sigma), preserve_range=True)
        if gaussian_sigma > 0
        else img2d
    )
    if np.isclose(float(np.min(smoothed)), float(np.max(smoothed))):
        return [], []

    method_key = str(threshold_method).strip().lower()
    thresh_fns = {
        "otsu": filters.threshold_otsu,
        "triangle": filters.threshold_triangle,
        "li": filters.threshold_li,
    }
    if method_key not in thresh_fns:
        raise ValueError(
            f"Unsupported threshold_method {threshold_method!r}. "
            f"Expected one of {tuple(thresh_fns.keys())}."
        )

    thresh = float(thresh_fns[method_key](smoothed))
    fg_mask = smoothed > thresh
    if min_area_px > 0:
        fg_mask = morphology.remove_small_objects(fg_mask, min_size=int(min_area_px))
    if not np.any(fg_mask):
        return [], []

    filled_mask = binary_fill_holes(fg_mask)
    inclusion_polys = _extract_contours_from_binary(
        filled_mask,
        min_poly_area=max(1.0, float(min_area_px) * 0.25),
    )

    exclusion_polys: List[np.ndarray] = []
    if subtract_holes:
        holes_mask = filled_mask & (~fg_mask)
        if min_hole_area_px > 0:
            holes_mask = morphology.remove_small_objects(
                holes_mask,
                min_size=int(min_hole_area_px),
            )
        holes_mask = binary_fill_holes(holes_mask)
        exclusion_polys = _extract_contours_from_binary(
            holes_mask,
            min_poly_area=max(1.0, float(min_hole_area_px) * 0.25),
        )

    return inclusion_polys, exclusion_polys


def _resolve_density_unit(unit: str) -> Tuple[str, float]:
    """Return ``(display_label, scale_from_mm2_factor)`` for a density unit string."""
    key = str(unit).strip().lower()
    if key not in _UNIT_SPECS:
        raise ValueError(
            f"Unsupported density unit {unit!r}. Expected one of "
            "('mm2', '10^4 um2', 'um2')."
        )
    return _UNIT_SPECS[key]


def _density_from_area(count: int, area_um2: float, unit_key: str) -> float:
    """Convert raw count and physical area (µm²) into calibrated cell density."""
    if area_um2 <= 0.0:
        return 0.0
    label, _ = _resolve_density_unit(unit_key)
    if label == "cells/mm²":
        return float(count) / (area_um2 / 1_000_000.0)
    if label == "cells/10^4 µm²":
        return float(count) / (area_um2 / 10_000.0)
    return float(count) / area_um2


def _resolve_sample_and_type_columns(
    df: pd.DataFrame,
    cell_type_col: str,
    sample_col: Optional[str],
) -> Tuple[pd.DataFrame, str, str]:
    """Canonicalize DataFrame and resolve sample and phenotype column names."""
    work = ensure_canonical_dataframe(df, default_sample="Sample_1")
    s_col = sample_col if (sample_col is not None and sample_col in work.columns) else None
    if s_col is None:
        s_col = find_sample_column(work) or COL_SAMPLE
    if s_col != COL_SAMPLE:
        work[COL_SAMPLE] = work[s_col].astype(str)

    if cell_type_col in work.columns:
        ct_col = cell_type_col
    elif COL_CELL_TYPE in work.columns:
        ct_col = COL_CELL_TYPE
    else:
        raise KeyError(f"Cell type column {cell_type_col!r} not found in DataFrame.")

    work[ct_col] = work[ct_col].fillna("Unassigned").astype(str)
    work[COL_SAMPLE] = work[COL_SAMPLE].fillna("Sample_1").astype(str)
    return work, COL_SAMPLE, ct_col


def _compute_group_stats_table(
    replicates_df: pd.DataFrame,
    value_col: str,
    categories: Sequence[str],
    group_map: Optional[Dict[str, str]],
    stat_test: str,
    p_adjust: str,
    *,
    include_all_cells: bool = False,
) -> pd.DataFrame:
    """Run sample-aggregated group comparisons across categories with FDR adjustment."""
    if replicates_df.empty or not group_map:
        return pd.DataFrame(columns=_STATS_COLUMNS)

    mapped_groups = {str(v) for v in group_map.values() if v is not None and str(v).strip()}
    if len(mapped_groups) < 2:
        return pd.DataFrame(columns=_STATS_COLUMNS)

    valid_rep = replicates_df[replicates_df["Group"].isin(mapped_groups)]
    if valid_rep["Group"].nunique() < 2:
        return pd.DataFrame(columns=_STATS_COLUMNS)

    test_targets = (["All Cells"] if include_all_cells else []) + list(categories)
    all_rows: List[Dict[str, Any]] = []
    per_type_indices: List[int] = []

    for cat in test_targets:
        sub = valid_rep[valid_rep["cell_type"] == cat]
        if sub["Group"].nunique() < 2:
            continue
        res_list = run_group_comparison(
            sub,
            value_col=value_col,
            group_col="Group",
            sample_col=COL_SAMPLE,
            aggregate_by_sample=True,
            test=stat_test,
            correction=p_adjust,
        )
        for item in res_list:
            row = item.to_dict()
            row["cell_type"] = cat
            if cat != "All Cells":
                per_type_indices.append(len(all_rows))
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame(columns=_STATS_COLUMNS)

    if len(per_type_indices) > 1 and p_adjust.lower().strip() != "tukey":
        raw_ps = [all_rows[idx]["p_value"] for idx in per_type_indices]
        adj_qs = adjust_p_values(raw_ps, method=p_adjust)
        for idx, q_val in zip(per_type_indices, adj_qs):
            q_clean = float(q_val) if np.isfinite(q_val) else 1.0
            all_rows[idx]["q_value"] = q_clean
            all_rows[idx]["significant"] = bool(q_clean < 0.05)
            all_rows[idx]["significance_label"] = format_significance(q_clean, mode="stars")

    stats_df = pd.DataFrame(all_rows)
    ordered_cols = [c for c in _STATS_COLUMNS if c in stats_df.columns]
    extra_cols = [c for c in stats_df.columns if c not in ordered_cols]
    return stats_df[ordered_cols + extra_cols]


def _auto_sample_area_px(sub_df: pd.DataFrame) -> Tuple[float, str]:
    """Estimate sample tissue area in px² from concave hull with bounding-box fallback."""
    if COL_X in sub_df.columns and COL_Y in sub_df.columns and len(sub_df) >= 3:
        hull_pts = generate_concave_hull_polygon(sub_df)
        if len(hull_pts) >= 3:
            _, px_area = build_shapely_region([hull_pts])
            if px_area > 0.0:
                return float(px_area), "concave_hull"

    if COL_X in sub_df.columns and COL_Y in sub_df.columns and len(sub_df) >= 1:
        xs = pd.to_numeric(sub_df[COL_X], errors="coerce").dropna().to_numpy(dtype=float)
        ys = pd.to_numeric(sub_df[COL_Y], errors="coerce").dropna().to_numpy(dtype=float)
        if len(xs) >= 2 and len(ys) >= 2:
            span = max(1.0, float(xs.max() - xs.min())) * max(1.0, float(ys.max() - ys.min()))
            return span, "bounding_box_fallback"
    return 1.0, "unit_fallback"


def compute_cell_density(
    df: pd.DataFrame,
    cell_type_col: str = "cell_type",
    sample_col: Optional[str] = None,
    scale_factor: float = 1.0,
    rois: Optional[Sequence[ROISpec]] = None,
    image_shapes: Optional[Dict[str, Tuple[int, int]]] = None,
    group_map: Optional[Dict[str, str]] = None,
    unit: str = "mm2",
    stat_test: str = "mann_whitney",
    p_adjust: str = "fdr_bh",
) -> DensityResult:
    """Compute calibrated cell densities per ROI/sample, group summaries, and cohort statistics.

    If neither ``rois`` nor ``image_shapes`` is provided, tissue area is
    automatically estimated per sample using ``generate_concave_hull_polygon``.
    """
    unit_label, _ = _resolve_density_unit(unit)
    scale = float(scale_factor)
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError(f"scale_factor must be positive and finite, got {scale_factor!r}.")

    norm_group_map: Optional[Dict[str, str]] = (
        {str(k): str(v) for k, v in group_map.items()} if group_map is not None else None
    )

    if df is None or df.empty:
        return DensityResult(
            summary_df=pd.DataFrame(),
            replicates_df=pd.DataFrame(),
            roi_areas_df=pd.DataFrame(),
            stats_df=pd.DataFrame(columns=_STATS_COLUMNS),
            cell_types=[],
            unit_label=unit_label,
            metadata={
                "scale_factor": scale,
                "unit": unit,
                "unit_label": unit_label,
                "n_samples": 0,
                "n_rois": 0,
                "n_cells": 0,
            },
        )

    work, s_col, ct_col = _resolve_sample_and_type_columns(df, cell_type_col, sample_col)
    cell_types = sorted(work[ct_col].unique().tolist())
    samples = list(dict.fromkeys(work[s_col].tolist()))

    def _resolve_group(sample_name: str) -> str:
        if norm_group_map is None:
            return "All"
        return norm_group_map.get(sample_name, "Unassigned")

    roi_units: List[Tuple[str, str, pd.DataFrame, float, float, float, float, str]] = []
    if rois:
        for roi in rois:
            roi_scale = roi.microns_per_pixel if roi.microns_per_pixel != 1.0 else scale
            target_sample = roi.sample
            if target_sample is None and len(samples) == 1:
                target_sample = samples[0]

            filtered, area_um2, area_mm2 = compute_polygon_area_and_filter(
                work,
                roi.inclusion_polygons,
                roi.exclusion_polygons,
                scale_factor=roi_scale,
                sample_col=s_col,
                sample_name=target_sample,
            )
            if target_sample is not None:
                sample_label = str(target_sample)
            elif not filtered.empty and filtered[s_col].nunique() == 1:
                sample_label = str(filtered[s_col].iloc[0])
            else:
                sample_label = roi.name
            pixel_area = area_um2 / (roi_scale ** 2)
            roi_units.append(
                (
                    sample_label,
                    roi.name,
                    filtered,
                    pixel_area,
                    area_um2,
                    area_mm2,
                    roi_scale,
                    "roi_polygon",
                )
            )
    else:
        for sample_name in samples:
            sub = work[work[s_col] == sample_name]
            if image_shapes is not None and sample_name in image_shapes:
                h, w = image_shapes[sample_name][:2]
                pixel_area = float(h) * float(w)
                area_source = "image_shape"
            else:
                pixel_area, area_source = _auto_sample_area_px(sub)
            area_um2 = float(pixel_area) * (scale ** 2)
            area_mm2 = area_um2 / 1_000_000.0
            roi_units.append(
                (
                    sample_name,
                    "Whole Sample",
                    sub,
                    pixel_area,
                    area_um2,
                    area_mm2,
                    scale,
                    area_source,
                )
            )

    area_rows: List[Dict[str, Any]] = []
    rep_rows: List[Dict[str, Any]] = []
    for sample_name, roi_name, sub_df, px_area, area_um2, area_mm2, r_scale, source in roi_units:
        grp = _resolve_group(sample_name)
        total_cells = int(len(sub_df))
        area_rows.append(
            {
                "Sample": sample_name,
                COL_SAMPLE: sample_name,
                "ROI": roi_name,
                "Group": grp,
                "Total_Cells": total_cells,
                "Pixel_Area": float(px_area),
                "Area_um2": float(area_um2),
                "Area_mm2": float(area_mm2),
                "Scale_um_per_px": float(r_scale),
                "Area_Source": source,
            }
        )

        counts_by_type = sub_df[ct_col].value_counts().to_dict() if total_cells > 0 else {}
        rep_rows.append(
            {
                COL_SAMPLE: sample_name,
                "Sample": sample_name,
                "ROI": roi_name,
                "Group": grp,
                "cell_type": "All Cells",
                "Count": total_cells,
                "Area_um2": float(area_um2),
                "Area_mm2": float(area_mm2),
                "Density": _density_from_area(total_cells, area_um2, unit),
                "Unit": unit_label,
            }
        )
        for ct in cell_types:
            cnt = int(counts_by_type.get(ct, 0))
            rep_rows.append(
                {
                    COL_SAMPLE: sample_name,
                    "Sample": sample_name,
                    "ROI": roi_name,
                    "Group": grp,
                    "cell_type": ct,
                    "Count": cnt,
                    "Area_um2": float(area_um2),
                    "Area_mm2": float(area_mm2),
                    "Density": _density_from_area(cnt, area_um2, unit),
                    "Unit": unit_label,
                }
            )

    roi_areas_df = pd.DataFrame(area_rows)
    replicates_df = pd.DataFrame(rep_rows)

    summary_rows: List[Dict[str, Any]] = []
    if not replicates_df.empty:
        all_cats = ["All Cells"] + cell_types
        groups_order = list(dict.fromkeys(replicates_df["Group"].tolist()))
        for grp in groups_order:
            for ct in all_cats:
                sub_rep = replicates_df[
                    (replicates_df["Group"] == grp) & (replicates_df["cell_type"] == ct)
                ]
                if sub_rep.empty:
                    continue
                sample_means = sub_rep.groupby(COL_SAMPLE, observed=True)["Density"].mean()
                vals = sample_means.to_numpy(dtype=float)
                n_samples = int(len(vals))
                sd_val = float(np.std(vals, ddof=1)) if n_samples > 1 else 0.0
                sem_val = sd_val / float(np.sqrt(n_samples)) if n_samples > 0 else 0.0
                q25, q75 = np.percentile(vals, [25.0, 75.0])
                summary_rows.append(
                    {
                        "Group": grp,
                        "cell_type": ct,
                        "N_Samples": n_samples,
                        "N_ROIs": int(len(sub_rep)),
                        "Total_Cells": int(sub_rep["Count"].sum()),
                        "Mean_Count": float(sub_rep["Count"].mean()),
                        "Mean_Density": float(np.mean(vals)),
                        "SD_Density": sd_val,
                        "SEM_Density": sem_val,
                        "Median_Density": float(np.median(vals)),
                        "IQR_Density": float(q75 - q25),
                        "Min_Density": float(np.min(vals)),
                        "Max_Density": float(np.max(vals)),
                        "Unit": unit_label,
                    }
                )
    summary_df = pd.DataFrame(summary_rows)

    stats_df = _compute_group_stats_table(
        replicates_df=replicates_df,
        value_col="Density",
        categories=cell_types,
        group_map=norm_group_map,
        stat_test=stat_test,
        p_adjust=p_adjust,
        include_all_cells=True,
    )

    return DensityResult(
        summary_df=summary_df,
        replicates_df=replicates_df,
        roi_areas_df=roi_areas_df,
        stats_df=stats_df,
        cell_types=cell_types,
        unit_label=unit_label,
        metadata={
            "scale_factor": scale,
            "unit": unit,
            "unit_label": unit_label,
            "stat_test": stat_test,
            "p_adjust": p_adjust,
            "n_samples": int(roi_areas_df[COL_SAMPLE].nunique()) if not roi_areas_df.empty else 0,
            "n_rois": int(len(roi_areas_df)),
            "n_cells": int(len(work)),
            "cell_type_col": ct_col,
            "sample_col": s_col,
            "has_groups": bool(not stats_df.empty),
        },
    )


__all__ = [
    "ROISpec",
    "DensityResult",
    "build_shapely_region",
    "compute_polygon_area_and_filter",
    "generate_concave_hull_polygon",
    "generate_intensity_tissue_polygons",
    "compute_cell_density",
]
