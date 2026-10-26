"""
IRIS-Viewer Geometry Engine

Algorithms for tissue boundary segmentation, convex hull extraction,
and polygonal area computation.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from skimage.draw import polygon


def ensure_spatial_coordinates(df: pd.DataFrame) -> Tuple[str, str]:
    """Validate and ensure canonical 'Pos_X' and 'Pos_Y' spatial coordinates exist on df.

    Adheres strictly to the IRIS canonical spatial column naming standard ('Pos_X', 'Pos_Y').
    If alternative coordinate columns exist in the DataFrame (e.g. 'Centroid_X'/'Centroid_Y',
    'x'/'y', or case variants), they are mapped to 'Pos_X' and 'Pos_Y' in place so that
    all downstream analytics, overlays, and exports use 'Pos_X' and 'Pos_Y'.

    Args:
        df: Input DataFrame containing spatial cell coordinates.

    Returns:
        Tuple[str, str]: ('Pos_X', 'Pos_Y')

    Raises:
        ValueError: If neither ('Pos_X', 'Pos_Y') nor acceptable alternatives are present.
    """
    if "Pos_X" in df.columns and "Pos_Y" in df.columns:
        return "Pos_X", "Pos_Y"

    x_col = next((c for c in ['Pos_X', 'Centroid_X', 'x', 'X'] if c in df.columns), None)
    y_col = next((c for c in ['Pos_Y', 'Centroid_Y', 'y', 'Y'] if c in df.columns), None)

    if x_col is None or y_col is None:
        raise ValueError("DataFrame missing spatial coordinate columns ('Pos_X', 'Pos_Y').")

    if "Pos_X" not in df.columns:
        df["Pos_X"] = df[x_col]
    if "Pos_Y" not in df.columns:
        df["Pos_Y"] = df[y_col]

    return "Pos_X", "Pos_Y"


def compute_tissue_boundary_mask(
    df: pd.DataFrame,
    mask_shape: Tuple[int, int],
    scale_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float, str]:
    """Compute a 2D convex hull around cell coordinates to delineate tissue boundaries.

    Args:
        df: Table containing spatial cell coordinates (Pos_Y, Pos_X or similar).
        mask_shape: (height, width) of the target mask canvas.
        scale_factor: Physical resolution in micrometers per pixel (µm/px).

    Returns:
        Tuple of:
            - hull_vertices: (N, 2) array of hull vertex coordinates in (Y, X) order.
            - tissue_mask: (H, W) uint8 binary rasterized polygon mask.
            - area_value: Calculated tissue area (physical µm² if calibrated, else px²).
            - area_text: Formatted human-readable area string with units.

    Raises:
        ValueError: If fewer than 3 spatial coordinates are present in df.
    """
    if len(df) < 3:
        raise ValueError("Convex hull generation requires at least 3 cell coordinates.")

    ensure_spatial_coordinates(df)
    points = df[["Pos_Y", "Pos_X"]].dropna().values
    if len(points) < 3:
        raise ValueError("Convex hull requires at least 3 non-NaN coordinate points.")

    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]

    if len(mask_shape) > 2: mask_shape = (mask_shape[-2], mask_shape[-1])

    tissue_mask = np.zeros(mask_shape, dtype=np.uint8)
    clamped_y = np.clip(hull_vertices[:, 0].astype(int), 0, mask_shape[0] - 1)
    clamped_x = np.clip(hull_vertices[:, 1].astype(int), 0, mask_shape[1] - 1)

    rr, cc = polygon(clamped_y, clamped_x, shape=mask_shape)
    tissue_mask[rr, cc] = 1

    pixel_count = float(np.sum(tissue_mask))
    physical_area = pixel_count * (scale_factor ** 2)

    area_value = physical_area if scale_factor != 1.0 else pixel_count
    unit_label = "µm²" if scale_factor != 1.0 else "px²"
    area_text = f"{area_value:.2f} {unit_label}" if scale_factor != 1.0 else f"{int(pixel_count)} {unit_label}"

    return hull_vertices, tissue_mask, area_value, area_text


def recompute_area_from_vertices(
    vertices: np.ndarray,
    mask_shape: Tuple[int, int],
    scale_factor: float = 1.0,
) -> Tuple[float, str]:
    """Recalculate geometric area when polygon boundary vertices are interactively edited.

    Uses the Shoelace formula for O(N) evaluation, bypassing full-canvas rasterization
    to prevent memory spikes and UI lag on gigapixel images.

    Args:
        vertices: (N, 2) array of vertex coordinates in (Y, X) order.
        mask_shape: (height, width) of the target mask canvas.
        scale_factor: Physical resolution in micrometers per pixel (µm/px).

    Returns:
        Tuple of (area_value, area_text).
    """
    if len(vertices) < 3:
        return 0.0, "--"

    # Shoelace formula for polygonal area: 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    y = vertices[:, 0]
    x = vertices[:, 1]
    pixel_count = 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    physical_area = pixel_count * (scale_factor ** 2)
    area_value = physical_area if scale_factor != 1.0 else pixel_count
    unit_label = "µm²" if scale_factor != 1.0 else "px²"
    area_text = f"{area_value:.2f} {unit_label}" if scale_factor != 1.0 else f"{int(round(pixel_count))} {unit_label}"

    return area_value, area_text
