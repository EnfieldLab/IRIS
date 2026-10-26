"""
Unit tests for iris_viewer.engine.geometry module:
- ensure_spatial_coordinates: Canonical 'Pos_X' and 'Pos_Y' coordinate validation and mapping
- recompute_area_from_vertices: Shoelace formula polygon area computation
- compute_tissue_boundary_mask: Convex hull boundary rasterization
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iris_viewer.engine.geometry import (
    compute_tissue_boundary_mask,
    ensure_spatial_coordinates,
    recompute_area_from_vertices,
)


def test_ensure_spatial_coordinates_with_canonical_pos_x_pos_y():
    """Verify ensure_spatial_coordinates accepts already canonical Pos_X and Pos_Y."""
    df_canonical = pd.DataFrame({
        "Pos_X": [10.0, 20.0],
        "Pos_Y": [15.0, 25.0],
        "cell_type": ["Tumor", "Stroma"],
    })
    x_col, y_col = ensure_spatial_coordinates(df_canonical)
    assert (x_col, y_col) == ("Pos_X", "Pos_Y")
    assert "Pos_X" in df_canonical.columns and "Pos_Y" in df_canonical.columns
    assert list(df_canonical["Pos_X"]) == [10.0, 20.0]
    assert list(df_canonical["Pos_Y"]) == [15.0, 25.0]


def test_ensure_spatial_coordinates_maps_alternative_centroid_columns():
    """Verify ensure_spatial_coordinates maps Centroid_X / Centroid_Y to Pos_X / Pos_Y in place."""
    df_centroid = pd.DataFrame({
        "Centroid_X": [100.0, 200.0],
        "Centroid_Y": [150.0, 250.0],
        "label": [1, 2],
    })
    x_col, y_col = ensure_spatial_coordinates(df_centroid)
    assert (x_col, y_col) == ("Pos_X", "Pos_Y")
    assert "Pos_X" in df_centroid.columns and "Pos_Y" in df_centroid.columns
    assert list(df_centroid["Pos_X"]) == [100.0, 200.0]
    assert list(df_centroid["Pos_Y"]) == [150.0, 250.0]


def test_ensure_spatial_coordinates_maps_lowercase_xy_columns():
    """Verify ensure_spatial_coordinates maps lowercase x / y to Pos_X / Pos_Y."""
    df_lower = pd.DataFrame({
        "x": [1.5, 2.5],
        "y": [3.5, 4.5],
    })
    ensure_spatial_coordinates(df_lower)
    assert "Pos_X" in df_lower.columns and "Pos_Y" in df_lower.columns
    assert list(df_lower["Pos_X"]) == [1.5, 2.5]
    assert list(df_lower["Pos_Y"]) == [3.5, 4.5]


def test_ensure_spatial_coordinates_raises_on_missing_columns():
    """Verify ensure_spatial_coordinates raises ValueError if spatial columns are missing."""
    df_missing = pd.DataFrame({"Gene": ["CD3", "CD8"], "Expression": [10.0, 5.0]})
    with pytest.raises(ValueError, match="DataFrame missing spatial coordinate columns"):
        ensure_spatial_coordinates(df_missing)


def test_recompute_area_from_vertices_shoelace_formula():
    """Verify Shoelace formula correctly calculates polygon areas without rasterization."""
    # Degenerate cases
    assert recompute_area_from_vertices(np.array([]), (100, 100)) == (0.0, "--")
    assert recompute_area_from_vertices(np.array([[0, 0], [10, 0]]), (100, 100)) == (0.0, "--")

    # 10x10 square vertices in (Y, X) order
    square_verts = np.array([
        [10.0, 10.0],
        [10.0, 20.0],
        [20.0, 20.0],
        [20.0, 10.0],
    ])

    # Unscaled (pixels)
    area_px, label_px = recompute_area_from_vertices(square_verts, (50, 50), scale_factor=1.0)
    assert pytest.approx(area_px, 0.01) == 100.0
    assert label_px == "100 px²"

    # Scaled (micrometers, scale_factor = 0.5 µm/px -> 100 px * 0.25 = 25 µm²)
    area_um, label_um = recompute_area_from_vertices(square_verts, (50, 50), scale_factor=0.5)
    assert pytest.approx(area_um, 0.01) == 25.0
    assert label_um == "25.00 µm²"


def test_compute_tissue_boundary_mask_convex_hull_generation():
    """Verify compute_tissue_boundary_mask constructs a binary convex hull mask."""
    # Triangle of points
    df = pd.DataFrame({
        "Pos_X": [10.0, 30.0, 20.0],
        "Pos_Y": [10.0, 10.0, 30.0],
    })
    hull_verts, mask, area_val, area_txt = compute_tissue_boundary_mask(df, (50, 50))
    assert len(hull_verts) == 3
    assert mask.shape == (50, 50)
    assert mask.dtype == np.uint8
    assert np.any(mask == 1)
    assert area_val > 0
    assert "px²" in area_txt
    # Interior point (20, 15) should be inside the triangle
    assert mask[15, 20] == 1
    # Point outside triangle (5, 5) should be 0
    assert mask[5, 5] == 0


def test_compute_tissue_boundary_mask_raises_on_insufficient_points():
    """Verify compute_tissue_boundary_mask raises ValueError when fewer than 3 points are provided."""
    df_few = pd.DataFrame({"Pos_X": [1.0, 2.0], "Pos_Y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="at least 3"):
        compute_tissue_boundary_mask(df_few, (50, 50))
