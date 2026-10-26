"""
Unit tests for P1 performance optimizations and geometry recalculations:
- Shoelace formula area computation in geometry.py
- Vectorized centroid / local 3x3 window ID resolution in data_adapter.py
- Fast categorical LUT mask generation by cell type in data_adapter.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iris_viewer.engine.geometry import recompute_area_from_vertices
from iris_viewer.io.data_adapter import IrisPhenotypeMapper


def test_recompute_area_from_vertices_shoelace():
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


def test_resolve_ids_vectorized_and_fallback():
    """Verify vectorized resolve_ids handles direct lookup, 3x3 window dilation fallback, and out-of-bounds."""
    # Create a 20x20 label mask with cell 101 at (5, 5) and cell 102 at (15, 15)
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[5:8, 5:8] = 101
    mask[15:18, 15:18] = 102

    df = pd.DataFrame({
        "Pos_X": [6.0, 4.0, 16.0, -5.0, 100.0, "invalid"],
        "Pos_Y": [6.0, 5.0, 16.0, 10.0, 10.0, 5.0],
        "cell_type": ["Tumor", "Tumor", "Stroma", "Immune", "Immune", "Other"],
    })

    mapper = IrisPhenotypeMapper(mask=mask, df=df, target_col="cell_type")
    resolved_df = mapper.resolve_ids()

    assert "ObjectNumber" in resolved_df.columns
    # Row 0: Direct hit at (6, 6) -> 101
    assert resolved_df.loc[0, "ObjectNumber"] == 101
    # Row 1: Centroid at (4, 5) is 1 px to the left of mask[5:8, 5:8], 3x3 window hits 101
    assert resolved_df.loc[1, "ObjectNumber"] == 101
    # Row 2: Direct hit at (16, 16) -> 102
    assert resolved_df.loc[2, "ObjectNumber"] == 102
    # Row 3, 4, 5: Out of bounds or invalid -> 0
    assert resolved_df.loc[3, "ObjectNumber"] == 0
    assert resolved_df.loc[4, "ObjectNumber"] == 0
    assert resolved_df.loc[5, "ObjectNumber"] == 0


def test_get_masks_by_type_fast_lut():
    """Verify categorical LUT path produces identical, accurate binary masks for each cell type."""
    h, w = 30, 30
    mask = np.zeros((h, w), dtype=np.int32)
    # Cells 1 and 2 are Tumor; Cell 3 is Stroma
    mask[2:5, 2:5] = 1
    mask[8:12, 8:12] = 2
    mask[20:25, 20:25] = 3

    df = pd.DataFrame({
        "ObjectNumber": [1, 2, 3],
        "cell_type": ["Tumor", "Tumor", "Stroma"],
    })

    mapper = IrisPhenotypeMapper(mask=mask, df=df, target_col="cell_type")
    type_masks = mapper.get_masks_by_type()

    assert "Tumor" in type_masks
    assert "Stroma" in type_masks

    tumor_mask, _ = type_masks["Tumor"]
    stroma_mask, _ = type_masks["Stroma"]

    # Tumor should have pixels from cell 1 (3x3 = 9) and cell 2 (4x4 = 16) = 25 pixels
    assert np.sum(tumor_mask) == 25
    assert np.all(tumor_mask[2:5, 2:5] == 1)
    assert np.all(tumor_mask[8:12, 8:12] == 1)
    assert np.all(tumor_mask[20:25, 20:25] == 0)

    # Stroma should have pixels from cell 3 (5x5 = 25 pixels)
    assert np.sum(stroma_mask) == 25
    assert np.all(stroma_mask[20:25, 20:25] == 1)
    assert np.all(stroma_mask[2:5, 2:5] == 0)
