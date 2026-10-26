"""
Unit tests for iris_viewer.io.data_adapter module:
- IrisPhenotypeMapper.resolve_ids: Vectorized ID lookup with 3x3 dilation window fallback
- IrisPhenotypeMapper.get_masks_by_type: Fast categorical LUT binary mask generation by cell type
- IrisDataAdapter: Annotation parsing and canonical Pos_X/Pos_Y coordinate standardisation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from iris_viewer.io.data_adapter import IrisDataAdapter, IrisPhenotypeMapper


def test_iris_phenotype_mapper_resolve_ids_vectorized_and_dilation_fallback():
    """Verify IrisPhenotypeMapper.resolve_ids handles direct lookup, 3x3 window dilation fallback, and out-of-bounds."""
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


def test_iris_phenotype_mapper_get_masks_by_type_categorical_lut():
    """Verify IrisPhenotypeMapper.get_masks_by_type produces accurate binary masks via fast categorical LUT."""
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


def test_iris_data_adapter_canonical_coordinates_and_sample_extraction(tmp_path):
    """Verify IrisDataAdapter produces canonical Pos_X, Pos_Y and per_sample slices."""
    csv_file = tmp_path / "annotations.csv"
    raw_df = pd.DataFrame({
        "Image": ["sample_1.tiff", "sample_1.tiff", "sample_2.tiff"],
        "ObjectNumber": [1, 2, 3],
        "Centroid_X": [15.5, 25.5, 35.5],
        "Centroid_Y": [10.0, 20.0, 30.0],
        "cell_type": ["B_cell", "T_cell", "Tumor"],
        "CD3": [100.0, 200.0, 50.0],
    })
    raw_df.to_csv(csv_file, index=False)

    adapter = IrisDataAdapter.from_file(csv_file)
    norm = adapter.normalised

    # Canonical columns standardisation
    assert "Pos_X" in norm.columns
    assert "Pos_Y" in norm.columns
    assert "ObjectNumber" in norm.columns
    assert "cell_type" in norm.columns
    assert list(norm["Pos_X"]) == [15.5, 25.5, 35.5]
    assert list(norm["Pos_Y"]) == [10.0, 20.0, 30.0]

    # Sample slicing
    samples = adapter.per_sample()
    assert "sample_1" in samples
    assert "sample_2" in samples
    assert len(samples["sample_1"]) == 2
    assert len(samples["sample_2"]) == 1
