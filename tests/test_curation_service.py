"""
Tests for iris_viewer.app.services.curation_service:
- Position-to-cell resolution in single-sample and multi-sample layouts.
- Safe phenotype retrieval across resolved_dfs, current_mapped_df, and master_df.
- Dynamic visual layer creation and mask data updating during phenotype reclassification.
- Quick curation click handling and Alt modifier support.
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import napari
import numpy as np
import pandas as pd
import pytest
from qtpy.QtWidgets import QApplication

from iris_viewer.app.services.curation_service import CurationService
from iris_viewer.core.state import ViewerAppState


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def _create_mock_context():
    """Create a mock application context with a headless Napari viewer."""
    viewer = napari.Viewer(show=False)
    state = ViewerAppState()

    # Create 2 samples: sample_A (at origin) and sample_B (translated x=200)
    mask_a = np.zeros((100, 100), dtype=np.uint32)
    mask_a[10:25, 10:25] = 101  # cell 101
    mask_a[50:65, 50:65] = 102  # cell 102

    mask_b = np.zeros((100, 100), dtype=np.uint32)
    mask_b[10:25, 10:25] = 201  # cell 201
    mask_b[50:65, 50:65] = 202  # cell 202

    state.masks_map["sample_A"] = mask_a
    state.masks_map["sample_B"] = mask_b

    lyr_a = viewer.add_labels(mask_a, name="Nuclear Segmentation", scale=(1.0, 1.0))
    lyr_a.translate = (0.0, 0.0)

    lyr_b = viewer.add_labels(mask_b, name="Nuclear Segmentation", scale=(1.0, 1.0))
    lyr_b.translate = (0.0, 200.0)

    # Initial phenotype overlay for sample_A (cell 101: Tumor, 102: Stroma)
    tumor_a = np.zeros((100, 100), dtype=np.uint8)
    tumor_a[10:25, 10:25] = 1
    lyr_tumor_a = viewer.add_image(tumor_a, name="Tumor", scale=(1.0, 1.0))
    lyr_tumor_a.translate = (0.0, 0.0)

    # Initial phenotype overlay for sample_B (cell 201: B cell, 202: T cell)
    bcell_b = np.zeros((100, 100), dtype=np.uint8)
    bcell_b[10:25, 10:25] = 1
    lyr_bcell_b = viewer.add_image(bcell_b, name="B cell", scale=(1.0, 1.0))
    lyr_bcell_b.translate = (0.0, 200.0)

    # Bounding boxes
    bbox_a = viewer.add_shapes([np.array([[0, 0], [0, 100], [100, 100], [100, 0]])], name="Bounds - sample_A", scale=(1.0, 1.0))
    bbox_a.translate = (0.0, 0.0)

    bbox_b = viewer.add_shapes([np.array([[0, 0], [0, 100], [100, 100], [100, 0]])], name="Bounds - sample_B", scale=(1.0, 1.0))
    bbox_b.translate = (0.0, 200.0)

    state.sample_layers_map["sample_A"] = [lyr_a, lyr_tumor_a, bbox_a]
    state.sample_layers_map["sample_B"] = [lyr_b, lyr_bcell_b, bbox_b]

    # DataFrames
    df_a = pd.DataFrame({
        "Image": ["sample_A", "sample_A"],
        "ObjectNumber": [101, 102],
        "Pos_X": [17.0, 57.0],
        "Pos_Y": [17.0, 57.0],
        "cell_type": ["Tumor", "Stroma"],
    })
    df_b = pd.DataFrame({
        "Image": ["sample_B", "sample_B"],
        "ObjectNumber": [201, 202],
        "Pos_X": [17.0, 57.0],
        "Pos_Y": [17.0, 57.0],
        "cell_type": ["B cell", "T cell"],
    })

    state.resolved_dfs["sample_A"] = df_a.copy()
    state.resolved_dfs["sample_B"] = df_b.copy()
    state.master_df = pd.concat([df_a, df_b], ignore_index=True)
    state.current_mapped_df = state.master_df.copy()

    # Mock control widget
    cw = MagicMock()
    cw.combo_ref.currentText.return_value = "cell_type"
    cw.combo_quick_type.currentText.return_value = "Tumor"
    cw.combo_quick_type.count.return_value = 4
    cw.combo_quick_type.itemText.side_effect = lambda i: ["Tumor", "Stroma", "B cell", "T cell"][i]
    cw.chk_quick_curate.isChecked.return_value = False

    # Mock comparison widget
    msw = MagicMock()
    msw.combo_ref.currentText.return_value = "cell_type"
    msw.combo_quick_type.currentText.return_value = "Tumor"
    msw.combo_quick_type.count.return_value = 4
    msw.combo_quick_type.itemText.side_effect = lambda i: ["Tumor", "Stroma", "B cell", "T cell"][i]
    msw.chk_quick_curate.isChecked.return_value = False

    ctx = types.SimpleNamespace(
        viewer=viewer,
        state=state,
        control_widget=cw,
        comparison_widget=msw,
        legend_widget=MagicMock(),
    )
    return ctx


class TestCurationServiceMultiSample:
    """Tests for multi-sample canvas curation workflows."""

    def test_get_cell_at_position_resolves_correct_sample(self, qapp):
        ctx = _create_mock_context()
        svc = CurationService(ctx)

        # Cursor in sample_A cell 101: position (15.0, 15.0)
        obj_id, sample_name = svc.get_cell_at_position((15.0, 15.0))
        assert obj_id == 101
        assert sample_name == "sample_A"

        # Cursor in sample_B cell 201: position (15.0, 215.0) -> data x = 215 - 200 = 15
        obj_id, sample_name = svc.get_cell_at_position((15.0, 215.0))
        assert obj_id == 201
        assert sample_name == "sample_B"

        # Cursor in background space (no cell): position (0.0, 0.0)
        obj_id, sample_name = svc.get_cell_at_position((0.0, 0.0))
        assert obj_id is None
        assert sample_name is None

    def test_get_cell_at_position_fallback_without_labels_layer(self, qapp):
        """When Labels layer is missing or removed, fallback to bbox translation works."""
        ctx = _create_mock_context()
        # Remove the Labels layer from sample_B
        ctx.state.sample_layers_map["sample_B"] = [
            l for l in ctx.state.sample_layers_map["sample_B"]
            if l.__class__.__name__ != "Labels"
        ]
        svc = CurationService(ctx)

        obj_id, sample_name = svc.get_cell_at_position((15.0, 215.0))
        assert obj_id == 201
        assert sample_name == "sample_B"

    def test_get_cell_phenotype_retrieval(self, qapp):
        ctx = _create_mock_context()
        svc = CurationService(ctx)

        # Cell 101 in sample_A is Tumor
        p_a = svc._get_cell_phenotype(101, "sample_A", "cell_type")
        assert p_a == "Tumor"

        # Cell 201 in sample_B is B cell
        p_b = svc._get_cell_phenotype(201, "sample_B", "cell_type")
        assert p_b == "B cell"

        # Non-existent cell returns Unassigned
        p_unknown = svc._get_cell_phenotype(999, "sample_B", "cell_type")
        assert p_unknown == "Unassigned"

    def test_apply_quick_phenotype_change_updates_dataframes(self, qapp):
        ctx = _create_mock_context()
        svc = CurationService(ctx)

        # Reclassify sample_B cell 201 from "B cell" to "Tumor"
        svc.apply_quick_phenotype_change(201, "Tumor", "cell_type", "sample_B")

        # Verify resolved_dfs for sample_B
        s_df = ctx.state.resolved_dfs["sample_B"]
        row = s_df[s_df["ObjectNumber"] == 201].iloc[0]
        assert row["cell_type"] == "Tumor"
        assert row["Verified_CellType"] == "Tumor"

        # Verify current_mapped_df
        c_df = ctx.state.current_mapped_df
        c_row = c_df[(c_df["ObjectNumber"] == 201) & (c_df["Image"] == "sample_B")].iloc[0]
        assert c_row["cell_type"] == "Tumor"
        assert c_row["Verified_CellType"] == "Tumor"

        # Verify master_df
        m_df = ctx.state.master_df
        m_row = m_df[(m_df["ObjectNumber"] == 201) & (m_df["Image"] == "sample_B")].iloc[0]
        assert m_row["cell_type"] == "Tumor"
        assert m_row["Verified_CellType"] == "Tumor"

    def test_apply_quick_phenotype_change_creates_new_layer_dynamically(self, qapp):
        """When target phenotype has no layer in sample_B, it must be created dynamically."""
        ctx = _create_mock_context()
        svc = CurationService(ctx)

        s_layers_b = ctx.state.sample_layers_map["sample_B"]
        # Confirm sample_B does not have a "Tumor" image layer yet
        assert not any(l.name.split(" [")[0] == "Tumor" for l in s_layers_b)

        # Reclassify cell 201 to "Tumor"
        svc.apply_quick_phenotype_change(201, "Tumor", "cell_type", "sample_B")

        # Now sample_B must have a "Tumor" layer
        tumor_layer = next((l for l in s_layers_b if l.name.split(" [")[0] == "Tumor"), None)
        assert tumor_layer is not None
        assert tumor_layer.data[15, 15] == 1
        assert np.array_equal(tumor_layer.translate, [0.0, 200.0])

        # And the old layer (B cell) must have cell 201 cleared
        bcell_layer = next((l for l in s_layers_b if l.name.split(" [")[0] == "B cell"), None)
        assert bcell_layer is not None
        assert bcell_layer.data[15, 15] == 0

    def test_quick_curate_click_via_comparison_widget(self, qapp):
        """Quick curation triggered from comparison_widget works seamlessly."""
        ctx = _create_mock_context()
        svc = CurationService(ctx)

        # Enable quick curate on comparison_widget
        ctx.comparison_widget.chk_quick_curate.isChecked.return_value = True
        ctx.comparison_widget.combo_quick_type.currentText.return_value = "Stroma"

        # Simulate left-click event at sample_A cell 101 (pos: 15.0, 15.0)
        event = types.SimpleNamespace(button=1, position=(15.0, 15.0), modifiers=[])
        gen = svc.on_quick_curate_click(ctx.viewer, event)
        # Advance generator through yield
        list(gen)

        # Verify cell 101 was reclassified to Stroma
        s_df = ctx.state.resolved_dfs["sample_A"]
        assert s_df.loc[s_df["ObjectNumber"] == 101, "cell_type"].iloc[0] == "Stroma"

    def test_get_available_phenotypes_excludes_lowercase_colors(self, qapp):
        """Available phenotypes must strictly reflect dataset cell types without lowercase color dictionary keys."""
        ctx = _create_mock_context()
        ctx.state.celltype_colors = {"tumor": "#FF0000", "stroma": "#00FF00", "b cell": "#0000FF"}
        svc = CurationService(ctx)

        types_list = svc._get_available_phenotypes("cell_type")
        # Dataset cell types: "Tumor", "Stroma", "B cell", "T cell"
        assert "tumor" not in types_list
        assert "stroma" not in types_list
        assert "b cell" not in types_list
        assert set(types_list) == {"B cell", "Stroma", "T cell", "Tumor"}

    def test_apply_quick_curate_with_dask_channel_layer(self, qapp):
        """Dask array image channels in sample_layers must not raise IndexError during curation."""
        import dask.array as da
        ctx = _create_mock_context()
        ctx.state.markers = ["DAPI", "CK"]

        # Add a 2D Dask array layer as DAPI
        dask_dapi = da.from_array(np.ones((100, 100), dtype=np.float32), chunks=(100, 100))
        dapi_layer = ctx.viewer.add_image(dask_dapi, name="DAPI", scale=(1.0, 1.0))
        ctx.state.sample_layers_map["sample_A"].insert(0, dapi_layer)

        svc = CurationService(ctx)
        # Reclassify cell 101 from Tumor to Stroma; must NOT raise IndexError from Dask slicing
        svc.apply_quick_phenotype_change(101, "Stroma", "cell_type", "sample_A")

        s_df = ctx.state.resolved_dfs["sample_A"]
        assert s_df.loc[s_df["ObjectNumber"] == 101, "cell_type"].iloc[0] == "Stroma"

    def test_apply_quick_curate_with_centroids_points_layer(self, qapp):
        """Points layer ('Centroids') with shape (N, 2) in sample_layers must not be indexed with 2D mask."""
        ctx = _create_mock_context()

        # Add a Points layer with shape (50, 2)
        pts_coords = np.random.rand(50, 2) * 100
        pts_layer = ctx.viewer.add_points(pts_coords, name="Centroids", size=5)
        ctx.state.sample_layers_map["sample_A"].append(pts_layer)

        svc = CurationService(ctx)
        # Reclassify cell 101 from Tumor to Stroma; must NOT raise IndexError from Centroids
        svc.apply_quick_phenotype_change(101, "Stroma", "cell_type", "sample_A")

        s_df = ctx.state.resolved_dfs["sample_A"]
        assert s_df.loc[s_df["ObjectNumber"] == 101, "cell_type"].iloc[0] == "Stroma"

    def test_multiple_successive_phenotype_changes_with_layer_mutual_exclusion(self, qapp):
        """Cell can be reclassified repeatedly without limit and layers maintain strict mutual exclusion."""
        ctx = _create_mock_context()
        svc = CurationService(ctx)

        # Cell 101 starts as Tumor
        assert svc._get_cell_phenotype(101, "sample_A", "cell_type") == "Tumor"

        # Sequence of successive phenotype changes
        sequence = ["Stroma", "B cell", "NovelPheno", "T cell", "Tumor"]

        for target_type in sequence:
            svc.apply_quick_phenotype_change(101, target_type, "cell_type", "sample_A")

            # 1. Verification in state DataFrames
            assert svc._get_cell_phenotype(101, "sample_A", "cell_type") == target_type
            s_df = ctx.state.resolved_dfs["sample_A"]
            assert s_df.loc[s_df["ObjectNumber"] == 101, "cell_type"].iloc[0] == target_type
            m_df = ctx.state.master_df
            m_row = m_df[(m_df["ObjectNumber"] == 101) & (m_df["Image"] == "sample_A")].iloc[0]
            assert m_row["cell_type"] == target_type

            # 2. Verification of strict visual layer mutual exclusion (no blending)
            s_layers = ctx.state.sample_layers_map["sample_A"]
            active_matching_layers = []
            for lyr in s_layers:
                if lyr.__class__.__name__ == 'Image' and lyr.name not in getattr(ctx.state, "markers", []):
                    b_name = lyr.name.split(" [")[0].strip()
                    if b_name in ("Nuclear Segmentation", "Cytoplasm Segmentation", "Bounds - sample_A"):
                        continue
                    if lyr.data[15, 15] > 0:
                        active_matching_layers.append(b_name)

            # Exactly ONE phenotype layer must contain cell 101, matching target_type
            assert len(active_matching_layers) == 1, f"Expected 1 active layer, found: {active_matching_layers}"
            assert active_matching_layers[0] == target_type

    def test_quick_curate_click_via_control_widget(self, qapp):
        """Quick curation triggered from primary control_widget works seamlessly."""
        ctx = _create_mock_context()
        svc = CurationService(ctx)

        # Enable quick curate on control_widget
        ctx.control_widget.chk_quick_curate.isChecked.return_value = True
        ctx.control_widget.combo_quick_type.currentText.return_value = "B cell"

        # Simulate left-click event at sample_A cell 101 (pos: 15.0, 15.0)
        event = types.SimpleNamespace(button=1, position=(15.0, 15.0), modifiers=[])
        gen = svc.on_quick_curate_click(ctx.viewer, event)
        list(gen)

        # Verify cell 101 was reclassified to B cell
        s_df = ctx.state.resolved_dfs["sample_A"]
        assert s_df.loc[s_df["ObjectNumber"] == 101, "cell_type"].iloc[0] == "B cell"


