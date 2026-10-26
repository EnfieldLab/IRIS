"""
Unit tests for iris_viewer.app.services.multi_sample_service module:
- get_sample_sub_df: Sample filtering and canonical 'Pos_X' / 'Pos_Y' coordinate validation
- MultiSampleService.update_phenotypes_on_canvas: Synchronizing phenotype layers across sample switches
- ViewerLegendWidget: Row deduplication and state synchronization
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import napari
import numpy as np
import pandas as pd
import pytest
from qtpy.QtWidgets import QApplication

from iris_viewer.app.services.image_service import ImageService
from iris_viewer.app.services.multi_sample_service import MultiSampleService, get_sample_sub_df
from iris_viewer.core.context import AppContext
from iris_viewer.core.state import ViewerAppState
from iris_viewer.widgets.legend import ViewerLegendWidget


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def test_get_sample_sub_df_ensures_canonical_pos_x_pos_y():
    """Verify get_sample_sub_df standardizes coordinates into Pos_X and Pos_Y."""
    df = pd.DataFrame({
        "Image": ["sample_a", "sample_a", "sample_b"],
        "Centroid_X": [12.5, 14.5, 99.0],
        "Centroid_Y": [22.0, 24.0, 99.0],
        "cell_type": ["B_cell", "T_cell", "Tumor"],
    })
    sub_df = get_sample_sub_df(df, "sample_a")
    assert len(sub_df) == 2
    assert "Pos_X" in sub_df.columns
    assert "Pos_Y" in sub_df.columns
    assert list(sub_df["Pos_X"]) == [12.5, 14.5]
    assert list(sub_df["Pos_Y"]) == [22.0, 24.0]


def test_get_sample_sub_df_tiered_sample_name_matching():
    """Verify get_sample_sub_df matches sample identifiers with stripped extensions or suffixes."""
    df = pd.DataFrame({
        "Image": ["TMA_Core_A1_image.tiff", "TMA_Core_A1_image.tiff", "TMA_Core_B2.tiff"],
        "Pos_X": [10.0, 20.0, 30.0],
        "Pos_Y": [15.0, 25.0, 35.0],
    })
    # Stripped match for TMA_Core_A1
    sub_a = get_sample_sub_df(df, "TMA_Core_A1")
    assert len(sub_a) == 2

    # Stripped match for TMA_Core_B2
    sub_b = get_sample_sub_df(df, "TMA_Core_B2")
    assert len(sub_b) == 1

    # Nonexistent sample returns empty DataFrame
    sub_c = get_sample_sub_df(df, "Nonexistent_Core")
    assert sub_c.empty


def _create_sample_switching_context(tmp_path: Path) -> AppContext:
    """Set up a test context with 2 distinct samples and annotation master_df."""
    viewer = napari.Viewer(show=False)
    state = ViewerAppState()
    legend_widget = ViewerLegendWidget(viewer)

    ctx = AppContext(
        viewer=viewer,
        state=state,
        legend_widget=legend_widget,
        added_legend_entries=set(),
        active_workers=[],
    )
    legend_widget.ctx = ctx

    # Mock ControlWidget
    cw = MagicMock()
    cw.combo_ref.currentText.return_value = "cell_type"
    cw.spin_scale.value.return_value = 1.0
    cw.update_dashboard = MagicMock()
    ctx.control_widget = cw

    # Services
    ms_svc = MultiSampleService(ctx)
    img_svc = ImageService(ctx)
    ctx.multi_sample_svc = ms_svc
    ctx.image_svc = img_svc

    # Sample A mask and df
    mask_a = np.zeros((100, 100), dtype=np.uint32)
    mask_a[10:20, 10:20] = 1
    mask_a[30:40, 30:40] = 2
    mask_a[50:60, 50:60] = 3

    # Sample B mask and df
    mask_b = np.zeros((100, 100), dtype=np.uint32)
    mask_b[15:25, 15:25] = 1
    mask_b[35:45, 35:45] = 2
    mask_b[55:65, 55:65] = 3

    master_df = pd.DataFrame({
        "Image": [
            "sample_A", "sample_A", "sample_A",
            "sample_B", "sample_B", "sample_B"
        ],
        "ObjectNumber": [1, 2, 3, 1, 2, 3],
        "cell_type": [
            "Tumor", "T cells", "B cells",
            "B cells", "Macrophage", "Epithelial"
        ],
        "Pos_X": [15, 35, 55, 20, 40, 60],
        "Pos_Y": [15, 35, 55, 20, 40, 60],
    })
    state.master_df = master_df

    ctx._sample_masks = {
        "sample_A": mask_a,
        "sample_B": mask_b,
    }

    return ctx


def test_multi_sample_service_single_sample_switch_synchronizes_phenotype_legends(tmp_path, qapp):
    """Verify switching between single samples correctly updates active cell phenotype legend rows."""
    ctx = _create_sample_switching_context(tmp_path)
    legend_widget = ctx.legend_widget
    ms_svc = ctx.multi_sample_svc

    # --- Step 1: Load Sample A ---
    sample_a = "sample_A"
    ctx.state.current_sample_name = sample_a
    ctx.state.masks_map[sample_a] = ctx._sample_masks[sample_a]
    ctx.state.sample_layers_map[sample_a] = []

    ms_svc.update_phenotypes_on_canvas()

    legend_names_a = [r.name for r in legend_widget.ref_ct_rows]
    assert sorted(legend_names_a) == ["B cells", "T cells", "Tumor"]
    assert any(r.name == "Centroids" for r in legend_widget.seg_rows)
    assert ctx.state.current_sample_name == "sample_A"

    # --- Step 2: Switch to Sample B ---
    sample_b = "sample_B"
    ctx.viewer.layers.clear()
    ctx.legend_widget.clear()
    ctx.added_legend_entries.clear()
    if hasattr(ctx, "multi_sample_svc") and hasattr(ctx.multi_sample_svc, "_added_master_legends"):
        ctx.multi_sample_svc._added_master_legends.clear()
    ctx.state.sample_layers_map.clear()
    ctx.state.reset_sample_state()
    ctx.state.current_sample_name = sample_b

    ctx.state.masks_map[sample_b] = ctx._sample_masks[sample_b]
    ctx.state.sample_layers_map[sample_b] = []

    ms_svc.update_phenotypes_on_canvas()

    legend_names_b = [r.name for r in legend_widget.ref_ct_rows]
    # ALL of sample_B's phenotypes MUST appear, including "B cells" which was also in Sample A
    assert sorted(legend_names_b) == ["B cells", "Epithelial", "Macrophage"]
    assert "T cells" not in legend_names_b
    assert "Tumor" not in legend_names_b
    assert any(r.name == "Centroids" for r in legend_widget.seg_rows)

    # --- Step 3: Switch back to Sample A ---
    ctx.viewer.layers.clear()
    ctx.legend_widget.clear()
    ctx.added_legend_entries.clear()
    if hasattr(ctx, "multi_sample_svc") and hasattr(ctx.multi_sample_svc, "_added_master_legends"):
        ctx.multi_sample_svc._added_master_legends.clear()
    ctx.state.sample_layers_map.clear()
    ctx.state.reset_sample_state()
    ctx.state.current_sample_name = sample_a

    ctx.state.masks_map[sample_a] = ctx._sample_masks[sample_a]
    ctx.state.sample_layers_map[sample_a] = []

    ms_svc.update_phenotypes_on_canvas()

    legend_names_a_again = [r.name for r in legend_widget.ref_ct_rows]
    assert sorted(legend_names_a_again) == ["B cells", "T cells", "Tumor"]
    assert "Macrophage" not in legend_names_a_again
    assert "Epithelial" not in legend_names_a_again


def test_multi_sample_service_legend_row_deduplication(tmp_path, qapp):
    """Verify adding the same phenotype row multiple times does not produce duplicates."""
    ctx = _create_sample_switching_context(tmp_path)
    legend_widget = ctx.legend_widget

    legend_widget.add_ref_ct_row(None, "B cells", "#FF0000")
    legend_widget.add_ref_ct_row(None, "B cells", "#FF0000")
    legend_widget.add_ref_ct_row(None, "B cells", "#00FF00")

    matching = [r for r in legend_widget.ref_ct_rows if r.name == "B cells"]
    assert len(matching) == 1
    assert matching[0].hex_color == "#00FF00"


def test_multi_sample_service_legend_clear_synchronizes_state(tmp_path, qapp):
    """Verify calling clear() on legend_widget resets rows and empties tracking sets."""
    ctx = _create_sample_switching_context(tmp_path)
    legend_widget = ctx.legend_widget
    ms_svc = ctx.multi_sample_svc

    legend_widget.add_ref_ct_row(None, "Plasma cells", "#123456")
    ms_svc._added_master_legends.add("Plasma cells")
    ctx.added_legend_entries.add("DAPI")

    assert len(legend_widget.ref_ct_rows) == 1
    assert "Plasma cells" in ms_svc._added_master_legends
    assert "DAPI" in ctx.added_legend_entries

    legend_widget.clear()

    assert len(legend_widget.ref_ct_rows) == 0
    assert len(ms_svc._added_master_legends) == 0
    assert len(ctx.added_legend_entries) == 0
