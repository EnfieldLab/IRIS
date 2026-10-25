"""
Tests for sample switching and legend synchronization:
- Verifies that switching between single samples correctly displays all active cell phenotype
  legend rows for each sample, including phenotypes shared across samples.
- Verifies that Centroids legend rows are preserved across sample switches.
- Verifies that legend rows are properly deduplicated and tracking sets remain synchronized.
"""
from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock

import napari
import numpy as np
import pandas as pd
import pytest
from qtpy.QtWidgets import QApplication

from iris_viewer.app.services.image_service import ImageService
from iris_viewer.app.services.multi_sample_service import MultiSampleService
from iris_viewer.core.state import ViewerAppState
from iris_viewer.widgets.legend import ViewerLegendWidget


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def _create_sample_switching_context(tmp_path: Path):
    """Set up a test context with 2 distinct samples and annotation master_df."""
    viewer = napari.Viewer(show=False)
    state = ViewerAppState()
    legend_widget = ViewerLegendWidget(viewer)

    ctx = types.SimpleNamespace()
    ctx.viewer = viewer
    ctx.state = state
    ctx.legend_widget = legend_widget
    ctx.added_legend_entries = set()
    ctx.active_workers = []

    legend_widget.ctx = ctx

    # Mock ControlWidget
    cw = MagicMock()
    cw.combo_ref.currentText.return_value = "cell_type"
    cw.spin_scale.value.return_value = 1.0
    cw.update_dashboard = MagicMock()
    ctx.control_widget = cw

    # Mock MultiSampleService and ImageService
    ms_svc = MultiSampleService(ctx)
    img_svc = ImageService(ctx)
    ctx.multi_sample_svc = ms_svc
    ctx.image_svc = img_svc

    # Sample A mask and df
    mask_a = np.zeros((100, 100), dtype=np.uint32)
    mask_a[10:20, 10:20] = 1  # B cells
    mask_a[30:40, 30:40] = 2  # T cells
    mask_a[50:60, 50:60] = 3  # Tumor

    df_a = pd.DataFrame({
        "Image": ["sample_A", "sample_A", "sample_A"],
        "ObjectNumber": [1, 2, 3],
        "Pos_X": [15.0, 35.0, 55.0],
        "Pos_Y": [15.0, 35.0, 55.0],
        "cell_type": ["B cells", "T cells", "Tumor"],
    })

    # Sample B mask and df (shares "B cells", has new "Macrophage" and "Epithelial")
    mask_b = np.zeros((100, 100), dtype=np.uint32)
    mask_b[10:20, 10:20] = 10  # B cells
    mask_b[30:40, 30:40] = 20  # Macrophage
    mask_b[50:60, 50:60] = 30  # Epithelial

    df_b = pd.DataFrame({
        "Image": ["sample_B", "sample_B", "sample_B"],
        "ObjectNumber": [10, 20, 30],
        "Pos_X": [15.0, 35.0, 55.0],
        "Pos_Y": [15.0, 35.0, 55.0],
        "cell_type": ["B cells", "Macrophage", "Epithelial"],
    })

    state.master_df = pd.concat([df_a, df_b], ignore_index=True)
    state.markers = ["DAPI", "CD45"]

    # Create dummy image files in tmp_path
    img_a_path = tmp_path / "sample_A.tiff"
    img_b_path = tmp_path / "sample_B.tiff"
    img_a_path.touch()
    img_b_path.touch()

    state.file_map = {
        "sample_A": img_a_path,
        "sample_B": img_b_path,
    }

    # Store masks for manual loading in test
    ctx._sample_masks = {
        "sample_A": mask_a,
        "sample_B": mask_b,
    }

    return ctx


def test_single_sample_switch_phenotype_legends(tmp_path, qapp):
    """When switching between single samples, all cell phenotype legend rows appear."""
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
    # on_sample_selected clears view, legend, and resets volatile state
    ctx.viewer.layers.clear()
    ctx.legend_widget.clear()
    ctx.added_legend_entries.clear()
    if hasattr(ctx, "multi_sample_svc") and hasattr(ctx.multi_sample_svc, "_added_master_legends"):
        ctx.multi_sample_svc._added_master_legends.clear()
    ctx.state.sample_layers_map.clear()
    ctx.state.reset_sample_state()
    ctx.state.current_sample_name = sample_b

    # Simulate loaded companion mask and layers for Sample B
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


def test_legend_deduplication(tmp_path, qapp):
    """Adding the same phenotype row multiple times does not produce duplicate rows in legend."""
    ctx = _create_sample_switching_context(tmp_path)
    legend_widget = ctx.legend_widget

    legend_widget.add_ref_ct_row(None, "B cells", "#FF0000")
    legend_widget.add_ref_ct_row(None, "B cells", "#FF0000")
    legend_widget.add_ref_ct_row(None, "B cells", "#00FF00")

    matching = [r for r in legend_widget.ref_ct_rows if r.name == "B cells"]
    assert len(matching) == 1
    assert matching[0].hex_color == "#00FF00"


def test_legend_clear_synchronizes_state(tmp_path, qapp):
    """Calling clear() on legend_widget empties GUI rows and clears tracking sets."""
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
