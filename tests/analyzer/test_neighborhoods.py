"""Tests for Cellular Neighborhoods, Spatial-LDA, and Neighborhood Studio UI."""
from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import openpyxl
import pandas as pd
import pytest
from qtpy.QtCore import Qt

from iris_analyzer.neighborhoods import (
    MovingWindowCNResult,
    SpatialLDAResult,
    compute_cn_cohort_stats,
    run_moving_window_cn,
    run_spatial_lda,
)
from iris_analyzer.widgets.neighborhoods_widget import (
    METHOD_MOVING_WINDOW,
    METHOD_SPATIAL_LDA,
    OVERLAY_DISCRETE,
    OVERLAY_ENTROPY,
    OVERLAY_TOPIC_PROB,
    CellularNeighborhoodStudioDialog,
    CellularNeighborhoodsWidget,
)


def _make_noisy_two_zone_tissue(
    *,
    seed: int = 42,
    grid_n: int = 20,
    noise_rate: float = 0.18,
    sample_name: str = "Sample_A",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build a synthetic 2D two-zone tissue with salt-and-pepper cell-type noise."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(
        np.linspace(10.0, 190.0, grid_n),
        np.linspace(10.0, 190.0, grid_n),
    )
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    n_cells = len(coords)

    cell_types = np.empty(n_cells, dtype=object)
    for i, (x_val, _y_val) in enumerate(coords):
        draw = rng.random()
        if x_val < 100.0:
            if draw < (1.0 - noise_rate):
                cell_types[i] = "Tumor"
            elif draw < (1.0 - noise_rate * 0.3):
                cell_types[i] = "Stromal"
            else:
                cell_types[i] = "CD8_T"
        else:
            if draw < (1.0 - noise_rate):
                cell_types[i] = "CD8_T"
            elif draw < (1.0 - noise_rate * 0.3):
                cell_types[i] = "Stromal"
            else:
                cell_types[i] = "Tumor"

    df = pd.DataFrame(
        {
            "Image": sample_name,
            "ObjectNumber": np.arange(1, n_cells + 1, dtype=np.int64),
            "Pos_X": coords[:, 0],
            "Pos_Y": coords[:, 1],
            "cell_type": cell_types,
        }
    )
    core_mask = (coords[:, 0] < 50.0) | (coords[:, 0] > 150.0)
    boundary_mask = (coords[:, 0] >= 82.0) & (coords[:, 0] <= 118.0)
    return df, core_mask, boundary_mask


def _make_multi_sample_cohort_df(seed: int = 123) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Build 4 samples across 2 cohort groups with distinct neighborhood ratios."""
    rng = np.random.default_rng(seed)
    frames = []
    group_map = {
        "S1": "Healthy",
        "S2": "Healthy",
        "S3": "Tumor",
        "S4": "Tumor",
    }

    for sname, grp in group_map.items():
        xs, ys = np.meshgrid(np.linspace(10.0, 150.0, 12), np.linspace(10.0, 150.0, 12))
        coords = np.column_stack([xs.ravel(), ys.ravel()])
        # Healthy samples mostly Zone A (x < 120), Tumor samples mostly Zone B (x >= 40)
        split_x = 115.0 if grp == "Healthy" else 45.0
        cts = []
        for x_val, _y_val in coords:
            u = rng.random()
            if x_val < split_x:
                cts.append("B_cell" if u < 0.85 else "Macrophage")
            else:
                cts.append("Tumor" if u < 0.85 else "CD8_T")
        frames.append(
            pd.DataFrame(
                {
                    "Image": sname,
                    "Group": grp,
                    "ObjectNumber": np.arange(1, len(coords) + 1, dtype=np.int64),
                    "Pos_X": coords[:, 0],
                    "Pos_Y": coords[:, 1],
                    "cell_type": cts,
                }
            )
        )

    return pd.concat(frames, ignore_index=True), group_map


def test_headless_import_without_napari_or_cvxpy() -> None:
    """neighborhoods.py must import cleanly without napari, qtpy, or cvxpy."""
    code = (
        "import sys; "
        "sys.modules['napari'] = None; "
        "sys.modules['cvxpy'] = None; "
        "import iris_analyzer.neighborhoods as nb; "
        "assert 'qtpy' not in sys.modules; "
        "assert 'PyQt5' not in sys.modules; "
        "assert hasattr(nb, 'run_spatial_lda') and hasattr(nb, 'run_moving_window_cn'); "
        "print('HEADLESS_OK')"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = "/Users/wwu/Documents/IRIS/src"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "HEADLESS_OK" in proc.stdout


def test_spatial_lda_simplex_smoothing_and_boundary_entropy() -> None:
    """Verify Spatial-LDA simplex, ADMM spatial smoothing, and elevated boundary entropy."""
    df, core_mask, boundary_mask = _make_noisy_two_zone_tissue(seed=11, grid_n=20, noise_rate=0.20)

    res_unsmoothed = run_spatial_lda(
        df,
        n_topics=3,
        difference_penalty=0.0,
        radius_um=28.0,
        n_iters=5,
        random_state=42,
    )
    res_mid = run_spatial_lda(
        df,
        n_topics=3,
        difference_penalty=0.25,
        radius_um=28.0,
        n_iters=5,
        random_state=42,
    )
    res_smooth = run_spatial_lda(
        df,
        n_topics=3,
        difference_penalty=2.0,
        radius_um=28.0,
        n_iters=5,
        random_state=42,
    )

    assert isinstance(res_smooth, SpatialLDAResult)
    probs = res_smooth.topic_probabilities
    assert probs.shape == (len(df), 3)
    assert np.all(probs >= 0.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-7)
    assert np.allclose(res_smooth.topic_word_matrix.sum(axis=1), 1.0, atol=1e-7)
    assert np.all((res_smooth.topic_entropy >= 0.0) & (res_smooth.topic_entropy <= 1.0))
    assert len(res_smooth.elbo_history) == 5

    expected_cols = {
        "Cellular_Neighborhood",
        "Cellular_Neighborhood_ID",
        "SpatialLDA_MaxProb",
        "SpatialLDA_Entropy",
        "Topic_0_Prob",
        "Topic_1_Prob",
        "Topic_2_Prob",
    }
    assert expected_cols.issubset(res_smooth.updated_df.columns)
    assert res_smooth.updated_df["Cellular_Neighborhood"].iloc[0].startswith("Topic ")

    # Spatial ADMM smoothing property: higher difference_penalty reduces ||D @ p||_1
    assert res_smooth.difference_l1_norm < res_unsmoothed.difference_l1_norm
    assert res_mid.difference_l1_norm < res_unsmoothed.difference_l1_norm
    assert res_smooth.difference_l1_norm <= res_mid.difference_l1_norm

    # Zone boundary transitions have strictly higher entropy than zone interiors
    core_ent = float(np.mean(res_smooth.topic_entropy[core_mask]))
    bnd_ent = float(np.mean(res_smooth.topic_entropy[boundary_mask]))
    assert bnd_ent > core_ent + 0.1


def test_spatial_lda_knn_multi_sample_and_edge_cases() -> None:
    """Verify Spatial-LDA k-NN mode, multi-sample isolation, 1-topic, and tiny samples."""
    df_multi, _ = _make_multi_sample_cohort_df(seed=9)
    res_knn = run_spatial_lda(
        df_multi,
        n_topics=3,
        featurization_mode="knn",
        k_neighbors=10,
        difference_penalty=0.5,
        n_iters=3,
    )
    assert res_knn.topic_probabilities.shape == (len(df_multi), 3)
    assert res_knn.metadata["n_samples"] == 4
    assert np.allclose(res_knn.topic_probabilities.sum(axis=1), 1.0)

    # Single-topic and tiny collinear sample fallback
    tiny_df = pd.DataFrame(
        {
            "Image": ["Tiny"] * 3,
            "Pos_X": [0.0, 10.0, 20.0],
            "Pos_Y": [0.0, 0.0, 0.0],
            "cell_type": ["A", "B", "A"],
        }
    )
    res_one = run_spatial_lda(tiny_df, n_topics=1, radius_um=15.0, n_iters=2)
    assert res_one.topic_probabilities.shape == (3, 1)
    assert np.allclose(res_one.topic_entropy, 0.0)

    empty_res = run_spatial_lda(tiny_df.iloc[0:0], n_topics=3)
    assert len(empty_res.updated_df) == 0
    assert empty_res.topic_probabilities.shape == (0, 3)


def test_moving_window_cn_enrichment_coordination_and_contexts() -> None:
    """Verify Moving Window clustering, log2 enrichment, coordination Z-scores, and Spatial Contexts."""
    df, core_mask, boundary_mask = _make_noisy_two_zone_tissue(seed=5, grid_n=20, noise_rate=0.05)

    for method in ("MiniBatchKMeans", "KMeans", "Leiden"):
        res = run_moving_window_cn(
            df,
            window_mode="knn",
            k_neighbors=10,
            n_niches=2 if method != "Leiden" else 4,
            cluster_method=method,
            resolution=0.3,
            compute_coordination=True,
            n_permutations=100,
            compute_spatial_contexts=True,
            context_k=40,
            context_threshold=0.85,
            random_state=42,
        )
        assert isinstance(res, MovingWindowCNResult)
        assert "Cellular_Neighborhood" in res.updated_df.columns
        assert "Cellular_Neighborhood_ID" in res.updated_df.columns
        assert "Spatial_Context" in res.updated_df.columns
        assert np.allclose(res.centers.sum(axis=1), 1.0, atol=1e-6)
        assert res.log2_enrichment.shape == res.centers.shape
        assert res.inter_cn_coordination is not None

    # Detailed verification on 2-cluster MiniBatchKMeans
    res2 = run_moving_window_cn(
        df,
        window_mode="knn",
        k_neighbors=10,
        n_niches=2,
        cluster_method="MiniBatchKMeans",
        compute_coordination=True,
        n_permutations=200,
        compute_spatial_contexts=True,
        context_k=40,
        context_threshold=0.85,
        random_state=42,
    )
    enrich_df = res2.log2_enrichment_df
    assert enrich_df.loc["CN 0", "Tumor"] * enrich_df.loc["CN 1", "Tumor"] < 0.0
    assert float(enrich_df["Tumor"].max()) > 0.5
    assert float(enrich_df["CD8_T"].max()) > 0.5

    # Inter-CN coordination symmetry & elevated diagonal self-coordination on separated zones
    coord = res2.inter_cn_coordination
    assert coord is not None and coord.shape == (2, 2)
    assert np.allclose(coord.values, coord.values.T, atol=1e-10)
    assert np.all(np.isfinite(coord.values))
    assert float(coord.iloc[0, 0]) > 2.0
    assert float(coord.iloc[1, 1]) > 2.0
    assert float(coord.iloc[0, 1]) < -2.0

    # Spatial Contexts: cores are pure single CN, boundary contains mixed CN context
    core_ctxs = res2.updated_df.loc[core_mask, "Spatial_Context"]
    bnd_ctxs = res2.updated_df.loc[boundary_mask, "Spatial_Context"]
    assert float((~core_ctxs.str.contains(" \\+ ")).mean()) > 0.85
    assert float((bnd_ctxs.str.contains(" \\+ ")).mean()) > 0.50


def test_moving_window_cohort_replicates_and_stats() -> None:
    """Verify replicate-level abundance sums to 1.0 and sample-level stats aggregate by sample."""
    df_multi, group_map = _make_multi_sample_cohort_df(seed=21)
    res = run_moving_window_cn(
        df_multi,
        window_mode="radius",
        radius_um=30.0,
        microns_per_pixel=0.5,
        n_niches=2,
        compute_coordination=False,
        compute_spatial_contexts=False,
    )
    assert res.inter_cn_coordination is None
    assert "Spatial_Context" not in res.updated_df.columns
    rep = res.replicates_df
    assert not rep.empty
    prop_sums = rep.groupby("Image")["proportion"].sum()
    assert np.allclose(prop_sums.values, 1.0, atol=1e-8)

    stats_df = compute_cn_cohort_stats(rep, group_map, stat_test="welch_t")
    assert not stats_df.empty
    assert (stats_df["n1"] == 2).all()
    assert (stats_df["n2"] == 2).all()
    assert (stats_df["aggregation_level"] == "sample").all()


def test_widget_async_track_worker_path(qapp) -> None:
    """Verify extract_neighborhoods uses napari thread_worker when ctx.track_worker exists."""
    df, _, _ = _make_noisy_two_zone_tissue(seed=3, grid_n=10)
    tracked_workers = []

    def _track_worker(worker, on_error=None, **_kwargs):
        tracked_workers.append(worker)
        if on_error is not None:
            worker.errored.connect(on_error)
        return worker

    ctx = SimpleNamespace(
        viewer=None,
        state=SimpleNamespace(master_df=df.copy(), current_mapped_df=df.copy(), resolved_dfs={}),
        track_worker=_track_worker,
    )
    widget = CellularNeighborhoodsWidget()
    try:
        widget.set_data(df, celltype_col="cell_type", ctx=ctx)
        widget.spin_mw_niches.setValue(2)
        widget.chk_mw_coordination.setChecked(False)
        widget.extract_neighborhoods()
        assert len(tracked_workers) == 1

        # Wait briefly for worker thread to complete and emit returned signal
        for _ in range(100):
            qapp.processEvents()
            if widget.last_result is not None:
                break
            import time

            time.sleep(0.02)

        assert widget.last_result is not None
        assert "Cellular_Neighborhood" in ctx.state.master_df.columns
    finally:
        widget.close()
        qapp.processEvents()


def test_widget_studio_overlays_and_exports(qapp, tmp_path) -> None:
    """Test CellularNeighborhoodsWidget, Napari overlays, 5-tab Studio, and exports."""
    import napari

    df_multi, group_map = _make_multi_sample_cohort_df(seed=7)
    # Build a synthetic mask for S1 (160x160) placing 4x4 cell blocks at S1 cell centroids
    mask_s1 = np.zeros((160, 160), dtype=np.int32)
    s1_sub = df_multi[df_multi["Image"] == "S1"]
    for row in s1_sub.itertuples(index=False):
        cx = int(round(row.Pos_X))
        cy = int(round(row.Pos_Y))
        mask_s1[max(0, cy - 2) : min(160, cy + 3), max(0, cx - 2) : min(160, cx + 3)] = int(
            row.ObjectNumber
        )

    viewer = napari.Viewer(show=False)
    widget = None
    studio = None
    try:
        state = SimpleNamespace(
            masks_map={"S1": mask_s1},
            current_mask=mask_s1,
            current_sample_name="S1",
            celltype_colors={"Tumor": "#E64B35", "B_cell": "#4DBBD5"},
            resolved_dfs={},
            master_df=df_multi.copy(),
            current_mapped_df=df_multi.copy(),
        )
        ctx = SimpleNamespace(viewer=viewer, state=state, track_worker=None)

        widget = CellularNeighborhoodsWidget()
        widget.set_data(df_multi, celltype_col="cell_type", scale_factor=1.0, ctx=ctx)
        widget.set_group_map(group_map)
        qapp.processEvents()

        # 1. Moving Window extraction + Discrete canvas overlay
        widget.combo_method.setCurrentText(METHOD_MOVING_WINDOW)
        widget.spin_mw_niches.setValue(3)
        widget.chk_mw_coordination.setChecked(True)
        widget.chk_mw_contexts.setChecked(True)
        mw_res = widget.extract_neighborhoods(sync=True)
        assert isinstance(mw_res, MovingWindowCNResult)
        assert widget.table_meta.rowCount() == mw_res.centers.shape[0]
        assert any(
            isinstance(getattr(lyr, "metadata", None), dict)
            and lyr.metadata.get("iris_neighborhood_overlay")
            for lyr in viewer.layers
        )

        # 2. Switch to Spatial-LDA and test all 3 Napari canvas overlay modes
        widget.combo_method.setCurrentText(METHOD_SPATIAL_LDA)
        widget.spin_lda_topics.setValue(3)
        widget.spin_lda_penalty.setValue(0.5)
        widget.spin_lda_iters.setValue(3)
        lda_res = widget.extract_neighborhoods(sync=True)
        assert isinstance(lda_res, SpatialLDAResult)

        # Rename Topic 0 and recolor cleanly
        widget.set_cn_name(0, "B-Cell Niche")
        widget.set_cn_color(0, "#00FFAA")
        assert widget.niche_metadata[0]["name"] == "B-Cell Niche"
        assert "B-Cell Niche" in set(widget.last_result.updated_df["Cellular_Neighborhood"])

        for mode_label in (OVERLAY_DISCRETE, OVERLAY_TOPIC_PROB, OVERLAY_ENTROPY):
            widget.combo_overlay_mode.setCurrentText(mode_label)
            qapp.processEvents()
            overlay_layers = [
                lyr
                for lyr in viewer.layers
                if isinstance(getattr(lyr, "metadata", None), dict)
                and lyr.metadata.get("iris_neighborhood_overlay")
            ]
            assert len(overlay_layers) >= 1
            assert overlay_layers[0].metadata["overlay_mode"] == mode_label

        # 3. Open 5-tab Studio Dialog, verify Qt.Window, theme switch, PDF & Excel export
        studio = widget.open_popup_window()
        assert isinstance(studio, CellularNeighborhoodStudioDialog)
        assert bool(studio.windowFlags() & Qt.Window) is True
        assert studio.tabs.count() == 5

        for tab_idx in range(5):
            studio.tabs.setCurrentIndex(tab_idx)
            qapp.processEvents()
            assert len(studio.current_figure().axes) >= 1

        studio.combo_heatmap_mode.setCurrentIndex(1)
        studio.chk_show_values.setChecked(False)
        studio.chk_pct_bars.setChecked(False)
        studio.tabs.setCurrentIndex(0)
        qapp.processEvents()

        studio.sidebar.combo_theme.setCurrentText("Classic White")
        qapp.processEvents()
        assert studio.current_figure().patch.get_facecolor()[:3] == pytest.approx((1.0, 1.0, 1.0))

        pdf_path = studio.export_figure_to_path(tmp_path / "cn_studio.pdf")
        svg_path = studio.export_figure_to_path(tmp_path / "cn_studio.svg")
        xlsx_path = studio.export_source_data_to_path(tmp_path / "cn_source_data.xlsx")
        assert pdf_path.exists() and pdf_path.stat().st_size > 0
        assert svg_path.exists() and svg_path.stat().st_size > 0
        assert xlsx_path.exists() and xlsx_path.stat().st_size > 0
        assert studio.copy_to_clipboard() is True

        # Verify Studio handles Moving Window result cleanly on all 5 tabs (including Tab 2)
        studio.update_result(mw_res, niche_metadata=widget.niche_metadata)
        studio.tabs.setCurrentIndex(2)
        qapp.processEvents()
        assert len(studio.current_figure().axes) >= 1

        wb = openpyxl.load_workbook(xlsx_path)
        assert set(wb.sheetnames) == {
            "Summary_Data",
            "Sample_Replicates",
            "Statistical_Tests",
            "Figure_Parameters",
        }
        wb.close()

        # 4. Also verify points-only fallback when ctx has no masks
        state.masks_map.clear()
        state.current_mask = None
        widget.combo_overlay_mode.setCurrentText(OVERLAY_ENTROPY)
        pts_layers = widget.render_canvas_overlay()
        assert len(pts_layers) == 1
        assert pts_layers[0].data.shape == (len(df_multi), 2)
    finally:
        if studio is not None:
            studio.close()
        if widget is not None:
            widget.close()
        viewer.close()
        qapp.processEvents()
