"""Unit and widget tests for iris_analyzer spatial graphs and pairwise interactions."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openpyxl
import pandas as pd
import scipy.sparse as sp
from qtpy.QtCore import Qt
from scipy.sparse.csgraph import shortest_path

from iris_analyzer.graph import (
    SpatialGraphResult,
    build_spatial_graph,
    extract_pair_edges_yx,
)
from iris_analyzer.interactions import InteractionResult, run_pairwise_interactions
from iris_analyzer.widgets.interactions_widget import (
    PairwiseInteractionStudioDialog,
    PairwiseInteractionsWidget,
)
from iris_common.widgets.publication_studio import PlotStyleConfig


def _make_grid_df(rows: int = 5, cols: int = 5, spacing: float = 10.0) -> pd.DataFrame:
    xs, ys = np.meshgrid(
        np.arange(cols, dtype=float) * spacing,
        np.arange(rows, dtype=float) * spacing,
    )
    return pd.DataFrame(
        {
            "Image": "FOV_Grid",
            "ObjectNumber": np.arange(1, rows * cols + 1, dtype=int),
            "Pos_X": xs.ravel(),
            "Pos_Y": ys.ravel(),
            "cell_type": ["TypeA" if (i % 2 == 0) else "TypeB" for i in range(rows * cols)],
        }
    )


def test_delaunay_hop_rings_on_5x5_grid():
    """Verify exact 1-hop, 2-hop, and 3-hop Delaunay rings on a 5x5 regular grid."""
    grid_df = _make_grid_df(rows=5, cols=5, spacing=1.0)

    # 1) Unconstrained Delaunay 1/2/3-hop against graph geodesic distance
    g1 = build_spatial_graph(grid_df, mode="delaunay", hop=1)
    g2 = build_spatial_graph(grid_df, mode="delaunay", hop=2, exact_ring=True)
    g3 = build_spatial_graph(grid_df, mode="delaunay", hop=3, exact_ring=True)
    g2_cum = build_spatial_graph(grid_df, mode="delaunay", hop=2, exact_ring=False)

    assert isinstance(g1, SpatialGraphResult)
    assert g1.n_cells == 25
    assert g1.adjacency.diagonal().sum() == 0
    assert (g1.adjacency - g1.adjacency.T).nnz == 0

    dist = shortest_path(g1.adjacency, directed=False, unweighted=True)
    assert np.array_equal(g2.adjacency.toarray(), dist == 2)
    assert np.array_equal(g3.adjacency.toarray(), dist == 3)
    assert np.array_equal(g2_cum.adjacency.toarray(), (dist >= 1) & (dist <= 2))

    # Strict non-overlap between distinct exact hop rings
    assert g1.adjacency.multiply(g2.adjacency).nnz == 0
    assert g1.adjacency.multiply(g3.adjacency).nnz == 0
    assert g2.adjacency.multiply(g3.adjacency).nnz == 0

    # 2) Filter diagonal chords (sqrt(2) ~= 1.414) -> exact 4-connected Manhattan lattice
    m1 = build_spatial_graph(grid_df, mode="delaunay", hop=1, max_edge_um=1.05)
    m2 = build_spatial_graph(grid_df, mode="delaunay", hop=2, exact_ring=True, max_edge_um=1.05)
    m3 = build_spatial_graph(grid_df, mode="delaunay", hop=3, exact_ring=True, max_edge_um=1.05)

    center_idx = 2 * 5 + 2
    corner_idx = 0
    assert m1.adjacency[center_idx].nnz == 4
    assert m1.adjacency[corner_idx].nnz == 2
    assert m2.adjacency[center_idx].nnz == 8
    assert m3.adjacency[center_idx].nnz == 8
    assert m1.adjacency.multiply(m2.adjacency).nnz == 0
    assert m2.adjacency.multiply(m3.adjacency).nnz == 0


def test_delaunay_small_and_degenerate_edge_cases():
    """Verify graceful handling of < 4 cells, collinear points, and duplicates."""
    empty_df = pd.DataFrame(columns=["Image", "Pos_X", "Pos_Y"])
    g0 = build_spatial_graph(empty_df, mode="delaunay")
    assert g0.n_cells == 0
    assert g0.edge_count == 0

    df1 = pd.DataFrame({"Image": ["S1"], "Pos_X": [5.0], "Pos_Y": [5.0]})
    g1 = build_spatial_graph(df1, mode="delaunay")
    assert g1.n_cells == 1
    assert g1.edge_count == 0
    assert g1.mean_degree == 0.0

    df2 = pd.DataFrame({"Image": ["S1", "S1"], "Pos_X": [0.0, 10.0], "Pos_Y": [0.0, 0.0]})
    g2 = build_spatial_graph(df2, mode="delaunay", hop=1)
    assert g2.edge_count == 1
    assert g2.mean_degree == 1.0
    g2_ring2 = build_spatial_graph(df2, mode="delaunay", hop=2, exact_ring=True)
    assert g2_ring2.edge_count == 0

    # 3 non-collinear cells -> triangle (3 edges)
    df3 = pd.DataFrame({"Pos_X": [0.0, 10.0, 0.0], "Pos_Y": [0.0, 0.0, 10.0]})
    g3 = build_spatial_graph(df3, mode="delaunay")
    assert g3.edge_count == 3

    # 4 strictly collinear cells -> Qhull fallback produces 1D chain (3 edges, zero crash)
    df4_col = pd.DataFrame(
        {"Pos_X": [0.0, 10.0, 20.0, 30.0], "Pos_Y": [5.0, 5.0, 5.0, 5.0]}
    )
    g4 = build_spatial_graph(df4_col, mode="delaunay", hop=1)
    assert g4.edge_count == 3
    assert g4.adjacency.diagonal().sum() == 0

    # Coincident duplicate cells
    df_dup = pd.DataFrame({"Pos_X": [1.0, 1.0, 1.0, 1.0], "Pos_Y": [2.0, 2.0, 2.0, 2.0]})
    g_dup = build_spatial_graph(df_dup, mode="delaunay")
    assert g_dup.adjacency.shape == (4, 4)
    assert g_dup.adjacency.diagonal().sum() == 0


def test_knn_vs_knn_radius_and_scaling():
    """Verify unconstrained knn connects distant cells while knn_radius disconnects them."""
    # Two clusters separated by 500 pixels; scale = 0.5 um/px -> 250 um apart
    df = pd.DataFrame(
        {
            "Image": ["FOV1"] * 4,
            "Pos_X": [0.0, 10.0, 500.0, 510.0],
            "Pos_Y": [0.0, 0.0, 0.0, 0.0],
        }
    )
    g_knn = build_spatial_graph(df, mode="knn", k=3, microns_per_pixel=0.5, symmetric=True)
    g_knn_rad = build_spatial_graph(
        df, mode="knn_radius", k=3, radius_um=20.0, microns_per_pixel=0.5, symmetric=True
    )
    g_rad = build_spatial_graph(df, mode="radius", radius_um=20.0, microns_per_pixel=0.5)

    # Unconstrained knn connects cluster 0-1 to distant cluster 2-3
    assert g_knn.adjacency[0, 2]
    assert g_knn.edge_count == 6

    # Constrained knn_radius and radius only connect pairs within 20 um (0-1 and 2-3 at 5 um)
    assert g_knn_rad.adjacency[0, 1]
    assert g_knn_rad.adjacency[2, 3]
    assert not g_knn_rad.adjacency[0, 2]
    assert not g_knn_rad.adjacency[1, 3]
    assert g_knn_rad.edge_count == 2
    assert np.array_equal(g_rad.adjacency.toarray(), g_knn_rad.adjacency.toarray())


def test_sample_boundary_isolation_zero_cross_sample_edges():
    """Verify zero edges ever cross sample boundaries across all graph modes."""
    df = pd.DataFrame(
        {
            "Image": ["Core_1", "Core_1", "Core_1", "Core_2", "Core_2", "Core_2"],
            "Pos_X": [0.0, 5.0, 2.5, 1.0, 6.0, 3.5],
            "Pos_Y": [0.0, 0.0, 4.0, 0.5, 0.5, 4.5],
            "cell_type": ["A", "B", "A", "B", "A", "B"],
        }
    )
    c1 = np.array([0, 1, 2])
    c2 = np.array([3, 4, 5])

    for mode in ("delaunay", "knn", "knn_radius", "radius"):
        res = build_spatial_graph(df, mode=mode, k=5, radius_um=100.0)
        sub_cross_12 = res.adjacency[c1][:, c2]
        sub_cross_21 = res.adjacency[c2][:, c1]
        assert sub_cross_12.nnz == 0, f"Cross-sample edge leaked in mode={mode}"
        assert sub_cross_21.nnz == 0, f"Cross-sample edge leaked in mode={mode}"
        assert set(res.sample_slices.keys()) == {"Core_1", "Core_2"}


def _make_colocalized_and_segregated_df(seed: int = 123) -> pd.DataFrame:
    """Create synthetic tissue with strong A<->B attraction and C segregation."""
    rng = np.random.default_rng(seed)
    rows = []

    # 30 paired A-B couples tightly clustered together in region [0, 300] x [0, 300]
    centers_ab = rng.uniform(10.0, 280.0, size=(30, 2))
    for cx, cy in centers_ab:
        rows.append((cx, cy, "TypeA"))
        rows.append((cx + 2.0, cy + 1.5, "TypeB"))

    # 50 TypeC cells isolated in a completely separate niche [500, 800] x [500, 800]
    centers_c = rng.uniform(500.0, 780.0, size=(50, 2))
    for cx, cy in centers_c:
        rows.append((cx, cy, "TypeC"))

    return pd.DataFrame(
        {
            "Image": "Tissue_1",
            "ObjectNumber": np.arange(1, len(rows) + 1, dtype=int),
            "Pos_X": [r[0] for r in rows],
            "Pos_Y": [r[1] for r in rows],
            "cell_type": [r[2] for r in rows],
        }
    )


def test_permutation_detects_attraction_and_avoidance():
    """Verify permutation test identifies Attraction on co-localized and Avoidance on segregated."""
    df = _make_colocalized_and_segregated_df()
    res = run_pairwise_interactions(
        df,
        cell_type_col="cell_type",
        mode="radius",
        radius_um=8.0,
        n_permutations=500,
        random_state=42,
    )
    assert isinstance(res, InteractionResult)
    assert res.z_matrix.shape == (3, 3)

    pw = res.pairwise_df.set_index(["source_type", "target_type"])
    ab = pw.loc[("TypeA", "TypeB")]
    ac = pw.loc[("TypeA", "TypeC")]

    assert ab["z_score"] > 3.0
    assert ab["log2_fc"] > 0.0
    assert ab["q_value"] < 0.05
    assert ab["interaction_class"] == "Attraction"

    assert ac["z_score"] < -2.0
    assert ac["log2_fc"] < 0.0
    assert ac["q_value"] < 0.05
    assert ac["interaction_class"] == "Avoidance"


def test_cohort_comparison_across_biological_groups():
    """Verify sample-level replicates_df and cohort_stats_df distinguish biological cohorts."""
    rng = np.random.default_rng(7)
    frames = []
    group_map = {}

    # Control: A and B spatially segregated (avoidance) across 4 samples
    for s_idx in range(4):
        s_name = f"Ctrl_{s_idx}"
        group_map[s_name] = "Control"
        pts_a = np.column_stack(
            [rng.uniform(0, 60, 20), rng.uniform(0, 60, 20)]
        )
        pts_b = np.column_stack(
            [rng.uniform(150, 210, 20), rng.uniform(150, 210, 20)]
        )
        frames.append(
            pd.DataFrame(
                {
                    "Image": s_name,
                    "Pos_X": np.concatenate([pts_a[:, 0], pts_b[:, 0]]),
                    "Pos_Y": np.concatenate([pts_a[:, 1], pts_b[:, 1]]),
                    "cell_type": ["A"] * 20 + ["B"] * 20,
                }
            )
        )

    # Treated: A and B tightly paired (attraction) across 4 samples
    for s_idx in range(4):
        s_name = f"Treat_{s_idx}"
        group_map[s_name] = "Treated"
        base = rng.uniform(10, 200, size=(20, 2))
        pts_a = base
        pts_b = base + rng.normal(0.0, 2.0, size=(20, 2))
        frames.append(
            pd.DataFrame(
                {
                    "Image": s_name,
                    "Pos_X": np.concatenate([pts_a[:, 0], pts_b[:, 0]]),
                    "Pos_Y": np.concatenate([pts_a[:, 1], pts_b[:, 1]]),
                    "cell_type": ["A"] * 20 + ["B"] * 20,
                }
            )
        )

    cohort_df = pd.concat(frames, ignore_index=True)
    res = run_pairwise_interactions(
        cohort_df,
        mode="radius",
        radius_um=12.0,
        n_permutations=200,
        group_map=group_map,
        stat_test="welch_t",
        random_state=42,
    )
    assert not res.replicates_df.empty
    assert not res.cohort_stats_df.empty

    ab_stats = res.cohort_stats_df[
        (res.cohort_stats_df["source_type"] == "A")
        & (res.cohort_stats_df["target_type"] == "B")
    ]
    assert len(ab_stats) == 1
    row = ab_stats.iloc[0]
    assert row["n1"] == 4 and row["n2"] == 4
    assert row["p_value"] < 0.05
    assert bool(row["significant"]) is True


def test_headless_import_without_napari_or_qt():
    """Verify graph.py and interactions.py import cleanly with napari=None and no Qt."""
    code = (
        "import sys; "
        "sys.modules['napari'] = None; "
        "import iris_analyzer.graph as g; "
        "import iris_analyzer.interactions as i; "
        "assert 'qtpy' not in sys.modules; "
        "assert 'PyQt5' not in sys.modules; "
        "assert hasattr(g, 'build_spatial_graph'); "
        "assert hasattr(i, 'run_pairwise_interactions'); "
        "print('HEADLESS_GRAPH_INTERACTIONS_OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "HEADLESS_GRAPH_INTERACTIONS_OK" in proc.stdout


def test_extract_pair_edges_and_widget_studio_exports(qapp, tmp_path: Path):
    """Verify edge extraction, dock widget, Studio tabs, and PDF/SVG/Excel export."""
    rng = np.random.default_rng(99)
    frames = []
    group_map = {}
    for s_idx in range(3):
        s_name = f"Ctrl_{s_idx}"
        group_map[s_name] = "Control"
        base = rng.uniform(10, 200, size=(15, 2))
        frames.append(
            pd.DataFrame(
                {
                    "Image": s_name,
                    "Pos_X": np.concatenate([base[:, 0], base[:, 0] + 20.0, base[:, 0] + 90.0]),
                    "Pos_Y": np.concatenate([base[:, 1], base[:, 1] + 20.0, base[:, 1] + 90.0]),
                    "cell_type": ["TypeA"] * 15 + ["TypeB"] * 15 + ["TypeC"] * 15,
                }
            )
        )
    for s_idx in range(3):
        s_name = f"Treat_{s_idx}"
        group_map[s_name] = "Treated"
        base = rng.uniform(10, 200, size=(15, 2))
        frames.append(
            pd.DataFrame(
                {
                    "Image": s_name,
                    "Pos_X": np.concatenate([base[:, 0], base[:, 0] + 2.0, base[:, 0] + 90.0]),
                    "Pos_Y": np.concatenate([base[:, 1], base[:, 1] + 2.0, base[:, 1] + 90.0]),
                    "cell_type": ["TypeA"] * 15 + ["TypeB"] * 15 + ["TypeC"] * 15,
                }
            )
        )
    full_df = pd.concat(frames, ignore_index=True)

    # Mock viewer capturing Shapes layer calls
    added_layers = []

    class _MockLayerList(list):
        def remove(self, item):
            super().remove(item)

    class _MockViewer:
        def __init__(self):
            self.layers = _MockLayerList()

        def add_shapes(self, data, **kwargs):
            lyr = SimpleNamespace(data=data, **kwargs)
            self.layers.append(lyr)
            added_layers.append(lyr)
            return lyr

    mock_ctx = SimpleNamespace(viewer=_MockViewer(), state=None)

    widget = PairwiseInteractionsWidget(ctx=mock_ctx)
    studio = None
    try:
        widget.set_data(full_df, celltype_col="cell_type", scale_factor=1.0, ctx=mock_ctx)
        widget.set_group_map(group_map)

        # Switch graph mode through UI controls and verify contextual visibility
        widget.combo_graph_mode.setCurrentText("Voronoi-Delaunay 2-Hop")
        assert not widget.chk_exact_ring.isHidden()
        assert not widget.spin_max_edge.isHidden()
        assert widget.spin_k.isHidden()

        widget.combo_graph_mode.setCurrentText("KNN + Radius (\u00b5m)")
        assert not widget.spin_k.isHidden()
        assert not widget.spin_radius.isHidden()
        assert widget.spin_max_edge.isHidden()

        widget.combo_graph_mode.setCurrentText("Fixed Radius (\u00b5m)")
        widget.spin_radius.setValue(8.0)
        widget.spin_permutations.setValue(200)

        res = widget.run_analysis()
        assert res is not None
        assert widget.last_result is res
        assert not res.cohort_stats_df.empty

        # Highlight TypeA <-> TypeB edges on canvas
        widget.combo_source_type.setCurrentText("TypeA")
        widget.combo_target_type.setCurrentText("TypeB")
        segs = widget.highlight_edges_on_canvas()
        assert segs.ndim == 3 and segs.shape[1:] == (2, 2)
        assert len(segs) > 0
        assert len(added_layers) == 1
        assert added_layers[0].shape_type == "line"

        # Verify direct extract_pair_edges_yx max_edges cap & homotypic deduplication
        src_mask = (full_df["cell_type"] == "TypeA").to_numpy()
        tgt_mask = (full_df["cell_type"] == "TypeB").to_numpy()
        capped = extract_pair_edges_yx(
            full_df, res.graph_result.adjacency, src_mask, tgt_mask, max_edges=5
        )
        assert capped.shape == (5, 2, 2)

        homotypic = extract_pair_edges_yx(
            full_df, res.graph_result.adjacency, src_mask, src_mask, max_edges=5000
        )
        if len(homotypic) > 0:
            # Verify no zero-length self-segments
            assert np.all(np.linalg.norm(homotypic[:, 0, :] - homotypic[:, 1, :], axis=1) > 0)

        # Open Studio Dialog and verify Qt.Window flag + all 5 tabs
        studio = widget.open_popup_window()
        assert isinstance(studio, PairwiseInteractionStudioDialog)
        assert bool(studio.windowFlags() & Qt.Window)
        assert studio.tabs.count() == 5

        for tab_idx in range(studio.tabs.count()):
            studio.tabs.setCurrentIndex(tab_idx)
            qapp.processEvents()

        # Test live style change via sidebar
        studio.tabs.setCurrentIndex(0)
        new_cfg = PlotStyleConfig.from_theme("Classic White", palette_name="coolwarm")
        studio.sidebar.set_config(new_cfg, emit=True)
        qapp.processEvents()
        assert studio.config.theme == "Classic White"

        # Test Vector PDF and SVG exports
        pdf_path = tmp_path / "interactions_heatmap.pdf"
        out_pdf = studio.export_figure_to_path(pdf_path)
        assert out_pdf.exists() and out_pdf.stat().st_size > 1000

        studio.tabs.setCurrentIndex(2)
        qapp.processEvents()
        svg_path = tmp_path / "interactions_network.svg"
        out_svg = studio.export_figure_to_path(svg_path)
        assert out_svg.exists() and out_svg.stat().st_size > 1000

        # Test Nature Source Data Excel export (including Statistical_Tests sheet)
        xlsx_path = tmp_path / "interactions_source_data.xlsx"
        out_xlsx = studio.export_source_data_to_path(xlsx_path)
        assert out_xlsx.exists() and out_xlsx.stat().st_size > 1000

        wb = openpyxl.load_workbook(out_xlsx)
        assert "Summary_Data" in wb.sheetnames
        assert "Sample_Replicates" in wb.sheetnames
        assert "Statistical_Tests" in wb.sheetnames
        assert "Figure_Parameters" in wb.sheetnames
        wb.close()
    finally:
        if studio is not None:
            studio.close()
        widget.close()
        qapp.processEvents()
        assert sp.issparse(res.graph_result.adjacency) if widget.last_result else True
