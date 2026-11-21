"""Tests for spatial cell density, proportion, diversity, and publication studios."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openpyxl
import pandas as pd
import pytest
from qtpy.QtCore import Qt

from iris_analyzer.density import (
    DensityResult,
    ROISpec,
    build_shapely_region,
    compute_cell_density,
    compute_polygon_area_and_filter,
    generate_concave_hull_polygon,
    generate_intensity_tissue_polygons,
)
from iris_analyzer.proportion import (
    ProportionResult,
    compute_cell_proportions,
    compute_sample_diversity,
)
from iris_analyzer.widgets.density_widget import (
    CellDensityStudioDialog,
    CellDensityWidget,
    SampleGroupAssignmentDialog,
)
from iris_analyzer.widgets.proportion_widget import (
    CellProportionStudioDialog,
    CellProportionWidget,
)


class _DummySignal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, fn) -> None:
        self._callbacks.append(fn)

    def emit(self, event=None) -> None:
        for cb in list(self._callbacks):
            cb(event)


class _DummyShapesLayer:
    def __init__(self, data, name: str = "Shapes", **kwargs) -> None:
        self.data = list(data)
        self.name = name
        self.kwargs = kwargs
        self.mode = "pan_zoom"
        self.events = SimpleNamespace(data=_DummySignal())


class _DummyViewer:
    def __init__(self, initial_layers=None) -> None:
        self.layers = list(initial_layers or [])

    def add_shapes(self, data, **kwargs):
        layer = _DummyShapesLayer(data, **kwargs)
        self.layers.append(layer)
        return layer


def _make_cohort_cells_df() -> tuple[pd.DataFrame, dict[str, str]]:
    """Build synthetic multi-sample cohort (4 samples/group, 60 cells/sample)."""
    rng = np.random.default_rng(123)
    rows = []
    group_map: dict[str, str] = {}

    # Responder samples: dense in smaller tissue region, high CD8_T fraction
    # NonResponder samples: sparser in larger tissue region, low CD8_T fraction
    nr_counts_base = {"Tumor": 30, "Macrophage": 12}
    configs = [
        ("Resp_1", "Responder", 100.0, {"CD8_T": 28, "CD4_T": 14, "Tumor": 12, "Macrophage": 6}),
        ("Resp_2", "Responder", 105.0, {"CD8_T": 26, "CD4_T": 16, "Tumor": 12, "Macrophage": 6}),
        ("Resp_3", "Responder", 98.0, {"CD8_T": 30, "CD4_T": 12, "Tumor": 12, "Macrophage": 6}),
        ("Resp_4", "Responder", 102.0, {"CD8_T": 27, "CD4_T": 15, "Tumor": 12, "Macrophage": 6}),
        ("NonResp_1", "NonResponder", 260.0, {"CD8_T": 6, "CD4_T": 12, **nr_counts_base}),
        ("NonResp_2", "NonResponder", 255.0, {"CD8_T": 7, "CD4_T": 11, **nr_counts_base}),
        ("NonResp_3", "NonResponder", 265.0, {"CD8_T": 5, "CD4_T": 13, **nr_counts_base}),
        ("NonResp_4", "NonResponder", 250.0, {"CD8_T": 8, "CD4_T": 10, **nr_counts_base}),
    ]

    for sample_id, grp, span, counts in configs:
        group_map[sample_id] = grp
        cell_idx = 1
        for ct, n_cells in counts.items():
            ys = rng.uniform(5.0, span - 5.0, size=n_cells)
            xs = rng.uniform(5.0, span - 5.0, size=n_cells)
            for y_val, x_val in zip(ys, xs):
                rows.append(
                    {
                        "Image": sample_id,
                        "ObjectNumber": cell_idx,
                        "Pos_Y": float(y_val),
                        "Pos_X": float(x_val),
                        "cell_type": ct,
                    }
                )
                cell_idx += 1

    return pd.DataFrame(rows), group_map


def test_polygon_hole_area_calibration_and_point_filtering():
    """Verify exact 100x100 - 40x40 hole area, calibration, union cleanup, and filtering."""
    outer = np.array([[0.0, 0.0], [0.0, 100.0], [100.0, 100.0], [100.0, 0.0]])
    hole = np.array([[30.0, 30.0], [30.0, 70.0], [70.0, 70.0], [70.0, 30.0]])

    geom, px_area = build_shapely_region([outer], [hole])
    assert not geom.is_empty
    assert px_area == pytest.approx(8400.0)

    # Overlapping inclusion squares should union cleanly without double-counting
    sq1 = np.array([[0.0, 0.0], [0.0, 20.0], [20.0, 20.0], [20.0, 0.0]])
    sq2 = np.array([[10.0, 0.0], [10.0, 20.0], [30.0, 20.0], [30.0, 0.0]])
    _, union_area = build_shapely_region([sq1, sq2])
    assert union_area == pytest.approx(600.0)

    # Self-intersecting bowtie polygon cleaned safely via buffer(0)
    bowtie = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
    bowtie_geom, bowtie_area = build_shapely_region([bowtie])
    assert bowtie_geom.is_valid
    assert bowtie_area >= 0.0

    cells = pd.DataFrame(
        {
            "Image": ["S1", "S1", "S1", "S1"],
            "ObjectNumber": [1, 2, 3, 4],
            "Pos_Y": [10.0, 50.0, 90.0, 120.0],
            "Pos_X": [10.0, 50.0, 90.0, 50.0],
            "cell_type": ["T", "Tumor", "T", "B"],
        }
    )
    filtered, area_um2, area_mm2 = compute_polygon_area_and_filter(
        cells,
        [outer],
        [hole],
        scale_factor=0.5,
        sample_name="S1",
    )
    assert area_um2 == pytest.approx(8400.0 * 0.25)
    assert area_mm2 == pytest.approx((8400.0 * 0.25) / 1_000_000.0)
    assert filtered["ObjectNumber"].tolist() == [1, 3]

    # Calibrated compute_cell_density over ROISpec in mm2, 10^4 um2, and um2
    roi = ROISpec(
        name="Tissue_With_Lumen",
        sample="S1",
        inclusion_polygons=[outer],
        exclusion_polygons=[hole],
        microns_per_pixel=0.5,
    )
    res_mm2 = compute_cell_density(cells, rois=[roi], unit="mm2")
    all_row = res_mm2.replicates_df[res_mm2.replicates_df["cell_type"] == "All Cells"].iloc[0]
    assert all_row["Count"] == 2
    assert all_row["Area_um2"] == pytest.approx(2100.0)
    assert all_row["Density"] == pytest.approx(2.0 / 0.0021)

    res_1e4 = compute_cell_density(cells, rois=[roi], unit="10^4 um2")
    all_1e4 = res_1e4.replicates_df[res_1e4.replicates_df["cell_type"] == "All Cells"].iloc[0]
    assert all_1e4["Density"] == pytest.approx(2.0 / 0.21)

    res_um2 = compute_cell_density(cells, rois=[roi], unit="um2")
    all_um2 = res_um2.replicates_df[res_um2.replicates_df["cell_type"] == "All Cells"].iloc[0]
    assert all_um2["Density"] == pytest.approx(2.0 / 2100.0)


def test_concave_hull_and_intensity_tissue_polygon_extraction():
    """Verify concave hull tightness and Otsu/Triangle/Li tissue+lumen hole extraction."""
    # U-shaped point cloud whose concave hull area is strictly below convex hull area
    rng = np.random.default_rng(7)
    left_arm = np.column_stack([rng.uniform(0, 100, 40), rng.uniform(0, 20, 40)])
    right_arm = np.column_stack([rng.uniform(0, 100, 40), rng.uniform(80, 100, 40)])
    bottom_bar = np.column_stack([rng.uniform(0, 20, 40), rng.uniform(0, 100, 40)])
    u_pts = np.vstack([left_arm, right_arm, bottom_bar])

    concave_verts = generate_concave_hull_polygon(u_pts, alpha_ratio=0.25)
    convex_verts = generate_concave_hull_polygon(u_pts, alpha_ratio=1.0)
    _, concave_area = build_shapely_region([concave_verts])
    _, convex_area = build_shapely_region([convex_verts])
    assert len(concave_verts) >= 4
    assert concave_area < convex_area * 0.85

    # Fallback on < 4 points
    tri_verts = generate_concave_hull_polygon(np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]))
    assert tri_verts.shape == (3, 2)

    # Synthetic (2, 240, 240) DAPI tissue ring (R=80) with interior lumen (R=30) + small noise
    img = np.zeros((2, 240, 240), dtype=np.float32)
    yy, xx = np.ogrid[:240, :240]
    ring = ((yy - 120) ** 2 + (xx - 120) ** 2 <= 80 ** 2) & (
        (yy - 120) ** 2 + (xx - 120) ** 2 > 30 ** 2
    )
    img[0, ring] = 220.0
    img[0, 10:15, 10:15] = 220.0  # tiny 25 px noise artifact (< min_area_px=500)

    incl_otsu, excl_otsu = generate_intensity_tissue_polygons(
        img,
        channel_idx=0,
        threshold_method="otsu",
        gaussian_sigma=1.5,
        min_area_px=500,
        subtract_holes=True,
        min_hole_area_px=200,
    )
    assert len(incl_otsu) == 1
    assert len(excl_otsu) == 1
    _, tissue_area = build_shapely_region(incl_otsu, excl_otsu)
    expected_ring_area = np.pi * (80 ** 2 - 30 ** 2)
    assert tissue_area == pytest.approx(expected_ring_area, rel=0.05)

    for method in ("triangle", "li"):
        incl_m, excl_m = generate_intensity_tissue_polygons(
            img,
            channel_idx=0,
            threshold_method=method,
            gaussian_sigma=1.5,
        )
        assert len(incl_m) >= 1 and len(excl_m) >= 1


def test_group_stats_sample_aggregation_parent_lineage_and_diversity():
    """Verify n=samples anti-pseudoreplication, parent-lineage denominator, CLR, and diversity."""
    df, group_map = _make_cohort_cells_df()

    # 1. Density with auto-derived concave hull areas and cohort stats
    dens_res = compute_cell_density(
        df,
        scale_factor=0.5,
        group_map=group_map,
        unit="mm2",
        stat_test="mann_whitney",
        p_adjust="fdr_bh",
    )
    assert isinstance(dens_res, DensityResult)
    assert dens_res.unit_label == "cells/mm²"
    assert not dens_res.stats_df.empty
    all_stat = dens_res.stats_df[dens_res.stats_df["cell_type"] == "All Cells"].iloc[0]
    # Sample N = 4 per group (NOT 240 cells!)
    assert all_stat["n1"] == 4
    assert all_stat["n2"] == 4
    assert all_stat["aggregation_level"] == "sample"
    assert all_stat["p_value"] < 0.05

    # 2. Total proportions + CLR + cohort stats
    prop_res = compute_cell_proportions(
        df,
        denominator_mode="total",
        group_map=group_map,
        stat_test="welch_t",
        p_adjust="fdr_bh",
    )
    assert isinstance(prop_res, ProportionResult)
    sample_sums = prop_res.replicates_df.groupby("Image")["proportion"].sum()
    assert np.allclose(sample_sums.to_numpy(dtype=float), 1.0)
    clr_sums = prop_res.clr_df[prop_res.cell_types].sum(axis=1).to_numpy(dtype=float)
    assert np.allclose(clr_sums, 0.0, atol=1e-10)

    cd8_stat = prop_res.stats_df[prop_res.stats_df["cell_type"] == "CD8_T"].iloc[0]
    assert cd8_stat["n1"] == 4 and cd8_stat["n2"] == 4
    assert cd8_stat["q_value"] < 0.05

    # 3. Parent-lineage denominator restricts to T-cell subset
    t_res = compute_cell_proportions(
        df,
        denominator_mode="parent_lineage",
        parent_subset=["CD4_T", "CD8_T"],
        group_map=group_map,
        stat_test="mann_whitney",
    )
    assert set(t_res.cell_types) == {"CD4_T", "CD8_T"}
    t_sums = t_res.replicates_df.groupby("Image")["proportion"].sum()
    assert np.allclose(t_sums.to_numpy(dtype=float), 1.0)
    r1_cd8 = t_res.replicates_df[
        (t_res.replicates_df["Image"] == "Resp_1") & (t_res.replicates_df["cell_type"] == "CD8_T")
    ].iloc[0]
    assert r1_cd8["denominator_count"] == 28 + 14
    assert r1_cd8["proportion"] == pytest.approx(28.0 / 42.0)

    # 4. ROI compartment proportions
    roi_comp = [
        ROISpec(
            name="Core",
            sample="Resp_1",
            inclusion_polygons=[np.array([[0.0, 0.0], [0.0, 60.0], [60.0, 60.0], [60.0, 0.0]])],
        )
    ]
    comp_res = compute_cell_proportions(df, denominator_mode="roi_compartment", rois=roi_comp)
    assert comp_res.denominator_mode == "roi_compartment"
    assert np.isclose(comp_res.replicates_df["proportion"].sum(), 1.0)

    # 5. Ecological diversity exact identities
    div_input = pd.DataFrame(
        {
            "Image": ["Uniform", "Uniform", "Uniform", "Uniform", "Mono"],
            "cell_type": ["A", "B", "C", "D", "A"],
            "count": [25, 25, 25, 25, 100],
        }
    )
    div_df = compute_sample_diversity(div_input).set_index("Image")
    assert div_df.loc["Uniform", "richness"] == 4
    assert div_df.loc["Uniform", "shannon_H"] == pytest.approx(np.log(4.0))
    assert div_df.loc["Uniform", "pielou_J"] == pytest.approx(1.0)
    assert div_df.loc["Uniform", "simpson_1_minus_D"] == pytest.approx(0.75)
    assert div_df.loc["Uniform", "inverse_simpson"] == pytest.approx(4.0)

    assert div_df.loc["Mono", "richness"] == 1
    assert div_df.loc["Mono", "shannon_H"] == pytest.approx(0.0)
    assert div_df.loc["Mono", "pielou_J"] == pytest.approx(0.0)
    assert div_df.loc["Mono", "simpson_1_minus_D"] == pytest.approx(0.0)
    assert div_df.loc["Mono", "inverse_simpson"] == pytest.approx(1.0)


def test_headless_imports_without_napari_or_qt():
    """Verify density.py and proportion.py import cleanly with napari=None and zero Qt."""
    cmd = (
        "import sys; sys.modules['napari'] = None; "
        "import iris_analyzer.density, iris_analyzer.proportion; "
        "assert 'qtpy' not in sys.modules; "
        "assert 'PyQt5' not in sys.modules; "
        "print('ANALYZER_HEADLESS_OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", cmd],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ANALYZER_HEADLESS_OK" in proc.stdout


def test_density_widget_and_studio_dialog_end_to_end(tmp_path: Path, qapp):
    """Test CellDensityWidget ROI tools, viewer shapes sync, Studio tabs, and PDF/XLSX export."""
    df, group_map = _make_cohort_cells_df()
    # Include numpy-backed Image/Labels layers to ensure viewer sync never fails on ndarray.data
    image_layer = SimpleNamespace(name="DAPI", data=np.zeros((140, 140), dtype=np.float32))
    labels_layer = SimpleNamespace(name="Nuclei", data=np.ones((140, 140), dtype=np.int32))
    viewer = _DummyViewer(initial_layers=[image_layer, labels_layer])
    ring_img = np.zeros((1, 140, 140), dtype=np.float32)
    yy, xx = np.ogrid[:140, :140]
    r2 = (yy - 70) ** 2 + (xx - 70) ** 2
    ring_img[0, (r2 <= 50 ** 2) & (r2 > 18 ** 2)] = 200.0
    ctx = SimpleNamespace(
        viewer=viewer,
        state=SimpleNamespace(raw_image=ring_img, current_sample_name="Resp_1"),
    )

    widget = CellDensityWidget()
    widget.set_data(df, celltype_col="cell_type", scale_factor=0.5, ctx=ctx)
    widget.set_group_map(group_map)

    # Viewer shapes drawing + live events.data sync
    incl_layer = widget.draw_inclusion_polygon()
    assert incl_layer.mode == "add_polygon"
    excl_layer = widget.draw_exclusion_hole()
    assert excl_layer.mode == "add_polygon"
    incl_layer.data = [np.array([[0.0, 0.0], [0.0, 100.0], [100.0, 100.0], [100.0, 0.0]])]
    incl_layer.events.data.emit()
    excl_layer.data = [np.array([[30.0, 30.0], [30.0, 70.0], [70.0, 70.0], [70.0, 30.0]])]
    excl_layer.events.data.emit()
    assert len(widget.rois) == 1
    assert len(widget.rois[0].exclusion_polygons) == 1

    # Auto concave hull, auto intensity mask, JSON import, and clear
    hull_rois = widget.run_auto_concave_hull()
    assert len(hull_rois) == 8
    int_rois = widget.run_auto_tissue_mask()
    assert len(int_rois) == 1

    json_path = tmp_path / "rois.json"
    json_path.write_text(
        json.dumps(
            {
                "rois": [
                    {
                        "name": "Resp1_ROI",
                        "sample": "Resp_1",
                        "inclusion_polygons": [[[0, 0], [0, 100], [100, 100], [100, 0]]],
                        "exclusion_polygons": [[[30, 30], [30, 70], [70, 70], [70, 30]]],
                        "microns_per_pixel": 0.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = widget.import_regions_from_path(json_path)
    assert len(loaded) == 1
    widget.clear_rois()

    res = widget.compute_density()
    assert res is not None and not res.stats_df.empty
    assert widget.btn_popout.text() == "⤢ Pop-up Window"

    dlg = widget.open_popup_window()
    assert isinstance(dlg, CellDensityStudioDialog)
    assert bool(dlg.windowFlags() & Qt.Window) is True
    qapp.processEvents()

    # Switch tabs, test focus selector, re-style via sidebar, export PDF & Excel Source Data
    for tab_idx in range(dlg.tabs.count()):
        dlg.tabs.setCurrentIndex(tab_idx)
        qapp.processEvents()
    dlg.tabs.setCurrentIndex(0)
    dlg.combo_focus_type.setCurrentText("CD8_T")
    dlg.sidebar.combo_theme.setCurrentText("Classic White")
    qapp.processEvents()
    assert dlg.active_figure().patch.get_facecolor()[:3] == pytest.approx((1.0, 1.0, 1.0))

    pdf_out = dlg.export_figure_to_path(tmp_path / "density_pub.pdf")
    xlsx_out = dlg.export_source_data_to_path(tmp_path / "density_source.xlsx")
    assert pdf_out.exists() and pdf_out.read_bytes().startswith(b"%PDF")
    assert xlsx_out.exists()
    assert dlg.copy_to_clipboard() is True

    wb = openpyxl.load_workbook(xlsx_out)
    expected_sheets = {
        "Summary_Data",
        "Sample_Replicates",
        "Statistical_Tests",
        "Figure_Parameters",
        "ROI_Areas",
    }
    assert expected_sheets.issubset(set(wb.sheetnames))

    grp_dlg = SampleGroupAssignmentDialog(["Resp_1", "NonResp_1"])
    grp_dlg._infer_from_prefix()
    assert grp_dlg.get_group_map() == {"Resp_1": "Resp", "NonResp_1": "NonResp"}
    grp_dlg.close()

    dlg.close()
    widget.close()
    qapp.processEvents()


def test_proportion_widget_and_studio_dialog_end_to_end(tmp_path: Path, qapp):
    """Test CellProportionWidget modes, Studio tabs, and vector PDF/XLSX export."""
    df, group_map = _make_cohort_cells_df()
    prop_viewer = _DummyViewer(
        initial_layers=[
            SimpleNamespace(name="DAPI", data=np.zeros((80, 80), dtype=np.float32)),
            SimpleNamespace(name="Mask", data=np.ones((80, 80), dtype=np.int32)),
        ]
    )
    prop_ctx_with_images = SimpleNamespace(viewer=prop_viewer, state=None)
    roi_ctx = SimpleNamespace(
        density_rois=[
            ROISpec(
                name="Core",
                sample="Resp_1",
                inclusion_polygons=[
                    np.array([[0.0, 0.0], [0.0, 80.0], [80.0, 80.0], [80.0, 0.0]])
                ],
            )
        ]
    )

    widget = CellProportionWidget()
    widget.set_data(df, celltype_col="cell_type", scale_factor=0.5, ctx=prop_ctx_with_images)
    widget.set_data(df, celltype_col="cell_type", scale_factor=0.5, ctx=roi_ctx)
    widget.set_group_map(group_map)

    # Test Parent Lineage Subset mode
    widget.combo_denominator.setCurrentText("Parent Lineage Subset")
    widget.set_parent_subset(["CD4_T", "CD8_T"])
    res_parent = widget.compute_proportions()
    assert res_parent is not None
    assert res_parent.denominator_mode == "parent_lineage"
    assert set(res_parent.cell_types) == {"CD4_T", "CD8_T"}

    # Test ROI Compartments mode synced from ctx.density_rois
    widget.combo_denominator.setCurrentText("ROI Compartments")
    res_roi = widget.compute_proportions()
    assert res_roi is not None and res_roi.denominator_mode == "roi_compartment"

    # Switch back to All Cells & preview modes
    widget.combo_denominator.setCurrentText("All Cells")
    res_all = widget.compute_proportions()
    assert res_all is not None and not res_all.stats_df.empty
    for mode_label in ("Stacked Composition %", "Grouped Box + Replicates", "Ecological Diversity"):
        widget.combo_preview_mode.setCurrentText(mode_label)
        qapp.processEvents()

    assert widget.btn_popout.text() == "⤢ Pop-up Window"
    dlg = widget.open_popup_window()
    assert isinstance(dlg, CellProportionStudioDialog)
    assert bool(dlg.windowFlags() & Qt.Window) is True
    qapp.processEvents()

    # Render all tabs, cycle statistics table views, and re-style
    for tab_idx in range(dlg.tabs.count()):
        dlg.tabs.setCurrentIndex(tab_idx)
        qapp.processEvents()

    for view_label in (
        "Cell-Type Proportion Statistics",
        "Sample Diversity Metrics",
        "Diversity Group Statistics",
        "CLR Matrix",
    ):
        dlg.combo_stat_view.setCurrentText(view_label)
        qapp.processEvents()

    dlg.tabs.setCurrentIndex(0)
    dlg.combo_stack_mode.setCurrentText("Per Cohort Group")
    dlg.sidebar.combo_theme.setCurrentText("Classic White")
    qapp.processEvents()

    dlg.tabs.setCurrentIndex(1)
    dlg.combo_box_focus.setCurrentText("CD8_T")
    qapp.processEvents()

    pdf_out = dlg.export_figure_to_path(tmp_path / "proportion_pub.pdf")
    xlsx_out = dlg.export_source_data_to_path(tmp_path / "proportion_source.xlsx")
    assert pdf_out.exists() and pdf_out.read_bytes().startswith(b"%PDF")
    assert xlsx_out.exists()
    assert dlg.copy_to_clipboard() is True

    wb = openpyxl.load_workbook(xlsx_out)
    assert {
        "Summary_Data",
        "Sample_Replicates",
        "Statistical_Tests",
        "Figure_Parameters",
        "Diversity_Metrics",
        "CLR_Matrix",
    }.issubset(set(wb.sheetnames))

    dlg.close()
    widget.close()
    qapp.processEvents()
