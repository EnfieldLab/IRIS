"""End-to-end Phase 3 & Phase 4 integration tests across segmentation disagreement,
ROI preview, multi-compartment features, 5-method phenotyping concordance,
IrisAnalyzerPanelWidget (Density -> Proportions -> Interactions -> Neighborhoods),
PublicationPlotStudioDialog PDF/XLSX export, controller wiring, and headless imports.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import matplotlib.axes
import numpy as np
import openpyxl
import pandas as pd
import pytest
from matplotlib.figure import Figure
from qtpy.QtCore import QEventLoop, Qt, QTimer
from qtpy.QtWidgets import QComboBox, QFileDialog, QMessageBox, QPushButton

from iris_analyzer import (
    IrisAnalyzerPanelWidget,
    MovingWindowCNResult,
    ROISpec,
    SpatialLDAResult,
    build_spatial_graph,
    compute_cell_density,
    compute_cell_proportions,
    extract_pair_edges_yx,
    run_moving_window_cn,
    run_pairwise_interactions,
    run_spatial_lda,
)
from iris_common.widgets.publication_studio import (
    PlotStyleConfig,
    PublicationPlotStudioDialog,
)
from iris_viewer.app.controller import ViewerController
from iris_viewer.app.services.review_service import ReviewService
from iris_viewer.core.context import AppContext
from iris_viewer.core.state import ViewerAppState
from iris_viewer.engine.disagreement import compute_model_disagreement
from iris_viewer.engine.feature_extraction import extract_cell_features
from iris_viewer.engine.pipeline import (
    SegmentationSequence,
    SegmentationStep,
    crop_roi_and_segment_sync,
)
from iris_viewer.io.exporters import export_disagreement_map, export_segmentation_mask
from iris_viewer.widgets.image_control import ImageControlWidget
from iris_viewer.widgets.legend import ViewerLegendWidget
from iris_viewer.widgets.phenotyping.panel import PhenotyperPanelWidget
from iris_viewer.widgets.segmentation_panel import SegmentationPanelWidget
from iris_viewer.widgets.segmentation_studio_dialog import SegmentationStudioDialog


class _DummySignal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, fn) -> None:
        self._callbacks.append(fn)

    def emit(self, event=None) -> None:
        for cb in list(self._callbacks):
            cb(event)


class _DummyLayer:
    def __init__(self, data, name: str, layer_type: str = "image", **kwargs) -> None:
        self.data = data
        self.name = name
        self.layer_type = layer_type
        self.visible = bool(kwargs.get("visible", True))
        self.opacity = float(kwargs.get("opacity", 1.0))
        self.scale = kwargs.get("scale", (1.0, 1.0))
        self.translate = kwargs.get("translate", (0.0, 0.0))
        self.metadata = dict(kwargs.get("metadata", {}))
        self.mode = "pan_zoom"
        self.events = SimpleNamespace(
            data=_DummySignal(),
            visible=_DummySignal(),
            opacity=_DummySignal(),
            contrast_limits=_DummySignal(),
        )


class _FakeLayerList(list):
    def __init__(self) -> None:
        super().__init__()
        self.selection = set()

    def __contains__(self, item) -> bool:
        if isinstance(item, str):
            return any(lyr.name == item for lyr in self)
        return super().__contains__(item)

    def __getitem__(self, item):
        if isinstance(item, str):
            for lyr in self:
                if lyr.name == item:
                    return lyr
            raise KeyError(item)
        return super().__getitem__(item)

    def remove(self, item) -> None:
        if isinstance(item, str):
            item = self[item]
        super().remove(item)


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _FakeLayerList()
        self.camera = SimpleNamespace(
            center=(48.0, 48.0), zoom=1.0, mouse_pan=True, mouse_zoom=True
        )
        self.mouse_drag_callbacks = []
        self.mouse_double_click_callbacks = []

    def add_image(self, data, name: str = "Image", **kwargs) -> _DummyLayer:
        lyr = _DummyLayer(data, name=name, layer_type="image", **kwargs)
        self.layers.append(lyr)
        return lyr

    def add_labels(self, data, name: str = "Labels", **kwargs) -> _DummyLayer:
        lyr = _DummyLayer(data, name=name, layer_type="labels", **kwargs)
        self.layers.append(lyr)
        return lyr

    def add_shapes(self, data, name: str = "Shapes", **kwargs) -> _DummyLayer:
        lyr = _DummyLayer(list(data), name=name, layer_type="shapes", **kwargs)
        self.layers.append(lyr)
        return lyr

    def add_points(self, data, name: str = "Points", **kwargs) -> _DummyLayer:
        lyr = _DummyLayer(data, name=name, layer_type="points", **kwargs)
        self.layers.append(lyr)
        return lyr


def _build_segmentation_test_scene():
    """Create (4, 96, 96) synthetic image and Cellpose vs StarDist masks covering all QC classes."""
    rng = np.random.default_rng(101)
    img = rng.uniform(1.0, 3.0, size=(4, 96, 96)).astype(np.float32)

    nuc_cp = np.zeros((96, 96), dtype=np.uint32)
    cyto_cp = np.zeros((96, 96), dtype=np.uint32)
    nuc_sd = np.zeros((96, 96), dtype=np.uint32)

    # 1. Agreed cell
    nuc_cp[6:16, 6:16] = 1
    cyto_cp[4:18, 4:18] = 1
    nuc_sd[6:16, 6:16] = 1

    # 2. Split cell (1 in Cellpose -> 2 in StarDist)
    nuc_cp[6:20, 26:46] = 2
    cyto_cp[4:22, 24:48] = 2
    nuc_sd[6:20, 26:36] = 2
    nuc_sd[6:20, 36:46] = 3

    # 3. Merge cell (2 in Cellpose -> 1 in StarDist)
    nuc_cp[30:44, 6:16] = 3
    nuc_cp[30:44, 16:26] = 4
    cyto_cp[28:46, 5:16] = 3
    cyto_cp[28:46, 16:27] = 4
    nuc_sd[30:44, 6:26] = 4

    # 4. Boundary Shift cell (10x10 shifted by 3 px -> IoU = 70/130 ~= 0.538)
    nuc_cp[56:66, 10:20] = 5
    cyto_cp[54:68, 8:22] = 5
    nuc_sd[56:66, 13:23] = 5

    # 5. Model-Exclusive in Cellpose + Orphan exclusive in StarDist
    nuc_cp[74:84, 10:20] = 6
    cyto_cp[72:86, 8:22] = 6
    nuc_sd[74:84, 70:80] = 6

    # Populate marker fluorescence signals (DAPI, PanCK, CD3, CD8)
    img[0, nuc_cp > 0] = 25.0
    img[1, (cyto_cp == 1) | (cyto_cp == 2)] = 30.0
    img[2, (cyto_cp >= 3) & (cyto_cp <= 5)] = 28.0
    img[3, cyto_cp == 6] = 32.0

    markers = ["DAPI", "PanCK", "CD3", "CD8"]
    seg_results = {
        "Cellpose": {"nuclear": nuc_cp, "cytoplasm": cyto_cp},
        "StarDist": {"nuclear": nuc_sd, "cytoplasm": None},
    }
    return img, markers, seg_results


def _build_cohort_spatial_df() -> tuple[pd.DataFrame, dict[str, str], list[ROISpec]]:
    """Build 6-sample cohort spatial DataFrame with density, proportion, and niche zones."""
    rng = np.random.default_rng(77)
    rows = []
    group_map = {}

    # Responder samples: compact tissue (dense CD8_T + Tumor co-localized left, Stromal right)
    # NonResponder samples: sparse tissue (low CD8_T, segregated Tumor vs CD8_T)
    specs = [
        ("Resp_1", "Responder", True),
        ("Resp_2", "Responder", True),
        ("Resp_3", "Responder", True),
        ("NonResp_1", "NonResponder", False),
        ("NonResp_2", "NonResponder", False),
        ("NonResp_3", "NonResponder", False),
    ]

    for sample_id, grp, is_resp in specs:
        group_map[sample_id] = grp
        obj_id = 1
        if is_resp:
            # 60 cells in a 100x100 region avoiding central [30:70, 30:70] hole
            # Co-localized Tumor + CD8_T in top/left band, CD4_T + Macrophage in bottom band
            for _ in range(24):
                y_val = float(rng.uniform(4.0, 26.0))
                x_val = float(rng.uniform(4.0, 96.0))
                ct = "CD8_T" if (obj_id % 2 == 0) else "Tumor"
                rows.append((sample_id, grp, obj_id, x_val, y_val, ct))
                obj_id += 1
            for _ in range(18):
                y_val = float(rng.uniform(74.0, 96.0))
                x_val = float(rng.uniform(4.0, 96.0))
                ct = "CD4_T" if (obj_id % 2 == 0) else "Macrophage"
                rows.append((sample_id, grp, obj_id, x_val, y_val, ct))
                obj_id += 1
            for _ in range(18):
                y_val = float(rng.uniform(30.0, 70.0))
                x_val = float(rng.uniform(4.0, 26.0))
                ct = "CD8_T"
                rows.append((sample_id, grp, obj_id, x_val, y_val, ct))
                obj_id += 1
        else:
            # 45 cells spread across 220x220 region with segregated domains
            for _ in range(25):
                y_val = float(rng.uniform(5.0, 90.0))
                x_val = float(rng.uniform(5.0, 90.0))
                ct = "Tumor"
                rows.append((sample_id, grp, obj_id, x_val, y_val, ct))
                obj_id += 1
            for _ in range(6):
                y_val = float(rng.uniform(140.0, 210.0))
                x_val = float(rng.uniform(140.0, 210.0))
                ct = "CD8_T"
                rows.append((sample_id, grp, obj_id, x_val, y_val, ct))
                obj_id += 1
            for _ in range(14):
                y_val = float(rng.uniform(140.0, 210.0))
                x_val = float(rng.uniform(10.0, 90.0))
                ct = "CD4_T" if (obj_id % 2 == 0) else "Macrophage"
                rows.append((sample_id, grp, obj_id, x_val, y_val, ct))
                obj_id += 1

    df = pd.DataFrame(
        rows,
        columns=["Image", "Group", "ObjectNumber", "Pos_X", "Pos_Y", "cell_type"],
    )

    outer = np.array([[0.0, 0.0], [0.0, 100.0], [100.0, 100.0], [100.0, 0.0]])
    hole = np.array([[30.0, 30.0], [30.0, 70.0], [70.0, 70.0], [70.0, 30.0]])
    outer_large = np.array([[0.0, 0.0], [0.0, 220.0], [220.0, 220.0], [220.0, 0.0]])

    rois = [
        ROISpec(
            name=f"{s}_ROI",
            sample=s,
            inclusion_polygons=[outer if is_resp else outer_large],
            exclusion_polygons=[hole] if is_resp else [],
            microns_per_pixel=0.5,
        )
        for s, _g, is_resp in specs
    ]
    return df, group_map, rois


def test_headless_imports_without_napari_or_qt():
    """Verify headless packages import cleanly with napari=None and zero eager Qt import."""
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    code = (
        "import sys\n"
        "sys.modules['napari'] = None\n"
        "sys.modules['napari.qt'] = None\n"
        "import iris_analyzer, iris_common, iris_phenotyper, iris_viewer.engine\n"
        "assert 'qtpy' not in sys.modules, 'iris_analyzer eagerly imported qtpy!'\n"
        "for sym in (\n"
        "    'ROISpec',\n"
        "    'compute_cell_density',\n"
        "    'compute_cell_proportions',\n"
        "    'build_spatial_graph',\n"
        "    'run_pairwise_interactions',\n"
        "    'run_spatial_lda',\n"
        "    'run_moving_window_cn',\n"
        "    'compute_cn_cohort_stats',\n"
        "):\n"
        "    assert hasattr(iris_analyzer, sym), sym\n"
        "print('HEADLESS_OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "HEADLESS_OK" in proc.stdout


def test_phase3_disagreement_roi_extraction_studio_and_controller(qapp, tmp_path):
    """Exercise multi-model disagreement QC, ROI preview, multi-compartment features, and wiring."""
    img, markers, seg_results = _build_segmentation_test_scene()
    viewer = _FakeViewer()
    state = ViewerAppState()
    state.raw_image = img
    state.markers = markers
    state.current_sample_name = "Core_A1"
    state.segmentation_results = seg_results

    cw = ImageControlWidget(viewer=None)
    assert cw.btn_export_confidence.text() == "Export Disagreement"
    assert "disagreement" in cw.btn_export_confidence.toolTip().lower()
    for btn in cw.findChildren(QPushButton):
        assert "..." not in btn.text() and "…" not in btn.text()

    seg_panel = SegmentationPanelWidget()
    dp = seg_panel.disagreement_panel
    fp = seg_panel.feature_panel
    legend = ViewerLegendWidget(viewer)

    ctx = AppContext(
        viewer=viewer,
        state=state,
        control_widget=cw,
        segmentation_panel=seg_panel,
        disagreement_panel=dp,
        feature_panel=fp,
        legend_widget=legend,
        added_legend_entries=set(),
        flagged_cells=[],
        current_issue_idx=-1,
    )
    review_svc = ReviewService(ctx)
    ctx.review_svc = review_svc
    seg_panel.ctx = ctx

    # 1. Multi-model disagreement QC + ReviewService overlays
    report = compute_model_disagreement(
        seg_results,
        compare_mode="Nuclear Masks",
        reference_method="Cellpose",
        iou_threshold=0.65,
    )
    assert report.category_counts["Agreed"] >= 1
    assert report.category_counts["Split"] >= 1
    assert report.category_counts["Merge"] >= 1
    assert report.category_counts["Boundary Shift"] >= 1
    assert report.category_counts["Model-Exclusive"] >= 1
    assert report.disagreement_heatmap.shape == (96, 96)

    review_svc.render_disagreement_overlays(report)
    assert "Disagreement Heatmap" in viewer.layers
    assert "Disagreement Contours" in viewer.layers
    assert len(legend.disagreement_rows) == 2

    review_svc.next_issue()
    assert "Review BBox" in viewer.layers
    assert "Active Disagreement Isolation" in viewer.layers

    shown = review_svc.flicker_models("Cellpose", "StarDist")
    assert shown in ("Cellpose", "StarDist")

    # 2. SegmentationStudioDialog + PublicationPlotStudioDialog export
    seg_panel.sync_available_results(["Cellpose", "StarDist"], active_method="Cellpose")
    studio = SegmentationStudioDialog(
        ctx=ctx,
        disagreement_panel=dp,
        feature_panel=fp,
    )
    assert bool(studio.windowFlags() & Qt.Window)
    studio.open_publication_studio()
    assert studio.publication_dialog is not None
    studio_pdf = tmp_path / "disagreement_studio.pdf"
    studio_xlsx = tmp_path / "disagreement_studio.xlsx"
    studio.publication_dialog.export_figure_to_path(studio_pdf)
    studio.publication_dialog.export_source_data_to_path(studio_xlsx)
    assert studio_pdf.is_file() and studio_pdf.stat().st_size > 100
    assert studio_xlsx.is_file()

    # 3. ROI preview sync cropping + clamping
    mock_backend = MagicMock()
    mock_backend.segment.side_effect = lambda image, **_kw: {
        "nuclear": np.ones(image.shape[-2:], dtype=np.uint32),
        "cytoplasm": np.full(image.shape[-2:], 2, dtype=np.uint32),
    }
    roi_res = crop_roi_and_segment_sync(
        sequence=SegmentationSequence(
            steps=[SegmentationStep("Cellpose", {"nuclear_channel": 0}, rule="Independent")]
        ),
        image=img,
        roi_bbox=(-10, 10, 40, 60),
        backends={"Cellpose": mock_backend},
        device="cpu",
    )
    assert roi_res["nuclear"].shape == (40, 50)
    assert roi_res["cytoplasm"].shape == (40, 50)

    # 4. Active mask selection & multi-compartment feature extraction
    nuc_mask, cyto_mask = state.set_active_segmentation_method("Cellpose")
    assert nuc_mask is seg_results["Cellpose"]["nuclear"]
    assert cyto_mask is seg_results["Cellpose"]["cytoplasm"]
    nuc_got, cyto_got = state.get_active_masks()
    assert nuc_got is nuc_mask and cyto_got is cyto_mask

    feat_df = extract_cell_features(
        mask=nuc_got,
        raw_image=img,
        markers=markers,
        scale_factor=0.5,
        cytoplasm_mask=cyto_got,
        compartments=["whole_cell", "nuclear", "cytoplasm", "membrane", "nc_ratio"],
    )
    assert len(feat_df) == 6
    assert "cyto_area" in feat_df.columns
    assert "nc_area_ratio" in feat_df.columns
    assert "PanCK_nuc_mean" in feat_df.columns
    assert "PanCK_cyto_mean" in feat_df.columns
    assert "PanCK_mem_mean" in feat_df.columns
    assert "PanCK_nc_ratio" in feat_df.columns

    # 5. Export active mask & disagreement map + controller helpers
    mask_out = tmp_path / "active_mask.tiff"
    dis_out = tmp_path / "disagreement_map.tiff"
    export_segmentation_mask(mask_out, nuc_got, cyto_got)
    export_disagreement_map(dis_out, report.disagreement_heatmap)
    assert mask_out.is_file()
    assert dis_out.is_file()

    # Controller helper verification (ROI creation, downstream sync, session v2)
    analyzer = IrisAnalyzerPanelWidget(ctx=ctx)
    marker_w = MagicMock()
    pheno_w = MagicMock()
    dummy_ctrl = SimpleNamespace(
        viewer=viewer,
        state=state,
        control_widget=cw,
        comparison_widget=cw.comparison_widget,
        segmentation_panel=seg_panel,
        disagreement_panel=dp,
        feature_panel=fp,
        phenotyper_panel=pheno_w,
        marker_widget=marker_w,
        analyzer_panel=analyzer,
        review_svc=review_svc,
        _ctx=ctx,
        _preview_roi_bound=set(),
    )
    dummy_ctrl._resolve_active_phenotype_col = (
        lambda col=None: ViewerController._resolve_active_phenotype_col(dummy_ctrl, col)
    )
    dummy_ctrl._get_active_image_for_roi = (
        lambda: ViewerController._get_active_image_for_roi(dummy_ctrl)
    )
    dummy_ctrl._compute_default_preview_bbox = (
        lambda hw: ViewerController._compute_default_preview_bbox(dummy_ctrl, hw)
    )
    dummy_ctrl._bbox_from_preview_layer = (
        lambda hw: ViewerController._bbox_from_preview_layer(dummy_ctrl, hw)
    )
    dummy_ctrl._sync_segmentation_sources = (
        lambda: ViewerController._sync_segmentation_sources(dummy_ctrl)
    )

    bbox = ViewerController._create_or_focus_preview_roi(dummy_ctrl)
    assert "Segmentation Preview ROI" in viewer.layers
    assert state.preview_roi_bbox == bbox
    assert bbox[2] > bbox[0] and bbox[3] > bbox[1]

    ViewerController._on_all_segmentations_complete(dummy_ctrl)
    assert "Ready for disagreement check: 2 pipeline(s)" in seg_panel.lbl_disagreement_status.text()

    feat_df["cell_type"] = ["Tumor", "Tumor", "CD8_T", "CD8_T", "Macrophage", "Macrophage"]
    cw.spin_scale.setValue(0.5)
    ViewerController._sync_downstream_panels(dummy_ctrl, df=feat_df, col="cell_type")
    pheno_w.set_data.assert_called()
    marker_w.set_data.assert_called()
    assert analyzer.density_widget.df is not None
    assert analyzer.density_widget.scale_factor == 0.5
    assert analyzer.interactions_widget.celltype_col == "cell_type"

    sess_path = tmp_path / "session_v2.json"
    dp.slider_threshold.setValue(72)
    with (
        patch.object(QFileDialog, "getSaveFileName", return_value=(str(sess_path), "")),
        patch.object(QMessageBox, "information"),
    ):
        ViewerController._save_session(dummy_ctrl)
    assert sess_path.is_file()

    dp.slider_threshold.setValue(50)
    with (
        patch.object(QFileDialog, "getOpenFileName", return_value=(str(sess_path), "")),
        patch.object(QMessageBox, "information"),
    ):
        ViewerController._load_session(dummy_ctrl, sess_path)
    assert dp.get_threshold() == pytest.approx(0.72)

    studio.close()
    analyzer.close()
    seg_panel.close()
    cw.close()
    legend.close()
    qapp.processEvents()


def test_phase3_five_method_phenotyping_and_diagnostics_concordance(qapp, tmp_path, monkeypatch):
    """Run all 5 phenotyping methods (ASTIR, MAPS, STELLAR, CellSighter, Nimbus) + Concordance."""
    import torch

    from iris_phenotyper.nimbus.pipeline import _get_unet_classes

    rng = np.random.default_rng(202)
    n_per = 6
    n_total = n_per * 2
    ref_a = ["T cells", "T cells", "B cells", "B cells", "Stromal", "Stromal"]
    ref_b = ["Unassigned"] * n_per
    true_all = ref_a * 2

    cd3 = np.tile([8.5, 8.2, 0.2, 0.3, 0.2, 0.2], 2) + rng.normal(0.0, 0.1, n_total)
    cd8 = np.tile([8.0, 7.8, 0.2, 0.2, 0.2, 0.2], 2) + rng.normal(0.0, 0.1, n_total)
    cd20 = np.tile([0.2, 0.2, 8.8, 8.4, 0.2, 0.2], 2) + rng.normal(0.0, 0.1, n_total)
    vim = np.tile([0.2, 0.2, 0.2, 0.2, 9.0, 8.9], 2) + rng.normal(0.0, 0.1, n_total)

    df = pd.DataFrame(
        {
            "ObjectNumber": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "Image": ["Sample_A"] * n_per + ["Sample_B"] * n_per,
            "Pos_X": np.tile([10.0, 22.0, 34.0, 10.0, 22.0, 34.0], 2),
            "Pos_Y": np.tile([10.0, 10.0, 10.0, 32.0, 32.0, 32.0], 2),
            "CD3_mean": np.clip(cd3, 0.05, None),
            "CD8_mean": np.clip(cd8, 0.05, None),
            "CD20_mean": np.clip(cd20, 0.05, None),
            "Vimentin_mean": np.clip(vim, 0.05, None),
            "Ref_CellType": ref_a + ref_b,
            "True_CellType": true_all,
        }
    )
    markers = ["CD3", "CD8", "CD20", "Vimentin"]

    img = rng.uniform(0.0, 1.0, size=(4, 48, 48)).astype(np.float32)
    mask = np.zeros((48, 48), dtype=np.int32)
    for idx in range(6):
        r0 = 6 + (idx // 3) * 20
        c0 = 6 + (idx % 3) * 12
        mask[r0 : r0 + 10, c0 : c0 + 10] = idx + 1
        if idx < 2:
            img[0, r0 : r0 + 10, c0 : c0 + 10] = 10.0
            img[1, r0 : r0 + 10, c0 : c0 + 10] = 9.0
        elif idx < 4:
            img[2, r0 : r0 + 10, c0 : c0 + 10] = 10.0
        else:
            img[3, r0 : r0 + 10, c0 : c0 + 10] = 10.0

    state = ViewerAppState()
    state.features_df = df
    state.master_df = df.copy()
    state.current_mapped_df = df.copy()
    state.markers = markers
    state.raw_image = None
    state.current_mask = None
    state.multi_sample_images = {"Sample_A": img, "Sample_B": img}
    state.masks_map = {"Sample_A": mask, "Sample_B": mask}
    state.current_sample_name = "Sample_A"

    combo_ref = QComboBox()
    combo_ref.addItems(["Ref_CellType", "True_CellType"])
    cw = SimpleNamespace(combo_ref=combo_ref)
    ctx = AppContext(state=state, control_widget=cw)

    panel = PhenotyperPanelWidget(ctx=ctx)
    panel.set_data(df=df, markers=markers)

    # 1. ASTIR
    yaml_path = tmp_path / "astir_markers.yml"
    yaml_path.write_text(
        "cell_types:\n"
        "  T cells:\n    - CD3\n    - CD8\n"
        "  B cells:\n    - CD20\n"
        "  Stromal:\n    - Vimentin\n",
        encoding="utf-8",
    )
    panel.load_astir_yaml(str(yaml_path))
    panel.spin_astir_epochs.setValue(8)
    panel._run_astir()
    assert "ASTIR_CellType" in df.columns

    # 2. MAPS
    panel.combo_maps_mode.setCurrentText("Train from Reference Column")
    panel.combo_maps_ref_col.setCurrentText("True_CellType")
    panel.spin_maps_hidden.setValue(32)
    panel.spin_maps_epochs.setValue(5)
    panel._run_maps()
    assert "MAPS_CellType" in df.columns

    # 3. STELLAR
    panel.combo_stellar_ref_col.setCurrentText("Ref_CellType")
    panel.spin_stellar_dist.setValue(40.0)
    panel.spin_stellar_epochs.setValue(3)
    panel.spin_stellar_seed_classes.setValue(2)
    panel._run_stellar()
    assert "STELLAR_CellType" in df.columns
    assert "STELLAR_CellType_IsNovel" in df.columns

    # 4. CellSighter (multi-FOV across Sample_A & Sample_B via ctx.state)
    panel.combo_cs_mode.setCurrentText("Train on Reference Column")
    panel.combo_cs_ref_col.setCurrentText("Ref_CellType")
    panel.spin_cs_crop_size.setValue(14)
    panel.spin_cs_crop_input_size.setValue(10)
    panel.spin_cs_epochs.setValue(1)
    panel._run_cellsighter()
    assert "CellSighter_CellType" in state.master_df.columns
    assert "CellSighter_CellType" in df.columns

    # 5. Nimbus (lightweight U-Net mock for fast CPU verification)
    tiny_unet = _get_unet_classes()["NimbusUNet"](filters_root=8, padding="reflect").eval()
    monkeypatch.setattr(
        "iris_phenotyper.nimbus.pipeline.load_nimbus_model",
        lambda **_kwargs: (tiny_unet, torch.device("cpu"), None),
    )
    panel.chk_nimbus_tta.setChecked(False)
    panel.combo_nimbus_mode.setCurrentText("Panel Marker Rules")
    panel._run_nimbus()
    assert "Nimbus_CellType" in state.master_df.columns
    assert "Nimbus_CellType" in df.columns
    for m in markers:
        assert f"{m}_nimbus_positivity" in df.columns

    # Concordance & Diagnostics Studio
    diag = panel.open_diagnostics_dialog()
    diag.refresh_data(df)
    conc = diag.compute_concordance("ASTIR_CellType", "MAPS_CellType")
    assert conc["n_cells"] == n_total
    assert 0.0 <= conc["accuracy"] <= 1.0
    assert "cohens_kappa" in conc

    diag.tabs.setCurrentIndex(2)
    diag.combo_conc_a.setCurrentText("ASTIR_CellType")
    diag.combo_conc_b.setCurrentText("MAPS_CellType")
    diag._render_concordance_tab()
    studio = diag.open_publication_studio()
    assert studio is not None
    pdf_out = tmp_path / "pheno_concordance.pdf"
    xlsx_out = tmp_path / "pheno_concordance.xlsx"
    studio.export_figure_to_path(pdf_out)
    studio.export_source_data_to_path(xlsx_out)
    assert pdf_out.is_file() and pdf_out.stat().st_size > 100
    assert xlsx_out.is_file()

    studio.close()
    diag.close()
    panel.close()
    qapp.processEvents()


def test_phase4_full_analyzer_panel_end_to_end_and_exports(qapp, tmp_path):
    """Exercise IrisAnalyzerPanelWidget across Density, Proportions, Interactions, Neighborhoods."""
    df, group_map, rois = _build_cohort_spatial_df()
    viewer = _FakeViewer()
    state = ViewerAppState()
    state.master_df = df.copy()
    state.current_mapped_df = df.copy()
    state.current_sample_name = "Resp_1"
    ctx = AppContext(viewer=viewer, state=state)

    panel = IrisAnalyzerPanelWidget(ctx=ctx)
    panel.set_data(
        df=df,
        markers=["CD3", "CD8", "CD20", "PanCK"],
        celltype_col="cell_type",
        scale_factor=0.5,
        ctx=ctx,
    )
    panel.set_group_map(group_map)
    assert panel.stack.count() == 4
    assert [b.text() for b in panel.nav_buttons] == [
        "Density",
        "Proportions",
        "Interactions",
        "Neighborhoods",
    ]
    assert f"{len(df)} cells" in panel.lbl_summary.text()

    # 1. Density: ROI with hole + concave hull + Studio PDF/XLSX export
    panel.switch_tab(0)
    dw = panel.density_widget
    dw.set_rois(rois)
    res_den = dw.compute_density()
    assert res_den is not None
    resp_area = res_den.roi_areas_df.loc[
        res_den.roi_areas_df["Sample"] == "Resp_1", "Area_um2"
    ].iloc[0]
    # 100x100 - 40x40 = 8400 px^2 * (0.5 um/px)^2 = 2100 um^2
    assert resp_area == pytest.approx(2100.0)
    assert not res_den.stats_df.empty

    hull_rois = dw.run_auto_concave_hull()
    assert len(hull_rois) == 6
    dw.set_rois(rois)
    dw.compute_density()

    den_studio = dw.open_popup_window()
    assert den_studio is not None and bool(den_studio.windowFlags() & Qt.Window)
    den_pdf = tmp_path / "density.pdf"
    den_xlsx = tmp_path / "density.xlsx"
    den_studio.export_figure_to_path(den_pdf)
    den_studio.export_source_data_to_path(den_xlsx)
    assert den_pdf.is_file() and den_pdf.stat().st_size > 100
    assert den_xlsx.is_file()
    den_studio.close()

    # 2. Proportions: total + parent_lineage + diversity + Studio export
    panel.switch_tab(1)
    pw = panel.proportion_widget
    pw.combo_denominator.setCurrentText("All Cells")
    res_prop_total = pw.compute_proportions()
    assert res_prop_total is not None
    assert not res_prop_total.diversity_df.empty
    assert {"shannon_H", "simpson_1_minus_D", "pielou_J"}.issubset(
        res_prop_total.diversity_df.columns
    )

    pw.combo_denominator.setCurrentText("Parent Lineage Subset")
    pw.set_parent_subset(["CD8_T", "CD4_T"])
    res_prop_parent = pw.compute_proportions()
    assert res_prop_parent is not None
    assert res_prop_parent.denominator_mode == "parent_lineage"
    assert set(res_prop_parent.cell_types) == {"CD4_T", "CD8_T"}

    prop_studio = pw.open_popup_window()
    assert prop_studio is not None
    prop_pdf = tmp_path / "proportions.pdf"
    prop_xlsx = tmp_path / "proportions.xlsx"
    prop_studio.export_figure_to_path(prop_pdf)
    prop_studio.export_source_data_to_path(prop_xlsx)
    assert prop_pdf.is_file() and prop_xlsx.is_file()
    prop_studio.close()

    # 3. Pairwise Interactions: Delaunay 1/2/3-hop, KNN, KNN+radius + extract_pair_edges_yx
    panel.switch_tab(2)
    iw = panel.interactions_widget
    iw.spin_permutations.setValue(60)
    for mode_label in (
        "Voronoi-Delaunay 1-Hop",
        "Voronoi-Delaunay 2-Hop",
        "Voronoi-Delaunay 3-Hop",
        "KNN (No Radius Cutoff)",
        "KNN + Radius (\u00b5m)",
    ):
        iw.combo_graph_mode.setCurrentText(mode_label)
        res_inter = iw.run_analysis()
        assert res_inter is not None
        assert not res_inter.pairwise_df.empty

    iw.combo_graph_mode.setCurrentText("Voronoi-Delaunay 1-Hop")
    iw.run_analysis()
    iw.combo_source_type.setCurrentText("Tumor")
    iw.combo_target_type.setCurrentText("CD8_T")
    segs = iw.highlight_edges_on_canvas()
    assert segs.ndim == 3 and segs.shape[1:] == (2, 2)
    assert len(segs) > 0
    assert any("Interaction Edges:" in lyr.name for lyr in viewer.layers)

    # Also verify direct headless graph + interaction calls with extract_pair_edges_yx
    g_res = build_spatial_graph(df, mode="delaunay", hop=2, exact_ring=True, microns_per_pixel=0.5)
    direct_edges = extract_pair_edges_yx(
        df,
        g_res.adjacency,
        (df["cell_type"] == "Tumor").to_numpy(),
        (df["cell_type"] == "CD8_T").to_numpy(),
    )
    assert direct_edges.ndim == 3 and direct_edges.shape[1:] == (2, 2)
    inter_direct = run_pairwise_interactions(
        df, graph_result=g_res, n_permutations=50, group_map=group_map
    )
    assert not inter_direct.cohort_stats_df.empty

    inter_studio = iw.open_popup_window()
    assert inter_studio is not None
    inter_pdf = tmp_path / "interactions.pdf"
    inter_xlsx = tmp_path / "interactions.xlsx"
    inter_studio.export_figure_to_path(inter_pdf)
    inter_studio.export_source_data_to_path(inter_xlsx)
    assert inter_pdf.is_file() and inter_xlsx.is_file()
    inter_studio.close()

    # 4. Neighborhoods: Moving Window + Spatial-LDA + Studio PDF/XLSX export
    panel.switch_tab(3)
    nw = panel.neighborhoods_widget

    nw.combo_method.setCurrentText("Moving Window")
    nw.spin_mw_niches.setValue(4)
    nw.chk_mw_contexts.setChecked(True)
    res_mw = nw.extract_neighborhoods(sync=True)
    assert isinstance(res_mw, MovingWindowCNResult)
    assert "Spatial_Context" in res_mw.updated_df.columns
    assert res_mw.inter_cn_coordination is not None

    nw.combo_method.setCurrentText("Spatial-LDA")
    nw.spin_lda_topics.setValue(4)
    nw.spin_lda_iters.setValue(3)
    res_lda = nw.extract_neighborhoods(sync=True)
    assert isinstance(res_lda, SpatialLDAResult)
    assert np.allclose(res_lda.topic_probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert "SpatialLDA_Entropy" in res_lda.updated_df.columns

    ov_layers = nw.render_canvas_overlay()
    assert len(ov_layers) >= 1

    nb_studio = nw.open_popup_window()
    assert nb_studio is not None
    nb_pdf = tmp_path / "neighborhoods.pdf"
    nb_xlsx = tmp_path / "neighborhoods.xlsx"
    nb_studio.export_figure_to_path(nb_pdf)
    nb_studio.export_source_data_to_path(nb_xlsx)
    assert nb_pdf.is_file() and nb_xlsx.is_file()
    wb = openpyxl.load_workbook(nb_xlsx)
    assert "Summary_Data" in wb.sheetnames
    wb.close()
    nb_studio.close()

    # 5. Direct PublicationPlotStudioDialog verification
    def _simple_plot(fig: Figure, _cfg: PlotStyleConfig) -> list[matplotlib.axes.Axes]:
        ax = fig.add_subplot(111)
        ax.bar(["Responder", "NonResponder"], [2.5, 1.0])
        ax.set_ylabel("Density")
        return [ax]

    pub_dlg = PublicationPlotStudioDialog(
        plot_fn=_simple_plot,
        summary_df=res_den.summary_df,
        replicates_df=res_den.replicates_df,
        stats_df=res_den.stats_df,
        categories=["Responder", "NonResponder"],
    )
    pub_pdf = tmp_path / "pub_studio.pdf"
    pub_xlsx = tmp_path / "pub_studio.xlsx"
    pub_dlg.export_figure_to_path(pub_pdf)
    pub_dlg.export_source_data_to_path(pub_xlsx)
    assert pub_pdf.is_file() and pub_pdf.stat().st_size > 100
    assert pub_xlsx.is_file()
    pub_dlg.close()

    # Verify headless compute_cell_density & compute_cell_proportions & direct engines
    h_den = compute_cell_density(df, rois=rois, scale_factor=0.5, group_map=group_map)
    h_prop = compute_cell_proportions(df, group_map=group_map)
    h_mw = run_moving_window_cn(df, n_niches=4, n_permutations=40)
    h_lda = run_spatial_lda(df, n_topics=3, n_iters=2)
    assert not h_den.summary_df.empty and not h_prop.summary_df.empty
    assert not h_mw.replicates_df.empty and h_lda.topic_probabilities.shape == (len(df), 3)

    panel.close()
    qapp.processEvents()


def test_controller_sync_segmentation_sources_and_auto_preview(qapp):
    """Verify active mask visibility sync, sampleSelected clearing, and 400ms auto-preview."""
    img, markers, seg_results = _build_segmentation_test_scene()
    viewer = _FakeViewer()
    state = ViewerAppState()
    state.raw_image = img
    state.markers = markers
    state.current_sample_name = "Core_A1"

    cw = ImageControlWidget(viewer=None)
    seg_panel = SegmentationPanelWidget()
    seg_panel.update_channel_dropdowns(markers)
    dp = seg_panel.disagreement_panel
    fp = seg_panel.feature_panel
    legend = ViewerLegendWidget(viewer)

    ctx = AppContext(
        viewer=viewer,
        state=state,
        control_widget=cw,
        segmentation_panel=seg_panel,
        disagreement_panel=dp,
        feature_panel=fp,
        legend_widget=legend,
        added_legend_entries=set(),
        flagged_cells=[],
        current_issue_idx=-1,
    )
    review_svc = ReviewService(ctx)
    ctx.review_svc = review_svc
    seg_panel.ctx = ctx

    class _ControllerHarness(ViewerController):
        def __init__(self) -> None:
            pass

    ctrl = _ControllerHarness()
    ctrl.viewer = viewer
    ctrl.state = state
    ctrl.control_widget = cw
    ctrl.comparison_widget = cw.comparison_widget
    ctrl.segmentation_panel = seg_panel
    ctrl.disagreement_panel = dp
    ctrl.feature_panel = fp
    ctrl.legend_widget = legend
    ctrl.added_legend_entries = ctx.added_legend_entries
    ctrl.phenotyper_panel = MagicMock()
    ctrl.marker_widget = MagicMock()
    ctrl.analyzer_panel = None
    ctrl.review_svc = review_svc
    ctrl._ctx = ctx
    ctrl.image_svc = MagicMock()
    ctrl.image_svc.on_sample_selected.side_effect = lambda _s: state.reset_sample_state()
    ctrl.color_svc = MagicMock()
    ctrl.annotation_svc = MagicMock()
    ctrl.boundary_svc = MagicMock()
    ctrl.batch_svc = MagicMock()
    ctrl.multi_sample_svc = MagicMock()

    ctrl._bind_control_signals()
    ctrl._bind_disagreement_signals()
    ctrl._bind_feature_signals()

    # Run group completions: _on_group_complete adds layers with visible=False, then
    # _sync_segmentation_sources() routes through review_svc.set_active_mask_layers().
    ctrl._on_group_complete("Cellpose", seg_results["Cellpose"])
    assert state.active_segmentation_method == "Cellpose"
    assert viewer.layers["Cellpose Nuclear Mask"].visible is True
    assert viewer.layers["Cellpose Cyto Mask"].visible is True

    ctrl._on_group_complete("StarDist", seg_results["StarDist"])
    ctrl._on_all_segmentations_complete()
    assert viewer.layers["Cellpose Nuclear Mask"].visible is True
    assert viewer.layers["Cellpose Cyto Mask"].visible is True
    assert viewer.layers["StarDist Nuclear Mask"].visible is False
    assert dp.get_active_method() == "Cellpose"
    assert fp.get_selected_mask_source() == "Cellpose"

    # Switching active method via DisagreementPanel updates viewer layer visibility
    dp.combo_active_method.setCurrentText("StarDist")
    assert state.active_segmentation_method == "StarDist"
    assert viewer.layers["StarDist Nuclear Mask"].visible is True
    assert viewer.layers["Cellpose Nuclear Mask"].visible is False
    assert viewer.layers["Cellpose Cyto Mask"].visible is False

    # Selecting a new sample clears stale mask source combos via _sync_segmentation_sources()
    cw.sampleSelected.emit("Core_B2")
    assert state.active_segmentation_method is None
    assert dp.combo_active_method.count() == 0
    assert dp.combo_ref_method.count() == 0
    assert fp.combo_active_mask.count() == 0

    # Verify PipelineBuilderWidget 400 ms chk_auto_preview debounce & coalescing
    pb = seg_panel.pipeline_builder
    emitted_previews = []
    pb.previewRoiRequested.connect(lambda g, s: emitted_previews.append((g, s)))

    step0 = pb.tab_widget.widget(0).step_widgets[0]
    pb.chk_auto_preview.setChecked(False)
    step0.spin_cp_diam.setValue(30)
    assert pb._preview_timer.interval() == 400
    assert not pb._preview_timer.isActive()
    assert len(emitted_previews) == 0

    pb.chk_auto_preview.setChecked(True)
    step0.spin_cp_diam.setValue(32)
    step0.spin_cp_diam.setValue(38)
    assert pb._preview_timer.isActive()
    assert len(emitted_previews) == 0

    loop = QEventLoop()
    pb.previewRoiRequested.connect(lambda *_args: loop.quit())
    QTimer.singleShot(1500, loop.quit)
    loop.exec_()
    assert len(emitted_previews) == 1
    assert emitted_previews[0][0] == pb.tab_widget.tabText(0)
    assert emitted_previews[0][1][0].params["diameter"] == 38.0

    pb.chk_auto_preview.setChecked(False)
    pb._preview_timer.stop()
    seg_panel.close()
    cw.close()
    legend.close()
    qapp.processEvents()
