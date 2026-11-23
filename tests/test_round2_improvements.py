"""
Comprehensive Round 2 Verification Suite
========================================
Validates all 6 Round 2 improvement streams across iris_common, iris_viewer,
iris_phenotyper, and iris_analyzer:
1. Control-reference theme, concise Figure Studio names, & zero journal branding.
2. Filled segmentation mask defaults (contour=0, nuclear opacity=0.8, cyto=0.5).
3. Low-IoU disagreement bug fix, Cellpose two-pass separation, rich metrics,
   per-method color highlighting, & Hotspot Patch Inspector.
4. Canonical 3-module terminal logging (IRIS-Viewer / IRIS-Phenotyper / IRIS-Analyzer).
5. Cross-module Sample Scope (Active Sample(s), Selected Samples, All Samples).
"""
from __future__ import annotations

import ast
import re
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QColorDialog, QGroupBox, QLabel, QPushButton

from iris_analyzer.density import ROISpec
from iris_analyzer.widgets import (
    CellDensityStudioDialog,
    CellDensityWidget,
    CellProportionStudioDialog,
    CellProportionWidget,
    CellularNeighborhoodStudioDialog,
    CellularNeighborhoodsWidget,
    IrisAnalyzerPanelWidget,
    PairwiseInteractionStudioDialog,
    PairwiseInteractionsWidget,
)
from iris_common.logging import (
    ALLOWED_MODULES,
    format_iris_log,
    iris_log,
    log_error,
    log_info,
    log_warn,
)
from iris_common.widgets import (
    SCOPE_ACTIVE,
    SCOPE_ALL,
    SCOPE_SELECTED,
    THEME_PRESETS,
    FigureStudioDialog,
    FigureStyleSidebar,
    PlotStyleConfig,
    PublicationPlotStudioDialog,
    PublicationStyleSidebar,
    SampleScopeBar,
)
from iris_common.widgets.publication_studio import (
    SCIENTIFIC_PALETTES,
    get_palette_colors,
)
from iris_common.widgets.theme import (
    BUTTON_CTA,
    BUTTON_PRIMARY,
    BUTTON_STUDIO,
    DIALOG_DARK_THEME,
    PUBLICATION_RC_PRESETS,
    SHARED_DARK_THEME,
)
from iris_viewer.app.controller import ViewerController
from iris_viewer.app.services.multi_sample_service import MultiSampleService
from iris_viewer.core.context import AppContext
from iris_viewer.core.layers import get_layer_base_name
from iris_viewer.core.state import ViewerAppState
from iris_viewer.engine.disagreement import (
    METHOD_QC_COLORS,
    compute_model_disagreement,
    render_method_comparison_rgba,
)
from iris_viewer.engine.pipeline import SegmentationStep
from iris_viewer.engine.segmentation.cellpose_backend import CellposeBackend
from iris_viewer.widgets.disagreement_panel import DisagreementPanelWidget
from iris_viewer.widgets.feature_panel import FeaturePanelWidget
from iris_viewer.widgets.image_control import ImageControlWidget
from iris_viewer.widgets.legend import ViewerLegendWidget
from iris_viewer.widgets.marker_expression import IrisMarkerExpressionWidget, MarkerExpressionDialog
from iris_viewer.widgets.phenotyping.diagnostics_dialog import PhenotyperDiagnosticDialog
from iris_viewer.widgets.phenotyping.gating_studio_dialog import GatingStudioDialog
from iris_viewer.widgets.phenotyping.panel import PhenotyperPanelWidget
from iris_viewer.widgets.phenotyping.sample_impact_dialog import SampleImpactDialog
from iris_viewer.widgets.segmentation_panel import SegmentationPanelWidget
from iris_viewer.widgets.segmentation_studio_dialog import SegmentationStudioDialog


def _pump_until(qapp, predicate, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        qapp.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for Qt/worker condition")
        time.sleep(0.02)
    qapp.processEvents()


def _close_controller(ctrl: ViewerController, qapp) -> None:
    try:
        sp = getattr(ctrl, "segmentation_panel", None)
        if sp is not None and getattr(sp, "popup_dialog", None) is not None:
            sp.popup_dialog.close()
    except Exception:
        pass
    try:
        ctrl.viewer.close()
    except Exception:
        pass
    qapp.processEvents()


def _make_sample_cohort_df() -> pd.DataFrame:
    rows = []
    specs = [
        ("SampleA", [(20, 20), (20, 60), (60, 20), (60, 60), (40, 40)], "Tumor", 5.0, 0.2),
        ("SampleB", [(22, 22), (22, 62), (62, 22), (62, 62), (42, 42)], "CD8_T", 0.3, 6.0),
        ("SampleC", [(24, 24), (24, 64), (64, 24), (64, 64), (44, 44)], "B_Cell", 1.5, 1.5),
    ]
    cid = 1
    for sample, coords, ctype, cd3_val, cd20_val in specs:
        for idx, (y, x) in enumerate(coords, start=1):
            rows.append(
                {
                    "Image": sample,
                    "ObjectNumber": idx,
                    "Global_ID": cid,
                    "Pos_Y": float(y),
                    "Pos_X": float(x),
                    "cell_type": ctype if idx % 2 == 1 else "Stroma",
                    "Ref_CellType": ctype if idx % 2 == 1 else "Stroma",
                    "CD3_mean": cd3_val + 0.1 * idx,
                    "CD20_mean": cd20_val + 0.1 * idx,
                    "PanCK_mean": 4.0 if ctype == "Tumor" else 0.4,
                    "DAPI_mean": 8.0 + 0.2 * idx,
                }
            )
            cid += 1
    return pd.DataFrame(rows)


def _make_2ch_tissue_and_mask(
    shape: tuple[int, int] = (96, 96),
    centers: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    if centers is None:
        centers = [(28, 28), (28, 68), (68, 48)]
    img = np.zeros((2, h, w), dtype=np.uint16)
    mask = np.zeros((2, h, w), dtype=np.uint16)
    yy, xx = np.ogrid[:h, :w]
    for cid, (cy, cx) in enumerate(centers, start=1):
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        nuc = dist2 <= 8**2
        cyto = dist2 <= 13**2
        ring = cyto & (~nuc)
        img[0][nuc] = 5000
        img[1][ring] = 3500
        mask[0][nuc] = cid
        mask[1][cyto] = cid
    return img, mask


# ---------------------------------------------------------------------------
# Test 1: Control-Reference Theme, Figure Studio Names, & Zero Journal Branding
# ---------------------------------------------------------------------------


def test_round2_theme_figure_studio_names_and_zero_branding(qapp):
    """Verify presets, palettes, studio titles, empty QGroupBox titles, and zero emojis."""
    expected_themes = ("IRIS Dark", "Classic White", "Boxed White", "Minimal Grid")
    assert tuple(THEME_PRESETS.keys()) == expected_themes
    assert tuple(PUBLICATION_RC_PRESETS.keys()) == expected_themes

    for key in list(THEME_PRESETS.keys()) + list(PUBLICATION_RC_PRESETS.keys()):
        for banned in ("Nature", "Cell Press", "Lancet"):
            assert banned not in key

    assert "Vibrant" in SCIENTIFIC_PALETTES
    assert "Clinical Blue" in SCIENTIFIC_PALETTES
    for key in SCIENTIFIC_PALETTES.keys():
        for banned in ("Nature NPG", "Lancet"):
            assert banned not in key

    # Legacy alias fallback resolves cleanly while UI dropdowns remain concise
    assert PlotStyleConfig.from_theme("Publication White (Nature)").theme == "Classic White"
    assert PlotStyleConfig.from_theme("Cell Press Boxed").theme == "Boxed White"
    assert get_palette_colors("Nature NPG", 4) == SCIENTIFIC_PALETTES["Vibrant"][:4]
    assert get_palette_colors("Lancet", 4) == SCIENTIFIC_PALETTES["Clinical Blue"][:4]

    # Unified Control-reference theme tokens
    for token in ("#1f232b", "#232832", "#3c4352"):
        assert token in SHARED_DARK_THEME
        assert token in DIALOG_DARK_THEME
    assert "#3A6A9A" in BUTTON_PRIMARY
    assert "#2D5F8B" in BUTTON_STUDIO
    assert "#2D6A4F" in BUTTON_CTA

    assert FigureStudioDialog is PublicationPlotStudioDialog
    assert FigureStyleSidebar is PublicationStyleSidebar

    cohort_df = _make_sample_cohort_df()
    widgets_to_close = []
    try:
        nw = CellularNeighborhoodsWidget()
        widgets_to_close.append(nw)
        method_items = [nw.combo_method.itemText(i) for i in range(nw.combo_method.count())]
        assert method_items == ["Moving Window", "Spatial-LDA"]
        for item in method_items:
            assert "Nolan Lab" not in item
            assert "Calico" not in item
        nw.set_method("Spatial-LDA (Calico)")
        assert nw.combo_method.currentText() == "Spatial-LDA"
        nw.set_method("Moving Window (Nolan Lab)")
        assert nw.combo_method.currentText() == "Moving Window"

        pub_dlg = FigureStudioDialog()
        seg_studio = SegmentationStudioDialog()
        pheno_panel = PhenotyperPanelWidget()
        pheno_diag = PhenotyperDiagnosticDialog()
        gating_studio = GatingStudioDialog()
        impact_dlg = SampleImpactDialog()
        expr_widget = IrisMarkerExpressionWidget()
        expr_dlg = MarkerExpressionDialog(
            df=cohort_df,
            markers=["CD3", "CD20", "PanCK"],
            celltype_col="cell_type",
        )
        ctrl_w = ImageControlWidget()
        seg_panel = SegmentationPanelWidget()
        dis_panel = DisagreementPanelWidget()
        feat_panel = FeaturePanelWidget()
        analyzer_panel = IrisAnalyzerPanelWidget()
        den_w = CellDensityWidget()
        prop_w = CellProportionWidget()
        inter_w = PairwiseInteractionsWidget()
        scope_bar = SampleScopeBar()

        widgets_to_close.extend(
            [
                pub_dlg,
                seg_studio,
                pheno_panel,
                pheno_diag,
                gating_studio,
                impact_dlg,
                expr_widget,
                expr_dlg,
                ctrl_w,
                seg_panel,
                dis_panel,
                feat_panel,
                analyzer_panel,
                den_w,
                prop_w,
                inter_w,
                scope_bar,
            ]
        )

        # Studio button text & dialog window titles
        assert pub_dlg.windowTitle() == "IRIS Figure Studio"
        assert seg_studio.windowTitle() == "Segmentation Disagreement QC"
        assert seg_studio.btn_publication_studio.text() == "Figure Studio"
        assert pheno_diag.btn_pub_studio.text() == "Figure Studio"
        assert expr_dlg.btn_pub_studio.text() == "Figure Studio"

        # Analyzer Studio dialogs
        analyzer_panel.set_data(cohort_df, markers=["CD3", "CD20"], celltype_col="cell_type")
        analyzer_panel.density_widget.compute_density()
        den_studio = analyzer_panel.density_widget.open_popup_window()
        analyzer_panel.proportion_widget.compute_proportions()
        prop_studio = analyzer_panel.proportion_widget.open_popup_window()
        analyzer_panel.interactions_widget.run_analysis()
        inter_studio = analyzer_panel.interactions_widget.open_popup_window()
        analyzer_panel.neighborhoods_widget.extract_neighborhoods(sync=True)
        nb_studio = analyzer_panel.neighborhoods_widget.open_popup_window()
        widgets_to_close.extend([den_studio, prop_studio, inter_studio, nb_studio])

        assert isinstance(den_studio, CellDensityStudioDialog)
        assert isinstance(prop_studio, CellProportionStudioDialog)
        assert isinstance(inter_studio, PairwiseInteractionStudioDialog)
        assert isinstance(nb_studio, CellularNeighborhoodStudioDialog)
        for sd in (den_studio, prop_studio, inter_studio, nb_studio):
            assert "Figure Studio" in sd.windowTitle()
            assert "Publication Studio" not in sd.windowTitle()

        # Zero emojis in PhenotyperPanelWidget labels & zero Publication Studio button labels
        for w in widgets_to_close:
            for box in w.findChildren(QGroupBox):
                assert box.title() == "", f"{type(w).__name__} has QGroupBox title {box.title()!r}"
            for btn in w.findChildren(QPushButton):
                assert "Publication Studio" not in btn.text()
                for banned in ("Nature", "Cell Press", "Lancet", "Nolan Lab", "Calico"):
                    assert banned not in btn.text()

        for lbl in pheno_panel.findChildren(QLabel):
            txt = lbl.text()
            assert "\u2713" not in txt
            assert "\u26a0" not in txt
            assert "Nature Communications" not in txt
    finally:
        for w in reversed(widgets_to_close):
            try:
                w.close()
            except Exception:
                pass
        qapp.processEvents()


# ---------------------------------------------------------------------------
# Test 2: Mask Visualization Defaults (contour=0, nuclear=0.8, cyto=0.5)
# ---------------------------------------------------------------------------


def test_round2_mask_visualization_defaults_across_viewer(qapp, tmp_path):
    """Verify contour=0, nuclear opacity=0.8, and cytoplasm opacity=0.5 everywhere."""
    img_2ch, mask_2ch = _make_2ch_tissue_and_mask((96, 96))
    mask_path = tmp_path / "SampleA_mask.tiff"
    tifffile.imwrite(str(mask_path), mask_2ch)

    ctrl = ViewerController()
    try:
        # 1. Single-sample ViewerController segmentation completion & ROI preview
        nuc_m = mask_2ch[0].astype(np.uint32)
        cyto_m = mask_2ch[1].astype(np.uint32)
        ctrl.state.current_sample_name = "SampleA"
        ctrl._on_group_complete("PipeA", {"nuclear": nuc_m, "cytoplasm": cyto_m})
        ctrl._on_roi_preview_complete(
            "PipeA",
            (0, 0, 64, 64),
            {"nuclear": nuc_m[:64, :64], "cytoplasm": cyto_m[:64, :64]},
        )

        for lyr_name in ("PipeA Nuclear Mask", "ROI Preview - Nuclear"):
            lyr = ctrl.viewer.layers[lyr_name]
            assert lyr.contour == 0
            assert lyr.opacity == pytest.approx(0.8)
        for lyr_name in ("PipeA Cyto Mask", "ROI Preview - Cytoplasm"):
            lyr = ctrl.viewer.layers[lyr_name]
            assert lyr.contour == 0
            assert lyr.opacity == pytest.approx(0.5)

        # 2. ImageService.load_and_display_mask (2-ch nuclear + cytoplasm TIFF)
        ctrl.viewer.layers.clear()
        ctrl.added_legend_entries.clear()
        ctrl.image_svc.load_and_display_mask(mask_path, "SampleA", scale_val=1.0)
        nuc_loaded = ctrl.viewer.layers["Nuclear Segmentation"]
        cyto_loaded = ctrl.viewer.layers["Cytoplasm Segmentation"]
        assert nuc_loaded.contour == 0
        assert nuc_loaded.opacity == pytest.approx(0.8)
        assert cyto_loaded.contour == 0
        assert cyto_loaded.opacity == pytest.approx(0.5)

        # 3. ReviewService.set_active_mask_layers & flicker_models
        ctrl.viewer.layers.clear()
        ctrl.state.segmentation_results = {
            "MethodA": {"nuclear": nuc_m, "cytoplasm": cyto_m},
            "MethodB": {"nuclear": nuc_m.copy(), "cytoplasm": cyto_m.copy()},
        }
        shown = ctrl.review_svc.flicker_models("MethodA", "MethodB")
        assert shown == "MethodA"
        for m in ("MethodA", "MethodB"):
            assert ctrl.viewer.layers[f"{m} Nuclear Mask"].contour == 0
            assert ctrl.viewer.layers[f"{m} Nuclear Mask"].opacity == pytest.approx(0.8)
            assert ctrl.viewer.layers[f"{m} Cyto Mask"].contour == 0
            assert ctrl.viewer.layers[f"{m} Cyto Mask"].opacity == pytest.approx(0.5)

        ctrl.review_svc.set_active_mask_layers("MethodB")
        assert ctrl.viewer.layers["MethodB Nuclear Mask"].visible is True
        assert ctrl.viewer.layers["MethodB Nuclear Mask"].contour == 0
        assert ctrl.viewer.layers["MethodB Nuclear Mask"].opacity == pytest.approx(0.8)
        assert ctrl.viewer.layers["MethodB Cyto Mask"].contour == 0
        assert ctrl.viewer.layers["MethodB Cyto Mask"].opacity == pytest.approx(0.5)

        # 4. MultiSampleService._on_samples_loaded
        ctrl.viewer.layers.clear()
        ctrl.state.sample_layers_map.clear()
        ctrl.state.multi_sample_images.clear()
        ms_svc = ctrl.multi_sample_svc
        assert isinstance(ms_svc, MultiSampleService)
        loaded_payload = [
            (
                "Core1",
                [("DAPI", img_2ch[0].astype(np.float32), [0.0, 5000.0])],
                [nuc_m, cyto_m],
                pd.DataFrame(),
                "cell_type",
                img_2ch,
            ),
            (
                "Core2",
                [("DAPI", img_2ch[0].astype(np.float32), [0.0, 5000.0])],
                [nuc_m, cyto_m],
                pd.DataFrame(),
                "cell_type",
                img_2ch,
            ),
        ]
        ms_svc._on_samples_loaded(loaded_payload, ["Core1", "Core2"], scale_val=1.0)
        nuc_layers = [
            lyr for lyr in ctrl.viewer.layers if get_layer_base_name(lyr) == "Nuclear Segmentation"
        ]
        cyto_layers = [
            lyr
            for lyr in ctrl.viewer.layers
            if get_layer_base_name(lyr) == "Cytoplasm Segmentation"
        ]
        assert len(nuc_layers) == 2
        assert len(cyto_layers) == 2
        for lyr in nuc_layers:
            assert lyr.contour == 0
            assert lyr.opacity == pytest.approx(0.8)
        for lyr in cyto_layers:
            assert lyr.contour == 0
            assert lyr.opacity == pytest.approx(0.5)
    finally:
        _close_controller(ctrl, qapp)


# ---------------------------------------------------------------------------
# Test 3: Low-IoU Bug Fixed + Method Color Highlighting + Hotspot Inspector
# ---------------------------------------------------------------------------


def test_round2_cellpose_two_pass_and_both_mode_iou_fix(monkeypatch):
    """Verify Cellpose two-pass separation and Compare='Both' nuclear consistency."""
    # Case A: Mocked CellposeBackend with has_membrane=False vs has_membrane=True
    nuc_expected = np.zeros((64, 64), dtype=np.uint32)
    nuc_expected[10:20, 10:20] = 1
    wc_expected = np.zeros((64, 64), dtype=np.uint32)
    wc_expected[5:25, 5:25] = 1

    calls: list[tuple[tuple[int, ...], dict]] = []

    class FakeCellposeModel:
        def __init__(self, gpu=False, device=None, pretrained_model=None, model_type=None):
            self.pretrained_model = pretrained_model
            self.model_type = model_type

        def eval(self, img, channel_axis=None, **kwargs):
            calls.append((tuple(img.shape), dict(kwargs)))
            if img.ndim == 2:
                return nuc_expected.copy(), None, None
            return wc_expected.copy(), None, None

    fake_models_mod = types.ModuleType("cellpose.models")
    fake_models_mod.CellposeModel = FakeCellposeModel
    fake_models_mod.MODEL_NAMES = ["cpsam", "cyto3", "nuclei"]
    fake_models_mod.MODEL_DIR = None
    fake_cp_mod = types.ModuleType("cellpose")
    fake_cp_mod.models = fake_models_mod

    monkeypatch.setitem(sys.modules, "cellpose", fake_cp_mod)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models_mod)

    backend = CellposeBackend()
    img = np.ones((2, 64, 64), dtype=np.float32)

    res_nuc_only = backend.segment(
        img, nuclear_channel=0, membrane_channel=None, model_type="cyto3", device="cpu"
    )
    assert res_nuc_only["nuclear"] is not None
    assert res_nuc_only["cytoplasm"] is None
    assert np.array_equal(res_nuc_only["nuclear"], nuc_expected)

    calls.clear()
    res_two_pass = backend.segment(
        img, nuclear_channel=0, membrane_channel=1, model_type="cyto3", device="cpu"
    )
    assert len(calls) == 2
    assert calls[0][0] == (64, 64)
    assert calls[1][0] == (2, 64, 64)
    assert res_two_pass["nuclear"] is not None
    assert res_two_pass["cytoplasm"] is not None
    assert np.array_equal(res_two_pass["nuclear"], nuc_expected)
    assert np.array_equal(res_two_pass["cytoplasm"], wc_expected)
    assert not np.array_equal(res_two_pass["nuclear"], res_two_pass["cytoplasm"])

    # Case B: Compare="Both" when MethodA has 75 px nuclear + 300 px cytoplasm and
    # MethodB has identical 75 px nuclear + cytoplasm=None -> IoU=1.0, 0 flagged!
    nuc_75 = np.zeros((64, 64), dtype=np.uint32)
    nuc_75[20:25, 20:35] = 1  # 5 * 15 = 75 px
    cyto_300 = np.zeros((64, 64), dtype=np.uint32)
    cyto_300[15:30, 15:35] = 1  # 15 * 20 = 300 px

    report_both = compute_model_disagreement(
        {
            "MethodA": {"nuclear": nuc_75, "cytoplasm": cyto_300},
            "MethodB": {"nuclear": nuc_75.copy(), "cytoplasm": None},
        },
        compare_mode="Both",
        iou_threshold=0.65,
    )
    assert report_both.mean_iou == pytest.approx(1.0)
    assert report_both.flagged_regions == 0
    assert report_both.agreed_cells == 1


def test_round2_disagreement_metrics_method_colors_and_hotspot_inspector(qapp, monkeypatch):
    """Verify Dice, Overlap %, method checkboxes/swatches, overlays, legend, & Studio."""
    nuc_a = np.zeros((80, 80), dtype=np.uint32)
    nuc_b = np.zeros((80, 80), dtype=np.uint32)
    nuc_loaded = np.zeros((80, 80), dtype=np.uint32)

    # Cell 1: MethodA is 4x5 (20 px) strictly inside MethodB 10x10 (100 px) -> overlap_coeff=1.0
    nuc_a[10:14, 10:15] = 1
    nuc_b[8:18, 8:18] = 1
    # Cell 2: Split in MethodB
    nuc_a[45:60, 40:60] = 2
    nuc_b[45:60, 40:49] = 2
    nuc_b[45:60, 50:60] = 3
    nuc_loaded[5:10, 5:10] = 1

    seg_results = {
        "Loaded": {"nuclear": nuc_loaded, "cytoplasm": None},
        "PipeA": {"nuclear": nuc_a, "cytoplasm": None},
        "PipeB": {"nuclear": nuc_b, "cytoplasm": None},
    }

    report = compute_model_disagreement(
        seg_results,
        compare_mode="Nuclear Masks",
        reference_method="PipeA",
        iou_threshold=0.65,
        selected_methods=["PipeA", "PipeB"],
    )
    assert report.methods == ["PipeA", "PipeB"]
    assert report.flagged_regions >= 1
    first_item = report.items[0]
    assert 0.0 < first_item.dice <= 1.0
    assert 0.0 < first_item.overlap_coeff <= 1.0
    nested_items = [it for it in report.items if 1 in it.method_cell_ids.get("PipeA", [])]
    assert nested_items
    assert nested_items[0].overlap_coeff == pytest.approx(1.0)

    rgba, resolved_colors = render_method_comparison_rgba(
        seg_results,
        compare_mode="Nuclear Masks",
        selected_methods=["PipeA", "PipeB"],
    )
    assert rgba.shape == (80, 80, 4)
    assert resolved_colors["PipeA"] == METHOD_QC_COLORS[0]
    assert resolved_colors["PipeB"] == METHOD_QC_COLORS[1]
    assert np.any(rgba[..., 3] > 0.9)

    # Wire real Viewer + DisagreementPanelWidget + ViewerLegendWidget + ReviewService + Studio
    ctrl = ViewerController()
    try:
        ctrl.state.raw_image = np.ones((2, 80, 80), dtype=np.float32) * 100.0
        ctrl.state.segmentation_results = seg_results
        ctrl.control_widget.chk_canvas_legend.setChecked(True)
        qapp.processEvents()

        dp = ctrl.disagreement_panel
        assert dp.get_overlay_color_mode() == "Color by Method"
        dp.update_available_methods(
            ["Loaded", "PipeA", "PipeB"],
            active_method="PipeA",
            cell_counts={"Loaded": 1, "PipeA": 2, "PipeB": 3},
        )
        # Auto-unchecked "Loaded" when >= 2 computed pipelines exist
        assert dp.get_selected_methods() == ["PipeA", "PipeB"]

        # Customize PipeB color via mocked QColorDialog
        color_events = []
        dp.methodColorsChanged.connect(color_events.append)
        monkeypatch.setattr(
            QColorDialog, "getColor", staticmethod(lambda *a, **kw: QColor("#12AAFF"))
        )
        dp._method_row_widgets["PipeB"]["swatch"].click()
        qapp.processEvents()
        assert dp.get_method_colors()["PipeB"] == "#12aaff"
        assert color_events

        ctrl.review_svc.render_disagreement_overlays(report)
        assert "Method Comparison Overlay" in ctrl.viewer.layers
        mc_lyr = ctrl.viewer.layers["Method Comparison Overlay"]
        assert mc_lyr.visible is True
        assert mc_lyr.metadata["method_colors"]["PipeB"] == "#12aaff"

        assert isinstance(ctrl.legend_widget, ViewerLegendWidget)
        rows_by_name = {r.name: r for r in ctrl.legend_widget.disagreement_method_rows}
        assert set(rows_by_name.keys()) == {"Loaded", "PipeA", "PipeB"}
        assert rows_by_name["Loaded"].checkbox.isChecked() is False
        assert rows_by_name["PipeA"].checkbox.isChecked() is True
        assert rows_by_name["PipeB"].checkbox.isChecked() is True

        ctrl.review_svc.show_competing_masks(first_item.method_cell_ids)
        assert "Active Disagreement Isolation" in ctrl.viewer.layers
        iso_lyr = ctrl.viewer.layers["Active Disagreement Isolation"]
        assert iso_lyr.metadata["method_colors"]["PipeB"] == "#12aaff"

        # CanvasLegendOverlay displays QC method rows cleanly
        ctrl.control_widget.canvas_legend.show()
        ctrl.control_widget.canvas_legend.refresh()
        canvas_texts = [
            lbl.text() for lbl in ctrl.control_widget.canvas_legend.findChildren(QLabel)
        ]
        assert any("PipeB (QC)" in t for t in canvas_texts)

        # SegmentationStudioDialog: Method Color Key pills + 8-col table + Hotspot Patch Inspector
        studio = SegmentationStudioDialog(
            ctx=ctrl._ctx,
            disagreement_panel=dp,
            feature_panel=ctrl.feature_panel,
        )
        try:
            assert len(studio._pill_widgets) == 3
            headers = [
                studio.table_issues.horizontalHeaderItem(c).text()
                for c in range(studio.table_issues.columnCount())
            ]
            assert headers == [
                "#",
                "Region ID",
                "Category",
                "Best IoU",
                "Dice",
                "Overlap %",
                "Models",
                "Picked",
            ]
            assert studio.table_issues.rowCount() == len(report.items)

            tab_titles = [
                studio.right_tabs.tabText(i) for i in range(studio.right_tabs.count())
            ]
            assert "Hotspot Patch Inspector" in tab_titles
            studio.right_tabs.setCurrentIndex(tab_titles.index("Hotspot Patch Inspector"))
            studio.table_issues.selectRow(0)
            qapp.processEvents()
            assert len(studio.fig_hotspot.axes) >= 1
            hs_ax = studio.fig_hotspot.axes[0]
            assert hs_ax.get_legend() is not None
            assert "Hotspot #" in hs_ax.get_title()
        finally:
            studio.close()
            qapp.processEvents()
    finally:
        _close_controller(ctrl, qapp)


# ---------------------------------------------------------------------------
# Test 4: Canonical 3-Module Terminal Logging (iris_common.logging)
# ---------------------------------------------------------------------------


def test_round2_canonical_3_module_terminal_logging(qapp, capsys, monkeypatch):
    """Verify strict 3-module logging API, live emissions, and AST zero stray print()."""
    assert ALLOWED_MODULES == ("IRIS-Viewer", "IRIS-Phenotyper", "IRIS-Analyzer")

    with pytest.raises(ValueError, match="Unknown IRIS module"):
        format_iris_log("Pipeline", "Step", "bad prefix")
    with pytest.raises(ValueError, match="Unknown IRIS module"):
        iris_log("IRIS-Unknown", "Sub", "bad prefix")

    assert format_iris_log("viewer", "Segmentation", "ok") == "[IRIS-Viewer] [Segmentation] ok"
    assert format_iris_log("phenotyper", "ASTIR", "fit") == "[IRIS-Phenotyper] [ASTIR] fit"
    assert format_iris_log("analyzer", "Density", "run") == "[IRIS-Analyzer] [Density] run"

    # Verify flush=True is forwarded cleanly
    flush_calls = []
    orig_print = print

    def _spy_print(*args, **kwargs):
        flush_calls.append(kwargs.get("flush"))
        return orig_print(*args, **kwargs)

    monkeypatch.setattr("builtins.print", _spy_print)
    capsys.readouterr()
    log_info("IRIS-Viewer", "ImageLoader", "Loaded")
    log_warn("IRIS-Phenotyper", "Nimbus", "Warn msg")
    log_error("IRIS-Analyzer", "Interactions", "Err msg")
    out = capsys.readouterr().out
    assert flush_calls == [True, True, True]
    pattern = re.compile(r"^\[(IRIS-Viewer|IRIS-Phenotyper|IRIS-Analyzer)\] \[[^\]]+\] ")
    for line in out.strip().splitlines():
        assert pattern.match(line), f"Non-canonical log line: {line!r}"

    # Verify live log emissions across Viewer, Phenotyper, and Analyzer operations
    df = _make_sample_cohort_df()
    capsys.readouterr()
    iris_log("IRIS-Viewer", "Features", "Extracted test features")
    ap = IrisAnalyzerPanelWidget()
    try:
        ap.set_data(df, markers=["CD3", "CD20"], celltype_col="cell_type")
        ap.density_widget.compute_density()
        ap.proportion_widget.compute_proportions()
        ap.interactions_widget.run_analysis()
        ap.neighborhoods_widget.extract_neighborhoods(sync=True)

        pp = PhenotyperPanelWidget()
        try:
            st = ViewerAppState()
            st.master_df = df.copy()
            st.markers = ["CD3", "CD20"]
            pp.ctx = AppContext(viewer=None, state=st)
            pp.set_data(st.master_df, markers=["CD3", "CD20"])
            pp._run_clustering()
        finally:
            pp.close()
    finally:
        ap.close()
        qapp.processEvents()

    live_lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert any(ln.startswith("[IRIS-Viewer] [") for ln in live_lines)
    assert any(ln.startswith("[IRIS-Phenotyper] [") for ln in live_lines)
    assert any(ln.startswith("[IRIS-Analyzer] [") for ln in live_lines)
    for ln in live_lines:
        assert pattern.match(ln), f"Unexpected terminal prefix: {ln!r}"

    # Static AST verification: zero stray print() calls in runtime modules
    repo_root = Path(__file__).resolve().parents[1]
    scan_paths = (
        list((repo_root / "src" / "iris_viewer").rglob("*.py"))
        + list((repo_root / "src" / "iris_analyzer").rglob("*.py"))
        + [
            repo_root / "src" / "iris_phenotyper" / m / "pipeline.py"
            for m in ("astir", "maps", "stellar", "cellsighter", "nimbus")
        ]
    )
    for fpath in scan_paths:
        tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print", f"Stray print() in {fpath}:{node.lineno}"


# ---------------------------------------------------------------------------
# Test 5: Sample Scope Across All Modules
# ---------------------------------------------------------------------------


def test_round2_sample_scope_bar_and_analyzer(qapp):
    """Verify SampleScopeBar and IrisAnalyzerPanelWidget sample scoping + ROI preservation."""
    bar = SampleScopeBar(default_mode=SCOPE_ALL, label="Scope:")
    try:
        bar.set_samples(["SampleA", "SampleB", "SampleC"], active_samples=["SampleB"])
        assert bar.resolve_samples() == ["SampleA", "SampleB", "SampleC"]

        bar.set_scope_mode(SCOPE_ACTIVE)
        assert bar.resolve_samples() == ["SampleB"]

        bar.set_scope_mode(SCOPE_SELECTED)
        bar._select_all_samples()
        assert bar.resolve_samples() == ["SampleA", "SampleB", "SampleC"]
        bar._clear_samples()
        assert bar.resolve_samples() == []

        bar.set_checked_samples(["SampleA", "SampleC"])
        df_ext = pd.DataFrame(
            {
                "Image": [
                    "SampleA_image.ome.tiff",
                    "SampleB.tiff",
                    "SampleC_mask.tif",
                ],
                "Val": [1, 2, 3],
            }
        )
        filtered = bar.filter_dataframe(df_ext)
        assert list(filtered["Val"]) == [1, 3]
    finally:
        bar.close()

    # IrisAnalyzerPanelWidget scoping across Density, Proportion, Interactions, Neighborhoods
    df = _make_sample_cohort_df()
    state = ViewerAppState()
    state.master_df = df.copy()
    state.current_sample_name = "SampleB"
    ctx = AppContext(viewer=None, state=state)

    rois = [
        ROISpec(
            name=f"ROI_{sn}",
            sample=sn,
            inclusion_polygons=[np.array([[0.0, 0.0], [0.0, 80.0], [80.0, 80.0], [80.0, 0.0]])],
        )
        for sn in ("SampleA", "SampleB", "SampleC")
    ]
    panel = IrisAnalyzerPanelWidget(ctx=ctx)
    try:
        panel.set_data(df, markers=["CD3", "CD20"], celltype_col="cell_type", ctx=ctx)
        panel.density_widget.set_rois(rois)

        # Scope to Active Sample(s) -> SampleB only
        panel.sample_scope_bar.set_scope_mode(SCOPE_ACTIVE)
        qapp.processEvents()
        assert len(panel.df) == 5
        assert set(panel.df["Image"].unique()) == {"SampleB"}

        res_den_active = panel.density_widget.compute_density()
        res_prop_active = panel.proportion_widget.compute_proportions()
        res_inter_active = panel.interactions_widget.run_analysis()
        res_nb_active = panel.neighborhoods_widget.extract_neighborhoods(sync=True)
        assert set(res_den_active.replicates_df["Image"].unique()) == {"SampleB"}
        assert set(res_prop_active.replicates_df["Image"].unique()) == {"SampleB"}
        assert res_inter_active.metadata["n_cells"] == 5
        assert len(res_nb_active.updated_df) == 5

        # Scope to Selected Samples -> SampleA + SampleC (10 cells)
        panel.sample_scope_bar.set_checked_samples(["SampleA", "SampleC"])
        panel.sample_scope_bar.set_scope_mode(SCOPE_SELECTED)
        qapp.processEvents()
        assert set(panel.df["Image"].unique()) == {"SampleA", "SampleC"}
        assert panel.density_widget.compute_density().replicates_df["Image"].nunique() == 2

        # Switch back to All Samples: all 3 ROIs preserved!
        panel.sample_scope_bar.set_scope_mode(SCOPE_ALL)
        qapp.processEvents()
        assert len(panel.density_widget.rois) == 3
        assert len(panel.df) == 15
        assert panel.density_widget.compute_density().replicates_df["Image"].nunique() == 3
    finally:
        panel.close()
        qapp.processEvents()


def test_round2_phenotyper_and_marker_expression_sample_scope(qapp, monkeypatch):
    """Verify Phenotyper (Clustering, MAPS, ASTIR, Nimbus, Gating) & Marker Expression scope."""
    import torch

    from iris_phenotyper.nimbus.pipeline import _get_unet_classes

    df = _make_sample_cohort_df()
    img, mask_2ch = _make_2ch_tissue_and_mask(
        (96, 96),
        centers=[(22, 22), (22, 62), (62, 22), (62, 62), (42, 42)],
    )
    nuc_mask = mask_2ch[0].astype(np.uint32)
    img_4ch = np.stack([img[0], img[1], img[0], img[1]], axis=0).astype(np.float32)

    state = ViewerAppState()
    state.master_df = df.copy()
    state.markers = ["CD3", "CD20", "PanCK", "DAPI"]
    state.current_sample_name = "SampleB"
    state.multi_sample_images = {"SampleA": img_4ch, "SampleB": img_4ch, "SampleC": img_4ch}
    state.masks_map = {"SampleA": nuc_mask, "SampleB": nuc_mask, "SampleC": nuc_mask}
    ctx = AppContext(viewer=None, state=state)

    tiny_unet = _get_unet_classes()["NimbusUNet"](filters_root=8, padding="reflect").eval()
    monkeypatch.setattr(
        "iris_phenotyper.nimbus.pipeline.load_nimbus_model",
        lambda **_kwargs: (tiny_unet, torch.device("cpu"), None),
    )

    panel = PhenotyperPanelWidget(ctx=ctx)
    expr = IrisMarkerExpressionWidget()
    try:
        panel.set_data(state.master_df, markers=state.markers)
        # Scope strictly to SampleB (rows 5..9)
        panel.sample_scope_bar.set_checked_samples(["SampleB"])
        panel.sample_scope_bar.set_scope_mode(SCOPE_SELECTED)
        qapp.processEvents()

        # Pre-populate non-scoped sample label to ensure no clobbering
        state.master_df["Cluster"] = "PreExisting"
        panel.combo_cluster_method.setCurrentText("KMeans")
        panel.spin_kmeans_k.setValue(2)
        panel._run_clustering()
        is_b = state.master_df["Image"] == "SampleB"
        assert (state.master_df.loc[is_b, "Cluster"] != "PreExisting").all()
        assert (state.master_df.loc[~is_b, "Cluster"] == "PreExisting").all()

        # ASTIR scoped to SampleB
        panel.spin_astir_epochs.setValue(2)
        panel._run_astir()
        assert (state.master_df.loc[is_b, "ASTIR_CellType"] != "Unassigned").all()
        assert (state.master_df.loc[~is_b, "ASTIR_CellType"] == "Unassigned").all()

        # MAPS scoped to SampleB
        panel.combo_maps_mode.setCurrentText("Marker-Rule Fallback")
        panel._run_maps()
        assert (state.master_df.loc[is_b, "MAPS_CellType"] != "Unassigned").all()
        assert (state.master_df.loc[~is_b, "MAPS_CellType"] == "Unassigned").all()

        # Nimbus scoped to SampleB
        panel.chk_nimbus_tta.setChecked(False)
        panel.spin_nimbus_pos_thresh.setValue(0.05)
        panel.combo_nimbus_mode.setCurrentText("Consensus Clustering")
        panel.spin_nimbus_clusters.setValue(2)
        panel._run_nimbus()
        assert (state.master_df.loc[is_b, "Nimbus_CellType"] != "Unassigned").all()
        assert (state.master_df.loc[~is_b, "Nimbus_CellType"] == "Unassigned").all()
        assert state.master_df.loc[is_b, "CD3_nimbus_positivity"].notna().all()
        assert state.master_df.loc[~is_b, "CD3_nimbus_positivity"].isna().all()

        # ManualGating scoped to SampleB
        state.master_df["ManualTest"] = "KeepOther"
        ctx.control_widget = SimpleNamespace(
            combo_ref=SimpleNamespace(
                currentText=lambda: "ManualTest",
                findText=lambda _t: 0,
                addItem=lambda _t: None,
                blockSignals=lambda _b: None,
                setCurrentText=lambda _t: None,
            )
        )
        panel.manual_gating.combo_x.setCurrentText("CD20")
        panel.manual_gating.combo_y.setCurrentText("CD3")
        panel.manual_gating.combo_scale.setCurrentText("Linear (Raw)")
        panel.manual_gating.combo_target_pop.setCurrentText("Q4 (CD20+ CD3-)")
        panel.manual_gating.spin_tx.setValue(2.0)
        panel.manual_gating.spin_ty.setValue(2.0)
        panel.manual_gating.txt_phenotype_name.setText("Scoped_B_Cell")
        panel.manual_gating._apply_gate()
        assert (state.master_df.loc[is_b, "ManualTest"] == "Scoped_B_Cell").all()
        assert (state.master_df.loc[~is_b, "ManualTest"] == "KeepOther").all()

        # IrisMarkerExpressionWidget scope filtering
        expr.combo_mode.setCurrentText("Raw Mean Intensity")
        expr.set_data(
            df=df,
            markers=["CD3", "CD20", "PanCK"],
            celltype_col="cell_type",
            selected_samples=["SampleA"],
        )
        expr.sample_scope_bar.set_scope_mode(SCOPE_ACTIVE)
        qapp.processEvents()
        ax_a = expr.figure.axes[0]
        labels_a = [t.get_text() for t in ax_a.get_yticklabels()]
        assert set(labels_a) == {"Tumor", "Stroma"}
        arr_a = np.asarray(ax_a.images[0].get_array())
        assert arr_a[labels_a.index("Tumor"), 0] > 4.0

        expr.sample_scope_bar.set_checked_samples(["SampleB"])
        expr.sample_scope_bar.set_scope_mode(SCOPE_SELECTED)
        qapp.processEvents()
        ax_b = expr.figure.axes[0]
        labels_b = [t.get_text() for t in ax_b.get_yticklabels()]
        assert set(labels_b) == {"CD8_T", "Stroma"}
        arr_b = np.asarray(ax_b.images[0].get_array())
        assert arr_b[labels_b.index("CD8_T"), 1] > 5.0

        expr.sample_scope_bar.set_scope_mode(SCOPE_ALL)
        qapp.processEvents()
        ax_all = expr.figure.axes[0]
        labels_all = [t.get_text() for t in ax_all.get_yticklabels()]
        assert set(labels_all) == {"Tumor", "CD8_T", "B_Cell", "Stroma"}
    finally:
        expr.close()
        panel.close()
        qapp.processEvents()


def test_round2_viewer_segmentation_and_feature_extraction_scopes(qapp, tmp_path):
    """Verify PipelineBuilder, FeaturePanel, and ViewerController across all 3 sample scopes."""
    img_a, mask_a = _make_2ch_tissue_and_mask((96, 96), centers=[(24, 24), (64, 64)])
    img_b, mask_b = _make_2ch_tissue_and_mask((96, 96), centers=[(30, 60), (70, 30), (50, 50)])
    img_c, mask_c = _make_2ch_tissue_and_mask((96, 96), centers=[(48, 48)])

    path_a = tmp_path / "SampleA_image.tiff"
    path_b = tmp_path / "SampleB_image.tiff"
    path_c = tmp_path / "SampleC_image.tiff"
    tifffile.imwrite(str(path_a), img_a)
    tifffile.imwrite(str(path_b), img_b)
    tifffile.imwrite(str(path_c), img_c)
    tifffile.imwrite(str(tmp_path / "SampleA_mask.tiff"), mask_a)
    tifffile.imwrite(str(tmp_path / "SampleB_mask.tiff"), mask_b)
    tifffile.imwrite(str(tmp_path / "SampleC_mask.tiff"), mask_c)

    ctrl = ViewerController()
    try:
        ctrl.state.markers = ["DAPI", "PanCK"]
        ctrl.state.file_map = {"SampleA": path_a, "SampleB": path_b, "SampleC": path_c}
        ctrl.segmentation_panel.update_channel_dropdowns(ctrl.state.markers)

        pb = ctrl.segmentation_panel.pipeline_builder
        fp = ctrl.feature_panel

        # 1. Single-Sample Active Scope
        ctrl.image_svc.on_sample_selected("SampleA")
        _pump_until(
            qapp,
            lambda: ctrl.state.raw_image is not None and len(ctrl._active_workers) == 0,
        )
        ctrl._sync_sample_scope_bars()
        assert pb.get_target_samples(["SampleA", "SampleB", "SampleC"], ["SampleA"]) == ["SampleA"]
        assert fp.get_target_samples(["SampleA", "SampleB", "SampleC"], ["SampleA"]) == ["SampleA"]

        fp.sample_scope_bar.set_scope_mode(SCOPE_ACTIVE)
        ctrl._run_feature_extraction(["area"], ["mean"])
        _pump_until(
            qapp,
            lambda: ctrl.state.features_df is not None and len(ctrl._active_workers) == 0,
        )
        assert set(ctrl.state.features_df["Image"].unique()) == {"SampleA"}

        # 2. Folder Selected Samples Scope (SampleB + SampleC without loading onto canvas)
        pb.sample_scope_bar.set_checked_samples(["SampleB", "SampleC"])
        pb.sample_scope_bar.set_scope_mode(SCOPE_SELECTED)
        ctrl._run_segmentation_pipeline(
            {"StarPipe": [SegmentationStep("StarDist", {"nuclear_channel": 0}, rule="Nuclear (Base)")]}
        )
        _pump_until(
            qapp,
            lambda: (
                len(ctrl.pipeline_service._active_workers) == 0
                and len(ctrl._active_workers) == 0
                and "SampleB" in ctrl.state.batch_results
                and "SampleC" in ctrl.state.batch_results
            ),
            timeout_s=45.0,
        )
        assert "SampleB" in ctrl.state.masks_map
        assert "SampleC" in ctrl.state.masks_map

        ctrl.state.features_df = None
        fp.sample_scope_bar.set_checked_samples(["SampleB", "SampleC"])
        fp.sample_scope_bar.set_scope_mode(SCOPE_SELECTED)
        ctrl._run_feature_extraction(["area"], ["mean"])
        _pump_until(
            qapp,
            lambda: ctrl.state.features_df is not None and len(ctrl._active_workers) == 0,
        )
        assert set(ctrl.state.features_df["Image"].unique()) == {"SampleB", "SampleC"}

        # 3. Folder All Samples Scope
        ctrl.state.features_df = None
        fp.sample_scope_bar.set_scope_mode(SCOPE_ALL)
        ctrl._run_feature_extraction(["area"], ["mean"])
        _pump_until(
            qapp,
            lambda: ctrl.state.features_df is not None and len(ctrl._active_workers) == 0,
        )
        assert set(ctrl.state.features_df["Image"].unique()) == {"SampleA", "SampleB", "SampleC"}

        # 4. Multi-Sample TMA Canvas Scope (Active Sample(s) covers canvas samples)
        ctrl.multi_sample_svc.add_samples_to_canvas(["SampleA", "SampleB"])
        _pump_until(
            qapp,
            lambda: (
                set(ctrl.state.multi_sample_images.keys()) == {"SampleA", "SampleB"}
                and len(ctrl._active_workers) == 0
            ),
        )
        ctrl.state.features_df = None
        fp.sample_scope_bar.set_scope_mode(SCOPE_ACTIVE)
        ctrl._run_feature_extraction(["area"], ["mean"])
        _pump_until(
            qapp,
            lambda: ctrl.state.features_df is not None and len(ctrl._active_workers) == 0,
        )
        assert set(ctrl.state.features_df["Image"].unique()) == {"SampleA", "SampleB"}
    finally:
        _close_controller(ctrl, qapp)
