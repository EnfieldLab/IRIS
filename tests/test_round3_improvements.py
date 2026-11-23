"""Regression and integration tests for Round 3 IRIS fixes & enhancements:

1. `iris` terminal CLI entrypoint alongside `iris-viewer` and `"IRIS"` UI window title.
2. `InstanSeg` float32 mask label handling in `_filter_mask_by_area` (no IndexError).
3. Untruncated QTabBar labels across Studios, Blink Compare clarity, and multi-channel
   fluorescence selection + Side-by-Side view in Hotspot Patch Inspector.
4. Main IRIS canvas Disagreement QC renders hollow contours (`fill_alpha=0.0`), keeps
   underlying fluorescence marker layers unobstructed by hiding opaque filled Labels
   masks during QC (restoring on exit or opt-in), and embeds a live Hotspot Patch preview.
"""
from __future__ import annotations

import importlib.metadata
import re
import sys
import types
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from iris_common.logging import ALLOWED_MODULES
from iris_common.widgets.publication_studio import PublicationPlotStudioDialog
from iris_viewer.app.controller import ViewerController
from iris_viewer.engine.disagreement import (
    DisagreementItem,
    DisagreementReport,
    compose_multichannel_patch_rgb,
    compute_model_disagreement,
    render_method_comparison_rgba,
)
from iris_viewer.engine.pipeline import (
    SegmentationSequence,
    SegmentationStep,
    crop_roi_and_segment_sync,
)
from iris_viewer.engine.segmentation.base import SegmentationBackend
from iris_viewer.engine.segmentation.instanseg_backend import InstanSegBackend
from iris_viewer.widgets.config_dialogs import ViewerPanelConfigDialog
from iris_viewer.widgets.phenotyping.diagnostics_dialog import PhenotyperDiagnosticDialog
from iris_viewer.widgets.phenotyping.gating_studio_dialog import GatingStudioDialog
from iris_viewer.widgets.segmentation_pipeline_builder import PipelineBuilderWidget
from iris_viewer.widgets.segmentation_studio_dialog import SegmentationStudioDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _close_controller(ctrl: ViewerController, qapp: QApplication) -> None:
    sp = getattr(ctrl, "segmentation_panel", None)
    if sp is not None and getattr(sp, "popup_dialog", None) is not None:
        try:
            sp.popup_dialog.close()
        except Exception:
            pass
    try:
        ctrl.viewer.close()
    except Exception:
        pass
    qapp.processEvents()


# ---------------------------------------------------------------------------
# Test 1: `iris` CLI Entrypoint & "IRIS" UI Platform Title
# ---------------------------------------------------------------------------


def test_round3_iris_cli_entrypoint_and_ui_title(qapp):
    """Verify `iris` CLI script registration and `"IRIS"` UI window titles."""
    repo_root = Path(__file__).resolve().parents[1]
    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    scripts_section = pyproject_text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    assert re.search(r'(?m)^\s*iris\s*=\s*"iris_viewer\.main:main"', scripts_section)
    assert re.search(r'(?m)^\s*iris-viewer\s*=\s*"iris_viewer\.main:main"', scripts_section)

    console_eps = importlib.metadata.entry_points(group="console_scripts")
    ep_names = {ep.name: ep.value for ep in console_eps}
    assert ep_names.get("iris") == "iris_viewer.main:main"
    assert ep_names.get("iris-viewer") == "iris_viewer.main:main"

    iris_bin = Path(sys.executable).parent / "iris"
    assert iris_bin.is_file()

    # Keep canonical 3-module logging prefixes intact
    assert ALLOWED_MODULES == ("IRIS-Viewer", "IRIS-Phenotyper", "IRIS-Analyzer")

    dlg = ViewerPanelConfigDialog(num_channels=3)
    ctrl = ViewerController()
    try:
        assert ctrl.viewer.title == "IRIS"
        assert dlg.windowTitle() == "IRIS — Marker Panel Configuration"
        assert dlg.windowTitle().startswith("IRIS ")
    finally:
        dlg.close()
        _close_controller(ctrl, qapp)


# ---------------------------------------------------------------------------
# Test 2: InstanSeg Float Label IndexError Fix
# ---------------------------------------------------------------------------


class _DummyBackend(SegmentationBackend):
    name = "Dummy"

    def is_available(self) -> bool:
        return True

    def get_default_params(self) -> dict:
        return {}

    def segment(self, image, nuclear_channel, membrane_channel, diameter=None, **kwargs):
        return {"nuclear": None, "cytoplasm": None}


def test_round3_filter_mask_by_area_float_and_instanseg(monkeypatch):
    """Verify _filter_mask_by_area & InstanSegBackend handle float32 masks cleanly."""
    dummy = _DummyBackend()

    # Reproduce exact float32 / float64 / NaN edge cases when min_area=15 > 0
    float_mask = np.zeros((100, 100), dtype=np.float32)
    float_mask[10:20, 10:20] = 1.0  # 100 px -> keep -> relabel 1
    float_mask[30:33, 30:33] = 2.0  # 9 px < 15 -> drop
    float_mask[50:60, 50:60] = 5.0  # 100 px -> keep -> relabel 2
    float_mask[0, 0] = np.nan
    float_mask[0, 1] = -3.0

    filtered = dummy._filter_mask_by_area(float_mask, min_area=15, max_area=5000)
    assert filtered.dtype == np.uint32
    assert set(np.unique(filtered).tolist()) == {0, 1, 2}
    assert filtered[15, 15] == 1
    assert filtered[55, 55] == 2
    assert filtered[31, 31] == 0

    # Reproduce full InstanSegBackend -> _filter_mask_by_area -> crop_roi_and_segment_sync
    import torch

    out_tensor = torch.zeros((1, 2, 128, 128), dtype=torch.float32)
    out_tensor[0, 0, 20:30, 20:30] = 1.0  # 100 px nucleus
    out_tensor[0, 0, 2:4, 2:4] = 2.0      # 4 px fragment (filtered out by min_cell_area=15)
    out_tensor[0, 1, 15:35, 15:35] = 1.0  # 400 px cell

    class FakeInstanSeg:
        def __init__(self, model_name, device="cpu", verbosity=0):
            self.model_name = model_name
            self.device = device

        def eval_medium_image(self, input_tensor, **_kwargs):
            return out_tensor.clone(), None

    fake_instanseg_mod = types.ModuleType("instanseg")
    fake_instanseg_mod.InstanSeg = FakeInstanSeg
    monkeypatch.setitem(sys.modules, "instanseg", fake_instanseg_mod)

    backend = InstanSegBackend()
    img = np.random.randint(0, 200, (2, 200, 200), dtype=np.uint16)
    seq = SegmentationSequence(
        steps=[
            SegmentationStep(
                method_name="InstanSeg",
                params={
                    "nuclear_channel": 0,
                    "membrane_channels": [1],
                    "model_name": "fluorescence_nuclei_and_cells",
                    "min_cell_area": 15,
                    "max_cell_area": 5000,
                },
                rule="Independent",
            )
        ]
    )
    res = crop_roi_and_segment_sync(
        sequence=seq,
        image=img,
        roi_bbox=(10, 10, 138, 138),
        backends={"InstanSeg": backend},
        device="cpu",
    )
    assert res["nuclear"] is not None
    assert res["cytoplasm"] is not None
    assert res["nuclear"].dtype == np.uint32
    assert res["cytoplasm"].dtype == np.uint32
    assert res["nuclear"].shape == (128, 128)
    assert set(np.unique(res["nuclear"]).tolist()) == {0, 1}


# ---------------------------------------------------------------------------
# Test 3: Untruncated Tab Bars, Blink Compare & Multi-Channel Hotspot Inspector
# ---------------------------------------------------------------------------


def test_round3_tabs_untruncated_and_multichannel_hotspot_inspector(qapp):
    """Verify full untruncated QTabBar text, lazy Dask RGB compositing, & channel UI."""
    # A. Tab bar non-elision across dialogs/widgets
    ctrl = ViewerController()
    pb = PipelineBuilderWidget()
    diag = PhenotyperDiagnosticDialog()
    gating = GatingStudioDialog()
    pub = PublicationPlotStudioDialog(plot_fn=lambda fig, cfg: None)
    try:
        for tab_w in (pb.tab_widget, diag.tabs, gating.tabs, pub.tabs):
            bar = tab_w.tabBar()
            assert bar.elideMode() == Qt.ElideNone
            assert bar.font().bold() is True

        # B. Setup multi-channel image, Napari marker layers, and 2-model disagreement
        h, w = 96, 96
        img_np = np.zeros((3, h, w), dtype=np.float32)
        img_np[0, 15:45, 15:45] = 800.0   # DAPI
        img_np[1, 10:50, 10:50] = 600.0   # PanCK
        img_np[2, 55:80, 55:80] = 700.0   # CD45
        dask_img = da.from_array(img_np, chunks=(1, 48, 48))

        ctrl.state.markers = ["DAPI", "PanCK", "CD45"]
        ctrl.state.raw_image = dask_img
        ctrl.viewer.add_image(img_np[0], name="DAPI", colormap="blue", visible=True)
        ctrl.viewer.add_image(img_np[1], name="PanCK", colormap="green", visible=False)
        ctrl.viewer.add_image(img_np[2], name="CD45", colormap="magenta", visible=True)

        # Headless engine patch composer: lazy Dask slice + custom channel combinations
        rgb_dapi_cd45 = compose_multichannel_patch_rgb(
            dask_img,
            (10, 10, 85, 85),
            channel_indices=[0, 2],
            channel_colors={0: "#4DA6FF", 2: "#FF4081"},
        )
        assert rgb_dapi_cd45.shape == (75, 75, 3)
        assert rgb_dapi_cd45[10, 10, 2] > 0.5  # DAPI blue signal present
        assert rgb_dapi_cd45[55, 55, 0] > 0.5  # CD45 magenta signal present

        rgb_empty = compose_multichannel_patch_rgb(
            dask_img, (10, 10, 50, 50), channel_indices=[]
        )
        assert rgb_empty.shape == (40, 40, 3)
        assert float(np.max(rgb_empty)) == 0.0

        nuc_a = np.zeros((h, w), dtype=np.uint32)
        nuc_b = np.zeros((h, w), dtype=np.uint32)
        nuc_a[20:40, 20:40] = 1
        nuc_b[24:46, 24:46] = 1

        seg_results = {
            "Cellpose": {"nuclear": nuc_a, "cytoplasm": None},
            "InstanSeg": {"nuclear": nuc_b, "cytoplasm": None},
        }
        ctrl.state.segmentation_results = seg_results
        report = compute_model_disagreement(
            seg_results, compare_mode="Nuclear Masks", iou_threshold=0.75
        )
        ctrl.state.disagreement_report = report

        studio = SegmentationStudioDialog(
            ctx=ctrl._ctx,
            disagreement_panel=ctrl.disagreement_panel,
            feature_panel=ctrl.feature_panel,
        )
        try:
            studio.show()
            qapp.processEvents()
            bar = studio.right_tabs.tabBar()
            assert bar.elideMode() == Qt.ElideNone
            fm = bar.fontMetrics()
            for i in range(bar.count()):
                txt_w = fm.horizontalAdvance(bar.tabText(i))
                assert bar.tabRect(i).width() >= txt_w, (
                    f"Tab {bar.tabText(i)!r} clipped: {bar.tabRect(i).width()} < {txt_w}"
                )

            # Streamlined Reference + Final Mask Builder controls (Blink Compare removed)
            assert not hasattr(studio, "btn_flicker")
            assert not hasattr(studio, "combo_flicker_a")
            assert not hasattr(studio, "combo_flicker_b")
            assert not hasattr(ctrl.disagreement_panel, "btn_flicker")
            assert studio.combo_ref_method.count() == 2
            assert hasattr(ctrl.disagreement_panel, "btn_use_single_model")
            assert hasattr(ctrl.disagreement_panel, "btn_auto_consensus")
            assert hasattr(studio, "btn_set_active_mask")
            assert hasattr(studio, "btn_auto_consensus")
            assert hasattr(studio, "btn_accept_selected_rows")
            assert hasattr(studio, "btn_accept_filtered")
            assert hasattr(studio, "btn_draw_curation_roi")
            assert hasattr(studio, "btn_apply_curation_roi")

            # Marker menu syncs active Napari channels (DAPI & CD45 True, PanCK False)
            assert set(studio._hotspot_channel_actions.keys()) == {0, 1, 2}
            assert studio._hotspot_channel_actions[0].isChecked() is True
            assert studio._hotspot_channel_actions[1].isChecked() is False
            assert studio._hotspot_channel_actions[2].isChecked() is True

            # Toggle PanCK on in Hotspot Patch Inspector and switch to Side-by-Side
            studio._hotspot_channel_actions[1].trigger()
            studio.combo_hotspot_view.setCurrentText("Side-by-Side Per Method")
            qapp.processEvents()
            assert len(studio.fig_hotspot.axes) == 2

            # Switch back to Composite Overlay -> 1 axis with legend & marker channels
            studio.combo_hotspot_view.setCurrentText("Composite Overlay")
            qapp.processEvents()
            assert len(studio.fig_hotspot.axes) == 1
        finally:
            studio.close()
            qapp.processEvents()
    finally:
        for wdg in (pb, diag, gating, pub):
            try:
                wdg.close()
            except Exception:
                pass
        _close_controller(ctrl, qapp)


# ---------------------------------------------------------------------------
# Test 4: Hollow QC Contours Leave Underlying Fluorescence Channels Visible
# ---------------------------------------------------------------------------


def test_round3_hollow_contours_preserve_underlying_channels_on_canvas(qapp):
    """Verify Disagreement QC hides opaque filled masks so fluorescence channels stay visible."""
    h, w = 80, 80
    nuc_a = np.zeros((h, w), dtype=np.uint32)
    nuc_b = np.zeros((h, w), dtype=np.uint32)
    nuc_a[15:35, 15:35] = 1
    nuc_b[20:42, 20:42] = 1

    seg_results = {
        "MethodA": {"nuclear": nuc_a, "cytoplasm": None},
        "MethodB": {"nuclear": nuc_b, "cytoplasm": None},
    }

    # Engine default fill_alpha is 0.0 -> interior pixels are completely transparent
    rgba, _colors = render_method_comparison_rgba(seg_results, compare_mode="Nuclear Masks")
    assert rgba[25, 25, 3] == pytest.approx(0.0)
    assert rgba[15, 25, 3] == pytest.approx(1.0)

    ctrl = ViewerController()
    try:
        dapi = np.ones((h, w), dtype=np.float32) * 500.0
        panck = np.ones((h, w), dtype=np.float32) * 300.0
        img_2ch = np.stack([dapi, panck], axis=0)
        ctrl.state.markers = ["DAPI", "PanCK"]
        ctrl.state.raw_image = img_2ch
        lyr_dapi = ctrl.viewer.add_image(dapi, name="DAPI", colormap="blue", visible=True)
        lyr_panck = ctrl.viewer.add_image(panck, name="PanCK", colormap="green", visible=True)

        # Simulate completed segmentation with filled Labels layers visible
        ctrl.state.segmentation_results = seg_results
        ctrl.state.set_active_segmentation_method("MethodA")
        lbl_a = ctrl.viewer.add_labels(nuc_a, name="MethodA Nuclear Mask", opacity=0.8, visible=True)
        lbl_b = ctrl.viewer.add_labels(nuc_b, name="MethodB Nuclear Mask", opacity=0.8, visible=False)
        lbl_a.contour = 0
        lbl_b.contour = 0

        dp = ctrl.disagreement_panel
        dp.update_available_methods(["MethodA", "MethodB"], active_method="MethodA")
        assert dp.chk_show_heatmap.isChecked() is False
        assert dp.chk_show_contours.isChecked() is True
        assert dp.chk_keep_filled_mask.isChecked() is False

        report: DisagreementReport = compute_model_disagreement(
            seg_results, compare_mode="Nuclear Masks", iou_threshold=0.75
        )
        ctrl.review_svc.render_disagreement_overlays(report)

        # Underlying fluorescence channels stay visible; solid filled Labels masks are hidden!
        assert lyr_dapi.visible is True
        assert lyr_panck.visible is True
        assert lbl_a.visible is False
        assert lbl_b.visible is False
        assert ctrl.viewer.layers["Disagreement Heatmap"].visible is False
        mc_lyr = ctrl.viewer.layers["Method Comparison Overlay"]
        assert mc_lyr.visible is True
        # Interior pixel of overlapped region has alpha=0.0 so fluorescence shines through
        assert mc_lyr.data[26, 26, 3] == pytest.approx(0.0)

        # Navigating to a hotspot renders hollow isolation contours & updates embedded dock preview
        ctrl.review_svc.next_issue()
        iso_lyr = ctrl.viewer.layers["Active Disagreement Isolation"]
        assert iso_lyr.visible is True
        assert iso_lyr.data[26, 26, 3] == pytest.approx(0.0)
        assert isinstance(report.items[0], DisagreementItem)
        assert len(dp.fig_mini_hotspot.axes) == 1
        assert "Hotspot #" in dp.fig_mini_hotspot.axes[0].get_title()

        # Opt-in to Keep Filled Mask keeps MethodA Nuclear Mask visible during QC
        dp.chk_keep_filled_mask.setChecked(True)
        qapp.processEvents()
        assert lbl_a.visible is True

        # Unchecking Show Contours (with Keep Filled Mask off) restores active filled mask
        dp.chk_keep_filled_mask.setChecked(False)
        dp.chk_show_contours.setChecked(False)
        qapp.processEvents()
        assert mc_lyr.visible is False
        assert lbl_a.visible is True
        assert lyr_dapi.visible is True
    finally:
        _close_controller(ctrl, qapp)
