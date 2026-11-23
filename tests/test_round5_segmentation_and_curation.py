"""Comprehensive verification tests for Round 5 Segmentation & Mask Curation:

1. ROI Preview layers registered in ViewerLegendWidget (`shapes` support, swatch -> edge_color,
   single-click selection -> `mode == 'select'`).
2. Pipeline Builder UI terminology (`Methods / Pipelines`, `+ New Pipeline`, `Run Active Pipeline`,
   `Run All Pipelines`, `Auto-Preview on parameter change` without `(400 ms)`), Completion
   Dilation Bar, editable model dropdowns, and Advanced Native Parameters toggle.
3. Native backend parameters + `extra_kwargs` JSON across Cellpose, StarDist, InstanSeg, CellSAM;
   CellSAM `(blank, nuclear, membrane)` composite ordering; StarDist default thresholds (`None`);
   and strict nuclear-anchored pairing invariants (`N_nuc >= N_cyto`, `N_nuc == N_cyto` on
   completion, 1-to-1 `ObjectNumber` matching in `extract_cell_features`).
4. Streamlined Curation UI, headless `curation.py` consensus adjudication + region splicing,
   and `ReviewService` / `ViewerController` `'Final Curated Mask'` integration.
5. Static hygiene: zero emojis, zero Qt stylesheet parse warnings, and zero top-level Qt/Napari
   imports in engine modules.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import re
import sys
import types

import matplotlib.colors as mcolors
import numpy as np
import pytest
from qtpy.QtCore import QItemSelectionModel, qInstallMessageHandler
from qtpy.QtWidgets import QAbstractItemView, QApplication

from iris_viewer.app.controller import ViewerController
from iris_viewer.engine.curation import (
    FINAL_CURATED_MASK_NAME,
    auto_adjudicate_consensus_mask,
    initialize_curated_mask,
    splice_method_into_regions,
)
from iris_viewer.engine.disagreement import DisagreementItem, compute_model_disagreement
from iris_viewer.engine.feature_extraction import extract_cell_features
from iris_viewer.engine.pipeline import (
    SegmentationSequence,
    SegmentationStep,
    iter_execute_sequence,
)
from iris_viewer.engine.segmentation.base import reconcile_nuclear_and_cytoplasm_masks
from iris_viewer.engine.segmentation.cellpose_backend import CellposeBackend
from iris_viewer.engine.segmentation.cellsam_backend import CellSAMBackend
from iris_viewer.engine.segmentation.instanseg_backend import (
    InstanSegBackend,
    _resolve_instanseg_model_and_target,
)
from iris_viewer.engine.segmentation.stardist_backend import StardistBackend
from iris_viewer.widgets.disagreement_panel import DisagreementPanelWidget
from iris_viewer.widgets.legend import ViewerLegendWidget
from iris_viewer.widgets.segmentation_pipeline_builder import PipelineBuilderWidget, StepWidget
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
# Item 1: ROI Preview Layers in ViewerLegendWidget
# ---------------------------------------------------------------------------


def test_round5_roi_preview_layers_registered_and_interactive_in_legend(qapp):
    """ROI preview layers appear in Legend, support swatch color -> edge_color, and select mode."""
    ctrl = ViewerController()
    try:
        img = np.zeros((2, 96, 96), dtype=np.uint16)
        img[0, 20:40, 20:40] = 1000
        img[1, 15:45, 15:45] = 800
        ctrl.state.raw_image = img
        ctrl.state.markers = ["DAPI", "Mem"]
        ctrl.viewer.add_image(img[0], name="DAPI", colormap="blue", visible=True)

        bbox = ctrl._create_or_focus_preview_roi()
        assert bbox is not None
        roi_name = "Segmentation Preview ROI"
        assert roi_name in ctrl.viewer.layers
        roi_layer = ctrl.viewer.layers[roi_name]

        seg_rows = {row.name: row for row in ctrl.legend_widget.seg_rows}
        assert roi_name in seg_rows
        roi_row = seg_rows[roi_name]
        assert roi_row._layer_type == "shapes"

        # Swatch color change updates Shapes edge_color & translucent face_color
        roi_row.update_swatch_display("#FF00AA")
        roi_row.colorChanged.emit(roi_name, "#FF00AA")
        expected_rgba = np.asarray(mcolors.to_rgba("#FF00AA"), dtype=float)
        edge_arr = np.asarray(roi_layer.edge_color, dtype=float)
        if edge_arr.ndim == 2:
            edge_arr = edge_arr[0]
        assert np.allclose(edge_arr[:3], expected_rgba[:3], atol=1e-2)
        face_arr = np.asarray(roi_layer.face_color, dtype=float)
        if face_arr.ndim == 2:
            face_arr = face_arr[0]
        assert np.allclose(face_arr[:3], expected_rgba[:3], atol=1e-2)
        assert face_arr[3] == pytest.approx(0.08, abs=1e-2)

        # Reset mode & visibility, then verify single-click triggers _select_layer -> mode='select'
        roi_layer.mode = "pan_zoom"
        roi_layer.visible = False
        roi_row.layerClicked.emit(roi_name)
        qapp.processEvents()
        assert roi_layer.visible is True
        assert roi_layer.mode == "select"
        assert roi_layer in ctrl.viewer.layers.selection

        # ROI Preview Nuclear & Cytoplasm completion registers both rows in Legend
        h = max(1, bbox[2] - bbox[0])
        w = max(1, bbox[3] - bbox[1])
        nuc_mask = np.zeros((h, w), dtype=np.uint32)
        cyto_mask = np.zeros((h, w), dtype=np.uint32)
        nuc_mask[2:6, 2:6] = 1
        cyto_mask[1:8, 1:8] = 1
        ctrl._on_roi_preview_complete("Pipeline 1", bbox, {"nuclear": nuc_mask, "cytoplasm": cyto_mask})

        seg_row_names = {r.name for r in ctrl.legend_widget.seg_rows}
        assert "ROI Preview - Nuclear" in seg_row_names
        assert "ROI Preview - Cytoplasm" in seg_row_names
    finally:
        _close_controller(ctrl, qapp)


# ---------------------------------------------------------------------------
# Items 2 & 4: Pipeline Builder Terminology & Completion Bar & Advanced UI
# ---------------------------------------------------------------------------


def test_round5_pipeline_builder_terminology_and_controls(qapp):
    """Verify updated Pipeline Builder labels, completion bar, editable combos & advanced UI."""
    pb = PipelineBuilderWidget()
    try:
        pb.show()
        qapp.processEvents()

        labels_text = [lbl.text() for lbl in pb.findChildren(type(pb.lbl_disagreement_status))]
        assert "Methods / Pipelines" in labels_text
        assert pb.btn_add_group.text() == "+ New Pipeline"
        assert pb.btn_run_active.text() == "Run Active Pipeline"
        assert pb.btn_run.text() == "Run All Pipelines"
        assert pb.chk_auto_preview.text() == "Auto-Preview on parameter change"
        assert "(400 ms)" not in pb.chk_auto_preview.text()

        group = pb.tab_widget.widget(0)
        assert group.chk_auto_complete_cyto.text() == "Auto-dilate unmatched nuclei"
        assert group.chk_auto_complete_cyto.isChecked() is True
        assert group.spin_fallback_dilate.value() == 5
        assert [
            group.combo_fallback_mode.itemText(i)
            for i in range(group.combo_fallback_mode.count())
        ] == ["distance", "watershed"]
        assert group.combo_fallback_mem.isVisible() is False

        group.spin_fallback_dilate.setValue(12)
        group.combo_fallback_mode.setCurrentText("watershed")
        qapp.processEvents()
        assert group.combo_fallback_mem.isVisible() is True
        seq = pb.get_active_sequence()
        assert seq is not None
        assert seq.auto_complete_cytoplasm is True
        assert seq.fallback_dilation_radius == 12
        assert seq.fallback_dilation_mode == "watershed"

        group.chk_auto_complete_cyto.setChecked(False)
        assert group.spin_fallback_dilate.isEnabled() is False
        assert group.combo_fallback_mode.isEnabled() is False
        assert group.combo_fallback_mem.isEnabled() is False
        seq2 = pb.get_all_sequences()[pb.tab_widget.tabText(0)]
        assert seq2.auto_complete_cytoplasm is False

        sw: StepWidget = group.step_widgets[0]
        assert sw.combo_cp_model.isEditable() is True
        assert sw.combo_sd_model.isEditable() is True
        assert sw.combo_is_model.isEditable() is True
        assert [sw.combo_type.itemText(i) for i in range(sw.combo_type.count())] == [
            "Nuclear (Base)",
            "Cytoplasm (Sequential)",
        ]
        cp_items = [sw.combo_cp_model.itemText(i) for i in range(sw.combo_cp_model.count())]
        for expected_model in ("cpsam", "cpsam_v2", "cpdino", "cpdino-vitb", "cyto3", "nuclei"):
            assert expected_model in cp_items

        # More Native Parameters collapsed by default, toggles and stays synced across both modes
        assert sw.btn_toggle_advanced.text() == "More Native Parameters"
        assert sw.advanced_container.isVisible() is False
        sw.btn_toggle_advanced.setChecked(True)
        qapp.processEvents()
        assert sw.advanced_container.isVisible() is True

        sw.combo_type.setCurrentText("Cytoplasm (Sequential)")
        qapp.processEvents()
        assert sw.btn_toggle_advanced.isHidden() is False
        assert sw.advanced_container.isVisible() is True
        sw.btn_toggle_advanced.setChecked(False)
        qapp.processEvents()
        assert sw.advanced_container.isVisible() is False
    finally:
        pb.close()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# Item 3a: All Native Parameters & Extra Kwargs Across All 4 Backends
# ---------------------------------------------------------------------------


def test_round5_stepwidget_serializes_native_params_and_extra_kwargs(qapp):
    """StepWidget serializes all native parameters + extra_kwargs JSON for all 4 methods."""
    markers = ["DAPI", "PanCK", "CD45"]
    sw = StepWidget(["Cellpose", "StarDist", "InstanSeg", "CellSAM"], markers)
    try:
        # Cellpose
        sw.combo_type.setCurrentText("Cytoplasm (Sequential)")
        sw.combo_method.setCurrentText("Cellpose")
        sw.combo_cp_model.setEditText("cpsam_v2")
        sw.spin_cp_diam.setValue(24)
        sw.spin_cp_flow.setValue(0.65)
        sw.spin_cp_cellprob.setValue(-1.2)
        sw.spin_cp_tile_overlap.setValue(0.25)
        sw.chk_cp_resample.setChecked(False)
        sw.chk_cp_augment.setChecked(True)
        sw.chk_cp_normalize.setChecked(False)
        sw.spin_cp_niter.setValue(300)
        sw.spin_cp_max_size_frac.setValue(0.6)
        sw.spin_cp_batch_size.setValue(16)
        sw.spin_cp_bsize.setValue(256)
        sw.spin_cp_stitch.setValue(0.35)
        sw.spin_max_cell_area.setValue(5000)
        sw.spin_gaussian_sigma.setValue(1.5)
        sw.spin_norm_pmin.setValue(2.0)
        sw.spin_norm_pmax.setValue(99.5)
        sw.spin_expand_radius.setValue(4)
        sw.combo_expansion_mode.setCurrentText("watershed")
        sw.txt_extra_kwargs.setText('{"anisotropy": 1.25}')

        cp_step = sw.get_step_config(markers)
        assert cp_step.rule == "Cytoplasm (Sequential)"
        assert cp_step.params["model_type"] == "cpsam_v2"
        assert cp_step.params["diameter"] == 24
        assert cp_step.params["resample"] is False
        assert cp_step.params["augment"] is True
        assert cp_step.params["cellpose_normalize"] is False
        assert cp_step.params["niter"] == 300
        assert cp_step.params["max_size_fraction"] == pytest.approx(0.6)
        assert cp_step.params["batch_size"] == 16
        assert cp_step.params["bsize"] == 256
        assert cp_step.params["stitch_threshold"] == pytest.approx(0.35)
        assert cp_step.params["extra_kwargs"] == {"anisotropy": 1.25}

        # StarDist with use_model_default_thresholds=True -> None
        sw.combo_method.setCurrentText("StarDist")
        sw.combo_sd_model.setEditText("/tmp/custom_stardist")
        sw.chk_sd_model_defaults.setChecked(True)
        sw.spin_sd_scale.setValue(1.5)
        sw.chk_sd_sparse.setChecked(False)
        sw.spin_sd_n_tiles.setValue(4)
        sd_step = sw.get_step_config(markers)
        assert sd_step.params["model_name"] == "/tmp/custom_stardist"
        assert sd_step.params["use_model_default_thresholds"] is True
        assert sd_step.params["prob_thresh"] is None
        assert sd_step.params["nms_thresh"] is None
        assert sd_step.params["scale"] == pytest.approx(1.5)
        assert sd_step.params["sparse"] is False
        assert sd_step.params["n_tiles_grid"] == 4

        # InstanSeg
        sw.combo_method.setCurrentText("InstanSeg")
        sw.combo_is_target.setCurrentText("all_outputs")
        sw.chk_is_normalise.setChecked(False)
        sw.chk_is_rescale_output.setChecked(False)
        sw.spin_is_tile_size.setValue(640)
        sw.spin_is_batch_size.setValue(2)
        sw.spin_is_seed_thresh.setValue(0.45)
        sw.spin_is_mask_thresh.setValue(0.55)
        sw.spin_is_peak_dist.setValue(5)
        sw.spin_is_overlap_thresh.setValue(0.30)
        sw.spin_is_window_size.setValue(128)
        is_step = sw.get_step_config(markers)
        assert is_step.params["target"] == "all_outputs"
        assert is_step.params["normalise"] is False
        assert is_step.params["rescale_output"] is False
        assert is_step.params["tile_size"] == 640
        assert is_step.params["batch_size"] == 2
        assert is_step.params["seed_threshold"] == pytest.approx(0.45)
        assert is_step.params["mask_threshold"] == pytest.approx(0.55)
        assert is_step.params["peak_distance"] == 5
        assert is_step.params["overlap_threshold"] == pytest.approx(0.30)
        assert is_step.params["window_size"] == 128

        # CellSAM + invalid extra_kwargs fallback
        sw.combo_method.setCurrentText("CellSAM")
        sw.txt_cs_model_path.setText("/tmp/cellsam.pt")
        sw.combo_cs_use_wsi.setCurrentText("true")
        sw.chk_cs_low_contrast.setChecked(True)
        sw.spin_cs_block_size.setValue(512)
        sw.spin_cs_overlap.setValue(64)
        sw.spin_cs_iou_depth.setValue(64)
        sw.spin_cs_iou_thresh.setValue(0.60)
        sw.chk_cs_gauge_size.setChecked(True)
        sw.chk_cs_postprocess.setChecked(True)
        sw.chk_cs_remove_bounds.setChecked(True)
        sw.chk_cs_fast.setChecked(True)
        sw.chk_cs_all_channels.setChecked(True)
        sw.txt_extra_kwargs.setText("{not-json")
        cs_step = sw.get_step_config(markers)
        assert cs_step.params["model_path"] == "/tmp/cellsam.pt"
        assert cs_step.params["use_wsi"] == "true"
        assert cs_step.params["low_contrast_enhancement"] is True
        assert cs_step.params["block_size"] == 512
        assert cs_step.params["overlap"] == 64
        assert cs_step.params["iou_depth"] == 64
        assert cs_step.params["iou_threshold"] == pytest.approx(0.60)
        assert cs_step.params["gauge_cell_size"] is True
        assert cs_step.params["postprocess"] is True
        assert cs_step.params["remove_boundaries"] is True
        assert cs_step.params["fast"] is True
        assert cs_step.params["use_all_channels"] is True
        assert cs_step.params["extra_kwargs"] == {}
    finally:
        sw.close()


def test_round5_backends_forward_native_params_and_cellsam_channel_order(monkeypatch, tmp_path):
    """Verify parameter forwarding, signature filtering, & CellSAM (blank, nuc, mem) order."""
    img = np.zeros((2, 64, 64), dtype=np.float32)
    img[0, 10:25, 10:25] = 100.0
    img[1, 8:28, 8:28] = 200.0

    # --- 1. CellposeBackend ---
    cp_calls = []
    cp_ckpt = tmp_path / "custom_cp.pt"
    cp_ckpt.write_text("weights", encoding="utf-8")

    class FakeCellposeModel:
        MODEL_NAMES = ["cpsam", "cpsam_v2", "cpdino", "cyto3"]

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs

        def eval(
            self,
            eval_img,
            diameter=None,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            batch_size=8,
            tile_overlap=0.1,
            resample=True,
            augment=False,
            niter=None,
            max_size_fraction=0.4,
            bsize=None,
            stitch_threshold=0.0,
            normalize=True,
            channel_axis=None,
        ):
            # Strict signature (no **kwargs) verifies unsupported extra_kwargs are filtered!
            cp_calls.append(
                {
                    "shape": eval_img.shape,
                    "diameter": diameter,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold,
                    "batch_size": batch_size,
                    "tile_overlap": tile_overlap,
                    "resample": resample,
                    "augment": augment,
                    "niter": niter,
                    "max_size_fraction": max_size_fraction,
                    "bsize": bsize,
                    "stitch_threshold": stitch_threshold,
                    "normalize": normalize,
                    "channel_axis": channel_axis,
                }
            )
            m = np.zeros((64, 64), dtype=np.uint32)
            if eval_img.ndim == 2:
                m[10:25, 10:25] = 1
            else:
                m[8:28, 8:28] = 5  # whole-cell gets reconciled to nuclear ID 1
            return m, None, None

    fake_cp_models = types.ModuleType("cellpose.models")
    fake_cp_models.CellposeModel = FakeCellposeModel
    fake_cp_models.MODEL_NAMES = FakeCellposeModel.MODEL_NAMES
    fake_cp_models.MODEL_DIR = str(tmp_path)
    fake_cp = types.ModuleType("cellpose")
    fake_cp.models = fake_cp_models
    monkeypatch.setitem(sys.modules, "cellpose", fake_cp)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_cp_models)

    cp = CellposeBackend()
    cp_res = cp.segment(
        image=img,
        nuclear_channel=0,
        membrane_channel=1,
        diameter=22.0,
        device="cpu",
        model_type=str(cp_ckpt),
        resample=False,
        augment=True,
        niter=250,
        max_size_fraction=0.55,
        batch_size=4,
        bsize=128,
        stitch_threshold=0.2,
        cellpose_normalize=False,
        extra_kwargs=json.dumps({"unsupported_kwarg": 999, "flow_threshold": 0.77}),
    )
    assert len(cp_calls) == 2
    assert cp_calls[0]["resample"] is False
    assert cp_calls[0]["augment"] is True
    assert cp_calls[0]["niter"] == 250
    assert cp_calls[0]["bsize"] == 128
    assert cp_calls[0]["normalize"] is False
    assert cp_calls[0]["flow_threshold"] == pytest.approx(0.77)
    assert cp._cached_model_cyto.init_kwargs["pretrained_model"] == str(cp_ckpt)
    assert set(np.unique(cp_res["nuclear"]).tolist()) == {0, 1}
    assert set(np.unique(cp_res["cytoplasm"]).tolist()) == {0, 1}

    # --- 2. StardistBackend ---
    sd_calls = []
    sd_dir = tmp_path / "custom_sd_model"
    sd_dir.mkdir()

    class FakeStarDist2D:
        def __init__(self, config, name=None, basedir=None):
            self.init_args = (config, name, basedir)

        @classmethod
        def from_pretrained(cls, name):
            return cls(None, name=name, basedir=None)

        def predict_instances(
            self, img_in, prob_thresh=0.5, nms_thresh=0.4, n_tiles=None, sparse=True, scale=None
        ):
            sd_calls.append(
                {
                    "prob_thresh": prob_thresh,
                    "nms_thresh": nms_thresh,
                    "n_tiles": n_tiles,
                    "sparse": sparse,
                    "scale": scale,
                }
            )
            m = np.zeros(img_in.shape, dtype=np.uint32)
            m[10:25, 10:25] = 1
            return m, {}

    fake_sd_models = types.ModuleType("stardist.models")
    fake_sd_models.StarDist2D = FakeStarDist2D
    fake_sd = types.ModuleType("stardist")
    fake_sd.models = fake_sd_models
    monkeypatch.setitem(sys.modules, "stardist", fake_sd)
    monkeypatch.setitem(sys.modules, "stardist.models", fake_sd_models)

    sd = StardistBackend()
    sd.segment(
        image=img,
        nuclear_channel=0,
        membrane_channel=None,
        device="cpu",
        model_name=str(sd_dir),
        use_model_default_thresholds=True,
        prob_thresh=0.8,
        nms_thresh=0.2,
        scale=1.25,
        sparse=False,
        n_tiles_grid=3,
        extra_kwargs={"ignored_extra": 123},
    )
    assert sd._cached_model.init_args == (None, "custom_sd_model", str(tmp_path))
    assert sd_calls[0]["prob_thresh"] is None
    assert sd_calls[0]["nms_thresh"] is None
    assert sd_calls[0]["n_tiles"] == (3, 3)
    assert sd_calls[0]["sparse"] is False
    assert sd_calls[0]["scale"] == pytest.approx(1.25)

    # --- 3. InstanSegBackend ---
    assert _resolve_instanseg_model_and_target("fluorescence_nuclei", "auto") == (
        "fluorescence_nuclei_and_cells",
        "nuclei",
    )
    assert _resolve_instanseg_model_and_target("fluorescence_cells", "") == (
        "fluorescence_nuclei_and_cells",
        "cells",
    )

    import torch

    is_calls = []

    class FakeInstanSeg:
        def __init__(self, model_name, device="cpu", verbosity=0):
            self.model_name = model_name

        def eval_medium_image(self, inp, **kw):
            is_calls.append(kw)
            out = torch.zeros((1, 2, 64, 64), dtype=torch.int32)
            out[0, 0, 10:25, 10:25] = 1
            out[0, 1, 8:28, 8:28] = 4
            return out, None

    fake_is = types.ModuleType("instanseg")
    fake_is.InstanSeg = FakeInstanSeg
    monkeypatch.setitem(sys.modules, "instanseg", fake_is)

    inst = InstanSegBackend()
    is_res = inst.segment(
        image=img,
        nuclear_channel=0,
        membrane_channel=1,
        device="cpu",
        target="all_outputs",
        normalise=False,
        rescale_output=False,
        tile_size=384,
        batch_size=2,
        seed_threshold=0.55,
        mask_threshold=0.65,
        peak_distance=4,
        overlap_threshold=0.25,
        window_size=96,
        extra_kwargs={"custom_flag": True},
    )
    assert is_calls[0]["normalise"] is False
    assert is_calls[0]["rescale_output"] is False
    assert is_calls[0]["tile_size"] == 384
    assert is_calls[0]["batch_size"] == 2
    assert is_calls[0]["seed_threshold"] == pytest.approx(0.55)
    assert is_calls[0]["mask_threshold"] == pytest.approx(0.65)
    assert is_calls[0]["peak_distance"] == 4
    assert is_calls[0]["overlap_threshold"] == pytest.approx(0.25)
    assert is_calls[0]["window_size"] == 96
    assert is_calls[0]["custom_flag"] is True
    assert set(np.unique(is_res["cytoplasm"]).tolist()) == {0, 1}

    # --- 4. CellSAMBackend: (blank, nuclear, whole-cell) channel order & local model + WSI ---
    cs_passes = []
    cs_local_calls = []
    cs_wsi_calls = []
    cs_ckpt = tmp_path / "cellsam_local.pt"
    cs_ckpt.write_text("ckpt", encoding="utf-8")

    class FakeSamModel:
        def to(self, _dev):
            return self

    def fake_seg_cellular(
        comp,
        model,
        bbox_threshold=0.4,
        normalize=True,
        postprocess=False,
        remove_boundaries=False,
        fast=False,
        device="cpu",
    ):
        cs_passes.append(
            {
                "comp": comp.copy(),
                "bbox_threshold": bbox_threshold,
                "normalize": normalize,
                "postprocess": postprocess,
                "remove_boundaries": remove_boundaries,
                "fast": fast,
            }
        )
        m = np.zeros(comp.shape[:2], dtype=np.uint32)
        if np.max(comp[..., 2]) == 0:
            m[10:25, 10:25] = 1
        else:
            m[8:28, 8:28] = 9
        return m, None, None

    def fake_segment_wsi(comp, block_size, overlap, iou_depth, iou_threshold, **kw):
        cs_wsi_calls.append(
            {
                "comp": comp.copy(),
                "block_size": block_size,
                "overlap": overlap,
                "iou_depth": iou_depth,
                "iou_threshold": iou_threshold,
                "kw": dict(kw),
            }
        )
        m = np.zeros(comp.shape[:2], dtype=np.uint32)
        if np.max(comp[..., 2]) == 0:
            m[10:25, 10:25] = 1
        else:
            m[8:28, 8:28] = 7
        return m

    fake_cs = types.ModuleType("cellSAM")
    fake_cs.get_model = lambda: FakeSamModel()
    fake_cs.get_local_model = lambda p: (cs_local_calls.append(p) or FakeSamModel())
    fake_cs.segment_cellular_image = fake_seg_cellular
    fake_cs_wsi = types.ModuleType("cellSAM.wsi")
    fake_cs_wsi.segment_wsi = fake_segment_wsi
    monkeypatch.setitem(sys.modules, "cellSAM", fake_cs)
    monkeypatch.setitem(sys.modules, "cellSAM.wsi", fake_cs_wsi)

    cs = CellSAMBackend()
    cs_res = cs.segment(
        image=img,
        nuclear_channel=0,
        membrane_channel=1,
        device="cpu",
        model_path=str(cs_ckpt),
        use_wsi="false",
        normalize=False,
        postprocess=True,
        remove_boundaries=True,
        fast=True,
        bbox_threshold=0.55,
    )
    assert cs_local_calls == [str(cs_ckpt)]
    assert len(cs_passes) == 2
    # Channel 0 is ALWAYS blank, Channel 1 is nuclear, Channel 2 is whole-cell
    nuc_pass_comp = cs_passes[0]["comp"]
    assert int(np.max(nuc_pass_comp[..., 0])) == 0
    assert int(np.max(nuc_pass_comp[..., 1])) > 0
    assert int(np.max(nuc_pass_comp[..., 2])) == 0

    wc_pass_comp = cs_passes[1]["comp"]
    assert int(np.max(wc_pass_comp[..., 0])) == 0
    assert int(np.max(wc_pass_comp[..., 1])) > 0
    assert int(np.max(wc_pass_comp[..., 2])) > 0
    assert cs_passes[0]["postprocess"] is True
    assert cs_passes[0]["remove_boundaries"] is True
    assert cs_passes[0]["fast"] is True
    assert set(np.unique(cs_res["nuclear"]).tolist()) == {0, 1}
    assert set(np.unique(cs_res["cytoplasm"]).tolist()) == {0, 1}

    # Verify WSI path forwards block_size, overlap, iou_depth, iou_threshold
    cs.segment(
        image=img,
        nuclear_channel=0,
        membrane_channel=1,
        device="cpu",
        use_wsi="true",
        block_size=256,
        overlap=32,
        iou_depth=48,
        iou_threshold=0.65,
    )
    assert len(cs_wsi_calls) == 2
    assert cs_wsi_calls[0]["block_size"] == 256
    assert cs_wsi_calls[0]["overlap"] == 32
    assert cs_wsi_calls[0]["iou_depth"] == 32  # clamped to <= overlap
    assert cs_wsi_calls[0]["iou_threshold"] == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# Item 3b: Strict Nuclear-Anchored Pairing & Feature Extraction
# ---------------------------------------------------------------------------


def test_round5_reconcile_nuclear_anchored_pairing_invariants():
    """Verify anucleate debris discarded, binucleate split, & neighbor nuclei protected."""
    nuc = np.zeros((80, 80), dtype=np.uint32)
    cyto = np.zeros((80, 80), dtype=np.uint32)

    # Nuclei 1 and 2 enclosed in one merged binucleate cytoplasm blob (ID 10)
    nuc[10:20, 10:20] = 1
    nuc[10:20, 30:40] = 2
    cyto[5:25, 5:45] = 10

    # Unmatched nucleus 3 adjacent to cell 4's sprawling cytoplasm blob (ID 20)
    nuc[50:60, 10:20] = 3
    nuc[50:60, 22:32] = 4
    cyto[45:65, 18:40] = 20  # overlaps part of nucleus 3 (2 px wide) and all of nucleus 4 (100 px)

    # Anucleate debris blob (ID 99) with no nuclear overlap
    cyto[65:75, 60:75] = 99

    out_nuc, out_cyto = reconcile_nuclear_and_cytoplasm_masks(nuc, cyto)
    assert out_nuc.dtype == np.uint32
    assert out_cyto is not None and out_cyto.dtype == np.uint32

    nuc_ids = set(np.unique(out_nuc).tolist()) - {0}
    cyto_ids = set(np.unique(out_cyto).tolist()) - {0}
    assert nuc_ids == {1, 2, 3, 4}
    # Debris discarded; binucleate blob 10 split into 1 and 2; blob 20 assigned to 4 only
    assert cyto_ids == {1, 2, 4}
    assert len(nuc_ids) >= len(cyto_ids)
    assert cyto_ids <= nuc_ids
    assert int(np.max(out_cyto[65:75, 60:75])) == 0
    # Neighbor unmatched nucleus 3 is protected (not overwritten by cytoplasm 4)
    assert np.all(out_cyto[nuc == 3] == 0)
    # Matched nuclei belong to their own IDs in cytoplasm
    assert np.all(out_cyto[nuc == 1] == 1)
    assert np.all(out_cyto[nuc == 2] == 2)
    assert np.all(out_cyto[nuc == 4] == 4)


def test_round5_pipeline_completion_and_feature_extraction_matching():
    """Pipeline enforces N_nuc >= N_cyto at each step, N_nuc == N_cyto on completion,
    and extract_cell_features preserves all nucleated ObjectNumbers when N_nuc > N_cyto."""
    img = np.zeros((2, 64, 64), dtype=np.float32)
    img[0, 8:16, 8:16] = 500.0
    img[0, 36:44, 36:44] = 500.0
    img[1, 5:20, 5:20] = 300.0

    class MockNucBackend:
        def segment(self, image, **_kw):
            m = np.zeros((64, 64), dtype=np.uint32)
            m[8:16, 8:16] = 1
            m[36:44, 36:44] = 2
            return {"nuclear": m, "cytoplasm": None}

    class MockCytoBackend:
        def segment(self, image, **_kw):
            c = np.zeros((64, 64), dtype=np.uint32)
            c[5:20, 5:20] = 77      # covers nucleus 1
            c[45:55, 5:15] = 88     # anucleate debris -> discarded
            return {"nuclear": None, "cytoplasm": c}

    backends = {"MockNuc": MockNucBackend(), "MockCyto": MockCytoBackend()}

    # 1. Partial cytoplasm (auto_complete_cytoplasm=False): N_nuc=2, N_cyto=1 -> no dropped cells!
    seq_partial = SegmentationSequence(
        steps=[
            SegmentationStep("MockNuc", {"nuclear_channel": 0}, rule="Nuclear (Base)"),
            SegmentationStep(
                "MockCyto", {"nuclear_channel": 0, "membrane_channels": [1]}, rule="Cytoplasm (Sequential)"
            ),
        ],
        auto_complete_cytoplasm=False,
    )
    events_partial = list(iter_execute_sequence("PartialPipe", seq_partial, img, backends, "cpu"))
    final_partial = [e[2] for e in events_partial if e[0] == "group_done"][0]
    p_nuc = final_partial["nuclear"]
    p_cyto = final_partial["cytoplasm"]
    assert set(np.unique(p_nuc).tolist()) - {0} == {1, 2}
    assert set(np.unique(p_cyto).tolist()) - {0} == {1}

    df_partial = extract_cell_features(
        mask=p_nuc,
        raw_image=img,
        markers=["DAPI", "Mem"],
        cytoplasm_mask=p_cyto,
    )
    assert sorted(df_partial["ObjectNumber"].astype(int).tolist()) == [1, 2]
    row1 = df_partial[df_partial["ObjectNumber"] == 1].iloc[0]
    row2 = df_partial[df_partial["ObjectNumber"] == 2].iloc[0]
    assert row1["cyto_area"] > 0
    assert row2["cyto_area"] == pytest.approx(0.0)

    # 2. Completed cytoplasm (auto_complete_cytoplasm=True): N_nuc == N_cyto == 2
    seq_complete = SegmentationSequence(
        steps=seq_partial.steps,
        auto_complete_cytoplasm=True,
        fallback_dilation_radius=4,
        fallback_dilation_mode="distance",
    )
    events_complete = list(
        iter_execute_sequence("CompletePipe", seq_complete, img, backends, "cpu")
    )
    final_complete = [e[2] for e in events_complete if e[0] == "group_done"][0]
    c_nuc = final_complete["nuclear"]
    c_cyto = final_complete["cytoplasm"]
    assert (set(np.unique(c_nuc).tolist()) - {0}) == {1, 2}
    assert (set(np.unique(c_cyto).tolist()) - {0}) == {1, 2}


# ---------------------------------------------------------------------------
# Item 5: Headless Curation Engine & Streamlined UI / ReviewService Integration
# ---------------------------------------------------------------------------


def test_round5_curation_consensus_and_region_splicing():
    """3-model consensus fixes Merge error and splicing supports hotspots, batch & polygon ROI."""
    h, w = 80, 80
    # ModelA has a Merge error in top-left (cells 1 & 2 merged into label 1) plus agreed cell 3
    nuc_a = np.zeros((h, w), dtype=np.uint32)
    cyto_a = np.zeros((h, w), dtype=np.uint32)
    nuc_a[10:22, 10:36] = 1
    cyto_a[8:24, 8:38] = 1
    nuc_a[50:62, 50:62] = 2
    cyto_a[48:64, 48:64] = 2

    # ModelB and ModelC both split top-left into two separate touching cells (labels 1 & 2) + cell 3
    nuc_b = np.zeros((h, w), dtype=np.uint32)
    cyto_b = np.zeros((h, w), dtype=np.uint32)
    nuc_b[10:22, 10:22] = 1
    nuc_b[10:22, 23:36] = 2
    cyto_b[8:24, 8:22] = 1
    cyto_b[8:24, 23:38] = 2
    nuc_b[50:62, 50:62] = 3
    cyto_b[48:64, 48:64] = 3

    nuc_c = np.zeros((h, w), dtype=np.uint32)
    cyto_c = np.zeros((h, w), dtype=np.uint32)
    nuc_c[10:22, 10:22] = 1
    nuc_c[10:22, 23:36] = 2
    cyto_c[8:24, 8:22] = 1
    cyto_c[8:24, 23:38] = 2
    nuc_c[50:62, 50:62] = 3
    cyto_c[48:64, 48:64] = 3

    seg_results = {
        "ModelA": {"nuclear": nuc_a, "cytoplasm": cyto_a},
        "ModelB": {"nuclear": nuc_b, "cytoplasm": cyto_b},
        "ModelC": {"nuclear": nuc_c, "cytoplasm": cyto_c},
    }
    report = compute_model_disagreement(
        seg_results,
        compare_mode="Nuclear Masks",
        reference_method="ModelA",
        iou_threshold=0.75,
    )
    consensus = auto_adjudicate_consensus_mask(seg_results, report)
    assert consensus.curated_regions_count >= 1
    # Merge resolved into 2 separate cells + preserved cell 3 -> 3 total cells, N_nuc == N_cyto
    assert set(np.unique(consensus.nuclear).tolist()) - {0} == {1, 2, 3}
    assert set(np.unique(consensus.cytoplasm).tolist()) - {0} == {1, 2, 3}

    # Test initialize + splice_method_into_regions with DisagreementItem & polygon_yx
    base_state = initialize_curated_mask(seg_results, "ModelA")
    assert set(np.unique(base_state.nuclear).tolist()) - {0} == {1, 2}

    item = DisagreementItem(
        region_id=1,
        category="Merge",
        severity=0.6,
        best_iou=0.4,
        bbox=(8, 8, 25, 40),
        centroid_yx=(16.0, 22.0),
        method_cell_ids={"ModelA": [1], "ModelB": [1, 2], "ModelC": [1, 2]},
        dice=0.5,
        overlap_coeff=0.6,
    )
    spliced_item = splice_method_into_regions(
        base_state, seg_results, "ModelB", items=[item]
    )
    assert set(np.unique(spliced_item.nuclear).tolist()) - {0} == {1, 2, 3}
    assert set(np.unique(spliced_item.cytoplasm).tolist()) - {0} == {1, 2, 3}

    # Test polygon ROI splicing from ModelC onto a ModelA base where ModelC has no cytoplasm
    seg_no_cyto = {
        "ModelA": {"nuclear": nuc_a, "cytoplasm": cyto_a},
        "ModelC": {"nuclear": nuc_c, "cytoplasm": None},
    }
    state_a = initialize_curated_mask(seg_no_cyto, "ModelA")
    poly = np.array([[5, 5], [5, 40], [26, 40], [26, 5]], dtype=float)
    spliced_poly = splice_method_into_regions(
        state_a, seg_no_cyto, "ModelC", polygon_yx=poly
    )
    # Spliced cells get missing cytoplasm filled from nuclear footprint -> N_nuc == N_cyto
    assert set(np.unique(spliced_poly.nuclear).tolist()) - {0} == {1, 2, 3}
    assert set(np.unique(spliced_poly.cytoplasm).tolist()) - {0} == {1, 2, 3}


def test_round5_end_to_end_final_curated_mask_in_review_service(qapp):
    """Streamlined UI + ReviewService ROI/hotspot/consensus curation updates Final Curated Mask."""
    ctrl = ViewerController()
    try:
        h, w = 80, 80
        img = np.zeros((2, h, w), dtype=np.float32)
        img[0, 10:22, 10:20] = 800.0
        img[0, 10:22, 26:36] = 800.0
        img[0, 50:62, 50:62] = 800.0
        ctrl.state.raw_image = img
        ctrl.state.markers = ["DAPI", "Mem"]
        ctrl.state.current_sample_name = "Sample1"

        nuc_a = np.zeros((h, w), dtype=np.uint32)
        cyto_a = np.zeros((h, w), dtype=np.uint32)
        nuc_a[10:22, 10:36] = 1
        cyto_a[8:24, 8:38] = 1
        nuc_a[50:62, 50:62] = 2
        cyto_a[48:64, 48:64] = 2
        # Add a second disputed cell (Model-Exclusive in ModelB) so multi-row selection has >=2 rows
        nuc_b = np.zeros((h, w), dtype=np.uint32)
        cyto_b = np.zeros((h, w), dtype=np.uint32)
        nuc_b[10:22, 10:22] = 1
        nuc_b[10:22, 23:36] = 2
        cyto_b[8:24, 8:22] = 1
        cyto_b[8:24, 23:38] = 2
        nuc_b[50:62, 50:62] = 3
        cyto_b[48:64, 48:64] = 3
        nuc_b[50:62, 10:22] = 4
        cyto_b[48:64, 8:24] = 4

        nuc_c = nuc_b.copy()
        cyto_c = cyto_b.copy()

        ctrl.state.segmentation_results = {
            "ModelA": {"nuclear": nuc_a, "cytoplasm": cyto_a},
            "ModelB": {"nuclear": nuc_b, "cytoplasm": cyto_b},
            "ModelC": {"nuclear": nuc_c, "cytoplasm": cyto_c},
        }
        ctrl.disagreement_panel.update_available_methods(
            ["ModelA", "ModelB", "ModelC"], active_method="ModelA"
        )
        ctrl.state.disagreement_report = compute_model_disagreement(
            ctrl.state.segmentation_results,
            compare_mode="Nuclear Masks",
            reference_method="ModelA",
            iou_threshold=0.75,
        )
        ctrl.review_svc.render_disagreement_overlays(ctrl.state.disagreement_report)

        studio = SegmentationStudioDialog(
            ctx=ctrl._ctx,
            disagreement_panel=ctrl.disagreement_panel,
            feature_panel=ctrl.feature_panel,
        )
        try:
            # Verify confusing controls are removed & ExtendedSelection enabled
            assert not hasattr(ctrl.disagreement_panel, "btn_flicker")
            assert not hasattr(studio, "btn_flicker")
            assert not hasattr(studio, "combo_flicker_a")
            assert not hasattr(studio, "combo_flicker_b")
            assert studio.table_issues.selectionMode() == QAbstractItemView.ExtendedSelection
            assert FINAL_CURATED_MASK_NAME not in [
                studio.combo_ref_method.itemText(i)
                for i in range(studio.combo_ref_method.count())
            ]

            # 1. Auto-adjudicate via consensus
            curated = ctrl.review_svc.auto_adjudicate_final_mask()
            assert curated is not None
            assert curated.curated_regions_count >= 1
            assert FINAL_CURATED_MASK_NAME in ctrl.state.segmentation_results
            assert ctrl.state.active_segmentation_method == FINAL_CURATED_MASK_NAME
            assert f"{FINAL_CURATED_MASK_NAME} Nuclear Mask" in ctrl.viewer.layers
            assert f"{FINAL_CURATED_MASK_NAME} Cyto Mask" in ctrl.viewer.layers
            assert (
                ctrl.viewer.layers[f"{FINAL_CURATED_MASK_NAME} Nuclear Mask"].opacity
                == pytest.approx(0.8)
            )
            assert (
                ctrl.viewer.layers[f"{FINAL_CURATED_MASK_NAME} Cyto Mask"].opacity
                == pytest.approx(0.5)
            )
            assert ctrl.feature_panel.get_selected_mask_source() == FINAL_CURATED_MASK_NAME

            # 2. Multi-row hotspot acceptance in Studio (ExtendedSelection)
            ctrl.review_svc.use_pipeline_as_final_mask("ModelA")
            studio.refresh_data()
            assert studio.table_issues.rowCount() >= 2
            sel_model = studio.table_issues.selectionModel()
            sel_model.clearSelection()
            for r_idx in range(min(2, studio.table_issues.rowCount())):
                sel_model.select(
                    studio.table_issues.model().index(r_idx, 0),
                    QItemSelectionModel.Select | QItemSelectionModel.Rows,
                )
            studio.combo_override_method.setCurrentText("ModelB")
            studio.btn_accept_selected_rows.click()
            assert set(
                np.unique(
                    ctrl.state.segmentation_results[FINAL_CURATED_MASK_NAME]["nuclear"]
                ).tolist()
            ) - {0} == {1, 2, 3, 4}

            # 3. Category-filtered batch acceptance in Studio
            ctrl.review_svc.use_pipeline_as_final_mask("ModelA")
            first_cat = ctrl.state.disagreement_report.items[0].category
            studio.combo_category_filter.setCurrentText(first_cat)
            studio.combo_override_method.setCurrentText("ModelB")
            studio.btn_accept_filtered.click()
            assert len(
                set(
                    np.unique(
                        ctrl.state.segmentation_results[FINAL_CURATED_MASK_NAME]["nuclear"]
                    ).tolist()
                )
                - {0}
            ) >= 3

            # 4. Reset to ModelA, draw 'Curation Selection ROI', and splice ModelB via ROI
            ctrl.review_svc.use_pipeline_as_final_mask("ModelA")
            assert set(
                np.unique(ctrl.state.segmentation_results[FINAL_CURATED_MASK_NAME]["nuclear"]).tolist()
            ) - {0} == {1, 2}

            roi_lyr = ctrl.review_svc.draw_curation_roi()
            assert roi_lyr is not None
            assert "Curation Selection ROI" in ctrl.viewer.layers
            assert roi_lyr.mode == "select"
            roi_lyr.data = [np.array([[5, 5], [5, 40], [26, 40], [26, 5]], dtype=float)]
            roi_curated = ctrl.review_svc.apply_roi_curation("ModelB")
            assert roi_curated is not None
            assert set(np.unique(roi_curated.nuclear).tolist()) - {0} == {1, 2, 3}
            assert set(np.unique(roi_curated.cytoplasm).tolist()) - {0} == {1, 2, 3}

            # 5. Verify feature extraction runs cleanly on Final Curated Mask
            nuc_final, cyto_final = ctrl.state.get_active_masks()
            df = extract_cell_features(
                mask=nuc_final,
                raw_image=img,
                markers=["DAPI", "Mem"],
                cytoplasm_mask=cyto_final,
            )
            assert len(df) == 3
            assert sorted(df["ObjectNumber"].astype(int).tolist()) == [1, 2, 3]
        finally:
            studio.close()
            qapp.processEvents()
    finally:
        _close_controller(ctrl, qapp)


# ---------------------------------------------------------------------------
# Static Hygiene: Zero Emojis, Zero QSS Warnings, Zero Top-Level Engine Qt Imports
# ---------------------------------------------------------------------------


def test_round5_static_hygiene_no_emojis_no_qss_warnings_headless_engine(qapp):
    """Ensure zero emojis in src/, zero Qt stylesheet warnings, & Qt-free engine imports."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    emoji_re = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")

    emoji_hits = []
    top_level_qt_hits = []
    for py_path in sorted(src_root.rglob("*.py")):
        text = py_path.read_text(encoding="utf-8")
        if emoji_re.search(text):
            emoji_hits.append(str(py_path.relative_to(repo_root)))
        if "engine" in py_path.parts:
            tree = ast.parse(text, filename=str(py_path))
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(
                            tok in alias.name.lower()
                            for tok in ("qtpy", "pyqt", "pyside", "napari")
                        ):
                            top_level_qt_hits.append(f"{py_path.name}:{alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if any(
                        tok in node.module.lower()
                        for tok in ("qtpy", "pyqt", "pyside", "napari")
                    ):
                        top_level_qt_hits.append(f"{py_path.name}:{node.module}")

    assert emoji_hits == []
    assert top_level_qt_hits == []

    # Headless curation module import check with qtpy/napari blocked
    saved_modules = {k: sys.modules.get(k) for k in ("qtpy", "napari") if k in sys.modules}
    sys.modules["qtpy"] = None  # type: ignore[assignment]
    sys.modules["napari"] = None  # type: ignore[assignment]
    try:
        spec = importlib.util.spec_from_file_location(
            "_headless_curation_check",
            src_root / "iris_viewer" / "engine" / "curation.py",
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        assert mod.FINAL_CURATED_MASK_NAME == "Final Curated Mask"
    finally:
        sys.modules.pop("_headless_curation_check", None)
        for k in ("qtpy", "napari"):
            if k in saved_modules:
                sys.modules[k] = saved_modules[k]
            else:
                sys.modules.pop(k, None)

    # Zero Qt stylesheet parse warnings across Round 5 widgets
    qss_warnings = []

    def _msg_handler(_mode, _ctx, msg):
        lower = msg.lower()
        if "stylesheet" in lower or "could not parse" in lower:
            qss_warnings.append(msg)

    prev_handler = qInstallMessageHandler(_msg_handler)
    try:
        dp = DisagreementPanelWidget()
        dp.update_available_methods(["M1", "M2"])
        pb = PipelineBuilderWidget()
        leg = ViewerLegendWidget(None)
        studio = SegmentationStudioDialog(disagreement_panel=dp)
        studio.show()
        qapp.processEvents()
        studio.close()
        leg.close()
        pb.close()
        dp.close()
    finally:
        qInstallMessageHandler(prev_handler)

    assert qss_warnings == []
