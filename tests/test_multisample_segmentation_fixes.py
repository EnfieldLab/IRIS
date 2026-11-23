"""Regression tests for multi-sample segmentation fixes, TMA QC alignment, and QSS syntax."""
from __future__ import annotations

import ast
import pathlib
import sys
import types

import numpy as np
import pytest
from qtpy.QtCore import QtMsgType, qInstallMessageHandler
from qtpy.QtWidgets import QApplication

from iris_common.widgets.publication_studio import PublicationPlotStudioDialog
from iris_common.widgets.sample_scope import SampleScopeBar
from iris_viewer.app.controller import ViewerController
from iris_viewer.engine.disagreement import compute_model_disagreement
from iris_viewer.engine.pipeline import (
    SegmentationSequence,
    SegmentationStep,
    _prepare_sequential_cytoplasm_inputs,
    crop_roi_and_segment_sync,
)
from iris_viewer.engine.segmentation.cellpose_backend import CellposeBackend
from iris_viewer.engine.segmentation.instanseg_backend import (
    InstanSegBackend,
    _resolve_instanseg_model_and_target,
)
from iris_viewer.engine.segmentation.stardist_backend import StardistBackend
from iris_viewer.widgets.disagreement_panel import DisagreementPanelWidget
from iris_viewer.widgets.segmentation_pipeline_builder import StepWidget
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
# Test 1: Cellpose tile_overlap clamping & sequential skip_nuclear optimization
# ---------------------------------------------------------------------------


def test_cellpose_tile_overlap_clamping_and_sequential_single_pass(qapp, monkeypatch):
    """Prevent `cannot reshape array` when tile_overlap < 0.05 & skip redundant nuclear pass."""
    # A. StepWidget UI enforces >= 0.05 minimum
    step = StepWidget(["Cellpose", "StarDist", "InstanSeg"], ["DAPI", "PanCK"])
    try:
        assert step.spin_cp_tile_overlap.minimum() == pytest.approx(0.05)
        assert step.spin_cp_tile_overlap.maximum() == pytest.approx(0.50)
        assert step.spin_cp_tile_overlap.value() == pytest.approx(0.10)
    finally:
        step.close()
        qapp.processEvents()

    # B. Backend clamps programmatic tile_overlap=0.0 -> 0.05 and honors skip_nuclear=True
    eval_calls = []

    class FakeCellposeModel:
        def __init__(self, gpu=False, device=None, pretrained_model=None, model_type=None):
            pass

        def eval(self, img, channel_axis=None, **kwargs):
            eval_calls.append((tuple(img.shape), dict(kwargs)))
            h, w = img.shape[-2], img.shape[-1]
            mask = np.zeros((h, w), dtype=np.uint32)
            mask[5:25, 5:25] = 1
            return mask, None, None

    fake_models_mod = types.ModuleType("cellpose.models")
    fake_models_mod.CellposeModel = FakeCellposeModel
    fake_models_mod.MODEL_NAMES = ["cpsam", "cyto3", "nuclei"]
    fake_models_mod.MODEL_DIR = None
    fake_cp_mod = types.ModuleType("cellpose")
    fake_cp_mod.models = fake_models_mod

    monkeypatch.setitem(sys.modules, "cellpose", fake_cp_mod)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models_mod)

    backend = CellposeBackend()
    img = np.random.randint(0, 500, (2, 80, 96), dtype=np.uint16)

    # Independent run with programmatic tile_overlap=0.0 clamps to 0.05
    res_ind = backend.segment(
        img, nuclear_channel=0, membrane_channel=1, tile_overlap=0.0, device="cpu"
    )
    assert res_ind["nuclear"] is not None and res_ind["cytoplasm"] is not None
    assert len(eval_calls) == 2
    assert eval_calls[0][1]["tile_overlap"] == pytest.approx(0.05)
    assert eval_calls[1][1]["tile_overlap"] == pytest.approx(0.05)

    # Sequential cytoplasm step skips the redundant nuclear pass when membrane is present
    eval_calls.clear()
    synth_img, kw = _prepare_sequential_cytoplasm_inputs(
        img,
        {"nuclear_channel": 0, "membrane_channels": [1], "tile_overlap": 0.0},
    )
    assert kw.get("skip_nuclear") is True
    res_seq = backend.segment(synth_img, device="cpu", **kw)
    assert len(eval_calls) == 1
    assert eval_calls[0][0] == (2, 80, 96)
    assert eval_calls[0][1]["tile_overlap"] == pytest.approx(0.05)
    assert res_seq["cytoplasm"] is not None


def test_cellpose_real_multi_tile_with_clamped_zero_overlap():
    """Verify real Cellpose tiling math and end-to-end execution with tile_overlap=0.0."""
    backend = CellposeBackend()
    if not backend.is_available():
        pytest.skip("Cellpose not installed")

    from cellpose import transforms

    # Ly=480, bsize=256 triggers ny=2 (unclamped 0.0) vs ny=3 (clamped 0.05) mismatch
    bsize = 256
    tile_grid = np.zeros((1, 480, 480), dtype=np.float32)
    img_tiles, *_ = transforms.make_tiles(tile_grid, bsize=bsize, tile_overlap=0.0)
    clamped_overlap = 0.05
    ny = int(np.ceil((1.0 + 2 * clamped_overlap) * 480 / bsize))
    nx = int(np.ceil((1.0 + 2 * clamped_overlap) * 480 / bsize))
    assert img_tiles.size == ny * nx * 1 * bsize * bsize

    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, size=(2, 280, 280), dtype=np.uint8)
    img[:, 50:70, 50:70] = 240
    res = backend.segment(
        img,
        nuclear_channel=0,
        membrane_channel=1,
        diameter=15.0,
        tile_overlap=0.0,
        device="cpu",
        model_type="cyto3",
        skip_nuclear=True,
    )
    assert res["cytoplasm"] is not None
    assert res["cytoplasm"].shape == (280, 280)


def test_stardist_backend_contiguity_and_gc(monkeypatch):
    """Verify StarDist preprocesses to contiguous float32 and returns uint32 mask."""
    calls = []

    class FakeStarDist2D:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def predict_instances(self, img, **kwargs):
            calls.append((img.dtype, img.flags["C_CONTIGUOUS"], dict(kwargs)))
            lbl = np.zeros(img.shape, dtype=np.int32)
            lbl[10:25, 10:25] = 1
            return lbl, {}

    fake_sd_models = types.ModuleType("stardist.models")
    fake_sd_models.StarDist2D = FakeStarDist2D
    fake_sd_mod = types.ModuleType("stardist")
    fake_sd_mod.models = fake_sd_models
    monkeypatch.setitem(sys.modules, "stardist", fake_sd_mod)
    monkeypatch.setitem(sys.modules, "stardist.models", fake_sd_models)

    backend = StardistBackend()
    img = np.random.randint(0, 500, (2, 70, 90), dtype=np.uint16)[:, ::-1, ::-1]
    res = backend.segment(img, nuclear_channel=0, membrane_channel=1, min_cell_area=15)
    assert calls[0][0] == np.float32
    assert calls[0][1] is True
    assert res["nuclear"].shape == (70, 90)
    assert res["nuclear"].dtype == np.uint32


# ---------------------------------------------------------------------------
# Test 2: Variable-Shape Multi-Sample Segmentation & TMA QC Sample Translation
# ---------------------------------------------------------------------------


def test_multisample_variable_shapes_and_tma_qc_sample_switching(qapp):
    """Verify variable-size TMA cores never mix shapes and QC overlays follow sample translate."""
    ctrl = ViewerController()
    try:
        # Core1 = (64, 80) at (0, 0); Core2 = (96, 72) at (0, 200)
        core1_nuc_a = np.zeros((64, 80), dtype=np.uint32)
        core1_nuc_b = np.zeros((64, 80), dtype=np.uint32)
        core1_nuc_a[10:25, 10:25] = 1
        core1_nuc_b[14:30, 14:30] = 1

        core2_nuc_a = np.zeros((96, 72), dtype=np.uint32)
        core2_nuc_b = np.zeros((96, 72), dtype=np.uint32)
        core2_nuc_a[20:40, 15:35] = 1
        core2_nuc_b[25:48, 20:42] = 1

        ctrl.state.multi_sample_images = {
            "Core1": np.ones((2, 64, 80), dtype=np.float32),
            "Core2": np.ones((2, 96, 72), dtype=np.float32),
        }
        ctrl.state.batch_results = {
            "Core1": {
                "Cellpose": {"nuclear": core1_nuc_a, "cytoplasm": None},
                "StarDist": {"nuclear": core1_nuc_b, "cytoplasm": None},
            },
            "Core2": {
                "Cellpose": {"nuclear": core2_nuc_a, "cytoplasm": None},
                "StarDist": {"nuclear": core2_nuc_b, "cytoplasm": None},
            },
        }

        b1 = ctrl.viewer.add_shapes(
            [np.array([[0, 0], [0, 80], [64, 80], [64, 0]], dtype=float)],
            shape_type="polygon",
            name="Bounds - Core1",
            translate=(0.0, 0.0),
        )
        b2 = ctrl.viewer.add_shapes(
            [np.array([[0, 0], [0, 72], [96, 72], [96, 0]], dtype=float)],
            shape_type="polygon",
            name="Bounds - Core2",
            translate=(0.0, 200.0),
        )
        ctrl.state.sample_layers_map = {"Core1": [b1], "Core2": [b2]}

        # Start inspecting Core2 (96x72 at translate=(0, 200))
        ctrl.state.current_sample_name = "Core2"
        ctrl.state.segmentation_results = dict(ctrl.state.batch_results["Core2"])
        ctrl._sync_segmentation_sources()
        dp = ctrl.disagreement_panel
        dp.update_available_samples(["Core1", "Core2"], active_sample="Core2")
        assert dp.qc_sample_row_widget.isHidden() is False
        assert dp.get_qc_sample() == "Core2"

        report_core2 = compute_model_disagreement(
            ctrl.state.segmentation_results,
            compare_mode="Nuclear Masks",
            iou_threshold=0.75,
        )
        ctrl.review_svc.render_disagreement_overlays(report_core2)
        ctrl.review_svc.next_issue()

        for lname in (
            "Method Comparison Overlay",
            "Disagreement Heatmap",
            "Review BBox",
            "Active Disagreement Isolation",
        ):
            t = tuple(ctrl.viewer.layers[lname].translate)
            assert (float(t[-2]), float(t[-1])) == pytest.approx((0.0, 200.0)), lname

        assert ctrl.viewer.layers["Method Comparison Overlay"].data.shape[:2] == (96, 72)

        # Open Studio dialog and switch QC sample from Core2 -> Core1 (64x80 at translate=(0, 0))
        studio = SegmentationStudioDialog(
            ctx=ctrl._ctx,
            disagreement_panel=dp,
            feature_panel=ctrl.feature_panel,
        )
        try:
            studio.show()
            qapp.processEvents()
            assert studio.combo_qc_sample.isVisible() is True
            assert studio.combo_qc_sample.currentText() == "Core2"

            studio.combo_qc_sample.setCurrentText("Core1")
            qapp.processEvents()

            assert ctrl.state.current_sample_name == "Core1"
            assert ctrl.viewer.layers["Method Comparison Overlay"].data.shape[:2] == (64, 80)
            for lname in ("Method Comparison Overlay", "Disagreement Heatmap"):
                t = tuple(ctrl.viewer.layers[lname].translate)
                assert (float(t[-2]), float(t[-1])) == pytest.approx((0.0, 0.0)), lname
        finally:
            studio.close()
            qapp.processEvents()
    finally:
        _close_controller(ctrl, qapp)


# ---------------------------------------------------------------------------
# Test 3: Zero QSS Parse Warnings Static AST + Runtime Qt Verification
# ---------------------------------------------------------------------------


def test_zero_concatenated_qss_selectors_and_no_runtime_parse_warnings(qapp):
    """Ensure all multi-rule Qt stylesheets separate rules with whitespace."""
    import re

    src_root = pathlib.Path(__file__).resolve().parents[1] / "src"
    bad_hits = []
    for py_path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value
            elif isinstance(node, ast.JoinedStr):
                text = "".join(
                    part.value
                    for part in node.values
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                )
            else:
                continue
            if re.search(r"\}(?:QPushButton|QToolButton|QLabel|QComboBox|QTabBar)", text):
                bad_hits.append(f"{py_path}:{node.lineno}")
    assert bad_hits == [], f"Found concatenated QSS rules: {bad_hits}"

    # Runtime Qt warning capture
    captured = []

    def _msg_handler(mode, context, message):
        if mode in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg):
            captured.append(str(message))

    prev_handler = qInstallMessageHandler(_msg_handler)
    dp = DisagreementPanelWidget()
    scope = SampleScopeBar()
    pub = PublicationPlotStudioDialog(plot_fn=lambda fig, cfg: None)
    try:
        dp.update_available_methods(["Cellpose", "StarDist", "InstanSeg"], active_method="Cellpose")
        qapp.processEvents()
    finally:
        dp.close()
        scope.close()
        pub.close()
        qapp.processEvents()
        qInstallMessageHandler(prev_handler)

    stylesheet_warnings = [m for m in captured if "Could not parse stylesheet" in m]
    assert stylesheet_warnings == [], f"Qt QSS warnings emitted: {stylesheet_warnings}"


# ---------------------------------------------------------------------------
# Test 4: InstanSeg Preset Resolution & ROI Preview Error Propagation
# ---------------------------------------------------------------------------


def test_instanseg_fluorescence_nuclei_and_cells_target_presets():
    """Map fluorescence_nuclei / fluorescence_cells presets to dual-head InstanSeg."""
    assert _resolve_instanseg_model_and_target("fluorescence_nuclei") == (
        "fluorescence_nuclei_and_cells",
        "nuclei",
    )
    assert _resolve_instanseg_model_and_target("fluorescence_cells") == (
        "fluorescence_nuclei_and_cells",
        "cells",
    )
    assert _resolve_instanseg_model_and_target(
        "fluorescence_nuclei_and_cells", skip_nuclear=True
    ) == ("fluorescence_nuclei_and_cells", "cells")
    assert _resolve_instanseg_model_and_target("brightfield_nuclei") == (
        "brightfield_nuclei",
        "all_outputs",
    )

    # Error propagation check on crop_roi_and_segment_sync
    from unittest.mock import MagicMock

    failing_backend = MagicMock()
    failing_backend.segment.side_effect = RuntimeError("synthetic model error")
    errs: list[str] = []
    img_dummy = np.ones((2, 64, 64), dtype=np.uint16)
    res_err = crop_roi_and_segment_sync(
        sequence=SegmentationSequence(
            steps=[SegmentationStep("InstanSeg", {"nuclear_channel": 0}, rule="Nuclear (Base)")]
        ),
        image=img_dummy,
        roi_bbox=(0, 0, 64, 64),
        backends={"InstanSeg": failing_backend},
        device="cpu",
        errors_out=errs,
    )
    assert res_err == {"nuclear": None, "cytoplasm": None}
    assert len(errs) == 1
    assert "synthetic model error" in errs[0]

    backend = InstanSegBackend()
    if not backend.is_available():
        pytest.skip("InstanSeg not installed")

    rng = np.random.default_rng(11)
    live_img = rng.integers(0, 120, size=(2, 128, 128), dtype=np.uint8)
    live_img[0, 30:50, 30:50] = 240
    live_img[1, 25:55, 25:55] = 220

    roi_errs_nuc: list[str] = []
    res_nuc = crop_roi_and_segment_sync(
        sequence=SegmentationSequence(
            steps=[
                SegmentationStep(
                    "InstanSeg",
                    {
                        "nuclear_channel": 0,
                        "model_name": "fluorescence_nuclei",
                        "min_cell_area": 15,
                    },
                    rule="Nuclear (Base)",
                )
            ]
        ),
        image=live_img,
        roi_bbox=(0, 0, 128, 128),
        backends={"InstanSeg": backend},
        device="cpu",
        errors_out=roi_errs_nuc,
    )
    assert roi_errs_nuc == []
    assert res_nuc["nuclear"] is not None
    assert res_nuc["nuclear"].shape == (128, 128)
    assert backend._cached_model_name == "fluorescence_nuclei_and_cells"

    roi_errs_cell: list[str] = []
    res_cell = crop_roi_and_segment_sync(
        sequence=SegmentationSequence(
            steps=[
                SegmentationStep(
                    "InstanSeg",
                    {
                        "nuclear_channel": 0,
                        "membrane_channels": [1],
                        "model_name": "fluorescence_cells",
                        "min_cell_area": 15,
                    },
                    rule="Independent",
                )
            ]
        ),
        image=live_img,
        roi_bbox=(0, 0, 128, 128),
        backends={"InstanSeg": backend},
        device="cpu",
        errors_out=roi_errs_cell,
    )
    assert roi_errs_cell == []
    assert res_cell["cytoplasm"] is not None
    assert res_cell["cytoplasm"].shape == (128, 128)
    assert backend._cached_model_name == "fluorescence_nuclei_and_cells"
