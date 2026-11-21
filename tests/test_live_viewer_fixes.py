"""Verification tests for live IRIS-Viewer fixes (Issues 1, 2, and 3):
1. Zero Napari ScaleBar.unit deprecation warnings.
2. Live single-sample segmentation on lazy Dask TIFF + flushed terminal/UI progress + contours.
3. Single-sample to multi-sample TMA transition & live multi-sample segmentation.
4. Live ROI preview on lazy Dask array without full-image materialization + TMA Bounds offset.
5. Simplified StepWidget UI (no Pre/Post drawer, inline Min Area, cpsam, watershed dilation).
6. Zero border-floating QGroupBox titles across Studios/Docks, headless engine safety, and hygiene.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import tifffile
from qtpy.QtWidgets import QGroupBox
from scipy.ndimage import gaussian_filter

from iris_analyzer import IrisAnalyzerPanelWidget
from iris_common.widgets.publication_studio import PublicationPlotStudioDialog
from iris_viewer.app.controller import ViewerController
from iris_viewer.core.layers import get_layer_base_name, get_layer_sample_name
from iris_viewer.engine.pipeline import (
    SegmentationSequence,
    SegmentationStep,
    execute_group_sync,
)
from iris_viewer.io.loaders import load_multichannel_tiff
from iris_viewer.widgets.marker_expression.dialog import MarkerExpressionDialog
from iris_viewer.widgets.phenotyping.diagnostics_dialog import PhenotyperDiagnosticDialog
from iris_viewer.widgets.phenotyping.gating_studio_dialog import GatingStudioDialog
from iris_viewer.widgets.phenotyping.panel import PhenotyperPanelWidget
from iris_viewer.widgets.phenotyping.sample_impact_dialog import SampleImpactDialog
from iris_viewer.widgets.segmentation_panel import SegmentationPanelWidget
from iris_viewer.widgets.segmentation_pipeline_builder import StepWidget
from iris_viewer.widgets.segmentation_studio_dialog import SegmentationStudioDialog


def _make_synthetic_tissue(
    shape_hw: tuple[int, int] = (128, 128),
    centers: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Create a (2, H, W) uint16 array with bright circular nuclei and membrane rings."""
    h, w = shape_hw
    if centers is None:
        centers = [(32, 32), (32, 92), (92, 62)]

    img = np.zeros((2, h, w), dtype=np.float32)
    yy, xx = np.ogrid[:h, :w]
    for cy, cx in centers:
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        img[0][r2 <= 10**2] = 4000.0
        img[1][(r2 >= 9**2) & (r2 <= 14**2)] = 3500.0

    img[0] = gaussian_filter(img[0], sigma=2.0) + 50.0
    img[1] = gaussian_filter(img[1], sigma=1.5) + 40.0
    return np.clip(img, 0, 65535).astype(np.uint16)


def _pump_until(qapp, predicate, timeout_s: float = 60.0) -> None:
    """Pump Qt events until *predicate()* is True or raise TimeoutError."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        qapp.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out after {timeout_s:.1f}s waiting for Qt condition.")
        time.sleep(0.02)
    qapp.processEvents()


def _close_controller(ctrl: ViewerController, qapp) -> None:
    """Shut down workers and close the Napari viewer cleanly on macOS PyQt5."""
    try:
        ctrl.pipeline_service.cancel_all()
    except Exception:
        pass
    ctrl.viewer.close()
    qapp.processEvents()


def test_zero_scale_bar_deprecation_warnings(qapp):
    """Test 1: ViewerController and update_scale emit zero ScaleBar.unit warnings."""
    ctrl = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ctrl = ViewerController()
            ctrl.image_svc.update_scale(0.5)
            ctrl.image_svc.update_scale(1.0)

        sb_warnings = [
            str(item.message)
            for item in caught
            if "scalebar" in str(item.message).lower()
            or "scale_bar.unit" in str(item.message).lower()
        ]
        assert sb_warnings == []
        assert ctrl.viewer.scale_bar.visible is False
        assert ctrl.viewer.scale_bar.unit is None
    finally:
        if ctrl is not None:
            _close_controller(ctrl, qapp)


def test_single_sample_segmentation_dask_tiff_progress_and_contours(qapp, tmp_path, capsys):
    """Test 2: Live single-sample segmentation on lazy Dask TIFF with StarDist & Cellpose."""
    tiff_path = tmp_path / "Sample_single_image.tiff"
    syn = _make_synthetic_tissue((128, 128))
    tifffile.imwrite(str(tiff_path), syn)

    raw_dask, n_ch = load_multichannel_tiff(tiff_path)
    assert n_ch == 2
    assert hasattr(raw_dask, "compute")

    ctrl = ViewerController()
    try:
        assert "StarDist" in ctrl.pipeline_service.backends
        assert "Cellpose" in ctrl.pipeline_service.backends

        ctrl.state.markers = ["DAPI", "PanCK"]
        ctrl.state.file_map = {"Sample_single": tiff_path}
        ctrl.segmentation_panel.update_channel_dropdowns(ctrl.state.markers)
        ctrl.image_svc.on_sample_selected("Sample_single")
        _pump_until(
            qapp,
            lambda: ctrl.state.raw_image is not None and len(ctrl._active_workers) == 0,
            timeout_s=20.0,
        )
        assert hasattr(ctrl.state.raw_image, "compute")

        capsys.readouterr()

        steps = [
            SegmentationStep("StarDist", {"nuclear_channel": 0}, rule="Nuclear (Base)"),
        ]
        ctrl._run_segmentation_pipeline({"Pipeline 1": steps})
        _pump_until(
            qapp,
            lambda: (
                "Pipeline 1" in ctrl.state.segmentation_results
                and len(ctrl.pipeline_service._active_workers) == 0
                and not ctrl.segmentation_panel.pipeline_builder.btn_cancel.isVisible()
            ),
            timeout_s=45.0,
        )

        out = capsys.readouterr().out
        assert "[IRIS-Viewer] [" in out and "[Segmentation]" in out
        assert ctrl.state.segmentation_status["Pipeline 1_0"] == "Done"
        sw0 = ctrl.segmentation_panel.pipeline_builder.tab_widget.widget(0).step_widgets[0]
        assert sw0.lbl_status.text() == "Done"

        seg_res = ctrl.state.segmentation_results["Pipeline 1"]
        nuc_mask = seg_res["nuclear"]
        assert nuc_mask is not None
        assert nuc_mask.dtype == np.uint32
        assert int(nuc_mask.max()) >= 2

        assert ctrl.state.current_mask is nuc_mask
        assert ctrl.state.masks_map["Sample_single"] is nuc_mask

        lyr = ctrl.viewer.layers["Pipeline 1 Nuclear Mask"]
        assert lyr.visible is True
        assert lyr.contour == 0
        assert lyr.opacity == pytest.approx(0.8)

        # Also exercise PipelineService.execute_pipeline_groups directly with on_step_start +
        # on_step_complete using the real Cellpose ('cpsam') backend.
        events_log: list[tuple] = []
        all_done = [False]
        cp_results: dict = {}

        ctrl.pipeline_service.execute_pipeline_groups(
            groups={
                "CellposeGrp": SegmentationSequence(
                    steps=[
                        SegmentationStep(
                            "Cellpose",
                            {
                                "nuclear_channel": 0,
                                "membrane_channels": [1],
                                "model_type": "cpsam",
                                "diameter": 22.0,
                            },
                            rule="Independent",
                        )
                    ]
                )
            },
            image=raw_dask,
            on_step_start=lambda g, idx, name: events_log.append(("start", g, idx, name)),
            on_step_complete=lambda g, idx, name, res: events_log.append(
                ("complete", g, idx, name, res)
            ),
            on_group_complete=lambda g, res: cp_results.update({g: res}),
            on_all_complete=lambda: all_done.__setitem__(0, True),
            on_error=lambda g, idx, name, err: events_log.append(("error", g, idx, name, err)),
        )
        _pump_until(
            qapp,
            lambda: all_done[0] and len(ctrl.pipeline_service._active_workers) == 0,
            timeout_s=60.0,
        )

        cp_out = capsys.readouterr().out
        assert "[IRIS-Viewer] [" in cp_out and "[Segmentation]" in cp_out
        assert [ev[0] for ev in events_log] == ["start", "complete"]
        assert events_log[0][:4] == ("start", "CellposeGrp", 0, "Cellpose")
        cp_nuc = cp_results["CellposeGrp"]["nuclear"]
        assert cp_nuc is not None
        assert cp_nuc.dtype == np.uint32
        assert int(cp_nuc.max()) >= 2
    finally:
        _close_controller(ctrl, qapp)


def test_single_to_multi_sample_transition_and_live_tma_segmentation(qapp, tmp_path):
    """Test 3: Transition from single-sample to multi-sample TMA and run StarDist on both."""
    path_a = tmp_path / "SampleA_image.tiff"
    path_b = tmp_path / "SampleB_image.tiff"
    tifffile.imwrite(str(path_a), _make_synthetic_tissue((128, 128)))
    tifffile.imwrite(
        str(path_b),
        _make_synthetic_tissue((128, 128), centers=[(28, 40), (64, 80), (96, 44)]),
    )

    ctrl = ViewerController()
    try:
        ctrl.state.markers = ["DAPI", "PanCK"]
        ctrl.state.file_map = {"SampleA": path_a, "SampleB": path_b}
        ctrl.segmentation_panel.update_channel_dropdowns(ctrl.state.markers)

        # Start in single-sample mode with SampleA loaded in raw_image & sample_layers_map
        ctrl.image_svc.on_sample_selected("SampleA")
        _pump_until(
            qapp,
            lambda: (
                ctrl.state.raw_image is not None
                and "SampleA" in ctrl.state.sample_layers_map
                and len(ctrl._active_workers) == 0
            ),
            timeout_s=20.0,
        )
        assert ctrl.state.raw_image is not None

        # Transition into Multi-Sample TMA mode containing both SampleA and SampleB
        ctrl.multi_sample_svc.add_samples_to_canvas(["SampleA", "SampleB"])
        _pump_until(
            qapp,
            lambda: (
                set(ctrl.state.multi_sample_images.keys()) == {"SampleA", "SampleB"}
                and len(ctrl._active_workers) == 0
            ),
            timeout_s=25.0,
        )

        assert "SampleA" in ctrl.state.multi_sample_images
        assert "SampleB" in ctrl.state.multi_sample_images
        assert ctrl.state.raw_image is None
        assert "Bounds - SampleA" in ctrl.viewer.layers
        assert "Bounds - SampleB" in ctrl.viewer.layers
        t_a = tuple(ctrl.viewer.layers["Bounds - SampleA"].translate)
        t_b = tuple(ctrl.viewer.layers["Bounds - SampleB"].translate)
        assert t_a != t_b

        ctrl._run_segmentation_pipeline(
            {"SegA": [SegmentationStep("StarDist", {"nuclear_channel": 0}, rule="Nuclear (Base)")]}
        )
        _pump_until(
            qapp,
            lambda: (
                len(ctrl.pipeline_service._active_workers) == 0
                and len(ctrl._active_workers) == 0
                and "SampleA" in ctrl.state.masks_map
                and "SampleB" in ctrl.state.masks_map
                and "SegA" in ctrl.state.segmentation_results
                and ctrl.segmentation_panel.disagreement_panel.get_active_method() == "SegA"
            ),
            timeout_s=45.0,
        )

        for sn in ("SampleA", "SampleB"):
            mask = ctrl.state.masks_map[sn]
            assert mask.dtype == np.uint32
            assert int(mask.max()) >= 2

        assert "SegA" in ctrl.state.segmentation_results
        assert ctrl.state.segmentation_results["SegA"]["nuclear"] is not None

        seg_layers = [
            lyr
            for lyr in ctrl.viewer.layers
            if get_layer_base_name(lyr) == "SegA Nuclear Mask"
        ]
        assert len(seg_layers) == 2
        by_sample = {get_layer_sample_name(lyr): lyr for lyr in seg_layers}
        assert set(by_sample.keys()) == {"SampleA", "SampleB"}
        assert tuple(by_sample["SampleA"].translate) == pytest.approx(t_a)
        assert tuple(by_sample["SampleB"].translate) == pytest.approx(t_b)
        for lyr in seg_layers:
            assert lyr.visible is True
            assert lyr.contour == 0
            assert lyr.opacity == pytest.approx(0.8)
    finally:
        _close_controller(ctrl, qapp)


def test_live_roi_preview_lazy_dask_and_multi_sample_offset(qapp, monkeypatch):
    """Test 4: ROI preview computes only cropped ROI region and honors TMA Bounds offset."""
    centers = [(96, 96), (96, 156), (156, 128)]
    np_img = _make_synthetic_tissue((256, 256), centers=centers)
    dask_img = da.from_array(np_img, chunks=(1, 64, 64))

    computed_shapes: list[tuple[int, ...]] = []
    orig_compute = da.Array.compute

    def _spy_compute(self, *args, **kwargs):
        computed_shapes.append(tuple(self.shape))
        return orig_compute(self, *args, **kwargs)

    monkeypatch.setattr(da.Array, "compute", _spy_compute)

    ctrl = ViewerController()
    try:
        ctrl.state.markers = ["DAPI", "PanCK"]
        ctrl.state.raw_image = dask_img
        ctrl.state.current_sample_name = "DaskSample"

        ctrl._create_or_focus_preview_roi()
        assert "Segmentation Preview ROI" in ctrl.viewer.layers

        # Resize ROI to a strict sub-region [64:192, 64:192] so full (2, 256, 256)
        # materialization is distinguishable from the 128x128 ROI crop.
        roi_lyr = ctrl.viewer.layers["Segmentation Preview ROI"]
        roi_lyr.data = [np.array([[64, 64], [64, 192], [192, 192], [192, 64]], dtype=float)]
        assert ctrl.state.preview_roi_bbox == (64, 64, 192, 192)

        computed_shapes.clear()
        ctrl._run_roi_preview(
            "PreviewGrp",
            [SegmentationStep("StarDist", {"nuclear_channel": 0}, rule="Nuclear (Base)")],
        )
        _pump_until(
            qapp,
            lambda: (
                "ROI Preview - Nuclear" in ctrl.viewer.layers
                and len(ctrl.pipeline_service._active_workers) == 0
                and not ctrl.segmentation_panel.pipeline_builder.progress_bar.isVisible()
            ),
            timeout_s=30.0,
        )

        assert len(computed_shapes) >= 1
        assert all(s == (128, 128) for s in computed_shapes)
        assert (2, 256, 256) not in computed_shapes
        assert (256, 256) not in computed_shapes

        preview_lyr = ctrl.viewer.layers["ROI Preview - Nuclear"]
        assert preview_lyr.visible is True
        assert preview_lyr.contour == 0
        assert preview_lyr.opacity == pytest.approx(0.8)
        assert preview_lyr.data.shape == (128, 128)
        assert int(np.max(preview_lyr.data)) >= 1
        assert tuple(preview_lyr.translate) == pytest.approx((64.0, 64.0))

        # Verify multi-sample TMA Bounds offset calculation with physical scale = 0.5 um/px
        ctrl.viewer.layers.clear()
        ctrl.state.raw_image = None
        ctrl.state.preview_roi_bbox = None
        ctrl.control_widget.spin_scale.setValue(0.5)
        scale_val = 0.5

        dask_a = da.from_array(_make_synthetic_tissue((256, 256), centers=centers), chunks=(1, 64, 64))
        dask_b = da.from_array(_make_synthetic_tissue((256, 256), centers=centers), chunks=(1, 64, 64))
        ctrl.state.multi_sample_images = {"SampleA": dask_a, "SampleB": dask_b}
        ctrl.state.current_sample_name = "SampleA"

        bounds_rect = np.array([[0, 0], [0, 256], [256, 256], [256, 0]], dtype=float)
        ctrl.viewer.add_shapes(
            [bounds_rect],
            shape_type="polygon",
            name="Bounds - SampleA",
            scale=(scale_val, scale_val),
            translate=(0.0, 0.0),
            metadata={"sample_name": "SampleA", "layer_type": "bounds"},
        )
        ctrl.viewer.add_shapes(
            [bounds_rect],
            shape_type="polygon",
            name="Bounds - SampleB",
            scale=(scale_val, scale_val),
            translate=(150.0, 200.0),  # (300 px, 400 px) in world coordinates at 0.5 um/px
            metadata={"sample_name": "SampleB", "layer_type": "bounds"},
        )
        # Focus camera inside SampleB's bounding box
        ctrl.viewer.camera.center = (150.0 + 64.0, 200.0 + 64.0)

        active_img, active_sn, offset_px = ctrl._get_active_sample_and_offset_for_roi()
        assert active_sn == "SampleB"
        assert active_img is dask_b
        assert offset_px == pytest.approx((300.0, 400.0))

        ctrl._create_or_focus_preview_roi()
        tma_roi_lyr = ctrl.viewer.layers["Segmentation Preview ROI"]
        assert tuple(tma_roi_lyr.translate) == pytest.approx((150.0, 200.0))
        tma_roi_lyr.data = [np.array([[64, 64], [64, 192], [192, 192], [192, 64]], dtype=float)]

        computed_shapes.clear()
        ctrl._run_roi_preview(
            "PreviewGrp",
            [SegmentationStep("StarDist", {"nuclear_channel": 0}, rule="Nuclear (Base)")],
        )
        _pump_until(
            qapp,
            lambda: (
                "ROI Preview - Nuclear" in ctrl.viewer.layers
                and len(ctrl.pipeline_service._active_workers) == 0
                and not ctrl.segmentation_panel.pipeline_builder.progress_bar.isVisible()
            ),
            timeout_s=30.0,
        )

        assert len(computed_shapes) >= 1
        assert all(s == (128, 128) for s in computed_shapes)
        tma_preview_lyr = ctrl.viewer.layers["ROI Preview - Nuclear"]
        assert tma_preview_lyr.visible is True
        assert tma_preview_lyr.contour == 0
        assert tma_preview_lyr.opacity == pytest.approx(0.8)
        assert int(np.max(tma_preview_lyr.data)) >= 1
        assert tuple(tma_preview_lyr.translate) == pytest.approx((150.0 + 32.0, 200.0 + 32.0))
    finally:
        _close_controller(ctrl, qapp)


def test_simplified_step_widget_ui_and_watershed_dilation(qapp):
    """Test 5: Simplified StepWidget UI and watershed Cytoplasm (Dilate Remaining) execution."""
    markers = ["DAPI", "PanCK", "CD3"]
    sw = StepWidget(["Cellpose", "StarDist", "InstanSeg", "CellSAM"], markers)
    try:
        sw.show()
        qapp.processEvents()

        assert not hasattr(sw, "btn_toggle_prepost")
        assert not hasattr(sw, "prepost_container")
        assert hasattr(sw, "spin_min_cell_area")

        cp_models = [sw.combo_cp_model.itemText(i) for i in range(sw.combo_cp_model.count())]
        assert "cpsam" in cp_models

        sw.combo_type.setCurrentText("Cytoplasm (Dilate Remaining)")
        qapp.processEvents()

        dilate_modes = [
            sw.combo_dilate_mode.itemText(i) for i in range(sw.combo_dilate_mode.count())
        ]
        assert "distance" in dilate_modes
        assert "watershed" in dilate_modes

        sw.combo_dilate_mode.setCurrentText("distance")
        qapp.processEvents()
        assert sw.combo_mem.isVisible() is False

        sw.combo_dilate_mode.setCurrentText("watershed")
        qapp.processEvents()
        assert sw.lbl_mem.isVisible() is True
        assert sw.combo_mem.isVisible() is True

        sw.spin_dilate.setValue(18)
        sw.combo_mem.set_checked_items(["PanCK"])
        step_cfg = sw.get_step_config(markers)
        assert step_cfg.rule == "Cytoplasm (Dilate Remaining)"
        assert step_cfg.params["dilation_mode"] == "watershed"
        assert step_cfg.params["membrane_channels"] == [1]
        assert step_cfg.params["dilation_radius"] == 18

        # Execute Cytoplasm (Dilate Remaining) in watershed mode via execute_group_sync
        nuc_mask = np.zeros((40, 60), dtype=np.uint32)
        nuc_mask[18:22, 13:17] = 1
        nuc_mask[18:22, 43:47] = 2

        mem_img = np.zeros((40, 60), dtype=np.float32)
        mem_img[10:30, 8:23] = 500.0
        mem_img[10:30, 25:52] = 500.0  # Dark cleft at x=23..25 shifts watershed left of x=30

        image = np.stack([(nuc_mask > 0).astype(np.float32) * 500.0, mem_img], axis=0)

        class _FixedNuclearBackend:
            name = "FixedNuc"

            def segment(self, **_kwargs):
                return {"nuclear": nuc_mask.copy(), "cytoplasm": None}

        seq_ws = SegmentationSequence(
            steps=[
                SegmentationStep("FixedNuc", {"nuclear_channel": 0}, rule="Nuclear (Base)"),
                step_cfg,
            ]
        )
        res_ws = execute_group_sync(
            "ws_dilate", seq_ws, image, {"FixedNuc": _FixedNuclearBackend()}, device="cpu"
        )
        cyto_ws = res_ws["cytoplasm"]
        assert cyto_ws is not None
        assert int(cyto_ws.max()) == 2
        assert cyto_ws[20, 28] == 2  # Assigned across dark cleft by watershed signal
    finally:
        sw.close()
        qapp.processEvents()


def test_zero_groupbox_titles_headless_engine_and_emoji_hygiene(qapp):
    """Test 6: Zero QGroupBox border titles, headless engine import safety, and no emojis."""
    small_df = pd.DataFrame(
        {
            "Image": ["S1", "S1", "S2", "S2"],
            "ObjectNumber": [1, 2, 1, 2],
            "Pos_X": [10.0, 20.0, 15.0, 25.0],
            "Pos_Y": [12.0, 22.0, 16.0, 26.0],
            "CD3_mean": [8.0, 0.5, 7.5, 0.4],
            "CD8_mean": [0.4, 8.2, 0.6, 8.5],
            "cell_type": ["T_cell", "B_cell", "T_cell", "B_cell"],
        }
    )

    widgets_to_close = []
    try:
        widgets = [
            SegmentationPanelWidget(),
            SegmentationStudioDialog(),
            PhenotyperPanelWidget(),
            PhenotyperDiagnosticDialog(),
            GatingStudioDialog(),
            SampleImpactDialog(),
            MarkerExpressionDialog(df=small_df, markers=["CD3", "CD8"], celltype_col="cell_type"),
            PublicationPlotStudioDialog(),
            IrisAnalyzerPanelWidget(),
        ]
        widgets_to_close.extend(widgets)

        for w in widgets:
            for box in w.findChildren(QGroupBox):
                assert box.title() == "", (
                    f"{type(w).__name__} contains QGroupBox with non-empty title {box.title()!r}"
                )
    finally:
        for w in widgets_to_close:
            w.close()
        qapp.processEvents()

    # Headless engine check: napari=None, iris_viewer.engine + PipelineService() leaves
    # qtpy, torch, and tensorflow unimported.
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = str(repo_root / "src")
    code = (
        "import sys\n"
        "sys.modules['napari'] = None\n"
        "sys.modules['napari.qt'] = None\n"
        "import iris_viewer.engine\n"
        "import iris_viewer.engine.pipeline\n"
        "svc = iris_viewer.engine.pipeline.PipelineService()\n"
        "for mod in ('qtpy', 'torch', 'tensorflow'):\n"
        "    assert mod not in sys.modules, f'Eagerly imported {mod}!'\n"
        "assert 'StarDist' in svc.get_available_methods()\n"
        "print('HEADLESS_ENGINE_OK')\n"
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
    assert "HEADLESS_ENGINE_OK" in proc.stdout

    # Check zero unnecessary emojis in all changed files (git diff --name-only)
    changed_files = subprocess.check_output(
        ["git", "-C", str(repo_root), "diff", "--name-only"],
        text=True,
    ).splitlines()
    for rel_path in changed_files:
        fpath = repo_root / rel_path
        if not fpath.is_file():
            continue
        text = fpath.read_text(encoding="utf-8")
        high_emojis = sorted({ch for ch in text if ord(ch) >= 0x1F300})
        assert high_emojis == [], f"Unexpected emoji(s) in {rel_path}: {high_emojis}"

        diff_text = subprocess.check_output(
            ["git", "-C", str(repo_root), "diff", "--", rel_path],
            text=True,
        )
        added_dingbats = [
            ch
            for line in diff_text.splitlines()
            if line.startswith("+") and not line.startswith("+++")
            for ch in line
            if 0x2600 <= ord(ch) <= 0x27BF
        ]
        assert added_dingbats == [], f"Unexpected added symbol(s) in {rel_path}: {added_dingbats}"
