"""Unit tests for Phase 3 segmentation backends, disagreement QC, ROI preview,
multi-compartment feature extraction, headless safety, and exporters.
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tifffile

from iris_viewer.engine.disagreement import (
    DisagreementItem,
    DisagreementReport,
    compute_model_disagreement,
    render_disagreement_contours_rgba,
)
from iris_viewer.engine.feature_extraction import extract_cell_features
from iris_viewer.engine.pipeline import (
    PipelineService,
    SegmentationSequence,
    SegmentationStep,
    crop_roi_and_segment_sync,
)
from iris_viewer.engine.segmentation.base import SegmentationBackend
from iris_viewer.engine.segmentation.cellpose_backend import CellposeBackend
from iris_viewer.engine.segmentation.cellsam_backend import CellSAMBackend
from iris_viewer.engine.segmentation.instanseg_backend import InstanSegBackend
from iris_viewer.engine.segmentation.stardist_backend import StardistBackend
from iris_viewer.io.exporters import (
    export_disagreement_map,
    export_segmentation_mask,
)


class _DummyBackend(SegmentationBackend):
    name = "Dummy"
    supports_cytoplasm = False

    def is_available(self) -> bool:
        return True

    def get_default_params(self):
        return {}

    def segment(self, image, nuclear_channel, membrane_channel, diameter, device, expand_radius=0, **kwargs):
        return {"nuclear": np.zeros(image.shape[-2:], dtype=np.uint32), "cytoplasm": None}


def _wrap_nuc(mask: np.ndarray) -> dict:
    return {"nuclear": mask.astype(np.uint32), "cytoplasm": None}


class TestModelDisagreement:
    def test_split_detection_and_heatmap_boost(self):
        ref = np.zeros((40, 40), dtype=np.uint32)
        ref[10:30, 10:30] = 1

        peer = np.zeros((40, 40), dtype=np.uint32)
        peer[10:30, 10:20] = 1
        peer[10:30, 20:30] = 2

        report = compute_model_disagreement(
            {"Ref": _wrap_nuc(ref), "Peer": _wrap_nuc(peer)},
            compare_mode="Nuclear Masks",
            reference_method="Ref",
            iou_threshold=0.65,
        )

        assert isinstance(report, DisagreementReport)
        assert report.flagged_regions == 1
        assert report.agreed_cells == 0
        assert report.category_counts["Split"] == 1
        assert len(report.items) == 1

        item = report.items[0]
        assert isinstance(item, DisagreementItem)
        assert item.category == "Split"
        assert item.method_cell_ids == {"Ref": [1], "Peer": [1, 2]}
        assert item.bbox == (10, 10, 30, 30)
        assert 0.0 <= item.severity <= 1.0
        # Split interior heatmap should be boosted despite both predicting foreground
        assert float(report.disagreement_heatmap[20, 15]) >= 0.75
        assert float(report.agreement_map[20, 15]) <= 0.25

        rgba = render_disagreement_contours_rgba(
            {"Ref": _wrap_nuc(ref), "Peer": _wrap_nuc(peer)}, report
        )
        assert rgba.shape == (40, 40, 4)
        assert rgba.dtype == np.float32
        # Red contour on split boundary
        expected_red = np.array([1.0, 59.0 / 255.0, 48.0 / 255.0, 1.0], dtype=np.float32)
        assert np.allclose(rgba[10, 15], expected_red, atol=1e-4)

    def test_merge_detection_and_contour_color(self):
        ref = np.zeros((40, 40), dtype=np.uint32)
        ref[10:30, 10:20] = 1
        ref[10:30, 20:30] = 2

        peer = np.zeros((40, 40), dtype=np.uint32)
        peer[10:30, 10:30] = 1

        report = compute_model_disagreement(
            {"Ref": _wrap_nuc(ref), "Peer": _wrap_nuc(peer)},
            reference_method="Ref",
            iou_threshold=0.65,
        )

        assert report.total_ref_cells == 2
        assert report.agreed_cells == 0
        assert report.flagged_regions == 1
        assert report.category_counts["Merge"] == 1
        item = report.items[0]
        assert item.category == "Merge"
        assert item.method_cell_ids == {"Ref": [1, 2], "Peer": [1]}

        rgba = render_disagreement_contours_rgba(
            {"Ref": _wrap_nuc(ref), "Peer": _wrap_nuc(peer)}, report
        )
        expected_purple = np.array(
            [191.0 / 255.0, 90.0 / 255.0, 242.0 / 255.0, 1.0], dtype=np.float32
        )
        assert np.allclose(rgba[15, 19], expected_purple, atol=1e-4)

    def test_boundary_shift_and_threshold_sensitivity(self):
        ref = np.zeros((40, 40), dtype=np.uint32)
        ref[10:20, 10:20] = 1  # 100 px

        peer = np.zeros((40, 40), dtype=np.uint32)
        peer[10:20, 13:23] = 1  # 100 px, inter = 70, union = 130, IoU = 70/130 ~= 0.5385

        res = {"Ref": _wrap_nuc(ref), "Peer": _wrap_nuc(peer)}
        rep_strict = compute_model_disagreement(res, reference_method="Ref", iou_threshold=0.65)
        assert rep_strict.category_counts["Boundary Shift"] == 1
        assert rep_strict.flagged_regions == 1
        assert rep_strict.items[0].category == "Boundary Shift"
        assert rep_strict.items[0].best_iou == pytest.approx(70.0 / 130.0, abs=1e-4)

        rgba = render_disagreement_contours_rgba(res, rep_strict)
        expected_amber = np.array([1.0, 204.0 / 255.0, 0.0, 1.0], dtype=np.float32)
        assert np.allclose(rgba[10, 15], expected_amber, atol=1e-4)

        rep_relaxed = compute_model_disagreement(res, reference_method="Ref", iou_threshold=0.50)
        assert rep_relaxed.category_counts["Agreed"] == 1
        assert rep_relaxed.flagged_regions == 0
        assert rep_relaxed.agreed_cells == 1

    def test_model_exclusive_ref_and_orphan_peer(self):
        ref = np.zeros((50, 50), dtype=np.uint32)
        ref[5:15, 5:15] = 1  # Exclusive to Ref
        ref[30:40, 30:40] = 2  # Shared

        peer = np.zeros((50, 50), dtype=np.uint32)
        peer[30:40, 30:40] = 1  # Shared
        peer[5:15, 35:45] = 2  # Exclusive orphan to Peer

        res = {"Ref": _wrap_nuc(ref), "Peer": _wrap_nuc(peer)}
        report = compute_model_disagreement(res, reference_method="Ref", iou_threshold=0.65)

        assert report.total_ref_cells == 2
        assert report.agreed_cells == 1
        assert report.category_counts["Agreed"] == 1
        assert report.category_counts["Model-Exclusive"] == 2
        assert report.flagged_regions == 2
        assert all(it.category == "Model-Exclusive" for it in report.items)
        present_methods = [set(it.method_cell_ids.keys()) for it in report.items]
        assert {"Ref"} in present_methods
        assert {"Peer"} in present_methods

        rgba = render_disagreement_contours_rgba(res, report, show_agreed=True)
        expected_cyan = np.array([0.0, 229.0 / 255.0, 1.0, 1.0], dtype=np.float32)
        expected_green = np.array(
            [52.0 / 255.0, 199.0 / 255.0, 89.0 / 255.0, 1.0], dtype=np.float32
        )
        assert np.allclose(rgba[5, 8], expected_cyan, atol=1e-4)
        assert np.allclose(rgba[30, 35], expected_green, atol=1e-4)

    def test_single_model_edge_case_and_pairwise_matrices(self):
        m = np.zeros((32, 32), dtype=np.uint32)
        m[4:12, 4:12] = 1
        m[18:26, 18:26] = 2

        report = compute_model_disagreement({"OnlyModel": _wrap_nuc(m)})
        assert report.methods == ["OnlyModel"]
        assert report.reference_method == "OnlyModel"
        assert report.total_ref_cells == 2
        assert report.agreed_cells == 2
        assert report.flagged_regions == 0
        assert report.items == []
        assert report.category_counts["Agreed"] == 2
        assert report.mean_iou == pytest.approx(1.0)
        assert float(report.disagreement_heatmap.max()) == 0.0
        assert float(report.agreement_map.min()) == 1.0
        assert report.pairwise_dice.shape == (1, 1)
        assert report.pairwise_iou.shape == (1, 1)
        assert float(report.pairwise_dice.iloc[0, 0]) == pytest.approx(1.0)
        assert float(report.pairwise_iou.iloc[0, 0]) == pytest.approx(1.0)
        assert list(report.model_summary.columns) == [
            "Method",
            "Cell_Count",
            "Median_Area_px",
            "Mean_Peer_IoU",
        ]

    def test_compare_modes_cytoplasm_and_both(self):
        nuc_a = np.zeros((32, 32), dtype=np.uint32)
        nuc_a[10:14, 10:14] = 1
        cyto_a = np.zeros((32, 32), dtype=np.uint32)
        cyto_a[8:18, 8:18] = 1

        nuc_b = nuc_a.copy()
        cyto_b = np.zeros((32, 32), dtype=np.uint32)
        cyto_b[8:18, 12:22] = 1  # Shifted cytoplasm only

        res = {
            "A": {"nuclear": nuc_a, "cytoplasm": cyto_a},
            "B": {"nuclear": nuc_b, "cytoplasm": cyto_b},
        }
        rep_nuc = compute_model_disagreement(res, compare_mode="Nuclear Masks")
        rep_cyto = compute_model_disagreement(res, compare_mode="Cytoplasm Masks")
        rep_both = compute_model_disagreement(res, compare_mode="Both")

        assert rep_nuc.agreed_cells == 1
        assert rep_cyto.category_counts["Boundary Shift"] == 1
        assert rep_both.category_counts["Boundary Shift"] == 1
        assert np.allclose(rep_cyto.pairwise_dice.values, rep_cyto.pairwise_dice.values.T)
        assert np.allclose(rep_cyto.pairwise_iou.values, rep_cyto.pairwise_iou.values.T)


class TestSegmentationBaseAndBackends:
    def test_preprocess_channel_smoothing_and_percentile_norm(self):
        backend = _DummyBackend()
        img = np.zeros((32, 32), dtype=np.uint16)
        img[16, 16] = 1000
        img[5:10, 5:10] = 500

        proc = backend._preprocess_channel(
            img, gaussian_sigma=1.0, norm_pmin=1.0, norm_pmax=99.0
        )
        assert proc.dtype == np.float32
        assert float(proc.min()) >= 0.0
        assert float(proc.max()) <= 1.0
        # Smoothing spreads peak into immediate neighbors
        assert proc[16, 17] > 0.0

        unnorm = backend._preprocess_channel(
            img, gaussian_sigma=0.0, norm_pmin=None, norm_pmax=None
        )
        assert unnorm.dtype == np.float32
        assert float(unnorm.max()) == pytest.approx(1000.0)

    def test_filter_mask_by_area_relabels_sequentially(self):
        backend = _DummyBackend()
        mask = np.zeros((30, 30), dtype=np.uint32)
        mask[1:3, 1:3] = 1  # 4 px (too small)
        mask[5:10, 5:10] = 4  # 25 px (keep -> 1)
        mask[12:18, 12:18] = 7  # 36 px (keep -> 2)
        mask[18:30, 18:30] = 9  # 144 px (too large)

        filtered = backend._filter_mask_by_area(mask, min_area=10, max_area=50)
        assert filtered is not None
        assert filtered.dtype == np.uint32
        assert set(np.unique(filtered).tolist()) == {0, 1, 2}
        assert np.count_nonzero(filtered == 1) == 25
        assert np.count_nonzero(filtered == 2) == 36
        assert backend._filter_mask_by_area(None, min_area=10) is None

    def test_watershed_vs_distance_cytoplasm_expansion(self):
        backend = _DummyBackend()
        nuc = np.zeros((40, 60), dtype=np.uint32)
        nuc[18:22, 13:17] = 1
        nuc[18:22, 43:47] = 2  # Midpoint x = 30

        dist_mask = backend._expand_nuclear_to_cytoplasm(nuc, expand_radius=20, mode="distance")
        # Boundary under Euclidean distance sits symmetrically at x=30
        assert dist_mask[20, 28] == 1
        assert dist_mask[20, 31] == 2

        # Fallback to distance when membrane_image is None
        ws_fallback = backend._expand_nuclear_to_cytoplasm(
            nuc, expand_radius=20, mode="watershed", membrane_image=None
        )
        np.testing.assert_array_equal(dist_mask, ws_fallback)

        # Membrane/cytoplasm channel with dark cleft at x=23..25 shifts watershed boundary
        mem_img = np.zeros((40, 60), dtype=np.float32)
        mem_img[10:30, 8:23] = 100.0  # Cell 1 domain
        mem_img[10:30, 25:52] = 100.0  # Cell 2 domain extending left of x=30

        ws_mask = backend._expand_nuclear_to_cytoplasm(
            nuc, expand_radius=20, mode="watershed", membrane_image=mem_img
        )
        assert not np.array_equal(ws_mask, dist_mask)
        # Pixel at (20, 28) is inside cell 2's bright domain so watershed assigns it to label 2
        assert ws_mask[20, 28] == 2

    def test_backend_default_params_and_mocked_wiring(self):
        cp = CellposeBackend()
        cp_defaults = cp.get_default_params()
        for key in (
            "model_type",
            "diameter",
            "flow_threshold",
            "cellprob_threshold",
            "tile_overlap",
            "gaussian_sigma",
            "norm_pmin",
            "norm_pmax",
            "min_cell_area",
            "max_cell_area",
            "expand_radius",
            "expansion_mode",
        ):
            assert key in cp_defaults

        sd = StardistBackend()
        sd_defaults = sd.get_default_params()
        for key in (
            "model_name",
            "prob_thresh",
            "nms_thresh",
            "normalize_input",
            "norm_pmin",
            "norm_pmax",
            "gaussian_sigma",
            "min_cell_area",
            "max_cell_area",
            "expand_radius",
            "expansion_mode",
        ):
            assert key in sd_defaults

        ins = InstanSegBackend()
        ins_defaults = ins.get_default_params()
        for key in (
            "model_name",
            "pixel_size",
            "resolve_cell_and_nucleus",
            "cleanup_fragments",
            "gaussian_sigma",
            "norm_pmin",
            "norm_pmax",
            "min_cell_area",
            "max_cell_area",
            "expand_radius",
            "expansion_mode",
        ):
            assert key in ins_defaults

        cs = CellSAMBackend()
        cs_defaults = cs.get_default_params()
        for key in (
            "token",
            "bbox_threshold",
            "use_all_channels",
            "gaussian_sigma",
            "norm_pmin",
            "norm_pmax",
            "min_cell_area",
            "max_cell_area",
            "expand_radius",
            "expansion_mode",
        ):
            assert key in cs_defaults

        # Mocked Cellpose nuclei + expansion + area filter execution
        fake_cp_masks = np.zeros((32, 32), dtype=np.int32)
        fake_cp_masks[2:4, 2:4] = 1  # 4px filtered out
        fake_cp_masks[10:20, 10:20] = 2  # 100px kept -> 1

        mock_cp_model = MagicMock()
        mock_cp_model.eval.return_value = (fake_cp_masks, None, None, None)

        def _fake_eval(self, x, channel_axis=None, **kw):
            return None

        fake_models_mod = types.ModuleType("cellpose.models")
        fake_models_mod.CellposeModel = MagicMock(return_value=mock_cp_model)
        fake_models_mod.CellposeModel.eval = _fake_eval
        fake_cp_pkg = types.ModuleType("cellpose")
        fake_cp_pkg.models = fake_models_mod
        fake_torch = types.ModuleType("torch")
        fake_torch.device = lambda name: name

        img = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "cellpose": fake_cp_pkg,
                "cellpose.models": fake_models_mod,
            },
        ):
            res_cp = cp.segment(
                img,
                nuclear_channel=0,
                membrane_channel=1,
                diameter=20.0,
                device="cpu",
                expand_radius=3,
                model_type="nuclei",
                flow_threshold=0.6,
                cellprob_threshold=0.2,
                tile_overlap=0.25,
                min_cell_area=10,
                expansion_mode="distance",
            )
        assert set(np.unique(res_cp["nuclear"]).tolist()) == {0, 1}
        assert res_cp["cytoplasm"] is not None
        assert np.count_nonzero(res_cp["cytoplasm"] == 1) > 100
        _, eval_kw = mock_cp_model.eval.call_args
        assert eval_kw["flow_threshold"] == pytest.approx(0.6)
        assert eval_kw["cellprob_threshold"] == pytest.approx(0.2)
        assert eval_kw["tile_overlap"] == pytest.approx(0.25)

        # Mocked StarDist parameter & watershed wiring
        mock_sd_model = MagicMock()
        mock_sd_model.predict_instances.return_value = (fake_cp_masks, {})
        fake_sd_models = types.ModuleType("stardist.models")
        fake_sd_models.StarDist2D = MagicMock()
        fake_sd_models.StarDist2D.from_pretrained.return_value = mock_sd_model
        fake_sd_pkg = types.ModuleType("stardist")
        fake_sd_pkg.models = fake_sd_models

        with patch.dict(
            sys.modules,
            {"stardist": fake_sd_pkg, "stardist.models": fake_sd_models},
        ):
            res_sd = sd.segment(
                img,
                nuclear_channel=0,
                membrane_channel=1,
                diameter=None,
                device="cpu",
                expand_radius=3,
                prob_thresh=0.65,
                nms_thresh=0.35,
                normalize_input=False,
                gaussian_sigma=1.0,
                min_cell_area=10,
                expansion_mode="watershed",
            )
        assert set(np.unique(res_sd["nuclear"]).tolist()) == {0, 1}
        assert res_sd["cytoplasm"] is not None
        _, sd_kw = mock_sd_model.predict_instances.call_args
        assert sd_kw["prob_thresh"] == pytest.approx(0.65)
        assert sd_kw["nms_thresh"] == pytest.approx(0.35)

        # Mocked InstanSeg parameter wiring
        fake_is_out = np.stack([fake_cp_masks, fake_cp_masks], axis=0)
        mock_is_instance = MagicMock()
        mock_is_instance.eval_medium_image.return_value = (fake_is_out,)
        fake_is_pkg = types.ModuleType("instanseg")
        fake_is_pkg.InstanSeg = MagicMock(return_value=mock_is_instance)

        fake_torch_is = types.ModuleType("torch")
        fake_torch_is.quantile = MagicMock()
        fake_torch_is.Tensor = type("FakeTensor", (), {})
        fake_torch_is.from_numpy = lambda arr: MagicMock(
            unsqueeze=lambda dim: arr[np.newaxis, ...]
        )

        with patch.dict(
            sys.modules,
            {"torch": fake_torch_is, "instanseg": fake_is_pkg},
        ):
            res_is = ins.segment(
                img,
                nuclear_channel=0,
                membrane_channel=1,
                diameter=None,
                device="cpu",
                pixel_size=0.4,
                resolve_cell_and_nucleus=False,
                cleanup_fragments=False,
                min_cell_area=10,
            )
        assert set(np.unique(res_is["nuclear"]).tolist()) == {0, 1}
        assert set(np.unique(res_is["cytoplasm"]).tolist()) == {0, 1}
        _, is_kw = mock_is_instance.eval_medium_image.call_args
        assert is_kw["pixel_size"] == pytest.approx(0.4)
        assert is_kw["resolve_cell_and_nucleus"] is False
        assert is_kw["cleanup_fragments"] is False

        # Mocked CellSAM bbox_threshold & use_all_channels wiring
        mock_cellsam_mod = types.ModuleType("cellSAM")
        mock_model_obj = MagicMock()
        mock_model_obj.to.return_value = mock_model_obj
        mock_cellsam_mod.get_model = MagicMock(return_value=mock_model_obj)
        mock_cellsam_mod.segment_cellular_image = MagicMock(
            return_value=(fake_cp_masks, None, None)
        )
        with patch.dict(sys.modules, {"cellSAM": mock_cellsam_mod}):
            res_cs = cs.segment(
                img,
                nuclear_channel=0,
                membrane_channel=None,
                diameter=None,
                device="cpu",
                bbox_threshold=0.55,
                use_all_channels=True,
                min_cell_area=10,
            )
        assert set(np.unique(res_cs["nuclear"]).tolist()) == {0, 1}
        _, cs_kw = mock_cellsam_mod.segment_cellular_image.call_args
        assert cs_kw["bbox_threshold"] == pytest.approx(0.55)


class TestROIPreviewAndMultiCompartmentFeatures:
    def test_crop_roi_and_segment_sync_clamping_and_2d(self):
        image_3d = np.arange(2 * 40 * 50, dtype=np.float32).reshape(2, 40, 50)
        image_2d = image_3d[0]

        def _seg_fn(image, **kwargs):
            h, w = image.shape[-2:]
            m = np.zeros((h, w), dtype=np.uint32)
            m[1:h - 1, 1:w - 1] = 1
            return {"nuclear": m, "cytoplasm": None}

        mock_backend = MagicMock()
        mock_backend.segment.side_effect = _seg_fn
        backends = {"MockSeg": mock_backend}
        seq = SegmentationSequence(
            steps=[SegmentationStep("MockSeg", {"nuclear_channel": 0}, rule="Nuclear (Base)")]
        )

        # Negative and out-of-bounds bbox should clamp cleanly to (0, 10, 25, 50) -> 25x40
        res_3d = crop_roi_and_segment_sync(
            seq, image_3d, (-5, 10, 25, 999), backends, device="cpu"
        )
        assert res_3d["nuclear"].shape == (25, 40)

        res_2d = crop_roi_and_segment_sync(
            seq, image_2d, (5, 5, 20, 30), backends, device="cpu"
        )
        assert res_2d["nuclear"].shape == (15, 25)

        empty_res = crop_roi_and_segment_sync(
            seq, image_2d, (100, 100, 120, 120), backends, device="cpu"
        )
        assert empty_res == {"nuclear": None, "cytoplasm": None}

    def test_pipeline_service_execute_roi_preview_lazy_worker(self):
        class _FakeSig:
            def __init__(self):
                self._fns = []

            def connect(self, fn):
                self._fns.append(fn)

            def emit(self, *args):
                for fn in self._fns:
                    fn(*args)

        class _FakeWorker:
            def __init__(self, fn):
                self._fn = fn
                self.returned = _FakeSig()
                self.finished = _FakeSig()
                self.errored = _FakeSig()

            def start(self):
                try:
                    out = self._fn()
                    self.returned.emit(out)
                except Exception as exc:
                    self.errored.emit(exc)
                finally:
                    self.finished.emit()

        def _fake_thread_worker(fn=None, **kwargs):
            def _wrap(f):
                return lambda: _FakeWorker(f)

            return _wrap(fn) if fn is not None else _wrap

        mock_backend = MagicMock()
        mock_backend.name = "MockSeg"
        mock_backend.is_available.return_value = True
        mock_backend.segment.side_effect = lambda image, **kw: {
            "nuclear": np.ones(image.shape[-2:], dtype=np.uint32),
            "cytoplasm": None,
        }

        with patch("iris_viewer.engine.pipeline.get_available_backends", return_value=[mock_backend]):
            svc = PipelineService()

        seq = SegmentationSequence(
            steps=[SegmentationStep("MockSeg", {"nuclear_channel": 0}, rule="Nuclear (Base)")]
        )
        img = np.zeros((2, 30, 30), dtype=np.float32)
        captured = []
        errors = []

        with patch("napari.qt.thread_worker", _fake_thread_worker):
            svc.execute_roi_preview(
                "Pipeline 1",
                seq,
                img,
                (-2, 5, 18, 25),
                on_complete=lambda grp, bbox, res: captured.append((grp, bbox, res)),
                on_error=lambda grp, msg: errors.append((grp, msg)),
            )

        assert errors == []
        assert len(captured) == 1
        grp, bbox, res = captured[0]
        assert grp == "Pipeline 1"
        assert bbox == (0, 5, 18, 25)
        assert res["nuclear"].shape == (18, 20)
        assert svc._active_workers == []

    def test_extract_cell_features_backward_compat_and_compartments(self):
        nuc = np.zeros((32, 32), dtype=np.uint32)
        nuc[12:16, 12:16] = 1  # 4x4 = 16 px

        cyto = np.zeros((32, 32), dtype=np.uint32)
        cyto[10:18, 10:18] = 1  # 8x8 = 64 px -> cyto ring = 48 px

        raw = np.zeros((2, 32, 32), dtype=np.float32)
        raw[0, 10:18, 10:18] = 20.0
        raw[0, 12:16, 12:16] = 100.0  # DAPI high in nucleus
        raw[1, 10:18, 10:18] = 50.0  # Membrane marker on whole cell

        # 1. Backward compatible mode (no extra compartment columns)
        df_legacy = extract_cell_features(nuc, raw, markers=["DAPI", "CD3"])
        assert "cyto_area" not in df_legacy.columns
        assert "DAPI_nuc_mean" not in df_legacy.columns
        assert float(df_legacy.loc[0, "DAPI_mean"]) == pytest.approx(100.0)

        # 2. Full multi-compartment extraction
        df_comp = extract_cell_features(
            nuc,
            raw,
            markers=["DAPI", "CD3"],
            cytoplasm_mask=cyto,
            compartments=["whole_cell", "nuclear", "cytoplasm", "membrane", "nc_ratio"],
        )
        assert float(df_comp.loc[0, "area"]) == pytest.approx(64.0)
        assert float(df_comp.loc[0, "cyto_area"]) == pytest.approx(48.0)
        assert float(df_comp.loc[0, "nc_area_ratio"]) == pytest.approx(16.0 / (48.0 + 1e-6))
        assert float(df_comp.loc[0, "DAPI_nuc_mean"]) == pytest.approx(100.0)
        assert float(df_comp.loc[0, "DAPI_cyto_mean"]) == pytest.approx(20.0)
        assert float(df_comp.loc[0, "DAPI_mem_mean"]) == pytest.approx(20.0)
        assert float(df_comp.loc[0, "CD3_mem_mean"]) == pytest.approx(50.0)
        assert float(df_comp.loc[0, "DAPI_nc_ratio"]) == pytest.approx(100.0 / (20.0 + 1e-6))

        # 3. Omitting whole_cell keeps base {marker}_{stat} matching nuclear mask
        df_no_whole = extract_cell_features(
            nuc,
            raw,
            markers=["DAPI"],
            cytoplasm_mask=cyto,
            compartments=["nuclear", "cytoplasm"],
        )
        assert float(df_no_whole.loc[0, "DAPI_mean"]) == pytest.approx(100.0)
        assert float(df_no_whole.loc[0, "DAPI_cyto_mean"]) == pytest.approx(20.0)


class TestHeadlessAndExporters:
    def test_headless_engine_import_without_napari(self):
        src_dir = str(Path(__file__).resolve().parents[1] / "src")
        code = (
            "import sys; "
            "sys.modules['napari'] = None; "
            "sys.modules['napari.qt'] = None; "
            "import iris_viewer.engine as eng; "
            "import iris_viewer.engine.disagreement as dis; "
            "import iris_viewer.engine.pipeline as pip; "
            "import iris_viewer.engine.feature_extraction as fe; "
            "import iris_viewer.engine.segmentation as seg; "
            "import iris_viewer.io as io_mod; "
            "assert hasattr(eng, 'compute_model_disagreement'); "
            "assert hasattr(eng, 'crop_roi_and_segment_sync'); "
            "assert hasattr(io_mod, 'export_segmentation_mask'); "
            "assert hasattr(io_mod, 'export_disagreement_map'); "
            "assert not hasattr(io_mod, 'export_consensus_mask'); "
            "assert hasattr(dis, 'DisagreementReport'); "
            "assert hasattr(pip, 'PipelineService'); "
            "assert hasattr(fe, 'extract_cell_features'); "
            "assert hasattr(seg, 'SegmentationBackend')"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert proc.returncode == 0, f"stderr: {proc.stderr}"

    def test_exporters_segmentation_and_disagreement_maps(self, tmp_path: Path):
        nuc = np.array([[0, 1], [2, 0]], dtype=np.uint32)
        cyto = np.array([[1, 1], [2, 2]], dtype=np.uint32)
        dmap = np.array([[0.0, 0.5], [0.9, 0.2]], dtype=np.float32)

        seg_tiff = tmp_path / "mask.tiff"
        export_segmentation_mask(seg_tiff, nuc, cyto)
        loaded_seg = tifffile.imread(str(seg_tiff))
        assert loaded_seg.shape == (2, 2, 2)
        np.testing.assert_array_equal(loaded_seg[0], nuc)
        np.testing.assert_array_equal(loaded_seg[1], cyto)

        seg_npy = tmp_path / "mask.npy"
        export_segmentation_mask(seg_npy, nuc)
        np.testing.assert_array_equal(np.load(str(seg_npy)), nuc)

        dis_tiff = tmp_path / "disagreement.tiff"
        export_disagreement_map(dis_tiff, dmap)
        loaded_dis = tifffile.imread(str(dis_tiff))
        np.testing.assert_allclose(loaded_dis, dmap, atol=1e-6)

        dis_npz = tmp_path / "disagreement.npz"
        export_disagreement_map(dis_npz, dmap)
        loaded_npz = np.load(str(dis_npz))
        np.testing.assert_allclose(loaded_npz["disagreement"], dmap, atol=1e-6)

        with pytest.raises(ValueError, match="mismatch"):
            export_segmentation_mask(tmp_path / "bad.tiff", nuc, np.zeros((3, 3), dtype=np.uint32))
