"""Comprehensive regression tests for IRIS Phase 1 bugfixes, decoupling, and UI wiring."""
from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QMessageBox, QPushButton

from iris_viewer.core.context import AppContext
from iris_viewer.core.state import ViewerAppState
from iris_viewer.core.types import CellTypePanel, CellTypeRule
from iris_viewer.engine.feature_extraction import extract_cell_features
from iris_viewer.engine.phenotyping.astir_backend import ASTIRBackend
from iris_viewer.engine.phenotyping.ensemble import PhenotypingEnsemble
from iris_viewer.engine.pipeline import (
    PipelineService,
    SegmentationSequence,
    SegmentationStep,
    execute_group_sync,
)


def _make_sample_features_df() -> pd.DataFrame:
    """Create synthetic single-cell extracted features using extract_cell_features."""
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[2:10, 2:10] = 1
    mask[12:20, 12:20] = 2
    mask[22:30, 22:30] = 3
    mask[2:10, 22:30] = 4
    mask[22:30, 2:10] = 5
    mask[32:38, 12:20] = 6

    raw = np.zeros((3, 40, 40), dtype=np.float32)
    # Channel 0: DAPI, Channel 1: CD3, Channel 2: CD8
    raw[0] = 50.0
    raw[1, 2:10, 2:10] = 100.0
    raw[1, 12:20, 12:20] = 95.0
    raw[1, 22:30, 22:30] = 5.0
    raw[2, 2:10, 2:10] = 90.0
    raw[2, 12:20, 12:20] = 5.0
    raw[2, 22:30, 22:30] = 85.0

    df = extract_cell_features(mask, raw, markers=["DAPI", "CD3", "CD8"])
    df["Image"] = ["Sample_A", "Sample_A", "Sample_A", "Sample_B", "Sample_B", "Sample_B"]
    return df


# 1. Headless Engine Decoupling


class TestHeadlessEngineDecoupling:
    def test_engine_imports_without_napari(self):
        """Importing iris_viewer.engine must succeed even when napari is blocked."""
        src_dir = str(Path(__file__).resolve().parents[1] / "src")
        code = (
            "import sys; "
            "sys.modules['napari'] = None; "
            "sys.modules['napari.qt'] = None; "
            "import iris_viewer.engine as eng; "
            "assert hasattr(eng, 'PipelineService'); "
            "assert hasattr(eng, 'extract_cell_features')"
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

    def test_get_available_backends_only_available_flag(self):
        import iris_viewer.engine.segmentation as seg_mod

        class _Avail:
            name = "Avail"

            def is_available(self) -> bool:
                return True

        class _Missing:
            name = "Missing"

            def is_available(self) -> bool:
                return False

        with patch.object(seg_mod, "AVAILABLE_BACKENDS", [_Avail, _Missing]):
            avail_only = seg_mod.get_available_backends()
            assert [b.name for b in avail_only] == ["Avail"]
            all_backends = seg_mod.get_available_backends(only_available=False)
            assert [b.name for b in all_backends] == ["Avail", "Missing"]

    def test_pipeline_independent_rule_sync_membrane_max_projection(self):
        img = np.zeros((3, 16, 16), dtype=np.float32)
        img[0] = 10.0
        img[1, 0:4, 0:4] = 30.0
        img[2, 4:8, 4:8] = 70.0

        nuc_mask = np.ones((16, 16), dtype=np.uint32)
        cyto_mask = np.full((16, 16), 2, dtype=np.uint32)
        captured: dict = {}

        backend = MagicMock()
        backend.name = "MockWholeCell"

        def _seg(image, nuclear_channel, membrane_channel, diameter, device, **kwargs):
            captured["image"] = image.copy()
            captured["nuc_ch"] = nuclear_channel
            captured["mem_ch"] = membrane_channel
            return {"nuclear": nuc_mask, "cytoplasm": cyto_mask}

        backend.segment.side_effect = _seg
        seq = SegmentationSequence(
            steps=[
                SegmentationStep(
                    "MockWholeCell",
                    {"nuclear_channel": 0, "membrane_channels": [1, 2]},
                    rule="Independent",
                )
            ]
        )
        res = execute_group_sync("grp1", seq, img, {"MockWholeCell": backend}, device="cpu")
        assert captured["nuc_ch"] == 0
        assert captured["mem_ch"] == 1
        assert captured["image"].shape == (2, 16, 16)
        assert captured["image"][1, 2, 2] == 30.0
        assert captured["image"][1, 5, 5] == 70.0
        np.testing.assert_array_equal(res["nuclear"], nuc_mask)
        np.testing.assert_array_equal(res["cytoplasm"], cyto_mask)

    def test_pipeline_independent_rule_sync_no_membrane(self):
        img = np.zeros((2, 12, 12), dtype=np.float32)
        nuc_mask = np.ones((12, 12), dtype=np.uint32)
        captured: dict = {}

        backend = MagicMock()

        def _seg(image, nuclear_channel, membrane_channel, diameter, device, **kwargs):
            captured["shape"] = image.shape
            captured["mem_ch"] = membrane_channel
            return {"nuclear": nuc_mask, "cytoplasm": None}

        backend.segment.side_effect = _seg
        seq = SegmentationSequence(
            steps=[SegmentationStep("B", {"nuclear_channel": 0}, rule="Independent")]
        )
        res = execute_group_sync("grp", seq, img, {"B": backend}, device="cpu")
        assert captured["shape"] == (2, 12, 12)
        assert captured["mem_ch"] is None
        np.testing.assert_array_equal(res["nuclear"], nuc_mask)
        assert res["cytoplasm"] is None

    def test_pipeline_independent_rule_async(self, monkeypatch):
        import napari.qt

        img = np.zeros((3, 10, 10), dtype=np.float32)
        img[1, 1, 1] = 12.0
        img[2, 2, 2] = 44.0
        nuc = np.ones((10, 10), dtype=np.uint32)
        cyto = np.full((10, 10), 2, dtype=np.uint32)

        backend = MagicMock()
        backend.name = "MockInd"
        backend.is_available.return_value = True
        backend.segment.return_value = {"nuclear": nuc, "cytoplasm": cyto}

        class _Sig:
            def __init__(self):
                self._cbs = []

            def connect(self, cb):
                self._cbs.append(cb)

            def emit(self, *args):
                for cb in self._cbs:
                    cb(*args)

        class _SyncWorker:
            def __init__(self, gen_fn):
                self._gen_fn = gen_fn
                self.yielded = _Sig()
                self.finished = _Sig()
                self.errored = _Sig()

            def start(self):
                try:
                    for item in self._gen_fn():
                        self.yielded.emit(item)
                except Exception as exc:
                    self.errored.emit(exc)
                else:
                    self.finished.emit()

        def _fake_thread_worker(*_args, **_kwargs):
            def _dec(fn):
                return lambda: _SyncWorker(fn)
            return _dec

        monkeypatch.setattr(napari.qt, "thread_worker", _fake_thread_worker)

        step_calls = []
        group_calls = []
        done_calls = []
        err_calls = []

        with patch("iris_viewer.engine.pipeline.get_available_backends", return_value=[backend]):
            svc = PipelineService()
            svc.execute_pipeline_groups(
                groups={
                    "G1": SegmentationSequence(
                        steps=[
                            SegmentationStep(
                                "MockInd",
                                {"nuclear_channel": 0, "membrane_channels": [1, 2]},
                                rule="Independent",
                            )
                        ]
                    )
                },
                image=img,
                on_step_complete=lambda g, idx, name, res: step_calls.append(
                    (g, idx, name, res)
                ),
                on_group_complete=lambda g, res: group_calls.append((g, res)),
                on_all_complete=lambda: done_calls.append(True),
                on_error=lambda *a: err_calls.append(a),
            )

        assert not err_calls
        assert len(step_calls) == 1
        assert step_calls[0][3]["nuclear"] is nuc
        assert step_calls[0][3]["cytoplasm"] is cyto
        assert len(group_calls) == 1
        np.testing.assert_array_equal(group_calls[0][1]["nuclear"], nuc)
        np.testing.assert_array_equal(group_calls[0][1]["cytoplasm"], cyto)
        assert done_calls == [True]


# 2. Crash & Logic Bug Fixes


class TestCrashAndLogicBugFixes:
    def test_segmentation_panel_open_studio_dialog_and_alias(self, qapp):
        from iris_viewer.widgets.segmentation_panel import SegmentationPanelWidget
        from iris_viewer.widgets.segmentation_studio_dialog import SegmentationStudioDialog

        assert SegmentationStudioDialog._sync_state is SegmentationStudioDialog.refresh_data

        panel = SegmentationPanelWidget()
        panel.open_studio_dialog()
        assert panel.popup_dialog is not None
        # Second call triggers refresh_data() cleanly without AttributeError
        panel.open_studio_dialog()
        panel.popup_dialog.close()
        panel.close()

    def test_batch_service_get_pipeline_groups(self, qapp):
        from iris_viewer.app.services.batch_service import BatchService
        from iris_viewer.widgets.segmentation_panel import SegmentationPanelWidget

        seg_panel = SegmentationPanelWidget()
        ctx = AppContext(segmentation_panel=seg_panel)
        svc = BatchService(ctx)
        groups = svc._get_pipeline_groups()
        assert len(groups) >= 1
        assert "Pipeline 1" in groups

        # Legacy mock panel exposing tab_widget directly
        legacy_panel = MagicMock(spec=["tab_widget"])
        legacy_panel.tab_widget = seg_panel.pipeline_builder.tab_widget
        ctx_legacy = AppContext(segmentation_panel=legacy_panel)
        svc_legacy = BatchService(ctx_legacy)
        assert "Pipeline 1" in svc_legacy._get_pipeline_groups()
        seg_panel.close()

    def test_ensemble_results_to_dataframe_without_and_with_object_number(self):
        ens = PhenotypingEnsemble(min_probability_threshold=0.0)
        preds = {
            "ASTIR": {
                10: {"Tumor": 0.9, "T cell": 0.1},
                20: {"Tumor": 0.2, "T cell": 0.8},
                30: {"Tumor": 0.7, "T cell": 0.3},
            }
        }
        final_types, confs, per_method = ens.vote(preds, strategy="soft_vote")

        # Without ObjectNumber, non-contiguous integer index
        df_idx = pd.DataFrame({"CD3": [0.1, 5.0, 0.2]}, index=[10, 20, 30])
        out_idx = ens.results_to_dataframe(df_idx, final_types, confs, per_method)
        assert list(out_idx["cell_type"]) == ["Tumor", "T cell", "Tumor"]
        assert list(out_idx["ASTIR_cell_type"]) == ["Tumor", "T cell", "Tumor"]
        assert list(out_idx["cell_type_confidence"]) == [1.0, 1.0, 1.0]

        # With ObjectNumber
        df_obj = pd.DataFrame({"ObjectNumber": [30, 10, 20], "CD3": [0.2, 0.1, 5.0]})
        out_obj = ens.results_to_dataframe(df_obj, final_types, confs, per_method)
        assert list(out_obj["cell_type"]) == ["Tumor", "Tumor", "T cell"]
        assert list(out_obj["ASTIR_cell_type"]) == ["Tumor", "Tumor", "T cell"]

    def test_astir_backend_refits_when_dataframe_changes(self, monkeypatch):
        fit_calls = []

        class _FakeAstirModel:
            def __init__(self, expr: pd.DataFrame, type_dict: dict):
                self.expr = expr
                self.type_dict = type_dict

            def fit_type(self, **_kwargs):
                fit_calls.append(self.expr.copy())

            def get_celltype_probabilities(self) -> pd.DataFrame:
                # Return probability proportional to CD3_mean vs CD20_mean of fitted expr
                p_t = np.where(self.expr["CD3"].to_numpy() > self.expr["CD20"].to_numpy(), 0.9, 0.1)
                return pd.DataFrame({"T_cell": p_t, "B_cell": 1.0 - p_t}, index=self.expr.index)

        fake_astir_mod = types.ModuleType("astir")
        fake_astir_mod.Astir = _FakeAstirModel

        panel = CellTypePanel(
            rules=[
                CellTypeRule("T_cell", markers_positive=["CD3"]),
                CellTypeRule("B_cell", markers_positive=["CD20"]),
            ]
        )
        df1 = pd.DataFrame(
            {"ObjectNumber": [1, 2], "CD3_mean": [10.0, 0.1], "CD20_mean": [0.1, 10.0]}
        )
        df2 = pd.DataFrame(
            {"ObjectNumber": [1, 2], "CD3_mean": [0.1, 10.0], "CD20_mean": [10.0, 0.1]}
        )

        monkeypatch.setitem(sys.modules, "astir", fake_astir_mod)
        backend = ASTIRBackend()
        backend.fit(df1, cell_type_panel=panel)
        assert len(fit_calls) == 1

        # Same df -> uses cached model without refitting
        p1 = backend.predict(df1, cell_type_panel=panel)
        assert len(fit_calls) == 1
        assert p1[1]["T_cell"] == pytest.approx(0.9)
        assert p1[2]["B_cell"] == pytest.approx(0.9)

        # Different values on df2 -> automatically refits instead of returning stale probs
        p2 = backend.predict(df2, cell_type_panel=panel)
        assert len(fit_calls) == 2
        assert p2[1]["B_cell"] == pytest.approx(0.9)
        assert p2[2]["T_cell"] == pytest.approx(0.9)


# 3. Column Aliasing, Scale Transformations, Nan Alignment, & Z-score CSV


class TestFeatureAliasingAndGatingFixes:
    def test_resolve_marker_col_and_transform_values(self):
        from iris_viewer.widgets.phenotyping.thresholding import (
            resolve_marker_col,
            transform_values,
        )

        df = pd.DataFrame(
            {
                "DAPI": [1.0, 2.0],
                "CD3_mean": [10.0, 20.0],
                "CD8_median": [5.0, 15.0],
                "PanCK_Mean": [3.0, 7.0],
            }
        )
        assert resolve_marker_col(df, "DAPI") == "DAPI"
        assert resolve_marker_col(df, "CD3") == "CD3_mean"
        assert resolve_marker_col(df, "CD8", preferred_stat="median") == "CD8_median"
        assert resolve_marker_col(df, "CD8", preferred_stat="mean") is None
        assert resolve_marker_col(df, "dapi") == "DAPI"
        assert resolve_marker_col(df, "panck") == "PanCK_Mean"
        assert resolve_marker_col(df, "MissingMarker") is None

        arr = np.array([0.0, 5.0, 10.0])
        np.testing.assert_allclose(
            transform_values(arr, "Arcsinh (Cofactor = 5)"),
            np.arcsinh(arr / 5.0),
        )
        np.testing.assert_allclose(
            transform_values(arr, "Log10(1 + x)"),
            np.log10(arr + 1.0),
        )
        np.testing.assert_allclose(transform_values(arr, "Linear (Raw)"), arr)
        np.testing.assert_allclose(
            transform_values(arr, "Min-Max Normalized"),
            np.array([0.0, 0.5, 1.0]),
        )
        # Constant array in Min-Max should not produce NaNs
        np.testing.assert_allclose(
            transform_values(np.array([4.0, 4.0]), "Min-Max Normalized"),
            np.array([0.0, 0.0]),
        )

    def test_extracted_features_work_end_to_end_in_gating_phenotyper_expression(self, qapp):
        from iris_viewer.widgets.marker_expression import (
            IrisMarkerExpressionWidget,
            MarkerExpressionDialog,
        )
        from iris_viewer.widgets.phenotyping.panel import PhenotyperPanelWidget

        df = _make_sample_features_df()
        markers = ["DAPI", "CD3", "CD8"]

        state = ViewerAppState()
        state.features_df = df
        state.markers = markers
        ctx = AppContext(state=state)

        # PhenotyperPanelWidget & ManualGatingWidget
        panel = PhenotyperPanelWidget(ctx=ctx)
        panel.set_data(df=df, markers=markers)
        mg = panel.manual_gating

        mg.combo_x.setCurrentText("CD3")
        mg.combo_y.setCurrentText("CD8")
        mg.combo_scale.setCurrentText("Linear (Raw)")
        mg._auto_threshold("x", "otsu")
        mg.spin_tx.setValue(50.0)
        mg.spin_ty.setValue(50.0)
        mg._apply_gate()
        assert "CellType" in df.columns
        assert (df["CellType"] == "CD3+ CD8+").sum() >= 1

        # GatingStudioDialog opens cleanly on extracted features
        mg.open_popup_window(0)
        assert mg.popup_dialog is not None
        mg.popup_dialog._render_plot()
        mg.popup_dialog.close()

        # Clustering, ASTIR fallback, MAPS fallback
        assert panel.list_cluster_markers.count() == 3
        panel.combo_cluster_method.setCurrentText("KMeans")
        panel.spin_kmeans_k.setValue(2)
        panel._run_clustering()
        assert "Cluster" in df.columns

        panel._run_astir()
        assert "ASTIR_CellType" in df.columns

        panel._run_maps()
        assert "MAPS_CellType" in df.columns

        # MarkerExpressionWidget & Dialog
        mw = IrisMarkerExpressionWidget()
        mw.set_data(df=df, markers=markers, celltype_col="Cluster")
        dlg = MarkerExpressionDialog(
            df=df,
            markers=markers,
            celltype_col="Cluster",
            current_dataset="All Samples",
        )
        assert dlg.cached_matrix is not None
        assert list(dlg.cached_matrix.columns) == markers

        dlg.close()
        mw.close()
        panel.close()

    def test_joint_nan_filtering_prevents_misalignment(self, qapp):
        from iris_viewer.widgets.phenotyping.manual_gating import ManualGatingWidget

        # Row 0: X is NaN, Y is 100. Row 1: X is 100, Y is NaN.
        # Rows 2-5: jointly finite.
        df = pd.DataFrame(
            {
                "CD3": [np.nan, 100.0, 0.0, 0.0, 10.0, 10.0],
                "CD8": [100.0, np.nan, 0.0, 0.0, 10.0, 10.0],
                "DAPI": [1.0, 1.0, 1.0, np.nan, 1.0, 1.0],
            }
        )
        state = ViewerAppState()
        state.features_df = df
        ctx = AppContext(state=state)
        mg = ManualGatingWidget(ctx=ctx)
        mg.set_data(df=df, markers=["CD3", "CD8", "DAPI"])
        mg.combo_scale.setCurrentText("Linear (Raw)")
        mg.spin_tx.setValue(5.0)
        mg.spin_ty.setValue(5.0)

        # In preview, only 4 jointly finite rows (rows 2..5) remain;
        # row 0 and row 1 must NOT be paired into a false (100, 100) cell!
        assert "Q2 (CD3+ CD8+): 2 " in mg.lbl_quad_info.text()

        mg.open_popup_window(0)
        studio = mg.popup_dialog
        studio.combo_scale.setCurrentText("Linear (Raw)")
        studio.spin_tx.setValue(5.0)
        studio.spin_ty.setValue(5.0)
        studio._render_plot()
        # Q2 cell text in quadrant table should show exactly 2 cells (50.0%), NOT 3!
        assert studio.table_quads.item(1, 1).text().startswith("2 (50.0%)")

        # Test 3D mode joint filtering doesn't raise and aligns all 3 axes
        studio.combo_mode.setCurrentText("3D Trivariate Scatter")
        studio._render_plot()
        coll = studio.ax.collections[0]
        assert len(coll.get_offsets()) == 3  # rows 2, 4, 5 have finite DAPI as well

        # Test 1D mode
        studio.combo_mode.setCurrentText("1D Histogram")
        studio._render_plot()

        studio.close()
        mg.close()

    def test_manual_gating_respects_combo_scale_and_view_tree_tab(self, qapp):
        from iris_viewer.widgets.phenotyping.manual_gating import ManualGatingWidget

        df = pd.DataFrame({"CD3": [0.0, 0.0, 100.0, 100.0], "CD8": [0.0, 0.0, 100.0, 100.0]})
        state = ViewerAppState()
        state.features_df = df
        ctx = AppContext(state=state)
        mg = ManualGatingWidget(ctx=ctx)
        mg.set_data(df=df, markers=["CD3", "CD8"])

        mg.combo_scale.setCurrentText("Linear (Raw)")
        assert mg.spin_tx.value() > 10.0  # linear threshold sits near ~50, not arcsinh ~2

        mg.combo_scale.setCurrentText("Min-Max Normalized")
        assert 0.0 <= mg.spin_tx.value() <= 1.0

        # Click Hierarchy Tree button -> opens tab index 1
        mg.btn_view_tree.click()
        assert mg.popup_dialog is not None
        assert mg.popup_dialog.tabs.currentIndex() == 1

        # Click Pop-up Window button -> opens tab index 0
        mg.btn_popout.click()
        assert mg.popup_dialog.tabs.currentIndex() == 0

        mg.popup_dialog.close()
        mg.close()

    def test_marker_expression_dialog_export_csv_zscore_vs_raw(self, qapp, tmp_path):
        from iris_viewer.widgets.marker_expression.dialog import MarkerExpressionDialog

        df = pd.DataFrame(
            {
                "Cluster": ["C0", "C0", "C1", "C1", "C2", "C2"],
                "CD3_mean": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0],
                "CD8_mean": [100.0, 100.0, 200.0, 200.0, 300.0, 300.0],
            }
        )
        dlg = MarkerExpressionDialog(
            df=df,
            markers=["CD3", "CD8"],
            celltype_col="Cluster",
            current_dataset="All Samples",
            current_mode="Z-Score (Per Marker)",
        )

        z_csv = tmp_path / "zscore.csv"
        with (
            patch.object(QFileDialog, "getSaveFileName", return_value=(str(z_csv), "")),
            patch.object(QMessageBox, "information"),
        ):
            dlg.export_csv()

        z_df = pd.read_csv(z_csv, index_col=0)
        assert z_df["CD3"].mean() == pytest.approx(0.0, abs=1e-6)
        assert z_df["CD8"].std(ddof=0) == pytest.approx(1.0, abs=1e-6)

        dlg.combo_mode.setCurrentText("Raw Mean Intensity")
        raw_csv = tmp_path / "raw.csv"
        with (
            patch.object(QFileDialog, "getSaveFileName", return_value=(str(raw_csv), "")),
            patch.object(QMessageBox, "information"),
        ):
            dlg.export_csv()

        raw_df = pd.read_csv(raw_csv, index_col=0)
        np.testing.assert_allclose(raw_df["CD3"].to_numpy(), [10.0, 20.0, 30.0])
        np.testing.assert_allclose(raw_df["CD8"].to_numpy(), [100.0, 200.0, 300.0])
        dlg.close()


# 4. Window Flags, Unwired UI Controls, Scalebar & Async Workers


class TestUIWiringWindowFlagsScalebarAndWorkers:
    def test_popup_dialogs_have_qt_window_flag(self, qapp):
        from iris_viewer.widgets.phenotyping.gating_studio_dialog import GatingStudioDialog
        from iris_viewer.widgets.phenotyping.sample_impact_dialog import SampleImpactDialog
        from iris_viewer.widgets.segmentation_studio_dialog import SegmentationStudioDialog

        for cls in (SegmentationStudioDialog, GatingStudioDialog, SampleImpactDialog):
            dlg = cls()
            assert bool(dlg.windowFlags() & Qt.Window), f"{cls.__name__} missing Qt.Window flag"
            dlg.close()

    def test_sample_impact_dialog_sample_selected_wired(self, qapp):
        from iris_viewer.widgets.image_control import ImageControlWidget
        from iris_viewer.widgets.phenotyping.manual_gating import ManualGatingWidget

        df = _make_sample_features_df()
        state = ViewerAppState()
        state.features_df = df
        cw = ImageControlWidget(viewer=None)
        cw.update_sample_list(["Sample_A", "Sample_B"])
        assert cw.combo_sample.currentText() == "Sample_A"

        ctx = AppContext(state=state, control_widget=cw)
        mg = ManualGatingWidget(ctx=ctx)
        mg.set_data(df=df, markers=["DAPI", "CD3", "CD8"])

        mg.open_impact_window()
        mg.impact_dialog.sampleSelected.emit("Sample_B")
        assert cw.combo_sample.currentText() == "Sample_B"

        mg.open_popup_window(0)
        mg.popup_dialog._open_impact_window()
        mg.popup_dialog.impact_dialog.sampleSelected.emit("Sample_A")
        assert cw.combo_sample.currentText() == "Sample_A"

        mg.popup_dialog.impact_dialog.close()
        mg.popup_dialog.close()
        mg.impact_dialog.close()
        mg.close()
        cw.close()

    def test_consensus_and_studio_undo_override_wired(self, qapp):
        from iris_viewer.app.controller import ViewerController
        from iris_viewer.widgets.consensus_panel import ConsensusPanelWidget
        from iris_viewer.widgets.segmentation_studio_dialog import SegmentationStudioDialog

        cp = ConsensusPanelWidget()
        review_svc = MagicMock()
        dummy_ctrl = types.SimpleNamespace(
            consensus_panel=cp,
            review_svc=review_svc,
            legend_widget=MagicMock(),
            _generate_consensus=MagicMock(),
            _toggle_interactive_review=MagicMock(),
            _select_layer=MagicMock(),
        )
        ViewerController._bind_consensus_signals(dummy_ctrl)

        cp.btn_undo_override.click()
        review_svc.undo_override.assert_called_once()

        ctx = AppContext(review_svc=review_svc)
        studio = SegmentationStudioDialog(ctx=ctx, consensus_panel=cp)
        studio.btn_undo_override.click()
        assert review_svc.undo_override.call_count == 2

        studio.close()
        cp.close()

    def test_neighborhood_export_buttons_wired(self, qapp, tmp_path):
        from iris_viewer.widgets.neighborhood import IrisCellularNeighborhoodsWidget

        nw = IrisCellularNeighborhoodsWidget()
        assert hasattr(nw, "btn_export_plot")
        assert hasattr(nw, "btn_export_csv")
        assert nw.btn_export_plot.text() == "Export Plot..."
        assert nw.btn_export_csv.text() == "Export CSV..."

        centers = np.array([[0.5, 0.5], [0.2, 0.8]])
        labels = ["Tumor", "T cell"]
        df = pd.DataFrame(
            {"Cellular_Neighborhood": ["Niche 0", "Niche 1"], "cell_type": ["Tumor", "T cell"]}
        )
        nw.initialize_metadata_table(2)
        nw.display_niche_compositions(centers, labels, updated_df=df)

        plot_out = tmp_path / "niche_plot.png"
        csv_out = tmp_path / "niche_profiles.csv"

        with (
            patch.object(QFileDialog, "getSaveFileName", return_value=(str(plot_out), "")),
            patch.object(QMessageBox, "information"),
        ):
            nw.btn_export_plot.click()
        assert plot_out.is_file()

        with (
            patch.object(QFileDialog, "getSaveFileName", return_value=(str(csv_out), "")),
            patch.object(QMessageBox, "information"),
        ):
            nw.btn_export_csv.click()
        assert csv_out.is_file()
        nw.close()

    def test_image_control_load_mask_and_session_buttons(self, qapp):
        from iris_viewer.widgets.image_control import ImageControlWidget

        cw = ImageControlWidget(viewer=None)
        assert cw.btn_load_mask.text() == "Load Mask File"
        assert cw.btn_save_session.text() == "Save Session"
        assert cw.btn_load_session.text() == "Load Session"

        for btn in cw.findChildren(QPushButton):
            assert "..." not in btn.text() and "…" not in btn.text()

        got_mask = []
        got_save = []
        got_load = []
        cw.loadExistingMaskTriggered.connect(got_mask.append)
        cw.saveSessionTriggered.connect(lambda: got_save.append(True))
        cw.loadSessionTriggered.connect(lambda: got_load.append(True))

        with patch.object(QFileDialog, "getOpenFileName", return_value=("/tmp/mask.tiff", "")):
            cw.btn_load_mask.click()
        cw.btn_save_session.click()
        cw.btn_load_session.click()

        assert got_mask == ["/tmp/mask.tiff"]
        assert got_save == [True]
        assert got_load == [True]
        cw.close()

    def test_canvas_scalebar_calibrated_at_scale_one(self, qapp):
        from iris_viewer.widgets.canvas_scalebar import CanvasScaleBarOverlay

        fake_camera = types.SimpleNamespace(
            zoom=1.0,
            events=types.SimpleNamespace(zoom=MagicMock()),
        )
        fake_viewer = types.SimpleNamespace(
            window=types.SimpleNamespace(_qt_viewer=None),
            scene=types.SimpleNamespace(camera=fake_camera),
        )
        sb = CanvasScaleBarOverlay(fake_viewer)
        assert sb._calibrated is True
        sb.set_physical_scale(1.0)
        assert sb.lbl_text.text().endswith("µm")

        sb.set_physical_scale(1.0, calibrated=False)
        assert sb.lbl_text.text().endswith("px")
        sb.set_calibrated(True)
        assert sb.lbl_text.text().endswith("µm")
        sb.close()

    def test_phenotyper_panel_async_workers_and_sync_fallback(self, qapp, monkeypatch):
        import napari.qt

        from iris_viewer.widgets.phenotyping.panel import PhenotyperPanelWidget

        df = _make_sample_features_df()
        markers = ["DAPI", "CD3", "CD8"]

        # 1. Synchronous fallback when ctx is None
        panel_sync = PhenotyperPanelWidget(ctx=None)
        panel_sync.manual_gating.set_data(df=df, markers=markers)
        # Provide active df via monkeypatch on manual_gating
        panel_sync.manual_gating.get_active_dataframe = lambda: df
        panel_sync._sync_clustering_markers()
        panel_sync.combo_cluster_method.setCurrentText("KMeans")
        panel_sync.spin_kmeans_k.setValue(2)
        panel_sync._run_clustering()
        assert panel_sync.btn_run_cluster.isEnabled() is True
        assert "Cluster" in df.columns

        panel_sync._run_astir()
        assert panel_sync.btn_run_astir.isEnabled() is True
        assert "ASTIR_CellType" in df.columns

        panel_sync._run_maps()
        assert panel_sync.btn_run_maps.isEnabled() is True
        assert "MAPS_CellType" in df.columns
        panel_sync.close()

        # 2. Async dispatch via thread_worker when ctx & track_worker are present
        tracked = []

        class _Sig:
            def __init__(self):
                self._cbs = []

            def connect(self, cb):
                self._cbs.append(cb)

            def emit(self, *args):
                for cb in self._cbs:
                    cb(*args)

        class _FakeWorker:
            def __init__(self, fn):
                self._fn = fn
                self.returned = _Sig()
                self.finished = _Sig()
                self.errored = _Sig()

            def start(self):
                try:
                    res = self._fn()
                except Exception as exc:
                    self.errored.emit(exc)
                else:
                    self.returned.emit(res)
                finally:
                    self.finished.emit()

        def _fake_thread_worker(fn):
            return lambda: _FakeWorker(fn)

        def _track_worker(worker, on_error=None, **_kwargs):
            tracked.append(worker)
            if on_error:
                worker.errored.connect(on_error)
            return worker

        df_async = _make_sample_features_df()
        state = ViewerAppState()
        state.features_df = df_async
        ctx = AppContext(state=state, track_worker=_track_worker)

        panel_async = PhenotyperPanelWidget(ctx=ctx)
        panel_async.set_data(df=df_async, markers=markers)
        panel_async.combo_cluster_method.setCurrentText("KMeans")
        panel_async.spin_kmeans_k.setValue(2)

        monkeypatch.setattr(napari.qt, "thread_worker", _fake_thread_worker)
        panel_async._run_clustering()
        panel_async._run_astir()
        panel_async._run_maps()

        assert len(tracked) == 3
        assert "Cluster" in df_async.columns
        assert "ASTIR_CellType" in df_async.columns
        assert "MAPS_CellType" in df_async.columns
        assert panel_async.btn_run_cluster.isEnabled() is True
        assert panel_async.btn_run_astir.isEnabled() is True
        assert panel_async.btn_run_maps.isEnabled() is True

        # 3. Verify error callback re-enables buttons on failure
        def _failing_astir(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "iris_viewer.engine.phenotyping.astir_backend.ASTIRBackend.predict",
            _failing_astir,
        )
        panel_async._run_astir()
        assert panel_async.btn_run_astir.isEnabled() is True
        assert "error" in panel_async.lbl_astir_status.text().lower()
        panel_async.close()
