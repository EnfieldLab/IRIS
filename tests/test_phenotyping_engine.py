"""
Tests for iris_viewer.engine.phenotyping: PhenotypingEnsemble, ASTIRBackend, MAPSBackend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iris_viewer.core.types import CellTypePanel, CellTypeRule
from iris_viewer.engine.phenotyping.astir_backend import ASTIRBackend
from iris_viewer.engine.phenotyping.ensemble import PhenotypingEnsemble
from iris_viewer.engine.phenotyping.maps_backend import MAPSBackend


def _make_predictions(n_cells: int = 10, cell_types: list[str] | None = None) -> dict:
    """Generate synthetic predictions for testing ensemble voting."""
    if cell_types is None:
        cell_types = ["Tumor", "T cell", "Macrophage"]
    rng = np.random.default_rng(42)

    preds_a = {}
    preds_b = {}
    for cid in range(1, n_cells + 1):
        probs_a = rng.dirichlet(np.ones(len(cell_types)))
        probs_b = rng.dirichlet(np.ones(len(cell_types)))
        preds_a[cid] = dict(zip(cell_types, probs_a))
        preds_b[cid] = dict(zip(cell_types, probs_b))

    return {"MethodA": preds_a, "MethodB": preds_b}


class TestPhenotypingEnsemble:
    """Tests for multi-method phenotyping ensemble voting strategies."""

    def _ensemble(self, min_conf: float = 0.0) -> PhenotypingEnsemble:
        return PhenotypingEnsemble(min_probability_threshold=min_conf)

    def test_consensus_phenotyping_soft_vote_returns_all_cells(self):
        preds = _make_predictions(20)
        ens = self._ensemble()
        final, conf, _per_method = ens.vote(preds, strategy="soft_vote")
        assert set(final.keys()) == set(range(1, 21))
        assert set(conf.keys()) == set(range(1, 21))

    def test_consensus_phenotyping_soft_vote_confidence_in_range(self):
        preds = _make_predictions(30)
        ens = self._ensemble()
        _, conf, _ = ens.vote(preds, strategy="soft_vote")
        for c in conf.values():
            assert 0.0 <= c <= 1.0

    def test_consensus_phenotyping_weighted_majority_respects_weights(self):
        """When one method has weight=100x, it should always win."""
        cell_ids = [1, 2, 3]
        preds = {
            "High": {cid: {"Tumor": 0.9, "T cell": 0.1} for cid in cell_ids},
            "Low":  {cid: {"Tumor": 0.1, "T cell": 0.9} for cid in cell_ids},
        }
        ens = PhenotypingEnsemble(method_weights={"High": 100.0, "Low": 1.0})
        final, _, _ = ens.vote(preds, strategy="weighted_majority")
        for cid in cell_ids:
            assert final[cid] == "Tumor"

    def test_consensus_phenotyping_borda_count_returns_all_cells(self):
        preds = _make_predictions(10)
        ens = self._ensemble()
        final, _conf, _ = ens.vote(preds, strategy="borda_count")
        assert len(final) == 10

    def test_consensus_phenotyping_min_probability_threshold_produces_unresolved(self):
        """With threshold=1.0 nothing should pass -> all Unresolved."""
        preds = _make_predictions(5)
        ens = PhenotypingEnsemble(min_probability_threshold=1.0)
        final, _, _ = ens.vote(preds, strategy="soft_vote")
        assert all(v == PhenotypingEnsemble.UNRESOLVED_LABEL for v in final.values())

    def test_consensus_phenotyping_invalid_strategy_raises(self):
        preds = _make_predictions(5)
        ens = self._ensemble()
        with pytest.raises(ValueError, match="Unknown strategy"):
            ens.vote(preds, strategy="magic_vote")

    def test_consensus_phenotyping_empty_predictions_raises(self):
        ens = self._ensemble()
        with pytest.raises(ValueError):
            ens.vote({}, strategy="soft_vote")

    def test_consensus_phenotyping_per_method_breakdown_contains_all_methods(self):
        preds = _make_predictions(5)
        ens = self._ensemble()
        _, _, per_method = ens.vote(preds, strategy="soft_vote")
        for breakdown in per_method.values():
            assert "MethodA" in breakdown
            assert "MethodB" in breakdown


class TestASTIRBackend:
    """Tests for ASTIR backend integration."""

    def test_astir_backend_availability_reflects_import(self):
        b = ASTIRBackend()
        try:
            import astir  # noqa: F401
            assert b.is_available() is True
        except ImportError:
            assert b.is_available() is False

    def test_astir_backend_predict_without_fit_raises_or_runs(self):
        b = ASTIRBackend()
        if not b.is_available():
            pytest.skip("astir not installed")
        df = pd.DataFrame({
            "Cell_ID": [1, 2],
            "CD3_mean": [1.0, 0.1],
            "CD8_mean": [0.9, 0.05],
        })
        panel = CellTypePanel(rules=[
            CellTypeRule("CD8 T cell", markers_positive=["CD3", "CD8"]),
        ])
        result = b.predict(df, panel)
        assert isinstance(result, dict)
        assert set(result.keys()) == {1, 2}

    def test_astir_backend_multiclass_inference(self):
        b = ASTIRBackend()
        if not b.is_available():
            pytest.skip("astir not installed")
        df = pd.DataFrame({
            "ObjectNumber": [1, 2, 3, 4, 5, 6],
            "CD3_mean": [5.0, 0.1, 4.2, 0.05, 4.8, 0.1],
            "CD8_mean": [4.0, 0.05, 3.8, 0.02, 4.1, 0.08],
            "CD20_mean": [0.1, 4.5, 0.08, 4.2, 0.15, 3.9],
        })
        panel = CellTypePanel(rules=[
            CellTypeRule("CD8_T_cell", markers_positive=["CD3", "CD8"]),
            CellTypeRule("B_cell", markers_positive=["CD20"]),
        ])
        result = b.predict(df, panel, max_epochs=5)
        assert isinstance(result, dict)
        assert len(result) == 6
        for cid in range(1, 7):
            assert "CD8_T_cell" in result[cid]
            assert "B_cell" in result[cid]
            assert sum(result[cid].values()) > 0.0


class TestMAPSFallback:
    """Tests for MAPS deep learning architecture."""

    def test_maps_backend_attributes(self):
        b = MAPSBackend()
        assert b.name == "MAPS"
        assert b.requires_spatial is True
        assert b.requires_training is False

    def test_maps_backend_predict_fallback_without_torch(self):
        """When PyTorch or PyG is missing or basic run, predicts correctly or gracefully falls back."""
        b = MAPSBackend()
        df = pd.DataFrame({
            "Cell_ID": [1, 2, 3],
            "X_centroid": [10.0, 20.0, 30.0],
            "Y_centroid": [10.0, 20.0, 30.0],
            "CD3_mean": [5.0, 0.1, 4.5],
            "CD8_mean": [4.0, 0.05, 0.1],
        })
        panel = CellTypePanel(rules=[
            CellTypeRule("T_cell", markers_positive=["CD3"]),
        ])
        preds = b.predict(df, panel)
        assert isinstance(preds, dict)
        assert len(preds) == 3


class TestUnsupervisedClusteringAndMarkerExpression:
    """Tests for cluster marker selection, cluster preservation, and robust expression profiling."""

    def test_marker_expression_widget_technical_marker_detection(self):
        from iris_viewer.widgets.phenotyping_panel import PhenotyperPanelWidget
        assert PhenotyperPanelWidget._is_technical_channel("DAPI") is True
        assert PhenotyperPanelWidget._is_technical_channel("dapi_nuclei") is True
        assert PhenotyperPanelWidget._is_technical_channel("TRITC") is True
        assert PhenotyperPanelWidget._is_technical_channel("Cy5") is True
        assert PhenotyperPanelWidget._is_technical_channel("Hoechst") is True
        assert PhenotyperPanelWidget._is_technical_channel("Area") is True
        assert PhenotyperPanelWidget._is_technical_channel("Eccentricity") is True
        assert PhenotyperPanelWidget._is_technical_channel("CD3") is False
        assert PhenotyperPanelWidget._is_technical_channel("CD8a") is False
        assert PhenotyperPanelWidget._is_technical_channel("CD20") is False
        assert PhenotyperPanelWidget._is_technical_channel("PanCK") is False
        assert PhenotyperPanelWidget._is_technical_channel("FoxP3") is False

    def test_marker_expression_widget_clustering_preservation(self, qapp):
        from iris_viewer.widgets.marker_expression import MarkerExpressionDialog
        # 8 clusters with multiple samples
        df = pd.DataFrame({
            "Cluster": [f"Cluster {i % 8}" for i in range(160)],
            "Sample": ["Sample_1"] * 80 + ["Sample_2"] * 80,
            "CD3": np.random.rand(160),
            "CD8": np.random.rand(160),
            "CD20": np.random.rand(160),
        })
        dialog = MarkerExpressionDialog(
            df=df,
            markers=["CD3", "CD8", "CD20"],
            celltype_col="Cluster",
            current_dataset="All Samples",
        )
        assert dialog.cached_matrix is not None
        assert len(dialog.cached_matrix) == 8
        for i in range(8):
            assert f"Cluster {i}" in dialog.cached_matrix.index
        # Test dataset scope switching
        dialog.set_current_dataset("All Samples")
        assert len(dialog.cached_matrix) == 8
        dialog.close()

    def test_phenotyper_panel_subpanel_navigation_order(self, qapp):
        from iris_viewer.widgets.phenotyping_panel import PhenotyperPanelWidget
        panel = PhenotyperPanelWidget(ctx=None)

        # Order must be: Manual Gating, Clustering, ASTIR, MAPS
        expected_labels = ["Manual Gating", "Clustering", "ASTIR", "MAPS"]
        actual_labels = [btn.text() for btn in panel.nav_buttons]
        assert actual_labels == expected_labels

        assert panel.stack.widget(0) == panel.manual_gating
        assert panel.stack.widget(1) == panel.cluster_widget
        assert panel.stack.widget(2) == panel.astir_widget
        assert panel.stack.widget(3) == panel.maps_widget
        panel.close()

    def test_phenotyper_panel_interactive_phenotype_assignment_studio(self, qapp):
        from iris_viewer.widgets.marker_expression import MarkerExpressionDialog
        df = pd.DataFrame({
            "Cluster": [f"Cluster {i % 4}" for i in range(80)],
            "Sample": ["Sample_A"] * 80,
            "CD3": np.random.rand(80),
            "CD8": np.random.rand(80),
            "CD20": np.random.rand(80),
        })

        assigned = []

        def on_assigned(clusters, label):
            assigned.append((clusters, label))

        dialog = MarkerExpressionDialog(
            df=df,
            markers=["CD3", "CD8", "CD20"],
            celltype_col="Cluster",
            current_dataset="All Samples",
            assignment_callback=on_assigned,
        )

        # Check cluster dropdown populated
        assert dialog.combo_active_row.count() == 4
        # Defaults to the first cluster
        assert dialog.combo_active_row.currentData() == "Cluster 0"
        assert dialog._active_row == "Cluster 0"
        assert "20 cells" in dialog.lbl_active_cell_count.text()
        assert dialog._active_row_patch is not None

        # Test stepping next and previous
        dialog._select_next_row()
        assert dialog._active_row == "Cluster 1"
        assert dialog.combo_active_row.currentData() == "Cluster 1"

        dialog._select_prev_row()
        assert dialog._active_row == "Cluster 0"

        # Test selecting a specific cluster
        dialog._set_active_row("Cluster 2", sync_combo=True)
        assert dialog.combo_active_row.currentData() == "Cluster 2"
        assert "Cluster 2" in dialog._active_row
        assert dialog._suggested_phenotype != ""

        # Test using suggestion chip
        dialog._use_suggested_label()
        assert dialog.txt_dialog_pheno.text() == dialog._suggested_phenotype

        # Test applying phenotype
        dialog._apply_phenotype_from_dialog()
        assert len(assigned) == 1
        assert assigned[0][0] == ["Cluster 2"]
        assert assigned[0][1] == dialog._suggested_phenotype
        assert "✓ Assigned" in dialog.lbl_dialog_assign_feedback.text()

        # Test click-on-heatmap row selection simulation
        # Simulate click on row 3 (Cluster 3)
        class DummyMouseEvent:
            inaxes = dialog.ax_heatmap
            button = 1
            x = 100
            y = 100
            xdata = 1.0
            ydata = 3.0

        press_evt = DummyMouseEvent()
        dialog._on_mouse_press(press_evt)
        assert dialog._mouse_click_candidate is not None

        release_evt = DummyMouseEvent()
        dialog._on_mouse_release(release_evt)
        assert dialog._active_row == "Cluster 3"
        assert dialog.combo_active_row.currentData() == "Cluster 3"

        dialog.close()

    def test_leiden_clustering_igraph_parameters(self):
        """Verify scanpy leiden executes cleanly with flavor='igraph', n_iterations=2, directed=False or fallback."""
        try:
            import anndata as ad
            import scanpy as sc
        except ImportError:
            pytest.skip("Scanpy/AnnData not installed in test environment")

        import importlib.util
        if not (importlib.util.find_spec("igraph") or importlib.util.find_spec("leidenalg")):
            pytest.skip("Neither igraph nor leidenalg installed in this environment")

        X = np.random.rand(40, 5)
        adata = ad.AnnData(X=X)
        sc.pp.neighbors(adata, n_neighbors=5)
        try:
            sc.tl.leiden(adata, resolution=0.5, flavor="igraph", n_iterations=2, directed=False)
        except (TypeError, ValueError, Exception):
            sc.tl.leiden(adata, resolution=0.5)
        assert "leiden" in adata.obs.columns


class TestModularPackageArchitecture:
    """Verifies that all modularized subpackages maintain clean, backward-compatible API contracts."""

    def test_phenotyping_modular_imports(self):
        # Direct modular subpackage imports
        from iris_viewer.widgets.phenotyping import (
            GatingHierarchyNode,
            GatingStudioDialog,
            ManualGatingWidget,
            PhenotyperPanelWidget,
            SampleImpactDialog,
            arcsinh_transform,
            assess_gating_quality,
            compute_optimal_threshold,
            find_sample_column,
            log10_transform,
        )
        assert GatingHierarchyNode is not None
        assert PhenotyperPanelWidget is not None
        assert ManualGatingWidget is not None
        assert GatingStudioDialog is not None
        assert SampleImpactDialog is not None
        assert arcsinh_transform is not None
        assert assess_gating_quality is not None
        assert compute_optimal_threshold is not None
        assert find_sample_column is not None
        assert log10_transform is not None

        # Re-export facade backwards compatibility
        from iris_viewer.widgets.phenotyping_panel import (
            GatingStudioDialog as FacadeStudio,
        )
        from iris_viewer.widgets.phenotyping_panel import (
            ManualGatingWidget as FacadeManual,
        )
        from iris_viewer.widgets.phenotyping_panel import (
            PhenotyperPanelWidget as FacadePanel,
        )
        from iris_viewer.widgets.phenotyping_panel import (
            SampleImpactDialog as FacadeImpact,
        )
        assert FacadePanel is PhenotyperPanelWidget
        assert FacadeManual is ManualGatingWidget
        assert FacadeStudio is GatingStudioDialog
        assert FacadeImpact is SampleImpactDialog

    def test_marker_expression_modular_imports(self):
        from iris_viewer.widgets.marker_expression import (
            IrisMarkerExpressionWidget,
            MarkerExpressionDialog,
            SampleSubsetDialog,
        )
        from iris_viewer.widgets.marker_expression.dialog import MarkerExpressionDialog as Dlg
        from iris_viewer.widgets.marker_expression.subset_dialog import SampleSubsetDialog as SubDlg
        from iris_viewer.widgets.marker_expression.widget import IrisMarkerExpressionWidget as Wgt

        assert MarkerExpressionDialog is Dlg
        assert SampleSubsetDialog is SubDlg
        assert IrisMarkerExpressionWidget is Wgt

    def test_segmentation_modular_imports(self):
        from iris_viewer.widgets.segmentation_panel import (
            IrisSegmentationWidget,
            MultiSelectButton,
            PipelineBuilderWidget,
            PipelineGroupWidget,
            SegmentationPanelWidget,
            SegmentationStudioDialog,
            StepWidget,
        )
        from iris_viewer.widgets.segmentation_pipeline_builder import (
            MultiSelectButton as MSB,
        )
        from iris_viewer.widgets.segmentation_pipeline_builder import (
            PipelineBuilderWidget as PBW,
        )
        from iris_viewer.widgets.segmentation_pipeline_builder import (
            PipelineGroupWidget as PGW,
        )
        from iris_viewer.widgets.segmentation_pipeline_builder import (
            StepWidget as SW,
        )
        from iris_viewer.widgets.segmentation_studio_dialog import (
            SegmentationStudioDialog as SSD,
        )

        assert MultiSelectButton is MSB
        assert StepWidget is SW
        assert PipelineGroupWidget is PGW
        assert PipelineBuilderWidget is PBW
        assert SegmentationStudioDialog is SSD
        assert IrisSegmentationWidget is SegmentationPanelWidget

    def test_phenotyping_thresholding_and_quality_with_nan_edge_cases(self):
        from iris_viewer.widgets.phenotyping.thresholding import (
            arcsinh_transform,
            assess_gating_quality,
            compute_optimal_threshold,
        )

        # Vector with NaNs, Infs, and zero variance
        v_dirty = np.array([np.nan, 1.0, 1.0, 2.0, 2.0, np.inf, 3.0, 3.0, np.nan])
        v_clean = arcsinh_transform(v_dirty)
        assert len(v_clean) == len(v_dirty)

        # Otsu should produce a finite float threshold despite NaNs
        th = compute_optimal_threshold(v_clean, method="otsu")
        assert np.isfinite(th)

        # Quality readout should produce robust finite score
        label, color, score = assess_gating_quality(v_clean, th)
        assert isinstance(label, str)
        assert color.startswith("#")
        assert np.isfinite(score)


