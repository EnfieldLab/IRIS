"""
Tests for iris_viewer.engine.phenotyping: PhenotypingEnsemble, ASTIRBackend, MAPSBackend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iris_viewer.engine.phenotyping.ensemble import PhenotypingEnsemble
from iris_viewer.engine.phenotyping.astir_backend import ASTIRBackend
from iris_viewer.engine.phenotyping.maps_backend import MAPSBackend
from iris_viewer.core.types import CellTypePanel, CellTypeRule, MarkerPanel


def _make_predictions(n_cells: int = 10, cell_types: list = None) -> dict:
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

    def test_soft_vote_returns_all_cells(self):
        preds = _make_predictions(20)
        ens = self._ensemble()
        final, conf, per_method = ens.vote(preds, strategy="soft_vote")
        assert set(final.keys()) == set(range(1, 21))
        assert set(conf.keys()) == set(range(1, 21))

    def test_soft_vote_confidence_in_range(self):
        preds = _make_predictions(30)
        ens = self._ensemble()
        _, conf, _ = ens.vote(preds, strategy="soft_vote")
        for c in conf.values():
            assert 0.0 <= c <= 1.0

    def test_weighted_majority_respects_weights(self):
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

    def test_borda_count_returns_all_cells(self):
        preds = _make_predictions(10)
        ens = self._ensemble()
        final, conf, _ = ens.vote(preds, strategy="borda_count")
        assert len(final) == 10

    def test_min_probability_threshold_produces_unresolved(self):
        """With threshold=1.0 nothing should pass -> all Unresolved."""
        preds = _make_predictions(5)
        ens = PhenotypingEnsemble(min_probability_threshold=1.0)
        final, _, _ = ens.vote(preds, strategy="soft_vote")
        assert all(v == PhenotypingEnsemble.UNRESOLVED_LABEL for v in final.values())

    def test_invalid_strategy_raises(self):
        preds = _make_predictions(5)
        ens = self._ensemble()
        with pytest.raises(ValueError, match="Unknown strategy"):
            ens.vote(preds, strategy="magic_vote")

    def test_empty_predictions_raises(self):
        ens = self._ensemble()
        with pytest.raises(ValueError):
            ens.vote({}, strategy="soft_vote")

    def test_per_method_breakdown_contains_all_methods(self):
        preds = _make_predictions(5)
        ens = self._ensemble()
        _, _, per_method = ens.vote(preds, strategy="soft_vote")
        for cid, breakdown in per_method.items():
            assert "MethodA" in breakdown
            assert "MethodB" in breakdown


class TestASTIRBackend:
    """Tests for ASTIR backend integration."""

    def test_availability_reflects_import(self):
        b = ASTIRBackend()
        try:
            import astir  # noqa: F401
            assert b.is_available() is True
        except ImportError:
            assert b.is_available() is False

    def test_predict_without_fit_raises_or_runs(self):
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


class TestMAPSFallback:
    """Tests for MAPS deep learning architecture."""

    def test_maps_attributes(self):
        b = MAPSBackend()
        assert b.name == "MAPS"
        assert b.requires_spatial is True
        assert b.requires_training is False

    def test_maps_predict_fallback_without_torch(self):
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
