"""
Tests for iris-phenotyper:
- PhenotypingEnsemble (all three strategies)
- CellTypePanel / MarkerPanel conversion helpers
- CELESTABackend fallback classifier
- PhenotyperAppState session persistence and undo
No Qt or napari required.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_features(n_cells: int = 20) -> pd.DataFrame:
    """Minimal features DataFrame with DAPI, CD45, CK, CD3, CD68 channels."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "ObjectNumber": range(1, n_cells + 1),
        "Pos_X": rng.uniform(0, 100, n_cells),
        "Pos_Y": rng.uniform(0, 100, n_cells),
        "DAPI_mean":  rng.uniform(100, 4000, n_cells),
        "CD45_mean":  rng.uniform(0, 500, n_cells),
        "CK_mean":    rng.uniform(0, 500, n_cells),
        "CD3_mean":   rng.uniform(0, 500, n_cells),
        "CD68_mean":  rng.uniform(0, 500, n_cells),
    })
    return df


def _make_predictions(n_cells: int = 20) -> dict:
    """Create two sets of stub predictions for testing."""
    rng = np.random.default_rng(0)
    cell_types = ["Tumor", "T cell", "Macrophage"]
    cell_ids = list(range(1, n_cells + 1))

    preds_a: dict = {}
    preds_b: dict = {}
    for cid in cell_ids:
        raw_a = rng.dirichlet(np.ones(3))
        raw_b = rng.dirichlet(np.ones(3))
        preds_a[cid] = dict(zip(cell_types, raw_a.tolist()))
        preds_b[cid] = dict(zip(cell_types, raw_b.tolist()))

    return {"MethodA": preds_a, "MethodB": preds_b}


# ── PhenotypingEnsemble ───────────────────────────────────────────────────────

class TestPhenotypingEnsemble:
    from iris_phenotyper.engine.ensemble import PhenotypingEnsemble

    def _ensemble(self, min_conf=0.0):
        from iris_phenotyper.engine.ensemble import PhenotypingEnsemble
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
        from iris_phenotyper.engine.ensemble import PhenotypingEnsemble
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
        """With threshold=1.0 nothing should pass → all Unresolved."""
        preds = _make_predictions(5)
        from iris_phenotyper.engine.ensemble import PhenotypingEnsemble
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
            ens.vote({})

    def test_per_method_breakdown_has_both_methods(self):
        preds = _make_predictions(5)
        ens = self._ensemble()
        _, _, per_method = ens.vote(preds, strategy="soft_vote")
        for breakdown in per_method.values():
            assert "MethodA" in breakdown or "MethodB" in breakdown

    def test_results_to_dataframe_adds_columns(self):
        preds = _make_predictions(20)
        ens = self._ensemble()
        final, conf, per_method = ens.vote(preds, strategy="soft_vote")
        features = _make_features(20)
        df = ens.results_to_dataframe(features, final, conf, per_method)
        assert "cell_type" in df.columns
        assert "cell_type_confidence" in df.columns

    def test_single_method_soft_vote(self):
        """Single-method predictions should still produce valid output."""
        preds = {"OnlyMethod": _make_predictions(5)["MethodA"]}
        ens = self._ensemble()
        final, conf, _ = ens.vote(preds, strategy="soft_vote")
        assert len(final) == 5


# ── CellTypePanel / MarkerPanel ───────────────────────────────────────────────

class TestCellTypePanel:

    def _panel(self):
        from iris_common.types import CellTypePanel, CellTypeRule, MarkerPanel
        rules = [
            CellTypeRule("Tumor",     markers_positive=["CK"],       markers_negative=["CD45"]),
            CellTypeRule("T cell",    markers_positive=["CD3"],       markers_negative=[]),
            CellTypeRule("Macrophage", markers_positive=["CD68"],     markers_negative=[]),
        ]
        return CellTypePanel(rules=rules)

    def test_to_astir_dict_returns_correct_keys(self):
        from iris_common.types import MarkerPanel
        panel = self._panel()
        mp = MarkerPanel(markers=["DAPI", "CD45", "CK", "CD3", "CD68"])
        d = panel.to_astir_dict(mp)
        assert "Tumor" in d
        assert "CK" in d["Tumor"]
        assert "T cell" in d
        assert "CD3" in d["T cell"]

    def test_to_astir_dict_excludes_absent_markers(self):
        from iris_common.types import MarkerPanel
        panel = self._panel()
        mp = MarkerPanel(markers=["DAPI"])   # none of the markers present
        d = panel.to_astir_dict(mp)
        assert all(len(v) == 0 for v in d.values())

    def test_to_celesta_anchor_matrix_shape(self):
        from iris_common.types import MarkerPanel
        panel = self._panel()
        mp = MarkerPanel(markers=["DAPI", "CD45", "CK", "CD3", "CD68"])
        anchor = panel.to_celesta_anchor_matrix(mp)
        # shape = (n_cell_types, n_markers + 1 parent column)
        assert anchor.shape[0] == 3  # 3 cell types
        assert "parent" in anchor.columns
        assert all(m in anchor.columns for m in mp.markers)

    def test_to_celesta_anchor_positive_equals_1(self):
        from iris_common.types import MarkerPanel
        panel = self._panel()
        mp = MarkerPanel(markers=["CK", "CD45"])
        anchor = panel.to_celesta_anchor_matrix(mp)
        # Tumor: CK=1, CD45=0 (negative marker)
        assert anchor.loc["Tumor", "CK"] == 1
        assert anchor.loc["Tumor", "CD45"] == 0

    def test_marker_panel_index_of(self):
        from iris_common.types import MarkerPanel
        mp = MarkerPanel(markers=["DAPI", "CD45", "CK"])
        assert mp.index_of("CD45") == 1
        assert mp.index_of("ck") == 2  # case-insensitive

    def test_marker_panel_index_of_missing_raises(self):
        from iris_common.types import MarkerPanel
        mp = MarkerPanel(markers=["DAPI"])
        with pytest.raises(ValueError):
            mp.index_of("CD3")


# ── CELESTA Fallback Classifier ───────────────────────────────────────────────

class TestCELESTAFallback:
    """Test the pure-Python rule-based fallback (no celesta install required)."""

    def _backend_and_panel(self):
        from iris_common.types import CellTypePanel, CellTypeRule, MarkerPanel
        from iris_phenotyper.engine.phenotyping.celesta_backend import CELESTABackend
        rules = [
            CellTypeRule("Tumor",     markers_positive=["CK"],  markers_negative=["CD45"]),
            CellTypeRule("Immune",    markers_positive=["CD45"], markers_negative=["CK"]),
            CellTypeRule("Macrophage", markers_positive=["CD68"], markers_negative=[]),
        ]
        panel = CellTypePanel(rules=rules)
        mp = MarkerPanel(markers=["DAPI", "CD45", "CK", "CD68"])
        return CELESTABackend(), panel, mp

    def test_fallback_returns_all_cells(self):
        backend, panel, mp = self._backend_and_panel()
        features = _make_features(20)
        result = backend.predict_fallback(features, cell_type_panel=panel, marker_panel=mp)
        assert len(result) == 20

    def test_fallback_probs_sum_to_one(self):
        backend, panel, mp = self._backend_and_panel()
        features = _make_features(10)
        result = backend.predict_fallback(features, cell_type_panel=panel, marker_panel=mp)
        for probs in result.values():
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-5

    def test_fallback_all_types_present_in_output(self):
        backend, panel, mp = self._backend_and_panel()
        features = _make_features(10)
        result = backend.predict_fallback(features, cell_type_panel=panel, marker_panel=mp)
        expected_types = {"Tumor", "Immune", "Macrophage"}
        for probs in result.values():
            assert expected_types == set(probs.keys())


# ── PhenotyperAppState ────────────────────────────────────────────────────────

class TestPhenotyperState:

    def _state(self):
        from iris_phenotyper.core.state import PhenotyperAppState
        return PhenotyperAppState()

    def test_snapshot_and_undo(self):
        st = self._state()
        st.final_cell_types = {1: "Tumor", 2: "T cell"}
        st.snapshot_override_state()
        st.final_cell_types[1] = "Macrophage"
        assert st.undo_override() is True
        assert st.final_cell_types[1] == "Tumor"

    def test_undo_when_empty_returns_false(self):
        st = self._state()
        assert st.undo_override() is False

    def test_session_roundtrip(self):
        st = self._state()
        st.final_cell_types = {1: "Tumor", 2: "T cell"}
        st.cell_type_confidence = {1: 0.9, 2: 0.4}
        st.cell_annotations = {5: "Macrophage"}

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "phenotyper_session.json"
            st.save_session(p)
            restored, _ = st.load_session(p)

        assert restored.final_cell_types == {1: "Tumor", 2: "T cell"}
        assert restored.cell_type_confidence == {1: 0.9, 2: 0.4}
        assert restored.cell_annotations == {5: "Macrophage"}

    def test_reset_sample_state(self):
        st = self._state()
        st.final_cell_types = {1: "Tumor"}
        st.reset_sample_state()
        assert st.final_cell_types == {}
        assert st.features_df is None
