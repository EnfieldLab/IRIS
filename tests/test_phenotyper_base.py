"""
Tests for iris_phenotyper.base and iris_phenotyper.evaluation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from iris_phenotyper.base import PhenotyperMethodBase, PhenotypingPredictionResult
from iris_phenotyper.evaluation import (
    EvaluationResult,
    evaluate_predictions,
    format_benchmark_report,
)


class _DummyPhenotyper(PhenotyperMethodBase):
    name = "DummyTab"
    input_modality = "tabular"
    requires_training = False

    def is_available(self) -> bool:
        return True

    def get_default_params(self) -> Dict[str, Any]:
        return {"threshold": 0.5}

    def fit(self, *args: Any, **kwargs: Any) -> "_DummyPhenotyper":
        return self

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> PhenotypingPredictionResult:
        ids = self.extract_object_ids(df)
        probs = pd.DataFrame(
            {
                "Tumor": np.where(df["CK_mean"] > 1.0, 0.9, 0.1),
                "T cell": np.where(df["CK_mean"] > 1.0, 0.1, 0.9),
            },
            index=ids,
        )
        labels = probs.idxmax(axis=1)
        conf = probs.max(axis=1)
        return PhenotypingPredictionResult(
            method_name=self.name,
            labels=labels,
            probabilities=probs,
            confidence=conf,
            metadata={"n_cells": len(df)},
        )


class TestPhenotyperBaseContract:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            PhenotyperMethodBase()  # type: ignore[abstract]

    def test_subclass_prediction_result_roundtrip(self):
        df = pd.DataFrame(
            {
                "ObjectNumber": [10, 20, 30],
                "CK_mean": [5.0, 0.2, 4.5],
            }
        )
        model = _DummyPhenotyper()
        assert model.is_available() is True
        assert model.fit(df) is model

        res = model.predict(df)
        res.validate()
        assert res.method_name == "DummyTab"
        assert res.labels.tolist() == ["Tumor", "T cell", "Tumor"]
        assert list(res.labels.index) == [10, 20, 30]
        assert res.confidence.tolist() == pytest.approx([0.9, 0.9, 0.9])

        out_df = res.to_dataframe()
        assert out_df["ObjectNumber"].tolist() == [10, 20, 30]
        assert out_df["cell_type"].tolist() == ["Tumor", "T cell", "Tumor"]
        assert "prob_Tumor" in out_df.columns

        legacy = res.to_legacy_dict()
        assert set(legacy.keys()) == {10, 20, 30}
        assert legacy[10]["Tumor"] == pytest.approx(0.9)

    def test_extract_object_ids_fallbacks(self):
        df_alias = pd.DataFrame({"Cell_ID": [5, 6]})
        assert PhenotyperMethodBase.extract_object_ids(df_alias).tolist() == [5, 6]

        df_no_id = pd.DataFrame({"CD3_mean": [1.0, 2.0, 3.0]})
        assert PhenotyperMethodBase.extract_object_ids(df_no_id).tolist() == [1, 2, 3]

    def test_validation_catches_bad_probabilities_and_index_mismatch(self):
        idx = pd.Index([1, 2], name="ObjectNumber")
        bad_prob = PhenotypingPredictionResult(
            method_name="Test",
            labels=pd.Series(["A", "B"], index=idx),
            probabilities=pd.DataFrame({"A": [1.2, 0.4], "B": [-0.2, 0.6]}, index=idx),
            confidence=pd.Series([1.0, 0.6], index=idx),
        )
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            bad_prob.validate()

        bad_idx = PhenotypingPredictionResult(
            method_name="Test",
            labels=pd.Series(["A", "B"], index=idx),
            probabilities=None,
            confidence=pd.Series([0.9, 0.8], index=pd.Index([1, 3])),
        )
        with pytest.raises(ValueError, match="must match"):
            bad_idx.validate()


class TestEvaluationAndReporting:
    def test_evaluate_predictions_metrics_and_probabilities(self):
        y_true = pd.Series(["Tumor", "Tumor", "T cell", "T cell", "B cell", "B cell"], index=[1, 2, 3, 4, 5, 6])
        y_pred = pd.Series(["Tumor", "T cell", "T cell", "T cell", "B cell", "B cell"], index=[1, 2, 3, 4, 5, 6])
        y_prob = pd.DataFrame(
            {
                "B cell": [0.05, 0.10, 0.05, 0.05, 0.90, 0.85],
                "T cell": [0.05, 0.60, 0.90, 0.85, 0.05, 0.10],
                "Tumor": [0.90, 0.30, 0.05, 0.10, 0.05, 0.05],
            },
            index=[1, 2, 3, 4, 5, 6],
        )
        res = evaluate_predictions(y_true, y_pred, y_prob=y_prob)
        assert isinstance(res, EvaluationResult)
        assert res.accuracy == pytest.approx(5.0 / 6.0)
        assert res["accuracy"] == pytest.approx(5.0 / 6.0)
        assert res.get("n_samples") == 6
        assert "macro_f1" in res
        assert 0.0 <= res.balanced_accuracy <= 1.0
        assert 0.0 <= res.macro_f1 <= 1.0
        assert 0.0 <= res.weighted_f1 <= 1.0
        assert -1.0 <= res.cohens_kappa <= 1.0
        assert res.log_loss is not None and res.log_loss > 0.0
        assert res.roc_auc_macro is not None and res.roc_auc_macro > 0.8

        assert list(res.per_class_df.columns) == ["precision", "recall", "f1", "support"]
        assert res.per_class_df.loc["Tumor", "support"] == 2
        assert res.confusion_matrix_raw.loc["Tumor", "T cell"] == 1
        np.testing.assert_allclose(res.confusion_matrix_norm.sum(axis=1).to_numpy(), [1.0, 1.0, 1.0])

        # Verify JSON serialization works cleanly
        payload = json.dumps(res.to_json_dict())
        assert '"accuracy"' in payload

    def test_evaluate_predictions_shuffled_series_alignment_and_edge_cases(self):
        y_true = pd.Series(["A", "A", "B", "B"], index=[10, 20, 30, 40])
        y_pred = pd.Series(["B", "A", "B", "A"], index=[40, 20, 30, 10])
        res = evaluate_predictions(y_true, y_pred)
        assert res.accuracy == pytest.approx(1.0)

        # Single-class edge case
        res_single = evaluate_predictions(["A", "A", "A"], ["A", "A", "A"])
        assert res_single.accuracy == 1.0
        assert res_single.cohens_kappa == 1.0

        # Extra predicted class not in y_true
        res_extra = evaluate_predictions(["A", "A", "B", "B"], ["A", "Unresolved", "B", "B"])
        assert "Unresolved" in res_extra.per_class_df.index
        assert res_extra.per_class_df.loc["Unresolved", "support"] == 0

        with pytest.raises(ValueError, match="length mismatch"):
            evaluate_predictions(["A", "B"], ["A"])

    def test_format_benchmark_report_single_and_multi(self):
        res_a = evaluate_predictions(["Tumor", "T cell", "Tumor", "T cell"], ["Tumor", "T cell", "Tumor", "Tumor"])
        res_b = evaluate_predictions(["Tumor", "T cell", "Tumor", "T cell"], ["Tumor", "T cell", "Tumor", "T cell"])

        single_md = format_benchmark_report(
            res_a,
            method_name="MAPS",
            dataset_name="cHL CODEX",
            paper_metrics={"accuracy": 0.90, "macro_f1": 0.88},
            metadata={"Split": "5-fold CV"},
        )
        assert "# Phenotyping Benchmark Report: MAPS - cHL CODEX" in single_md
        assert "Published Paper" in single_md
        assert "Per-Class Performance" in single_md
        assert "Confusion Matrix" in single_md

        multi_md = format_benchmark_report(
            {"MAPS": res_a, "STELLAR": res_b},
            dataset_name="Synthetic CODEX",
        )
        assert "# Phenotyping Benchmark Comparison: Synthetic CODEX" in multi_md
        assert "| MAPS |" in multi_md
        assert "| STELLAR |" in multi_md

    def test_top_level_lazy_adapter_imports_without_napari(self):
        cmd = (
            "import sys; sys.modules['napari'] = None; "
            "import iris_phenotyper; "
            "assert 'torch' not in sys.modules, 'torch must not load eagerly on import iris_phenotyper'; "
            "from iris_phenotyper import ("
            "MAPSPhenotyper, STELLARPhenotyper, CellSighterPhenotyper, "
            "ASTIRPhenotyper, NimbusPhenotyper); "
            "names = [MAPSPhenotyper.__name__, STELLARPhenotyper.__name__, "
            "CellSighterPhenotyper.__name__, ASTIRPhenotyper.__name__, "
            "NimbusPhenotyper.__name__]; "
            "assert names == ['MAPSPhenotyper', 'STELLARPhenotyper', "
            "'CellSighterPhenotyper', 'ASTIRPhenotyper', 'NimbusPhenotyper']; "
            "print('LAZY_ADAPTERS_OK')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", cmd],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "LAZY_ADAPTERS_OK" in proc.stdout
