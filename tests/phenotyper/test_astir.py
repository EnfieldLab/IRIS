"""
Unit & Integration Tests for iris_phenotyper.astir
==================================================
Covers marker schema normalization (YAML, dict, CellTypePanel), arcsinh
preprocessing, Bayesian variational inference for cell types & cell states,
Unknown gating, hierarchy aggregation, diagnostics, and PhenotyperMethodBase
contract compliance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iris_common.types import CellTypePanel, CellTypeRule, MarkerPanel
from iris_phenotyper.astir import (
    ASTIRPhenotyper,
    ASTIRPipeline,
    assign_celltypes_from_probabilities,
    normalize_marker_config,
    preprocess_expression,
)
from iris_phenotyper.base import PhenotyperMethodBase, PhenotypingPredictionResult


def _build_synthetic_dataset(
    n_per_type: int = 45,
    *,
    raw_scale: bool = False,
    with_mean_suffix: bool = False,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Generate a separable 3-type + state synthetic single-cell dataset."""
    rng = np.random.default_rng(seed)
    n_total = n_per_type * 3
    true_labels = (
        ["T cells"] * n_per_type + ["B cells"] * n_per_type + ["Epithelial"] * n_per_type
    )

    # Low vs high log-expression levels
    cd3 = np.concatenate(
        [
            rng.normal(3.2, 0.25, n_per_type),
            rng.normal(0.15, 0.08, n_per_type),
            rng.normal(0.15, 0.08, n_per_type),
        ]
    )
    cd45 = np.concatenate(
        [
            rng.normal(3.0, 0.25, n_per_type),
            rng.normal(3.1, 0.25, n_per_type),
            rng.normal(0.15, 0.08, n_per_type),
        ]
    )
    cd20 = np.concatenate(
        [
            rng.normal(0.15, 0.08, n_per_type),
            rng.normal(3.3, 0.25, n_per_type),
            rng.normal(0.15, 0.08, n_per_type),
        ]
    )
    panck = np.concatenate(
        [
            rng.normal(0.15, 0.08, n_per_type),
            rng.normal(0.15, 0.08, n_per_type),
            rng.normal(3.4, 0.25, n_per_type),
        ]
    )
    ecad = np.concatenate(
        [
            rng.normal(0.2, 0.08, n_per_type),
            rng.normal(0.2, 0.08, n_per_type),
            rng.normal(3.1, 0.25, n_per_type),
        ]
    )
    ki67 = rng.uniform(0.1, 2.5, n_total)
    ps6 = rng.uniform(0.1, 2.5, n_total)

    mat = np.column_stack([cd3, cd45, cd20, panck, ecad, ki67, ps6])
    mat = np.clip(mat, 0.0, None)
    if raw_scale:
        mat = 5.0 * np.sinh(mat)

    markers = ["CD3", "CD45", "CD20", "PanCK", "E-Cadherin", "Ki-67", "phospho S6"]
    cols = [f"{m}_mean" for m in markers] if with_mean_suffix else markers

    df = pd.DataFrame(mat, columns=cols)
    df.insert(0, "ObjectNumber", np.arange(1, n_total + 1, dtype=int))
    df.insert(1, "Image", ["FOV_1"] * (n_total // 2) + ["FOV_2"] * (n_total - n_total // 2))
    df.insert(2, "Pos_X", rng.uniform(0, 500, n_total))
    df.insert(3, "Pos_Y", rng.uniform(0, 500, n_total))

    marker_cfg = {
        "cell_types": {
            "T cells": ["CD45", "CD3"],
            "B cells": ["CD45", "CD20"],
            "Epithelial": ["PanCK", "E-Cadherin"],
        },
        "cell_states": {
            "Proliferation": ["Ki-67"],
            "mTOR": ["phospho S6"],
        },
        "hierarchy": {
            "Immune": ["T cells", "B cells"],
            "Epithelial": ["Epithelial"],
        },
    }
    return df, pd.Series(true_labels, index=df["ObjectNumber"], name="cell_type"), marker_cfg


class TestMarkerAndPreprocessing:
    def test_normalize_marker_config_variants(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "markers.yml"
        yaml_file.write_text(
            "cell_types:\n  T cells:\n    - CD3\n    - CD45\n  B cells:\n    - CD20\n",
            encoding="utf-8",
        )
        t_yaml, s_yaml, h_yaml = normalize_marker_config(yaml_file)
        assert t_yaml == {"T cells": ["CD3", "CD45"], "B cells": ["CD20"]}
        assert s_yaml is None
        assert h_yaml is None

        flat_dict = {"T cells": ["CD3"], "B cells": ["CD20"]}
        t_flat, _, _ = normalize_marker_config(flat_dict)
        assert t_flat == flat_dict

        m_panel = MarkerPanel(markers=["CD3", "CD45", "CD20"])
        c_panel = CellTypePanel(
            rules=[
                CellTypeRule("T cells", markers_positive=["cd3", "cd45"], parent="Immune"),
                CellTypeRule("B cells", markers_positive=["cd20"], parent="Immune"),
            ]
        )
        t_panel, _, h_panel = normalize_marker_config(c_panel, marker_panel=m_panel)
        assert t_panel == {"T cells": ["CD3", "CD45"], "B cells": ["CD20"]}
        assert h_panel == {"Immune": ["T cells", "B cells"]}

    def test_preprocess_expression_arcsinh_and_winsorize(self) -> None:
        df = pd.DataFrame({"A": [0.0, 5.0, 50.0, 500.0], "B": [1.0, 2.0, 3.0, 4.0]})
        proc_auto, did_auto = preprocess_expression(df, apply_arcsinh="auto", arcsinh_cofactor=5.0)
        assert did_auto is True
        assert np.allclose(proc_auto["A"].to_numpy(), np.arcsinh(df["A"].to_numpy() / 5.0))

        small_df = pd.DataFrame({"A": [0.1, 1.2, 3.4], "B": [0.2, 2.1, 4.0]})
        proc_small, did_small = preprocess_expression(small_df, apply_arcsinh="auto")
        assert did_small is False
        assert np.allclose(proc_small.to_numpy(), small_df.to_numpy())

        proc_win, _ = preprocess_expression(
            small_df, apply_arcsinh=False, winsorize_percentiles=(25.0, 75.0)
        )
        assert proc_win["A"].max() <= np.percentile(small_df["A"], 75.0) + 1e-12
        assert proc_win["A"].min() >= np.percentile(small_df["A"], 25.0) - 1e-12


class TestASTIRPhenotyperAndPipeline:
    def test_adapter_contract_and_defaults(self) -> None:
        model = ASTIRPhenotyper(engine="iris")
        assert isinstance(model, PhenotyperMethodBase)
        assert model.name == "ASTIR"
        assert model.input_modality == "tabular"
        assert model.requires_training is False
        assert model.is_available() is True
        defaults = model.get_default_params()
        assert defaults["arcsinh_cofactor"] == 5.0
        assert "max_epochs" in defaults

    def test_fit_predict_synthetic_with_mean_suffix_and_states(self) -> None:
        df, y_true, cfg = _build_synthetic_dataset(
            n_per_type=40, raw_scale=True, with_mean_suffix=True, seed=11
        )
        model = ASTIRPhenotyper(
            marker_config=cfg,
            max_epochs=60,
            learning_rate=5e-3,
            batch_size=32,
            n_init=2,
            n_init_epochs=3,
            threshold=0.5,
            apply_arcsinh="auto",
            batch_col="Image",
            fit_states=True,
            engine="iris",
            random_seed=1234,
        )
        res = model.predict(df)
        assert isinstance(res, PhenotypingPredictionResult)
        res.validate()
        assert res.method_name == "ASTIR"
        assert res.metadata["did_arcsinh"] is True
        assert res.metadata["engine_used"] == "iris_pytorch"
        assert list(res.labels.index) == df["ObjectNumber"].tolist()
        assert res.probabilities is not None
        assert list(res.probabilities.columns) == ["T cells", "B cells", "Epithelial", "Other"]
        assert np.allclose(res.probabilities.sum(axis=1).to_numpy(), 1.0, atol=1e-5)

        acc = float(np.mean(res.labels.to_numpy() == y_true.to_numpy()))
        assert acc >= 0.85

        assert "cell_states" in res.metadata
        states_df = res.metadata["cell_states"]
        assert list(states_df.columns) == ["Proliferation", "mTOR"]
        assert float(states_df.min().min()) >= -1e-9
        assert float(states_df.max().max()) <= 1.0 + 1e-9

    def test_unknown_gating_and_hierarchy_and_diagnostics(self) -> None:
        df, _, cfg = _build_synthetic_dataset(n_per_type=30, raw_scale=False, seed=7)
        pipe = ASTIRPipeline(
            df,
            cfg,
            apply_arcsinh=False,
            create_design_mat=False,
            engine="iris",
            random_seed=42,
        )
        pipe.fit_type(max_epochs=30, learning_rate=5e-3, batch_size=32, n_init=2, n_init_epochs=2)
        pipe.fit_state(max_epochs=20, learning_rate=5e-3, batch_size=32, n_init=1, n_init_epochs=2)

        probs = pipe.get_celltype_probabilities()
        high_gate = assign_celltypes_from_probabilities(probs, threshold=0.999999)
        assert "Unknown" in set(high_gate["cell_type"])

        hier = pipe.assign_celltype_hierarchy(depth=1)
        assert list(hier.columns) == ["Immune", "Epithelial"]
        assert np.allclose(
            hier["Immune"].to_numpy(),
            (probs["T cells"] + probs["B cells"]).to_numpy(),
        )

        diag_type = pipe.diagnostics_celltype(threshold=0.5, alpha=0.01)
        assert list(diag_type.columns) == [
            "feature",
            "should be expressed higher in",
            "than",
            "mean cell type 1",
            "mean cell type 2",
            "p-value",
            "note",
        ]
        assert len(diag_type) == 0

        diag_state = pipe.diagnostics_cellstate()
        assert list(diag_state.columns) == [
            "pathway",
            "protein A",
            "correlation of protein A",
            "protein B",
            "correlation of protein B",
            "note",
        ]

    def test_anndata_input_and_transfer_vs_autorefit(self) -> None:
        import anndata as ad

        df, _, cfg = _build_synthetic_dataset(n_per_type=30, raw_scale=False, seed=99)
        marker_cols = ["CD3", "CD45", "CD20", "PanCK", "E-Cadherin", "Ki-67", "phospho S6"]
        adata = ad.AnnData(
            X=df[marker_cols].to_numpy(),
            obs=pd.DataFrame(
                {"cell_id": [f"cell_{i}" for i in range(len(df))], "batch": df["Image"].to_numpy()},
                index=[f"cell_{i}" for i in range(len(df))],
            ),
        )
        adata.var_names = marker_cols

        pheno = ASTIRPhenotyper(
            marker_config=cfg,
            max_epochs=35,
            learning_rate=5e-3,
            batch_size=32,
            n_init=1,
            n_init_epochs=2,
            apply_arcsinh=False,
            engine="iris",
        )
        res1 = pheno.predict(adata)
        res1.validate()
        assert list(res1.labels.index) == list(range(1, len(df) + 1))
        assert res1.metadata["cell_ids"][0] == "cell_0"

        # Transfer prediction without refitting
        res_transfer = pheno.predict(adata[:20].copy(), refit=False)
        res_transfer.validate()
        assert len(res_transfer.labels) == 20

        # Auto-refit on new DataFrame with CellTypePanel
        panel = CellTypePanel(
            rules=[
                CellTypeRule("T cells", markers_positive=["CD45", "CD3"]),
                CellTypeRule("B cells", markers_positive=["CD45", "CD20"]),
            ]
        )
        df_new, _, _ = _build_synthetic_dataset(n_per_type=20, raw_scale=False, seed=55)
        res_new = pheno.predict(df_new, cell_type_panel=panel)
        res_new.validate()
        assert set(res_new.probabilities.columns) == {"T cells", "B cells", "Other"}

    def test_edge_case_shared_marker_and_single_type_raises(self) -> None:
        df = pd.DataFrame(
            {
                "ObjectNumber": [1, 2, 3, 4],
                "Shared_mean": [1.0, 2.0, 1.0, 2.0],
                "Const_mean": [0.5, 0.5, 0.5, 0.5],
            }
        )
        # Single cell type must raise
        with pytest.raises(RuntimeError):
            ASTIRPipeline(df, {"OnlyOne": ["Shared"]}, engine="iris")

        # Shared marker present in all cell types should still initialize safely without crashing
        pipe = ASTIRPipeline(
            df,
            {"TypeA": ["Shared", "Const"], "TypeB": ["Shared"]},
            apply_arcsinh=False,
            engine="iris",
        )
        pipe.fit_type(max_epochs=2, batch_size=2, n_init=1, n_init_epochs=1)
        probs = pipe.get_celltype_probabilities()
        assert probs.shape == (4, 3)

    def test_dtype_float32_max_assignment_and_cached_fit_predict(self) -> None:
        df, _, cfg = _build_synthetic_dataset(n_per_type=20, raw_scale=False, seed=21)
        pheno = ASTIRPhenotyper(
            marker_config=cfg,
            max_epochs=10,
            batch_size=16,
            n_init=1,
            n_init_epochs=2,
            assignment_type="max",
            apply_arcsinh=False,
            dtype="float32",
            engine="iris",
        )
        pheno.fit(df)
        res1 = pheno.predict(df)
        res2 = pheno.predict(df)
        res1.validate()
        res2.validate()
        assert res1.labels.equals(res2.labels)
        assert res1.metadata["assignment_type"] == "max"

