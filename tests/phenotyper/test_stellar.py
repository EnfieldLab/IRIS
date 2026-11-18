"""
Unit and Integration Tests for IRIS STELLAR Phenotyper
====================================================
Covers spatial graph construction, PyG numerical equivalence, supervised +
unsupervised training, cross-region label transfer, novel cell-type discovery,
and ``PhenotyperMethodBase`` / ``PhenotypingPredictionResult`` compliance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iris_phenotyper.base import PhenotypingPredictionResult
from iris_phenotyper.evaluation import evaluate_predictions
from iris_phenotyper.stellar import (
    STELLARPhenotyper,
    build_spatial_graph,
    is_pyg_available,
    partition_graph,
)
from iris_phenotyper.stellar.pipeline import STELLARGraph, _build_model_classes


def _make_spatial_codex_dataset(
    n_per_type: int = 45,
    include_novel: bool = False,
    seed: int = 123,
    canonical_cols: bool = True,
) -> pd.DataFrame:
    """Create a synthetic spatially structured multiplexed single-cell dataset."""
    rng = np.random.default_rng(seed)

    # 3 known phenotypes + optional 4th novel phenotype with distinct markers & spatial niche.
    centers = {
        "T_cell": np.array([3.0, -1.0, -1.0, 2.5, -1.0, -0.5], dtype=np.float32),
        "B_cell": np.array([-1.0, 3.2, -1.0, -0.8, 2.4, -0.5], dtype=np.float32),
        "Macrophage": np.array([-1.0, -1.0, 3.3, -0.5, -0.8, 2.6], dtype=np.float32),
    }
    spatial_niches = {
        "T_cell": np.array([40.0, 40.0]),
        "B_cell": np.array([150.0, 40.0]),
        "Macrophage": np.array([40.0, 150.0]),
    }
    if include_novel:
        centers["Novel_Epithelial"] = np.array(
            [-1.2, -1.2, -1.2, -1.0, -1.0, 3.5], dtype=np.float32
        )
        # Add extra separation on marker 0 vs 5 using a signature with high marker 5 & marker 1
        centers["Novel_Epithelial"] = np.array(
            [2.8, 2.8, -1.5, -1.5, 2.8, 2.8], dtype=np.float32
        )
        spatial_niches["Novel_Epithelial"] = np.array([160.0, 160.0])

    rows = []
    obj_id = 1
    for ct, feat_center in centers.items():
        niche = spatial_niches[ct]
        feats = feat_center[None, :] + rng.normal(0.0, 0.30, size=(n_per_type, len(feat_center)))
        coords = niche[None, :] + rng.normal(0.0, 15.0, size=(n_per_type, 2))
        for i in range(n_per_type):
            row = {
                f"M{m + 1}": float(feats[i, m]) for m in range(len(feat_center))
            }
            if canonical_cols:
                row["ObjectNumber"] = obj_id
                row["Image"] = "FOV_1" if i < n_per_type // 2 else "FOV_2"
                row["Pos_X"] = float(coords[i, 0])
                row["Pos_Y"] = float(coords[i, 1])
                row["cell_type"] = ct
            else:
                row["x"] = float(coords[i, 0])
                row["y"] = float(coords[i, 1])
                row["unique_region"] = "regA" if i < n_per_type // 2 else "regB"
                row["cell_type_A"] = ct
            rows.append(row)
            obj_id += 1
    return pd.DataFrame(rows)


class TestSpatialGraphConstruction:
    """Tests for O(N log N) spatial graph building and partitioning."""

    def test_strict_distance_threshold_and_no_self_loops(self):
        coords = np.array(
            [
                [0.0, 0.0],
                [30.0, 0.0],   # dist 30 < 50 -> connected to 0
                [50.0, 0.0],   # dist 50 == 50 from 0 (not < 50), dist 20 from 1 -> connected to 1
                [200.0, 200.0],  # isolated
            ]
        )
        edges = build_spatial_graph(coords, distance_thres=50.0)
        edge_set = {(int(edges[0, k]), int(edges[1, k])) for k in range(edges.shape[1])}
        assert edge_set == {(0, 1), (1, 0), (1, 2), (2, 1)}
        for u, v in edge_set:
            assert u != v

    def test_region_isolation_prevents_cross_region_edges(self):
        coords = np.array([[10.0, 10.0], [15.0, 10.0], [12.0, 10.0], [18.0, 10.0]])
        regions = ["R1", "R1", "R2", "R2"]
        edges = build_spatial_graph(coords, distance_thres=50.0, regions=regions)
        edge_set = {(int(edges[0, k]), int(edges[1, k])) for k in range(edges.shape[1])}
        assert edge_set == {(0, 1), (1, 0), (2, 3), (3, 2)}

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError, match="distance_thres"):
            build_spatial_graph(np.zeros((4, 2)), distance_thres=-1.0)
        with pytest.raises(ValueError, match="NaN"):
            build_spatial_graph(np.array([[0.0, 0.0], [np.nan, 1.0]]), distance_thres=10.0)

    def test_partition_graph_preserves_total_nodes(self):
        rng = np.random.default_rng(1)
        coords = rng.uniform(0, 200, size=(80, 2))
        edges = build_spatial_graph(coords, distance_thres=40.0)
        g = STELLARGraph(
            x=rng.normal(size=(80, 6)).astype(np.float32),
            edge_index=edges,
            coords=coords,
            y=rng.integers(0, 3, size=80),
        )
        parts = partition_graph(g, num_parts=5, seed=1)
        assert len(parts) == 5
        assert sum(p.x.shape[0] for p in parts) == 80
        for p in parts:
            if p.edge_index.shape[1] > 0:
                assert int(p.edge_index.max()) < p.x.shape[0]


class TestSTELLAREquivalence:
    """Verify mathematical equivalence between PureSAGEConv and PyG SAGEConv."""

    def test_pure_pytorch_matches_pyg_encoder_if_installed(self):
        if not is_pyg_available():
            pytest.skip("torch_geometric not installed in current environment")
        import torch

        classes = _build_model_classes()
        torch.manual_seed(7)
        enc_pure = classes["STELLAREncoder"](12, 5, hidden_dim=32, use_pyg=False)
        enc_pyg = classes["STELLAREncoder"](12, 5, hidden_dim=32, use_pyg=True)
        with torch.no_grad():
            enc_pure.load_state_dict(enc_pyg.state_dict())

        x = torch.randn(40, 12)
        ei = torch.randint(0, 40, (2, 120), dtype=torch.long)
        out_pure, f1_pure, f2_pure = enc_pure(x, ei)
        out_pyg, f1_pyg, f2_pyg = enc_pyg(x, ei)

        assert torch.max(torch.abs(out_pure - out_pyg)).item() < 1e-6
        assert torch.max(torch.abs(f1_pure - f1_pyg)).item() < 1e-6
        assert torch.max(torch.abs(f2_pure - f2_pyg)).item() < 1e-6


class TestSTELLARPhenotyper:
    """End-to-end tests for STELLARPhenotyper adapter and novel discovery."""

    def test_metadata_and_availability(self):
        pheno = STELLARPhenotyper()
        assert pheno.name == "STELLAR"
        assert pheno.input_modality == "spatial_tabular"
        assert pheno.requires_training is True
        assert pheno.is_available() is True
        defaults = pheno.get_default_params()
        assert defaults["distance_thres"] == 50.0
        assert defaults["epochs"] == 20

    def test_label_transfer_canonical_schema(self):
        train_df = _make_spatial_codex_dataset(n_per_type=40, include_novel=False, seed=10)
        test_df = _make_spatial_codex_dataset(n_per_type=35, include_novel=False, seed=99)
        test_unlabeled = test_df.drop(columns=["cell_type"])

        pheno = STELLARPhenotyper(
            distance_thres=60.0,
            epochs=12,
            num_parts=6,
            backend="pytorch",
            seed=1,
        )
        result = pheno.fit(train_df).predict(test_unlabeled)
        assert isinstance(result, PhenotypingPredictionResult)
        result.validate()
        assert result.method_name == "STELLAR"
        assert result.labels.index.equals(pd.Index(test_df["ObjectNumber"], name="ObjectNumber"))
        assert result.probabilities is not None
        assert np.allclose(result.probabilities.sum(axis=1).to_numpy(), 1.0, atol=1e-5)
        assert result.is_novel is not None
        assert not bool(result.is_novel.any())

        ev = evaluate_predictions(test_df["cell_type"], result.labels, result.probabilities)
        assert ev.accuracy >= 0.85
        assert ev.macro_f1 >= 0.85

        # Inductive inference on a new batch without retraining
        extra_df = _make_spatial_codex_dataset(n_per_type=15, include_novel=False, seed=202)
        inductive_res = pheno.predict(extra_df.drop(columns=["cell_type"]), retrain=False)
        inductive_res.validate()
        assert len(inductive_res.labels) == len(extra_df)

    def test_novel_cell_type_discovery_and_dryad_aliases(self):
        train_df = _make_spatial_codex_dataset(
            n_per_type=40, include_novel=False, seed=5, canonical_cols=False
        )
        test_df = _make_spatial_codex_dataset(
            n_per_type=40, include_novel=True, seed=77, canonical_cols=False
        )
        test_inputs = test_df.drop(columns=["cell_type_A"])

        pheno = STELLARPhenotyper(
            distance_thres=60.0,
            epochs=15,
            seed_epochs=12,
            num_seed_class=1,
            num_parts=6,
            backend="pytorch",
            seed=1,
        )
        result = pheno.fit_predict(train_df, test_inputs)
        result.validate()
        assert result.is_novel is not None
        assert result.is_novel.dtype == bool

        # Target contains an unseen Novel_Epithelial cluster; verify STELLAR detects novel cells.
        true_novel_mask = (test_df["cell_type_A"] == "Novel_Epithelial").to_numpy()
        pred_novel_mask = result.is_novel.to_numpy()
        assert int(pred_novel_mask.sum()) > 0
        novel_recall = float(np.mean(pred_novel_mask[true_novel_mask]))
        assert novel_recall >= 0.60
        assert all(
            lbl.startswith("Novel_")
            for lbl, is_nov in zip(result.labels, pred_novel_mask)
            if is_nov
        )
