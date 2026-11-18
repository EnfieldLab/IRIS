"""
STELLAR Paper Reproduction Script
=================================
Reproduces the key benchmarks from Brbic et al., Nature Methods 2022
("Annotation of spatially resolved single-cell data with STELLAR",
DOI: 10.1038/s41592-022-01651-8) using the Dryad CODEX datasets
(DOI: 10.5061/dryad.g4f4qrfrc):

1. Cross-tissue transfer + novel cell-type discovery:
   Tonsil -> Barrett's Esophagus (``BE_Tonsil_l3_dryad.csv``)
2. HuBMAP healthy intestine intra-donor cross-region transfer:
   B004 Colon regions 1-3 -> region 4 (``B004_training_dryad.csv``)
3. HuBMAP cross-donor transfer + marker expression profile concordance:
   B004 Colon -> unannotated B005 Colon (``B0056_unnanotated_dryad.csv``)
4. Exact numerical equivalence check between pure-PyTorch ``PureSAGEConv`` and
   PyTorch Geometric ``SAGEConv`` (when ``torch_geometric`` is available).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from iris_phenotyper.evaluation import (
    EvaluationResult,
    evaluate_predictions,
    format_benchmark_report,
)
from iris_phenotyper.stellar.adapter import STELLARPhenotyper
from iris_phenotyper.stellar.pipeline import (
    _build_model_classes,
    _entropy_loss,
    build_spatial_graph,
    is_pyg_available,
)

DEFAULT_BENCHMARK_DIR = Path("/Users/wwu/Documents/IRIS_Benchmarks/stellar")
DEFAULT_DATA_SUBDIR = "doi_10_5061_dryad_g4f4qrfrc__v20220719"

EXPECTED_FILES: Dict[str, Dict[str, Any]] = {
    "B004_training_dryad.csv": {
        "size": 165599687,
        "sha256": "487a74b1d2d61fa830b0c1ad9e05f0ff4e90cd913de69671fdabdac0235f6344",
    },
    "B0056_unnanotated_dryad.csv": {
        "size": 387756050,
        "sha256": "54a48af81f5e428b123960d93510be81c6fd7040041d830d58b445612ae21393",
    },
    "BE_Tonsil_l3_dryad.csv": {
        "size": 127298344,
        "sha256": "3b308d8cb11782cc628fc658a8cb9541643638a5c545a2b5b589639c6d456729",
    },
    "README_CODEXdata_STELLAR.docx": {
        "size": 13991,
        "sha256": "f1a270015b3f9ba862f2dbb2d85449dcff7db59f7175f1b0c7b2d19de865b3a8",
    },
}


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file using 1 MiB streaming chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset_files(data_dir: Path) -> Dict[str, Any]:
    """Verify presence, file sizes, and SHA-256 digests of Dryad CODEX CSVs."""
    records: Dict[str, Any] = {}
    all_present = True
    all_verified = True
    for fname, expected in EXPECTED_FILES.items():
        fpath = data_dir / fname
        if not fpath.is_file():
            all_present = False
            all_verified = False
            records[fname] = {"exists": False, "path": str(fpath)}
            continue
        size = int(fpath.stat().st_size)
        sha = compute_sha256(fpath)
        ok = (size == expected["size"]) and (sha == expected["sha256"])
        if not ok:
            all_verified = False
        records[fname] = {
            "exists": True,
            "path": str(fpath),
            "size_bytes": size,
            "sha256": sha,
            "sha256_matches_dryad": ok,
        }
    return {
        "data_dir": str(data_dir),
        "all_present": all_present,
        "all_verified": all_verified,
        "files": records,
    }


def _hungarian_map_novel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    known_classes: set[str],
    novel_pred_classes: List[str],
) -> Tuple[np.ndarray, Dict[str, str]]:
    """Optimally align discovered novel cluster heads to unseen ground-truth types."""
    unseen_true = sorted(set(y_true.tolist()) - known_classes)
    if not novel_pred_classes or not unseen_true:
        return y_pred.copy(), {}

    cost = np.zeros((len(novel_pred_classes), len(unseen_true)), dtype=np.int64)
    for i, np_cls in enumerate(novel_pred_classes):
        for j, ut_cls in enumerate(unseen_true):
            cost[i, j] = -int(np.sum((y_pred == np_cls) & (y_true == ut_cls)))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {novel_pred_classes[int(r)]: unseen_true[int(c)] for r, c in zip(row_ind, col_ind)}
    mapped = np.array([mapping.get(lbl, lbl) for lbl in y_pred], dtype=object)
    return mapped, mapping


def run_equivalence_check(upstream_repo: Optional[Path] = None) -> Dict[str, Any]:
    """Verify exact numerical equivalence between PureSAGEConv and PyG / upstream STELLAR."""
    if not is_pyg_available():
        return {
            "checked": False,
            "reason": "torch_geometric not installed in current Python environment",
        }
    import torch
    import torch.nn.functional as F

    classes = _build_model_classes()
    torch.manual_seed(42)
    rng = np.random.default_rng(7)
    coords = rng.uniform(0.0, 300.0, size=(200, 2))
    regions = np.array(["regA"] * 100 + ["regB"] * 100)
    edges = build_spatial_graph(coords, distance_thres=50.0, regions=regions)

    enc_pure = classes["STELLAREncoder"](44, 13, hidden_dim=128, use_pyg=False)
    enc_pyg = classes["STELLAREncoder"](44, 13, hidden_dim=128, use_pyg=True)
    with torch.no_grad():
        enc_pure.load_state_dict(enc_pyg.state_dict())

    x = torch.randn(200, 44)
    ei = torch.tensor(edges, dtype=torch.long)
    y = torch.randint(0, 13, (200,))

    out_pure, f1_pure, f2_pure = enc_pure(x, ei)
    out_pyg, f1_pyg, f2_pyg = enc_pyg(x, ei)
    logits_diff = float(torch.max(torch.abs(out_pure - out_pyg)).item())
    feat_diff = float(torch.max(torch.abs(f2_pure - f2_pyg)).item())

    ml = classes["MarginLoss"](m=-0.25)
    loss_pure = ml(out_pure, y) - 0.3 * _entropy_loss(F.softmax(out_pure, dim=1).mean(dim=0))
    loss_pyg = ml(out_pyg, y) - 0.3 * _entropy_loss(F.softmax(out_pyg, dim=1).mean(dim=0))
    loss_diff = abs(float(loss_pure.item()) - float(loss_pyg.item()))

    loss_pure.backward()
    loss_pyg.backward()
    grad_diff = max(
        float(torch.max(torch.abs(p1.grad - p2.grad)).item())
        for p1, p2 in zip(enc_pure.parameters(), enc_pyg.parameters())
    )

    result: Dict[str, Any] = {
        "checked": True,
        "max_logits_abs_diff": logits_diff,
        "max_embedding_abs_diff": feat_diff,
        "max_loss_abs_diff": loss_diff,
        "max_grad_abs_diff": grad_diff,
        "numerically_equivalent": logits_diff < 1e-6 and feat_diff < 1e-6 and grad_diff < 1e-6,
    }

    if upstream_repo is not None and upstream_repo.is_dir():
        sys_path_added = False
        repo_str = str(upstream_repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
            sys_path_added = True
        try:
            import datasets as up_datasets
            import models as up_models

            up_edges = up_datasets.get_hubmap_edge_index(coords, regions, 50.0)
            up_set = {(int(u), int(v)) for u, v in up_edges}
            our_set = {(int(edges[0, k]), int(edges[1, k])) for k in range(edges.shape[1])}
            up_enc = up_models.Encoder(44, 13)
            with torch.no_grad():
                up_enc.load_state_dict(enc_pyg.state_dict())
            from torch_geometric.data import Data

            up_out, _, up_f2 = up_enc(Data(x=x, edge_index=ei))
            result["upstream_edge_set_exact_match"] = up_set == our_set
            result["upstream_logits_max_abs_diff"] = float(
                torch.max(torch.abs(up_out - out_pure)).item()
            )
            result["upstream_embedding_max_abs_diff"] = float(
                torch.max(torch.abs(up_f2 - f2_pure)).item()
            )
        finally:
            if sys_path_added and repo_str in sys.path:
                sys.path.remove(repo_str)

    return result


def run_tonsil_to_be_benchmark(
    csv_path: Path,
    *,
    sample_rate: float = 0.5,
    epochs: int = 50,
    seed_epochs: int = 20,
    distance_thres: float = 50.0,
    backend: str = "auto",
    seed: int = 1,
) -> Tuple[Dict[str, Any], EvaluationResult, pd.DataFrame]:
    """Run Tonsil -> Barrett's Esophagus cross-tissue novel-discovery benchmark."""
    df = pd.read_csv(csv_path, index_col=0)
    tonsil_df = df.loc[df["sample_name"] == "tonsil"].copy()
    if sample_rate < 1.0:
        tonsil_df = tonsil_df.sample(n=round(sample_rate * len(tonsil_df)), random_state=seed)
    tonsil_df = tonsil_df.reset_index(drop=True)
    be_df = df.loc[df["sample_name"] == "Barretts Esophagus"].reset_index(drop=True)

    tonsil_df["ObjectNumber"] = np.arange(1, len(tonsil_df) + 1, dtype=np.int64)
    be_df["ObjectNumber"] = np.arange(1, len(be_df) + 1, dtype=np.int64)

    marker_cols = [c for c in df.columns if c not in ("x", "y", "sample_name", "cell_type")]

    pheno = STELLARPhenotyper(
        distance_thres=distance_thres,
        epochs=epochs,
        seed_epochs=seed_epochs,
        num_seed_class=3,
        num_heads=13,
        backend=backend,
        seed=seed,
        feature_columns=marker_cols,
        x_col="x",
        y_col="y",
        region_col="sample_name",
        label_col="cell_type",
    )
    pred_res = pheno.fit_predict(tonsil_df, be_df.drop(columns=["cell_type"]))

    y_true = be_df["cell_type"].astype(str).to_numpy()
    y_pred_raw = pred_res.labels.to_numpy().astype(str)
    known_set = set(tonsil_df["cell_type"].astype(str).unique())
    novel_pred_classes = list(pred_res.metadata["novel_classes"])

    y_pred_mapped, novel_mapping = _hungarian_map_novel(
        y_true, y_pred_raw, known_set, novel_pred_classes
    )
    eval_res = evaluate_predictions(y_true, y_pred_mapped)

    known_mask = np.isin(y_true, list(known_set))
    known_acc = float(np.mean(y_pred_raw[known_mask] == y_true[known_mask]))
    known_eval = evaluate_predictions(y_true[known_mask], y_pred_raw[known_mask])

    unseen_mask = ~known_mask
    pred_novel_mask = pred_res.is_novel.to_numpy(dtype=bool)
    novel_recall = float(np.mean(pred_novel_mask[unseen_mask])) if np.any(unseen_mask) else 0.0
    novel_prec = (
        float(np.mean(unseen_mask[pred_novel_mask])) if np.any(pred_novel_mask) else 0.0
    )
    novel_f1 = (
        2.0 * novel_prec * novel_recall / (novel_prec + novel_recall)
        if (novel_prec + novel_recall) > 0.0
        else 0.0
    )
    ari_unseen = (
        float(adjusted_rand_score(y_true[unseen_mask], y_pred_raw[unseen_mask]))
        if np.any(unseen_mask)
        else 0.0
    )
    nmi_unseen = (
        float(normalized_mutual_info_score(y_true[unseen_mask], y_pred_raw[unseen_mask]))
        if np.any(unseen_mask)
        else 0.0
    )

    summary = {
        "dataset": "BE_Tonsil_l3_dryad.csv",
        "n_train_cells": int(len(tonsil_df)),
        "n_test_cells": int(len(be_df)),
        "n_markers": len(marker_cols),
        "sample_rate": float(sample_rate),
        "epochs": int(epochs),
        "distance_thres": float(distance_thres),
        "backend_used": pred_res.metadata["backend"],
        "train_time_sec": float(pred_res.metadata["train_time_sec"]),
        "open_world_accuracy": float(eval_res.accuracy),
        "open_world_balanced_accuracy": float(eval_res.balanced_accuracy),
        "open_world_macro_f1": float(eval_res.macro_f1),
        "open_world_weighted_f1": float(eval_res.weighted_f1),
        "open_world_cohens_kappa": float(eval_res.cohens_kappa),
        "accuracy_excluding_rare_tonsil_smoothmuscle": float(
            np.mean(y_pred_mapped[y_true != "SmoothMuscle"] == y_true[y_true != "SmoothMuscle"])
        ),
        "known_classes_accuracy": known_acc,
        "known_classes_macro_f1": float(known_eval.macro_f1),
        "novel_binary_precision": novel_prec,
        "novel_binary_recall": novel_recall,
        "novel_binary_f1": novel_f1,
        "unseen_ari": ari_unseen,
        "unseen_nmi": nmi_unseen,
        "n_predicted_novel_cells": int(np.sum(pred_novel_mask)),
        "n_true_unseen_cells": int(np.sum(unseen_mask)),
        "novel_hungarian_mapping": novel_mapping,
        "published_paper_accuracy": 0.93,
    }
    crosstab_df = pd.crosstab(
        pd.Series(y_pred_raw, name="predicted"),
        pd.Series(y_true, name="true"),
    )
    return summary, eval_res, crosstab_df


def run_hubmap_b004_region_benchmark(
    csv_path: Path,
    *,
    tissue: str = "CL",
    holdout_region: str = "reg004_CL_B004",
    sample_rate: float = 1.0,
    epochs: int = 20,
    distance_thres: float = 50.0,
    backend: str = "auto",
    seed: int = 1,
) -> Tuple[Dict[str, Any], EvaluationResult]:
    """Run HuBMAP B004 intra-donor cross-region holdout benchmark (3 regions -> 4th)."""
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    sub = df.loc[(df["donor"] == "B004") & (df["tissue"] == tissue)].copy()
    if sample_rate < 1.0:
        sub = sub.sample(n=round(sample_rate * len(sub)), random_state=seed)

    train_df = sub.loc[sub["unique_region"] != holdout_region].reset_index(drop=True)
    test_df = sub.loc[sub["unique_region"] == holdout_region].reset_index(drop=True)
    train_df["ObjectNumber"] = np.arange(1, len(train_df) + 1, dtype=np.int64)
    test_df["ObjectNumber"] = np.arange(1, len(test_df) + 1, dtype=np.int64)

    exclude = {"x", "y", "tissue", "cell_type_A", "donor", "unique_region", "ObjectNumber"}
    marker_cols = [c for c in sub.columns if c not in exclude]

    pheno = STELLARPhenotyper(
        distance_thres=distance_thres,
        epochs=epochs,
        num_seed_class=0,
        backend=backend,
        seed=seed,
        feature_columns=marker_cols,
        x_col="x",
        y_col="y",
        region_col="unique_region",
        label_col="cell_type_A",
    )
    pred_res = pheno.fit_predict(train_df, test_df.drop(columns=["cell_type_A"]))

    y_true = test_df["cell_type_A"].astype(str)
    eval_res = evaluate_predictions(y_true, pred_res.labels, pred_res.probabilities)
    summary = {
        "dataset": "B004_training_dryad.csv",
        "tissue": tissue,
        "holdout_region": holdout_region,
        "n_train_cells": int(len(train_df)),
        "n_test_cells": int(len(test_df)),
        "n_markers": len(marker_cols),
        "sample_rate": float(sample_rate),
        "epochs": int(epochs),
        "distance_thres": float(distance_thres),
        "backend_used": pred_res.metadata["backend"],
        "train_time_sec": float(pred_res.metadata["train_time_sec"]),
        "accuracy": float(eval_res.accuracy),
        "balanced_accuracy": float(eval_res.balanced_accuracy),
        "macro_f1": float(eval_res.macro_f1),
        "weighted_f1": float(eval_res.weighted_f1),
        "cohens_kappa": float(eval_res.cohens_kappa),
        "roc_auc_macro": eval_res.roc_auc_macro,
        "log_loss": eval_res.log_loss,
    }
    return summary, eval_res


def run_hubmap_cross_donor_transfer(
    b004_csv: Path,
    b0056_csv: Path,
    *,
    tissue: str = "CL",
    target_donor: str = "B005",
    sample_rate: float = 0.5,
    epochs: int = 20,
    distance_thres: float = 50.0,
    backend: str = "auto",
    seed: int = 1,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run B004 -> B005 unannotated transfer and compute marker-profile concordance."""
    tr = pd.read_csv(b004_csv, index_col=0, low_memory=False)
    te = pd.read_csv(b0056_csv, index_col=0, low_memory=False)
    if sample_rate < 1.0:
        tr = tr.sample(n=round(sample_rate * len(tr)), random_state=seed)
        te = te.sample(n=round(sample_rate * len(te)), random_state=seed)
    tr = tr.loc[(tr["tissue"] == tissue) & (tr["donor"] == "B004")].reset_index(drop=True)
    te = te.loc[(te["tissue"] == tissue) & (te["donor"] == target_donor)].reset_index(drop=True)
    tr["ObjectNumber"] = np.arange(1, len(tr) + 1, dtype=np.int64)
    te["ObjectNumber"] = np.arange(1, len(te) + 1, dtype=np.int64)

    exclude = {"x", "y", "tissue", "cell_type_A", "donor", "unique_region", "ObjectNumber"}
    marker_cols = [c for c in tr.columns if c not in exclude]

    pheno = STELLARPhenotyper(
        distance_thres=distance_thres,
        epochs=epochs,
        num_seed_class=0,
        backend=backend,
        seed=seed,
        feature_columns=marker_cols,
        x_col="x",
        y_col="y",
        region_col="unique_region",
        label_col="cell_type_A",
    )
    pred_res = pheno.fit_predict(tr, te)

    ref_means = tr.groupby("cell_type_A")[marker_cols].mean()
    te_with_pred = te[marker_cols].copy()
    te_with_pred["pred_cell_type"] = pred_res.labels.to_numpy()
    tgt_means = te_with_pred.groupby("pred_cell_type")[marker_cols].mean()
    pred_counts = te_with_pred["pred_cell_type"].value_counts().to_dict()

    shared_types = sorted(set(ref_means.index) & set(tgt_means.index))
    rows: List[Dict[str, Any]] = []
    for ct in shared_types:
        v_ref = ref_means.loc[ct].to_numpy(dtype=np.float64)
        v_tgt = tgt_means.loc[ct].to_numpy(dtype=np.float64)
        r_pearson = float(np.corrcoef(v_ref, v_tgt)[0, 1])
        denom = float(np.linalg.norm(v_ref) * np.linalg.norm(v_tgt))
        cos_sim = float(np.dot(v_ref, v_tgt) / denom) if denom > 0 else 0.0
        rows.append(
            {
                "cell_type": ct,
                "ref_count": int((tr["cell_type_A"] == ct).sum()),
                "pred_count": int(pred_counts.get(ct, 0)),
                "pearson_r": r_pearson,
                "cosine_sim": cos_sim,
            }
        )
    concordance_df = pd.DataFrame(rows)
    summary = {
        "train_dataset": "B004_training_dryad.csv",
        "target_dataset": "B0056_unnanotated_dryad.csv",
        "tissue": tissue,
        "target_donor": target_donor,
        "n_train_cells": int(len(tr)),
        "n_target_cells": int(len(te)),
        "sample_rate": float(sample_rate),
        "epochs": int(epochs),
        "backend_used": pred_res.metadata["backend"],
        "train_time_sec": float(pred_res.metadata["train_time_sec"]),
        "n_predicted_types": len(shared_types),
        "mean_marker_profile_pearson_r": (
            float(concordance_df["pearson_r"].mean()) if not concordance_df.empty else 0.0
        ),
        "mean_marker_profile_cosine_sim": (
            float(concordance_df["cosine_sim"].mean()) if not concordance_df.empty else 0.0
        ),
        "mean_prediction_confidence": float(pred_res.confidence.mean()),
    }
    return summary, concordance_df


def run_synthetic_fallback_benchmark(
    epochs: int = 15,
    backend: str = "auto",
    seed: int = 1,
) -> Tuple[Dict[str, Any], EvaluationResult]:
    """Run a synthetic multi-region CODEX benchmark when Dryad files are absent."""
    n_markers = 20
    n_per_cls = 150
    known_names = ["Enterocyte", "Goblet", "CD8T", "Macrophage", "SmoothMuscle"]

    def _make_split(split_seed: int, include_novel: bool) -> pd.DataFrame:
        local_rng = np.random.default_rng(split_seed)
        rows = []
        cls_names = list(known_names) + (["Novel_Paneth"] if include_novel else [])
        for c_idx, cname in enumerate(cls_names):
            proto = np.zeros(n_markers, dtype=np.float32)
            proto[(c_idx * 3) % n_markers : ((c_idx * 3) % n_markers) + 3] = 3.0
            feats = proto[None, :] + local_rng.normal(0.0, 0.35, size=(n_per_cls, n_markers))
            center = np.array([(c_idx % 3) * 120.0 + 40.0, (c_idx // 3) * 120.0 + 40.0])
            coords = center[None, :] + local_rng.normal(0.0, 18.0, size=(n_per_cls, 2))
            for i in range(n_per_cls):
                row = {f"M{m + 1}": float(feats[i, m]) for m in range(n_markers)}
                row["Pos_X"] = float(coords[i, 0])
                row["Pos_Y"] = float(coords[i, 1])
                row["Image"] = f"reg_{(i % 2) + 1}"
                row["cell_type"] = cname
                rows.append(row)
        out = pd.DataFrame(rows)
        out["ObjectNumber"] = np.arange(1, len(out) + 1, dtype=np.int64)
        return out

    train_df = _make_split(seed + 10, include_novel=False)
    test_df = _make_split(seed + 20, include_novel=True)
    pheno = STELLARPhenotyper(
        distance_thres=50.0,
        epochs=epochs,
        seed_epochs=15,
        num_seed_class=1,
        num_parts=12,
        backend=backend,
        seed=seed,
    )
    pred = pheno.fit_predict(train_df, test_df.drop(columns=["cell_type"]))
    y_true = test_df["cell_type"].astype(str).to_numpy()
    mapped, mapping = _hungarian_map_novel(
        y_true,
        pred.labels.to_numpy().astype(str),
        set(known_names),
        list(pred.metadata["novel_classes"]),
    )
    ev = evaluate_predictions(y_true, mapped)
    summary = {
        "dataset": "synthetic_multi_region_codex",
        "n_train_cells": int(len(train_df)),
        "n_test_cells": int(len(test_df)),
        "accuracy": float(ev.accuracy),
        "macro_f1": float(ev.macro_f1),
        "novel_mapping": mapping,
        "backend_used": pred.metadata["backend"],
    }
    return summary, ev


def run_stellar_paper_reproduction(
    benchmark_dir: Path = DEFAULT_BENCHMARK_DIR,
    *,
    sample_rate: float = 0.5,
    tonsil_epochs: int = 20,
    hubmap_epochs: int = 20,
    backend: str = "pytorch",
    seed: int = 1,
    quick: bool = False,
) -> Dict[str, Any]:
    """Run full STELLAR paper reproduction and save reports + manifest updates."""
    t_start = time.perf_counter()
    data_dir = benchmark_dir / "data" / DEFAULT_DATA_SUBDIR
    results_dir = benchmark_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    upstream_repo = benchmark_dir / "upstream_repo"

    if quick:
        sample_rate = min(sample_rate, 0.25)
        tonsil_epochs = min(tonsil_epochs, 20)
        hubmap_epochs = min(hubmap_epochs, 12)

    file_check = verify_dataset_files(data_dir)
    equiv_summary = run_equivalence_check(upstream_repo)
    (results_dir / "equivalence_verification.json").write_text(
        json.dumps(equiv_summary, indent=2) + "\n", encoding="utf-8"
    )

    if not file_check["all_present"]:
        syn_summary, syn_ev = run_synthetic_fallback_benchmark(
            epochs=hubmap_epochs, backend=backend, seed=seed
        )
        rep_md = format_benchmark_report(
            syn_ev,
            method_name="STELLAR",
            dataset_name="Synthetic Multi-Region CODEX",
            metadata={"status": "awaiting_manual_data_pipeline_verified_on_synthetic"},
        )
        (results_dir / "synthetic_verification_report.md").write_text(rep_md, encoding="utf-8")
        out = {
            "status": "awaiting_manual_data_pipeline_verified_on_synthetic",
            "data_verification": file_check,
            "equivalence_verification": equiv_summary,
            "synthetic_benchmark": syn_summary,
        }
        (results_dir / "stellar_reproduction_summary.json").write_text(
            json.dumps(out, indent=2) + "\n", encoding="utf-8"
        )
        return out

    # 1. Tonsil -> Barrett's Esophagus (Cross-Tissue Novel Cell Discovery)
    tonsil_summary, tonsil_ev, tonsil_ct = run_tonsil_to_be_benchmark(
        data_dir / "BE_Tonsil_l3_dryad.csv",
        sample_rate=sample_rate,
        epochs=tonsil_epochs,
        backend=backend,
        seed=seed,
    )
    tonsil_ev.per_class_df.to_csv(results_dir / "tonsil_be_per_class.csv")
    tonsil_ev.confusion_matrix_raw.to_csv(results_dir / "tonsil_be_confusion_matrix.csv")
    tonsil_ct.to_csv(results_dir / "tonsil_be_raw_pred_crosstab.csv")
    tonsil_md = format_benchmark_report(
        tonsil_ev,
        method_name="STELLAR",
        dataset_name="Tonsil -> Barrett's Esophagus (Dryad CODEX)",
        paper_metrics={"accuracy": 0.93},
        metadata={
            "known_classes_accuracy": f"{tonsil_summary['known_classes_accuracy']:.4f}",
            "novel_binary_precision": f"{tonsil_summary['novel_binary_precision']:.4f}",
            "novel_binary_recall": f"{tonsil_summary['novel_binary_recall']:.4f}",
            "novel_hungarian_mapping": tonsil_summary["novel_hungarian_mapping"],
            "backend": tonsil_summary["backend_used"],
            "epochs": tonsil_epochs,
            "sample_rate": sample_rate,
        },
    )
    (results_dir / "tonsil_be_report.md").write_text(tonsil_md, encoding="utf-8")

    # 2. HuBMAP B004 Colon Intra-Donor Cross-Region Holdout
    b004_summary, b004_ev = run_hubmap_b004_region_benchmark(
        data_dir / "B004_training_dryad.csv",
        tissue="CL",
        holdout_region="reg004_CL_B004",
        sample_rate=1.0 if not quick else sample_rate,
        epochs=hubmap_epochs,
        backend=backend,
        seed=seed,
    )
    b004_ev.per_class_df.to_csv(results_dir / "hubmap_b004_per_class.csv")
    b004_ev.confusion_matrix_raw.to_csv(results_dir / "hubmap_b004_confusion_matrix.csv")
    b004_md = format_benchmark_report(
        b004_ev,
        method_name="STELLAR",
        dataset_name="HuBMAP B004 Colon Cross-Region (reg1-3 -> reg4)",
        metadata={
            "backend": b004_summary["backend_used"],
            "epochs": hubmap_epochs,
            "distance_thres": 50.0,
        },
    )
    (results_dir / "hubmap_b004_region_holdout_report.md").write_text(b004_md, encoding="utf-8")

    # 3. HuBMAP Cross-Donor Unannotated Transfer (B004 CL -> B005 CL)
    b005_summary, b005_conc_df = run_hubmap_cross_donor_transfer(
        data_dir / "B004_training_dryad.csv",
        data_dir / "B0056_unnanotated_dryad.csv",
        tissue="CL",
        target_donor="B005",
        sample_rate=sample_rate,
        epochs=hubmap_epochs,
        backend=backend,
        seed=seed,
    )
    b005_conc_df.to_csv(results_dir / "hubmap_b004_to_b005_concordance.csv", index=False)

    summary = {
        "status": "verified_on_paper_data",
        "total_runtime_sec": round(time.perf_counter() - t_start, 2),
        "data_verification": file_check,
        "equivalence_verification": equiv_summary,
        "benchmarks": {
            "tonsil_to_barretts_esophagus": tonsil_summary,
            "hubmap_b004_colon_cross_region": b004_summary,
            "hubmap_b004_to_b005_colon_transfer": b005_summary,
        },
    }
    (results_dir / "stellar_reproduction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce STELLAR benchmarks on Dryad CODEX data")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="External benchmark root containing data/, upstream_repo/, results/",
    )
    parser.add_argument("--sample-rate", type=float, default=0.5)
    parser.add_argument("--tonsil-epochs", type=int, default=20)
    parser.add_argument("--hubmap-epochs", type=int, default=20)
    parser.add_argument("--backend", choices=["auto", "pyg", "pytorch"], default="pytorch")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    summary = run_stellar_paper_reproduction(
        benchmark_dir=args.benchmark_dir,
        sample_rate=args.sample_rate,
        tonsil_epochs=args.tonsil_epochs,
        hubmap_epochs=args.hubmap_epochs,
        backend=args.backend,
        seed=args.seed,
        quick=args.quick,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
