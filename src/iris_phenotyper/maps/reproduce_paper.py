"""
MAPS Paper Reproduction Script
==============================
Reproduces the cHL CODEX and cHL2 MIBI cell-type phenotyping benchmarks from
Shaban et al., *Nature Communications* 2024
(DOI: 10.1038/s41467-023-44188-w, repo: ``mahmoodlab/MAPS``).

This script:
1. Reconstructs the exact 5-fold stratified splits of the 143,730-cell cHL CODEX
   cohort from ``Source_Data.zip`` / ``cHL_CODEX_MAPS.csv`` and the exact 5-fold
   patient leave-one-out splits of the 230,889-cell cHL2 MIBI cohort from
   ``cHL_2_MIBI.zip``.
2. Evaluates all official ``mahmoodlab/MAPS`` fold checkpoints on their exact
   held-out test splits via ``MAPSPhenotyper`` + ``evaluate_predictions``.
3. Trains fresh IRIS ``MAPSPhenotyper`` models from scratch on Fold 0 of both
   ``cHL_CODEX`` and ``cHL2_MIBI`` and compares directly against the official
   checkpoints and published Nature Communications Source Data metrics.
4. Saves standardized JSON metrics, Markdown reports (``format_benchmark_report``),
   and prediction CSVs to ``/Users/wwu/Documents/IRIS_Benchmarks/maps/results/``
   and updates ``dataset_manifest.json`` if present.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from iris_phenotyper.evaluation import (
    EvaluationResult,
    evaluate_predictions,
    format_benchmark_report,
)
from iris_phenotyper.maps import MAPSPhenotyper

DEFAULT_BENCHMARK_DIR = Path("/Users/wwu/Documents/IRIS_Benchmarks/maps")

CHL_CODEX_CLASS_NAMES: List[str] = [
    "B",
    "CD4 T",
    "CD8 T",
    "DC",
    "Endothelial",
    "Epithelial",
    "Lymphatic",
    "M1",
    "M2",
    "Mast",
    "Monocyte",
    "NK",
    "Neutrophil",
    "Other",
    "Treg",
    "Tumor",
]

CHL2_MIBI_CLASS_NAMES: List[str] = [
    "B",
    "CD4 T",
    "CD4 Treg",
    "CD8 T",
    "DC",
    "Endothelial",
    "M1",
    "M2",
    "NK",
    "Neutrophil",
    "Other",
    "Tumor",
]


def _load_chl_codex_full_table(benchmark_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load the 143,730-cell cHL CODEX dataset and canonical 50 feature columns."""
    extracted_csv = benchmark_dir / "data" / "source_data_csvs" / "cHL_CODEX_MAPS.csv"
    source_zip = benchmark_dir / "data" / "Source_Data.zip"

    if extracted_csv.exists():
        df = pd.read_csv(extracted_csv)
    elif source_zip.exists():
        extracted_csv.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_zip) as zf:
            raw = zf.read("Source Data/Figure 2A/cHL_CODEX_MAPS.csv")
        extracted_csv.write_bytes(raw)
        df = pd.read_csv(extracted_csv)
    else:
        raise FileNotFoundError(
            f"Neither {extracted_csv} nor {source_zip} found. "
            "Download Source_Data.zip from Zenodo record 10067010 first."
        )

    valid_csv = benchmark_dir / "upstream_repo" / "data" / "cell_phenotyping" / "valid.csv"
    if valid_csv.exists():
        header_cols = list(pd.read_csv(valid_csv, nrows=0).columns)
        feat_cols = [c for c in header_cols if c != "cell_label"]
    else:
        non_feat = {
            "cellLabel",
            "X_cent",
            "Y_cent",
            "cellSize",
            "cellType",
            "gt_labels",
            "pred_labels",
            "gt_classes",
            "pred_classes",
        }
        markers = sorted(c for c in df.columns if c not in non_feat)
        feat_cols = markers + ["cellSize"]

    return df, feat_cols


def _load_chl2_mibi_full_table(
    benchmark_dir: Path,
) -> Optional[Tuple[pd.DataFrame, List[str], List[Any]]]:
    """Load the 230,889-cell cHL2 MIBI dataset and non-control image list from cHL_2_MIBI.zip."""
    chl2_zip = benchmark_dir / "data" / "cHL_2_MIBI.zip"
    if not chl2_zip.exists():
        return None

    non_marker_cols = {
        "cellLabel",
        "X_cent",
        "Y_cent",
        "cellSize",
        "pointNum",
        "Annotation",
        "Annotation_num",
        "tile_name",
        "image_name",
    }
    with zipfile.ZipFile(chl2_zip) as zf:
        meta = pd.read_csv(zf.open("cHL_2_MIBI/cHL2_metadata.csv"))
        control_samples = set(meta.loc[meta["control_sample"] == "Yes", "image_id"].tolist())
        image_list = meta["image_id"].tolist()

        dfs: List[pd.DataFrame] = []
        for img_id in image_list:
            df_i = pd.read_csv(zf.open(f"cHL_2_MIBI/annotation_csv/{img_id}.csv"))
            df_i["image_name"] = img_id
            dfs.append(df_i)

    data = pd.concat(dfs, axis=0, ignore_index=True)
    markers = sorted(c for c in dfs[0].columns if c not in non_marker_cols)
    feat_cols = markers + ["cellSize"]
    non_control_images = [img for img in image_list if img not in control_samples]
    return data, feat_cols, non_control_images


def _load_published_paper_metrics(benchmark_dir: Path) -> Dict[str, Any]:
    """Read official Nature Communications 2024 Source Data metric CSVs."""
    tables_dir = benchmark_dir / "data" / "source_data_tables"
    source_zip = benchmark_dir / "data" / "Source_Data.zip"
    if not (tables_dir / "Figure2B_F1_score.csv").exists() and source_zip.exists():
        tables_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_zip) as zf:
            for info in zf.infolist():
                if info.filename.endswith(".csv") and info.file_size < 100_000:
                    dest = tables_dir / Path(info.filename).name
                    dest.write_bytes(zf.read(info.filename))

    out: Dict[str, Any] = {}
    for metric_key, fname in (
        ("macro_f1", "Figure2B_F1_score.csv"),
        ("macro_precision", "Figure2B_Precision.csv"),
        ("macro_recall", "Figure2B_Recall.csv"),
    ):
        csv_path = tables_dir / fname
        if not csv_path.exists():
            continue
        tbl = pd.read_csv(csv_path)
        maps_rows = tbl[tbl["Method"] == "MAPS"]
        for _, row in maps_rows.iterrows():
            ds = str(row["Datasets"])
            ds_dict = out.setdefault(ds, {})
            ds_dict[metric_key] = {
                "folds": [float(row[f"Fold {k}"]) for k in range(5)],
                "mean": float(row["Mean"]),
                "std": float(row["Std"]),
            }
    return out


def _reconstruct_chl_codex_folds(
    df: pd.DataFrame,
    feat_cols: Sequence[str],
    seed: int = 7325111,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Reproduce exact 5-fold train/valid/test indices matching cHL_CODEX.py."""
    X = df[list(feat_cols)].to_numpy(dtype=np.float64)
    y = df["gt_labels"].to_numpy(dtype=np.int64)

    skf_outer = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for train_full_idx, test_idx in skf_outer.split(X, y):
        skf_inner = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        tr_sub, va_sub = next(skf_inner.split(X[train_full_idx], y[train_full_idx]))
        folds.append((train_full_idx[tr_sub], train_full_idx[va_sub], test_idx))
    return folds


def _reconstruct_chl2_mibi_folds(
    df: pd.DataFrame,
    feat_cols: Sequence[str],
    non_control_images: Sequence[Any],
    seed: int = 7325111,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Reproduce exact 5-fold patient-level splits matching cHL2_MIBI.py."""
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for test_img in non_control_images:
        train_full_idx = df.index[df["image_name"] != test_img].to_numpy()
        test_idx = df.index[df["image_name"] == test_img].to_numpy()

        X_tr = df.iloc[train_full_idx][list(feat_cols)].to_numpy(dtype=np.float64)
        y_tr = df.iloc[train_full_idx]["Annotation"].to_numpy()
        skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        tr_sub, va_sub = next(skf.split(X_tr, y_tr))
        folds.append((train_full_idx[tr_sub], train_full_idx[va_sub], test_idx))
    return folds


def _make_iris_frame(df: pd.DataFrame, feat_cols: Sequence[str]) -> pd.DataFrame:
    """Format a slice of a single-cell table with 1-indexed ObjectNumber and features."""
    out = df[list(feat_cols)].copy().reset_index(drop=True)
    if "cellLabel" in df.columns:
        obj_ids = pd.to_numeric(df["cellLabel"], errors="coerce").fillna(0).astype(int)
        if np.all(obj_ids > 0) and obj_ids.is_unique:
            out.insert(0, "ObjectNumber", obj_ids.to_numpy())
            return out
    out.insert(0, "ObjectNumber", np.arange(1, len(out) + 1, dtype=int))
    return out


def _evaluate_5fold_checkpoints(
    ckpt_dir: Path,
    df: pd.DataFrame,
    feat_cols: Sequence[str],
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    class_names: Sequence[str],
    y_all_str: np.ndarray,
    device: str = "auto",
) -> Tuple[EvaluationResult, List[Dict[str, Any]], pd.DataFrame]:
    """Evaluate 5 official fold checkpoints on their respective held-out test splits."""
    oof_true: List[str] = []
    oof_pred: List[str] = []
    oof_prob_frames: List[pd.DataFrame] = []
    oof_records: List[pd.DataFrame] = []
    fold_summaries: List[Dict[str, Any]] = []

    for fid, (_, _, te_idx) in enumerate(folds):
        ckpt = ckpt_dir / f"fold_{fid}" / "best_checkpoint.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing official checkpoint: {ckpt}")

        te_slice = df.iloc[te_idx]
        te_features = _make_iris_frame(te_slice, feat_cols)
        y_true = y_all_str[te_idx].tolist()

        model = MAPSPhenotyper.from_checkpoint(
            ckpt,
            class_names=class_names,
            feature_cols=feat_cols,
            device=device,
        )
        res = model.predict(te_features)
        ev = evaluate_predictions(
            y_true,
            res.labels.to_numpy(),
            res.probabilities,
            labels=class_names,
        )
        macro_p, macro_r, _, _ = precision_recall_fscore_support(
            y_true,
            res.labels.to_numpy(),
            labels=list(class_names),
            average="macro",
            zero_division=0,
        )

        fold_summaries.append(
            {
                "fold": fid,
                "n_test": int(len(te_idx)),
                "accuracy": float(ev.accuracy),
                "balanced_accuracy": float(ev.balanced_accuracy),
                "macro_f1": float(ev.macro_f1),
                "weighted_f1": float(ev.weighted_f1),
                "cohens_kappa": float(ev.cohens_kappa),
                "macro_precision": float(macro_p),
                "macro_recall": float(macro_r),
            }
        )

        oof_true.extend(y_true)
        oof_pred.extend(res.labels.astype(str).tolist())
        oof_prob_frames.append(res.probabilities.reset_index(drop=True))

        fold_pred_df = res.to_dataframe(include_probabilities=False)
        fold_pred_df.insert(1, "fold", fid)
        fold_pred_df.insert(2, "true_cell_type", y_true)
        oof_records.append(fold_pred_df)

    oof_probs = pd.concat(oof_prob_frames, axis=0, ignore_index=True)
    oof_eval = evaluate_predictions(
        oof_true,
        oof_pred,
        oof_probs,
        labels=class_names,
    )
    oof_df = pd.concat(oof_records, axis=0, ignore_index=True)
    return oof_eval, fold_summaries, oof_df


def evaluate_official_tutorial_checkpoint(
    benchmark_dir: Path,
    device: str = "auto",
) -> Optional[EvaluationResult]:
    """Evaluate official cHL_CODEX.pt on upstream_repo/data/cell_phenotyping/valid.csv."""
    valid_csv = benchmark_dir / "upstream_repo" / "data" / "cell_phenotyping" / "valid.csv"
    ckpt = (
        benchmark_dir
        / "upstream_repo"
        / "models"
        / "cell_phenotyping"
        / "cHL_CODEX.pt"
    )
    if not valid_csv.exists() or not ckpt.exists():
        return None

    va_df = pd.read_csv(valid_csv)
    feat_cols = [c for c in va_df.columns if c != "cell_label"]
    y_true = [CHL_CODEX_CLASS_NAMES[int(i)] for i in va_df["cell_label"].to_numpy()]

    va_iris = _make_iris_frame(va_df, feat_cols)
    model = MAPSPhenotyper.from_checkpoint(
        ckpt,
        class_names=CHL_CODEX_CLASS_NAMES,
        feature_cols=feat_cols,
        device=device,
    )
    res = model.predict(va_iris)
    return evaluate_predictions(
        y_true,
        res.labels.to_numpy(),
        res.probabilities,
        labels=CHL_CODEX_CLASS_NAMES,
    )


def _train_and_eval_fold0(
    results_dir: Path,
    dataset_tag: str,
    df: pd.DataFrame,
    feat_cols: Sequence[str],
    fold0: Tuple[np.ndarray, np.ndarray, np.ndarray],
    class_names: Sequence[str],
    y_all_str: np.ndarray,
    *,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    device: str,
    seed: int = 7325111,
) -> Tuple[EvaluationResult, Dict[str, Any], Path]:
    """Train IRIS MAPSPhenotyper from scratch on Fold 0 and evaluate on held-out test."""
    tr_idx, va_idx, te_idx = fold0

    tr_df = _make_iris_frame(df.iloc[tr_idx], feat_cols)
    tr_df["cell_type"] = y_all_str[tr_idx].tolist()

    va_df = _make_iris_frame(df.iloc[va_idx], feat_cols)
    va_df["cell_type"] = y_all_str[va_idx].tolist()

    te_df = _make_iris_frame(df.iloc[te_idx], feat_cols)
    y_test = y_all_str[te_idx].tolist()

    model = MAPSPhenotyper(
        hidden_dim=512,
        dropout=0.25,
        batch_size=128,
        learning_rate=0.001,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        patience=patience,
        scale_divisor=255.0,
        seed=seed,
        device=device,
        verbose=1,
    )
    model.fit(
        tr_df,
        label_col="cell_type",
        valid_features=va_df,
        feature_cols=feat_cols,
        class_names=class_names,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    saved_ckpt = model.save(results_dir / f"maps_{dataset_tag}_iris_fold0.pt")

    res = model.predict(te_df)
    ev = evaluate_predictions(
        y_test,
        res.labels.to_numpy(),
        res.probabilities,
        labels=class_names,
    )
    macro_p, macro_r, _, _ = precision_recall_fscore_support(
        y_test,
        res.labels.to_numpy(),
        labels=list(class_names),
        average="macro",
        zero_division=0,
    )

    pred_df = res.to_dataframe(include_probabilities=True)
    pred_df.insert(1, "true_cell_type", y_test)
    pred_df.to_csv(results_dir / f"maps_{dataset_tag}_fold0_iris_predictions.csv", index=False)

    train_meta = {
        "n_train": int(len(tr_idx)),
        "n_valid": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "best_epoch": int(model.fit_summary_["best_epoch"]),
        "best_valid_loss": float(model.fit_summary_["best_valid_loss"]),
        "epochs_run": int(model.fit_summary_["epochs_run"]),
        "elapsed_sec": float(model.fit_summary_["elapsed_sec"]),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "device": str(res.metadata.get("device")),
        "dtype": str(res.metadata.get("dtype")),
    }
    return ev, train_meta, saved_ckpt


def _reproduce_dataset(
    *,
    benchmark_dir: Path,
    results_dir: Path,
    dataset_tag: str,
    paper_key: str,
    ckpt_subdir: str,
    df: pd.DataFrame,
    feat_cols: Sequence[str],
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    class_names: Sequence[str],
    y_all_str: np.ndarray,
    paper_metrics: Dict[str, Any],
    max_epochs: int,
    min_epochs: int,
    patience: int,
    device: str,
    skip_train: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    """Run official 5-fold verification + optional fresh Fold 0 training for one dataset."""
    oof_eval, fold_summaries, oof_df = _evaluate_5fold_checkpoints(
        benchmark_dir / "upstream_repo" / "models" / ckpt_subdir,
        df,
        feat_cols,
        folds,
        class_names,
        y_all_str,
        device=device,
    )
    oof_csv_path = results_dir / f"maps_{dataset_tag}_oof_predictions.csv"
    oof_df.to_csv(oof_csv_path, index=False)

    fresh_eval: Optional[EvaluationResult] = None
    fresh_meta: Optional[Dict[str, Any]] = None
    fresh_ckpt: Optional[Path] = None
    if not skip_train:
        fresh_eval, fresh_meta, fresh_ckpt = _train_and_eval_fold0(
            results_dir,
            dataset_tag,
            df,
            feat_cols,
            folds[0],
            class_names,
            y_all_str,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            patience=patience,
            device=device,
        )

    ds_paper = paper_metrics.get(paper_key, {})
    paper_f1_mean = ds_paper.get("macro_f1", {}).get("mean")
    paper_f1_std = ds_paper.get("macro_f1", {}).get("std")
    paper_prec_mean = ds_paper.get("macro_precision", {}).get("mean")
    paper_rec_mean = ds_paper.get("macro_recall", {}).get("mean")
    paper_f1_fold0 = (
        ds_paper.get("macro_f1", {}).get("folds", [None])[0]
        if ds_paper.get("macro_f1")
        else None
    )

    fold_macro_f1s = [f["macro_f1"] for f in fold_summaries]
    fold_accs = [f["accuracy"] for f in fold_summaries]
    fold_kappas = [f["cohens_kappa"] for f in fold_summaries]

    payload: Dict[str, Any] = {
        "dataset": dataset_tag,
        "paper_table_key": paper_key,
        "n_cells_total": int(len(df)),
        "n_cells_evaluated_oof": int(oof_eval.n_samples),
        "n_features": int(len(feat_cols)),
        "n_classes": len(class_names),
        "classes": list(class_names),
        "oof_predictions_csv": str(oof_csv_path),
        "published_paper": {
            "macro_f1_mean": paper_f1_mean,
            "macro_f1_std": paper_f1_std,
            "macro_f1_fold0": paper_f1_fold0,
            "macro_precision_mean": paper_prec_mean,
            "macro_recall_mean": paper_rec_mean,
        },
        "official_5fold_checkpoints": {
            "per_fold": fold_summaries,
            "mean_accuracy": float(np.mean(fold_accs)),
            "std_accuracy": float(np.std(fold_accs)),
            "mean_macro_f1": float(np.mean(fold_macro_f1s)),
            "std_macro_f1": float(np.std(fold_macro_f1s)),
            "mean_cohens_kappa": float(np.mean(fold_kappas)),
            "pooled_oof": oof_eval.to_json_dict(),
        },
    }
    if fresh_eval is not None and fresh_meta is not None:
        payload["iris_fresh_trained_fold0"] = {
            "training": fresh_meta,
            "checkpoint_path": str(fresh_ckpt) if fresh_ckpt else None,
            "evaluation": fresh_eval.to_json_dict(),
        }

    sections: List[str] = [
        format_benchmark_report(
            oof_eval,
            method_name="MAPS (Official 5-Fold OOF via IRIS)",
            dataset_name=f"{paper_key} ({oof_eval.n_samples:,} test cells, {len(class_names)} classes)",
            paper_metrics={"macro_f1": paper_f1_mean} if paper_f1_mean is not None else None,
            metadata={
                "5-Fold Mean Macro F1": (
                    f"{np.mean(fold_macro_f1s):.4f} +/- {np.std(fold_macro_f1s):.4f}"
                ),
                "5-Fold Mean Accuracy": (
                    f"{np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}"
                ),
                "Published Paper 5-Fold Macro F1": (
                    f"{paper_f1_mean:.4f} +/- {paper_f1_std:.4f}"
                    if paper_f1_mean is not None and paper_f1_std is not None
                    else "N/A"
                ),
            },
        )
    ]
    if fresh_eval is not None and fresh_meta is not None:
        sections.append(
            format_benchmark_report(
                fresh_eval,
                method_name="MAPS (IRIS Freshly Trained Fold 0)",
                dataset_name=f"{paper_key} Fold 0 Test ({fresh_eval.n_samples:,} cells)",
                paper_metrics={"macro_f1": paper_f1_fold0} if paper_f1_fold0 is not None else None,
                metadata={
                    "Device": fresh_meta["device"],
                    "Epochs Run": fresh_meta["epochs_run"],
                    "Best Epoch": fresh_meta["best_epoch"],
                    "Training Time (s)": f"{fresh_meta['elapsed_sec']:.1f}",
                },
            )
        )
    return payload, sections


def run_reproduction(
    benchmark_dir: Path = DEFAULT_BENCHMARK_DIR,
    *,
    max_epochs: int = 160,
    min_epochs: int = 80,
    patience: int = 30,
    device: str = "auto",
    skip_train: bool = False,
) -> Dict[str, Any]:
    """Run full MAPS reproduction pipeline and write metrics, report, and CSV outputs."""
    results_dir = benchmark_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    paper_metrics = _load_published_paper_metrics(benchmark_dir)

    all_sections: List[str] = []
    datasets_out: Dict[str, Any] = {}

    # 1. cHL CODEX (143,730 cells, 16 classes, 50 features)
    codex_df, codex_feats = _load_chl_codex_full_table(benchmark_dir)
    codex_folds = _reconstruct_chl_codex_folds(codex_df, codex_feats)
    codex_y_str = np.asarray(
        [CHL_CODEX_CLASS_NAMES[int(i)] for i in codex_df["gt_labels"].to_numpy()],
        dtype=object,
    )
    codex_payload, codex_sections = _reproduce_dataset(
        benchmark_dir=benchmark_dir,
        results_dir=results_dir,
        dataset_tag="cHL_CODEX",
        paper_key="cHL (CODEX)",
        ckpt_subdir="cHL CODEX",
        df=codex_df,
        feat_cols=codex_feats,
        folds=codex_folds,
        class_names=CHL_CODEX_CLASS_NAMES,
        y_all_str=codex_y_str,
        paper_metrics=paper_metrics,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        patience=patience,
        device=device,
        skip_train=skip_train,
    )
    datasets_out["cHL_CODEX"] = codex_payload
    all_sections.extend(codex_sections)

    tutorial_eval = evaluate_official_tutorial_checkpoint(benchmark_dir, device=device)
    if tutorial_eval is not None:
        codex_payload["official_tutorial_checkpoint_valid"] = tutorial_eval.to_json_dict()

    # 2. cHL2 MIBI (230,889 cells across 6 FOVs / 192,795 test cells, 12 classes, 47 features)
    chl2_loaded = _load_chl2_mibi_full_table(benchmark_dir)
    if chl2_loaded is not None:
        chl2_df, chl2_feats, non_ctrl_imgs = chl2_loaded
        chl2_folds = _reconstruct_chl2_mibi_folds(chl2_df, chl2_feats, non_ctrl_imgs)
        chl2_y_str = chl2_df["Annotation"].astype(str).to_numpy(dtype=object)
        chl2_payload, chl2_sections = _reproduce_dataset(
            benchmark_dir=benchmark_dir,
            results_dir=results_dir,
            dataset_tag="cHL2_MIBI",
            paper_key="cHL2 (MIBI)",
            ckpt_subdir="cHL2 MIBI",
            df=chl2_df,
            feat_cols=chl2_feats,
            folds=chl2_folds,
            class_names=CHL2_MIBI_CLASS_NAMES,
            y_all_str=chl2_y_str,
            paper_metrics=paper_metrics,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            patience=patience,
            device=device,
            skip_train=skip_train,
        )
        datasets_out["cHL2_MIBI"] = chl2_payload
        all_sections.extend(chl2_sections)

    metrics_payload: Dict[str, Any] = {
        "method": "MAPS",
        "paper_doi": "10.1038/s41467-023-44188-w",
        "datasets": datasets_out,
    }

    metrics_json_path = results_dir / "maps_reproduction_metrics.json"
    metrics_json_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")

    report_path = results_dir / "maps_reproduction_report.md"
    report_path.write_text("\n---\n\n".join(all_sections), encoding="utf-8")

    manifest_path = benchmark_dir / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary: Dict[str, Any] = {
            "metrics_json": str(metrics_json_path),
            "report_md": str(report_path),
        }
        for tag, ds_data in datasets_out.items():
            oof_blk = ds_data["official_5fold_checkpoints"]
            pooled = oof_blk["pooled_oof"]
            pub = ds_data["published_paper"]
            ds_summary: Dict[str, Any] = {
                "n_cells_evaluated_oof": ds_data["n_cells_evaluated_oof"],
                "official_5fold_mean_macro_f1": oof_blk["mean_macro_f1"],
                "official_5fold_std_macro_f1": oof_blk["std_macro_f1"],
                "official_5fold_mean_accuracy": oof_blk["mean_accuracy"],
                "official_5fold_pooled_accuracy": pooled["accuracy"],
                "official_5fold_pooled_weighted_f1": pooled["weighted_f1"],
                "official_5fold_pooled_cohens_kappa": pooled["cohens_kappa"],
                "paper_5fold_macro_f1_mean": pub.get("macro_f1_mean"),
                "paper_5fold_macro_f1_std": pub.get("macro_f1_std"),
                "oof_predictions_csv": ds_data["oof_predictions_csv"],
            }
            if "iris_fresh_trained_fold0" in ds_data:
                fr = ds_data["iris_fresh_trained_fold0"]["evaluation"]
                ds_summary["fresh_iris_fold0_accuracy"] = fr["accuracy"]
                ds_summary["fresh_iris_fold0_macro_f1"] = fr["macro_f1"]
                ds_summary["fresh_iris_fold0_weighted_f1"] = fr["weighted_f1"]
                ds_summary["fresh_iris_fold0_cohens_kappa"] = fr["cohens_kappa"]
                ds_summary["paper_fold0_macro_f1"] = pub.get("macro_f1_fold0")
            summary[tag] = ds_summary
        manifest["reproduction_summary"] = summary
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce mahmoodlab/MAPS cHL CODEX and cHL2 MIBI benchmarks."
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="Path to /Users/wwu/Documents/IRIS_Benchmarks/maps",
    )
    parser.add_argument("--max-epochs", type=int, default=160)
    parser.add_argument("--min-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    metrics = run_reproduction(
        benchmark_dir=args.benchmark_dir,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        device=args.device,
        skip_train=args.skip_train,
    )
    for tag, ds_data in metrics["datasets"].items():
        oof = ds_data["official_5fold_checkpoints"]
        paper_f1 = ds_data["published_paper"].get("macro_f1_mean")
        print(
            f"[{tag}] Official 5-Fold: mean_macro_f1={oof['mean_macro_f1']:.4f} "
            f"(paper={paper_f1:.4f}), mean_acc={oof['mean_accuracy']:.4f}, "
            f"kappa={oof['mean_cohens_kappa']:.4f}"
        )
        if "iris_fresh_trained_fold0" in ds_data:
            fr = ds_data["iris_fresh_trained_fold0"]["evaluation"]
            print(
                f"[{tag}] IRIS Fresh Fold 0: acc={fr['accuracy']:.4f}, "
                f"macro_f1={fr['macro_f1']:.4f}, weighted_f1={fr['weighted_f1']:.4f}, "
                f"kappa={fr['cohens_kappa']:.4f}"
            )


if __name__ == "__main__":
    main()
