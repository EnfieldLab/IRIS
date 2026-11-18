"""
Nimbus Paper Reproduction Benchmark Script
==========================================
Reproduces Nimbus zero-shot marker positivity classification and downstream
cell clustering on the Pan-Multiplex-Gold-Standard benchmark:

    Rumberger, J. L. et al., "Automated classification of cellular expression
    in multiplexed imaging data with Nimbus", Nature Methods 22, 2161-2170
    (2025), bioRxiv DOI: 10.1101/2024.06.02.597062.

Evaluates pretrained ``V1.pt`` predictions against expert proofread ground
truth (0 = negative, 1 = positive; 2/3 = ambiguous excluded as in paper
``figure_4_.ipynb``) and computes overall, per-dataset, and per-marker
AUROC, AUPRC, F1, precision, recall, specificity, and downstream cluster
metrics.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    silhouette_score,
)

from iris_phenotyper.evaluation import (
    _format_df_markdown,
    evaluate_predictions,
    format_benchmark_report,
)
from iris_phenotyper.nimbus.adapter import NimbusPhenotyper
from iris_phenotyper.nimbus.pipeline import (
    cluster_positivity_profiles,
    is_upstream_available,
)

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_ROOT = Path("/Users/wwu/Documents/IRIS_Benchmarks/nimbus")
DEFAULT_DATA_DIR = DEFAULT_BENCHMARK_ROOT / "data" / "gold_standard_labelled"
DEFAULT_RESULTS_DIR = DEFAULT_BENCHMARK_ROOT / "results"
DEFAULT_CHECKPOINT = DEFAULT_BENCHMARK_ROOT / "checkpoints" / "V1.pt"

# Published Pan-Multiplex Gold-Standard metrics for Nimbus (noise_robust Res-U-Net)
# from angelolab/publications 2024-Rumberger_Greenwald_etal_Nimbus/data/experimental_results.csv
PUBLISHED_PAPER_METRICS: Dict[str, Any] = {
    "paper_title": (
        "Automated classification of cellular expression in multiplexed imaging data with Nimbus"
    ),
    "biorxiv_doi": "10.1101/2024.06.02.597062",
    "journal_doi": "10.1038/s41592-025-02826-9",
    "source": "JLrumberger/Pan-Multiplex experimental_results.csv (noise_robust Res-U-Net)",
    "precision": 0.8541787242637255,
    "recall": 0.8497555525408883,
    "specificity": 0.9853679125984568,
    "f1": 0.8501606480725389,
    "threshold": 0.5,
}

# Dataset folder -> default image suffix (from 3_interactive_viewer.ipynb)
DATASET_SUFFIX_MAP: Dict[str, str] = {
    "codex_colon": ".ome.tif",
    "mibi_decidua": ".tif",
    "vectra_colon": ".ome.tif",
    "vectra_pancreas": ".ome.tif",
    "mibi_breast": ".tiff",
}

# Dataset-specific default magnifications matching Pan-M metadata (20x standard, 10x model ref)
DATASET_MAGNIFICATION_MAP: Dict[str, float] = {
    "codex_colon": 20.0,
    "mibi_decidua": 20.0,
    "vectra_colon": 20.0,
    "vectra_pancreas": 20.0,
    "mibi_breast": 20.0,
}


def _find_ground_truth_csv(data_dir: Path) -> Path:
    candidates = [
        data_dir / "gold_standard_groundtruth.csv",
        data_dir.parent / "gold_standard_groundtruth.csv",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    found = list(data_dir.rglob("*groundtruth*.csv"))
    if found:
        return found[0]
    raise FileNotFoundError(f"Could not find gold_standard_groundtruth.csv under {data_dir}")


def _resolve_fov_paths(dataset_dir: Path, suffix: str) -> List[Path]:
    fovs_dir = dataset_dir / "fovs"
    if not fovs_dir.is_dir():
        return []
    entries = sorted(
        p for p in fovs_dir.iterdir() if not p.name.startswith(".") and (p.is_dir() or p.name.endswith(suffix))
    )
    return entries


def _canonicalize_fov_name(raw_name: str, suffix: str) -> str:
    name = str(raw_name)
    for ext in (suffix, ".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_bin_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin_pred, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2.0 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    accuracy = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0
    auroc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None
    auprc = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None
    return {
        "n_total": int(len(y_true)),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "auroc": auroc,
        "auprc": auprc,
    }


def _detect_gt_columns(gt_df: pd.DataFrame) -> Dict[str, str]:
    cols = {c.lower(): c for c in gt_df.columns}
    mapping: Dict[str, str] = {}
    for target, cands in (
        ("dataset", ("dataset", "subset")),
        ("fov", ("fov", "image", "sample")),
        ("cell_id", ("cell_id", "objectnumber", "label", "labels")),
        ("channel", ("channel", "marker")),
        ("activity", ("activity", "gt_proofread", "gold_standard", "label_gt")),
    ):
        for cand in cands:
            if cand in cols:
                mapping[target] = cols[cand]
                break
        if target not in mapping:
            raise KeyError(
                f"Ground truth CSV missing expected '{target}' column. Found: {list(gt_df.columns)}"
            )
    return mapping


def run_benchmark(
    data_dir: Path = DEFAULT_DATA_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    checkpoint: Union[str, Path] = DEFAULT_CHECKPOINT,
    *,
    datasets: Optional[Sequence[str]] = None,
    max_fovs_per_dataset: Optional[int] = None,
    device: str = "auto",
    test_time_aug: bool = True,
    batch_size: int = 4,
    threshold: float = 0.5,
    use_upstream: Union[bool, str] = "auto",
) -> Dict[str, Any]:
    """Run Nimbus pretrained inference on Pan-Multiplex-Gold-Standard and save evaluation."""
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    gt_csv = _find_ground_truth_csv(data_dir)
    gt_raw = pd.read_csv(gt_csv)
    col_map = _detect_gt_columns(gt_raw)
    gt_df = gt_raw.rename(
        columns={
            col_map["dataset"]: "dataset",
            col_map["fov"]: "fov",
            col_map["cell_id"]: "cell_id",
            col_map["channel"]: "channel",
            col_map["activity"]: "activity",
        }
    ).copy()
    gt_df["cell_id"] = pd.to_numeric(gt_df["cell_id"], errors="coerce").astype("Int64")
    gt_df["activity"] = pd.to_numeric(gt_df["activity"], errors="coerce")

    available_datasets = [
        d for d in DATASET_SUFFIX_MAP if (data_dir / d / "fovs").is_dir() and (data_dir / d / "masks").is_dir()
    ]
    if datasets is not None:
        target_datasets = [d for d in datasets if d in available_datasets]
    else:
        target_datasets = available_datasets
    if not target_datasets:
        raise FileNotFoundError(
            f"No Gold-Standard datasets with fovs/ and masks/ found under {data_dir}."
        )

    ckpt_arg: Union[str, Path] = checkpoint if Path(checkpoint).is_file() else "latest"
    merged_records: List[pd.DataFrame] = []
    cluster_records: List[Dict[str, Any]] = []

    for dset in target_datasets:
        suffix = DATASET_SUFFIX_MAP.get(dset, ".ome.tif")
        dset_dir = data_dir / dset
        fov_paths = _resolve_fov_paths(dset_dir, suffix=suffix)
        if not fov_paths:
            continue
        if max_fovs_per_dataset is not None:
            fov_paths = fov_paths[: int(max_fovs_per_dataset)]

        dset_gt = gt_df[gt_df["dataset"] == dset].copy()
        if dset_gt.empty:
            continue

        target_fov_names = {_canonicalize_fov_name(p.name, suffix) for p in fov_paths}
        dset_gt["fov_key"] = dset_gt["fov"].astype(str).map(lambda x: _canonicalize_fov_name(x, suffix))
        dset_gt_sub = dset_gt[dset_gt["fov_key"].isin(target_fov_names)]
        channels = sorted(dset_gt_sub["channel"].dropna().astype(str).unique().tolist())
        if not channels:
            channels = sorted(dset_gt["channel"].dropna().astype(str).unique().tolist())

        logger.info(
            "Running Nimbus on dataset=%s (%d FOVs, %d GT channels, upstream=%s)",
            dset,
            len(fov_paths),
            len(channels),
            use_upstream,
        )
        pheno = NimbusPhenotyper(
            checkpoint=ckpt_arg,
            dataset_magnification=DATASET_MAGNIFICATION_MAP.get(dset, 20.0),
            model_magnification=10.0,
            batch_size=batch_size,
            test_time_aug=test_time_aug,
            device=device,
            use_upstream=use_upstream,
            return_pixel_maps=False,
        )
        pred_res = pheno.predict(
            fov_paths=fov_paths,
            mask_dir=dset_dir / "masks",
            approx_mask_match=True,
            channel_names=channels,
            suffix=suffix,
        )
        cell_table = pred_res.metadata["cell_table"].copy()
        cell_table["fov_key"] = cell_table["Image"].astype(str).map(
            lambda x: _canonicalize_fov_name(x, suffix)
        )
        present_channels = [c for c in channels if c in cell_table.columns]

        # Downstream consensus clustering (mirroring 4_Generic_Cell_Clustering.ipynb)
        if len(cell_table) >= 10 and len(present_channels) >= 2:
            pos_mat = cell_table[present_channels].copy()
            pos_mat.index = pd.Index(cell_table["ObjectNumber"].astype(int), name="ObjectNumber")
            cl_labels, cl_probs, cl_conf = cluster_positivity_profiles(
                pos_mat, max_k=min(15, len(present_channels) + 2)
            )
            n_unique_cl = int(cl_labels.nunique())
            sil = (
                float(silhouette_score(pos_mat.to_numpy(dtype=float), cl_labels.to_numpy()))
                if n_unique_cl >= 2 and len(pos_mat) > n_unique_cl
                else None
            )
            cluster_records.append(
                {
                    "dataset": dset,
                    "n_cells": int(len(cell_table)),
                    "n_markers": int(len(present_channels)),
                    "n_clusters": n_unique_cl,
                    "mean_cluster_confidence": float(cl_conf.mean()),
                    "silhouette_score": sil,
                }
            )

        melted = cell_table.melt(
            id_vars=["fov_key", "ObjectNumber"],
            value_vars=present_channels,
            var_name="channel",
            value_name="nimbus_score",
        )
        melted = melted.rename(columns={"ObjectNumber": "cell_id"})
        merged = dset_gt_sub.merge(
            melted,
            on=["fov_key", "cell_id", "channel"],
            how="inner",
        )
        merged["dataset"] = dset
        merged_records.append(merged)

    if not merged_records:
        raise RuntimeError("No ground-truth annotations matched predicted FOVs/cells.")

    all_eval_df = pd.concat(merged_records, ignore_index=True)
    # Exclude ambiguous annotations (activity >= 2: 2 or 3) per Pan-Multiplex spec
    clean_df = all_eval_df[all_eval_df["activity"].isin([0, 1])].copy()
    clean_df["activity"] = clean_df["activity"].astype(int)
    clean_df["nimbus_score"] = clean_df["nimbus_score"].astype(float)

    # Save full cell-level predictions
    clean_df.to_csv(results_dir / "nimbus_gold_standard_predictions.csv", index=False)

    # Overall pooled evaluation
    y_true_all = clean_df["activity"].to_numpy(dtype=int)
    y_score_all = clean_df["nimbus_score"].to_numpy(dtype=float)
    overall_metrics = _binary_metrics(y_true_all, y_score_all, threshold=threshold)

    # Standardized IRIS EvaluationResult for Negative vs Positive marker calling
    y_true_str = np.where(y_true_all == 1, "Positive", "Negative")
    y_pred_str = np.where(y_score_all >= threshold, "Positive", "Negative")
    y_prob_mat = np.column_stack([1.0 - y_score_all, y_score_all])
    iris_eval = evaluate_predictions(
        y_true_str,
        y_pred_str,
        y_prob=y_prob_mat,
        labels=["Negative", "Positive"],
    )

    # Cleaned subset matching figure_4_error_analysis_supp2.ipynb (drops decidua Ki67/IDO/PDL1 errors)
    paper_clean_mask = ~(
        (clean_df["dataset"] == "mibi_decidua")
        & (clean_df["channel"].isin(["Ki67", "IDO", "PDL1"]))
    )
    paper_clean_df = clean_df[paper_clean_mask]
    paper_clean_metrics = _binary_metrics(
        paper_clean_df["activity"].to_numpy(dtype=int),
        paper_clean_df["nimbus_score"].to_numpy(dtype=float),
        threshold=threshold,
    )

    # Per-dataset metrics
    dset_rows: List[Dict[str, Any]] = []
    for dset_name, grp in clean_df.groupby("dataset", sort=True):
        m = _binary_metrics(
            grp["activity"].to_numpy(dtype=int),
            grp["nimbus_score"].to_numpy(dtype=float),
            threshold=threshold,
        )
        m["dataset"] = str(dset_name)
        dset_rows.append(m)
    per_dataset_df = pd.DataFrame(dset_rows).set_index("dataset")
    per_dataset_df.to_csv(results_dir / "per_dataset_metrics.csv")

    # Per-marker metrics (across dataset, channel)
    marker_rows: List[Dict[str, Any]] = []
    for (dset_name, ch_name), grp in clean_df.groupby(["dataset", "channel"], sort=True):
        m = _binary_metrics(
            grp["activity"].to_numpy(dtype=int),
            grp["nimbus_score"].to_numpy(dtype=float),
            threshold=threshold,
        )
        m["dataset"] = str(dset_name)
        m["channel"] = str(ch_name)
        marker_rows.append(m)
    per_marker_df = pd.DataFrame(marker_rows)
    per_marker_df.to_csv(results_dir / "per_marker_metrics.csv", index=False)

    valid_aurocs = per_marker_df["auroc"].dropna()
    valid_f1s = per_marker_df.loc[per_marker_df["n_pos"] > 0, "f1"]

    clustering_df = pd.DataFrame(cluster_records)
    if not clustering_df.empty:
        clustering_df.to_csv(results_dir / "downstream_clustering_metrics.csv", index=False)

    summary: Dict[str, Any] = {
        "method": "Nimbus",
        "checkpoint": str(ckpt_arg),
        "upstream_nimbus_inference_used": bool(
            is_upstream_available() if use_upstream == "auto" else use_upstream
        ),
        "test_time_aug": bool(test_time_aug),
        "threshold": float(threshold),
        "datasets_evaluated": target_datasets,
        "overall_pooled": overall_metrics,
        "overall_paper_cleaned": paper_clean_metrics,
        "per_marker_macro": {
            "mean_auroc": float(valid_aurocs.mean()) if not valid_aurocs.empty else None,
            "median_auroc": float(valid_aurocs.median()) if not valid_aurocs.empty else None,
            "mean_f1": float(valid_f1s.mean()) if not valid_f1s.empty else None,
            "n_markers_evaluated": int(len(per_marker_df)),
        },
        "iris_evaluation": {
            "accuracy": float(iris_eval.accuracy),
            "balanced_accuracy": float(iris_eval.balanced_accuracy),
            "macro_f1": float(iris_eval.macro_f1),
            "weighted_f1": float(iris_eval.weighted_f1),
            "cohens_kappa": float(iris_eval.cohens_kappa),
            "roc_auc": iris_eval.roc_auc_macro,
        },
        "downstream_clustering": cluster_records,
        "published_paper": PUBLISHED_PAPER_METRICS,
    }

    with open(results_dir / "summary_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Generate Markdown benchmark report
    report = format_benchmark_report(
        iris_eval,
        method_name="Nimbus",
        dataset_name="Pan-Multiplex-Gold-Standard",
        paper_metrics=None,
        metadata={
            "Checkpoint": str(ckpt_arg),
            "Test-Time Augmentation": test_time_aug,
            "Datasets": ", ".join(target_datasets),
            "Pooled Positive-Class F1 (Paper Metric)": (
                f"{overall_metrics['f1']:.4f} (Paper: {PUBLISHED_PAPER_METRICS['f1']:.4f})"
            ),
            "Pooled Precision": (
                f"{overall_metrics['precision']:.4f} "
                f"(Paper: {PUBLISHED_PAPER_METRICS['precision']:.4f})"
            ),
            "Pooled Recall": (
                f"{overall_metrics['recall']:.4f} (Paper: {PUBLISHED_PAPER_METRICS['recall']:.4f})"
            ),
            "Pooled Specificity": (
                f"{overall_metrics['specificity']:.4f} "
                f"(Paper: {PUBLISHED_PAPER_METRICS['specificity']:.4f})"
            ),
            "Pooled AUROC": (
                f"{overall_metrics['auroc']:.4f}"
                if overall_metrics["auroc"] is not None
                else "N/A"
            ),
            "Pooled AUPRC": (
                f"{overall_metrics['auprc']:.4f}"
                if overall_metrics["auprc"] is not None
                else "N/A"
            ),
        },
    )
    paper_comp_rows = [
        ("Precision (Positive)", overall_metrics["precision"], PUBLISHED_PAPER_METRICS["precision"]),
        ("Recall (Positive)", overall_metrics["recall"], PUBLISHED_PAPER_METRICS["recall"]),
        ("Specificity", overall_metrics["specificity"], PUBLISHED_PAPER_METRICS["specificity"]),
        ("F1 Score (Positive)", overall_metrics["f1"], PUBLISHED_PAPER_METRICS["f1"]),
    ]
    with open(results_dir / "benchmark_report.md", "w", encoding="utf-8") as handle:
        handle.write(report)
        handle.write("\n## Paper Benchmark Comparison (Positive Class @ threshold=0.5)\n\n")
        handle.write("| Metric | IRIS Reproduced | Published Paper | Delta |\n")
        handle.write("| --- | ---: | ---: | ---: |\n")
        for label, repro_val, paper_val in paper_comp_rows:
            delta = float(repro_val) - float(paper_val)
            handle.write(f"| {label} | {repro_val:.4f} | {paper_val:.4f} | {delta:+.4f} |\n")
        handle.write("\n## Per-Dataset Metrics (Positive Class @ threshold=0.5)\n\n")
        table_df = per_dataset_df[
            ["n_total", "n_pos", "precision", "recall", "specificity", "f1", "auroc", "auprc"]
        ]
        handle.write("\n".join(_format_df_markdown(table_df)))
        handle.write("\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Nimbus on Pan-Multiplex-Gold-Standard")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max-fovs", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--native", action="store_true", help="Force native PyTorch pipeline over upstream package")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run_benchmark(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        checkpoint=args.checkpoint,
        datasets=args.datasets,
        max_fovs_per_dataset=args.max_fovs,
        device=args.device,
        test_time_aug=not args.no_tta,
        batch_size=args.batch_size,
        use_upstream=False if args.native else "auto",
    )
    print(json.dumps(summary["overall_pooled"], indent=2))
    print("Per-marker macro:", json.dumps(summary["per_marker_macro"], indent=2))


if __name__ == "__main__":
    main()
