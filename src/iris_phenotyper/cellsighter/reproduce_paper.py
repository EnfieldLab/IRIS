"""
CellSighter Paper Reproduction on Melanoma Lymph Node MIBI Dataset
================================================================
Reproduces CellSighter (Amitay et al., Nature Communications 2023,
DOI: 10.1038/s41467-023-40066-7) on the official Mendeley Data benchmark
"Multiplexed Images of Melanoma Lymph Node Metastases"
(DOI: 10.17632/zcz8743fcv.2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

from iris_phenotyper.cellsighter.adapter import CellSighterPhenotyper
from iris_phenotyper.cellsighter.pipeline import (
    load_directory_fovs,
    load_labels_mapping,
    resolve_device,
)
from iris_phenotyper.evaluation import evaluate_predictions, format_benchmark_report

DEFAULT_DATA_DIR = Path("/Users/wwu/Documents/IRIS_Benchmarks/cellsighter/data")
DEFAULT_RESULTS_DIR = Path("/Users/wwu/Documents/IRIS_Benchmarks/cellsighter/results")

DEFAULT_TRAIN_FOVS: List[str] = [str(i) for i in range(1, 11)] + ["12"]
DEFAULT_VAL_FOVS: List[str] = ["11"]
DEFAULT_TEST_FOVS: List[str] = ["13", "14", "15", "16"]

# Biological lineage grouping matching KerenLab/CellSighter config.json hierarchy_match.
MELANOMA_HIERARCHY: Dict[str, str] = {
    "0": "B cell",
    "1": "T cell",
    "2": "T cell",
    "3": "T cell",
    "4": "Myeloid",
    "5": "B cell",
    "6": "Endothelial_Stroma",
    "7": "Other",
    "8": "Myeloid",
    "9": "T cell",
    "10": "NK",
    "11": "Myeloid",
    "12": "Endothelial_Stroma",
    "13": "Tumor",
    "14": "Endothelial_Stroma",
}

# Published benchmark reference figures from Amitay et al., Nature Communications 2023.
PUBLISHED_PAPER_METRICS: Dict[str, Any] = {
    "accuracy": 0.85,
    "notes": (
        "Amitay et al. (Nat. Commun. 2023) report >80% overall accuracy (~85% across major "
        "subsets, matching inter-observer concordance), with recall ~0.93 (CD8 T cell), "
        "~0.88 (CD4 T cell), ~0.83 (CD4 Treg), and ~0.82 (Endothelial/Vessels)."
    ),
}


def run_melanoma_reproduction(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    train_fovs: Sequence[str] = DEFAULT_TRAIN_FOVS,
    val_fovs: Sequence[str] = DEFAULT_VAL_FOVS,
    test_fovs: Sequence[str] = DEFAULT_TEST_FOVS,
    crop_size: int = 128,
    crop_input_size: int = 60,
    backbone: str = "resnet50",
    normalization: str = "quantile",
    clip_quantile: float = 0.99,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    ensemble_size: int = 2,
    samples_per_epoch: Optional[int] = 10240,
    sampling_power: float = 0.5,
    device: str = "auto",
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a CellSighter ensemble on Mendeley Melanoma FOVs and evaluate on held-out FOVs."""
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    channels_file = data_dir / "channels.txt"
    labels_file = data_dir / "labels_strings.txt"
    if not channels_file.exists() or not labels_file.exists():
        raise FileNotFoundError(
            f"Missing channels.txt or labels_strings.txt in {data_dir}. "
            "Download Mendeley dataset 10.17632/zcz8743fcv.2 first."
        )

    labels_map = load_labels_mapping(labels_file)
    resolved_dev = str(resolve_device(device))
    ckpt_path = results_dir / "cellsighter_melanoma_ensemble.pt"

    print(
        f"[CellSighter] Starting Melanoma MIBI reproduction on device={resolved_dev}, "
        f"backbone={backbone}, ensemble_size={ensemble_size}, epochs={epochs}, "
        f"samples_per_epoch={samples_per_epoch}"
    )

    t0 = time.time()
    model = CellSighterPhenotyper(
        crop_size=crop_size,
        crop_input_size=crop_input_size,
        to_pad=True,
        normalization=normalization,
        clip_quantile=clip_quantile,
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        sample_batch=True,
        hierarchy_match=MELANOMA_HIERARCHY,
        sampling_power=sampling_power,
        samples_per_epoch=samples_per_epoch,
        augment=True,
        poisson_aug="auto",
        shift_max=5,
        ensemble_size=ensemble_size,
        device=resolved_dev,
        seed=seed,
        verbose=1,
    )

    model.fit(
        root_dir=data_dir,
        train_set=list(train_fovs),
        val_set=list(val_fovs),
        channels_path=channels_file,
        labels_mapping_path=labels_file,
        save_path=ckpt_path,
    )
    train_elapsed = time.time() - t0
    print(f"[CellSighter] Training complete in {train_elapsed:.1f}s; saved {ckpt_path}")

    # Load labeled held-out test crops to obtain exact ground-truth labels alongside predictions.
    t_eval = time.time()
    _, test_crops, _ = load_directory_fovs(
        data_dir,
        list(test_fovs),
        channels_path=channels_file,
        crop_size=crop_size,
        to_pad=True,
        load_labels=True,
        drop_unlabeled=True,
    )
    assert model.trainer is not None
    probs_arr = model.trainer.predict_crops(test_crops, batch_size=batch_size, device=resolved_dev)
    eval_elapsed = time.time() - t_eval

    class_names = list(model.trainer.class_names)
    idx_to_label = model.trainer.index_to_label
    raw_to_name = {int(k): str(v) for k, v in labels_map.items()}
    if not raw_to_name:
        raw_to_name = {idx_to_label[i]: name for i, name in enumerate(class_names)}

    y_true = [raw_to_name[int(c.label)] for c in test_crops]
    pred_idx = probs_arr.argmax(axis=1)
    y_pred = [class_names[int(i)] for i in pred_idx]
    y_prob_df = pd.DataFrame(probs_arr, columns=class_names)

    present_classes = sorted(set(y_true) | set(y_pred))
    eval_res = evaluate_predictions(y_true, y_pred, y_prob=y_prob_df, labels=present_classes)

    pred_df = pd.DataFrame(
        {
            "image_id": [c.image_id for c in test_crops],
            "ObjectNumber": [c.cell_id for c in test_crops],
            "true_label": y_true,
            "predicted_label": y_pred,
            "confidence": probs_arr.max(axis=1).astype(float),
        }
    )
    pred_df.to_csv(results_dir / "test_predictions.csv", index=False)
    eval_res.per_class_df.to_csv(results_dir / "per_class_metrics.csv")
    eval_res.confusion_matrix_raw.to_csv(results_dir / "confusion_matrix_raw.csv")

    report_md = format_benchmark_report(
        eval_res,
        method_name="CellSighter",
        dataset_name="Melanoma Lymph Node MIBI (10.17632/zcz8743fcv.2)",
        paper_metrics={"accuracy": PUBLISHED_PAPER_METRICS["accuracy"]},
        metadata={
            "Device": resolved_dev,
            "Backbone": backbone,
            "Ensemble Size": ensemble_size,
            "Epochs": epochs,
            "Samples Per Epoch": samples_per_epoch,
            "Normalization": f"{normalization} (q={clip_quantile})",
            "Train FOVs": ",".join(train_fovs),
            "Val FOVs": ",".join(val_fovs),
            "Test FOVs": ",".join(test_fovs),
            "Train Time (s)": f"{train_elapsed:.1f}",
            "Eval Time (s)": f"{eval_elapsed:.1f}",
        },
    )
    (results_dir / "BENCHMARK_REPORT.md").write_text(report_md, encoding="utf-8")

    summary: Dict[str, Any] = {
        "method": "CellSighter",
        "dataset_doi": "10.17632/zcz8743fcv.2",
        "paper_doi": "10.1038/s41467-023-40066-7",
        "device": resolved_dev,
        "config": {
            "crop_size": crop_size,
            "crop_input_size": crop_input_size,
            "backbone": backbone,
            "normalization": normalization,
            "clip_quantile": clip_quantile,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "ensemble_size": ensemble_size,
            "samples_per_epoch": samples_per_epoch,
            "sampling_power": sampling_power,
            "train_fovs": list(train_fovs),
            "val_fovs": list(val_fovs),
            "test_fovs": list(test_fovs),
        },
        "train_elapsed_s": round(train_elapsed, 2),
        "eval_elapsed_s": round(eval_elapsed, 2),
        "n_test_cells": len(test_crops),
        "metrics": eval_res.to_json_dict(),
        "published_paper_reference": PUBLISHED_PAPER_METRICS,
        "training_history": model.trainer.history,
        "checkpoint_path": str(ckpt_path),
    }
    (results_dir / "reproduction_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"[CellSighter] Held-out Test (n={len(test_crops)}): "
        f"accuracy={eval_res.accuracy:.4f}, balanced_acc={eval_res.balanced_accuracy:.4f}, "
        f"macro_f1={eval_res.macro_f1:.4f}, weighted_f1={eval_res.weighted_f1:.4f}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce CellSighter on the Mendeley Melanoma MIBI benchmark."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--ensemble-size", type=int, default=2)
    parser.add_argument("--samples-per-epoch", type=int, default=10240)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sampling-power", type=float, default=0.5)
    parser.add_argument("--normalization", type=str, default="quantile")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_melanoma_reproduction(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        ensemble_size=args.ensemble_size,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        sampling_power=args.sampling_power,
        normalization=args.normalization,
        backbone=args.backbone,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
