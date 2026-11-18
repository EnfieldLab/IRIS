"""
ASTIR Paper Reproduction & Benchmark Evaluation
==============================================
Reproduces the CamLab ASTIR vignette and Basel IMC benchmark on
``basel_22k_subset.h5ad`` using both the official ``camlab-bioml/astir``
package (via ``iris_astir``) and IRIS's modernized PyTorch 2.x/NumPy 2.x
``ASTIRPhenotyper``.

Evaluates:
1. Official ``astir`` vs IRIS ``ASTIRPhenotyper`` concordance (F1, Cohen's kappa,
   posterior probability Pearson r, and cell-type proportion agreement).
2. Published ``Astir-Vignette`` notebook proportions on ``basel_22k_subset.h5ad``.
3. Concordance against Jackson et al. (2020) ground-truth IMC annotations
   from ``astir-manuscript`` (Zenodo ``10.5281/zenodo.5156034``).
4. Posterior confidence calibration and canonical marker enrichment.
"""
from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats

DEFAULT_BENCHMARK_DIR = Path("/Users/wwu/Documents/IRIS_Benchmarks/astir")
DEFAULT_OFFICIAL_PYTHON = Path("/Users/wwu/.conda/envs/iris_astir/bin/python")

# Cell counts recorded in camlab-bioml/Astir-Vignette/astir_tutorial.ipynb (Cell 16, threshold=0.7)
VIGNETTE_PUBLISHED_COUNTS_T07: Dict[str, int] = {
    "Stromal": 6410,
    "Epithelial (luminal)": 4698,
    "Unknown": 4438,
    "Other": 3108,
    "Epithelial (other)": 1643,
    "Epithelial (basal)": 1321,
    "Macrophage": 416,
    "Endothelial": 336,
    "T cells": 270,
    "B cells": 77,
}

CLASS_LINEAGE_MAP: Dict[str, str] = {
    "Epithelial (basal)": "Tumor",
    "Epithelial (luminal)": "Tumor",
    "Epithelial (other)": "Tumor",
    "Stromal": "Stroma",
    "T cells": "Immune",
    "B cells": "Immune",
    "Macrophage": "Immune",
    "Endothelial": "Vessel",
    "Other": "Unclear",
    "Unknown": "Unclear",
}


def _run_official_astir_inprocess(
    h5ad_path: Path,
    marker_yaml: Path,
    out_probs_csv: Path,
    out_meta_json: Path,
    *,
    max_epochs: int = 1000,
    learning_rate: float = 2e-3,
    n_init: int = 5,
    n_init_epochs: int = 3,
    delta_loss: float = 1e-3,
    random_seed: int = 1234,
) -> Dict[str, Any]:
    """Run upstream ``astir.data.from_anndata_yaml`` and save probabilities + metadata."""
    from astir.data import from_anndata_yaml

    t0 = time.perf_counter()
    ast = from_anndata_yaml(
        str(h5ad_path),
        str(marker_yaml),
        create_design_mat=True,
        random_seed=random_seed,
    )
    n_cells = ast.get_type_dataset().get_exprs_df().shape[0]
    batch_size = max(1, int(n_cells / 100))
    ast.fit_type(
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        delta_loss=delta_loss,
        n_init=n_init,
        n_init_epochs=n_init_epochs,
    )
    elapsed_s = float(time.perf_counter() - t0)

    probs_df = ast.get_celltype_probabilities()
    probs_df.to_csv(out_probs_csv)

    labels_t07 = ast.get_celltypes(threshold=0.7)["cell_type"]
    labels_t05 = ast.get_celltypes(threshold=0.5)["cell_type"]
    losses = ast.get_type_losses().detach().cpu().numpy().tolist()

    meta = {
        "engine": "official_camlab_astir",
        "n_cells": int(n_cells),
        "batch_size": int(batch_size),
        "n_epochs_final": int(len(losses)),
        "final_loss": float(losses[-1]),
        "elapsed_s": elapsed_s,
        "counts_threshold_0_7": {str(k): int(v) for k, v in labels_t07.value_counts().items()},
        "counts_threshold_0_5": {str(k): int(v) for k, v in labels_t05.value_counts().items()},
    }
    out_meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _ensure_official_predictions(
    h5ad_path: Path,
    marker_yaml: Path,
    results_dir: Path,
    official_python: Path,
    *,
    max_epochs: int,
    learning_rate: float,
    n_init: int,
    n_init_epochs: int,
    delta_loss: float,
    random_seed: int,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Run official ``astir`` either in-process or via dedicated ``iris_astir`` env."""
    probs_csv = results_dir / "basel_22k_official_probabilities.csv"
    meta_json = results_dir / "basel_22k_official_meta.json"

    try:
        import astir  # noqa: F401

        meta = _run_official_astir_inprocess(
            h5ad_path,
            marker_yaml,
            probs_csv,
            meta_json,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            n_init=n_init,
            n_init_epochs=n_init_epochs,
            delta_loss=delta_loss,
            random_seed=random_seed,
        )
        probs_df = pd.read_csv(probs_csv, index_col=0)
        return probs_df, meta
    except ImportError:
        pass

    if official_python.exists():
        cmd = [
            str(official_python),
            str(Path(__file__).resolve()),
            "--official-only",
            "--h5ad",
            str(h5ad_path),
            "--marker-yaml",
            str(marker_yaml),
            "--results-dir",
            str(results_dir),
            "--max-epochs",
            str(max_epochs),
            "--learning-rate",
            str(learning_rate),
            "--n-init",
            str(n_init),
            "--n-init-epochs",
            str(n_init_epochs),
            "--delta-loss",
            str(delta_loss),
            "--random-seed",
            str(random_seed),
        ]
        subprocess.run(cmd, check=True)
        probs_df = pd.read_csv(probs_csv, index_col=0)
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        return probs_df, meta

    return None, None


def _compute_marker_enrichment(
    expr_df: pd.DataFrame,
    labels: pd.Series,
    cell_types_markers: Mapping[str, List[str]],
) -> pd.DataFrame:
    """Compute mean marker intensity in target type vs all other cells + Welch t-test."""
    rows: List[Dict[str, Any]] = []
    label_arr = labels.to_numpy(dtype=object)
    for ct, markers in cell_types_markers.items():
        mask_in = label_arr == ct
        mask_out = ~mask_in
        n_in = int(mask_in.sum())
        if n_in < 2 or int(mask_out.sum()) < 2:
            continue
        for marker in markers:
            if marker not in expr_df.columns:
                continue
            vals = expr_df[marker].to_numpy(dtype=np.float64)
            v_in = vals[mask_in]
            v_out = vals[mask_out]
            m_in = float(np.mean(v_in))
            m_out = float(np.mean(v_out))
            tt = stats.ttest_ind(v_in, v_out, equal_var=False)
            rows.append(
                {
                    "cell_type": ct,
                    "marker": marker,
                    "n_cells": n_in,
                    "mean_in_type": m_in,
                    "mean_other_cells": m_out,
                    "delta_expr": m_in - m_out,
                    "fold_enrichment": float((m_in + 1e-6) / (m_out + 1e-6)),
                    "t_stat": float(tt.statistic),
                    "p_value": float(tt.pvalue),
                }
            )
    return pd.DataFrame(rows)


def _compute_calibration_bins(
    confidence: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> List[Dict[str, Any]]:
    """Evaluate accuracy stratified by posterior confidence bins."""
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]
    conf_arr = confidence.to_numpy(dtype=float)
    yt_arr = y_true.to_numpy(dtype=object)
    yp_arr = y_pred.to_numpy(dtype=object)
    records: List[Dict[str, Any]] = []
    for lo, hi in bins:
        mask = (conf_arr >= lo) & (conf_arr < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            records.append(
                {
                    "bin": f"[{lo:.1f}, {min(hi, 1.0):.1f}]",
                    "n_cells": 0,
                    "fraction": 0.0,
                    "mean_confidence": None,
                    "lineage_accuracy": None,
                }
            )
            continue
        records.append(
            {
                "bin": f"[{lo:.1f}, {min(hi, 1.0):.1f}]",
                "n_cells": n_bin,
                "fraction": float(n_bin / len(conf_arr)),
                "mean_confidence": float(np.mean(conf_arr[mask])),
                "lineage_accuracy": float(np.mean(yt_arr[mask] == yp_arr[mask])),
            }
        )
    return records


def run_reproduction(
    benchmark_dir: Path = DEFAULT_BENCHMARK_DIR,
    official_python: Path = DEFAULT_OFFICIAL_PYTHON,
    *,
    max_epochs: int = 1000,
    learning_rate: float = 2e-3,
    n_init: int = 5,
    n_init_epochs: int = 3,
    delta_loss: float = 1e-3,
    random_seed: int = 1234,
) -> Dict[str, Any]:
    """Run full ASTIR reproduction on ``basel_22k_subset.h5ad`` and write artifacts."""
    import anndata

    from iris_phenotyper.astir.adapter import ASTIRPhenotyper
    from iris_phenotyper.astir.pipeline import assign_celltypes_from_probabilities
    from iris_phenotyper.evaluation import evaluate_predictions, format_benchmark_report

    data_dir = benchmark_dir / "data"
    results_dir = benchmark_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = data_dir / "basel_22k_subset.h5ad"
    marker_yaml = data_dir / "jackson-2020-markers-v4.yml"
    manuscript_dir = data_dir / "astir_manuscript" / "camlab-bioml-astir-manuscript-045a2a5"
    jackson_csv = manuscript_dir / "data-raw" / "annotations" / "basel" / "jackson_annotation_fixed.csv"
    cluster_map_yaml = manuscript_dir / "data-raw" / "annotations" / "cluster-mapping.yml"
    coarse_map_yaml = manuscript_dir / "data-raw" / "annotations" / "coarse-fine-mapping.yml"

    ad = anndata.read_h5ad(h5ad_path)
    n_cells = int(ad.n_obs)
    batch_size = max(1, int(n_cells / 100))

    with marker_yaml.open("r", encoding="utf-8") as fh:
        marker_cfg = yaml.safe_load(fh)

    # 1. Official camlab-bioml/astir reference run
    official_probs, official_meta = _ensure_official_predictions(
        h5ad_path,
        marker_yaml,
        results_dir,
        official_python,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        n_init=n_init,
        n_init_epochs=n_init_epochs,
        delta_loss=delta_loss,
        random_seed=random_seed,
    )

    # 2. IRIS ASTIRPhenotyper run (PyTorch 2.x / NumPy 2.x native VI)
    model = ASTIRPhenotyper(
        marker_config=marker_yaml,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        delta_loss=delta_loss,
        n_init=n_init,
        n_init_epochs=n_init_epochs,
        threshold=0.7,
        apply_arcsinh=False,  # basel_22k_subset.h5ad is already arcsinh(x / 5)
        batch_col="batch",
        create_design_mat=True,
        fit_states=True,
        run_diagnostics=True,
        random_seed=random_seed,
        engine="iris",
    )
    res_t07 = model.predict(ad, threshold=0.7)
    res_t05 = model.predict(ad, threshold=0.5)

    assert res_t07.probabilities is not None
    assert model.pipeline is not None

    iris_probs = res_t07.probabilities.copy()
    cell_ids = list(ad.obs["cell_id"])
    iris_probs_by_id = iris_probs.copy()
    iris_probs_by_id.index = cell_ids

    # Save prediction & diagnostic tables
    pred_out = pd.DataFrame(
        {
            "ObjectNumber": res_t07.labels.index.to_numpy(dtype=int),
            "cell_id": cell_ids,
            "batch": ad.obs["batch"].astype(str).to_numpy(),
            "cell_type_t07": res_t07.labels.to_numpy(dtype=str),
            "cell_type_t05": res_t05.labels.to_numpy(dtype=str),
            "confidence": res_t07.confidence.to_numpy(dtype=float),
        }
    )
    for col in iris_probs.columns:
        pred_out[f"prob_{col}"] = iris_probs[col].to_numpy(dtype=float)
    pred_out.to_csv(results_dir / "basel_22k_iris_predictions.csv", index=False)

    diag_df = model.pipeline.diagnostics_celltype(threshold=0.5, alpha=0.01)
    diag_df.to_csv(results_dir / "basel_22k_diagnostics_celltype.csv", index=False)

    state_df = model.pipeline.get_cellstates()
    state_df.to_csv(results_dir / "basel_22k_cellstate_scores.csv")

    # 3. Canonical marker enrichment & calibration
    x_arr = ad.X.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X)
    expr_full = pd.DataFrame(x_arr, index=cell_ids, columns=list(ad.var_names))
    enrichment_df = _compute_marker_enrichment(
        expr_full,
        pd.Series(res_t07.labels.to_numpy(dtype=str), index=cell_ids),
        marker_cfg["cell_types"],
    )
    enrichment_df.to_csv(results_dir / "basel_22k_marker_enrichment.csv", index=False)

    # 4. Compare IRIS vs Official camlab-bioml/astir & Astir-Vignette
    iris_counts_t07 = {str(k): int(v) for k, v in res_t07.labels.value_counts().items()}
    iris_counts_t05 = {str(k): int(v) for k, v in res_t05.labels.value_counts().items()}

    concordance: Dict[str, Any] = {}
    if official_probs is not None:
        official_aligned = official_probs.reindex(index=cell_ids, columns=iris_probs_by_id.columns)
        off_labels_t07 = assign_celltypes_from_probabilities(official_aligned, threshold=0.7)[
            "cell_type"
        ]
        off_labels_t05 = assign_celltypes_from_probabilities(official_aligned, threshold=0.5)[
            "cell_type"
        ]
        off_pred_out = pd.DataFrame(
            {
                "cell_id": cell_ids,
                "cell_type_t07": off_labels_t07.to_numpy(dtype=str),
                "cell_type_t05": off_labels_t05.to_numpy(dtype=str),
                "confidence": official_aligned.to_numpy(dtype=float).max(axis=1),
            }
        )
        for col in official_aligned.columns:
            off_pred_out[f"prob_{col}"] = official_aligned[col].to_numpy(dtype=float)
        off_pred_out.to_csv(results_dir / "basel_22k_official_predictions.csv", index=False)

        eval_off_vs_iris_t07 = evaluate_predictions(
            off_labels_t07.to_numpy(dtype=str),
            res_t07.labels.to_numpy(dtype=str),
            y_prob=iris_probs_by_id,
        )
        eval_off_vs_iris_t05 = evaluate_predictions(
            off_labels_t05.to_numpy(dtype=str),
            res_t05.labels.to_numpy(dtype=str),
            y_prob=iris_probs_by_id,
        )

        p_iris = iris_probs_by_id.to_numpy(dtype=float)
        p_off = official_aligned.to_numpy(dtype=float)
        pearson_overall = float(np.corrcoef(p_iris.ravel(), p_off.ravel())[0, 1])
        spearman_overall = float(stats.spearmanr(p_iris.ravel(), p_off.ravel()).statistic)
        mae_prob = float(np.mean(np.abs(p_iris - p_off)))

        per_col_corr = {}
        for col in iris_probs_by_id.columns:
            per_col_corr[str(col)] = float(
                np.corrcoef(iris_probs_by_id[col].to_numpy(), official_aligned[col].to_numpy())[
                    0, 1
                ]
            )

        concordance = {
            "threshold_0_7": {
                "accuracy": eval_off_vs_iris_t07.accuracy,
                "balanced_accuracy": eval_off_vs_iris_t07.balanced_accuracy,
                "macro_f1": eval_off_vs_iris_t07.macro_f1,
                "weighted_f1": eval_off_vs_iris_t07.weighted_f1,
                "cohens_kappa": eval_off_vs_iris_t07.cohens_kappa,
            },
            "threshold_0_5": {
                "accuracy": eval_off_vs_iris_t05.accuracy,
                "balanced_accuracy": eval_off_vs_iris_t05.balanced_accuracy,
                "macro_f1": eval_off_vs_iris_t05.macro_f1,
                "weighted_f1": eval_off_vs_iris_t05.weighted_f1,
                "cohens_kappa": eval_off_vs_iris_t05.cohens_kappa,
            },
            "posterior_pearson_r": pearson_overall,
            "posterior_spearman_rho": spearman_overall,
            "posterior_mae": mae_prob,
            "per_class_posterior_pearson_r": per_col_corr,
        }

    # Proportion agreement against Astir-Vignette published counts
    all_cats = sorted(set(VIGNETTE_PUBLISHED_COUNTS_T07) | set(iris_counts_t07))
    vig_vec = np.array([VIGNETTE_PUBLISHED_COUNTS_T07.get(c, 0) for c in all_cats], dtype=float)
    iris_vec = np.array([iris_counts_t07.get(c, 0) for c in all_cats], dtype=float)
    proportion_corr = float(np.corrcoef(vig_vec, iris_vec)[0, 1])

    # 5. Ground-truth evaluation against Jackson et al. (2020)
    gt_results: Dict[str, Any] = {}
    calibration_bins: List[Dict[str, Any]] = []
    report_sections: List[str] = []

    if jackson_csv.exists() and cluster_map_yaml.exists() and coarse_map_yaml.exists():
        ann_df = pd.read_csv(jackson_csv).set_index("cell_id").loc[cell_ids]
        with cluster_map_yaml.open("r", encoding="utf-8") as fh:
            cluster_map = yaml.safe_load(fh)
        with coarse_map_yaml.open("r", encoding="utf-8") as fh:
            coarse_map = yaml.safe_load(fh)

        rev_cluster = {ct: coarse for coarse, cts in cluster_map.items() for ct in cts}
        rev_coarse = {fine: coarse for coarse, fines in coarse_map.items() for fine in fines}
        rev_coarse["Other"] = "Unclear"
        rev_coarse["Unknown"] = "Unclear"

        gt_coarse = ann_df["cell_type"].map(rev_cluster).fillna("Unclear")
        gt_class = ann_df["class"].astype(str)

        iris_coarse_t05 = pd.Series(res_t05.labels.to_numpy(dtype=str)).map(rev_coarse).fillna("Unclear")
        iris_lineage_t05 = pd.Series(res_t05.labels.to_numpy(dtype=str)).map(CLASS_LINEAGE_MAP).fillna("Unclear")

        # 5a. Non-Unclear structural cell types (Epithelial/Stromal/Endothelial/Macrophage/Lymphocyte)
        valid_mask = gt_coarse.to_numpy(dtype=object) != "Unclear"
        assigned_mask_iris = valid_mask & (iris_coarse_t05.to_numpy(dtype=object) != "Unclear")

        eval_coarse_all = evaluate_predictions(
            gt_coarse.to_numpy(dtype=str)[valid_mask],
            iris_coarse_t05.to_numpy(dtype=str)[valid_mask],
        )
        eval_coarse_assigned = evaluate_predictions(
            gt_coarse.to_numpy(dtype=str)[assigned_mask_iris],
            iris_coarse_t05.to_numpy(dtype=str)[assigned_mask_iris],
        )

        # 5b. 4-class major lineage across all 22,717 cells (Tumor/Stroma/Immune/Vessel)
        assigned_lineage_mask = iris_lineage_t05.to_numpy(dtype=object) != "Unclear"
        eval_lineage_all = evaluate_predictions(
            gt_class.to_numpy(dtype=str),
            iris_lineage_t05.to_numpy(dtype=str),
        )
        eval_lineage_assigned = evaluate_predictions(
            gt_class.to_numpy(dtype=str)[assigned_lineage_mask],
            iris_lineage_t05.to_numpy(dtype=str)[assigned_lineage_mask],
        )

        calibration_bins = _compute_calibration_bins(
            res_t05.confidence,
            gt_class,
            iris_lineage_t05,
        )

        gt_results = {
            "coarse_structural_non_unclear_n": int(valid_mask.sum()),
            "coarse_assigned_n": int(assigned_mask_iris.sum()),
            "coarse_all_non_unclear": eval_coarse_all.to_json_dict(),
            "coarse_assigned_only": eval_coarse_assigned.to_json_dict(),
            "major_lineage_all_22k": eval_lineage_all.to_json_dict(),
            "major_lineage_assigned_only": eval_lineage_assigned.to_json_dict(),
        }

        if official_probs is not None:
            official_aligned = official_probs.reindex(index=cell_ids, columns=iris_probs_by_id.columns)
            off_labels_t05 = assign_celltypes_from_probabilities(official_aligned, threshold=0.5)[
                "cell_type"
            ]
            off_coarse_t05 = off_labels_t05.map(rev_coarse).fillna("Unclear")
            off_lineage_t05 = off_labels_t05.map(CLASS_LINEAGE_MAP).fillna("Unclear")

            off_assigned_coarse = valid_mask & (off_coarse_t05.to_numpy(dtype=object) != "Unclear")
            off_assigned_lin = off_lineage_t05.to_numpy(dtype=object) != "Unclear"

            eval_off_coarse_all = evaluate_predictions(
                gt_coarse.to_numpy(dtype=str)[valid_mask],
                off_coarse_t05.to_numpy(dtype=str)[valid_mask],
            )
            eval_off_coarse_assigned = evaluate_predictions(
                gt_coarse.to_numpy(dtype=str)[off_assigned_coarse],
                off_coarse_t05.to_numpy(dtype=str)[off_assigned_coarse],
            )
            eval_off_lin_assigned = evaluate_predictions(
                gt_class.to_numpy(dtype=str)[off_assigned_lin],
                off_lineage_t05.to_numpy(dtype=str)[off_assigned_lin],
            )
            gt_results["official_coarse_all_non_unclear"] = eval_off_coarse_all.to_json_dict()
            gt_results["official_coarse_assigned_only"] = eval_off_coarse_assigned.to_json_dict()
            gt_results["official_major_lineage_assigned_only"] = eval_off_lin_assigned.to_json_dict()

            report_sections.append(
                format_benchmark_report(
                    {
                        "Official astir (coarse assigned)": eval_off_coarse_assigned,
                        "IRIS ASTIR (coarse assigned)": eval_coarse_assigned,
                        "Official astir (all non-Unclear)": eval_off_coarse_all,
                        "IRIS ASTIR (all non-Unclear)": eval_coarse_all,
                    },
                    dataset_name="Jackson et al. Basel 22k Subset (Coarse Structural Types)",
                    metadata={
                        "Cells (Total)": n_cells,
                        "Non-Unclear Ground Truth Cells": int(valid_mask.sum()),
                        "Threshold": 0.5,
                    },
                )
            )
        else:
            report_sections.append(
                format_benchmark_report(
                    eval_coarse_assigned,
                    method_name="IRIS ASTIR",
                    dataset_name="Jackson et al. Basel 22k Subset (Coarse Assigned)",
                )
            )

    summary_payload: Dict[str, Any] = {
        "dataset": "basel_22k_subset.h5ad",
        "n_cells": n_cells,
        "n_markers_total": int(ad.n_vars),
        "hyperparameters": {
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_init": n_init,
            "n_init_epochs": n_init_epochs,
            "delta_loss": delta_loss,
            "random_seed": random_seed,
        },
        "iris_run": {
            "engine_used": res_t07.metadata["engine_used"],
            "n_epochs_final": res_t07.metadata["n_epochs_type"],
            "final_elbo_loss": res_t07.metadata["final_elbo_loss"],
            "elapsed_s": res_t07.metadata["elapsed_s"],
            "counts_threshold_0_7": iris_counts_t07,
            "counts_threshold_0_5": iris_counts_t05,
            "diagnostics_issues_t05": int(len(diag_df)),
            "cell_states_inferred": list(state_df.columns),
        },
        "official_run": official_meta,
        "vignette_published_counts_t07": VIGNETTE_PUBLISHED_COUNTS_T07,
        "vignette_proportion_pearson_r": proportion_corr,
        "official_vs_iris_concordance": concordance,
        "jackson_ground_truth_eval": gt_results,
        "calibration_bins": calibration_bins,
        "marker_enrichment_summary": {
            "n_pairs": int(len(enrichment_df)),
            "all_positive_delta": bool((enrichment_df["delta_expr"] > 0).all())
            if not enrichment_df.empty
            else False,
            "min_fold_enrichment": float(enrichment_df["fold_enrichment"].min())
            if not enrichment_df.empty
            else 0.0,
            "median_fold_enrichment": float(enrichment_df["fold_enrichment"].median())
            if not enrichment_df.empty
            else 0.0,
        },
    }

    (results_dir / "reproduction_metrics.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    # Build comprehensive markdown report
    md_lines: List[str] = [
        "# ASTIR Reproduction Report (`basel_22k_subset.h5ad`)",
        "",
        "## Run Summary",
        "",
        f"- **Dataset**: `basel_22k_subset.h5ad` ({n_cells:,} cells, {ad.n_vars} markers, 4 TMA batches)",
        "- **Marker Schema**: `jackson-2020-markers-v4.yml` (8 cell types + 13 cell states)",
        f"- **IRIS Engine**: `{res_t07.metadata['engine_used']}` ({res_t07.metadata['n_epochs_type']} final epochs, elapsed {res_t07.metadata['elapsed_s']:.2f} s, ELBO loss {res_t07.metadata['final_elbo_loss']:.1f})",
    ]
    if official_meta:
        md_lines.append(
            f"- **Official `camlab-bioml/astir` Run**: {official_meta['n_epochs_final']} final epochs, elapsed {official_meta['elapsed_s']:.2f} s, ELBO loss {official_meta['final_loss']:.1f}"
        )
    md_lines.extend(
        [
            "",
            "## Cell-Type Counts (Threshold = 0.7): Vignette vs Official `astir` vs IRIS ASTIR",
            "",
            "| Cell Type | Vignette Published | Official `astir` | IRIS ASTIR | IRIS % |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    off_counts = (official_meta or {}).get("counts_threshold_0_7", {})
    for ct in sorted(all_cats, key=lambda x: iris_counts_t07.get(x, 0), reverse=True):
        vc = VIGNETTE_PUBLISHED_COUNTS_T07.get(ct, 0)
        oc = off_counts.get(ct, 0)
        ic = iris_counts_t07.get(ct, 0)
        pct = 100.0 * ic / n_cells
        md_lines.append(f"| {ct} | {vc:,} | {oc:,} | {ic:,} | {pct:.2f}% |")

    md_lines.extend(
        [
            "",
            f"- **Cell-Type Count Pearson Correlation (Vignette vs IRIS)**: `{proportion_corr:.6f}`",
        ]
    )
    if concordance:
        c07 = concordance["threshold_0_7"]
        c05 = concordance["threshold_0_5"]
        md_lines.extend(
            [
                f"- **Official vs IRIS Label Agreement (`threshold=0.7`)**: Accuracy = `{c07['accuracy']:.4f}`, Macro-F1 = `{c07['macro_f1']:.4f}`, Weighted-F1 = `{c07['weighted_f1']:.4f}`, Cohen's kappa = `{c07['cohens_kappa']:.4f}`",
                f"- **Official vs IRIS Label Agreement (`threshold=0.5`)**: Accuracy = `{c05['accuracy']:.4f}`, Macro-F1 = `{c05['macro_f1']:.4f}`, Weighted-F1 = `{c05['weighted_f1']:.4f}`, Cohen's kappa = `{c05['cohens_kappa']:.4f}`",
                f"- **Posterior Probability Matrix Correlation (Official vs IRIS)**: Pearson r = `{concordance['posterior_pearson_r']:.6f}`, Spearman rho = `{concordance['posterior_spearman_rho']:.6f}`, MAE = `{concordance['posterior_mae']:.6f}`",
            ]
        )

    if calibration_bins:
        md_lines.extend(
            [
                "",
                "## Posterior Calibration (vs Jackson et al. Major Lineage)",
                "",
                "| Confidence Bin | Cells | Fraction | Mean Confidence | Lineage Accuracy |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in calibration_bins:
            mc = "-" if row["mean_confidence"] is None else f"{row['mean_confidence']:.4f}"
            la = "-" if row["lineage_accuracy"] is None else f"{row['lineage_accuracy']:.4f}"
            md_lines.append(
                f"| {row['bin']} | {row['n_cells']:,} | {100.0 * row['fraction']:.2f}% | {mc} | {la} |"
            )

    if not enrichment_df.empty:
        md_lines.extend(
            [
                "",
                "## Canonical Marker Enrichment (Assigned Cell Type vs Other Cells)",
                "",
                "| Cell Type | Marker | N Cells | Mean In-Type | Mean Out-Type | Fold Enrichment | p-value |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in enrichment_df.iterrows():
            md_lines.append(
                f"| {row['cell_type']} | {row['marker']} | {int(row['n_cells']):,} | "
                f"{row['mean_in_type']:.3f} | {row['mean_other_cells']:.3f} | "
                f"{row['fold_enrichment']:.2f}x | {row['p_value']:.2e} |"
            )

    if report_sections:
        md_lines.extend(["", *report_sections])

    (results_dir / "reproduction_report.md").write_text(
        "\n".join(md_lines).rstrip() + "\n",
        encoding="utf-8",
    )

    manifest_path = benchmark_dir / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["reproduction_summary"] = {
            "dataset": summary_payload["dataset"],
            "n_cells": summary_payload["n_cells"],
            "vignette_proportion_pearson_r": summary_payload["vignette_proportion_pearson_r"],
            "official_vs_iris_concordance": summary_payload["official_vs_iris_concordance"],
            "jackson_ground_truth_eval": {
                k: v
                for k, v in summary_payload["jackson_ground_truth_eval"].items()
                if k.startswith("coarse_") or k.startswith("official_coarse_")
            },
            "marker_enrichment_summary": summary_payload["marker_enrichment_summary"],
            "results_files": sorted(p.name for p in results_dir.iterdir() if p.is_file()),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return summary_payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce ASTIR paper/vignette benchmark.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--official-python", type=Path, default=DEFAULT_OFFICIAL_PYTHON)
    parser.add_argument("--official-only", action="store_true")
    parser.add_argument("--h5ad", type=Path, default=None)
    parser.add_argument("--marker-yaml", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--n-init-epochs", type=int, default=3)
    parser.add_argument("--delta-loss", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=1234)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    if args.official_only:
        h5ad = args.h5ad or (args.benchmark_dir / "data" / "basel_22k_subset.h5ad")
        yaml_p = args.marker_yaml or (args.benchmark_dir / "data" / "jackson-2020-markers-v4.yml")
        res_dir = args.results_dir or (args.benchmark_dir / "results")
        res_dir.mkdir(parents=True, exist_ok=True)
        meta = _run_official_astir_inprocess(
            h5ad,
            yaml_p,
            res_dir / "basel_22k_official_probabilities.csv",
            res_dir / "basel_22k_official_meta.json",
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            n_init=args.n_init,
            n_init_epochs=args.n_init_epochs,
            delta_loss=args.delta_loss,
            random_seed=args.random_seed,
        )
        print(json.dumps(meta, indent=2))
        return

    summary = run_reproduction(
        benchmark_dir=args.benchmark_dir,
        official_python=args.official_python,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        n_init=args.n_init,
        n_init_epochs=args.n_init_epochs,
        delta_loss=args.delta_loss,
        random_seed=args.random_seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
