"""
IRIS Compositional Cell Proportion & Ecological Diversity Engine
================================================================
Headless-safe compositional analysis supporting total-cell, parent-lineage,
and polygonal ROI-compartment denominators, centered log-ratio (CLR)
transformation, Shannon/Simpson/Pielou diversity metrics, and sample-level
cohort hypothesis testing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from iris_analyzer.density import (
    _STATS_COLUMNS,
    ROISpec,
    _compute_group_stats_table,
    _resolve_sample_and_type_columns,
    compute_polygon_area_and_filter,
)
from iris_common.engine.statistics import adjust_p_values, format_significance, run_group_comparison
from iris_common.schema import COL_SAMPLE

_DENOMINATOR_ALIASES: Dict[str, str] = {
    "total": "total",
    "all": "total",
    "all cells": "total",
    "all_cells": "total",
    "parent_lineage": "parent_lineage",
    "parent": "parent_lineage",
    "parent lineage": "parent_lineage",
    "parent lineage subset": "parent_lineage",
    "roi_compartment": "roi_compartment",
    "roi": "roi_compartment",
    "roi compartments": "roi_compartment",
    "roi_compartments": "roi_compartment",
}


@dataclass
class ProportionResult:
    """Container for cell-type proportions, CLR transform, diversity, and cohort stats."""

    summary_df: pd.DataFrame
    replicates_df: pd.DataFrame
    diversity_df: pd.DataFrame
    clr_df: pd.DataFrame
    stats_df: pd.DataFrame
    cell_types: List[str]
    denominator_mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_sample_diversity(
    replicates_df: pd.DataFrame,
    sample_col: str = COL_SAMPLE,
    cell_type_col: str = "cell_type",
    count_col: str = "count",
) -> pd.DataFrame:
    """Compute ecological diversity indices per sample from replicate counts.

    Metrics computed per sample:
    - ``richness``: count of cell types with ``count > 0``
    - ``total_count``: sum of cells across cell types
    - ``shannon_H``: Shannon entropy :math:`-\\sum p_i \\ln(p_i)`
    - ``pielou_J``: Pielou evenness :math:`H / \\ln(S)` (0.0 when ``richness <= 1``)
    - ``simpson_1_minus_D``: Gini-Simpson index :math:`1 - \\sum p_i^2`
    - ``inverse_simpson``: inverse Simpson index :math:`1 / \\sum p_i^2`
    """
    base_cols = [
        sample_col,
        "richness",
        "total_count",
        "shannon_H",
        "pielou_J",
        "simpson_1_minus_D",
        "inverse_simpson",
    ]
    if replicates_df is None or replicates_df.empty:
        return pd.DataFrame(columns=base_cols)

    resolved_sample_col = sample_col
    if resolved_sample_col not in replicates_df.columns:
        if "Sample" in replicates_df.columns:
            resolved_sample_col = "Sample"
        else:
            raise KeyError(f"Sample column {sample_col!r} not found in replicates_df.")

    resolved_count_col = count_col
    if resolved_count_col not in replicates_df.columns:
        if "Count" in replicates_df.columns:
            resolved_count_col = "Count"
        else:
            raise KeyError(f"Count column {count_col!r} not found in replicates_df.")

    if cell_type_col not in replicates_df.columns:
        raise KeyError(f"Cell type column {cell_type_col!r} not found in replicates_df.")

    has_group = "Group" in replicates_df.columns
    samples = list(dict.fromkeys(replicates_df[resolved_sample_col].astype(str).tolist()))

    rows: List[Dict[str, Any]] = []
    for sample_name in samples:
        sub = replicates_df[replicates_df[resolved_sample_col].astype(str) == sample_name]
        type_counts = (
            sub.groupby(cell_type_col, observed=True)[resolved_count_col]
            .sum()
            .to_numpy(dtype=float)
        )
        pos_counts = type_counts[type_counts > 0.0]
        total = float(np.sum(pos_counts))
        richness = int(len(pos_counts))

        if total <= 0.0 or richness == 0:
            shannon_h = 0.0
            pielou_j = 0.0
            simpson = 0.0
            inv_simpson = 0.0
        else:
            probs = pos_counts / total
            shannon_h = float(-np.sum(probs * np.log(probs)))
            pielou_j = float(shannon_h / np.log(richness)) if richness > 1 else 0.0
            dominance = float(np.sum(probs ** 2))
            simpson = float(1.0 - dominance)
            inv_simpson = float(1.0 / dominance) if dominance > 0.0 else 0.0

        row: Dict[str, Any] = {sample_col: sample_name}
        if has_group:
            row["Group"] = str(sub["Group"].iloc[0])
        row.update(
            {
                "richness": richness,
                "total_count": int(round(total)),
                "shannon_H": shannon_h,
                "pielou_J": pielou_j,
                "simpson_1_minus_D": simpson,
                "inverse_simpson": inv_simpson,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _compute_diversity_stats(
    diversity_df: pd.DataFrame,
    group_map: Optional[Dict[str, str]],
    stat_test: str,
    p_adjust: str,
) -> pd.DataFrame:
    """Compare diversity indices across sample groups when >= 2 groups are present."""
    if diversity_df.empty or not group_map or "Group" not in diversity_df.columns:
        return pd.DataFrame()

    mapped_groups = {str(v) for v in group_map.values() if v is not None and str(v).strip()}
    if len(mapped_groups) < 2:
        return pd.DataFrame()

    sub = diversity_df[diversity_df["Group"].isin(mapped_groups)]
    if sub["Group"].nunique() < 2:
        return pd.DataFrame()

    metrics = ("shannon_H", "pielou_J", "simpson_1_minus_D", "inverse_simpson")
    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        res_list = run_group_comparison(
            sub,
            value_col=metric,
            group_col="Group",
            sample_col=COL_SAMPLE,
            aggregate_by_sample=True,
            test=stat_test,
            correction=p_adjust,
        )
        for item in res_list:
            row = item.to_dict()
            row["metric"] = metric
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    if len(rows) > 1 and p_adjust.lower().strip() != "tukey":
        raw_ps = [r["p_value"] for r in rows]
        adj_qs = adjust_p_values(raw_ps, method=p_adjust)
        for row, q_val in zip(rows, adj_qs):
            q_clean = float(q_val) if np.isfinite(q_val) else 1.0
            row["q_value"] = q_clean
            row["significant"] = bool(q_clean < 0.05)
            row["significance_label"] = format_significance(q_clean, mode="stars")

    return pd.DataFrame(rows)


def compute_cell_proportions(
    df: pd.DataFrame,
    cell_type_col: str = "cell_type",
    sample_col: Optional[str] = None,
    denominator_mode: str = "total",
    parent_subset: Optional[Sequence[str]] = None,
    rois: Optional[Sequence[ROISpec]] = None,
    scale_factor: float = 1.0,
    group_map: Optional[Dict[str, str]] = None,
    stat_test: str = "mann_whitney",
    p_adjust: str = "fdr_bh",
    pseudocount: float = 1e-4,
) -> ProportionResult:
    """Compute per-sample or per-ROI cell-type proportions, CLR transform, diversity, and stats.

    Parameters
    ----------
    denominator_mode : str
        - ``'total'``: denominator is all cells in each sample.
        - ``'parent_lineage'``: restrict denominator and numerator to ``parent_subset``.
        - ``'roi_compartment'``: compute compartment proportions inside each ``ROISpec``.
    """
    mode_key = str(denominator_mode).strip().lower()
    if mode_key not in _DENOMINATOR_ALIASES:
        raise ValueError(
            f"Unsupported denominator_mode {denominator_mode!r}. "
            "Expected 'total', 'parent_lineage', or 'roi_compartment'."
        )
    mode = _DENOMINATOR_ALIASES[mode_key]

    norm_group_map: Optional[Dict[str, str]] = (
        {str(k): str(v) for k, v in group_map.items()} if group_map is not None else None
    )

    if df is None or df.empty:
        return ProportionResult(
            summary_df=pd.DataFrame(),
            replicates_df=pd.DataFrame(),
            diversity_df=compute_sample_diversity(pd.DataFrame()),
            clr_df=pd.DataFrame(),
            stats_df=pd.DataFrame(columns=_STATS_COLUMNS),
            cell_types=[],
            denominator_mode=mode,
            metadata={"n_samples": 0, "n_units": 0, "n_cells": 0},
        )

    work, s_col, ct_col = _resolve_sample_and_type_columns(df, cell_type_col, sample_col)
    all_samples = list(dict.fromkeys(work[s_col].tolist()))

    subset_list: List[str] = []
    if parent_subset is not None:
        subset_list = list(dict.fromkeys(str(x) for x in parent_subset if str(x).strip()))

    if mode == "parent_lineage":
        if not subset_list:
            raise ValueError(
                "parent_subset must be a non-empty list when denominator_mode='parent_lineage'."
            )
        work_eval = work[work[ct_col].isin(set(subset_list))].copy()
        present_types = sorted(work_eval[ct_col].unique().tolist())
        cell_types = present_types if present_types else sorted(subset_list)
    else:
        if subset_list:
            work_eval = work[work[ct_col].isin(set(subset_list))].copy()
        else:
            work_eval = work
        cell_types = sorted(work_eval[ct_col].unique().tolist())

    if mode == "roi_compartment" and not rois:
        raise ValueError("rois must be provided when denominator_mode='roi_compartment'.")

    def _resolve_group(sample_name: str) -> str:
        if norm_group_map is None:
            return "All"
        return norm_group_map.get(sample_name, "Unassigned")

    units: List[Tuple[str, str, pd.DataFrame]] = []
    if mode == "roi_compartment":
        scale = float(scale_factor)
        for roi in rois or ():
            roi_scale = roi.microns_per_pixel if roi.microns_per_pixel != 1.0 else scale
            target_sample = roi.sample
            if target_sample is None and len(all_samples) == 1:
                target_sample = all_samples[0]
            sub_roi, _, _ = compute_polygon_area_and_filter(
                work_eval,
                roi.inclusion_polygons,
                roi.exclusion_polygons,
                scale_factor=roi_scale,
                sample_col=s_col,
                sample_name=target_sample,
            )
            if target_sample is not None:
                sample_label = str(target_sample)
            elif not sub_roi.empty and sub_roi[s_col].nunique() == 1:
                sample_label = str(sub_roi[s_col].iloc[0])
            else:
                sample_label = roi.name
            units.append((sample_label, roi.name, sub_roi))
    else:
        for sample_name in all_samples:
            sub = work_eval[work_eval[s_col] == sample_name]
            units.append((sample_name, "Whole Sample", sub))

    eps = float(pseudocount) if pseudocount > 0.0 else 1e-4
    rep_rows: List[Dict[str, Any]] = []
    clr_rows: List[Dict[str, Any]] = []

    for sample_name, roi_name, sub_df in units:
        grp = _resolve_group(sample_name)
        denom = int(len(sub_df))
        counts_map = sub_df[ct_col].value_counts().to_dict() if denom > 0 else {}
        counts_vec = np.array([int(counts_map.get(ct, 0)) for ct in cell_types], dtype=float)
        props_vec = (
            counts_vec / float(denom)
            if denom > 0
            else np.zeros(len(cell_types), dtype=float)
        )
        pcts_vec = props_vec * 100.0

        if len(cell_types) > 0:
            log_p = np.log(props_vec + eps)
            clr_vec = log_p - float(np.mean(log_p))
        else:
            clr_vec = np.empty(0, dtype=float)

        clr_entry: Dict[str, Any] = {
            COL_SAMPLE: sample_name,
            "Sample": sample_name,
            "ROI": roi_name,
            "Group": grp,
        }
        for idx, ct in enumerate(cell_types):
            clr_entry[ct] = float(clr_vec[idx])
            rep_rows.append(
                {
                    COL_SAMPLE: sample_name,
                    "Sample": sample_name,
                    "ROI": roi_name,
                    "Group": grp,
                    "cell_type": ct,
                    "count": int(counts_vec[idx]),
                    "denominator_count": denom,
                    "proportion": float(props_vec[idx]),
                    "percentage": float(pcts_vec[idx]),
                    "clr": float(clr_vec[idx]),
                }
            )
        clr_rows.append(clr_entry)

    replicates_df = pd.DataFrame(rep_rows)
    clr_df = pd.DataFrame(clr_rows)
    diversity_df = compute_sample_diversity(
        replicates_df,
        sample_col=COL_SAMPLE,
        cell_type_col="cell_type",
        count_col="count",
    )

    summary_rows: List[Dict[str, Any]] = []
    if not replicates_df.empty:
        groups_order = list(dict.fromkeys(replicates_df["Group"].tolist()))
        for grp in groups_order:
            for ct in cell_types:
                sub_rep = replicates_df[
                    (replicates_df["Group"] == grp) & (replicates_df["cell_type"] == ct)
                ]
                if sub_rep.empty:
                    continue
                sample_agg = sub_rep.groupby(COL_SAMPLE, observed=True)[
                    ["proportion", "percentage", "clr"]
                ].mean()
                p_vals = sample_agg["proportion"].to_numpy(dtype=float)
                pct_vals = sample_agg["percentage"].to_numpy(dtype=float)
                clr_vals = sample_agg["clr"].to_numpy(dtype=float)
                n_samples = int(len(p_vals))
                sd_p = float(np.std(p_vals, ddof=1)) if n_samples > 1 else 0.0
                sem_p = sd_p / float(np.sqrt(n_samples)) if n_samples > 0 else 0.0
                sd_pct = float(np.std(pct_vals, ddof=1)) if n_samples > 1 else 0.0
                sem_pct = sd_pct / float(np.sqrt(n_samples)) if n_samples > 0 else 0.0
                summary_rows.append(
                    {
                        "Group": grp,
                        "cell_type": ct,
                        "N_Samples": n_samples,
                        "N_Units": int(len(sub_rep)),
                        "Total_Count": int(sub_rep["count"].sum()),
                        "Mean_Proportion": float(np.mean(p_vals)),
                        "SD_Proportion": sd_p,
                        "SEM_Proportion": sem_p,
                        "Median_Proportion": float(np.median(p_vals)),
                        "Mean_Percentage": float(np.mean(pct_vals)),
                        "SD_Percentage": sd_pct,
                        "SEM_Percentage": sem_pct,
                        "Median_Percentage": float(np.median(pct_vals)),
                        "Mean_CLR": float(np.mean(clr_vals)),
                    }
                )
    summary_df = pd.DataFrame(summary_rows)

    stats_df = _compute_group_stats_table(
        replicates_df=replicates_df,
        value_col="proportion",
        categories=cell_types,
        group_map=norm_group_map,
        stat_test=stat_test,
        p_adjust=p_adjust,
        include_all_cells=False,
    )
    diversity_stats_df = _compute_diversity_stats(
        diversity_df=diversity_df,
        group_map=norm_group_map,
        stat_test=stat_test,
        p_adjust=p_adjust,
    )

    return ProportionResult(
        summary_df=summary_df,
        replicates_df=replicates_df,
        diversity_df=diversity_df,
        clr_df=clr_df,
        stats_df=stats_df,
        cell_types=cell_types,
        denominator_mode=mode,
        metadata={
            "denominator_mode": mode,
            "parent_subset": subset_list,
            "stat_test": stat_test,
            "p_adjust": p_adjust,
            "pseudocount": eps,
            "n_samples": len(all_samples),
            "n_units": len(units),
            "n_cells": int(len(work_eval)),
            "cell_type_col": ct_col,
            "sample_col": s_col,
            "has_groups": bool(not stats_df.empty),
            "diversity_stats_df": diversity_stats_df,
        },
    )


__all__ = [
    "ProportionResult",
    "compute_sample_diversity",
    "compute_cell_proportions",
]
