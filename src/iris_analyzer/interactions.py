"""
IRIS Analyzer Pairwise Spatial Interaction Engine
=================================================
Headless-safe permutation-based spatial cell-cell interaction analysis
using vectorized sparse matrix multiplication (SpMM) stratified by sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from iris_analyzer.graph import SpatialGraphResult, build_spatial_graph
from iris_common.engine.statistics import adjust_p_values, run_group_comparison
from iris_common.schema import (
    CELL_TYPE_ALIASES,
    COL_CELL_TYPE,
    COL_SAMPLE,
    find_sample_column,
)

PAIRWISE_COLUMNS: Tuple[str, ...] = (
    "source_type",
    "target_type",
    "observed_count",
    "expected_count",
    "conditional_prob",
    "log2_fc",
    "z_score",
    "p_enrich",
    "p_avoid",
    "p_two_sided",
    "q_value",
    "interaction_class",
)

COHORT_STATS_COLUMNS: Tuple[str, ...] = (
    "source_type",
    "target_type",
    "pair",
    "group1",
    "group2",
    "test_name",
    "statistic",
    "p_value",
    "q_value",
    "n1",
    "n2",
    "effect_size",
    "effect_size_name",
    "mean1",
    "mean2",
    "median1",
    "median2",
    "significant",
    "significance_label",
)


@dataclass
class InteractionResult:
    """Results from stratified permutation-based pairwise spatial interaction analysis."""

    pairwise_df: pd.DataFrame
    z_matrix: pd.DataFrame
    log2fc_matrix: pd.DataFrame
    q_matrix: pd.DataFrame
    mean_count_matrix: pd.DataFrame
    replicates_df: pd.DataFrame
    cohort_stats_df: pd.DataFrame
    cell_types: List[str]
    graph_result: SpatialGraphResult
    metadata: Dict[str, Any]


def _resolve_cell_type_column(df: pd.DataFrame, cell_type_col: str) -> str:
    """Resolve the cell-type column name in *df*, checking canonical aliases."""
    if cell_type_col in df.columns:
        return cell_type_col
    lower_map = {str(c).strip().lower(): str(c) for c in df.columns}
    if cell_type_col.lower() in lower_map:
        return lower_map[cell_type_col.lower()]
    if cell_type_col == COL_CELL_TYPE:
        for alias in CELL_TYPE_ALIASES:
            if alias in df.columns and not pd.api.types.is_numeric_dtype(df[alias]):
                return alias
            if alias.lower() in lower_map:
                cand = lower_map[alias.lower()]
                if not pd.api.types.is_numeric_dtype(df[cand]):
                    return cand
    raise KeyError(f"Cell type column {cell_type_col!r} not found in DataFrame.")


def _extract_cell_types(series: pd.Series) -> Tuple[List[str], np.ndarray]:
    """Extract ordered unique cell types and integer codes (``-1`` for missing/empty)."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        cats = [str(c).strip() for c in series.cat.categories if str(c).strip()]
        str_vals = series.astype("string").str.strip()
        present = {str(v) for v in str_vals.dropna().unique() if str(v)}
        cell_types = [c for c in cats if c in present]
    else:
        str_vals = series.astype("string").str.strip()
        cell_types = sorted({str(v) for v in str_vals.dropna().unique() if str(v)})

    type_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    codes = np.full(len(series), -1, dtype=np.int64)
    na_mask = str_vals.isna().to_numpy()
    raw_strs = str_vals.fillna("").to_numpy()
    for idx in np.flatnonzero(~na_mask):
        code = type_to_idx.get(str(raw_strs[idx]), -1)
        codes[idx] = code
    return cell_types, codes


def _build_indicator_matrix(codes: np.ndarray, n_types: int) -> np.ndarray:
    """Construct ``(N, T)`` one-hot float64 indicator matrix ignoring ``-1`` codes."""
    n = codes.size
    mat = np.zeros((n, n_types), dtype=np.float64)
    valid = codes >= 0
    if np.any(valid):
        rows = np.flatnonzero(valid)
        mat[rows, codes[rows]] = 1.0
    return mat


def _check_block_diagonal(
    adj: sp.csr_matrix,
    sample_slices: Dict[str, np.ndarray],
    n_cells: int,
) -> bool:
    """Verify whether *adj* contains zero edges crossing ``sample_slices``."""
    if len(sample_slices) <= 1 or adj.nnz == 0:
        return True
    sample_ids = np.full(n_cells, -1, dtype=np.int64)
    for sid, idx in enumerate(sample_slices.values()):
        sample_ids[idx] = sid
    r, c = adj.nonzero()
    return bool(np.all((sample_ids[r] == sample_ids[c]) & (sample_ids[r] >= 0)))


def _classify_interaction(
    q_val: float,
    z_val: float,
    fdr_threshold: float,
    z_threshold: float,
) -> str:
    """Classify a pair into Attraction, Avoidance, or Non-significant."""
    sig_q = (q_val <= fdr_threshold) if fdr_threshold >= 1.0 else (q_val < fdr_threshold)
    if not (np.isfinite(q_val) and np.isfinite(z_val)):
        return "Non-significant"
    if sig_q and abs(z_val) >= z_threshold:
        if z_val > 0.0:
            return "Attraction"
        if z_val < 0.0:
            return "Avoidance"
    return "Non-significant"


def _run_sample_permutations(
    adj_s: sp.csr_matrix,
    labels_s: np.ndarray,
    n_types: int,
    n_permutations: int,
    rng: np.random.Generator,
    store_all: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run vectorized batched SpMM permutations for one sample subgraph.

    Returns ``(obs_s, sum_s, sum_sq_s, ge_s, le_s, perms_s_or_none)``.
    """
    n_s = labels_s.size
    y_s = _build_indicator_matrix(labels_s, n_types)
    obs_s = y_s.T @ (adj_s @ y_s)

    valid_labels = labels_s[labels_s >= 0]
    distinct_valid = np.unique(valid_labels).size
    if adj_s.nnz == 0 or n_s < 2 or distinct_valid < 2:
        sum_s = obs_s * float(n_permutations)
        sum_sq_s = np.square(obs_s) * float(n_permutations)
        ge_s = np.full((n_types, n_types), n_permutations, dtype=np.float64)
        le_s = np.full((n_types, n_types), n_permutations, dtype=np.float64)
        perms_s = (
            np.broadcast_to(obs_s, (n_permutations, n_types, n_types)).copy()
            if store_all
            else None
        )
        return obs_s, sum_s, sum_sq_s, ge_s, le_s, perms_s

    sum_s = np.zeros((n_types, n_types), dtype=np.float64)
    sum_sq_s = np.zeros((n_types, n_types), dtype=np.float64)
    ge_s = np.zeros((n_types, n_types), dtype=np.float64)
    le_s = np.zeros((n_types, n_types), dtype=np.float64)
    perms_s = (
        np.empty((n_permutations, n_types, n_types), dtype=np.float64)
        if store_all
        else None
    )

    max_batch = max(1, min(50, n_permutations, 8_000_000 // max(1, n_s * n_types)))
    has_missing = bool(np.any(labels_s < 0))
    n_idx_base = np.arange(n_s, dtype=np.int64)[:, None]

    for start in range(0, n_permutations, max_batch):
        b_len = min(max_batch, n_permutations - start)
        p_labels = rng.permuted(np.broadcast_to(labels_s, (b_len, n_s)), axis=1).T
        y_nbt = np.zeros((n_s, b_len, n_types), dtype=np.float64)
        b_idx = np.broadcast_to(np.arange(b_len, dtype=np.int64)[None, :], (n_s, b_len))
        n_idx = np.broadcast_to(n_idx_base, (n_s, b_len))
        if has_missing:
            vmask = p_labels >= 0
            y_nbt[n_idx[vmask], b_idx[vmask], p_labels[vmask]] = 1.0
        else:
            y_nbt[n_idx, b_idx, p_labels] = 1.0

        y_2d = y_nbt.reshape(n_s, b_len * n_types)
        m_2d = adj_s @ y_2d
        m_nbt = m_2d.reshape(n_s, b_len, n_types)
        o_batch = np.matmul(y_nbt.transpose(1, 2, 0), m_nbt.transpose(1, 0, 2))

        if perms_s is not None:
            perms_s[start : start + b_len] = o_batch

        sum_s += o_batch.sum(axis=0)
        sum_sq_s += np.square(o_batch).sum(axis=0)
        ge_s += np.sum(o_batch >= (obs_s[None, :, :] - 1e-12), axis=0)
        le_s += np.sum(o_batch <= (obs_s[None, :, :] + 1e-12), axis=0)

    return obs_s, sum_s, sum_sq_s, ge_s, le_s, perms_s


def _run_global_stratified_permutations(
    adj: sp.csr_matrix,
    labels: np.ndarray,
    sample_slices: Dict[str, np.ndarray],
    n_types: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback batched SpMM across full graph when external graph has cross-sample edges."""
    n = labels.size
    y_all = _build_indicator_matrix(labels, n_types)
    obs = y_all.T @ (adj @ y_all)

    sum_g = np.zeros((n_types, n_types), dtype=np.float64)
    sum_sq_g = np.zeros((n_types, n_types), dtype=np.float64)
    ge_g = np.zeros((n_types, n_types), dtype=np.float64)
    le_g = np.zeros((n_types, n_types), dtype=np.float64)

    max_batch = max(1, min(50, n_permutations, 8_000_000 // max(1, n * n_types)))
    has_missing = bool(np.any(labels < 0))
    n_idx_base = np.arange(n, dtype=np.int64)[:, None]

    for start in range(0, n_permutations, max_batch):
        b_len = min(max_batch, n_permutations - start)
        p_global = np.empty((n, b_len), dtype=np.int64)
        for idx_s in sample_slices.values():
            p_global[idx_s] = rng.permuted(
                np.broadcast_to(labels[idx_s], (b_len, idx_s.size)), axis=1
            ).T

        y_nbt = np.zeros((n, b_len, n_types), dtype=np.float64)
        b_idx = np.broadcast_to(np.arange(b_len, dtype=np.int64)[None, :], (n, b_len))
        n_idx = np.broadcast_to(n_idx_base, (n, b_len))
        if has_missing:
            vmask = p_global >= 0
            y_nbt[n_idx[vmask], b_idx[vmask], p_global[vmask]] = 1.0
        else:
            y_nbt[n_idx, b_idx, p_global] = 1.0

        y_2d = y_nbt.reshape(n, b_len * n_types)
        m_2d = adj @ y_2d
        m_nbt = m_2d.reshape(n, b_len, n_types)
        o_batch = np.matmul(y_nbt.transpose(1, 2, 0), m_nbt.transpose(1, 0, 2))

        sum_g += o_batch.sum(axis=0)
        sum_sq_g += np.square(o_batch).sum(axis=0)
        ge_g += np.sum(o_batch >= (obs[None, :, :] - 1e-12), axis=0)
        le_g += np.sum(o_batch <= (obs[None, :, :] + 1e-12), axis=0)

    return obs, sum_g, sum_sq_g, ge_g, le_g


def _compute_stats_from_moments(
    obs: np.ndarray,
    sum_null: np.ndarray,
    sum_sq_null: np.ndarray,
    ge_cnt: np.ndarray,
    le_cnt: np.ndarray,
    n_permutations: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute expected, z_score, log2_fc, p_enrich, p_avoid, p_two_sided from moments."""
    n_p = float(n_permutations)
    expected = sum_null / n_p
    denom_var = float(max(1, n_permutations - 1)) if n_permutations > 1 else 1.0
    var = np.maximum(0.0, (sum_sq_null - (np.square(sum_null) / n_p)) / denom_var)
    std = np.sqrt(var)
    z_mat = np.divide(
        obs - expected,
        std,
        out=np.zeros_like(obs, dtype=np.float64),
        where=std > 1e-12,
    )
    log2fc_mat = np.log2((obs + 0.5) / (expected + 0.5))
    p_enrich = (ge_cnt + 1.0) / (n_p + 1.0)
    p_avoid = (le_cnt + 1.0) / (n_p + 1.0)
    p_two = np.minimum(1.0, 2.0 * np.minimum(p_enrich, p_avoid))
    return expected, z_mat, log2fc_mat, p_enrich, p_avoid, p_two


def run_pairwise_interactions(
    df: pd.DataFrame,
    cell_type_col: str = "cell_type",
    sample_col: Optional[str] = None,
    graph_result: Optional[SpatialGraphResult] = None,
    mode: str = "delaunay",
    hop: int = 1,
    exact_ring: bool = True,
    k: int = 10,
    radius_um: float = 50.0,
    max_edge_um: Optional[float] = None,
    microns_per_pixel: float = 1.0,
    n_permutations: int = 1000,
    p_adjust: str = "fdr_bh",
    fdr_threshold: float = 0.05,
    z_threshold: float = 1.96,
    group_map: Optional[Dict[str, str]] = None,
    stat_test: str = "mann_whitney",
    random_state: Optional[int] = 42,
) -> InteractionResult:
    """Run stratified spatial cell-cell pairwise interaction permutation testing.

    Permutations shuffle phenotype labels strictly within each sample boundary
    before computing ``O = Y^T @ (A @ Y)`` using chunked sparse-dense matrix
    multiplication.
    """
    if int(n_permutations) < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations!r}.")

    n_perm = int(n_permutations)
    if graph_result is None:
        graph_res = build_spatial_graph(
            df,
            mode=mode,
            hop=hop,
            exact_ring=exact_ring,
            k=k,
            radius_um=radius_um,
            max_edge_um=max_edge_um,
            microns_per_pixel=microns_per_pixel,
            sample_col=sample_col,
        )
    else:
        graph_res = graph_result
        if graph_res.adjacency.shape != (len(df), len(df)):
            raise ValueError(
                f"graph_result adjacency shape {graph_res.adjacency.shape} does not match "
                f"DataFrame length {len(df)}."
            )

    resolved_sample_col = (
        sample_col
        or graph_res.params.get("sample_col")
        or find_sample_column(df)
        or COL_SAMPLE
    )

    if len(df) == 0:
        empty_mat = pd.DataFrame(dtype=np.float64)
        return InteractionResult(
            pairwise_df=pd.DataFrame(columns=PAIRWISE_COLUMNS),
            z_matrix=empty_mat,
            log2fc_matrix=empty_mat.copy(),
            q_matrix=empty_mat.copy(),
            mean_count_matrix=empty_mat.copy(),
            replicates_df=pd.DataFrame(),
            cohort_stats_df=pd.DataFrame(columns=COHORT_STATS_COLUMNS),
            cell_types=[],
            graph_result=graph_res,
            metadata={
                "cell_type_col": cell_type_col,
                "sample_col": resolved_sample_col,
                "n_permutations": n_perm,
                "p_adjust": p_adjust,
                "fdr_threshold": float(fdr_threshold),
                "z_threshold": float(z_threshold),
            },
        )

    resolved_ct_col = _resolve_cell_type_column(df, cell_type_col)
    cell_types, codes = _extract_cell_types(df[resolved_ct_col])
    n_types = len(cell_types)

    if n_types == 0:
        empty_mat = pd.DataFrame(dtype=np.float64)
        return InteractionResult(
            pairwise_df=pd.DataFrame(columns=PAIRWISE_COLUMNS),
            z_matrix=empty_mat,
            log2fc_matrix=empty_mat.copy(),
            q_matrix=empty_mat.copy(),
            mean_count_matrix=empty_mat.copy(),
            replicates_df=pd.DataFrame(),
            cohort_stats_df=pd.DataFrame(columns=COHORT_STATS_COLUMNS),
            cell_types=[],
            graph_result=graph_res,
            metadata={
                "cell_type_col": resolved_ct_col,
                "sample_col": resolved_sample_col,
                "n_permutations": n_perm,
                "p_adjust": p_adjust,
                "fdr_threshold": float(fdr_threshold),
                "z_threshold": float(z_threshold),
            },
        )

    adj_f64 = graph_res.adjacency.astype(np.float64)
    sample_slices = (
        graph_res.sample_slices
        if graph_res.sample_slices
        else {COL_SAMPLE: np.arange(len(df), dtype=np.int64)}
    )
    is_block_diag = _check_block_diagonal(graph_res.adjacency, sample_slices, len(df))
    is_symmetric = bool(graph_res.params.get("symmetric", True)) and (
        (graph_res.adjacency - graph_res.adjacency.T).nnz == 0
    )

    rng = np.random.default_rng(random_state)

    # Store full (n_perm, T, T) sum buffer when comfortably bounded in memory.
    can_accumulate_perms = is_block_diag and (n_perm * n_types * n_types <= 25_000_000)
    total_obs = np.zeros((n_types, n_types), dtype=np.float64)
    total_perms = (
        np.zeros((n_perm, n_types, n_types), dtype=np.float64)
        if can_accumulate_perms
        else None
    )

    rep_records: List[Dict[str, Any]] = []
    sample_obs_list: List[np.ndarray] = []

    for s_name, idx_s in sample_slices.items():
        idx_arr = np.asarray(idx_s, dtype=np.int64)
        if idx_arr.size == 0:
            continue
        adj_s = adj_f64[idx_arr][:, idx_arr]
        labels_s = codes[idx_arr]
        obs_s, sum_s, sum_sq_s, ge_s, le_s, perms_s = _run_sample_permutations(
            adj_s=adj_s,
            labels_s=labels_s,
            n_types=n_types,
            n_permutations=n_perm,
            rng=rng,
            store_all=can_accumulate_perms,
        )
        total_obs += obs_s
        if total_perms is not None and perms_s is not None:
            total_perms += perms_s
        elif not can_accumulate_perms:
            sample_obs_list.append(obs_s)

        exp_s, z_s, log2fc_s, pe_s, pa_s, pt_s = _compute_stats_from_moments(
            obs_s, sum_s, sum_sq_s, ge_s, le_s, n_perm
        )

        # Track which cell types are actually present in this sample.
        present_counts = np.bincount(labels_s[labels_s >= 0], minlength=n_types)
        sample_group = str(group_map.get(s_name, "Unassigned")) if group_map else "All"

        # Per-sample FDR adjustment for replicate classification.
        if is_symmetric:
            iu, ju = np.triu_indices(n_types, k=0)
            q_s_ut = adjust_p_values(pt_s[iu, ju], method=p_adjust)
            q_s = np.ones((n_types, n_types), dtype=np.float64)
            q_s[iu, ju] = q_s_ut
            q_s[ju, iu] = q_s_ut
        else:
            q_s = adjust_p_values(pt_s.ravel(), method=p_adjust).reshape(n_types, n_types)

        for i in range(n_types):
            src_name = cell_types[i]
            for j in range(n_types):
                tgt_name = cell_types[j]
                both_present = (present_counts[i] > 0) and (present_counts[j] > 0)
                z_val = float(z_s[i, j]) if both_present else np.nan
                fc_val = float(log2fc_s[i, j]) if both_present else np.nan
                q_val = float(q_s[i, j]) if both_present else np.nan
                cls_val = (
                    _classify_interaction(q_val, z_val, fdr_threshold, z_threshold)
                    if both_present
                    else "Non-significant"
                )
                rep_records.append(
                    {
                        resolved_sample_col: s_name,
                        "group": sample_group,
                        "source_type": src_name,
                        "target_type": tgt_name,
                        "pair": f"{src_name} <-> {tgt_name}",
                        "observed_count": float(obs_s[i, j]),
                        "expected_count": float(exp_s[i, j]),
                        "log2_fc": fc_val,
                        "z_score": z_val,
                        "p_enrich": float(pe_s[i, j]) if both_present else np.nan,
                        "p_avoid": float(pa_s[i, j]) if both_present else np.nan,
                        "p_two_sided": float(pt_s[i, j]) if both_present else np.nan,
                        "q_value": q_val,
                        "interaction_class": cls_val,
                    }
                )

    if total_perms is not None:
        sum_g = total_perms.sum(axis=0)
        sum_sq_g = np.square(total_perms).sum(axis=0)
        ge_g = np.sum(total_perms >= (total_obs[None, :, :] - 1e-12), axis=0)
        le_g = np.sum(total_perms <= (total_obs[None, :, :] + 1e-12), axis=0)
    else:
        total_obs, sum_g, sum_sq_g, ge_g, le_g = _run_global_stratified_permutations(
            adj=adj_f64,
            labels=codes,
            sample_slices=sample_slices,
            n_types=n_types,
            n_permutations=n_perm,
            rng=rng,
        )

    expected_g, z_g, log2fc_g, pe_g, pa_g, pt_g = _compute_stats_from_moments(
        total_obs, sum_g, sum_sq_g, ge_g, le_g, n_perm
    )

    if is_symmetric:
        iu, ju = np.triu_indices(n_types, k=0)
        q_ut = adjust_p_values(pt_g[iu, ju], method=p_adjust)
        q_g = np.ones((n_types, n_types), dtype=np.float64)
        q_g[iu, ju] = q_ut
        q_g[ju, iu] = q_ut
    else:
        q_g = adjust_p_values(pt_g.ravel(), method=p_adjust).reshape(n_types, n_types)

    row_edge_totals = total_obs.sum(axis=1, keepdims=True)
    cond_prob_g = np.divide(
        total_obs,
        row_edge_totals,
        out=np.zeros_like(total_obs, dtype=np.float64),
        where=row_edge_totals > 0.0,
    )

    global_type_counts = np.bincount(codes[codes >= 0], minlength=n_types).astype(np.float64)
    mean_count_g = np.divide(
        total_obs,
        global_type_counts[:, None],
        out=np.zeros_like(total_obs, dtype=np.float64),
        where=global_type_counts[:, None] > 0.0,
    )

    pair_records: List[Dict[str, Any]] = []
    for i, src in enumerate(cell_types):
        for j, tgt in enumerate(cell_types):
            q_val = float(q_g[i, j])
            z_val = float(z_g[i, j])
            pair_records.append(
                {
                    "source_type": src,
                    "target_type": tgt,
                    "observed_count": float(total_obs[i, j]),
                    "expected_count": float(expected_g[i, j]),
                    "conditional_prob": float(cond_prob_g[i, j]),
                    "log2_fc": float(log2fc_g[i, j]),
                    "z_score": z_val,
                    "p_enrich": float(pe_g[i, j]),
                    "p_avoid": float(pa_g[i, j]),
                    "p_two_sided": float(pt_g[i, j]),
                    "q_value": q_val,
                    "interaction_class": _classify_interaction(
                        q_val, z_val, fdr_threshold, z_threshold
                    ),
                }
            )

    pairwise_df = pd.DataFrame(pair_records, columns=PAIRWISE_COLUMNS)
    z_matrix = pd.DataFrame(z_g, index=cell_types, columns=cell_types)
    log2fc_matrix = pd.DataFrame(log2fc_g, index=cell_types, columns=cell_types)
    q_matrix = pd.DataFrame(q_g, index=cell_types, columns=cell_types)
    mean_count_matrix = pd.DataFrame(mean_count_g, index=cell_types, columns=cell_types)
    replicates_df = pd.DataFrame(rep_records)

    cohort_stats_df = _compute_cohort_stats(
        replicates_df=replicates_df,
        cell_types=cell_types,
        sample_col=resolved_sample_col,
        group_map=group_map,
        stat_test=stat_test,
        p_adjust=p_adjust,
        fdr_threshold=fdr_threshold,
        is_symmetric=is_symmetric,
    )

    metadata = {
        "cell_type_col": resolved_ct_col,
        "sample_col": resolved_sample_col,
        "n_cells": len(df),
        "n_samples": len(sample_slices),
        "n_permutations": n_perm,
        "p_adjust": p_adjust,
        "fdr_threshold": float(fdr_threshold),
        "z_threshold": float(z_threshold),
        "stat_test": stat_test,
        "symmetric": is_symmetric,
        "graph_mode": graph_res.mode,
        "graph_params": dict(graph_res.params),
    }

    return InteractionResult(
        pairwise_df=pairwise_df,
        z_matrix=z_matrix,
        log2fc_matrix=log2fc_matrix,
        q_matrix=q_matrix,
        mean_count_matrix=mean_count_matrix,
        replicates_df=replicates_df,
        cohort_stats_df=cohort_stats_df,
        cell_types=cell_types,
        graph_result=graph_res,
        metadata=metadata,
    )


def _compute_cohort_stats(
    replicates_df: pd.DataFrame,
    cell_types: List[str],
    sample_col: str,
    group_map: Optional[Dict[str, str]],
    stat_test: str,
    p_adjust: str,
    fdr_threshold: float,
    is_symmetric: bool,
    value_col: str = "z_score",
) -> pd.DataFrame:
    """Compute sample-level biological group comparisons per cell-type pair."""
    if group_map is None or replicates_df.empty:
        return pd.DataFrame(columns=COHORT_STATS_COLUMNS)

    mapped_df = replicates_df[replicates_df["group"] != "Unassigned"].copy()
    if mapped_df.empty or mapped_df["group"].nunique() < 2:
        return pd.DataFrame(columns=COHORT_STATS_COLUMNS)

    stat_rows: List[Dict[str, Any]] = []
    for i, src in enumerate(cell_types):
        start_j = i if is_symmetric else 0
        for j in range(start_j, len(cell_types)):
            tgt = cell_types[j]
            sub = mapped_df[
                (mapped_df["source_type"] == src) & (mapped_df["target_type"] == tgt)
            ]
            if sub.empty or sub[value_col].dropna().empty:
                continue
            comp_list = run_group_comparison(
                sub,
                value_col=value_col,
                group_col="group",
                sample_col=sample_col,
                aggregate_by_sample=True,
                test=stat_test,
                correction=p_adjust,
                alpha=fdr_threshold,
            )
            for item in comp_list:
                row = item.to_dict()
                row["source_type"] = src
                row["target_type"] = tgt
                row["pair"] = f"{src} <-> {tgt}"
                stat_rows.append(row)
                if is_symmetric and i != j:
                    mirror = dict(row)
                    mirror["source_type"] = tgt
                    mirror["target_type"] = src
                    mirror["pair"] = f"{tgt} <-> {src}"
                    stat_rows.append(mirror)

    if not stat_rows:
        return pd.DataFrame(columns=COHORT_STATS_COLUMNS)

    out_df = pd.DataFrame(stat_rows)
    ordered_cols = [c for c in COHORT_STATS_COLUMNS if c in out_df.columns]
    extra_cols = [c for c in out_df.columns if c not in ordered_cols]
    return out_df[ordered_cols + extra_cols]


__all__ = [
    "COHORT_STATS_COLUMNS",
    "InteractionResult",
    "PAIRWISE_COLUMNS",
    "run_pairwise_interactions",
]
