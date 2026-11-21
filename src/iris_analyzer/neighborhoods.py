"""
IRIS Analyzer Cellular Neighborhoods & Spatial-LDA Engine
==========================================================
Headless-safe (pure NumPy/SciPy/scikit-learn, zero Qt/Napari/cvxpy imports)
spatial microenvironment discovery supporting:

1. Moving-Window Cellular Neighborhoods (Nolan Lab; Schuerch et al., Cell 2020):
   - Per-sample k-NN or fixed-radius moving-window cell-type composition
   - Clustering via MiniBatchKMeans, KMeans, or Leiden
   - Log2 fold enrichment over global cell-type baseline proportions
   - Sample-stratified Delaunay inter-CN coordination Z-score matrix
   - Multi-CN Spatial Contexts across wider windows

2. Spatial-LDA (Calico; Chen et al., PSB 2020 / JCB 2020):
   - Per-sample bag-of-cell-types neighborhood count featurization
   - Sample-isolated Delaunay spatial difference matrix D
   - Variational EM (digamma updates) coupled with sparse ADMM spatial prior
     smoothing via SuperLU factorization (splu) and edge soft-thresholding
   - Cell-level topic probabilities, normalized transition entropy, and ELBO
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln, polygamma, softmax
from sklearn.cluster import KMeans, MiniBatchKMeans

from iris_analyzer.graph import build_spatial_graph
from iris_analyzer.interactions import _extract_cell_types, _resolve_cell_type_column
from iris_common.engine.statistics import run_group_comparison
from iris_common.schema import (
    COL_SAMPLE,
    COL_X,
    COL_Y,
    X_COORD_ALIASES,
    Y_COORD_ALIASES,
    ensure_canonical_dataframe,
    find_sample_column,
)


@dataclass
class SpatialLDAResult:
    """Results from Spatial-LDA spatial topic modeling."""

    updated_df: pd.DataFrame
    topic_probabilities: np.ndarray
    topic_word_matrix: np.ndarray
    cell_types: List[str]
    topic_entropy: np.ndarray
    difference_l1_norm: float
    elbo_history: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MovingWindowCNResult:
    """Results from Moving-Window Cellular Neighborhood discovery."""

    updated_df: pd.DataFrame
    centers: np.ndarray
    log2_enrichment: np.ndarray
    cell_types: List[str]
    replicates_df: pd.DataFrame
    inter_cn_coordination: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    baseline_proportions: Optional[np.ndarray] = None

    @property
    def log2_enrichment_df(self) -> pd.DataFrame:
        """Return ``log2_enrichment`` labeled with CN names and cell types."""
        cn_names = self.metadata.get(
            "cn_names", [f"CN {i}" for i in range(self.centers.shape[0])]
        )
        return pd.DataFrame(
            self.log2_enrichment,
            index=cn_names,
            columns=self.cell_types,
        )


def _resolve_xy_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract float64 ``(x_px, y_px)`` arrays from *df* without mutating it."""
    if COL_X in df.columns and COL_Y in df.columns:
        x_col, y_col = COL_X, COL_Y
    else:
        lower_map = {str(c).strip().lower(): str(c) for c in df.columns}
        x_col = None
        y_col = None
        for alias in X_COORD_ALIASES:
            if alias in df.columns and pd.api.types.is_numeric_dtype(df[alias]):
                x_col = alias
                break
            if alias.lower() in lower_map and pd.api.types.is_numeric_dtype(
                df[lower_map[alias.lower()]]
            ):
                x_col = lower_map[alias.lower()]
                break
        for alias in Y_COORD_ALIASES:
            if alias in df.columns and pd.api.types.is_numeric_dtype(df[alias]):
                y_col = alias
                break
            if alias.lower() in lower_map and pd.api.types.is_numeric_dtype(
                df[lower_map[alias.lower()]]
            ):
                y_col = lower_map[alias.lower()]
                break
        if x_col is None or y_col is None:
            canon = ensure_canonical_dataframe(df, inplace=False, require_coordinates=True)
            x_col, y_col = COL_X, COL_Y
            return (
                pd.to_numeric(canon[x_col], errors="coerce").to_numpy(dtype=np.float64),
                pd.to_numeric(canon[y_col], errors="coerce").to_numpy(dtype=np.float64),
            )
    return (
        pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float64),
    )


def _resolve_sample_slices(
    df: pd.DataFrame,
    sample_col: Optional[str],
) -> Tuple[Optional[str], Dict[str, np.ndarray]]:
    """Resolve sample column and positional row index arrays per sample."""
    resolved_sample = sample_col if (sample_col and sample_col in df.columns) else None
    if resolved_sample is None:
        resolved_sample = find_sample_column(df)

    if resolved_sample is None or len(df) == 0:
        return resolved_sample, {"Sample_0": np.arange(len(df), dtype=np.int64)}

    sample_vals = df[resolved_sample].fillna("Sample_0").astype(str).to_numpy()
    slices: Dict[str, np.ndarray] = {}
    for pos, name in enumerate(sample_vals):
        slices.setdefault(name, []).append(pos)
    return resolved_sample, {
        k: np.asarray(v, dtype=np.int64) for k, v in slices.items()
    }


def _build_window_count_matrix(
    x_px: np.ndarray,
    y_px: np.ndarray,
    codes: np.ndarray,
    n_classes: int,
    sample_slices: Dict[str, np.ndarray],
    *,
    mode: str = "knn",
    k_neighbors: int = 10,
    radius_um: float = 50.0,
    include_self: bool = True,
    microns_per_pixel: float = 1.0,
) -> np.ndarray:
    """Build per-sample neighborhood count matrix ``W`` of shape ``(N, n_classes)``."""
    n_cells = len(codes)
    counts = np.zeros((n_cells, n_classes), dtype=np.float64)
    if n_cells == 0 or n_classes == 0:
        return counts

    one_hot = np.zeros((n_cells, n_classes), dtype=np.float64)
    valid_cls = (codes >= 0) & (codes < n_classes)
    if np.any(valid_cls):
        rows = np.flatnonzero(valid_cls)
        one_hot[rows, codes[rows]] = 1.0

    norm_mode = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    is_radius = norm_mode in ("radius", "radius_um", "radius_(um)", "radius_(µm)")
    scale = float(microns_per_pixel) if microns_per_pixel > 0 else 1.0

    for idx in sample_slices.values():
        if idx.size == 0:
            continue
        coords_um = np.column_stack([x_px[idx], y_px[idx]]) * scale
        valid_xy = np.all(np.isfinite(coords_um), axis=1)
        valid_local = np.flatnonzero(valid_xy)
        n_valid = valid_local.size
        if n_valid == 0:
            continue

        sub_coords = coords_um[valid_local]
        tree = cKDTree(sub_coords)

        row_parts: List[np.ndarray] = []
        col_parts: List[np.ndarray] = []

        if is_radius:
            r_eff = max(float(radius_um), 0.0)
            nbs_list = tree.query_ball_point(sub_coords, r=r_eff)
            for q_pos, nb_list in enumerate(nbs_list):
                if not include_self:
                    nb_arr = [nb for nb in nb_list if nb != q_pos]
                else:
                    nb_arr = nb_list
                if nb_arr:
                    q_global = int(idx[valid_local[q_pos]])
                    nb_globals = idx[valid_local[np.asarray(nb_arr, dtype=np.int64)]]
                    row_parts.append(np.full(nb_globals.size, q_global, dtype=np.int64))
                    col_parts.append(nb_globals)
        else:
            k_req = max(1, int(k_neighbors))
            if include_self:
                k_eff = min(k_req, n_valid)
                _, nn_idx = tree.query(sub_coords, k=k_eff)
                if nn_idx.ndim == 1:
                    nn_idx = nn_idx[:, np.newaxis]
                q_globals = idx[valid_local]
                nb_globals = idx[valid_local[nn_idx.ravel()]]
                row_parts.append(np.repeat(q_globals, k_eff))
                col_parts.append(nb_globals)
            else:
                if n_valid <= 1:
                    continue
                k_query = min(k_req + 1, n_valid)
                _, nn_idx = tree.query(sub_coords, k=k_query)
                if nn_idx.ndim == 1:
                    nn_idx = nn_idx[:, np.newaxis]
                k_keep = min(k_req, n_valid - 1)
                for q_pos in range(n_valid):
                    cand = [int(nb) for nb in nn_idx[q_pos] if int(nb) != q_pos][:k_keep]
                    if cand:
                        q_global = int(idx[valid_local[q_pos]])
                        nb_globals = idx[valid_local[np.asarray(cand, dtype=np.int64)]]
                        row_parts.append(np.full(nb_globals.size, q_global, dtype=np.int64))
                        col_parts.append(nb_globals)

        if row_parts:
            r_all = np.concatenate(row_parts)
            c_all = np.concatenate(col_parts)
            adj_win = sp.csr_matrix(
                (np.ones(r_all.size, dtype=np.float64), (r_all, c_all)),
                shape=(n_cells, n_cells),
            )
            counts += adj_win @ one_hot

    return counts


def _build_delaunay_difference_matrix(
    df: pd.DataFrame,
    sample_col: Optional[str],
    microns_per_pixel: float,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, Dict[str, np.ndarray]]:
    """Build undirected Delaunay incidence matrix ``D`` ``(E, N)`` and adjacency ``A``."""
    n_cells = len(df)
    if n_cells == 0:
        empty_d = sp.csr_matrix((0, 0), dtype=np.float64)
        empty_a = sp.csr_matrix((0, 0), dtype=bool)
        return empty_d, empty_a, {}

    graph_res = build_spatial_graph(
        df,
        mode="delaunay",
        hop=1,
        microns_per_pixel=microns_per_pixel,
        sample_col=sample_col,
        symmetric=True,
    )
    adj = graph_res.adjacency
    tri_u = sp.triu(adj, k=1)
    u_idx, v_idx = tri_u.nonzero()
    n_edges = u_idx.size
    if n_edges == 0:
        return sp.csr_matrix((0, n_cells), dtype=np.float64), adj, graph_res.sample_slices

    rows = np.concatenate([np.arange(n_edges, dtype=np.int64), np.arange(n_edges, dtype=np.int64)])
    cols = np.concatenate([u_idx.astype(np.int64), v_idx.astype(np.int64)])
    vals = np.concatenate([np.ones(n_edges, dtype=np.float64), -np.ones(n_edges, dtype=np.float64)])
    diff_mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_edges, n_cells))
    return diff_mat, adj, graph_res.sample_slices


def _normalized_shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute normalized Shannon entropy in ``[0, 1]`` for rows of ``probs``."""
    n_cells, n_topics = probs.shape
    if n_cells == 0:
        return np.zeros(0, dtype=np.float64)
    if n_topics <= 1:
        return np.zeros(n_cells, dtype=np.float64)
    p_safe = np.clip(probs, 1e-16, 1.0)
    raw_ent = -np.sum(probs * np.log(p_safe), axis=1)
    norm_ent = raw_ent / math.log(float(n_topics))
    return np.clip(norm_ent, 0.0, 1.0)


def compute_cn_replicates_df(
    df: pd.DataFrame,
    *,
    cn_col: str = "Cellular_Neighborhood",
    cn_id_col: str = "Cellular_Neighborhood_ID",
    sample_col: Optional[str] = None,
    cn_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Aggregate per-sample CN counts and proportions including zero-count entries."""
    cols = [
        COL_SAMPLE,
        "Cellular_Neighborhood",
        "Cellular_Neighborhood_ID",
        "cell_count",
        "total_sample_cells",
        "proportion",
        "percentage",
    ]
    if df.empty or cn_col not in df.columns:
        return pd.DataFrame(columns=cols)

    resolved_sample, sample_slices = _resolve_sample_slices(df, sample_col)
    if cn_names is None:
        if cn_id_col in df.columns:
            pairs = (
                df[[cn_id_col, cn_col]]
                .dropna()
                .drop_duplicates()
                .sort_values(cn_id_col)
            )
            ordered_ids = [int(v) for v in pairs[cn_id_col].tolist()]
            ordered_names = [str(v) for v in pairs[cn_col].tolist()]
        else:
            ordered_names = sorted(df[cn_col].dropna().astype(str).unique().tolist())
            ordered_ids = list(range(len(ordered_names)))
    else:
        ordered_names = [str(n) for n in cn_names]
        ordered_ids = list(range(len(ordered_names)))

    rows: List[Dict[str, Any]] = []
    for sample_name, idx in sample_slices.items():
        sub = df.iloc[idx]
        total_cells = int(len(sub))
        if cn_id_col in sub.columns:
            vc = pd.to_numeric(sub[cn_id_col], errors="coerce").value_counts().to_dict()
            use_id = True
        else:
            vc = sub[cn_col].astype(str).value_counts().to_dict()
            use_id = False

        for cid, cname in zip(ordered_ids, ordered_names):
            cnt = int(vc.get(cid if use_id else cname, 0))
            prop = float(cnt / total_cells) if total_cells > 0 else 0.0
            rows.append(
                {
                    COL_SAMPLE: sample_name,
                    "Cellular_Neighborhood": cname,
                    "Cellular_Neighborhood_ID": int(cid),
                    "cell_count": cnt,
                    "total_sample_cells": total_cells,
                    "proportion": prop,
                    "percentage": prop * 100.0,
                }
            )

    return pd.DataFrame(rows, columns=cols)


def compute_cn_cohort_stats(
    replicates_df: pd.DataFrame,
    group_map: Dict[str, str],
    *,
    value_col: str = "percentage",
    stat_test: str = "mann_whitney",
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Run sample-level statistical comparisons across groups for each CN."""
    if replicates_df is None or replicates_df.empty or not group_map:
        return pd.DataFrame()

    work = replicates_df.copy()
    work["_CohortGroup"] = work[COL_SAMPLE].astype(str).map(
        {str(k): str(v) for k, v in group_map.items()}
    )
    work = work.dropna(subset=["_CohortGroup"])
    if work.empty or work["_CohortGroup"].nunique() < 2:
        return pd.DataFrame()

    stat_tables: List[pd.DataFrame] = []
    ordered_cns = list(dict.fromkeys(work["Cellular_Neighborhood"].astype(str).tolist()))
    for cn_name in ordered_cns:
        sub = work[work["Cellular_Neighborhood"].astype(str) == cn_name].copy()
        if sub["_CohortGroup"].nunique() < 2:
            continue
        res_list = run_group_comparison(
            sub,
            value_col=value_col,
            group_col="_CohortGroup",
            sample_col=COL_SAMPLE,
            aggregate_by_sample=True,
            test=stat_test,
            correction=correction,
        )
        res_df = res_list.to_dataframe()
        if not res_df.empty:
            res_df.insert(0, "Cellular_Neighborhood", cn_name)
            stat_tables.append(res_df)

    if not stat_tables:
        return pd.DataFrame()
    return pd.concat(stat_tables, ignore_index=True)


def compute_inter_cn_coordination(
    df: pd.DataFrame,
    cn_col: str = "Cellular_Neighborhood_ID",
    sample_col: Optional[str] = None,
    microns_per_pixel: float = 1.0,
    n_permutations: int = 200,
    random_state: int = 42,
    cn_names: Optional[Sequence[str]] = None,
    adjacency: Optional[sp.csr_matrix] = None,
    sample_slices: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """Compute sample-stratified Delaunay contact Z-scores between CN labels."""
    if df.empty or cn_col not in df.columns:
        return pd.DataFrame()

    raw_ids = pd.to_numeric(df[cn_col], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    valid_ids = raw_ids[raw_ids >= 0]
    if valid_ids.size == 0:
        return pd.DataFrame()

    k_max = int(valid_ids.max()) + 1
    if cn_names is not None:
        k_count = max(k_max, len(cn_names))
        labels = [
            str(cn_names[i]) if i < len(cn_names) else f"CN {i}"
            for i in range(k_count)
        ]
    else:
        k_count = k_max
        labels = [f"CN {i}" for i in range(k_count)]

    if adjacency is None or sample_slices is None:
        graph_res = build_spatial_graph(
            df,
            mode="delaunay",
            hop=1,
            microns_per_pixel=microns_per_pixel,
            sample_col=sample_col,
            symmetric=True,
        )
        adjacency = graph_res.adjacency
        sample_slices = graph_res.sample_slices

    tri_u = sp.triu(adjacency, k=1)
    u_all, v_all = tri_u.nonzero()
    if u_all.size == 0:
        return pd.DataFrame(
            np.zeros((k_count, k_count), dtype=np.float64),
            index=labels,
            columns=labels,
        )

    valid_edge_mask = (raw_ids[u_all] >= 0) & (raw_ids[v_all] >= 0)
    u_edges = u_all[valid_edge_mask]
    v_edges = v_all[valid_edge_mask]
    if u_edges.size == 0:
        return pd.DataFrame(
            np.zeros((k_count, k_count), dtype=np.float64),
            index=labels,
            columns=labels,
        )

    def _contact_matrix(lbl_arr: np.ndarray) -> np.ndarray:
        flat_idx = lbl_arr[u_edges] * k_count + lbl_arr[v_edges]
        counts = np.bincount(flat_idx, minlength=k_count * k_count).reshape(k_count, k_count)
        return (counts + counts.T).astype(np.float64)

    obs = _contact_matrix(raw_ids)
    n_perms = max(1, int(n_permutations))
    rng = np.random.default_rng(random_state)

    # Precompute valid sample index slices for fast stratified permutation.
    valid_sample_slices: List[np.ndarray] = []
    for s_idx in sample_slices.values():
        sub_valid = s_idx[raw_ids[s_idx] >= 0]
        if sub_valid.size > 1:
            valid_sample_slices.append(sub_valid)

    sum_mat = np.zeros((k_count, k_count), dtype=np.float64)
    sum_sq_mat = np.zeros((k_count, k_count), dtype=np.float64)
    perm_ids = raw_ids.copy()

    for _ in range(n_perms):
        for s_idx in valid_sample_slices:
            perm_ids[s_idx] = rng.permutation(raw_ids[s_idx])
        c_perm = _contact_matrix(perm_ids)
        sum_mat += c_perm
        sum_sq_mat += c_perm * c_perm

    mean_null = sum_mat / float(n_perms)
    if n_perms > 1:
        var_null = np.maximum(
            0.0, (sum_sq_mat - float(n_perms) * (mean_null * mean_null)) / float(n_perms - 1)
        )
        std_null = np.sqrt(var_null)
    else:
        std_null = np.zeros_like(mean_null)

    z_mat = np.divide(
        obs - mean_null,
        std_null,
        out=np.zeros_like(obs, dtype=np.float64),
        where=std_null > 1e-12,
    )
    z_mat = 0.5 * (z_mat + z_mat.T)
    return pd.DataFrame(z_mat, index=labels, columns=labels)


def _dirichlet_prior_obj_grad_hess(
    xi_mat: np.ndarray,
    cs_mat: np.ndarray,
    target_conc: float,
    l2_anchor: float = 1e-3,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Negative Dirichlet log-likelihood and derivatives w.r.t. log-space prior ``xi``."""
    n_cells = xi_mat.shape[0]
    probs = softmax(xi_mat, axis=1)
    alpha = np.maximum(1e-6, target_conc * probs)
    obj = (
        np.sum(gammaln(alpha))
        - n_cells * gammaln(target_conc)
        - np.sum((alpha - 1.0) * cs_mat)
        + 0.5 * l2_anchor * np.sum(xi_mat * xi_mat)
    )
    diff = digamma(alpha) - digamma(target_conc) - cs_mat
    grad = alpha * (diff - np.sum(probs * diff, axis=1, keepdims=True)) + l2_anchor * xi_mat
    hess_diag = np.maximum(
        1e-4,
        (alpha * alpha) * polygamma(1, alpha) * (1.0 - probs) + l2_anchor,
    )
    return float(obj), grad, hess_diag


def _admm_update_xi(
    xi_init: np.ndarray,
    gamma: np.ndarray,
    diff_mat: sp.csr_matrix,
    dtd: sp.csc_matrix,
    difference_penalty: float,
    target_conc: float,
    *,
    admm_rho: float = 0.1,
    max_dirichlet_iters: int = 20,
    max_admm_iters: int = 40,
) -> np.ndarray:
    """Update spatially smoothed log-space prior ``xi`` via sparse LU ADMM."""
    xi = xi_init.copy()
    n_cells, n_topics = xi.shape
    if n_cells == 0 or n_topics <= 1:
        return xi

    cs = digamma(gamma) - digamma(np.sum(gamma, axis=1, keepdims=True))
    n_edges = diff_mat.shape[0]
    pen = max(float(difference_penalty), 0.0)

    if pen <= 0.0 or n_edges == 0:
        iters = max(1, min(int(max_dirichlet_iters), 20))
        for _ in range(iters):
            obj, grad, hess = _dirichlet_prior_obj_grad_hess(xi, cs, target_conc)
            step = grad / hess
            step_size = 1.0
            for _ in range(8):
                xi_cand = np.clip(xi - step_size * step, -8.0, 8.0)
                obj_cand, _, _ = _dirichlet_prior_obj_grad_hess(xi_cand, cs, target_conc)
                if obj_cand <= obj + 1e-9:
                    xi = xi_cand
                    break
                step_size *= 0.5
        return xi

    z = diff_mat @ xi
    u = np.zeros_like(z)
    rho_cur = max(float(admm_rho), 0.1 * pen, 1e-4)
    n_admm = max(1, int(max_admm_iters))
    n_newton = 1 if n_cells > 800 else 2

    for _ in range(n_admm):
        for _ in range(n_newton):
            obj_dir, g_dir, h_dir = _dirichlet_prior_obj_grad_hess(xi, cs, target_conc)
            resid = diff_mat @ xi - z + u
            aug_obj = obj_dir + 0.5 * rho_cur * float(np.sum(resid * resid))
            g_tot = g_dir + rho_cur * (diff_mat.T @ resid)

            step_mat = np.zeros_like(xi)
            for k in range(n_topics):
                mat_k = sp.diags(h_dir[:, k], format="csc") + rho_cur * dtd
                lu = spla.splu(mat_k)
                step_mat[:, k] = lu.solve(g_tot[:, k])

            step_size = 1.0
            for _ in range(8):
                xi_cand = np.clip(xi - step_size * step_mat, -8.0, 8.0)
                o_cand, _, _ = _dirichlet_prior_obj_grad_hess(xi_cand, cs, target_conc)
                r_cand = diff_mat @ xi_cand - z + u
                if o_cand + 0.5 * rho_cur * float(np.sum(r_cand * r_cand)) <= aug_obj + 1e-9:
                    xi = xi_cand
                    break
                step_size *= 0.5

        dx = diff_mat @ xi
        v = dx + u
        kappa = pen / rho_cur
        z_old = z
        z = np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)
        u = u + (dx - z)

        p_res = float(np.linalg.norm(dx - z))
        d_res = float(rho_cur * np.linalg.norm(diff_mat.T @ (z - z_old)))
        if p_res < 1e-4 and d_res < 1e-4:
            break
        if p_res > 4.0 * max(d_res, 1e-12):
            rho_cur = min(rho_cur * 2.0, 1e4)
            u *= 0.5
        elif d_res > 4.0 * max(p_res, 1e-12):
            rho_cur = max(rho_cur * 0.5, 1e-4)
            u *= 2.0

    return xi


def _compute_spatial_lda_elbo(
    counts: np.ndarray,
    alpha: np.ndarray,
    gamma: np.ndarray,
    lam: np.ndarray,
    eta: float,
    diff_mat: sp.csr_matrix,
    xi: np.ndarray,
    difference_penalty: float,
) -> float:
    """Compute variational ELBO minus spatial difference L1 regularization."""
    elog_theta = digamma(gamma) - digamma(np.sum(gamma, axis=1, keepdims=True))
    elog_beta = digamma(lam) - digamma(np.sum(lam, axis=1, keepdims=True))
    exp_theta = np.exp(elog_theta)
    exp_beta = np.exp(elog_beta)

    norm_phi = np.maximum(exp_theta @ exp_beta, 1e-300)
    elbo = float(np.sum(counts * np.log(norm_phi)))

    # E[log p(theta | alpha)] - E[log q(theta | gamma)]
    elbo += float(
        np.sum(gammaln(np.sum(alpha, axis=1)) - np.sum(gammaln(alpha), axis=1))
        + np.sum((alpha - gamma) * elog_theta)
        + np.sum(np.sum(gammaln(gamma), axis=1) - gammaln(np.sum(gamma, axis=1)))
    )

    # E[log p(beta | eta)] - E[log q(beta | lambda)]
    n_topics, n_vocab = lam.shape
    elbo += float(
        n_topics * (gammaln(n_vocab * eta) - n_vocab * gammaln(eta))
        + np.sum((eta - lam) * elog_beta)
        + np.sum(np.sum(gammaln(lam), axis=1) - gammaln(np.sum(lam, axis=1)))
    )

    if difference_penalty > 0.0 and diff_mat.shape[0] > 0:
        elbo -= float(difference_penalty) * float(np.sum(np.abs(diff_mat @ xi)))
    return elbo


def run_spatial_lda(
    df: pd.DataFrame,
    cell_type_col: str = "cell_type",
    sample_col: Optional[str] = None,
    n_topics: int = 6,
    difference_penalty: float = 0.25,
    radius_um: float = 50.0,
    k_neighbors: Optional[int] = None,
    featurization_mode: str = "radius",
    microns_per_pixel: float = 1.0,
    admm_rho: float = 0.1,
    n_iters: int = 5,
    max_dirichlet_iters: int = 20,
    max_admm_iters: int = 40,
    topic_prior_alpha: float = 0.1,
    word_prior_eta: float = 0.1,
    random_state: int = 42,
) -> SpatialLDAResult:
    """Fit Spatial-LDA topic model with sparse ADMM spatial prior smoothing.

    Uses pure NumPy/SciPy variational Bayes EM and SuperLU-backed ADMM fusion
    over sample-isolated Delaunay adjacency edges (zero ``cvxpy`` dependency).
    """
    if int(n_topics) < 1:
        raise ValueError(f"n_topics must be >= 1, got {n_topics}.")
    k_topics = int(n_topics)

    resolved_ct = _resolve_cell_type_column(df, cell_type_col)
    if df.empty:
        out_df = df.copy()
        out_df["Cellular_Neighborhood"] = pd.Series(dtype=object)
        out_df["Cellular_Neighborhood_ID"] = pd.Series(dtype=np.int64)
        out_df["SpatialLDA_MaxProb"] = pd.Series(dtype=np.float64)
        out_df["SpatialLDA_Entropy"] = pd.Series(dtype=np.float64)
        for k in range(k_topics):
            out_df[f"Topic_{k}_Prob"] = pd.Series(dtype=np.float64)
        return SpatialLDAResult(
            updated_df=out_df,
            topic_probabilities=np.zeros((0, k_topics), dtype=np.float64),
            topic_word_matrix=np.zeros((k_topics, 0), dtype=np.float64),
            cell_types=[],
            topic_entropy=np.zeros(0, dtype=np.float64),
            difference_l1_norm=0.0,
            elbo_history=[],
            metadata={"n_cells": 0, "n_topics": k_topics},
        )

    x_px, y_px = _resolve_xy_arrays(df)
    cell_types, codes = _extract_cell_types(df[resolved_ct])
    n_cells = len(df)
    n_vocab = len(cell_types)

    if n_vocab == 0:
        raise ValueError(f"No valid non-empty phenotype labels found in {resolved_ct!r}.")

    diff_mat, adj_delaunay, sample_slices = _build_delaunay_difference_matrix(
        df, sample_col=sample_col, microns_per_pixel=microns_per_pixel
    )
    dtd = (diff_mat.T @ diff_mat).tocsc()

    norm_feat = str(featurization_mode).strip().lower().replace("-", "_")
    use_knn = norm_feat in ("knn", "k_nn")
    counts = _build_window_count_matrix(
        x_px,
        y_px,
        codes,
        n_vocab,
        sample_slices,
        mode="knn" if use_knn else "radius",
        k_neighbors=int(k_neighbors or 10),
        radius_um=float(radius_um),
        include_self=True,
        microns_per_pixel=microns_per_pixel,
    )

    eta = max(float(word_prior_eta), 1e-6)
    alpha0 = max(float(topic_prior_alpha), 1e-4)
    pen = max(float(difference_penalty), 0.0)

    if k_topics == 1:
        topic_probs = np.ones((n_cells, 1), dtype=np.float64)
        word_totals = counts.sum(axis=0, keepdims=True) + eta
        topic_word = word_totals / np.sum(word_totals, axis=1, keepdims=True)
        entropy = np.zeros(n_cells, dtype=np.float64)
        diff_l1 = 0.0
        elbo_hist: List[float] = []
    else:
        rng = np.random.default_rng(random_state)
        row_totals = counts.sum(axis=1, keepdims=True)
        scale_init = max(float(np.sum(counts)), 1.0) / float(k_topics * n_vocab * 10.0)
        lam = eta + rng.gamma(1.0, 1.0, size=(k_topics, n_vocab)) * max(scale_init, 1.0)
        xi = np.zeros((n_cells, k_topics), dtype=np.float64)

        valid_sums = row_totals[row_totals > 0]
        mean_w = float(np.mean(valid_sums)) if valid_sums.size else 1.0
        target_conc = max(alpha0 * float(k_topics), mean_w)

        def _run_e_step(
            alpha_mat: np.ndarray, lam_mat: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            exp_beta = np.exp(digamma(lam_mat) - digamma(np.sum(lam_mat, axis=1, keepdims=True)))
            gamma_cur = alpha_mat + row_totals / float(k_topics)
            inner_max = max(1, int(max_dirichlet_iters))
            for _ in range(inner_max):
                exp_theta = np.exp(
                    digamma(gamma_cur) - digamma(np.sum(gamma_cur, axis=1, keepdims=True))
                )
                norm_phi = exp_theta @ exp_beta + 1e-16
                sstats_doc = (counts / norm_phi) @ exp_beta.T
                gamma_next = alpha_mat + exp_theta * sstats_doc
                if float(np.mean(np.abs(gamma_next - gamma_cur))) < 1e-4:
                    gamma_cur = gamma_next
                    break
                gamma_cur = gamma_next
            return gamma_cur, exp_beta

        elbo_hist = []
        outer_iters = max(1, int(n_iters))
        for outer_idx in range(outer_iters):
            if outer_idx == 0:
                alpha_cur = np.full((n_cells, k_topics), alpha0, dtype=np.float64)
            else:
                alpha_cur = np.maximum(1e-4, target_conc * softmax(xi, axis=1))

            for _lda_step in range(5):
                gamma, exp_beta = _run_e_step(alpha_cur, lam)
                exp_theta = np.exp(digamma(gamma) - digamma(np.sum(gamma, axis=1, keepdims=True)))
                norm_phi = exp_theta @ exp_beta + 1e-16
                suff_stats = exp_beta * (exp_theta.T @ (counts / norm_phi))
                lam = eta + suff_stats

            gamma, _ = _run_e_step(alpha_cur, lam)
            xi = _admm_update_xi(
                xi,
                gamma,
                diff_mat,
                dtd,
                difference_penalty=pen,
                target_conc=target_conc,
                admm_rho=admm_rho,
                max_dirichlet_iters=max_dirichlet_iters,
                max_admm_iters=max_admm_iters,
            )

            alpha_post = np.maximum(1e-4, target_conc * softmax(xi, axis=1))
            gamma_eval, _ = _run_e_step(alpha_post, lam)
            elbo_hist.append(
                _compute_spatial_lda_elbo(
                    counts, alpha_post, gamma_eval, lam, eta, diff_mat, xi, pen
                )
            )

        alpha_final = np.maximum(1e-4, target_conc * softmax(xi, axis=1))
        gamma, _ = _run_e_step(alpha_final, lam)
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        topic_probs = np.divide(
            gamma,
            gamma_sum,
            out=np.full_like(gamma, 1.0 / float(k_topics)),
            where=gamma_sum > 0,
        )
        topic_probs = np.clip(topic_probs, 0.0, 1.0)
        topic_probs /= np.sum(topic_probs, axis=1, keepdims=True)

        topic_word = lam / np.sum(lam, axis=1, keepdims=True)
        entropy = _normalized_shannon_entropy(topic_probs)
        diff_l1 = (
            float(np.sum(np.abs(diff_mat @ topic_probs)))
            if diff_mat.shape[0] > 0
            else 0.0
        )

    dom_ids = np.argmax(topic_probs, axis=1).astype(np.int64)
    max_probs = np.max(topic_probs, axis=1).astype(np.float64)
    topic_names = [f"Topic {k}" for k in range(k_topics)]

    updated_df = df.copy()
    updated_df["Cellular_Neighborhood"] = [topic_names[int(idx)] for idx in dom_ids]
    updated_df["Cellular_Neighborhood_ID"] = dom_ids
    updated_df["SpatialLDA_MaxProb"] = max_probs
    updated_df["SpatialLDA_Entropy"] = entropy
    for k in range(k_topics):
        updated_df[f"Topic_{k}_Prob"] = topic_probs[:, k]

    # Global baseline and log2 enrichment of topics over tissue baseline
    baseline = np.zeros(n_vocab, dtype=np.float64)
    valid_codes = codes[(codes >= 0) & (codes < n_vocab)]
    if valid_codes.size > 0:
        baseline = np.bincount(valid_codes, minlength=n_vocab).astype(np.float64)
        baseline /= float(valid_codes.size)
    else:
        baseline[:] = 1.0 / float(n_vocab)

    eps = 1e-4
    log2_enrich = np.log2((topic_word + eps) / (baseline[np.newaxis, :] + eps))

    resolved_sample, _ = _resolve_sample_slices(df, sample_col)
    replicates_df = compute_cn_replicates_df(
        updated_df,
        cn_col="Cellular_Neighborhood",
        cn_id_col="Cellular_Neighborhood_ID",
        sample_col=resolved_sample,
        cn_names=topic_names,
    )
    inter_cn_coord = compute_inter_cn_coordination(
        updated_df,
        cn_col="Cellular_Neighborhood_ID",
        sample_col=resolved_sample,
        microns_per_pixel=microns_per_pixel,
        n_permutations=100,
        random_state=random_state,
        cn_names=topic_names,
        adjacency=adj_delaunay,
        sample_slices=sample_slices,
    )

    diff_xi_l1 = float(np.sum(np.abs(diff_mat @ xi))) if (k_topics > 1 and diff_mat.shape[0] > 0) else 0.0

    return SpatialLDAResult(
        updated_df=updated_df,
        topic_probabilities=topic_probs,
        topic_word_matrix=topic_word,
        cell_types=cell_types,
        topic_entropy=entropy,
        difference_l1_norm=diff_l1,
        elbo_history=elbo_hist,
        metadata={
            "method": "Spatial-LDA",
            "n_cells": n_cells,
            "n_samples": len(sample_slices),
            "n_topics": k_topics,
            "n_edges": int(diff_mat.shape[0]),
            "difference_penalty": pen,
            "difference_xi_l1_norm": diff_xi_l1,
            "featurization_mode": "knn" if use_knn else "radius",
            "radius_um": float(radius_um),
            "k_neighbors": int(k_neighbors) if k_neighbors is not None else None,
            "microns_per_pixel": float(microns_per_pixel),
            "cn_names": topic_names,
            "baseline_proportions": baseline,
            "log2_enrichment": log2_enrich,
            "replicates_df": replicates_df,
            "inter_cn_coordination": inter_cn_coord,
        },
    )


def _cluster_window_profiles(
    profiles: np.ndarray,
    *,
    n_niches: int,
    cluster_method: str,
    resolution: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster normalized window composition profiles ``(N, V)`` into discrete CNs."""
    n_cells, n_vocab = profiles.shape
    if n_cells == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, n_vocab), dtype=np.float64)

    method_norm = str(cluster_method).strip().lower().replace("_", "").replace("-", "")
    if method_norm == "leiden":
        labels = _run_leiden_clustering(
            profiles, resolution=float(resolution), random_state=int(random_state)
        )
    elif method_norm == "kmeans":
        k_eff = max(1, min(int(n_niches), n_cells))
        km = KMeans(n_clusters=k_eff, random_state=int(random_state), n_init=10)
        labels = km.fit_predict(profiles).astype(np.int64)
    else:
        # Default Nolan Lab MiniBatchKMeans
        k_eff = max(1, min(int(n_niches), n_cells))
        batch_sz = min(1024, max(1, n_cells))
        mbk = MiniBatchKMeans(
            n_clusters=k_eff,
            random_state=int(random_state),
            n_init=5,
            batch_size=batch_sz,
        )
        labels = mbk.fit_predict(profiles).astype(np.int64)

    # Relabel cleanly 0..K_actual-1 preserving deterministic ordering
    unique_raw = sorted(np.unique(labels).tolist())
    remap = {old: new_id for new_id, old in enumerate(unique_raw)}
    relabeled = np.asarray([remap[int(v)] for v in labels], dtype=np.int64)
    n_clusters = len(unique_raw)
    centers = np.zeros((n_clusters, n_vocab), dtype=np.float64)
    for cid in range(n_clusters):
        mask = relabeled == cid
        if np.any(mask):
            centers[cid] = profiles[mask].mean(axis=0)
    return relabeled, centers


def _run_leiden_clustering(
    profiles: np.ndarray,
    *,
    resolution: float,
    random_state: int,
) -> np.ndarray:
    """Run Leiden clustering via igraph/leidenalg, scanpy, or deterministic fallback."""
    n_cells = profiles.shape[0]
    if n_cells <= 2:
        return np.zeros(n_cells, dtype=np.int64)

    k_graph = min(15, n_cells - 1)
    tree = cKDTree(profiles)
    dists, nn_idx = tree.query(profiles, k=k_graph + 1)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, np.newaxis]

    try:
        import igraph as ig
        import leidenalg as la

        edge_set: set[Tuple[int, int]] = set()
        for i in range(n_cells):
            for nb in nn_idx[i, 1:]:
                j = int(nb)
                if i != j:
                    edge_set.add((min(i, j), max(i, j)))
        graph = ig.Graph(n=n_cells, edges=sorted(edge_set), directed=False)
        partition = la.find_partition(
            graph,
            la.RBConfigurationVertexPartition,
            resolution_parameter=float(resolution),
            seed=int(random_state),
        )
        return np.asarray(partition.membership, dtype=np.int64)
    except ImportError:
        pass

    try:
        import anndata as ad
        import scanpy as sc

        adata = ad.AnnData(X=profiles.astype(np.float32))
        sc.pp.neighbors(adata, n_neighbors=max(2, min(15, n_cells - 1)), use_rep="X")
        sc.tl.leiden(
            adata,
            resolution=float(resolution),
            random_state=int(random_state),
            flavor="igraph",
            directed=False,
        )
        return adata.obs["leiden"].astype(int).to_numpy(dtype=np.int64)
    except Exception:
        pass

    # Deterministic fallback when graph community packages are unavailable
    est_k = max(2, min(n_cells, int(round(4.0 + 6.0 * max(float(resolution), 0.1)))))
    km = KMeans(n_clusters=est_k, random_state=int(random_state), n_init=5)
    return km.fit_predict(profiles).astype(np.int64)


def _compute_spatial_contexts(
    x_px: np.ndarray,
    y_px: np.ndarray,
    cn_ids: np.ndarray,
    cn_names: Sequence[str],
    sample_slices: Dict[str, np.ndarray],
    *,
    context_k: int = 40,
    context_threshold: float = 0.85,
    microns_per_pixel: float = 1.0,
) -> List[str]:
    """Compute Schuerch et al. minimal CN set reaching ``context_threshold`` in wider windows."""
    n_cells = len(cn_ids)
    n_cns = len(cn_names)
    if n_cells == 0 or n_cns == 0:
        return []

    win_counts = _build_window_count_matrix(
        x_px,
        y_px,
        cn_ids,
        n_cns,
        sample_slices,
        mode="knn",
        k_neighbors=max(1, int(context_k)),
        include_self=True,
        microns_per_pixel=microns_per_pixel,
    )
    row_sums = win_counts.sum(axis=1, keepdims=True)
    win_props = np.divide(
        win_counts,
        row_sums,
        out=np.zeros_like(win_counts),
        where=row_sums > 0,
    )

    thresh = float(np.clip(context_threshold, 1e-3, 1.0))
    contexts: List[str] = []
    for i in range(n_cells):
        props = win_props[i]
        if props.sum() <= 0.0:
            cid = int(cn_ids[i])
            contexts.append(cn_names[cid] if 0 <= cid < n_cns else "Unknown")
            continue
        order = np.argsort(-props, kind="mergesort")
        cum = 0.0
        selected_ids: List[int] = []
        for cid in order:
            p = float(props[cid])
            if p <= 0.0:
                break
            selected_ids.append(int(cid))
            cum += p
            if cum >= thresh - 1e-12:
                break
        if not selected_ids:
            selected_ids = [int(cn_ids[i])]
        selected_ids = sorted(set(selected_ids))
        contexts.append(" + ".join(cn_names[cid] for cid in selected_ids))
    return contexts


def run_moving_window_cn(
    df: pd.DataFrame,
    cell_type_col: str = "cell_type",
    sample_col: Optional[str] = None,
    window_mode: str = "knn",
    k_neighbors: int = 10,
    radius_um: float = 50.0,
    include_self: bool = True,
    n_niches: int = 10,
    cluster_method: str = "MiniBatchKMeans",
    resolution: float = 0.5,
    microns_per_pixel: float = 1.0,
    compute_coordination: bool = True,
    n_permutations: int = 200,
    compute_spatial_contexts: bool = False,
    context_k: int = 40,
    context_threshold: float = 0.85,
    random_state: int = 42,
) -> MovingWindowCNResult:
    """Identify Cellular Neighborhoods via moving-window cell-type compositions.

    Implements Nolan Lab's moving-window composition clustering (Schuerch et al.,
    *Cell* 2020), log2 fold enrichment over global tissue baseline, per-sample
    replicate abundance tables, Delaunay inter-CN coordination Z-scores, and
    wider-window Spatial Context labels.
    """
    resolved_ct = _resolve_cell_type_column(df, cell_type_col)
    if df.empty:
        out_df = df.copy()
        out_df["Cellular_Neighborhood"] = pd.Series(dtype=object)
        out_df["Cellular_Neighborhood_ID"] = pd.Series(dtype=np.int64)
        if compute_spatial_contexts:
            out_df["Spatial_Context"] = pd.Series(dtype=object)
        return MovingWindowCNResult(
            updated_df=out_df,
            centers=np.zeros((0, 0), dtype=np.float64),
            log2_enrichment=np.zeros((0, 0), dtype=np.float64),
            cell_types=[],
            replicates_df=compute_cn_replicates_df(out_df),
            inter_cn_coordination=pd.DataFrame() if compute_coordination else None,
            metadata={"method": "Moving Window", "n_cells": 0, "n_niches": 0},
            baseline_proportions=np.zeros(0, dtype=np.float64),
        )

    x_px, y_px = _resolve_xy_arrays(df)
    cell_types, codes = _extract_cell_types(df[resolved_ct])
    n_cells = len(df)
    n_vocab = len(cell_types)
    if n_vocab == 0:
        raise ValueError(f"No valid non-empty phenotype labels found in {resolved_ct!r}.")

    resolved_sample, sample_slices = _resolve_sample_slices(df, sample_col)
    win_counts = _build_window_count_matrix(
        x_px,
        y_px,
        codes,
        n_vocab,
        sample_slices,
        mode=window_mode,
        k_neighbors=int(k_neighbors),
        radius_um=float(radius_um),
        include_self=bool(include_self),
        microns_per_pixel=float(microns_per_pixel),
    )
    row_sums = win_counts.sum(axis=1, keepdims=True)
    win_props = np.divide(
        win_counts,
        row_sums,
        out=np.zeros_like(win_counts, dtype=np.float64),
        where=row_sums > 0,
    )

    cn_ids, centers = _cluster_window_profiles(
        win_props,
        n_niches=int(n_niches),
        cluster_method=cluster_method,
        resolution=float(resolution),
        random_state=int(random_state),
    )
    n_actual_cns = centers.shape[0]
    cn_names = [f"CN {i}" for i in range(n_actual_cns)]

    # Baseline proportions across valid typed cells
    valid_codes = codes[(codes >= 0) & (codes < n_vocab)]
    if valid_codes.size > 0:
        baseline = np.bincount(valid_codes, minlength=n_vocab).astype(np.float64)
        baseline /= float(valid_codes.size)
    else:
        baseline = np.full(n_vocab, 1.0 / float(n_vocab), dtype=np.float64)

    eps = 1e-4
    log2_enrichment = np.log2((centers + eps) / (baseline[np.newaxis, :] + eps))

    updated_df = df.copy()
    updated_df["Cellular_Neighborhood"] = [cn_names[int(cid)] for cid in cn_ids]
    updated_df["Cellular_Neighborhood_ID"] = cn_ids

    spatial_contexts_summary: Dict[str, int] = {}
    if compute_spatial_contexts:
        ctx_labels = _compute_spatial_contexts(
            x_px,
            y_px,
            cn_ids,
            cn_names,
            sample_slices,
            context_k=int(context_k),
            context_threshold=float(context_threshold),
            microns_per_pixel=float(microns_per_pixel),
        )
        updated_df["Spatial_Context"] = ctx_labels
        spatial_contexts_summary = pd.Series(ctx_labels).value_counts().to_dict()

    replicates_df = compute_cn_replicates_df(
        updated_df,
        cn_col="Cellular_Neighborhood",
        cn_id_col="Cellular_Neighborhood_ID",
        sample_col=resolved_sample,
        cn_names=cn_names,
    )

    inter_cn_coord: Optional[pd.DataFrame] = None
    if compute_coordination:
        inter_cn_coord = compute_inter_cn_coordination(
            updated_df,
            cn_col="Cellular_Neighborhood_ID",
            sample_col=resolved_sample,
            microns_per_pixel=float(microns_per_pixel),
            n_permutations=int(n_permutations),
            random_state=int(random_state),
            cn_names=cn_names,
        )

    return MovingWindowCNResult(
        updated_df=updated_df,
        centers=centers,
        log2_enrichment=log2_enrichment,
        cell_types=cell_types,
        replicates_df=replicates_df,
        inter_cn_coordination=inter_cn_coord,
        baseline_proportions=baseline,
        metadata={
            "method": "Moving Window",
            "n_cells": n_cells,
            "n_samples": len(sample_slices),
            "n_niches": n_actual_cns,
            "cn_names": cn_names,
            "window_mode": window_mode,
            "k_neighbors": int(k_neighbors),
            "radius_um": float(radius_um),
            "include_self": bool(include_self),
            "cluster_method": cluster_method,
            "resolution": float(resolution),
            "microns_per_pixel": float(microns_per_pixel),
            "baseline_proportions": baseline,
            "spatial_contexts_summary": spatial_contexts_summary,
        },
    )


__all__ = [
    "MovingWindowCNResult",
    "SpatialLDAResult",
    "compute_cn_cohort_stats",
    "compute_cn_replicates_df",
    "compute_inter_cn_coordination",
    "run_moving_window_cn",
    "run_spatial_lda",
]
