"""
IRIS Analyzer Spatial Graph Construction
========================================
Headless-safe per-sample spatial cell graph construction and edge-segment
extraction for downstream spatial proteomics analytics.

Supports:
- Voronoi-Delaunay triangulation (1-hop direct and exact/cumulative 2/3-hop rings)
- Unconstrained k-nearest neighbors (``knn``)
- Radius-constrained k-nearest neighbors (``knn_radius``)
- Fixed physical distance radius graphs (``radius``)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import Delaunay, QhullError, cKDTree

from iris_common.schema import (
    COL_SAMPLE,
    COL_X,
    COL_Y,
    X_COORD_ALIASES,
    Y_COORD_ALIASES,
    ensure_canonical_dataframe,
    find_sample_column,
)

VALID_GRAPH_MODES: Tuple[str, ...] = ("delaunay", "knn", "knn_radius", "radius")

_MODE_ALIASES: Dict[str, Tuple[str, Optional[int]]] = {
    "delaunay": ("delaunay", None),
    "voronoi": ("delaunay", None),
    "voronoi_delaunay": ("delaunay", None),
    "voronoi-delaunay": ("delaunay", None),
    "voronoi-delaunay 1-hop": ("delaunay", 1),
    "voronoi-delaunay 2-hop": ("delaunay", 2),
    "voronoi-delaunay 3-hop": ("delaunay", 3),
    "delaunay_1hop": ("delaunay", 1),
    "delaunay_2hop": ("delaunay", 2),
    "delaunay_3hop": ("delaunay", 3),
    "knn": ("knn", None),
    "knn (no radius cutoff)": ("knn", None),
    "knn_unconstrained": ("knn", None),
    "knn_radius": ("knn_radius", None),
    "knn + radius": ("knn_radius", None),
    "knn + radius (um)": ("knn_radius", None),
    "knn + radius (\u00b5m)": ("knn_radius", None),
    "knn_radius_um": ("knn_radius", None),
    "radius": ("radius", None),
    "fixed radius": ("radius", None),
    "fixed radius (um)": ("radius", None),
    "fixed radius (\u00b5m)": ("radius", None),
    "fixed_radius": ("radius", None),
}


@dataclass
class SpatialGraphResult:
    """Container holding a per-sample block-diagonal spatial adjacency graph."""

    adjacency: sp.csr_matrix
    n_cells: int
    mode: str
    params: Dict[str, Any]
    mean_degree: float
    edge_count: int
    sample_slices: Dict[str, np.ndarray]
    edge_lengths_um: Optional[np.ndarray] = None


def normalize_graph_mode(mode: str, hop: int = 1) -> Tuple[str, int]:
    """Normalize canonical or friendly UI graph mode strings into ``(mode, hop)``."""
    raw = str(mode).strip()
    key = raw.lower()
    if key in _MODE_ALIASES:
        norm_mode, implied_hop = _MODE_ALIASES[key]
        eff_hop = implied_hop if (implied_hop is not None and hop == 1) else int(hop)
        return norm_mode, eff_hop

    compact = key.replace("µ", "u").replace(" ", "_")
    if compact in _MODE_ALIASES:
        norm_mode, implied_hop = _MODE_ALIASES[compact]
        eff_hop = implied_hop if (implied_hop is not None and hop == 1) else int(hop)
        return norm_mode, eff_hop

    raise ValueError(
        f"Unsupported spatial graph mode {mode!r}. Expected one of {VALID_GRAPH_MODES}."
    )


def _resolve_coord_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Resolve X and Y coordinate column names without modifying *df*."""
    if COL_X in df.columns and COL_Y in df.columns:
        return COL_X, COL_Y

    lower_map = {str(c).strip().lower(): str(c) for c in df.columns}
    x_col: Optional[str] = None
    y_col: Optional[str] = None

    for alias in X_COORD_ALIASES:
        if alias in df.columns and pd.api.types.is_numeric_dtype(df[alias]):
            x_col = alias
            break
        low = alias.lower()
        if low in lower_map and pd.api.types.is_numeric_dtype(df[lower_map[low]]):
            x_col = lower_map[low]
            break

    for alias in Y_COORD_ALIASES:
        if alias in df.columns and pd.api.types.is_numeric_dtype(df[alias]):
            y_col = alias
            break
        low = alias.lower()
        if low in lower_map and pd.api.types.is_numeric_dtype(df[lower_map[low]]):
            y_col = lower_map[low]
            break

    if x_col is None or y_col is None:
        canon = ensure_canonical_dataframe(df, inplace=False, require_coordinates=True)
        if COL_X in canon.columns and COL_Y in canon.columns:
            return COL_X, COL_Y
        raise ValueError(
            f"DataFrame missing spatial coordinate columns ({COL_X!r}, {COL_Y!r})."
        )
    return x_col, y_col


def _extract_xy_pixels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(x_px, y_px)`` float64 arrays from *df*."""
    if COL_X in df.columns and COL_Y in df.columns:
        x_col, y_col = COL_X, COL_Y
    else:
        x_col, y_col = _resolve_coord_columns(df)
    x_arr = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=np.float64)
    y_arr = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float64)
    return x_arr, y_arr


def _build_delaunay_pairs(
    coords_um: np.ndarray,
    max_edge_um: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct undirected 1-hop Delaunay pairs on valid 2D coordinates."""
    n = len(coords_um)
    if n < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    if n == 2:
        u = np.array([0], dtype=np.int64)
        v = np.array([1], dtype=np.int64)
    else:
        try:
            tri = Delaunay(coords_um)
            simplices = tri.simplices
            edges = np.vstack(
                [
                    simplices[:, [0, 1]],
                    simplices[:, [1, 2]],
                    simplices[:, [0, 2]],
                ]
            )
            edges.sort(axis=1)
            unique_edges = np.unique(edges, axis=0)
            u = unique_edges[:, 0].astype(np.int64, copy=False)
            v = unique_edges[:, 1].astype(np.int64, copy=False)
        except QhullError:
            u, v = _fallback_degenerate_delaunay_pairs(coords_um)

    if u.size == 0:
        return u, v

    if max_edge_um is not None:
        limit = float(max_edge_um)
        lengths = np.hypot(
            coords_um[u, 0] - coords_um[v, 0],
            coords_um[u, 1] - coords_um[v, 1],
        )
        keep = lengths <= limit
        u = u[keep]
        v = v[keep]

    return u, v


def _fallback_degenerate_delaunay_pairs(
    coords_um: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Handle collinear, coincident, or low-rank coordinates on Qhull failure."""
    n = len(coords_um)
    if n < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    centered = coords_um - np.mean(coords_um, axis=0, keepdims=True)
    _, singular_vals, vt = np.linalg.svd(centered, full_matrices=False)
    if singular_vals.size > 0 and singular_vals[0] > 1e-12:
        proj = centered @ vt[0]
        order = np.argsort(proj, kind="mergesort")
        pairs: List[Tuple[int, int]] = []
        for idx in range(n - 1):
            a, b = int(order[idx]), int(order[idx + 1])
            if a != b:
                pairs.append((min(a, b), max(a, b)))
        # Also connect any coincident points that share identical coordinates.
        i = 0
        while i < n:
            j = i + 1
            while j < n and abs(proj[order[j]] - proj[order[i]]) <= 1e-12:
                for p in range(i, j):
                    a, b = int(order[p]), int(order[j])
                    if a != b:
                        pairs.append((min(a, b), max(a, b)))
                j += 1
            i = j
        if not pairs:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty
        arr = np.unique(np.asarray(pairs, dtype=np.int64), axis=0)
        return arr[:, 0], arr[:, 1]

    # All coordinates are identical: connect a compact small graph or 2-NN ring.
    if n <= 4:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = [(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)]
    arr = np.unique(np.asarray(pairs, dtype=np.int64), axis=0)
    return arr[:, 0], arr[:, 1]


def _pairs_to_symmetric_csr(
    n: int,
    u: np.ndarray,
    v: np.ndarray,
) -> sp.csr_matrix:
    """Build a symmetric zero-diagonal boolean CSR matrix from undirected pairs ``(u, v)``."""
    if n == 0 or u.size == 0:
        return sp.csr_matrix((n, n), dtype=bool)
    mask = u != v
    u = u[mask]
    v = v[mask]
    if u.size == 0:
        return sp.csr_matrix((n, n), dtype=bool)
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    data = np.ones(rows.size, dtype=bool)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=bool)
    adj.sum_duplicates()
    adj.setdiag(False)
    adj.eliminate_zeros()
    return adj


def _expand_delaunay_hops(
    adj1: sp.csr_matrix,
    hop: int,
    exact_ring: bool,
) -> sp.csr_matrix:
    """Compute exact or cumulative geodesic hop neighborhoods from 1-hop ``adj1``."""
    n = adj1.shape[0]
    if n == 0:
        return adj1.copy()
    if hop == 1:
        out = ((adj1 + adj1.T) > 0).astype(bool)
        out.setdiag(False)
        out.eliminate_zeros()
        return out

    ident = sp.eye(n, dtype=bool, format="csr")
    reach_1 = (adj1 + ident).astype(bool)
    prev_reach = reach_1
    cur_reach = reach_1
    for _ in range(2, hop + 1):
        prev_reach = cur_reach
        cur_reach = (cur_reach @ reach_1).astype(bool)

    if exact_ring:
        ring = cur_reach.astype(np.int8) - prev_reach.astype(np.int8)
    else:
        ring = cur_reach.astype(np.int8) - ident.astype(np.int8)

    ring.eliminate_zeros()
    out = ring.astype(bool)
    out = (out + out.T).astype(bool)
    out.setdiag(False)
    out.eliminate_zeros()
    return out


def _build_knn_csr(
    coords_um: np.ndarray,
    k: int,
    radius_um: Optional[float],
    symmetric: bool,
) -> sp.csr_matrix:
    """Build unconstrained ``knn`` or ``knn_radius`` adjacency CSR matrix."""
    n = len(coords_um)
    k_eff = min(max(0, int(k)), max(0, n - 1))
    if n < 2 or k_eff == 0:
        return sp.csr_matrix((n, n), dtype=bool)
    if radius_um is not None and float(radius_um) <= 0.0:
        return sp.csr_matrix((n, n), dtype=bool)

    tree = cKDTree(coords_um)
    query_k = min(n, k_eff + 1)
    query_kwargs: Dict[str, Any] = {"k": query_k}
    if radius_um is not None:
        query_kwargs["distance_upper_bound"] = float(radius_um)

    dists, indices = tree.query(coords_um, **query_kwargs)
    dists = np.atleast_2d(dists)
    indices = np.atleast_2d(indices)
    if dists.shape[0] != n:
        dists = dists.reshape(n, query_k)
        indices = indices.reshape(n, query_k)

    rows_list: List[np.ndarray] = []
    cols_list: List[np.ndarray] = []

    row_arange = np.arange(n, dtype=np.int64)[:, None]
    valid = (indices < n) & (indices != row_arange) & np.isfinite(dists)
    if radius_um is not None:
        valid &= dists <= float(radius_um)

    # Keep at most k_eff non-self neighbors per row (handles coincident points).
    for row_i in range(n):
        nbrs = indices[row_i][valid[row_i]]
        if nbrs.size > k_eff:
            nbrs = nbrs[:k_eff]
        if nbrs.size > 0:
            rows_list.append(np.full(nbrs.size, row_i, dtype=np.int64))
            cols_list.append(nbrs.astype(np.int64, copy=False))

    if not rows_list:
        return sp.csr_matrix((n, n), dtype=bool)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    adj = sp.csr_matrix((np.ones(rows.size, dtype=bool), (rows, cols)), shape=(n, n), dtype=bool)
    adj.sum_duplicates()
    adj.setdiag(False)
    adj.eliminate_zeros()

    if symmetric:
        adj = (adj + adj.T).astype(bool)
        adj.setdiag(False)
        adj.eliminate_zeros()
    return adj


def _build_radius_csr(coords_um: np.ndarray, radius_um: float) -> sp.csr_matrix:
    """Build symmetric zero-diagonal adjacency connecting all pairs within ``radius_um``."""
    n = len(coords_um)
    if n < 2 or float(radius_um) <= 0.0:
        return sp.csr_matrix((n, n), dtype=bool)

    tree = cKDTree(coords_um)
    pairs = tree.query_pairs(r=float(radius_um), output_type="ndarray")
    if pairs.size == 0:
        return sp.csr_matrix((n, n), dtype=bool)
    return _pairs_to_symmetric_csr(n, pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64))


def _build_sample_subgraph(
    coords_um: np.ndarray,
    mode: str,
    hop: int,
    exact_ring: bool,
    k: int,
    radius_um: float,
    max_edge_um: Optional[float],
    symmetric: bool,
) -> sp.csr_matrix:
    """Construct adjacency matrix for one sample's coordinates (in um)."""
    n = len(coords_um)
    if n < 2:
        return sp.csr_matrix((n, n), dtype=bool)

    valid_mask = np.isfinite(coords_um).all(axis=1)
    if not np.any(valid_mask):
        return sp.csr_matrix((n, n), dtype=bool)

    valid_idx = np.flatnonzero(valid_mask)
    valid_coords = coords_um[valid_idx]

    if mode == "delaunay":
        u, v = _build_delaunay_pairs(valid_coords, max_edge_um=max_edge_um)
        adj1 = _pairs_to_symmetric_csr(len(valid_idx), u, v)
        sub_valid = _expand_delaunay_hops(adj1, hop=hop, exact_ring=exact_ring)
    elif mode == "knn":
        sub_valid = _build_knn_csr(
            valid_coords,
            k=k,
            radius_um=None,
            symmetric=symmetric,
        )
    elif mode == "knn_radius":
        sub_valid = _build_knn_csr(
            valid_coords,
            k=k,
            radius_um=radius_um,
            symmetric=symmetric,
        )
    elif mode == "radius":
        sub_valid = _build_radius_csr(valid_coords, radius_um=radius_um)
    else:
        raise ValueError(f"Unsupported graph mode: {mode!r}")

    if len(valid_idx) == n:
        return sub_valid

    vr, vc = sub_valid.nonzero()
    if vr.size == 0:
        return sp.csr_matrix((n, n), dtype=bool)
    global_r = valid_idx[vr]
    global_c = valid_idx[vc]
    out = sp.csr_matrix(
        (np.ones(global_r.size, dtype=bool), (global_r, global_c)),
        shape=(n, n),
        dtype=bool,
    )
    out.setdiag(False)
    out.eliminate_zeros()
    return out


def build_spatial_graph(
    df: pd.DataFrame,
    mode: str = "delaunay",
    hop: int = 1,
    exact_ring: bool = True,
    k: int = 10,
    radius_um: float = 50.0,
    max_edge_um: Optional[float] = None,
    microns_per_pixel: float = 1.0,
    sample_col: Optional[str] = None,
    symmetric: bool = True,
) -> SpatialGraphResult:
    """Build a strictly per-sample block-diagonal spatial cell graph.

    Parameters
    ----------
    df:
        Single-cell DataFrame containing spatial coordinates (``Pos_X``, ``Pos_Y`` or aliases).
    mode:
        One of ``"delaunay"``, ``"knn"``, ``"knn_radius"``, or ``"radius"``
        (friendly UI mode labels are also accepted).
    hop:
        Topological hop distance for ``"delaunay"`` mode (e.g., ``1``, ``2``, ``3``).
    exact_ring:
        When ``True`` and ``hop >= 2``, returns exact geodesic ring ``d_G == hop``;
        otherwise returns cumulative neighborhood ``1 <= d_G <= hop``.
    k:
        Number of neighbors for ``"knn"`` and ``"knn_radius"`` modes.
    radius_um:
        Physical distance cutoff in micrometers for ``"knn_radius"`` and ``"radius"``.
    max_edge_um:
        Optional maximum chord length (in micrometers) retained on 1-hop Delaunay edges
        prior to hop expansion.
    microns_per_pixel:
        Coordinate scale factor converting pixels to micrometers.
    sample_col:
        Optional column identifying biological samples/FOVs. Auto-detected via
        ``iris_common.schema.find_sample_column`` when ``None``.
    symmetric:
        Whether to symmetrize ``knn`` and ``knn_radius`` graphs. (``delaunay`` and
        ``radius`` are always symmetric.)
    """
    if not np.isfinite(microns_per_pixel) or float(microns_per_pixel) <= 0.0:
        raise ValueError(
            f"microns_per_pixel must be positive and finite, got {microns_per_pixel!r}."
        )

    norm_mode, eff_hop = normalize_graph_mode(mode, hop=hop)
    if eff_hop < 1:
        raise ValueError(f"hop must be >= 1, got {eff_hop!r}.")
    if max_edge_um is not None and float(max_edge_um) <= 0.0:
        max_edge_um = None

    n_cells = len(df)
    if sample_col is not None:
        if sample_col not in df.columns:
            raise KeyError(f"Sample column {sample_col!r} not found in DataFrame.")
        resolved_sample_col: Optional[str] = sample_col
    else:
        resolved_sample_col = find_sample_column(df)

    if n_cells == 0:
        return SpatialGraphResult(
            adjacency=sp.csr_matrix((0, 0), dtype=bool),
            n_cells=0,
            mode=norm_mode,
            params={
                "mode": norm_mode,
                "hop": eff_hop,
                "exact_ring": bool(exact_ring),
                "k": int(k),
                "radius_um": float(radius_um),
                "max_edge_um": float(max_edge_um) if max_edge_um is not None else None,
                "microns_per_pixel": float(microns_per_pixel),
                "sample_col": resolved_sample_col,
                "symmetric": bool(symmetric or norm_mode in ("delaunay", "radius")),
            },
            mean_degree=0.0,
            edge_count=0,
            sample_slices={},
            edge_lengths_um=np.empty(0, dtype=np.float64),
        )

    x_px, y_px = _extract_xy_pixels(df)
    scale = float(microns_per_pixel)
    coords_um_all = np.column_stack([x_px * scale, y_px * scale])

    sample_slices: Dict[str, np.ndarray] = {}
    if resolved_sample_col is not None:
        sample_series = df[resolved_sample_col].astype(str)
        # Preserve first-seen sample order cleanly across Python/pandas versions.
        seen_samples = list(dict.fromkeys(sample_series.tolist()))
        raw_vals = sample_series.to_numpy()
        for s_name in seen_samples:
            sample_slices[s_name] = np.flatnonzero(raw_vals == s_name).astype(np.int64)
    else:
        sample_slices[COL_SAMPLE] = np.arange(n_cells, dtype=np.int64)

    eff_symmetric = bool(symmetric or norm_mode in ("delaunay", "radius"))
    global_rows: List[np.ndarray] = []
    global_cols: List[np.ndarray] = []

    for s_idx in sample_slices.values():
        if s_idx.size < 2:
            continue
        sub_adj = _build_sample_subgraph(
            coords_um_all[s_idx],
            mode=norm_mode,
            hop=eff_hop,
            exact_ring=bool(exact_ring),
            k=int(k),
            radius_um=float(radius_um),
            max_edge_um=max_edge_um,
            symmetric=eff_symmetric,
        )
        sr, sc = sub_adj.nonzero()
        if sr.size > 0:
            global_rows.append(s_idx[sr])
            global_cols.append(s_idx[sc])

    if global_rows:
        r_all = np.concatenate(global_rows)
        c_all = np.concatenate(global_cols)
        adjacency = sp.csr_matrix(
            (np.ones(r_all.size, dtype=bool), (r_all, c_all)),
            shape=(n_cells, n_cells),
            dtype=bool,
        )
        adjacency.sum_duplicates()
        adjacency.setdiag(False)
        adjacency.eliminate_zeros()
    else:
        adjacency = sp.csr_matrix((n_cells, n_cells), dtype=bool)

    if eff_symmetric:
        edge_r, edge_c = sp.triu(adjacency, k=1).nonzero()
        edge_count = int(edge_r.size)
    else:
        edge_r, edge_c = adjacency.nonzero()
        edge_count = int(edge_r.size)

    if edge_count > 0:
        edge_lengths_um = np.hypot(
            coords_um_all[edge_r, 0] - coords_um_all[edge_c, 0],
            coords_um_all[edge_r, 1] - coords_um_all[edge_c, 1],
        ).astype(np.float64, copy=False)
    else:
        edge_lengths_um = np.empty(0, dtype=np.float64)

    mean_degree = float(adjacency.nnz / n_cells) if n_cells > 0 else 0.0

    params = {
        "mode": norm_mode,
        "hop": eff_hop,
        "exact_ring": bool(exact_ring),
        "k": int(k),
        "radius_um": float(radius_um),
        "max_edge_um": float(max_edge_um) if max_edge_um is not None else None,
        "microns_per_pixel": scale,
        "sample_col": resolved_sample_col,
        "symmetric": eff_symmetric,
    }

    return SpatialGraphResult(
        adjacency=adjacency,
        n_cells=n_cells,
        mode=norm_mode,
        params=params,
        mean_degree=mean_degree,
        edge_count=edge_count,
        sample_slices=sample_slices,
        edge_lengths_um=edge_lengths_um,
    )


def extract_pair_edges_yx(
    df: pd.DataFrame,
    adjacency: sp.spmatrix,
    source_mask: np.ndarray | pd.Series | Sequence[bool] | Iterable[bool],
    target_mask: np.ndarray | pd.Series | Sequence[bool] | Iterable[bool],
    max_edges: Optional[int] = 5000,
) -> np.ndarray:
    """Extract ``(E, 2, 2)`` line segments ``[[y0, x0], [y1, x1]]`` for Napari Shapes.

    Selects edges ``(u, v)`` in ``adjacency`` where ``source_mask[u]`` and
    ``target_mask[v]`` are True, deduplicating symmetric reverse segments so
    each rendered segment appears once.
    """
    n_cells = len(df)
    if adjacency.shape != (n_cells, n_cells):
        raise ValueError(
            f"Adjacency shape {adjacency.shape} does not match DataFrame length {n_cells}."
        )
    if n_cells == 0:
        return np.empty((0, 2, 2), dtype=np.float64)

    src = np.asarray(list(source_mask) if not hasattr(source_mask, "__len__") else source_mask,
                     dtype=bool).ravel()
    tgt = np.asarray(list(target_mask) if not hasattr(target_mask, "__len__") else target_mask,
                     dtype=bool).ravel()
    if src.size != n_cells or tgt.size != n_cells:
        raise ValueError(
            f"Mask lengths ({src.size}, {tgt.size}) must match DataFrame length {n_cells}."
        )

    x_px, y_px = _extract_xy_pixels(df)
    u, v = adjacency.nonzero()
    if u.size == 0:
        return np.empty((0, 2, 2), dtype=np.float64)

    keep = (u != v) & src[u] & tgt[v]
    if not np.any(keep):
        return np.empty((0, 2, 2), dtype=np.float64)

    u = u[keep]
    v = v[keep]

    finite_keep = (
        np.isfinite(x_px[u]) & np.isfinite(y_px[u])
        & np.isfinite(x_px[v]) & np.isfinite(y_px[v])
    )
    u = u[finite_keep]
    v = v[finite_keep]
    if u.size == 0:
        return np.empty((0, 2, 2), dtype=np.float64)

    # Deduplicate unordered node pairs {u, v} while preserving first source->target order.
    pair_keys = np.column_stack([np.minimum(u, v), np.maximum(u, v)])
    _, first_idx = np.unique(pair_keys, axis=0, return_index=True)
    first_idx.sort()
    u = u[first_idx]
    v = v[first_idx]

    if max_edges is not None and int(max_edges) > 0 and u.size > int(max_edges):
        sel = np.linspace(0, u.size - 1, int(max_edges), dtype=np.int64)
        u = u[sel]
        v = v[sel]

    segments = np.empty((u.size, 2, 2), dtype=np.float64)
    segments[:, 0, 0] = y_px[u]
    segments[:, 0, 1] = x_px[u]
    segments[:, 1, 0] = y_px[v]
    segments[:, 1, 1] = x_px[v]
    return segments


__all__ = [
    "SpatialGraphResult",
    "VALID_GRAPH_MODES",
    "build_spatial_graph",
    "extract_pair_edges_yx",
    "normalize_graph_mode",
]
