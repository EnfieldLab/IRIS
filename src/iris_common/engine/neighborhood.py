"""
IRIS Common Cellular Neighborhoods Engine
=========================================
Identifies and characterizes local spatial microenvironments (cellular niches)
via k-nearest neighbor graph extraction and clustering (K-Means or Leiden).
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans


def identify_cellular_neighborhoods(
    df: pd.DataFrame,
    cell_type_col: str,
    k_neighbors: int = 5,
    n_niches: int = 5,
    microns_per_pixel: float = 1.0,
    method: str = "KMeans",
    resolution: float = 1.0,
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Computes local cell-type composition profiles and clusters them into spatial niches.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing cell coordinates ('Pos_X', 'Pos_Y') and cell phenotypes.
    cell_type_col : str
        Column name holding cell phenotype designations.
    k_neighbors : int
        Number of spatial nearest neighbors to consider (default 5).
    n_niches : int
        Target number of clusters for KMeans (default 5).
    microns_per_pixel : float
        Physical pixel scale factor (default 1.0).
    method : str
        Clustering algorithm ('KMeans' or 'Leiden').
    resolution : float
        Resolution parameter for Leiden clustering (default 1.0).
    n_jobs : int
        Parallel worker count for KD-Tree search (default -1).

    Returns
    -------
    updated_df : pd.DataFrame
        Input DataFrame augmented with 'Cellular_Neighborhood' cluster labels.
    centers : np.ndarray
        (N_niches, N_cell_types) matrix of cell-type proportions per niche.
    unique_types : List[str]
        Sorted list of cell types corresponding to columns of `centers`.
    """
    if df.empty or cell_type_col not in df.columns:
        return df.copy(), np.zeros((1, 1)), []

    x_col = next((c for c in ['Pos_X', 'Centroid_X', 'x', 'X'] if c in df.columns), None)
    y_col = next((c for c in ['Pos_Y', 'Centroid_Y', 'y', 'Y'] if c in df.columns), None)
    if x_col is None or y_col is None:
        raise ValueError("DataFrame missing spatial coordinate columns ('Pos_X', 'Pos_Y').")

    if len(df) <= k_neighbors:
        updated_df = df.copy()
        updated_df['Cellular_Neighborhood'] = 0
        return updated_df, np.zeros((1, 1)), []

    unique_types = sorted([str(t) for t in df[cell_type_col].dropna().unique() if str(t).strip()])
    if not unique_types:
        return df.copy(), np.zeros((1, 1)), []

    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    ntypes = len(unique_types)

    images = df['Image'].unique() if 'Image' in df.columns else ['Sample']

    all_profiles = []
    df_indices = []

    for img in images:
        img_mask = (df['Image'] == img) if 'Image' in df.columns else np.ones(len(df), dtype=bool)
        sub_df = df[img_mask]
        if sub_df.empty:
            continue

        sub_coords = sub_df[[x_col, y_col]].values * microns_per_pixel

        if len(sub_df) <= k_neighbors:
            all_profiles.append(np.zeros((len(sub_df), ntypes)))
            df_indices.extend(sub_df.index)
            continue

        tree = cKDTree(sub_coords)
        _, neighbor_idxs = tree.query(sub_coords, k=k_neighbors + 1, workers=n_jobs)
        neighbor_idxs = neighbor_idxs[:, 1:]

        sub_profiles = np.zeros((len(sub_df), ntypes))
        phenotypes = sub_df[cell_type_col].astype(str).values
        for i in range(len(sub_df)):
            n_types_local = phenotypes[neighbor_idxs[i]]
            for t in n_types_local:
                if t in type_to_idx:
                    sub_profiles[i, type_to_idx[t]] += 1

        row_sums = sub_profiles.sum(axis=1, keepdims=True)
        sub_profiles = np.divide(
            sub_profiles, row_sums, out=np.zeros_like(sub_profiles), where=row_sums != 0
        )

        all_profiles.append(sub_profiles)
        df_indices.extend(sub_df.index)

    if not all_profiles:
        return df.copy(), np.zeros((1, 1)), []

    composition_profiles = np.vstack(all_profiles)

    # Realign profiles to original DataFrame index order
    sort_idx = np.argsort(df_indices)
    composition_profiles = composition_profiles[sort_idx]

    if method == "Leiden":
        try:
            import igraph as ig
            import leidenalg as la
        except ImportError:
            raise ImportError(
                "Leiden clustering requires 'igraph' and 'leidenalg' packages. "
                "Install them or switch to KMeans."
            )

        profile_tree = cKDTree(composition_profiles)
        k_prof = min(15, len(composition_profiles))
        _, prof_nn = profile_tree.query(composition_profiles, k=k_prof, workers=n_jobs)
        edges = []
        for i in range(len(composition_profiles)):
            for nb in prof_nn[i, 1:]:
                edges.append((i, nb))
        g = ig.Graph(n=len(composition_profiles), edges=edges, directed=False)
        g.simplify()
        partition = la.find_partition(
            g, la.RBConfigurationVertexPartition, resolution_parameter=resolution, seed=42
        )
        niche_labels = np.array(partition.membership)

        unique_labels = sorted(np.unique(niche_labels))
        computed_centers = []
        for lbl in unique_labels:
            lbl_mask = (niche_labels == lbl)
            computed_centers.append(composition_profiles[lbl_mask].mean(axis=0))
        centers = np.array(computed_centers)
    else:
        # MiniBatchKMeans default
        actual_clusters = min(n_niches, len(composition_profiles))
        actual_clusters = max(1, actual_clusters)
        batch_sz = min(1024, len(composition_profiles))
        kmeans = MiniBatchKMeans(
            n_clusters=actual_clusters,
            random_state=42,
            n_init=5,
            batch_size=batch_sz,
        )
        niche_labels = kmeans.fit_predict(composition_profiles)
        centers = kmeans.cluster_centers_

    updated_df = df.copy()
    updated_df['Cellular_Neighborhood'] = niche_labels

    return updated_df, centers, unique_types
