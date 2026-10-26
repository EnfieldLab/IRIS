"""
IRIS-Viewer Cellular Neighborhoods Engine

Identifies and characterizes local spatial microenvironments (cellular niches)
via k-nearest neighbor graph extraction and clustering (K-Means or Leiden).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans

from iris_viewer.engine.geometry import ensure_spatial_coordinates


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
    """Compute local cell-type composition profiles and cluster them into spatial niches.

    Args:
        df: Table containing spatial cell coordinates (Pos_X, Pos_Y) and cell phenotypes.
        cell_type_col: Column name containing cell phenotype labels.
        k_neighbors: Number of spatial nearest neighbors for local window profiling.
        n_niches: Target number of spatial microenvironment clusters (niches).
        microns_per_pixel: Physical pixel scale factor (µm/px).
        method: Clustering algorithm ('KMeans' or 'Leiden').
        resolution: Resolution parameter for Leiden graph partitioning.
        n_jobs: Number of parallel CPU workers for KD-Tree nearest neighbor queries.

    Returns:
        Tuple of:
            - updated_df: DataFrame augmented with 'Cellular_Neighborhood' and 'Cellular_Neighborhood_ID'.
            - centers: (N_niches, N_cell_types) matrix of mean phenotype proportions per niche.
            - unique_types: Sorted list of cell types corresponding to columns of centers.

    Raises:
        ValueError: If spatial coordinate columns are missing.
        ImportError: If Leiden clustering is requested but igraph/leidenalg are not installed.
    """
    if df.empty or cell_type_col not in df.columns:
        return df.copy(), np.zeros((1, 1)), []

    ensure_spatial_coordinates(df)
    x_col, y_col = "Pos_X", "Pos_Y"

    if len(df) <= k_neighbors:
        updated_df = df.copy()
        updated_df['Cellular_Neighborhood'] = 0
        return updated_df, np.zeros((1, 1)), []

    unique_types = sorted([str(t) for t in df[cell_type_col].dropna().unique() if str(t).strip()])
    if not unique_types: return df.copy(), np.zeros((1, 1)), []

    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    ntypes = len(unique_types)
    images = df['Image'].unique() if 'Image' in df.columns else ['Sample']

    row_positions = np.arange(len(df))
    all_profiles = []
    processed_positions = []

    for img in images:
        img_mask = (df['Image'] == img) if 'Image' in df.columns else np.ones(len(df), dtype=bool)
        sub_df, sub_pos = df[img_mask], row_positions[img_mask]
        if sub_df.empty: continue

        sub_coords = sub_df[[x_col, y_col]].values * microns_per_pixel
        valid_coords = ~np.isnan(sub_coords).any(axis=1) & ~np.isinf(sub_coords).any(axis=1)

        if len(sub_df) <= k_neighbors or not np.any(valid_coords):
            all_profiles.append(np.zeros((len(sub_df), ntypes)))
            processed_positions.extend(sub_pos)
            continue

        if not np.all(valid_coords):
            valid_tree_coords = sub_coords[valid_coords]
            if len(valid_tree_coords) <= k_neighbors:
                all_profiles.append(np.zeros((len(sub_df), ntypes)))
                processed_positions.extend(sub_pos)
                continue
            tree = cKDTree(valid_tree_coords)
            safe_coords = np.where(np.isnan(sub_coords) | np.isinf(sub_coords), 0.0, sub_coords)
            _, neighbor_idxs = tree.query(safe_coords, k=k_neighbors + 1, workers=n_jobs)
            valid_pheno = sub_df.loc[sub_df.index[valid_coords], cell_type_col].astype(str).values
            neighbor_idxs = neighbor_idxs[:, 1:]
            sub_profiles = np.zeros((len(sub_df), ntypes))
            for i in range(len(sub_df)):
                if not valid_coords[i]: continue
                n_types_local = valid_pheno[neighbor_idxs[i]]
                for t in n_types_local:
                    if t in type_to_idx: sub_profiles[i, type_to_idx[t]] += 1
        else:
            tree = cKDTree(sub_coords)
            _, neighbor_idxs = tree.query(sub_coords, k=k_neighbors + 1, workers=n_jobs)
            neighbor_idxs = neighbor_idxs[:, 1:]
            sub_profiles = np.zeros((len(sub_df), ntypes))
            phenotypes = sub_df[cell_type_col].astype(str).values
            for i in range(len(sub_df)):
                n_types_local = phenotypes[neighbor_idxs[i]]
                for t in n_types_local:
                    if t in type_to_idx: sub_profiles[i, type_to_idx[t]] += 1

        row_sums = sub_profiles.sum(axis=1, keepdims=True)
        sub_profiles = np.divide(sub_profiles, row_sums, out=np.zeros_like(sub_profiles), where=row_sums != 0)
        all_profiles.append(sub_profiles)
        processed_positions.extend(sub_pos)

    if not all_profiles: return df.copy(), np.zeros((1, 1)), []

    composition_profiles = np.vstack(all_profiles)
    processed_positions = np.array(processed_positions)

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
            for nb in prof_nn[i, 1:]: edges.append((i, nb))
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
        actual_clusters = max(1, min(n_niches, len(composition_profiles)))
        batch_sz = min(1024, len(composition_profiles))
        kmeans = MiniBatchKMeans(
            n_clusters=actual_clusters,
            random_state=42,
            n_init=5,
            batch_size=batch_sz,
        )
        niche_labels = kmeans.fit_predict(composition_profiles)
        centers = kmeans.cluster_centers_

    final_labels = np.zeros(len(df), dtype=int)
    final_labels[processed_positions] = niche_labels

    updated_df = df.copy()
    updated_df['Cellular_Neighborhood'] = [f"Niche {lbl}" for lbl in final_labels]
    updated_df['Cellular_Neighborhood_ID'] = final_labels
    return updated_df, centers, unique_types
