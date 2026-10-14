"""
IRIS-Viewer Consensus Engine

Generates an ensemble consensus segmentation mask from multiple segmentation
method outputs using spatial proximity, bounding box slicing, and directional
overlap agreement.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table


def _extract_cell_centroids(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract cell IDs, centroids, and bounding boxes from a label mask.

    Args:
        mask: 2D integer array containing segmented object labels.

    Returns:
        Tuple of (cell_ids, centroids, bounding_boxes).
    """
    props = regionprops_table(mask, properties=('label', 'centroid', 'bbox'))
    if len(props['label']) == 0:
        return np.array([], dtype=np.uint32), np.zeros((0, 2), dtype=np.float64), np.zeros((0, 4), dtype=int)
        
    cell_ids = np.array(props['label'], dtype=np.uint32)
    centroids = np.column_stack([props['centroid-0'], props['centroid-1']])
    bboxes = np.column_stack([props['bbox-0'], props['bbox-1'], props['bbox-2'], props['bbox-3']])
    return cell_ids, centroids, bboxes


def _compute_directional_overlaps(
    mask_a: np.ndarray,
    id_a: int,
    mask_b: np.ndarray,
    id_b: int,
    bbox: np.ndarray,
    padding: int = 15,
) -> Tuple[float, float]:
    """Compute directional overlap ratios between two cell masks within a bounding box.

    Args:
        mask_a: First segmentation label mask.
        id_a: Target label ID in mask_a.
        mask_b: Second segmentation label mask.
        id_b: Target label ID in mask_b.
        bbox: Bounding box coordinates [min_row, min_col, max_row, max_col].
        padding: Pixel padding applied around the bounding box.

    Returns:
        Tuple of (overlap_a_in_b, overlap_b_in_a).
    """
    minr, minc, maxr, maxc = bbox
    minr, maxr = max(0, minr - padding), min(mask_a.shape[0], maxr + padding)
    minc, maxc = max(0, minc - padding), min(mask_a.shape[1], maxc + padding)
    
    region_a = (mask_a[minr:maxr, minc:maxc] == id_a)
    region_b = (mask_b[minr:maxr, minc:maxc] == id_b)
    
    intersection = float(np.sum(region_a & region_b))
    area_a = float(np.sum(region_a))
    area_b = float(np.sum(region_b))
    
    overlap_a_in_b = intersection / area_a if area_a > 0 else 0.0
    overlap_b_in_a = intersection / area_b if area_b > 0 else 0.0
    return overlap_a_in_b, overlap_b_in_a


def compute_consensus_mask(
    results_dict: Dict[str, Dict[str, Optional[np.ndarray]]],
    min_overlap_iou: float = 0.3,
    centroid_distance_threshold: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, float], Dict[int, Dict[str, int]]]:
    """Compute ensemble consensus mask, confidence map, and cross-method mappings.

    Combines multiple segmentation outputs into a unified consensus representation.
    The method yielding the highest cell count is selected as the initial reference.
    Spatial nearest neighbors and directional overlaps are computed using KD-trees
    and bounding box slices to score inter-method agreement.

    Args:
        results_dict: Dictionary mapping method name to segmentation masks
            dict containing 'nuclear' and/or 'cytoplasm' arrays.
        min_overlap_iou: Minimum overlap threshold for candidate association.
        centroid_distance_threshold: Maximum distance in pixels between centroids
            to consider candidate cell matches across methods.

    Returns:
        Tuple of:
            - consensus_mask: (H, W) uint32 label array.
            - confidence_map: (H, W) float32 array scaled [0.0, 1.0].
            - per_cell_confidence: Dict mapping consensus cell ID to confidence score.
            - cell_method_map: Dict mapping consensus cell ID to per-method original IDs.

    Raises:
        ValueError: If results_dict is empty.
    """
    method_names = list(results_dict.keys())
    n_methods = len(method_names)
    if n_methods == 0: raise ValueError("At least one segmentation result is required.")
        
    masks = {}
    for mname, res in results_dict.items():
        nuc, cyto = res.get('nuclear'), res.get('cytoplasm')
        if nuc is not None and cyto is not None: masks[mname] = np.maximum(nuc, cyto)
        elif nuc is not None: masks[mname] = nuc
        elif cyto is not None: masks[mname] = cyto
        
    if n_methods == 1:
        single_mask = masks[method_names[0]].astype(np.uint32)
        confidence = np.where(single_mask > 0, 1.0, 0.0).astype(np.float32)
        ids = np.unique(single_mask)
        ids = ids[ids > 0]
        per_cell = {int(cid): 1.0 for cid in ids}
        cell_map = {int(cid): {method_names[0]: int(cid)} for cid in ids}
        return single_mask, confidence, per_cell, cell_map

    # Extract centroids and bounding boxes
    method_data = {}
    for mname in method_names:
        cell_ids, centroids, bboxes = _extract_cell_centroids(masks[mname])
        method_data[mname] = {'ids': cell_ids, 'centroids': centroids, 'bboxes': bboxes, 'mask': masks[mname]}

    # Primary method (highest cell yield) serves as reference anchor
    reference_method = max(method_names, key=lambda m: len(method_data[m]['ids']))
    ref_data = method_data[reference_method]
    ref_ids, ref_centroids, ref_bboxes = ref_data['ids'], ref_data['centroids'], ref_data['bboxes']
    
    h, w = masks[reference_method].shape
    consensus_mask = np.zeros((h, w), dtype=np.uint32)
    per_cell_confidence: Dict[int, float] = {}
    cell_method_map: Dict[int, Dict[str, int]] = {}
    next_consensus_id = 1
    other_trees = {}
    
    for mname in method_names:
        if mname != reference_method and len(method_data[mname]['centroids']) > 0:
            other_trees[mname] = cKDTree(method_data[mname]['centroids'])
            
    matched_other = {mname: set() for mname in method_names if mname != reference_method}

    # Pre-query all trees simultaneously for vectorized spatial proximity
    queries = {}
    for mname, tree in other_trees.items():
        queries[mname] = tree.query(ref_centroids, distance_upper_bound=centroid_distance_threshold)

    # Match reference cells against other methods using BBOX slicing
    for ref_idx, ref_cid in enumerate(ref_ids):
        ref_bbox = ref_bboxes[ref_idx]
        agreements = {reference_method: int(ref_cid)}
        score_sum = 0.0
        
        for mname in other_trees:
            dist = queries[mname][0][ref_idx]
            idx = queries[mname][1][ref_idx]
            if dist == float('inf'): continue
                
            other_cid = int(method_data[mname]['ids'][idx])
            if other_cid in matched_other[mname]: continue
                
            overlap_ref_in_other, overlap_other_in_ref = _compute_directional_overlaps(
                masks[reference_method], ref_cid, masks[mname], other_cid, ref_bbox
            )
            
            # High overlap threshold (70% mutual inclusion)
            inclusion_thresh = 0.7 
            if overlap_ref_in_other >= inclusion_thresh and overlap_other_in_ref >= inclusion_thresh:
                score = 1.0  # Bidirectional match
            elif overlap_ref_in_other >= 0.1 or overlap_other_in_ref >= 0.1:
                score = 0.5  # Unidirectional partial match
            else:
                score = 0.0  # Disagreement / negligible overlap
                
            score_sum += score
            if score > 0:
                agreements[mname] = other_cid
                matched_other[mname].add(other_cid)
                
        confidence = score_sum / (n_methods - 1)
        
        minr, minc, maxr, maxc = ref_bbox
        cell_pixels = (masks[reference_method][minr:maxr, minc:maxc] == ref_cid)
        consensus_mask[minr:maxr, minc:maxc][cell_pixels] = next_consensus_id
        
        per_cell_confidence[next_consensus_id] = float(confidence)
        cell_method_map[next_consensus_id] = agreements
        next_consensus_id += 1

    # Add unmatched cells from non-reference methods
    for mname in method_names:
        if mname == reference_method: continue
        for idx, cid in enumerate(method_data[mname]['ids']):
            cid_int = int(cid)
            if cid_int in matched_other.get(mname, set()): continue
            
            minr, minc, maxr, maxc = method_data[mname]['bboxes'][idx]
            m_slice = masks[mname][minr:maxr, minc:maxc]
            cons_slice = consensus_mask[minr:maxr, minc:maxc]
            
            cell_pixels = (m_slice == cid_int)
            available_pixels = cell_pixels & (cons_slice == 0)
            
            if np.sum(available_pixels) > 0:
                cons_slice[available_pixels] = next_consensus_id
                per_cell_confidence[next_consensus_id] = 0.0  # Unmatched across methods
                cell_method_map[next_consensus_id] = {mname: cid_int}
                next_consensus_id += 1

    # Build cell-level agreement heatmap
    confidence_map = np.zeros((h, w), dtype=np.float32)
    for cid, conf in per_cell_confidence.items():
        confidence_map[consensus_mask == cid] = max(0.01, float(conf))
        
    return consensus_mask, confidence_map, per_cell_confidence, cell_method_map


def apply_manual_override(
    consensus_mask: np.ndarray,
    confidence_map: np.ndarray,
    per_cell_confidence: Dict[int, float],
    cell_method_map: Dict[int, Dict[str, int]],
    cell_id: int,
    preferred_method: str,
    masks: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, float]]:
    """Apply manual curation override to replace or remove a consensus cell footprint.

    Args:
        consensus_mask: Active consensus label mask.
        confidence_map: Active confidence heatmap array.
        per_cell_confidence: Active dictionary of per-cell confidence values.
        cell_method_map: Dictionary mapping consensus ID to method predictions.
        cell_id: Target consensus cell ID to modify.
        preferred_method: Name of method whose mask should be adopted, or 'Remove'.
        masks: Dictionary of full image masks per method.

    Returns:
        Updated tuple of (consensus_mask, confidence_map, per_cell_confidence).
    """
    if cell_id not in cell_method_map: return consensus_mask, confidence_map, per_cell_confidence
    
    # Clear the old cell footprint
    old_pixels = (consensus_mask == cell_id)
    consensus_mask[old_pixels] = 0
    confidence_map[old_pixels] = 0.0
    
    if preferred_method == "Remove":
        per_cell_confidence.pop(cell_id, None)
        cell_method_map.pop(cell_id, None)
        return consensus_mask, confidence_map, per_cell_confidence

    method_info = cell_method_map[cell_id]
    if preferred_method not in method_info: return consensus_mask, confidence_map, per_cell_confidence
    
    original_id = method_info[preferred_method]
    preferred_mask = masks[preferred_method]
    
    new_pixels = (preferred_mask == original_id)
    available = new_pixels & (consensus_mask == 0)
    
    consensus_mask[available] = cell_id
    confidence_map[available] = 1.0
    per_cell_confidence[cell_id] = 1.0
    return consensus_mask, confidence_map, per_cell_confidence