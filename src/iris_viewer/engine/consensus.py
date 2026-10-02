""" IRIS-Viewer Consensus Engine """
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
from typing import Dict, Tuple, Optional

def _extract_cell_centroids(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract cell IDs, centroids, and bounding boxes from a label mask."""
    props = regionprops_table(mask, properties=('label', 'centroid', 'bbox'))
    if len(props['label']) == 0:
        return np.array([], dtype=np.uint32), np.zeros((0, 2), dtype=np.float64), np.zeros((0, 4), dtype=int)
        
    cell_ids = np.array(props['label'], dtype=np.uint32)
    centroids = np.column_stack([props['centroid-0'], props['centroid-1']])
    bboxes = np.column_stack([props['bbox-0'], props['bbox-1'], props['bbox-2'], props['bbox-3']])
    return cell_ids, centroids, bboxes

def _compute_directional_overlaps(mask_a: np.ndarray, id_a: int, mask_b: np.ndarray, id_b: int, bbox: np.ndarray) -> Tuple[float, float]:
    """Compute directional overlap ratios (Intersection / Area A, Intersection / Area B)."""
    minr, minc, maxr, maxc = bbox
    minr = max(0, minr - 15)
    maxr = min(mask_a.shape[0], maxr + 15)
    minc = max(0, minc - 15)
    maxc = min(mask_a.shape[1], maxc + 15)
    
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
) -> Tuple[np.ndarray, np.ndarray, Dict[int, float], Dict[int, Dict]]:
    
    method_names = list(results_dict.keys())
    n_methods = len(method_names)
    
    if n_methods == 0:
        raise ValueError("At least one segmentation result is required.")
        
    masks = {}
    for mname, res in results_dict.items():
        nuc = res.get('nuclear')
        cyto = res.get('cytoplasm')
        if nuc is not None and cyto is not None:
            masks[mname] = np.maximum(nuc, cyto)
        elif nuc is not None:
            masks[mname] = nuc
        elif cyto is not None:
            masks[mname] = cyto
        
    if n_methods == 1:
        single_mask = masks[method_names[0]].astype(np.uint32)
        confidence = np.where(single_mask > 0, 1.0, 0.0).astype(np.float32)
        ids = np.unique(single_mask)
        ids = ids[ids > 0]
        per_cell = {int(cid): 1.0 for cid in ids}
        cell_map = {int(cid): {method_names[0]: int(cid)} for cid in ids}
        return single_mask, confidence, per_cell, cell_map

    # Step 1: Extract centroids and bboxes
    method_data = {}
    for mname in method_names:
        cell_ids, centroids, bboxes = _extract_cell_centroids(masks[mname])
        method_data[mname] = {'ids': cell_ids, 'centroids': centroids, 'bboxes': bboxes, 'mask': masks[mname]}

    # Step 2: Use primary method (most cells) as reference
    reference_method = max(method_names, key=lambda m: len(method_data[m]['ids']))
    ref_data = method_data[reference_method]
    ref_ids = ref_data['ids']
    ref_centroids = ref_data['centroids']
    ref_bboxes = ref_data['bboxes']
    
    h, w = masks[reference_method].shape
    consensus_mask = np.zeros((h, w), dtype=np.uint32)
    per_cell_confidence = {}
    cell_method_map = {}
    next_consensus_id = 1
    other_trees = {}
    
    for mname in method_names:
        if mname != reference_method and len(method_data[mname]['centroids']) > 0:
            other_trees[mname] = cKDTree(method_data[mname]['centroids'])
            
    matched_other = {mname: set() for mname in method_names if mname != reference_method}

    # FIX: Pre-query all trees simultaneously to utilize C-level vectorization and eliminate the freezing loop
    queries = {}
    for mname, tree in other_trees.items():
        queries[mname] = tree.query(ref_centroids, distance_upper_bound=centroid_distance_threshold)

    # Step 3: Match reference cells against other methods using BBOX slicing
    for ref_idx, ref_cid in enumerate(ref_ids):
        ref_centroid = ref_centroids[ref_idx]
        ref_bbox = ref_bboxes[ref_idx]
        agreements = {reference_method: int(ref_cid)}
        
        score_sum = 0.0
        
        for mname in other_trees:
            dist = queries[mname][0][ref_idx]
            idx = queries[mname][1][ref_idx]
            
            if dist == float('inf'): 
                continue
                
            other_cid = int(method_data[mname]['ids'][idx])
            if other_cid in matched_other[mname]: 
                continue
                
            overlap_ref_in_other, overlap_other_in_ref = _compute_directional_overlaps(
                masks[reference_method], ref_cid, masks[mname], other_cid, ref_bbox
            )
            
            # Bidirectional/Unidirectional Logic
            # High overlap threshold (e.g., 70% inclusion)
            T = 0.7 
            if overlap_ref_in_other >= T and overlap_other_in_ref >= T:
                score = 1.0  # Bidirectional match (Green)
            elif overlap_ref_in_other >= 0.1 or overlap_other_in_ref >= 0.1:
                score = 0.5  # Unidirectional match (Yellow)
            else:
                score = 0.0  # No Match (Red)
                
            score_sum += score
            
            if score > 0: # Keep track of it if there's any meaningful overlap
                agreements[mname] = other_cid
                matched_other[mname].add(other_cid)
                
        # Confidence score based on Bidirectional classification
        confidence = score_sum / (n_methods - 1)
        
        # SLICED ASSIGNMENT (Lightning Fast)
        minr, minc, maxr, maxc = ref_bbox
        cell_pixels = (masks[reference_method][minr:maxr, minc:maxc] == ref_cid)
        consensus_mask[minr:maxr, minc:maxc][cell_pixels] = next_consensus_id
        
        per_cell_confidence[next_consensus_id] = float(confidence)
        cell_method_map[next_consensus_id] = agreements
        next_consensus_id += 1

    # Step 4: Add unmatched cells using BBOX slicing
    for mname in method_names:
        if mname == reference_method: continue
        for idx, cid in enumerate(method_data[mname]['ids']):
            cid_int = int(cid)
            if cid_int in matched_other.get(mname, set()): continue
            
            confidence = 0.0 # Unmatched cell has 0% IoU with reference
            minr, minc, maxr, maxc = method_data[mname]['bboxes'][idx]
            
            m_slice = masks[mname][minr:maxr, minc:maxc]
            cons_slice = consensus_mask[minr:maxr, minc:maxc]
            
            cell_pixels = (m_slice == cid_int)
            available_pixels = cell_pixels & (cons_slice == 0)
            
            if np.sum(available_pixels) > 0:
                cons_slice[available_pixels] = next_consensus_id
                per_cell_confidence[next_consensus_id] = float(confidence)
                cell_method_map[next_consensus_id] = {mname: cid_int}
                next_consensus_id += 1

    # Step 5: Build cell-level agreement heatmap
    confidence_map = np.zeros((h, w), dtype=np.float32)
    for cid, conf in per_cell_confidence.items():
        # Ensure 0 confidence (No match) gets colored red (0.01) instead of transparent
        display_conf = max(0.01, float(conf))
        confidence_map[consensus_mask == cid] = display_conf
        
    return consensus_mask, confidence_map, per_cell_confidence, cell_method_map

def apply_manual_override(
    consensus_mask: np.ndarray, confidence_map: np.ndarray, per_cell_confidence: Dict[int, float],
    cell_method_map: Dict[int, Dict], cell_id: int, preferred_method: str, masks: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, float]]:
    
    if cell_id not in cell_method_map: 
        return consensus_mask, confidence_map, per_cell_confidence
    
    # Always clear the old cell footprint first
    old_pixels = (consensus_mask == cell_id)
    consensus_mask[old_pixels] = 0
    confidence_map[old_pixels] = 0.0
    
    if preferred_method == "Remove":
        per_cell_confidence.pop(cell_id, None)
        cell_method_map.pop(cell_id, None)
        return consensus_mask, confidence_map, per_cell_confidence

    method_info = cell_method_map[cell_id]
    if preferred_method not in method_info: 
        return consensus_mask, confidence_map, per_cell_confidence
    
    original_id = method_info[preferred_method]
    preferred_mask = masks[preferred_method]
    
    new_pixels = (preferred_mask == original_id)
    available = new_pixels & (consensus_mask == 0)
    
    consensus_mask[available] = cell_id
    confidence_map[available] = 1.0
    per_cell_confidence[cell_id] = 1.0
    
    return consensus_mask, confidence_map, per_cell_confidence