"""
IRIS-Viewer Multi-Model Disagreement Engine

Compares label masks produced by multiple segmentation backends/pipelines,
quantifies pixel and instance agreement, classifies spatial discrepancies
(Split, Merge, Boundary Shift, Model-Exclusive), and renders diagnostic overlays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, find_objects
from skimage.segmentation import find_boundaries

DISAGREEMENT_CATEGORIES: Tuple[str, ...] = (
    "Agreed",
    "Boundary Shift",
    "Split",
    "Merge",
    "Model-Exclusive",
)

METHOD_QC_COLORS: Tuple[str, ...] = (
    "#00E5FF",
    "#FF2D95",
    "#FFD60A",
    "#32D74B",
    "#BF5AF2",
    "#FF9F0A",
)

# RGBA float32 colors for contour rendering
_CATEGORY_COLORS_RGBA: Dict[str, np.ndarray] = {
    "Split": np.array([1.0, 59.0 / 255.0, 48.0 / 255.0, 1.0], dtype=np.float32),
    "Merge": np.array([191.0 / 255.0, 90.0 / 255.0, 242.0 / 255.0, 1.0], dtype=np.float32),
    "Boundary Shift": np.array([1.0, 204.0 / 255.0, 0.0, 1.0], dtype=np.float32),
    "Model-Exclusive": np.array([0.0, 229.0 / 255.0, 1.0, 1.0], dtype=np.float32),
    "Agreed": np.array([52.0 / 255.0, 199.0 / 255.0, 89.0 / 255.0, 1.0], dtype=np.float32),
}


@dataclass
class DisagreementItem:
    region_id: int
    category: str  # "Boundary Shift" | "Split" | "Merge" | "Model-Exclusive"
    severity: float  # [0.0, 1.0]
    best_iou: float  # [0.0, 1.0]
    bbox: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    centroid_yx: Tuple[float, float]
    method_cell_ids: Dict[str, List[int]]  # method_name -> overlapping label IDs
    dice: float = 0.0
    overlap_coeff: float = 0.0


@dataclass
class DisagreementReport:
    methods: List[str]
    reference_method: str
    compare_mode: str  # "Nuclear Masks" | "Cytoplasm Masks" | "Both"
    iou_threshold: float
    disagreement_heatmap: np.ndarray  # (H, W) float32 in [0, 1]
    agreement_map: np.ndarray  # (H, W) float32 in [0, 1]
    pairwise_dice: pd.DataFrame  # (M, M) pixel foreground Dice
    pairwise_iou: pd.DataFrame  # (M, M) mean matched-instance IoU
    model_summary: pd.DataFrame  # ["Method", "Cell_Count", "Median_Area_px", "Mean_Peer_IoU"]
    per_cell_iou: Dict[int, float]  # ref_cell_id -> mean best peer IoU
    items: List[DisagreementItem]  # sorted descending by severity (disputed regions only)
    category_counts: Dict[str, int]
    total_ref_cells: int
    agreed_cells: int
    flagged_regions: int
    mean_iou: float


@dataclass
class _MethodMaskInfo:
    mask: np.ndarray
    max_label: int
    areas: np.ndarray
    cell_ids: np.ndarray
    bboxes: Dict[int, Tuple[int, int, int, int]]
    centroids: Dict[int, Tuple[float, float]]


@dataclass
class _OverlapEdge:
    a_id: int
    b_id: int
    inter: int
    iou: float
    frac_a: float
    frac_b: float
    dice: float
    overlap_coeff: float


def _extract_masks_for_mode(
    segmentation_results: Dict[str, Dict[str, Optional[np.ndarray]]],
    compare_mode: str,
) -> Dict[str, np.ndarray]:
    """Select and validate 2D integer label masks according to compare_mode."""
    valid_modes = {"Nuclear Masks", "Cytoplasm Masks", "Both"}
    if compare_mode not in valid_modes:
        raise ValueError(
            f"Unsupported compare_mode '{compare_mode}'. Expected one of {sorted(valid_modes)}."
        )

    valid_entries = [
        (method_name, res)
        for method_name, res in segmentation_results.items()
        if isinstance(res, dict)
        and (res.get("nuclear") is not None or res.get("cytoplasm") is not None)
    ]
    all_have_cyto = bool(valid_entries) and all(
        res.get("cytoplasm") is not None for _, res in valid_entries
    )

    masks: Dict[str, np.ndarray] = {}
    expected_shape: Optional[Tuple[int, int]] = None

    for method_name, res in valid_entries:
        nuc = res.get("nuclear")
        cyto = res.get("cytoplasm")

        chosen: Optional[np.ndarray] = None
        if compare_mode == "Nuclear Masks":
            chosen = nuc
        elif compare_mode == "Cytoplasm Masks":
            chosen = cyto
        else:
            if all_have_cyto:
                chosen = np.where(cyto > 0, cyto, nuc) if nuc is not None else cyto
            else:
                chosen = nuc

        if chosen is None:
            continue

        arr = np.asarray(chosen)
        if arr.ndim != 2:
            raise ValueError(
                f"Segmentation mask for '{method_name}' must be 2D, got shape {arr.shape}."
            )
        if expected_shape is None:
            expected_shape = arr.shape
        elif arr.shape != expected_shape:
            raise ValueError(
                f"Mask shape mismatch: '{method_name}' has {arr.shape}, expected {expected_shape}."
            )

        if np.any(arr < 0):
            arr = np.maximum(arr, 0)
        masks[method_name] = arr.astype(np.int64, copy=False)

    if not masks:
        raise ValueError(
            f"No valid segmentation masks found for compare_mode='{compare_mode}'."
        )
    return masks


def _extract_method_info(mask: np.ndarray) -> _MethodMaskInfo:
    """Compute vectorized areas, bboxes (via find_objects), and centroids for one mask."""
    max_lbl = int(mask.max()) if mask.size > 0 else 0
    if max_lbl <= 0:
        return _MethodMaskInfo(
            mask=mask,
            max_label=0,
            areas=np.zeros(1, dtype=np.int64),
            cell_ids=np.array([], dtype=np.int64),
            bboxes={},
            centroids={},
        )

    areas = np.bincount(mask.ravel(), minlength=max_lbl + 1)
    cell_ids = np.flatnonzero(areas[1:] > 0) + 1
    slices = find_objects(mask)

    y_nz, x_nz = np.nonzero(mask > 0)
    lbl_nz = mask[y_nz, x_nz]
    sum_y = np.bincount(lbl_nz, weights=y_nz.astype(np.float64), minlength=max_lbl + 1)
    sum_x = np.bincount(lbl_nz, weights=x_nz.astype(np.float64), minlength=max_lbl + 1)

    bboxes: Dict[int, Tuple[int, int, int, int]] = {}
    centroids: Dict[int, Tuple[float, float]] = {}
    for cid in cell_ids:
        cid_int = int(cid)
        sl = slices[cid_int - 1]
        if sl is not None:
            bboxes[cid_int] = (sl[0].start, sl[1].start, sl[0].stop, sl[1].stop)
        else:
            bboxes[cid_int] = (0, 0, mask.shape[0], mask.shape[1])
        area_val = float(areas[cid_int])
        centroids[cid_int] = (float(sum_y[cid_int] / area_val), float(sum_x[cid_int] / area_val))

    return _MethodMaskInfo(
        mask=mask,
        max_label=max_lbl,
        areas=areas,
        cell_ids=cell_ids,
        bboxes=bboxes,
        centroids=centroids,
    )


def _compute_sparse_overlaps(
    info_a: _MethodMaskInfo,
    info_b: _MethodMaskInfo,
) -> Tuple[Dict[int, List[_OverlapEdge]], Dict[int, List[_OverlapEdge]], List[_OverlapEdge]]:
    """Compute sparse cell-pair intersections and IoUs in O(H*W) without quadratic loops."""
    a_to_b: Dict[int, List[_OverlapEdge]] = {}
    b_to_a: Dict[int, List[_OverlapEdge]] = {}
    edges: List[_OverlapEdge] = []

    if info_a.cell_ids.size == 0 or info_b.cell_ids.size == 0:
        return a_to_b, b_to_a, edges

    both = (info_a.mask > 0) & (info_b.mask > 0)
    if not np.any(both):
        return a_to_b, b_to_a, edges

    la = info_a.mask[both]
    lb = info_b.mask[both]
    stride = int(info_b.max_label) + 1
    keys = la * stride + lb
    uniq_keys, counts = np.unique(keys, return_counts=True)

    a_ids = uniq_keys // stride
    b_ids = uniq_keys % stride
    areas_a = info_a.areas[a_ids]
    areas_b = info_b.areas[b_ids]
    unions = areas_a + areas_b - counts

    ious = counts.astype(np.float64) / np.maximum(unions, 1)
    frac_a_arr = counts.astype(np.float64) / np.maximum(areas_a, 1)
    frac_b_arr = counts.astype(np.float64) / np.maximum(areas_b, 1)
    dice_arr = (2.0 * counts.astype(np.float64)) / np.maximum(areas_a + areas_b, 1)
    ov_arr = counts.astype(np.float64) / np.maximum(np.minimum(areas_a, areas_b), 1)

    for idx in range(len(uniq_keys)):
        aid = int(a_ids[idx])
        bid = int(b_ids[idx])
        edge = _OverlapEdge(
            a_id=aid,
            b_id=bid,
            inter=int(counts[idx]),
            iou=float(ious[idx]),
            frac_a=float(frac_a_arr[idx]),
            frac_b=float(frac_b_arr[idx]),
            dice=float(dice_arr[idx]),
            overlap_coeff=float(ov_arr[idx]),
        )
        edges.append(edge)
        a_to_b.setdefault(aid, []).append(edge)
        b_to_a.setdefault(bid, []).append(edge)

    return a_to_b, b_to_a, edges


def _matched_instance_iou(
    info_a: _MethodMaskInfo,
    info_b: _MethodMaskInfo,
    edges: List[_OverlapEdge],
) -> float:
    """Compute symmetric 1-to-1 matched-instance IoU between two masks."""
    n_a = int(info_a.cell_ids.size)
    n_b = int(info_b.cell_ids.size)
    if n_a == 0 and n_b == 0:
        return 1.0
    if n_a == 0 or n_b == 0 or not edges:
        return 0.0

    used_a = set()
    used_b = set()
    iou_sum = 0.0
    for edge in sorted(edges, key=lambda e: e.iou, reverse=True):
        if edge.a_id not in used_a and edge.b_id not in used_b:
            used_a.add(edge.a_id)
            used_b.add(edge.b_id)
            iou_sum += edge.iou

    return float(iou_sum / max(n_a, n_b))


def _union_bboxes(bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    min_r = min(b[0] for b in bboxes)
    min_c = min(b[1] for b in bboxes)
    max_r = max(b[2] for b in bboxes)
    max_c = max(b[3] for b in bboxes)
    return (min_r, min_c, max_r, max_c)


def _weighted_centroid(
    cids_and_infos: List[Tuple[int, _MethodMaskInfo]],
) -> Tuple[float, float]:
    total_area = 0.0
    sum_y = 0.0
    sum_x = 0.0
    for cid, info in cids_and_infos:
        area = float(info.areas[cid])
        cy, cx = info.centroids[cid]
        total_area += area
        sum_y += cy * area
        sum_x += cx * area
    if total_area <= 0.0:
        return (0.0, 0.0)
    return (sum_y / total_area, sum_x / total_area)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        p = self.parent.setdefault(x, x)
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def compute_model_disagreement(
    segmentation_results: Dict[str, Dict[str, Optional[np.ndarray]]],
    compare_mode: str = "Nuclear Masks",
    reference_method: Optional[str] = None,
    iou_threshold: float = 0.65,
    selected_methods: Optional[Sequence[str]] = None,
) -> DisagreementReport:
    """Compare multiple segmentation masks and identify spatial model disagreements."""
    if selected_methods is not None:
        filtered_results = {
            m: segmentation_results[m]
            for m in selected_methods
            if m in segmentation_results
        }
    else:
        filtered_results = segmentation_results

    masks = _extract_masks_for_mode(filtered_results, compare_mode)
    methods = list(masks.keys())
    n_methods = len(methods)
    h, w = next(iter(masks.values())).shape

    infos: Dict[str, _MethodMaskInfo] = {m: _extract_method_info(masks[m]) for m in methods}

    if reference_method is not None:
        if reference_method in masks:
            ref_method = reference_method
        elif reference_method in segmentation_results:
            ref_method = max(methods, key=lambda m: int(infos[m].cell_ids.size))
        else:
            raise ValueError(f"Reference method '{reference_method}' not found in results.")
    else:
        ref_method = max(methods, key=lambda m: int(infos[m].cell_ids.size))

    ref_info = infos[ref_method]
    peers = [m for m in methods if m != ref_method]

    # Pairwise pixel foreground Dice and matched-instance IoU
    dice_mat = np.eye(n_methods, dtype=np.float64)
    iou_mat = np.eye(n_methods, dtype=np.float64)

    pair_overlaps: Dict[
        Tuple[str, str],
        Tuple[Dict[int, List[_OverlapEdge]], Dict[int, List[_OverlapEdge]], List[_OverlapEdge]],
    ] = {}

    fg_masks = {m: (masks[m] > 0) for m in methods}
    fg_counts = {m: int(np.count_nonzero(fg_masks[m])) for m in methods}

    for i in range(n_methods):
        m_i = methods[i]
        for j in range(i + 1, n_methods):
            m_j = methods[j]
            inter_fg = int(np.count_nonzero(fg_masks[m_i] & fg_masks[m_j]))
            denom_fg = fg_counts[m_i] + fg_counts[m_j]
            dice = (2.0 * inter_fg / denom_fg) if denom_fg > 0 else 1.0
            dice_mat[i, j] = dice
            dice_mat[j, i] = dice

            a_to_b, b_to_a, edges = _compute_sparse_overlaps(infos[m_i], infos[m_j])
            pair_overlaps[(m_i, m_j)] = (a_to_b, b_to_a, edges)
            pair_overlaps[(m_j, m_i)] = (b_to_a, a_to_b, edges)

            inst_iou = _matched_instance_iou(infos[m_i], infos[m_j], edges)
            iou_mat[i, j] = inst_iou
            iou_mat[j, i] = inst_iou

    pairwise_dice = pd.DataFrame(dice_mat, index=methods, columns=methods)
    pairwise_iou = pd.DataFrame(iou_mat, index=methods, columns=methods)

    # Model summary table
    summary_rows = []
    for i, m in enumerate(methods):
        m_areas = infos[m].areas[infos[m].cell_ids] if infos[m].cell_ids.size > 0 else np.array([])
        med_area = float(np.median(m_areas)) if m_areas.size > 0 else 0.0
        if n_methods == 1:
            mean_peer_iou = 1.0
        else:
            peer_vals = [iou_mat[i, j] for j in range(n_methods) if j != i]
            mean_peer_iou = float(np.mean(peer_vals))
        summary_rows.append(
            {
                "Method": m,
                "Cell_Count": int(infos[m].cell_ids.size),
                "Median_Area_px": med_area,
                "Mean_Peer_IoU": mean_peer_iou,
            }
        )
    model_summary = pd.DataFrame(
        summary_rows,
        columns=["Method", "Cell_Count", "Median_Area_px", "Mean_Peer_IoU"],
    )

    category_counts: Dict[str, int] = {cat: 0 for cat in DISAGREEMENT_CATEGORIES}
    ref_cell_ids = [int(c) for c in ref_info.cell_ids]
    total_ref_cells = len(ref_cell_ids)

    # Single-model edge case
    if n_methods == 1:
        per_cell_iou = {cid: 1.0 for cid in ref_cell_ids}
        category_counts["Agreed"] = total_ref_cells
        disagreement_heatmap = np.zeros((h, w), dtype=np.float32)
        agreement_map = np.ones((h, w), dtype=np.float32)
        return DisagreementReport(
            methods=methods,
            reference_method=ref_method,
            compare_mode=compare_mode,
            iou_threshold=float(iou_threshold),
            disagreement_heatmap=disagreement_heatmap,
            agreement_map=agreement_map,
            pairwise_dice=pairwise_dice,
            pairwise_iou=pairwise_iou,
            model_summary=model_summary,
            per_cell_iou=per_cell_iou,
            items=[],
            category_counts=category_counts,
            total_ref_cells=total_ref_cells,
            agreed_cells=total_ref_cells,
            flagged_regions=0,
            mean_iou=1.0,
        )

    # Compute per-reference-cell best peer IoUs, Dice, and Overlap Coefficients
    per_cell_iou: Dict[int, float] = {}
    per_cell_max_iou: Dict[int, float] = {}
    per_cell_dice: Dict[int, float] = {}
    per_cell_overlap: Dict[int, float] = {}
    for rid in ref_cell_ids:
        peer_best_ious: List[float] = []
        peer_best_dices: List[float] = []
        peer_best_ovs: List[float] = []
        for p in peers:
            r_to_p = pair_overlaps[(ref_method, p)][0]
            r_edges = r_to_p.get(rid, [])
            if r_edges:
                best_edge = max(r_edges, key=lambda e: e.iou)
                peer_best_ious.append(best_edge.iou)
                peer_best_dices.append(best_edge.dice)
                peer_best_ovs.append(best_edge.overlap_coeff)
            else:
                peer_best_ious.append(0.0)
                peer_best_dices.append(0.0)
                peer_best_ovs.append(0.0)
        per_cell_iou[rid] = float(np.mean(peer_best_ious))
        per_cell_max_iou[rid] = float(max(peer_best_ious, default=0.0))
        per_cell_dice[rid] = float(np.mean(peer_best_dices))
        per_cell_overlap[rid] = float(np.mean(peer_best_ovs))

    # Helper to gather overlapping cell IDs across all methods for a group of ref cells
    def _gather_method_ids(r_ids: List[int]) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {ref_method: sorted(r_ids)}
        for p in peers:
            r_to_p = pair_overlaps[(ref_method, p)][0]
            pids = set()
            for rid in r_ids:
                for edge in r_to_p.get(rid, []):
                    if edge.iou >= 0.10 or edge.frac_a >= 0.15 or edge.frac_b >= 0.15:
                        pids.add(edge.b_id)
            if pids:
                out[p] = sorted(pids)
        return out

    def _region_bbox_and_centroid(
        method_cids: Dict[str, List[int]],
    ) -> Tuple[Tuple[int, int, int, int], Tuple[float, float]]:
        bboxes: List[Tuple[int, int, int, int]] = []
        weighted_items: List[Tuple[int, _MethodMaskInfo]] = []
        for m_name, cids in method_cids.items():
            m_info = infos[m_name]
            for cid in cids:
                bboxes.append(m_info.bboxes[cid])
                weighted_items.append((cid, m_info))
        return _union_bboxes(bboxes), _weighted_centroid(weighted_items)

    items: List[DisagreementItem] = []
    classified_ref: set = set()
    covered_peer_cells: Dict[str, set] = {p: set() for p in peers}

    # 1. Detect Split reference cells (one ref cell overlaps >= 2 distinct peer cells)
    for rid in ref_cell_ids:
        is_split = False
        for p in peers:
            r_to_p = pair_overlaps[(ref_method, p)][0]
            split_targets = [
                e
                for e in r_to_p.get(rid, [])
                if e.frac_b >= 0.15 and (e.iou >= 0.10 or e.frac_a >= 0.10)
            ]
            if len(split_targets) >= 2:
                is_split = True
                break

        if is_split:
            classified_ref.add(rid)
            m_ids = _gather_method_ids([rid])
            for p in peers:
                covered_peer_cells[p].update(m_ids.get(p, ()))
            bbox, centroid = _region_bbox_and_centroid(m_ids)
            best_iou_val = per_cell_iou[rid]
            severity = float(min(1.0, max(0.75, 1.0 - best_iou_val + 0.15)))
            items.append(
                DisagreementItem(
                    region_id=0,
                    category="Split",
                    severity=severity,
                    best_iou=best_iou_val,
                    bbox=bbox,
                    centroid_yx=centroid,
                    method_cell_ids=m_ids,
                    dice=per_cell_dice[rid],
                    overlap_coeff=per_cell_overlap[rid],
                )
            )

    # 2. Detect Merge groups (>= 2 ref cells map into the same single peer cell)
    merge_uf = _UnionFind()
    merged_ref_candidates: set = set()
    for p in peers:
        r_to_p = pair_overlaps[(ref_method, p)][0]
        p_to_ref_primary: Dict[int, List[int]] = {}
        for rid in ref_cell_ids:
            if rid in classified_ref:
                continue
            r_edges = r_to_p.get(rid, [])
            if not r_edges:
                continue
            best_edge = max(r_edges, key=lambda e: e.inter)
            if best_edge.frac_a >= 0.15 and (
                best_edge.iou >= 0.10 or best_edge.frac_a >= 0.50
            ):
                p_to_ref_primary.setdefault(best_edge.b_id, []).append(rid)

        for r_group in p_to_ref_primary.values():
            if len(r_group) >= 2:
                head = r_group[0]
                merged_ref_candidates.add(head)
                for other_rid in r_group[1:]:
                    merged_ref_candidates.add(other_rid)
                    merge_uf.union(head, other_rid)

    merge_groups: Dict[int, List[int]] = {}
    for rid in sorted(merged_ref_candidates):
        merge_groups.setdefault(merge_uf.find(rid), []).append(rid)

    for r_group in merge_groups.values():
        for rid in r_group:
            classified_ref.add(rid)
        m_ids = _gather_method_ids(r_group)
        for p in peers:
            covered_peer_cells[p].update(m_ids.get(p, ()))
        bbox, centroid = _region_bbox_and_centroid(m_ids)
        best_iou_val = float(np.mean([per_cell_iou[r] for r in r_group]))
        dice_val = float(np.mean([per_cell_dice[r] for r in r_group]))
        ov_val = float(np.mean([per_cell_overlap[r] for r in r_group]))
        severity = float(min(1.0, max(0.75, 1.0 - best_iou_val + 0.15)))
        items.append(
            DisagreementItem(
                region_id=0,
                category="Merge",
                severity=severity,
                best_iou=best_iou_val,
                bbox=bbox,
                centroid_yx=centroid,
                method_cell_ids=m_ids,
                dice=dice_val,
                overlap_coeff=ov_val,
            )
        )

    # 3. Classify remaining reference cells: Model-Exclusive, Boundary Shift, or Agreed
    agreed_cells = 0
    for rid in ref_cell_ids:
        if rid in classified_ref:
            continue
        mean_best = per_cell_iou[rid]
        max_best = per_cell_max_iou[rid]

        if max_best < 0.10:
            m_ids = {ref_method: [rid]}
            bbox, centroid = _region_bbox_and_centroid(m_ids)
            severity = float(max(0.85, 1.0 - mean_best))
            items.append(
                DisagreementItem(
                    region_id=0,
                    category="Model-Exclusive",
                    severity=min(1.0, severity),
                    best_iou=mean_best,
                    bbox=bbox,
                    centroid_yx=centroid,
                    method_cell_ids=m_ids,
                    dice=per_cell_dice[rid],
                    overlap_coeff=per_cell_overlap[rid],
                )
            )
        elif mean_best < float(iou_threshold):
            m_ids = _gather_method_ids([rid])
            for p in peers:
                covered_peer_cells[p].update(m_ids.get(p, ()))
            bbox, centroid = _region_bbox_and_centroid(m_ids)
            severity = float(np.clip(1.0 - mean_best, 0.0, 1.0))
            items.append(
                DisagreementItem(
                    region_id=0,
                    category="Boundary Shift",
                    severity=severity,
                    best_iou=mean_best,
                    bbox=bbox,
                    centroid_yx=centroid,
                    method_cell_ids=m_ids,
                    dice=per_cell_dice[rid],
                    overlap_coeff=per_cell_overlap[rid],
                )
            )
        else:
            agreed_cells += 1
            for p in peers:
                r_to_p = pair_overlaps[(ref_method, p)][0]
                for edge in r_to_p.get(rid, []):
                    if edge.iou >= 0.10:
                        covered_peer_cells[p].add(edge.b_id)

    # 4. Orphan peer cells overlapping no reference cell -> Model-Exclusive
    orphan_nodes: List[Tuple[str, int]] = []
    orphan_index: Dict[Tuple[str, int], int] = {}
    for p in peers:
        p_to_ref = pair_overlaps[(p, ref_method)][0]
        for pid in infos[p].cell_ids:
            pid_int = int(pid)
            if pid_int in covered_peer_cells[p]:
                continue
            ref_edges = p_to_ref.get(pid_int, [])
            max_ref_iou = max((e.iou for e in ref_edges), default=0.0)
            if max_ref_iou < 0.10:
                idx = len(orphan_nodes)
                node = (p, pid_int)
                orphan_nodes.append(node)
                orphan_index[node] = idx

    if orphan_nodes:
        orphan_uf = _UnionFind()
        for idx in range(len(orphan_nodes)):
            orphan_uf.find(idx)
        for i_p in range(len(peers)):
            p1 = peers[i_p]
            for j_p in range(i_p + 1, len(peers)):
                p2 = peers[j_p]
                p1_to_p2 = pair_overlaps[(p1, p2)][0]
                for idx, (m_name, cid) in enumerate(orphan_nodes):
                    if m_name != p1:
                        continue
                    for edge in p1_to_p2.get(cid, []):
                        other_node = (p2, edge.b_id)
                        if other_node in orphan_index and edge.iou >= 0.10:
                            orphan_uf.union(idx, orphan_index[other_node])

        orphan_groups: Dict[int, Dict[str, List[int]]] = {}
        for idx, (m_name, cid) in enumerate(orphan_nodes):
            root = orphan_uf.find(idx)
            orphan_groups.setdefault(root, {}).setdefault(m_name, []).append(cid)

        for grp in orphan_groups.values():
            for m_name in grp:
                grp[m_name].sort()
            bbox, centroid = _region_bbox_and_centroid(grp)
            # Severity higher when isolated to fewer peers
            n_peers_present = len(grp)
            severity = float(1.0 if n_peers_present == 1 else max(0.85, 1.0 - 0.05 * n_peers_present))
            items.append(
                DisagreementItem(
                    region_id=0,
                    category="Model-Exclusive",
                    severity=min(1.0, severity),
                    best_iou=0.0,
                    bbox=bbox,
                    centroid_yx=centroid,
                    method_cell_ids=grp,
                )
            )

    # Sort descending by severity and assign 1-based region IDs
    items.sort(key=lambda it: (-it.severity, it.bbox[0], it.bbox[1]))
    for idx, item in enumerate(items, start=1):
        item.region_id = idx
        category_counts[item.category] += 1

    category_counts["Agreed"] = agreed_cells

    # Per-pixel disagreement heatmap: binary foreground vote entropy + split/merge boost
    votes = np.zeros((h, w), dtype=np.float32)
    for m in methods:
        votes += fg_masks[m].astype(np.float32)
    prob = votes / float(n_methods)

    heatmap = np.zeros((h, w), dtype=np.float32)
    partial = (prob > 0.0) & (prob < 1.0)
    if np.any(partial):
        p_val = prob[partial]
        heatmap[partial] = -(p_val * np.log2(p_val) + (1.0 - p_val) * np.log2(1.0 - p_val))

    for item in items:
        if item.category not in ("Split", "Merge"):
            continue
        r0, c0, r1, c1 = item.bbox
        pr0, pc0 = max(0, r0 - 1), max(0, c0 - 1)
        pr1, pc1 = min(h, r1 + 1), min(w, c1 + 1)
        interior = np.zeros((pr1 - pr0, pc1 - pc0), dtype=bool)
        seam = np.zeros((pr1 - pr0, pc1 - pc0), dtype=bool)
        for m_name, cids in item.method_cell_ids.items():
            sub_mask = masks[m_name][pr0:pr1, pc0:pc1]
            sel = np.isin(sub_mask, cids)
            interior |= sel
            if len(cids) >= 2:
                sub_sel = np.where(sel, sub_mask, 0)
                seam |= find_boundaries(sub_sel, mode="inner")

        boost_val = float(max(0.75, item.severity))
        patch = heatmap[pr0:pr1, pc0:pc1]
        patch[interior] = np.maximum(patch[interior], boost_val)
        patch[seam] = 1.0

    disagreement_heatmap = np.clip(heatmap, 0.0, 1.0).astype(np.float32)
    agreement_map = np.clip(1.0 - disagreement_heatmap, 0.0, 1.0).astype(np.float32)

    if per_cell_iou:
        mean_iou = float(np.mean(list(per_cell_iou.values())))
    elif any(infos[m].cell_ids.size > 0 for m in peers):
        mean_iou = 0.0
    else:
        mean_iou = 1.0

    return DisagreementReport(
        methods=methods,
        reference_method=ref_method,
        compare_mode=compare_mode,
        iou_threshold=float(iou_threshold),
        disagreement_heatmap=disagreement_heatmap,
        agreement_map=agreement_map,
        pairwise_dice=pairwise_dice,
        pairwise_iou=pairwise_iou,
        model_summary=model_summary,
        per_cell_iou=per_cell_iou,
        items=items,
        category_counts=category_counts,
        total_ref_cells=total_ref_cells,
        agreed_cells=agreed_cells,
        flagged_regions=len(items),
        mean_iou=mean_iou,
    )


def render_disagreement_contours_rgba(
    segmentation_results: Dict[str, Dict[str, Optional[np.ndarray]]],
    report: DisagreementReport,
    show_agreed: bool = False,
) -> np.ndarray:
    """Render color-coded inner cell boundaries by disagreement category."""
    h, w = report.disagreement_heatmap.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    masks = _extract_masks_for_mode(segmentation_results, report.compare_mode)

    if show_agreed and report.reference_method in masks:
        ref_mask = masks[report.reference_method]
        max_ref = int(ref_mask.max()) if ref_mask.size > 0 else 0
        if max_ref > 0:
            disputed_ref_ids = set()
            for item in report.items:
                disputed_ref_ids.update(item.method_cell_ids.get(report.reference_method, ()))

            agreed_lookup = np.zeros(max_ref + 1, dtype=bool)
            for cid in report.per_cell_iou:
                if 0 < cid <= max_ref and cid not in disputed_ref_ids:
                    agreed_lookup[cid] = True

            ref_bounds = find_boundaries(ref_mask, mode="inner")
            agreed_pixels = ref_bounds & agreed_lookup[ref_mask]
            rgba[agreed_pixels] = _CATEGORY_COLORS_RGBA["Agreed"]

    # Draw lowest-severity items first so highest-severity contours sit on top
    for item in sorted(report.items, key=lambda it: it.severity):
        color = _CATEGORY_COLORS_RGBA.get(item.category)
        if color is None:
            continue
        r0, c0, r1, c1 = item.bbox
        pr0, pc0 = max(0, r0 - 1), max(0, c0 - 1)
        pr1, pc1 = min(h, r1 + 1), min(w, c1 + 1)
        if pr1 <= pr0 or pc1 <= pc0:
            continue

        patch_rgba = rgba[pr0:pr1, pc0:pc1]
        for m_name, cids in item.method_cell_ids.items():
            if m_name not in masks or not cids:
                continue
            sub = masks[m_name][pr0:pr1, pc0:pc1]
            sel = (sub == cids[0]) if len(cids) == 1 else np.isin(sub, cids)
            if not np.any(sel):
                continue
            sub_sel = np.where(sel, sub, 0)
            bounds = find_boundaries(sub_sel, mode="inner")
            patch_rgba[bounds] = color

    return rgba


def _is_valid_hex_color(value: Optional[str]) -> bool:
    """Return True if *value* is a valid #RGB or #RRGGBB hex color string."""
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s.startswith("#"):
        return False
    hex_digits = s[1:]
    if len(hex_digits) not in (3, 6):
        return False
    return all(ch in "0123456789abcdefABCDEF" for ch in hex_digits)


def _hex_to_rgb_float(hex_color: str) -> Tuple[float, float, float]:
    """Convert a #RGB or #RRGGBB hex string into normalized (r, g, b) floats in [0, 1]."""
    s = hex_color.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)


def render_method_comparison_rgba(
    segmentation_results: Dict[str, Dict[str, Optional[np.ndarray]]],
    compare_mode: str = "Nuclear Masks",
    selected_methods: Optional[Sequence[str]] = None,
    method_colors: Optional[Dict[str, str]] = None,
    thickness: int = 2,
    fill_alpha: float = 0.18,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """Render per-method color-coded boundaries and light interior fills into RGBA."""
    if selected_methods is not None:
        filtered_results = {
            m: segmentation_results[m]
            for m in selected_methods
            if m in segmentation_results
        }
    else:
        filtered_results = segmentation_results

    if not filtered_results:
        fallback_hw: Optional[Tuple[int, int]] = None
        for res in segmentation_results.values():
            if not isinstance(res, dict):
                continue
            for key in ("nuclear", "cytoplasm"):
                arr = res.get(key)
                if arr is not None and np.asarray(arr).ndim == 2:
                    fallback_hw = np.asarray(arr).shape[:2]
                    break
            if fallback_hw is not None:
                break
        if fallback_hw is None:
            raise ValueError(
                f"No valid segmentation masks found for compare_mode='{compare_mode}'."
            )
        return np.zeros((fallback_hw[0], fallback_hw[1], 4), dtype=np.float32), {}

    masks = _extract_masks_for_mode(filtered_results, compare_mode)
    h, w = next(iter(masks.values())).shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    resolved_colors: Dict[str, str] = {}
    for idx, method_name in enumerate(masks.keys()):
        custom = method_colors.get(method_name) if method_colors else None
        if _is_valid_hex_color(custom):
            resolved_colors[method_name] = str(custom).strip()
        else:
            resolved_colors[method_name] = METHOD_QC_COLORS[idx % len(METHOD_QC_COLORS)]

    alpha_fill = float(np.clip(fill_alpha, 0.0, 1.0))
    if alpha_fill > 0.0:
        for method_name, mask in masks.items():
            fg = mask > 0
            if not np.any(fg):
                continue
            r, g, b = _hex_to_rgb_float(resolved_colors[method_name])
            src_rgb = np.array([r, g, b], dtype=np.float32)
            dst_rgb = rgba[fg, :3]
            dst_a = rgba[fg, 3:4]
            out_a = alpha_fill + dst_a * (1.0 - alpha_fill)
            out_rgb = (
                dst_rgb * dst_a * (1.0 - alpha_fill) + src_rgb * alpha_fill
            ) / np.maximum(out_a, 1e-6)
            rgba[fg, :3] = out_rgb
            rgba[fg, 3:4] = out_a

    thick = max(1, int(thickness))
    for method_name, mask in masks.items():
        if not np.any(mask > 0):
            continue
        bounds = find_boundaries(mask, mode="inner")
        if not np.any(bounds):
            continue
        if thick > 1:
            bounds = binary_dilation(bounds, iterations=thick - 1)
        r, g, b = _hex_to_rgb_float(resolved_colors[method_name])
        rgba[bounds, 0] = r
        rgba[bounds, 1] = g
        rgba[bounds, 2] = b
        rgba[bounds, 3] = 1.0

    return np.clip(rgba, 0.0, 1.0).astype(np.float32), resolved_colors
