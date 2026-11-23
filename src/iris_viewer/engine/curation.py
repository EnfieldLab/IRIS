"""
IRIS-Viewer Headless Mask Curation & Adjudication Engine
========================================================

Pure NumPy/SciPy/scikit-image engine for building a unified ``Final Curated Mask``
from multi-model segmentation results via:
- Peer-consensus auto-adjudication across >= 3 models
- Selective splicing of a chosen model into disputed hotspot items, bounding boxes,
  or freehand/rectangular ROI polygons
- Strict 1-to-1 nuclear/cytoplasm ID pairing (N_nuc == N_cyto)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from skimage.draw import polygon as draw_polygon

from iris_viewer.engine.disagreement import (
    DisagreementReport,
    _compute_sparse_overlaps,
    _extract_masks_for_mode,
    _extract_method_info,
    _matched_instance_iou,
)

FINAL_CURATED_MASK_NAME = "Final Curated Mask"
NO_CELL_CHOICE = "No Cell"


@dataclass
class CuratedMaskState:
    """Mutable snapshot of the curated unified segmentation mask and provenance."""

    nuclear: np.ndarray
    cytoplasm: Optional[np.ndarray]
    base_method: str
    method_legend: List[str]
    source_map: np.ndarray
    curated_regions_count: int = 0
    cell_source_methods: Dict[int, str] = field(default_factory=dict)
    region_decisions: Dict[int, str] = field(default_factory=dict)
    roi_splices_count: int = 0
    cell_region_ids: Dict[int, int] = field(default_factory=dict)


def _sanitize_uint32_mask(mask: np.ndarray) -> np.ndarray:
    """Return a contiguous 2D uint32 copy with negative labels clamped to 0."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label mask, got shape {arr.shape}.")
    if np.issubdtype(arr.dtype, np.signedinteger) or np.issubdtype(arr.dtype, np.floating):
        if np.any(arr < 0):
            arr = np.where(arr > 0, arr, 0)
    return arr.astype(np.uint32, copy=True)


def _remap_with_sorted_ids(
    arr: np.ndarray,
    pos_ids: np.ndarray,
    new_ids: np.ndarray,
) -> np.ndarray:
    """Remap labels in *arr* using sorted *pos_ids* -> *new_ids*; unknown labels -> 0."""
    if arr.size == 0 or pos_ids.size == 0:
        return np.zeros(arr.shape, dtype=np.uint32)
    max_pos = int(pos_ids[-1])
    if max_pos <= 1_000_000:
        lut = np.zeros(max_pos + 1, dtype=np.uint32)
        lut[pos_ids] = new_ids
        out = np.zeros(arr.shape, dtype=np.uint32)
        in_range = arr <= max_pos
        out[in_range] = lut[arr[in_range]]
        return out

    out = np.zeros(arr.shape, dtype=np.uint32)
    nz = arr > 0
    vals = arr[nz]
    idx = np.searchsorted(pos_ids, vals)
    clipped = np.minimum(idx, pos_ids.size - 1)
    valid = (idx < pos_ids.size) & (pos_ids[clipped] == vals)
    mapped = np.zeros_like(vals, dtype=np.uint32)
    mapped[valid] = new_ids[idx[valid]]
    out[nz] = mapped
    return out


def _relabel_paired_with_map(
    nuc: np.ndarray,
    cyto: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[int, int]]:
    """Relabel *nuc* contiguously 1..N uint32 and synchronize *cyto* so N_nuc == N_cyto."""
    nuc_u = _sanitize_uint32_mask(nuc)
    uniq_nuc = np.unique(nuc_u)
    pos_nuc = uniq_nuc[uniq_nuc > 0].astype(np.uint32, copy=False)

    if pos_nuc.size == 0:
        nuc_out = np.zeros(nuc_u.shape, dtype=np.uint32)
        cyto_out = np.zeros(nuc_u.shape, dtype=np.uint32) if cyto is not None else None
        return nuc_out, cyto_out, {}

    n_cells = int(pos_nuc.size)
    new_ids = np.arange(1, n_cells + 1, dtype=np.uint32)
    already_contiguous = int(pos_nuc[0]) == 1 and int(pos_nuc[-1]) == n_cells

    if already_contiguous:
        nuc_out = nuc_u
    else:
        nuc_out = _remap_with_sorted_ids(nuc_u, pos_nuc, new_ids)

    old_to_new = {int(old_id): int(new_id) for old_id, new_id in zip(pos_nuc, new_ids)}

    if cyto is None:
        return nuc_out, None, old_to_new

    cyto_u = _sanitize_uint32_mask(cyto)
    if cyto_u.shape != nuc_out.shape:
        raise ValueError(
            f"Cytoplasm mask shape {cyto_u.shape} does not match nuclear shape {nuc_out.shape}."
        )
    cyto_out = _remap_with_sorted_ids(cyto_u, pos_nuc, new_ids)

    counts = np.bincount(cyto_out.ravel(), minlength=n_cells + 1)
    missing = np.flatnonzero(counts[1:] == 0) + 1
    if missing.size > 0:
        miss_lut = np.zeros(n_cells + 1, dtype=bool)
        miss_lut[missing] = True
        fill = miss_lut[nuc_out]
        cyto_out[fill] = nuc_out[fill]

    return nuc_out, cyto_out, old_to_new


def _ensure_paired_masks(
    nuc: np.ndarray,
    cyto: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Enforce contiguous 1..N uint32 IDs and 1-to-1 nuclear/cytoplasm matching."""
    nuc_out, cyto_out, _ = _relabel_paired_with_map(nuc, cyto)
    return nuc_out, cyto_out


def _compute_centroids(mask: np.ndarray) -> Dict[int, Tuple[float, float]]:
    """Compute vectorized (cy, cx) centroids for all positive cell labels in *mask*."""
    if mask.size == 0:
        return {}
    max_lbl = int(mask.max())
    if max_lbl <= 0:
        return {}
    y_nz, x_nz = np.nonzero(mask > 0)
    lbl_nz = mask[y_nz, x_nz]
    counts = np.bincount(lbl_nz, minlength=max_lbl + 1)
    sum_y = np.bincount(lbl_nz, weights=y_nz.astype(np.float64), minlength=max_lbl + 1)
    sum_x = np.bincount(lbl_nz, weights=x_nz.astype(np.float64), minlength=max_lbl + 1)

    centroids: Dict[int, Tuple[float, float]] = {}
    for cid in np.flatnonzero(counts[1:] > 0) + 1:
        c = int(cid)
        centroids[c] = (float(sum_y[c] / counts[c]), float(sum_x[c] / counts[c]))
    return centroids


def initialize_curated_mask(
    segmentation_results: Dict[str, Dict[str, Optional[np.ndarray]]],
    base_method: str,
) -> CuratedMaskState:
    """Initialize a ``CuratedMaskState`` by copying paired masks from *base_method*."""
    if base_method not in segmentation_results:
        raise ValueError(f"Base method '{base_method}' not found in segmentation results.")
    entry = segmentation_results[base_method]
    if not isinstance(entry, dict) or entry.get("nuclear") is None:
        raise ValueError(f"Base method '{base_method}' has no nuclear mask.")

    nuc_u32, cyto_u32 = _ensure_paired_masks(entry["nuclear"], entry.get("cytoplasm"))

    fg = nuc_u32 > 0
    if cyto_u32 is not None:
        fg = fg | (cyto_u32 > 0)

    source_map = np.zeros(nuc_u32.shape, dtype=np.uint16)
    source_map[fg] = 1

    max_id = int(nuc_u32.max()) if nuc_u32.size > 0 else 0
    cell_source_methods = {cid: str(base_method) for cid in range(1, max_id + 1)}

    return CuratedMaskState(
        nuclear=nuc_u32,
        cytoplasm=cyto_u32,
        base_method=str(base_method),
        method_legend=[str(base_method)],
        source_map=source_map,
        curated_regions_count=0,
        cell_source_methods=cell_source_methods,
    )


def _extract_region_id(item: Any) -> Optional[int]:
    raw = item.get("region_id") if isinstance(item, dict) else getattr(item, "region_id", None)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _extract_item_fields(item: Any) -> Tuple[Tuple[int, int, int, int], Dict[str, List[int]]]:
    if isinstance(item, dict):
        bbox = item.get("bbox", (0, 0, 0, 0))
        m_ids = dict(item.get("method_cell_ids", {}))
    else:
        bbox = getattr(item, "bbox", (0, 0, 0, 0))
        m_ids = dict(getattr(item, "method_cell_ids", {}))
    r0, c0, r1, c1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    return (r0, c0, r1, c1), m_ids


def _normalize_polygons(polygon_yx: Any) -> List[np.ndarray]:
    if polygon_yx is None:
        return []
    if isinstance(polygon_yx, np.ndarray):
        if polygon_yx.ndim == 2 and polygon_yx.shape[0] >= 3 and polygon_yx.shape[1] >= 2:
            return [polygon_yx.astype(float, copy=False)]
        if polygon_yx.ndim == 3 and polygon_yx.shape[1] >= 3 and polygon_yx.shape[2] >= 2:
            return [polygon_yx[i].astype(float, copy=False) for i in range(polygon_yx.shape[0])]
        return []

    if isinstance(polygon_yx, (list, tuple)):
        if not polygon_yx:
            return []
        first = polygon_yx[0]
        if isinstance(first, (int, float, np.number)) or (
            isinstance(first, (list, tuple, np.ndarray))
            and np.asarray(first).ndim == 1
            and len(first) in (2, 3)
        ):
            arr = np.asarray(polygon_yx, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 2:
                return [arr]
            return []
        polys: List[np.ndarray] = []
        for elem in polygon_yx:
            p_arr = np.asarray(elem, dtype=float)
            if p_arr.ndim == 2 and p_arr.shape[0] >= 3 and p_arr.shape[1] >= 2:
                polys.append(p_arr)
        return polys

    return []


def _centroid_in_roi(cy: float, cx: float, roi_mask: np.ndarray) -> bool:
    h, w = roi_mask.shape
    if h == 0 or w == 0:
        return False
    ry = min(h - 1, max(0, int(round(cy))))
    rx = min(w - 1, max(0, int(round(cx))))
    return bool(roi_mask[ry, rx])


def splice_method_into_regions(
    curated: CuratedMaskState,
    segmentation_results: Dict[str, Dict[str, Optional[np.ndarray]]],
    source_method: Optional[str],
    *,
    items: Optional[Sequence[Any]] = None,
    bboxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    polygon_yx: Optional[Any] = None,
) -> CuratedMaskState:
    """Splice cells from *source_method* (or clear cells for ``NO_CELL_CHOICE``) in regions."""
    raw_method = "" if source_method is None else str(source_method).strip()
    is_no_cell = raw_method in ("", NO_CELL_CHOICE)
    decision_label = NO_CELL_CHOICE if is_no_cell else raw_method

    h, w = curated.nuclear.shape[:2]
    src_nuc: Optional[np.ndarray] = None
    src_cyto: Optional[np.ndarray] = None
    if not is_no_cell:
        if decision_label not in segmentation_results:
            raise ValueError(
                f"Source method '{decision_label}' not found in segmentation results."
            )
        src_entry = segmentation_results[decision_label]
        if not isinstance(src_entry, dict) or src_entry.get("nuclear") is None:
            raise ValueError(f"Source method '{decision_label}' has no nuclear mask.")

        src_nuc = _sanitize_uint32_mask(src_entry["nuclear"])
        if src_nuc.shape != (h, w):
            raise ValueError(
                f"Source nuclear shape {src_nuc.shape} does not match curated shape {(h, w)}."
            )

        src_cyto_raw = src_entry.get("cytoplasm")
        if src_cyto_raw is not None:
            src_cyto = _sanitize_uint32_mask(src_cyto_raw)
            if src_cyto.shape != (h, w):
                raise ValueError(
                    f"Source cytoplasm shape {src_cyto.shape} does not match curated shape "
                    f"{(h, w)}."
                )

    roi_mask = np.zeros((h, w), dtype=bool)
    disputed_base_ids: set[int] = set()
    item_src_ids: set[int] = set()
    target_rids: set[int] = set()
    item_entries: List[Tuple[Optional[int], np.ndarray]] = []

    if items is not None:
        # Precompute sanitized masks, centroids, and pixel areas across all methods once
        method_cache: Dict[
            str,
            Tuple[np.ndarray, Optional[np.ndarray], Dict[int, Tuple[float, float]], np.ndarray],
        ] = {}
        for m_name, m_entry in segmentation_results.items():
            if m_name == FINAL_CURATED_MASK_NAME or not isinstance(m_entry, dict):
                continue
            m_nuc_raw = m_entry.get("nuclear")
            m_cyto_raw = m_entry.get("cytoplasm")
            m_main_raw = m_nuc_raw if m_nuc_raw is not None else m_cyto_raw
            if m_main_raw is None or np.asarray(m_main_raw).shape[:2] != (h, w):
                continue
            m_main = _sanitize_uint32_mask(m_main_raw)
            m_cyto = (
                _sanitize_uint32_mask(m_cyto_raw)
                if m_cyto_raw is not None and np.asarray(m_cyto_raw).shape[:2] == (h, w)
                else None
            )
            method_cache[m_name] = (
                m_main,
                m_cyto,
                _compute_centroids(m_main),
                np.bincount(m_main.ravel()),
            )

        for item in items:
            rid = _extract_region_id(item)
            if rid is not None:
                target_rids.add(rid)
            (r0, c0, r1, c1), m_ids = _extract_item_fields(item)
            item_mask = np.zeros((h, w), dtype=bool)
            cr0 = max(0, min(r0, h))
            cc0 = max(0, min(c0, w))
            cr1 = max(0, min(r1, h))
            cc1 = max(0, min(c1, w))
            if cr1 > cr0 and cc1 > cc0:
                item_mask[cr0:cr1, cc0:cc1] = True

            for m_name, cids in m_ids.items():
                if not cids:
                    continue
                if m_name == curated.base_method:
                    disputed_base_ids.update(int(c) for c in cids)
                if not is_no_cell and m_name == decision_label:
                    item_src_ids.update(int(c) for c in cids)
                if m_name in method_cache:
                    m_main, m_cyto, _, _ = method_cache[m_name]
                    hit_mask = np.isin(m_main, cids)
                    if not np.any(hit_mask) and m_cyto is not None:
                        hit_mask = np.isin(m_cyto, cids)
                    item_mask |= hit_mask

            # Union competing cell masks across ALL methods (even unchecked/unlisted ones)
            nz_y, nz_x = np.nonzero(item_mask)
            if nz_y.size > 0:
                competing_unions = np.zeros((h, w), dtype=bool)
                for m_name, (m_main, _m_cyto, m_centroids, m_areas) in method_cache.items():
                    if m_name in m_ids and m_ids[m_name]:
                        continue
                    overlap_vals = m_main[nz_y, nz_x]
                    pos_vals = overlap_vals[overlap_vals > 0]
                    if pos_vals.size == 0:
                        continue
                    counts = np.bincount(pos_vals)
                    comp_ids: List[int] = []
                    for cid in np.flatnonzero(counts > 0):
                        cid_int = int(cid)
                        cy, cx = m_centroids.get(cid_int, (-1.0, -1.0))
                        area = float(m_areas[cid_int]) if cid_int < m_areas.size else 0.0
                        if _centroid_in_roi(cy, cx, item_mask) or (
                            area > 0.0 and float(counts[cid_int]) >= 0.25 * area
                        ):
                            comp_ids.append(cid_int)
                    if comp_ids:
                        competing_unions |= np.isin(m_main, comp_ids)
                item_mask |= competing_unions

            roi_mask |= item_mask
            if rid is not None:
                item_entries.append((rid, item_mask))

    has_bbox_or_poly_splice = False
    if bboxes is not None:
        for bbox in bboxes:
            r0, c0, r1, c1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            cr0 = max(0, min(r0, h))
            cc0 = max(0, min(c0, w))
            cr1 = max(0, min(r1, h))
            cc1 = max(0, min(c1, w))
            if cr1 > cr0 and cc1 > cc0:
                roi_mask[cr0:cr1, cc0:cc1] = True
                has_bbox_or_poly_splice = True

    for poly in _normalize_polygons(polygon_yx):
        rr, cc = draw_polygon(poly[:, -2], poly[:, -1], shape=(h, w))
        roi_mask[rr, cc] = True
        has_bbox_or_poly_splice = True

    cur_nuc = _sanitize_uint32_mask(curated.nuclear)
    cur_cyto = _sanitize_uint32_mask(curated.cytoplasm) if curated.cytoplasm is not None else None

    method_legend = list(curated.method_legend)
    src_code = 0
    if not is_no_cell:
        if decision_label not in method_legend:
            method_legend.append(decision_label)
        src_code = int(method_legend.index(decision_label) + 1)

    cur_centroids = _compute_centroids(cur_nuc)
    src_centroids = _compute_centroids(src_nuc) if src_nuc is not None else {}

    # Identify curated cells to remove
    remove_cur_ids: set[int] = set()
    base_entry = segmentation_results.get(curated.base_method, {})
    base_nuc_raw = base_entry.get("nuclear") if isinstance(base_entry, dict) else None
    base_unshifted = (
        curated.curated_regions_count == 0
        and base_nuc_raw is not None
        and np.asarray(base_nuc_raw).shape == (h, w)
        and int(np.max(cur_nuc)) == int(np.max(base_nuc_raw))
    )
    prev_cell_regions = dict(getattr(curated, "cell_region_ids", {}) or {})
    for cid, (cy, cx) in cur_centroids.items():
        if _centroid_in_roi(cy, cx, roi_mask):
            remove_cur_ids.add(cid)
        elif base_unshifted and cid in disputed_base_ids:
            remove_cur_ids.add(cid)
        elif target_rids and prev_cell_regions.get(cid) in target_rids:
            remove_cur_ids.add(cid)

    # Identify source cells to insert
    insert_src_ids: List[int] = []
    if not is_no_cell:
        insert_src_ids = sorted(
            sid
            for sid, (cy, cx) in src_centroids.items()
            if _centroid_in_roi(cy, cx, roi_mask) or sid in item_src_ids
        )

    keep_cur_ids = sorted(cid for cid in cur_centroids if cid not in remove_cur_ids)
    max_cur_id = int(cur_nuc.max()) if cur_nuc.size > 0 else 0
    if keep_cur_ids and max_cur_id > 0:
        keep_lut = np.zeros(max_cur_id + 1, dtype=bool)
        keep_lut[keep_cur_ids] = True
        retained_nuc = np.where(keep_lut[cur_nuc], cur_nuc, 0).astype(np.uint32)
        if cur_cyto is not None:
            retained_cyto: Optional[np.ndarray] = np.where(
                keep_lut[cur_cyto], cur_cyto, 0
            ).astype(np.uint32)
        else:
            retained_cyto = None
    else:
        retained_nuc = np.zeros((h, w), dtype=np.uint32)
        retained_cyto = np.zeros((h, w), dtype=np.uint32) if cur_cyto is not None else None

    fg_retained = retained_nuc > 0
    if retained_cyto is not None:
        fg_retained = fg_retained | (retained_cyto > 0)
    new_source_map = np.where(fg_retained, curated.source_map, 0).astype(np.uint16)

    temp_method_map: Dict[int, str] = {
        cid: curated.cell_source_methods.get(cid, curated.base_method) for cid in keep_cur_ids
    }
    temp_region_map: Dict[int, int] = {
        cid: prev_cell_regions[cid] for cid in keep_cur_ids if cid in prev_cell_regions
    }

    merged_nuc = retained_nuc.copy()
    merged_cyto = retained_cyto.copy() if retained_cyto is not None else None

    if insert_src_ids and src_nuc is not None:
        offset = int(merged_nuc.max()) if merged_nuc.size > 0 else 0
        src_pos_arr = np.array(insert_src_ids, dtype=np.uint32)
        temp_ids_arr = np.arange(offset + 1, offset + 1 + len(insert_src_ids), dtype=np.uint32)

        single_rid = item_entries[0][0] if len(item_entries) == 1 else None
        for sid, temp_id in zip(insert_src_ids, temp_ids_arr):
            temp_method_map[int(temp_id)] = decision_label
            if single_rid is not None:
                temp_region_map[int(temp_id)] = single_rid
            elif item_entries:
                cy, cx = src_centroids.get(sid, (-1.0, -1.0))
                for e_rid, e_mask in item_entries:
                    if e_rid is not None and _centroid_in_roi(cy, cx, e_mask):
                        temp_region_map[int(temp_id)] = e_rid
                        break

        cand_src_nuc = _remap_with_sorted_ids(src_nuc, src_pos_arr, temp_ids_arr)
        place_nuc = (cand_src_nuc > 0) & (retained_nuc == 0)
        merged_nuc[place_nuc] = cand_src_nuc[place_nuc]
        new_source_map[place_nuc] = src_code

        if merged_cyto is not None:
            # Clear retained cytoplasm strictly under newly inserted nuclei
            merged_cyto[place_nuc] = 0
            if src_cyto is not None:
                cand_src_cyto = _remap_with_sorted_ids(src_cyto, src_pos_arr, temp_ids_arr)
                place_cyto = (cand_src_cyto > 0) & (merged_nuc == 0) & (merged_cyto == 0)
                merged_cyto[place_cyto] = cand_src_cyto[place_cyto]
                new_source_map[place_cyto] = src_code

                own_nuc_cyto = place_nuc & (cand_src_cyto == cand_src_nuc)
                merged_cyto[own_nuc_cyto] = cand_src_nuc[own_nuc_cyto]
            else:
                merged_cyto[place_nuc] = cand_src_nuc[place_nuc]

    final_nuc, final_cyto, old_to_new = _relabel_paired_with_map(merged_nuc, merged_cyto)

    fg_final = final_nuc > 0
    if final_cyto is not None:
        fg_final = fg_final | (final_cyto > 0)
    new_source_map = np.where(fg_final, new_source_map, 0).astype(np.uint16)

    new_cell_source_methods: Dict[int, str] = {
        new_id: temp_method_map.get(old_id, curated.base_method)
        for old_id, new_id in old_to_new.items()
    }
    new_cell_region_ids: Dict[int, int] = {
        new_id: temp_region_map[old_id]
        for old_id, new_id in old_to_new.items()
        if old_id in temp_region_map
    }

    new_region_decisions = dict(getattr(curated, "region_decisions", {}) or {})
    if items is not None:
        for item in items:
            rid = _extract_region_id(item)
            if rid is not None:
                new_region_decisions[rid] = decision_label

    new_roi_splices = int(getattr(curated, "roi_splices_count", 0))
    if has_bbox_or_poly_splice or (items is not None and len(items) > 0 and not target_rids):
        new_roi_splices += 1

    return CuratedMaskState(
        nuclear=final_nuc,
        cytoplasm=final_cyto,
        base_method=curated.base_method,
        method_legend=method_legend,
        source_map=new_source_map,
        curated_regions_count=len(new_region_decisions) + new_roi_splices,
        cell_source_methods=new_cell_source_methods,
        region_decisions=new_region_decisions,
        roi_splices_count=new_roi_splices,
        cell_region_ids=new_cell_region_ids,
    )


def auto_adjudicate_consensus_mask(
    segmentation_results: Dict[str, Dict[str, Optional[np.ndarray]]],
    report: DisagreementReport,
    min_peer_iou: float = 0.60,
    existing_curated: Optional[CuratedMaskState] = None,
) -> CuratedMaskState:
    """Resolve uncurated hotspots via >= 3 model peer consensus, preserving manual picks."""
    candidates = [
        m
        for m in (getattr(report, "methods", None) or list(segmentation_results.keys()))
        if m != FINAL_CURATED_MASK_NAME
        and m in segmentation_results
        and isinstance(segmentation_results[m], dict)
        and segmentation_results[m].get("nuclear") is not None
    ]

    ref_method = getattr(report, "reference_method", None)
    if not ref_method or ref_method == FINAL_CURATED_MASK_NAME or ref_method not in candidates:
        ref_method = candidates[0] if candidates else ""

    curated = initialize_curated_mask(segmentation_results, ref_method)

    items = list(getattr(report, "items", []) or [])
    manual_decisions: Dict[int, str] = {}
    if existing_curated is not None and getattr(existing_curated, "region_decisions", None):
        for rid, dec in existing_curated.region_decisions.items():
            dec_str = str(dec)
            if dec_str and not dec_str.startswith("Consensus:"):
                manual_decisions[int(rid)] = dec_str

    item_by_rid: Dict[int, Any] = {}
    for item in items:
        rid = _extract_region_id(item)
        if rid is not None:
            item_by_rid[rid] = item

    if len(candidates) >= 3 and items:
        compare_mode = getattr(report, "compare_mode", "Nuclear Masks")
        cand_seg = {m: segmentation_results[m] for m in candidates}
        try:
            cand_masks = _extract_masks_for_mode(cand_seg, compare_mode)
        except ValueError:
            cand_masks = _extract_masks_for_mode(cand_seg, "Nuclear Masks")

        h, w = curated.nuclear.shape[:2]
        n_cand = len(candidates)

        for item in items:
            rid = _extract_region_id(item)
            if rid is not None and rid in manual_decisions:
                continue

            (r0, c0, r1, c1), m_ids = _extract_item_fields(item)
            cr0 = max(0, min(r0, h))
            cc0 = max(0, min(c0, w))
            cr1 = max(0, min(r1, h))
            cc1 = max(0, min(c1, w))
            if cr1 <= cr0 or cc1 <= cc0:
                continue

            patch_infos = {}
            for m in candidates:
                cids = m_ids.get(m, [])
                if cids:
                    sub = cand_masks[m][cr0:cr1, cc0:cc1]
                    patch = np.where(np.isin(sub, cids), sub, 0)
                else:
                    patch = np.zeros((cr1 - cr0, cc1 - cc0), dtype=np.int64)
                patch_infos[m] = _extract_method_info(patch)

            pair_iou = np.ones((n_cand, n_cand), dtype=np.float64)
            for i in range(n_cand):
                mi = candidates[i]
                for j in range(i + 1, n_cand):
                    mj = candidates[j]
                    _, _, edges = _compute_sparse_overlaps(patch_infos[mi], patch_infos[mj])
                    val = _matched_instance_iou(patch_infos[mi], patch_infos[mj], edges)
                    pair_iou[i, j] = val
                    pair_iou[j, i] = val

            scores: Dict[str, float] = {}
            for i, m in enumerate(candidates):
                peer_vals = [pair_iou[i, j] for j in range(n_cand) if j != i]
                scores[m] = float(np.mean(peer_vals)) if peer_vals else 0.0

            ref_score = scores.get(ref_method, 0.0)
            best_method = max(
                candidates, key=lambda m: (scores[m], 1 if m == ref_method else 0)
            )
            best_score = scores[best_method]

            if (
                best_method != ref_method
                and best_score >= float(min_peer_iou)
                and best_score > ref_score + 0.05
            ):
                curated = splice_method_into_regions(
                    curated,
                    segmentation_results,
                    best_method,
                    items=[item],
                )
                if rid is not None:
                    curated.region_decisions[rid] = f"Consensus: {best_method}"

    for rid, dec in manual_decisions.items():
        item = item_by_rid.get(rid)
        if item is None:
            continue
        if dec != NO_CELL_CHOICE and (
            dec not in segmentation_results
            or not isinstance(segmentation_results[dec], dict)
            or segmentation_results[dec].get("nuclear") is None
        ):
            continue
        curated = splice_method_into_regions(
            curated,
            segmentation_results,
            dec,
            items=[item],
        )

    return curated


__all__ = [
    "FINAL_CURATED_MASK_NAME",
    "NO_CELL_CHOICE",
    "CuratedMaskState",
    "_ensure_paired_masks",
    "initialize_curated_mask",
    "splice_method_into_regions",
    "auto_adjudicate_consensus_mask",
]
