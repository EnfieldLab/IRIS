"""Pipeline Service for Segmentation Sequences."""
from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from iris_common.logging import iris_log, log_error
from iris_viewer.engine.segmentation import get_available_backends
from iris_viewer.engine.segmentation.base import reconcile_nuclear_and_cytoplasm_masks

_DILATION_RULES = (
    "Cytoplasm (Dilate Remaining)",
    "Cytoplasm Expansion (Dilate / Watershed)",
)


@dataclass
class SegmentationStep:
    method_name: str
    params: dict
    rule: str = "Independent"


@dataclass
class SegmentationSequence:
    steps: List[SegmentationStep]
    auto_complete_cytoplasm: Optional[bool] = None
    fallback_dilation_radius: int = 5
    fallback_dilation_mode: str = "distance"
    fallback_membrane_channels: Optional[List[int]] = None


def _count_labels(mask: Optional[np.ndarray]) -> int:
    """Count distinct positive instance labels in *mask*."""
    if mask is None or mask.size == 0:
        return 0
    return int((np.unique(mask) > 0).sum())


def _extract_channel_2d(image: np.ndarray, ch_idx: int) -> np.ndarray:
    """Materialize a single 2D channel plane from a NumPy or lazy Dask/Zarr array."""
    if image.ndim == 2:
        plane = image
    else:
        idx = max(0, min(int(ch_idx), int(image.shape[0]) - 1))
        plane = image[idx, :, :]
    if hasattr(plane, "compute"):
        plane = plane.compute()
    return np.asarray(plane)


def _extract_max_membrane_2d(image: np.ndarray, mem_cs: list) -> Optional[np.ndarray]:
    """Materialize and max-project selected membrane channels into a 2D array."""
    if not mem_cs or image.ndim < 3:
        return None
    valid_indices = [int(c) for c in mem_cs if 0 <= int(c) < int(image.shape[0])]
    if not valid_indices:
        return None
    planes = [_extract_channel_2d(image, c) for c in valid_indices]
    if len(planes) == 1:
        return planes[0]
    return np.max(np.stack(planes, axis=0), axis=0)


def _merge_label_masks(
    base_mask: Optional[np.ndarray],
    new_mask: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Merge a new label mask into an existing one without label collisions."""
    if new_mask is None:
        return base_mask
    if base_mask is None:
        return new_mask.copy()
    max_label = int(base_mask.max())
    shifted = new_mask.copy()
    shifted[shifted > 0] += max_label
    fill = (base_mask == 0) & (shifted > 0)
    merged = base_mask.copy()
    merged[fill] = shifted[fill]
    return merged


def _prepare_nuclear_inputs(
    image: np.ndarray,
    params: dict,
) -> Tuple[np.ndarray, dict]:
    """Materialize only the selected nuclear channel into a (1, H, W) array."""
    kw = params.copy()
    kw.setdefault("diameter", None)
    nuc_c = int(kw.get("nuclear_channel", 0))
    kw.pop("membrane_channels", None)
    kw["membrane_channel"] = None
    nuc_plane = _extract_channel_2d(image, nuc_c)
    kw["nuclear_channel"] = 0
    return nuc_plane[np.newaxis, ...], kw


def _prepare_independent_inputs(
    image: np.ndarray,
    params: dict,
) -> Tuple[np.ndarray, dict]:
    """Build input image and kwargs for an Independent segmentation step."""
    kw = params.copy()
    kw.setdefault("diameter", None)
    nuc_c = int(kw.get("nuclear_channel", 0))
    mem_cs = kw.pop("membrane_channels", None)
    if mem_cs and image.ndim >= 3:
        mem_img = _extract_max_membrane_2d(image, mem_cs)
        nuc_img = _extract_channel_2d(image, nuc_c)
        if mem_img is not None:
            seg_img = np.stack([nuc_img, mem_img], axis=0)
            kw["nuclear_channel"] = 0
            kw["membrane_channel"] = 1
        elif hasattr(image, "compute"):
            seg_img = nuc_img[np.newaxis, ...]
            kw["nuclear_channel"] = 0
            kw.setdefault("membrane_channel", None)
        else:
            seg_img = image
            kw.setdefault("membrane_channel", None)
    elif hasattr(image, "compute"):
        nuc_img = _extract_channel_2d(image, nuc_c)
        seg_img = nuc_img[np.newaxis, ...]
        kw["nuclear_channel"] = 0
        kw.setdefault("membrane_channel", None)
    else:
        seg_img = image
        kw.setdefault("membrane_channel", None)
    return seg_img, kw


def _prepare_sequential_cytoplasm_inputs(
    image: np.ndarray,
    params: dict,
) -> Tuple[np.ndarray, dict]:
    """Build 2-channel [nuclear, max-projected membrane] image for sequential cytoplasm."""
    nuc_c = int(params.get("nuclear_channel", 0))
    mem_cs = params.get("membrane_channels", [])
    nuc_plane = _extract_channel_2d(image, nuc_c)
    mem_img = _extract_max_membrane_2d(image, mem_cs)
    has_valid_mem = mem_img is not None
    if mem_img is None:
        mem_img = np.zeros_like(nuc_plane)
    synth_img = np.stack([nuc_plane, mem_img], axis=0)
    kw = params.copy()
    kw.setdefault("diameter", None)
    kw.pop("membrane_channels", None)
    kw["nuclear_channel"] = 0
    kw["membrane_channel"] = 1
    if has_valid_mem:
        kw["skip_nuclear"] = True
    return synth_img, kw


def _clamp_roi_bbox(
    roi_bbox: Tuple[int, int, int, int],
    image_hw: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Normalize and clamp (y0, x0, y1, x1) bounding box within image height/width."""
    h, w = int(image_hw[0]), int(image_hw[1])
    y0, x0, y1, x1 = (int(roi_bbox[0]), int(roi_bbox[1]), int(roi_bbox[2]), int(roi_bbox[3]))
    if y0 > y1:
        y0, y1 = y1, y0
    if x0 > x1:
        x0, x1 = x1, x0
    y0 = max(0, min(y0, h))
    y1 = max(0, min(y1, h))
    x0 = max(0, min(x0, w))
    x1 = max(0, min(x1, w))
    return (y0, x0, y1, x1)


def _dilate_remaining_cytoplasm(
    global_nuc: np.ndarray,
    global_cyto: Optional[np.ndarray],
    radius: int,
    mode: str = "distance",
    membrane_image: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Dilate nuclear labels lacking cytoplasm coverage up to *radius* pixels."""
    if global_cyto is None:
        out_dtype = (
            global_nuc.dtype if np.issubdtype(global_nuc.dtype, np.integer) else np.uint32
        )
        global_cyto = np.zeros(global_nuc.shape, dtype=out_dtype)

    matched_ids = np.unique(global_cyto[global_cyto > 0])
    matched_ids = matched_ids[matched_ids > 0]
    if matched_ids.size > 0:
        unseg_mask = (global_nuc > 0) & np.isin(global_nuc, matched_ids, invert=True)
    else:
        unseg_mask = global_nuc > 0

    if not np.any(unseg_mask):
        return None, global_cyto

    unsegmented_nuc_mask = np.where(unseg_mask, global_nuc, 0)
    mode_norm = (mode or "distance").strip().lower()
    radius_px = max(0, int(radius))

    if (
        mode_norm == "watershed"
        and membrane_image is not None
        and np.asarray(membrane_image).shape == global_nuc.shape
    ):
        from scipy.ndimage import gaussian_filter
        from skimage.segmentation import expand_labels, watershed

        support = expand_labels(unsegmented_nuc_mask, distance=radius_px) > 0
        mem = np.asarray(membrane_image, dtype=np.float32)
        smoothed = gaussian_filter(mem, sigma=1.0)
        elevation = float(smoothed.max()) - smoothed
        new_cyto = watershed(
            elevation,
            markers=unsegmented_nuc_mask,
            mask=support,
        ).astype(global_cyto.dtype, copy=False)
    else:
        from scipy.ndimage import distance_transform_edt

        dists, (nearest_y, nearest_x) = distance_transform_edt(
            unsegmented_nuc_mask == 0,
            return_indices=True,
        )
        new_cyto = unsegmented_nuc_mask.copy()
        expand_mask = (dists > 0) & (dists <= radius_px)
        new_cyto[expand_mask] = unsegmented_nuc_mask[
            nearest_y[expand_mask], nearest_x[expand_mask]
        ]

    other_nuc = (global_nuc > 0) & (~unseg_mask)
    valid_expansion = (global_cyto == 0) & (new_cyto > 0) & (~other_nuc)
    global_cyto[valid_expansion] = new_cyto[valid_expansion]
    global_cyto[unseg_mask] = global_nuc[unseg_mask]
    return new_cyto, global_cyto


def _assign_sequential_cytoplasm(
    global_nuc: np.ndarray,
    global_cyto: Optional[np.ndarray],
    new_cyto: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Map cytoplasm labels onto unclaimed nuclear labels via reconciled overlap."""
    if global_cyto is None:
        out_dtype = (
            global_nuc.dtype if np.issubdtype(global_nuc.dtype, np.integer) else np.uint32
        )
        global_cyto = np.zeros(global_nuc.shape, dtype=out_dtype)
    if new_cyto is None:
        return None, global_cyto

    claimed_ids = np.unique(global_cyto[global_cyto > 0])
    claimed_ids = claimed_ids[claimed_ids > 0]

    _, reconciled = reconcile_nuclear_and_cytoplasm_masks(global_nuc, new_cyto)
    if reconciled is None or int(reconciled.max()) == 0:
        return np.zeros_like(global_cyto), global_cyto

    step_cyto = reconciled.copy()
    if claimed_ids.size > 0:
        max_id = max(int(step_cyto.max()), int(claimed_ids.max()))
        keep_lut = np.ones(max_id + 1, dtype=bool)
        keep_lut[0] = False
        keep_lut[claimed_ids] = False
        step_cyto = np.where(keep_lut[step_cyto], step_cyto, np.uint32(0))

    # Protect already-assigned cytoplasm and any neighbor nuclei from overwrite.
    step_cyto[global_cyto > 0] = 0
    other_nuc = (global_nuc > 0) & (global_nuc != step_cyto)
    step_cyto[other_nuc] = 0

    write_mask = step_cyto > 0
    global_cyto[write_mask] = step_cyto[write_mask]
    return step_cyto.astype(global_cyto.dtype, copy=False), global_cyto


def iter_execute_sequence(
    group_name: str,
    sequence: SegmentationSequence,
    image: np.ndarray,
    backends: Dict,
    device: str,
    gpu_lock: Optional[threading.Lock] = None,
    log_prefix: str = "[Pipeline]",
) -> Iterator[tuple]:
    """Execute one segmentation sequence step-by-step, yielding progress tuples."""
    global_nuc: Optional[np.ndarray] = None
    global_cyto: Optional[np.ndarray] = None
    n_steps = len(sequence.steps)

    def _call_segment(backend, seg_img: np.ndarray, kw: dict) -> dict:
        if gpu_lock is not None:
            with gpu_lock:
                return backend.segment(image=seg_img, device=device, **kw)
        return backend.segment(image=seg_img, device=device, **kw)

    tag = f"{log_prefix} " if log_prefix and log_prefix != "[Pipeline]" else ""
    iris_log(
        "IRIS-Viewer",
        "Segmentation",
        f"{tag}[{group_name}] Starting group ({n_steps} step(s), device={device})...",
    )

    for step_idx, step in enumerate(sequence.steps):
        step_label = "Dilation" if step.rule in _DILATION_RULES else step.method_name
        yield ("step_start", group_name, step_idx, step_label)
        t0 = time.perf_counter()

        try:
            if step.rule == "Independent":
                backend = backends.get(step.method_name)
                if not backend:
                    err = f"Backend {step.method_name} not available"
                    log_error(
                        "IRIS-Viewer",
                        "Segmentation",
                        f"{tag}[{group_name}] Step {step_idx + 1}/{n_steps} failed: {err}",
                    )
                    yield ("error", group_name, step_idx, step.method_name, err)
                    continue

                nuc_orig = step.params.get("nuclear_channel", 0)
                mem_cs = step.params.get("membrane_channels", [])
                iris_log(
                    "IRIS-Viewer",
                    "Segmentation",
                    f"{tag}[{group_name}] Step {step_idx + 1}/{n_steps}: "
                    f"'Independent' ({step.method_name}, Nuc:{nuc_orig}, Mem:{mem_cs})...",
                )
                seg_img, kw = _prepare_independent_inputs(image, step.params)
                res = _call_segment(backend, seg_img, kw)

                res_step = dict(res)
                is_cells_only = (
                    bool(step.params.get("skip_nuclear", False))
                    or str(step.params.get("target", "")).strip() == "cells"
                    or str(step.params.get("model_name", "")).strip() == "fluorescence_cells"
                )
                if (
                    not is_cells_only
                    and res_step.get("nuclear") is not None
                    and res_step.get("cytoplasm") is not None
                ):
                    rnuc, rcyto = reconcile_nuclear_and_cytoplasm_masks(
                        res_step["nuclear"], res_step["cytoplasm"]
                    )
                    res_step["nuclear"] = rnuc
                    res_step["cytoplasm"] = rcyto

                global_nuc = _merge_label_masks(global_nuc, res_step.get("nuclear"))
                global_cyto = _merge_label_masks(global_cyto, res_step.get("cytoplasm"))
                if not is_cells_only and global_nuc is not None and global_cyto is not None:
                    global_nuc, global_cyto = reconcile_nuclear_and_cytoplasm_masks(
                        global_nuc, global_cyto
                    )

                elapsed = time.perf_counter() - t0
                iris_log(
                    "IRIS-Viewer",
                    "Segmentation",
                    f"{tag}[{group_name}] Step {step_idx + 1}/{n_steps} "
                    f"'{step.method_name}' done in {elapsed:.2f}s "
                    f"(nuc={_count_labels(res_step.get('nuclear'))}, "
                    f"cyto={_count_labels(res_step.get('cytoplasm'))})",
                )
                yield ("step", group_name, step_idx, step.method_name, res_step)

            elif step.rule == "Nuclear (Base)":
                backend = backends.get(step.method_name)
                if not backend:
                    err = f"Backend {step.method_name} not available"
                    log_error(
                        "IRIS-Viewer",
                        "Segmentation",
                        f"{tag}[{group_name}] Step {step_idx + 1}/{n_steps} failed: {err}",
                    )
                    yield ("error", group_name, step_idx, step.method_name, err)
                    continue

                nuc_orig = step.params.get("nuclear_channel", 0)
                iris_log(
                    "IRIS-Viewer",
                    "Segmentation",
                    f"{tag}[{group_name}] Step {step_idx + 1}/{n_steps}: "
                    f"'Nuclear (Base)' ({step.method_name}, Nuc:{nuc_orig})...",
                )
                seg_img, kw = _prepare_nuclear_inputs(image, step.params)
                res = _call_segment(backend, seg_img, kw)

                global_nuc = _merge_label_masks(global_nuc, res.get("nuclear"))
                res_step = dict(res)
                res_step["cytoplasm"] = None
                elapsed = time.perf_counter() - t0
                iris_log(
                    "IRIS-Viewer",
                    "Segmentation",
                    f"{tag}[{group_name}] Step {step_idx + 1}/{n_steps} "
                    f"'{step.method_name}' done in {elapsed:.2f}s "
                    f"(nuc={_count_labels(res_step.get('nuclear'))})",
                )
                yield ("step", group_name, step_idx, step.method_name, res_step)

            elif step.rule == "Cytoplasm (Sequential)":
                backend = backends.get(step.method_name)
                if not backend:
                    err = f"Backend {step.method_name} not available"
                    log_error(
                        "IRIS-Viewer",
                        "Segmentation",
                        f"{tag}[{group_name}] Step {step_idx + 1}/{n_steps} failed: {err}",
                    )
                    yield ("error", group_name, step_idx, step.method_name, err)
                    continue

                if global_nuc is None:
                    err = "Cannot run sequential cytoplasm without a base nuclear mask!"
                    log_error(
                        "IRIS-Viewer",
                        "Segmentation",
                        f"{tag}[{group_name}] Step {step_idx + 1}/{n_steps} failed: {err}",
                    )
                    yield ("error", group_name, step_idx, step.method_name, err)
                    continue

                nuc_c = step.params.get("nuclear_channel", 0)
                mem_cs = step.params.get("membrane_channels", [])
                iris_log(
                    "IRIS-Viewer",
                    "Segmentation",
                    f"{tag}[{group_name}] Step {step_idx + 1}/{n_steps}: "
                    f"'Cytoplasm (Sequential)' ({step.method_name}, "
                    f"Nuc:{nuc_c}, Mem:{mem_cs})...",
                )

                synth_img, kw = _prepare_sequential_cytoplasm_inputs(image, step.params)
                res = _call_segment(backend, synth_img, kw)

                new_cyto = res.get("cytoplasm")
                if new_cyto is None:
                    new_cyto = res.get("nuclear")

                final_cyto_for_this_step, global_cyto = _assign_sequential_cytoplasm(
                    global_nuc, global_cyto, new_cyto
                )
                res_step = dict(res)
                res_step["cytoplasm"] = final_cyto_for_this_step
                res_step["nuclear"] = None
                elapsed = time.perf_counter() - t0
                iris_log(
                    "IRIS-Viewer",
                    "Segmentation",
                    f"{tag}[{group_name}] Step {step_idx + 1}/{n_steps} "
                    f"'{step.method_name}' done in {elapsed:.2f}s "
                    f"(cyto={_count_labels(final_cyto_for_this_step)})",
                )
                yield ("step", group_name, step_idx, step.method_name, res_step)

            elif step.rule in _DILATION_RULES:
                if global_nuc is None:
                    err = "No nuclear mask to dilate!"
                    log_error(
                        "IRIS-Viewer",
                        "Segmentation",
                        f"{tag}[{group_name}] Step {step_idx + 1}/{n_steps} failed: {err}",
                    )
                    yield ("error", group_name, step_idx, "Dilation", err)
                    continue

                radius = int(step.params.get("dilation_radius", 5))
                dilation_mode = str(step.params.get("dilation_mode", "distance"))
                mem_cs = step.params.get("membrane_channels", [])
                iris_log(
                    "IRIS-Viewer",
                    "Segmentation",
                    f"{tag}[{group_name}] Step {step_idx + 1}/{n_steps}: "
                    f"'{step.rule}' (radius={radius}px, mode={dilation_mode})...",
                )
                mem_img = (
                    _extract_max_membrane_2d(image, mem_cs)
                    if dilation_mode == "watershed" and mem_cs
                    else None
                )
                new_cyto, global_cyto = _dilate_remaining_cytoplasm(
                    global_nuc,
                    global_cyto,
                    radius,
                    mode=dilation_mode,
                    membrane_image=mem_img,
                )
                res_step = {"nuclear": None, "cytoplasm": new_cyto}
                elapsed = time.perf_counter() - t0
                iris_log(
                    "IRIS-Viewer",
                    "Segmentation",
                    f"{tag}[{group_name}] Step {step_idx + 1}/{n_steps} "
                    f"'Dilation' done in {elapsed:.2f}s "
                    f"(expanded={_count_labels(new_cyto)})",
                )
                yield ("step", group_name, step_idx, "Dilation", res_step)

        except Exception as exc:
            traceback.print_exc()
            log_error(
                "IRIS-Viewer",
                "Segmentation",
                f"{tag}[{group_name}] Step {step_idx + 1}/{n_steps} ({step_label}) raised: {exc}",
            )
            yield ("error", group_name, step_idx, step_label, str(exc))

    last_params = sequence.steps[-1].params if sequence.steps else {}
    auto_complete = (
        bool(sequence.auto_complete_cytoplasm)
        if sequence.auto_complete_cytoplasm is not None
        else bool(last_params.get("auto_complete_cytoplasm", False))
    )
    fb_radius = int(
        sequence.fallback_dilation_radius
        if sequence.fallback_dilation_radius != 5
        else last_params.get("fallback_dilation_radius", sequence.fallback_dilation_radius)
    )
    fb_mode = str(
        sequence.fallback_dilation_mode
        if sequence.fallback_dilation_mode != "distance"
        else last_params.get("fallback_dilation_mode", sequence.fallback_dilation_mode)
    )
    if (
        auto_complete
        and global_nuc is not None
        and int(global_nuc.max()) > 0
        and (global_cyto is None or _count_labels(global_cyto) < _count_labels(global_nuc))
    ):
        mem_img = None
        if fb_mode.strip().lower() == "watershed":
            fb_mem_cs = (
                sequence.fallback_membrane_channels
                if sequence.fallback_membrane_channels is not None
                else last_params.get("fallback_membrane_channels")
            )
            if fb_mem_cs:
                mem_img = _extract_max_membrane_2d(image, fb_mem_cs)
            else:
                for st in reversed(sequence.steps):
                    mem_cs = st.params.get("membrane_channels")
                    if mem_cs:
                        mem_img = _extract_max_membrane_2d(image, mem_cs)
                        break
        iris_log(
            "IRIS-Viewer",
            "Segmentation",
            f"{tag}[{group_name}] Auto-completing unmatched cytoplasm "
            f"(radius={fb_radius}px, mode={fb_mode})...",
        )
        _, global_cyto = _dilate_remaining_cytoplasm(
            global_nuc,
            global_cyto,
            fb_radius,
            mode=fb_mode,
            membrane_image=mem_img,
        )

    iris_log(
        "IRIS-Viewer",
        "Segmentation",
        f"{tag}[{group_name}] Group complete: "
        f"{_count_labels(global_nuc)} nuclei, {_count_labels(global_cyto)} cytoplasm",
    )
    yield ("group_done", group_name, {"nuclear": global_nuc, "cytoplasm": global_cyto})


class PipelineService:
    """Manage execution of segmentation models off the main UI thread.

    Decouples backend invocation and GPU management from the ViewerController.
    """

    def __init__(self) -> None:
        self.backends = {b.name: b for b in get_available_backends() if b.is_available()}
        self._active_workers: list = []
        self._gpu_lock = threading.Lock()

    @staticmethod
    def _get_device() -> str:
        """Detect best available torch device ('cuda', 'mps', or 'cpu')."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def get_available_methods(self) -> List[str]:
        return list(self.backends.keys())

    def execute_pipeline_groups(
        self,
        groups: Dict[str, SegmentationSequence],
        image: np.ndarray,
        on_step_complete: Callable[[str, int, str, dict], None],
        on_group_complete: Callable[[str, dict], None],
        on_all_complete: Callable[[], None],
        on_error: Callable[[str, int, str, str], None],
        on_step_start: Optional[Callable[[str, int, str], None]] = None,
    ) -> None:
        """Execute segmentation pipeline groups asynchronously on a worker thread.

        Args:
            groups: Mapping of group name to SegmentationSequence.
            image: Multi-channel input image array (C, H, W).
            on_step_complete: Callback invoked after each individual step finishes.
            on_group_complete: Callback invoked when a group finishes all steps.
            on_all_complete: Callback invoked when all sequences conclude.
            on_error: Error handler callback receiving step error details.
            on_step_start: Optional callback invoked before each step starts.
        """
        from napari.qt import thread_worker

        device = self._get_device()
        total_steps = sum(len(seq.steps) for seq in groups.values())

        @thread_worker(progress={"total": total_steps, "desc": "Segmentation Pipeline"})
        def _run_sequence():
            for group_name, sequence in groups.items():
                yield from iter_execute_sequence(
                    group_name=group_name,
                    sequence=sequence,
                    image=image,
                    backends=self.backends,
                    device=device,
                    gpu_lock=self._gpu_lock,
                    log_prefix="[Pipeline]",
                )

        self.sequence_worker = _run_sequence()

        def _on_yield(val):
            msg_type = val[0]
            if msg_type == "step_start":
                _, grp, idx, name = val
                if on_step_start is not None:
                    on_step_start(grp, idx, name)
            elif msg_type == "step":
                _, grp, idx, name, res = val
                on_step_complete(grp, idx, name, res)
            elif msg_type == "group_done":
                _, grp, res = val
                on_group_complete(grp, res)
            elif msg_type == "error":
                _, grp, idx, name, err = val
                on_error(grp, idx, name, err)

        worker = self.sequence_worker

        def _on_finished():
            if worker in self._active_workers:
                self._active_workers.remove(worker)
            on_all_complete()

        def _on_pipeline_worker_error(exc: Exception):
            if worker in self._active_workers:
                self._active_workers.remove(worker)
            on_error("Pipeline", -1, "General", str(exc))

        worker.yielded.connect(_on_yield)
        worker.finished.connect(_on_finished)
        worker.errored.connect(_on_pipeline_worker_error)

        self._active_workers.append(worker)
        worker.start()

    def cancel_all(self) -> None:
        """Cancel all running workers in the current sequence."""
        for worker in list(self._active_workers):
            try:
                worker.quit()
            except Exception:
                pass
        self._active_workers.clear()

    def execute_roi_preview(
        self,
        group_name: str,
        sequence: SegmentationSequence,
        image: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        on_complete: Callable[[str, Tuple[int, int, int, int], dict], None],
        on_error: Callable[[str, str], None],
    ) -> None:
        """Run a single pipeline sequence asynchronously on a cropped ROI bounding box."""
        from napari.qt import thread_worker

        device = self._get_device()
        clamped_bbox = _clamp_roi_bbox(roi_bbox, image.shape[-2:])

        @thread_worker
        def _run_roi() -> Dict[str, Optional[np.ndarray]]:
            step_errors: List[str] = []
            res = crop_roi_and_segment_sync(
                sequence=sequence,
                image=image,
                roi_bbox=clamped_bbox,
                backends=self.backends,
                device=device,
                gpu_lock=self._gpu_lock,
                errors_out=step_errors,
            )
            if (
                step_errors
                and res.get("nuclear") is None
                and res.get("cytoplasm") is None
            ):
                raise RuntimeError("; ".join(step_errors))
            return res

        worker = _run_roi()

        def _on_returned(res: Dict[str, Optional[np.ndarray]]) -> None:
            on_complete(group_name, clamped_bbox, res)

        def _on_finished() -> None:
            if worker in self._active_workers:
                self._active_workers.remove(worker)

        def _on_worker_error(exc: Exception) -> None:
            if worker in self._active_workers:
                self._active_workers.remove(worker)
            on_error(group_name, str(exc))

        worker.returned.connect(_on_returned)
        worker.finished.connect(_on_finished)
        worker.errored.connect(_on_worker_error)

        self._active_workers.append(worker)
        worker.start()


def crop_roi_and_segment_sync(
    sequence: SegmentationSequence,
    image: np.ndarray,
    roi_bbox: Tuple[int, int, int, int],
    backends: Dict,
    device: str,
    gpu_lock: Optional[threading.Lock] = None,
    errors_out: Optional[List[str]] = None,
) -> Dict[str, Optional[np.ndarray]]:
    """Clamp ``roi_bbox`` (y0, x0, y1, x1), crop ``image``, and segment synchronously."""
    if image.ndim < 2:
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}.")

    y0, x0, y1, x1 = _clamp_roi_bbox(roi_bbox, image.shape[-2:])
    if y1 <= y0 or x1 <= x0:
        return {"nuclear": None, "cytoplasm": None}

    t0 = time.perf_counter()
    cropped = image[y0:y1, x0:x1] if image.ndim == 2 else image[..., y0:y1, x0:x1]
    iris_log(
        "IRIS-Viewer",
        "ROI Preview",
        f"Cropped shape={cropped.shape} (y={y0}:{y1}, x={x0}:{x1}); "
        f"running {len(sequence.steps)} step(s) on {device}...",
    )
    res = execute_group_sync(
        group_name="ROI_Preview",
        sequence=sequence,
        image=cropped,
        backends=backends,
        device=device,
        gpu_lock=gpu_lock,
        log_prefix="[Pipeline]",
        errors_out=errors_out,
    )
    iris_log(
        "IRIS-Viewer",
        "ROI Preview",
        f"Completed in {time.perf_counter() - t0:.2f}s "
        f"(nuclear={_count_labels(res.get('nuclear'))}, "
        f"cytoplasm={_count_labels(res.get('cytoplasm'))})",
    )
    return res


# Synchronous Group Execution


def execute_group_sync(
    group_name: str,
    sequence: SegmentationSequence,
    image: np.ndarray,
    backends: Dict,
    device: str,
    gpu_lock: Optional[threading.Lock] = None,
    log_prefix: str = "[Batch]",
    errors_out: Optional[List[str]] = None,
) -> Dict[str, Optional[np.ndarray]]:
    """Execute a segmentation sequence synchronously for batch or ROI processing.

    Used by BatchService and ROI Preview to sequentially process segmentation steps
    without nesting napari thread workers.

    Args:
        group_name: Identifier for log reporting.
        sequence: Ordered sequence of SegmentationStep objects to execute.
        image: Multi-channel input image array (C, H, W).
        backends: Dictionary mapping method name to available backend instance.
        device: Target compute device ('cpu', 'cuda', 'mps').
        gpu_lock: Optional threading.Lock to synchronize GPU access across workers.
        log_prefix: Prefix tag forwarded to step logs.
        errors_out: Optional list populated with formatted step error strings.

    Returns:
        Dict with keys 'nuclear' and 'cytoplasm', each containing an ndarray or None.
    """
    final_res: Dict[str, Optional[np.ndarray]] = {"nuclear": None, "cytoplasm": None}
    for event in iter_execute_sequence(
        group_name=group_name,
        sequence=sequence,
        image=image,
        backends=backends,
        device=device,
        gpu_lock=gpu_lock,
        log_prefix=log_prefix,
    ):
        if event[0] == "error" and errors_out is not None:
            errors_out.append(f"{event[3]} (step {event[2] + 1}): {event[4]}")
        elif event[0] == "group_done":
            final_res = event[2]
    return final_res
