"""Pipeline Service for Segmentation Sequences."""
from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from iris_viewer.engine.segmentation import get_available_backends


@dataclass
class SegmentationStep:
    method_name: str
    params: dict
    rule: str = "Independent"


@dataclass
class SegmentationSequence:
    steps: List[SegmentationStep]


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


def _prepare_independent_inputs(
    image: np.ndarray,
    params: dict,
) -> Tuple[np.ndarray, dict]:
    """Build input image and kwargs for an Independent segmentation step."""
    kw = params.copy()
    kw.setdefault("diameter", None)
    nuc_c = kw.get("nuclear_channel", 0)
    mem_cs = kw.pop("membrane_channels", None)
    if mem_cs and image.ndim >= 3:
        mem_img = image[mem_cs, :, :].max(axis=0)
        seg_img = np.stack([image[nuc_c, :, :], mem_img], axis=0)
        kw["nuclear_channel"] = 0
        kw["membrane_channel"] = 1
    else:
        seg_img = image
        kw.setdefault("membrane_channel", None)
    return seg_img, kw


def _prepare_sequential_cytoplasm_inputs(
    image: np.ndarray,
    params: dict,
) -> Tuple[np.ndarray, dict]:
    """Build 2-channel [nuclear, max-projected membrane] image for sequential cytoplasm."""
    nuc_c = params.get("nuclear_channel", 0)
    mem_cs = params.get("membrane_channels", [])
    if mem_cs and len(mem_cs) > 0:
        mem_img = image[mem_cs, :, :].max(axis=0)
    else:
        mem_img = np.zeros_like(image[0])
    synth_img = np.stack([image[nuc_c, :, :], mem_img], axis=0)
    kw = params.copy()
    kw.setdefault("diameter", None)
    kw.pop("membrane_channels", None)
    kw["nuclear_channel"] = 0
    kw["membrane_channel"] = 1
    return synth_img, kw


def _dilate_remaining_cytoplasm(
    global_nuc: np.ndarray,
    global_cyto: Optional[np.ndarray],
    radius: int,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Dilate nuclear labels lacking cytoplasm coverage up to *radius* pixels."""
    if global_cyto is None:
        global_cyto = np.zeros_like(global_nuc)

    unsegmented_nuc_mask = np.zeros_like(global_nuc)
    for n_lbl in np.unique(global_nuc)[1:]:
        if not np.any(global_cyto[global_nuc == n_lbl] > 0):
            unsegmented_nuc_mask[global_nuc == n_lbl] = n_lbl

    if not np.any(unsegmented_nuc_mask > 0):
        return None, global_cyto

    from scipy.ndimage import distance_transform_edt

    dists, (nearest_y, nearest_x) = distance_transform_edt(
        unsegmented_nuc_mask == 0,
        return_indices=True,
    )
    new_cyto = unsegmented_nuc_mask.copy()
    expand_mask = (dists > 0) & (dists <= radius)
    new_cyto[expand_mask] = unsegmented_nuc_mask[
        nearest_y[expand_mask], nearest_x[expand_mask]
    ]
    valid_expansion = (global_cyto == 0) & (new_cyto > 0)
    global_cyto[valid_expansion] = new_cyto[valid_expansion]
    return new_cyto, global_cyto


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
    ) -> None:
        """Execute segmentation pipeline groups asynchronously on a worker thread.

        Args:
            groups: Mapping of group name to SegmentationSequence.
            image: Multi-channel input image array (C, H, W).
            on_step_complete: Callback invoked after each individual step finishes.
            on_group_complete: Callback invoked when a group finishes all steps.
            on_all_complete: Callback invoked when all sequences conclude.
            on_error: Error handler callback receiving step error details.
        """
        from napari.qt import thread_worker

        device = self._get_device()
        total_steps = sum(len(seq.steps) for seq in groups.values())

        @thread_worker(progress={"total": total_steps, "desc": "Segmentation Pipeline"})
        def _run_sequence():
            for group_name, sequence in groups.items():
                global_nuc: Optional[np.ndarray] = None
                global_cyto: Optional[np.ndarray] = None

                for step_idx, step in enumerate(sequence.steps):
                    try:
                        if step.rule == "Independent":
                            backend = self.backends.get(step.method_name)
                            if not backend:
                                yield (
                                    "error",
                                    group_name,
                                    step_idx,
                                    step.method_name,
                                    f"Backend {step.method_name} not available",
                                )
                                continue

                            seg_img, kw = _prepare_independent_inputs(image, step.params)
                            with self._gpu_lock:
                                res = backend.segment(image=seg_img, device=device, **kw)

                            global_nuc = _merge_label_masks(global_nuc, res.get("nuclear"))
                            global_cyto = _merge_label_masks(global_cyto, res.get("cytoplasm"))
                            yield ("step", group_name, step_idx, step.method_name, res)

                        elif step.rule == "Nuclear (Base)":
                            backend = self.backends.get(step.method_name)
                            if not backend:
                                yield (
                                    "error",
                                    group_name,
                                    step_idx,
                                    step.method_name,
                                    f"Backend {step.method_name} not available",
                                )
                                continue

                            kw = step.params.copy()
                            kw.setdefault("diameter", None)
                            nuc_orig = kw.get("nuclear_channel", 0)
                            kw.pop("membrane_channels", None)
                            kw["membrane_channel"] = None

                            print(
                                f"[Pipeline] [{group_name}] Executing 'Nuclear (Base)' "
                                f"with '{step.method_name}' on Original Channel Nuc:{nuc_orig}..."
                            )

                            with self._gpu_lock:
                                res = backend.segment(image=image, device=device, **kw)

                            global_nuc = _merge_label_masks(global_nuc, res.get("nuclear"))
                            # Enforce that Nuclear (Base) steps only yield a nuclear mask
                            res["cytoplasm"] = None

                            yield ("step", group_name, step_idx, step.method_name, res)

                        elif step.rule == "Cytoplasm (Sequential)":
                            backend = self.backends.get(step.method_name)
                            if not backend:
                                yield (
                                    "error",
                                    group_name,
                                    step_idx,
                                    step.method_name,
                                    f"Backend {step.method_name} not available",
                                )
                                continue

                            if global_nuc is None:
                                yield (
                                    "error",
                                    group_name,
                                    step_idx,
                                    step.method_name,
                                    "Cannot run sequential cytoplasm without a base nuclear mask!",
                                )
                                continue

                            nuc_c = step.params.get("nuclear_channel", 0)
                            mem_cs = step.params.get("membrane_channels", [])
                            print(
                                f"[Pipeline] [{group_name}] Executing 'Cytoplasm (Sequential)' "
                                f"with '{step.method_name}' on Original Channels "
                                f"Nuc:{nuc_c} Mem:{mem_cs}..."
                            )

                            synth_img, kw = _prepare_sequential_cytoplasm_inputs(
                                image, step.params
                            )
                            with self._gpu_lock:
                                res = backend.segment(image=synth_img, device=device, **kw)

                            new_cyto = res.get("cytoplasm")
                            if new_cyto is None:
                                new_cyto = res.get("nuclear")

                            if global_cyto is None:
                                global_cyto = np.zeros_like(global_nuc)

                            final_cyto_for_this_step = np.zeros_like(new_cyto)
                            new_labels = np.unique(new_cyto)
                            new_labels = new_labels[new_labels > 0]

                            for lbl in new_labels:
                                mask = new_cyto == lbl
                                nuc_overlaps = global_nuc[mask]
                                nuc_overlaps = nuc_overlaps[nuc_overlaps > 0]
                                if len(nuc_overlaps) > 0:
                                    nuc_lbl = int(np.bincount(nuc_overlaps).argmax())
                                    if not np.any(global_cyto[global_nuc == nuc_lbl] > 0):
                                        final_cyto_for_this_step[mask] = nuc_lbl
                                        global_cyto[mask] = nuc_lbl

                            res["cytoplasm"] = final_cyto_for_this_step
                            res["nuclear"] = None

                            yield ("step", group_name, step_idx, step.method_name, res)

                        elif step.rule == "Cytoplasm (Dilate Remaining)":
                            if global_nuc is None:
                                yield (
                                    "error",
                                    group_name,
                                    step_idx,
                                    "Dilation",
                                    "No nuclear mask to dilate!",
                                )
                                continue

                            radius = step.params.get("dilation_radius", 5)
                            new_cyto, global_cyto = _dilate_remaining_cytoplasm(
                                global_nuc, global_cyto, radius
                            )
                            res = {"nuclear": None, "cytoplasm": new_cyto}
                            yield ("step", group_name, step_idx, "Dilation", res)

                    except Exception as exc:
                        traceback.print_exc()
                        yield (
                            "error",
                            group_name,
                            step_idx,
                            getattr(step, "method_name", "Unknown"),
                            str(exc),
                        )

                yield (
                    "group_done",
                    group_name,
                    {"nuclear": global_nuc, "cytoplasm": global_cyto},
                )

        self.sequence_worker = _run_sequence()

        def _on_yield(val):
            msg_type = val[0]
            if msg_type == "step":
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


# Synchronous Group Execution


def execute_group_sync(
    group_name: str,
    sequence: SegmentationSequence,
    image: np.ndarray,
    backends: Dict,
    device: str,
    gpu_lock: Optional[threading.Lock] = None,
) -> Dict[str, Optional[np.ndarray]]:
    """Execute a segmentation sequence synchronously for batch processing.

    Used by BatchService to sequentially process segmentation steps without nesting
    napari thread workers.

    Args:
        group_name: Identifier for log reporting.
        sequence: Ordered sequence of SegmentationStep objects to execute.
        image: Multi-channel input image array (C, H, W).
        backends: Dictionary mapping method name to available backend instance.
        device: Target compute device ('cpu', 'cuda', 'mps').
        gpu_lock: Optional threading.Lock to synchronize GPU access across workers.

    Returns:
        Dict with keys 'nuclear' and 'cytoplasm', each containing an ndarray or None.
    """
    global_nuc: Optional[np.ndarray] = None
    global_cyto: Optional[np.ndarray] = None

    def _call_segment(backend, seg_img: np.ndarray, kw: dict) -> dict:
        if gpu_lock:
            with gpu_lock:
                return backend.segment(image=seg_img, device=device, **kw)
        return backend.segment(image=seg_img, device=device, **kw)

    for step_idx, step in enumerate(sequence.steps):
        try:
            if step.rule == "Independent":
                backend = backends.get(step.method_name)
                if not backend:
                    print(
                        f"[Batch/{group_name}] Backend '{step.method_name}' "
                        f"not available -- skipping step {step_idx}"
                    )
                    continue

                seg_img, kw = _prepare_independent_inputs(image, step.params)
                res = _call_segment(backend, seg_img, kw)
                global_nuc = _merge_label_masks(global_nuc, res.get("nuclear"))
                global_cyto = _merge_label_masks(global_cyto, res.get("cytoplasm"))

            elif step.rule == "Nuclear (Base)":
                backend = backends.get(step.method_name)
                if not backend:
                    print(
                        f"[Batch/{group_name}] Backend '{step.method_name}' "
                        f"not available -- skipping step {step_idx}"
                    )
                    continue

                kw = step.params.copy()
                kw.setdefault("diameter", None)
                kw.pop("membrane_channels", None)
                kw["membrane_channel"] = None

                res = _call_segment(backend, image, kw)
                global_nuc = _merge_label_masks(global_nuc, res.get("nuclear"))

            elif step.rule == "Cytoplasm (Sequential)":
                backend = backends.get(step.method_name)
                if not backend or global_nuc is None:
                    print(
                        f"[Batch/{group_name}] Skipping sequential cytoplasm step {step_idx} "
                        f"(backend={backend is not None}, nuc_ready={global_nuc is not None})"
                    )
                    continue

                synth_img, kw = _prepare_sequential_cytoplasm_inputs(image, step.params)
                res = _call_segment(backend, synth_img, kw)

                new_cyto = (
                    res.get("cytoplasm")
                    if res.get("cytoplasm") is not None
                    else res.get("nuclear")
                )
                if new_cyto is None:
                    continue

                if global_cyto is None:
                    global_cyto = np.zeros_like(global_nuc)

                new_labels = np.unique(new_cyto)
                new_labels = new_labels[new_labels > 0]
                for lbl in new_labels:
                    px = new_cyto == lbl
                    overlaps = global_nuc[px]
                    overlaps = overlaps[overlaps > 0]
                    if len(overlaps) > 0:
                        nuc_lbl = int(np.bincount(overlaps).argmax())
                        if nuc_lbl > 0 and not np.any(global_cyto[global_nuc == nuc_lbl] > 0):
                            global_cyto[px] = nuc_lbl

            elif step.rule == "Cytoplasm (Dilate Remaining)":
                if global_nuc is None:
                    print(
                        f"[Batch/{group_name}] Skipping dilation step {step_idx} "
                        "-- no nuclear mask"
                    )
                    continue

                radius = step.params.get("dilation_radius", 5)
                _, global_cyto = _dilate_remaining_cytoplasm(
                    global_nuc, global_cyto, radius
                )

        except Exception as exc:
            print(f"[Batch/{group_name}] Step {step_idx} ({step.method_name}) raised: {exc}")
            traceback.print_exc()

    return {"nuclear": global_nuc, "cytoplasm": global_cyto}
