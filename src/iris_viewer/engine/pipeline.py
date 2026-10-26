""" Pipeline Service for Segmentation Sequences """
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
from napari.qt import thread_worker

from iris_viewer.engine.segmentation import get_available_backends


@dataclass
class SegmentationStep:
    method_name: str
    params: dict
    rule: str = "Independent"  # "Nuclear (Base)", "Cytoplasm (Sequential)", "Cytoplasm (Dilate Remaining)"

@dataclass
class SegmentationSequence:
    steps: List[SegmentationStep]

class PipelineService:
    """
    Manages the execution of segmentation models off the main UI thread.
    Decouples backend invocation and GPU management from the ViewerController.
    """
    def __init__(self):
        self.backends = {b.name: b for b in get_available_backends() if b.is_available()}
        self._active_workers = []
        self._gpu_lock = threading.Lock()

    def get_available_methods(self) -> List[str]:
        return list(self.backends.keys())

    def execute_pipeline_groups(
        self,
        groups: Dict[str, SegmentationSequence],
        image: np.ndarray,
        on_step_complete: Callable[[str, int, str, dict], None],
        on_group_complete: Callable[[str, dict], None],
        on_all_complete: Callable[[], None],
        on_error: Callable[[str, int, str, str], None]
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
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        total_steps = sum(len(seq.steps) for seq in groups.values())
        @thread_worker(progress={'total': total_steps, 'desc': "Segmentation Pipeline"})
        def _run_sequence():
            for group_name, sequence in groups.items():
                global_nuc = None
                global_cyto = None

                for step_idx, step in enumerate(sequence.steps):
                    try:
                        if step.rule == "Nuclear (Base)":
                            backend = self.backends.get(step.method_name)
                            if not backend:
                                yield ('error', group_name, step_idx, step.method_name, f"Backend {step.method_name} not available")
                                continue

                            kw = step.params.copy()
                            kw.setdefault('diameter', None)
                            nuc_orig = kw.get('nuclear_channel', 0)

                            if 'membrane_channels' in kw: del kw['membrane_channels']
                            kw['membrane_channel'] = None

                            print(f"[Pipeline] [{group_name}] Executing 'Nuclear (Base)' with '{step.method_name}' on Original Channel Nuc:{nuc_orig}...")

                            with self._gpu_lock:
                                res = backend.segment(image=image, device=device, **kw)

                            if global_nuc is None:
                                global_nuc = res.get('nuclear')
                            else:
                                n_new = res.get('nuclear')
                                if n_new is not None:
                                    max_label = global_nuc.max()
                                    n_new_shifted = n_new.copy()
                                    n_new_shifted[n_new_shifted > 0] += max_label
                                    mask = (global_nuc == 0) & (n_new_shifted > 0)
                                    global_nuc[mask] = n_new_shifted[mask]

                            # Enforce that Nuclear (Base) steps only yield a nuclear mask
                            res['cytoplasm'] = None

                            yield ('step', group_name, step_idx, step.method_name, res)

                        elif step.rule == "Cytoplasm (Sequential)":
                            backend = self.backends.get(step.method_name)
                            if not backend:
                                yield ('error', group_name, step_idx, step.method_name, f"Backend {step.method_name} not available")
                                continue

                            if global_nuc is None:
                                yield ('error', group_name, step_idx, step.method_name, "Cannot run sequential cytoplasm without a base nuclear mask!")
                                continue

                            nuc_c = step.params.get('nuclear_channel', 0)
                            mem_cs = step.params.get('membrane_channels', [])

                            print(f"[Pipeline] [{group_name}] Executing 'Cytoplasm (Sequential)' with '{step.method_name}' on Original Channels Nuc:{nuc_c} Mem:{mem_cs}...")

                            if mem_cs and len(mem_cs) > 0:
                                mem_img = image[mem_cs, :, :].max(axis=0)
                            else:
                                mem_img = np.zeros_like(image[0])

                            nuc_img = image[nuc_c, :, :]
                            synth_img = np.stack([nuc_img, mem_img], axis=0)

                            kw = step.params.copy()
                            kw.setdefault('diameter', None)
                            if 'membrane_channels' in kw: del kw['membrane_channels']
                            kw['nuclear_channel'] = 0
                            kw['membrane_channel'] = 1

                            with self._gpu_lock:
                                res = backend.segment(image=synth_img, device=device, **kw)

                            new_cyto = res.get('cytoplasm')
                            if new_cyto is None:
                                new_cyto = res.get('nuclear')

                            if global_cyto is None:
                                global_cyto = np.zeros_like(global_nuc)

                            final_cyto_for_this_step = np.zeros_like(new_cyto)
                            new_labels = np.unique(new_cyto)
                            new_labels = new_labels[new_labels > 0]

                            for L in new_labels:
                                mask = (new_cyto == L)
                                nuc_overlaps = global_nuc[mask]
                                nuc_overlaps = nuc_overlaps[nuc_overlaps > 0]
                                if len(nuc_overlaps) > 0:
                                    nuc_L = np.bincount(nuc_overlaps).argmax()
                                    if not np.any(global_cyto[global_nuc == nuc_L] > 0):
                                        final_cyto_for_this_step[mask] = nuc_L
                                        global_cyto[mask] = nuc_L

                            res['cytoplasm'] = final_cyto_for_this_step
                            res['nuclear'] = None

                            yield ('step', group_name, step_idx, step.method_name, res)

                        elif step.rule == "Cytoplasm (Dilate Remaining)":
                            if global_nuc is None:
                                yield ('error', group_name, step_idx, "Dilation", "No nuclear mask to dilate!")
                                continue

                            if global_cyto is None:
                                global_cyto = np.zeros_like(global_nuc)

                            radius = step.params.get('dilation_radius', 5)
                            unsegmented_nuc_mask = np.zeros_like(global_nuc)
                            nuc_labels = np.unique(global_nuc)

                            for nL in nuc_labels[1:]:
                                if not np.any(global_cyto[global_nuc == nL] > 0):
                                    unsegmented_nuc_mask[global_nuc == nL] = nL

                            if np.any(unsegmented_nuc_mask > 0):
                                from scipy.ndimage import distance_transform_edt

                                bg = (unsegmented_nuc_mask == 0)
                                dists, indices = distance_transform_edt(bg, return_indices=True)

                                new_cyto = unsegmented_nuc_mask.copy()
                                expand_mask = (dists <= radius) & (dists > 0)

                                nearest_y, nearest_x = indices
                                nearest_labels = unsegmented_nuc_mask[nearest_y[expand_mask], nearest_x[expand_mask]]

                                new_cyto[expand_mask] = nearest_labels
                                valid_expansion = (global_cyto == 0) & (new_cyto > 0)
                                global_cyto[valid_expansion] = new_cyto[valid_expansion]

                                res = {'nuclear': None, 'cytoplasm': new_cyto}
                            else:
                                res = {'nuclear': None, 'cytoplasm': None}

                            yield ('step', group_name, step_idx, "Dilation", res)

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        yield ('error', group_name, step_idx, getattr(step, 'method_name', 'Unknown'), str(e))

                # Yield the final compiled mask for this group
                yield ('group_done', group_name, {'nuclear': global_nuc, 'cytoplasm': global_cyto})

        self.sequence_worker = _run_sequence()

        def _on_yield(val):
            msg_type = val[0]
            if msg_type == 'step':
                _, grp, idx, name, res = val
                on_step_complete(grp, idx, name, res)
            elif msg_type == 'group_done':
                _, grp, res = val
                on_group_complete(grp, res)
            elif msg_type == 'error':
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

    def cancel_all(self):
        """Cancels all running workers in the current sequence."""
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

    for step_idx, step in enumerate(sequence.steps):
        try:
            # Nuclear (Base)
            if step.rule == "Nuclear (Base)":
                backend = backends.get(step.method_name)
                if not backend:
                    print(f"[Batch/{group_name}] Backend '{step.method_name}' not available — skipping step {step_idx}")
                    continue

                kw = step.params.copy()
                kw.setdefault('diameter', None)
                kw.pop('membrane_channels', None)
                kw['membrane_channel'] = None

                if gpu_lock:
                    with gpu_lock:
                        res = backend.segment(image=image, device=device, **kw)
                else:
                    res = backend.segment(image=image, device=device, **kw)

                n_new = res.get('nuclear')
                if n_new is not None:
                    if global_nuc is None:
                        global_nuc = n_new
                    else:
                        # Merge by adding new labels above the current max
                        shifted = n_new.copy()
                        shifted[shifted > 0] += int(global_nuc.max())
                        fill = (global_nuc == 0) & (shifted > 0)
                        global_nuc[fill] = shifted[fill]

            # Cytoplasm (Sequential)
            elif step.rule == "Cytoplasm (Sequential)":
                backend = backends.get(step.method_name)
                if not backend or global_nuc is None:
                    print(f"[Batch/{group_name}] Skipping sequential cytoplasm step {step_idx} "
                          f"(backend={backend is not None}, nuc_ready={global_nuc is not None})")
                    continue

                nuc_c = step.params.get('nuclear_channel', 0)
                mem_cs = step.params.get('membrane_channels', [])
                mem_img = image[mem_cs, :, :].max(axis=0) if mem_cs else np.zeros_like(image[0])
                synth_img = np.stack([image[nuc_c, :, :], mem_img], axis=0)

                kw = step.params.copy()
                kw.setdefault('diameter', None)
                kw.pop('membrane_channels', None)
                kw['nuclear_channel'] = 0
                kw['membrane_channel'] = 1

                if gpu_lock:
                    with gpu_lock:
                        res = backend.segment(image=synth_img, device=device, **kw)
                else:
                    res = backend.segment(image=synth_img, device=device, **kw)

                new_cyto = res.get('cytoplasm') if res.get('cytoplasm') is not None else res.get('nuclear')
                if new_cyto is None:
                    continue

                if global_cyto is None:
                    global_cyto = np.zeros_like(global_nuc)

                new_labels = np.unique(new_cyto)
                new_labels = new_labels[new_labels > 0]
                for L in new_labels:
                    px = new_cyto == L
                    overlaps = global_nuc[px]
                    overlaps = overlaps[overlaps > 0]
                    if len(overlaps) > 0:
                        nuc_L = int(np.bincount(overlaps).argmax())
                        if nuc_L > 0 and not np.any(global_cyto[global_nuc == nuc_L] > 0):
                            global_cyto[px] = nuc_L

            # Cytoplasm (Dilate Remaining)
            elif step.rule == "Cytoplasm (Dilate Remaining)":
                if global_nuc is None:
                    print(f"[Batch/{group_name}] Skipping dilation step {step_idx} — no nuclear mask")
                    continue

                if global_cyto is None:
                    global_cyto = np.zeros_like(global_nuc)

                radius = step.params.get('dilation_radius', 5)
                unseg = np.zeros_like(global_nuc)
                for nL in np.unique(global_nuc)[1:]:
                    if not np.any(global_cyto[global_nuc == nL] > 0):
                        unseg[global_nuc == nL] = nL

                if np.any(unseg > 0):
                    from scipy.ndimage import distance_transform_edt
                    dists, (ny, nx) = distance_transform_edt(unseg == 0, return_indices=True)
                    exp = (dists > 0) & (dists <= radius)
                    nearest = unseg[ny[exp], nx[exp]]
                    new_cyto = unseg.copy()
                    new_cyto[exp] = nearest
                    fill = (global_cyto == 0) & (new_cyto > 0)
                    global_cyto[fill] = new_cyto[fill]

        except Exception as exc:
            import traceback
            print(f"[Batch/{group_name}] Step {step_idx} ({step.method_name}) raised: {exc}")
            traceback.print_exc()

    return {'nuclear': global_nuc, 'cytoplasm': global_cyto}
