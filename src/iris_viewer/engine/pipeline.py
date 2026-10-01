""" Pipeline Service for Segmentation Sequences """
import threading
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
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

    def execute_sequence(
        self,
        sequence: SegmentationSequence,
        image: np.ndarray,
        on_step_complete: Callable[[int, str, dict], None],
        on_sequence_complete: Callable[[], None],
        on_error: Callable[[int, str, str], None]
    ):
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        @thread_worker(progress={'total': len(sequence.steps), 'desc': "Segmentation Pipeline"})
        def _run_sequence():
            global_nuc = None
            global_cyto = None

            for step_idx, step in enumerate(sequence.steps):
                try:
                    if step.rule == "Nuclear (Base)":
                        backend = self.backends.get(step.method_name)
                        if not backend:
                            yield ('error', step_idx, step.method_name, f"Backend {step.method_name} not available")
                            continue
                            
                        kw = step.params.copy()
                        kw.setdefault('diameter', None)
                        nuc_orig = kw.get('nuclear_channel', 0)
                        
                        if 'membrane_channels' in kw: del kw['membrane_channels']
                        kw['membrane_channel'] = None
                        
                        print(f"[Pipeline] Executing 'Nuclear (Base)' with '{step.method_name}' on Original Channel Nuc:{nuc_orig}...")
                        
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
                            
                        yield ('step', step_idx, step.method_name, res)

                    elif step.rule == "Cytoplasm (Sequential)":
                        backend = self.backends.get(step.method_name)
                        if not backend:
                            yield ('error', step_idx, step.method_name, f"Backend {step.method_name} not available")
                            continue
                            
                        if global_nuc is None:
                            yield ('error', step_idx, step.method_name, "Cannot run sequential cytoplasm without a base nuclear mask!")
                            continue
                            
                        nuc_c = step.params.get('nuclear_channel', 0)
                        mem_cs = step.params.get('membrane_channels', [])
                        
                        print(f"[Pipeline] Executing 'Cytoplasm (Sequential)' with '{step.method_name}' on Original Channels Nuc:{nuc_c} Mem:{mem_cs}...")
                        
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
                        
                        yield ('step', step_idx, step.method_name, res)
                        
                    elif step.rule == "Cytoplasm (Dilate Remaining)":
                        if global_nuc is None:
                            yield ('error', step_idx, "Dilation", "No nuclear mask to dilate!")
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
                            
                        yield ('step', step_idx, "Dilation", res)
                        
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    yield ('error', step_idx, getattr(step, 'method_name', 'Unknown'), str(e))

        self.sequence_worker = _run_sequence()
        
        def _on_yield(val):
            msg_type = val[0]
            if msg_type == 'step':
                _, idx, name, res = val
                on_step_complete(idx, name, res)
            elif msg_type == 'error':
                _, idx, name, err = val
                on_error(idx, name, err)
                
        self.sequence_worker.yielded.connect(_on_yield)
        self.sequence_worker.finished.connect(on_sequence_complete)
        
        self._active_workers.append(self.sequence_worker)
        self.sequence_worker.start()

    def cancel_all(self):
        """Cancels all running workers in the current sequence."""
        for worker in self._active_workers:
            worker.quit()
        self._active_workers.clear()
