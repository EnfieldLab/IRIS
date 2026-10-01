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
    # Later additions for Phase 4: target_markers, prior_mask, dilation_rules, etc.

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
        on_step_complete: Callable[[str, dict], None],
        on_sequence_complete: Callable[[], None],
        on_error: Callable[[str, str], None]
    ):
        """
        Executes a SegmentationSequence.
        Currently executes steps in parallel (like legacy behavior) but is designed 
        to be refactored into a sequential cascade in Phase 4.
        """
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

        self._pending_count = len(sequence.steps)
        self._completed_count = 0
        
        if self._pending_count == 0:
            on_sequence_complete()
            return

        for step in sequence.steps:
            backend = self.backends.get(step.method_name)
            if not backend:
                on_error(step.method_name, f"Backend {step.method_name} not available")
                self._completed_count += 1
                if self._completed_count == self._pending_count:
                    on_sequence_complete()
                continue
            
            @thread_worker(progress={'total': 0, 'desc': f"Seg: {step.method_name}"})
            def _run_single(b=backend, img=image, d=device, kw=step.params, mn=step.method_name):
                # Ensure sequential GPU access to prevent VRAM overflow
                with self._gpu_lock:
                    return mn, b.segment(image=img, device=d, **kw)
                
            worker = _run_single()
            
            # We must use default arguments in lambdas to capture the current worker in the loop
            worker.returned.connect(
                lambda res, w=worker: self._handle_result(res, w, on_step_complete, on_sequence_complete)
            )
            worker.errored.connect(
                lambda err, w=worker, mn=step.method_name: self._handle_error(err, w, mn, on_error, on_sequence_complete)
            )
            
            self._active_workers.append(worker)
            worker.start()

    def _handle_result(self, res, worker, on_step_complete, on_sequence_complete):
        if worker in self._active_workers:
            self._active_workers.remove(worker)
        method_name, seg_result = res
        on_step_complete(method_name, seg_result)
        
        self._completed_count += 1
        if self._completed_count >= self._pending_count:
            on_sequence_complete()

    def _handle_error(self, err, worker, method_name, on_error, on_sequence_complete):
        if worker in self._active_workers:
            self._active_workers.remove(worker)
        on_error(method_name, str(err))
        
        self._completed_count += 1
        if self._completed_count >= self._pending_count:
            on_sequence_complete()

    def cancel_all(self):
        """Cancels all running workers in the current sequence."""
        for worker in self._active_workers:
            worker.quit()
        self._active_workers.clear()
