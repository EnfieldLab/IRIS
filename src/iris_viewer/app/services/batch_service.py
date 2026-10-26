"""
IRIS-Viewer Batch Service
Sequential multi-sample batch processing using the configured pipeline groups.

Key fixes over the old controller implementation:
- Does NOT reference Qt widget state from inside the thread (bug fix).
- Uses the current tab-based pipeline groups (not the removed method_checkboxes).
- Calls execute_group_sync() so segmentation runs synchronously inside a single
  thread_worker (nested thread_workers are not allowed by napari).
"""
from __future__ import annotations

import threading
import types

import numpy as np
from napari.qt import thread_worker
from qtpy.QtWidgets import QMessageBox

from iris_viewer.engine.consensus import compute_consensus_mask
from iris_viewer.engine.feature_extraction import extract_cell_features
from iris_viewer.engine.pipeline import SegmentationSequence, execute_group_sync
from iris_viewer.io.exporters import (
    export_batch_summary,
    export_consensus_mask,
    export_features_csv,
)
from iris_viewer.io.loaders import load_multichannel_tiff


def _detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on available hardware."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if getattr(getattr(torch, 'backends', None), 'mps', None) and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


class BatchService:
    """
    Handles sequential multi-sample batch processing.

    One ``thread_worker`` is spawned per sample.  The worker executes all
    configured pipeline groups synchronously via ``execute_group_sync``, then
    computes consensus and extracts features before signalling completion.
    The next sample is triggered from the Qt main thread via the
    ``returned`` signal — no nested workers, no Qt widget access from threads.
    """

    def __init__(self, ctx: types.SimpleNamespace) -> None:
        self.ctx = ctx
        self._gpu_lock = threading.Lock()

    # Public API

    def run_batch_processing(self) -> None:
        """Entry point: validate state, snapshot pipeline config, start the queue."""
        ctx = self.ctx

        if not ctx.state.file_map:
            QMessageBox.warning(
                ctx.control_widget, "No Directory",
                "Load an image directory first."
            )
            return

        # Snapshot pipeline groups from the UI before spawning any thread
        groups = self._get_pipeline_groups()
        if not groups:
            QMessageBox.warning(
                ctx.control_widget, "No Pipeline Configured",
                "Configure at least one pipeline group with steps before running batch.\n\n"
                "Go to the Segmentation tab and add groups/steps."
            )
            return

        sample_names = sorted(ctx.state.file_map.keys())
        ctx.state.batch_queue = sample_names.copy()
        ctx.state.batch_results.clear()

        n = len(sample_names)
        ctx.control_widget.btn_batch.setText(f"Processing 0/{n}…")
        ctx.control_widget.btn_batch.setEnabled(False)

        self._batch_process_next(groups)

    # Internal Queue Loop

    def _batch_process_next(self, groups: dict) -> None:
        """Pop the next sample and start its worker; call _on_batch_complete when done."""
        ctx = self.ctx

        if not ctx.state.batch_queue:
            self._on_batch_complete()
            return

        sample_name = ctx.state.batch_queue.pop(0)
        img_path = ctx.state.file_map[sample_name]

        # Snapshot everything needed by the worker BEFORE entering the thread
        # This is the critical fix: no Qt widget reads inside _process_sample().
        scale_val: float = ctx.control_widget.spin_scale.value()
        markers: list = list(ctx.state.markers)
        device: str = _detect_device()
        backends: dict = dict(ctx.pipeline_service.backends)

        # Deep-copy the step lists so the worker has stable, immutable configs
        group_sequences: dict = {
            name: SegmentationSequence(steps=list(steps))
            for name, steps in groups.items()
        }
        threshold: float = ctx.consensus_panel.get_threshold()
        gpu_lock = self._gpu_lock

        @thread_worker
        def _process_sample():
            raw_image, _ = load_multichannel_tiff(img_path)

            batch_seg: dict = {}
            for grp_name, sequence in group_sequences.items():
                result = execute_group_sync(
                    group_name=grp_name,
                    sequence=sequence,
                    image=raw_image,
                    backends=backends,
                    device=device,
                    gpu_lock=gpu_lock,
                )
                if result.get('nuclear') is not None or result.get('cytoplasm') is not None:
                    batch_seg[grp_name] = result

            if not batch_seg:
                return sample_name, None

            consensus_mask, confidence_map, per_cell_conf, _ = compute_consensus_mask(batch_seg)

            features_df = extract_cell_features(
                mask=consensus_mask,
                raw_image=raw_image,
                markers=markers,
                scale_factor=scale_val,
            )

            out_dir = img_path.parent
            cyto = next(
                (r['cytoplasm'] for r in batch_seg.values() if r.get('cytoplasm') is not None),
                None,
            )
            export_consensus_mask(out_dir / f"{sample_name}_consensus_mask.tiff", consensus_mask, cyto)
            export_features_csv(out_dir / f"{sample_name}_features.csv", features_df, sample_name)

            confs = list(per_cell_conf.values())
            half = threshold * 0.5
            return sample_name, {
                'total_cells': len(per_cell_conf),
                'high_conf':  sum(1 for c in confs if c >= threshold),
                'med_conf':   sum(1 for c in confs if half <= c < threshold),
                'low_conf':   sum(1 for c in confs if c < half),
                'mean_confidence': float(np.mean(confs)) if confs else 0.0,
                'methods': list(batch_seg.keys()),
            }

        worker = _process_sample()
        worker.returned.connect(lambda res: self._on_sample_done(res, groups))
        if hasattr(ctx, "track_worker"):
            ctx.track_worker(
                worker,
                on_error=lambda exc: self._on_sample_error(sample_name, exc, groups),
                error_title="Batch Error",
                error_prefix=f"Processing sample '{sample_name}' failed",
            )
        else:
            worker.errored.connect(lambda exc: self._on_sample_error(sample_name, exc, groups))
            if hasattr(ctx, "active_workers"):
                ctx.active_workers.append(worker)
        worker.start()

    def _on_sample_error(self, sample_name: str, exc: Exception, groups: dict) -> None:
        """Handle error for a single sample so the remaining batch queue does not freeze."""
        ctx = self.ctx
        print(f"[Batch Error] Sample '{sample_name}' failed: {exc}")
        ctx.state.batch_results[sample_name] = {
            'total_cells': 0, 'high_conf': 0, 'med_conf': 0,
            'low_conf': 0, 'mean_confidence': 0.0, 'methods': [],
            'error': str(exc),
        }

        done = len(ctx.state.batch_results)
        total = done + len(ctx.state.batch_queue)
        ctx.control_widget.btn_batch.setText(f"Processing {done}/{total}…")

        self._batch_process_next(groups)

    def _on_sample_done(self, result: tuple, groups: dict) -> None:
        ctx = self.ctx
        sample_name, stats = result

        ctx.state.batch_results[sample_name] = stats or {
            'total_cells': 0, 'high_conf': 0, 'med_conf': 0,
            'low_conf': 0, 'mean_confidence': 0.0, 'methods': [],
        }

        done = len(ctx.state.batch_results)
        total = done + len(ctx.state.batch_queue)
        ctx.control_widget.btn_batch.setText(f"Processing {done}/{total}…")

        self._batch_process_next(groups)

    def _on_batch_complete(self) -> None:
        ctx = self.ctx
        ctx.control_widget.btn_batch.setText("Batch Process All Samples")
        ctx.control_widget.btn_batch.setEnabled(True)

        if ctx.state.batch_results and ctx.state.image_dir:
            summary_path = ctx.state.image_dir / "batch_summary.csv"
            export_batch_summary(summary_path, ctx.state.batch_results)

        n_samples = len(ctx.state.batch_results)
        n_success = sum(
            1 for s in ctx.state.batch_results.values()
            if s and s.get('total_cells', 0) > 0
        )

        QMessageBox.information(
            ctx.control_widget, "Batch Complete",
            f"Processed {n_samples} samples ({n_success} successful).\n"
            f"Results saved alongside source images."
            + (f"\nSummary: {ctx.state.image_dir / 'batch_summary.csv'}"
               if ctx.state.image_dir else "")
        )

    # Helpers

    def _get_pipeline_groups(self) -> dict:
        """Read the current tab-based pipeline configuration from the UI.
        Returns a dict of group_name -> list[SegmentationStep].
        Must be called from the main thread before any worker is spawned.
        """
        ctx = self.ctx
        panel = ctx.segmentation_panel
        groups: dict = {}
        for i in range(panel.tab_widget.count()):
            name = panel.tab_widget.tabText(i)
            steps = panel.tab_widget.widget(i).get_steps()
            if steps:
                groups[name] = steps
        return groups
