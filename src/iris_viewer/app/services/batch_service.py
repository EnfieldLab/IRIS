"""
IRIS-Viewer Batch Service
Sequential multi-sample batch processing using the configured pipeline groups.

Runs all configured segmentation groups synchronously inside a single
``thread_worker`` per sample (never accessing Qt widgets inside threads), selects
the active/first group mask for feature extraction and mask export, and computes
multi-model disagreement summaries + heatmaps when >= 2 groups succeed.
"""
from __future__ import annotations

import threading
import types
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from napari.qt import thread_worker
from qtpy.QtWidgets import QMessageBox

from iris_common.logging import iris_log, log_error
from iris_viewer.engine.disagreement import compute_model_disagreement
from iris_viewer.engine.feature_extraction import extract_cell_features
from iris_viewer.engine.pipeline import SegmentationSequence, execute_group_sync
from iris_viewer.io.exporters import (
    export_batch_summary,
    export_disagreement_map,
    export_features_csv,
    export_segmentation_mask,
)
from iris_viewer.io.loaders import load_multichannel_tiff


def _detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on available hardware."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class BatchService:
    """Sequential multi-sample batch processor using configured pipeline groups."""

    def __init__(self, ctx: types.SimpleNamespace) -> None:
        self.ctx = ctx
        self._gpu_lock = threading.Lock()

    # Public API

    def run_batch_processing(self) -> None:
        """Entry point: validate state, snapshot pipeline config, start the queue."""
        ctx = self.ctx

        if not ctx.state.file_map:
            QMessageBox.warning(
                ctx.control_widget,
                "No Directory",
                "Load an image directory first.",
            )
            return

        groups = self._get_pipeline_groups()
        if not groups:
            QMessageBox.warning(
                ctx.control_widget,
                "No Pipeline Configured",
                "Configure at least one pipeline group with steps before running batch.\n\n"
                "Go to the Segmentation tab and add groups/steps.",
            )
            return

        sample_names = sorted(ctx.state.file_map.keys())
        ctx.state.batch_queue = sample_names.copy()
        ctx.state.batch_results.clear()

        n = len(sample_names)
        iris_log("IRIS-Viewer", "Batch", f"Starting batch processing across {n} sample(s)...")
        ctx.control_widget.btn_batch.setText(f"Processing 0/{n}...")
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

        # Snapshot all UI configuration on main thread BEFORE entering worker
        scale_val: float = (
            float(ctx.control_widget.spin_scale.value())
            if getattr(ctx, "control_widget", None) is not None
            else 1.0
        )
        markers: list = list(ctx.state.markers)
        device: str = _detect_device()
        backends: dict = dict(ctx.pipeline_service.backends)

        group_sequences: dict = {
            name: SegmentationSequence(steps=list(steps))
            for name, steps in groups.items()
        }

        dp = getattr(ctx, "disagreement_panel", None)
        threshold: float = dp.get_threshold() if dp is not None else 0.65
        compare_mode: str = (
            dp.combo_compare.currentText()
            if dp is not None and hasattr(dp, "combo_compare")
            else "Nuclear Masks"
        )
        preferred_method: Optional[str] = getattr(ctx.state, "active_segmentation_method", None)
        if not preferred_method and dp is not None and hasattr(dp, "get_active_method"):
            preferred_method = dp.get_active_method() or None
        selected_qc_methods: Optional[List[str]] = None
        if dp is not None and hasattr(dp, "get_selected_methods") and getattr(dp, "_methods", []):
            sel = dp.get_selected_methods()
            if sel:
                selected_qc_methods = sel

        fp = getattr(ctx, "feature_panel", None)
        compartments: Optional[List[str]] = (
            fp.get_selected_compartments()
            if fp is not None and hasattr(fp, "get_selected_compartments")
            else None
        )
        selected_morphology: Optional[List[str]] = (
            [k for k, chk in fp.morph_checkboxes.items() if chk.isChecked()]
            if fp is not None and hasattr(fp, "morph_checkboxes")
            else None
        )
        selected_intensity: Optional[List[str]] = (
            [k for k, chk in fp.intensity_checkboxes.items() if chk.isChecked()]
            if fp is not None and hasattr(fp, "intensity_checkboxes")
            else None
        )

        gpu_lock = self._gpu_lock

        @thread_worker
        def _process_sample():
            raw_image, _ = load_multichannel_tiff(img_path)
            raw_image = np.asarray(raw_image)

            batch_seg: Dict[str, Dict[str, Optional[object]]] = {}
            for grp_name, sequence in group_sequences.items():
                result = execute_group_sync(
                    group_name=grp_name,
                    sequence=sequence,
                    image=raw_image,
                    backends=backends,
                    device=device,
                    gpu_lock=gpu_lock,
                )
                if result.get("nuclear") is not None or result.get("cytoplasm") is not None:
                    batch_seg[grp_name] = result

            if not batch_seg:
                return sample_name, None

            active_name = (
                preferred_method
                if preferred_method and preferred_method in batch_seg
                else next(iter(batch_seg))
            )
            active_res = batch_seg[active_name]
            nuc = active_res.get("nuclear")
            cyto = active_res.get("cytoplasm")
            primary_mask = nuc if nuc is not None else cyto
            cyto_arg = cyto if nuc is not None else None

            features_df = extract_cell_features(
                mask=primary_mask,
                raw_image=raw_image,
                markers=markers,
                scale_factor=scale_val,
                selected_morphology=selected_morphology,
                selected_intensity=selected_intensity,
                cytoplasm_mask=cyto_arg,
                compartments=compartments,
            )

            out_dir = img_path.parent
            export_segmentation_mask(
                out_dir / f"{sample_name}_segmentation_mask.tiff",
                primary_mask,
                cyto_arg,
            )
            export_features_csv(
                out_dir / f"{sample_name}_features.csv",
                features_df,
                sample_name,
            )

            total_cells = int(len(features_df))
            methods_used = list(batch_seg.keys())

            if len(batch_seg) >= 2:
                # Filter segmentation dicts per compare_mode so missing compartments don't crash
                filtered_seg: Dict[str, Dict[str, Optional[object]]] = {}
                for m_name, res in batch_seg.items():
                    if compare_mode == "Nuclear Masks" and res.get("nuclear") is not None:
                        filtered_seg[m_name] = {"nuclear": res["nuclear"], "cytoplasm": None}
                    elif compare_mode == "Cytoplasm Masks" and res.get("cytoplasm") is not None:
                        filtered_seg[m_name] = {"nuclear": None, "cytoplasm": res["cytoplasm"]}
                    elif compare_mode == "Both":
                        filtered_seg[m_name] = res

                if len(filtered_seg) >= 2:
                    ref = active_name if active_name in filtered_seg else None
                    report = compute_model_disagreement(
                        filtered_seg,
                        compare_mode=compare_mode,
                        reference_method=ref,
                        iou_threshold=threshold,
                        selected_methods=selected_qc_methods,
                    )
                    export_disagreement_map(
                        out_dir / f"{sample_name}_disagreement_map.tiff",
                        report.disagreement_heatmap,
                    )
                    n_split = int(report.category_counts.get("Split", 0))
                    n_merge = int(report.category_counts.get("Merge", 0))
                    n_shift = int(report.category_counts.get("Boundary Shift", 0))
                    n_excl = int(report.category_counts.get("Model-Exclusive", 0))
                    return sample_name, {
                        "active_method": active_name,
                        "total_cells": total_cells,
                        "agreed_cells": int(report.agreed_cells),
                        "flagged_regions": int(report.flagged_regions),
                        "mean_iou": float(report.mean_iou),
                        "split_count": n_split,
                        "merge_count": n_merge,
                        "boundary_shift_count": n_shift,
                        "model_exclusive_count": n_excl,
                        "high_conf": int(report.agreed_cells),
                        "med_conf": n_shift,
                        "low_conf": n_split + n_merge + n_excl,
                        "mean_confidence": float(report.mean_iou),
                        "methods": methods_used,
                    }

            return sample_name, {
                "active_method": active_name,
                "total_cells": total_cells,
                "agreed_cells": total_cells,
                "flagged_regions": 0,
                "mean_iou": 1.0,
                "split_count": 0,
                "merge_count": 0,
                "boundary_shift_count": 0,
                "model_exclusive_count": 0,
                "high_conf": total_cells,
                "med_conf": 0,
                "low_conf": 0,
                "mean_confidence": 1.0,
                "methods": methods_used,
            }

        worker = _process_sample()
        worker.returned.connect(lambda res: self._on_sample_done(res, groups))
        if hasattr(ctx, "track_worker") and callable(ctx.track_worker):
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
        log_error("IRIS-Viewer", "Batch", f"Sample '{sample_name}' failed: {exc}")
        ctx.state.batch_results[sample_name] = {
            "active_method": "",
            "total_cells": 0,
            "agreed_cells": 0,
            "flagged_regions": 0,
            "mean_iou": 0.0,
            "split_count": 0,
            "merge_count": 0,
            "boundary_shift_count": 0,
            "model_exclusive_count": 0,
            "high_conf": 0,
            "med_conf": 0,
            "low_conf": 0,
            "mean_confidence": 0.0,
            "methods": [],
            "error": str(exc),
        }

        done = len(ctx.state.batch_results)
        total = done + len(ctx.state.batch_queue)
        ctx.control_widget.btn_batch.setText(f"Processing {done}/{total}...")

        self._batch_process_next(groups)

    def _on_sample_done(self, result: tuple, groups: dict) -> None:
        ctx = self.ctx
        sample_name, stats = result

        ctx.state.batch_results[sample_name] = stats or {
            "active_method": "",
            "total_cells": 0,
            "agreed_cells": 0,
            "flagged_regions": 0,
            "mean_iou": 0.0,
            "split_count": 0,
            "merge_count": 0,
            "boundary_shift_count": 0,
            "model_exclusive_count": 0,
            "high_conf": 0,
            "med_conf": 0,
            "low_conf": 0,
            "mean_confidence": 0.0,
            "methods": [],
        }

        done = len(ctx.state.batch_results)
        total = done + len(ctx.state.batch_queue)
        n_cells = int(stats.get("total_cells", 0)) if stats else 0
        iris_log(
            "IRIS-Viewer",
            "Batch",
            f"[{done}/{total}] '{sample_name}' complete ({n_cells:,} cells).",
        )
        ctx.control_widget.btn_batch.setText(f"Processing {done}/{total}...")

        self._batch_process_next(groups)

    def _on_batch_complete(self) -> None:
        ctx = self.ctx
        ctx.control_widget.btn_batch.setText("Batch Process All Samples")
        ctx.control_widget.btn_batch.setEnabled(True)

        if ctx.state.batch_results and ctx.state.image_dir:
            summary_path = ctx.state.image_dir / "batch_summary.csv"
            export_batch_summary(summary_path, ctx.state.batch_results)
            if summary_path.is_file():
                rows = []
                for sample_name, res in ctx.state.batch_results.items():
                    rows.append({
                        "Sample": sample_name,
                        "Active_Method": res.get("active_method", ""),
                        "Total_Cells": res.get("total_cells", 0),
                        "Agreed_Cells": res.get("agreed_cells", res.get("high_conf", 0)),
                        "Flagged_Regions": res.get("flagged_regions", res.get("low_conf", 0)),
                        "Mean_IoU": res.get("mean_iou", res.get("mean_confidence", 0.0)),
                        "Split_Count": res.get("split_count", 0),
                        "Merge_Count": res.get("merge_count", 0),
                        "Boundary_Shift_Count": res.get("boundary_shift_count", 0),
                        "Model_Exclusive_Count": res.get("model_exclusive_count", 0),
                        "Methods_Used": ", ".join(res.get("methods", [])),
                    })
                pd.DataFrame(rows).to_csv(str(summary_path), index=False)

        n_samples = len(ctx.state.batch_results)
        n_success = sum(
            1 for s in ctx.state.batch_results.values() if s and s.get("total_cells", 0) > 0
        )
        iris_log(
            "IRIS-Viewer",
            "Batch",
            f"Batch complete: {n_success}/{n_samples} sample(s) succeeded.",
        )

        QMessageBox.information(
            ctx.control_widget,
            "Batch Complete",
            f"Processed {n_samples} samples ({n_success} successful).\n"
            "Results saved alongside source images."
            + (
                f"\nSummary: {ctx.state.image_dir / 'batch_summary.csv'}"
                if ctx.state.image_dir
                else ""
            ),
        )

    # Helpers

    def _get_pipeline_groups(self) -> dict:
        """Read the current tab-based pipeline configuration from the UI."""
        ctx = self.ctx
        panel = ctx.segmentation_panel
        tw = (
            panel.pipeline_builder.tab_widget
            if hasattr(panel, "pipeline_builder")
            else panel.tab_widget
        )
        groups: dict = {}
        for i in range(tw.count()):
            name = tw.tabText(i)
            steps = tw.widget(i).get_steps()
            if steps:
                groups[name] = steps
        return groups
