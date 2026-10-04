"""
IRIS-Viewer Controller
Thin coordinator: owns the napari viewer, creates widgets and services,
wires Qt signals, and drives the segmentation pipeline, consensus generation,
feature extraction, export, and session save/restore.

Heavy per-domain logic has been moved to:
  app/services/image_service.py   - image loading & display
  app/services/batch_service.py   - multi-sample batch loop
  app/services/review_service.py  - confidence overlay & interactive review
"""
from __future__ import annotations

import types as pytypes
from pathlib import Path

import numpy as np
import napari
from napari.qt import thread_worker
from napari.utils.colormaps import Colormap
from qtpy.QtWidgets import QFileDialog, QMessageBox

from iris_viewer.core.state import ViewerAppState
from iris_viewer.engine.pipeline import PipelineService, SegmentationSequence
from iris_viewer.engine.consensus import compute_consensus_mask
from iris_viewer.engine.feature_extraction import extract_cell_features
from iris_viewer.io_utils.exporters import (
    export_confidence_map,
    export_consensus_mask,
    export_features_csv,
)

from iris_viewer.widgets.control import ViewerControlWidget
from iris_viewer.widgets.segmentation_panel import SegmentationPanelWidget
from iris_viewer.widgets.consensus_panel import ConsensusPanelWidget
from iris_viewer.widgets.feature_panel import FeaturePanelWidget
from iris_viewer.widgets.legend import ViewerLegendWidget
from iris_viewer.widgets.multi_sample import MultiSampleComparisonWidget
from iris_viewer.widgets.training_panel import TrainingDataWidget

from iris_viewer.app.services import BatchService, ImageService, ReviewService, MultiSampleService, TrainingService


class ViewerController:
    """
    Thin MVC coordinator for IRIS-Viewer.

    Responsibilities:
    - Instantiate and dock all Qt widgets.
    - Assemble the shared context namespace (ctx) passed to services.
    - Bind Qt signals to service methods or local handler methods.
    - Drive the segmentation pipeline, consensus, feature extraction, export,
      and session persistence (logic too tightly coupled to remain here cleanly).
    """

    def __init__(self) -> None:
        self.state = ViewerAppState()
        self.pipeline_service = PipelineService()
        self.added_legend_entries: set = set()
        self._active_workers: list = []

        # ── napari Viewer ─────────────────────────────────────────────────
        self.viewer = napari.Viewer(title="IRIS-Viewer")
        try:
            self.viewer.scale_bar.visible = True
            self.viewer.scale_bar.ticks = False
            self.viewer.scale_bar.box = False
            self.viewer.scale_bar.font_size = 26.0
            self.viewer.scale_bar.position = "bottom_left"
        except AttributeError:
            pass

        # ── Widgets ───────────────────────────────────────────────────────
        self.control_widget     = ViewerControlWidget(viewer=self.viewer)
        self.segmentation_panel = SegmentationPanelWidget()
        self.consensus_panel    = ConsensusPanelWidget()
        self.feature_panel      = FeaturePanelWidget()
        self.legend_widget      = ViewerLegendWidget(self.viewer)
        self.comparison_widget  = MultiSampleComparisonWidget()
        self.training_panel     = TrainingDataWidget()

        for widget, name in [
            (self.control_widget,     "Control"),
            (self.segmentation_panel, "Segmentation"),
            (self.consensus_panel,    "Consensus"),
            (self.feature_panel,      "Features"),
            (self.comparison_widget,  "Multi-Sample"),
            (self.training_panel,     "Training"),
            (self.legend_widget,      "Legend"),
        ]:
            self.viewer.window.add_dock_widget(widget, name=name, area="right")

        dw = self.viewer.window.dock_widgets
        qt = self.viewer.window._qt_window
        qt.tabifyDockWidget(dw["Control"].parent(),       dw["Segmentation"].parent())
        qt.tabifyDockWidget(dw["Segmentation"].parent(),  dw["Consensus"].parent())
        qt.tabifyDockWidget(dw["Consensus"].parent(),     dw["Features"].parent())
        qt.tabifyDockWidget(dw["Features"].parent(),      dw["Multi-Sample"].parent())
        qt.tabifyDockWidget(dw["Multi-Sample"].parent(),  dw["Training"].parent())
        qt.tabifyDockWidget(dw["Training"].parent(),      dw["Legend"].parent())
        dw["Control"].parent().raise_()

        # ── Shared Context namespace ───────────────────────────────────────
        # Both the controller and all services read/write the same objects.
        # Mutable containers (lists, sets, dicts) are shared by reference.
        self._ctx = pytypes.SimpleNamespace(
            viewer=self.viewer,
            state=self.state,
            pipeline_service=self.pipeline_service,
            control_widget=self.control_widget,
            segmentation_panel=self.segmentation_panel,
            consensus_panel=self.consensus_panel,
            feature_panel=self.feature_panel,
            legend_widget=self.legend_widget,
            comparison_widget=self.comparison_widget,
            training_panel=self.training_panel,
            added_legend_entries=self.added_legend_entries,  # same set object
            active_workers=self._active_workers,             # same list object
            flagged_cells=[],
            current_issue_idx=-1,
        )

        # ── Services ──────────────────────────────────────────────────────
        self.image_svc        = ImageService(self._ctx)
        self.batch_svc        = BatchService(self._ctx)
        self.review_svc       = ReviewService(self._ctx)
        self.multi_sample_svc = MultiSampleService(self._ctx)
        self.training_svc     = TrainingService(self._ctx)

        # ── Init ──────────────────────────────────────────────────────────
        self._configure_backend_availability()
        self._bind_control_signals()
        self._bind_segmentation_signals()
        self._bind_consensus_signals()
        self._bind_feature_signals()
        self._bind_multi_sample_signals()
        self._bind_training_signals()

    # ── Configuration ──────────────────────────────────────────────────────

    def _configure_backend_availability(self) -> None:
        self.segmentation_panel.update_available_methods(
            self.pipeline_service.get_available_methods()
        )

    # ── Signal Binding ─────────────────────────────────────────────────────

    def _bind_control_signals(self) -> None:
        cw = self.control_widget
        cw.loadDirectoryTriggered.connect(self.image_svc.on_load_directory)
        cw.sampleSelected.connect(self.image_svc.on_sample_selected)
        cw.loadExistingMaskTriggered.connect(self.image_svc.on_load_existing_mask)
        cw.loadChannelColorTriggered.connect(self.image_svc.load_channel_colors)
        cw.scaleFactorChanged.connect(self.image_svc.update_scale)
        cw.viewLockToggled.connect(self._toggle_view_lock)
        cw.exportMaskTriggered.connect(self._export_mask)
        cw.exportFeaturesTriggered.connect(self._export_features)
        cw.exportConfidenceTriggered.connect(self._export_confidence)
        cw.batchProcessTriggered.connect(self.batch_svc.run_batch_processing)
        # Optional session signals (added in a future control_widget update)
        if hasattr(cw, 'saveSessionTriggered'):
            cw.saveSessionTriggered.connect(self._save_session)
        if hasattr(cw, 'loadSessionTriggered'):
            cw.loadSessionTriggered.connect(self._load_session)

    def _bind_segmentation_signals(self) -> None:
        self.segmentation_panel.runSequence.connect(self._run_segmentation_pipeline)
        self.segmentation_panel.cancelSequence.connect(self._cancel_segmentation)

    def _bind_consensus_signals(self) -> None:
        cp = self.consensus_panel
        cp.generateConsensusRequested.connect(self._generate_consensus)
        cp.thresholdChanged.connect(self.review_svc.update_confidence_overlay)
        cp.cellOverrideRequested.connect(self.review_svc.apply_manual_override)
        cp.btn_next_issue.clicked.connect(self.review_svc.next_issue)
        cp.btn_prev_issue.clicked.connect(self.review_svc.prev_issue)
        cp.chk_enable_review.toggled.connect(self._toggle_interactive_review)
        self.legend_widget.layerSelected.connect(self._select_layer)

    def _bind_feature_signals(self) -> None:
        self.feature_panel.extractTriggered.connect(self._run_feature_extraction)
        self.feature_panel.exportTriggered.connect(self._export_features)

    def _bind_multi_sample_signals(self) -> None:
        """Wire the MultiSampleComparisonWidget signals to MultiSampleService."""
        cw = self.comparison_widget
        cw.masterFileTriggered.connect(self.multi_sample_svc.set_master_file)
        cw.addSamplesTriggered.connect(self.multi_sample_svc.add_samples_to_canvas)
        cw.clearCanvasTriggered.connect(self.multi_sample_svc.clear_canvas)
        # Channel colour loading can also come from the viewer control widget:
        if hasattr(self.control_widget, 'loadCelltypeColorTriggered'):
            self.control_widget.loadCelltypeColorTriggered.connect(
                self.multi_sample_svc.load_celltype_colors
            )

    def _bind_training_signals(self) -> None:
        """Wire the TrainingDataWidget to TrainingService."""
        tp = self.training_panel
        tp.autoSelectTriggered.connect(self.training_svc.auto_select)
        tp.exportTriggered.connect(self.training_svc.export)

    # ── Misc Controls ──────────────────────────────────────────────────────

    def _toggle_interactive_review(self, enabled: bool) -> None:
        cb = self.review_svc.on_mouse_click
        if enabled and cb not in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.append(cb)
        elif not enabled and cb in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.remove(cb)

    def _select_layer(self, layer_name: str) -> None:
        self.viewer.layers.selection.clear()
        for layer in self.viewer.layers:
            base = layer.name.split(" [")[0] if " [" in layer.name else layer.name
            if base == layer_name:
                self.viewer.layers.selection.add(layer)

    def _toggle_view_lock(self, is_locked: bool) -> None:
        self.viewer.camera.mouse_pan  = not is_locked
        self.viewer.camera.mouse_zoom = not is_locked

    # ── Segmentation Pipeline ───────────────────────────────────────────────

    def _run_segmentation_pipeline(self, groups: dict) -> None:
        if self.state.raw_image is None:
            QMessageBox.warning(
                self.control_widget, "No Image",
                "Load an image before running segmentation."
            )
            return

        self.state.segmentation_results.clear()
        self.state.segmentation_status.clear()

        for group_name, steps in groups.items():
            for idx in range(len(steps)):
                step_key = f"{group_name}_{idx}"
                self.state.segmentation_status[step_key] = "Queued"
                self.segmentation_panel.update_step_status(group_name, idx, "Queued")

        self.segmentation_panel.set_running_state(True)

        self.pipeline_service.execute_pipeline_groups(
            groups={g: SegmentationSequence(steps=s) for g, s in groups.items()},
            image=self.state.raw_image,
            on_step_complete=self._on_single_segmentation_done,
            on_group_complete=self._on_group_complete,
            on_all_complete=self._on_all_segmentations_complete,
            on_error=self._on_segmentation_error,
        )

    def _on_segmentation_error(
        self, group_name: str, step_idx: int, method_name: str, error_msg: str
    ) -> None:
        step_key = f"{group_name}_{step_idx}"
        self.state.segmentation_status[step_key] = "Failed"
        self.segmentation_panel.update_step_status(group_name, step_idx, "Failed")
        print(f"[Pipeline Error] {group_name}/{method_name} (step {step_idx}): {error_msg}")

    def _on_single_segmentation_done(
        self, group_name: str, step_idx: int, method_name: str, seg_result: dict
    ) -> None:
        step_key = f"{group_name}_{step_idx}"
        self.state.segmentation_status[step_key] = "Done"
        self.segmentation_panel.update_step_status(group_name, step_idx, "Done")

    def _on_group_complete(self, group_name: str, seg_result: dict) -> None:
        self.state.segmentation_results[group_name] = seg_result
        scale_val = self.control_widget.spin_scale.value()

        def _add_mask(data: np.ndarray | None, suffix: str) -> None:
            if data is None:
                return
            layer_name = f"{group_name} {suffix}"
            try:
                lyr = self.viewer.add_labels(
                    data, name=layer_name, opacity=0.3,
                    visible=False, contour=1, scale=(scale_val, scale_val),
                )
            except TypeError:
                lyr = self.viewer.add_labels(
                    data, name=layer_name, opacity=0.3,
                    visible=False, scale=(scale_val, scale_val),
                )
            if layer_name not in self.added_legend_entries:
                self.legend_widget.add_segmentation_row(lyr, layer_name, '#888888')
                self.added_legend_entries.add(layer_name)

        _add_mask(seg_result.get('nuclear'), "Nuclear Mask")
        _add_mask(seg_result.get('cytoplasm'), "Cyto Mask")

    def _on_all_segmentations_complete(self) -> None:
        self.segmentation_panel.set_running_state(False)
        masks = {
            m: res for m, res in self.state.segmentation_results.items()
            if res.get('nuclear') is not None or res.get('cytoplasm') is not None
        }
        if not masks:
            QMessageBox.warning(
                self.control_widget, "No Results",
                "All segmentation groups failed.  Check dependencies and parameters."
            )
            return

        n = len(masks)
        self.control_widget.update_dashboard(seg_status=f"{n} pipeline(s) applied")
        self.segmentation_panel.lbl_consensus_status.setText(
            f"Ready for consensus: {n} pipeline(s)"
        )

    def _cancel_segmentation(self) -> None:
        self.pipeline_service.cancel_all()
        self.segmentation_panel.set_running_state(False)

        for key in list(self.state.segmentation_status.keys()):
            if self.state.segmentation_status[key] in ("Running", "Queued"):
                self.state.segmentation_status[key] = "Cancelled"
                # Key format: "{group_name}_{step_idx}"
                # rsplit from the right ensures group names containing underscores
                # are preserved correctly.
                parts = key.rsplit("_", 1)
                if len(parts) == 2:
                    group_name, step_str = parts
                    try:
                        step_idx = int(step_str)
                        self.segmentation_panel.update_step_status(
                            group_name, step_idx, "Cancelled"
                        )
                    except ValueError:
                        pass

        self.state.batch_queue.clear()
        self.control_widget.btn_batch.setText("Batch Process All Samples")
        self.control_widget.btn_batch.setEnabled(True)

        QMessageBox.information(
            self.control_widget, "Cancellation Sent",
            "Segmentation cancellation requested.\n\n"
            "Note: a running deep-learning operation must finish its current "
            "tensor pass before the worker thread actually closes."
        )

    # ── Consensus Generation ────────────────────────────────────────────────

    def _generate_consensus(self) -> None:
        compare_mode = self.consensus_panel.combo_compare.currentText()

        seg_results: dict = {}
        for m, res in self.state.segmentation_results.items():
            if compare_mode == "Nuclear Masks":
                if res.get('nuclear') is not None:
                    seg_results[m] = {'nuclear': res['nuclear'], 'cytoplasm': None}
            elif compare_mode == "Cytoplasm Masks":
                if res.get('cytoplasm') is not None:
                    seg_results[m] = {'nuclear': None, 'cytoplasm': res['cytoplasm']}
            else:
                if res.get('nuclear') is not None or res.get('cytoplasm') is not None:
                    seg_results[m] = res

        if not seg_results:
            QMessageBox.warning(
                self.control_widget, "No Masks",
                "Run segmentation before generating consensus."
            )
            return

        self.consensus_panel.set_generating_state(True)

        @thread_worker
        def _run() -> tuple:
            return compute_consensus_mask(seg_results)

        worker = _run()
        worker.returned.connect(self._on_consensus_complete)
        self._active_workers.append(worker)
        worker.start()

    def _on_consensus_complete(self, result: tuple) -> None:
        self.consensus_panel.set_generating_state(False)
        consensus_mask, confidence_map, per_cell_conf, cell_method_map = result

        self.state.consensus_mask    = consensus_mask
        self.state.confidence_map    = confidence_map
        self.state.per_cell_confidence = per_cell_conf
        self.state.cell_method_map   = cell_method_map   # properly typed public field

        scale_val = self.control_widget.spin_scale.value()

        if "Consensus Mask" in self.viewer.layers:
            self.viewer.layers["Consensus Mask"].data = consensus_mask
        else:
            try:
                cl = self.viewer.add_labels(
                    consensus_mask, name="Consensus Mask",
                    opacity=1.0, visible=True, contour=1,
                    scale=(scale_val, scale_val),
                )
            except TypeError:
                cl = self.viewer.add_labels(
                    consensus_mask, name="Consensus Mask",
                    opacity=1.0, visible=True,
                    scale=(scale_val, scale_val),
                )
            if "Consensus Mask" not in self.added_legend_entries:
                self.legend_widget.add_consensus_row(cl, "Consensus Mask", '#44AA66')
                self.added_legend_entries.add("Consensus Mask")

        conf_cmap = Colormap(
            colors=np.array([
                [0.0, 0.0, 0.0, 0.0],
                [0.8, 0.2, 0.2, 1.0],
                [0.8, 0.8, 0.2, 1.0],
                [0.2, 0.8, 0.2, 1.0],
            ]),
            controls=np.array([0.0, 0.001, 0.5, 1.0]),
            name='confidence_ryg',
        )
        if "Confidence Map" in self.viewer.layers:
            self.viewer.layers["Confidence Map"].data = confidence_map
        else:
            conf_lyr = self.viewer.add_image(
                confidence_map, name="Confidence Map",
                colormap=conf_cmap, blending='additive',
                visible=False, scale=(scale_val, scale_val),
                contrast_limits=[0, 1],
            )
            if "Confidence Map" not in self.added_legend_entries:
                self.legend_widget.add_consensus_row(conf_lyr, "Confidence Map", '#FFAA00')
                self.added_legend_entries.add("Confidence Map")

        threshold = self.consensus_panel.get_threshold()
        self.review_svc.update_confidence_overlay(threshold)

        n_cells = len(per_cell_conf)
        n_methods = sum(
            1 for r in self.state.segmentation_results.values()
            if r.get('nuclear') is not None
        )
        self.control_widget.update_dashboard(total_cells=f"{n_cells:,}")
        self.segmentation_panel.lbl_consensus_status.setText(
            f"Consensus: {n_cells:,} cells from {n_methods} method(s)"
        )
        QMessageBox.information(
            self.control_widget, "Success",
            "Consensus mask and confidence contours generated successfully."
        )

    # ── Feature Extraction ──────────────────────────────────────────────────

    def _run_feature_extraction(
        self, selected_morphology: list, selected_intensity: list
    ) -> None:
        if self.state.consensus_mask is None:
            QMessageBox.warning(self.control_widget, "No Mask", "Run segmentation first.")
            return
        if self.state.raw_image is None:
            QMessageBox.warning(self.control_widget, "No Image", "Load an image first.")
            return

        self.feature_panel.set_extracting_state(True)
        scale_val = self.control_widget.spin_scale.value()

        @thread_worker
        def _extract():
            return extract_cell_features(
                mask=self.state.consensus_mask,
                raw_image=self.state.raw_image,
                markers=self.state.markers,
                scale_factor=scale_val,
                selected_morphology=selected_morphology,
                selected_intensity=selected_intensity,
            )

        worker = _extract()
        worker.returned.connect(self._on_features_extracted)
        self._active_workers.append(worker)
        worker.start()

    def _on_features_extracted(self, df) -> None:
        self.state.features_df = df
        self.feature_panel.set_extracting_state(False)
        self.feature_panel.populate_preview(df)

    # ── Exports ─────────────────────────────────────────────────────────────

    def _export_mask(self) -> None:
        if self.state.consensus_mask is None:
            QMessageBox.warning(self.control_widget, "No Mask", "No consensus mask to export.")
            return
        sample = self.state.current_sample_name or "sample"
        filepath, _ = QFileDialog.getSaveFileName(
            self.control_widget, "Export Consensus Mask",
            f"{sample}_mask.tiff", "TIFF (*.tiff)"
        )
        if filepath:
            cyto = next(
                (r['cytoplasm'] for r in self.state.segmentation_results.values()
                 if r.get('cytoplasm') is not None),
                None,
            )
            export_consensus_mask(Path(filepath), self.state.consensus_mask, cyto)
            QMessageBox.information(self.control_widget, "Export Complete", "Consensus mask saved.")

    def _export_features(self) -> None:
        if self.state.features_df is None or self.state.features_df.empty:
            QMessageBox.warning(self.control_widget, "No Features", "Extract features first.")
            return
        sample = self.state.current_sample_name or "sample"
        filepath, _ = QFileDialog.getSaveFileName(
            self.control_widget, "Export Features CSV",
            f"{sample}_features.csv", "CSV (*.csv)"
        )
        if filepath:
            export_features_csv(Path(filepath), self.state.features_df, sample)
            QMessageBox.information(self.control_widget, "Export Complete", "Features CSV saved.")

    def _export_confidence(self) -> None:
        if self.state.confidence_map is None:
            QMessageBox.warning(self.control_widget, "No Confidence Map", "Run consensus first.")
            return
        sample = self.state.current_sample_name or "sample"
        filepath, _ = QFileDialog.getSaveFileName(
            self.control_widget, "Export Confidence Map",
            f"{sample}_confidence.tiff", "TIFF (*.tiff)"
        )
        if filepath:
            export_confidence_map(Path(filepath), self.state.confidence_map)
            QMessageBox.information(self.control_widget, "Export Complete", "Confidence map saved.")

    # ── Session ─────────────────────────────────────────────────────────────

    def _save_session(self) -> None:
        sample = self.state.current_sample_name or "session"
        filepath, _ = QFileDialog.getSaveFileName(
            self.control_widget, "Save Session",
            f"{sample}_session.json", "JSON (*.json)"
        )
        if filepath:
            threshold = self.consensus_panel.get_threshold()
            self.state.save_session(Path(filepath), consensus_threshold=threshold)
            QMessageBox.information(
                self.control_widget, "Session Saved",
                f"Session saved to:\n{filepath}"
            )

    def _load_session(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self.control_widget, "Load Session", "", "JSON (*.json)"
        )
        if not filepath:
            return
        try:
            restored, raw_data = ViewerAppState.load_session(Path(filepath))

            # Restore non-array fields into the live state object
            self.state.markers             = restored.markers
            self.state.per_cell_confidence = restored.per_cell_confidence
            self.state.manual_overrides    = restored.manual_overrides
            self.state.persistent_settings = restored.persistent_settings

            # Restore consensus threshold slider
            thresh = raw_data.get("consensus_threshold", 0.66)
            self.consensus_panel.slider_threshold.setValue(int(thresh * 100))

            # Reload image directory (triggers discover + sample load)
            if restored.image_dir and restored.image_dir.is_dir():
                self.image_svc.on_load_directory(str(restored.image_dir))

            QMessageBox.information(
                self.control_widget, "Session Loaded",
                f"Session restored from:\n{filepath}"
            )
        except Exception as exc:
            QMessageBox.critical(
                self.control_widget, "Error",
                f"Failed to load session:\n{exc}"
            )