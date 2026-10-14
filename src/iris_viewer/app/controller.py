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
from iris_viewer.io.exporters import (
    export_confidence_map,
    export_consensus_mask,
    export_features_csv,
)

from iris_viewer.widgets import (
    ImageControlWidget,
    SegmentationPanelWidget,
    ConsensusPanelWidget,
    FeaturePanelWidget,
    PhenotyperPanelWidget,
    ViewerLegendWidget,
    MultiSampleComparisonWidget,
    IrisCellularNeighborhoodsWidget,
    IrisMarkerExpressionWidget,
)

from iris_viewer.app.services import (
    BatchService,
    ReviewService,
    BoundaryService,
    NeighborhoodService,
    ImageService,
    MultiSampleService,
    CurationService,
)


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
            self.viewer.scale_bar.visible = False
            self.viewer.scale_bar.unit = "px"
            self.viewer.scale_bar.ticks = False
            self.viewer.scale_bar.box = False
            self.viewer.scale_bar.font_size = 26.0
            self.viewer.scale_bar.position = "bottom_left"
        except AttributeError:
            pass

        # ── Widgets ───────────────────────────────────────────────────────
        self.control_widget      = ImageControlWidget(viewer=self.viewer)
        self.segmentation_panel  = SegmentationPanelWidget()
        self.consensus_panel     = self.segmentation_panel.consensus_panel
        self.feature_panel       = self.segmentation_panel.feature_panel
        self.phenotyper_panel    = PhenotyperPanelWidget()
        self.legend_widget       = ViewerLegendWidget(self.viewer)
        self.comparison_widget   = self.control_widget.comparison_widget
        self.marker_widget       = IrisMarkerExpressionWidget()
        self.neighborhood_widget = IrisCellularNeighborhoodsWidget()
        self.neighborhood_widget.legend_widget_ref = self.legend_widget

        from qtpy.QtWidgets import QDockWidget

        docks = {}
        for widget, name in [
            (self.control_widget,      "Control"),
            (self.comparison_widget,   "TMA"),
            (self.segmentation_panel,  "Segmentation"),
            (self.phenotyper_panel,    "Phenotyper"),
            (self.marker_widget,       "Marker"),
            (self.neighborhood_widget, "Neighborhoods"),
        ]:
            docks[name] = self.viewer.window.add_dock_widget(widget, name=name, area="right")
            
        legend_dock = self.viewer.window.add_dock_widget(self.legend_widget, name="Legend", area="left")
        
        self.comparison_widget.show()

        qt = self.viewer.window._qt_window

        def _get_dock_wrapper(d):
            if isinstance(d, QDockWidget):
                return d
            if hasattr(d, 'parent') and isinstance(d.parent(), QDockWidget):
                return d.parent()
            return d

        control_dock = _get_dock_wrapper(docks["Control"])
        tma_dock     = _get_dock_wrapper(docks["TMA"])
        seg_dock     = _get_dock_wrapper(docks["Segmentation"])
        pheno_dock   = _get_dock_wrapper(docks["Phenotyper"])
        marker_dock  = _get_dock_wrapper(docks["Marker"])
        neigh_dock   = _get_dock_wrapper(docks["Neighborhoods"])

        qt.tabifyDockWidget(control_dock, tma_dock)
        qt.tabifyDockWidget(tma_dock,     seg_dock)
        qt.tabifyDockWidget(seg_dock,     pheno_dock)
        qt.tabifyDockWidget(pheno_dock,   marker_dock)
        qt.tabifyDockWidget(marker_dock,  neigh_dock)
        control_dock.raise_()

        # Tabify Legend with napari's layer list
        leg_wrapper = _get_dock_wrapper(legend_dock)
        for d in qt.findChildren(QDockWidget):
            if d.windowTitle().lower() == "layer list":
                qt.tabifyDockWidget(d, leg_wrapper)
                d.raise_()
                break

        # ── Adjust Main Window Size ───────────────────────────────────────
        from qtpy.QtCore import QTimer
        def enforce_window_size():
            screen = qt.screen()
            if screen:
                geom = screen.availableGeometry()
                # Enforce a reasonable default size (e.g., 85% of screen)
                qt.resize(int(geom.width() * 0.85), int(geom.height() * 0.85))
                qt.move(geom.center() - qt.rect().center())
            qt.showMaximized()
            
        # Delay the resize slightly so it overrides napari's internal settings restoration
        QTimer.singleShot(200, enforce_window_size)

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
            phenotyper_panel=self.phenotyper_panel,
            legend_widget=self.legend_widget,
            comparison_widget=self.comparison_widget,
            neighborhood_widget=self.neighborhood_widget,
            marker_widget=self.marker_widget,
            expression_widget=self.marker_widget,
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
        self.curation_svc     = CurationService(self._ctx)
        self.boundary_svc     = BoundaryService(self._ctx)
        self.neighborhood_svc = NeighborhoodService(self._ctx)
        self._ctx.review_svc = self.review_svc
        self.segmentation_panel.ctx = self._ctx
        self.phenotyper_panel.ctx = self._ctx

        # ── Replace Napari Welcome Text ───────────────────────────────────────
        from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget, QApplication
        from qtpy.QtCore import Qt
        
        # Suppress default napari welcome widgets
        try:
            for top_widget in QApplication.instance().topLevelWidgets():
                for child in top_widget.findChildren(QWidget):
                    if "Welcome" in type(child).__name__ or type(child).__name__ == "QtWelcomeWidget":
                        child.hide()
                        child.setVisible = lambda x: None
                    elif isinstance(child, QLabel):
                        text = child.text().lower()
                        if "napari" in text or "drag" in text or "shortcuts" in text:
                            child.hide()
                            child.setVisible = lambda x: None
        except Exception:
            pass

        # Suppress Vispy TextVisual elements for legacy napari backends
        try:
            if hasattr(self.viewer.window._qt_viewer, "canvas"):
                scene = self.viewer.window._qt_viewer.canvas.scene
                def hide_vispy_text(node):
                    if type(node).__name__ == "TextVisual":
                        text = getattr(node, "text", "").lower()
                        if "napari" in text or "drag" in text:
                            node.visible = False
                    if hasattr(node, "children"):
                        for c in node.children:
                            hide_vispy_text(c)
                hide_vispy_text(scene)
        except Exception:
            pass

        # Initialize centered IRIS branding animation canvas overlay
        canvas_native = self.viewer.window._qt_viewer.canvas.native
        
        self.iris_welcome = QLabel()
        # Use an opaque black background
        self.iris_welcome.setStyleSheet("background-color: black;")
        self.iris_welcome.setAlignment(Qt.AlignCenter)
        self.iris_welcome.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        from qtpy.QtGui import QMovie
        import os
        gif_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../image/iris_blink.gif"))
        movie = QMovie(gif_path)
        
        # Start and jump to frame 0 to read the original size synchronously
        movie.start()
        movie.jumpToFrame(0)
        orig_size = movie.currentImage().size()
        if not orig_size.isEmpty():
            from qtpy.QtCore import QSize
            scale_factor = 0.5
            movie.setScaledSize(QSize(int(orig_size.width() * scale_factor), int(orig_size.height() * scale_factor)))
            
        self.iris_welcome.setMovie(movie)
        # Keep a reference to prevent garbage collection
        self.iris_welcome._movie = movie

        # Place it in a layout so it centers automatically and covers the canvas
        layout = canvas_native.layout()
        if not layout:
            layout = QVBoxLayout(canvas_native)
            layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(self.iris_welcome) # No alignment flag makes it expand

        # Hide it when a layer is added
        @self.viewer.layers.events.inserted.connect
        def _hide_iris_text(*args):
            self.iris_welcome.hide()
            
        @self.viewer.layers.events.removed.connect
        def _show_iris_text(*args):
            if len(self.viewer.layers) == 0:
                self.iris_welcome.show()

        # ── Init ──────────────────────────────────────────────────────────
        self._configure_backend_availability()
        self._bind_control_signals()
        self._bind_segmentation_signals()
        self._bind_consensus_signals()
        self._bind_feature_signals()
        self._bind_multi_sample_signals()
        self._bind_curation_signals()
        self._bind_neighborhood_signals()

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
        cw.loadChannelColorTriggered.connect(self._load_colors_smart)
        cw.loadCellTypeColorTriggered.connect(self.multi_sample_svc.load_celltype_colors)
        cw.scaleFactorChanged.connect(self.image_svc.update_scale)
        cw.viewLockToggled.connect(self._toggle_view_lock)
        cw.loadAnnotationDataTriggered.connect(self.multi_sample_svc.set_master_file)
        cw.autoSegmentTriggered.connect(self.boundary_svc.auto_segment_boundary)
        cw.saveBoundsTriggered.connect(self.boundary_svc.save_sample_boundaries)
        cw.exportMaskTriggered.connect(self._export_mask)
        cw.exportFeaturesTriggered.connect(self._export_features)
        cw.exportConfidenceTriggered.connect(self._export_confidence)
        cw.exportUpdatedPhenotypesTriggered.connect(self._export_updated_phenotypes)
        cw.batchProcessTriggered.connect(self.batch_svc.run_batch_processing)
        cw.colChanged.connect(self.multi_sample_svc.update_phenotypes_on_canvas)
        cw.colChanged.connect(self._on_phenotype_column_changed)


        # Optional session signals (added in a future control_widget update)
        if hasattr(cw, 'saveSessionTriggered'):
            cw.saveSessionTriggered.connect(self._save_session)
        if hasattr(cw, 'loadSessionTriggered'):
            cw.loadSessionTriggered.connect(self._load_session)

    def _on_phenotype_column_changed(self, col: str) -> None:
        active_df = (
            self.state.master_df
            if (self.state.master_df is not None and not self.state.master_df.empty)
            else self.state.current_mapped_df
        )
        if hasattr(self, "phenotyper_panel") and active_df is not None and not active_df.empty:
            self.phenotyper_panel.set_data(
                df=active_df,
                markers=self.state.markers,
            )
        if hasattr(self, "marker_widget"):
            selected_s = (
                list(self.state.sample_layers_map.keys())
                or ([self.state.current_sample_name] if self.state.current_sample_name else [])
            )
            self.marker_widget.set_data(
                df=active_df,
                markers=self.state.markers,
                celltype_col=col,
                selected_samples=selected_s,
            )


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
        cw.addSamplesTriggered.connect(self.multi_sample_svc.add_samples_to_canvas)
        cw.clearCanvasTriggered.connect(self.multi_sample_svc.clear_canvas)

    def _bind_curation_signals(self) -> None:
        """Bind mouse drag callbacks for curation interactions."""
        self.viewer.mouse_drag_callbacks.append(self.curation_svc.on_right_click)
        self.viewer.mouse_drag_callbacks.append(self.curation_svc.on_quick_curate_click)

    def _bind_neighborhood_signals(self) -> None:
        """Wire the IrisCellularNeighborhoodsWidget signals to NeighborhoodService."""
        nw = self.neighborhood_widget
        nw.runNicheAnalysis.connect(self.neighborhood_svc.execute_niche_analysis)
        nw.nicheMetadataChanged.connect(self.neighborhood_svc.update_niche_canvas_coloration)


    def _load_colors_smart(self, filepath: str) -> None:
        """Parse and apply color map definitions for markers or cell phenotypes.

        Automatically inspects tabular columns to detect whether color codes correspond
        to imaging channels/markers or phenotypic cell types, delegating to the
        appropriate service subsystem.

        Args:
            filepath: Path to the Excel (.xlsx) or CSV color configuration table.
        """
        import re
        import pandas as pd
        path_obj = Path(filepath)
        try:
            df = pd.read_excel(filepath) if path_obj.suffix.lower() in ('.xlsx', '.xls') else pd.read_csv(filepath)
            cols = [re.sub(r'[^a-z0-9]', '', str(c).lower()) for c in df.columns]
        except Exception as e:
            QMessageBox.critical(self.control_widget, "Error", f"Failed to read file:\n{e}")
            return
            
        has_color = any(k in cols for k in ['colors', 'color', 'colours', 'colour', 'hex', 'code'])
        has_markers = has_color and any(k in cols for k in ['markers', 'marker', 'channel', 'channels', 'target', 'targets'])
        has_celltypes = has_color and any(k in cols for k in ['celltypes', 'celltype', 'phenotype', 'phenotypes', 'label', 'labels', 'cluster', 'clusters'])
        
        if not has_markers and not has_celltypes:
            QMessageBox.warning(self.control_widget, "Invalid Format", "File must contain a color column and either marker or cell type names.")
            return
            
        if has_markers:
            self.image_svc.load_channel_colors(filepath)
        if has_celltypes:
            self.multi_sample_svc.load_celltype_colors(filepath)

    # ── Interactive Review & Canvas Controls ──────────────────────────────────

    def _toggle_interactive_review(self, enabled: bool) -> None:
        """Toggle mouse click callbacks for interactive confidence review.

        Args:
            enabled: If True, binds click handler to review low-confidence boundaries.
        """
        cb = self.review_svc.on_mouse_click
        if enabled and cb not in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.append(cb)
        elif not enabled and cb in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.remove(cb)

    def _select_layer(self, layer_name: str) -> None:
        """Set active layer selection in napari's layer list by name.

        Args:
            layer_name: Channel name or layer label to select.
        """
        self.viewer.layers.selection.clear()
        for layer in self.viewer.layers:
            base = layer.name.split(" [")[0] if " [" in layer.name else layer.name
            if base == layer_name:
                self.viewer.layers.selection.add(layer)
                break

    def _toggle_view_lock(self, is_locked: bool) -> None:
        """Lock or unlock canvas pan and zoom interactions.

        Args:
            is_locked: If True, disables camera panning and zooming.
        """
        self.viewer.camera.mouse_pan = not is_locked
        self.viewer.camera.mouse_zoom = not is_locked
        if is_locked:
            self._old_double_clicks = list(self.viewer.mouse_double_click_callbacks)
            self.viewer.mouse_double_click_callbacks.clear()
        elif hasattr(self, "_old_double_clicks"):
            self.viewer.mouse_double_click_callbacks.extend(self._old_double_clicks)
            del self._old_double_clicks

    # ── Segmentation Pipeline ───────────────────────────────────────────────

    def _run_segmentation_pipeline(self, groups: dict) -> None:
        is_multi = len(self.state.multi_sample_images) > 0
        if self.state.raw_image is None and not is_multi:
            QMessageBox.warning(
                self.control_widget, "No Image",
                "Load an image or multiple samples before running segmentation."
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

        if is_multi:
            self._run_multi_segmentation_pipeline(groups)
        else:
            self.pipeline_service.execute_pipeline_groups(
                groups={g: SegmentationSequence(steps=s) for g, s in groups.items()},
                image=self.state.raw_image,
                on_step_complete=self._on_single_segmentation_done,
                on_group_complete=self._on_group_complete,
                on_all_complete=self._on_all_segmentations_complete,
                on_error=self._on_segmentation_error,
            )

    def _run_multi_segmentation_pipeline(self, groups: dict) -> None:
        from iris_viewer.engine.pipeline import execute_group_sync, SegmentationSequence
        from napari.qt.threading import thread_worker
        
        backends = self.pipeline_service.backends
        device = "cpu"
        try:
            device = self.pipeline_service._get_device()
        except Exception:
            pass
        scale_val = self.control_widget.spin_scale.value()
        
        @thread_worker
        def _process_all():
            results = {}
            for sample_name, image in self.state.multi_sample_images.items():
                results[sample_name] = {}
                for group_name, steps in groups.items():
                    seq = SegmentationSequence(steps=steps)
                    res = execute_group_sync(group_name, seq, image, backends, device=device)
                    results[sample_name][group_name] = res
            return results
            
        def _on_done(results):
            self.segmentation_panel.set_running_state(False)
            has_results = False
            for sample_name, group_res in results.items():
                sample_layers = self.state.sample_layers_map.get(sample_name, [])
                for group_name, res in group_res.items():
                    nuc = res.get('nuclear')
                    cyto = res.get('cytoplasm')
                    if nuc is not None or cyto is not None:
                        has_results = True
                    
                    bbox_lyr = next((l for l in self.viewer.layers if l.name == f"Bounds - {sample_name}"), None)
                    t = bbox_lyr.translate if bbox_lyr else (0, 0)
                    
                    if nuc is not None:
                        lname = f"{group_name} Nuclear Mask"
                        lyr = self.viewer.add_labels(
                            nuc, name=lname, translate=t, 
                            scale=(scale_val, scale_val), visible=True, opacity=0.3
                        )
                        sample_layers.append(lyr)
                        if lname not in self.added_legend_entries:
                            self.legend_widget.add_segmentation_row(lyr, lname, "#888888")
                            self.added_legend_entries.add(lname)
                    
                    if cyto is not None:
                        lname = f"{group_name} Cyto Mask"
                        lyr = self.viewer.add_labels(
                            cyto, name=lname, translate=t, 
                            scale=(scale_val, scale_val), visible=True, opacity=0.3
                        )
                        sample_layers.append(lyr)
                        if lname not in self.added_legend_entries:
                            self.legend_widget.add_segmentation_row(lyr, lname, "#888888")
                            self.added_legend_entries.add(lname)
                            
            if not has_results:
                QMessageBox.warning(self.control_widget, "No Results", "Multi-sample segmentation failed.")

        worker = _process_all()
        worker.returned.connect(_on_done)
        self._active_workers.append(worker)
        worker.start()

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
        if hasattr(self, "phenotyper_panel") and df is not None and not df.empty:
            self.phenotyper_panel.set_data(df=df, markers=self.state.markers)

    # ── Exports ─────────────────────────────────────────────────────────────

    def _export_mask(self, fmt: str = "") -> None:
        if self.state.consensus_mask is None:
            QMessageBox.warning(self.control_widget, "No Mask", "No consensus mask to export.")
            return
        sample = self.state.current_sample_name or "sample"
        ext = fmt if fmt.startswith(".") else f".{fmt}" if fmt else ".tiff"
        default_name = f"{sample}_mask{ext}"
        
        filters = "TIFF Image (*.tiff *.tif);;OME-TIFF Image (*.ome.tif *.ome.tiff);;PNG Image (*.png);;NumPy Array (*.npy);;NumPy Compressed (*.npz);;All Files (*)"
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self.control_widget, "Export Consensus Mask",
            default_name, filters
        )
        if filepath:
            cyto = next(
                (r['cytoplasm'] for r in self.state.segmentation_results.values()
                 if r.get('cytoplasm') is not None),
                None,
            )
            try:
                export_consensus_mask(Path(filepath), self.state.consensus_mask, cyto)
                QMessageBox.information(self.control_widget, "Export Complete", f"Consensus mask saved:\n{Path(filepath).name}")
            except Exception as e:
                QMessageBox.critical(self.control_widget, "Export Error", f"Failed to save mask: {e}")

    def _export_features(self, fmt: str = "") -> None:
        if self.state.features_df is None or self.state.features_df.empty:
            QMessageBox.warning(self.control_widget, "No Features", "Extract features first.")
            return
        sample = self.state.current_sample_name or "sample"
        ext = fmt if fmt.startswith(".") else f".{fmt}" if fmt else ".csv"
        default_name = f"{sample}_features{ext}"
        
        filters = "CSV (*.csv);;Excel Workbook (*.xlsx);;Apache Parquet (*.parquet);;Tab-Separated Values (*.tsv);;JSON (*.json);;All Files (*)"
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self.control_widget, "Export Features",
            default_name, filters
        )
        if filepath:
            try:
                export_features_csv(Path(filepath), self.state.features_df, sample)
                QMessageBox.information(self.control_widget, "Export Complete", f"Features saved:\n{Path(filepath).name}")
            except Exception as e:
                QMessageBox.critical(self.control_widget, "Export Error", f"Failed to save features: {e}")

    def _export_confidence(self, fmt: str = "") -> None:
        if self.state.confidence_map is None:
            QMessageBox.warning(self.control_widget, "No Confidence Map", "Run consensus first.")
            return
        sample = self.state.current_sample_name or "sample"
        ext = fmt if fmt.startswith(".") else f".{fmt}" if fmt else ".tiff"
        default_name = f"{sample}_confidence{ext}"
        
        filters = "TIFF Image (*.tiff *.tif);;PNG Heatmap (*.png);;NumPy Array (*.npy);;NumPy Compressed (*.npz);;All Files (*)"
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self.control_widget, "Export Confidence Map",
            default_name, filters
        )
        if filepath:
            try:
                export_confidence_map(Path(filepath), self.state.confidence_map)
                QMessageBox.information(self.control_widget, "Export Complete", f"Confidence map saved:\n{Path(filepath).name}")
            except Exception as e:
                QMessageBox.critical(self.control_widget, "Export Error", f"Failed to save confidence map: {e}")

    def _export_updated_phenotypes(self, fmt: str = "") -> None:
        if getattr(self.state, 'master_df', None) is None or self.state.master_df.empty:
            QMessageBox.warning(self.control_widget, "No Annotation Data", "Load annotation data first.")
            return
            
        ext = fmt if fmt.startswith(".") else f".{fmt}" if fmt else ".csv"
        default_name = f"updated_phenotypes{ext}"
        filters = "CSV (*.csv);;Excel Workbook (*.xlsx);;Apache Parquet (*.parquet);;Tab-Separated Values (*.tsv);;JSON (*.json);;All Files (*)"
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self.control_widget, "Export Updated Phenotypes",
            default_name, filters
        )
        if filepath:
            ref_col = self.control_widget.combo_ref.currentText()
            df_export = self.state.master_df.copy()
            updated_col_name = f"{ref_col}_updated"
            
            if 'Verified_CellType' in df_export.columns:
                df_export[updated_col_name] = df_export['Verified_CellType']
                df_export.drop(columns=['Verified_CellType'], inplace=True)
            else:
                df_export[updated_col_name] = df_export[ref_col]
                
            try:
                lower = filepath.lower()
                if lower.endswith('.xlsx'):
                    df_export.to_excel(filepath, index=False)
                elif lower.endswith('.parquet'):
                    df_export.to_parquet(filepath, index=False)
                elif lower.endswith('.tsv') or lower.endswith('.txt'):
                    df_export.to_csv(filepath, sep='\t', index=False)
                elif lower.endswith('.json'):
                    df_export.to_json(filepath, orient='records', indent=2)
                else:
                    df_export.to_csv(filepath, index=False)
                QMessageBox.information(self.control_widget, "Export Complete", f"Updated phenotypes saved:\n{Path(filepath).name}")
            except Exception as e:
                QMessageBox.critical(self.control_widget, "Export Error", f"Failed to save file: {e}")

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