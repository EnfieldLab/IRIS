"""
IRIS-Viewer Controller
Thin coordinator: owns the napari viewer, creates widgets and services,
wires Qt signals, and drives the segmentation pipeline, multi-model disagreement
analysis, ROI preview, feature extraction, export, and session save/restore.

Heavy per-domain logic lives in:
  app/services/image_service.py   - image loading & display
  app/services/batch_service.py   - multi-sample batch loop
  app/services/review_service.py  - disagreement overlay & interactive review
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import napari
import numpy as np
import pandas as pd
from napari.qt import thread_worker
from qtpy.QtCore import QSize, Qt, QTimer
from qtpy.QtGui import QMovie
from qtpy.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from iris_analyzer.widgets import IrisAnalyzerPanelWidget
from iris_common.logging import iris_log, log_error
from iris_common.widgets.sample_scope import SCOPE_ACTIVE, SCOPE_ALL, SCOPE_SELECTED
from iris_viewer.app.services import (
    AnnotationService,
    BatchService,
    BoundaryService,
    ColorService,
    CurationService,
    ImageService,
    MultiSampleService,
    NeighborhoodService,
    ReviewService,
)
from iris_viewer.core import (
    AppContext,
    ViewerAppState,
    get_layer_base_name,
    get_layer_sample_name,
    get_layer_type,
)
from iris_viewer.engine.disagreement import compute_model_disagreement
from iris_viewer.engine.feature_extraction import extract_cell_features
from iris_viewer.engine.pipeline import (
    PipelineService,
    SegmentationSequence,
    _clamp_roi_bbox,
    _count_labels,
    iter_execute_sequence,
)
from iris_viewer.io.exporters import (
    export_disagreement_map,
    export_features_csv,
    export_segmentation_mask,
)
from iris_viewer.io.loaders import load_existing_mask, load_multichannel_tiff
from iris_viewer.widgets import (
    ImageControlWidget,
    IrisCellularNeighborhoodsWidget,
    IrisMarkerExpressionWidget,
    PhenotyperPanelWidget,
    SegmentationPanelWidget,
    ViewerLegendWidget,
)


class ViewerController:
    """
    Thin MVC coordinator for IRIS-Viewer.

    Responsibilities:
    - Instantiate and dock all Qt widgets.
    - Assemble the shared context namespace (ctx) passed to services.
    - Bind Qt signals to service methods or local handler methods.
    - Drive the segmentation pipeline, ROI preview, model disagreement QC,
      feature extraction, export, and session persistence.
    """

    def __init__(self) -> None:
        self.state = ViewerAppState()
        self.pipeline_service = PipelineService()
        self.added_legend_entries: set = set()
        self._active_workers: list = []
        self._preview_roi_bound: set = set()
        self._seg_errors: list[str] = []
        self._seg_total_steps: int = 0
        self._seg_completed_steps: int = 0
        self._roi_sample_offset: Tuple[float, float] = (0.0, 0.0)

        # napari Viewer
        self.viewer = napari.Viewer(title="IRIS-Viewer")
        try:
            self.viewer.scale_bar.visible = False
            self.viewer.scale_bar.ticks = False
            self.viewer.scale_bar.box = False
            self.viewer.scale_bar.font_size = 26.0
            self.viewer.scale_bar.position = "bottom_left"
        except AttributeError:
            pass

        # Widgets
        self.control_widget = ImageControlWidget(viewer=self.viewer)
        self.segmentation_panel = SegmentationPanelWidget()
        self.disagreement_panel = self.segmentation_panel.disagreement_panel
        self.feature_panel = self.segmentation_panel.feature_panel
        self.phenotyper_panel = PhenotyperPanelWidget()
        self.legend_widget = ViewerLegendWidget(self.viewer)
        self.comparison_widget = self.control_widget.comparison_widget
        self.marker_widget = IrisMarkerExpressionWidget()
        self.analyzer_panel = IrisAnalyzerPanelWidget()
        self.neighborhood_widget = IrisCellularNeighborhoodsWidget()
        self.neighborhood_widget.legend_widget_ref = self.legend_widget

        docks = {}
        for widget, name in [
            (self.control_widget, "Control"),
            (self.comparison_widget, "TMA"),
            (self.segmentation_panel, "Segmentation"),
            (self.phenotyper_panel, "Phenotyper"),
            (self.marker_widget, "Marker"),
            (self.analyzer_panel, "Analyzer"),
        ]:
            docks[name] = self.viewer.window.add_dock_widget(widget, name=name, area="right")

        legend_dock = self.viewer.window.add_dock_widget(
            self.legend_widget, name="Legend", area="left"
        )

        self.comparison_widget.show()

        qt = self.viewer.window._qt_window

        def _get_dock_wrapper(d):
            if isinstance(d, QDockWidget):
                return d
            if hasattr(d, "parent") and isinstance(d.parent(), QDockWidget):
                return d.parent()
            return d

        control_dock = _get_dock_wrapper(docks["Control"])
        tma_dock = _get_dock_wrapper(docks["TMA"])
        seg_dock = _get_dock_wrapper(docks["Segmentation"])
        pheno_dock = _get_dock_wrapper(docks["Phenotyper"])
        marker_dock = _get_dock_wrapper(docks["Marker"])
        analyzer_dock = _get_dock_wrapper(docks["Analyzer"])

        qt.tabifyDockWidget(control_dock, tma_dock)
        qt.tabifyDockWidget(tma_dock, seg_dock)
        qt.tabifyDockWidget(seg_dock, pheno_dock)
        qt.tabifyDockWidget(pheno_dock, marker_dock)
        qt.tabifyDockWidget(marker_dock, analyzer_dock)
        control_dock.raise_()

        # Tabify Legend with napari's layer list
        leg_wrapper = _get_dock_wrapper(legend_dock)
        for d in qt.findChildren(QDockWidget):
            if d.windowTitle().lower() == "layer list":
                qt.tabifyDockWidget(d, leg_wrapper)
                d.raise_()
                break

        # Adjust Main Window Size
        def enforce_window_size():
            screen = qt.screen()
            if screen:
                geom = screen.availableGeometry()
                qt.resize(int(geom.width() * 0.85), int(geom.height() * 0.85))
                qt.move(geom.center() - qt.rect().center())
            qt.showMaximized()

        # Delay the resize slightly so it overrides napari's internal settings restoration
        QTimer.singleShot(200, enforce_window_size)

        # Shared Context namespace
        self._ctx = AppContext(
            viewer=self.viewer,
            state=self.state,
            pipeline_service=self.pipeline_service,
            control_widget=self.control_widget,
            segmentation_panel=self.segmentation_panel,
            disagreement_panel=self.disagreement_panel,
            feature_panel=self.feature_panel,
            phenotyper_panel=self.phenotyper_panel,
            analyzer_panel=self.analyzer_panel,
            legend_widget=self.legend_widget,
            comparison_widget=self.comparison_widget,
            neighborhood_widget=self.neighborhood_widget,
            marker_widget=self.marker_widget,
            expression_widget=self.marker_widget,
            added_legend_entries=self.added_legend_entries,
            active_workers=self._active_workers,
            flagged_cells=[],
            current_issue_idx=-1,
        )

        # Services
        self.color_svc = ColorService(self._ctx)
        self.annotation_svc = AnnotationService(self._ctx)
        self.image_svc = ImageService(self._ctx)
        self.batch_svc = BatchService(self._ctx)
        self.review_svc = ReviewService(self._ctx)
        self.multi_sample_svc = MultiSampleService(self._ctx)
        self.curation_svc = CurationService(self._ctx)
        self.boundary_svc = BoundaryService(self._ctx)
        self.neighborhood_svc = NeighborhoodService(self._ctx)
        self._ctx.color_svc = self.color_svc
        self._ctx.annotation_svc = self.annotation_svc
        self._ctx.review_svc = self.review_svc
        self._ctx.multi_sample_svc = self.multi_sample_svc
        self._ctx.image_svc = self.image_svc
        self._ctx.curation_svc = self.curation_svc
        self._ctx.boundary_svc = self.boundary_svc
        self._ctx.batch_svc = self.batch_svc
        self._ctx.neighborhood_svc = self.neighborhood_svc
        self._ctx.track_worker = self.track_worker
        self.legend_widget.ctx = self._ctx
        self.segmentation_panel.ctx = self._ctx
        self.phenotyper_panel.ctx = self._ctx
        self.analyzer_panel.ctx = self._ctx

        # Suppress default napari welcome widgets
        try:
            for top_widget in QApplication.instance().topLevelWidgets():
                for child in top_widget.findChildren(QWidget):
                    cls_name = type(child).__name__
                    if "Welcome" in cls_name or cls_name == "QtWelcomeWidget":
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
        self.iris_welcome.setStyleSheet("background-color: black;")
        self.iris_welcome.setAlignment(Qt.AlignCenter)
        self.iris_welcome.setAttribute(Qt.WA_TransparentForMouseEvents)

        gif_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../image/iris_blink.gif")
        )
        movie = QMovie(gif_path)

        movie.start()
        movie.jumpToFrame(0)
        orig_size = movie.currentImage().size()
        if not orig_size.isEmpty():
            scale_factor = 0.5
            movie.setScaledSize(
                QSize(
                    int(orig_size.width() * scale_factor),
                    int(orig_size.height() * scale_factor),
                )
            )

        self.iris_welcome.setMovie(movie)
        self.iris_welcome._movie = movie

        layout = canvas_native.layout()
        if not layout:
            layout = QVBoxLayout(canvas_native)
            layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.iris_welcome)

        @self.viewer.layers.events.inserted.connect
        def _hide_iris_text(*args):
            self.iris_welcome.hide()

        @self.viewer.layers.events.removed.connect
        def _show_iris_text(*args):
            if len(self.viewer.layers) == 0:
                self.iris_welcome.show()

        # Init
        self._configure_backend_availability()
        self._bind_control_signals()
        self._bind_segmentation_signals()
        self._bind_disagreement_signals()
        self._bind_feature_signals()
        self._bind_multi_sample_signals()
        self._bind_curation_signals()
        self._bind_neighborhood_signals()

    # Configuration

    def _configure_backend_availability(self) -> None:
        self.segmentation_panel.update_available_methods(
            self.pipeline_service.get_available_methods()
        )

    # Signal Binding

    def _bind_control_signals(self) -> None:
        cw = self.control_widget
        cw.loadDirectoryTriggered.connect(self.image_svc.on_load_directory)
        if hasattr(self, "_sync_sample_scope_bars"):
            cw.loadDirectoryTriggered.connect(lambda _d: self._sync_sample_scope_bars())
        cw.sampleSelected.connect(self.image_svc.on_sample_selected)
        cw.sampleSelected.connect(lambda _s: self._sync_segmentation_sources())
        if hasattr(self, "_sync_sample_scope_bars"):
            cw.sampleSelected.connect(lambda _s: self._sync_sample_scope_bars())
        cw.sampleSelected.connect(lambda _s: self._sync_downstream_panels())
        cw.loadExistingMaskTriggered.connect(self.image_svc.on_load_existing_mask)
        cw.loadExistingMaskTriggered.connect(lambda _p: self._sync_segmentation_sources())
        cw.loadChannelColorTriggered.connect(self._load_colors_smart)
        cw.loadCellTypeColorTriggered.connect(self.color_svc.load_celltype_colors)
        cw.scaleFactorChanged.connect(self.image_svc.update_scale)
        cw.scaleFactorChanged.connect(lambda _val: self._sync_downstream_panels())
        cw.viewLockToggled.connect(self._toggle_view_lock)
        cw.loadAnnotationDataTriggered.connect(self.annotation_svc.load_annotation_file)
        cw.loadAnnotationDataTriggered.connect(lambda _p: self._sync_downstream_panels())
        cw.autoSegmentTriggered.connect(self.boundary_svc.auto_segment_boundary)
        cw.saveBoundsTriggered.connect(self.boundary_svc.save_sample_boundaries)
        cw.exportMaskTriggered.connect(self._export_mask)
        cw.exportFeaturesTriggered.connect(self._export_features)
        cw.exportConfidenceTriggered.connect(self._export_confidence)
        cw.exportUpdatedPhenotypesTriggered.connect(self._export_updated_phenotypes)
        cw.batchProcessTriggered.connect(self.batch_svc.run_batch_processing)
        cw.colChanged.connect(self.multi_sample_svc.update_phenotypes_on_canvas)
        cw.colChanged.connect(self._on_phenotype_column_changed)

        if hasattr(cw, "saveSessionTriggered"):
            cw.saveSessionTriggered.connect(self._save_session)
        if hasattr(cw, "loadSessionTriggered"):
            cw.loadSessionTriggered.connect(self._load_session)

    def _resolve_active_phenotype_col(self, col: Optional[str] = None) -> str:
        if col and isinstance(col, str):
            return col
        resolved = ""
        if hasattr(self, "control_widget") and hasattr(self.control_widget, "combo_ref"):
            resolved = self.control_widget.combo_ref.currentText().strip()
        if (
            not resolved
            and hasattr(self, "comparison_widget")
            and hasattr(self.comparison_widget, "combo_ref")
        ):
            resolved = self.comparison_widget.combo_ref.currentText().strip()
        return resolved or "cell_type"

    def _sync_downstream_panels(
        self,
        df: Optional[Any] = None,
        col: Optional[str] = None,
    ) -> None:
        """Propagate active DataFrame, markers, phenotype col, and scale to downstream tabs."""
        resolved_col = self._resolve_active_phenotype_col(col)
        active_df = df
        if active_df is None and hasattr(self, "state"):
            for attr in ("master_df", "current_mapped_df", "features_df"):
                cand = getattr(self.state, attr, None)
                if cand is not None and not cand.empty:
                    active_df = cand
                    break

        markers = list(self.state.markers) if hasattr(self, "state") else []
        scale_val = (
            float(self.control_widget.spin_scale.value())
            if hasattr(self, "control_widget") and hasattr(self.control_widget, "spin_scale")
            else 1.0
        )

        if (
            hasattr(self, "phenotyper_panel")
            and self.phenotyper_panel is not None
            and active_df is not None
            and not active_df.empty
        ):
            self.phenotyper_panel.set_data(df=active_df, markers=markers)

        if hasattr(self, "marker_widget") and self.marker_widget is not None:
            selected_s = (
                list(self.state.sample_layers_map.keys())
                or (
                    [self.state.current_sample_name]
                    if getattr(self.state, "current_sample_name", None)
                    else []
                )
            )
            self.marker_widget.set_data(
                df=active_df,
                markers=markers,
                celltype_col=resolved_col,
                selected_samples=selected_s,
            )

        if hasattr(self, "analyzer_panel") and self.analyzer_panel is not None:
            self.analyzer_panel.set_data(
                df=active_df,
                markers=markers,
                celltype_col=resolved_col,
                scale_factor=scale_val,
                ctx=getattr(self, "_ctx", None),
            )

    def _on_phenotype_column_changed(self, col: Optional[str] = None) -> None:
        resolved_col = self._resolve_active_phenotype_col(col)
        active_df = None
        for attr in ("master_df", "current_mapped_df", "features_df"):
            cand = getattr(self.state, attr, None)
            if cand is not None and not cand.empty:
                active_df = cand
                break

        if (
            active_df is not None
            and not active_df.empty
            and resolved_col in active_df.columns
        ):
            unique_types = sorted(
                str(t) for t in active_df[resolved_col].dropna().unique() if str(t).strip()
            )
            for w in (
                getattr(self, "control_widget", None),
                getattr(self, "comparison_widget", None),
            ):
                if w and hasattr(w, "combo_quick_type"):
                    w.combo_quick_type.blockSignals(True)
                    w.combo_quick_type.clear()
                    w.combo_quick_type.addItems(unique_types)
                    w.combo_quick_type.blockSignals(False)

        self._sync_downstream_panels(df=active_df, col=resolved_col)

    def _bind_segmentation_signals(self) -> None:
        self.segmentation_panel.runSequence.connect(self._run_segmentation_pipeline)
        self.segmentation_panel.cancelSequence.connect(self._cancel_segmentation)
        self.segmentation_panel.drawPreviewRoiRequested.connect(self._create_or_focus_preview_roi)
        self.segmentation_panel.previewRoiRequested.connect(self._run_roi_preview)

    def _bind_disagreement_signals(self) -> None:
        dp = self.disagreement_panel
        dp.computeDisagreementRequested.connect(self._compute_disagreement)
        dp.thresholdChanged.connect(self.review_svc.update_disagreement_overlay)
        dp.activeMethodChanged.connect(self._on_active_segmentation_method_changed)
        dp.hotspotSelected.connect(self.review_svc.jump_to_issue)
        dp.flickerPairRequested.connect(self.review_svc.flicker_models)
        dp.categoryFilterChanged.connect(self.review_svc.set_category_filter)
        dp.overlayVisibilityChanged.connect(self.review_svc.toggle_overlay_visibility)
        dp.btn_next_issue.clicked.connect(self.review_svc.next_issue)
        dp.btn_prev_issue.clicked.connect(self.review_svc.prev_issue)
        dp.chk_enable_review.toggled.connect(self._toggle_interactive_review)
        if hasattr(dp, "overlayColorModeChanged") and hasattr(
            self, "_on_overlay_color_or_palette_changed"
        ):
            dp.overlayColorModeChanged.connect(self._on_overlay_color_or_palette_changed)
        if hasattr(dp, "methodColorsChanged") and hasattr(
            self, "_on_overlay_color_or_palette_changed"
        ):
            dp.methodColorsChanged.connect(self._on_overlay_color_or_palette_changed)
        if hasattr(dp, "selectedMethodsChanged") and hasattr(
            self, "_on_selected_qc_methods_changed"
        ):
            dp.selectedMethodsChanged.connect(self._on_selected_qc_methods_changed)
        self.legend_widget.layerSelected.connect(self._select_layer)

    def _bind_feature_signals(self) -> None:
        self.feature_panel.extractTriggered.connect(self._run_feature_extraction)
        self.feature_panel.exportTriggered.connect(self._export_features)
        self.feature_panel.maskSourceChanged.connect(self._on_active_segmentation_method_changed)

    def _bind_multi_sample_signals(self) -> None:
        """Wire the MultiSampleComparisonWidget signals to MultiSampleService and state."""
        cw = self.comparison_widget
        cw.addSamplesTriggered.connect(self.multi_sample_svc.add_samples_to_canvas)
        cw.clearCanvasTriggered.connect(self.multi_sample_svc.clear_canvas)
        cw.clearCanvasTriggered.connect(self._sync_segmentation_sources)
        if hasattr(self, "_sync_sample_scope_bars"):
            cw.clearCanvasTriggered.connect(lambda: self._sync_sample_scope_bars())

    def _bind_curation_signals(self) -> None:
        """Bind mouse drag callbacks for curation interactions."""
        self.viewer.mouse_drag_callbacks.append(self.curation_svc.on_right_click)
        self.viewer.mouse_drag_callbacks.append(self.curation_svc.on_quick_curate_click)

    def _bind_neighborhood_signals(self) -> None:
        """Wire the legacy IrisCellularNeighborhoodsWidget signals to NeighborhoodService."""
        nw = self.neighborhood_widget
        nw.runNicheAnalysis.connect(self.neighborhood_svc.execute_niche_analysis)
        nw.nicheMetadataChanged.connect(self.neighborhood_svc.update_niche_canvas_coloration)

    def _load_colors_smart(self, filepath: str) -> None:
        """Parse and apply color map definitions for markers or cell phenotypes."""
        import re

        import pandas as pd

        path_obj = Path(filepath)
        try:
            df = (
                pd.read_excel(filepath)
                if path_obj.suffix.lower() in (".xlsx", ".xls")
                else pd.read_csv(filepath)
            )
            cols = [re.sub(r"[^a-z0-9]", "", str(c).lower()) for c in df.columns]
        except Exception as e:
            QMessageBox.critical(self.control_widget, "Error", f"Failed to read file:\n{e}")
            return

        has_color = any(k in cols for k in ["colors", "color", "colours", "colour", "hex", "code"])
        has_markers = has_color and any(
            k in cols for k in ["markers", "marker", "channel", "channels", "target", "targets"]
        )
        has_celltypes = has_color and any(
            k in cols
            for k in [
                "celltypes",
                "celltype",
                "phenotype",
                "phenotypes",
                "label",
                "labels",
                "cluster",
                "clusters",
            ]
        )

        if not has_markers and not has_celltypes:
            QMessageBox.warning(
                self.control_widget,
                "Invalid Format",
                "File must contain a color column and either marker or cell type names.",
            )
            return

        if has_markers and has_celltypes:
            self.color_svc.load_channel_colors(filepath, show_dialog=False)
            self.color_svc.load_celltype_colors(filepath, show_dialog=False)
            QMessageBox.information(
                self.control_widget,
                "Success",
                "Updated channel marker and cell type colors from file.",
            )
        elif has_markers:
            self.color_svc.load_channel_colors(filepath, show_dialog=True)
        elif has_celltypes:
            self.color_svc.load_celltype_colors(filepath, show_dialog=True)

    # Interactive Review & Canvas Controls

    def _toggle_interactive_review(self, enabled: bool) -> None:
        """Toggle mouse click callbacks for interactive disagreement region inspection."""
        cb = self.review_svc.on_mouse_click
        if enabled and cb not in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.append(cb)
        elif not enabled and cb in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.remove(cb)

    def _select_layer(self, layer_name: str) -> None:
        """Set active layer selection in napari's layer list by name."""
        self.viewer.layers.selection.clear()
        for layer in self.viewer.layers:
            base = get_layer_base_name(layer)
            if base == layer_name:
                self.viewer.layers.selection.add(layer)
                break

    def _toggle_view_lock(self, is_locked: bool) -> None:
        """Lock or unlock canvas pan and zoom interactions."""
        self.viewer.camera.mouse_pan = not is_locked
        self.viewer.camera.mouse_zoom = not is_locked
        if is_locked:
            self._old_double_clicks = list(self.viewer.mouse_double_click_callbacks)
            self.viewer.mouse_double_click_callbacks.clear()
        elif hasattr(self, "_old_double_clicks"):
            self.viewer.mouse_double_click_callbacks.extend(self._old_double_clicks)
            del self._old_double_clicks

    # ROI Preview Controls

    def _get_scale_val(self) -> float:
        scale_val = (
            float(self.control_widget.spin_scale.value())
            if hasattr(self, "control_widget") and hasattr(self.control_widget, "spin_scale")
            else 1.0
        )
        return scale_val if scale_val > 0 else 1.0

    def _get_bounds_layer_for_sample(self, sample_name: str) -> Any:
        for lyr in self.viewer.layers:
            if (
                get_layer_type(lyr) == "bounds"
                and get_layer_sample_name(lyr) == sample_name
            ) or getattr(lyr, "name", "") == f"Bounds - {sample_name}":
                return lyr
        return None

    def _get_active_sample_and_offset_for_roi(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[str], Tuple[float, float]]:
        """Return ``(image, sample_name, (ty_px, tx_px))`` for the active ROI sample."""
        if self.state.raw_image is not None and not self.state.multi_sample_images:
            return self.state.raw_image, self.state.current_sample_name, (0.0, 0.0)

        if self.state.multi_sample_images:
            scale_val = ViewerController._get_scale_val(self)
            cam = getattr(self.viewer, "camera", None)
            center = getattr(cam, "center", None)
            cam_y = cam_x = None
            if center is not None and len(center) >= 2:
                cy, cx = float(center[-2]), float(center[-1])
                if np.isfinite(cy) and np.isfinite(cx):
                    cam_y, cam_x = cy, cx

            if cam_y is not None and cam_x is not None:
                for sn, img in self.state.multi_sample_images.items():
                    bbox_lyr = ViewerController._get_bounds_layer_for_sample(self, sn)
                    if bbox_lyr is None:
                        continue
                    t = tuple(getattr(bbox_lyr, "translate", (0.0, 0.0)))
                    ty_w = float(t[-2]) if len(t) >= 2 else 0.0
                    tx_w = float(t[-1]) if len(t) >= 2 else 0.0
                    data = getattr(bbox_lyr, "data", None)
                    if data:
                        pts = np.asarray(data[0], dtype=float)
                        if pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] >= 2:
                            y_min = float(np.min(pts[:, -2])) * scale_val + ty_w
                            y_max = float(np.max(pts[:, -2])) * scale_val + ty_w
                            x_min = float(np.min(pts[:, -1])) * scale_val + tx_w
                            x_max = float(np.max(pts[:, -1])) * scale_val + tx_w
                            if y_min <= cam_y <= y_max and x_min <= cam_x <= x_max:
                                return img, sn, (ty_w / scale_val, tx_w / scale_val)

            cur = self.state.current_sample_name
            if not cur or cur not in self.state.multi_sample_images:
                cur = next(iter(self.state.multi_sample_images))
            bbox_lyr = ViewerController._get_bounds_layer_for_sample(self, cur)
            if bbox_lyr is not None:
                t = tuple(getattr(bbox_lyr, "translate", (0.0, 0.0)))
                ty_w = float(t[-2]) if len(t) >= 2 else 0.0
                tx_w = float(t[-1]) if len(t) >= 2 else 0.0
                offset_px = (ty_w / scale_val, tx_w / scale_val)
            else:
                offset_px = (0.0, 0.0)
            return self.state.multi_sample_images[cur], cur, offset_px

        return None, None, (0.0, 0.0)

    def _get_active_image_for_roi(self) -> Optional[np.ndarray]:
        img, _sample_name, _offset = ViewerController._get_active_sample_and_offset_for_roi(
            self
        )
        return img

    def _compute_default_preview_bbox(
        self,
        image_hw: Optional[Tuple[int, int]],
        sample_offset_px: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[int, int, int, int]:
        scale_val = ViewerController._get_scale_val(self)

        h, w = image_hw if image_hw is not None else (1024, 1024)
        box_h = min(512, max(1, int(h)))
        box_w = min(512, max(1, int(w)))

        cy_px = float(h) / 2.0
        cx_px = float(w) / 2.0
        ty_px, tx_px = float(sample_offset_px[0]), float(sample_offset_px[1])
        cam = getattr(self.viewer, "camera", None)
        center = getattr(cam, "center", None)
        if center is not None and len(center) >= 2:
            cand_y = float(center[-2]) / scale_val - ty_px
            cand_x = float(center[-1]) / scale_val - tx_px
            if (
                np.isfinite(cand_y)
                and np.isfinite(cand_x)
                and (float(center[-2]) != 0.0 or float(center[-1]) != 0.0)
                and 0.0 <= cand_y < float(h)
                and 0.0 <= cand_x < float(w)
            ):
                cy_px, cx_px = cand_y, cand_x

        y0 = int(round(cy_px - box_h / 2.0))
        x0 = int(round(cx_px - box_w / 2.0))
        y0 = max(0, min(y0, max(0, int(h) - box_h)))
        x0 = max(0, min(x0, max(0, int(w) - box_w)))
        y1 = min(int(h), y0 + box_h)
        x1 = min(int(w), x0 + box_w)
        return (y0, x0, y1, x1)

    def _bbox_from_preview_layer(
        self, image_hw: Optional[Tuple[int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        roi_name = "Segmentation Preview ROI"
        if roi_name not in self.viewer.layers:
            return None
        lyr = self.viewer.layers[roi_name]
        data = getattr(lyr, "data", None)
        if not data:
            return None
        pts = np.asarray(data[-1], dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
            return None
        y0 = int(np.floor(np.min(pts[:, -2])))
        y1 = int(np.ceil(np.max(pts[:, -2])))
        x0 = int(np.floor(np.min(pts[:, -1])))
        x1 = int(np.ceil(np.max(pts[:, -1])))
        bbox = (y0, x0, y1, x1)
        if image_hw is not None:
            bbox = _clamp_roi_bbox(bbox, image_hw)
        return bbox

    def _compute_preview_roi_bbox(
        self,
        image_hw: Tuple[int, int],
        sample_offset_px: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[int, int, int, int]:
        from_layer = self._bbox_from_preview_layer(image_hw)
        if (
            from_layer is not None
            and from_layer[2] > from_layer[0]
            and from_layer[3] > from_layer[1]
        ):
            return from_layer
        if self.state.preview_roi_bbox is not None:
            clamped = _clamp_roi_bbox(self.state.preview_roi_bbox, image_hw)
            if clamped[2] > clamped[0] and clamped[3] > clamped[1]:
                return clamped
        return self._compute_default_preview_bbox(image_hw, sample_offset_px)

    def _create_or_focus_preview_roi(self) -> Tuple[int, int, int, int]:
        """Create or focus editable 512x512 'Segmentation Preview ROI' on Napari Shapes."""
        if hasattr(self, "_get_active_sample_and_offset_for_roi"):
            img, _sample_name, (ty_px, tx_px) = self._get_active_sample_and_offset_for_roi()
        else:
            img = self._get_active_image_for_roi()
            ty_px, tx_px = 0.0, 0.0
        image_hw = (int(img.shape[-2]), int(img.shape[-1])) if img is not None else None
        scale_val = ViewerController._get_scale_val(self)
        translate = (float(ty_px) * scale_val, float(tx_px) * scale_val)

        roi_name = "Segmentation Preview ROI"
        layer_moved_sample = False
        if roi_name in self.viewer.layers:
            old_t = tuple(getattr(self.viewer.layers[roi_name], "translate", (0.0, 0.0)))
            if len(old_t) >= 2 and (
                abs(float(old_t[-2]) - translate[0]) > 1e-3
                or abs(float(old_t[-1]) - translate[1]) > 1e-3
            ):
                layer_moved_sample = True

        existing = None if layer_moved_sample else self._bbox_from_preview_layer(image_hw)
        if existing is not None and existing[2] > existing[0] and existing[3] > existing[1]:
            bbox = existing
        else:
            try:
                bbox = self._compute_default_preview_bbox(image_hw, (ty_px, tx_px))
            except TypeError:
                bbox = self._compute_default_preview_bbox(image_hw)

        y0, x0, y1, x1 = bbox
        rect = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)

        if roi_name in self.viewer.layers:
            lyr = self.viewer.layers[roi_name]
            if layer_moved_sample or not getattr(lyr, "data", None):
                lyr.data = [rect]
            lyr.visible = True
            lyr.scale = (scale_val, scale_val)
            lyr.translate = translate
        else:
            lyr = self.viewer.add_shapes(
                [rect],
                shape_type="rectangle",
                name=roi_name,
                edge_color="#00E5FF",
                face_color=[0.0, 0.9, 1.0, 0.08],
                edge_width=2,
                scale=(scale_val, scale_val),
                translate=translate,
            )

        try:
            lyr.mode = "select"
        except Exception:
            pass

        if id(lyr) not in self._preview_roi_bound:
            events = getattr(lyr, "events", None)
            data_ev = getattr(events, "data", None)
            if data_ev is not None and hasattr(data_ev, "connect"):

                def _on_roi_changed(_event=None):
                    cur_img = self._get_active_image_for_roi()
                    hw = (
                        (int(cur_img.shape[-2]), int(cur_img.shape[-1]))
                        if cur_img is not None
                        else None
                    )
                    updated = self._bbox_from_preview_layer(hw)
                    if updated is not None:
                        self.state.preview_roi_bbox = updated

                data_ev.connect(_on_roi_changed)
                self._preview_roi_bound.add(id(lyr))

        try:
            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.add(lyr)
        except Exception:
            pass

        self.state.preview_roi_bbox = bbox
        return bbox

    def _bring_layer_to_top(self, layer: Any) -> None:
        """Move *layer* to the top of napari's layer stack when supported."""
        if not hasattr(self.viewer.layers, "move"):
            return
        try:
            idx = list(self.viewer.layers).index(layer)
            top = len(self.viewer.layers) - 1
            if 0 <= idx < top:
                self.viewer.layers.move(idx, top)
        except Exception:
            pass

    def _upsert_labels_layer(
        self,
        data: np.ndarray,
        layer_name: str,
        scale_val: float,
        translate: Tuple[float, float] = (0.0, 0.0),
        visible: bool = True,
        metadata: Optional[dict] = None,
        opacity: Optional[float] = None,
    ) -> Any:
        """Add or update a filled Labels segmentation mask layer."""
        resolved_opacity = (
            float(opacity)
            if opacity is not None
            else (0.5 if "cyto" in layer_name.lower() else 0.8)
        )
        if layer_name in self.viewer.layers and metadata is None:
            lyr = self.viewer.layers[layer_name]
            lyr.data = data
            lyr.scale = (scale_val, scale_val)
            lyr.translate = translate
            lyr.opacity = resolved_opacity
            lyr.visible = visible
            try:
                lyr.contour = 0
            except Exception:
                pass
            ViewerController._bring_layer_to_top(self, lyr)
            return lyr

        kwargs = {
            "name": layer_name,
            "opacity": resolved_opacity,
            "visible": visible,
            "scale": (scale_val, scale_val),
            "translate": translate,
        }
        if metadata is not None:
            kwargs["metadata"] = metadata

        lyr = self.viewer.add_labels(data, **kwargs)
        try:
            lyr.contour = 0
        except Exception:
            pass
        ViewerController._bring_layer_to_top(self, lyr)
        return lyr

    def _run_roi_preview(self, group_name: str, steps: list) -> None:
        img, sample_name, (ty_px, tx_px) = self._get_active_sample_and_offset_for_roi()
        if img is None:
            QMessageBox.warning(
                self.control_widget,
                "No Image",
                "Load an image before running ROI segmentation preview.",
            )
            return

        scale_val = self._get_scale_val()
        self._roi_sample_offset = (float(ty_px), float(tx_px))
        image_hw = (int(img.shape[-2]), int(img.shape[-1]))

        roi_name = "Segmentation Preview ROI"
        translate = (float(ty_px) * scale_val, float(tx_px) * scale_val)
        if roi_name in self.viewer.layers:
            lyr = self.viewer.layers[roi_name]
            old_t = tuple(getattr(lyr, "translate", (0.0, 0.0)))
            if len(old_t) >= 2 and (
                abs(float(old_t[-2]) - translate[0]) > 1e-3
                or abs(float(old_t[-1]) - translate[1]) > 1e-3
            ):
                bbox = self._create_or_focus_preview_roi()
            else:
                lyr.scale = (scale_val, scale_val)
                lyr.translate = translate
                bbox = self._compute_preview_roi_bbox(image_hw, (ty_px, tx_px))
        else:
            bbox = self._compute_preview_roi_bbox(image_hw, (ty_px, tx_px))

        self.state.preview_roi_bbox = bbox

        pb = getattr(self.segmentation_panel, "pipeline_builder", None)
        if pb is not None and hasattr(pb, "btn_preview_roi"):
            pb.btn_preview_roi.setEnabled(False)

        sample_tag = f" [{sample_name}]" if sample_name else ""
        status_msg = (
            f"Previewing ROI{sample_tag} ({group_name}): "
            f"y={bbox[0]}:{bbox[2]}, x={bbox[1]}:{bbox[3]}..."
        )
        self.segmentation_panel.set_progress(0, 0, status_msg)
        iris_log(
            "IRIS-Viewer",
            "ROI Preview",
            f"Executing '{group_name}'{sample_tag} on ROI y={bbox[0]}:{bbox[2]}, x={bbox[1]}:{bbox[3]}...",
        )
        self.pipeline_service.execute_roi_preview(
            group_name=group_name,
            sequence=SegmentationSequence(steps=steps),
            image=img,
            roi_bbox=bbox,
            on_complete=self._on_roi_preview_complete,
            on_error=self._on_roi_preview_error,
        )

    def _on_roi_preview_complete(
        self,
        group_name: str,
        clamped_bbox: Tuple[int, int, int, int],
        res: dict,
    ) -> None:
        self.state.preview_roi_bbox = clamped_bbox
        y0, x0, y1, x1 = clamped_bbox
        scale_val = self._get_scale_val()
        ty, tx = getattr(self, "_roi_sample_offset", (0.0, 0.0))
        translate = (ty * scale_val + y0 * scale_val, tx * scale_val + x0 * scale_val)

        pb = getattr(self.segmentation_panel, "pipeline_builder", None)
        if pb is not None and hasattr(pb, "btn_preview_roi"):
            pb.btn_preview_roi.setEnabled(True)

        for key, layer_name, opac in (
            ("nuclear", "ROI Preview - Nuclear", 0.8),
            ("cytoplasm", "ROI Preview - Cytoplasm", 0.5),
        ):
            mask_arr = res.get(key)
            if mask_arr is None:
                if layer_name in self.viewer.layers:
                    self.viewer.layers[layer_name].visible = False
                continue
            self._upsert_labels_layer(
                mask_arr,
                layer_name=layer_name,
                scale_val=scale_val,
                translate=translate,
                visible=True,
                opacity=opac,
            )

        n_nuc = _count_labels(res.get("nuclear"))
        n_cyto = _count_labels(res.get("cytoplasm"))
        summary = (
            f"ROI Preview ({group_name}): {n_nuc} nuclei, {n_cyto} cytoplasm | "
            f"y={y0}:{y1}, x={x0}:{x1}"
        )
        self.segmentation_panel.stop_progress(summary)

    def _on_roi_preview_error(self, group_name: str, error_msg: str) -> None:
        pb = getattr(self.segmentation_panel, "pipeline_builder", None)
        if pb is not None and hasattr(pb, "btn_preview_roi"):
            pb.btn_preview_roi.setEnabled(True)
        log_error("IRIS-Viewer", "ROI Preview", f"{group_name}: {error_msg}")
        self.segmentation_panel.stop_progress(
            f"ROI Preview Error ({group_name}): {error_msg}"
        )

    # Segmentation Pipeline

    def _sync_masks_map_for_method(
        self, method: Optional[str], nuc: Optional[np.ndarray]
    ) -> None:
        """Keep ``state.masks_map`` aligned for single-sample or multi-sample TMA."""
        if not method:
            return
        if getattr(self.state, "multi_sample_images", None):
            for sn in self.state.multi_sample_images:
                entry = self.state.batch_results.get(sn)
                if isinstance(entry, dict) and isinstance(entry.get(method), dict):
                    sn_nuc = entry[method].get("nuclear")
                    if sn_nuc is not None:
                        self.state.masks_map[sn] = sn_nuc
        elif self.state.current_sample_name and nuc is not None:
            self.state.masks_map[self.state.current_sample_name] = nuc

    def _resolve_sample_scope_lists(self) -> Tuple[List[str], List[str]]:
        """Return ``(all_samples, active_samples)`` from ViewerAppState."""
        all_samples: List[str] = []
        seen: set[str] = set()
        candidates = (
            sorted(getattr(self.state, "file_map", {}).keys()),
            list(getattr(self.state, "multi_sample_images", {}).keys()),
            list(getattr(self.state, "sample_layers_map", {}).keys()),
            list(getattr(self.state, "masks_map", {}).keys()),
            (
                [self.state.current_sample_name]
                if getattr(self.state, "current_sample_name", None)
                else []
            ),
        )
        for group in candidates:
            for sn in group:
                if sn and sn not in seen:
                    seen.add(sn)
                    all_samples.append(sn)

        if getattr(self.state, "multi_sample_images", None):
            active_samples = list(self.state.multi_sample_images.keys())
        elif getattr(self.state, "sample_layers_map", None):
            active_samples = list(self.state.sample_layers_map.keys())
        elif getattr(self.state, "current_sample_name", None):
            active_samples = [self.state.current_sample_name]
        else:
            active_samples = []

        return all_samples, active_samples

    def _sync_sample_scope_bars(self) -> Tuple[List[str], List[str]]:
        """Sync SampleScopeBar widgets in PipelineBuilder and FeaturePanel."""
        all_samples, active_samples = ViewerController._resolve_sample_scope_lists(self)
        sp = getattr(self, "segmentation_panel", None)
        if sp is not None and hasattr(sp, "sync_sample_scope"):
            sp.sync_sample_scope(all_samples, active_samples)
        return all_samples, active_samples

    def _sync_segmentation_sources(self) -> None:
        """Auto-select active method and sync Disagreement & Feature panels."""
        methods = [
            m
            for m, r in self.state.segmentation_results.items()
            if r.get("nuclear") is not None or r.get("cytoplasm") is not None
        ]
        if not methods:
            if hasattr(self, "review_svc") and hasattr(self.review_svc, "set_active_mask_layers"):
                self.review_svc.set_active_mask_layers("")
            else:
                self.state.set_active_segmentation_method(None)
            self.segmentation_panel.sync_available_results([], active_method=None)
            return

        cell_counts = {
            m: _count_labels(
                self.state.segmentation_results[m].get("nuclear")
                if self.state.segmentation_results[m].get("nuclear") is not None
                else self.state.segmentation_results[m].get("cytoplasm")
            )
            for m in methods
        }

        active = self.state.active_segmentation_method
        if not active or active not in methods:
            active = methods[0]
        if hasattr(self, "review_svc") and hasattr(self.review_svc, "set_active_mask_layers"):
            nuc, _cyto = self.review_svc.set_active_mask_layers(active)
        else:
            nuc, _cyto = self.state.set_active_segmentation_method(active)
        ViewerController._sync_masks_map_for_method(self, active, nuc)
        try:
            self.segmentation_panel.sync_available_results(
                methods, active_method=active, cell_counts=cell_counts
            )
        except TypeError:
            self.segmentation_panel.sync_available_results(methods, active_method=active)

    def _run_segmentation_pipeline(self, groups: dict) -> None:
        pb = getattr(self.segmentation_panel, "pipeline_builder", None)
        scope_mode = SCOPE_ACTIVE
        target_samples: List[str] = []
        if pb is not None and hasattr(pb, "sample_scope_bar"):
            all_samples, active_samples = ViewerController._sync_sample_scope_bars(self)
            scope_mode = pb.sample_scope_bar.get_scope_mode()
            target_samples = pb.get_target_samples(all_samples, active_samples)

        is_multi_canvas = len(self.state.multi_sample_images) > 0
        use_scoped_batch = False

        if scope_mode in (SCOPE_SELECTED, SCOPE_ALL):
            if not target_samples:
                QMessageBox.warning(
                    self.control_widget,
                    "No Samples Selected",
                    "Select at least one sample in Target Scope before running segmentation.",
                )
                return
            is_single_active_only = (
                len(target_samples) == 1
                and target_samples[0] == self.state.current_sample_name
                and self.state.raw_image is not None
                and not is_multi_canvas
            )
            if not is_single_active_only:
                valid_targets = [
                    sn
                    for sn in target_samples
                    if (
                        sn in self.state.multi_sample_images
                        or (sn == self.state.current_sample_name and self.state.raw_image is not None)
                        or sn in self.state.file_map
                    )
                ]
                if not valid_targets:
                    QMessageBox.warning(
                        self.control_widget,
                        "No Image",
                        "No image files found for the selected target samples.",
                    )
                    return
                target_samples = valid_targets
                use_scoped_batch = True
        else:
            if self.state.raw_image is None and not is_multi_canvas:
                QMessageBox.warning(
                    self.control_widget,
                    "No Image",
                    "Load an image or multiple samples before running segmentation.",
                )
                return
            if is_multi_canvas:
                target_samples = list(self.state.multi_sample_images.keys())
                use_scoped_batch = True

        for group_name, steps in groups.items():
            self.state.segmentation_results.pop(group_name, None)
            prefix = f"{group_name}_"
            for old_key in [k for k in self.state.segmentation_status if k.startswith(prefix)]:
                self.state.segmentation_status.pop(old_key, None)
            for idx in range(len(steps)):
                step_key = f"{group_name}_{idx}"
                self.state.segmentation_status[step_key] = "Queued"
                self.segmentation_panel.update_step_status(group_name, idx, "Queued")

        if len(groups) == 1:
            self.state.active_segmentation_method = next(iter(groups))

        self.state.disagreement_report = None
        self.state.disagreement_heatmap = None
        self._seg_errors = []
        steps_per_sample = max(1, sum(len(s) for s in groups.values()))
        n_target = len(target_samples) if use_scoped_batch else 1
        self._seg_total_steps = n_target * steps_per_sample
        self._seg_completed_steps = 0

        iris_log(
            "IRIS-Viewer",
            "Segmentation",
            f"Launching {len(groups)} pipeline group(s) across {n_target} sample(s) "
            f"(scope='{scope_mode}')...",
        )

        self.segmentation_panel.set_running_state(True)
        self.segmentation_panel.set_progress(
            0,
            self._seg_total_steps,
            f"Starting segmentation (0/{self._seg_total_steps})...",
        )

        if use_scoped_batch:
            self._run_multi_segmentation_pipeline(groups, target_samples=target_samples)
        else:
            self.pipeline_service.execute_pipeline_groups(
                groups={g: SegmentationSequence(steps=s) for g, s in groups.items()},
                image=self.state.raw_image,
                on_step_start=self._on_segmentation_step_start,
                on_step_complete=self._on_single_segmentation_done,
                on_group_complete=self._on_group_complete,
                on_all_complete=self._on_all_segmentations_complete,
                on_error=self._on_segmentation_error,
            )

    def track_worker(
        self,
        worker: Any,
        on_error: Optional[Callable[[Exception], None]] = None,
        error_title: str = "Worker Error",
        error_prefix: str = "An error occurred during background processing",
    ) -> Any:
        """Register a worker in active_workers, connect automatic cleanup on finished,
        and connect an errored handler to notify the user and reset UI states.
        """
        def _cleanup():
            if worker in self._active_workers:
                self._active_workers.remove(worker)

        worker.finished.connect(_cleanup)

        def _handle_error(exc: Exception):
            _cleanup()
            log_error("IRIS-Viewer", "Segmentation", f"{error_title}: {exc}")
            if on_error:
                try:
                    on_error(exc)
                except Exception as inner_e:
                    log_error(
                        "IRIS-Viewer",
                        "Segmentation",
                        f"{error_title} callback raised: {inner_e}",
                    )
            else:
                QMessageBox.critical(
                    self.control_widget, error_title, f"{error_prefix}:\n{exc}"
                )

        worker.errored.connect(_handle_error)
        self._active_workers.append(worker)
        return worker

    def _run_multi_segmentation_pipeline(
        self,
        groups: dict,
        target_samples: Optional[List[str]] = None,
    ) -> None:
        backends = self.pipeline_service.backends
        device = "cpu"
        try:
            device = self.pipeline_service._get_device()
        except Exception:
            pass
        gpu_lock = self.pipeline_service._gpu_lock
        scale_val = self._get_scale_val()

        target_list = (
            list(target_samples)
            if target_samples is not None
            else list(self.state.multi_sample_images.keys())
        )
        sample_sources: List[Tuple[str, Optional[np.ndarray], Optional[Path]]] = []
        for sn in target_list:
            img = self.state.multi_sample_images.get(sn)
            if img is None and sn == self.state.current_sample_name:
                img = self.state.raw_image
            img_path = self.state.file_map.get(sn)
            if img is not None or img_path is not None:
                sample_sources.append((sn, img, img_path))

        displayed_samples = set(self.state.multi_sample_images.keys()) | set(
            self.state.sample_layers_map.keys()
        )
        if self.state.raw_image is not None and self.state.current_sample_name:
            displayed_samples.add(self.state.current_sample_name)

        groups_snapshot = {
            g_name: SegmentationSequence(steps=list(steps))
            for g_name, steps in groups.items()
        }
        n_samples = len(sample_sources)
        n_groups = len(groups_snapshot)
        total_steps = max(1, n_samples * sum(len(seq.steps) for seq in groups_snapshot.values()))
        active_group = (
            self.state.active_segmentation_method
            if self.state.active_segmentation_method in groups_snapshot
            else next(iter(groups_snapshot))
        )
        rep_sample = (
            self.state.current_sample_name
            if self.state.current_sample_name in {s[0] for s in sample_sources}
            else (sample_sources[0][0] if sample_sources else None)
        )

        self._multi_done_steps = 0
        self._multi_any_masks = False
        fatal_error = [False]

        @thread_worker
        def _process_all():
            for s_idx, (sample_name, image, img_path) in enumerate(sample_sources, start=1):
                if image is None and img_path is not None:
                    image, _ = load_multichannel_tiff(img_path)
                if image is None:
                    continue

                for group_name, seq in groups_snapshot.items():
                    group_res = {"nuclear": None, "cytoplasm": None}
                    for ev in iter_execute_sequence(
                        group_name=group_name,
                        sequence=seq,
                        image=image,
                        backends=backends,
                        device=device,
                        gpu_lock=gpu_lock,
                        log_prefix=f"[{s_idx}/{n_samples}] {sample_name}",
                    ):
                        etype = ev[0]
                        if etype == "step_start":
                            _, gname, step_idx, label = ev
                            yield (
                                "step_start",
                                s_idx,
                                n_samples,
                                sample_name,
                                gname,
                                step_idx,
                                label,
                            )
                        elif etype == "step":
                            _, gname, step_idx, label, _res = ev
                            yield (
                                "step_done",
                                s_idx,
                                n_samples,
                                sample_name,
                                gname,
                                step_idx,
                                label,
                            )
                        elif etype == "error":
                            _, gname, step_idx, label, err_msg = ev
                            yield (
                                "step_error",
                                s_idx,
                                n_samples,
                                sample_name,
                                gname,
                                step_idx,
                                label,
                                err_msg,
                            )
                        elif etype == "group_done":
                            _, gname, group_res = ev

                    n_nuc = _count_labels(group_res.get("nuclear"))
                    n_cyto = _count_labels(group_res.get("cytoplasm"))
                    iris_log(
                        "IRIS-Viewer",
                        "Segmentation",
                        f"[{s_idx}/{n_samples}] {sample_name} | "
                        f"{group_name} -> {n_nuc} nuclei, {n_cyto} cytoplasm",
                    )
                    yield ("group_done", s_idx, n_samples, sample_name, group_name, group_res)

        def _upsert_sample_mask_layer(
            sample_name: str,
            group_name: str,
            mask_arr: np.ndarray,
            lname: str,
            translate: Tuple[float, float],
            is_vis: bool,
            layer_opacity: float,
        ) -> None:
            sample_layers = self.state.sample_layers_map.setdefault(sample_name, [])
            existing_lyr = None
            for lyr in list(sample_layers):
                if lyr not in self.viewer.layers:
                    sample_layers.remove(lyr)
                    continue
                if get_layer_base_name(lyr) == lname and (
                    get_layer_sample_name(lyr) in (sample_name, None)
                ):
                    existing_lyr = lyr
                    break

            if existing_lyr is not None:
                existing_lyr.data = mask_arr
                existing_lyr.translate = translate
                existing_lyr.scale = (scale_val, scale_val)
                existing_lyr.opacity = layer_opacity
                existing_lyr.visible = is_vis
                try:
                    existing_lyr.contour = 0
                except Exception:
                    pass
                self._bring_layer_to_top(existing_lyr)
                lyr = existing_lyr
            else:
                meta = {
                    "sample_name": sample_name,
                    "layer_type": "segmentation",
                    "base_name": lname,
                    "segmentation_name": lname,
                }
                lyr = self._upsert_labels_layer(
                    mask_arr,
                    layer_name=lname,
                    scale_val=scale_val,
                    translate=translate,
                    visible=is_vis,
                    metadata=meta,
                    opacity=layer_opacity,
                )
                sample_layers.append(lyr)
                self.multi_sample_svc._register_layer(lyr, lname)
                lyr.opacity = layer_opacity
                lyr.visible = is_vis
                try:
                    lyr.contour = 0
                except Exception:
                    pass
                self.state.persistent_settings.setdefault(lname, {})["visible"] = is_vis
                self.state.persistent_settings[lname]["opacity"] = layer_opacity
                if lname not in self.added_legend_entries:
                    self.legend_widget.add_segmentation_row(lyr, lname, "#888888")
                    self.added_legend_entries.add(lname)

        def _on_yield(val: tuple) -> None:
            etype = val[0]
            if etype == "step_start":
                _, s_idx, n_s, sample_name, gname, step_idx, label = val
                self.state.segmentation_status[f"{gname}_{step_idx}"] = "Running"
                self.segmentation_panel.update_step_status(gname, step_idx, "Running")
                self.segmentation_panel.set_progress(
                    self._multi_done_steps,
                    total_steps,
                    f"[{s_idx}/{n_s}] {sample_name} | {gname}: {label}...",
                )
            elif etype == "step_done":
                _, s_idx, n_s, sample_name, gname, step_idx, label = val
                self._multi_done_steps += 1
                self.state.segmentation_status[f"{gname}_{step_idx}"] = "Done"
                self.segmentation_panel.update_step_status(gname, step_idx, "Done")
                self.segmentation_panel.set_progress(
                    self._multi_done_steps,
                    total_steps,
                    f"[{s_idx}/{n_s}] {sample_name} | {gname}: {label} done",
                )
            elif etype == "step_error":
                _, s_idx, n_s, sample_name, gname, step_idx, label, err_msg = val
                self._multi_done_steps += 1
                self.state.segmentation_status[f"{gname}_{step_idx}"] = "Failed"
                self.segmentation_panel.update_step_status(gname, step_idx, "Failed")
                self._seg_errors.append(
                    f"[{sample_name}] {gname} / {label} (step {step_idx + 1}): {err_msg}"
                )
                self.segmentation_panel.set_progress(
                    self._multi_done_steps,
                    total_steps,
                    f"[{s_idx}/{n_s}] {sample_name} | {gname}: {label} failed",
                )
            elif etype == "group_done":
                _, _s_idx, _n_s, sample_name, group_name, res = val
                nuc = res.get("nuclear")
                cyto = res.get("cytoplasm")
                if nuc is not None or cyto is not None:
                    self._multi_any_masks = True

                entry = self.state.batch_results.get(sample_name)
                if not isinstance(entry, dict):
                    entry = {}
                    self.state.batch_results[sample_name] = entry
                entry[group_name] = res

                if group_name == active_group and nuc is not None:
                    self.state.masks_map[sample_name] = nuc
                if sample_name == rep_sample or group_name not in self.state.segmentation_results:
                    self.state.segmentation_results[group_name] = res

                if sample_name in displayed_samples:
                    bbox_lyr = self._get_bounds_layer_for_sample(sample_name)
                    t = tuple(bbox_lyr.translate) if bbox_lyr is not None else (0.0, 0.0)
                    is_vis = group_name == active_group
                    if nuc is not None:
                        _upsert_sample_mask_layer(
                            sample_name,
                            group_name,
                            nuc,
                            f"{group_name} Nuclear Mask",
                            t,
                            is_vis,
                            0.8,
                        )
                    if cyto is not None:
                        _upsert_sample_mask_layer(
                            sample_name,
                            group_name,
                            cyto,
                            f"{group_name} Cyto Mask",
                            t,
                            is_vis,
                            0.5,
                        )

        def _on_multi_all_done() -> None:
            if worker in self.pipeline_service._active_workers:
                self.pipeline_service._active_workers.remove(worker)
            self.segmentation_panel.set_running_state(False)
            if fatal_error[0]:
                return

            self._sync_segmentation_sources()

            if not self._multi_any_masks:
                detail = (
                    "\n".join(self._seg_errors)
                    if self._seg_errors
                    else "Multi-sample segmentation failed. Check dependencies and parameters."
                )
                QMessageBox.warning(self.control_widget, "No Results", detail)
                return

            total_cells = sum(
                _count_labels(self.state.masks_map.get(sn))
                for sn, _img, _p in sample_sources
                if self.state.masks_map.get(sn) is not None
            )
            self.control_widget.update_dashboard(
                seg_status=f"{n_groups} pipeline(s) x {n_samples} samples",
                total_cells=f"{total_cells:,}",
            )
            if n_groups >= 2:
                self.segmentation_panel.lbl_disagreement_status.setText(
                    f"Ready for disagreement check: {n_groups} pipeline(s) ({n_samples} samples)"
                )
            else:
                self.segmentation_panel.lbl_disagreement_status.setText(
                    f"Completed {active_group} across {n_samples} sample(s) ({total_cells:,} cells)"
                )
            self.segmentation_panel.lbl_studio_summary.setText(
                f"Pipelines: {n_groups} x {n_samples} samples ({total_cells:,} cells) | "
                "Disagreements: Ready | Features: Idle"
            )
            if self._seg_errors:
                QMessageBox.warning(
                    self.control_widget,
                    "Partial Segmentation Errors",
                    "Some multi-sample steps reported errors:\n\n"
                    + "\n".join(self._seg_errors[:10]),
                )

        def _on_multi_seg_error(exc: Exception) -> None:
            fatal_error[0] = True
            if worker in self.pipeline_service._active_workers:
                self.pipeline_service._active_workers.remove(worker)
            self.segmentation_panel.set_running_state(False)
            QMessageBox.critical(
                self.control_widget,
                "Segmentation Error",
                f"Multi-sample segmentation pipeline failed:\n{exc}",
            )

        worker = _process_all()
        worker.yielded.connect(_on_yield)
        worker.finished.connect(_on_multi_all_done)
        self.pipeline_service._active_workers.append(worker)
        self.track_worker(
            worker,
            on_error=_on_multi_seg_error,
            error_title="Segmentation Error",
            error_prefix="Multi-sample segmentation failed",
        )
        worker.start()

    def _on_segmentation_step_start(
        self, group_name: str, step_idx: int, method_name: str
    ) -> None:
        step_key = f"{group_name}_{step_idx}"
        self.state.segmentation_status[step_key] = "Running"
        self.segmentation_panel.update_step_status(group_name, step_idx, "Running")
        done = getattr(self, "_seg_completed_steps", 0)
        total = getattr(self, "_seg_total_steps", 0)
        msg = f"Running {group_name} step {step_idx + 1} ({method_name})..."
        self.segmentation_panel.set_progress(done, total, msg)
        iris_log("IRIS-Viewer", "Segmentation", msg)

    def _on_segmentation_error(
        self, group_name: str, step_idx: int, method_name: str, error_msg: str
    ) -> None:
        step_key = f"{group_name}_{step_idx}"
        self.state.segmentation_status[step_key] = "Failed"
        self.segmentation_panel.update_step_status(group_name, step_idx, "Failed")
        self._seg_completed_steps = getattr(self, "_seg_completed_steps", 0) + 1
        if not hasattr(self, "_seg_errors") or self._seg_errors is None:
            self._seg_errors = []
        self._seg_errors.append(
            f"{group_name} / {method_name} (step {step_idx + 1}): {error_msg}"
        )
        total = getattr(self, "_seg_total_steps", 0)
        self.segmentation_panel.set_progress(
            self._seg_completed_steps,
            total,
            f"Failed {group_name}: {method_name} ({self._seg_completed_steps}/{total})",
        )
        log_error(
            "IRIS-Viewer",
            "Segmentation",
            f"{group_name}/{method_name} (step {step_idx}): {error_msg}",
        )

    def _on_single_segmentation_done(
        self, group_name: str, step_idx: int, method_name: str, seg_result: dict
    ) -> None:
        step_key = f"{group_name}_{step_idx}"
        self.state.segmentation_status[step_key] = "Done"
        self.segmentation_panel.update_step_status(group_name, step_idx, "Done")
        self._seg_completed_steps = getattr(self, "_seg_completed_steps", 0) + 1
        total = getattr(self, "_seg_total_steps", 0)
        self.segmentation_panel.set_progress(
            self._seg_completed_steps,
            total,
            f"Completed {group_name}: {method_name} ({self._seg_completed_steps}/{total})",
        )

    def _on_group_complete(self, group_name: str, seg_result: dict) -> None:
        self.state.segmentation_results[group_name] = seg_result
        scale_val = ViewerController._get_scale_val(self)

        def _add_or_update_mask(data: np.ndarray | None, suffix: str) -> None:
            if data is None:
                return
            layer_name = f"{group_name} {suffix}"
            opac = 0.5 if "cyto" in suffix.lower() else 0.8
            lyr = ViewerController._upsert_labels_layer(
                self,
                data,
                layer_name=layer_name,
                scale_val=scale_val,
                translate=(0.0, 0.0),
                visible=False,
                opacity=opac,
            )
            if self.state.current_sample_name:
                sl = self.state.sample_layers_map.setdefault(self.state.current_sample_name, [])
                if lyr not in sl:
                    sl.append(lyr)
            if layer_name not in self.added_legend_entries:
                self.legend_widget.add_segmentation_row(lyr, layer_name, "#888888")
                self.added_legend_entries.add(layer_name)

        _add_or_update_mask(seg_result.get("nuclear"), "Nuclear Mask")
        _add_or_update_mask(seg_result.get("cytoplasm"), "Cyto Mask")
        self._sync_segmentation_sources()

    def _on_all_segmentations_complete(self) -> None:
        self.segmentation_panel.set_running_state(False)
        masks = {
            m: res
            for m, res in self.state.segmentation_results.items()
            if res.get("nuclear") is not None or res.get("cytoplasm") is not None
        }
        seg_errors = getattr(self, "_seg_errors", [])
        if not masks:
            detail = (
                "\n".join(seg_errors)
                if seg_errors
                else "All segmentation groups failed.  Check dependencies and parameters."
            )
            QMessageBox.warning(
                self.control_widget,
                "Segmentation Failed" if seg_errors else "No Results",
                detail,
            )
            return

        self._sync_segmentation_sources()
        n = len(masks)
        active = self.state.active_segmentation_method or next(iter(masks))
        nuc_active, cyto_active = self.state.get_active_masks()
        n_cells = _count_labels(nuc_active if nuc_active is not None else cyto_active)
        self.control_widget.update_dashboard(
            seg_status=f"{n} pipeline(s) applied",
            total_cells=f"{n_cells:,}",
        )
        if n >= 2:
            self.segmentation_panel.lbl_disagreement_status.setText(
                f"Ready for disagreement check: {n} pipeline(s)"
            )
        else:
            self.segmentation_panel.lbl_disagreement_status.setText(
                f"Completed 1 pipeline ({active}); run >= 2 for disagreement check"
            )
        self.segmentation_panel.lbl_studio_summary.setText(
            f"Pipelines: {n} applied ({n_cells:,} cells) | Disagreements: Ready | Features: Idle"
        )
        if seg_errors:
            QMessageBox.warning(
                self.control_widget,
                "Partial Segmentation Errors",
                "Some segmentation steps failed:\n\n" + "\n".join(seg_errors[:10]),
            )

    def _cancel_segmentation(self) -> None:
        self.pipeline_service.cancel_all()
        self.segmentation_panel.set_running_state(False)

        for key in list(self.state.segmentation_status.keys()):
            if self.state.segmentation_status[key] in ("Running", "Queued"):
                self.state.segmentation_status[key] = "Cancelled"
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
            self.control_widget,
            "Cancellation Sent",
            "Segmentation cancellation requested.\n\n"
            "Note: a running deep-learning operation must finish its current "
            "tensor pass before the worker thread actually closes.",
        )

    # Active Mask & Disagreement QC

    def _on_active_segmentation_method_changed(self, method: str) -> None:
        method = str(method).strip()
        if not method or method not in self.state.segmentation_results:
            return
        if hasattr(self, "review_svc") and hasattr(self.review_svc, "set_active_mask_layers"):
            nuc, _cyto = self.review_svc.set_active_mask_layers(method)
        else:
            nuc, _cyto = self.state.set_active_segmentation_method(method)

        ViewerController._sync_masks_map_for_method(self, method, nuc)

        methods = [
            m
            for m, r in self.state.segmentation_results.items()
            if r.get("nuclear") is not None or r.get("cytoplasm") is not None
        ]
        if hasattr(self, "feature_panel") and self.feature_panel is not None:
            self.feature_panel.update_mask_sources(methods, active=method)
        if hasattr(self, "disagreement_panel") and self.disagreement_panel is not None:
            if self.disagreement_panel.get_active_method() != method:
                self.disagreement_panel.update_available_methods(methods, active_method=method)

    def _on_overlay_color_or_palette_changed(self, *_args: Any) -> None:
        if hasattr(self, "review_svc") and self.review_svc is not None:
            self.review_svc.render_disagreement_overlays()
        sp = getattr(self, "segmentation_panel", None)
        if sp is not None and sp.popup_dialog is not None and sp.popup_dialog.isVisible():
            sp.popup_dialog.refresh_data()

    def _on_selected_qc_methods_changed(self, selected_methods: list) -> None:
        if not hasattr(self, "review_svc") or self.review_svc is None:
            return
        if getattr(self.state, "disagreement_report", None) is not None and len(selected_methods) >= 1:
            self.review_svc.update_disagreement_overlay(self.disagreement_panel.get_threshold())
        else:
            self.review_svc.render_disagreement_overlays()
        sp = getattr(self, "segmentation_panel", None)
        if sp is not None and sp.popup_dialog is not None and sp.popup_dialog.isVisible():
            sp.popup_dialog.refresh_data()

    def _compute_disagreement(self) -> None:
        dp = self.disagreement_panel
        compare_mode = dp.combo_compare.currentText()
        ref_method = dp.get_reference_method() or None
        iou_threshold = dp.get_threshold()
        selected_methods: Optional[List[str]] = None
        if hasattr(dp, "get_selected_methods") and getattr(dp, "_methods", []):
            selected_methods = dp.get_selected_methods()
            if not selected_methods:
                QMessageBox.warning(
                    self.control_widget,
                    "No Methods Selected",
                    "Check at least one segmentation method in Method Comparison & Color Key.",
                )
                return

        seg_results: dict = {}
        for m, res in self.state.segmentation_results.items():
            if selected_methods is not None and m not in selected_methods:
                continue
            if compare_mode == "Nuclear Masks":
                if res.get("nuclear") is not None:
                    seg_results[m] = {"nuclear": res["nuclear"], "cytoplasm": None}
            elif compare_mode == "Cytoplasm Masks":
                if res.get("cytoplasm") is not None:
                    seg_results[m] = {"nuclear": None, "cytoplasm": res["cytoplasm"]}
            else:
                if res.get("nuclear") is not None or res.get("cytoplasm") is not None:
                    seg_results[m] = res

        if not seg_results:
            QMessageBox.warning(
                self.control_widget,
                "No Masks",
                "Run segmentation before computing model disagreements.",
            )
            return

        if ref_method not in seg_results:
            ref_method = None

        iris_log(
            "IRIS-Viewer",
            "Disagreement QC",
            f"Comparing {len(seg_results)} method(s) ({compare_mode}, IoU>={iou_threshold:.2f})...",
        )
        dp.set_computing_state(True)

        @thread_worker
        def _run():
            return compute_model_disagreement(
                seg_results,
                compare_mode=compare_mode,
                reference_method=ref_method,
                iou_threshold=iou_threshold,
                selected_methods=selected_methods,
            )

        worker = _run()
        worker.returned.connect(self._on_disagreement_complete)
        self.track_worker(
            worker,
            on_error=self._on_disagreement_error,
            error_title="Disagreement Error",
            error_prefix="Failed to compute model disagreements",
        )
        worker.start()

    def _on_disagreement_error(self, exc: Exception) -> None:
        self.disagreement_panel.set_computing_state(False)
        log_error("IRIS-Viewer", "Disagreement QC", f"Failed to compute disagreements: {exc}")
        QMessageBox.critical(
            self.control_widget,
            "Disagreement Error",
            f"Failed to compute model disagreements:\n{exc}",
        )

    def _on_disagreement_complete(self, report: Any) -> None:
        self.disagreement_panel.set_computing_state(False)
        self.state.disagreement_report = report
        self.state.disagreement_heatmap = report.disagreement_heatmap
        self.review_svc.render_disagreement_overlays(report)

        if not self.state.active_segmentation_method and report.reference_method:
            self._on_active_segmentation_method_changed(report.reference_method)

        iris_log(
            "IRIS-Viewer",
            "Disagreement QC",
            f"Complete: {report.agreed_cells}/{report.total_ref_cells} agreed, "
            f"{report.flagged_regions} flagged, mean IoU={report.mean_iou:.3f}.",
        )

        self.control_widget.update_dashboard(
            total_cells=f"{report.total_ref_cells:,}",
            confidence=f"{report.mean_iou:.3f} (mean IoU)",
        )
        self.segmentation_panel.lbl_disagreement_status.setText(
            f"Disagreements: {report.flagged_regions:,} region(s) | "
            f"Mean IoU: {report.mean_iou:.3f} ({len(report.methods)} model(s))"
        )
        self.segmentation_panel.lbl_studio_summary.setText(
            f"Pipelines: Ready | Disagreements: {report.flagged_regions:,} flagged | "
            "Features: Idle"
        )
        if (
            self.segmentation_panel.popup_dialog is not None
            and self.segmentation_panel.popup_dialog.isVisible()
        ):
            self.segmentation_panel.popup_dialog.refresh_data()

    # Feature Extraction

    def _run_feature_extraction(
        self, selected_morphology: list, selected_intensity: list
    ) -> None:
        fp = self.feature_panel
        scope_mode = SCOPE_ACTIVE
        target_samples: List[str] = []
        if hasattr(fp, "sample_scope_bar"):
            all_samples, active_samples = ViewerController._sync_sample_scope_bars(self)
            scope_mode = fp.sample_scope_bar.get_scope_mode()
            target_samples = fp.get_target_samples(all_samples, active_samples)

        active_method = fp.get_selected_mask_source() or self.state.active_segmentation_method
        compartments = fp.get_selected_compartments()
        scale_val = float(self.control_widget.spin_scale.value())
        markers = list(self.state.markers)

        is_single_active = (
            scope_mode == SCOPE_ACTIVE
            and not self.state.multi_sample_images
        )

        if is_single_active:
            nuc_mask, cyto_mask = self.state.get_active_masks()
            if active_method and active_method in self.state.segmentation_results:
                entry = self.state.segmentation_results[active_method]
                nuc_mask = entry.get("nuclear")
                cyto_mask = entry.get("cytoplasm")
            if nuc_mask is None and cyto_mask is None:
                QMessageBox.warning(
                    self.control_widget,
                    "No Mask",
                    "Run segmentation or load a mask first.",
                )
                return
            if self.state.raw_image is None:
                QMessageBox.warning(self.control_widget, "No Image", "Load an image first.")
                return

            primary_mask = nuc_mask if nuc_mask is not None else cyto_mask
            cyto_arg = cyto_mask if nuc_mask is not None else None
            sample_tag = self.state.current_sample_name or "Sample"
            raw_img = self.state.raw_image

            self.feature_panel.set_extracting_state(True)
            iris_log(
                "IRIS-Viewer",
                "Features",
                f"Extracting features for '{sample_tag}'...",
            )

            @thread_worker
            def _extract_single():
                df = extract_cell_features(
                    mask=primary_mask,
                    raw_image=np.asarray(raw_img),
                    markers=markers,
                    scale_factor=scale_val,
                    selected_morphology=selected_morphology,
                    selected_intensity=selected_intensity,
                    cytoplasm_mask=cyto_arg,
                    compartments=compartments,
                )
                if df is not None and not df.empty and "Image" not in df.columns:
                    df["Image"] = sample_tag
                return df

            worker = _extract_single()
            worker.returned.connect(self._on_features_extracted)
            self.track_worker(
                worker,
                on_error=self._on_features_extraction_error,
                error_title="Feature Extraction Error",
                error_prefix="Failed to extract features",
            )
            worker.start()
            return

        # Multi-sample canvas or Folder Scope (Selected Samples / All Samples)
        if scope_mode == SCOPE_ACTIVE:
            target_samples = list(self.state.multi_sample_images.keys())

        if not target_samples:
            QMessageBox.warning(
                self.control_widget,
                "No Samples Selected",
                "Select at least one sample in Feature Scope before extracting.",
            )
            return

        jobs: List[Tuple[str, Any, Any, Any]] = []
        any_image = False
        for sn in target_samples:
            img_src: Any = self.state.multi_sample_images.get(sn)
            if img_src is None and sn == self.state.current_sample_name:
                img_src = self.state.raw_image
            if img_src is None:
                img_src = self.state.file_map.get(sn)
            if img_src is None:
                continue
            any_image = True

            nuc_src: Any = None
            cyto_src: Any = None
            b_entry = self.state.batch_results.get(sn)
            if (
                isinstance(b_entry, dict)
                and active_method
                and isinstance(b_entry.get(active_method), dict)
            ):
                nuc_src = b_entry[active_method].get("nuclear")
                cyto_src = b_entry[active_method].get("cytoplasm")
            if (
                nuc_src is None
                and cyto_src is None
                and sn == self.state.current_sample_name
                and active_method
                and active_method in self.state.segmentation_results
            ):
                s_entry = self.state.segmentation_results[active_method]
                nuc_src = s_entry.get("nuclear")
                cyto_src = s_entry.get("cytoplasm")
            if nuc_src is None and sn in self.state.masks_map:
                m_entry = self.state.masks_map[sn]
                if isinstance(m_entry, dict):
                    nuc_src = m_entry.get("nuclear")
                    cyto_src = m_entry.get("cytoplasm")
                else:
                    nuc_src = m_entry
                    if sn == self.state.current_sample_name:
                        cyto_src = self.state.cytoplasm_mask
            if nuc_src is None and cyto_src is None and sn in self.state.file_map:
                img_p = Path(self.state.file_map[sn])
                for cand_name in (f"{sn}_mask.tiff", f"{sn}_segmentation_mask.tiff"):
                    cand_p = img_p.with_name(cand_name)
                    if cand_p.exists():
                        nuc_src = cand_p
                        break

            if nuc_src is not None or cyto_src is not None:
                jobs.append((sn, img_src, nuc_src, cyto_src))

        if not any_image:
            QMessageBox.warning(self.control_widget, "No Image", "Load an image first.")
            return
        if not jobs:
            QMessageBox.warning(
                self.control_widget,
                "No Mask",
                "Run segmentation or load masks for the target samples first.",
            )
            return

        self.feature_panel.set_extracting_state(True)
        iris_log(
            "IRIS-Viewer",
            "Features",
            f"Extracting features across {len(jobs)} sample(s) (scope='{scope_mode}')...",
        )

        @thread_worker
        def _extract_multi():
            dfs: List[pd.DataFrame] = []
            for sn, img_src, nuc_src, cyto_src in jobs:
                if isinstance(img_src, (Path, str)):
                    raw_arr, _ = load_multichannel_tiff(Path(img_src))
                    raw_np = np.asarray(raw_arr)
                else:
                    raw_np = np.asarray(img_src)

                nuc_arr = nuc_src
                cyto_arr = cyto_src
                if isinstance(nuc_src, Path):
                    loaded = load_existing_mask(nuc_src)
                    if loaded is None:
                        continue
                    if loaded.ndim == 3:
                        nuc_arr = loaded[0]
                        cyto_arr = loaded[1] if loaded.shape[0] > 1 else None
                    else:
                        nuc_arr = loaded
                        cyto_arr = None

                primary = nuc_arr if nuc_arr is not None else cyto_arr
                cyto_arg = cyto_arr if nuc_arr is not None else None
                if primary is None:
                    continue

                sub_df = extract_cell_features(
                    mask=primary,
                    raw_image=raw_np,
                    markers=markers,
                    scale_factor=scale_val,
                    selected_morphology=selected_morphology,
                    selected_intensity=selected_intensity,
                    cytoplasm_mask=cyto_arg,
                    compartments=compartments,
                )
                if sub_df is not None and not sub_df.empty:
                    sub_df["Image"] = sn
                    dfs.append(sub_df)
                    iris_log(
                        "IRIS-Viewer",
                        "Features",
                        f"Extracted {len(sub_df):,} cells from '{sn}'.",
                    )

            if not dfs:
                return pd.DataFrame()
            return pd.concat(dfs, ignore_index=True)

        worker = _extract_multi()
        worker.returned.connect(self._on_features_extracted)
        self.track_worker(
            worker,
            on_error=self._on_features_extraction_error,
            error_title="Feature Extraction Error",
            error_prefix="Failed to extract features",
        )
        worker.start()

    def _on_features_extraction_error(self, exc: Exception) -> None:
        self.feature_panel.set_extracting_state(False)
        log_error("IRIS-Viewer", "Features", f"Feature extraction failed: {exc}")
        QMessageBox.critical(
            self.control_widget,
            "Feature Extraction Error",
            f"Failed to extract features:\n{exc}",
        )

    def _on_features_extracted(self, df) -> None:
        self.state.features_df = df
        self.feature_panel.set_extracting_state(False)
        self.feature_panel.populate_preview(df)
        self._sync_downstream_panels(df=df)
        if df is not None and not df.empty:
            n_samples = int(df["Image"].nunique()) if "Image" in df.columns else 1
            iris_log(
                "IRIS-Viewer",
                "Features",
                f"Complete: {len(df):,} cells x {len(df.columns)} features "
                f"across {n_samples} sample(s).",
            )
            self.segmentation_panel.lbl_studio_summary.setText(
                f"Pipelines: Ready | Disagreements: Ready | Features: {len(df):,} cells"
            )

    # Exports

    def _export_mask(self, fmt: str = "") -> None:
        nuc_mask, cyto_mask = self.state.get_active_masks()
        if nuc_mask is None and cyto_mask is None:
            QMessageBox.warning(
                self.control_widget,
                "No Mask",
                "Run segmentation or load a mask before exporting.",
            )
            return

        primary_mask = nuc_mask if nuc_mask is not None else cyto_mask
        cyto_arg = cyto_mask if nuc_mask is not None else None
        sample = self.state.current_sample_name or "sample"
        ext = fmt if fmt.startswith(".") else f".{fmt}" if fmt else ".tiff"
        default_name = f"{sample}_mask{ext}"

        filters = (
            "TIFF Image (*.tiff *.tif);;OME-TIFF Image (*.ome.tif *.ome.tiff);;"
            "PNG Image (*.png);;NumPy Array (*.npy);;NumPy Compressed (*.npz);;All Files (*)"
        )
        filepath, _selected_filter = QFileDialog.getSaveFileName(
            self.control_widget,
            "Export Segmentation Mask",
            default_name,
            filters,
        )
        if filepath:
            try:
                export_segmentation_mask(Path(filepath), primary_mask, cyto_arg)
                iris_log("IRIS-Viewer", "Export", f"Saved mask to {Path(filepath).name}.")
                QMessageBox.information(
                    self.control_widget,
                    "Export Complete",
                    f"Segmentation mask saved:\n{Path(filepath).name}",
                )
            except Exception as e:
                QMessageBox.critical(
                    self.control_widget, "Export Error", f"Failed to save mask: {e}"
                )

    def _export_features(self, fmt: str = "") -> None:
        if self.state.features_df is None or self.state.features_df.empty:
            QMessageBox.warning(self.control_widget, "No Features", "Extract features first.")
            return
        sample = self.state.current_sample_name or "sample"
        ext = fmt if fmt.startswith(".") else f".{fmt}" if fmt else ".csv"
        default_name = f"{sample}_features{ext}"

        filters = (
            "CSV (*.csv);;Excel Workbook (*.xlsx);;Apache Parquet (*.parquet);;"
            "Tab-Separated Values (*.tsv);;JSON (*.json);;All Files (*)"
        )
        filepath, _selected_filter = QFileDialog.getSaveFileName(
            self.control_widget,
            "Export Features",
            default_name,
            filters,
        )
        if filepath:
            try:
                export_features_csv(Path(filepath), self.state.features_df, sample)
                iris_log("IRIS-Viewer", "Export", f"Saved features to {Path(filepath).name}.")
                QMessageBox.information(
                    self.control_widget,
                    "Export Complete",
                    f"Features saved:\n{Path(filepath).name}",
                )
            except Exception as e:
                QMessageBox.critical(
                    self.control_widget, "Export Error", f"Failed to save features: {e}"
                )

    def _export_confidence(self, fmt: str = "") -> None:
        heatmap = self.state.disagreement_heatmap
        if heatmap is None and self.state.disagreement_report is not None:
            heatmap = getattr(self.state.disagreement_report, "disagreement_heatmap", None)
        if heatmap is None:
            QMessageBox.warning(
                self.control_widget,
                "No Disagreement Map",
                "Compute multi-model disagreements first before exporting the disagreement map.",
            )
            return
        sample = self.state.current_sample_name or "sample"
        ext = fmt if fmt.startswith(".") else f".{fmt}" if fmt else ".tiff"
        default_name = f"{sample}_disagreement{ext}"

        filters = (
            "TIFF Image (*.tiff *.tif);;PNG Heatmap (*.png);;"
            "NumPy Array (*.npy);;NumPy Compressed (*.npz);;All Files (*)"
        )
        filepath, _selected_filter = QFileDialog.getSaveFileName(
            self.control_widget,
            "Export Disagreement Map",
            default_name,
            filters,
        )
        if filepath:
            try:
                export_disagreement_map(Path(filepath), heatmap)
                iris_log("IRIS-Viewer", "Export", f"Saved disagreement map to {Path(filepath).name}.")
                QMessageBox.information(
                    self.control_widget,
                    "Export Complete",
                    f"Disagreement heatmap saved:\n{Path(filepath).name}",
                )
            except Exception as e:
                QMessageBox.critical(
                    self.control_widget,
                    "Export Error",
                    f"Failed to save disagreement map: {e}",
                )

    def _export_updated_phenotypes(self, fmt: str = "") -> None:
        if getattr(self.state, "master_df", None) is None or self.state.master_df.empty:
            QMessageBox.warning(
                self.control_widget, "No Annotation Data", "Load annotation data first."
            )
            return

        ext = fmt if fmt.startswith(".") else f".{fmt}" if fmt else ".csv"
        default_name = f"updated_phenotypes{ext}"
        filters = (
            "CSV (*.csv);;Excel Workbook (*.xlsx);;Apache Parquet (*.parquet);;"
            "Tab-Separated Values (*.tsv);;JSON (*.json);;All Files (*)"
        )
        filepath, _selected_filter = QFileDialog.getSaveFileName(
            self.control_widget,
            "Export Updated Phenotypes",
            default_name,
            filters,
        )
        if filepath:
            ref_col = self.control_widget.combo_ref.currentText()
            df_export = self.state.master_df.copy()
            updated_col_name = f"{ref_col}_updated"

            if "Verified_CellType" in df_export.columns:
                df_export[updated_col_name] = df_export["Verified_CellType"]
                df_export.drop(columns=["Verified_CellType"], inplace=True)
            else:
                df_export[updated_col_name] = df_export[ref_col]

            try:
                lower = filepath.lower()
                if lower.endswith(".xlsx"):
                    df_export.to_excel(filepath, index=False)
                elif lower.endswith(".parquet"):
                    df_export.to_parquet(filepath, index=False)
                elif lower.endswith(".tsv") or lower.endswith(".txt"):
                    df_export.to_csv(filepath, sep="\t", index=False)
                elif lower.endswith(".json"):
                    df_export.to_json(filepath, orient="records", indent=2)
                else:
                    df_export.to_csv(filepath, index=False)
                QMessageBox.information(
                    self.control_widget,
                    "Export Complete",
                    f"Updated phenotypes saved:\n{Path(filepath).name}",
                )
            except Exception as e:
                QMessageBox.critical(
                    self.control_widget, "Export Error", f"Failed to save file: {e}"
                )

    # Session

    def _save_session(self) -> None:
        sample = self.state.current_sample_name or "session"
        filepath, _ = QFileDialog.getSaveFileName(
            self.control_widget,
            "Save Session",
            f"{sample}_session.json",
            "JSON (*.json)",
        )
        if filepath:
            threshold = self.disagreement_panel.get_threshold()
            self.state.save_session(Path(filepath), disagreement_threshold=threshold)
            QMessageBox.information(
                self.control_widget,
                "Session Saved",
                f"Session saved to:\n{filepath}",
            )

    def _load_session(self, filepath: Optional[Union[str, Path]] = None) -> None:
        if not filepath:
            selected_path, _ = QFileDialog.getOpenFileName(
                self.control_widget, "Load Session", "", "JSON (*.json)"
            )
            if not selected_path:
                return
            filepath = selected_path

        filepath = Path(filepath)
        if not filepath.is_file():
            QMessageBox.warning(
                self.control_widget,
                "File Not Found",
                f"Session file not found:\n{filepath}",
            )
            return

        try:
            restored, raw_data = ViewerAppState.load_session(filepath)

            self.state.markers = restored.markers
            self.state.active_segmentation_method = restored.active_segmentation_method
            self.state.persistent_settings = restored.persistent_settings
            self.state.channel_colors = restored.channel_colors
            self.state.celltype_colors = restored.celltype_colors
            if hasattr(self, "color_svc"):
                self.color_svc.default_colors.update(restored.channel_colors)
                self.color_svc.celltype_colors.update(restored.celltype_colors)
            if hasattr(self, "multi_sample_svc"):
                self.multi_sample_svc.DEFAULT_COLORS.update(restored.channel_colors)
                self.multi_sample_svc.CELLTYPE_COLORS.update(restored.celltype_colors)

            thresh = float(raw_data.get("disagreement_threshold", 0.65))
            self.disagreement_panel.slider_threshold.setValue(int(round(thresh * 100)))

            if restored.image_dir and restored.image_dir.is_dir():
                self.image_svc.on_load_directory(str(restored.image_dir))

            QMessageBox.information(
                self.control_widget,
                "Session Loaded",
                f"Session restored from:\n{filepath}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self.control_widget,
                "Error",
                f"Failed to load session:\n{exc}",
            )
