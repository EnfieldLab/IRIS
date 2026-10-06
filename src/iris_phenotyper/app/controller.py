"""
IRIS-Phenotyper Controller
Thin coordinator: owns the napari viewer, creates widgets, wires signals,
and drives phenotyping, ensemble voting, UMAP, and export.
"""
from __future__ import annotations

import types as pytypes
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import napari
from napari.qt import thread_worker
from qtpy.QtWidgets import QFileDialog, QMessageBox

from iris_common.io import export_features_csv, export_consensus_mask, export_confidence_map
from iris_common.types import CellTypePanel, MarkerPanel

from iris_phenotyper.core.state import PhenotyperAppState
from iris_phenotyper.engine.ensemble import PhenotypingEnsemble
from iris_phenotyper.engine.phenotyping.astir_backend import ASTIRBackend
from iris_phenotyper.engine.phenotyping.maps_backend import MAPSBackend
from iris_phenotyper.widgets.control import PhenotyperControlWidget
from iris_phenotyper.widgets.marker_type_editor import MarkerTypeEditorWidget
from iris_phenotyper.widgets.phenotyping_panel import PhenotypingPanelWidget
from iris_phenotyper.widgets.ensemble_panel import EnsemblePanelWidget
from iris_common.app.services.image_service import ImageService
from iris_common.app.services.multi_sample_service import MultiSampleService
from iris_common.widgets.legend import ViewerLegendWidget


# Cell type colour palette — matches EnsemblePanelWidget._TYPE_PALETTE
_TYPE_PALETTE = [
    "#4477EE", "#44AA44", "#EE4444", "#FFAA22", "#AA44EE",
    "#22AACC", "#FF6688", "#88CC44", "#AA8822", "#44CCAA",
    "#CC4488", "#6688FF", "#FFCC44", "#44DDFF", "#FF8844",
    "#AADDAA", "#DD88FF", "#88AAFF", "#FFDD88", "#CC88AA",
]


def _hex_to_rgba(hex_color: str, alpha: float = 1.0):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return [r, g, b, alpha]


class PhenotyperController:
    """Thin coordinator for IRIS-Phenotyper."""

    def __init__(self) -> None:
        self.state = PhenotyperAppState()
        self._active_workers: list = []
        self._backends = {"ASTIR": ASTIRBackend(), "MAPS": MAPSBackend()}

        # ── Viewer ────────────────────────────────────────────────────────────
        self.viewer = napari.Viewer(title="IRIS-Phenotyper")

        # ── Widgets ───────────────────────────────────────────────────────────
        self.control_widget    = PhenotyperControlWidget(viewer=self.viewer)
        self.marker_editor     = MarkerTypeEditorWidget()
        self.phenotyping_panel = PhenotypingPanelWidget()
        self.ensemble_panel    = EnsemblePanelWidget()
        self.legend_widget     = ViewerLegendWidget(self.viewer)
        self.comparison_widget = self.control_widget.comparison_widget

        self.added_legend_entries = set()
        
        self._ctx = pytypes.SimpleNamespace(
            viewer=self.viewer,
            state=self.state,
            control_widget=self.control_widget,
            legend_widget=self.legend_widget,
            comparison_widget=self.comparison_widget,
            added_legend_entries=self.added_legend_entries,
            active_workers=self._active_workers,
        )

        self.image_svc = ImageService(self._ctx)
        self.multi_sample_svc = MultiSampleService(self._ctx)

        from iris_phenotyper.widgets.unsupervised_panel import UnsupervisedPanelWidget
        self.unsupervised_panel = UnsupervisedPanelWidget()

        for w, name in [
            (self.control_widget,       "Control"),
            (self.comparison_widget,    "TMA"),
            (self.marker_editor,        "Marker Panel"),
            (self.phenotyping_panel,    "Phenotyping"),
            (self.unsupervised_panel,   "Unsupervised"),
            (self.ensemble_panel,       "Ensemble"),
            (self.legend_widget,        "Legend"),
        ]:
            self.viewer.window.add_dock_widget(w, name=name, area="right")

        self.comparison_widget.show()

        dw = self.viewer.window.dock_widgets
        qt = self.viewer.window._qt_window
        qt.tabifyDockWidget(dw["Control"].parent(),      dw["TMA"].parent())
        qt.tabifyDockWidget(dw["TMA"].parent(),          dw["Unsupervised"].parent())
        
        dw["Marker Panel"].parent().hide()
        dw["Phenotyping"].parent().hide()
        dw["Ensemble"].parent().hide()

        dw["Control"].parent().raise_()

        self._bind_signals()

    # ── Signals ───────────────────────────────────────────────────────────────

    def _bind_signals(self) -> None:
        cw = self.control_widget

        # Base ImageControlWidget signals
        cw.loadDirectoryTriggered.connect(self.image_svc.on_load_directory)
        cw.sampleSelected.connect(self.image_svc.on_sample_selected)
        cw.loadExistingMaskTriggered.connect(self.image_svc.on_load_existing_mask)
        cw.loadChannelColorTriggered.connect(self._load_colors_smart)
        cw.scaleFactorChanged.connect(self.image_svc.update_scale)
        cw.viewLockToggled.connect(self._toggle_view_lock)
        
        # Phenotyper Annotation
        if hasattr(cw, 'loadAnnotationDataTriggered'):
            cw.loadAnnotationDataTriggered.connect(self._load_annotation_data)
            cw.loadAnnotationDataTriggered.connect(self.multi_sample_svc.set_master_file)
        
        # MultiSample comparison widget signals
        ms_cw = self.comparison_widget
        ms_cw.addSamplesTriggered.connect(self.multi_sample_svc.add_samples_to_canvas)
        ms_cw.clearCanvasTriggered.connect(self.multi_sample_svc.clear_canvas)

        # Output/Export signals
        cw.exportMaskTriggered.connect(self._export_mask)
        cw.exportFeaturesTriggered.connect(self._export_features)
        cw.exportConfidenceTriggered.connect(self._export_confidence)
        cw.batchProcessTriggered.connect(self._run_batch)

        self.legend_widget.layerSelected.connect(self._select_layer)

        self.phenotyping_panel.runRequested.connect(self._run_phenotyping)
        self.phenotyping_panel.cancelRequested.connect(self._cancel)

        self.ensemble_panel.cellOverrideRequested.connect(self._apply_override)
        self.ensemble_panel.undoRequested.connect(self._undo_override)
        self.ensemble_panel.showUMAPRequested.connect(self._run_umap)

        self.unsupervised_panel.runClusteringRequested.connect(self._run_unsupervised_clustering)
        self.unsupervised_panel.showHeatmapRequested.connect(self._show_unsupervised_heatmap)

    def _run_unsupervised_clustering(self, method: str, k: int) -> None:
        if self.state.features_df is None:
            self._show_error("No Data", "Please load an Annotation file first.")
            return

        import scanpy as sc
        import phenograph
        import anndata
        from qtpy.QtWidgets import QApplication

        self.unsupervised_panel.set_running_state(True)
        QApplication.processEvents()

        try:
            # We use the selected markers in self.state.markers
            if not self.state.markers:
                raise ValueError("No markers selected for clustering.")

            X = self.state.features_df[self.state.markers].values
            
            if method == "Leiden":
                adata = anndata.AnnData(X=X)
                sc.pp.neighbors(adata, n_neighbors=k)
                sc.tl.leiden(adata)
                clusters = adata.obs['leiden'].values.astype(int)
            elif method == "PhenoGraph":
                clusters, graph, Q = phenograph.cluster(X, k=k)
            else:
                raise ValueError(f"Unknown method {method}")

            # Store the cluster assignment
            self.state.features_df['Unsupervised_Cluster'] = clusters
            
            unique_clusters = np.unique(clusters)
            self.unsupervised_panel.populate_clusters(unique_clusters.tolist())

            # Update napari points layer if it exists
            # (optional, we could color by cluster if user wanted)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._show_error("Clustering Failed", str(exc))
        finally:
            self.unsupervised_panel.set_running_state(False)

    def _show_unsupervised_heatmap(self, selected_clusters: list) -> None:
        if self.state.features_df is None or 'Unsupervised_Cluster' not in self.state.features_df.columns:
            return

        # Filter by selected clusters
        df = self.state.features_df
        mask = df['Unsupervised_Cluster'].isin(selected_clusters)
        filtered = df[mask]
        
        if len(filtered) == 0:
            self._show_error("No Data", "Selected clusters contain no cells.")
            return

        # Group by cluster and calculate mean marker expression
        grouped = filtered.groupby('Unsupervised_Cluster')[self.state.markers].mean()
        
        # We want clusters on rows, markers on columns, this is already the grouped shape!
        
        from iris_phenotyper.widgets.heatmap_window import HeatmapWindow
        self._heatmap_win = HeatmapWindow(grouped, parent=self.control_widget)
        self._heatmap_win.show()

    def _toggle_view_lock(self, is_locked: bool) -> None:
        self.viewer.camera.mouse_pan  = not is_locked
        self.viewer.camera.mouse_zoom = not is_locked
        if is_locked:
            self._old_double_clicks = list(self.viewer.mouse_double_click_callbacks)
            self.viewer.mouse_double_click_callbacks.clear()
        elif hasattr(self, "_old_double_clicks"):
            self.viewer.mouse_double_click_callbacks.extend(self._old_double_clicks)
            del self._old_double_clicks

    def _select_layer(self, layer_name: str) -> None:
        self.viewer.layers.selection.clear()
        for layer in self.viewer.layers:
            base = layer.name.split(" [")[0] if " [" in layer.name else layer.name
            if base == layer_name:
                self.viewer.layers.selection.add(layer)
                break

    # ── Outputs & Exports (From ImageControlWidget) ──────────────────────────

    def _export_mask(self) -> None:
        if self.state.current_mask is None:
            QMessageBox.warning(self.control_widget, "No Mask", "No mask loaded to export.")
            return
        sample = self.state.current_sample_name or "sample"
        filepath, _ = QFileDialog.getSaveFileName(
            self.control_widget, "Export Mask",
            f"{sample}_mask.tiff", "TIFF (*.tiff)"
        )
        if filepath:
            export_consensus_mask(Path(filepath), self.state.current_mask, self.state.cytoplasm_mask)
            QMessageBox.information(self.control_widget, "Export Complete", "Mask saved.")

    def _export_features(self) -> None:
        if self.state.features_df is None or self.state.features_df.empty:
            QMessageBox.warning(self.control_widget, "No Features", "Load or extract features first.")
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
        QMessageBox.information(self.control_widget, "Not Applicable", "Phenotyper does not generate confidence maps. Use IRIS-Viewer.")

    def _run_batch(self) -> None:
        QMessageBox.information(self.control_widget, "Not Applicable", "Batch processing is handled directly in the Phenotyping tab.")

    # ── File Loaders ─────────────────────────────────────────────────────────
    def _load_colors_smart(self, filepath: str) -> None:
        import pandas as pd
        path_obj = Path(filepath)
        try:
            df = pd.read_excel(filepath) if path_obj.suffix.lower() in ('.xlsx', '.xls') else pd.read_csv(filepath)
            cols = [str(c).lower().strip() for c in df.columns]
        except Exception as e:
            QMessageBox.critical(self.control_widget, "Error", f"Failed to read file:\n{e}")
            return
            
        has_markers = 'markers' in cols and 'colors' in cols
        has_celltypes = 'celltypes' in cols and 'colors' in cols
        
        if not has_markers and not has_celltypes:
            QMessageBox.warning(self.control_widget, "Invalid Format", "File must contain 'colors' and either 'markers' or 'celltypes'.")
            return
            
        if has_markers:
            self.image_svc.load_channel_colors(filepath)
        if has_celltypes:
            self.multi_sample_svc.load_celltype_colors(filepath)

    def _load_annotation_data(self, filepath: str) -> None:
        import pandas as pd
        path_obj = Path(filepath)
        try:
            if path_obj.suffix.lower() == '.h5ad':
                import anndata as ad
                adata = ad.read_h5ad(filepath)
                df = adata.to_df()
                # If there are important metadata columns (like spatial coords) in adata.obs, merge them
                if not adata.obs.empty:
                    df = pd.concat([df, adata.obs], axis=1)
            elif path_obj.suffix.lower() in ('.xlsx', '.xls'):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)
        except Exception as exc:
            QMessageBox.critical(self.control_widget, "Error", f"Failed to load annotation data:\n{exc}")
            return

        self.state.features_df = df

        # Auto-detect markers from column names
        markers = []
        # The user said: "the features do not need to have _mean. Update that."
        # We look for _mean first just in case
        for suffix in ["_mean", "_Mean", "_intensity_mean", "mean"]:
            markers = [c.replace(suffix, '') for c in df.columns if c.endswith(suffix)]
            if markers:
                break
                
        if not markers:
            # Fallback: all numeric columns not looking like spatial or ID metadata
            exclude = {"cell_id", "objectnumber", "pos_x", "pos_y", "x", "y", "centroid-0", "centroid-1", "area", "perimeter", "eccentricity"}
            import pandas.api.types as ptypes
            markers = [c for c in df.columns if c.lower() not in exclude and ptypes.is_numeric_dtype(df[c])]

        # Pop up dialog to let user filter markers
        if markers:
            from qtpy.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, QDialogButtonBox, QLabel
            from qtpy.QtCore import Qt
            dlg = QDialog(self.control_widget)
            dlg.setWindowTitle("Select Markers")
            dlg.resize(350, 450)
            layout = QVBoxLayout(dlg)
            layout.addWidget(QLabel("Include/Exclude detected markers:"))
            
            btn_layout = QHBoxLayout()
            btn_all = QPushButton("Select All")
            btn_none = QPushButton("Deselect All")
            btn_layout.addWidget(btn_all)
            btn_layout.addWidget(btn_none)
            layout.addLayout(btn_layout)

            list_widget = QListWidget()
            for m in markers:
                item = QListWidgetItem(m)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                list_widget.addItem(item)
            layout.addWidget(list_widget)

            def _select_all():
                for i in range(list_widget.count()):
                    list_widget.item(i).setCheckState(Qt.Checked)
            def _deselect_all():
                for i in range(list_widget.count()):
                    list_widget.item(i).setCheckState(Qt.Unchecked)

            btn_all.clicked.connect(_select_all)
            btn_none.clicked.connect(_deselect_all)

            bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            bbox.accepted.connect(dlg.accept)
            bbox.rejected.connect(dlg.reject)
            layout.addWidget(bbox)
            
            if dlg.exec_():
                selected_markers = []
                for i in range(list_widget.count()):
                    item = list_widget.item(i)
                    if item.checkState() == Qt.Checked:
                        selected_markers.append(item.text())
                if selected_markers:
                    markers = selected_markers

        self.state.markers = markers
        self.state.marker_panel = MarkerPanel(markers=markers, panel_file=Path(filepath))
        self.marker_editor.set_available_markers(markers)
        self.state.phenotyping_results.clear()
        self.state.umap_embedding = None

        QMessageBox.information(
            self.control_widget, "Loaded",
            f"Loaded {len(df):,} cells with {len(markers)} marker channels."
        )
        self._update_phenotyping_panel()



    # ── Phenotyping Pipeline ──────────────────────────────────────────────────

    def _run_phenotyping(self, configs: Dict) -> None:
        if self.state.features_df is None:
            QMessageBox.warning(
                self.control_widget, "No Data",
                "Load a features CSV first."
            )
            return

        cell_type_panel = self.marker_editor.get_cell_type_panel()
        if not cell_type_panel.rules:
            QMessageBox.warning(
                self.control_widget, "No Panel",
                "Define at least one cell type in the Marker Panel tab."
            )
            return

        self.state.cell_type_panel = cell_type_panel
        ensemble_cfg = configs.pop("_ensemble", {})
        method_configs = configs

        self.state.phenotyping_results.clear()
        self.phenotyping_panel.set_running_state(True)

        # Snapshot everything needed
        features_snapshot = self.state.features_df.copy()
        marker_panel = self.state.marker_panel
        backends = dict(self._backends)

        results = {}
        for method_name, params in method_configs.items():
            backend = backends.get(method_name)
            if backend is None or not backend.is_available():
                # MAPS fallback: use the pure-Python rule-based method
                if method_name == "MAPS":
                    backend = backends.get("MAPS")
                    if backend is not None:
                        try:
                            preds = backend.predict_fallback(
                                features_snapshot,
                                cell_type_panel=cell_type_panel,
                                marker_panel=marker_panel,
                            )
                            results[f"{method_name} (fallback)"] = preds
                            continue
                        except Exception as exc:
                            print(f"[Phenotyper] MAPS fallback failed: {exc}")
                print(f"[Phenotyper] {method_name} not available — skipping")
                continue

            try:
                from qtpy.QtWidgets import QApplication
                QApplication.processEvents() # Keep UI minimally alive
                
                preds = backend.predict(
                    features_snapshot,
                    cell_type_panel=cell_type_panel,
                    marker_panel=marker_panel,
                    **params,
                )
                results[method_name] = preds
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"[Phenotyper] {method_name} failed: {exc}")

        self._on_phenotyping_complete((results, ensemble_cfg))

    def _on_phenotyping_complete(self, result: tuple) -> None:
        predictions, ensemble_cfg = result
        self.state.phenotyping_results = predictions
        self.phenotyping_panel.set_running_state(False)

        if not predictions:
            QMessageBox.warning(
                self.control_widget, "No Results",
                "All phenotyping methods failed or were unavailable.\n"
                "Check that ASTIR is installed: pip install astir"
            )
            return

        strategy = ensemble_cfg.get("strategy", "soft_vote")
        min_conf = ensemble_cfg.get("min_probability_threshold", 0.3)

        ensemble = PhenotypingEnsemble(min_probability_threshold=min_conf)
        final_types, confidences, per_method = ensemble.vote(predictions, strategy=strategy)

        self.state.final_cell_types    = final_types
        self.state.cell_type_confidence = confidences
        self.state.per_method_breakdown = per_method

        # Merge results into features DataFrame
        self.state.ensemble_df = ensemble.results_to_dataframe(
            self.state.features_df,
            final_types, confidences, per_method,
        )

        cell_types = list(self.state.cell_type_panel.cell_types)
        self.ensemble_panel.update_results(final_types, confidences, cell_types)
        self._render_cell_type_layer()

        QMessageBox.information(
            self.control_widget, "Done",
            f"Phenotyping complete.\n"
            f"{len(final_types):,} cells assigned via {strategy}.\n"
            f"Methods: {', '.join(predictions.keys())}"
        )

    # ── Visualization ─────────────────────────────────────────────────────────

    def _render_cell_type_layer(self) -> None:
        """Add a napari Points layer coloured by cell type."""
        df = self.state.ensemble_df
        if df is None or "cell_type" not in df.columns:
            return
        if "Pos_X" not in df.columns or "Pos_Y" not in df.columns:
            return

        coords = df[["Pos_Y", "Pos_X"]].values
        cell_types = df["cell_type"].values
        unique_types = list(dict.fromkeys(cell_types))  # preserve order

        color_map = {ct: _TYPE_PALETTE[i % len(_TYPE_PALETTE)]
                     for i, ct in enumerate(unique_types)}

        face_colors = np.array([
            _hex_to_rgba(color_map.get(ct, "#888888"))
            for ct in cell_types
        ])

        layer_name = "Cell Types"
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)

        self.viewer.add_points(
            coords, name=layer_name,
            face_color=face_colors,
            size=6.0, opacity=0.85,
            symbol="disc",
        )

    # ── UMAP ──────────────────────────────────────────────────────────────────

    def _run_umap(self) -> None:
        if self.state.ensemble_df is None:
            QMessageBox.warning(self.control_widget, "No Data", "Run phenotyping first.")
            return

        suffix = "_mean"
        feat_cols = [c for c in self.state.ensemble_df.columns if c.endswith(suffix)]
        if not feat_cols:
            QMessageBox.warning(self.control_widget, "No Features", "No intensity columns found.")
            return

        X = self.state.ensemble_df[feat_cols].values.astype(np.float32)
        cell_types = self.state.ensemble_df["cell_type"].values

        @thread_worker
        def _compute_umap():
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            return reducer.fit_transform(X)

        def _on_umap_done(embedding: np.ndarray) -> None:
            self.state.umap_embedding = embedding
            unique_types = list(dict.fromkeys(cell_types))
            color_map = {ct: _TYPE_PALETTE[i % len(_TYPE_PALETTE)]
                         for i, ct in enumerate(unique_types)}
            face_colors = np.array([
                _hex_to_rgba(color_map.get(ct, "#888888"))
                for ct in cell_types
            ])
            umap_layer_name = "UMAP"
            if umap_layer_name in self.viewer.layers:
                self.viewer.layers.remove(umap_layer_name)
            # Open a second viewer window for the UMAP plot
            viewer2 = napari.Viewer(title="IRIS-Phenotyper UMAP")
            viewer2.add_points(
                embedding, name="UMAP",
                face_color=face_colors,
                size=3.0, opacity=0.7, symbol="disc",
            )
            viewer2.dims.ndisplay = 2

        worker = _compute_umap()
        worker.returned.connect(_on_umap_done)
        self._active_workers.append(worker)
        worker.start()

    # ── Overrides ─────────────────────────────────────────────────────────────

    def _apply_override(self, cell_id: int, new_type: str) -> None:
        self.state.snapshot_override_state()
        if new_type == "Remove":
            self.state.final_cell_types.pop(cell_id, None)
            self.state.cell_type_confidence.pop(cell_id, None)
        else:
            self.state.final_cell_types[cell_id] = new_type
            self.state.cell_type_confidence[cell_id] = 1.0
        self._render_cell_type_layer()

    def _undo_override(self) -> None:
        if self.state.undo_override():
            self._render_cell_type_layer()

    # ── Cancel ────────────────────────────────────────────────────────────────

    def _cancel(self) -> None:
        for w in self._active_workers:
            try:
                w.quit()
            except Exception:
                pass
        self._active_workers.clear()
        self.phenotyping_panel.set_running_state(False)

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_results(self) -> None:
        if self.state.ensemble_df is None:
            QMessageBox.warning(self.control_widget, "No Results", "Run phenotyping first.")
            return
        sample = self.state.current_sample_name or "sample"
        filepath, _ = QFileDialog.getSaveFileName(
            self.control_widget, "Export Phenotyping Results",
            f"{sample}_phenotypes.csv", "CSV (*.csv)"
        )
        if filepath:
            export_features_csv(Path(filepath), self.state.ensemble_df, sample)
            QMessageBox.information(self.control_widget, "Saved", f"Results saved to:\n{filepath}")

    # ── Session ───────────────────────────────────────────────────────────────

    def _save_session(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self.control_widget, "Save Session", "", "JSON (*.json)"
        )
        if filepath:
            self.state.save_session(Path(filepath))
            QMessageBox.information(self.control_widget, "Saved", f"Session saved to:\n{filepath}")

    # ── UI Updates ────────────────────────────────────────────────────────────

    def _update_phenotyping_panel(self) -> None:
        # Phenotyping panel has no summary view anymore.
        pass

    def _update_ensemble_panel(self) -> None:
        if not self.state.phenotyping_results:
            self.ensemble_panel.update_results(None, [])
            return
        methods = list(self.state.phenotyping_results.keys())
        self.ensemble_panel.update_results(self.state.ensemble_df, methods)

    def _show_error(self, title: str, msg: str) -> None:
        QMessageBox.critical(self.control_widget, title, msg)
