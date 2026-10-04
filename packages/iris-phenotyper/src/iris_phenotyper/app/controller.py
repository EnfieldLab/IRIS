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

from iris_common.io import export_features_csv
from iris_common.types import CellTypePanel, MarkerPanel

from iris_phenotyper.core.state import PhenotyperAppState
from iris_phenotyper.engine.ensemble import PhenotypingEnsemble
from iris_phenotyper.engine.phenotyping.astir_backend import ASTIRBackend
from iris_phenotyper.engine.phenotyping.celesta_backend import CELESTABackend
from iris_phenotyper.widgets.control import PhenotyperControlWidget
from iris_phenotyper.widgets.marker_type_editor import MarkerTypeEditorWidget
from iris_phenotyper.widgets.phenotyping_panel import PhenotypingPanelWidget
from iris_phenotyper.widgets.ensemble_panel import EnsemblePanelWidget


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
        self._backends = {"ASTIR": ASTIRBackend(), "CELESTA": CELESTABackend()}

        # ── Viewer ────────────────────────────────────────────────────────────
        self.viewer = napari.Viewer(title="IRIS-Phenotyper")

        # ── Widgets ───────────────────────────────────────────────────────────
        self.control_widget    = PhenotyperControlWidget(viewer=self.viewer)
        self.marker_editor     = MarkerTypeEditorWidget()
        self.phenotyping_panel = PhenotypingPanelWidget()
        self.ensemble_panel    = EnsemblePanelWidget()

        for w, name in [
            (self.control_widget,    "Control"),
            (self.marker_editor,     "Marker Panel"),
            (self.phenotyping_panel, "Phenotyping"),
            (self.ensemble_panel,    "Ensemble"),
        ]:
            self.viewer.window.add_dock_widget(w, name=name, area="right")

        dw = self.viewer.window.dock_widgets
        qt = self.viewer.window._qt_window
        qt.tabifyDockWidget(dw["Control"].parent(),    dw["Marker Panel"].parent())
        qt.tabifyDockWidget(dw["Marker Panel"].parent(), dw["Phenotyping"].parent())
        qt.tabifyDockWidget(dw["Phenotyping"].parent(), dw["Ensemble"].parent())
        dw["Control"].parent().raise_()

        self._bind_signals()

    # ── Signals ───────────────────────────────────────────────────────────────

    def _bind_signals(self) -> None:
        cw = self.control_widget
        cw.loadFeaturesTriggered.connect(self._load_features)
        cw.exportResultsTriggered.connect(self._export_results)
        if hasattr(cw, 'saveSessionTriggered'):
            cw.saveSessionTriggered.connect(self._save_session)

        self.phenotyping_panel.runRequested.connect(self._run_phenotyping)
        self.phenotyping_panel.cancelRequested.connect(self._cancel)

        self.ensemble_panel.cellOverrideRequested.connect(self._apply_override)
        self.ensemble_panel.undoRequested.connect(self._undo_override)
        self.ensemble_panel.showUMAPRequested.connect(self._run_umap)

    # ── Features Loading ──────────────────────────────────────────────────────

    def _load_features(self, filepath: str) -> None:
        import pandas as pd
        try:
            df = pd.read_csv(filepath)
        except Exception as exc:
            QMessageBox.critical(self.control_widget, "Error", f"Failed to load CSV:\n{exc}")
            return

        self.state.features_df = df

        # Auto-detect markers from column names
        suffix = "_mean"
        markers = [c.replace(suffix, '') for c in df.columns if c.endswith(suffix)]
        self.state.marker_panel = MarkerPanel(markers=markers, panel_file=Path(filepath))
        self.marker_editor.set_available_markers(markers)

        n = len(df)
        self.control_widget.update_dashboard(
            total_cells=f"{n:,}",
            features_path=filepath,
        )
        QMessageBox.information(
            self.control_widget, "Loaded",
            f"Loaded {n:,} cells with {len(markers)} marker channels."
        )

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

        # Snapshot everything needed by the worker
        features_snapshot = self.state.features_df.copy()
        marker_panel = self.state.marker_panel
        backends = dict(self._backends)

        @thread_worker
        def _run():
            results = {}
            for method_name, params in method_configs.items():
                backend = backends.get(method_name)
                if backend is None or not backend.is_available():
                    # CELESTA fallback: use the pure-Python rule-based method
                    if method_name == "CELESTA":
                        backend = backends.get("CELESTA")
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
                                print(f"[Phenotyper] CELESTA fallback failed: {exc}")
                    print(f"[Phenotyper] {method_name} not available — skipping")
                    continue

                try:
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

            return results, ensemble_cfg

        worker = _run()
        worker.returned.connect(self._on_phenotyping_complete)
        self._active_workers.append(worker)
        worker.start()

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
