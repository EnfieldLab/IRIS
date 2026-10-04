"""
IRIS-Viewer Training Service

Service for filtering and exporting high confidence training sets from
manually annotated expression profiles.
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from qtpy.QtWidgets import QMessageBox, QFileDialog
from napari.qt import thread_worker

from iris_common.io.data_adapter import IrisDataAdapter


class TrainingService:
    """
    Handles the calculation of high-confidence cell sets based on morphological
    and expression-purity filters, and exports the refined set to AnnData or CSV.
    """

    def __init__(self, ctx: types.SimpleNamespace) -> None:
        self.ctx = ctx
        self._layer_name = "High Confidence Training Set"

    def auto_select(self, params: Dict[str, Any]) -> None:
        """Runs the auto selection logic in a thread worker."""
        ctx = self.ctx
        df = ctx.state.current_mapped_df
        adapter = ctx.state._data_adapter
        
        if df is None or df.empty or adapter is None:
            QMessageBox.warning(
                ctx.control_widget, "No Data",
                "Please load an annotation file (master file) and add samples to canvas first."
            )
            ctx.training_panel.reset_select_btn()
            return
            
        if "cell_type" not in df.columns:
            QMessageBox.warning(
                ctx.control_widget, "No Cell Type",
                "The loaded data does not have a 'cell_type' column."
            )
            ctx.training_panel.reset_select_btn()
            return

        # Snapshot for worker
        df_snap = df.copy()
        markers = list(adapter.markers)
        intensity_stat = adapter._intensity_stat
        marker_cols = [f"{m}_{intensity_stat}" for m in markers if f"{m}_{intensity_stat}" in df_snap.columns]
        
        # Get canvas translations for each sample
        translations = {}
        for sample, layers in ctx.state.sample_layers_map.items():
            bbox_lyr = next((l for l in layers if l.name == f"Bounds - {sample}"), None)
            if bbox_lyr is not None:
                translations[sample] = tuple(bbox_lyr.translate)

        @thread_worker
        def _compute():
            filtered = df_snap.copy()
            
            # 1. Morphological Filter
            if "area" in filtered.columns:
                min_a = params.get("min_area", 0)
                max_a = params.get("max_area", 100000)
                filtered = filtered[(filtered["area"] >= min_a) & (filtered["area"] <= max_a)]
                
            if filtered.empty:
                return pd.DataFrame()
                
            # 2. Expression Purity Filter
            # Calculate median profile per cell type
            keep_top_pct = params.get("keep_top_percent", 75) / 100.0
            
            if len(marker_cols) > 0:
                selected_indices = []
                for ctype, group in filtered.groupby("cell_type"):
                    if len(group) == 0:
                        continue
                    
                    # Normalize expression
                    expr = group[marker_cols].values
                    # Add small epsilon to avoid divide by zero
                    norms = np.linalg.norm(expr, axis=1, keepdims=True) + 1e-8
                    expr_norm = expr / norms
                    
                    median_profile = np.median(expr_norm, axis=0)
                    median_norm = np.linalg.norm(median_profile) + 1e-8
                    median_profile = median_profile / median_norm
                    
                    # Cosine similarity
                    sims = np.dot(expr_norm, median_profile)
                    
                    # Top percentile threshold
                    threshold = np.quantile(sims, max(0.0, 1.0 - keep_top_pct))
                    
                    # Keep indices above threshold
                    valid_idx = group.index[sims >= threshold]
                    selected_indices.extend(valid_idx)
                    
                filtered = filtered.loc[selected_indices]
                
            # 3. Calculate Canvas Coordinates
            canvas_y = []
            canvas_x = []
            for _, row in filtered.iterrows():
                sy, sx = translations.get(row.get("Image", ""), (0, 0))
                canvas_y.append(row["Pos_Y"] + sy)
                canvas_x.append(row["Pos_X"] + sx)
                
            filtered["_canvas_y"] = canvas_y
            filtered["_canvas_x"] = canvas_x
            
            return filtered

        def _on_done(result_df: pd.DataFrame):
            ctx.training_panel.reset_select_btn()
            
            if result_df.empty:
                QMessageBox.information(
                    ctx.control_widget, "No Cells",
                    "No cells passed the filtering criteria."
                )
                return
                
            coords = result_df[["_canvas_y", "_canvas_x"]].values
            cell_types = result_df["cell_type"].values
            
            # Remove existing layer if present
            if self._layer_name in ctx.viewer.layers:
                ctx.viewer.layers.remove(self._layer_name)
                
            # Create a point layer with metadata
            pts = ctx.viewer.add_points(
                coords,
                name=self._layer_name,
                size=10,
                face_color="green",
                edge_color="white",
                edge_width=2,
                properties={"cell_type": cell_types, "original_index": result_df.index.values},
                text={
                    "string": "{cell_type}",
                    "color": "white",
                    "size": 8,
                    "anchor": "upper_left",
                    "visible": False
                }
            )
            
            # Add to legend
            ctx.legend_widget.add_ref_row(pts, self._layer_name, "green", layer_type="points")
            
            QMessageBox.information(
                ctx.control_widget, "Selection Complete",
                f"Selected {len(result_df)} high-confidence cells.\n\n"
                "A new Points layer has been added. You can manually Add/Delete points "
                "using the Napari tools before exporting."
            )

        worker = _compute()
        worker.returned.connect(_on_done)
        ctx.active_workers.append(worker)
        worker.start()

    def export(self, fmt_str: str) -> None:
        """Exports the manually refined points back to a training set."""
        ctx = self.ctx
        
        if self._layer_name not in ctx.viewer.layers:
            QMessageBox.warning(
                ctx.control_widget, "No Training Set",
                f"Please generate the '{self._layer_name}' first."
            )
            return
            
        pts_layer = ctx.viewer.layers[self._layer_name]
        if len(pts_layer.data) == 0:
            QMessageBox.warning(
                ctx.control_widget, "Empty Set",
                "The training set is empty."
            )
            return
            
        # We need to map the points back to original cells.
        # Since the user might have deleted points, we only keep the ones that are left.
        # If the user added points, we would need a spatial KDTree to find the nearest cell.
        df = ctx.state.current_mapped_df
        if df is None:
            return
            
        # Get canvas translations for each sample
        translations = {}
        for sample, layers in ctx.state.sample_layers_map.items():
            bbox_lyr = next((l for l in layers if l.name == f"Bounds - {sample}"), None)
            if bbox_lyr is not None:
                translations[sample] = tuple(bbox_lyr.translate)
                
        # Build KDTree of all cells in canvas coordinates
        from scipy.spatial import cKDTree
        all_y = []
        all_x = []
        for _, row in df.iterrows():
            sy, sx = translations.get(row.get("Image", ""), (0, 0))
            all_y.append(row["Pos_Y"] + sy)
            all_x.append(row["Pos_X"] + sx)
            
        tree = cKDTree(np.column_stack((all_y, all_x)))
        
        # Query nearest neighbors for current points
        pts_coords = pts_layer.data
        distances, indices = tree.query(pts_coords)
        
        # Filter out points that are too far from any cell (e.g. > 20 pixels)
        valid_mask = distances < 20.0
        valid_indices = indices[valid_mask]
        
        final_df = df.iloc[valid_indices].copy()
        final_df = final_df.reset_index(drop=True)
        
        if len(final_df) == 0:
            QMessageBox.warning(ctx.control_widget, "Export Failed", "No valid cells found near the points.")
            return

        # Export based on format
        if "AnnData" in fmt_str:
            path, _ = QFileDialog.getSaveFileName(
                ctx.control_widget, "Save AnnData", "", "H5AD (*.h5ad)"
            )
            if path:
                if not path.endswith(".h5ad"):
                    path += ".h5ad"
                try:
                    import anndata as ad
                    # Use adapter to get AnnData format
                    adapter = IrisDataAdapter(final_df, intensity_stat=ctx.state._data_adapter._intensity_stat)
                    adata = adapter.to_anndata()
                    adata.write(path)
                    QMessageBox.information(ctx.control_widget, "Success", f"Saved {len(final_df)} cells to {Path(path).name}")
                except Exception as e:
                    QMessageBox.critical(ctx.control_widget, "Error", str(e))
                    
        elif "CSV" in fmt_str:
            path, _ = QFileDialog.getSaveFileName(
                ctx.control_widget, "Save CSV", "", "CSV (*.csv)"
            )
            if path:
                if not path.endswith(".csv"):
                    path += ".csv"
                try:
                    final_df.to_csv(path, index=False)
                    QMessageBox.information(ctx.control_widget, "Success", f"Saved {len(final_df)} cells to {Path(path).name}")
                except Exception as e:
                    QMessageBox.critical(ctx.control_widget, "Error", str(e))
