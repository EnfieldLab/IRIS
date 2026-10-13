"""
IRIS Common Neighborhood Service
================================
Coordinates spatial cellular neighborhood analysis, asynchronous worker execution,
napari canvas overlay coloration, and legend synchronization.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from napari.qt.threading import thread_worker
from napari.utils.colormaps import Colormap
from qtpy.QtWidgets import QMessageBox

from iris_common.engine.neighborhood import identify_cellular_neighborhoods
from iris_common.app.services.multi_sample_service import get_sample_sub_df


class NeighborhoodService:
    """Service managing cellular neighborhood profiling and canvas visualization."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.last_cn_k: int = 5
        self.last_cn_method: str = "KMeans"
        self.last_total_cn: int = 5

    def execute_niche_analysis(
        self, k_neighbors: int, n_niches: int, method: str, resolution: float
    ) -> None:
        """Triggers asynchronous spatial neighborhood clustering across all cells."""
        ctx = self.ctx
        df = None
        # Prioritize master_df so KMeans runs through all cells in the annotation file
        if ctx.state.master_df is not None and not ctx.state.master_df.empty:
            df = ctx.state.master_df
        elif ctx.state.current_mapped_df is not None and not ctx.state.current_mapped_df.empty:
            df = ctx.state.current_mapped_df
        elif hasattr(ctx.state, "features_df") and ctx.state.features_df is not None and not ctx.state.features_df.empty:
            df = ctx.state.features_df
        else:
            QMessageBox.warning(
                ctx.control_widget,
                "No Data",
                "Load sample data and cell annotations before extracting neighborhoods.",
            )
            return

        ref_col = ctx.control_widget.combo_ref.currentText()
        if not ref_col or ref_col not in df.columns:
            for fallback in ("cell_type", "Verified_CellType", "CellType", "GROUP", "Phenotype", "Label"):
                if fallback in df.columns:
                    ref_col = fallback
                    break
        if not ref_col or ref_col not in df.columns:
            QMessageBox.warning(
                ctx.control_widget,
                "Invalid Phenotype Column",
                f"Cell phenotype column '{ref_col}' was not found in annotation data.",
            )
            return

        scale_val = ctx.control_widget.spin_scale.value()
        n_jobs = -1

        widget = getattr(ctx, "neighborhood_widget", None)
        if widget is not None:
            widget.btn_niche.setText("Processing...")
            widget.btn_niche.setEnabled(False)

        self.last_cn_k = k_neighbors
        self.last_cn_method = method
        self.last_total_cn = n_niches

        @thread_worker
        def _niche_worker():
            return identify_cellular_neighborhoods(
                df=df,
                cell_type_col=ref_col,
                k_neighbors=k_neighbors,
                n_niches=n_niches,
                microns_per_pixel=scale_val,
                method=method,
                resolution=resolution,
                n_jobs=n_jobs,
            )

        worker = _niche_worker()
        worker.returned.connect(self._on_niche_complete)
        worker.errored.connect(self._on_niche_error)
        if hasattr(ctx, "_active_workers"):
            ctx._active_workers.append(worker)
        worker.start()

    def _on_niche_complete(self, result: tuple) -> None:
        """Handles completion of neighborhood computation on main Qt thread."""
        ctx = self.ctx
        updated_df, centers, labels = result

        # 1. Propagate clusters across all cells in master_df
        ctx.state.master_df = updated_df

        # 2. Update current_mapped_df for loaded canvas samples
        loaded_samples = list(ctx.state.sample_layers_map.keys())
        if loaded_samples and "Image" in updated_df.columns:
            matched = []
            for sn in loaded_samples:
                sub = get_sample_sub_df(updated_df, sn)
                if not sub.empty:
                    matched.append(sub)
            if matched:
                ctx.state.current_mapped_df = pd.concat(matched, ignore_index=True)
            else:
                ctx.state.current_mapped_df = updated_df
        else:
            ctx.state.current_mapped_df = updated_df

        # 3. Store into resolved_dfs for each loaded sample
        if hasattr(ctx.state, "resolved_dfs"):
            for sname in list(ctx.state.sample_layers_map.keys()):
                sub = get_sample_sub_df(updated_df, sname)
                if not sub.empty:
                    ctx.state.resolved_dfs[sname] = sub

        total_cn = centers.shape[0]
        widget = getattr(ctx, "neighborhood_widget", None)
        if widget is not None:
            widget.initialize_metadata_table(total_cn)
            widget.display_niche_compositions(centers, labels, updated_df)
            self.update_niche_canvas_coloration(widget.niche_metadata, total_cn)
            widget.btn_niche.setText("Extract Spatial Neighborhoods")
            widget.btn_niche.setEnabled(True)

    def _on_niche_error(self, exc: Exception) -> None:
        """Handles background calculation error."""
        ctx = self.ctx
        widget = getattr(ctx, "neighborhood_widget", None)
        if widget is not None:
            widget.btn_niche.setText("Extract Spatial Neighborhoods")
            widget.btn_niche.setEnabled(True)
        QMessageBox.critical(
            ctx.control_widget,
            "Neighborhood Analysis Error",
            f"An error occurred during neighborhood extraction: {str(exc)}",
        )

    def update_niche_canvas_coloration(
        self, meta_dict: Dict[int, Dict[str, str]], total_cn: Optional[int] = None
    ) -> None:
        """Creates or refreshes napari overlay layers for each identified niche."""
        ctx = self.ctx
        if ctx.state.current_mapped_df is None:
            return

        scale_val = ctx.control_widget.spin_scale.value()
        k_neighbors = self.last_cn_k
        method = self.last_cn_method
        if total_cn is None:
            total_cn = len(meta_dict)

        current_param_prefix = f"{method} - {k_neighbors} KNN - {total_cn} CNs - "
        flush_prefix = f"{method} - {k_neighbors} KNN - "

        # Remove previous niche layers from napari
        layers_to_flush = [l for l in ctx.viewer.layers if l.name.startswith(flush_prefix)]
        for layer in layers_to_flush:
            ctx.viewer.layers.remove(layer)

        for s_name, s_layers in ctx.state.sample_layers_map.items():
            ctx.state.sample_layers_map[s_name] = [
                l for l in s_layers if not l.name.startswith(flush_prefix)
            ]

        # Flush legend rows
        if hasattr(ctx.legend_widget, "cn_rows"):
            rows_to_flush = [
                row for row in ctx.legend_widget.cn_rows if row.name.startswith(flush_prefix)
            ]
            for row in rows_to_flush:
                ctx.legend_widget.cn_rows.remove(row)
                ctx.legend_widget.cn_layout.removeWidget(row)
                row.deleteLater()

        sample_names = list(ctx.state.sample_layers_map.keys())
        if not sample_names and ctx.state.current_sample_name:
            sample_names = [ctx.state.current_sample_name]

        for sample_name in sample_names:
            mask = ctx.state.masks_map.get(sample_name)
            if mask is None:
                mask = ctx.state.current_mask
            if mask is None:
                continue

            sub_df = get_sample_sub_df(ctx.state.current_mapped_df, sample_name)
            if sub_df.empty and ctx.state.master_df is not None:
                sub_df = get_sample_sub_df(ctx.state.master_df, sample_name)
            if sub_df.empty and len(sample_names) == 1:
                sub_df = ctx.state.current_mapped_df

            if sub_df.empty or 'Cellular_Neighborhood' not in sub_df.columns:
                continue

            for idx, item in meta_dict.items():
                niche_name = item["name"]
                hex_color = item["color"]
                layer_title = f"{current_param_prefix}{niche_name}"

                # Extract cell IDs for this niche (matching by index, name, ID column, or string)
                cond = (
                    (sub_df['Cellular_Neighborhood'] == idx) |
                    (sub_df['Cellular_Neighborhood'] == f"Niche {idx}") |
                    (sub_df['Cellular_Neighborhood'] == niche_name) |
                    (sub_df['Cellular_Neighborhood'].astype(str) == str(idx))
                )
                if 'Cellular_Neighborhood_ID' in sub_df.columns:
                    cond = cond | (sub_df['Cellular_Neighborhood_ID'] == idx)
                niche_cells = sub_df[cond]
                if niche_cells.empty:
                    continue

                if 'ObjectNumber' in niche_cells.columns:
                    niche_objs = pd.to_numeric(niche_cells['ObjectNumber'], errors='coerce').fillna(0).astype(np.int64).values
                else:
                    niche_objs = niche_cells.index.values + 1

                niche_objs = niche_objs[niche_objs > 0]
                if len(niche_objs) == 0:
                    continue

                niche_mask = np.isin(mask, niche_objs).astype(np.uint8)
                if not np.any(niche_mask):
                    continue

                colors = np.array([[0, 0, 0, 0], list(mcolors.to_rgb(hex_color)) + [1]])
                layer = ctx.viewer.add_image(
                    niche_mask,
                    name=layer_title,
                    opacity=0.5,
                    blending='translucent',
                    colormap=Colormap(colors),
                    contrast_limits=[0, 1],
                    scale=(scale_val, scale_val),
                )

                bounds_layer = next(
                    (l for l in ctx.viewer.layers if l.name == f"Bounds - {sample_name}"), None
                )
                if bounds_layer:
                    layer.translate = bounds_layer.translate

                if sample_name in ctx.state.sample_layers_map:
                    ctx.state.sample_layers_map[sample_name].append(layer)

                if hasattr(ctx, "multi_sample_svc"):
                    ctx.multi_sample_svc._register_layer(layer, layer_title)

                if hasattr(ctx.legend_widget, "cn_rows"):
                    already_in_legend = any(
                        row.name == layer_title for row in ctx.legend_widget.cn_rows
                    )
                    if not already_in_legend:
                        ctx.legend_widget.add_cn_row(
                            layer, layer_title, hex_color, layer_type='image'
                        )
