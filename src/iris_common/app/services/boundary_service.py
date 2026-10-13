"""
IRIS Common Boundary Service
============================
Orchestrates tissue boundary auto-segmentation, manual vertex adjustment,
polygon visualization in napari, and sidecar JSON persistence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from qtpy.QtWidgets import QMessageBox

from iris_common.engine.geometry import (
    compute_tissue_boundary_mask,
    recompute_area_from_vertices,
)


class BoundaryService:
    """Service handling tissue boundary segmentation and persistence."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx

    def auto_segment_boundary(self) -> None:
        """Computes convex-hull tissue boundaries across all active samples."""
        ctx = self.ctx
        df = ctx.state.current_mapped_df
        if df is None or df.empty:
            if ctx.state.master_df is not None and not ctx.state.master_df.empty:
                df = ctx.state.master_df
            elif hasattr(ctx.state, "features_df") and ctx.state.features_df is not None and not ctx.state.features_df.empty:
                df = ctx.state.features_df
            else:
                QMessageBox.warning(
                    ctx.control_widget,
                    "No Sample Loaded",
                    "Load image data and annotations before computing tissue boundaries.",
                )
                return

        scale_val = ctx.control_widget.spin_scale.value()
        samples = list(ctx.state.sample_layers_map.keys())
        if not samples and ctx.state.current_sample_name:
            samples = [ctx.state.current_sample_name]

        if not samples:
            QMessageBox.warning(
                ctx.control_widget,
                "No Sample Active",
                "Please load at least one sample onto the canvas.",
            )
            return

        segmented_count = 0
        for sample_name in samples:
            sub_df = pd.DataFrame()
            if hasattr(ctx.state, "resolved_dfs") and sample_name in ctx.state.resolved_dfs:
                sub_df = ctx.state.resolved_dfs[sample_name]
            elif df is not None and not df.empty:
                if 'Image' in df.columns:
                    sub_df = df[df['Image'] == sample_name]
                else:
                    sub_df = df

            mask = ctx.state.masks_map.get(sample_name)
            if mask is None:
                mask = ctx.state.current_mask
            if mask is None or sub_df.empty:
                continue

            try:
                m_shape = (mask.shape[-2], mask.shape[-1]) if mask.ndim >= 2 else mask.shape
                hull_vertices, _, area_val, area_text = compute_tissue_boundary_mask(
                    sub_df, m_shape, scale_val
                )
                ctx.state.areas_map[sample_name] = area_val

                layer_name = f"Boundary - {sample_name}"
                bounds_layer = next(
                    (l for l in ctx.viewer.layers if l.name == f"Bounds - {sample_name}"), None
                )

                if layer_name in ctx.viewer.layers:
                    t_layer = ctx.viewer.layers[layer_name]
                    t_layer.data = [hull_vertices]
                    t_layer.scale = (scale_val, scale_val)
                    if bounds_layer:
                        t_layer.translate = bounds_layer.translate
                else:
                    t_layer = ctx.viewer.add_shapes(
                        [hull_vertices],
                        shape_type='polygon',
                        name=layer_name,
                        edge_width=2,
                        edge_color='white',
                        face_color='white',
                        opacity=0.5,
                        scale=(scale_val, scale_val),
                    )
                    if bounds_layer:
                        t_layer.translate = bounds_layer.translate

                    ctx.legend_widget.add_ref_row(
                        t_layer, layer_name, "#FFFFFF", layer_type='shapes'
                    )

                    # Dynamic recomputation on vertex edit
                    self._bind_boundary_change_event(t_layer, sample_name, m_shape)

                if sample_name in ctx.state.sample_layers_map:
                    if t_layer not in ctx.state.sample_layers_map[sample_name]:
                        ctx.state.sample_layers_map[sample_name].append(t_layer)

                segmented_count += 1
            except Exception as exc:
                print(f"[IRIS-Viewer] Boundary computation error for {sample_name}: {exc}")

        if segmented_count > 0:
            print(f"[IRIS-Viewer] Auto-segmented tissue boundary for {segmented_count} sample(s).")
        else:
            QMessageBox.warning(
                ctx.control_widget,
                "Boundary Generation Incomplete",
                "Unable to generate boundary. Verify cell coordinates and mask availability.",
            )

    def _bind_boundary_change_event(self, t_layer, sample_name: str, mask_shape: tuple) -> None:
        """Connects a callback to update areas_map when the polygon is edited in napari."""
        ctx = self.ctx

        @t_layer.events.data.connect
        def _on_polygon_data_change(event):
            if len(t_layer.data) > 0:
                vertices = t_layer.data[0]
                scale_val = ctx.control_widget.spin_scale.value()
                new_area, _ = recompute_area_from_vertices(vertices, mask_shape, scale_val)
                ctx.state.areas_map[sample_name] = new_area

    def save_sample_boundaries(self) -> None:
        """Serializes tissue boundary polygon vertices to {sample_name}_boundary.json."""
        ctx = self.ctx
        target_dir = ctx.state.image_dir
        if not target_dir and ctx.state.file_map:
            first_path = next(iter(ctx.state.file_map.values()), None)
            if first_path:
                target_dir = first_path.parent

        if not target_dir:
            QMessageBox.warning(
                ctx.control_widget,
                "No Directory",
                "No image directory is currently loaded.",
            )
            return

        target_dir = Path(target_dir)
        saved_count = 0

        # Check all active sample layers
        sample_dict = ctx.state.sample_layers_map
        if not sample_dict and ctx.state.current_sample_name:
            sample_dict = {ctx.state.current_sample_name: list(ctx.viewer.layers)}

        for sample_name, layers in sample_dict.items():
            boundary_layer = next(
                (l for l in layers if l.name == f"Boundary - {sample_name}"), None
            )
            if boundary_layer is None:
                # Also check directly in viewer layers
                boundary_layer = next(
                    (l for l in ctx.viewer.layers if l.name == f"Boundary - {sample_name}"), None
                )

            if boundary_layer and len(boundary_layer.data) > 0:
                vertices = np.asarray(boundary_layer.data[0]).tolist()
                scale_val = (
                    boundary_layer.scale[0]
                    if hasattr(boundary_layer, 'scale') and len(boundary_layer.scale) > 0
                    else 1.0
                )
                out_dir = target_dir
                if getattr(ctx.state, "file_map", None) and sample_name in ctx.state.file_map:
                    out_dir = Path(ctx.state.file_map[sample_name]).parent
                out_path = Path(out_dir) / f"{sample_name}_boundary.json"
                try:
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump(
                            {
                                'sample_name': sample_name,
                                'vertices': vertices,
                                'scale': float(scale_val),
                            },
                            f,
                            indent=4,
                        )
                    saved_count += 1
                except Exception as exc:
                    print(f"[IRIS-Viewer] Failed saving boundary for {sample_name}: {exc}")

        if saved_count > 0:
            QMessageBox.information(
                ctx.control_widget,
                "Success",
                f"Saved {saved_count} tissue boundary file(s) to image folder.",
            )
        else:
            QMessageBox.warning(
                ctx.control_widget,
                "No Boundaries",
                "No tissue boundaries found on canvas. Run 'Auto-Segment Boundary' first.",
            )
