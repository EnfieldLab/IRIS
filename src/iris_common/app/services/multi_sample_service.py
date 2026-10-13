"""
iris_common.app.services.multi_sample_service
=============================================
Orchestrates loading of TMA cores / tissue sections onto the napari canvas.

Responsibilities
----------------
- Loading image + mask + annotations for multiple samples
- TMA grid auto-alignment using the ``_rxcx_`` sample-name convention
- Bounding-box construction and per-sample layer synchronisation
- Cell-type phenotype overlay rendering via ``IrisPhenotypeMapper``
- Tissue boundary save / restore

Annotation loading is handled by :class:`AnnotationService`.
Colour file loading is handled by :class:`ColorService`.
All state mutations go through ``self.ctx`` (the shared ``SimpleNamespace``).
Thread-safety: all napari calls happen on the Qt main thread;
pixel-level I/O runs inside a ``thread_worker``.
"""
from __future__ import annotations

import json
import re
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import matplotlib.colors as mcolors
from napari.qt import thread_worker
from napari.utils.colormaps import Colormap
from qtpy.QtWidgets import QFileDialog, QMessageBox

from iris_common.io.data_adapter import IrisDataAdapter, IrisPhenotypeMapper
from iris_common.widgets.styles import DEFAULT_MARKER_COLORS
from iris_common.app.services.annotation_service import AnnotationService
from iris_common.app.services.color_service import ColorService


class MultiSampleService:
    """
    Manages multi-sample canvas operations for the IRIS-Viewer.

    Context requirements (ctx fields accessed)
    -------------------------------------------
    viewer, state, control_widget, legend_widget, comparison_widget
    state.image_dir, state.file_map, state.markers
    state.master_df, state.resolved_dfs, state.masks_map,
    state.sample_layers_map, state.areas_map, state.current_mapped_df,
    state.cytoplasm_mask, state.current_mask, state.persistent_settings
    """

    def __init__(self, ctx: types.SimpleNamespace) -> None:
        self.ctx = ctx
        self._added_master_legends: set = set()
        self._syncing_visibility = False
        self._syncing_colormap   = False
        self._syncing_opacity    = False
        self._syncing_contrast   = False
        # Colour maps — mutated in place by ColorService
        self.CELLTYPE_COLORS: Dict[str, str] = {}
        self.DEFAULT_COLORS: Dict[str, str] = dict(DEFAULT_MARKER_COLORS)
        self._annotation_svc = AnnotationService(ctx)
        self._color_svc = ColorService(ctx, self.DEFAULT_COLORS, self.CELLTYPE_COLORS)

    # ── Master Annotation File ─────────────────────────────────────────────────

    def set_master_file(self, file_path: str) -> None:
        """Load and normalise *file_path* (CSV/XLSX) into ``ctx.state.master_df``."""
        self._annotation_svc.load_annotation_file(file_path)

        # Draw overlays for any samples that were already loaded on the canvas
        self.update_phenotypes_on_canvas()

    def update_phenotypes_on_canvas(self) -> None:
        """
        Scans existing canvas samples and adds phenotype overlays if missing,
        based on the currently loaded master_df and selected reference column.
        """
        ctx = self.ctx
        if ctx.state.master_df is None or ctx.state.master_df.empty:
            return

        cw = getattr(ctx, "control_widget", None)
        ref_col = cw.combo_ref.currentText() if hasattr(cw, "combo_ref") else "cell_type"
        scale_val = cw.spin_scale.value() if hasattr(cw, "spin_scale") else 1.0

        for sample_name, sample_layers in list(ctx.state.sample_layers_map.items()):
            mask = ctx.state.masks_map.get(sample_name)
            if mask is None:
                continue

            sub_df = pd.DataFrame()
            if "Image" in ctx.state.master_df.columns:
                sub_df = ctx.state.master_df[
                    ctx.state.master_df["Image"] == sample_name
                ].copy()
            
            # Fallback: if empty but there's only one sample on canvas, assume the CSV is for this sample
            if sub_df.empty and len(ctx.state.sample_layers_map) == 1:
                sub_df = ctx.state.master_df.copy()

            print(f"[IRIS-Viewer] sample_name={sample_name}, mask is {'None' if mask is None else 'Present'}, sub_df.shape={sub_df.shape}")

            if sub_df.empty:
                print(f"[IRIS-Viewer] sub_df is empty! master_df cols: {ctx.state.master_df.columns.tolist() if ctx.state.master_df is not None else 'None'}")
                if ctx.state.master_df is not None and "Image" in ctx.state.master_df.columns:
                    print(f"[IRIS-Viewer] Unique images in master_df: {ctx.state.master_df['Image'].unique()}")
                continue
                
            # If a bounding box exists, get its translation to align overlays perfectly
            bbox_lyr = next((l for l in sample_layers if l.name == f"Bounds - {sample_name}"), None)
            translation = bbox_lyr.translate if bbox_lyr else (0.0, 0.0)

            # Check and add Centroids if they don't exist yet
            has_centroids = any(l.name == "Centroids" for l in sample_layers)
            if not has_centroids and {"Pos_X", "Pos_Y"}.issubset(sub_df.columns):
                coords = sub_df[["Pos_Y", "Pos_X"]].values
                pts = ctx.viewer.add_points(
                    coords, name="Centroids", size=5,
                    visible=True, scale=(scale_val, scale_val),
                )
                pts.face_color = "white"
                pts.translate = translation
                sample_layers.append(pts)
                self._register_layer(pts, "Centroids")
                if "Centroids" not in self._added_master_legends:
                    ctx.legend_widget.add_ref_row(pts, "Centroids", "#FFFFFF", layer_type="points")
                    self._added_master_legends.add("Centroids")

            # Remove existing phenotype layers for this sample before adding new ones
            base_markers = set(ctx.state.markers) if ctx.state.markers else set()
            layers_to_remove = []
            for l in list(sample_layers):
                l_base = l.name.split(" [")[0]
                if l.__class__.__name__ == 'Image' and l_base not in base_markers:
                    layers_to_remove.append(l)
                    
            for l in layers_to_remove:
                if l in ctx.viewer.layers:
                    ctx.viewer.layers.remove(l)
                if l in sample_layers:
                    sample_layers.remove(l)
                
                l_base = l.name.split(" [")[0]
                if hasattr(ctx, "legend_widget") and hasattr(ctx.legend_widget, "remove_row"):
                    ctx.legend_widget.remove_row(l_base)
                if hasattr(self, "_added_master_legends") and l_base in self._added_master_legends:
                    self._added_master_legends.remove(l_base)
                if hasattr(ctx, "added_legend_entries") and l_base in ctx.added_legend_entries:
                    ctx.added_legend_entries.remove(l_base)

            # Check and add Cell Phenotypes via IrisPhenotypeMapper
            mapper = IrisPhenotypeMapper(mask, sub_df, target_col=ref_col)
            layer_data_ref = mapper.get_masks_by_type()
            
            print(f"[IRIS-Viewer] mapper returned layer_data_ref with keys: {list(layer_data_ref.keys())}")

            for type_name in sorted(layer_data_ref.keys()):
                # Skip if a layer with this phenotype name already exists in this sample's group
                if any(l.name.split(" [")[0] == type_name for l in sample_layers):
                    continue

                mask_bin, hex_color = layer_data_ref[type_name]
                if type_name.lower() in self.CELLTYPE_COLORS:
                    hex_color = self.CELLTYPE_COLORS[type_name.lower()]
                
                colors = np.array([[0, 0, 0, 0], list(mcolors.to_rgb(hex_color)) + [1]])
                lyr = ctx.viewer.add_image(
                    mask_bin, name=type_name, opacity=0.8, visible=True,
                    blending="translucent", colormap=Colormap(colors),
                    contrast_limits=[0, 1], scale=(scale_val, scale_val),
                )
                lyr.translate = translation
                sample_layers.append(lyr)
                self._register_layer(lyr, type_name)
                
                try:
                    actual_rgba = (
                        lyr.colormap.colors[-1]
                        if hasattr(lyr.colormap, "colors") and lyr.colormap.colors is not None
                        else lyr.colormap.map([1.0])[0]
                    )
                    actual_hex = mcolors.to_hex(actual_rgba)
                except Exception:
                    actual_hex = hex_color
                
                if type_name not in self._added_master_legends:
                    ctx.legend_widget.add_ref_ct_row(lyr, type_name, actual_hex)
                    self._added_master_legends.add(type_name)
                    
            # Update the resolved dataframes cache for analytics/export
            ctx.state.resolved_dfs[sample_name] = mapper.df if 'mapper' in locals() else sub_df
            
        # Rebuild the unified mapped DataFrame
        all_dfs = [ctx.state.resolved_dfs[sn] for sn in ctx.state.sample_layers_map.keys() if sn in ctx.state.resolved_dfs]
        if all_dfs:
            ctx.state.current_mapped_df = pd.concat(all_dfs, ignore_index=True)
            if hasattr(ctx, "expression_widget"):
                ctx.expression_widget.set_data(ctx.state.current_mapped_df)

    # ── Add Samples to Canvas ─────────────────────────────────────────────────

    def add_samples_to_canvas(self, sample_names: List[str]) -> None:
        """
        Load image + mask + annotations for each sample in the list and
        add them to the napari canvas.  Runs pixel I/O in a thread worker.
        """
        ctx = self.ctx
        
        # 1. Do not reload samples that are already loaded
        sample_names = [sn for sn in sample_names if sn not in ctx.state.sample_layers_map]
        if not sample_names:
            return
            
        scale_val = ctx.control_widget.spin_scale.value()

        # Snapshot everything the worker needs to avoid Qt calls in the thread
        file_map_snap = dict(ctx.state.file_map)
        master_df_snap = ctx.state.master_df.copy() if ctx.state.master_df is not None else None
        markers_snap   = list(ctx.state.markers)
        ref_col = getattr(ctx.control_widget, "combo_ref", None)
        ref_col_text = ref_col.currentText() if ref_col else "cell_type"

        @thread_worker
        def _load_all():
            results = []
            for sample_name in sample_names:
                img_path = file_map_snap.get(sample_name)
                if img_path is None:
                    continue
                mask_path = img_path.parent / f"{sample_name}_mask.tiff"

                # ── Sub-DataFrame ─────────────────────────────────────────────
                sub_df = pd.DataFrame()
                if master_df_snap is not None and "Image" in master_df_snap.columns:
                    sub_df = master_df_snap[
                        master_df_snap["Image"] == sample_name
                    ].copy()

                # ── Image ────────────────────────────────────────────────────
                img_data = []
                raw = None
                if img_path.exists():
                    raw = tifffile.imread(str(img_path))
                    if raw.ndim == 2:
                        raw = raw[np.newaxis]

                    for i, m in enumerate(markers_snap):
                        if i >= raw.shape[0]:
                            break
                        chan = raw[i]
                        lo = float(np.percentile(chan, 1))
                        hi = float(np.percentile(chan, 99.9))
                        if lo >= hi:
                            hi = lo + 1.0
                        img_data.append((m, chan, [lo, hi]))

                # ── Mask ─────────────────────────────────────────────────────
                mask_channels = []
                if mask_path.exists():
                    mask_raw = tifffile.imread(str(mask_path)).astype(np.uint32)
                    if mask_raw.ndim == 2:
                        mask_channels = [mask_raw]
                    else:
                        mask_channels = [mask_raw[i] for i in range(mask_raw.shape[0])]

                results.append((sample_name, img_data, mask_channels, sub_df, ref_col_text, raw))
            return results

        def _on_loaded(results):
            self._on_samples_loaded(results, sample_names, scale_val)
            if hasattr(ctx, "comparison_widget"):
                ctx.comparison_widget.set_loading_done()

        worker = _load_all()
        worker.returned.connect(_on_loaded)
        ctx.active_workers.append(worker)
        worker.start()

    # ── On-load: build napari layers ──────────────────────────────────────────

    def _on_samples_loaded(
        self,
        loaded_results: list,
        target_sample_names: List[str],
        scale_val: float,
    ) -> None:
        ctx = self.ctx

        for (sample_name, img_data, mask_channels, sub_df, ref_col_text, raw) in loaded_results:
            ctx.state.resolved_dfs[sample_name] = sub_df
            ctx.state.multi_sample_images[sample_name] = raw
            sample_layers = []
            sample_h, sample_w = 2000, 2000

            # ── Image channels ────────────────────────────────────────────────
            for m, chan_data, clims in img_data:
                sample_h, sample_w = chan_data.shape
                cname = self.DEFAULT_COLORS.get(m, "gray")
                is_vis = m in ("DAPI", "CK", "aSMA", "CD45")
                layer = ctx.viewer.add_image(
                    chan_data, name=m, colormap=cname, blending="additive",
                    visible=is_vis, contrast_limits=clims, scale=(scale_val, scale_val),
                )
                sample_layers.append(layer)
                self._register_layer(layer, m)
                if m not in self._added_master_legends:
                    try:
                        init_color = layer.colormap.map([1.0])[0]
                    except Exception:
                        init_color = "#FFFFFF"
                    ctx.legend_widget.add_grouped_channel_row(layer, m, init_color, "Channels")
                    self._added_master_legends.add(m)

            # ── Segmentation masks ────────────────────────────────────────────
            if mask_channels:
                sample_h, sample_w = mask_channels[0].shape
                ctx.state.masks_map[sample_name] = mask_channels[0]
                for i, m_chan in enumerate(mask_channels):
                    lname = (
                        f"Segmentation Ch{i}"
                        if len(mask_channels) > 1
                        else "Segmentation Mask"
                    )
                    try:
                        seg_lyr = ctx.viewer.add_labels(
                            m_chan, name=lname, opacity=1.0,
                            visible=(i == 0), contour=1, scale=(scale_val, scale_val),
                        )
                    except TypeError:
                        seg_lyr = ctx.viewer.add_labels(
                            m_chan, name=lname, opacity=1.0,
                            visible=(i == 0), scale=(scale_val, scale_val),
                        )
                    sample_layers.append(seg_lyr)
                    self._register_layer(seg_lyr, lname)
                    if lname not in self._added_master_legends:
                        ctx.legend_widget.add_ref_row(seg_lyr, lname, "#888888")
                        self._added_master_legends.add(lname)

                if ctx.state.current_mask is None:
                    ctx.state.cytoplasm_mask = (
                        mask_channels[1] if len(mask_channels) > 1 else mask_channels[0]
                    )
                ctx.state.current_mask = mask_channels[0]

            # ── Centroids ─────────────────────────────────────────────────────
            if not sub_df.empty and {"Pos_X", "Pos_Y"}.issubset(sub_df.columns):
                coords = sub_df[["Pos_Y", "Pos_X"]].values
                pts = ctx.viewer.add_points(
                    coords, name="Centroids", size=5,
                    visible=True, scale=(scale_val, scale_val),
                )
                pts.face_color = "white"
                sample_layers.append(pts)
                self._register_layer(pts, "Centroids")
                if "Centroids" not in self._added_master_legends:
                    ctx.legend_widget.add_ref_row(pts, "Centroids", "#FFFFFF", layer_type="points")
                    self._added_master_legends.add("Centroids")

            # ── Cell type overlays ────────────────────────────────────────────
            if not sub_df.empty and mask_channels:
                mapper = IrisPhenotypeMapper(
                    mask_channels[0], sub_df, target_col=ref_col_text
                )
                layer_data_ref = mapper.get_masks_by_type()

                for type_name in sorted(layer_data_ref.keys()):
                    mask_bin, hex_color = layer_data_ref[type_name]
                    if type_name.lower() in self.CELLTYPE_COLORS:
                        hex_color = self.CELLTYPE_COLORS[type_name.lower()]
                    colors = np.array(
                        [[0, 0, 0, 0], list(mcolors.to_rgb(hex_color)) + [1]]
                    )
                    lyr = ctx.viewer.add_image(
                        mask_bin, name=type_name, opacity=0.8, visible=True,
                        blending="translucent", colormap=Colormap(colors),
                        contrast_limits=[0, 1], scale=(scale_val, scale_val),
                    )
                    sample_layers.append(lyr)
                    self._register_layer(lyr, type_name)
                    try:
                        actual_rgba = (
                            lyr.colormap.colors[-1]
                            if hasattr(lyr.colormap, "colors") and lyr.colormap.colors is not None
                            else lyr.colormap.map([1.0])[0]
                        )
                        actual_hex = mcolors.to_hex(actual_rgba)
                    except Exception:
                        actual_hex = hex_color
                    if type_name not in self._added_master_legends:
                        ctx.legend_widget.add_ref_ct_row(lyr, type_name, actual_hex)
                        self._added_master_legends.add(type_name)

            # ── Bounding box ──────────────────────────────────────────────────
            pad = 40
            if not sub_df.empty and {"Pos_X", "Pos_Y"}.issubset(sub_df.columns):
                min_y = sub_df["Pos_Y"].min() - pad
                max_y = sub_df["Pos_Y"].max() + pad
                min_x = sub_df["Pos_X"].min() - pad
                max_x = sub_df["Pos_X"].max() + pad
            elif mask_channels:
                ys, xs = np.nonzero(mask_channels[0])
                if len(ys):
                    min_y, max_y = ys.min() - pad, ys.max() + pad
                    min_x, max_x = xs.min() - pad, xs.max() + pad
                else:
                    min_y, min_x, max_y, max_x = 0, 0, sample_h, sample_w
            else:
                min_y, min_x, max_y, max_x = 0, 0, sample_h, sample_w

            min_y = max(0, min_y)
            min_x = max(0, min_x)

            bbox_verts = np.array([
                [min_y, min_x], [min_y, max_x],
                [max_y, max_x], [max_y, min_x],
            ])
            bbox_name = f"Bounds - {sample_name}"
            bbox_lyr = ctx.viewer.add_shapes(
                [bbox_verts], shape_type="polygon",
                edge_width=5, edge_color="yellow", face_color="transparent",
                name=bbox_name,
                features={"name": [sample_name]},
                text={"string": "{name}", "color": "white",
                      "size": 12, "anchor": "upper_left"},
                scale=(scale_val, scale_val),
            )
            ctx.legend_widget.add_ref_row(
                bbox_lyr, bbox_name, "yellow", layer_type="shapes"
            )
            sample_layers.append(bbox_lyr)
            ctx.state.sample_layers_map[sample_name] = sample_layers

            # Bind bbox translate → sync all sibling layers
            @bbox_lyr.events.translate.connect
            def _on_translate(event, bl=bbox_lyr, sl=sample_layers):
                t = tuple(bl.translate)
                for lyr in sl:
                    if lyr in ctx.viewer.layers and lyr is not bl:
                        lyr.translate = t

            # ── Auto-load saved boundary ──────────────────────────────────────
            boundary_path = ctx.state.file_map.get(sample_name)
            if not boundary_path and ctx.state.image_dir:
                boundary_path = Path(ctx.state.image_dir) / f"{sample_name}_image.tiff"
            if boundary_path:
                self._try_load_saved_boundary(
                    sample_name, boundary_path, mask_channels, scale_val, sample_layers
                )

        # ── Align and finalise ────────────────────────────────────────────────
        self._finalize_canvas(target_sample_names)

    # ── Canvas Alignment ──────────────────────────────────────────────────────

    def align_samples_on_canvas(
        self, sample_names: List[str], gap: Optional[int] = None
    ) -> None:
        """
        Parse _rxcx_ metadata from sample names and translate them onto
        a proportionally spaced grid.

        Samples whose names don't contain _rxcx_ are appended at the bottom.
        """
        ctx = self.ctx
        if gap is None:
            gap = (
                ctx.comparison_widget.get_gap()
                if hasattr(ctx, "comparison_widget")
                else 50
            )

        parsed = []
        unmatched = 0
        for s_name in sample_names:
            layers = ctx.state.sample_layers_map.get(s_name, [])
            bbox_lyr = next(
                (l for l in layers if l.name == f"Bounds - {s_name}"), None
            )
            if bbox_lyr is None or len(bbox_lyr.data) == 0:
                continue

            bbox = bbox_lyr.data[0]
            min_y, min_x = np.min(bbox, axis=0)
            max_y, max_x = np.max(bbox, axis=0)
            sy, sx = bbox_lyr.scale
            h = (max_y - min_y) * sy
            w = (max_x - min_x) * sx
            native_cy = (min_y + max_y) / 2.0
            native_cx = (min_x + max_x) / 2.0

            match = re.search(r"_r(\d+)c(\d+)", s_name, re.IGNORECASE)
            if match:
                r, c = int(match.group(1)), int(match.group(2))
            else:
                r = 999 + unmatched
                c = 0
                unmatched += 1

            parsed.append(dict(
                name=s_name, r=r, c=c, h=h, w=w,
                native_cy=native_cy, native_cx=native_cx,
                scale_y=sy, scale_x=sx, bbox_lyr=bbox_lyr,
            ))

        if not parsed:
            return

        unique_rows = sorted({s["r"] for s in parsed})
        unique_cols = sorted({s["c"] for s in parsed})

        row_heights = {r: max((s["h"] for s in parsed if s["r"] == r), default=0)
                       for r in unique_rows}
        col_widths  = {c: max((s["w"] for s in parsed if s["c"] == c), default=0)
                       for c in unique_cols}

        target_y: Dict[int, float] = {}
        cy = 0.0
        for r in unique_rows:
            target_y[r] = cy + row_heights[r] / 2.0
            cy += row_heights[r] + gap

        target_x: Dict[int, float] = {}
        cx = 0.0
        for c in unique_cols:
            target_x[c] = cx + col_widths[c] / 2.0
            cx += col_widths[c] + gap

        for s in parsed:
            ty = target_y[s["r"]] - s["native_cy"] * s["scale_y"]
            tx = target_x[s["c"]] - s["native_cx"] * s["scale_x"]
            s["bbox_lyr"].translate = (ty, tx)

            # Explicitly sync all sibling layers for this sample to guarantee exact alignment
            layers = ctx.state.sample_layers_map.get(s["name"], [])
            for lyr in layers:
                if lyr in ctx.viewer.layers and lyr is not s["bbox_lyr"]:
                    lyr.translate = (ty, tx)

    # ── Clear Canvas ──────────────────────────────────────────────────────────

    def clear_canvas(self) -> None:
        """Remove all layers and legend entries, then reset per-sample state."""
        ctx = self.ctx
        ctx.viewer.layers.clear()
        ctx.legend_widget.clear()
        self._added_master_legends.clear()
        ctx.state.sample_layers_map.clear()
        if hasattr(ctx.state, "multi_sample_images"):
            ctx.state.multi_sample_images.clear()
        ctx.state.reset_sample_state()
        ctx.state.areas_map.clear()
        ctx.state.current_mapped_df = None

    # ── Finalise canvas ───────────────────────────────────────────────────────

    def _finalize_canvas(self, sample_names: List[str]) -> None:
        ctx = self.ctx
        all_loaded = list(ctx.state.sample_layers_map.keys())
        self.align_samples_on_canvas(all_loaded)

        all_dfs = []
        for sn in sample_names:
            if hasattr(ctx.state, "resolved_dfs") and sn in ctx.state.resolved_dfs:
                df = ctx.state.resolved_dfs[sn]
            elif ctx.state.master_df is not None:
                df = ctx.state.master_df[
                    ctx.state.master_df.get("Image", pd.Series()) == sn
                ].copy() if "Image" in ctx.state.master_df.columns else pd.DataFrame()
            else:
                df = pd.DataFrame()
            if not df.empty:
                all_dfs.append(df)

        if all_dfs:
            ctx.state.current_mapped_df = pd.concat(all_dfs, ignore_index=True)
            if hasattr(ctx, "expression_widget"):
                ctx.expression_widget.set_data(ctx.state.current_mapped_df)

    # ── Saved boundary ────────────────────────────────────────────────────────

    def _try_load_saved_boundary(
        self, sample_name, img_path, mask_channels, scale_val, sample_layers
    ) -> None:
        ctx = self.ctx
        boundary_json = Path(img_path).parent / f"{sample_name}_boundary.json"
        if not boundary_json.exists():
            return
        try:
            with open(boundary_json, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            hull_verts = np.array(saved["vertices"])
            lname = f"Boundary - {sample_name}"
            bounds_lyr = next(
                (l for l in ctx.viewer.layers if l.name == f"Bounds - {sample_name}"), None
            )
            t_lyr = ctx.viewer.add_shapes(
                [hull_verts], shape_type="polygon", name=lname,
                edge_width=2, edge_color="white", face_color="white",
                opacity=0.5, scale=(scale_val, scale_val),
            )
            if bounds_lyr is not None:
                t_lyr.translate = bounds_lyr.translate
            ctx.legend_widget.add_ref_row(t_lyr, lname, "#FFFFFF", layer_type="shapes")
            if t_lyr not in sample_layers:
                sample_layers.append(t_lyr)

            if mask_channels:
                m_shape = (
                    (mask_channels[0].shape[-2], mask_channels[0].shape[-1])
                    if mask_channels[0].ndim >= 2
                    else mask_channels[0].shape
                )
                from iris_common.engine.geometry import recompute_area_from_vertices
                area_val, _ = recompute_area_from_vertices(hull_verts, m_shape, scale_val)
                ctx.state.areas_map[sample_name] = area_val

                @t_lyr.events.data.connect
                def _on_polygon_change(event):
                    if len(t_lyr.data) > 0:
                        s_val = ctx.control_widget.spin_scale.value()
                        new_a, _ = recompute_area_from_vertices(t_lyr.data[0], m_shape, s_val)
                        ctx.state.areas_map[sample_name] = new_a
        except Exception as exc:
            print(f"[IRIS-Viewer] Failed to load boundary for {sample_name}: {exc}")

    # ── Layer registration & sync ─────────────────────────────────────────────

    def _register_layer(self, layer, base_name: str) -> None:
        """Attach visibility / colormap / opacity / contrast sync callbacks."""
        ctx = self.ctx
        settings = ctx.state.persistent_settings

        if base_name not in settings:
            settings[base_name] = {"visible": layer.visible, "opacity": layer.opacity}
            if hasattr(layer, "colormap"):
                settings[base_name]["colormap"] = layer.colormap
            if hasattr(layer, "contrast_limits"):
                settings[base_name]["contrast_limits"] = layer.contrast_limits
        else:
            cfg = settings[base_name]
            if "visible" in cfg:
                layer.visible = cfg["visible"]
            if "opacity" in cfg:
                layer.opacity = cfg["opacity"]
            if "colormap" in cfg and hasattr(layer, "colormap"):
                try:
                    layer.colormap = cfg["colormap"]
                except Exception:
                    pass
            if "contrast_limits" in cfg and hasattr(layer, "contrast_limits"):
                try:
                    layer.contrast_limits = cfg["contrast_limits"]
                except Exception:
                    pass

        @layer.events.visible.connect
        def _vis(event):
            if self._syncing_visibility:
                return
            self._syncing_visibility = True
            try:
                is_vis = layer.visible
                settings.setdefault(base_name, {})["visible"] = is_vis
                for l in ctx.viewer.layers:
                    lb = l.name.split(" [")[0] if " [" in l.name else l.name
                    if lb == base_name and l.visible != is_vis:
                        l.visible = is_vis
                self._sync_legend_checkbox(base_name, is_vis)
            finally:
                self._syncing_visibility = False

        if hasattr(layer.events, "colormap"):
            @layer.events.colormap.connect
            def _cmap(event):
                if self._syncing_colormap:
                    return
                self._syncing_colormap = True
                try:
                    cmap = layer.colormap
                    settings.setdefault(base_name, {})["colormap"] = cmap
                    for l in ctx.viewer.layers:
                        lb = l.name.split(" [")[0] if " [" in l.name else l.name
                        if lb == base_name and l is not layer and hasattr(l, "colormap"):
                            l.colormap = cmap
                finally:
                    self._syncing_colormap = False

        if hasattr(layer.events, "opacity"):
            @layer.events.opacity.connect
            def _opac(event):
                if self._syncing_opacity:
                    return
                self._syncing_opacity = True
                try:
                    op = layer.opacity
                    settings.setdefault(base_name, {})["opacity"] = op
                    for l in ctx.viewer.layers:
                        lb = l.name.split(" [")[0] if " [" in l.name else l.name
                        if lb == base_name and l is not layer:
                            l.opacity = op
                finally:
                    self._syncing_opacity = False

        if hasattr(layer.events, "contrast_limits"):
            @layer.events.contrast_limits.connect
            def _clim(event):
                if self._syncing_contrast:
                    return
                self._syncing_contrast = True
                try:
                    clims = layer.contrast_limits
                    settings.setdefault(base_name, {})["contrast_limits"] = clims
                    for l in ctx.viewer.layers:
                        lb = l.name.split(" [")[0] if " [" in l.name else l.name
                        if lb == base_name and l is not layer and hasattr(l, "contrast_limits"):
                            l.contrast_limits = clims
                except Exception:
                    pass
                finally:
                    self._syncing_contrast = False

    def _sync_legend_checkbox(self, base_name: str, is_vis: bool) -> None:
        lw = self.ctx.legend_widget
        for r_list in [
            getattr(lw, "ref_rows", []),
            getattr(lw, "ref_ct_rows", []),
            getattr(lw, "cn_rows", []),
        ]:
            for row in r_list:
                if row.name == base_name:
                    row.checkbox.blockSignals(True)
                    row.checkbox.setChecked(is_vis)
                    row.checkbox.blockSignals(False)
        if hasattr(lw, "custom_channel_groups") and "Channels" in lw.custom_channel_groups:
            _, _, row_list = lw.custom_channel_groups["Channels"]
            for row in row_list:
                if row.name == base_name:
                    row.checkbox.blockSignals(True)
                    row.checkbox.setChecked(is_vis)
                    row.checkbox.blockSignals(False)
                    break

    # ── Colour loading ────────────────────────────────────────────────────────

    def load_channel_colors(self, filepath: str) -> None:
        """Load marker colour mapping from *filepath* and apply to the canvas."""
        self._color_svc.load_channel_colors(filepath)

    def load_celltype_colors(self, filepath: str) -> None:
        """Load cell-type colour mapping from *filepath* and apply to the canvas."""
        self._color_svc.load_celltype_colors(filepath)
