"""
IRIS-Viewer Multi-Sample Service

Drives:
- Loading multiple TMA cores or tissue sections onto the napari canvas
- TMA grid auto-alignment using _rxcx_ sample name convention
- Bounding box construction and per-sample layer synchronisation
- Master annotation file ingestion (xlsx/csv) via IrisDataAdapter
- Cell type overlay rendering using IrisPhenotypeMapper
- Tissue boundary save/restore
- Batch statistics export

All state mutations go through self.ctx (the shared SimpleNamespace).
Thread-safety: all napari calls happen on the Qt main thread; only
pixel-level computation runs in thread_workers.
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
        # CELLTYPE_COLORS: user-overridable via load_celltype_colors()
        self.CELLTYPE_COLORS: Dict[str, str] = {}
        self.DEFAULT_COLORS: Dict[str, str] = dict(DEFAULT_MARKER_COLORS)

    # ── Master Annotation File ─────────────────────────────────────────────────

    def set_master_file(self, file_path: str) -> None:
        """
        Load an xlsx/csv annotation file through IrisDataAdapter and
        store the normalised DataFrame in state.master_df.

        Columns expected (auto-detected):
          voron_x, voron_y, GROUP, DAPI…Ki67, area, elongation, …
        """
        ctx = self.ctx
        try:
            adapter = IrisDataAdapter.from_file(file_path)
        except Exception as exc:
            QMessageBox.critical(
                ctx.control_widget, "Annotation Load Error",
                f"Failed to parse annotation file:\n{exc}"
            )
            return

        ctx.state.master_df = adapter.normalised
        ctx.state._data_adapter = adapter          # stash for phenotyper re-use

        # Populate marker list from detected markers (if viewer hasn't set them yet)
        if not ctx.state.markers and adapter.markers:
            ctx.state.markers = adapter.markers
            ctx.segmentation_panel.update_channel_dropdowns(ctx.state.markers)

        # Populate cell type reference combo
        if "cell_type" in adapter.normalised.columns:
            unique_types = sorted(
                adapter.normalised["cell_type"].dropna().unique().tolist()
            )
            cw = ctx.control_widget
            if hasattr(cw, "combo_ref"):
                cols = adapter.normalised.columns.tolist()
                cw.combo_ref.blockSignals(True)
                cw.combo_ref.clear()
                cw.combo_ref.addItems(cols)
                cw.combo_ref.setCurrentText("cell_type")
                cw.combo_ref.blockSignals(False)
            if hasattr(cw, "combo_quick_type"):
                cw.combo_quick_type.blockSignals(True)
                cw.combo_quick_type.clear()
                cw.combo_quick_type.addItems(unique_types)
                cw.combo_quick_type.blockSignals(False)

        if hasattr(ctx, "control_widget") and hasattr(ctx.control_widget, "lbl_annotation"):
            ctx.control_widget.lbl_annotation.setText(
                Path(file_path).name
            )

        QMessageBox.information(
            ctx.control_widget, "Annotation Loaded",
            f"Loaded {len(adapter.normalised):,} cells from {Path(file_path).name}.\n"
            f"Markers detected: {', '.join(adapter.markers[:8])}{'…' if len(adapter.markers) > 8 else ''}\n"
            f"Samples: {adapter.normalised['Image'].nunique():,}"
            if "Image" in adapter.normalised.columns else ""
        )

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

                # Determine crop bounds from sub_df
                crop_y_min, crop_y_max = None, None
                crop_x_min, crop_x_max = None, None
                pad = 20
                if not sub_df.empty:
                    crop_y_min = max(0, int(sub_df["Pos_Y"].min()) - pad)
                    crop_y_max = int(sub_df["Pos_Y"].max()) + pad
                    crop_x_min = max(0, int(sub_df["Pos_X"].min()) - pad)
                    crop_x_max = int(sub_df["Pos_X"].max()) + pad

                # ── Image ────────────────────────────────────────────────────
                img_data = []
                if img_path.exists():
                    raw = tifffile.imread(str(img_path))
                    if raw.ndim == 2:
                        raw = raw[np.newaxis]
                    
                    if crop_y_min is None:
                        crop_y_min, crop_x_min = 0, 0
                        crop_y_max, crop_x_max = raw.shape[-2], raw.shape[-1]
                    else:
                        crop_y_max = min(raw.shape[-2], crop_y_max)
                        crop_x_max = min(raw.shape[-1], crop_x_max)

                    # Crop
                    raw = raw[..., crop_y_min:crop_y_max, crop_x_min:crop_x_max]

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
                        mask_raw = mask_raw[np.newaxis]
                    
                    if crop_y_min is not None and crop_y_max is not None:
                        mask_raw = mask_raw[..., crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                        
                    mask_channels = [mask_raw[i] for i in range(mask_raw.shape[0])]

                # ── Shift Coordinates ─────────────────────────────────────────
                if not sub_df.empty and crop_y_min is not None:
                    sub_df["Pos_Y"] -= crop_y_min
                    sub_df["Pos_X"] -= crop_x_min

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
            if s_name in ctx.state.resolved_dfs and not ctx.state.resolved_dfs[s_name].empty:
                df = ctx.state.resolved_dfs[s_name]
                native_cy = df["Pos_Y"].mean()
                native_cx = df["Pos_X"].mean()
            else:
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

    # ── Clear Canvas ──────────────────────────────────────────────────────────

    def clear_canvas(self) -> None:
        ctx = self.ctx
        ctx.viewer.layers.clear()
        ctx.legend_widget.clear()
        self._added_master_legends.clear()
        ctx.state.sample_layers_map.clear()
        ctx.state.reset_sample_state()
        if hasattr(ctx.state, "resolved_dfs"):
            ctx.state.resolved_dfs.clear()
        if hasattr(ctx.state, "multi_sample_images"):
            ctx.state.multi_sample_images.clear()
        if hasattr(ctx.state, "areas_map"):
            ctx.state.areas_map.clear()
        if hasattr(ctx.state, "current_mapped_df"):
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
        boundary_json = img_path.parent / f"{sample_name}_boundary.json"
        if not boundary_json.exists():
            return
        try:
            with open(boundary_json) as fh:
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
            sample_layers.append(t_lyr)
        except Exception as exc:
            print(f"[MultiSampleService] Failed to load boundary for {sample_name}: {exc}")

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
        ctx = self.ctx
        try:
            import pandas as pd
            df = (
                pd.read_excel(filepath)
                if Path(filepath).suffix.lower() in (".xlsx", ".xls")
                else pd.read_csv(filepath)
            )
            df.columns = [str(c).lower().strip() for c in df.columns]
            if "markers" not in df.columns or "colors" not in df.columns:
                QMessageBox.warning(ctx.control_widget, "Invalid Format",
                                    "File must contain 'markers' and 'colors' columns.")
                return
            color_map = {
                str(r["markers"]).strip().lower(): mcolors.to_hex(str(r["colors"]).strip())
                for _, r in df.iterrows()
                if pd.notna(r["markers"]) and mcolors.is_color_like(str(r["colors"]).strip())
            }
            for m in ctx.state.markers:
                if m.lower() in color_map:
                    self.DEFAULT_COLORS[m] = color_map[m.lower()]
            updated = 0
            lw = ctx.legend_widget
            if hasattr(lw, "custom_channel_groups") and "Channels" in lw.custom_channel_groups:
                _, _, row_list = lw.custom_channel_groups["Channels"]
                for row in row_list:
                    if row.name.lower() in color_map:
                        c = color_map[row.name.lower()]
                        row.update_swatch_display(c)
                        row.colorChanged.emit(row.name, c)
                        updated += 1
            QMessageBox.information(ctx.control_widget, "Success",
                                    f"Updated {updated} marker colors.")
        except Exception as exc:
            QMessageBox.critical(ctx.control_widget, "Error",
                                 f"Failed to load channel colors:\n{exc}")

    def load_celltype_colors(self, filepath: str) -> None:
        ctx = self.ctx
        try:
            import pandas as pd
            df = (
                pd.read_excel(filepath)
                if Path(filepath).suffix.lower() in (".xlsx", ".xls")
                else pd.read_csv(filepath)
            )
            df.columns = [str(c).lower().strip() for c in df.columns]
            if "celltypes" not in df.columns or "colors" not in df.columns:
                return
            for _, row in df.iterrows():
                ct = str(row["celltypes"]).strip().lower()
                cv = str(row["colors"]).strip()
                if mcolors.is_color_like(cv):
                    self.CELLTYPE_COLORS[ct] = mcolors.to_hex(cv)
            updated = 0
            for row in getattr(ctx.legend_widget, "ref_ct_rows", []):
                if row.name.lower() in self.CELLTYPE_COLORS:
                    c = self.CELLTYPE_COLORS[row.name.lower()]
                    row.update_swatch_display(c)
                    row.colorChanged.emit(row.name, c)
                    updated += 1
            QMessageBox.information(ctx.control_widget, "Success",
                                    f"Updated {updated} cell type colors.")
        except Exception as exc:
            QMessageBox.critical(ctx.control_widget, "Error",
                                 f"Failed to load cell type colors:\n{exc}")
