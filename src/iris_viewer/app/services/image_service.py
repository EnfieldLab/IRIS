"""
IRIS-Viewer Image Service
Handles image directory discovery, sample loading, multi-channel display,
companion mask auto-loading, channel-colour CSV loading, and scale updates.
All methods receive a shared ``ctx`` namespace from the controller.
"""
from __future__ import annotations

import types
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
from napari.qt import thread_worker
from napari.utils.colormaps import Colormap
from qtpy.QtWidgets import QMessageBox

from iris_viewer.app.services.color_service import ColorService
from iris_viewer.io.loaders import (
    discover_image_files,
    load_existing_mask,
    load_multichannel_tiff,
    probe_channel_count,
)
from iris_viewer.widgets.config_dialogs import ViewerPanelConfigDialog
from iris_viewer.widgets.styles import DEFAULT_MARKER_COLORS


class ImageService:
    """
    Responsible for all image-related operations:
    directory discovery, sample selection, multi-channel layer creation,
    existing-mask loading, channel-colour CSV import, and scale management.

    Uses a shared ``ctx`` namespace so that every state field and widget
    reference is accessed from a single, authoritative location.
    """

    def __init__(self, ctx: types.SimpleNamespace) -> None:
        self.ctx = ctx

    def get_marker_color(self, marker: str) -> str | Colormap:
        """Resolve hex or named colour for a marker, respecting user overrides."""
        ctx = self.ctx
        if hasattr(ctx, "color_svc"):
            return ctx.color_svc.get_marker_color(marker)
        if hasattr(ctx, "multi_sample_svc"):
            return ctx.multi_sample_svc.get_marker_color(marker)
        if hasattr(ctx, "state") and hasattr(ctx.state, "channel_colors"):
            cc = ctx.state.channel_colors
            for k in (marker, marker.lower(), marker.upper()):
                if k in cc:
                    return cc[k]
        if hasattr(ctx, "state") and hasattr(ctx.state, "persistent_settings"):
            for k in (marker, marker.lower(), marker.upper()):
                if k in ctx.state.persistent_settings and "colormap" in ctx.state.persistent_settings[k]:
                    cmap = ctx.state.persistent_settings[k]["colormap"]
                    if hasattr(cmap, "colors") and len(cmap.colors) > 0:
                        try:
                            return mcolors.to_hex(cmap.colors[-1][:3])
                        except Exception:
                            pass
                    return cmap
        return DEFAULT_MARKER_COLORS.get(marker.upper(), 'gray')

    # Directory & Sample Loading

    def on_load_directory(self, dir_path: str) -> None:
        """Discover images and prompt the user to assign marker channels."""
        ctx = self.ctx
        ctx.state.image_dir = Path(dir_path)
        ctx.state.file_map = discover_image_files(ctx.state.image_dir)

        if not ctx.state.file_map:
            QMessageBox.warning(
                ctx.control_widget, "No Images Found",
                f"No TIFF images found in:\n{dir_path}\n\n"
                "Expected files matching *_image.tiff or *.tiff / *.tif."
            )
            return

        sample_names = sorted(ctx.state.file_map.keys())

        # Only re-run the channel assignment dialog when markers are not yet set
        if not ctx.state.markers:
            first_path = ctx.state.file_map[sample_names[0]]
            num_channels = probe_channel_count(first_path)
            dialog = ViewerPanelConfigDialog(num_channels, parent=ctx.control_widget)
            if not dialog.exec_():
                QMessageBox.warning(
                    ctx.control_widget, "Cancelled",
                    "Channel assignment was cancelled.  No images loaded."
                )
                return

            ctx.state.markers = dialog.marker_panel
            if hasattr(ctx, 'segmentation_panel'):
                ctx.segmentation_panel.update_channel_dropdowns(ctx.state.markers)
            if hasattr(ctx, 'expression_widget'):
                ctx.expression_widget.set_data(markers=ctx.state.markers)
            if hasattr(ctx, 'phenotyper_panel'):
                ctx.phenotyper_panel.set_data(markers=ctx.state.markers)

        ctx.control_widget.update_sample_list(sample_names)
        if hasattr(ctx, 'comparison_widget'):
            ctx.comparison_widget.update_sample_list(sample_names)

        if sample_names:
            ctx.control_widget.combo_sample.setCurrentIndex(0)
            self.on_sample_selected(sample_names[0])

    def on_sample_selected(self, sample_name: str) -> None:
        """Clear the current view and asynchronously load the selected sample."""
        ctx = self.ctx
        if not sample_name or sample_name not in ctx.state.file_map:
            return

        ctx.viewer.layers.clear()
        ctx.legend_widget.clear()
        ctx.added_legend_entries.clear()
        if hasattr(ctx, "multi_sample_svc") and hasattr(ctx.multi_sample_svc, "_added_master_legends"):
            ctx.multi_sample_svc._added_master_legends.clear()
        ctx.state.sample_layers_map.clear()
        ctx.state.reset_sample_state()
        ctx.state.current_sample_name = sample_name
        img_path = ctx.state.file_map[sample_name]

        @thread_worker
        def _load() -> tuple:
            return load_multichannel_tiff(img_path)

        worker = _load()
        worker.returned.connect(lambda res: self._on_image_loaded(res, sample_name))
        if hasattr(ctx, "track_worker"):
            ctx.track_worker(
                worker,
                on_error=lambda exc: self._on_image_load_error(exc, sample_name),
                error_title="Image Load Error",
                error_prefix=f"Failed to load image for sample '{sample_name}'",
            )
        else:
            worker.errored.connect(lambda exc: self._on_image_load_error(exc, sample_name))
            if hasattr(ctx, "active_workers"):
                ctx.active_workers.append(worker)
        worker.start()

    def _on_image_load_error(self, exc: Exception, sample_name: str) -> None:
        """Handle image loading failure gracefully with user feedback."""
        ctx = self.ctx
        print(f"[Image Error] Failed to load sample '{sample_name}': {exc}")
        if hasattr(ctx, "control_widget"):
            QMessageBox.critical(
                ctx.control_widget,
                "Image Load Error",
                f"Failed to load image for sample '{sample_name}':\n{exc}"
            )

    def _on_image_loaded(self, result: tuple, sample_name: str) -> None:
        """Add multi-channel layers to the viewer after the dask array is loaded."""
        ctx = self.ctx
        raw_image, n_channels = result
        ctx.state.raw_image = raw_image
        scale_val = ctx.control_widget.spin_scale.value()

        if sample_name not in ctx.state.sample_layers_map:
            ctx.state.sample_layers_map[sample_name] = []

        for i, marker in enumerate(ctx.state.markers):
            if i >= raw_image.shape[0]:
                break

            chan_data = raw_image[i]

            # Compute contrast limits from a sub-sampled view (avoids computing
            # the full dask array) and cache them in state for later reuse.
            step_h = max(1, chan_data.shape[0] // 500)
            step_w = max(1, chan_data.shape[1] // 500)
            small_chan = chan_data[::step_h, ::step_w]
            if hasattr(small_chan, 'compute'):
                small_chan = small_chan.compute()

            lo = float(np.percentile(small_chan, 1))
            hi = float(np.percentile(small_chan, 99.9))
            if lo >= hi:
                hi = lo + 1.0
            clims = [lo, hi]
            ctx.state.channel_contrast_limits[marker] = clims

            cname = self.get_marker_color(marker)
            is_vis = marker.upper() in {
                'DAPI', 'HOECHST', 'CK', 'PANCK', 'ASMA', 'SMA', 'CD45'
            }

            if isinstance(cname, Colormap):
                cmap = cname
            elif isinstance(cname, str) and (
                cname.startswith("#")
                or (mcolors.is_color_like(cname) and cname.lower() not in ('gray', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow'))
            ):
                try:
                    rgb = list(mcolors.to_rgb(cname))
                    cmap = Colormap(np.array([[0, 0, 0, 0], rgb + [1.0]]), name=f"custom_{marker}")
                except Exception:
                    cmap = cname
            else:
                cmap = cname

            layer = ctx.viewer.add_image(
                chan_data,
                name=marker,
                colormap=cmap,
                blending='additive',
                visible=is_vis,
                contrast_limits=clims,
                scale=(scale_val, scale_val),
                metadata={"sample_name": sample_name, "marker": marker, "layer_type": "channel"},
            )
            ctx.state.sample_layers_map[sample_name].append(layer)

            # Persist layer colormap so it survives sample switching
            if hasattr(ctx.state, "persistent_settings"):
                ctx.state.persistent_settings.setdefault(marker, {})["colormap"] = layer.colormap
                ctx.state.persistent_settings.setdefault(marker.lower(), {})["colormap"] = layer.colormap
                ctx.state.persistent_settings.setdefault(marker.upper(), {})["colormap"] = layer.colormap

            if marker not in ctx.added_legend_entries:
                if isinstance(cname, str) and (cname.startswith("#") or mcolors.is_color_like(cname)):
                    init_color = mcolors.to_hex(cname)
                else:
                    try:
                        mapped = layer.colormap.map([1.0])[0]
                        init_color = mcolors.to_hex(mapped) if not isinstance(mapped, str) or not mapped.startswith("#") else mapped
                    except Exception:
                        init_color = '#FFFFFF'
                ctx.legend_widget.add_channel_row(layer, marker, init_color)
                ctx.added_legend_entries.add(marker)

        # Auto-load companion mask if it exists alongside the image
        img_path = ctx.state.file_map[sample_name]
        mask_path = img_path.with_name(f"{sample_name}_mask.tiff")
        if mask_path.exists():
            self.load_and_display_mask(mask_path, sample_name, scale_val)

        # Auto-load companion tissue boundary if it exists
        boundary_path = img_path.with_name(f"{sample_name}_boundary.json")
        if boundary_path.exists():
            try:
                import json
                with open(boundary_path, "r", encoding="utf-8") as fh:
                    saved = json.load(fh)
                hull_verts = np.array(saved["vertices"])
                lname = f"Boundary - {sample_name}"
                t_lyr = ctx.viewer.add_shapes(
                    [hull_verts], shape_type="polygon", name=lname,
                    edge_width=2, edge_color="white", face_color="white",
                    opacity=0.5, scale=(scale_val, scale_val),
                    metadata={"sample_name": sample_name, "layer_type": "tissue_boundary"},
                )
                ctx.legend_widget.add_ref_row(t_lyr, lname, "#FFFFFF", layer_type="shapes")
                if sample_name not in ctx.state.sample_layers_map:
                    ctx.state.sample_layers_map[sample_name] = []
                ctx.state.sample_layers_map[sample_name].append(t_lyr)

                if ctx.state.current_mask is not None:
                    m_shape = ctx.state.current_mask.shape
                    from iris_viewer.engine.geometry import recompute_area_from_vertices
                    area_val, _ = recompute_area_from_vertices(hull_verts, m_shape, scale_val)
                    ctx.state.areas_map[sample_name] = area_val

                    @t_lyr.events.data.connect
                    def _on_single_polygon_change(event):
                        if len(t_lyr.data) > 0:
                            s_val = ctx.control_widget.spin_scale.value()
                            new_a, _ = recompute_area_from_vertices(t_lyr.data[0], m_shape, s_val)
                            ctx.state.areas_map[sample_name] = new_a
            except Exception as exc:
                print(f"[IRIS-Viewer] Failed to load boundary for {sample_name}: {exc}")

        h, w = raw_image.shape[-2], raw_image.shape[-1]
        ctx.control_widget.update_dashboard(
            image_dims=f"{w} × {h}",
            num_channels=str(n_channels),
        )

        if ctx.state.master_df is not None and hasattr(ctx, "multi_sample_svc"):
            ctx.multi_sample_svc.update_phenotypes_on_canvas()
        elif hasattr(ctx, "expression_widget"):
            cw = getattr(ctx, "control_widget", None)
            ref_col = cw.combo_ref.currentText() if cw and hasattr(cw, "combo_ref") else "cell_type"
            active_df = (
                ctx.state.master_df
                if (ctx.state.master_df is not None and not ctx.state.master_df.empty)
                else ctx.state.current_mapped_df
            )
            ctx.expression_widget.set_data(
                df=active_df,
                markers=ctx.state.markers,
                celltype_col=ref_col,
                selected_samples=[sample_name],
            )

    # Mask Loading

    def on_load_existing_mask(self, filepath: str) -> None:
        """Public slot — load a mask chosen by the user via the file dialog."""
        ctx = self.ctx
        scale_val = ctx.control_widget.spin_scale.value()
        sample_name = ctx.state.current_sample_name or "External"
        self.load_and_display_mask(Path(filepath), sample_name, scale_val)

    def load_and_display_mask(
        self, mask_path: Path, sample_name: str, scale_val: float
    ) -> None:
        """Load a mask TIFF and add it as a Labels layer (or layers for multi-channel)."""
        ctx = self.ctx
        mask_data = load_existing_mask(mask_path)
        if mask_data is None:
            return

        def _add_labels(data: np.ndarray, name: str):
            try:
                layer = ctx.viewer.add_labels(
                    data, name=name, opacity=1.0, visible=True,
                    contour=1, scale=(scale_val, scale_val),
                    metadata={"sample_name": sample_name, "segmentation_name": name, "layer_type": "segmentation"},
                )
            except TypeError:
                layer = ctx.viewer.add_labels(
                    data, name=name, opacity=1.0, visible=True,
                    scale=(scale_val, scale_val),
                    metadata={"sample_name": sample_name, "segmentation_name": name, "layer_type": "segmentation"},
                )
            if sample_name not in ctx.state.sample_layers_map:
                ctx.state.sample_layers_map[sample_name] = []
            ctx.state.sample_layers_map[sample_name].append(layer)
            if name not in ctx.added_legend_entries:
                ctx.legend_widget.add_segmentation_row(layer, name, '#888888')
                ctx.added_legend_entries.add(name)
            return layer

        if mask_data.ndim == 3:
            for ch in range(mask_data.shape[0]):
                if ch == 0:
                    lname = "Nuclear Segmentation"
                elif ch == 1:
                    lname = "Cytoplasm Segmentation"
                else:
                    lname = f"Segmentation Ch{ch}"
                lyr = _add_labels(mask_data[ch], lname)
                if ch > 0:
                    lyr.visible = False
            ctx.state.segmentation_results["Loaded"] = {
                'nuclear': mask_data[0],
                'cytoplasm': mask_data[1] if mask_data.shape[0] > 1 else None,
            }
            ctx.state.current_mask = mask_data[0]
            ctx.state.masks_map[sample_name] = mask_data[0]
        else:
            _add_labels(mask_data, "Nuclear Segmentation")
            ctx.state.segmentation_results["Loaded"] = {
                'nuclear': mask_data, 'cytoplasm': None,
            }
            ctx.state.current_mask = mask_data
            ctx.state.masks_map[sample_name] = mask_data

        n_cells = int(np.unique(mask_data).size) - 1
        ctx.control_widget.update_dashboard(
            seg_status="Loaded from file",
            total_cells=f"{n_cells:,}",
        )

    # Channel Colours

    def load_channel_colors(self, filepath: str) -> None:
        """Import a marker→colour mapping from CSV/XLSX, updating state, legend, and canvas."""
        if hasattr(self.ctx, "color_svc"):
            self.ctx.color_svc.load_channel_colors(filepath)
        elif hasattr(self.ctx, "multi_sample_svc"):
            self.ctx.multi_sample_svc.load_channel_colors(filepath)
        else:
            ColorService(self.ctx).load_channel_colors(filepath)

    # Scale

    def update_scale(self, scale_val: float) -> None:
        """Apply a new physical scale (µm/px) to all existing viewer layers."""
        ctx = self.ctx
        try:
            ctx.viewer.scale_bar.visible = False
            ctx.viewer.scale_bar.unit = "µm" if scale_val != 1.0 else "px"
        except AttributeError:
            pass

        for layer in ctx.viewer.layers:
            if hasattr(layer, 'scale'):
                layer.scale = (scale_val, scale_val)
