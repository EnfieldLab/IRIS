"""
IRIS-Viewer Image Service
Handles image directory discovery, sample loading, multi-channel display,
companion mask auto-loading, channel-colour CSV loading, and scale updates.
All methods receive a shared ``ctx`` namespace from the controller.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors

from napari.qt import thread_worker
from qtpy.QtWidgets import QMessageBox

from iris_viewer.io_utils.image_loaders import (
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

    # ── Directory & Sample Loading ────────────────────────────────────────

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
            ctx.segmentation_panel.update_channel_dropdowns(ctx.state.markers)

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
        ctx.state.reset_sample_state()
        ctx.state.current_sample_name = sample_name
        img_path = ctx.state.file_map[sample_name]

        @thread_worker
        def _load() -> tuple:
            return load_multichannel_tiff(img_path)

        worker = _load()
        worker.returned.connect(lambda res: self._on_image_loaded(res, sample_name))
        ctx.active_workers.append(worker)
        worker.start()

    def _on_image_loaded(self, result: tuple, sample_name: str) -> None:
        """Add multi-channel layers to the viewer after the dask array is loaded."""
        ctx = self.ctx
        raw_image, n_channels = result
        ctx.state.raw_image = raw_image
        scale_val = ctx.control_widget.spin_scale.value()

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

            cname = DEFAULT_MARKER_COLORS.get(marker.upper(), 'gray')
            is_vis = marker.upper() in {
                'DAPI', 'HOECHST', 'CK', 'PANCK', 'ASMA', 'SMA', 'CD45'
            }

            layer = ctx.viewer.add_image(
                chan_data,
                name=marker,
                colormap=cname,
                blending='additive',
                visible=is_vis,
                contrast_limits=clims,
                scale=(scale_val, scale_val),
            )

            if marker not in ctx.added_legend_entries:
                try:
                    init_color = layer.colormap.map([1.0])[0]
                except Exception:
                    init_color = '#FFFFFF'
                ctx.legend_widget.add_channel_row(layer, marker, init_color)
                ctx.added_legend_entries.add(marker)

        # Auto-load companion mask if it exists alongside the image
        img_path = ctx.state.file_map[sample_name]
        mask_path = img_path.with_name(f"{sample_name}_mask.tiff")
        if mask_path.exists():
            self.load_and_display_mask(mask_path, sample_name, scale_val)

        h, w = raw_image.shape[-2], raw_image.shape[-1]
        ctx.control_widget.update_dashboard(
            image_dims=f"{w} × {h}",
            num_channels=str(n_channels),
        )

    # ── Mask Loading ───────────────────────────────────────────────────────

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
                )
            except TypeError:
                layer = ctx.viewer.add_labels(
                    data, name=name, opacity=1.0, visible=True,
                    scale=(scale_val, scale_val),
                )
            if name not in ctx.added_legend_entries:
                ctx.legend_widget.add_segmentation_row(layer, name, '#888888')
                ctx.added_legend_entries.add(name)
            return layer

        if mask_data.ndim == 3:
            for ch in range(mask_data.shape[0]):
                lname = f"Loaded Mask Ch{ch}" if mask_data.shape[0] > 1 else "Loaded Mask"
                lyr = _add_labels(mask_data[ch], lname)
                if ch > 0:
                    lyr.visible = False
            ctx.state.segmentation_results["Loaded"] = {
                'nuclear': mask_data[0],
                'cytoplasm': mask_data[1] if mask_data.shape[0] > 1 else None,
            }
        else:
            _add_labels(mask_data, "Loaded Mask")
            ctx.state.segmentation_results["Loaded"] = {
                'nuclear': mask_data, 'cytoplasm': None,
            }

        n_cells = int(np.unique(mask_data).size) - 1
        ctx.control_widget.update_dashboard(
            seg_status="Loaded from file",
            total_cells=f"{n_cells:,}",
        )

    # ── Channel Colours ────────────────────────────────────────────────────

    def load_channel_colors(self, filepath: str) -> None:
        """Import a marker→colour mapping from a CSV/XLSX and update legend rows."""
        ctx = self.ctx
        try:
            import pandas as pd
            path_obj = Path(filepath)
            df = (
                pd.read_excel(filepath)
                if path_obj.suffix.lower() == '.xlsx'
                else pd.read_csv(filepath)
            )
            df.columns = [str(c).lower().strip() for c in df.columns]

            if 'markers' not in df.columns or 'colors' not in df.columns:
                QMessageBox.warning(
                    ctx.control_widget, "Invalid Format",
                    "The file must contain 'markers' and 'colors' columns."
                )
                return

            color_map: dict = {}
            for _, row in df.iterrows():
                m = str(row['markers']).strip().lower()
                c = str(row['colors']).strip()
                if pd.notna(row['markers']) and pd.notna(row['colors']) and mcolors.is_color_like(c):
                    color_map[m] = mcolors.to_hex(c)

            updated = 0
            for marker in ctx.state.markers:
                if marker.lower() in color_map:
                    new_hex = color_map[marker.lower()]
                    for row in getattr(ctx.legend_widget, 'channel_rows', []):
                        if row.name.strip().lower() == marker.lower():
                            row.update_swatch_display(new_hex)
                            row.colorChanged.emit(row.name, new_hex)
                            updated += 1
                            break

            QMessageBox.information(
                ctx.control_widget, "Colours Updated",
                f"Updated {updated} marker colour(s)."
            )
        except Exception as exc:
            QMessageBox.critical(
                ctx.control_widget, "Error",
                f"Failed to load marker colours:\n{exc}"
            )

    # ── Scale ──────────────────────────────────────────────────────────────

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
