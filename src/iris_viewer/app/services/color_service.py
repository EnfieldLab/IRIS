"""
iris_viewer.app.services.color_service
=======================================
Handles loading user-supplied colour mapping files and applying them
to marker channel layers and cell-type phenotype layers on the canvas.

Separated from ``MultiSampleService`` to keep each module focused.
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Optional

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from napari.utils.colormaps import Colormap
from qtpy.QtWidgets import QMessageBox

from iris_viewer.core.layers import get_layer_base_name
from iris_viewer.widgets.styles import DEFAULT_MARKER_COLORS


class ColorService:
    """
    Loads CSV/XLSX colour maps and applies them to the napari canvas and Legend.
    Provides centralized colour resolution for marker channels and cell-type phenotypes.

    Parameters
    ----------
    ctx :
        Application context namespace exposing ``viewer``, ``state``,
        ``control_widget``, and ``legend_widget``.
    default_colors :
        Mutable dict mapping marker names to default hex colours (optional;
        defaults to a copy of ``DEFAULT_MARKER_COLORS``).
    celltype_colors :
        Mutable dict mapping cell-type names (lowercase) to hex colours
        (optional; defaults to an empty dict).
    """

    def __init__(
        self,
        ctx: types.SimpleNamespace,
        default_colors: Optional[dict] = None,
        celltype_colors: Optional[dict] = None,
    ) -> None:
        self.ctx = ctx
        self.default_colors: dict = (
            default_colors if default_colors is not None else dict(DEFAULT_MARKER_COLORS)
        )
        self.celltype_colors: dict = (
            celltype_colors if celltype_colors is not None else {}
        )

    # Colour Resolution & Colormap Helpers

    def get_marker_color(self, marker: str) -> str | Colormap:
        """Resolve hex or named colour for a marker, respecting user overrides."""
        ctx = self.ctx
        # 1. State channel colors
        if hasattr(ctx, "state") and hasattr(ctx.state, "channel_colors"):
            cc = ctx.state.channel_colors
            for k in (marker, marker.lower(), marker.upper()):
                if k in cc:
                    return cc[k]
        # 2. default_colors dict
        for k in (marker, marker.lower(), marker.upper()):
            if k in self.default_colors:
                return self.default_colors[k]
        # 3. Persistent settings
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
        # 4. Fallback default
        return DEFAULT_MARKER_COLORS.get(marker.upper(), "gray")

    def get_layer_colormap(self, color_val: Any, name: str) -> Any:
        """Convert a hex string or Colormap into a valid napari colormap."""
        if isinstance(color_val, Colormap):
            return color_val
        if isinstance(color_val, str):
            if color_val.startswith("#") or (
                mcolors.is_color_like(color_val)
                and color_val.lower() not in ("gray", "red", "green", "blue", "cyan", "magenta", "yellow")
            ):
                try:
                    rgb = list(mcolors.to_rgb(color_val))
                    colors = np.array([[0, 0, 0, 0], rgb + [1.0]])
                    return Colormap(colors, name=f"custom_{name}")
                except Exception:
                    pass
        return color_val

    def get_celltype_color(self, cell_type: str, fallback: str = "gray") -> str:
        """Resolve hex colour for a cell type, respecting user overrides."""
        ctx = self.ctx
        if hasattr(ctx, "state") and hasattr(ctx.state, "celltype_colors"):
            ctc = ctx.state.celltype_colors
            for k in (cell_type, cell_type.lower()):
                if k in ctc:
                    return ctc[k]
        for k in (cell_type, cell_type.lower()):
            if k in self.celltype_colors:
                return self.celltype_colors[k]
        return fallback

    # Channel / Marker colours

    def load_channel_colors(self, filepath: str, show_dialog: bool = True) -> None:
        """
        Load a marker → colour mapping from *filepath* (CSV or XLSX).
        Matching legend rows and canvas layers are updated in place,
        and settings are persisted to state for multi-sample viewing.
        """
        ctx = self.ctx
        try:
            import re
            df = self._read_color_file(filepath)
            if df is None:
                return

            col_norm = {re.sub(r'[^a-z0-9]', '', str(c).lower()): c for c in df.columns}
            m_col = next((col_norm[k] for k in ["markers", "marker", "channel", "channels", "target", "targets", "name"] if k in col_norm), None)
            c_col = next((col_norm[k] for k in ["colors", "color", "colours", "colour", "hex", "code"] if k in col_norm), None)

            if m_col is None or c_col is None:
                if show_dialog:
                    QMessageBox.warning(
                        ctx.control_widget, "Invalid Format",
                        f"File must contain a marker column (e.g. 'markers', 'marker') and a color column (e.g. 'colors', 'color').\nFound columns: {list(df.columns)}",
                    )
                return

            color_map: dict[str, str] = {}
            for _, row in df.iterrows():
                val_m = row[m_col]
                val_c = row[c_col]
                if pd.notna(val_m) and pd.notna(val_c):
                    m = str(val_m).strip()
                    c = str(val_c).strip()
                    if mcolors.is_color_like(c):
                        hex_val = mcolors.to_hex(c)
                        color_map[m.lower()] = hex_val
                        color_map[m] = hex_val
                        color_map[m.upper()] = hex_val

            # 1. Update shared DEFAULT_COLORS dict
            for m_key, hex_c in color_map.items():
                self.default_colors[m_key] = hex_c

            # 2. Update ctx.state.channel_colors and persistent_settings
            if hasattr(ctx, "state"):
                if hasattr(ctx.state, "channel_colors"):
                    ctx.state.channel_colors.update(color_map)
                if hasattr(ctx.state, "markers"):
                    for orig_m in ctx.state.markers:
                        if orig_m.lower() in color_map:
                            self.default_colors[orig_m] = color_map[orig_m.lower()]
                if hasattr(ctx.state, "persistent_settings"):
                    for m_key, hex_c in color_map.items():
                        colors = np.array([[0, 0, 0, 0], list(mcolors.to_rgb(hex_c)) + [1]])
                        cmap = Colormap(colors, name=f"custom_{m_key}")
                        ctx.state.persistent_settings.setdefault(m_key, {})["colormap"] = cmap

            # 3. Update legend rows
            updated = 0
            lw = getattr(ctx, "legend_widget", None)
            if lw:
                row_lists = [getattr(lw, "channel_rows", [])]
                if hasattr(lw, "custom_channel_groups") and "Channels" in lw.custom_channel_groups:
                    _, _, custom_rows = lw.custom_channel_groups["Channels"]
                    row_lists.append(custom_rows)

                for r_list in row_lists:
                    for row in r_list:
                        r_name = getattr(row, "name", "")
                        if r_name.lower() in color_map:
                            c = color_map[r_name.lower()]
                            row.update_swatch_display(c)
                            row.colorChanged.emit(row.name, c)
                            updated += 1

            # 4. Update any existing viewer layers
            for layer in ctx.viewer.layers:
                layer_base = get_layer_base_name(layer)
                if layer_base.lower() in color_map and hasattr(layer, "colormap"):
                    c = color_map[layer_base.lower()]
                    self._apply_colormap_to_layer(layer, c)

            if show_dialog:
                unique_markers = len({k.lower() for k in color_map.keys()})
                QMessageBox.information(
                    ctx.control_widget, "Success",
                    f"Updated {unique_markers} marker color(s).",
                )
        except Exception as exc:
            if show_dialog:
                QMessageBox.critical(
                    ctx.control_widget, "Error",
                    f"Failed to load channel colors:\n{exc}",
                )

    # Cell-type colours

    def load_celltype_colors(self, filepath: str, show_dialog: bool = True) -> None:
        """
        Load a cell-type → colour mapping from *filepath* (CSV or XLSX).
        Matching legend rows and canvas layers are updated in place,
        and settings are persisted to state for multi-sample viewing.
        """
        ctx = self.ctx
        try:
            import re
            df = self._read_color_file(filepath)
            if df is None:
                return

            col_norm = {re.sub(r'[^a-z0-9]', '', str(c).lower()): c for c in df.columns}
            ct_col = next((col_norm[k] for k in ["celltypes", "celltype", "phenotype", "phenotypes", "label", "labels", "cluster", "clusters"] if k in col_norm), None)
            c_col = next((col_norm[k] for k in ["colors", "color", "colours", "colour", "hex", "code"] if k in col_norm), None)

            if ct_col is None or c_col is None:
                if show_dialog:
                    QMessageBox.warning(
                        ctx.control_widget, "Invalid Format",
                        f"File must contain a cell type column (e.g. 'cell_type', 'celltypes', 'phenotype') and a color column (e.g. 'colors', 'color', 'hex').\nFound columns: {list(df.columns)}",
                    )
                return

            color_map: dict[str, str] = {}
            for _, row in df.iterrows():
                val_ct = row[ct_col]
                val_c = row[c_col]
                if pd.notna(val_ct) and pd.notna(val_c):
                    ct = str(val_ct).strip()
                    cv = str(val_c).strip()
                    if mcolors.is_color_like(cv):
                        hex_val = mcolors.to_hex(cv)
                        self.celltype_colors[ct.lower()] = hex_val
                        self.celltype_colors[ct] = hex_val
                        color_map[ct.lower()] = hex_val
                        color_map[ct] = hex_val

            # Persist to state
            if hasattr(ctx, "state"):
                if hasattr(ctx.state, "celltype_colors"):
                    ctx.state.celltype_colors.update(color_map)
                if hasattr(ctx.state, "persistent_settings"):
                    for ct_key, hex_val in color_map.items():
                        colors = np.array([[0, 0, 0, 0], list(mcolors.to_rgb(hex_val)) + [1]])
                        cmap = Colormap(colors, name=f"custom_{ct_key}")
                        ctx.state.persistent_settings.setdefault(ct_key, {})["colormap"] = cmap

            updated = 0
            lw = getattr(ctx, "legend_widget", None)
            if lw:
                for r_list in [
                    getattr(lw, "ref_ct_rows", []),
                    getattr(lw, "seg_rows", []),
                    getattr(lw, "channel_rows", []),
                ]:
                    for row in r_list:
                        r_name = getattr(row, "name", "")
                        if r_name.lower() in self.celltype_colors:
                            c = self.celltype_colors[r_name.lower()]
                            row.update_swatch_display(c)
                            row.colorChanged.emit(row.name, c)
                            updated += 1

            for layer in ctx.viewer.layers:
                layer_base = get_layer_base_name(layer)
                if layer_base.lower() in self.celltype_colors and hasattr(layer, "colormap"):
                    c = self.celltype_colors[layer_base.lower()]
                    self._apply_colormap_to_layer(layer, c)

            if show_dialog:
                unique_cts = len({k.lower() for k in color_map.keys()})
                QMessageBox.information(
                    ctx.control_widget, "Success",
                    f"Updated {unique_cts} cell type color(s).",
                )
        except Exception as exc:
            if show_dialog:
                QMessageBox.critical(
                    ctx.control_widget, "Error",
                    f"Failed to load cell type colors:\n{exc}",
                )

    # Helpers

    def _read_color_file(self, filepath: str) -> "pd.DataFrame | None":
        """Return a DataFrame from *filepath* (CSV or XLSX)."""
        path = Path(filepath)
        try:
            return (
                pd.read_excel(filepath)
                if path.suffix.lower() in (".xlsx", ".xls")
                else pd.read_csv(filepath)
            )
        except Exception as exc:
            QMessageBox.critical(
                self.ctx.control_widget, "Read Error",
                f"Could not read colour file:\n{exc}",
            )
            return None

    def _apply_colormap_to_layer(self, layer: Any, hex_color: str) -> None:
        """Push a new two-stop colormap to a specific canvas layer."""
        try:
            colors = np.array([[0, 0, 0, 0], list(mcolors.to_rgb(hex_color)) + [1]])
            layer.colormap = Colormap(colors, name=f"custom_{layer.name}")
            layer_base = get_layer_base_name(layer)
            if hasattr(self.ctx.state, "persistent_settings"):
                self.ctx.state.persistent_settings.setdefault(layer_base, {})["colormap"] = layer.colormap
                self.ctx.state.persistent_settings.setdefault(layer_base.lower(), {})["colormap"] = layer.colormap
                self.ctx.state.persistent_settings.setdefault(layer_base.upper(), {})["colormap"] = layer.colormap
        except Exception as exc:
            print(f"[IRIS-Viewer] Could not set colormap for {layer.name}: {exc}")

    def _apply_colormap_to_layers(self, cell_type: str, hex_color: str) -> None:
        """Push a new two-stop colormap to every canvas layer matching *cell_type*."""
        for layer in self.ctx.viewer.layers:
            layer_base = get_layer_base_name(layer)
            if layer_base.lower() == cell_type.lower() and hasattr(layer, "colormap"):
                self._apply_colormap_to_layer(layer, hex_color)
