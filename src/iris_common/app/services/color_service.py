"""
iris_common.app.services.color_service
=======================================
Handles loading user-supplied colour mapping files and applying them
to marker channel layers and cell-type phenotype layers on the canvas.

Separated from ``MultiSampleService`` to keep each module focused.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors
from napari.utils.colormaps import Colormap
from qtpy.QtWidgets import QMessageBox

import pandas as pd


class ColorService:
    """
    Loads CSV/XLSX colour maps and applies them to the napari canvas and Legend.

    Parameters
    ----------
    ctx :
        Application context namespace exposing ``viewer``, ``state``,
        ``control_widget``, and ``legend_widget``.
    default_colors :
        Mutable dict mapping marker names to default hex colours (shared
        reference with ``MultiSampleService``).
    celltype_colors :
        Mutable dict mapping cell-type names (lowercase) to hex colours
        (shared reference with ``MultiSampleService``).
    """

    def __init__(
        self,
        ctx: types.SimpleNamespace,
        default_colors: dict,
        celltype_colors: dict,
    ) -> None:
        self.ctx = ctx
        self.default_colors = default_colors
        self.celltype_colors = celltype_colors

    # ── Channel / Marker colours ────────────────────────────────────────────

    # ── Channel / Marker colours ────────────────────────────────────────────

    def load_channel_colors(self, filepath: str) -> None:
        """
        Load a marker → colour mapping from *filepath* (CSV or XLSX).
        Matching legend rows and canvas layers are updated in place.
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
                QMessageBox.warning(
                    ctx.control_widget, "Invalid Format",
                    f"File must contain a marker column (e.g. 'markers', 'marker') and a color column (e.g. 'colors', 'color').\nFound columns: {list(df.columns)}",
                )
                return

            color_map = {}
            for _, row in df.iterrows():
                val_m = row[m_col]
                val_c = row[c_col]
                if pd.notna(val_m) and pd.notna(val_c):
                    m = str(val_m).strip().lower()
                    c = str(val_c).strip()
                    if mcolors.is_color_like(c):
                        color_map[m] = mcolors.to_hex(c)

            for m_key, hex_c in color_map.items():
                self.default_colors[m_key] = hex_c
                for orig_m in getattr(ctx.state, "markers", []):
                    if orig_m.lower() == m_key:
                        self.default_colors[orig_m] = hex_c

            updated = 0
            lw = getattr(ctx, "legend_widget", None)
            if lw:
                row_lists = [getattr(lw, "channel_rows", [])]
                if hasattr(lw, "custom_channel_groups") and "Channels" in lw.custom_channel_groups:
                    _, _, custom_rows = lw.custom_channel_groups["Channels"]
                    row_lists.append(custom_rows)

                for r_list in row_lists:
                    for row in r_list:
                        if row.name.lower() in color_map:
                            c = color_map[row.name.lower()]
                            row.update_swatch_display(c)
                            row.colorChanged.emit(row.name, c)
                            updated += 1

            for layer in ctx.viewer.layers:
                layer_base = layer.name.split(" [")[0]
                if layer_base.lower() in color_map and hasattr(layer, "colormap"):
                    c = color_map[layer_base.lower()]
                    self._apply_colormap_to_layer(layer, c)

            QMessageBox.information(
                ctx.control_widget, "Success",
                f"Updated {len(color_map)} marker color(s).",
            )
        except Exception as exc:
            QMessageBox.critical(
                ctx.control_widget, "Error",
                f"Failed to load channel colors:\n{exc}",
            )

    # ── Cell-type colours ───────────────────────────────────────────────────

    def load_celltype_colors(self, filepath: str) -> None:
        """
        Load a cell-type → colour mapping from *filepath* (CSV or XLSX).
        Matching legend rows and canvas layers are updated in place.
        """
        ctx = self.ctx
        try:
            import re
            df = self._read_color_file(filepath)
            if df is None:
                return

            col_norm = {re.sub(r'[^a-z0-9]', '', str(c).lower()): c for c in df.columns}
            ct_col = next((col_norm[k] for k in ["celltypes", "celltype", "celltype", "celltypes", "phenotype", "phenotypes", "label", "labels", "cluster", "clusters"] if k in col_norm), None)
            c_col = next((col_norm[k] for k in ["colors", "color", "colours", "colour", "hex", "code"] if k in col_norm), None)

            if ct_col is None or c_col is None:
                QMessageBox.warning(
                    ctx.control_widget, "Invalid Format",
                    f"File must contain a cell type column (e.g. 'cell_type', 'celltypes', 'phenotype') and a color column (e.g. 'colors', 'color', 'hex').\nFound columns: {list(df.columns)}",
                )
                return

            color_map = {}
            for _, row in df.iterrows():
                val_ct = row[ct_col]
                val_c = row[c_col]
                if pd.notna(val_ct) and pd.notna(val_c):
                    ct = str(val_ct).strip().lower()
                    cv = str(val_c).strip()
                    if mcolors.is_color_like(cv):
                        hex_val = mcolors.to_hex(cv)
                        self.celltype_colors[ct] = hex_val
                        color_map[ct] = hex_val

            updated = 0
            lw = getattr(ctx, "legend_widget", None)
            if lw:
                for r_list in [
                    getattr(lw, "ref_ct_rows", []),
                    getattr(lw, "seg_rows", []),
                    getattr(lw, "channel_rows", []),
                ]:
                    for row in r_list:
                        if row.name.lower() in self.celltype_colors:
                            c = self.celltype_colors[row.name.lower()]
                            row.update_swatch_display(c)
                            row.colorChanged.emit(row.name, c)
                            updated += 1

            for layer in ctx.viewer.layers:
                layer_base = layer.name.split(" [")[0]
                if layer_base.lower() in self.celltype_colors and hasattr(layer, "colormap"):
                    c = self.celltype_colors[layer_base.lower()]
                    self._apply_colormap_to_layer(layer, c)

            QMessageBox.information(
                ctx.control_widget, "Success",
                f"Updated {len(color_map)} cell type color(s).",
            )
        except Exception as exc:
            QMessageBox.critical(
                ctx.control_widget, "Error",
                f"Failed to load cell type colors:\n{exc}",
            )

    # ── Helpers ─────────────────────────────────────────────────────────────

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
            layer_base = layer.name.split(" [")[0]
            if hasattr(self.ctx.state, "persistent_settings"):
                self.ctx.state.persistent_settings.setdefault(layer_base, {})["colormap"] = layer.colormap
        except Exception as exc:
            print(f"[IRIS-Viewer] Could not set colormap for {layer.name}: {exc}")

    def _apply_colormap_to_layers(self, cell_type: str, hex_color: str) -> None:
        """Push a new two-stop colormap to every canvas layer matching *cell_type*."""
        for layer in self.ctx.viewer.layers:
            layer_base = layer.name.split(" [")[0]
            if layer_base.lower() == cell_type.lower() and hasattr(layer, "colormap"):
                self._apply_colormap_to_layer(layer, hex_color)
