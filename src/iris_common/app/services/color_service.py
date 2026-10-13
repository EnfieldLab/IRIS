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

    def load_channel_colors(self, filepath: str) -> None:
        """
        Load a marker → colour mapping from *filepath* (CSV or XLSX).

        The file must contain ``markers`` and ``colors`` columns.
        Matching legend rows are updated in place.
        """
        ctx = self.ctx
        try:
            df = self._read_color_file(filepath)
            if df is None:
                return
            df.columns = [str(c).lower().strip() for c in df.columns]
            if "markers" not in df.columns or "colors" not in df.columns:
                QMessageBox.warning(
                    ctx.control_widget, "Invalid Format",
                    "File must contain 'markers' and 'colors' columns.",
                )
                return
            color_map = {
                str(r["markers"]).strip().lower(): mcolors.to_hex(str(r["colors"]).strip())
                for _, r in df.iterrows()
                if pd.notna(r["markers"]) and mcolors.is_color_like(str(r["colors"]).strip())
            }
            for m in ctx.state.markers:
                if m.lower() in color_map:
                    self.default_colors[m] = color_map[m.lower()]

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
            QMessageBox.information(
                ctx.control_widget, "Success",
                f"Updated {updated} marker colors.",
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

        The file must contain ``celltypes`` and ``colors`` columns.
        Matching legend rows and canvas layers are updated in place.
        """
        ctx = self.ctx
        try:
            df = self._read_color_file(filepath)
            if df is None:
                return
            df.columns = [str(c).lower().strip() for c in df.columns]
            if "celltypes" not in df.columns or "colors" not in df.columns:
                QMessageBox.warning(
                    ctx.control_widget, "Invalid Format",
                    "File must contain 'celltypes' and 'colors' columns.",
                )
                return
            for _, row in df.iterrows():
                ct = str(row["celltypes"]).strip().lower()
                cv = str(row["colors"]).strip()
                if mcolors.is_color_like(cv):
                    self.celltype_colors[ct] = mcolors.to_hex(cv)

            updated = 0
            for row in getattr(ctx.legend_widget, "seg_rows", []):
                if row.name.lower() in self.celltype_colors:
                    c = self.celltype_colors[row.name.lower()]
                    row.update_swatch_display(c)
                    row.colorChanged.emit(row.name, c)
                    updated += 1
                    self._apply_colormap_to_layers(row.name, c)

            QMessageBox.information(
                ctx.control_widget, "Success",
                f"Updated {updated} cell type colors.",
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

    def _apply_colormap_to_layers(self, cell_type: str, hex_color: str) -> None:
        """Push a new two-stop colormap to every canvas layer matching *cell_type*."""
        for layer in self.ctx.viewer.layers:
            layer_base = layer.name.split(" [")[0]
            if layer_base.lower() == cell_type.lower() and hasattr(layer, "colormap"):
                try:
                    colors = np.array([[0, 0, 0, 0], list(mcolors.to_rgb(hex_color)) + [1]])
                    layer.colormap = Colormap(colors, name=f"custom_{layer.name}")
                except Exception as exc:
                    print(f"[IRIS-Viewer] Could not set colormap for {layer.name}: {exc}")
