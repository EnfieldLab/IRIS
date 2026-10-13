"""
iris_common.app.services.annotation_service
============================================
Handles loading and normalising annotation CSV/XLSX files into the
shared application state via ``IrisDataAdapter``.

Separated from ``MultiSampleService`` to keep each module focused.
"""
from __future__ import annotations

import types
from pathlib import Path

from qtpy.QtWidgets import QMessageBox

from iris_common.io.data_adapter import IrisDataAdapter


class AnnotationService:
    """
    Loads annotation spreadsheets and populates ``ctx.state.master_df``.

    Parameters
    ----------
    ctx :
        Application context namespace exposing ``state``, ``control_widget``,
        and optionally ``segmentation_panel``.
    """

    def __init__(self, ctx: types.SimpleNamespace) -> None:
        self.ctx = ctx

    def load_annotation_file(self, file_path: str) -> None:
        """
        Parse *file_path* (CSV or XLSX) through ``IrisDataAdapter`` and store
        the normalised DataFrame in ``ctx.state.master_df``.

        On success the control widget's cell-type combo boxes are populated
        and a reference to the adapter is cached on ``ctx.state._data_adapter``
        for downstream re-use.
        """
        ctx = self.ctx
        try:
            adapter = IrisDataAdapter.from_file(file_path)
        except Exception as exc:
            QMessageBox.critical(
                ctx.control_widget, "Annotation Load Error",
                f"Failed to parse annotation file:\n{exc}",
            )
            return

        ctx.state.master_df = adapter.normalised
        # Normalise sample name variants in the Image column
        if "Image" in ctx.state.master_df.columns:
            ctx.state.master_df["Image"] = (
                ctx.state.master_df["Image"]
                .astype(str)
                .str.replace(".tiff", "", regex=False)
                .str.replace("_image", "", regex=False)
            )

        ctx.state._data_adapter = adapter

        # Propagate detected markers to the segmentation channel dropdowns
        if not ctx.state.markers and adapter.markers:
            ctx.state.markers = adapter.markers
            if hasattr(ctx, "segmentation_panel"):
                ctx.segmentation_panel.update_channel_dropdowns(ctx.state.markers)

        # Populate the cell-type reference combo and quick-curation dropdown
        self._populate_celltype_combos(adapter)

        if hasattr(ctx, "control_widget") and hasattr(ctx.control_widget, "lbl_annotation"):
            ctx.control_widget.lbl_annotation.setText(Path(file_path).name)

        QMessageBox.information(
            ctx.control_widget, "Annotation Loaded",
            f"Loaded {len(adapter.normalised):,} cells from {Path(file_path).name}.\n"
            f"Markers detected: {', '.join(adapter.markers[:8])}"
            f"{'…' if len(adapter.markers) > 8 else ''}\n"
            f"Samples: {adapter.normalised['Image'].nunique():,}"
            if "Image" in adapter.normalised.columns else "",
        )

    def _populate_celltype_combos(self, adapter: IrisDataAdapter) -> None:
        """Fill the Cell Phenotype and Quick Curation dropdowns from *adapter*."""
        cw = self.ctx.control_widget
        cols = adapter.normalised.columns.tolist()

        if hasattr(cw, "combo_ref"):
            cw.combo_ref.blockSignals(True)
            cw.combo_ref.clear()
            cw.combo_ref.addItems(cols)

            best_col = next(
                (c for c in ("cell_type", "Verified_CellType", "CellType", "GROUP", "Phenotype", "Label")
                 if c in cols),
                None,
            )
            if best_col:
                cw.combo_ref.setCurrentText(best_col)
            elif cols:
                cw.combo_ref.setCurrentText(cols[0])
            cw.combo_ref.blockSignals(False)

        ref_col = cw.combo_ref.currentText() if hasattr(cw, "combo_ref") else "cell_type"
        if ref_col in adapter.normalised.columns and hasattr(cw, "combo_quick_type"):
            unique_types = sorted(
                str(t) for t in adapter.normalised[ref_col].dropna().unique()
            )
            cw.combo_quick_type.blockSignals(True)
            cw.combo_quick_type.clear()
            cw.combo_quick_type.addItems(unique_types)
            cw.combo_quick_type.blockSignals(False)
