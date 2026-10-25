"""
iris_viewer.app.services.curation_service
=========================================
Handles interactive user curation workflows, including:
- Right-click context menu to check cell identity and reclassify phenotype.
- Left-click (or Alt+click) quick curation to reclassify cells into the selected phenotype.
- Cross-referencing 2D world cursor positions against multi-sample canvas masks and bounding boxes.
- Dynamically creating and updating phenotypic mask image layers on napari canvas.
- Synchronizing cell classification updates across current_mapped_df, resolved_dfs, and master_df.
"""
from __future__ import annotations

import types
from typing import Any, List, Optional, Tuple

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from napari.utils.colormaps import Colormap
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QInputDialog, QMenu

from iris_viewer.app.services.multi_sample_service import get_sample_sub_df
from iris_viewer.widgets.styles import DEFAULT_CELLTYPE_PALETTE


class CurationService:
    """
    Handles interactive user curation inputs on the Napari canvas.
    """

    def __init__(self, ctx: types.SimpleNamespace) -> None:
        self.ctx = ctx

    def on_right_click(self, viewer: Any, event: Any):
        """Right-click opens a context menu for inspecting and reassigning phenotypes."""
        if event.button != 2:
            return

        obj_id, sample_name = self.get_cell_at_position(event.position)

        if obj_id is not None:
            if hasattr(self.ctx, "canvas_manager") and hasattr(self.ctx.canvas_manager, "highlight_cell_outline"):
                self.ctx.canvas_manager.highlight_cell_outline(obj_id, sample_name)
            if hasattr(self.ctx, "expression_widget") and hasattr(self.ctx.expression_widget, "highlight_cell"):
                self.ctx.expression_widget.highlight_cell(obj_id, sample_name)

        yield
        while event.type == 'mouse_move':
            yield
        if event.type == 'mouse_release' and obj_id is not None:
            self.show_context_menu(obj_id, sample_name)

    def on_quick_curate_click(self, viewer: Any, event: Any):
        """Left-click with quick curate toggled instantly applies selected phenotype."""
        if event.button != 1:
            return

        cw = getattr(self.ctx, "control_widget", None)
        msw = getattr(self.ctx, "comparison_widget", None)

        is_checked = False
        target_type = None
        ref_col = None

        if cw and hasattr(cw, "chk_quick_curate") and cw.chk_quick_curate.isChecked():
            is_checked = True
            target_type = cw.combo_quick_type.currentText().strip() if hasattr(cw, "combo_quick_type") else None
            ref_col = cw.combo_ref.currentText().strip() if hasattr(cw, "combo_ref") else None
        elif msw and hasattr(msw, "chk_quick_curate") and msw.chk_quick_curate.isChecked():
            is_checked = True
            target_type = msw.combo_quick_type.currentText().strip() if hasattr(msw, "combo_quick_type") else None
            ref_col = msw.combo_ref.currentText().strip() if hasattr(msw, "combo_ref") else None

        alt_pressed = hasattr(event, "modifiers") and any("Alt" in str(m) for m in event.modifiers)
        if not is_checked and not alt_pressed:
            return

        if not target_type:
            target_type = (
                (cw.combo_quick_type.currentText().strip() if cw and hasattr(cw, "combo_quick_type") else None)
                or (msw.combo_quick_type.currentText().strip() if msw and hasattr(msw, "combo_quick_type") else None)
            )
        if not ref_col:
            ref_col = (
                (cw.combo_ref.currentText().strip() if cw and hasattr(cw, "combo_ref") else None)
                or (msw.combo_ref.currentText().strip() if msw and hasattr(msw, "combo_ref") else None)
                or "cell_type"
            )

        obj_id, sample_name = self.get_cell_at_position(event.position)

        if obj_id is not None:
            if hasattr(self.ctx, "canvas_manager") and hasattr(self.ctx.canvas_manager, "highlight_cell_outline"):
                self.ctx.canvas_manager.highlight_cell_outline(obj_id, sample_name)
            if hasattr(self.ctx, "expression_widget") and hasattr(self.ctx.expression_widget, "highlight_cell"):
                self.ctx.expression_widget.highlight_cell(obj_id, sample_name)

            if target_type and ref_col:
                self.apply_quick_phenotype_change(obj_id, target_type, ref_col, sample_name)

        yield

    def get_cell_at_position(self, position: Any) -> Tuple[Optional[int], Optional[str]]:
        """Cross-references world cursor position with translated segmentation layers and masks.

        Returns:
            Tuple[Optional[int], Optional[str]]: (cell_id, sample_name) or (None, None)
        """
        if position is None or len(position) < 2:
            return None, None

        # Gather all active samples on canvas
        sample_names = list(self.ctx.state.sample_layers_map.keys())
        if not sample_names and hasattr(self.ctx.state, "masks_map"):
            sample_names = list(self.ctx.state.masks_map.keys())

        for sample_name in sample_names:
            layers = self.ctx.state.sample_layers_map.get(sample_name, [])
            mask = self.ctx.state.masks_map.get(sample_name)

            bbox_lyr = next((l for l in layers if l.name == f"Bounds - {sample_name}"), None)
            seg_layer = next(
                (l for l in layers if l.__class__.__name__ == 'Labels' and ("segmentation" in l.name.lower() or "mask" in l.name.lower())),
                None,
            )

            if mask is None and seg_layer is not None:
                mask = seg_layer.data

            if mask is None:
                continue

            if hasattr(mask, "compute"):
                mask = mask.compute()
            mask = np.asarray(mask)

            y, x = None, None
            # Attempt napari native world_to_data translation if layer is available
            if seg_layer is not None and hasattr(seg_layer, "world_to_data"):
                try:
                    data_coord = seg_layer.world_to_data(position)
                    if len(data_coord) >= 2:
                        y, x = int(round(data_coord[-2])), int(round(data_coord[-1]))
                except Exception:
                    y, x = None, None

            # Fallback using bounding box or reference layer translation and scale
            if y is None or x is None:
                ref_lyr = bbox_lyr or seg_layer or (layers[0] if layers else None)
                ty, tx = getattr(ref_lyr, "translate", (0.0, 0.0)) if ref_lyr else (0.0, 0.0)
                sy, sx = getattr(ref_lyr, "scale", (1.0, 1.0)) if ref_lyr else (1.0, 1.0)
                sy = sy if sy != 0 else 1.0
                sx = sx if sx != 0 else 1.0
                y = int(round((position[-2] - ty) / sy))
                x = int(round((position[-1] - tx) / sx))

            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                obj_id = int(mask[y, x])
                if obj_id > 0:
                    return obj_id, sample_name

        return None, None

    def show_context_menu(self, obj_id: int, sample_name: str) -> None:
        """Pops up a dropdown menu near the mouse cursor to inspect and reclassify phenotype."""
        cw = getattr(self.ctx, "control_widget", None)
        msw = getattr(self.ctx, "comparison_widget", None)

        ref_col = (
            (cw.combo_ref.currentText().strip() if cw and hasattr(cw, "combo_ref") else None)
            or (msw.combo_ref.currentText().strip() if msw and hasattr(msw, "combo_ref") else None)
            or "cell_type"
        )

        current_type = self._get_cell_phenotype(obj_id, sample_name, ref_col)

        parent_widget = (
            (cw if cw and hasattr(cw, "winId") else None)
            or (msw if msw and hasattr(msw, "winId") else None)
            or (self.ctx.viewer.window._qt_window if hasattr(self.ctx.viewer, "window") else None)
        )
        menu = QMenu(parent_widget)
        menu.setStyleSheet(
            "QMenu { background-color: #222; color: #FFF; border: 1px solid #555; padding: 5px; } "
            "QMenu::item { padding: 4px 20px; } "
            "QMenu::item:selected { background-color: #3A6A9A; }"
        )

        sample_label = f"{sample_name} | " if sample_name else ""
        title_action = menu.addAction(f"{sample_label}Cell #{obj_id} | Phenotype: {current_type}")
        title_action.setEnabled(False)
        menu.addSeparator()

        unique_types = self._get_available_phenotypes(ref_col)
        for ctype in unique_types:
            if ctype != current_type:
                action = menu.addAction(ctype)
                action.triggered.connect(
                    lambda checked=False, t=ctype, oid=obj_id, s=sample_name, c=ref_col:
                    self.apply_quick_phenotype_change(oid, t, c, s)
                )

        menu.addSeparator()
        add_new_action = menu.addAction("Add New Phenotype Label...")
        add_new_action.triggered.connect(
            lambda: self.prompt_new_phenotype(obj_id, ref_col, sample_name)
        )
        menu.exec_(QCursor.pos())

    def prompt_new_phenotype(self, obj_id: int, ref_col: str, sample_name: str) -> None:
        """Allows creating a brand new phenotypic label ad-hoc."""
        parent_widget = getattr(self.ctx, "control_widget", None) or getattr(self.ctx, "comparison_widget", None)
        new_type, ok = QInputDialog.getText(parent_widget, "New Phenotype", "Enter label designation:")
        if ok and new_type.strip():
            label = new_type.strip()
            self.apply_quick_phenotype_change(obj_id, label, ref_col, sample_name)
            for w in (getattr(self.ctx, "control_widget", None), getattr(self.ctx, "comparison_widget", None)):
                if w and hasattr(w, "combo_quick_type"):
                    idx = w.combo_quick_type.findText(label)
                    if idx >= 0:
                        w.combo_quick_type.setCurrentIndex(idx)

    def _get_cell_phenotype(self, obj_id: int, sample_name: str, ref_col: str) -> str:
        """Find the cell phenotype across resolved_dfs, current_mapped_df, master_df, or canvas layers."""
        # 1. Check resolved_dfs[sample_name]
        if hasattr(self.ctx.state, "resolved_dfs") and sample_name in self.ctx.state.resolved_dfs:
            s_df = self.ctx.state.resolved_dfs[sample_name]
            if s_df is not None and not s_df.empty and "ObjectNumber" in s_df.columns:
                m = s_df["ObjectNumber"] == obj_id
                if m.any() and ref_col in s_df.columns:
                    val = s_df.loc[m, ref_col].iloc[0]
                    if pd.notna(val) and str(val).strip():
                        return str(val).strip()

        # 2. Check current_mapped_df
        df_sub = getattr(self.ctx.state, "current_mapped_df", None)
        if df_sub is not None and not df_sub.empty and "ObjectNumber" in df_sub.columns:
            if "Image" in df_sub.columns and sample_name:
                matched_sub = get_sample_sub_df(df_sub, sample_name)
                if not matched_sub.empty:
                    m = matched_sub["ObjectNumber"] == obj_id
                    if m.any() and ref_col in matched_sub.columns:
                        val = matched_sub.loc[m, ref_col].iloc[0]
                        if pd.notna(val) and str(val).strip():
                            return str(val).strip()
            else:
                m = df_sub["ObjectNumber"] == obj_id
                if m.any() and ref_col in df_sub.columns:
                    val = df_sub.loc[m, ref_col].iloc[0]
                    if pd.notna(val) and str(val).strip():
                        return str(val).strip()

        # 3. Check master_df
        m_df = getattr(self.ctx.state, "master_df", None)
        if m_df is not None and not m_df.empty and "ObjectNumber" in m_df.columns:
            if "Image" in m_df.columns and sample_name:
                matched_sub = get_sample_sub_df(m_df, sample_name)
                if not matched_sub.empty:
                    m = matched_sub["ObjectNumber"] == obj_id
                    if m.any() and ref_col in matched_sub.columns:
                        val = matched_sub.loc[m, ref_col].iloc[0]
                        if pd.notna(val) and str(val).strip():
                            return str(val).strip()
            else:
                m = m_df["ObjectNumber"] == obj_id
                if m.any() and ref_col in m_df.columns:
                    val = m_df.loc[m, ref_col].iloc[0]
                    if pd.notna(val) and str(val).strip():
                        return str(val).strip()

        # 4. Check active phenotype visual layers for this sample
        s_layers = self.ctx.state.sample_layers_map.get(sample_name, [])
        mask = self.ctx.state.masks_map.get(sample_name)
        if mask is None and sample_name == getattr(self.ctx.state, "current_sample_name", None):
            mask = getattr(self.ctx.state, "current_mask", None)

        if mask is not None:
            if hasattr(mask, "compute"):
                mask = mask.compute()
            mask_np = np.asarray(mask)
            mask_bool = (mask_np == obj_id)
            base_markers = set(self.ctx.state.markers) if getattr(self.ctx.state, "markers", None) else set()
            for lyr in s_layers:
                if lyr.__class__.__name__ != 'Image':
                    continue
                if getattr(lyr.data, "shape", None) != mask_bool.shape:
                    continue
                if not isinstance(lyr.data, np.ndarray) or hasattr(lyr.data, "compute"):
                    continue
                b_name = lyr.name.split(" [")[0].strip()
                if (
                    b_name in base_markers
                    or "segmentation" in b_name.lower()
                    or "mask" in b_name.lower()
                    or "bounds" in b_name.lower()
                    or "centroid" in b_name.lower()
                ):
                    continue
                if np.any(lyr.data[mask_bool] > 0):
                    return b_name

        return "Unassigned"

    def _get_available_phenotypes(self, ref_col: str) -> List[str]:
        """Aggregate unique phenotype labels from active annotation dataframes, dropdowns, and canvas layers."""
        types_set = set()

        m_df = getattr(self.ctx.state, "master_df", None)
        if m_df is not None and not m_df.empty and ref_col in m_df.columns:
            types_set.update(str(x).strip() for x in m_df[ref_col].dropna().unique() if str(x).strip())

        curr_df = getattr(self.ctx.state, "current_mapped_df", None)
        if curr_df is not None and not curr_df.empty and ref_col in curr_df.columns:
            types_set.update(str(x).strip() for x in curr_df[ref_col].dropna().unique() if str(x).strip())

        if hasattr(self.ctx.state, "resolved_dfs"):
            for s_df in self.ctx.state.resolved_dfs.values():
                if s_df is not None and not s_df.empty and ref_col in s_df.columns:
                    types_set.update(str(x).strip() for x in s_df[ref_col].dropna().unique() if str(x).strip())

        for w in (getattr(self.ctx, "control_widget", None), getattr(self.ctx, "comparison_widget", None)):
            if w and hasattr(w, "combo_quick_type"):
                types_set.update(
                    w.combo_quick_type.itemText(i).strip()
                    for i in range(w.combo_quick_type.count())
                    if w.combo_quick_type.itemText(i).strip()
                )

        return sorted(x for x in types_set if x)

    def _update_df_row(self, df: pd.DataFrame, obj_id: int, sample_name: str, new_type: str, ref_col: str) -> Optional[str]:
        """Update a cell's phenotype in a DataFrame, returning its previous phenotype if found."""
        if df is None or df.empty or "ObjectNumber" not in df.columns:
            return None

        target_indices = None
        if "Image" in df.columns and sample_name:
            sub = get_sample_sub_df(df, sample_name)
            if not sub.empty and "ObjectNumber" in sub.columns:
                m = sub["ObjectNumber"] == obj_id
                if m.any():
                    target_indices = sub[m].index

        if target_indices is None or len(target_indices) == 0:
            m = df["ObjectNumber"] == obj_id
            if m.any():
                target_indices = df[m].index

        if target_indices is None or len(target_indices) == 0:
            return None

        old_val = None
        if ref_col in df.columns:
            val = df.loc[target_indices[0], ref_col]
            if pd.notna(val) and str(val).strip():
                old_val = str(val).strip()

            if df[ref_col].dtype.name == 'category' and new_type not in df[ref_col].cat.categories:
                df[ref_col] = df[ref_col].cat.add_categories([new_type])
            df.loc[target_indices, ref_col] = new_type

        if 'Verified_CellType' not in df.columns:
            df['Verified_CellType'] = df[ref_col] if ref_col in df.columns else new_type
        if df['Verified_CellType'].dtype.name == 'category' and new_type not in df['Verified_CellType'].cat.categories:
            df['Verified_CellType'] = df['Verified_CellType'].cat.add_categories([new_type])
        df.loc[target_indices, 'Verified_CellType'] = new_type

        return old_val

    def apply_quick_phenotype_change(self, obj_id: int, new_type: str, ref_col: str, sample_name: str) -> None:
        """Updates the DataFrame and swaps the active visual mask overlay."""
        new_type = str(new_type).strip()
        if not new_type:
            return

        # 1. Update UI dropdowns
        for w in (getattr(self.ctx, "control_widget", None), getattr(self.ctx, "comparison_widget", None)):
            if w and hasattr(w, "combo_quick_type"):
                items = [w.combo_quick_type.itemText(i).strip() for i in range(w.combo_quick_type.count())]
                if new_type not in items:
                    w.combo_quick_type.addItem(new_type)

        old_type = self._get_cell_phenotype(obj_id, sample_name, ref_col)
        if old_type == "Unassigned":
            old_type = None

        if old_type == new_type:
            return

        # 2. Update resolved_dfs for sample_name
        if hasattr(self.ctx.state, "resolved_dfs") and sample_name in self.ctx.state.resolved_dfs:
            s_df = self.ctx.state.resolved_dfs[sample_name]
            prev = self._update_df_row(s_df, obj_id, sample_name, new_type, ref_col)
            if prev and not old_type:
                old_type = prev

        # 3. Update current_mapped_df
        curr_df = getattr(self.ctx.state, "current_mapped_df", None)
        if curr_df is not None:
            prev = self._update_df_row(curr_df, obj_id, sample_name, new_type, ref_col)
            if prev and not old_type:
                old_type = prev

        # 4. Update master_df
        m_df = getattr(self.ctx.state, "master_df", None)
        if m_df is not None:
            prev = self._update_df_row(m_df, obj_id, sample_name, new_type, ref_col)
            if prev and not old_type:
                old_type = prev

        # Synchronize current_mapped_df from resolved_dfs if in multi-sample mode
        if hasattr(self.ctx.state, "resolved_dfs") and self.ctx.state.resolved_dfs:
            loaded_keys = [sn for sn in self.ctx.state.sample_layers_map.keys() if sn in self.ctx.state.resolved_dfs]
            if loaded_keys:
                all_dfs = [self.ctx.state.resolved_dfs[sn] for sn in loaded_keys]
                self.ctx.state.current_mapped_df = pd.concat(all_dfs, ignore_index=True)

        # 5. Visual Mask Updating on Napari Canvas
        mask = self.ctx.state.masks_map.get(sample_name)
        if mask is None and sample_name == getattr(self.ctx.state, "current_sample_name", None):
            mask = getattr(self.ctx.state, "current_mask", None)

        if mask is not None:
            if hasattr(mask, "compute"):
                mask = mask.compute()
            mask_np = np.asarray(mask)
            mask_bool = (mask_np == obj_id)
            s_layers = self.ctx.state.sample_layers_map.get(sample_name, [])
            if not s_layers:
                s_layers = list(self.ctx.viewer.layers)

            base_markers = set(self.ctx.state.markers) if getattr(self.ctx.state, "markers", None) else set()

            def _is_phenotype_layer(l, target_name: Optional[str] = None) -> bool:
                if l.__class__.__name__ != 'Image':
                    return False
                if getattr(l.data, "shape", None) != mask_bool.shape:
                    return False
                if not isinstance(l.data, np.ndarray) or hasattr(l.data, "compute"):
                    return False
                b_name = l.name.split(" [")[0].strip()
                if target_name is not None and b_name.lower() != target_name.strip().lower():
                    return False
                if (
                    b_name in base_markers
                    or "segmentation" in b_name.lower()
                    or "mask" in b_name.lower()
                    or "bounds" in b_name.lower()
                    or "centroid" in b_name.lower()
                ):
                    return False
                return True

            # 1. Unconditionally clear this cell from ALL other phenotype layers for this sample
            for lyr in s_layers:
                if _is_phenotype_layer(lyr):
                    b_name = lyr.name.split(" [")[0].strip()
                    if b_name.lower() != new_type.lower():
                        if np.any(lyr.data[mask_bool] > 0):
                            new_data = lyr.data.copy()
                            new_data[mask_bool] = 0
                            lyr.data = new_data

            # 2. Add this cell to the target phenotype layer for this sample
            new_layer = next((l for l in s_layers if _is_phenotype_layer(l, new_type)), None)
            if new_layer is not None:
                new_data = new_layer.data.copy()
                new_data[mask_bool] = 1
                new_layer.data = new_data
                new_layer.visible = True
            else:
                self._create_phenotype_layer(sample_name, new_type, mask_bool, s_layers)

        # 6. Trigger metrics and expression widget refresh
        if hasattr(self.ctx, "_update_dashboard_metrics"):
            self.ctx._update_dashboard_metrics()
        if hasattr(self.ctx, "expression_widget") and hasattr(self.ctx.expression_widget, "refresh"):
            self.ctx.expression_widget.refresh()

    def _create_phenotype_layer(self, sample_name: str, type_name: str, mask_bool: np.ndarray, s_layers: list) -> None:
        """Create a new binary mask overlay layer for a previously unrepresented phenotype."""
        ctx = self.ctx
        hex_color = "#3A6A9A"
        if hasattr(ctx.state, "celltype_colors") and type_name.lower() in ctx.state.celltype_colors:
            hex_color = ctx.state.celltype_colors[type_name.lower()]
        elif hasattr(ctx.state, "celltype_colors") and type_name in ctx.state.celltype_colors:
            hex_color = ctx.state.celltype_colors[type_name]
        elif hasattr(ctx, "multi_sample_svc") and type_name.lower() in ctx.multi_sample_svc.CELLTYPE_COLORS:
            hex_color = ctx.multi_sample_svc.CELLTYPE_COLORS[type_name.lower()]
        else:
            idx = abs(hash(type_name)) % len(DEFAULT_CELLTYPE_PALETTE)
            hex_color = DEFAULT_CELLTYPE_PALETTE[idx]

        # Persist color mapping for this phenotype
        if hasattr(ctx.state, "celltype_colors"):
            ctx.state.celltype_colors[type_name] = hex_color
            ctx.state.celltype_colors[type_name.lower()] = hex_color
        if hasattr(ctx, "multi_sample_svc"):
            ctx.multi_sample_svc.CELLTYPE_COLORS[type_name.lower()] = hex_color

        ref_lyr = next((l for l in s_layers if f"Bounds - {sample_name}" in l.name), None) or (s_layers[0] if s_layers else None)
        scale_val = ref_lyr.scale if ref_lyr is not None else (1.0, 1.0)
        translation = ref_lyr.translate if ref_lyr is not None else (0.0, 0.0)

        mask_bin = np.zeros(mask_bool.shape, dtype=np.uint8)
        mask_bin[mask_bool] = 1

        colors = np.array([[0, 0, 0, 0], list(mcolors.to_rgb(hex_color)) + [1.0]])
        lyr = ctx.viewer.add_image(
            mask_bin, name=type_name, opacity=0.8, visible=True,
            blending="translucent", colormap=Colormap(colors),
            contrast_limits=[0, 1], scale=scale_val,
        )
        lyr.translate = translation

        if lyr not in s_layers:
            s_layers.append(lyr)
        if sample_name:
            if sample_name not in ctx.state.sample_layers_map:
                ctx.state.sample_layers_map[sample_name] = []
            if lyr not in ctx.state.sample_layers_map[sample_name]:
                ctx.state.sample_layers_map[sample_name].append(lyr)

        if hasattr(ctx, "multi_sample_svc"):
            ctx.multi_sample_svc._register_layer(lyr, type_name)
            already_in_legend = False
            if hasattr(ctx, "legend_widget") and hasattr(ctx.legend_widget, "ref_ct_rows"):
                already_in_legend = any(r.name == type_name for r in ctx.legend_widget.ref_ct_rows)
            elif hasattr(ctx.multi_sample_svc, "_added_master_legends"):
                already_in_legend = type_name in ctx.multi_sample_svc._added_master_legends

            if not already_in_legend:
                ctx.legend_widget.add_ref_ct_row(lyr, type_name, hex_color)
                ctx.multi_sample_svc._added_master_legends.add(type_name)
        elif hasattr(ctx, "legend_widget"):
            already_in_legend = any(r.name == type_name for r in getattr(ctx.legend_widget, "ref_ct_rows", []))
            if not already_in_legend:
                ctx.legend_widget.add_ref_ct_row(lyr, type_name, hex_color)
