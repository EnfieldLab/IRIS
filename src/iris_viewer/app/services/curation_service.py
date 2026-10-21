from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QInputDialog, QMenu, QMessageBox


class CurationService:
    """
    Handles interactive user inputs, such as right-click phenotype curation.
    """
    def __init__(self, ctx):
        self.ctx = ctx

    def on_right_click(self, viewer, event):
        """Right-click opens a context menu for reassigning phenotypes to cells."""
        if event.button != 2: return
        obj_id, sample_name = self.get_cell_at_position(event.position)

        if obj_id is not None:
            # Highlight logic - optional, if expression_widget / canvas_manager exists
            if hasattr(self.ctx, "canvas_manager") and hasattr(self.ctx.canvas_manager, "highlight_cell_outline"):
                self.ctx.canvas_manager.highlight_cell_outline(obj_id, sample_name)
            if hasattr(self.ctx, "expression_widget") and hasattr(self.ctx.expression_widget, "highlight_cell"):
                self.ctx.expression_widget.highlight_cell(obj_id, sample_name)

        yield
        while event.type == 'mouse_move': yield
        if event.type == 'mouse_release' and obj_id is not None:
            self.show_context_menu(obj_id, sample_name)

    def on_quick_curate_click(self, viewer, event):
        """Left-click with quick curate toggled instantly applies selected phenotype."""
        # Ensure that it's a left click, and that the quick_curate checkbox exists and is checked
        cw = self.ctx.control_widget
        if event.button != 1 or not hasattr(cw, "chk_quick_curate") or not cw.chk_quick_curate.isChecked():
            return

        obj_id, sample_name = self.get_cell_at_position(event.position)

        if obj_id is not None:
            if hasattr(self.ctx, "canvas_manager") and hasattr(self.ctx.canvas_manager, "highlight_cell_outline"):
                self.ctx.canvas_manager.highlight_cell_outline(obj_id, sample_name)

            target_type = cw.combo_quick_type.currentText()
            ref_col = cw.combo_ref.currentText()

            if target_type and ref_col:
                self.apply_quick_phenotype_change(obj_id, target_type, ref_col, sample_name)

        yield

    def get_cell_at_position(self, position):
        """Cross-references world cursor position with translated segmentation layers."""
        if getattr(self.ctx.state, "current_mapped_df", None) is None: return None, None

        for sample_name, layers in self.ctx.state.sample_layers_map.items():
            seg_layer = next((l for l in layers if l.__class__.__name__ == 'Labels' and ("Segmentation" in l.name or "Mask" in l.name)), None)

            if seg_layer is None:
                continue

            data_coord = seg_layer.world_to_data(position)

            if len(data_coord) >= 2:
                y, x = int(data_coord[-2]), int(data_coord[-1])
                mask_data = seg_layer.data

                if 0 <= y < mask_data.shape[0] and 0 <= x < mask_data.shape[1]:
                    obj_id = mask_data[y, x]
                    if obj_id > 0:
                        df_sub = self.ctx.state.current_mapped_df
                        if 'ObjectNumber' not in df_sub.columns:
                            continue

                        if 'Image' in df_sub.columns:
                            matched = df_sub[(df_sub['Image'].astype(str) == str(sample_name)) & (df_sub['ObjectNumber'] == obj_id)]
                        else:
                            matched = df_sub[df_sub['ObjectNumber'] == obj_id]

                        if not matched.empty:
                            return obj_id, sample_name

        return None, None

    def show_context_menu(self, obj_id, sample_name):
        """Pops up a dropdown menu near the mouse cursor to select a new phenotype."""
        if sample_name and 'Image' in self.ctx.state.current_mapped_df.columns:
            row_mask = (self.ctx.state.current_mapped_df['ObjectNumber'] == obj_id) & (self.ctx.state.current_mapped_df['Image'] == sample_name)
        else:
            row_mask = self.ctx.state.current_mapped_df['ObjectNumber'] == obj_id

        if not row_mask.any(): return
        ref_col = self.ctx.control_widget.combo_ref.currentText()
        if not ref_col or ref_col not in self.ctx.state.current_mapped_df.columns: return
        current_type = self.ctx.state.current_mapped_df.loc[row_mask, ref_col].values[0]

        menu = QMenu(self.ctx.control_widget)
        menu.setStyleSheet("QMenu { background-color: #222; color: #FFF; border: 1px solid #555; padding: 5px;} QMenu::item { padding: 4px 20px; } QMenu::item:selected { background-color: #3A6A9A; }")

        sample_label = f"{sample_name} | " if sample_name else ""
        title_action = menu.addAction(f"{sample_label}ID: {obj_id} | Phenotype: {current_type}")
        title_action.setEnabled(False)
        menu.addSeparator()

        unique_types = sorted([str(t) for t in self.ctx.state.master_df[ref_col].dropna().unique()])
        for ctype in unique_types:
            if ctype != current_type:
                action = menu.addAction(ctype)
                action.triggered.connect(lambda checked, t=ctype, oid=obj_id, s=sample_name: self.apply_quick_phenotype_change(oid, t, ref_col, s))

        menu.addSeparator()
        add_new_action = menu.addAction("Add New Phenotype Label...")
        add_new_action.triggered.connect(lambda: self.prompt_new_phenotype(obj_id, ref_col, sample_name))
        menu.exec_(QCursor.pos())

    def prompt_new_phenotype(self, obj_id, ref_col, sample_name):
        """Allows creating a brand new phenotypic label ad-hoc."""
        new_type, ok = QInputDialog.getText(self.ctx.control_widget, "New Phenotype", "Enter label designation:")
        if ok and new_type.strip():
            self.apply_quick_phenotype_change(obj_id, new_type.strip(), ref_col, sample_name)

    def apply_quick_phenotype_change(self, obj_id, new_type, ref_col, sample_name):
        """Updates the dataframe and swaps the active visual mask overlay."""
        if not new_type or self.ctx.state.current_mapped_df is None: return

        df_sub = self.ctx.state.current_mapped_df

        if df_sub[ref_col].dtype.name == 'category' and new_type not in df_sub[ref_col].cat.categories:
            df_sub[ref_col] = df_sub[ref_col].cat.add_categories([new_type])

        if 'Verified_CellType' in df_sub.columns and df_sub['Verified_CellType'].dtype.name == 'category':
            if new_type not in df_sub['Verified_CellType'].cat.categories:
                df_sub['Verified_CellType'] = df_sub['Verified_CellType'].cat.add_categories([new_type])

        if self.ctx.state.master_df is not None:
            if ref_col in self.ctx.state.master_df.columns and self.ctx.state.master_df[ref_col].dtype.name == 'category':
                if new_type not in self.ctx.state.master_df[ref_col].cat.categories:
                    self.ctx.state.master_df[ref_col] = self.ctx.state.master_df[ref_col].cat.add_categories([new_type])
            if 'Verified_CellType' in self.ctx.state.master_df.columns and self.ctx.state.master_df['Verified_CellType'].dtype.name == 'category':
                if new_type not in self.ctx.state.master_df['Verified_CellType'].cat.categories:
                    self.ctx.state.master_df['Verified_CellType'] = self.ctx.state.master_df['Verified_CellType'].cat.add_categories([new_type])

        existing_items = [self.ctx.control_widget.combo_quick_type.itemText(i) for i in range(self.ctx.control_widget.combo_quick_type.count())]
        if new_type not in existing_items:
            self.ctx.control_widget.combo_quick_type.addItem(new_type)

        if sample_name and 'Image' in df_sub.columns:
            row_mask = (df_sub['ObjectNumber'] == obj_id) & (df_sub['Image'].astype(str) == str(sample_name))
        else:
            row_mask = df_sub['ObjectNumber'] == obj_id

        if not row_mask.any(): return

        target_row = df_sub[row_mask].iloc[0]
        pos_x = target_row.get('Pos_X')
        pos_y = target_row.get('Pos_Y')

        orig_idx = df_sub[row_mask].index[0]
        old_type = str(df_sub.loc[orig_idx, ref_col])
        new_type = str(new_type)
        if old_type == new_type: return

        # Ensure Verified_CellType exists in both dataframes
        if 'Verified_CellType' not in self.ctx.state.current_mapped_df.columns:
            self.ctx.state.current_mapped_df['Verified_CellType'] = self.ctx.state.current_mapped_df[ref_col]
        if self.ctx.state.master_df is not None and 'Verified_CellType' not in self.ctx.state.master_df.columns:
            self.ctx.state.master_df['Verified_CellType'] = self.ctx.state.master_df[ref_col]

        self.ctx.state.current_mapped_df.loc[orig_idx, ref_col] = new_type
        self.ctx.state.current_mapped_df.loc[orig_idx, 'Verified_CellType'] = new_type

        if self.ctx.state.master_df is not None:
            m_df = self.ctx.state.master_df
            if sample_name and 'Image' in m_df.columns:
                m_mask = (m_df['Image'].astype(str) == str(sample_name)) & (m_df['Pos_X'] == pos_x) & (m_df['Pos_Y'] == pos_y)
            else:
                m_mask = (m_df['Pos_X'] == pos_x) & (m_df['Pos_Y'] == pos_y)

            if m_mask.any():
                m_idx = m_df[m_mask].index[0]
                self.ctx.state.master_df.loc[m_idx, ref_col] = new_type
                self.ctx.state.master_df.loc[m_idx, 'Verified_CellType'] = new_type

        mask = self.ctx.state.masks_map.get(sample_name)
        if mask is not None:
            mask_bool = (mask == obj_id)
            s_layers = self.ctx.state.sample_layers_map.get(sample_name, [])

            old_layer = next((l for l in s_layers if l.name.split(" [")[0] == old_type), None)
            new_layer = next((l for l in s_layers if l.name.split(" [")[0] == new_type), None)

            if old_layer is not None:
                new_data = old_layer.data.copy()
                new_data[mask_bool] = 0
                old_layer.data = new_data

            if new_layer is not None:
                new_data = new_layer.data.copy()
                new_data[mask_bool] = 1
                new_layer.data = new_data
            else:
                QMessageBox.information(
                    self.ctx.control_widget,
                    "New Phenotype Added",
                    f"Label '{new_type}' saved to memory. Reload the canvas to generate its new visual layer."
                )

        # Try to trigger a UI refresh if there's a specific method
        if hasattr(self.ctx, "_update_dashboard_metrics"):
            self.ctx._update_dashboard_metrics()
