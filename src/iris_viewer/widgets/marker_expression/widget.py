"""
iris_viewer.widgets.marker_expression.widget
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dockable tab widget for Marker Expression Profiling with embedded scientific heatmap
and interactive pop-up window inspection.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from iris_common.logging import iris_log
from iris_common.schema import find_sample_column
from iris_common.widgets.sample_scope import (
    SCOPE_ACTIVE,
    SCOPE_ALL,
    SCOPE_SELECTED,
    SampleScopeBar,
)
from iris_viewer.widgets.marker_expression.dialog import MarkerExpressionDialog
from iris_viewer.widgets.marker_expression.subset_dialog import SampleSubsetDialog
from iris_viewer.widgets.phenotyping.thresholding import resolve_marker_col
from iris_viewer.widgets.styles import (
    POPOUT_BUTTON_STYLE,
    SECTION_TITLE_STYLE,
    SHARED_DARK_THEME,
    create_section_card,
)


class IrisMarkerExpressionWidget(QWidget):
    """
    Dockable tab widget for Marker Expression Profiling with embedded scientific heatmap
    and interactive pop-up window inspection.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.df: Optional[pd.DataFrame] = None
        self.markers: List[str] = []
        self.celltype_col: str = "cell_type"
        self.popup_dialog: Optional[MarkerExpressionDialog] = None
        self.selected_subset: Set[str] = set()
        self.selected_samples: List[str] = []
        self.assignment_callback: Optional[Callable[[List[str], str], None]] = None
        self._user_customized_subset: bool = False
        self._dataset_none: bool = False

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(12)
        self.setLayout(self.main_layout)

        self.setStyleSheet(SHARED_DARK_THEME)

        self._build_ui()
        self.update_plot()

    def _get_image_col(self, df: Optional[pd.DataFrame]) -> Optional[str]:
        return find_sample_column(df)

    def _build_ui(self) -> None:
        # Configuration Card
        config_box, cb = create_section_card(
            "Marker Expression Analysis",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        # Sample scope bar (Active Sample(s) / Selected Samples dropdown / All Samples)
        self.sample_scope_bar = SampleScopeBar(
            default_mode=SCOPE_ACTIVE,
            label="Dataset:",
            compact=True,
            parent=config_box,
        )
        self.combo_dataset = self.sample_scope_bar.combo_scope
        self.sample_scope_bar.scopeChanged.connect(self._on_scope_changed)
        cb.addWidget(self.sample_scope_bar)

        # Row 1: Group By & Custom Subset Filter
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(QLabel("Group By:"))
        self.combo_group_by = QComboBox()
        self.combo_group_by.currentIndexChanged.connect(self._on_group_by_changed)
        row1.addWidget(self.combo_group_by, 1)

        self.btn_subset = QPushButton("Subsets...")
        self.btn_subset.setMinimumHeight(28)
        self.btn_subset.setToolTip("Open sample subset selection dialog")
        self.btn_subset.clicked.connect(self._open_subset_dialog)
        row1.addWidget(self.btn_subset)
        cb.addLayout(row1)

        # Row 2: Mode & Colormap
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Raw Mean Intensity", "Z-Score (Per Marker)"])
        self.combo_mode.currentIndexChanged.connect(self._on_filter_changed)
        row2.addWidget(self.combo_mode, 1)

        row2.addWidget(QLabel("Cmap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["coolwarm", "viridis", "plasma", "magma", "RdBu_r", "YlGnBu"])
        self.combo_cmap.currentIndexChanged.connect(self._on_filter_changed)
        row2.addWidget(self.combo_cmap)
        cb.addLayout(row2)

        self.main_layout.addWidget(config_box, 0)

        # Heatmap Plot Card
        self.plot_group, pg = create_section_card(
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        plot_hdr = QHBoxLayout()
        lbl_title = QLabel("Marker Expression Profile")
        lbl_title.setStyleSheet(SECTION_TITLE_STYLE)
        plot_hdr.addWidget(lbl_title)
        plot_hdr.addStretch(1)

        self.btn_popout = QPushButton("⤢ Pop-up Window")
        self.btn_popout.setToolTip("Open full-size interactive window to inspect and organize rows and columns")
        self.btn_popout.setStyleSheet(POPOUT_BUTTON_STYLE)
        self.btn_popout.setMinimumHeight(28)
        self.btn_popout.clicked.connect(self.open_popup_window)
        plot_hdr.addWidget(self.btn_popout)
        pg.addLayout(plot_hdr)

        self.figure = Figure(figsize=(6, 5))
        self.figure.patch.set_facecolor("#14161d")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(320)
        self.canvas.setToolTip("Double-click heatmap to open large pop-up window")
        self.canvas.mouseDoubleClickEvent = lambda event: self.open_popup_window()
        pg.addWidget(self.canvas, 1)

        self.main_layout.addWidget(self.plot_group, 1)

    def set_data(
        self,
        df: Optional[pd.DataFrame] = None,
        markers: Optional[List[str]] = None,
        celltype_col: Optional[str] = None,
        selected_samples: Optional[List[str]] = None,
    ) -> None:
        """Updates internal data references and refreshes the heatmap display."""
        if df is not None:
            self.df = df
        if markers is not None:
            self.markers = list(markers)
        elif self.df is not None and not self.markers:
            skip = {"pos_x", "pos_y", "objectnumber", "cell_id", "area", "eccentricity"}
            self.markers = [
                c for c in self.df.columns
                if c.lower() not in skip and pd.api.types.is_numeric_dtype(self.df[c])
            ]

        if celltype_col:
            self.celltype_col = celltype_col

        if selected_samples is not None:
            self.selected_samples = list(selected_samples)

        self._sync_sample_dropdown()
        self._sync_grouping_dropdown()
        self.update_plot()

        if self.popup_dialog is not None and self.popup_dialog.isVisible() and self.df is not None:
            self.popup_dialog.update_data(
                df=self.df,
                markers=self.markers,
                celltype_col=self.celltype_col,
                selected_samples=self.selected_samples,
            )
            if hasattr(self, 'assignment_callback') and self.assignment_callback:
                self.popup_dialog.set_assignment_callback(self.assignment_callback)
            self.popup_dialog.set_group_by(self.celltype_col)

    def _sync_grouping_dropdown(self) -> None:
        if not hasattr(self, 'combo_group_by'):
            return
        curr = self.combo_group_by.currentText() if self.combo_group_by.count() > 0 else self.celltype_col
        self.combo_group_by.blockSignals(True)
        self.combo_group_by.clear()

        candidates = []
        if self.df is not None:
            priority = ["Cluster", "cell_type", "Verified_CellType", "CellType", "Phenotype", "GROUP", "Label"]
            for col in priority:
                if col in self.df.columns and col not in candidates:
                    candidates.append(col)
            skip_terms = {"x", "y", "z", "pos_x", "pos_y", "area", "eccentricity", "objectnumber", "cell_id"}
            resolved_marker_cols = {
                c for m in self.markers if (c := resolve_marker_col(self.df, m)) is not None
            }
            for col in self.df.columns:
                if (
                    col.lower() in skip_terms
                    or col in self.markers
                    or col in resolved_marker_cols
                    or col in candidates
                ):
                    continue
                if (
                    self.df[col].dtype == 'object'
                    or self.df[col].dtype.name == 'category'
                    or self.df[col].dtype.name == 'string'
                ):
                    candidates.append(col)
                elif pd.api.types.is_integer_dtype(self.df[col]) and self.df[col].nunique() <= 60:
                    candidates.append(col)

        if not candidates:
            candidates = [self.celltype_col or "Cluster"]

        self.combo_group_by.addItems(candidates)
        idx = self.combo_group_by.findText(curr)
        if idx >= 0:
            self.combo_group_by.setCurrentIndex(idx)
        else:
            idx2 = self.combo_group_by.findText(self.celltype_col)
            self.combo_group_by.setCurrentIndex(idx2 if idx2 >= 0 else 0)
        self.celltype_col = self.combo_group_by.currentText()
        self.combo_group_by.blockSignals(False)

    def _on_group_by_changed(self) -> None:
        col = self.combo_group_by.currentText().strip()
        if col:
            self.celltype_col = col
            self.update_plot()

    def set_group_by(self, col: str) -> None:
        self.celltype_col = col
        if hasattr(self, 'combo_group_by'):
            idx = self.combo_group_by.findText(col)
            if idx >= 0:
                self.combo_group_by.blockSignals(True)
                self.combo_group_by.setCurrentIndex(idx)
                self.combo_group_by.blockSignals(False)
            else:
                self.combo_group_by.addItem(col)
                self.combo_group_by.setCurrentText(col)
        self.update_plot()

    def set_current_dataset(self, dataset_name: str, subset_samples: Optional[List[str]] = None) -> None:
        """Sets the active dataset or cohort sample filter and updates the plot."""
        self._dataset_none = False
        ds = str(dataset_name or "").strip()

        if subset_samples:
            self.selected_subset = set(subset_samples)
            self._user_customized_subset = True
            self.sample_scope_bar.set_checked_samples(list(subset_samples), emit=False)
            self.sample_scope_bar.set_scope_mode(SCOPE_SELECTED, emit=False)
        elif ds == "None":
            self._dataset_none = True
        elif ds == SCOPE_ALL or ds.lower() == "all":
            self.sample_scope_bar.set_scope_mode(SCOPE_ALL, emit=False)
        elif ds == SCOPE_ACTIVE or ds.lower().startswith("active"):
            self.sample_scope_bar.set_scope_mode(SCOPE_ACTIVE, emit=False)
        elif ds.startswith("Sample: "):
            sname = ds.replace("Sample: ", "").strip()
            self.selected_subset = {sname}
            self._user_customized_subset = True
            self.sample_scope_bar.set_checked_samples([sname], emit=False)
            self.sample_scope_bar.set_scope_mode(SCOPE_SELECTED, emit=False)
        elif ds == SCOPE_SELECTED or ds.startswith("Subset ("):
            if self.selected_subset:
                self.sample_scope_bar.set_checked_samples(list(self.selected_subset), emit=False)
            elif self.selected_samples:
                self.sample_scope_bar.set_checked_samples(list(self.selected_samples), emit=False)
            self.sample_scope_bar.set_scope_mode(SCOPE_SELECTED, emit=False)

        if self.popup_dialog is not None and self.popup_dialog.isVisible():
            self.popup_dialog.set_current_dataset(
                self.sample_scope_bar.get_scope_mode() if not self._dataset_none else "None",
                subset_samples=subset_samples or list(self.selected_subset) or None,
            )
        self.update_plot()

    def set_assignment_callback(self, cb: Callable[[List[str], str], None]) -> None:
        self.assignment_callback = cb
        if self.popup_dialog is not None:
            self.popup_dialog.set_assignment_callback(cb)

    def _sync_sample_dropdown(self) -> None:
        img_col = self._get_image_col(self.df)
        all_samples: List[str] = []
        if self.df is not None and img_col is not None:
            all_samples = list(dict.fromkeys(str(x).strip() for x in self.df[img_col].dropna().unique() if str(x).strip()))
        for s in self.selected_samples:
            s_str = str(s).strip()
            if s_str and s_str not in all_samples:
                all_samples.append(s_str)

        self.sample_scope_bar.set_samples(
            all_samples,
            active_samples=self.selected_samples,
            emit=False,
        )
        if not self._user_customized_subset and self.selected_samples:
            self.sample_scope_bar.set_checked_samples(self.selected_samples, emit=False)
            self.selected_subset = set(self.sample_scope_bar.get_checked_samples())

    def _on_scope_changed(self, mode: str, resolved_samples: List[str]) -> None:
        self._dataset_none = False
        if mode == SCOPE_SELECTED:
            self._user_customized_subset = True
            self.selected_subset = set(resolved_samples)
        iris_log(
            "IRIS-Viewer",
            "Marker Expression",
            f"Scope set to '{mode}' ({len(resolved_samples)} sample(s)).",
        )
        if self.popup_dialog is not None and self.popup_dialog.isVisible():
            self.popup_dialog.set_current_dataset(
                mode,
                subset_samples=resolved_samples if mode == SCOPE_SELECTED else None,
            )
        self.update_plot()

    def _open_subset_dialog(self) -> None:
        img_col = self._get_image_col(self.df)
        if self.df is None or img_col is None:
            QMessageBox.information(self, "No Sample Info", "Data does not contain an 'Image' or sample column.")
            return

        all_samples = sorted([str(s) for s in self.df[img_col].dropna().unique()])
        current_sel = set(self.sample_scope_bar.get_checked_samples()) or set(self.selected_subset)
        dialog = SampleSubsetDialog(all_samples, current_sel, parent=self)
        if dialog.exec_():
            self._dataset_none = False
            self._user_customized_subset = True
            self.selected_subset = set(dialog.selected_samples)
            self.sample_scope_bar.set_checked_samples(sorted(self.selected_subset), emit=False)
            self.sample_scope_bar.set_scope_mode(SCOPE_SELECTED, emit=False)
            iris_log(
                "IRIS-Viewer",
                "Marker Expression",
                f"Selected {len(self.selected_subset)}/{len(all_samples)} samples via subset dialog.",
            )
            self.update_plot()

    def _on_filter_changed(self) -> None:
        self.update_plot()

    def _render_empty(self, message: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#14161d")
        ax.text(0.5, 0.5, message, color="#9AA4B2", ha="center", va="center", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()

    def update_plot(self) -> None:
        """Renders the embedded dock heatmap."""
        if self.df is None or self.df.empty or not self.markers:
            self._render_empty("Load sample image and annotations to view marker profile.")
            return

        if getattr(self, "_dataset_none", False):
            self._render_empty("Dataset selection set to 'None'.")
            return

        df_work = self.sample_scope_bar.filter_dataframe(self.df)
        if df_work is None or df_work.empty:
            self._render_empty(
                f"No cells match scope '{self.sample_scope_bar.get_scope_mode()}'."
            )
            return

        ct_col = self.combo_group_by.currentText().strip() if hasattr(self, 'combo_group_by') and self.combo_group_by.count() > 0 else self.celltype_col
        if not ct_col or ct_col not in df_work.columns:
            for fallback in ("Cluster", "cell_type", "Verified_CellType", "CellType", "GROUP", "Phenotype", "Label"):
                if fallback in df_work.columns:
                    ct_col = fallback
                    break

        if ct_col not in df_work.columns or df_work.empty:
            self._render_empty("No grouping column available.")
            return

        self.celltype_col = ct_col

        resolved_pairs = []
        for m in self.markers:
            col = resolve_marker_col(df_work, m)
            if col is not None and pd.api.types.is_numeric_dtype(df_work[col]):
                resolved_pairs.append((m, col))
        if not resolved_pairs:
            self._render_empty("No numeric marker columns resolved.")
            return
        valid_markers = [m for m, _ in resolved_pairs]
        col_names = [col for _, col in resolved_pairs]

        if ct_col.lower() == "cluster":
            if pd.api.types.is_numeric_dtype(df_work[ct_col]):
                df_work[ct_col] = df_work[ct_col].apply(
                    lambda v: f"Cluster {int(v)}" if pd.notnull(v) and v >= 0 else ("Noise / Outlier" if v == -1 else "Unassigned")
                )
            else:
                def _fmt_cluster_val_dock(v):
                    if pd.isnull(v):
                        return "Unassigned"
                    s = str(v).strip()
                    if s.isdigit():
                        return f"Cluster {int(s)}"
                    return s
                df_work[ct_col] = df_work[ct_col].apply(_fmt_cluster_val_dock)
            df_work = df_work[df_work[ct_col] != "Unassigned"]

        if df_work.empty:
            self._render_empty("No assigned cells in selected scope.")
            return

        matrix = df_work.groupby(ct_col, observed=True)[col_names].mean().fillna(0)
        matrix.columns = valid_markers
        matrix.index = matrix.index.astype(str)
        if ct_col.lower() == "cluster":
            sorted_idx = sorted(
                matrix.index,
                key=lambda s: (0, int(re.search(r'\d+', s).group())) if re.search(r'\d+', s) else (1, s)
            )
            matrix = matrix.reindex(sorted_idx)

        is_zscore = "Z-Score" in self.combo_mode.currentText()

        if is_zscore:
            col_mean = matrix.mean(axis=0)
            col_std = matrix.std(axis=0, ddof=0).replace(0, 1e-8).fillna(1.0)
            col_std[col_std == 0] = 1e-8
            plot_values = np.nan_to_num(((matrix - col_mean) / col_std).values, nan=0.0)
            cbar_label = "Z-Score"
            finite_abs = np.abs(plot_values)[np.isfinite(plot_values)]
            max_abs = max(1.0, float(np.max(finite_abs))) if finite_abs.size > 0 else 1.0
            vmin, vmax = -min(3.0, max_abs), min(3.0, max_abs)
        else:
            plot_values = np.nan_to_num(matrix.values, nan=0.0)
            cbar_label = "Mean"
            finite_vals = plot_values[np.isfinite(plot_values)]
            if finite_vals.size > 0:
                vmin = float(np.min(finite_vals))
                vmax = float(np.max(finite_vals))
            else:
                vmin, vmax = 0.0, 1.0
            if vmin == vmax:
                vmax += 1.0

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#14161d')

        cmap = plt.get_cmap(self.combo_cmap.currentText())
        im = ax.imshow(plot_values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

        n_rows, n_cols = plot_values.shape
        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))
        ax.set_xticklabels(list(matrix.columns), rotation=45, ha='right', fontsize=9, color='white')
        ax.set_yticklabels(list(matrix.index.astype(str)), fontsize=9, color='white')

        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which='minor', color='#222630', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', bottom=False, left=False)

        cbar = self.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
        cbar.set_label(cbar_label, color='white', fontsize=9)
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

        ax.set_xlabel("Markers", color='#DDD', fontsize=9, fontweight='bold', labelpad=4)
        ax.set_ylabel(ct_col, color='#DDD', fontsize=9, fontweight='bold', labelpad=4)

        for spine in ax.spines.values():
            spine.set_color('#444')

        max_row_len = max([len(str(r)) for r in matrix.index]) if not matrix.empty else 10
        left_m = float(np.clip((max_row_len + 2) * 0.014, 0.20, 0.45))
        self.figure.subplots_adjust(left=left_m, right=0.92, bottom=0.22, top=0.94)
        self.canvas.draw()

    def open_popup_window(self) -> None:
        """Opens or activates the full-size interactive Marker Expression Dialog."""
        if self.df is None or self.df.empty:
            QMessageBox.information(self, "No Data", "Please load sample data and annotations first.")
            return

        mode = "None" if getattr(self, "_dataset_none", False) else self.sample_scope_bar.get_scope_mode()
        checked = set(self.sample_scope_bar.get_checked_samples()) or set(self.selected_subset)

        if self.popup_dialog is None:
            self.popup_dialog = MarkerExpressionDialog(
                df=self.df,
                markers=self.markers,
                celltype_col=self.celltype_col,
                current_dataset=mode,
                current_mode=self.combo_mode.currentText(),
                current_cmap=self.combo_cmap.currentText(),
                selected_subset=checked,
                selected_samples=self.selected_samples,
                assignment_callback=getattr(self, 'assignment_callback', None),
                parent=self.window(),
            )
        else:
            self.popup_dialog.update_data(
                df=self.df,
                markers=self.markers,
                celltype_col=self.celltype_col,
                selected_samples=self.selected_samples,
            )
            if hasattr(self, 'assignment_callback') and self.assignment_callback:
                self.popup_dialog.set_assignment_callback(self.assignment_callback)
            self.popup_dialog.set_group_by(self.celltype_col)
            self.popup_dialog.set_current_dataset(
                mode,
                subset_samples=list(checked) if mode == SCOPE_SELECTED and checked else None,
            )

        self.popup_dialog.show()
        self.popup_dialog.raise_()
        self.popup_dialog.activateWindow()
