"""
IRIS Common Marker Expression Profiler
======================================
Provides interactive docked widget and pop-up dialog for visualizing
and organizing cell phenotype vs. marker expression heatmaps with
hierarchical clustering dendrograms and dynamic dataset filtering.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
try:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    NavigationToolbar = None

from matplotlib.figure import Figure
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


class SampleSubsetDialog(QDialog):
    """Modal dialog allowing users to select a custom subset of samples."""

    def __init__(self, available_samples: List[str], current_selected: Set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Sample Subset")
        self.resize(380, 440)
        self.selected_samples: Set[str] = set(current_selected)

        self.setStyleSheet("""
            QDialog { background-color: #1f232b; color: #E0E0E0; font-size: 12px; }
            QPushButton {
                background-color: #2D5F8B; color: white; font-weight: bold;
                border: 1px solid #4B82B5; border-radius: 3px; padding: 5px 10px; font-size: 11px;
            }
            QPushButton:hover { background-color: #3D75A8; }
            QListWidget {
                background-color: #14161d; color: white; border: 1px solid #444;
                border-radius: 3px; padding: 4px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose samples to include in marker expression profile:"))

        btn_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_none = QPushButton("Select None")
        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        layout.addLayout(btn_row)

        self.list_widget = QListWidget()
        for s in available_samples:
            item = QListWidgetItem(s)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if s in self.selected_samples else Qt.Unchecked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        action_row = QHBoxLayout()
        action_row.addStretch()
        btn_ok = QPushButton("Apply")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(self._on_apply)
        btn_cancel.clicked.connect(self.reject)
        action_row.addWidget(btn_cancel)
        action_row.addWidget(btn_ok)
        layout.addLayout(action_row)

    def _select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Checked)

    def _select_none(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Unchecked)

    def _on_apply(self):
        self.selected_samples = {
            self.list_widget.item(i).text()
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == Qt.Checked
        }
        self.accept()


class MarkerExpressionDialog(QDialog):
    """
    High-resolution interactive pop-up window for exploring, reordering,
    hiding/showing, and exporting Marker Expression Profile heatmaps with dendrograms.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        markers: List[str],
        celltype_col: str,
        current_dataset: str = "Selected Samples",
        current_mode: str = "Raw Mean Intensity",
        current_cmap: str = "coolwarm",
        selected_subset: Optional[Set[str]] = None,
        selected_samples: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Marker Expression Profile Heatmap")
        self.resize(1300, 840)
        self.setMinimumSize(980, 600)
        self.setWindowFlags(self.windowFlags() | Qt.Window)

        self.df = df
        self.markers = list(markers)
        self.celltype_col = celltype_col
        self.current_dataset = current_dataset
        self.current_mode = current_mode
        self.current_cmap = current_cmap
        self.selected_subset: Set[str] = set(selected_subset) if selected_subset else set()
        self.selected_samples: List[str] = list(selected_samples or [])

        # Custom user order and visibility sets
        self.ordered_cell_types: List[str] = []
        self.ordered_markers: List[str] = list(markers)
        self.visible_cell_types: Set[str] = set()
        self.visible_markers: Set[str] = set(markers)
        self.cached_matrix: Optional[pd.DataFrame] = None
        self.row_linkage: Optional[np.ndarray] = None
        self.col_linkage: Optional[np.ndarray] = None
        self.ax_heatmap = None
        self.ax_col_dendro = None
        self.ax_row_dendro = None
        self._pan_start = None
        self._pan_xlim = None
        self._pan_ylim = None
        self._orig_limits = None

        self.setStyleSheet("""
            QDialog { background-color: #1a1d24; color: #E0E0E0; font-size: 12px; }
            QGroupBox { border: 1px solid #444; margin-top: 6px; font-weight: bold; color: #DDD; }
            QPushButton {
                background-color: #2D5F8B; color: white; font-weight: bold;
                border: 1px solid #4B82B5; border-radius: 3px; padding: 5px 12px; font-size: 12px;
            }
            QPushButton:hover { background-color: #3D75A8; }
            QComboBox, QSpinBox {
                background-color: #2A2A2A; color: white; border: 1px solid #555;
                border-radius: 3px; padding: 3px 6px; font-size: 12px;
            }
            QLabel { color: #DDD; font-size: 12px; }
            QCheckBox { color: #DDD; font-size: 12px; }
            QListWidget {
                background-color: #14161d; color: white; border: 1px solid #444;
                border-radius: 3px; font-size: 12px;
            }
            QListWidget::item:selected { background-color: #3A6A9A; }
        """)

        self._build_ui()
        self._sync_available_samples()
        self.recompute_and_render()

    def _get_image_col(self, df: Optional[pd.DataFrame]) -> Optional[str]:
        if df is None:
            return None
        for col in (
            "Image", "image", "Sample", "sample", "Sample_Name", "sample_name",
            "SampleID", "sample_id", "Core", "core", "TMA_Core", "tma_core",
            "Slide", "slide", "filename", "Filename", "File", "file"
        ):
            if col in df.columns:
                return col
        return None

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # ── Top Toolbar: Row 1 (Parameters & Settings) ───────────────────
        top_bar1 = QHBoxLayout()
        top_bar1.setSpacing(10)

        top_bar1.addWidget(QLabel("Dataset:"))
        self.combo_dataset = QComboBox()
        self.combo_dataset.currentIndexChanged.connect(self._on_dataset_changed)
        top_bar1.addWidget(self.combo_dataset)

        self.btn_subset = QPushButton("Filter Subsets...")
        self.btn_subset.clicked.connect(self._open_subset_dialog)
        top_bar1.addWidget(self.btn_subset)

        top_bar1.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Raw Mean Intensity", "Z-Score (Per Marker)"])
        self.combo_mode.setCurrentText(self.current_mode)
        self.combo_mode.currentTextChanged.connect(self._on_mode_changed)
        top_bar1.addWidget(self.combo_mode)

        top_bar1.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["coolwarm", "viridis", "plasma", "magma", "RdBu_r", "YlGnBu", "Spectral_r"])
        self.combo_cmap.setCurrentText(self.current_cmap)
        self.combo_cmap.currentTextChanged.connect(self._on_cmap_changed)
        top_bar1.addWidget(self.combo_cmap)

        self.chk_values = QCheckBox("Show Values")
        self.chk_values.setChecked(False)
        self.chk_values.toggled.connect(self.render_heatmap)
        top_bar1.addWidget(self.chk_values)

        self.chk_dendrograms = QCheckBox("Show Dendrograms")
        self.chk_dendrograms.setChecked(False)
        self.chk_dendrograms.toggled.connect(self._on_dendrograms_toggled)
        top_bar1.addWidget(self.chk_dendrograms)

        top_bar1.addWidget(QLabel("Font:"))
        self.spin_font = QSpinBox()
        self.spin_font.setRange(7, 20)
        self.spin_font.setValue(10)
        self.spin_font.valueChanged.connect(self.render_heatmap)
        top_bar1.addWidget(self.spin_font)

        top_bar1.addWidget(QLabel("Angle:"))
        self.spin_angle = QSpinBox()
        self.spin_angle.setRange(0, 90)
        self.spin_angle.setValue(45)
        self.spin_angle.valueChanged.connect(self.render_heatmap)
        top_bar1.addWidget(self.spin_angle)

        top_bar1.addStretch(1)
        main_layout.addLayout(top_bar1)

        # ── Top Toolbar: Row 2 (Actions & Export) ────────────────────────
        top_bar2 = QHBoxLayout()
        top_bar2.setSpacing(10)

        self.btn_toggle_drawer = QPushButton("Organize Rows/Cols")
        self.btn_toggle_drawer.setCheckable(True)
        self.btn_toggle_drawer.setChecked(True)
        self.btn_toggle_drawer.clicked.connect(self._toggle_organizer_panel)
        top_bar2.addWidget(self.btn_toggle_drawer)

        self.btn_cluster_both = QPushButton("Hierarchical Cluster Both")
        self.btn_cluster_both.setStyleSheet("""
            QPushButton {
                background-color: #5E35B1; color: white; font-weight: bold;
                border: 1px solid #7E57C2; border-radius: 3px; padding: 5px 14px;
            }
            QPushButton:hover { background-color: #6A1B9A; }
        """)
        self.btn_cluster_both.clicked.connect(self.cluster_both)
        top_bar2.addWidget(self.btn_cluster_both)

        self.btn_reset_order = QPushButton("Reset Order")
        self.btn_reset_order.clicked.connect(self._reset_order)
        top_bar2.addWidget(self.btn_reset_order)

        self.btn_zoom_in = QPushButton("Zoom In")
        self.btn_zoom_in.setToolTip("Zoom in towards heatmap center (or use Mouse Wheel)")
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        top_bar2.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QPushButton("Zoom Out")
        self.btn_zoom_out.setToolTip("Zoom out from heatmap center (or use Mouse Wheel)")
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        top_bar2.addWidget(self.btn_zoom_out)

        self.btn_reset_zoom = QPushButton("Reset View")
        self.btn_reset_zoom.setToolTip("Reset zoom and pan back to full heatmap view")
        self.btn_reset_zoom.clicked.connect(self.reset_view)
        top_bar2.addWidget(self.btn_reset_zoom)

        top_bar2.addStretch(1)

        self.btn_copy = QPushButton("Copy Figure")
        self.btn_copy.clicked.connect(self.copy_to_clipboard)
        top_bar2.addWidget(self.btn_copy)

        self.btn_save_fig = QPushButton("Export Figure...")
        self.btn_save_fig.clicked.connect(self.export_figure)
        top_bar2.addWidget(self.btn_save_fig)

        self.btn_save_csv = QPushButton("Export CSV...")
        self.btn_save_csv.clicked.connect(self.export_csv)
        top_bar2.addWidget(self.btn_save_csv)

        main_layout.addLayout(top_bar2)

        # ── Hover Details Status Bar ──────────────────────────────────────
        self.lbl_hover = QLabel("Hover over any cell in the heatmap for values. Scroll mouse wheel to zoom, click and drag to pan.")
        self.lbl_hover.setStyleSheet("color: #AAA; font-size: 12px; font-style: italic; padding: 2px 4px;")
        main_layout.addWidget(self.lbl_hover)

        # ── Main Splitter: Canvas & Reorder Organizer ─────────────────────
        self.splitter = QSplitter(Qt.Horizontal)

        # Canvas container
        canvas_container = QWidget()
        cc_layout = QVBoxLayout(canvas_container)
        cc_layout.setContentsMargins(0, 0, 0, 0)
        cc_layout.setSpacing(4)

        self.figure = Figure(figsize=(8.5, 6.5), dpi=100)
        self.figure.patch.set_facecolor('#14161d')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        if NavigationToolbar is not None:
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setStyleSheet("""
                QToolBar { background-color: #1f232b; border: 1px solid #333; spacing: 4px; padding: 2px; }
                QToolButton { background-color: transparent; border: 1px solid transparent; border-radius: 3px; padding: 3px; }
                QToolButton:hover { background-color: #3A6A9A; border: 1px solid #558; }
            """)
            cc_layout.addWidget(self.toolbar)

        cc_layout.addWidget(self.canvas, 1)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('scroll_event', self._on_scroll_zoom)

        self.splitter.addWidget(canvas_container)

        # ── Organize Drawer Panel ─────────────────────────────────────────
        self.organizer_panel = QGroupBox()
        op_layout = QVBoxLayout(self.organizer_panel)
        op_layout.setContentsMargins(10, 4, 10, 10)
        op_layout.setSpacing(6)

        title_organize = QLabel("Organize Rows & Columns")
        title_organize.setStyleSheet("color: #DDD; font-size: 13px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        op_layout.addWidget(title_organize)

        lbl_hint = QLabel("Check/uncheck to show/hide items. Drag or use buttons to reorder.")
        lbl_hint.setStyleSheet("color: #9E9E9E; font-size: 11px; font-style: italic;")
        lbl_hint.setWordWrap(True)
        op_layout.addWidget(lbl_hint)

        lists_layout = QHBoxLayout()
        lists_layout.setSpacing(10)

        # Row (Cell Types) Box
        row_box = QVBoxLayout()
        row_box.setSpacing(4)
        row_box.addWidget(QLabel("<b>Cell Types (Rows):</b>"))

        r_quick = QHBoxLayout()
        r_quick.setSpacing(4)
        btn_r_all = QPushButton("Select All")
        btn_r_none = QPushButton("Select None")
        btn_r_all.clicked.connect(lambda: self._set_all_checks(is_row=True, checked=True))
        btn_r_none.clicked.connect(lambda: self._set_all_checks(is_row=True, checked=False))
        r_quick.addWidget(btn_r_all)
        r_quick.addWidget(btn_r_none)
        row_box.addLayout(r_quick)

        self.list_rows = QListWidget()
        self.list_rows.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_rows.model().rowsMoved.connect(self._on_rows_drag_moved)
        self.list_rows.itemChanged.connect(self._on_row_item_changed)
        row_box.addWidget(self.list_rows)

        r_btn_row = QHBoxLayout()
        r_btn_row.setSpacing(4)
        btn_r_up = QPushButton("Up")
        btn_r_down = QPushButton("Down")
        btn_r_az = QPushButton("A-Z")
        btn_r_up.clicked.connect(lambda: self._move_item(self.list_rows, -1, is_row=True))
        btn_r_down.clicked.connect(lambda: self._move_item(self.list_rows, 1, is_row=True))
        btn_r_az.clicked.connect(lambda: self._sort_alphabetical(is_row=True))
        r_btn_row.addWidget(btn_r_up)
        r_btn_row.addWidget(btn_r_down)
        r_btn_row.addWidget(btn_r_az)
        row_box.addLayout(r_btn_row)
        lists_layout.addLayout(row_box)

        # Column (Markers) Box
        col_box = QVBoxLayout()
        col_box.setSpacing(4)
        col_box.addWidget(QLabel("<b>Markers (Columns):</b>"))

        c_quick = QHBoxLayout()
        c_quick.setSpacing(4)
        btn_c_all = QPushButton("Select All")
        btn_c_none = QPushButton("Select None")
        btn_c_all.clicked.connect(lambda: self._set_all_checks(is_row=False, checked=True))
        btn_c_none.clicked.connect(lambda: self._set_all_checks(is_row=False, checked=False))
        c_quick.addWidget(btn_c_all)
        c_quick.addWidget(btn_c_none)
        col_box.addLayout(c_quick)

        self.list_cols = QListWidget()
        self.list_cols.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_cols.model().rowsMoved.connect(self._on_cols_drag_moved)
        self.list_cols.itemChanged.connect(self._on_col_item_changed)
        col_box.addWidget(self.list_cols)

        c_btn_row = QHBoxLayout()
        c_btn_row.setSpacing(4)
        btn_c_up = QPushButton("Left")
        btn_c_down = QPushButton("Right")
        btn_c_az = QPushButton("A-Z")
        btn_c_up.clicked.connect(lambda: self._move_item(self.list_cols, -1, is_row=False))
        btn_c_down.clicked.connect(lambda: self._move_item(self.list_cols, 1, is_row=False))
        btn_c_az.clicked.connect(lambda: self._sort_alphabetical(is_row=False))
        c_btn_row.addWidget(btn_c_up)
        c_btn_row.addWidget(btn_c_down)
        c_btn_row.addWidget(btn_c_az)
        col_box.addLayout(c_btn_row)
        lists_layout.addLayout(col_box)

        op_layout.addLayout(lists_layout)
        self.organizer_panel.setMinimumWidth(380)

        self.splitter.addWidget(self.organizer_panel)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self.splitter, 1)

    def _toggle_organizer_panel(self) -> None:
        is_visible = self.btn_toggle_drawer.isChecked()
        self.organizer_panel.setVisible(is_visible)

    def _sync_available_samples(self) -> None:
        prev = self.combo_dataset.currentText() if self.combo_dataset.count() > 0 else self.current_dataset
        self.combo_dataset.blockSignals(True)
        self.combo_dataset.clear()
        self.combo_dataset.addItem("Selected Samples")
        self.combo_dataset.addItem("All Samples")
        self.combo_dataset.addItem("None")

        img_col = self._get_image_col(self.df)
        if self.df is not None and img_col is not None:
            samples = sorted([str(s) for s in self.df[img_col].dropna().unique()])
            for s in samples:
                self.combo_dataset.addItem(f"Sample: {s}")

        idx = self.combo_dataset.findText(prev)
        if idx >= 0:
            self.combo_dataset.setCurrentIndex(idx)
        else:
            self.combo_dataset.setCurrentIndex(0)
        self.current_dataset = self.combo_dataset.currentText()
        self.combo_dataset.blockSignals(False)

    def _open_subset_dialog(self) -> None:
        img_col = self._get_image_col(self.df)
        if self.df is None or img_col is None:
            QMessageBox.information(self, "No Sample Info", "Current data does not contain a sample or Image column.")
            return

        all_samples = sorted([str(s) for s in self.df[img_col].dropna().unique()])
        dialog = SampleSubsetDialog(all_samples, self.selected_subset, parent=self)
        if dialog.exec_():
            self.selected_subset = dialog.selected_samples
            n_sel = len(self.selected_subset)
            custom_txt = f"Subset ({n_sel} samples)"
            idx = self.combo_dataset.findText(custom_txt)
            if idx < 0:
                self.combo_dataset.addItem(custom_txt)
                idx = self.combo_dataset.count() - 1
            self.combo_dataset.setCurrentIndex(idx)
            self.current_dataset = custom_txt
            self.recompute_and_render()

    def _on_dataset_changed(self) -> None:
        self.current_dataset = self.combo_dataset.currentText()
        # Invalidate dendrogram linkages so they freshly recompute on the selected sample
        self.row_linkage = None
        self.col_linkage = None
        self.recompute_and_render()

    def _on_mode_changed(self, mode: str) -> None:
        self.current_mode = mode
        if self.chk_dendrograms.isChecked():
            self._perform_hierarchical_clustering()
        self.render_heatmap()

    def _on_cmap_changed(self, cmap: str) -> None:
        self.current_cmap = cmap
        self.render_heatmap()

    def recompute_and_render(self) -> None:
        """Filters data, computes raw mean matrix, populates order lists, and renders."""
        if self.df is None or self.df.empty or not self.markers:
            self._render_empty_canvas("No data available.")
            return

        # Filter dataset or cohort subset
        df_work = self.df.copy()
        ds = self.current_dataset
        img_col = self._get_image_col(df_work)

        def _clean_str(val):
            return re.sub(r'\.tiff?$', '', re.sub(r'_image$', '', str(val).lower().strip()))

        if ds == "None":
            self._render_empty_canvas("Dataset selection set to 'None'.")
            return
        elif ds == "Selected Samples":
            if img_col and self.selected_samples:
                clean_selected = [str(s).strip() for s in self.selected_samples]
                match_mask = df_work[img_col].astype(str).str.strip().isin(clean_selected)
                if not match_mask.any():
                    norm_targets = set(_clean_str(s) for s in clean_selected)
                    norm_cols = df_work[img_col].apply(_clean_str)
                    match_mask = norm_cols.isin(norm_targets)
                if match_mask.any():
                    df_work = df_work[match_mask]
        elif ds.startswith("Sample: "):
            sname = ds.replace("Sample: ", "").strip()
            if img_col:
                match_mask = (df_work[img_col].astype(str).str.strip() == sname)
                if not match_mask.any():
                    target_c = _clean_str(sname)
                    norm_cols = df_work[img_col].apply(_clean_str)
                    match_mask = (norm_cols == target_c)
                df_work = df_work[match_mask]
        elif ds.startswith("Subset (") and self.selected_subset:
            if img_col:
                clean_subset = [str(s).strip() for s in self.selected_subset]
                match_mask = df_work[img_col].astype(str).str.strip().isin(clean_subset)
                if not match_mask.any():
                    norm_targets = set(_clean_str(s) for s in clean_subset)
                    norm_cols = df_work[img_col].apply(_clean_str)
                    match_mask = norm_cols.isin(norm_targets)
                if match_mask.any():
                    df_work = df_work[match_mask]

        if df_work.empty:
            self._render_empty_canvas(f"No cells found for selection: {ds}")
            return

        # Identify cell type column
        ct_col = self.celltype_col
        if ct_col not in df_work.columns:
            for fallback in ("cell_type", "Verified_CellType", "CellType", "GROUP", "Phenotype", "Label"):
                if fallback in df_work.columns:
                    ct_col = fallback
                    break

        if ct_col not in df_work.columns:
            self._render_empty_canvas("Cell type column not found in data.")
            return

        # Filter valid numeric markers
        valid_markers = [m for m in self.markers if m in df_work.columns and pd.api.types.is_numeric_dtype(df_work[m])]
        if not valid_markers:
            self._render_empty_canvas("No valid numeric marker columns found.")
            return

        # Aggregate expression across phenotypes by mean
        raw_matrix = df_work.groupby(ct_col)[valid_markers].mean()
        raw_matrix = raw_matrix.dropna(how='all').fillna(0)

        self.cached_matrix = raw_matrix

        # Sync ordering and visibility
        cell_types = list(raw_matrix.index.astype(str))
        common_types = [ct for ct in self.ordered_cell_types if ct in cell_types]
        new_types = [ct for ct in cell_types if ct not in common_types]
        self.ordered_cell_types = common_types + new_types

        if not self.visible_cell_types:
            self.visible_cell_types = set(cell_types)
        else:
            self.visible_cell_types.update(new_types)

        if not self.ordered_markers or set(self.ordered_markers) != set(valid_markers):
            self.ordered_markers = valid_markers
            self.visible_markers = set(valid_markers)

        if not self.visible_markers:
            self.visible_markers = set(self.ordered_markers)

        if self.chk_dendrograms.isChecked():
            self._perform_hierarchical_clustering()
        else:
            self._populate_list_widgets()
            self.render_heatmap()

        # Display active dataset stats in hover label
        self.lbl_hover.setText(f"Showing {ds} ({len(df_work):,} cells across {len(cell_types)} phenotypes). Hover for values.")

    def _populate_list_widgets(self) -> None:
        self.list_rows.blockSignals(True)
        self.list_rows.clear()
        for ct in self.ordered_cell_types:
            item = QListWidgetItem(ct)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setCheckState(Qt.Checked if ct in self.visible_cell_types else Qt.Unchecked)
            self.list_rows.addItem(item)
        self.list_rows.blockSignals(False)

        self.list_cols.blockSignals(True)
        self.list_cols.clear()
        for m in self.ordered_markers:
            item = QListWidgetItem(m)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setCheckState(Qt.Checked if m in self.visible_markers else Qt.Unchecked)
            self.list_cols.addItem(item)
        self.list_cols.blockSignals(False)

    def _set_all_checks(self, is_row: bool, checked: bool) -> None:
        lw = self.list_rows if is_row else self.list_cols
        target_set = self.visible_cell_types if is_row else self.visible_markers
        lw.blockSignals(True)
        for i in range(lw.count()):
            item = lw.item(i)
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            if checked:
                target_set.add(item.text())
            else:
                target_set.discard(item.text())
        lw.blockSignals(False)
        if self.chk_dendrograms.isChecked():
            self._perform_hierarchical_clustering()
        self.render_heatmap()

    def _on_row_item_changed(self, item: QListWidgetItem) -> None:
        if item.checkState() == Qt.Checked:
            self.visible_cell_types.add(item.text())
        else:
            self.visible_cell_types.discard(item.text())
        if self.chk_dendrograms.isChecked():
            self._perform_hierarchical_clustering()
        self.render_heatmap()

    def _on_col_item_changed(self, item: QListWidgetItem) -> None:
        if item.checkState() == Qt.Checked:
            self.visible_markers.add(item.text())
        else:
            self.visible_markers.discard(item.text())
        if self.chk_dendrograms.isChecked():
            self._perform_hierarchical_clustering()
        self.render_heatmap()

    def _on_rows_drag_moved(self) -> None:
        self.ordered_cell_types = [self.list_rows.item(i).text() for i in range(self.list_rows.count())]
        self.row_linkage = None
        self.chk_dendrograms.setChecked(False)
        self.render_heatmap()

    def _on_cols_drag_moved(self) -> None:
        self.ordered_markers = [self.list_cols.item(i).text() for i in range(self.list_cols.count())]
        self.col_linkage = None
        self.chk_dendrograms.setChecked(False)
        self.render_heatmap()

    def _move_item(self, list_widget: QListWidget, direction: int, is_row: bool) -> None:
        curr_row = list_widget.currentRow()
        if curr_row < 0:
            return
        new_row = curr_row + direction
        if 0 <= new_row < list_widget.count():
            item = list_widget.takeItem(curr_row)
            list_widget.insertItem(new_row, item)
            list_widget.setCurrentRow(new_row)
            if is_row:
                self._on_rows_drag_moved()
            else:
                self._on_cols_drag_moved()

    def _sort_alphabetical(self, is_row: bool) -> None:
        if is_row:
            self.ordered_cell_types.sort()
            self.row_linkage = None
            self.chk_dendrograms.setChecked(False)
            self._populate_list_widgets()
            self.render_heatmap()
        else:
            self.ordered_markers.sort()
            self.col_linkage = None
            self.chk_dendrograms.setChecked(False)
            self._populate_list_widgets()
            self.render_heatmap()

    def _perform_hierarchical_clustering(self) -> None:
        """
        Computes hierarchical clustering on currently visible rows and columns,
        reorders them to match exact dendrogram leaf positions (top-to-bottom and left-to-right),
        and caches the linkage matrices for rendering.
        """
        if self.cached_matrix is None or self.cached_matrix.empty:
            return

        safe_rows = [r for r in self.ordered_cell_types if r in self.cached_matrix.index and r in self.visible_cell_types]
        safe_cols = [c for c in self.ordered_markers if c in self.cached_matrix.columns and c in self.visible_markers]
        if len(safe_rows) < 2 or len(safe_cols) < 2:
            self.row_linkage = None
            self.col_linkage = None
            self.chk_dendrograms.setChecked(False)
            self.lbl_hover.setText("At least 2 rows and 2 columns must be visible to run hierarchical clustering.")
            return

        try:
            from scipy.cluster import hierarchy
            mat_df = self.cached_matrix.loc[safe_rows, safe_cols].copy()
            is_zscore = "Z-Score" in self.current_mode
            if is_zscore:
                col_mean = mat_df.mean(axis=0)
                col_std = mat_df.std(axis=0)
                col_std[col_std == 0] = 1e-8
                mat = ((mat_df - col_mean) / col_std).values
            else:
                mat = mat_df.values

            # Cluster rows
            row_link = hierarchy.linkage(mat, method='average', metric='euclidean')
            dendro_r = hierarchy.dendrogram(row_link, no_plot=True)
            ordered_rows = [safe_rows[i] for i in dendro_r['leaves']]
            remaining_rows = [r for r in self.ordered_cell_types if r not in safe_rows]
            self.ordered_cell_types = ordered_rows + remaining_rows
            self.row_linkage = row_link

            # Cluster columns
            col_link = hierarchy.linkage(mat.T, method='average', metric='euclidean')
            dendro_c = hierarchy.dendrogram(col_link, no_plot=True)
            ordered_cols = [safe_cols[i] for i in dendro_c['leaves']]
            remaining_cols = [c for c in self.ordered_markers if c not in safe_cols]
            self.ordered_markers = ordered_cols + remaining_cols
            self.col_linkage = col_link

            self._populate_list_widgets()
            self.lbl_hover.setText("Hierarchical clustering applied. Row dendrogram shown on right, column dendrogram on top.")
        except Exception as exc:
            self.row_linkage = None
            self.col_linkage = None
            self.chk_dendrograms.setChecked(False)
            self.lbl_hover.setText(f"Clustering fallback applied: {exc}")

    def _on_dendrograms_toggled(self, checked: bool) -> None:
        if checked:
            if self.row_linkage is None or self.col_linkage is None:
                self._perform_hierarchical_clustering()
        self.render_heatmap()

    def cluster_both(self) -> None:
        """Applies hierarchical clustering to order both rows and columns and generates dendrograms."""
        self.chk_dendrograms.blockSignals(True)
        self.chk_dendrograms.setChecked(True)
        self.chk_dendrograms.blockSignals(False)
        self._perform_hierarchical_clustering()
        self.render_heatmap()

    def _reset_order(self) -> None:
        if self.cached_matrix is not None:
            self.ordered_cell_types = list(self.cached_matrix.index.astype(str))
            self.ordered_markers = list(self.cached_matrix.columns.astype(str))
            self.visible_cell_types = set(self.ordered_cell_types)
            self.visible_markers = set(self.ordered_markers)
            self.row_linkage = None
            self.col_linkage = None
            self.chk_dendrograms.setChecked(False)
            self._populate_list_widgets()
            self.render_heatmap()
            self.lbl_hover.setText("Reset order and visibility to default.")

    def _render_empty_canvas(self, msg: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#14161d')
        ax.text(0.5, 0.5, msg, color='#888', ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#333')
        self.canvas.draw()

    def render_heatmap(self) -> None:
        """
        Renders scientific heatmap with cell types on Y, markers on X,
        column dendrogram on the top, and row dendrogram on the right.
        Uses a locked GridSpec to guarantee 100% pixel-perfect leaf alignment.
        """
        if self.cached_matrix is None or self.cached_matrix.empty:
            return

        safe_rows = [r for r in self.ordered_cell_types if r in self.cached_matrix.index and r in self.visible_cell_types]
        safe_cols = [c for c in self.ordered_markers if c in self.cached_matrix.columns and c in self.visible_markers]
        if not safe_rows or not safe_cols:
            self._render_empty_canvas("No rows or columns selected for display. Use the organizer to select items.")
            return

        matrix = self.cached_matrix.loc[safe_rows, safe_cols].copy()
        is_zscore = "Z-Score" in self.current_mode

        if is_zscore:
            col_mean = matrix.mean(axis=0)
            col_std = matrix.std(axis=0)
            col_std[col_std == 0] = 1e-8
            plot_values = ((matrix - col_mean) / col_std).values
            cbar_label = "Z-Score (Per Marker)"
            max_abs = max(1.0, float(np.nanmax(np.abs(plot_values))))
            vmin, vmax = -min(3.0, max_abs), min(3.0, max_abs)
        else:
            plot_values = matrix.values
            cbar_label = "Mean Intensity"
            vmin = float(np.nanmin(plot_values))
            vmax = float(np.nanmax(plot_values))
            if vmin == vmax:
                vmax += 1.0

        n_rows, n_cols = plot_values.shape
        show_dendro = (
            self.chk_dendrograms.isChecked()
            and n_rows >= 2
            and n_cols >= 2
        )

        if show_dendro:
            if (
                self.row_linkage is None
                or self.row_linkage.shape[0] != n_rows - 1
                or self.col_linkage is None
                or self.col_linkage.shape[0] != n_cols - 1
            ):
                self._perform_hierarchical_clustering()
                safe_rows = [r for r in self.ordered_cell_types if r in self.cached_matrix.index and r in self.visible_cell_types]
                safe_cols = [c for c in self.ordered_markers if c in self.cached_matrix.columns and c in self.visible_markers]
                matrix = self.cached_matrix.loc[safe_rows, safe_cols].copy()
                if is_zscore:
                    col_mean = matrix.mean(axis=0)
                    col_std = matrix.std(axis=0)
                    col_std[col_std == 0] = 1e-8
                    plot_values = ((matrix - col_mean) / col_std).values
                else:
                    plot_values = matrix.values
                n_rows, n_cols = plot_values.shape

            show_dendro = (
                self.row_linkage is not None
                and self.col_linkage is not None
                and self.row_linkage.shape[0] == n_rows - 1
                and self.col_linkage.shape[0] == n_cols - 1
            )

        self.figure.clear()
        cmap = plt.get_cmap(self.current_cmap)

        # Dynamically size left and bottom margins based on maximum row and col label lengths
        import math
        max_row_len = max([len(str(r)) for r in safe_rows]) if safe_rows else 10
        max_col_len = max([len(str(c)) for c in safe_cols]) if safe_cols else 10
        fs = self.spin_font.value()
        fig_w, fig_h = self.figure.get_size_inches()

        # Dynamic left margin: each char at font size fs takes approx (fs * 0.58 / 72) inches
        char_w_pts = fs * 0.58
        row_label_w_in = (max_row_len + 3.0) * (char_w_pts / 72.0)
        left_margin = float(np.clip(row_label_w_in / fig_w, 0.16, 0.44))

        angle = self.spin_angle.value()
        rad = math.radians(max(15, angle))
        col_label_h_in = (max_col_len * math.sin(rad) + 2.5) * (char_w_pts / 72.0)
        bottom_margin = float(np.clip(col_label_h_in / fig_h, 0.12, 0.36))

        if show_dendro and self.row_linkage is not None and self.col_linkage is not None:
            # 2 rows x 3 columns:
            # (0, 0): Column dendrogram (above heatmap)
            # (0, 1): Blank spacer
            # (0, 2): Blank spacer
            # (1, 0): Heatmap (cells align 1-to-1 with column dendrogram)
            # (1, 1): Row dendrogram on the RIGHT (aligns 1-to-1 with rows)
            # (1, 2): Dedicated colorbar column
            gs = gridspec.GridSpec(
                2, 3,
                figure=self.figure,
                left=left_margin,
                right=0.96,
                bottom=bottom_margin,
                top=0.96,
                width_ratios=[0.80, 0.16, 0.04],
                height_ratios=[0.14, 0.86],
                wspace=0.015,
                hspace=0.015,
            )
            ax_col_dendro = self.figure.add_subplot(gs[0, 0])
            ax_blank1 = self.figure.add_subplot(gs[0, 1])
            ax_blank1.axis('off')
            ax_blank1.set_facecolor('#14161d')

            ax_blank2 = self.figure.add_subplot(gs[0, 2])
            ax_blank2.axis('off')
            ax_blank2.set_facecolor('#14161d')

            ax = self.figure.add_subplot(gs[1, 0])
            ax_row_dendro = self.figure.add_subplot(gs[1, 1])
            ax_cbar = self.figure.add_subplot(gs[1, 2])

            from scipy.cluster import hierarchy
            # Top Column Dendrogram: spans 0 to 10*n_cols matching column 0..n_cols-1
            hierarchy.dendrogram(
                self.col_linkage,
                orientation='top',
                ax=ax_col_dendro,
                no_labels=True,
                color_threshold=0,
                above_threshold_color='#64B5F6',
            )
            ax_col_dendro.set_facecolor('#14161d')
            ax_col_dendro.set_xlim(0, 10 * n_cols)
            ax_col_dendro.axis('off')

            # Right Row Dendrogram: orientation='right' puts leaves on left, tree extending right
            # Inverted y-limits (10*n_rows, 0) align leaf 0 at the top with row 0 of imshow!
            hierarchy.dendrogram(
                self.row_linkage,
                orientation='right',
                ax=ax_row_dendro,
                no_labels=True,
                color_threshold=0,
                above_threshold_color='#81C784',
            )
            ax_row_dendro.set_facecolor('#14161d')
            ax_row_dendro.set_ylim(10 * n_rows, 0)
            ax_row_dendro.axis('off')

            self.ax_col_dendro = ax_col_dendro
            self.ax_row_dendro = ax_row_dendro
        else:
            gs = gridspec.GridSpec(
                1, 2,
                figure=self.figure,
                left=left_margin,
                right=0.96,
                bottom=bottom_margin,
                top=0.96,
                width_ratios=[0.96, 0.04],
                wspace=0.02,
            )
            ax = self.figure.add_subplot(gs[0, 0])
            ax_cbar = self.figure.add_subplot(gs[0, 1])
            self.ax_col_dendro = None
            self.ax_row_dendro = None

        self.ax_heatmap = ax
        ax.set_facecolor('#14161d')
        im = ax.imshow(plot_values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels(safe_cols, rotation=self.spin_angle.value(), ha='right', fontsize=self.spin_font.value(), color='white')
        ax.set_yticklabels(safe_rows, fontsize=self.spin_font.value(), color='white')

        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which='minor', color='#222630', linestyle='-', linewidth=0.75)
        ax.tick_params(which='minor', bottom=False, left=False)

        # Values overlay if toggled
        if self.chk_values.isChecked() and n_rows <= 35 and n_cols <= 40:
            font_sz = max(7, self.spin_font.value() - 2)
            for r in range(n_rows):
                for c in range(n_cols):
                    val = plot_values[r, c]
                    txt = f"{val:.2f}" if is_zscore else f"{val:.1f}"
                    norm_v = (val - vmin) / max(1e-8, (vmax - vmin))
                    try:
                        rgba = cmap(norm_v)
                        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        tc = "black" if lum > 0.55 else "white"
                    except Exception:
                        tc = "white"
                    ax.text(c, r, txt, ha='center', va='center', fontsize=font_sz, color=tc, fontweight='bold')

        # Dedicated colorbar axis
        cbar = self.figure.colorbar(im, cax=ax_cbar)
        cbar.set_label(cbar_label, color='white', fontsize=self.spin_font.value())
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

        ax.set_xlabel("Markers", color='#DDD', fontsize=self.spin_font.value() + 1, fontweight='bold', labelpad=8)
        ax.set_ylabel("Cell Types", color='#DDD', fontsize=self.spin_font.value() + 1, fontweight='bold', labelpad=8)

        for spine in ax.spines.values():
            spine.set_color('#444')

        # Store metadata for hover interaction
        self.hover_info = {
            "rows": safe_rows,
            "cols": safe_cols,
            "raw": matrix.values,
            "plot": plot_values,
            "is_zscore": is_zscore,
        }

        # Track default full heatmap bounds for Reset View
        self._orig_limits = ((-0.5, n_cols - 0.5), (n_rows - 0.5, -0.5))

        # Auto-sync dendrograms whenever heatmap axis bounds change
        ax.callbacks.connect('xlim_changed', lambda a: self._sync_dendrogram_axes())
        ax.callbacks.connect('ylim_changed', lambda a: self._sync_dendrogram_axes())

        self.canvas.draw()

    def _sync_dendrogram_axes(self) -> None:
        """Synchronizes top and right dendrograms with the current heatmap axis limits."""
        if self.ax_heatmap is None:
            return
        cur_xlim = self.ax_heatmap.get_xlim()
        cur_ylim = self.ax_heatmap.get_ylim()

        if getattr(self, "ax_col_dendro", None) is not None:
            d_x0 = (cur_xlim[0] + 0.5) * 10.0
            d_x1 = (cur_xlim[1] + 0.5) * 10.0
            self.ax_col_dendro.set_xlim(d_x0, d_x1)

        if getattr(self, "ax_row_dendro", None) is not None:
            d_y0 = (cur_ylim[0] + 0.5) * 10.0
            d_y1 = (cur_ylim[1] + 0.5) * 10.0
            self.ax_row_dendro.set_ylim(d_y0, d_y1)

    def _zoom_step(self, scale: float) -> None:
        """Zooms into or out of the heatmap center by scale factor."""
        if self.ax_heatmap is None:
            return
        cur_xlim = self.ax_heatmap.get_xlim()
        cur_ylim = self.ax_heatmap.get_ylim()

        cx = (cur_xlim[0] + cur_xlim[1]) / 2.0
        cy = (cur_ylim[0] + cur_ylim[1]) / 2.0
        half_w = (cur_xlim[1] - cur_xlim[0]) * scale / 2.0
        half_h = (cur_ylim[1] - cur_ylim[0]) * scale / 2.0

        self.ax_heatmap.set_xlim(cx - half_w, cx + half_w)
        self.ax_heatmap.set_ylim(cy - half_h, cy + half_h)
        self._sync_dendrogram_axes()
        self.canvas.draw_idle()

    def zoom_in(self) -> None:
        self._zoom_step(0.8)

    def zoom_out(self) -> None:
        self._zoom_step(1.25)

    def reset_view(self) -> None:
        if self.ax_heatmap is not None and getattr(self, "_orig_limits", None) is not None:
            self.ax_heatmap.set_xlim(self._orig_limits[0])
            self.ax_heatmap.set_ylim(self._orig_limits[1])
            self._sync_dendrogram_axes()
            self.canvas.draw_idle()

    def _on_scroll_zoom(self, event) -> None:
        """Handles mouse wheel zooming centered at the cursor position."""
        if self.ax_heatmap is None or event.inaxes != self.ax_heatmap:
            return
        if event.xdata is None or event.ydata is None:
            return

        scale = 0.85 if event.button == "up" else 1.18
        cur_xlim = self.ax_heatmap.get_xlim()
        cur_ylim = self.ax_heatmap.get_ylim()

        xd, yd = event.xdata, event.ydata
        new_w = (cur_xlim[1] - cur_xlim[0]) * scale
        new_h = (cur_ylim[1] - cur_ylim[0]) * scale

        frac_x = (xd - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0]) if cur_xlim[1] != cur_xlim[0] else 0.5
        frac_y = (yd - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0]) if cur_ylim[1] != cur_ylim[0] else 0.5

        self.ax_heatmap.set_xlim(xd - frac_x * new_w, xd + (1.0 - frac_x) * new_w)
        self.ax_heatmap.set_ylim(yd - frac_y * new_h, yd + (1.0 - frac_y) * new_h)
        self._sync_dendrogram_axes()
        self.canvas.draw_idle()

    def _on_mouse_press(self, event) -> None:
        """Initiates click-and-drag panning on the heatmap."""
        if self.ax_heatmap is not None and event.inaxes == self.ax_heatmap and event.button in (1, 2, 3):
            self._pan_start = (event.x, event.y)
            self._pan_xlim = self.ax_heatmap.get_xlim()
            self._pan_ylim = self.ax_heatmap.get_ylim()

    def _on_mouse_release(self, event) -> None:
        """Ends click-and-drag panning."""
        self._pan_start = None

    def _on_mouse_motion(self, event) -> None:
        """Handles panning motion or hover inspection."""
        if (
            getattr(self, "_pan_start", None) is not None
            and self.ax_heatmap is not None
            and event.x is not None
            and event.y is not None
        ):
            inv = self.ax_heatmap.transData.inverted()
            p0 = inv.transform(self._pan_start)
            p1 = inv.transform((event.x, event.y))
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]

            self.ax_heatmap.set_xlim(self._pan_xlim[0] - dx, self._pan_xlim[1] - dx)
            self.ax_heatmap.set_ylim(self._pan_ylim[0] - dy, self._pan_ylim[1] - dy)
            self._sync_dendrogram_axes()
            self.canvas.draw_idle()
            return

        self._on_mouse_move(event)

    def _on_mouse_move(self, event) -> None:
        if (
            event.inaxes is None
            or not hasattr(self, "ax_heatmap")
            or event.inaxes != self.ax_heatmap
            or not hasattr(self, "hover_info")
        ):
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        c = int(round(x))
        r = int(round(y))
        rows = self.hover_info["rows"]
        cols = self.hover_info["cols"]

        if 0 <= r < len(rows) and 0 <= c < len(cols):
            ct = rows[r]
            m = cols[c]
            raw_v = self.hover_info["raw"][r, c]
            plot_v = self.hover_info["plot"][r, c]
            if self.hover_info["is_zscore"]:
                z_txt = f"{plot_v:+.2f}"
                self.lbl_hover.setText(
                    f"<b>Cell Type:</b> {ct} &nbsp;|&nbsp; <b>Marker:</b> {m} &nbsp;|&nbsp; "
                    f"<b>Z-Score:</b> <span style='color:#00E5FF;'>{z_txt}</span> (Raw Mean: {raw_v:.2f})"
                )
            else:
                self.lbl_hover.setText(
                    f"<b>Cell Type:</b> {ct} &nbsp;|&nbsp; <b>Marker:</b> {m} &nbsp;|&nbsp; "
                    f"<b>Mean Intensity:</b> <span style='color:#76FF03;'>{raw_v:.2f}</span>"
                )

    def copy_to_clipboard(self) -> None:
        try:
            pixmap = self.canvas.grab()
            QApplication.clipboard().setPixmap(pixmap)
            self.lbl_hover.setText("Copied heatmap figure to clipboard!")
            self.lbl_hover.setStyleSheet("color: #76FF03; font-size: 12px; font-weight: bold; padding: 2px 4px;")
        except Exception as exc:
            QMessageBox.warning(self, "Copy Failed", f"Could not copy to clipboard: {exc}")

    def export_figure(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Marker Heatmap Figure",
            "marker_expression_heatmap.pdf",
            "Vector PDF (*.pdf);;Scalable Vector Graphics (*.svg);;High-Res PNG (*.png)",
        )
        if filepath:
            self.figure.savefig(
                filepath,
                dpi=300,
                bbox_inches='tight',
                facecolor=self.figure.get_facecolor(),
            )
            QMessageBox.information(self, "Export Complete", f"Figure saved to {Path(filepath).name}.")

    def export_csv(self) -> None:
        if self.cached_matrix is None:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Marker Expression Matrix",
            "marker_expression_matrix.csv",
            "CSV Document (*.csv)",
        )
        if filepath:
            safe_rows = [r for r in self.ordered_cell_types if r in self.cached_matrix.index and r in self.visible_cell_types]
            safe_cols = [c for c in self.ordered_markers if c in self.cached_matrix.columns and c in self.visible_markers]
            out_df = self.cached_matrix.loc[safe_rows, safe_cols]
            out_df.to_csv(filepath)
            QMessageBox.information(self, "Export Complete", f"Matrix saved to {Path(filepath).name}.")

    def update_data(
        self,
        df: pd.DataFrame,
        markers: List[str],
        celltype_col: str,
        selected_samples: Optional[List[str]] = None,
    ) -> None:
        self.df = df
        self.markers = list(markers)
        self.celltype_col = celltype_col
        if selected_samples is not None:
            self.selected_samples = list(selected_samples)
        self._sync_available_samples()
        self.recompute_and_render()


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

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)
        self.setLayout(self.main_layout)

        self.setStyleSheet("""
            QWidget { background-color: #1f232b; color: #E0E0E0; font-size: 12px; }
            QGroupBox { border: 1px solid #444; border-radius: 8px; margin-top: 6px; padding-top: 6px; }
            QLabel { color: #DDD; }
            QSpinBox, QComboBox {
                background-color: #2A2A2A; color: white; border: 1px solid #555;
                border-radius: 4px; padding: 2px 4px; height: 22px;
            }
            QPushButton {
                background-color: #3A6A9A; color: white; font-weight: bold;
                border: 1px solid #558; border-radius: 4px; padding: 5px;
            }
            QPushButton:hover { background-color: #4A7FBA; }
        """)

        self._build_ui()
        self.update_plot()

    def _get_image_col(self, df: Optional[pd.DataFrame]) -> Optional[str]:
        if df is None:
            return None
        for col in (
            "Image", "image", "Sample", "sample", "Sample_Name", "sample_name",
            "SampleID", "sample_id", "Core", "core", "TMA_Core", "tma_core",
            "Slide", "slide", "filename", "Filename", "File", "file"
        ):
            if col in df.columns:
                return col
        return None

    def _build_ui(self) -> None:
        # ── Configuration Box ─────────────────────────────────────────────
        config_box = QGroupBox()
        cb = QVBoxLayout()
        cb.setContentsMargins(10, 3, 10, 10)
        cb.setSpacing(6)
        config_box.setLayout(cb)

        title = QLabel("Marker Expression Analysis")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        cb.addWidget(title)

        # Row 1: Dataset & Filter
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Dataset:"))
        self.combo_dataset = QComboBox()
        self.combo_dataset.addItem("Selected Samples")
        self.combo_dataset.addItem("All Samples")
        self.combo_dataset.addItem("None")
        self.combo_dataset.currentIndexChanged.connect(self._on_filter_changed)
        row1.addWidget(self.combo_dataset, 1)

        self.btn_subset = QPushButton("Filter Subsets...")
        self.btn_subset.setToolTip("Select custom subset of samples")
        self.btn_subset.clicked.connect(self._open_subset_dialog)
        row1.addWidget(self.btn_subset)
        cb.addLayout(row1)

        # Row 2: Mode & Colormap
        row2 = QHBoxLayout()
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

        # ── Heatmap Plot Group ────────────────────────────────────────────
        self.plot_group = QGroupBox()
        pg = QVBoxLayout()
        pg.setContentsMargins(6, 6, 6, 6)
        pg.setSpacing(6)
        self.plot_group.setLayout(pg)

        plot_hdr = QHBoxLayout()
        lbl_title = QLabel("Marker Expression Profile")
        lbl_title.setStyleSheet("font-weight: bold; color: #EEE; font-size: 13px;")
        plot_hdr.addWidget(lbl_title)
        plot_hdr.addStretch(1)

        # Enlargement symbol on pop-up window button as requested
        self.btn_popout = QPushButton("⤢ Pop-up Window")
        self.btn_popout.setToolTip("Open full-size interactive window to inspect and organize rows and columns")
        self.btn_popout.setStyleSheet("""
            QPushButton {
                background-color: #2D5F8B; color: white; font-weight: bold;
                border: 1px solid #4B82B5; border-radius: 3px; padding: 3px 8px; font-size: 11px;
            }
            QPushButton:hover { background-color: #3D75A8; }
        """)
        self.btn_popout.clicked.connect(self.open_popup_window)
        plot_hdr.addWidget(self.btn_popout)
        pg.addLayout(plot_hdr)

        self.figure = Figure(figsize=(6, 5))
        self.figure.patch.set_facecolor('#1f232b')
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
        self.update_plot()

        if self.popup_dialog is not None and self.popup_dialog.isVisible() and self.df is not None:
            self.popup_dialog.update_data(
                df=self.df,
                markers=self.markers,
                celltype_col=self.celltype_col,
                selected_samples=self.selected_samples,
            )

    def _sync_sample_dropdown(self) -> None:
        curr = self.combo_dataset.currentText() if self.combo_dataset.count() > 0 else "Selected Samples"
        self.combo_dataset.blockSignals(True)
        self.combo_dataset.clear()
        self.combo_dataset.addItem("Selected Samples")
        self.combo_dataset.addItem("All Samples")
        self.combo_dataset.addItem("None")

        img_col = self._get_image_col(self.df)
        if self.df is not None and img_col is not None:
            for s in sorted([str(x) for x in self.df[img_col].dropna().unique()]):
                self.combo_dataset.addItem(f"Sample: {s}")

        idx = self.combo_dataset.findText(curr)
        self.combo_dataset.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_dataset.blockSignals(False)

    def _open_subset_dialog(self) -> None:
        img_col = self._get_image_col(self.df)
        if self.df is None or img_col is None:
            QMessageBox.information(self, "No Sample Info", "Data does not contain an 'Image' or sample column.")
            return

        all_samples = sorted([str(s) for s in self.df[img_col].dropna().unique()])
        dialog = SampleSubsetDialog(all_samples, self.selected_subset, parent=self)
        if dialog.exec_():
            self.selected_subset = dialog.selected_samples
            n_sel = len(self.selected_subset)
            custom_txt = f"Subset ({n_sel} samples)"
            idx = self.combo_dataset.findText(custom_txt)
            if idx < 0:
                self.combo_dataset.addItem(custom_txt)
                idx = self.combo_dataset.count() - 1
            self.combo_dataset.setCurrentIndex(idx)
            self.update_plot()

    def _on_filter_changed(self) -> None:
        self.update_plot()

    def update_plot(self) -> None:
        """Renders the embedded dock heatmap."""
        if self.df is None or self.df.empty or not self.markers:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#14161d')
            ax.text(0.5, 0.5, "Load sample image and annotations to view marker profile.", color='#888', ha='center', va='center', fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            return

        ds = self.combo_dataset.currentText()
        df_work = self.df.copy()
        img_col = self._get_image_col(df_work)

        def _clean_str(val):
            return re.sub(r'\.tiff?$', '', re.sub(r'_image$', '', str(val).lower().strip()))

        if ds == "None":
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#14161d')
            ax.text(0.5, 0.5, "Dataset selection set to 'None'.", color='#888', ha='center', va='center', fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            return
        elif ds == "Selected Samples":
            if img_col and self.selected_samples:
                clean_selected = [str(s).strip() for s in self.selected_samples]
                match_mask = df_work[img_col].astype(str).str.strip().isin(clean_selected)
                if not match_mask.any():
                    norm_targets = set(_clean_str(s) for s in clean_selected)
                    norm_cols = df_work[img_col].apply(_clean_str)
                    match_mask = norm_cols.isin(norm_targets)
                if match_mask.any():
                    df_work = df_work[match_mask]
        elif ds.startswith("Sample: "):
            sname = ds.replace("Sample: ", "").strip()
            if img_col:
                match_mask = (df_work[img_col].astype(str).str.strip() == sname)
                if not match_mask.any():
                    target_c = _clean_str(sname)
                    norm_cols = df_work[img_col].apply(_clean_str)
                    match_mask = (norm_cols == target_c)
                df_work = df_work[match_mask]
        elif ds.startswith("Subset (") and self.selected_subset:
            if img_col:
                clean_subset = [str(s).strip() for s in self.selected_subset]
                match_mask = df_work[img_col].astype(str).str.strip().isin(clean_subset)
                if not match_mask.any():
                    norm_targets = set(_clean_str(s) for s in clean_subset)
                    norm_cols = df_work[img_col].apply(_clean_str)
                    match_mask = norm_cols.isin(norm_targets)
                if match_mask.any():
                    df_work = df_work[match_mask]

        ct_col = self.celltype_col
        if ct_col not in df_work.columns:
            for fallback in ("cell_type", "Verified_CellType", "CellType", "GROUP", "Phenotype", "Label"):
                if fallback in df_work.columns:
                    ct_col = fallback
                    break

        if ct_col not in df_work.columns or df_work.empty:
            return

        valid_markers = [m for m in self.markers if m in df_work.columns and pd.api.types.is_numeric_dtype(df_work[m])]
        if not valid_markers:
            return

        matrix = df_work.groupby(ct_col)[valid_markers].mean().fillna(0)
        is_zscore = "Z-Score" in self.combo_mode.currentText()

        if is_zscore:
            col_mean = matrix.mean(axis=0)
            col_std = matrix.std(axis=0)
            col_std[col_std == 0] = 1e-8
            plot_values = ((matrix - col_mean) / col_std).values
            cbar_label = "Z-Score"
            max_abs = max(1.0, float(np.nanmax(np.abs(plot_values))))
            vmin, vmax = -min(3.0, max_abs), min(3.0, max_abs)
        else:
            plot_values = matrix.values
            cbar_label = "Mean"
            vmin = float(np.nanmin(plot_values))
            vmax = float(np.nanmax(plot_values))
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
        ax.set_ylabel("Cell Types", color='#DDD', fontsize=9, fontweight='bold', labelpad=4)

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

        if self.popup_dialog is None:
            self.popup_dialog = MarkerExpressionDialog(
                df=self.df,
                markers=self.markers,
                celltype_col=self.celltype_col,
                current_dataset=self.combo_dataset.currentText(),
                current_mode=self.combo_mode.currentText(),
                current_cmap=self.combo_cmap.currentText(),
                selected_subset=self.selected_subset,
                selected_samples=self.selected_samples,
                parent=self.window(),
            )
        else:
            self.popup_dialog.update_data(
                df=self.df,
                markers=self.markers,
                celltype_col=self.celltype_col,
                selected_samples=self.selected_samples,
            )

        self.popup_dialog.show()
        self.popup_dialog.raise_()
        self.popup_dialog.activateWindow()
