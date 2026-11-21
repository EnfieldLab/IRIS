"""
iris_viewer.widgets.marker_expression.dialog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

High-resolution interactive pop-up window for exploring, reordering,
hiding/showing, and exporting Marker Expression Profile heatmaps with dendrograms.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Set

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

try:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    NavigationToolbar = None

from matplotlib.figure import Figure
from qtpy.QtCore import Qt
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
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from iris_common.logging import iris_log
from iris_common.schema import find_sample_column
from iris_common.widgets.sample_scope import (
    SCOPE_ACTIVE,
    SCOPE_ALL,
    SCOPE_SELECTED,
)
from iris_common.widgets.theme import (
    BUTTON_CTA,
    BUTTON_PRIMARY,
    BUTTON_SECONDARY,
    BUTTON_STUDIO,
    DIALOG_DARK_THEME,
    create_section_card,
    style_section_header,
)
from iris_viewer.widgets.marker_expression.subset_dialog import SampleSubsetDialog
from iris_viewer.widgets.phenotyping.thresholding import resolve_marker_col


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
        assignment_callback: Optional[Callable[[List[str], str], None]] = None,
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
        self.assignment_callback = assignment_callback
        self._active_row: Optional[str] = None
        self._active_row_patch = None
        self._suggested_phenotype: str = ""
        self.cached_row_counts: dict = {}
        self.cached_total_cells: int = 0

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
        self.pub_studio_dialog = None

        self.setStyleSheet(DIALOG_DARK_THEME)

        self._build_ui()
        self._sync_available_samples()
        self._sync_grouping_dropdown()
        self.recompute_and_render()

    def _get_image_col(self, df: Optional[pd.DataFrame]) -> Optional[str]:
        return find_sample_column(df)

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # Top Toolbar: Row 1 (Parameters & Settings)
        top_bar1 = QHBoxLayout()
        top_bar1.setSpacing(10)

        top_bar1.addWidget(QLabel("Dataset:"))
        self.combo_dataset = QComboBox()
        self.combo_dataset.currentIndexChanged.connect(self._on_dataset_changed)
        top_bar1.addWidget(self.combo_dataset)

        self.btn_subset = QPushButton("Choose Samples...")
        self.btn_subset.setMinimumHeight(28)
        self.btn_subset.clicked.connect(self._open_subset_dialog)
        top_bar1.addWidget(self.btn_subset)

        top_bar1.addWidget(QLabel("Group By:"))
        self.combo_group_by = QComboBox()
        self.combo_group_by.currentIndexChanged.connect(self._on_group_by_changed)
        top_bar1.addWidget(self.combo_group_by)

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

        # Top Toolbar: Row 2 (Actions & Export)
        top_bar2 = QHBoxLayout()
        top_bar2.setSpacing(10)

        self.btn_toggle_drawer = QPushButton("Organize Rows/Cols")
        self.btn_toggle_drawer.setCheckable(True)
        self.btn_toggle_drawer.setChecked(True)
        self.btn_toggle_drawer.clicked.connect(self._toggle_organizer_panel)
        top_bar2.addWidget(self.btn_toggle_drawer)

        self.btn_cluster_both = QPushButton("Hierarchical Cluster Both")
        self.btn_cluster_both.setStyleSheet(BUTTON_PRIMARY)
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

        self.btn_pub_studio = QPushButton("Figure Studio")
        self.btn_pub_studio.setStyleSheet(BUTTON_STUDIO)
        self.btn_pub_studio.setMinimumHeight(28)
        self.btn_pub_studio.clicked.connect(self.open_publication_studio)
        top_bar2.addWidget(self.btn_pub_studio)

        main_layout.addLayout(top_bar2)

        # Hover Details Status Bar
        self.lbl_hover = QLabel(
            "Hover over any cell in the heatmap for values. "
            "Scroll mouse wheel to zoom, click and drag to pan."
        )
        self.lbl_hover.setStyleSheet(
            "color: #9AA4B2; font-size: 12px; font-style: italic; padding: 2px 4px;"
        )
        main_layout.addWidget(self.lbl_hover)

        # Main Splitter: Canvas & Reorder Organizer
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
            self.toolbar.setStyleSheet(
                "QToolBar { background-color: #1f232b; border: 1px solid #3c4352; "
                "spacing: 4px; padding: 2px; }"
                "QToolButton { background-color: transparent; border: 1px solid transparent; "
                "border-radius: 3px; padding: 3px; }"
                "QToolButton:hover { background-color: #3A6A9A; border: 1px solid #4B82B5; }"
            )
            cc_layout.addWidget(self.toolbar)

        cc_layout.addWidget(self.canvas, 1)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('scroll_event', self._on_scroll_zoom)

        self.splitter.addWidget(canvas_container)

        # Organize Drawer Panel
        self.organizer_panel, op_layout = create_section_card(
            "Organize Rows & Columns",
            "Check/uncheck to show/hide items. Drag or use buttons to reorder.",
        )

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
        self.list_rows.currentItemChanged.connect(self._on_row_selection_changed)
        self.list_rows.itemClicked.connect(self._on_row_selection_changed)
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

        # Directed Phenotype Assignment Studio
        self.assign_bar = QGroupBox()
        self.assign_bar.setStyleSheet(
            "QGroupBox { border: 1px solid #3A6A9A; border-radius: 8px; "
            "background-color: #232832; margin-top: 2px; padding: 6px; }"
        )
        ab_layout = QVBoxLayout(self.assign_bar)
        ab_layout.setContentsMargins(10, 8, 10, 8)
        ab_layout.setSpacing(6)

        lbl_assign_title = QLabel("Assign Phenotype from Expression Profile")
        style_section_header(lbl_assign_title, subtitle=False)
        ab_layout.addWidget(lbl_assign_title)

        # Row 1: Target Cluster Selection & Marker Profile Characterization
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        lbl_target = QLabel("Target Cluster / Row:")
        lbl_target.setStyleSheet("font-weight: bold; color: #F0F2F5; font-size: 11px;")
        row1.addWidget(lbl_target)

        self.btn_prev_row = QPushButton("◀")
        self.btn_prev_row.setToolTip("Previous Cluster / Row")
        self.btn_prev_row.setFixedWidth(28)
        self.btn_prev_row.setStyleSheet(
            "QPushButton { background-color: #2A2E39; color: #F0F2F5; font-weight: bold; "
            "border: 1px solid #4B5364; border-radius: 4px; padding: 3px; font-size: 11px; } "
            "QPushButton:hover { background-color: #3A6A9A; border-color: #4B82B5; }"
        )
        self.btn_prev_row.clicked.connect(self._select_prev_row)
        row1.addWidget(self.btn_prev_row)

        self.combo_active_row = QComboBox()
        self.combo_active_row.setMinimumWidth(200)
        self.combo_active_row.setToolTip(
            "Select the active cluster or click directly on any heatmap row"
        )
        self.combo_active_row.setStyleSheet(
            "QComboBox { background-color: #2A2E39; color: #00E5FF; font-weight: bold; "
            "border: 1px solid #3A6A9A; border-radius: 4px; padding: 3px 8px; font-size: 11px; } "
            "QComboBox QAbstractItemView { background-color: #2A2E39; color: #F0F2F5; "
            "selection-background-color: #3A6A9A; selection-color: #FFFFFF; }"
        )
        self.combo_active_row.currentIndexChanged.connect(self._on_combo_active_row_changed)
        row1.addWidget(self.combo_active_row)

        self.btn_next_row = QPushButton("▶")
        self.btn_next_row.setToolTip("Next Cluster / Row")
        self.btn_next_row.setFixedWidth(28)
        self.btn_next_row.setStyleSheet(
            "QPushButton { background-color: #2A2E39; color: #F0F2F5; font-weight: bold; "
            "border: 1px solid #4B5364; border-radius: 4px; padding: 3px; font-size: 11px; } "
            "QPushButton:hover { background-color: #3A6A9A; border-color: #4B82B5; }"
        )
        self.btn_next_row.clicked.connect(self._select_next_row)
        row1.addWidget(self.btn_next_row)

        self.lbl_selected_row = QLabel("Active: None")
        self.lbl_selected_row.setVisible(False)  # Backward-compatibility alias

        self.lbl_active_cell_count = QLabel("")
        self.lbl_active_cell_count.setStyleSheet(
            "color: #FFD54F; font-size: 11px; font-weight: bold; margin-left: 4px;"
        )
        row1.addWidget(self.lbl_active_cell_count)

        self.lbl_enriched_profile = QLabel("")
        self.lbl_enriched_profile.setStyleSheet(
            "color: #64B5F6; font-size: 11px; font-style: italic; margin-left: 6px;"
        )
        row1.addWidget(self.lbl_enriched_profile, 1)

        ab_layout.addLayout(row1)

        # Row 2: Phenotype Annotation Input, Suggestion Chip, & Application Button
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        lbl_new_pheno = QLabel("Phenotype Label:")
        lbl_new_pheno.setStyleSheet("font-weight: bold; color: #F0F2F5; font-size: 11px;")
        row2.addWidget(lbl_new_pheno)

        self.txt_dialog_pheno = QLineEdit()
        self.txt_dialog_pheno.setPlaceholderText("Enter phenotype name (e.g., CD8+ T cells)...")
        self.txt_dialog_pheno.setStyleSheet(
            "QLineEdit { background-color: #2A2E39; color: #F0F2F5; "
            "border: 1px solid #4B5364; border-radius: 4px; padding: 4px 8px; font-size: 11px; } "
            "QLineEdit:focus { border: 1px solid #4B82B5; }"
        )
        self.txt_dialog_pheno.returnPressed.connect(self._apply_phenotype_from_dialog)
        row2.addWidget(self.txt_dialog_pheno, 1)

        self.btn_suggest_label = QPushButton("Use Suggestion")
        self.btn_suggest_label.setToolTip(
            "Insert the auto-suggested marker signature into the phenotype field"
        )
        self.btn_suggest_label.setStyleSheet(BUTTON_SECONDARY)
        self.btn_suggest_label.clicked.connect(self._use_suggested_label)
        row2.addWidget(self.btn_suggest_label)

        self.btn_dialog_assign = QPushButton("Apply Phenotype to Cluster")
        self.btn_dialog_assign.setStyleSheet(BUTTON_CTA)
        self.btn_dialog_assign.setMinimumHeight(28)
        self.btn_dialog_assign.clicked.connect(self._apply_phenotype_from_dialog)
        row2.addWidget(self.btn_dialog_assign)

        self.lbl_dialog_assign_feedback = QLabel("")
        self.lbl_dialog_assign_feedback.setStyleSheet("color: #66BB6A; font-weight: bold; font-size: 11px; margin-left: 4px;")
        row2.addWidget(self.lbl_dialog_assign_feedback)

        ab_layout.addLayout(row2)

        main_layout.addWidget(self.assign_bar)

    def _toggle_organizer_panel(self) -> None:
        is_visible = self.btn_toggle_drawer.isChecked()
        self.organizer_panel.setVisible(is_visible)

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
            self.row_linkage = None
            self.recompute_and_render()

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
        self.row_linkage = None
        self.recompute_and_render()

    def set_assignment_callback(self, cb: Callable[[List[str], str], None]) -> None:
        self.assignment_callback = cb

    def _select_prev_row(self) -> None:
        if not hasattr(self, "combo_active_row") or self.combo_active_row.count() == 0:
            return
        cur = self.combo_active_row.currentIndex()
        new_idx = max(0, cur - 1)
        self.combo_active_row.setCurrentIndex(new_idx)

    def _select_next_row(self) -> None:
        if not hasattr(self, "combo_active_row") or self.combo_active_row.count() == 0:
            return
        cur = self.combo_active_row.currentIndex()
        new_idx = min(self.combo_active_row.count() - 1, cur + 1)
        self.combo_active_row.setCurrentIndex(new_idx)

    def _on_combo_active_row_changed(self, index: int) -> None:
        if index < 0 or not hasattr(self, "combo_active_row"):
            return
        row_key = self.combo_active_row.itemData(index)
        if not row_key:
            full_txt = self.combo_active_row.itemText(index)
            row_key = full_txt.split('(')[0].strip() if '(' in full_txt else full_txt.strip()
        if row_key:
            self._set_active_row(str(row_key), sync_combo=False)

    def _sync_active_rows_combo(self) -> None:
        if not hasattr(self, "combo_active_row"):
            return
        self.combo_active_row.blockSignals(True)
        self.combo_active_row.clear()

        rows = [r for r in self.ordered_cell_types if r in self.cached_matrix.index and r in self.visible_cell_types] if self.cached_matrix is not None else []
        counts = getattr(self, "cached_row_counts", {})
        total = getattr(self, "cached_total_cells", 0)

        for r in rows:
            cnt = counts.get(r, 0)
            if total > 0 and cnt > 0:
                pct = (cnt / total) * 100.0
                label = f"{r}  ({cnt:,} cells, {pct:.1f}%)"
            elif cnt > 0:
                label = f"{r}  ({cnt:,} cells)"
            else:
                label = str(r)
            self.combo_active_row.addItem(label, userData=r)

        self.combo_active_row.blockSignals(False)

        # Retain current active row or select the first row
        active = getattr(self, "_active_row", None)
        if active in rows:
            self._set_active_row(active, sync_combo=True)
        elif rows:
            self._set_active_row(rows[0], sync_combo=True)
        else:
            self._set_active_row("", sync_combo=True)

    def _use_suggested_label(self) -> None:
        sugg = getattr(self, '_suggested_phenotype', '')
        if not sugg:
            sugg = self.txt_dialog_pheno.placeholderText().strip()
        if sugg:
            self.txt_dialog_pheno.setText(sugg)

    def _on_row_selection_changed(self, current, previous=None) -> None:
        if current is not None:
            self._set_active_row(current.text().strip(), sync_combo=True)

    def _set_active_row(self, row_name: str, sync_combo: bool = True) -> None:
        self._active_row = row_name
        self.lbl_selected_row.setText(f"Active: {row_name}" if row_name else "Active: None")
        self.lbl_dialog_assign_feedback.setText("")

        if sync_combo and hasattr(self, "combo_active_row"):
            self.combo_active_row.blockSignals(True)
            for i in range(self.combo_active_row.count()):
                ud = self.combo_active_row.itemData(i)
                if ud == row_name or (ud is None and self.combo_active_row.itemText(i).startswith(row_name)):
                    self.combo_active_row.setCurrentIndex(i)
                    break
            self.combo_active_row.blockSignals(False)

        # Synchronize list_rows in drawer if present
        if hasattr(self, "list_rows"):
            self.list_rows.blockSignals(True)
            for i in range(self.list_rows.count()):
                item = self.list_rows.item(i)
                if item.text().strip() == row_name:
                    self.list_rows.setCurrentItem(item)
                    break
            self.list_rows.blockSignals(False)

        # Update cell count & percentage badge
        counts = getattr(self, "cached_row_counts", {})
        total = getattr(self, "cached_total_cells", 0)
        cnt = counts.get(row_name, 0)
        if total > 0 and cnt > 0:
            pct = (cnt / total) * 100.0
            self.lbl_active_cell_count.setText(f"{cnt:,} cells ({pct:.1f}%)")
        elif cnt > 0:
            self.lbl_active_cell_count.setText(f"{cnt:,} cells")
        else:
            self.lbl_active_cell_count.setText("")

        # Update top enriched markers profile
        if self.cached_matrix is not None and row_name in self.cached_matrix.index:
            row_vals = self.cached_matrix.loc[row_name]
            top_m = row_vals.sort_values(ascending=False).head(4)
            desc = ", ".join([f"{m}+" for m in top_m.index[:2]])
            raw_str = ", ".join([f"{m}: {v:.1f}" for m, v in top_m.head(3).items()])
            self.lbl_enriched_profile.setText(f"Top Markers: {desc} ({raw_str})")
            sugg = f"{desc} Population" if desc else f"{row_name} Population"
            self._suggested_phenotype = sugg
            self.txt_dialog_pheno.setPlaceholderText(sugg)
        else:
            self.lbl_enriched_profile.setText("")
            self._suggested_phenotype = ""

        # Draw highlight border and axis tick glow on heatmap
        self._highlight_active_row_on_heatmap(row_name)

    def _highlight_active_row_on_heatmap(self, row_name: str) -> None:
        if not hasattr(self, "ax_heatmap") or self.ax_heatmap is None:
            return
        if not hasattr(self, "hover_info") or not self.hover_info:
            return

        rows = self.hover_info.get("rows", [])
        if not rows:
            return

        r = None
        if row_name in rows:
            r = rows.index(row_name)
        else:
            for i, r_cand in enumerate(rows):
                if str(r_cand).strip() == str(row_name).strip():
                    r = i
                    break

        # Remove previous highlight rectangle if present
        if hasattr(self, "_active_row_patch") and self._active_row_patch is not None:
            try:
                self._active_row_patch.remove()
            except Exception:
                pass
            self._active_row_patch = None

        if r is not None and 0 <= r < len(rows):
            n_cols = len(self.hover_info.get("cols", []))
            if n_cols > 0:
                rect = patches.Rectangle(
                    (-0.49, r - 0.49),
                    n_cols - 0.02,
                    0.98,
                    linewidth=2.5,
                    edgecolor='#00E5FF',
                    facecolor='none',
                    linestyle='-',
                    zorder=20,
                )
                self.ax_heatmap.add_patch(rect)
                self._active_row_patch = rect

            # Highlight the matching y-tick label in cyan, others in white
            try:
                for i, tick_label in enumerate(self.ax_heatmap.get_yticklabels()):
                    if i == r:
                        tick_label.set_color('#00E5FF')
                        tick_label.set_fontweight('bold')
                    else:
                        tick_label.set_color('white')
                        tick_label.set_fontweight('normal')
            except Exception:
                pass

            self.canvas.draw_idle()

    def _apply_phenotype_from_dialog(self) -> None:
        row_name = getattr(self, '_active_row', None)
        if not row_name and hasattr(self, 'combo_active_row') and self.combo_active_row.count() > 0:
            row_name = self.combo_active_row.itemData(self.combo_active_row.currentIndex())
            if not row_name:
                txt = self.combo_active_row.currentText()
                row_name = txt.split('(')[0].strip() if '(' in txt else txt.strip()
        if not row_name:
            curr = self.list_rows.currentItem()
            if curr:
                row_name = curr.text().strip()
        if not row_name:
            self.lbl_dialog_assign_feedback.setText("Select a cluster or row first.")
            self.lbl_dialog_assign_feedback.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        label_name = self.txt_dialog_pheno.text().strip() or self.txt_dialog_pheno.placeholderText().strip()
        if not label_name:
            self.lbl_dialog_assign_feedback.setText("Enter a phenotype name.")
            self.lbl_dialog_assign_feedback.setStyleSheet("color: #E57373; font-size: 11px;")
            return

        if self.assignment_callback is not None:
            self.assignment_callback([row_name], label_name)
            self.lbl_dialog_assign_feedback.setText(f"Assigned {row_name} -> '{label_name}'")
            self.lbl_dialog_assign_feedback.setStyleSheet("color: #66BB6A; font-weight: bold; font-size: 11px;")
            iris_log(
                "IRIS-Viewer",
                "Marker Expression",
                f"Assigned '{row_name}' to phenotype '{label_name}'.",
            )
            if hasattr(self, "combo_active_row"):
                idx = self.combo_active_row.currentIndex()
                if idx >= 0:
                    c_txt = self.combo_active_row.itemText(idx)
                    if " -> " not in c_txt:
                        self.combo_active_row.setItemText(idx, f"{c_txt} -> '{label_name}'")
        else:
            if self.df is not None:
                ct_col = self.celltype_col
                target_col = "CellType" if "CellType" in self.df.columns else ct_col
                m = (self.df[ct_col].astype(str).str.strip() == row_name)
                if not m.any() and re.findall(r'\d+', row_name):
                    c_num = int(re.findall(r'\d+', row_name)[0])
                    m = (self.df[ct_col] == c_num)
                if m.any():
                    if target_col not in self.df.columns:
                        self.df[target_col] = "Unassigned"
                    if self.df[target_col].dtype.name == 'category':
                        if label_name not in self.df[target_col].cat.categories:
                            self.df[target_col] = self.df[target_col].cat.add_categories([label_name])
                    self.df.loc[m, target_col] = label_name
                    n_assigned = int(m.sum())
                    self.lbl_dialog_assign_feedback.setText(f"Assigned {n_assigned:,} cells -> '{label_name}'")
                    self.lbl_dialog_assign_feedback.setStyleSheet("color: #66BB6A; font-weight: bold; font-size: 11px;")
                    iris_log(
                        "IRIS-Viewer",
                        "Marker Expression",
                        f"Assigned {n_assigned:,} cells in '{row_name}' -> '{label_name}' ({target_col}).",
                    )
                    self.recompute_and_render()

    def _sync_available_samples(self) -> None:
        prev = self.combo_dataset.currentText() if self.combo_dataset.count() > 0 else self.current_dataset
        self.combo_dataset.blockSignals(True)
        self.combo_dataset.clear()
        self.combo_dataset.addItem(SCOPE_ACTIVE)
        self.combo_dataset.addItem(SCOPE_SELECTED)
        self.combo_dataset.addItem(SCOPE_ALL)
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
            self.combo_dataset.setCurrentIndex(1)  # Selected Samples default fallback
        self.current_dataset = self.combo_dataset.currentText()
        self.combo_dataset.blockSignals(False)

    def _open_subset_dialog(self) -> None:
        img_col = self._get_image_col(self.df)
        if self.df is None or img_col is None:
            QMessageBox.information(self, "No Sample Info", "Current data does not contain a sample or Image column.")
            return

        all_samples = sorted([str(s) for s in self.df[img_col].dropna().unique()])
        current_sel = set(self.selected_subset) or set(self.selected_samples)
        dialog = SampleSubsetDialog(all_samples, current_sel, parent=self)
        if dialog.exec_():
            self.selected_subset = set(dialog.selected_samples)
            self._subset_explicitly_set = True
            self.combo_dataset.blockSignals(True)
            self.combo_dataset.setCurrentText(SCOPE_SELECTED)
            self.combo_dataset.blockSignals(False)
            self.current_dataset = SCOPE_SELECTED
            self.btn_subset.setText(f"Samples ({len(self.selected_subset)}/{len(all_samples)})")
            self.row_linkage = None
            self.col_linkage = None
            self.recompute_and_render()

    def set_current_dataset(self, dataset_name: str, subset_samples: Optional[List[str]] = None) -> None:
        """Sets the active dataset or cohort sample filter and triggers recomputation."""
        if subset_samples is not None:
            self.selected_subset = set(subset_samples)
            self._subset_explicitly_set = True
            self.combo_dataset.blockSignals(True)
            self.combo_dataset.setCurrentText(SCOPE_SELECTED)
            self.combo_dataset.blockSignals(False)
            self.current_dataset = SCOPE_SELECTED
        else:
            idx = self.combo_dataset.findText(dataset_name)
            if idx >= 0:
                self.combo_dataset.blockSignals(True)
                self.combo_dataset.setCurrentIndex(idx)
                self.combo_dataset.blockSignals(False)
            elif str(dataset_name).startswith("Subset ("):
                self.combo_dataset.blockSignals(True)
                self.combo_dataset.setCurrentText(SCOPE_SELECTED)
                self.combo_dataset.blockSignals(False)
            self.current_dataset = self.combo_dataset.currentText()
        self.row_linkage = None
        self.col_linkage = None
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

        def _filter_by_names(names: Sequence[str]) -> pd.DataFrame:
            if not img_col:
                return df_work
            clean_targets = [str(s).strip() for s in names if str(s).strip()]
            if not clean_targets:
                return df_work.iloc[0:0]
            match_mask = df_work[img_col].astype(str).str.strip().isin(clean_targets)
            if not match_mask.any():
                norm_targets = {_clean_str(s) for s in clean_targets}
                norm_cols = df_work[img_col].apply(_clean_str)
                match_mask = norm_cols.isin(norm_targets)
            return df_work[match_mask]

        if ds == "None":
            self._render_empty_canvas("Dataset selection set to 'None'.")
            return
        elif ds == SCOPE_ACTIVE:
            if img_col:
                targets = list(self.selected_samples)
                if not targets:
                    uniques = [str(s).strip() for s in df_work[img_col].dropna().unique() if str(s).strip()]
                    targets = uniques[:1]
                df_work = _filter_by_names(targets)
        elif ds == SCOPE_SELECTED:
            if getattr(self, "_subset_explicitly_set", False):
                df_work = _filter_by_names(sorted(self.selected_subset))
            elif self.selected_subset:
                df_work = _filter_by_names(sorted(self.selected_subset))
            elif self.selected_samples:
                df_work = _filter_by_names(self.selected_samples)
        elif ds.startswith("Sample: "):
            sname = ds.replace("Sample: ", "").strip()
            df_work = _filter_by_names([sname])
        elif ds.startswith("Subset (") and self.selected_subset:
            df_work = _filter_by_names(sorted(self.selected_subset))

        if df_work.empty:
            self._render_empty_canvas(f"No cells found for selection: {ds}")
            return

        # Identify grouping column
        ct_col = self.combo_group_by.currentText().strip() if hasattr(self, 'combo_group_by') and self.combo_group_by.count() > 0 else self.celltype_col
        if not ct_col or ct_col not in df_work.columns:
            for fallback in ("Cluster", "cell_type", "Verified_CellType", "CellType", "GROUP", "Phenotype", "Label"):
                if fallback in df_work.columns:
                    ct_col = fallback
                    break

        if ct_col not in df_work.columns:
            self._render_empty_canvas("Grouping column not found in data.")
            return

        self.celltype_col = ct_col

        # Filter valid numeric markers via column aliasing
        resolved_pairs = []
        for m in self.markers:
            col = resolve_marker_col(df_work, m)
            if col is not None and pd.api.types.is_numeric_dtype(df_work[col]):
                resolved_pairs.append((m, col))
        if not resolved_pairs:
            self._render_empty_canvas("No valid numeric marker columns found.")
            return
        valid_markers = [m for m, _ in resolved_pairs]
        col_names = [col for _, col in resolved_pairs]

        # Format cluster column labels if numeric or digit-string
        if ct_col.lower() == "cluster":
            if pd.api.types.is_numeric_dtype(df_work[ct_col]):
                df_work[ct_col] = df_work[ct_col].apply(
                    lambda v: f"Cluster {int(v)}" if pd.notnull(v) and v >= 0 else ("Noise / Outlier" if v == -1 else "Unassigned")
                )
            else:
                def _fmt_cluster_val(v):
                    if pd.isnull(v):
                        return "Unassigned"
                    s = str(v).strip()
                    if s.isdigit():
                        return f"Cluster {int(s)}"
                    return s
                df_work[ct_col] = df_work[ct_col].apply(_fmt_cluster_val)

            # Exclude unclustered cells from the cluster profile heatmap
            df_work = df_work[df_work[ct_col] != "Unassigned"]

        if df_work.empty:
            self._render_empty_canvas(f"No clustered cells found for selection: {ds}")
            return

        # Aggregate expression across phenotypes by mean
        raw_matrix = df_work.groupby(ct_col, observed=True)[col_names].mean().fillna(0)
        raw_matrix.columns = valid_markers
        raw_matrix.index = raw_matrix.index.astype(str)

        # If grouping by cluster, sort rows numerically
        if ct_col.lower() == "cluster":
            sorted_idx = sorted(
                raw_matrix.index,
                key=lambda s: (0, int(re.search(r'\d+', s).group())) if re.search(r'\d+', s) else (1, s)
            )
            raw_matrix = raw_matrix.reindex(sorted_idx)

        self.cached_matrix = raw_matrix
        self.cached_row_counts = df_work[ct_col].astype(str).value_counts().to_dict()
        self.cached_total_cells = len(df_work)

        # Sync ordering and visibility: ensure all clusters in raw_matrix are visible and ordered
        cell_types = list(raw_matrix.index)
        self.ordered_cell_types = cell_types
        self.visible_cell_types = set(cell_types)

        self.ordered_markers = valid_markers
        self.visible_markers = set(valid_markers)

        if self.chk_dendrograms.isChecked():
            self._perform_hierarchical_clustering()
            self.render_heatmap()
        else:
            self._populate_list_widgets()
            self.render_heatmap()

        self._sync_active_rows_combo()

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

    @staticmethod
    def _zscore_matrix(mat_df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-marker column-wise Z-scores matching heatmap display."""
        col_mean = mat_df.mean(axis=0)
        col_std = mat_df.std(axis=0, ddof=0).replace(0, 1e-8).fillna(1.0)
        col_std[col_std == 0] = 1e-8
        z_vals = np.nan_to_num(((mat_df - col_mean) / col_std).values, nan=0.0)
        return pd.DataFrame(z_vals, index=mat_df.index, columns=mat_df.columns)

    def _get_export_matrix(self) -> Optional[pd.DataFrame]:
        """Return the visible submatrix, Z-scored if Z-Score mode is active."""
        if self.cached_matrix is None:
            return None
        safe_rows = [
            r for r in self.ordered_cell_types
            if r in self.cached_matrix.index and r in self.visible_cell_types
        ]
        safe_cols = [
            c for c in self.ordered_markers
            if c in self.cached_matrix.columns and c in self.visible_markers
        ]
        out_df = self.cached_matrix.loc[safe_rows, safe_cols].copy()
        if "Z-Score" in self.current_mode:
            return self._zscore_matrix(out_df)
        return out_df

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
                mat = self._zscore_matrix(mat_df).values
            else:
                mat = np.nan_to_num(mat_df.values, nan=0.0)

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
        self._sync_active_rows_combo()

    def cluster_both(self) -> None:
        """Applies hierarchical clustering to order both rows and columns and generates dendrograms."""
        self.chk_dendrograms.blockSignals(True)
        self.chk_dendrograms.setChecked(True)
        self.chk_dendrograms.blockSignals(False)
        self._perform_hierarchical_clustering()
        self.render_heatmap()
        self._sync_active_rows_combo()

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
            self._sync_active_rows_combo()
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
            plot_values = self._zscore_matrix(matrix).values
            cbar_label = "Z-Score (Per Marker)"
            finite_abs = np.abs(plot_values)[np.isfinite(plot_values)]
            max_abs = max(1.0, float(np.max(finite_abs))) if finite_abs.size > 0 else 1.0
            vmin, vmax = -min(3.0, max_abs), min(3.0, max_abs)
        else:
            plot_values = np.nan_to_num(matrix.values, nan=0.0)
            cbar_label = "Mean Intensity"
            finite_vals = plot_values[np.isfinite(plot_values)]
            if finite_vals.size > 0:
                vmin = float(np.min(finite_vals))
                vmax = float(np.max(finite_vals))
            else:
                vmin, vmax = 0.0, 1.0
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
                    plot_values = self._zscore_matrix(matrix).values
                else:
                    plot_values = np.nan_to_num(matrix.values, nan=0.0)
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
        # Highlight active row if selected
        if getattr(self, '_active_row', None):
            self._highlight_active_row_on_heatmap(self._active_row)

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
        """Initiates click-and-drag panning on the heatmap, tracking start point for row click selection."""
        if self.ax_heatmap is not None and event.inaxes == self.ax_heatmap and event.button in (1, 2, 3):
            self._pan_start = (event.x, event.y)
            self._pan_xlim = self.ax_heatmap.get_xlim()
            self._pan_ylim = self.ax_heatmap.get_ylim()
            self._mouse_click_candidate = (event.x, event.y, event.ydata)

    def _on_mouse_release(self, event) -> None:
        """Ends click-and-drag panning, or selects row on single click."""
        cand = getattr(self, "_mouse_click_candidate", None)
        if cand is not None and event.x is not None and event.y is not None:
            dx = abs(event.x - cand[0])
            dy = abs(event.y - cand[1])
            # If mouse didn't drag more than 5 pixels, treat as a row selection click!
            if dx <= 5 and dy <= 5 and hasattr(self, "hover_info") and self.hover_info:
                y_val = cand[2]
                if y_val is not None:
                    r = int(round(y_val))
                    rows = self.hover_info.get("rows", [])
                    if 0 <= r < len(rows):
                        clicked_row = rows[r]
                        self._set_active_row(str(clicked_row), sync_combo=True)
        self._pan_start = None
        self._mouse_click_candidate = None

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
                    f"<b>Mean Intensity:</b> <span style='color:#66BB6A;'>{raw_v:.2f}</span>"
                )

    def copy_to_clipboard(self) -> None:
        try:
            pixmap = self.canvas.grab()
            QApplication.clipboard().setPixmap(pixmap)
            self.lbl_hover.setText("Copied heatmap figure to clipboard!")
            self.lbl_hover.setStyleSheet("color: #66BB6A; font-size: 12px; font-weight: bold; padding: 2px 4px;")
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
            iris_log("IRIS-Viewer", "Marker Expression", f"Exported heatmap figure to {Path(filepath).name}.")
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
            out_df = self._get_export_matrix()
            if out_df is not None:
                out_df.to_csv(filepath)
                iris_log("IRIS-Viewer", "Marker Expression", f"Exported matrix ({out_df.shape[0]}x{out_df.shape[1]}) to {Path(filepath).name}.")
                QMessageBox.information(
                    self, "Export Complete", f"Matrix saved to {Path(filepath).name}."
                )

    def open_publication_studio(self):
        """Open Figure Studio populated with the current expression matrix."""
        export_df = self._get_export_matrix()
        if export_df is None or export_df.empty:
            QMessageBox.information(
                self,
                "No Matrix Data",
                "No expression matrix is available to send to Figure Studio.",
            )
            return None

        from iris_common.widgets.publication_studio import (
            THEME_PRESETS,
            PlotStyleConfig,
            PublicationPlotStudioDialog,
        )

        mat_df = export_df.copy()
        categories = [str(r) for r in mat_df.index]
        is_zscore = "Z-Score" in self.current_mode
        show_values = self.chk_values.isChecked()
        show_dendro = (
            self.chk_dendrograms.isChecked()
            and self.row_linkage is not None
            and self.col_linkage is not None
            and mat_df.shape[0] >= 2
            and mat_df.shape[1] >= 2
            and self.row_linkage.shape[0] == mat_df.shape[0] - 1
            and self.col_linkage.shape[0] == mat_df.shape[1] - 1
        )
        row_link = self.row_linkage.copy() if show_dendro and self.row_linkage is not None else None
        col_link = self.col_linkage.copy() if show_dendro and self.col_linkage is not None else None
        base_cmap = self.current_cmap
        celltype_col = self.celltype_col
        dataset_name = self.current_dataset

        palette_init = (
            base_cmap if base_cmap in ("coolwarm", "RdBu_r")
            else ("viridis" if base_cmap == "viridis" else "coolwarm")
        )
        cfg = PlotStyleConfig(
            theme="IRIS Dark",
            palette_name=palette_init,
            tick_rotation=int(self.spin_angle.value()),
            grid_mode="None",
        )

        def _plot_heatmap(fig: Figure, style_cfg: PlotStyleConfig) -> None:
            import math

            preset = THEME_PRESETS.get(style_cfg.theme, THEME_PRESETS["IRIS Dark"])
            bg = preset["figure_facecolor"]
            fg = preset["text_color"]
            fig.patch.set_facecolor(bg)

            cmap_map = {
                "viridis": "viridis",
                "Viridis": "viridis",
                "RdBu_r": "RdBu_r",
                "coolwarm": "coolwarm",
            }
            cmap_name = cmap_map.get(style_cfg.palette_name, base_cmap)
            cmap = plt.get_cmap(cmap_name)

            safe_rows = [style_cfg.custom_labels.get(str(r), str(r)) for r in mat_df.index]
            safe_cols = [style_cfg.custom_labels.get(str(c), str(c)) for c in mat_df.columns]
            plot_vals = np.nan_to_num(mat_df.to_numpy(dtype=float), nan=0.0)
            n_rows, n_cols = plot_vals.shape

            if is_zscore:
                finite_abs = np.abs(plot_vals)[np.isfinite(plot_vals)]
                max_abs = max(1.0, float(np.max(finite_abs))) if finite_abs.size > 0 else 1.0
                vmin, vmax = -min(3.0, max_abs), min(3.0, max_abs)
                cbar_label = "Z-Score (Per Marker)"
            else:
                finite_vals = plot_vals[np.isfinite(plot_vals)]
                vmin = float(np.min(finite_vals)) if finite_vals.size > 0 else 0.0
                vmax = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
                if vmin == vmax:
                    vmax += 1.0
                cbar_label = "Mean Intensity"

            fig_w, fig_h = fig.get_size_inches()
            fs = float(style_cfg.tick_size)
            max_row_len = max((len(r) for r in safe_rows), default=8)
            max_col_len = max((len(c) for c in safe_cols), default=8)
            char_w_pts = fs * 0.58
            left_margin = float(np.clip((max_row_len + 3.0) * (char_w_pts / 72.0) / fig_w, 0.16, 0.44))
            rad = math.radians(max(15, int(style_cfg.tick_rotation)))
            bottom_margin = float(
                np.clip((max_col_len * math.sin(rad) + 2.5) * (char_w_pts / 72.0) / fig_h, 0.14, 0.36)
            )

            if show_dendro and row_link is not None and col_link is not None:
                from scipy.cluster import hierarchy

                gs = gridspec.GridSpec(
                    2,
                    3,
                    figure=fig,
                    left=left_margin,
                    right=0.92,
                    bottom=bottom_margin,
                    top=0.90,
                    width_ratios=[0.80, 0.15, 0.05],
                    height_ratios=[0.14, 0.86],
                    wspace=0.02,
                    hspace=0.02,
                )
                ax_col_d = fig.add_subplot(gs[0, 0])
                ax_blank1 = fig.add_subplot(gs[0, 1])
                ax_blank2 = fig.add_subplot(gs[0, 2])
                ax = fig.add_subplot(gs[1, 0])
                ax_row_d = fig.add_subplot(gs[1, 1])
                ax_cbar = fig.add_subplot(gs[1, 2])

                for b_ax in (ax_blank1, ax_blank2):
                    b_ax.axis("off")
                    b_ax.set_facecolor(bg)

                dendro_color = "#64B5F6" if preset.get("is_dark", True) else "#2B5B84"
                hierarchy.dendrogram(
                    col_link,
                    orientation="top",
                    ax=ax_col_d,
                    no_labels=True,
                    color_threshold=0,
                    above_threshold_color=dendro_color,
                )
                ax_col_d.set_facecolor(bg)
                ax_col_d.set_xlim(0, 10 * n_cols)
                ax_col_d.axis("off")

                hierarchy.dendrogram(
                    row_link,
                    orientation="right",
                    ax=ax_row_d,
                    no_labels=True,
                    color_threshold=0,
                    above_threshold_color=dendro_color,
                )
                ax_row_d.set_facecolor(bg)
                ax_row_d.set_ylim(10 * n_rows, 0)
                ax_row_d.axis("off")
            else:
                gs = gridspec.GridSpec(
                    1,
                    2,
                    figure=fig,
                    left=left_margin,
                    right=0.92,
                    bottom=bottom_margin,
                    top=0.90,
                    width_ratios=[0.95, 0.05],
                    wspace=0.03,
                )
                ax = fig.add_subplot(gs[0, 0])
                ax_cbar = fig.add_subplot(gs[0, 1])

            ax.set_facecolor(bg)
            im = ax.imshow(
                plot_vals,
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xticks(np.arange(n_cols))
            ax.set_yticks(np.arange(n_rows))
            ax.set_xticklabels(
                safe_cols,
                rotation=int(style_cfg.tick_rotation),
                ha="right",
                fontsize=fs,
                fontfamily=style_cfg.font_family,
                color=fg,
            )
            ax.set_yticklabels(
                safe_rows,
                fontsize=fs,
                fontfamily=style_cfg.font_family,
                color=fg,
            )

            if show_values and n_rows <= 35 and n_cols <= 40:
                val_fs = max(6.5, fs - 2.0)
                for r in range(n_rows):
                    for c in range(n_cols):
                        val = plot_vals[r, c]
                        txt = f"{val:.2f}" if is_zscore else f"{val:.1f}"
                        rgba = cmap((val - vmin) / max(1e-8, vmax - vmin))
                        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        tc = "black" if lum > 0.55 else "white"
                        ax.text(
                            c,
                            r,
                            txt,
                            ha="center",
                            va="center",
                            fontsize=val_fs,
                            color=tc,
                            fontweight="bold",
                            fontfamily=style_cfg.font_family,
                        )

            cbar = fig.colorbar(im, cax=ax_cbar)
            cbar.set_label(
                cbar_label,
                color=fg,
                fontsize=float(style_cfg.label_size),
                fontfamily=style_cfg.font_family,
            )
            cbar.ax.yaxis.set_tick_params(color=fg, labelcolor=fg, labelsize=fs)

            title_txt = style_cfg.title or f"Marker Expression ({dataset_name})"
            ax.set_title(
                title_txt,
                color=fg,
                fontsize=float(style_cfg.title_size),
                fontweight="bold" if style_cfg.bold_title else "normal",
                fontfamily=style_cfg.font_family,
            )
            ax.set_xlabel(
                style_cfg.xlabel or "Markers",
                color=fg,
                fontsize=float(style_cfg.label_size),
                fontweight="bold" if style_cfg.bold_labels else "normal",
                fontfamily=style_cfg.font_family,
            )
            ax.set_ylabel(
                style_cfg.ylabel or celltype_col,
                color=fg,
                fontsize=float(style_cfg.label_size),
                fontweight="bold" if style_cfg.bold_labels else "normal",
                fontfamily=style_cfg.font_family,
            )

        if self.pub_studio_dialog is not None:
            self.pub_studio_dialog.close()

        self.pub_studio_dialog = PublicationPlotStudioDialog(
            plot_fn=_plot_heatmap,
            summary_df=mat_df,
            categories=categories,
            config=cfg,
            title="Marker Expression Heatmap — Figure Studio",
            metadata={"Dataset": dataset_name, "Mode": self.current_mode, "GroupBy": celltype_col},
            parent=self,
        )
        iris_log("IRIS-Viewer", "Marker Expression", f"Opened Figure Studio ({mat_df.shape[0]}x{mat_df.shape[1]}).")
        self.pub_studio_dialog.show()
        self.pub_studio_dialog.raise_()
        self.pub_studio_dialog.activateWindow()
        return self.pub_studio_dialog

    open_figure_studio = open_publication_studio

    def closeEvent(self, event) -> None:
        if getattr(self, "pub_studio_dialog", None) is not None:
            try:
                self.pub_studio_dialog.close()
            except Exception:
                pass
        super().closeEvent(event)

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

