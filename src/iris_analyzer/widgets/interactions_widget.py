"""
IRIS Analyzer Pairwise Spatial Interaction Dock Widget & Figure Studio Dialog
===========================================================================
Interactive dock widget and multi-tab Figure Studio for spatial
cell-cell interaction discovery, Napari canvas edge highlighting,
cohort comparison, and Source Data export.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy.cluster.hierarchy import leaves_list, linkage

from iris_analyzer.graph import (
    SpatialGraphResult,
    build_spatial_graph,
    extract_pair_edges_yx,
)
from iris_analyzer.interactions import InteractionResult, run_pairwise_interactions
from iris_common.engine.statistics import run_group_comparison
from iris_common.logging import iris_log
from iris_common.schema import (
    CELL_TYPE_ALIASES,
    COL_CELL_TYPE,
    COL_SAMPLE,
    find_sample_column,
)
from iris_common.widgets.publication_studio import (
    STUDIO_DARK_STYLESHEET,
    THEME_PRESETS,
    PlotStyleConfig,
    PublicationStyleSidebar,
    apply_plot_style,
    copy_figure_to_clipboard,
    draw_significance_brackets,
    export_publication_figure,
    export_source_data_excel,
    format_significance_label,
    resolve_category_colors,
)
from iris_common.widgets.theme import (
    BUTTON_CTA,
    POPOUT_BUTTON_STYLE,
    SECTION_SUBTITLE_STYLE,
    SHARED_DARK_THEME,
    create_section_card,
    style_section_header,
)

try:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    try:
        from matplotlib.backends.backend_qt5agg import (
            NavigationToolbar2QT as NavigationToolbar,
        )
    except ImportError:
        NavigationToolbar = None


GRAPH_MODE_SPECS: Dict[str, Tuple[str, int]] = {
    "Voronoi-Delaunay 1-Hop": ("delaunay", 1),
    "Voronoi-Delaunay 2-Hop": ("delaunay", 2),
    "Voronoi-Delaunay 3-Hop": ("delaunay", 3),
    "KNN (No Radius Cutoff)": ("knn", 1),
    "KNN + Radius (\u00b5m)": ("knn_radius", 1),
    "Fixed Radius (\u00b5m)": ("radius", 1),
}

_COHORT_COL_CANDIDATES: Tuple[str, ...] = (
    "group",
    "Group",
    "condition",
    "Condition",
    "cohort",
    "Cohort",
    "treatment",
    "Treatment",
    "phenotype_group",
)


def _resolve_cell_type_col_safe(df: Optional[pd.DataFrame], preferred: str) -> str:
    """Resolve an existing cell-type column from *preferred* or canonical aliases."""
    if df is None or len(df.columns) == 0:
        return preferred or COL_CELL_TYPE
    if preferred in df.columns:
        return preferred
    lower_map = {str(c).strip().lower(): str(c) for c in df.columns}
    if preferred and preferred.lower() in lower_map:
        return lower_map[preferred.lower()]
    for alias in CELL_TYPE_ALIASES:
        if alias in df.columns and not pd.api.types.is_numeric_dtype(df[alias]):
            return alias
        if alias.lower() in lower_map:
            cand = lower_map[alias.lower()]
            if not pd.api.types.is_numeric_dtype(df[cand]):
                return cand
    return preferred or COL_CELL_TYPE


def _extract_unique_types(df: Optional[pd.DataFrame], col: str) -> List[str]:
    """Return sorted unique non-empty phenotype names in *df[col]*."""
    if df is None or col not in df.columns or df.empty:
        return []
    vals = df[col].astype("string").str.strip().dropna()
    return sorted({str(v) for v in vals if str(v)})


def _infer_group_map(df: Optional[pd.DataFrame]) -> Dict[str, str]:
    """Infer sample -> group mapping when *df* has a clear biological group column."""
    if df is None or df.empty:
        return {}
    sample_col = find_sample_column(df)
    if sample_col is None:
        return {}
    lower_map = {str(c).strip().lower(): str(c) for c in df.columns}
    group_col: Optional[str] = None
    for cand in _COHORT_COL_CANDIDATES:
        if cand in df.columns:
            group_col = cand
            break
        if cand.lower() in lower_map:
            group_col = lower_map[cand.lower()]
            break
    if group_col is None or group_col == sample_col:
        return {}

    grouped = (
        df[[sample_col, group_col]]
        .dropna()
        .assign(
            _s=lambda d: d[sample_col].astype(str).str.strip(),
            _g=lambda d: d[group_col].astype(str).str.strip(),
        )
    )
    grouped = grouped[(grouped["_s"] != "") & (grouped["_g"] != "")]
    if grouped.empty:
        return {}

    mapping: Dict[str, str] = {}
    for s_name, sub in grouped.groupby("_s", sort=False):
        mapping[str(s_name)] = str(sub["_g"].iloc[0])
    return mapping if len(set(mapping.values())) >= 2 else {}


def _cluster_order(matrix: pd.DataFrame, symmetric: bool) -> Tuple[List[str], List[str]]:
    """Compute Ward hierarchical leaf ordering for matrix rows and columns."""
    rows = list(matrix.index)
    cols = list(matrix.columns)
    if len(rows) < 2 or len(cols) < 2:
        return rows, cols

    vals = np.nan_to_num(matrix.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        if np.ptp(vals) <= 1e-12:
            return rows, cols
        row_link = linkage(vals, method="ward", metric="euclidean")
        row_order = [rows[i] for i in leaves_list(row_link)]
        if symmetric and rows == cols:
            return row_order, row_order
        col_link = linkage(vals.T, method="ward", metric="euclidean")
        col_order = [cols[i] for i in leaves_list(col_link)]
        return row_order, col_order
    except Exception:
        return rows, cols


def _resolve_diverging_cmap(palette_name: str) -> str:
    """Resolve a diverging Matplotlib colormap name from a PlotStyleConfig palette."""
    if palette_name in ("RdBu_r", "coolwarm"):
        return palette_name
    if palette_name in ("viridis", "Viridis"):
        return "viridis"
    return "RdBu_r"


def _style_colorbar(cbar: Any, config: PlotStyleConfig, label: str = "") -> None:
    """Style a Matplotlib Colorbar without running full axes spine/grid overrides."""
    preset = THEME_PRESETS.get(config.theme, THEME_PRESETS["IRIS Dark"])
    cax = cbar.ax
    cax.tick_params(
        colors=preset["tick_color"],
        labelsize=float(config.tick_size),
    )
    for lbl in cax.get_yticklabels():
        lbl.set_fontfamily(config.font_family)
        lbl.set_fontsize(float(config.tick_size))
        lbl.set_color(preset["tick_color"])
    if label:
        cbar.set_label(
            label,
            fontsize=float(config.label_size),
            fontfamily=config.font_family,
            color=preset["text_color"],
        )
    if hasattr(cbar, "outline") and cbar.outline is not None:
        cbar.outline.set_edgecolor(preset["spine_color"])
        cbar.outline.set_linewidth(max(0.5, float(config.spine_linewidth)))


class _CohortGroupDialog(QDialog):
    """Independent dialog for assigning biological cohort groups to samples."""

    def __init__(
        self,
        samples: Sequence[str],
        group_map: Dict[str, str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Assign Sample Cohort Groups")
        self.resize(440, 360)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setStyleSheet(STUDIO_DARK_STYLESHEET)

        self._samples = list(samples)
        self._group_map = dict(group_map)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        hint = QLabel(
            "Assign each biological sample/FOV to a group (e.g., Control, Treated) "
            "for sample-level cohort interaction testing."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.table = QTableWidget(len(self._samples), 2)
        self.table.setHorizontalHeaderLabels(["Sample", "Group"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        for idx, s_name in enumerate(self._samples):
            s_item = QTableWidgetItem(s_name)
            s_item.setFlags(s_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(idx, 0, s_item)

            g_item = QTableWidgetItem(self._group_map.get(s_name, "Group_A"))
            self.table.setItem(idx, 1, g_item)

        layout.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_apply = QPushButton("Apply Groups")
        btn_apply.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_apply)
        layout.addLayout(btn_row)

    def get_group_map(self) -> Dict[str, str]:
        """Read edited sample -> group mapping from the table."""
        out: Dict[str, str] = {}
        for row_idx in range(self.table.rowCount()):
            s_item = self.table.item(row_idx, 0)
            g_item = self.table.item(row_idx, 1)
            if s_item is None:
                continue
            s_name = s_item.text().strip()
            g_name = g_item.text().strip() if g_item is not None else ""
            if s_name and g_name:
                out[s_name] = g_name
        return out


class PairwiseInteractionStudioDialog(QDialog):
    """Multi-tab Figure Studio for pairwise spatial interaction results."""

    def __init__(
        self,
        result: InteractionResult,
        style_config: Optional[PlotStyleConfig] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pairwise Interactions \u2014 Figure Studio")
        self.resize(1340, 840)
        self.setMinimumSize(960, 600)
        self.setWindowFlags(self.windowFlags() | Qt.Window)

        self.result = result
        self.config: PlotStyleConfig = (
            style_config.copy()
            if style_config is not None
            else PlotStyleConfig(
                theme="IRIS Dark",
                palette_name="RdBu_r",
                tick_rotation=45,
                width_mm=183.0,
                height_mm=130.0,
                dpi=300,
            )
        )
        self._tab_figures: Dict[int, Figure] = {}
        self._tab_canvases: Dict[int, FigureCanvas] = {}
        self._cohort_stats_cache: Optional[pd.DataFrame] = None

        self.setStyleSheet(STUDIO_DARK_STYLESHEET)
        self._build_ui()
        self._populate_controls()
        self._populate_pairwise_table()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.render_active_tab()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(6)

        self.splitter = QSplitter(Qt.Horizontal)
        self.tabs = QTabWidget()

        # Tab 0: Clustered Z-Score / Log2FC Heatmap
        tab0 = QWidget()
        t0_layout = QVBoxLayout(tab0)
        t0_layout.setContentsMargins(6, 6, 6, 6)
        t0_layout.setSpacing(6)
        t0_bar = QHBoxLayout()
        t0_bar.addWidget(QLabel("Metric:"))
        self.combo_heatmap_metric = QComboBox()
        self.combo_heatmap_metric.addItems(["Z-Score", "Log2 Fold Change"])
        self.combo_heatmap_metric.currentIndexChanged.connect(self.render_active_tab)
        t0_bar.addWidget(self.combo_heatmap_metric)

        self.chk_heatmap_cluster = QCheckBox("Hierarchical Clustering")
        self.chk_heatmap_cluster.setChecked(True)
        self.chk_heatmap_cluster.toggled.connect(self.render_active_tab)
        t0_bar.addWidget(self.chk_heatmap_cluster)

        self.chk_heatmap_stars = QCheckBox("Show Significance (*)")
        self.chk_heatmap_stars.setChecked(True)
        self.chk_heatmap_stars.toggled.connect(self.render_active_tab)
        t0_bar.addWidget(self.chk_heatmap_stars)
        t0_bar.addStretch(1)
        t0_layout.addLayout(t0_bar)
        self._attach_plot_canvas(tab0, t0_layout, tab_idx=0)
        self.tabs.addTab(tab0, "Clustered Z-Score / Log2FC Heatmap")

        # Tab 1: Significance Bubble Matrix
        tab1 = QWidget()
        t1_layout = QVBoxLayout(tab1)
        t1_layout.setContentsMargins(6, 6, 6, 6)
        t1_layout.setSpacing(6)
        t1_bar = QHBoxLayout()
        t1_bar.addWidget(QLabel("Color Metric:"))
        self.combo_bubble_metric = QComboBox()
        self.combo_bubble_metric.addItems(["Z-Score", "Log2 Fold Change"])
        self.combo_bubble_metric.currentIndexChanged.connect(self.render_active_tab)
        t1_bar.addWidget(self.combo_bubble_metric)

        self.chk_bubble_cluster = QCheckBox("Match Clustered Order")
        self.chk_bubble_cluster.setChecked(True)
        self.chk_bubble_cluster.toggled.connect(self.render_active_tab)
        t1_bar.addWidget(self.chk_bubble_cluster)
        t1_bar.addStretch(1)
        t1_layout.addLayout(t1_bar)
        self._attach_plot_canvas(tab1, t1_layout, tab_idx=1)
        self.tabs.addTab(tab1, "Significance Bubble Matrix")

        # Tab 2: Spatial Interaction Network
        tab2 = QWidget()
        t2_layout = QVBoxLayout(tab2)
        t2_layout.setContentsMargins(6, 6, 6, 6)
        t2_layout.setSpacing(6)
        t2_bar = QHBoxLayout()
        t2_bar.addWidget(QLabel("Layout:"))
        self.combo_network_layout = QComboBox()
        self.combo_network_layout.addItems(["Circular", "Spring"])
        self.combo_network_layout.currentIndexChanged.connect(self.render_active_tab)
        t2_bar.addWidget(self.combo_network_layout)

        self.chk_show_ns_edges = QCheckBox("Show Non-Significant Edges")
        self.chk_show_ns_edges.setChecked(False)
        self.chk_show_ns_edges.toggled.connect(self.render_active_tab)
        t2_bar.addWidget(self.chk_show_ns_edges)

        t2_bar.addWidget(QLabel("Edge Width Scale:"))
        self.spin_edge_scale = QDoubleSpinBox()
        self.spin_edge_scale.setRange(0.5, 4.0)
        self.spin_edge_scale.setSingleStep(0.25)
        self.spin_edge_scale.setValue(1.5)
        self.spin_edge_scale.valueChanged.connect(self.render_active_tab)
        t2_bar.addWidget(self.spin_edge_scale)
        t2_bar.addStretch(1)
        t2_layout.addLayout(t2_bar)
        self._attach_plot_canvas(tab2, t2_layout, tab_idx=2)
        self.tabs.addTab(tab2, "Spatial Interaction Network")

        # Tab 3: Cohort Comparison & Stats
        tab3 = QWidget()
        t3_layout = QVBoxLayout(tab3)
        t3_layout.setContentsMargins(6, 6, 6, 6)
        t3_layout.setSpacing(6)
        t3_bar = QHBoxLayout()
        t3_bar.addWidget(QLabel("Pair:"))
        self.combo_cohort_pair = QComboBox()
        self.combo_cohort_pair.currentIndexChanged.connect(self.render_active_tab)
        t3_bar.addWidget(self.combo_cohort_pair, 1)

        t3_bar.addWidget(QLabel("Metric:"))
        self.combo_cohort_metric = QComboBox()
        self.combo_cohort_metric.addItems(["z_score", "log2_fc"])
        self.combo_cohort_metric.currentIndexChanged.connect(self.render_active_tab)
        t3_bar.addWidget(self.combo_cohort_metric)

        t3_bar.addWidget(QLabel("Test:"))
        self.combo_cohort_test = QComboBox()
        self.combo_cohort_test.addItems(
            ["mann_whitney", "welch_t", "student_t", "kruskal", "anova_oneway"]
        )
        self.combo_cohort_test.currentIndexChanged.connect(self.render_active_tab)
        t3_bar.addWidget(self.combo_cohort_test)
        t3_layout.addLayout(t3_bar)
        self._attach_plot_canvas(tab3, t3_layout, tab_idx=3)
        self.tabs.addTab(tab3, "Cohort Comparison & Stats")

        # Tab 4: Pairwise Table
        tab4 = QWidget()
        t4_layout = QVBoxLayout(tab4)
        t4_layout.setContentsMargins(6, 6, 6, 6)
        t4_layout.setSpacing(6)
        t4_bar = QHBoxLayout()
        t4_bar.addWidget(QLabel("Filter:"))
        self.edit_table_search = QLineEdit()
        self.edit_table_search.setPlaceholderText("Filter source or target cell type...")
        self.edit_table_search.textChanged.connect(self._populate_pairwise_table)
        t4_bar.addWidget(self.edit_table_search, 1)

        t4_bar.addWidget(QLabel("Class:"))
        self.combo_table_class = QComboBox()
        self.combo_table_class.addItems(
            ["All", "Attraction", "Avoidance", "Non-significant"]
        )
        self.combo_table_class.currentIndexChanged.connect(self._populate_pairwise_table)
        t4_bar.addWidget(self.combo_table_class)
        t4_layout.addLayout(t4_bar)

        self.pairwise_table = QTableWidget(0, 0)
        self.pairwise_table.setAlternatingRowColors(True)
        self.pairwise_table.setEditTriggers(QTableWidget.NoEditTriggers)
        t4_layout.addWidget(self.pairwise_table, 1)
        self.tabs.addTab(tab4, "Pairwise Table")

        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        left_layout.addWidget(self.tabs, 1)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #9AA4B2; font-size: 11px; padding: 2px 4px;")
        left_layout.addWidget(self.lbl_status)

        self.splitter.addWidget(left_pane)

        self.sidebar = PublicationStyleSidebar(
            config=self.config,
            categories=self.result.cell_types,
            parent=self,
        )
        self.sidebar.setMinimumWidth(310)
        self.sidebar.styleChanged.connect(self._on_style_changed)
        self.sidebar.copyClipboardRequested.connect(self.copy_to_clipboard)
        self.sidebar.exportFigureRequested.connect(self.export_figure)
        self.sidebar.exportSourceDataRequested.connect(self.export_source_data)

        self.splitter.addWidget(self.sidebar)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        main_layout.addWidget(self.splitter, 1)

    def _attach_plot_canvas(
        self,
        container: QWidget,
        layout: QVBoxLayout,
        tab_idx: int,
    ) -> None:
        fig = Figure(figsize=self.config.size_inches, dpi=100)
        preset = THEME_PRESETS.get(self.config.theme, THEME_PRESETS["IRIS Dark"])
        fig.patch.set_facecolor(preset["figure_facecolor"])
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        if NavigationToolbar is not None:
            toolbar = NavigationToolbar(canvas, container)
            toolbar.setStyleSheet(
                "QToolBar { background-color: #1f232b; border: 1px solid #3c4352; "
                "spacing: 4px; padding: 2px; }"
                "QToolButton { background-color: transparent; border: 1px solid transparent; "
                "border-radius: 3px; padding: 3px; }"
                "QToolButton:hover { background-color: #3A6A9A; border: 1px solid #4B82B5; }"
            )
            layout.addWidget(toolbar)

        layout.addWidget(canvas, 1)
        self._tab_figures[tab_idx] = fig
        self._tab_canvases[tab_idx] = canvas

    def _populate_controls(self) -> None:
        self.combo_cohort_pair.blockSignals(True)
        self.combo_cohort_pair.clear()
        cts = self.result.cell_types
        is_sym = bool(self.result.metadata.get("symmetric", True))
        for i, src in enumerate(cts):
            start_j = i if is_sym else 0
            for j in range(start_j, len(cts)):
                tgt = cts[j]
                sep = " <-> " if is_sym else " -> "
                self.combo_cohort_pair.addItem(f"{src}{sep}{tgt}", (src, tgt))
        self.combo_cohort_pair.blockSignals(False)

        stat_test = str(self.result.metadata.get("stat_test", "mann_whitney"))
        idx_t = self.combo_cohort_test.findText(stat_test)
        if idx_t >= 0:
            self.combo_cohort_test.blockSignals(True)
            self.combo_cohort_test.setCurrentIndex(idx_t)
            self.combo_cohort_test.blockSignals(False)

    def update_result(self, result: InteractionResult) -> None:
        """Replace the active InteractionResult and refresh all views."""
        self.result = result
        self._cohort_stats_cache = None
        self.sidebar.set_categories(result.cell_types)
        self._populate_controls()
        self._populate_pairwise_table()
        self.render_active_tab()

    def current_figure(self) -> Figure:
        """Return the active Matplotlib Figure (falling back to Tab 0 on table tab)."""
        idx = self.tabs.currentIndex()
        return self._tab_figures.get(idx, self._tab_figures[0])

    def _on_tab_changed(self, _idx: int) -> None:
        self.render_active_tab()

    def _on_style_changed(self, new_config: PlotStyleConfig) -> None:
        self.config = new_config.copy()
        self.render_active_tab()

    def render_active_tab(self) -> None:
        """Render the currently selected plot tab with current styling."""
        idx = self.tabs.currentIndex()
        if idx == 0:
            self._render_heatmap_tab(self._tab_figures[0], self.config)
            self._draw_canvas(0)
        elif idx == 1:
            self._render_bubble_tab(self._tab_figures[1], self.config)
            self._draw_canvas(1)
        elif idx == 2:
            self._render_network_tab(self._tab_figures[2], self.config)
            self._draw_canvas(2)
        elif idx == 3:
            self._render_cohort_tab(self._tab_figures[3], self.config)
            self._draw_canvas(3)

        self.lbl_status.setText(
            f"{self.config.width_mm:.1f} x {self.config.height_mm:.1f} mm | "
            f"{self.config.dpi} DPI | Theme: {self.config.theme} | "
            f"Palette: {self.config.palette_name} | "
            f"Cells: {self.result.graph_result.n_cells} | "
            f"Edges: {self.result.graph_result.edge_count}"
        )

    def _draw_canvas(self, tab_idx: int) -> None:
        fig = self._tab_figures[tab_idx]
        try:
            fig.tight_layout()
        except Exception:
            pass
        self._tab_canvases[tab_idx].draw_idle()

    def _render_heatmap_tab(self, fig: Figure, config: PlotStyleConfig) -> None:
        fig.clear()
        ax = fig.add_subplot(111)

        use_z = self.combo_heatmap_metric.currentText() == "Z-Score"
        mat = self.result.z_matrix if use_z else self.result.log2fc_matrix
        if mat.empty:
            ax.text(0.5, 0.5, "No interaction matrix available.", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            apply_plot_style(fig, [ax], config)
            return

        is_sym = bool(self.result.metadata.get("symmetric", True))
        if self.chk_heatmap_cluster.isChecked():
            row_order, col_order = _cluster_order(mat, symmetric=is_sym)
        else:
            row_order, col_order = list(mat.index), list(mat.columns)

        ordered_mat = mat.loc[row_order, col_order]
        ordered_q = self.result.q_matrix.loc[row_order, col_order]
        vals = ordered_mat.to_numpy(dtype=np.float64)
        q_vals = ordered_q.to_numpy(dtype=np.float64)

        finite_vals = vals[np.isfinite(vals)]
        vmax = float(np.max(np.abs(finite_vals))) if finite_vals.size else 1.0
        if vmax <= 1e-9:
            vmax = 1.0

        cmap_name = _resolve_diverging_cmap(config.palette_name)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(vals, cmap=cmap_name, norm=norm, aspect="equal")

        n_r, n_c = vals.shape
        ax.set_xticks(np.arange(n_c))
        ax.set_xticklabels(col_order)
        ax.set_yticks(np.arange(n_r))
        ax.set_yticklabels(row_order)

        fdr_thr = float(self.result.metadata.get("fdr_threshold", 0.05))
        z_thr = float(self.result.metadata.get("z_threshold", 1.96))
        if self.chk_heatmap_stars.isChecked():
            z_ordered = self.result.z_matrix.loc[row_order, col_order].to_numpy(dtype=np.float64)
            for r_i in range(n_r):
                for c_j in range(n_c):
                    qv = float(q_vals[r_i, c_j])
                    zv = float(z_ordered[r_i, c_j])
                    if np.isfinite(qv) and qv < fdr_thr and abs(zv) >= z_thr:
                        stars = format_significance_label(qv, qv, mode="stars")
                        if stars != "ns":
                            v_cell = float(vals[r_i, c_j])
                            txt_c = "white" if abs(v_cell) > 0.55 * vmax else "black"
                            ax.text(
                                c_j,
                                r_i,
                                stars,
                                ha="center",
                                va="center",
                                fontsize=float(config.stat_size),
                                fontfamily=config.font_family,
                                fontweight="bold",
                                color=txt_c,
                            )

        if not config.title:
            ax.set_title(
                "Pairwise Spatial Interaction Z-Score"
                if use_z
                else "Pairwise Spatial Interaction Log2 Fold Change"
            )
        if not config.xlabel:
            ax.set_xlabel("Target Cell Type")
        if not config.ylabel:
            ax.set_ylabel("Source Cell Type")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        apply_plot_style(fig, [ax], config)
        _style_colorbar(
            cbar,
            config,
            label="Interaction Z-Score" if use_z else "Log2 Fold Change",
        )

    def _render_bubble_tab(self, fig: Figure, config: PlotStyleConfig) -> None:
        fig.clear()
        ax = fig.add_subplot(111)

        use_z = self.combo_bubble_metric.currentText() == "Z-Score"
        mat = self.result.z_matrix if use_z else self.result.log2fc_matrix
        if mat.empty:
            ax.text(0.5, 0.5, "No interaction data available.", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            apply_plot_style(fig, [ax], config)
            return

        is_sym = bool(self.result.metadata.get("symmetric", True))
        if self.chk_bubble_cluster.isChecked():
            row_order, col_order = _cluster_order(mat, symmetric=is_sym)
        else:
            row_order, col_order = list(mat.index), list(mat.columns)

        ordered_val = mat.loc[row_order, col_order].to_numpy(dtype=np.float64)
        ordered_q = self.result.q_matrix.loc[row_order, col_order].to_numpy(dtype=np.float64)
        n_r, n_c = ordered_val.shape

        yy, xx = np.meshgrid(np.arange(n_r), np.arange(n_c), indexing="ij")
        flat_x = xx.ravel()
        flat_y = yy.ravel()
        flat_val = ordered_val.ravel()
        flat_q = np.clip(np.nan_to_num(ordered_q.ravel(), nan=1.0), 1e-10, 1.0)
        neg_log10_q = -np.log10(flat_q)

        max_nlq = max(float(np.max(neg_log10_q)), 2.0)
        sizes = 35.0 + 380.0 * (neg_log10_q / max_nlq)

        finite_vals = flat_val[np.isfinite(flat_val)]
        vmax = float(np.max(np.abs(finite_vals))) if finite_vals.size else 1.0
        if vmax <= 1e-9:
            vmax = 1.0

        cmap_name = _resolve_diverging_cmap(config.palette_name)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        preset = THEME_PRESETS.get(config.theme, THEME_PRESETS["IRIS Dark"])
        fdr_thr = float(self.result.metadata.get("fdr_threshold", 0.05))
        sig_mask = flat_q < fdr_thr
        edge_colors = [
            preset["text_color"] if is_sig else preset["spine_color"]
            for is_sig in sig_mask
        ]
        line_widths = [1.2 if is_sig else 0.4 for is_sig in sig_mask]

        sc = ax.scatter(
            flat_x,
            flat_y,
            s=sizes,
            c=flat_val,
            cmap=cmap_name,
            norm=norm,
            edgecolors=edge_colors,
            linewidths=line_widths,
            alpha=0.92,
        )

        ax.set_xlim(-0.6, n_c - 0.4)
        ax.set_ylim(n_r - 0.4, -0.6)
        ax.set_xticks(np.arange(n_c))
        ax.set_xticklabels(col_order)
        ax.set_yticks(np.arange(n_r))
        ax.set_yticklabels(row_order)

        if not config.title:
            ax.set_title("Significance Bubble Matrix (-log10 q vs Effect)")
        if not config.xlabel:
            ax.set_xlabel("Target Cell Type")
        if not config.ylabel:
            ax.set_ylabel("Source Cell Type")

        ref_qs = [0.05, 0.01, 0.001]
        leg_handles = []
        for rq in ref_qs:
            nlq = -np.log10(rq)
            msz = np.sqrt(35.0 + 380.0 * (min(nlq, max_nlq) / max_nlq))
            leg_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="#888888",
                    markeredgecolor=preset["text_color"],
                    markersize=float(msz) * 0.55,
                    label=f"q = {rq}",
                )
            )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.legend(
            handles=leg_handles,
            title="-log10(q)",
            bbox_to_anchor=(1.26, 1.0),
            loc="upper left",
            borderaxespad=0.0,
        )
        apply_plot_style(fig, [ax], config)
        _style_colorbar(
            cbar,
            config,
            label="Interaction Z-Score" if use_z else "Log2 Fold Change",
        )

    def _render_network_tab(self, fig: Figure, config: PlotStyleConfig) -> None:
        fig.clear()
        ax = fig.add_subplot(111)

        cts = self.result.cell_types
        if not cts:
            ax.text(0.5, 0.5, "No cell types to display.", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            apply_plot_style(fig, [ax], config)
            return

        is_sym = bool(self.result.metadata.get("symmetric", True))
        graph = nx.Graph() if is_sym else nx.DiGraph()
        graph.add_nodes_from(cts)

        pw_df = self.result.pairwise_df
        show_ns = self.chk_show_ns_edges.isChecked()
        edge_scale = float(self.spin_edge_scale.value())

        edges_to_draw: List[Dict[str, Any]] = []
        self_loops: Dict[str, Tuple[str, float]] = {}

        seen_undir: set[Tuple[str, str]] = set()
        for row in pw_df.itertuples(index=False):
            src = str(row.source_type)
            tgt = str(row.target_type)
            cls_name = str(row.interaction_class)
            z_val = float(row.z_score) if np.isfinite(row.z_score) else 0.0

            if cls_name == "Non-significant" and not show_ns:
                continue

            if src == tgt:
                if cls_name != "Non-significant":
                    self_loops[src] = (cls_name, abs(z_val))
                continue

            if is_sym:
                key = (min(src, tgt), max(src, tgt))
                if key in seen_undir:
                    continue
                seen_undir.add(key)

            graph.add_edge(src, tgt, weight=max(abs(z_val), 0.1))
            edges_to_draw.append(
                {
                    "src": src,
                    "tgt": tgt,
                    "cls": cls_name,
                    "z": z_val,
                }
            )

        if self.combo_network_layout.currentText() == "Spring" and len(cts) > 1:
            pos = nx.spring_layout(graph, seed=42, k=1.2)
        else:
            pos = nx.circular_layout(graph)

        preset = THEME_PRESETS.get(config.theme, THEME_PRESETS["IRIS Dark"])
        node_palette_cfg = config.copy()
        if node_palette_cfg.palette_name in ("RdBu_r", "coolwarm", "Viridis", "viridis"):
            node_palette_cfg.palette_name = "Vibrant"
        node_colors = resolve_category_colors(cts, node_palette_cfg)

        attr_color = "#E64B35"
        avoid_color = "#3C78D8"
        ns_color = "#777777"

        for edge in edges_to_draw:
            p0 = pos[edge["src"]]
            p1 = pos[edge["tgt"]]
            cls_name = edge["cls"]
            z_abs = min(abs(edge["z"]), 10.0)
            lw = (0.7 + 2.8 * (z_abs / 10.0)) * edge_scale
            if cls_name == "Attraction":
                ec, ls, alpha, zorder = attr_color, "-", 0.88, 2
            elif cls_name == "Avoidance":
                ec, ls, alpha, zorder = avoid_color, "--", 0.85, 2
            else:
                ec, ls, alpha, zorder = ns_color, ":", 0.25, 1
                lw = 0.8

            if is_sym:
                ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    color=ec,
                    linestyle=ls,
                    linewidth=lw,
                    alpha=alpha,
                    zorder=zorder,
                )
            else:
                arrow = FancyArrowPatch(
                    p0,
                    p1,
                    connectionstyle="arc3,rad=0.14",
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color=ec,
                    linestyle=ls,
                    linewidth=lw,
                    alpha=alpha,
                    zorder=zorder,
                )
                ax.add_patch(arrow)

        xs = [float(pos[ct][0]) for ct in cts]
        ys = [float(pos[ct][1]) for ct in cts]
        cols = [node_colors[ct] for ct in cts]

        for ct in cts:
            if ct in self_loops:
                loop_cls, loop_z = self_loops[ct]
                ring_c = attr_color if loop_cls == "Attraction" else avoid_color
                ring_ls = "-" if loop_cls == "Attraction" else "--"
                ring = Circle(
                    (float(pos[ct][0]), float(pos[ct][1])),
                    radius=0.15,
                    fill=False,
                    edgecolor=ring_c,
                    linestyle=ring_ls,
                    linewidth=max(1.5, min(loop_z, 8.0) * 0.4),
                    zorder=3,
                )
                ax.add_patch(ring)

        ax.scatter(
            xs,
            ys,
            s=360,
            c=cols,
            edgecolors=preset["text_color"],
            linewidths=1.4,
            zorder=4,
        )

        cx = float(np.mean(xs)) if xs else 0.0
        cy = float(np.mean(ys)) if ys else 0.0
        for ct in cts:
            disp = config.custom_labels.get(ct, ct)
            px, py = float(pos[ct][0]), float(pos[ct][1])
            dx, dy = px - cx, py - cy
            norm_d = float(np.hypot(dx, dy))
            if norm_d > 1e-6:
                ox = px + (dx / norm_d) * 0.22
                oy = py + (dy / norm_d) * 0.22
                ha = "left" if dx > 0.25 else ("right" if dx < -0.25 else "center")
                va = "bottom" if dy > 0.2 else ("top" if dy < -0.2 else "center")
            else:
                ox, oy, ha, va = px, py + 0.18, "center", "bottom"
            ax.text(
                ox,
                oy,
                disp,
                ha=ha,
                va=va,
                fontsize=float(config.tick_size),
                fontfamily=config.font_family,
                fontweight="bold",
                color=preset["text_color"],
                zorder=5,
            )

        margin = 0.52
        x_min, x_max = (min(xs) - margin, max(xs) + margin) if xs else (-1.0, 1.0)
        y_min, y_max = (min(ys) - margin, max(ys) + margin) if ys else (-1.0, 1.0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")

        handles = [
            Line2D([0], [0], color=attr_color, lw=2.2, ls="-", label="Attraction (Z > 0)"),
            Line2D([0], [0], color=avoid_color, lw=2.2, ls="--", label="Avoidance (Z < 0)"),
        ]
        if show_ns:
            handles.append(
                Line2D([0], [0], color=ns_color, lw=1.2, ls=":", label="Non-significant")
            )
        ax.legend(handles=handles, loc="upper right")

        if not config.title:
            ax.set_title("Spatial Cell-Type Interaction Network")
        apply_plot_style(fig, [ax], config)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp_name in ("bottom", "left", "top", "right"):
            if sp_name in ax.spines:
                ax.spines[sp_name].set_visible(False)

    def _compute_pair_cohort_stats(
        self,
        src: str,
        tgt: str,
        metric: str,
        stat_test: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(pair_replicates_df, pair_stats_df)`` for Cohort Comparison tab."""
        rep = self.result.replicates_df
        if rep.empty:
            return pd.DataFrame(), pd.DataFrame()

        sub = rep[(rep["source_type"] == src) & (rep["target_type"] == tgt)].copy()
        if sub.empty:
            return sub, pd.DataFrame()

        mapped = sub[sub["group"] != "Unassigned"].copy()
        if mapped.empty:
            mapped = sub.copy()

        mapped[metric] = pd.to_numeric(mapped[metric], errors="coerce")
        mapped = mapped.dropna(subset=[metric])
        if mapped.empty:
            return mapped, pd.DataFrame()

        groups = sorted({str(g) for g in mapped["group"].unique()})
        if len(groups) < 2:
            return mapped, pd.DataFrame()

        sample_col = str(self.result.metadata.get("sample_col", COL_SAMPLE))
        p_adj = str(self.result.metadata.get("p_adjust", "fdr_bh"))
        fdr_thr = float(self.result.metadata.get("fdr_threshold", 0.05))
        comp = run_group_comparison(
            mapped,
            value_col=metric,
            group_col="group",
            sample_col=sample_col,
            aggregate_by_sample=True,
            test=stat_test,
            correction=p_adj,
            alpha=fdr_thr,
        )
        return mapped, comp.to_dataframe()

    def _render_cohort_tab(self, fig: Figure, config: PlotStyleConfig) -> None:
        fig.clear()
        ax = fig.add_subplot(111)

        pair_data = self.combo_cohort_pair.currentData()
        if pair_data is None:
            ax.text(0.5, 0.5, "No cell-type pairs available.", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            apply_plot_style(fig, [ax], config)
            return

        src, tgt = pair_data
        metric = self.combo_cohort_metric.currentText()
        stat_test = self.combo_cohort_test.currentText()
        pair_df, stats_df = self._compute_pair_cohort_stats(src, tgt, metric, stat_test)
        self._cohort_stats_cache = stats_df

        if pair_df.empty:
            ax.text(
                0.5,
                0.5,
                f"No sample-level values available for {src} <-> {tgt}.",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            apply_plot_style(fig, [ax], config)
            return

        groups = sorted({str(g) for g in pair_df["group"].unique()})
        group_arrays = [
            pair_df.loc[pair_df["group"] == g, metric].to_numpy(dtype=np.float64)
            for g in groups
        ]

        cat_cfg = config.copy()
        if cat_cfg.palette_name in ("RdBu_r", "coolwarm", "Viridis", "viridis"):
            cat_cfg.palette_name = "Vibrant"
        group_colors = resolve_category_colors(groups, cat_cfg)
        preset = THEME_PRESETS.get(config.theme, THEME_PRESETS["IRIS Dark"])

        positions = np.arange(len(groups), dtype=np.float64)
        bp = ax.boxplot(
            group_arrays,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            showfliers=False,
        )
        for idx_g, box in enumerate(bp["boxes"]):
            col = group_colors[groups[idx_g]]
            box.set_facecolor(mcolors.to_rgba(col, alpha=0.35))
            box.set_edgecolor(col)
            box.set_linewidth(1.5)
        for med in bp["medians"]:
            med.set_color(preset["text_color"])
            med.set_linewidth(1.8)
        for whisk in bp["whiskers"]:
            whisk.set_color(preset["tick_color"])
        for cap in bp["caps"]:
            cap.set_color(preset["tick_color"])

        rng = np.random.default_rng(0)
        y_maxes: Dict[str, float] = {}
        for idx_g, g in enumerate(groups):
            vals = group_arrays[idx_g]
            jitter = rng.uniform(-0.12, 0.12, size=vals.size) if vals.size > 1 else np.zeros(1)
            ax.scatter(
                np.full(vals.size, positions[idx_g]) + jitter,
                vals,
                s=46,
                color=group_colors[g],
                edgecolors=preset["text_color"],
                linewidths=0.8,
                zorder=5,
                label=g,
            )
            y_maxes[g] = float(np.max(vals)) if vals.size else 0.0

        ax.axhline(0.0, color=preset["spine_color"], linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(groups)

        if not stats_df.empty:
            x_map = {g: float(positions[idx_g]) for idx_g, g in enumerate(groups)}
            draw_significance_brackets(ax, stats_df, x_map, y_maxes, config)

        if not config.title:
            ax.set_title(f"{src} <-> {tgt} across Sample Groups")
        if not config.xlabel:
            ax.set_xlabel("Biological Group")
        if not config.ylabel:
            ax.set_ylabel("Interaction Z-Score" if metric == "z_score" else "Log2 Fold Change")

        apply_plot_style(fig, [ax], config)

    def _populate_pairwise_table(self) -> None:
        df = self.result.pairwise_df
        if df is None or df.empty:
            self.pairwise_table.setRowCount(0)
            self.pairwise_table.setColumnCount(0)
            return

        query = self.edit_table_search.text().strip().lower()
        cls_filter = self.combo_table_class.currentText()

        view = df.copy()
        if query:
            mask = (
                view["source_type"].astype(str).str.lower().str.contains(query, regex=False)
                | view["target_type"].astype(str).str.lower().str.contains(query, regex=False)
            )
            view = view[mask]
        if cls_filter != "All":
            view = view[view["interaction_class"] == cls_filter]

        self.pairwise_table.setSortingEnabled(False)
        self.pairwise_table.setRowCount(len(view))
        self.pairwise_table.setColumnCount(len(view.columns))
        self.pairwise_table.setHorizontalHeaderLabels([str(c) for c in view.columns])

        for r_idx, row_vals in enumerate(view.itertuples(index=False, name=None)):
            cls_name = str(row_vals[-1])
            for c_idx, val in enumerate(row_vals):
                if isinstance(val, (float, np.floating)):
                    if not np.isfinite(val):
                        text = ""
                    elif abs(float(val)) < 1e-3 and float(val) != 0.0:
                        text = f"{float(val):.2e}"
                    else:
                        text = f"{float(val):.4f}"
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                if cls_name == "Attraction":
                    item.setForeground(QColor("#FF8A75"))
                elif cls_name == "Avoidance":
                    item.setForeground(QColor("#75B8FF"))
                self.pairwise_table.setItem(r_idx, c_idx, item)

        self.pairwise_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.pairwise_table.setSortingEnabled(True)

    def active_stats_df(self) -> Optional[pd.DataFrame]:
        """Return cohort statistics DataFrame (or current pair comparison if populated)."""
        if self.result.cohort_stats_df is not None and not self.result.cohort_stats_df.empty:
            return self.result.cohort_stats_df
        if self._cohort_stats_cache is not None and not self._cohort_stats_cache.empty:
            return self._cohort_stats_cache
        return None

    def copy_to_clipboard(self) -> bool:
        """Copy the active figure at configured DPI into the system clipboard."""
        ok = copy_figure_to_clipboard(
            self.current_figure(),
            config=self.config,
            dpi=self.config.dpi,
        )
        self.lbl_status.setText(
            f"Copied figure to clipboard at {self.config.dpi} DPI."
            if ok
            else "Clipboard unavailable in current headless environment."
        )
        return ok

    def export_figure_to_path(self, filepath: Union[str, Path]) -> Path:
        """Export the active figure directly to *filepath*."""
        out = export_publication_figure(self.current_figure(), filepath, self.config)
        self.lbl_status.setText(f"Exported figure to {out.name}.")
        iris_log("IRIS-Analyzer", "Interactions", f"Exported figure to {out}.", flush=True)
        return out

    def export_figure(self) -> Optional[Path]:
        """Prompt for an output path and export the active figure."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Figure",
            "iris_figure.pdf",
            "Vector PDF (*.pdf);;Scalable Vector Graphics (*.svg);;"
            "Encapsulated PostScript (*.eps);;LZW TIFF (*.tiff *.tif);;PNG (*.png)",
        )
        if not filepath:
            return None
        try:
            out = self.export_figure_to_path(filepath)
            QMessageBox.information(self, "Export Complete", f"Figure saved to {out.name}.")
            return out
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Could not export figure: {exc}")
            return None

    def export_source_data_to_path(self, filepath: Union[str, Path]) -> Path:
        """Export multi-sheet Source Data Excel workbook directly to *filepath*."""
        meta: Dict[str, Any] = dict(self.result.metadata)
        meta.update({f"style.{k}": v for k, v in self.config.to_dict().items()})
        replicates = (
            self.result.replicates_df
            if not self.result.replicates_df.empty
            else None
        )
        out = export_source_data_excel(
            filepath,
            summary_df=self.result.pairwise_df,
            replicates_df=replicates,
            stats_df=self.active_stats_df(),
            metadata=meta,
        )
        self.lbl_status.setText(f"Exported Source Data workbook to {out.name}.")
        iris_log(
            "IRIS-Analyzer",
            "Interactions",
            f"Exported source data workbook to {out}.",
            flush=True,
        )
        return out

    def export_source_data(self) -> Optional[Path]:
        """Prompt for an Excel path and export Source Data."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Source Data (.xlsx)",
            "iris_source_data.xlsx",
            "Excel Workbook (*.xlsx)",
        )
        if not filepath:
            return None
        try:
            out = self.export_source_data_to_path(filepath)
            QMessageBox.information(
                self, "Export Complete", f"Source Data workbook saved to {out.name}."
            )
            return out
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Could not export Source Data: {exc}")
            return None

    def closeEvent(self, event: Any) -> None:
        """Clear Matplotlib figure references cleanly on dialog close."""
        for fig in self._tab_figures.values():
            fig.clear()
        super().closeEvent(event)


class PairwiseInteractionsWidget(QWidget):
    """Dockable widget for configuring and running pairwise spatial interaction analysis."""

    analysisCompleted = Signal(object)
    edgesHighlighted = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None, ctx: Any = None) -> None:
        super().__init__(parent)
        self.df: Optional[pd.DataFrame] = None
        self.markers: List[str] = []
        self.celltype_col: str = COL_CELL_TYPE
        self.scale_factor: float = 1.0
        self.ctx: Any = ctx
        self.group_map: Dict[str, str] = {}
        self.last_result: Optional[InteractionResult] = None
        self.last_graph: Optional[SpatialGraphResult] = None
        self.last_highlighted_edges: Optional[np.ndarray] = None
        self.popup_dialog: Optional[PairwiseInteractionStudioDialog] = None

        self.setStyleSheet(SHARED_DARK_THEME)
        self._build_ui()
        self._sync_parameter_visibility()
        self._render_empty_preview()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        # Graph & Permutation Controls
        config_box, cb_layout = create_section_card("Spatial Graph & Permutation Test")

        form = QFormLayout()
        form.setSpacing(6)

        self.combo_graph_mode = QComboBox()
        self.combo_graph_mode.addItems(list(GRAPH_MODE_SPECS.keys()))
        form.addRow("Graph Mode:", self.combo_graph_mode)

        self.chk_exact_ring = QCheckBox("Exact Geodesic Ring (d = hop)")
        self.chk_exact_ring.setChecked(True)
        self.lbl_exact_ring = QLabel("Ring Mode:")
        form.addRow(self.lbl_exact_ring, self.chk_exact_ring)

        self.spin_max_edge = QDoubleSpinBox()
        self.spin_max_edge.setRange(0.0, 10000.0)
        self.spin_max_edge.setDecimals(1)
        self.spin_max_edge.setSingleStep(10.0)
        self.spin_max_edge.setSpecialValueText("Off (None)")
        self.spin_max_edge.setValue(0.0)
        self.spin_max_edge.setSuffix(" \u00b5m")
        self.lbl_max_edge = QLabel("Max Edge (\u00b5m):")
        form.addRow(self.lbl_max_edge, self.spin_max_edge)

        self.spin_k = QSpinBox()
        self.spin_k.setRange(1, 200)
        self.spin_k.setValue(10)
        self.lbl_k = QLabel("K Neighbors:")
        form.addRow(self.lbl_k, self.spin_k)

        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(1.0, 10000.0)
        self.spin_radius.setDecimals(1)
        self.spin_radius.setSingleStep(5.0)
        self.spin_radius.setValue(50.0)
        self.spin_radius.setSuffix(" \u00b5m")
        self.lbl_radius = QLabel("Radius (\u00b5m):")
        form.addRow(self.lbl_radius, self.spin_radius)

        self.spin_permutations = QSpinBox()
        self.spin_permutations.setRange(50, 5000)
        self.spin_permutations.setSingleStep(50)
        self.spin_permutations.setValue(500)
        form.addRow("Permutations:", self.spin_permutations)

        self.spin_fdr = QDoubleSpinBox()
        self.spin_fdr.setRange(0.001, 0.50)
        self.spin_fdr.setDecimals(3)
        self.spin_fdr.setSingleStep(0.01)
        self.spin_fdr.setValue(0.05)
        form.addRow("FDR Threshold:", self.spin_fdr)
        self.combo_graph_mode.currentTextChanged.connect(self._sync_parameter_visibility)

        cb_layout.addLayout(form)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.btn_run = QPushButton("Run Interaction Analysis")
        self.btn_run.setStyleSheet(BUTTON_CTA)
        self.btn_run.setMinimumHeight(28)
        self.btn_run.clicked.connect(self.run_analysis)
        action_row.addWidget(self.btn_run, 2)

        self.btn_assign_groups = QPushButton("Cohort Groups...")
        self.btn_assign_groups.setMinimumHeight(28)
        self.btn_assign_groups.setToolTip("Assign sample cohort groups for statistical comparison")
        self.btn_assign_groups.clicked.connect(self._open_group_dialog)
        action_row.addWidget(self.btn_assign_groups, 1)
        cb_layout.addLayout(action_row)

        root.addWidget(config_box)

        # Canvas Edge Highlighting Controls
        edge_box, edge_layout = create_section_card("Canvas Edge Highlighting")

        pair_row = QHBoxLayout()
        pair_row.setSpacing(8)
        pair_row.addWidget(QLabel("Source:"))
        self.combo_source_type = QComboBox()
        pair_row.addWidget(self.combo_source_type, 1)
        pair_row.addWidget(QLabel("Target:"))
        self.combo_target_type = QComboBox()
        pair_row.addWidget(self.combo_target_type, 1)
        edge_layout.addLayout(pair_row)

        self.btn_highlight_edges = QPushButton("Highlight Interacting Pairs on Canvas")
        self.btn_highlight_edges.setMinimumHeight(28)
        self.btn_highlight_edges.clicked.connect(self.highlight_edges_on_canvas)
        edge_layout.addWidget(self.btn_highlight_edges)
        root.addWidget(edge_box)

        # Embedded Heatmap Preview + Popout
        plot_box, plot_layout = create_section_card(margins=(10, 6, 10, 10), spacing=8)

        hdr_row = QHBoxLayout()
        lbl_title = QLabel("Interaction Z-Score Matrix")
        style_section_header(lbl_title, subtitle=False)
        hdr_row.addWidget(lbl_title)
        hdr_row.addStretch(1)

        self.btn_popout = QPushButton("\u2922 Pop-up Window")
        self.btn_popout.setStyleSheet(POPOUT_BUTTON_STYLE)
        self.btn_popout.setToolTip("Open Pairwise Interactions Figure Studio")
        self.btn_popout.clicked.connect(self.open_popup_window)
        hdr_row.addWidget(self.btn_popout)
        plot_layout.addLayout(hdr_row)

        self.figure = Figure(figsize=(5.5, 4.2), dpi=100)
        self.figure.patch.set_facecolor("#14161d")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(240)
        self.canvas.setToolTip("Double-click to open Pairwise Interactions Figure Studio")
        self.canvas.mouseDoubleClickEvent = lambda _event: self.open_popup_window()
        plot_layout.addWidget(self.canvas, 1)

        self.lbl_status = QLabel("Load single-cell data to analyze spatial interactions.")
        self.lbl_status.setStyleSheet(SECTION_SUBTITLE_STYLE)
        self.lbl_status.setWordWrap(True)
        plot_layout.addWidget(self.lbl_status)

        root.addWidget(plot_box, 1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

    def _sync_parameter_visibility(self) -> None:
        label = self.combo_graph_mode.currentText()
        mode, hop = GRAPH_MODE_SPECS.get(label, ("delaunay", 1))

        show_ring = mode == "delaunay" and hop >= 2
        show_max_edge = mode == "delaunay"
        show_k = mode in ("knn", "knn_radius")
        show_radius = mode in ("knn_radius", "radius")

        self.lbl_exact_ring.setVisible(show_ring)
        self.chk_exact_ring.setVisible(show_ring)
        self.lbl_max_edge.setVisible(show_max_edge)
        self.spin_max_edge.setVisible(show_max_edge)
        self.lbl_k.setVisible(show_k)
        self.spin_k.setVisible(show_k)
        self.lbl_radius.setVisible(show_radius)
        self.spin_radius.setVisible(show_radius)

    def set_group_map(self, group_map: Optional[Dict[str, str]]) -> None:
        """Set explicit sample -> biological group mapping for cohort comparisons."""
        self.group_map = dict(group_map or {})

    def set_data(
        self,
        df: Optional[pd.DataFrame],
        markers: Optional[Sequence[str]] = None,
        celltype_col: str = "cell_type",
        scale_factor: float = 1.0,
        ctx: Any = None,
    ) -> None:
        """Set active single-cell DataFrame, phenotype column, scale factor, and context."""
        self.df = df
        if markers is not None:
            self.markers = list(markers)
        self.celltype_col = _resolve_cell_type_col_safe(df, celltype_col)
        self.scale_factor = float(scale_factor) if scale_factor and scale_factor > 0 else 1.0
        if ctx is not None:
            self.ctx = ctx

        if not self.group_map:
            inferred = _infer_group_map(self.df)
            if inferred:
                self.group_map = inferred

        types = _extract_unique_types(self.df, self.celltype_col)
        self.combo_source_type.blockSignals(True)
        self.combo_target_type.blockSignals(True)
        self.combo_source_type.clear()
        self.combo_target_type.clear()
        self.combo_source_type.addItems(types)
        self.combo_target_type.addItems(types)
        if len(types) >= 2:
            self.combo_target_type.setCurrentIndex(1)
        self.combo_source_type.blockSignals(False)
        self.combo_target_type.blockSignals(False)

        n_rows = len(self.df) if self.df is not None else 0
        if self.last_graph is not None and (self.df is None or self.last_graph.n_cells != n_rows):
            self.last_graph = None
            self.last_result = None
            self.last_highlighted_edges = None
            self._render_empty_preview()

        self.lbl_status.setText(
            f"Ready: {n_rows} cells, {len(types)} cell types."
            if n_rows > 0
            else "No cells loaded."
        )

    def _current_graph_kwargs(self) -> Dict[str, Any]:
        label = self.combo_graph_mode.currentText()
        mode, hop = GRAPH_MODE_SPECS.get(label, ("delaunay", 1))
        max_edge_val = float(self.spin_max_edge.value())
        max_edge_um = max_edge_val if (mode == "delaunay" and max_edge_val > 0.0) else None
        return {
            "mode": mode,
            "hop": hop,
            "exact_ring": bool(self.chk_exact_ring.isChecked()),
            "k": int(self.spin_k.value()),
            "radius_um": float(self.spin_radius.value()),
            "max_edge_um": max_edge_um,
            "microns_per_pixel": float(self.scale_factor),
        }

    def run_analysis(self) -> Optional[InteractionResult]:
        """Run pairwise spatial interaction testing and update embedded heatmap."""
        if self.df is None or self.df.empty:
            self.lbl_status.setText("No single-cell data available.")
            return None

        kwargs = self._current_graph_kwargs()
        n_perm = int(self.spin_permutations.value())
        res = run_pairwise_interactions(
            self.df,
            cell_type_col=self.celltype_col,
            n_permutations=n_perm,
            fdr_threshold=float(self.spin_fdr.value()),
            group_map=self.group_map if self.group_map else None,
            **kwargs,
        )
        self.last_result = res
        self.last_graph = res.graph_result
        self._render_compact_heatmap(res)

        n_attr = int((res.pairwise_df["interaction_class"] == "Attraction").sum())
        n_avoid = int((res.pairwise_df["interaction_class"] == "Avoidance").sum())
        n_samples = len(res.graph_result.sample_slices) or 1
        mode_label = self.combo_graph_mode.currentText()
        iris_log(
            "IRIS-Analyzer",
            "Interactions",
            f"Ran {mode_label} ({n_perm} perms): {n_samples} sample(s), "
            f"{res.graph_result.n_cells:,} cells, {res.graph_result.edge_count:,} edges, "
            f"{n_attr} Attraction / {n_avoid} Avoidance pairs.",
            flush=True,
        )
        self.lbl_status.setText(
            f"{res.graph_result.n_cells} cells | {res.graph_result.edge_count} edges | "
            f"{n_attr} Attraction, {n_avoid} Avoidance pairs"
        )

        if self.popup_dialog is not None and self.popup_dialog.isVisible():
            self.popup_dialog.update_result(res)

        self.analysisCompleted.emit(res)
        return res

    def _render_empty_preview(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#14161d")
        ax.text(
            0.5,
            0.5,
            "Run Interaction Analysis to preview Z-score matrix",
            ha="center",
            va="center",
            color="#999999",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw_idle()

    def _render_compact_heatmap(self, result: InteractionResult) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#14161d")

        z_mat = result.z_matrix
        if z_mat.empty:
            self._render_empty_preview()
            return

        vals = z_mat.to_numpy(dtype=np.float64)
        q_vals = result.q_matrix.to_numpy(dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
        if vmax <= 1e-9:
            vmax = 1.0

        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(vals, cmap="RdBu_r", norm=norm, aspect="equal")

        labels = list(z_mat.columns)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", color="#E0E0E0", fontsize=8)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, color="#E0E0E0", fontsize=8)

        fdr_thr = float(result.metadata.get("fdr_threshold", 0.05))
        z_thr = float(result.metadata.get("z_threshold", 1.96))
        for r_i in range(vals.shape[0]):
            for c_j in range(vals.shape[1]):
                qv = float(q_vals[r_i, c_j])
                zv = float(vals[r_i, c_j])
                if np.isfinite(qv) and qv < fdr_thr and abs(zv) >= z_thr:
                    stars = format_significance_label(qv, qv, mode="stars")
                    if stars != "ns":
                        txt_c = "white" if abs(zv) > 0.55 * vmax else "black"
                        ax.text(
                            c_j,
                            r_i,
                            stars,
                            ha="center",
                            va="center",
                            color=txt_c,
                            fontsize=8,
                            fontweight="bold",
                        )

        cbar = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="#D0D4DE", labelsize=8)
        cbar.set_label("Z-Score", color="#E0E0E0", fontsize=9)
        for spine in ax.spines.values():
            spine.set_color("#4A5162")

        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

    def _open_group_dialog(self) -> None:
        if self.df is None or self.df.empty:
            QMessageBox.information(self, "No Data", "Load single-cell data first.")
            return
        sample_col = find_sample_column(self.df)
        if sample_col is None:
            QMessageBox.information(
                self,
                "Single Sample",
                "No sample column detected in DataFrame for cohort grouping.",
            )
            return
        samples = list(dict.fromkeys(self.df[sample_col].astype(str).tolist()))
        dlg = _CohortGroupDialog(samples, self.group_map, parent=self.window())
        if dlg.exec_():
            self.group_map = dlg.get_group_map()
            self.lbl_status.setText(
                f"Assigned {len(self.group_map)} samples across "
                f"{len(set(self.group_map.values()))} cohort groups."
            )
        dlg.close()

    def highlight_edges_on_canvas(self) -> np.ndarray:
        """Extract interacting pair segments and render them on the Napari viewer."""
        if self.df is None or self.df.empty:
            return np.empty((0, 2, 2), dtype=np.float64)

        src_type = self.combo_source_type.currentText().strip()
        tgt_type = self.combo_target_type.currentText().strip()
        if not src_type or not tgt_type or self.celltype_col not in self.df.columns:
            return np.empty((0, 2, 2), dtype=np.float64)

        if self.last_result is not None and self.last_result.graph_result.n_cells == len(self.df):
            graph_res = self.last_result.graph_result
        elif self.last_graph is not None and self.last_graph.n_cells == len(self.df):
            graph_res = self.last_graph
        else:
            graph_res = build_spatial_graph(self.df, **self._current_graph_kwargs())
            self.last_graph = graph_res

        ct_series = self.df[self.celltype_col].astype(str).str.strip()
        src_mask = (ct_series == src_type).to_numpy()
        tgt_mask = (ct_series == tgt_type).to_numpy()
        segments = extract_pair_edges_yx(
            self.df,
            graph_res.adjacency,
            src_mask,
            tgt_mask,
            max_edges=5000,
        )
        self.last_highlighted_edges = segments
        self.edgesHighlighted.emit(segments)

        viewer = getattr(self.ctx, "viewer", None) if self.ctx is not None else None
        if viewer is not None and hasattr(viewer, "layers") and hasattr(viewer, "add_shapes"):
            to_remove = [
                lyr
                for lyr in list(viewer.layers)
                if getattr(lyr, "name", "").startswith("Interaction Edges")
                or (
                    isinstance(getattr(lyr, "metadata", None), dict)
                    and lyr.metadata.get("layer_type") == "interaction_edges"
                )
            ]
            for lyr in to_remove:
                viewer.layers.remove(lyr)

            if len(segments) > 0:
                layer_name = f"Interaction Edges: {src_type} <-> {tgt_type}"
                render_segs = self._apply_canvas_sample_translations(segments, graph_res, src_mask)
                scale_tuple = (
                    (1.0, 1.0)
                    if render_segs is not segments
                    else (self.scale_factor, self.scale_factor)
                )
                viewer.add_shapes(
                    render_segs,
                    shape_type="line",
                    edge_color="#00E5FF",
                    edge_width=1.5,
                    name=layer_name,
                    scale=scale_tuple,
                    metadata={
                        "layer_type": "interaction_edges",
                        "source_type": src_type,
                        "target_type": tgt_type,
                    },
                )

        self.lbl_status.setText(
            f"Highlighted {len(segments)} edges ({src_type} <-> {tgt_type})."
        )
        n_samples = len(graph_res.sample_slices) or 1
        iris_log(
            "IRIS-Analyzer",
            "Interactions",
            f"Highlighted {len(segments)} canvas edge(s) for {src_type} <-> {tgt_type} "
            f"across {n_samples} sample(s).",
            flush=True,
        )
        return segments

    def _apply_canvas_sample_translations(
        self,
        segments: np.ndarray,
        graph_res: SpatialGraphResult,
        src_mask: np.ndarray,
    ) -> np.ndarray:
        """Offset edge segments when multi-sample Napari Bounds layers are translated."""
        viewer = getattr(self.ctx, "viewer", None)
        if viewer is None or not hasattr(viewer, "layers") or len(graph_res.sample_slices) <= 1:
            return segments

        translations: Dict[str, Tuple[float, float]] = {}
        for lyr in viewer.layers:
            lname = str(getattr(lyr, "name", ""))
            if lname.startswith("Bounds - "):
                s_name = lname[len("Bounds - ") :].strip()
                tr = tuple(getattr(lyr, "translate", (0.0, 0.0)))
                if len(tr) >= 2 and (abs(float(tr[0])) > 1e-9 or abs(float(tr[1])) > 1e-9):
                    translations[s_name] = (float(tr[0]), float(tr[1]))

        if not translations:
            return segments

        # Map each sample's local edges in world space using sample_slices.
        out_chunks: List[np.ndarray] = []
        adj = graph_res.adjacency
        tgt_type = self.combo_target_type.currentText().strip()
        ct_series = self.df[self.celltype_col].astype(str).str.strip()
        tgt_mask = (ct_series == tgt_type).to_numpy()

        for s_name, s_idx in graph_res.sample_slices.items():
            sub_df = self.df.iloc[s_idx]
            sub_adj = adj[s_idx][:, s_idx]
            sub_segs = extract_pair_edges_yx(
                sub_df,
                sub_adj,
                src_mask[s_idx],
                tgt_mask[s_idx],
                max_edges=5000,
            )
            if len(sub_segs) == 0:
                continue
            ty, tx = translations.get(s_name, (0.0, 0.0))
            scaled = sub_segs * float(self.scale_factor)
            scaled[:, :, 0] += ty
            scaled[:, :, 1] += tx
            out_chunks.append(scaled)

        return np.vstack(out_chunks) if out_chunks else segments

    def open_popup_window(self) -> Optional[PairwiseInteractionStudioDialog]:
        """Open or focus the Pairwise Interactions Figure Studio dialog."""
        if self.last_result is None:
            if self.df is not None and not self.df.empty:
                self.run_analysis()
            else:
                QMessageBox.information(
                    self,
                    "No Interaction Data",
                    "Load single-cell data and run interaction analysis first.",
                )
                return None
        if self.last_result is None:
            return None

        if self.popup_dialog is None:
            self.popup_dialog = PairwiseInteractionStudioDialog(
                self.last_result,
                parent=self.window(),
            )
        else:
            self.popup_dialog.update_result(self.last_result)

        self.popup_dialog.show()
        self.popup_dialog.raise_()
        self.popup_dialog.activateWindow()
        return self.popup_dialog

    def closeEvent(self, event: Any) -> None:
        """Close child dialog and clear Matplotlib figure on widget close."""
        if self.popup_dialog is not None:
            self.popup_dialog.close()
        self.figure.clear()
        super().closeEvent(event)


__all__ = [
    "GRAPH_MODE_SPECS",
    "PairwiseInteractionStudioDialog",
    "PairwiseInteractionsWidget",
]
