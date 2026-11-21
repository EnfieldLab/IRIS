"""
IRIS Analyzer Cellular Neighborhoods & Spatial-LDA Dock Widget & Figure Studio Dialog
===================================================================================
Interactive dock widget and 5-tab Figure Studio supporting:
- Moving Window Cellular Neighborhoods (MiniBatchKMeans / KMeans / Leiden)
- Spatial-LDA Spatial Topic Modeling (sparse ADMM spatial smoothing)
- Editable CN/Topic metadata table with live color customization
- Multi-mode Napari canvas overlays (Discrete CN, Topic Probability, Entropy)
- Studio figure styling, vector PDF/SVG export, and Source Data .xlsx export
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import maximum_filter

from iris_analyzer.neighborhoods import (
    MovingWindowCNResult,
    SpatialLDAResult,
    compute_cn_cohort_stats,
    compute_cn_replicates_df,
    compute_inter_cn_coordination,
    run_moving_window_cn,
    run_spatial_lda,
)
from iris_common.logging import iris_log
from iris_common.schema import (
    CELL_TYPE_ALIASES,
    COL_CELL_TYPE,
    COL_SAMPLE,
    COL_X,
    COL_Y,
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
    resolve_category_colors,
)
from iris_common.widgets.theme import (
    BUTTON_CTA,
    DEFAULT_CELLTYPE_PALETTE,
    POPOUT_BUTTON_STYLE,
    SECTION_SUBTITLE_STYLE,
    SHARED_DARK_THEME,
    configure_independent_window,
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

NeighborhoodResult = Union[MovingWindowCNResult, SpatialLDAResult]

METHOD_MOVING_WINDOW = "Moving Window"
METHOD_SPATIAL_LDA = "Spatial-LDA"

_METHOD_ALIASES: Dict[str, str] = {
    METHOD_MOVING_WINDOW: METHOD_MOVING_WINDOW,
    "Moving Window (Nolan Lab)": METHOD_MOVING_WINDOW,
    "moving window": METHOD_MOVING_WINDOW,
    "moving window (nolan lab)": METHOD_MOVING_WINDOW,
    "moving_window": METHOD_MOVING_WINDOW,
    METHOD_SPATIAL_LDA: METHOD_SPATIAL_LDA,
    "Spatial-LDA (Calico)": METHOD_SPATIAL_LDA,
    "spatial-lda": METHOD_SPATIAL_LDA,
    "spatial-lda (calico)": METHOD_SPATIAL_LDA,
    "spatial_lda": METHOD_SPATIAL_LDA,
}


def _normalize_method_name(method: Optional[str]) -> str:
    """Resolve canonical neighborhood algorithm name from current or legacy labels."""
    if not method:
        return METHOD_MOVING_WINDOW
    cleaned = str(method).strip()
    if cleaned in _METHOD_ALIASES:
        return _METHOD_ALIASES[cleaned]
    return _METHOD_ALIASES.get(cleaned.lower(), METHOD_MOVING_WINDOW)

OVERLAY_DISCRETE = "Discrete CN / Dominant Topic"
OVERLAY_TOPIC_PROB = "Topic k Continuous Probability [0-1]"
OVERLAY_ENTROPY = "Transition Zone Entropy [0-1]"

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

CN_PALETTE: List[str] = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#E69F00",
    "#CC79A7",
    "#0072B2",
    "#D55E00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#7E6148",
]


def _resolve_celltype_col_safe(df: Optional[pd.DataFrame], preferred: str) -> str:
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


def _infer_group_map(df: Optional[pd.DataFrame]) -> Dict[str, str]:
    """Infer sample -> cohort group mapping from common metadata columns."""
    if df is None or df.empty:
        return {}
    sample_col = find_sample_column(df)
    if sample_col is None:
        return {}
    group_col: Optional[str] = None
    lower_map = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in _COHORT_COL_CANDIDATES:
        if cand in df.columns:
            group_col = cand
            break
        if cand.lower() in lower_map:
            group_col = lower_map[cand.lower()]
            break
    if group_col is None:
        return {}
    pairs = df[[sample_col, group_col]].dropna().drop_duplicates(subset=[sample_col])
    return {str(r[sample_col]): str(r[group_col]) for _, r in pairs.iterrows()}


def _resolve_mask_object_ids(mask: np.ndarray, sub_df: pd.DataFrame) -> np.ndarray:
    """Resolve 1-indexed segmentation ``ObjectNumber`` per row of ``sub_df``."""
    h, w = mask.shape[-2], mask.shape[-1]
    n_rows = len(sub_df)
    if n_rows == 0:
        return np.zeros(0, dtype=np.int64)

    existing = (
        pd.to_numeric(sub_df["ObjectNumber"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        if "ObjectNumber" in sub_df.columns
        else np.zeros(n_rows, dtype=np.int64)
    )
    max_mask_id = int(mask.max()) if mask.size > 0 else 0
    if np.any(existing > 0) and np.all(existing <= max_mask_id):
        return existing

    if {COL_X, COL_Y}.issubset(sub_df.columns):
        xs = np.round(
            pd.to_numeric(sub_df[COL_X], errors="coerce").fillna(-1).to_numpy()
        ).astype(np.int64)
        ys = np.round(
            pd.to_numeric(sub_df[COL_Y], errors="coerce").fillna(-1).to_numpy()
        ).astype(np.int64)
        in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        looked_up = np.zeros(n_rows, dtype=np.int64)
        if np.any(in_bounds):
            v_idx = np.flatnonzero(in_bounds)
            direct = mask[ys[v_idx], xs[v_idx]].astype(np.int64)
            looked_up[v_idx] = direct
            zeros = direct == 0
            if np.any(zeros):
                dilated = maximum_filter(mask, size=3)
                looked_up[v_idx[zeros]] = dilated[ys[v_idx[zeros]], xs[v_idx[zeros]]].astype(
                    np.int64
                )
        return np.where(looked_up > 0, looked_up, existing)

    return existing


def _style_colorbar(cbar: Any, config: PlotStyleConfig, label: str = "") -> None:
    """Apply theme colors and font sizing to a Matplotlib colorbar."""
    preset = THEME_PRESETS.get(config.theme, THEME_PRESETS["IRIS Dark"])
    cbar.ax.tick_params(
        colors=preset["tick_color"],
        labelsize=float(config.tick_size),
    )
    if label:
        cbar.set_label(
            label,
            color=preset["text_color"],
            fontsize=float(config.label_size),
            fontfamily=config.font_family,
        )
    if hasattr(cbar, "outline") and cbar.outline is not None:
        cbar.outline.set_edgecolor(preset["spine_color"])
        cbar.outline.set_linewidth(max(0.5, float(config.spine_linewidth)))


def _extract_result_matrices(
    result: NeighborhoodResult,
    niche_metadata: Dict[int, Dict[str, str]],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Return ``(composition_matrix, log2_enrichment, cn_names, cell_types)``."""
    if isinstance(result, SpatialLDAResult):
        comp = result.topic_word_matrix
        k_count = comp.shape[0]
        default_names = [f"Topic {i}" for i in range(k_count)]
        enrichment = result.metadata.get("log2_enrichment")
        if enrichment is None:
            baseline = result.metadata.get("baseline_proportions")
            if baseline is None or len(baseline) != comp.shape[1]:
                baseline = np.full(
                    comp.shape[1], 1.0 / max(1, comp.shape[1]), dtype=np.float64
                )
            enrichment = np.log2((comp + 1e-4) / (np.asarray(baseline)[np.newaxis, :] + 1e-4))
    else:
        comp = result.centers
        k_count = comp.shape[0]
        default_names = [f"CN {i}" for i in range(k_count)]
        enrichment = result.log2_enrichment

    cn_names = [
        niche_metadata.get(i, {}).get("name", default_names[i])
        for i in range(k_count)
    ]
    return comp, np.asarray(enrichment, dtype=np.float64), cn_names, list(result.cell_types)


def _extract_replicates_with_custom_names(
    result: NeighborhoodResult,
    niche_metadata: Dict[int, Dict[str, str]],
) -> pd.DataFrame:
    """Return per-sample CN/Topic abundance DataFrame reflecting custom CN names."""
    if isinstance(result, SpatialLDAResult):
        rep_df = result.metadata.get("replicates_df")
        if rep_df is None or rep_df.empty:
            rep_df = compute_cn_replicates_df(result.updated_df)
        else:
            rep_df = rep_df.copy()
    else:
        rep_df = result.replicates_df.copy()

    if not rep_df.empty and "Cellular_Neighborhood_ID" in rep_df.columns:
        rep_df["Cellular_Neighborhood"] = rep_df["Cellular_Neighborhood_ID"].apply(
            lambda cid: niche_metadata.get(int(cid), {}).get(
                "name",
                f"Topic {int(cid)}" if isinstance(result, SpatialLDAResult) else f"CN {int(cid)}",
            )
        )
    return rep_df


class _CohortGroupDialog(QDialog):
    """Lightweight modal dialog assigning samples to biological cohort groups."""

    def __init__(
        self,
        samples: Sequence[str],
        group_map: Dict[str, str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        configure_independent_window(self)
        self.setWindowTitle("Assign Sample Cohort Groups")
        self.resize(440, 340)
        self.setStyleSheet(STUDIO_DARK_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Assign a biological group name to each sample:"))

        self.table = QTableWidget(len(samples), 2)
        self.table.setHorizontalHeaderLabels(["Sample", "Cohort Group"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)

        for r_idx, sample in enumerate(samples):
            s_item = QTableWidgetItem(str(sample))
            s_item.setFlags(s_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r_idx, 0, s_item)
            default_grp = group_map.get(str(sample), "Group_A")
            self.table.setItem(r_idx, 1, QTableWidgetItem(str(default_grp)))
        layout.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("Apply")
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

    def get_group_map(self) -> Dict[str, str]:
        """Return ``{sample_name: group_name}`` mapping from edited cells."""
        out: Dict[str, str] = {}
        for r_idx in range(self.table.rowCount()):
            s_item = self.table.item(r_idx, 0)
            g_item = self.table.item(r_idx, 1)
            if s_item is not None and g_item is not None:
                g_val = g_item.text().strip()
                if g_val:
                    out[s_item.text().strip()] = g_val
        return out


class CellularNeighborhoodStudioDialog(QDialog):
    """Multi-tab Figure Studio for Cellular Neighborhoods and Spatial-LDA."""

    def __init__(
        self,
        result: NeighborhoodResult,
        niche_metadata: Optional[Dict[int, Dict[str, str]]] = None,
        phenotype_colors: Optional[Dict[str, str]] = None,
        group_map: Optional[Dict[str, str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        configure_independent_window(self)
        self.setWindowTitle("Cellular Neighborhoods \u2014 Figure Studio")
        self.resize(1320, 840)
        self.setMinimumSize(940, 580)
        self.setStyleSheet(STUDIO_DARK_STYLESHEET)

        self.result: NeighborhoodResult = result
        self.niche_metadata: Dict[int, Dict[str, str]] = dict(niche_metadata or {})
        self.phenotype_colors: Dict[str, str] = dict(phenotype_colors or {})
        self.group_map: Dict[str, str] = dict(group_map or {})
        self.style_config: PlotStyleConfig = PlotStyleConfig(
            palette_name="Vibrant",
            tick_rotation=25,
        )
        self._tab_figures: List[Figure] = []
        self._tab_canvases: List[FigureCanvas] = []
        self._tab_toolbars: List[Any] = []
        self._latest_stats_df: pd.DataFrame = pd.DataFrame()

        self._build_ui()
        self.render_all_tabs()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)

        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Studio Control Toolbar
        ctrl_bar = QHBoxLayout()
        ctrl_bar.setSpacing(8)

        ctrl_bar.addWidget(QLabel("Heatmap Mode:"))
        self.combo_heatmap_mode = QComboBox()
        self.combo_heatmap_mode.addItems(["Log2 Enrichment", "Composition / Topic Weights"])
        self.combo_heatmap_mode.currentIndexChanged.connect(self.render_active_tab)
        ctrl_bar.addWidget(self.combo_heatmap_mode)

        self.chk_show_values = QCheckBox("Show Matrix Values")
        self.chk_show_values.setChecked(True)
        self.chk_show_values.toggled.connect(self.render_active_tab)
        ctrl_bar.addWidget(self.chk_show_values)

        self.chk_pct_bars = QCheckBox("Stacked % (0-100)")
        self.chk_pct_bars.setChecked(True)
        self.chk_pct_bars.toggled.connect(self.render_active_tab)
        ctrl_bar.addWidget(self.chk_pct_bars)

        ctrl_bar.addWidget(QLabel("Cohort Stat:"))
        self.combo_stat_test = QComboBox()
        self.combo_stat_test.addItems(["mann_whitney", "welch_t", "student_t", "kruskal"])
        self.combo_stat_test.currentTextChanged.connect(self.render_active_tab)
        ctrl_bar.addWidget(self.combo_stat_test)

        ctrl_bar.addStretch(1)
        left_layout.addLayout(ctrl_bar)

        self.tabs = QTabWidget()
        tab_titles = (
            "Log2 Enrichment / Topic Weights Heatmap",
            "Stacked Cell-Type Composition",
            "Spatial-LDA Probabilities & Transition Entropy",
            "Inter-CN Coordination Matrix",
            "Cohort CN Abundance & Stats",
        )
        for title in tab_titles:
            tab_w = QWidget()
            tab_l = QVBoxLayout(tab_w)
            tab_l.setContentsMargins(4, 4, 4, 4)
            tab_l.setSpacing(4)

            fig = Figure(figsize=self.style_config.size_inches, dpi=100)
            preset = THEME_PRESETS.get(self.style_config.theme, THEME_PRESETS["IRIS Dark"])
            fig.patch.set_facecolor(preset["figure_facecolor"])
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            toolbar = None
            if NavigationToolbar is not None:
                toolbar = NavigationToolbar(canvas, tab_w)
                toolbar.setStyleSheet(
                    "QToolBar { background-color: #1f232b; border: 1px solid #3c4352; "
                    "spacing: 4px; padding: 2px; }"
                    "QToolButton { background-color: transparent; border: 1px solid transparent; "
                    "border-radius: 3px; padding: 3px; }"
                    "QToolButton:hover { background-color: #3A6A9A; border: 1px solid #4B82B5; }"
                )
                tab_l.addWidget(toolbar)

            tab_l.addWidget(canvas, 1)
            self._tab_figures.append(fig)
            self._tab_canvases.append(canvas)
            self._tab_toolbars.append(toolbar)
            self.tabs.addTab(tab_w, title)

        self.tabs.currentChanged.connect(self._on_tab_changed)
        left_layout.addWidget(self.tabs, 1)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #9AA4B2; font-size: 11px; padding: 2px 4px;")
        left_layout.addWidget(self.lbl_status)

        splitter.addWidget(left_pane)

        # Right Pane: Embedded Figure Style Sidebar
        self.sidebar = PublicationStyleSidebar(
            config=self.style_config,
            categories=self._categories_for_tab(0),
            parent=self,
        )
        self.sidebar.setMinimumWidth(315)
        self.sidebar.styleChanged.connect(self._on_style_changed)
        self.sidebar.copyClipboardRequested.connect(self.copy_to_clipboard)
        self.sidebar.exportFigureRequested.connect(self.export_figure)
        self.sidebar.exportSourceDataRequested.connect(self.export_source_data)
        splitter.addWidget(self.sidebar)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

    def current_figure(self) -> Figure:
        """Return the Matplotlib Figure for the currently selected tab."""
        idx = max(0, min(self.tabs.currentIndex(), len(self._tab_figures) - 1))
        return self._tab_figures[idx]

    def _categories_for_tab(self, tab_idx: int) -> List[str]:
        comp, _, cn_names, cell_types = _extract_result_matrices(
            self.result, self.niche_metadata
        )
        if tab_idx == 1:
            return list(cell_types)
        if tab_idx == 4 and self.group_map:
            return sorted(set(self.group_map.values()))
        return list(cn_names)

    def _on_tab_changed(self, tab_idx: int) -> None:
        self.sidebar.set_categories(self._categories_for_tab(tab_idx))
        self.render_tab(tab_idx)

    def _on_style_changed(self, new_config: PlotStyleConfig) -> None:
        self.style_config = new_config.copy()
        self.render_active_tab()

    def update_result(
        self,
        result: NeighborhoodResult,
        niche_metadata: Optional[Dict[int, Dict[str, str]]] = None,
        phenotype_colors: Optional[Dict[str, str]] = None,
        group_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Update cached analysis results and refresh all Studio tabs."""
        self.result = result
        if niche_metadata is not None:
            self.niche_metadata = dict(niche_metadata)
        if phenotype_colors is not None:
            self.phenotype_colors = dict(phenotype_colors)
        if group_map is not None:
            self.group_map = dict(group_map)
        self.sidebar.set_categories(self._categories_for_tab(self.tabs.currentIndex()))
        self.render_all_tabs()

    def render_active_tab(self) -> None:
        """Re-render only the active Studio tab."""
        self.render_tab(self.tabs.currentIndex())

    def render_all_tabs(self) -> None:
        """Render every Studio tab canvas cleanly."""
        for idx in range(len(self._tab_figures)):
            self.render_tab(idx)

    def render_tab(self, tab_idx: int) -> None:
        """Render tab *tab_idx* using the active ``PlotStyleConfig``."""
        if not (0 <= tab_idx < len(self._tab_figures)):
            return
        fig = self._tab_figures[tab_idx]
        canvas = self._tab_canvases[tab_idx]
        cfg = self.style_config

        if tab_idx == 0:
            self._render_heatmap_tab(fig, cfg)
        elif tab_idx == 1:
            self._render_stacked_composition_tab(fig, cfg)
        elif tab_idx == 2:
            self._render_spatial_lda_tab(fig, cfg)
        elif tab_idx == 3:
            self._render_coordination_tab(fig, cfg)
        elif tab_idx == 4:
            self._render_cohort_stats_tab(fig, cfg)

        try:
            fig.tight_layout()
        except Exception:
            pass
        canvas.draw_idle()

        self.lbl_status.setText(
            f"{cfg.width_mm:.1f} x {cfg.height_mm:.1f} mm | {cfg.dpi} DPI | "
            f"Theme: {cfg.theme} | Method: {self.result.metadata.get('method', 'CN')}"
        )

    def _render_heatmap_tab(self, fig: Figure, cfg: PlotStyleConfig) -> None:
        fig.clear()
        ax = fig.add_subplot(111)
        comp, enrichment, cn_names, cell_types = _extract_result_matrices(
            self.result, self.niche_metadata
        )
        if comp.size == 0 or not cell_types:
            ax.text(0.5, 0.5, "No cellular neighborhood composition data.", ha="center", va="center")
            apply_plot_style(fig, [ax], cfg)
            return

        use_weights = self.combo_heatmap_mode.currentIndex() == 1
        mat = comp if use_weights else enrichment
        preset = THEME_PRESETS.get(cfg.theme, THEME_PRESETS["IRIS Dark"])

        if use_weights:
            vmax = max(float(np.nanmax(mat)), 1e-6)
            im = ax.imshow(mat, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto")
            cbar_lbl = "Mean Proportion / Topic Weight"
            default_title = (
                "Spatial-LDA Topic-Phenotype Weights"
                if isinstance(self.result, SpatialLDAResult)
                else "Cellular Neighborhood Composition Weights"
            )
        else:
            finite = mat[np.isfinite(mat)]
            vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
            vmax = max(vmax, 0.5)
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
            cbar_lbl = "Log2 Fold Enrichment"
            default_title = "Cell-Type Log2 Fold Enrichment Across Neighborhoods"

        ax.set_xticks(np.arange(len(cell_types)))
        ax.set_xticklabels(cell_types)
        ax.set_yticks(np.arange(len(cn_names)))
        ax.set_yticklabels(cn_names)

        if self.chk_show_values.isChecked():
            for r_i in range(mat.shape[0]):
                for c_j in range(mat.shape[1]):
                    val = float(mat[r_i, c_j])
                    if not np.isfinite(val):
                        continue
                    if use_weights:
                        txt_c = "white" if val > 0.55 * vmax else "black"
                        label_str = f"{val:.2f}"
                    else:
                        txt_c = "white" if abs(val) > 0.55 * vmax else "black"
                        label_str = f"{val:+.2f}"
                    ax.text(
                        c_j,
                        r_i,
                        label_str,
                        ha="center",
                        va="center",
                        color=txt_c,
                        fontsize=max(6.0, float(cfg.tick_size) - 2.0),
                    )

        if not cfg.title:
            ax.set_title(default_title, color=preset["text_color"])
        if not cfg.xlabel:
            ax.set_xlabel("Cell Type", color=preset["text_color"])
        if not cfg.ylabel:
            ax.set_ylabel("Neighborhood / Topic", color=preset["text_color"])

        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        apply_plot_style(fig, [ax], cfg)
        _style_colorbar(cbar, cfg, label=cbar_lbl)

    def _render_stacked_composition_tab(self, fig: Figure, cfg: PlotStyleConfig) -> None:
        fig.clear()
        ax = fig.add_subplot(111)
        comp, _, cn_names, cell_types = _extract_result_matrices(
            self.result, self.niche_metadata
        )
        if comp.size == 0 or not cell_types:
            ax.text(0.5, 0.5, "No composition data available.", ha="center", va="center")
            apply_plot_style(fig, [ax], cfg)
            return

        is_pct = self.chk_pct_bars.isChecked()
        scale = 100.0 if is_pct else 1.0
        n_niches = comp.shape[0]
        bottom = np.zeros(n_niches, dtype=np.float64)
        cat_colors = resolve_category_colors(cell_types, cfg)
        x = np.arange(n_niches)

        for idx, ct in enumerate(cell_types):
            vals = comp[:, idx] * scale
            color = (
                cfg.custom_colors.get(ct)
                or self.phenotype_colors.get(ct)
                or self.phenotype_colors.get(ct.lower())
                or cat_colors[ct]
            )
            ax.bar(
                x,
                vals,
                bottom=bottom,
                label=ct,
                color=color,
                edgecolor="#111111",
                linewidth=0.5,
            )
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(cn_names)
        ax.set_ylim(0.0, 100.0 if is_pct else 1.0)
        if not cfg.title:
            ax.set_title("Cellular Neighborhood Cell-Type Composition")
        if not cfg.xlabel:
            ax.set_xlabel("Cellular Neighborhood / Topic")
        if not cfg.ylabel:
            ax.set_ylabel("Cell Type Composition (%)" if is_pct else "Cell Type Proportion")
        ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)
        apply_plot_style(fig, [ax], cfg)

    def _render_spatial_lda_tab(self, fig: Figure, cfg: PlotStyleConfig) -> None:
        fig.clear()
        if not isinstance(self.result, SpatialLDAResult):
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Switch algorithm to 'Spatial-LDA' and run extraction\n"
                "to inspect per-cell topic probabilities and transition zone entropy.",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            apply_plot_style(fig, [ax], cfg)
            return

        probs = self.result.topic_probabilities
        entropy = self.result.topic_entropy
        if probs.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No cells available.", ha="center", va="center")
            apply_plot_style(fig, [ax], cfg)
            return

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        k_topics = probs.shape[1]
        topic_names = [
            self.niche_metadata.get(i, {}).get("name", f"Topic {i}")
            for i in range(k_topics)
        ]
        cn_colors = [
            self.niche_metadata.get(i, {}).get("color", CN_PALETTE[i % len(CN_PALETTE)])
            for i in range(k_topics)
        ]

        bp = ax1.boxplot(
            [probs[:, k] for k in range(k_topics)],
            patch_artist=True,
            tick_labels=topic_names,
            showfliers=False,
        )
        for patch, color in zip(bp["boxes"], cn_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax1.set_ylim(-0.02, 1.05)
        ax1.set_title("Per-Cell Topic Probability Distribution")
        ax1.set_ylabel("Topic Posterior Probability")
        ax1.set_xlabel("Topic")

        dom_ids = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)

        # Subsample scatter cleanly if dataset is huge while keeping deterministic order
        if len(max_probs) > 5000:
            step = max(1, len(max_probs) // 5000)
            sl = slice(None, None, step)
            mp_plot, ent_plot, dom_plot = max_probs[sl], entropy[sl], dom_ids[sl]
        else:
            mp_plot, ent_plot, dom_plot = max_probs, entropy, dom_ids

        for k in range(k_topics):
            mask = dom_plot == k
            if np.any(mask):
                ax2.scatter(
                    mp_plot[mask],
                    ent_plot[mask],
                    s=14,
                    alpha=0.65,
                    color=cn_colors[k],
                    label=topic_names[k],
                    edgecolors="none",
                )

        ax2.set_xlim(0.0, 1.02)
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_title("Transition Zone Entropy vs Topic Dominance")
        ax2.set_xlabel("Max Topic Probability (Dominance)")
        ax2.set_ylabel("Normalized Shannon Entropy [0-1]")
        ax2.legend(loc="upper right", markerscale=1.3)

        apply_plot_style(fig, [ax1, ax2], cfg)

    def _render_coordination_tab(self, fig: Figure, cfg: PlotStyleConfig) -> None:
        fig.clear()
        ax = fig.add_subplot(111)

        if isinstance(self.result, SpatialLDAResult):
            coord_df = self.result.metadata.get("inter_cn_coordination")
        else:
            coord_df = self.result.inter_cn_coordination

        if coord_df is None or coord_df.empty:
            coord_df = compute_inter_cn_coordination(
                self.result.updated_df,
                cn_col="Cellular_Neighborhood_ID",
                n_permutations=100,
            )

        if coord_df is None or coord_df.empty:
            ax.text(
                0.5,
                0.5,
                "Inter-CN coordination matrix not available for this dataset.",
                ha="center",
                va="center",
            )
            apply_plot_style(fig, [ax], cfg)
            return

        vals = coord_df.to_numpy(dtype=np.float64)
        k_count = vals.shape[0]
        names = [
            self.niche_metadata.get(i, {}).get("name", str(coord_df.index[i]))
            for i in range(k_count)
        ]
        finite = vals[np.isfinite(vals)]
        vmax = float(np.max(np.abs(finite))) if finite.size else 2.0
        vmax = max(vmax, 1.0)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(vals, cmap="RdBu_r", norm=norm, aspect="equal")

        ax.set_xticks(np.arange(k_count))
        ax.set_xticklabels(names)
        ax.set_yticks(np.arange(k_count))
        ax.set_yticklabels(names)

        if self.chk_show_values.isChecked():
            for r_i in range(k_count):
                for c_j in range(k_count):
                    zv = float(vals[r_i, c_j])
                    if not np.isfinite(zv):
                        continue
                    txt_c = "white" if abs(zv) > 0.55 * vmax else "black"
                    ax.text(
                        c_j,
                        r_i,
                        f"{zv:+.1f}",
                        ha="center",
                        va="center",
                        color=txt_c,
                        fontsize=max(6.0, float(cfg.tick_size) - 2.0),
                    )

        if not cfg.title:
            ax.set_title("Inter-CN Spatial Coordination (Delaunay Contact Z-Score)")
        if not cfg.xlabel:
            ax.set_xlabel("Neighboring Neighborhood")
        if not cfg.ylabel:
            ax.set_ylabel("Reference Neighborhood")

        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        apply_plot_style(fig, [ax], cfg)
        _style_colorbar(cbar, cfg, label="Contact Z-Score")

    def _render_cohort_stats_tab(self, fig: Figure, cfg: PlotStyleConfig) -> None:
        fig.clear()
        ax = fig.add_subplot(111)
        rep_df = _extract_replicates_with_custom_names(self.result, self.niche_metadata)
        if rep_df.empty:
            ax.text(0.5, 0.5, "No sample replicate data available.", ha="center", va="center")
            apply_plot_style(fig, [ax], cfg)
            return

        effective_gmap = dict(self.group_map)
        if not effective_gmap:
            effective_gmap = _infer_group_map(self.result.updated_df)

        cn_order = list(dict.fromkeys(rep_df["Cellular_Neighborhood"].astype(str).tolist()))
        n_cns = len(cn_order)
        x_base = np.arange(n_cns, dtype=np.float64)

        if not effective_gmap or len(set(effective_gmap.values())) < 2:
            # Plot overall per-sample abundance distribution across CNs
            means = [
                float(rep_df.loc[rep_df["Cellular_Neighborhood"] == cn, "percentage"].mean())
                for cn in cn_order
            ]
            stds = [
                float(rep_df.loc[rep_df["Cellular_Neighborhood"] == cn, "percentage"].std(ddof=0))
                for cn in cn_order
            ]
            colors = [
                self.niche_metadata.get(i, {}).get("color", CN_PALETTE[i % len(CN_PALETTE)])
                for i in range(n_cns)
            ]
            ax.bar(
                x_base,
                means,
                yerr=stds,
                capsize=3,
                color=colors,
                edgecolor="#111111",
                linewidth=0.6,
                alpha=0.85,
            )
            for idx, cn in enumerate(cn_order):
                vals = rep_df.loc[rep_df["Cellular_Neighborhood"] == cn, "percentage"].to_numpy()
                if vals.size:
                    jitter = np.linspace(-0.14, 0.14, vals.size) if vals.size > 1 else np.array([0.0])
                    ax.scatter(
                        np.full(vals.size, x_base[idx]) + jitter,
                        vals,
                        color="#FFFFFF",
                        edgecolor="#111111",
                        s=22,
                        zorder=4,
                    )
            self._latest_stats_df = pd.DataFrame()
            ax.set_xticks(x_base)
            ax.set_xticklabels(cn_order)
            if not cfg.title:
                ax.set_title("Per-Sample CN Abundance (%)")
            if not cfg.ylabel:
                ax.set_ylabel("CN Abundance (% of Sample Cells)")
            apply_plot_style(fig, [ax], cfg)
            return

        rep_df["Group"] = rep_df[COL_SAMPLE].astype(str).map(
            {str(k): str(v) for k, v in effective_gmap.items()}
        )
        rep_grouped = rep_df.dropna(subset=["Group"])
        groups = sorted(rep_grouped["Group"].astype(str).unique().tolist())
        n_grp = len(groups)
        grp_colors = resolve_category_colors(groups, cfg)

        bar_width = min(0.36, 0.78 / max(1, n_grp))
        offsets = (np.arange(n_grp) - (n_grp - 1) / 2.0) * bar_width

        cn_ymax: Dict[str, float] = {cn: 0.0 for cn in cn_order}
        for g_idx, grp in enumerate(groups):
            sub_g = rep_grouped[rep_grouped["Group"] == grp]
            means = []
            sems = []
            for cn in cn_order:
                vals = sub_g.loc[sub_g["Cellular_Neighborhood"] == cn, "percentage"].to_numpy(
                    dtype=np.float64
                )
                means.append(float(np.mean(vals)) if vals.size else 0.0)
                sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
                sems.append(sem)
                if vals.size:
                    cn_ymax[cn] = max(cn_ymax[cn], float(np.max(vals)))

            xs = x_base + offsets[g_idx]
            ax.bar(
                xs,
                means,
                width=bar_width * 0.9,
                yerr=sems,
                capsize=2.5,
                label=grp,
                color=grp_colors[grp],
                edgecolor="#111111",
                linewidth=0.5,
                alpha=0.85,
            )
            for c_idx, cn in enumerate(cn_order):
                vals = sub_g.loc[sub_g["Cellular_Neighborhood"] == cn, "percentage"].to_numpy(
                    dtype=np.float64
                )
                if vals.size:
                    jitter = (
                        np.linspace(-bar_width * 0.22, bar_width * 0.22, vals.size)
                        if vals.size > 1
                        else np.array([0.0])
                    )
                    ax.scatter(
                        np.full(vals.size, xs[c_idx]) + jitter,
                        vals,
                        color="#FFFFFF",
                        edgecolor="#111111",
                        s=18,
                        zorder=4,
                    )

        stat_method = self.combo_stat_test.currentText()
        stats_df = compute_cn_cohort_stats(
            rep_df,
            effective_gmap,
            value_col="percentage",
            stat_test=stat_method,
            correction="fdr_bh",
        )
        self._latest_stats_df = stats_df

        ax.set_xticks(x_base)
        ax.set_xticklabels(cn_order)
        if not cfg.title:
            ax.set_title("Cohort Cellular Neighborhood Abundance & Statistics")
        if not cfg.xlabel:
            ax.set_xlabel("Cellular Neighborhood / Topic")
        if not cfg.ylabel:
            ax.set_ylabel("Sample Abundance (%)")
        ax.legend(loc="upper right")

        apply_plot_style(fig, [ax], cfg)

        # Overlay collision-free significance brackets across all CNs uniformly
        if not stats_df.empty:
            group_pos = {g: i for i, g in enumerate(groups)}
            comps_all: List[Tuple[str, str, float, float]] = []
            x_pos_all: Dict[str, float] = {}
            y_max_all: Dict[str, float] = {}
            for c_idx, cn in enumerate(cn_order):
                for grp in groups:
                    key = f"{c_idx}::{grp}"
                    x_pos_all[key] = float(x_base[c_idx] + offsets[group_pos[grp]])
                    y_max_all[key] = max(cn_ymax.get(cn, 1.0), 1.0)
                sub_st = stats_df[stats_df["Cellular_Neighborhood"] == cn]
                for row in sub_st.itertuples(index=False):
                    comps_all.append(
                        (
                            f"{c_idx}::{row.group1}",
                            f"{c_idx}::{row.group2}",
                            float(row.p_value),
                            float(row.q_value),
                        )
                    )
            if comps_all:
                draw_significance_brackets(
                    ax,
                    comps_all,
                    x_positions=x_pos_all,
                    y_maxes=y_max_all,
                    config=cfg,
                )

    def copy_to_clipboard(self) -> bool:
        """Copy the active tab figure to the Qt clipboard."""
        ok = copy_figure_to_clipboard(
            self.current_figure(), config=self.style_config, dpi=self.style_config.dpi
        )
        self.lbl_status.setText(
            f"Copied active tab figure to clipboard at {self.style_config.dpi} DPI."
            if ok
            else "Clipboard unavailable in headless mode."
        )
        return ok

    def export_figure_to_path(self, filepath: Union[str, Path]) -> Path:
        """Export the active tab figure directly to *filepath*."""
        out = export_publication_figure(self.current_figure(), filepath, self.style_config)
        self.lbl_status.setText(f"Exported figure to {out.name}.")
        iris_log("IRIS-Analyzer", "Neighborhoods", f"Exported figure to {out}.", flush=True)
        return out

    def export_figure(self) -> Optional[Path]:
        """Prompt user for an export path and save the active figure."""
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
        """Export multi-sheet Source Data workbook (.xlsx) to *filepath*."""
        comp, enrichment, cn_names, cell_types = _extract_result_matrices(
            self.result, self.niche_metadata
        )
        summary_df = pd.DataFrame(enrichment, index=cn_names, columns=cell_types)
        summary_df.index.name = "Cellular_Neighborhood"
        if comp.size > 0 and cell_types:
            for j, ct in enumerate(cell_types):
                summary_df[f"{ct}__Weight"] = comp[:, j]

        replicates_df = _extract_replicates_with_custom_names(self.result, self.niche_metadata)
        effective_gmap = dict(self.group_map) or _infer_group_map(self.result.updated_df)
        stats_df = self._latest_stats_df
        if stats_df.empty and effective_gmap:
            stats_df = compute_cn_cohort_stats(
                replicates_df,
                effective_gmap,
                value_col="percentage",
                stat_test=self.combo_stat_test.currentText(),
            )

        meta: Dict[str, Any] = {
            k: v
            for k, v in self.result.metadata.items()
            if not isinstance(v, (pd.DataFrame, np.ndarray))
        }
        meta.update({f"style.{k}": v for k, v in self.style_config.to_dict().items()})
        out = export_source_data_excel(
            filepath,
            summary_df=summary_df,
            replicates_df=replicates_df,
            stats_df=stats_df if not stats_df.empty else None,
            metadata=meta,
            include_empty_sheets=True,
        )
        self.lbl_status.setText(f"Exported Source Data workbook to {out.name}.")
        iris_log(
            "IRIS-Analyzer",
            "Neighborhoods",
            f"Exported source data workbook to {out}.",
            flush=True,
        )
        return out

    def export_source_data(self) -> Optional[Path]:
        """Prompt user for an Excel path and write Source Data workbook."""
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
        """Clear figures cleanly when closing the Studio dialog."""
        for fig in self._tab_figures:
            fig.clear()
        super().closeEvent(event)


class CellularNeighborhoodsWidget(QWidget):
    """Dock widget for Moving Window CN and Spatial-LDA microenvironment analysis."""

    analysisCompleted = Signal(object)
    nicheMetadataChanged = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None, ctx: Any = None) -> None:
        super().__init__(parent)
        self.ctx: Any = ctx
        self.df: Optional[pd.DataFrame] = None
        self.markers: List[str] = []
        self.celltype_col: str = COL_CELL_TYPE
        self.scale_factor: float = 1.0
        self.group_map: Dict[str, str] = {}
        self.phenotype_colors: Dict[str, str] = {}
        self.niche_metadata: Dict[int, Dict[str, str]] = {}
        self.last_result: Optional[NeighborhoodResult] = None
        self.popup_dialog: Optional[CellularNeighborhoodStudioDialog] = None
        self._table_updating: bool = False

        self.setStyleSheet(SHARED_DARK_THEME)
        self._build_ui()
        self._sync_parameter_visibility()

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

        # Method & Parameters Box
        cfg_box, cfg_layout = create_section_card("Cellular Neighborhood Analysis")

        method_row = QHBoxLayout()
        method_row.setSpacing(8)
        method_row.addWidget(QLabel("Algorithm:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems([METHOD_MOVING_WINDOW, METHOD_SPATIAL_LDA])
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        method_row.addWidget(self.combo_method, 1)
        cfg_layout.addLayout(method_row)

        self.param_stack = QStackedWidget()
        self.param_stack.setStyleSheet(
            "QStackedWidget, QWidget#irisCardBody { background-color: transparent; }"
        )

        # Page 0: Moving Window
        mw_page = QWidget()
        mw_page.setObjectName("irisCardBody")
        mw_form = QFormLayout(mw_page)
        mw_form.setContentsMargins(2, 2, 2, 2)
        mw_form.setSpacing(6)

        self.combo_mw_mode = QComboBox()
        self.combo_mw_mode.addItems(["k-NN", "Radius (\u00b5m)"])
        self.combo_mw_mode.currentTextChanged.connect(self._sync_parameter_visibility)
        mw_form.addRow("Window Mode:", self.combo_mw_mode)

        self.lbl_mw_k = QLabel("Window k:")
        self.spin_mw_k = QSpinBox()
        self.spin_mw_k.setRange(1, 250)
        self.spin_mw_k.setValue(10)
        mw_form.addRow(self.lbl_mw_k, self.spin_mw_k)

        self.lbl_mw_radius = QLabel("Radius (\u00b5m):")
        self.spin_mw_radius = QDoubleSpinBox()
        self.spin_mw_radius.setRange(1.0, 2000.0)
        self.spin_mw_radius.setValue(50.0)
        self.spin_mw_radius.setDecimals(1)
        mw_form.addRow(self.lbl_mw_radius, self.spin_mw_radius)

        self.combo_mw_cluster = QComboBox()
        self.combo_mw_cluster.addItems(["MiniBatchKMeans", "KMeans", "Leiden"])
        self.combo_mw_cluster.currentTextChanged.connect(self._sync_parameter_visibility)
        mw_form.addRow("Clustering:", self.combo_mw_cluster)

        self.lbl_mw_niches = QLabel("Target CNs:")
        self.spin_mw_niches = QSpinBox()
        self.spin_mw_niches.setRange(2, 50)
        self.spin_mw_niches.setValue(10)
        mw_form.addRow(self.lbl_mw_niches, self.spin_mw_niches)

        self.lbl_mw_res = QLabel("Resolution:")
        self.spin_mw_resolution = QDoubleSpinBox()
        self.spin_mw_resolution.setRange(0.05, 10.0)
        self.spin_mw_resolution.setSingleStep(0.05)
        self.spin_mw_resolution.setValue(0.50)
        self.spin_mw_resolution.setDecimals(2)
        mw_form.addRow(self.lbl_mw_res, self.spin_mw_resolution)

        self.chk_mw_include_self = QCheckBox("Include Self in Window")
        self.chk_mw_include_self.setChecked(True)
        mw_form.addRow("", self.chk_mw_include_self)

        self.chk_mw_coordination = QCheckBox("Inter-CN Coordination")
        self.chk_mw_coordination.setChecked(True)
        mw_form.addRow("", self.chk_mw_coordination)

        self.chk_mw_contexts = QCheckBox("Spatial Contexts (k=40, thr=0.85)")
        self.chk_mw_contexts.setChecked(False)
        mw_form.addRow("", self.chk_mw_contexts)

        self.param_stack.addWidget(mw_page)

        # Page 1: Spatial-LDA
        lda_page = QWidget()
        lda_page.setObjectName("irisCardBody")
        lda_form = QFormLayout(lda_page)
        lda_form.setContentsMargins(2, 2, 2, 2)
        lda_form.setSpacing(6)

        self.spin_lda_topics = QSpinBox()
        self.spin_lda_topics.setRange(2, 30)
        self.spin_lda_topics.setValue(6)
        lda_form.addRow("Topics (K):", self.spin_lda_topics)

        self.spin_lda_penalty = QDoubleSpinBox()
        self.spin_lda_penalty.setRange(0.0, 5.0)
        self.spin_lda_penalty.setSingleStep(0.05)
        self.spin_lda_penalty.setValue(0.25)
        self.spin_lda_penalty.setDecimals(2)
        lda_form.addRow("Difference Penalty:", self.spin_lda_penalty)

        self.combo_lda_window = QComboBox()
        self.combo_lda_window.addItems(["Radius (\u00b5m)", "k-NN"])
        self.combo_lda_window.currentTextChanged.connect(self._sync_parameter_visibility)
        lda_form.addRow("Feature Window:", self.combo_lda_window)

        self.lbl_lda_radius = QLabel("Radius (\u00b5m):")
        self.spin_lda_radius = QDoubleSpinBox()
        self.spin_lda_radius.setRange(1.0, 2000.0)
        self.spin_lda_radius.setValue(50.0)
        self.spin_lda_radius.setDecimals(1)
        lda_form.addRow(self.lbl_lda_radius, self.spin_lda_radius)

        self.lbl_lda_k = QLabel("Neighbors (k):")
        self.spin_lda_k = QSpinBox()
        self.spin_lda_k.setRange(1, 250)
        self.spin_lda_k.setValue(10)
        lda_form.addRow(self.lbl_lda_k, self.spin_lda_k)

        self.spin_lda_iters = QSpinBox()
        self.spin_lda_iters.setRange(1, 50)
        self.spin_lda_iters.setValue(5)
        lda_form.addRow("EM Iterations:", self.spin_lda_iters)

        self.param_stack.addWidget(lda_page)
        cfg_layout.addWidget(self.param_stack)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.btn_extract = QPushButton("Extract Cellular Neighborhoods")
        self.btn_extract.setStyleSheet(BUTTON_CTA)
        self.btn_extract.setMinimumHeight(28)
        self.btn_extract.clicked.connect(lambda: self.extract_neighborhoods())
        action_row.addWidget(self.btn_extract, 2)

        self.btn_groups = QPushButton("Cohort Groups...")
        self.btn_groups.setMinimumHeight(28)
        self.btn_groups.clicked.connect(self._open_group_dialog)
        action_row.addWidget(self.btn_groups, 1)
        cfg_layout.addLayout(action_row)

        root.addWidget(cfg_box)

        # Canvas Overlay Mode + Editable CN/Topic Metadata Table
        overlay_box, ov_layout = create_section_card("Canvas Overlay & Neighborhood Table")

        ov_row = QHBoxLayout()
        ov_row.setSpacing(8)
        ov_row.addWidget(QLabel("Overlay:"))
        self.combo_overlay_mode = QComboBox()
        self.combo_overlay_mode.addItems([
            OVERLAY_DISCRETE,
            OVERLAY_TOPIC_PROB,
            OVERLAY_ENTROPY,
        ])
        self.combo_overlay_mode.currentTextChanged.connect(self._on_overlay_mode_changed)
        ov_row.addWidget(self.combo_overlay_mode, 1)
        ov_layout.addLayout(ov_row)

        topic_row = QHBoxLayout()
        topic_row.setSpacing(8)
        self.lbl_overlay_topic = QLabel("Topic Channel:")
        self.combo_overlay_topic = QComboBox()
        self.combo_overlay_topic.currentIndexChanged.connect(self.render_canvas_overlay)
        topic_row.addWidget(self.lbl_overlay_topic)
        topic_row.addWidget(self.combo_overlay_topic, 1)
        ov_layout.addLayout(topic_row)

        self.table_meta = QTableWidget(0, 4)
        self.table_meta.setHorizontalHeaderLabels(["ID", "Name", "Color", "Cells"])
        self.table_meta.verticalHeader().setVisible(False)
        self.table_meta.setMinimumHeight(130)
        hdr = self.table_meta.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_meta.itemChanged.connect(self._on_table_item_changed)
        ov_layout.addWidget(self.table_meta)

        root.addWidget(overlay_box)

        # Preview Plot & Pop-out Studio Button
        plot_box, plot_layout = create_section_card(margins=(10, 6, 10, 10), spacing=8)

        hdr_row = QHBoxLayout()
        lbl_preview = QLabel("Neighborhood Composition Preview")
        style_section_header(lbl_preview, subtitle=False)
        hdr_row.addWidget(lbl_preview)
        hdr_row.addStretch(1)

        self.btn_popout = QPushButton("\u2922 Pop-up Window")
        self.btn_popout.setStyleSheet(POPOUT_BUTTON_STYLE)
        self.btn_popout.setToolTip("Open Cellular Neighborhoods Figure Studio")
        self.btn_popout.clicked.connect(self.open_popup_window)
        hdr_row.addWidget(self.btn_popout)
        plot_layout.addLayout(hdr_row)

        self.figure = Figure(figsize=(5.4, 3.8), dpi=100)
        self.figure.patch.set_facecolor("#14161d")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(220)
        self.canvas.setToolTip("Double-click to open Cellular Neighborhoods Figure Studio")
        self.canvas.mouseDoubleClickEvent = lambda _e: self.open_popup_window()
        plot_layout.addWidget(self.canvas, 1)

        self.lbl_status = QLabel("Load single-cell data to extract cellular neighborhoods.")
        self.lbl_status.setStyleSheet(SECTION_SUBTITLE_STYLE)
        self.lbl_status.setWordWrap(True)
        plot_layout.addWidget(self.lbl_status)

        root.addWidget(plot_box, 1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

    def set_method(self, method: str) -> None:
        """Select neighborhood extraction algorithm (accepting legacy aliases)."""
        canonical = _normalize_method_name(method)
        self.combo_method.setCurrentText(canonical)

    def _on_method_changed(self, idx: int) -> None:
        self.param_stack.setCurrentIndex(idx)
        self._sync_parameter_visibility()

    def _sync_parameter_visibility(self) -> None:
        is_mw_radius = "Radius" in self.combo_mw_mode.currentText()
        self.lbl_mw_k.setVisible(not is_mw_radius)
        self.spin_mw_k.setVisible(not is_mw_radius)
        self.lbl_mw_radius.setVisible(is_mw_radius)
        self.spin_mw_radius.setVisible(is_mw_radius)

        is_leiden = self.combo_mw_cluster.currentText() == "Leiden"
        self.lbl_mw_niches.setVisible(not is_leiden)
        self.spin_mw_niches.setVisible(not is_leiden)
        self.lbl_mw_res.setVisible(is_leiden)
        self.spin_mw_resolution.setVisible(is_leiden)

        is_lda_knn = self.combo_lda_window.currentText() == "k-NN"
        self.lbl_lda_radius.setVisible(not is_lda_knn)
        self.spin_lda_radius.setVisible(not is_lda_knn)
        self.lbl_lda_k.setVisible(is_lda_knn)
        self.spin_lda_k.setVisible(is_lda_knn)

        is_prob_mode = self.combo_overlay_mode.currentText() == OVERLAY_TOPIC_PROB
        self.lbl_overlay_topic.setVisible(is_prob_mode)
        self.combo_overlay_topic.setVisible(is_prob_mode)

    def set_data(
        self,
        df: Optional[pd.DataFrame],
        markers: Optional[Sequence[str]] = None,
        celltype_col: str = "cell_type",
        scale_factor: float = 1.0,
        ctx: Any = None,
    ) -> None:
        """Bind active single-cell DataFrame, markers, scale factor, and viewer context."""
        self.df = df
        if markers is not None:
            self.markers = list(markers)
        self.celltype_col = _resolve_celltype_col_safe(df, celltype_col)
        self.scale_factor = float(scale_factor) if scale_factor and scale_factor > 0 else 1.0
        if ctx is not None:
            self.ctx = ctx

        if not self.group_map:
            inferred = _infer_group_map(self.df)
            if inferred:
                self.group_map = inferred

        self._refresh_phenotype_colors()

        n_rows = len(self.df) if self.df is not None else 0
        if self.last_result is not None and len(self.last_result.updated_df) != n_rows:
            self.last_result = None
            self.figure.clear()
            self.canvas.draw_idle()

        self.lbl_status.setText(
            f"Ready: {n_rows} cells (phenotype col: {self.celltype_col})."
            if n_rows > 0
            else "No single-cell data loaded."
        )

    def _refresh_phenotype_colors(self) -> None:
        if self.ctx is None:
            return
        state = getattr(self.ctx, "state", None)
        if state is not None and isinstance(getattr(state, "celltype_colors", None), dict):
            for k, v in state.celltype_colors.items():
                self.phenotype_colors[str(k)] = str(v)
                self.phenotype_colors[str(k).lower()] = str(v)

        legend_w = getattr(self.ctx, "legend_widget", None)
        if legend_w is not None:
            rows = getattr(legend_w, "ref_ct_rows", []) or getattr(legend_w, "seg_rows", [])
            for row in rows:
                name = str(getattr(row, "name", "")).split(" [")[0].strip()
                hex_c = getattr(row, "hex_color", None)
                if name and hex_c:
                    self.phenotype_colors[name] = str(hex_c)
                    self.phenotype_colors[name.lower()] = str(hex_c)

    def set_group_map(self, group_map: Optional[Dict[str, str]]) -> None:
        """Set explicit sample -> biological cohort group mapping."""
        self.group_map = dict(group_map or {})
        if self.popup_dialog is not None and self.popup_dialog.isVisible():
            self.popup_dialog.update_result(
                self.last_result,
                niche_metadata=self.niche_metadata,
                phenotype_colors=self.phenotype_colors,
                group_map=self.group_map,
            )

    def _build_extraction_callable(self, method: Optional[str] = None):
        """Capture UI parameters and return a zero-arg callable executing analysis."""
        df_snap = self.df.copy()
        ct_col = self.celltype_col
        scale = float(self.scale_factor)
        canonical_method = _normalize_method_name(
            method if method is not None else self.combo_method.currentText()
        )

        if canonical_method == METHOD_SPATIAL_LDA:
            n_topics = int(self.spin_lda_topics.value())
            penalty = float(self.spin_lda_penalty.value())
            feat_mode = "knn" if self.combo_lda_window.currentText() == "k-NN" else "radius"
            radius_um = float(self.spin_lda_radius.value())
            k_nn = int(self.spin_lda_k.value()) if feat_mode == "knn" else None
            n_iters = int(self.spin_lda_iters.value())

            def _run() -> NeighborhoodResult:
                return run_spatial_lda(
                    df_snap,
                    cell_type_col=ct_col,
                    n_topics=n_topics,
                    difference_penalty=penalty,
                    radius_um=radius_um,
                    k_neighbors=k_nn,
                    featurization_mode=feat_mode,
                    microns_per_pixel=scale,
                    n_iters=n_iters,
                )

            return _run

        mw_mode = "radius" if "Radius" in self.combo_mw_mode.currentText() else "knn"
        k_val = int(self.spin_mw_k.value())
        rad_val = float(self.spin_mw_radius.value())
        n_niches = int(self.spin_mw_niches.value())
        cluster_method = self.combo_mw_cluster.currentText()
        resolution = float(self.spin_mw_resolution.value())
        include_self = bool(self.chk_mw_include_self.isChecked())
        do_coord = bool(self.chk_mw_coordination.isChecked())
        do_ctx = bool(self.chk_mw_contexts.isChecked())

        def _run_mw() -> NeighborhoodResult:
            return run_moving_window_cn(
                df_snap,
                cell_type_col=ct_col,
                window_mode=mw_mode,
                k_neighbors=k_val,
                radius_um=rad_val,
                include_self=include_self,
                n_niches=n_niches,
                cluster_method=cluster_method,
                resolution=resolution,
                microns_per_pixel=scale,
                compute_coordination=do_coord,
                compute_spatial_contexts=do_ctx,
            )

        return _run_mw

    def extract_neighborhoods(
        self,
        *,
        sync: Optional[bool] = None,
        method: Optional[str] = None,
    ) -> Optional[NeighborhoodResult]:
        """Run Moving Window CN or Spatial-LDA extraction (async via Napari if available)."""
        if self.df is None or self.df.empty:
            self.lbl_status.setText("Load single-cell data before extracting neighborhoods.")
            return None

        canonical_method = _normalize_method_name(
            method if method is not None else self.combo_method.currentText()
        )
        s_col = find_sample_column(self.df)
        n_samples = (
            int(self.df[s_col].nunique())
            if s_col and s_col in self.df.columns
            else 1
        )
        if canonical_method == METHOD_SPATIAL_LDA:
            win_desc = (
                f"k={int(self.spin_lda_k.value())}"
                if self.combo_lda_window.currentText() == "k-NN"
                else f"r={float(self.spin_lda_radius.value()):.1f}um"
            )
            iris_log(
                "IRIS-Analyzer",
                "Neighborhoods",
                f"Running Spatial-LDA (topics={int(self.spin_lda_topics.value())}, "
                f"{win_desc}, {n_samples} sample(s), {len(self.df):,} cells)...",
                flush=True,
            )
        else:
            win_desc = (
                f"r={float(self.spin_mw_radius.value()):.1f}um"
                if "Radius" in self.combo_mw_mode.currentText()
                else f"k={int(self.spin_mw_k.value())}"
            )
            cluster_method = self.combo_mw_cluster.currentText()
            target_desc = (
                f"res={float(self.spin_mw_resolution.value()):.2f}"
                if cluster_method == "Leiden"
                else f"target={int(self.spin_mw_niches.value())}"
            )
            iris_log(
                "IRIS-Analyzer",
                "Neighborhoods",
                f"Running Moving Window ({cluster_method}, "
                f"{win_desc}, {target_desc}, {n_samples} sample(s), {len(self.df):,} cells)...",
                flush=True,
            )

        run_fn = self._build_extraction_callable(method=canonical_method)
        has_tracker = (
            self.ctx is not None
            and hasattr(self.ctx, "track_worker")
            and callable(self.ctx.track_worker)
        )
        use_async = (not sync) if sync is not None else bool(has_tracker)

        if use_async and has_tracker:
            from napari.qt import thread_worker

            self.btn_extract.setEnabled(False)
            self.btn_extract.setText("Processing...")

            @thread_worker
            def _worker():
                return run_fn()

            worker = _worker()
            worker.returned.connect(self._on_extraction_complete)
            self.ctx.track_worker(
                worker,
                on_error=self._on_extraction_error,
                error_title="Neighborhood Error",
                error_prefix="Failed to extract cellular neighborhoods",
            )
            worker.start()
            return None

        try:
            result = run_fn()
        except Exception as exc:
            self._on_extraction_error(exc)
            return None
        self._on_extraction_complete(result)
        return result

    def _on_extraction_complete(self, result: NeighborhoodResult) -> None:
        """Handle finished CN / Spatial-LDA computation on the Qt main thread."""
        self.btn_extract.setEnabled(True)
        self.btn_extract.setText("Extract Cellular Neighborhoods")
        self.last_result = result
        self.df = result.updated_df

        self._propagate_results_to_ctx(result)
        self._populate_metadata_table(result)
        self._populate_topic_selector(result)
        self._render_compact_preview(result)
        self.render_canvas_overlay()

        s_col = find_sample_column(result.updated_df)
        n_samples = (
            int(result.updated_df[s_col].nunique())
            if s_col and s_col in result.updated_df.columns
            else 1
        )
        if isinstance(result, SpatialLDAResult):
            k_topics = result.topic_probabilities.shape[1]
            mean_ent = float(np.mean(result.topic_entropy))
            self.lbl_status.setText(
                f"Spatial-LDA complete: {k_topics} topics | "
                f"diff L1={result.difference_l1_norm:.2f} | "
                f"mean entropy={mean_ent:.3f}"
            )
            iris_log(
                "IRIS-Analyzer",
                "Neighborhoods",
                f"Spatial-LDA complete: {k_topics} topic(s), {n_samples} sample(s), "
                f"{len(result.updated_df):,} cells, mean entropy={mean_ent:.3f}.",
                flush=True,
            )
        else:
            n_cns = result.centers.shape[0]
            self.lbl_status.setText(
                f"Moving Window complete: {n_cns} CNs across "
                f"{len(result.updated_df)} cells."
            )
            iris_log(
                "IRIS-Analyzer",
                "Neighborhoods",
                f"Moving Window complete: {n_cns} cluster(s), {n_samples} sample(s), "
                f"{len(result.updated_df):,} cells.",
                flush=True,
            )

        if self.popup_dialog is not None and self.popup_dialog.isVisible():
            self.popup_dialog.update_result(
                result,
                niche_metadata=self.niche_metadata,
                phenotype_colors=self.phenotype_colors,
                group_map=self.group_map,
            )

        self.analysisCompleted.emit(result)

    def _on_extraction_error(self, exc: Exception) -> None:
        self.btn_extract.setEnabled(True)
        self.btn_extract.setText("Extract Cellular Neighborhoods")
        self.lbl_status.setText(f"Extraction error: {exc}")
        iris_log(
            "IRIS-Analyzer",
            "Neighborhoods",
            f"Extraction failed: {exc}",
            level="ERROR",
            flush=True,
        )

    def _propagate_results_to_ctx(self, result: NeighborhoodResult) -> None:
        """Sync neighborhood columns back into ``ctx.state`` if attached."""
        state = getattr(self.ctx, "state", None) if self.ctx is not None else None
        if state is None:
            return
        updated_df = result.updated_df
        sample_col = find_sample_column(updated_df)
        new_cols = [
            c
            for c in updated_df.columns
            if c in ("Cellular_Neighborhood", "Cellular_Neighborhood_ID", "Spatial_Context")
            or c.startswith("SpatialLDA_")
            or (c.startswith("Topic_") and c.endswith("_Prob"))
        ]

        master = getattr(state, "master_df", None)
        if master is None or len(master) == len(updated_df):
            state.master_df = updated_df
            state.current_mapped_df = updated_df
        else:
            state.current_mapped_df = updated_df
            if sample_col and sample_col in master.columns:
                for sname in updated_df[sample_col].astype(str).unique():
                    sub_new = updated_df[updated_df[sample_col].astype(str) == str(sname)]
                    mask = master[sample_col].astype(str) == str(sname)
                    if int(mask.sum()) == len(sub_new) and len(sub_new) > 0:
                        for col in new_cols:
                            master.loc[mask, col] = sub_new[col].to_numpy()

        resolved_dfs = getattr(state, "resolved_dfs", None)
        if isinstance(resolved_dfs, dict) and sample_col is not None:
            for sname, sdf in list(resolved_dfs.items()):
                sub = updated_df[updated_df[sample_col].astype(str) == str(sname)]
                if len(sub) == len(sdf) and len(sub) > 0:
                    for col in new_cols:
                        sdf[col] = sub[col].to_numpy()

    def _populate_metadata_table(self, result: NeighborhoodResult) -> None:
        comp, _, _, _ = _extract_result_matrices(result, {})
        n_cns = comp.shape[0]
        is_lda = isinstance(result, SpatialLDAResult)
        counts = result.updated_df["Cellular_Neighborhood_ID"].value_counts().to_dict()

        self._table_updating = True
        try:
            self.niche_metadata.clear()
            self.table_meta.setRowCount(n_cns)
            for i in range(n_cns):
                name_str = f"Topic {i}" if is_lda else f"CN {i}"
                hex_color = CN_PALETTE[i % len(CN_PALETTE)]
                self.niche_metadata[i] = {"name": name_str, "color": hex_color}

                id_item = QTableWidgetItem(str(i))
                id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
                id_item.setTextAlignment(Qt.AlignCenter)
                self.table_meta.setItem(i, 0, id_item)

                name_item = QTableWidgetItem(name_str)
                self.table_meta.setItem(i, 1, name_item)

                swatch = QPushButton()
                swatch.setFixedSize(26, 20)
                swatch.setStyleSheet(
                    f"background-color: {hex_color}; border: 1px solid #888; border-radius: 3px;"
                )
                swatch.clicked.connect(lambda _c=False, cid=i: self._pick_cn_color(cid))
                self.table_meta.setCellWidget(i, 2, swatch)

                cnt_item = QTableWidgetItem(str(int(counts.get(i, 0))))
                cnt_item.setFlags(cnt_item.flags() & ~Qt.ItemIsEditable)
                cnt_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table_meta.setItem(i, 3, cnt_item)
        finally:
            self._table_updating = False

        self.nicheMetadataChanged.emit(dict(self.niche_metadata))

    def _populate_topic_selector(self, result: NeighborhoodResult) -> None:
        self.combo_overlay_topic.blockSignals(True)
        self.combo_overlay_topic.clear()
        if isinstance(result, SpatialLDAResult):
            k_topics = result.topic_probabilities.shape[1]
            self.combo_overlay_topic.addItems([
                self.niche_metadata.get(k, {}).get("name", f"Topic {k}")
                for k in range(k_topics)
            ])
        elif self.combo_overlay_mode.currentText() != OVERLAY_DISCRETE:
            self.combo_overlay_mode.blockSignals(True)
            self.combo_overlay_mode.setCurrentText(OVERLAY_DISCRETE)
            self.combo_overlay_mode.blockSignals(False)
        self.combo_overlay_topic.blockSignals(False)
        self._sync_parameter_visibility()

    def _pick_cn_color(self, cid: int) -> None:
        current = self.niche_metadata.get(cid, {}).get("color", "#3A6A9A")
        chosen = QColorDialog.getColor(QColor(current), self, f"Pick Color for ID {cid}")
        if chosen.isValid():
            self.set_cn_color(cid, chosen.name())

    def set_cn_color(self, cid: int, hex_color: str) -> None:
        """Programmatically assign a custom hex color to CN/Topic *cid*."""
        if cid not in self.niche_metadata:
            return
        self.niche_metadata[cid]["color"] = hex_color
        btn = self.table_meta.cellWidget(cid, 2)
        if isinstance(btn, QPushButton):
            btn.setStyleSheet(
                f"background-color: {hex_color}; border: 1px solid #888; border-radius: 3px;"
            )
        self._after_metadata_edited()

    def set_cn_name(self, cid: int, name: str) -> None:
        """Programmatically rename CN/Topic *cid*."""
        clean = name.strip()
        if not clean or cid not in self.niche_metadata:
            return
        self._table_updating = True
        try:
            item = self.table_meta.item(cid, 1)
            if item is not None:
                item.setText(clean)
        finally:
            self._table_updating = False
        self.niche_metadata[cid]["name"] = clean
        if self.last_result is not None and "Cellular_Neighborhood_ID" in self.last_result.updated_df.columns:
            ids = self.last_result.updated_df["Cellular_Neighborhood_ID"].to_numpy()
            mask = ids == cid
            if np.any(mask):
                self.last_result.updated_df.loc[mask, "Cellular_Neighborhood"] = clean
        self._after_metadata_edited()

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._table_updating or item.column() != 1:
            return
        cid = item.row()
        new_name = item.text().strip()
        if new_name and cid in self.niche_metadata:
            self.niche_metadata[cid]["name"] = new_name
            if (
                self.last_result is not None
                and "Cellular_Neighborhood_ID" in self.last_result.updated_df.columns
            ):
                mask = self.last_result.updated_df["Cellular_Neighborhood_ID"].to_numpy() == cid
                if np.any(mask):
                    self.last_result.updated_df.loc[mask, "Cellular_Neighborhood"] = new_name
            self._after_metadata_edited()

    def _after_metadata_edited(self) -> None:
        self.nicheMetadataChanged.emit(dict(self.niche_metadata))
        if self.last_result is not None:
            self._populate_topic_selector(self.last_result)
            self._render_compact_preview(self.last_result)
            self.render_canvas_overlay()
            if self.popup_dialog is not None and self.popup_dialog.isVisible():
                self.popup_dialog.update_result(
                    self.last_result,
                    niche_metadata=self.niche_metadata,
                    phenotype_colors=self.phenotype_colors,
                    group_map=self.group_map,
                )

    def _on_overlay_mode_changed(self, _mode: str) -> None:
        self._sync_parameter_visibility()
        self.render_canvas_overlay()

    def _render_compact_preview(self, result: NeighborhoodResult) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#14161d")

        comp, _, cn_names, cell_types = _extract_result_matrices(result, self.niche_metadata)
        if comp.size == 0 or not cell_types:
            ax.text(0.5, 0.5, "No composition data.", ha="center", va="center", color="#999")
            self.canvas.draw_idle()
            return

        n_cns = comp.shape[0]
        bottom = np.zeros(n_cns, dtype=np.float64)
        x = np.arange(n_cns)
        cmap = plt.colormaps.get_cmap("tab20")

        for idx, ct in enumerate(cell_types):
            vals = comp[:, idx]
            color = (
                self.phenotype_colors.get(ct)
                or self.phenotype_colors.get(ct.lower())
                or DEFAULT_CELLTYPE_PALETTE[idx % len(DEFAULT_CELLTYPE_PALETTE)]
                or cmap(idx % 20)
            )
            ax.bar(
                x,
                vals,
                bottom=bottom,
                label=ct,
                color=color,
                edgecolor="#111111",
                linewidth=0.4,
            )
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(cn_names, rotation=25, ha="right", color="#E0E0E0", fontsize=8)
        ax.tick_params(axis="y", colors="#E0E0E0", labelsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Proportion", color="#E0E0E0", fontsize=9)
        ax.legend(
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            fontsize=7,
            facecolor="#1f232b",
            edgecolor="#3c4352",
            labelcolor="white",
        )
        for spine in ax.spines.values():
            spine.set_color("#4A5162")

        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

    def render_canvas_overlay(self) -> List[Any]:
        """Render the selected CN/Topic/Entropy overlay onto ``ctx.viewer`` if present."""
        viewer = getattr(self.ctx, "viewer", None) if self.ctx is not None else None
        if viewer is None or not hasattr(viewer, "layers") or self.last_result is None:
            return []

        # Remove previously rendered neighborhood overlays cleanly
        for lyr in list(viewer.layers):
            meta = getattr(lyr, "metadata", None)
            if isinstance(meta, dict) and meta.get("iris_neighborhood_overlay"):
                viewer.layers.remove(lyr)

        df = self.last_result.updated_df
        if df.empty:
            return []

        mode = self.combo_overlay_mode.currentText()
        state = getattr(self.ctx, "state", None)
        sample_col = find_sample_column(df)

        # Gather per-sample masks if available
        masks_map: Dict[str, np.ndarray] = {}
        if state is not None:
            if isinstance(getattr(state, "masks_map", None), dict) and state.masks_map:
                masks_map = {str(k): v for k, v in state.masks_map.items() if v is not None}
            else:
                nuc_mask = None
                if hasattr(state, "get_active_masks"):
                    nuc_mask, _ = state.get_active_masks()
                if nuc_mask is None:
                    nuc_mask = getattr(state, "current_mask", None)
                if nuc_mask is not None:
                    sname = str(getattr(state, "current_sample_name", None) or "Sample_0")
                    masks_map = {sname: nuc_mask}

        created_layers: List[Any] = []
        if masks_map and hasattr(viewer, "add_image"):
            from napari.utils.colormaps import Colormap

            for sname, mask_arr in masks_map.items():
                mask = mask_arr[0] if mask_arr.ndim > 2 else mask_arr
                if sample_col is not None and sname in set(df[sample_col].astype(str)):
                    sub_df = df[df[sample_col].astype(str) == sname]
                elif len(masks_map) == 1:
                    sub_df = df
                else:
                    continue

                obj_ids = _resolve_mask_object_ids(mask, sub_df)
                valid_mask = obj_ids > 0
                if not np.any(valid_mask):
                    continue

                bounds_lyr = next(
                    (l for l in viewer.layers if getattr(l, "name", "") == f"Bounds - {sname}"),
                    None,
                )
                translation = getattr(bounds_lyr, "translate", None) if bounds_lyr else None

                if mode == OVERLAY_DISCRETE:
                    cn_ids = sub_df["Cellular_Neighborhood_ID"].to_numpy(dtype=np.int64)
                    for cid in sorted(np.unique(cn_ids[valid_mask]).tolist()):
                        target_objs = obj_ids[valid_mask & (cn_ids == cid)]
                        if target_objs.size == 0:
                            continue
                        bin_mask = np.isin(mask, target_objs).astype(np.uint8)
                        if not np.any(bin_mask):
                            continue
                        cinfo = self.niche_metadata.get(cid, {})
                        cname = cinfo.get("name", f"CN {cid}")
                        hex_c = cinfo.get("color", CN_PALETTE[cid % len(CN_PALETTE)])
                        colors = np.array([[0, 0, 0, 0], list(mcolors.to_rgb(hex_c)) + [1]])
                        lyr = viewer.add_image(
                            bin_mask,
                            name=f"CN Overlay [{sname}] - {cname}",
                            opacity=0.6,
                            blending="translucent",
                            colormap=Colormap(colors),
                            contrast_limits=[0, 1],
                            scale=(self.scale_factor, self.scale_factor),
                            metadata={
                                "iris_neighborhood_overlay": True,
                                "overlay_mode": mode,
                                "sample_name": sname,
                                "niche": cname,
                            },
                        )
                        if translation is not None:
                            lyr.translate = translation
                        created_layers.append(lyr)
                else:
                    if mode == OVERLAY_TOPIC_PROB:
                        t_idx = max(0, self.combo_overlay_topic.currentIndex())
                        col_name = f"Topic_{t_idx}_Prob"
                        if col_name not in sub_df.columns:
                            continue
                        scores = pd.to_numeric(sub_df[col_name], errors="coerce").fillna(0.0).to_numpy(
                            dtype=np.float32
                        )
                        cmap_name = "viridis"
                        layer_label = f"Topic {t_idx} Probability [{sname}]"
                    else:
                        if "SpatialLDA_Entropy" not in sub_df.columns:
                            continue
                        scores = pd.to_numeric(
                            sub_df["SpatialLDA_Entropy"], errors="coerce"
                        ).fillna(0.0).to_numpy(dtype=np.float32)
                        cmap_name = "magma"
                        layer_label = f"Transition Zone Entropy [{sname}]"

                    max_id = int(max(int(mask.max()), int(obj_ids[valid_mask].max())))
                    lut = np.zeros(max_id + 1, dtype=np.float32)
                    lut[obj_ids[valid_mask]] = scores[valid_mask]
                    score_img = lut[mask]
                    lyr = viewer.add_image(
                        score_img,
                        name=layer_label,
                        opacity=0.75,
                        blending="translucent",
                        colormap=cmap_name,
                        contrast_limits=[0.0, 1.0],
                        scale=(self.scale_factor, self.scale_factor),
                        metadata={
                            "iris_neighborhood_overlay": True,
                            "overlay_mode": mode,
                            "sample_name": sname,
                        },
                    )
                    if translation is not None:
                        lyr.translate = translation
                    created_layers.append(lyr)
        elif hasattr(viewer, "add_points") and {COL_X, COL_Y}.issubset(df.columns):
            coords_yx = np.column_stack([
                pd.to_numeric(df[COL_Y], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
                pd.to_numeric(df[COL_X], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
            ])
            if mode == OVERLAY_DISCRETE:
                cn_ids = df["Cellular_Neighborhood_ID"].to_numpy(dtype=np.int64)
                face_colors = [
                    self.niche_metadata.get(int(cid), {}).get(
                        "color", CN_PALETTE[int(cid) % len(CN_PALETTE)]
                    )
                    for cid in cn_ids
                ]
                lyr = viewer.add_points(
                    coords_yx,
                    face_color=face_colors,
                    size=8.0,
                    name="CN Overlay - Discrete",
                    scale=(self.scale_factor, self.scale_factor),
                    metadata={"iris_neighborhood_overlay": True, "overlay_mode": mode},
                )
                created_layers.append(lyr)
            else:
                if mode == OVERLAY_TOPIC_PROB:
                    t_idx = max(0, self.combo_overlay_topic.currentIndex())
                    col_name = f"Topic_{t_idx}_Prob"
                    if col_name not in df.columns:
                        return []
                    vals = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0).to_numpy(
                        dtype=np.float64
                    )
                    cmap = plt.colormaps.get_cmap("viridis")
                    layer_title = f"Topic {t_idx} Probability [0-1]"
                else:
                    if "SpatialLDA_Entropy" not in df.columns:
                        return []
                    vals = pd.to_numeric(
                        df["SpatialLDA_Entropy"], errors="coerce"
                    ).fillna(0.0).to_numpy(dtype=np.float64)
                    cmap = plt.colormaps.get_cmap("magma")
                    layer_title = "Transition Zone Entropy [0-1]"

                rgba = cmap(np.clip(vals, 0.0, 1.0))
                lyr = viewer.add_points(
                    coords_yx,
                    face_color=rgba,
                    size=8.0,
                    name=layer_title,
                    scale=(self.scale_factor, self.scale_factor),
                    metadata={"iris_neighborhood_overlay": True, "overlay_mode": mode},
                )
                created_layers.append(lyr)

        return created_layers

    def _open_group_dialog(self) -> None:
        if self.df is None or self.df.empty:
            QMessageBox.information(self, "No Data", "Load single-cell data first.")
            return
        sample_col = find_sample_column(self.df)
        if sample_col is None:
            QMessageBox.information(
                self, "Single Sample", "No sample identifier column detected in DataFrame."
            )
            return
        samples = list(dict.fromkeys(self.df[sample_col].astype(str).tolist()))
        dlg = _CohortGroupDialog(samples, self.group_map, parent=self.window())
        if dlg.exec_():
            self.set_group_map(dlg.get_group_map())
        dlg.close()

    def open_popup_window(self) -> Optional[CellularNeighborhoodStudioDialog]:
        """Open or focus the 5-tab Cellular Neighborhoods Figure Studio dialog."""
        if self.last_result is None:
            if self.df is not None and not self.df.empty:
                self.extract_neighborhoods(sync=True)
            else:
                QMessageBox.information(
                    self,
                    "No Neighborhood Data",
                    "Load single-cell data and extract neighborhoods first.",
                )
                return None
        if self.last_result is None:
            return None

        if self.popup_dialog is None:
            self.popup_dialog = CellularNeighborhoodStudioDialog(
                self.last_result,
                niche_metadata=self.niche_metadata,
                phenotype_colors=self.phenotype_colors,
                group_map=self.group_map,
                parent=self.window(),
            )
        else:
            self.popup_dialog.update_result(
                self.last_result,
                niche_metadata=self.niche_metadata,
                phenotype_colors=self.phenotype_colors,
                group_map=self.group_map,
            )

        self.popup_dialog.show()
        self.popup_dialog.raise_()
        self.popup_dialog.activateWindow()
        return self.popup_dialog

    def closeEvent(self, event: Any) -> None:
        """Close pop-up Studio dialog and release figure resources."""
        if self.popup_dialog is not None:
            self.popup_dialog.close()
        self.figure.clear()
        super().closeEvent(event)


__all__ = [
    "CN_PALETTE",
    "CellularNeighborhoodStudioDialog",
    "CellularNeighborhoodsWidget",
    "METHOD_MOVING_WINDOW",
    "METHOD_SPATIAL_LDA",
    "OVERLAY_DISCRETE",
    "OVERLAY_ENTROPY",
    "OVERLAY_TOPIC_PROB",
]
