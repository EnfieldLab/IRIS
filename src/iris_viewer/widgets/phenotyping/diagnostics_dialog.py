"""Phenotyping Diagnostics Studio Dialog.

Interactive multi-tab diagnostic window for single-cell phenotyping outputs:
  - Confidence & Novelty: posterior confidence distribution + STELLAR novel cell summary
  - Nimbus Positivity Heatmap: per-phenotype mean marker positivity clustermap
  - Method Concordance Matrix: cross-method confusion matrix, Accuracy, Cohen's Kappa
  - Direct handoff to Figure Studio for export
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from iris_common.logging import iris_log
from iris_common.widgets.publication_studio import (
    THEME_PRESETS,
    PlotStyleConfig,
    PublicationPlotStudioDialog,
)
from iris_common.widgets.theme import BUTTON_STUDIO, DIALOG_DARK_THEME, configure_tab_widget
from iris_viewer.widgets.phenotyping.thresholding import resolve_marker_col

try:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    NavigationToolbar = None


_SKIP_CAT_COLS = {
    "x",
    "y",
    "z",
    "pos_x",
    "pos_y",
    "x_centroid",
    "y_centroid",
    "centroid_x",
    "centroid_y",
    "area",
    "eccentricity",
    "perimeter",
    "solidity",
    "objectnumber",
    "cell_id",
    "id",
}


class PhenotyperDiagnosticDialog(QDialog):
    """Pop-up diagnostics window for confidence, Nimbus positivity, and method concordance."""

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        ctx: Any = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Phenotyping Diagnostics")
        self.resize(1180, 780)
        self.setMinimumSize(860, 560)
        self.setWindowFlags(self.windowFlags() | Qt.Window)

        self.ctx = ctx
        self.df: Optional[pd.DataFrame] = df if df is not None else self._resolve_df_from_ctx()
        self.studio_dialog: Optional[PublicationPlotStudioDialog] = None

        self._cached_conf_df: Optional[pd.DataFrame] = None
        self._cached_nimbus_df: Optional[pd.DataFrame] = None
        self._cached_conc_raw: Optional[pd.DataFrame] = None
        self._cached_conc_norm: Optional[pd.DataFrame] = None
        self._cached_conc_metrics: Dict[str, Any] = {}

        self.setStyleSheet(DIALOG_DARK_THEME)

        self._build_ui()
        self.refresh_data(self.df)

    def _resolve_df_from_ctx(self) -> Optional[pd.DataFrame]:
        state = getattr(self.ctx, "state", None) if self.ctx is not None else None
        if state is None:
            return None
        for attr in ("master_df", "current_mapped_df", "features_df"):
            cand = getattr(state, attr, None)
            if cand is not None and not cand.empty:
                return cand
        return None

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(8)

        # Header Bar
        hdr = QHBoxLayout()
        hdr.setSpacing(10)

        self.lbl_overview = QLabel("Diagnostics: No cell data loaded.")
        self.lbl_overview.setStyleSheet(
            "color: #F0F2F5; font-size: 12px; font-weight: bold; background-color: transparent;"
        )
        hdr.addWidget(self.lbl_overview, 1)

        self.btn_refresh = QPushButton("Refresh Data")
        self.btn_refresh.setMinimumHeight(28)
        self.btn_refresh.clicked.connect(lambda: self.refresh_data(None))
        hdr.addWidget(self.btn_refresh)

        self.btn_pub_studio = QPushButton("Figure Studio")
        self.btn_pub_studio.setStyleSheet(BUTTON_STUDIO)
        self.btn_pub_studio.setMinimumHeight(28)
        self.btn_pub_studio.clicked.connect(self.open_publication_studio)
        hdr.addWidget(self.btn_pub_studio)

        root.addLayout(hdr)

        # Main Tabs
        self.tabs = QTabWidget()
        configure_tab_widget(self.tabs)
        self.tabs.addTab(self._build_confidence_tab(), "Confidence & Novelty")
        self.tabs.addTab(self._build_nimbus_tab(), "Nimbus Positivity Heatmap")
        self.tabs.addTab(self._build_concordance_tab(), "Method Concordance Matrix")
        root.addWidget(self.tabs, 1)

    def _make_canvas_block(
        self,
        parent_layout: QVBoxLayout,
        figsize: Tuple[float, float] = (8.5, 5.5),
    ) -> Tuple[Figure, FigureCanvas]:
        fig = Figure(figsize=figsize, dpi=100)
        fig.patch.set_facecolor("#14161d")
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if NavigationToolbar is not None:
            toolbar = NavigationToolbar(canvas, self)
            toolbar.setStyleSheet(
                "QToolBar { background-color: #1f232b; border: 1px solid #3c4352; "
                "spacing: 4px; padding: 2px; } "
                "QToolButton { background-color: transparent; border: none; padding: 3px; } "
                "QToolButton:hover { background-color: #3A6A9A; border: 1px solid #4B82B5; }"
            )
            parent_layout.addWidget(toolbar)
        parent_layout.addWidget(canvas, 1)
        return fig, canvas

    # Tab 0: Confidence & Novelty

    def _build_confidence_tab(self) -> QWidget:
        tab = QWidget()
        lyt = QVBoxLayout(tab)
        lyt.setContentsMargins(10, 8, 10, 10)
        lyt.setSpacing(8)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.addWidget(QLabel("Confidence Column:"))
        self.combo_conf_col = QComboBox()
        self.combo_conf_col.setMinimumWidth(180)
        self.combo_conf_col.currentIndexChanged.connect(self._render_confidence_tab)
        ctrl.addWidget(self.combo_conf_col)

        ctrl.addWidget(QLabel("Novelty Column:"))
        self.combo_novel_col = QComboBox()
        self.combo_novel_col.setMinimumWidth(160)
        self.combo_novel_col.currentIndexChanged.connect(self._render_confidence_tab)
        ctrl.addWidget(self.combo_novel_col)

        ctrl.addWidget(QLabel("Bins:"))
        self.spin_conf_bins = QSpinBox()
        self.spin_conf_bins.setRange(5, 100)
        self.spin_conf_bins.setValue(25)
        self.spin_conf_bins.valueChanged.connect(self._render_confidence_tab)
        ctrl.addWidget(self.spin_conf_bins)
        ctrl.addStretch(1)
        lyt.addLayout(ctrl)

        self.lbl_conf_summary = QLabel("No confidence columns found yet.")
        self.lbl_conf_summary.setWordWrap(True)
        self.lbl_conf_summary.setStyleSheet(
            "color: #4FC3F7; font-size: 12px; font-weight: bold; padding: 4px;"
        )
        lyt.addWidget(self.lbl_conf_summary)

        self.fig_conf, self.canvas_conf = self._make_canvas_block(lyt)
        return tab

    # Tab 1: Nimbus Positivity Heatmap

    def _build_nimbus_tab(self) -> QWidget:
        tab = QWidget()
        lyt = QVBoxLayout(tab)
        lyt.setContentsMargins(10, 8, 10, 10)
        lyt.setSpacing(8)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.addWidget(QLabel("Group By Phenotype:"))
        self.combo_nimbus_group = QComboBox()
        self.combo_nimbus_group.setMinimumWidth(200)
        self.combo_nimbus_group.currentIndexChanged.connect(self._render_nimbus_tab)
        ctrl.addWidget(self.combo_nimbus_group)

        self.chk_nimbus_cluster = QCheckBox("Hierarchical Cluster Rows & Markers")
        self.chk_nimbus_cluster.setChecked(True)
        self.chk_nimbus_cluster.toggled.connect(self._render_nimbus_tab)
        ctrl.addWidget(self.chk_nimbus_cluster)
        ctrl.addStretch(1)
        lyt.addLayout(ctrl)

        self.lbl_nimbus_summary = QLabel("No *_nimbus_positivity columns found yet.")
        self.lbl_nimbus_summary.setStyleSheet(
            "color: #81C784; font-size: 12px; font-weight: bold; padding: 4px;"
        )
        lyt.addWidget(self.lbl_nimbus_summary)

        self.fig_nimbus, self.canvas_nimbus = self._make_canvas_block(lyt)
        return tab

    # Tab 2: Method Concordance Matrix

    def _build_concordance_tab(self) -> QWidget:
        tab = QWidget()
        lyt = QVBoxLayout(tab)
        lyt.setContentsMargins(10, 8, 10, 10)
        lyt.setSpacing(8)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.addWidget(QLabel("Method A (Rows):"))
        self.combo_conc_a = QComboBox()
        self.combo_conc_a.setMinimumWidth(180)
        self.combo_conc_a.currentIndexChanged.connect(self._render_concordance_tab)
        ctrl.addWidget(self.combo_conc_a)

        ctrl.addWidget(QLabel("Method B (Columns):"))
        self.combo_conc_b = QComboBox()
        self.combo_conc_b.setMinimumWidth(180)
        self.combo_conc_b.currentIndexChanged.connect(self._render_concordance_tab)
        ctrl.addWidget(self.combo_conc_b)

        self.chk_conc_norm = QCheckBox("Row-Normalize (%)")
        self.chk_conc_norm.setChecked(True)
        self.chk_conc_norm.toggled.connect(self._render_concordance_tab)
        ctrl.addWidget(self.chk_conc_norm)

        self.btn_compute_conc = QPushButton("Compute Concordance")
        self.btn_compute_conc.clicked.connect(self._render_concordance_tab)
        ctrl.addWidget(self.btn_compute_conc)
        ctrl.addStretch(1)
        lyt.addLayout(ctrl)

        self.lbl_conc_metrics = QLabel("Select two categorical annotation columns to compare.")
        self.lbl_conc_metrics.setStyleSheet(
            "color: #FFD54F; font-size: 12px; font-weight: bold; padding: 4px;"
        )
        lyt.addWidget(self.lbl_conc_metrics)

        self.fig_conc, self.canvas_conc = self._make_canvas_block(lyt)
        return tab

    # Data Discovery & Refresh

    @staticmethod
    def _discover_categorical_columns(df: Optional[pd.DataFrame]) -> List[str]:
        if df is None or df.empty:
            return []
        priority = [
            "cell_type",
            "CellType",
            "Verified_CellType",
            "Cluster",
            "ASTIR_CellType",
            "MAPS_CellType",
            "STELLAR_CellType",
            "CellSighter_CellType",
            "Nimbus_CellType",
        ]
        found: List[str] = []
        for col in priority:
            if col in df.columns and col not in found:
                found.append(col)

        for col in df.columns:
            c_str = str(col)
            c_low = c_str.lower()
            if (
                c_str in found
                or c_low in _SKIP_CAT_COLS
                or c_low.endswith("_confidence")
                or c_low.endswith("_isnovel")
                or c_low.endswith("_nimbus_positivity")
                or c_low.endswith("_mean")
                or c_low.endswith("_median")
            ):
                continue
            if resolve_marker_col(df, c_str) == c_str and pd.api.types.is_float_dtype(df[col]):
                continue
            series = df[col]
            if (
                series.dtype == object
                or series.dtype.name in ("category", "string", "bool")
                or (pd.api.types.is_integer_dtype(series) and series.nunique() <= 60)
            ):
                found.append(c_str)
        return found

    @staticmethod
    def _discover_confidence_columns(df: Optional[pd.DataFrame]) -> List[str]:
        if df is None or df.empty:
            return []
        cols: List[str] = []
        for col in df.columns:
            low = str(col).lower()
            if (
                low.endswith("_confidence") or low == "confidence"
            ) and pd.api.types.is_numeric_dtype(df[col]):
                cols.append(str(col))
        return cols

    @staticmethod
    def _discover_novelty_columns(df: Optional[pd.DataFrame]) -> List[str]:
        if df is None or df.empty:
            return []
        cols: List[str] = []
        for col in df.columns:
            low = str(col).lower()
            if low.endswith("_isnovel") or low == "is_novel":
                cols.append(str(col))
        return cols

    @staticmethod
    def _discover_nimbus_columns(df: Optional[pd.DataFrame]) -> List[str]:
        if df is None or df.empty:
            return []
        return [
            str(c)
            for c in df.columns
            if str(c).endswith("_nimbus_positivity") and pd.api.types.is_numeric_dtype(df[c])
        ]

    @staticmethod
    def _populate_combo(combo: QComboBox, items: List[str], preferred: Optional[str] = None) -> None:
        curr = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        if preferred and preferred in items:
            combo.setCurrentText(preferred)
        elif curr in items:
            combo.setCurrentText(curr)
        elif items:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def refresh_data(self, df: Optional[pd.DataFrame] = None) -> None:
        """Refresh column selectors and re-render all diagnostic tabs."""
        if df is not None:
            self.df = df
        elif self.ctx is not None:
            resolved = self._resolve_df_from_ctx()
            if resolved is not None:
                self.df = resolved

        cur_df = self.df
        conf_cols = self._discover_confidence_columns(cur_df)
        novel_cols = self._discover_novelty_columns(cur_df)
        nimbus_cols = self._discover_nimbus_columns(cur_df)
        cat_cols = self._discover_categorical_columns(cur_df)

        self._populate_combo(self.combo_conf_col, conf_cols)
        self._populate_combo(
            self.combo_novel_col,
            (["None"] + novel_cols) if novel_cols else ["None"],
            preferred=novel_cols[0] if novel_cols else "None",
        )
        nimbus_pref = "Nimbus_CellType" if "Nimbus_CellType" in cat_cols else None
        self._populate_combo(self.combo_nimbus_group, cat_cols, preferred=nimbus_pref)
        self._populate_combo(self.combo_conc_a, cat_cols)
        second_cat = cat_cols[1] if len(cat_cols) > 1 else (cat_cols[0] if cat_cols else None)
        self._populate_combo(self.combo_conc_b, cat_cols, preferred=second_cat)

        n_cells = len(cur_df) if cur_df is not None else 0
        self.lbl_overview.setText(
            f"Cells: {n_cells:,} | Annotations: {len(cat_cols)} | "
            f"Confidence cols: {len(conf_cols)} | Novelty cols: {len(novel_cols)} | "
            f"Nimbus markers: {len(nimbus_cols)}"
        )

        self._render_confidence_tab()
        self._render_nimbus_tab()
        self._render_concordance_tab()

    # Empty Canvas Helper

    @staticmethod
    def _draw_empty(fig: Figure, canvas: FigureCanvas, msg: str) -> None:
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor("#14161d")
        ax.text(0.5, 0.5, msg, color="#999", ha="center", va="center", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#333")
        canvas.draw_idle()

    # Rendering: Tab 0 Confidence & Novelty

    def _draw_confidence_figure(
        self,
        fig: Figure,
        vals: np.ndarray,
        novel_mask: Optional[np.ndarray],
        conf_col: str,
        bins: int,
        *,
        is_dark: bool = True,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ) -> None:
        bg = "#14161d" if is_dark else "#FFFFFF"
        fg = "#E0E0E0" if is_dark else "#111111"
        fig.patch.set_facecolor(bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)

        bin_edges = np.linspace(0.0, 1.0, int(bins) + 1)
        if novel_mask is not None and np.any(novel_mask):
            known_vals = vals[~novel_mask]
            novel_vals = vals[novel_mask]
            if known_vals.size > 0:
                ax.hist(
                    known_vals,
                    bins=bin_edges,
                    alpha=0.75,
                    color="#42A5F5",
                    edgecolor="#14161d" if is_dark else "#FFFFFF",
                    label=f"Known ({known_vals.size:,})",
                )
            ax.hist(
                novel_vals,
                bins=bin_edges,
                alpha=0.80,
                color="#FF7043",
                edgecolor="#14161d" if is_dark else "#FFFFFF",
                label=f"Novel ({novel_vals.size:,})",
            )
            ax.legend(frameon=False, labelcolor=fg)
        else:
            ax.hist(
                vals,
                bins=bin_edges,
                alpha=0.85,
                color="#42A5F5",
                edgecolor="#14161d" if is_dark else "#FFFFFF",
            )

        ax.axvline(0.50, color="#FFD54F", linestyle="--", linewidth=1.4, label="Threshold 0.50")
        ax.set_xlim(0.0, 1.0)
        ax.set_title(title or f"Confidence Distribution: {conf_col}", color=fg, fontweight="bold")
        ax.set_xlabel(xlabel or f"{conf_col} Score", color=fg)
        ax.set_ylabel(ylabel or "Cell Count", color=fg)
        ax.tick_params(colors=fg)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#555" if is_dark else "#222")

    def _render_confidence_tab(self) -> None:
        conf_col = self.combo_conf_col.currentText().strip()
        if self.df is None or self.df.empty or not conf_col or conf_col not in self.df.columns:
            self._cached_conf_df = None
            self.lbl_conf_summary.setText("No confidence scores available. Run a phenotyper first.")
            self._draw_empty(
                self.fig_conf,
                self.canvas_conf,
                "No <Method>_Confidence column available.",
            )
            return

        vals_series = pd.to_numeric(self.df[conf_col], errors="coerce")
        valid_mask = vals_series.notna().to_numpy()
        vals = vals_series.to_numpy(dtype=float)[valid_mask]
        if vals.size == 0:
            self._cached_conf_df = None
            self._draw_empty(self.fig_conf, self.canvas_conf, f"{conf_col} contains no finite values.")
            return

        novel_col = self.combo_novel_col.currentText().strip()
        novel_mask: Optional[np.ndarray] = None
        if novel_col and novel_col != "None" and novel_col in self.df.columns:
            novel_mask = self.df[novel_col].fillna(False).astype(bool).to_numpy()[valid_mask]
        else:
            # Also inspect base label column for Novel_* prefix
            base_col = conf_col[: -len("_Confidence")] if conf_col.endswith("_Confidence") else ""
            if base_col and base_col in self.df.columns:
                novel_flags = self.df[base_col].astype(str).str.startswith("Novel_").to_numpy()
                if np.any(novel_flags):
                    novel_mask = novel_flags[valid_mask]

        n_low = int(np.sum(vals < 0.50))
        pct_low = 100.0 * n_low / max(1, vals.size)
        novel_txt = "N/A"
        if novel_mask is not None:
            n_nov = int(np.sum(novel_mask))
            novel_txt = f"{n_nov:,} / {vals.size:,} ({100.0 * n_nov / max(1, vals.size):.1f}%)"

        self.lbl_conf_summary.setText(
            f"Mean: {float(np.mean(vals)):.3f} | Median: {float(np.median(vals)):.3f} | "
            f"Low Conf (<0.50): {n_low:,} ({pct_low:.1f}%) | STELLAR Novel Cells: {novel_txt}"
        )

        # Cache binned summary for Figure Studio / Source Data
        bins = self.spin_conf_bins.value()
        counts, edges = np.histogram(vals, bins=np.linspace(0.0, 1.0, bins + 1))
        summary_rows = {
            "bin_start": edges[:-1],
            "bin_end": edges[1:],
            "cell_count": counts,
        }
        if novel_mask is not None:
            nov_counts, _ = np.histogram(vals[novel_mask], bins=edges)
            summary_rows["novel_count"] = nov_counts
        self._cached_conf_df = pd.DataFrame(summary_rows)

        self.fig_conf.clear()
        self._draw_confidence_figure(self.fig_conf, vals, novel_mask, conf_col, bins, is_dark=True)
        self.fig_conf.tight_layout()
        self.canvas_conf.draw_idle()

    # Rendering: Tab 1 Nimbus Positivity Heatmap

    def _compute_nimbus_matrix(self) -> Optional[pd.DataFrame]:
        pos_cols = self._discover_nimbus_columns(self.df)
        group_col = self.combo_nimbus_group.currentText().strip()
        if (
            self.df is None
            or self.df.empty
            or not pos_cols
            or not group_col
            or group_col not in self.df.columns
        ):
            return None

        mat = self.df.groupby(group_col, observed=True)[pos_cols].mean().fillna(0.0)
        mat.index = mat.index.astype(str)
        mat.columns = [c[: -len("_nimbus_positivity")] for c in pos_cols]

        if self.chk_nimbus_cluster.isChecked() and mat.shape[0] >= 2 and mat.shape[1] >= 2:
            try:
                from scipy.cluster import hierarchy

                r_link = hierarchy.linkage(mat.to_numpy(dtype=float), method="average")
                c_link = hierarchy.linkage(mat.to_numpy(dtype=float).T, method="average")
                r_order = hierarchy.dendrogram(r_link, no_plot=True)["leaves"]
                c_order = hierarchy.dendrogram(c_link, no_plot=True)["leaves"]
                mat = mat.iloc[r_order, c_order]
            except Exception:
                pass
        return mat

    @staticmethod
    def _draw_heatmap_matrix(
        fig: Figure,
        mat: pd.DataFrame,
        *,
        cmap_name: str = "YlGnBu",
        vmin: Optional[float] = 0.0,
        vmax: Optional[float] = 1.0,
        cbar_label: str = "Value",
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        fmt: str = ".2f",
        annot_matrix: Optional[np.ndarray] = None,
        is_dark: bool = True,
        tick_rotation: int = 40,
    ) -> None:
        bg = "#14161d" if is_dark else "#FFFFFF"
        fg = "#E0E0E0" if is_dark else "#111111"
        fig.patch.set_facecolor(bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)

        vals = mat.to_numpy(dtype=float)
        cmap = plt.get_cmap(cmap_name)
        im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

        n_rows, n_cols = vals.shape
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels(
            list(mat.columns), rotation=tick_rotation, ha="right", color=fg, fontsize=9
        )
        ax.set_yticklabels(list(mat.index), color=fg, fontsize=9)

        if n_rows <= 25 and n_cols <= 25:
            lo = float(np.nanmin(vals)) if vmin is None else float(vmin)
            hi = float(np.nanmax(vals)) if vmax is None else float(vmax)
            denom = max(1e-8, hi - lo)
            for r in range(n_rows):
                for c in range(n_cols):
                    v = vals[r, c]
                    txt = str(annot_matrix[r, c]) if annot_matrix is not None else format(v, fmt)
                    rgba = cmap((v - lo) / denom)
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    tc = "black" if lum > 0.55 else "white"
                    ax.text(c, r, txt, ha="center", va="center", color=tc, fontsize=8, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, color=fg)
        cbar.ax.yaxis.set_tick_params(color=fg, labelcolor=fg)

        ax.set_title(title, color=fg, fontweight="bold")
        ax.set_xlabel(xlabel, color=fg)
        ax.set_ylabel(ylabel, color=fg)
        for sp in ax.spines.values():
            sp.set_color("#444" if is_dark else "#222")

    def _render_nimbus_tab(self) -> None:
        mat = self._compute_nimbus_matrix()
        self._cached_nimbus_df = mat
        if mat is None or mat.empty:
            self.lbl_nimbus_summary.setText("No Nimbus *_nimbus_positivity columns available yet.")
            self._draw_empty(
                self.fig_nimbus,
                self.canvas_nimbus,
                "Run Nimbus Phenotyper to populate per-marker positivity heatmaps.",
            )
            return

        group_col = self.combo_nimbus_group.currentText().strip()
        self.lbl_nimbus_summary.setText(
            f"Showing mean Nimbus positivity across {mat.shape[0]} phenotypes ({group_col}) "
            f"and {mat.shape[1]} markers."
        )
        self.fig_nimbus.clear()
        self._draw_heatmap_matrix(
            self.fig_nimbus,
            mat,
            cmap_name="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            cbar_label="Mean Nimbus Positivity",
            title=f"Nimbus Marker Positivity by {group_col}",
            xlabel="Marker",
            ylabel=group_col,
            is_dark=True,
        )
        self.fig_nimbus.tight_layout()
        self.canvas_nimbus.draw_idle()

    # Rendering: Tab 2 Method Concordance Matrix

    def compute_concordance(self, col_a: str, col_b: str) -> Dict[str, Any]:
        """Compute confusion matrix, Accuracy, Agreement Rate, and Cohen's Kappa."""
        if self.df is None or self.df.empty:
            raise ValueError("No DataFrame loaded.")
        if col_a not in self.df.columns or col_b not in self.df.columns:
            raise ValueError(f"Columns ({col_a!r}, {col_b!r}) not found in DataFrame.")

        sub = self.df[[col_a, col_b]].dropna()
        if sub.empty:
            raise ValueError("No non-null rows shared between selected columns.")

        y_a = sub[col_a].astype(str).to_numpy()
        y_b = sub[col_b].astype(str).to_numpy()
        labels = sorted(set(y_a.tolist()) | set(y_b.tolist()))

        acc = float(accuracy_score(y_a, y_b))
        if len(labels) <= 1:
            kappa = 1.0 if acc == 1.0 else 0.0
        else:
            raw_k = float(cohen_kappa_score(y_a, y_b, labels=labels))
            kappa = 0.0 if np.isnan(raw_k) else raw_k

        cm_raw = confusion_matrix(y_a, y_b, labels=labels)
        cm_raw_df = pd.DataFrame(cm_raw, index=labels, columns=labels)
        row_sums = cm_raw.sum(axis=1, keepdims=True).astype(float)
        cm_norm = np.divide(
            cm_raw.astype(float),
            row_sums,
            out=np.zeros_like(cm_raw, dtype=float),
            where=row_sums > 0.0,
        )
        cm_norm_df = pd.DataFrame(cm_norm * 100.0, index=labels, columns=labels)

        return {
            "n_cells": int(len(sub)),
            "accuracy": acc,
            "agreement_rate": acc,
            "cohens_kappa": kappa,
            "labels": labels,
            "confusion_raw": cm_raw_df,
            "confusion_norm": cm_norm_df,
        }

    def _render_concordance_tab(self) -> None:
        col_a = self.combo_conc_a.currentText().strip()
        col_b = self.combo_conc_b.currentText().strip()
        if (
            self.df is None
            or self.df.empty
            or not col_a
            or not col_b
            or col_a not in self.df.columns
            or col_b not in self.df.columns
        ):
            self._cached_conc_raw = None
            self._cached_conc_norm = None
            self._cached_conc_metrics = {}
            self.lbl_conc_metrics.setText("Need at least one categorical annotation column.")
            self._draw_empty(
                self.fig_conc,
                self.canvas_conc,
                "Run phenotyping methods to compare method concordance.",
            )
            return

        try:
            res = self.compute_concordance(col_a, col_b)
        except Exception as exc:
            self._cached_conc_raw = None
            self._cached_conc_norm = None
            self._cached_conc_metrics = {}
            self.lbl_conc_metrics.setText(f"Concordance error: {exc}")
            self._draw_empty(self.fig_conc, self.canvas_conc, str(exc))
            return

        self._cached_conc_raw = res["confusion_raw"]
        self._cached_conc_norm = res["confusion_norm"]
        self._cached_conc_metrics = res

        self.lbl_conc_metrics.setText(
            f"N={res['n_cells']:,} | Accuracy: {res['accuracy']:.3f} | "
            f"Agreement Rate: {res['agreement_rate']:.1%} | "
            f"Cohen's Kappa: {res['cohens_kappa']:.3f}"
        )

        use_norm = self.chk_conc_norm.isChecked()
        plot_df = res["confusion_norm"] if use_norm else res["confusion_raw"].astype(float)
        raw_arr = res["confusion_raw"].to_numpy()
        norm_arr = res["confusion_norm"].to_numpy()

        annot = np.empty(raw_arr.shape, dtype=object)
        for r in range(raw_arr.shape[0]):
            for c in range(raw_arr.shape[1]):
                annot[r, c] = (
                    f"{raw_arr[r, c]}\n({norm_arr[r, c]:.0f}%)"
                    if use_norm
                    else f"{raw_arr[r, c]}"
                )

        self.fig_conc.clear()
        self._draw_heatmap_matrix(
            self.fig_conc,
            plot_df,
            cmap_name="viridis",
            vmin=0.0,
            vmax=100.0 if use_norm else None,
            cbar_label="Row Agreement (%)" if use_norm else "Cell Count",
            title=(
                f"{col_a} vs {col_b}  "
                f"(Acc={res['accuracy']:.3f}, Kappa={res['cohens_kappa']:.3f})"
            ),
            xlabel=col_b,
            ylabel=col_a,
            annot_matrix=annot,
            is_dark=True,
        )
        self.fig_conc.tight_layout()
        self.canvas_conc.draw_idle()

    # Figure Studio Integration

    @staticmethod
    def _resolve_theme_preset(theme_name: str) -> Dict[str, Any]:
        aliases = {
            "Publication White (Nature)": "Classic White",
            "Cell Press Boxed": "Boxed White",
        }
        canonical = aliases.get(theme_name, theme_name)
        return THEME_PRESETS.get(canonical, THEME_PRESETS["IRIS Dark"])

    def open_publication_studio(self) -> Optional[PublicationPlotStudioDialog]:
        """Launch Figure Studio for the active diagnostic tab."""
        tab_idx = self.tabs.currentIndex()
        plot_fn: Optional[Callable[[Figure, PlotStyleConfig], Any]] = None
        summary_df: Optional[pd.DataFrame] = None
        stats_df: Optional[pd.DataFrame] = None
        categories: Optional[List[str]] = None
        title = "Phenotyping Diagnostics — Figure Studio"
        cfg = PlotStyleConfig(theme="IRIS Dark")

        if tab_idx == 0:
            if self._cached_conf_df is None:
                QMessageBox.information(
                    self, "No Data", "No confidence distribution is available to export."
                )
                return None
            conf_col = self.combo_conf_col.currentText().strip()
            novel_col = self.combo_novel_col.currentText().strip()
            bins = self.spin_conf_bins.value()
            vals_series = pd.to_numeric(self.df[conf_col], errors="coerce")
            valid_mask = vals_series.notna().to_numpy()
            vals = vals_series.to_numpy(dtype=float)[valid_mask]
            novel_mask: Optional[np.ndarray] = None
            if novel_col and novel_col != "None" and novel_col in self.df.columns:
                novel_mask = self.df[novel_col].fillna(False).astype(bool).to_numpy()[valid_mask]
            summary_df = self._cached_conf_df.copy()
            categories = ["Known", "Novel"] if novel_mask is not None else [conf_col]
            default_title = f"Confidence Distribution: {conf_col}"

            def _plot_conf(fig: Figure, style_cfg: PlotStyleConfig) -> None:
                preset = self._resolve_theme_preset(style_cfg.theme)
                self._draw_confidence_figure(
                    fig,
                    vals,
                    novel_mask,
                    conf_col,
                    bins,
                    is_dark=bool(preset.get("is_dark", True)),
                    title=style_cfg.title or default_title,
                    xlabel=style_cfg.xlabel or f"{conf_col} Score",
                    ylabel=style_cfg.ylabel or "Cell Count",
                )

            plot_fn = _plot_conf
            title = f"{conf_col} — Figure Studio"

        elif tab_idx == 1:
            if self._cached_nimbus_df is None or self._cached_nimbus_df.empty:
                QMessageBox.information(
                    self, "No Data", "No Nimbus positivity matrix is available to export."
                )
                return None
            mat = self._cached_nimbus_df.copy()
            group_col = self.combo_nimbus_group.currentText().strip()
            summary_df = mat
            categories = list(mat.index)
            cfg.grid_mode = "None"
            cfg.tick_rotation = 40
            cfg.palette_name = "viridis"
            default_title = f"Nimbus Marker Positivity by {group_col}"

            def _plot_nimbus(fig: Figure, style_cfg: PlotStyleConfig) -> None:
                preset = self._resolve_theme_preset(style_cfg.theme)
                cmap_map = {
                    "viridis": "viridis",
                    "Viridis": "viridis",
                    "RdBu_r": "RdBu_r",
                    "coolwarm": "coolwarm",
                }
                cmap_choice = cmap_map.get(style_cfg.palette_name, "YlGnBu")
                self._draw_heatmap_matrix(
                    fig,
                    mat,
                    cmap_name=cmap_choice,
                    vmin=0.0,
                    vmax=1.0,
                    cbar_label="Mean Nimbus Positivity",
                    title=style_cfg.title or default_title,
                    xlabel=style_cfg.xlabel or "Marker",
                    ylabel=style_cfg.ylabel or group_col,
                    is_dark=bool(preset.get("is_dark", True)),
                    tick_rotation=int(style_cfg.tick_rotation),
                )

            plot_fn = _plot_nimbus
            title = "Nimbus Positivity Heatmap — Figure Studio"

        else:
            if self._cached_conc_raw is None or not self._cached_conc_metrics:
                QMessageBox.information(
                    self, "No Data", "No method concordance matrix is available to export."
                )
                return None
            res = dict(self._cached_conc_metrics)
            col_a = self.combo_conc_a.currentText().strip()
            col_b = self.combo_conc_b.currentText().strip()
            use_norm = self.chk_conc_norm.isChecked()
            plot_df = (
                res["confusion_norm"].copy() if use_norm else res["confusion_raw"].astype(float)
            )
            raw_arr = res["confusion_raw"].to_numpy()
            norm_arr = res["confusion_norm"].to_numpy()
            annot = np.empty(raw_arr.shape, dtype=object)
            for r in range(raw_arr.shape[0]):
                for c in range(raw_arr.shape[1]):
                    annot[r, c] = (
                        f"{raw_arr[r, c]}\n({norm_arr[r, c]:.0f}%)"
                        if use_norm
                        else f"{raw_arr[r, c]}"
                    )
            summary_df = res["confusion_raw"].copy()
            stats_df = pd.DataFrame(
                [
                    {
                        "Method_A": col_a,
                        "Method_B": col_b,
                        "N_Cells": res["n_cells"],
                        "Accuracy": res["accuracy"],
                        "Agreement_Rate": res["agreement_rate"],
                        "Cohens_Kappa": res["cohens_kappa"],
                    }
                ]
            )
            categories = list(res["labels"])
            cfg.grid_mode = "None"
            cfg.tick_rotation = 35
            cfg.palette_name = "viridis"
            default_title = (
                f"{col_a} vs {col_b} "
                f"(Acc={res['accuracy']:.3f}, Kappa={res['cohens_kappa']:.3f})"
            )

            def _plot_conc(fig: Figure, style_cfg: PlotStyleConfig) -> None:
                preset = self._resolve_theme_preset(style_cfg.theme)
                cmap_map = {
                    "viridis": "viridis",
                    "Viridis": "viridis",
                    "RdBu_r": "RdBu_r",
                    "coolwarm": "coolwarm",
                }
                cmap_choice = cmap_map.get(style_cfg.palette_name, "viridis")
                self._draw_heatmap_matrix(
                    fig,
                    plot_df,
                    cmap_name=cmap_choice,
                    vmin=0.0,
                    vmax=100.0 if use_norm else None,
                    cbar_label="Row Agreement (%)" if use_norm else "Cell Count",
                    title=style_cfg.title or default_title,
                    xlabel=style_cfg.xlabel or col_b,
                    ylabel=style_cfg.ylabel or col_a,
                    annot_matrix=annot,
                    is_dark=bool(preset.get("is_dark", True)),
                    tick_rotation=int(style_cfg.tick_rotation),
                )

            plot_fn = _plot_conc
            title = f"{col_a} vs {col_b} — Figure Studio"

        if self.studio_dialog is not None:
            self.studio_dialog.close()

        self.studio_dialog = PublicationPlotStudioDialog(
            plot_fn=plot_fn,
            summary_df=summary_df,
            stats_df=stats_df,
            categories=categories,
            config=cfg,
            title=title,
            parent=self,
        )
        iris_log("IRIS-Phenotyper", "Diagnostics", f"Opened Figure Studio: {title}.")
        self.studio_dialog.show()
        self.studio_dialog.raise_()
        self.studio_dialog.activateWindow()
        return self.studio_dialog

    open_figure_studio = open_publication_studio

    def closeEvent(self, event: Any) -> None:
        if self.studio_dialog is not None:
            try:
                self.studio_dialog.close()
            except Exception:
                pass
        super().closeEvent(event)
