"""
IRIS Cell Proportion & Ecological Diversity Dock Widget & Figure Studio Dialog
===========================================================================
Interactive compositional cell-type proportion quantification (All Cells,
Parent Lineage Subset, and ROI Compartments), cohort group comparisons,
centered log-ratio (CLR) dot-plots, Shannon/Simpson diversity analysis, and
figure/workbook export.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.axes
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from iris_analyzer.density import ROISpec
from iris_analyzer.proportion import ProportionResult, compute_cell_proportions
from iris_analyzer.widgets.density_widget import (
    STAT_TEST_CHOICES,
    SampleGroupAssignmentDialog,
    TabbedPublicationStudioDialog,
    extract_viewer_roi_polygons,
    filter_rois_for_samples,
    normalize_sample_key,
    populate_table_widget,
    resolve_scoped_samples,
    stat_test_code_from_label,
)
from iris_common.logging import iris_log
from iris_common.schema import COL_SAMPLE, ensure_canonical_dataframe, find_sample_column
from iris_common.widgets.publication_studio import (
    THEME_PRESETS,
    PlotStyleConfig,
    apply_plot_style,
    draw_significance_brackets,
    resolve_category_colors,
)
from iris_common.widgets.sample_scope import SCOPE_ACTIVE, SCOPE_ALL, SCOPE_SELECTED
from iris_common.widgets.theme import (
    BUTTON_CTA,
    POPOUT_BUTTON_STYLE,
    SHARED_DARK_THEME,
    create_section_card,
    style_section_header,
)

DENOMINATOR_MODE_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("All Cells", "total"),
    ("Parent Lineage Subset", "parent_lineage"),
    ("ROI Compartments", "roi_compartment"),
)

PREVIEW_PLOT_MODES: Tuple[str, ...] = (
    "Stacked Composition %",
    "Grouped Box + Replicates",
    "Ecological Diversity",
)


def denominator_mode_code_from_label(label: str) -> str:
    """Translate UI denominator mode label into engine mode key."""
    return dict(DENOMINATOR_MODE_CHOICES).get(label, "total")


def _draw_box_with_replicates(
    ax: matplotlib.axes.Axes,
    grouped_values: List[Tuple[str, np.ndarray]],
    colors: Dict[str, str],
    config: PlotStyleConfig,
    *,
    ylabel: str,
    title: str,
    stats_df: Optional[pd.DataFrame] = None,
    ylim_zero: bool = True,
    seed: int = 21,
) -> None:
    """Draw boxplots with overlaid biological replicate scatter points and stat brackets."""
    if not grouped_values or all(len(v) == 0 for _, v in grouped_values):
        ax.text(0.5, 0.5, "No data available.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    preset = THEME_PRESETS.get(config.theme, THEME_PRESETS["IRIS Dark"])
    edge_col = "#FFFFFF" if preset["is_dark"] else "#111111"
    rng = np.random.default_rng(seed)

    positions = list(range(len(grouped_values)))
    arrays = [vals if len(vals) > 0 else np.array([0.0]) for _, vals in grouped_values]
    bp = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": edge_col, "linewidth": 1.4},
        whiskerprops={"color": edge_col, "linewidth": 1.0},
        capprops={"color": edge_col, "linewidth": 1.0},
        boxprops={"linewidth": 1.0},
    )

    x_positions: Dict[str, float] = {}
    y_maxes: Dict[str, float] = {}
    for idx, ((grp, vals), box_patch) in enumerate(zip(grouped_values, bp["boxes"])):
        col = colors.get(grp, "#3A6A9A")
        box_patch.set_facecolor(col)
        box_patch.set_alpha(0.55)
        box_patch.set_edgecolor(edge_col)
        x_positions[grp] = float(idx)
        if len(vals) > 0:
            jitter = rng.uniform(-0.13, 0.13, size=len(vals)) if len(vals) > 1 else np.zeros(1)
            ax.scatter(
                np.full(len(vals), float(idx)) + jitter,
                vals,
                color=col,
                edgecolor=edge_col,
                linewidth=0.85,
                s=34,
                zorder=4,
                label=grp,
            )
            y_maxes[grp] = float(np.max(vals))
        else:
            y_maxes[grp] = 0.0

    ax.set_xticks(positions)
    ax.set_xticklabels([grp for grp, _ in grouped_values])
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    max_y = max(y_maxes.values()) if y_maxes else 1.0
    min_y = min(float(np.min(v)) for _, v in grouped_values if len(v) > 0)
    low = 0.0 if ylim_zero else min(0.0, min_y * 1.1)
    ax.set_ylim(low, max(max_y * 1.18, 1e-6))

    if stats_df is not None and not stats_df.empty:
        valid_pairs = stats_df[
            stats_df["group1"].astype(str).isin(x_positions)
            & stats_df["group2"].astype(str).isin(x_positions)
        ]
        if not valid_pairs.empty:
            draw_significance_brackets(ax, valid_pairs, x_positions, y_maxes, config=config)


def _draw_stacked_composition(
    ax: matplotlib.axes.Axes,
    result: ProportionResult,
    config: PlotStyleConfig,
    *,
    by_group: bool = False,
) -> None:
    """Render 100% stacked compositional bar chart per sample or per cohort group."""
    rep = result.replicates_df
    if rep.empty or not result.cell_types:
        ax.text(0.5, 0.5, "No compositional data available.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    if by_group:
        pivot = (
            rep.groupby(["Group", "cell_type"], observed=True)["percentage"]
            .mean()
            .unstack(fill_value=0.0)
        )
        # Re-normalize mean percentages so stacked bars sum cleanly to 100%
        row_sums = pivot.sum(axis=1).replace(0.0, 1.0)
        pivot = pivot.div(row_sums, axis=0) * 100.0
        x_labels = [str(idx) for idx in pivot.index]
        title = "Mean Cell-Type Composition by Cohort"
    else:
        has_multi_rois = rep["ROI"].nunique() > 1
        unit_col = "_UnitLabel"
        work = rep.copy()
        work[unit_col] = (
            work[COL_SAMPLE].astype(str) + " | " + work["ROI"].astype(str)
            if has_multi_rois
            else work[COL_SAMPLE].astype(str)
        )
        order_df = work[[unit_col, "Group"]].drop_duplicates()
        ordered_units = order_df.sort_values(["Group", unit_col])[unit_col].tolist()
        pivot = (
            work.pivot_table(
                index=unit_col,
                columns="cell_type",
                values="percentage",
                aggfunc="mean",
                fill_value=0.0,
            )
            .reindex(ordered_units)
        )
        x_labels = ordered_units
        title = "Cell-Type Composition per Sample"

    colors = resolve_category_colors(result.cell_types, config)
    preset = THEME_PRESETS.get(config.theme, THEME_PRESETS["IRIS Dark"])
    edge_col = "#1F232B" if preset["is_dark"] else "#FFFFFF"

    x = np.arange(len(x_labels), dtype=float)
    bottom = np.zeros(len(x_labels), dtype=float)
    for ct in result.cell_types:
        vals = (
            pivot[ct].to_numpy(dtype=float)
            if ct in pivot.columns
            else np.zeros(len(x_labels), dtype=float)
        )
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.72,
            color=colors[ct],
            edgecolor=edge_col,
            linewidth=0.5,
            label=ct,
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(
        x_labels,
        rotation=35 if len(x_labels) > 4 else 0,
        ha="right" if len(x_labels) > 4 else "center",
    )
    ax.set_ylim(0.0, 105.0)
    ax.set_ylabel("Proportion (% of Denominator)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True)


class CellProportionStudioDialog(TabbedPublicationStudioDialog):
    """Multi-tab Figure Studio for proportions, CLR dot-plots, and diversity."""

    def __init__(
        self,
        result: ProportionResult,
        *,
        config: Optional[PlotStyleConfig] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            "Cell Proportions \u2014 Figure Studio",
            categories=result.cell_types,
            config=config,
            log_submodule="Cell Proportion",
            parent=parent,
        )
        self.result: ProportionResult = result

        # Tab 0: Stacked Proportions
        tab0_header = QWidget()
        h0_layout = QHBoxLayout(tab0_header)
        h0_layout.setContentsMargins(0, 0, 0, 0)
        h0_layout.addWidget(QLabel("Grouping:"))
        self.combo_stack_mode = QComboBox()
        self.combo_stack_mode.addItems(["Per Sample", "Per Cohort Group"])
        self.combo_stack_mode.currentTextChanged.connect(lambda _: self._render_tab_index(0))
        h0_layout.addWidget(self.combo_stack_mode)
        h0_layout.addStretch(1)
        self.add_plot_tab(
            "Stacked Proportions",
            self._plot_stacked_proportions,
            header_widget=tab0_header,
        )

        # Tab 1: Grouped Comparison + Significance
        tab1_header = QWidget()
        h1_layout = QHBoxLayout(tab1_header)
        h1_layout.setContentsMargins(0, 0, 0, 0)
        h1_layout.addWidget(QLabel("Cell Type:"))
        self.combo_box_focus = QComboBox()
        self.combo_box_focus.currentTextChanged.connect(lambda _: self._render_tab_index(1))
        h1_layout.addWidget(self.combo_box_focus)
        h1_layout.addStretch(1)
        self.add_plot_tab(
            "Grouped Comparison + Significance",
            self._plot_grouped_comparison,
            header_widget=tab1_header,
        )

        # Tab 2: Compositional Dot-Plot / CLR
        self.add_plot_tab(
            "Compositional Dot-Plot / CLR",
            self._plot_clr_dotplot,
        )

        # Tab 3: Shannon & Simpson Diversity
        self.add_plot_tab(
            "Shannon & Simpson Diversity",
            self._plot_diversity,
        )

        # Tab 4: Statistics Table
        tab4_widget = QWidget()
        t4_layout = QVBoxLayout(tab4_widget)
        t4_layout.setContentsMargins(6, 6, 6, 6)
        t4_hdr = QHBoxLayout()
        t4_hdr.addWidget(QLabel("View Table:"))
        self.combo_stat_view = QComboBox()
        self.combo_stat_view.addItems(
            [
                "Cell-Type Proportion Statistics",
                "Sample Diversity Metrics",
                "Diversity Group Statistics",
                "CLR Matrix",
            ]
        )
        self.combo_stat_view.currentTextChanged.connect(self._refresh_stats_table_view)
        t4_hdr.addWidget(self.combo_stat_view)
        t4_hdr.addStretch(1)
        t4_layout.addLayout(t4_hdr)

        self.table_stats = QTableWidget(0, 0)
        self.table_stats.setAlternatingRowColors(True)
        self.table_stats.setEditTriggers(QTableWidget.NoEditTriggers)
        t4_layout.addWidget(self.table_stats, 1)
        self.add_table_tab("Statistics Table", tab4_widget)

        self.update_data(result)

    def update_data(self, result: ProportionResult) -> None:
        """Refresh all proportion tabs, tables, and Source Data workbook payload."""
        self.result = result

        prev_focus = self.combo_box_focus.currentText()
        self.combo_box_focus.blockSignals(True)
        self.combo_box_focus.clear()
        items = ["All Cell Types"] + list(result.cell_types)
        self.combo_box_focus.addItems(items)
        if prev_focus in items:
            self.combo_box_focus.setCurrentText(prev_focus)
        self.combo_box_focus.blockSignals(False)

        self._refresh_stats_table_view()

        self.set_export_payload(
            summary_df=result.summary_df,
            replicates_df=result.replicates_df,
            stats_df=result.stats_df,
            metadata=result.metadata,
            extra_sheets={
                "Diversity_Metrics": result.diversity_df,
                "CLR_Matrix": result.clr_df,
            },
            categories=result.cell_types,
        )
        self.render_all_tabs()

    def _refresh_stats_table_view(self) -> None:
        view = self.combo_stat_view.currentText()
        if view == "Sample Diversity Metrics":
            populate_table_widget(
                self.table_stats,
                self.result.diversity_df,
                empty_note="No diversity metrics computed.",
            )
        elif view == "Diversity Group Statistics":
            div_stats = self.result.metadata.get("diversity_stats_df")
            populate_table_widget(
                self.table_stats,
                div_stats if isinstance(div_stats, pd.DataFrame) else pd.DataFrame(),
                empty_note="Assign >= 2 sample groups to compute diversity statistics.",
            )
        elif view == "CLR Matrix":
            populate_table_widget(
                self.table_stats,
                self.result.clr_df,
                empty_note="No CLR matrix available.",
            )
        else:
            populate_table_widget(
                self.table_stats,
                self.result.stats_df,
                empty_note="Assign >= 2 sample groups to compute proportion statistics.",
            )

    def _plot_stacked_proportions(
        self,
        fig: Figure,
        cfg: PlotStyleConfig,
    ) -> List[matplotlib.axes.Axes]:
        ax = fig.add_subplot(111)
        by_group = self.combo_stack_mode.currentText() == "Per Cohort Group"
        _draw_stacked_composition(ax, self.result, cfg, by_group=by_group)
        return [ax]

    def _plot_grouped_comparison(
        self,
        fig: Figure,
        cfg: PlotStyleConfig,
    ) -> List[matplotlib.axes.Axes]:
        ax = fig.add_subplot(111)
        rep = self.result.replicates_df
        if rep.empty or not self.result.cell_types:
            ax.text(0.5, 0.5, "No proportion data available.", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            return [ax]

        focus = self.combo_box_focus.currentText() or "All Cell Types"
        sample_df = (
            rep.groupby(["Group", COL_SAMPLE, "cell_type"], observed=True, as_index=False)[
                "percentage"
            ]
            .mean()
        )
        groups = list(dict.fromkeys(sample_df["Group"].astype(str).tolist()))
        group_colors = resolve_category_colors(groups, cfg)

        if focus != "All Cell Types":
            sub = sample_df[sample_df["cell_type"] == focus]
            grouped_vals = [
                (
                    grp,
                    sub.loc[sub["Group"].astype(str) == grp, "percentage"].to_numpy(dtype=float),
                )
                for grp in groups
            ]
            sub_stats = (
                self.result.stats_df[self.result.stats_df["cell_type"] == focus]
                if not self.result.stats_df.empty
                else pd.DataFrame()
            )
            _draw_box_with_replicates(
                ax,
                grouped_vals,
                group_colors,
                cfg,
                ylabel="Cell Percentage (%)",
                title=f"{focus} Proportion by Cohort ({self.result.denominator_mode})",
                stats_df=sub_stats,
                ylim_zero=True,
            )
            return [ax]

        ctypes = list(self.result.cell_types)
        n_groups = max(len(groups), 1)
        width = min(0.75 / n_groups, 0.32)
        x_base = np.arange(len(ctypes), dtype=float)
        preset = THEME_PRESETS.get(cfg.theme, THEME_PRESETS["IRIS Dark"])
        edge_col = "#FFFFFF" if preset["is_dark"] else "#111111"
        rng = np.random.default_rng(31)
        cell_max_y = np.zeros(len(ctypes), dtype=float)

        for g_idx, grp in enumerate(groups):
            offset = (g_idx - (n_groups - 1) / 2.0) * width
            pos = x_base + offset
            arrays = []
            for c_idx, ct in enumerate(ctypes):
                vals = sample_df.loc[
                    (sample_df["Group"].astype(str) == grp) & (sample_df["cell_type"] == ct),
                    "percentage",
                ].to_numpy(dtype=float)
                arrays.append(vals if vals.size else np.array([0.0]))
                if vals.size:
                    jitter = rng.uniform(-width * 0.2, width * 0.2, size=vals.size)
                    ax.scatter(
                        np.full(vals.size, pos[c_idx]) + jitter,
                        vals,
                        color=group_colors[grp],
                        edgecolor=edge_col,
                        linewidth=0.7,
                        s=18,
                        zorder=4,
                    )
                    cell_max_y[c_idx] = max(cell_max_y[c_idx], float(np.max(vals)))

            bp = ax.boxplot(
                arrays,
                positions=pos,
                widths=width * 0.85,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": edge_col, "linewidth": 1.2},
                whiskerprops={"color": edge_col, "linewidth": 0.9},
                capprops={"color": edge_col, "linewidth": 0.9},
                boxprops={"linewidth": 0.9},
            )
            for idx_b, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(group_colors[grp])
                patch.set_alpha(0.55)
                patch.set_edgecolor(edge_col)
                if idx_b == 0:
                    patch.set_label(grp)

        y_top = float(np.max(cell_max_y)) if cell_max_y.size else 10.0
        if not self.result.stats_df.empty:
            for c_idx, ct in enumerate(ctypes):
                ct_stats = self.result.stats_df[self.result.stats_df["cell_type"] == ct]
                sig = ct_stats[ct_stats["significant"].astype(bool)]
                if sig.empty:
                    continue
                lbl = str(sig.sort_values("q_value").iloc[0]["significance_label"])
                if lbl and lbl != "ns":
                    ax.text(
                        x_base[c_idx],
                        cell_max_y[c_idx] + max(y_top * 0.04, 1.5),
                        lbl,
                        ha="center",
                        va="bottom",
                        color=preset["bracket_color"],
                        fontsize=float(cfg.stat_size),
                        fontweight="bold",
                    )

        ax.set_xticks(x_base)
        ax.set_xticklabels(ctypes, rotation=30 if len(ctypes) > 4 else 0, ha="right")
        ax.set_ylabel("Proportion (%)")
        ax.set_title(f"Cell-Type Proportions Across Cohorts ({self.result.denominator_mode})")
        ax.set_ylim(0.0, max(y_top * 1.18, 5.0))
        if len(groups) > 1 or groups != ["All"]:
            ax.legend(loc="best", frameon=True)
        return [ax]

    def _plot_clr_dotplot(
        self,
        fig: Figure,
        cfg: PlotStyleConfig,
    ) -> List[matplotlib.axes.Axes]:
        ax = fig.add_subplot(111)
        rep = self.result.replicates_df
        if rep.empty or not self.result.cell_types:
            ax.text(0.5, 0.5, "No compositional CLR data available.", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            return [ax]

        use_groups = "Group" in rep.columns and rep["Group"].nunique() >= 2
        row_col = "Group" if use_groups else COL_SAMPLE
        agg = (
            rep.groupby([row_col, "cell_type"], observed=True, as_index=False)[
                ["percentage", "clr"]
            ]
            .mean()
        )
        rows = list(dict.fromkeys(agg[row_col].astype(str).tolist()))
        ctypes = list(self.result.cell_types)
        row_idx = {r: i for i, r in enumerate(rows)}
        col_idx = {c: j for j, c in enumerate(ctypes)}

        xs = np.array([col_idx[str(c)] for c in agg["cell_type"]], dtype=float)
        ys = np.array([row_idx[str(r)] for r in agg[row_col]], dtype=float)
        pcts = agg["percentage"].to_numpy(dtype=float)
        clrs = agg["clr"].to_numpy(dtype=float)

        sizes = np.clip(pcts * 8.5, 25.0, 520.0)
        max_abs = max(float(np.max(np.abs(clrs))) if clrs.size else 1.0, 0.5)
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

        preset = THEME_PRESETS.get(cfg.theme, THEME_PRESETS["IRIS Dark"])
        edge_col = "#FFFFFF" if preset["is_dark"] else "#222222"

        sc = ax.scatter(
            xs,
            ys,
            s=sizes,
            c=clrs,
            cmap="RdBu_r",
            norm=norm,
            edgecolor=edge_col,
            linewidth=0.8,
            zorder=3,
        )

        if use_groups and not self.result.stats_df.empty:
            for c_i, ct in enumerate(ctypes):
                ct_stats = self.result.stats_df[self.result.stats_df["cell_type"] == ct]
                sig = ct_stats[ct_stats["significant"].astype(bool)]
                if not sig.empty:
                    lbl = str(sig.sort_values("q_value").iloc[0]["significance_label"])
                    if lbl and lbl != "ns":
                        ax.text(
                            c_i,
                            len(rows) - 0.35,
                            lbl,
                            ha="center",
                            va="bottom",
                            color=preset["bracket_color"],
                            fontsize=float(cfg.stat_size),
                            fontweight="bold",
                        )

        ax.set_xticks(range(len(ctypes)))
        ax.set_xticklabels(ctypes, rotation=30 if len(ctypes) > 4 else 0, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
        ax.set_xlim(-0.6, len(ctypes) - 0.4)
        ax.set_ylim(-0.6, len(rows) - 0.1)
        ax.set_title("Compositional Dot-Plot (Size = Mean %, Color = CLR)")

        cbar = fig.colorbar(sc, ax=ax, pad=0.03, fraction=0.04)
        cbar.set_label(
            "Centered Log-Ratio (CLR)",
            color=preset["text_color"],
            fontsize=cfg.label_size,
        )
        cbar.ax.yaxis.set_tick_params(color=preset["tick_color"], labelcolor=preset["tick_color"])
        return [ax]

    def _plot_diversity(
        self,
        fig: Figure,
        cfg: PlotStyleConfig,
    ) -> List[matplotlib.axes.Axes]:
        div = self.result.diversity_df
        if div.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No diversity data available.", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            return [ax]

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        has_group = "Group" in div.columns
        groups = (
            list(dict.fromkeys(div["Group"].astype(str).tolist()))
            if has_group
            else ["All"]
        )
        colors = resolve_category_colors(groups, cfg)

        div_stats = self.result.metadata.get("diversity_stats_df")
        div_stats_df = div_stats if isinstance(div_stats, pd.DataFrame) else pd.DataFrame()

        for ax_obj, metric, label_str, title_str in (
            (ax1, "shannon_H", "Shannon Entropy (H)", "Shannon Diversity (H)"),
            (ax2, "simpson_1_minus_D", "Simpson Index (1 - D)", "Simpson Diversity (1 - D)"),
        ):
            grouped_vals = [
                (
                    grp,
                    (
                        div.loc[div["Group"].astype(str) == grp, metric].to_numpy(dtype=float)
                        if has_group
                        else div[metric].to_numpy(dtype=float)
                    ),
                )
                for grp in groups
            ]
            m_stats = (
                div_stats_df[div_stats_df["metric"] == metric]
                if not div_stats_df.empty and "metric" in div_stats_df.columns
                else pd.DataFrame()
            )
            _draw_box_with_replicates(
                ax_obj,
                grouped_vals,
                colors,
                cfg,
                ylabel=label_str,
                title=title_str,
                stats_df=m_stats,
                ylim_zero=True,
            )

        return [ax1, ax2]


class CellProportionWidget(QWidget):
    """Dockable widget for compositional cell-type proportion and diversity analysis."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(SHARED_DARK_THEME)

        self.df: Optional[pd.DataFrame] = None
        self.markers: List[str] = []
        self.celltype_col: str = "cell_type"
        self.scale_factor: float = 1.0
        self.ctx: Any = None
        self.rois: List[ROISpec] = []
        self.group_map: Dict[str, str] = {}
        self.last_result: Optional[ProportionResult] = None
        self.popup_dialog: Optional[CellProportionStudioDialog] = None
        self._scope_mode: str = SCOPE_ALL
        self._scoped_samples: List[str] = []
        self._all_scope_samples: List[str] = []

        self._build_ui()
        self._render_preview()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # Denominator Configuration Box
        denom_box, denom_layout = create_section_card("Denominator & Compartment Scope")

        mode_form = QFormLayout()
        mode_form.setSpacing(6)
        self.combo_denominator = QComboBox()
        self.combo_denominator.addItems([lbl for lbl, _ in DENOMINATOR_MODE_CHOICES])
        self.combo_denominator.currentTextChanged.connect(self._on_denominator_changed)
        mode_form.addRow("Denominator Mode:", self.combo_denominator)
        denom_layout.addLayout(mode_form)

        # Parent Lineage Subset Selector (shown for parent_lineage)
        self.parent_container = QWidget()
        self.parent_container.setObjectName("irisCardBody")
        self.parent_container.setStyleSheet(
            "QWidget#irisCardBody { background-color: transparent; }"
        )
        p_layout = QVBoxLayout(self.parent_container)
        p_layout.setContentsMargins(0, 0, 0, 0)
        p_layout.setSpacing(6)
        p_hdr = QHBoxLayout()
        p_hdr.setSpacing(8)
        p_hdr.addWidget(QLabel("Parent Lineage Subset:"))
        btn_p_all = QPushButton("All")
        btn_p_all.setMinimumHeight(28)
        btn_p_all.clicked.connect(lambda: self._set_all_parent_checks(True))
        btn_p_none = QPushButton("None")
        btn_p_none.setMinimumHeight(28)
        btn_p_none.clicked.connect(lambda: self._set_all_parent_checks(False))
        p_hdr.addWidget(btn_p_all)
        p_hdr.addWidget(btn_p_none)
        p_layout.addLayout(p_hdr)

        self.list_parent_types = QListWidget()
        self.list_parent_types.setMaximumHeight(110)
        p_layout.addWidget(self.list_parent_types)
        self.parent_container.setVisible(False)
        denom_layout.addWidget(self.parent_container)

        # ROI Compartment Controls (shown for roi_compartment)
        self.roi_container = QWidget()
        self.roi_container.setObjectName("irisCardBody")
        self.roi_container.setStyleSheet(
            "QWidget#irisCardBody { background-color: transparent; }"
        )
        r_layout = QVBoxLayout(self.roi_container)
        r_layout.setContentsMargins(0, 0, 0, 0)
        r_layout.setSpacing(6)
        r_btn_row = QHBoxLayout()
        r_btn_row.setSpacing(8)
        self.btn_sync_rois = QPushButton("Sync Viewer ROIs")
        self.btn_sync_rois.setMinimumHeight(28)
        self.btn_sync_rois.clicked.connect(self.sync_rois_from_context)
        r_btn_row.addWidget(self.btn_sync_rois)
        self.btn_import_rois = QPushButton("Import Regions JSON...")
        self.btn_import_rois.setMinimumHeight(28)
        self.btn_import_rois.clicked.connect(self._prompt_import_rois)
        r_btn_row.addWidget(self.btn_import_rois)
        r_layout.addLayout(r_btn_row)
        self.lbl_roi_info = QLabel("Active Compartment ROIs: 0")
        style_section_header(self.lbl_roi_info, subtitle=True)
        r_layout.addWidget(self.lbl_roi_info)
        self.roi_container.setVisible(False)
        denom_layout.addWidget(self.roi_container)

        layout.addWidget(denom_box)

        # Cohort & Statistics Box
        cfg_box, cfg_layout = create_section_card("Cohort, Statistics & Preview")
        form = QFormLayout()
        form.setSpacing(6)

        grp_row = QHBoxLayout()
        grp_row.setSpacing(8)
        self.btn_assign_groups = QPushButton("Assign Cohort Groups...")
        self.btn_assign_groups.setMinimumHeight(28)
        self.btn_assign_groups.clicked.connect(self.open_group_assignment_dialog)
        grp_row.addWidget(self.btn_assign_groups)
        self.lbl_groups_status = QLabel("Ungrouped")
        style_section_header(self.lbl_groups_status, subtitle=True)
        grp_row.addWidget(self.lbl_groups_status, 1)
        form.addRow("Cohorts:", grp_row)

        self.combo_stat_test = QComboBox()
        self.combo_stat_test.addItems([label for label, _ in STAT_TEST_CHOICES])
        form.addRow("Statistical Test:", self.combo_stat_test)

        self.combo_preview_mode = QComboBox()
        self.combo_preview_mode.addItems(list(PREVIEW_PLOT_MODES))
        self.combo_preview_mode.currentTextChanged.connect(lambda _: self._render_preview())
        form.addRow("Preview Mode:", self.combo_preview_mode)
        cfg_layout.addLayout(form)

        self.btn_compute = QPushButton("Compute Proportions")
        self.btn_compute.setStyleSheet(BUTTON_CTA)
        self.btn_compute.setMinimumHeight(28)
        self.btn_compute.clicked.connect(self.compute_proportions)
        cfg_layout.addWidget(self.btn_compute)
        layout.addWidget(cfg_box)

        # Embedded Plot Preview Box
        plot_box, pb_layout = create_section_card(margins=(10, 6, 10, 10), spacing=8)

        hdr = QHBoxLayout()
        lbl_plot_title = QLabel("Cell-Type Composition")
        style_section_header(lbl_plot_title, subtitle=False)
        hdr.addWidget(lbl_plot_title)
        hdr.addStretch(1)
        self.btn_popout = QPushButton("\u2922 Pop-up Window")
        self.btn_popout.setStyleSheet(POPOUT_BUTTON_STYLE)
        self.btn_popout.setToolTip("Open Cell Proportions Figure Studio")
        self.btn_popout.clicked.connect(self.open_popup_window)
        hdr.addWidget(self.btn_popout)
        pb_layout.addLayout(hdr)

        self.figure = Figure(figsize=(5.5, 4.0), dpi=100)
        self.figure.patch.set_facecolor("#14161d")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(260)
        pb_layout.addWidget(self.canvas, 1)

        layout.addWidget(plot_box, 1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

    def set_sample_scope(
        self,
        mode: str,
        resolved_samples: Sequence[str],
        all_samples: Optional[Sequence[str]] = None,
    ) -> None:
        """Store current sample scope context pushed from ``IrisAnalyzerPanelWidget``."""
        self._scope_mode = str(mode or SCOPE_ALL)
        self._scoped_samples = [str(s).strip() for s in resolved_samples if str(s).strip()]
        self._all_scope_samples = [
            str(s).strip() for s in (all_samples or []) if str(s).strip()
        ]
        self._update_roi_info_label()

    def _active_scope_samples(self) -> List[str]:
        return resolve_scoped_samples(self.df, self._scoped_samples, ctx=self.ctx)

    def _filter_rois_for_scope(self, rois: Sequence[ROISpec]) -> List[ROISpec]:
        if not rois:
            return []
        if self.df is not None and self.df.empty:
            return []
        if self._scope_mode in (SCOPE_ACTIVE, SCOPE_SELECTED) and not self._scoped_samples:
            return []
        active = self._active_scope_samples()
        return filter_rois_for_samples(rois, active)

    def _update_roi_info_label(self) -> None:
        scoped = self._filter_rois_for_scope(self.rois)
        if len(scoped) != len(self.rois):
            self.lbl_roi_info.setText(
                f"Active Compartment ROIs: {len(scoped)}/{len(self.rois)} in scope"
            )
        else:
            self.lbl_roi_info.setText(f"Active Compartment ROIs: {len(self.rois)}")

    def set_data(
        self,
        df: Optional[pd.DataFrame],
        markers: Optional[Sequence[str]] = None,
        celltype_col: str = "cell_type",
        scale_factor: float = 1.0,
        ctx: Any = None,
    ) -> None:
        """Bind single-cell table, physical calibration scale, and optional viewer context."""
        if ctx is not None:
            self.ctx = ctx
        if df is None and self.ctx is not None and getattr(self.ctx, "state", None) is not None:
            state = self.ctx.state
            df = (
                state.current_mapped_df
                if getattr(state, "current_mapped_df", None) is not None
                else getattr(state, "master_df", None)
            )
        self.df = ensure_canonical_dataframe(df) if df is not None and not df.empty else df
        self.markers = list(markers or [])
        self.celltype_col = celltype_col or "cell_type"
        self.scale_factor = float(scale_factor) if scale_factor and scale_factor > 0 else 1.0

        self._populate_parent_types()
        self.sync_rois_from_context()
        self._update_roi_info_label()
        self._render_preview()

    def _populate_parent_types(self) -> None:
        prev_checked = set(self.get_parent_subset())
        self.list_parent_types.clear()
        if self.df is None or self.df.empty:
            return
        ct_col = self.celltype_col if self.celltype_col in self.df.columns else "cell_type"
        if ct_col not in self.df.columns:
            return
        types = sorted(self.df[ct_col].dropna().astype(str).unique().tolist())
        for ct in types:
            item = QListWidgetItem(ct)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            state = Qt.Checked if (not prev_checked or ct in prev_checked) else Qt.Unchecked
            item.setCheckState(state)
            self.list_parent_types.addItem(item)

    def _set_all_parent_checks(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        for idx in range(self.list_parent_types.count()):
            self.list_parent_types.item(idx).setCheckState(state)

    def get_parent_subset(self) -> List[str]:
        """Return currently checked cell types in the parent lineage selector."""
        selected: List[str] = []
        for idx in range(self.list_parent_types.count()):
            item = self.list_parent_types.item(idx)
            if item.checkState() == Qt.Checked:
                selected.append(item.text().strip())
        return selected

    def set_parent_subset(self, subset: Sequence[str]) -> None:
        """Programmatically check only cell types in *subset*."""
        target = {str(x) for x in subset}
        for idx in range(self.list_parent_types.count()):
            item = self.list_parent_types.item(idx)
            item.setCheckState(Qt.Checked if item.text() in target else Qt.Unchecked)

    def set_rois(self, rois: Sequence[ROISpec]) -> None:
        """Programmatically set compartment ``ROISpec`` objects."""
        self.rois = list(rois)
        self._update_roi_info_label()

    def sync_rois_from_context(self) -> List[ROISpec]:
        """Pull active ROIs from ``ctx.density_rois`` or napari Shapes layers."""
        if self.ctx is None:
            return list(self.rois)
        shared = getattr(self.ctx, "density_rois", None)
        if shared:
            self.set_rois(list(shared))
            return list(self.rois)
        viewer = getattr(self.ctx, "viewer", None)
        if viewer is not None:
            incl, excl = extract_viewer_roi_polygons(viewer)
            if incl:
                active_samples = self._active_scope_samples()
                cur_sample = getattr(
                    getattr(self.ctx, "state", None), "current_sample_name", None
                )
                target_sample: Optional[str] = None
                if len(active_samples) == 1:
                    if cur_sample and normalize_sample_key(cur_sample) == normalize_sample_key(
                        active_samples[0]
                    ):
                        target_sample = active_samples[0]
                    elif self._scope_mode in (SCOPE_ACTIVE, SCOPE_SELECTED) or not cur_sample:
                        target_sample = active_samples[0]
                    else:
                        target_sample = str(cur_sample)
                elif cur_sample:
                    if not active_samples or normalize_sample_key(cur_sample) in {
                        normalize_sample_key(s) for s in active_samples
                    }:
                        target_sample = str(cur_sample)
                self.set_rois(
                    [
                        ROISpec(
                            name="Viewer Compartment",
                            sample=target_sample,
                            inclusion_polygons=incl,
                            exclusion_polygons=excl,
                            microns_per_pixel=self.scale_factor,
                        )
                    ]
                )
        return list(self.rois)

    def _prompt_import_rois(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Import ROI Compartments JSON",
            "",
            "JSON Files (*.json)",
        )
        if filepath:
            self.import_rois_from_path(filepath)

    def import_rois_from_path(self, filepath: Union[str, Path]) -> List[ROISpec]:
        """Load compartment ``ROISpec`` entries from a JSON file."""
        with open(Path(filepath), "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        items = payload.get("rois", []) if isinstance(payload, dict) else payload
        loaded: List[ROISpec] = []
        for idx, item in enumerate(items or []):
            if not isinstance(item, dict):
                continue
            loaded.append(
                ROISpec(
                    name=str(item.get("name", f"ROI_{idx + 1}")),
                    sample=item.get("sample"),
                    inclusion_polygons=[
                        np.asarray(p, dtype=float) for p in item.get("inclusion_polygons", [])
                    ],
                    exclusion_polygons=[
                        np.asarray(p, dtype=float) for p in item.get("exclusion_polygons", [])
                    ],
                    microns_per_pixel=float(item.get("microns_per_pixel", self.scale_factor)),
                )
            )
        self.set_rois(loaded)
        return loaded

    def set_group_map(self, group_map: Optional[Dict[str, str]]) -> None:
        """Programmatically assign sample-to-cohort mappings."""
        self.group_map = {str(k): str(v) for k, v in (group_map or {}).items() if str(v).strip()}
        if self.group_map:
            n_groups = len(set(self.group_map.values()))
            self.lbl_groups_status.setText(
                f"{len(self.group_map)} samples across {n_groups} groups"
            )
        else:
            self.lbl_groups_status.setText("Ungrouped")

    def open_group_assignment_dialog(self) -> Optional[Dict[str, str]]:
        """Open sample-to-cohort assignment dialog and store resulting group map."""
        samples: List[str] = []
        if self.df is not None and not self.df.empty:
            s_col = find_sample_column(self.df) or COL_SAMPLE
            if s_col in self.df.columns:
                samples = sorted(self.df[s_col].dropna().astype(str).unique().tolist())
        dlg = SampleGroupAssignmentDialog(samples, self.group_map, parent=self)
        if dlg.exec_():
            self.set_group_map(dlg.get_group_map())
            return dict(self.group_map)
        return None

    def _on_denominator_changed(self, text: str) -> None:
        mode = denominator_mode_code_from_label(text)
        self.parent_container.setVisible(mode == "parent_lineage")
        self.roi_container.setVisible(mode == "roi_compartment")

    def compute_proportions(self) -> Optional[ProportionResult]:
        """Compute cell proportions, CLR matrix, diversity, and cohort statistics."""
        if self.df is None or self.df.empty:
            self._render_preview()
            return None

        mode = denominator_mode_code_from_label(self.combo_denominator.currentText())
        test_code = stat_test_code_from_label(self.combo_stat_test.currentText())
        parent_subset = self.get_parent_subset() if mode == "parent_lineage" else None

        if mode == "parent_lineage" and not parent_subset:
            QMessageBox.warning(
                self,
                "Parent Lineage Required",
                "Select at least one cell type for the Parent Lineage Subset denominator.",
            )
            return None

        effective_rois: Optional[List[ROISpec]] = None
        if mode == "roi_compartment":
            if not self.rois:
                self.sync_rois_from_context()
            effective_rois = self._filter_rois_for_scope(self.rois)
            if not effective_rois:
                QMessageBox.warning(
                    self,
                    "ROIs Required",
                    "Define or import ROI compartments matching the active sample scope.",
                )
                return None

        self.last_result = compute_cell_proportions(
            self.df,
            cell_type_col=self.celltype_col,
            denominator_mode=mode,
            parent_subset=parent_subset,
            rois=effective_rois,
            scale_factor=self.scale_factor,
            group_map=self.group_map if self.group_map else None,
            stat_test=test_code,
        )
        n_cells = int(self.last_result.metadata.get("n_cells", 0))
        n_samples = int(self.last_result.metadata.get("n_samples", 0))
        n_types = len(self.last_result.cell_types)
        iris_log(
            "IRIS-Analyzer",
            "Cell Proportion",
            f"Computed proportions (mode={mode}, test={test_code}): "
            f"{n_samples} sample(s), {n_types} cell type(s), {n_cells:,} cells.",
            flush=True,
        )
        self._render_preview()
        if self.popup_dialog is not None and self.popup_dialog.isVisible():
            self.popup_dialog.update_data(self.last_result)
        return self.last_result

    def _render_preview(self) -> None:
        self.figure.clear()
        cfg = PlotStyleConfig(theme="IRIS Dark")
        if self.last_result is None or self.last_result.replicates_df.empty:
            ax = self.figure.add_subplot(111)
            ax.set_facecolor("#14161d")
            ax.text(
                0.5,
                0.5,
                "Load data and click 'Compute Proportions' to analyze.",
                color="#888888",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            apply_plot_style(self.figure, [ax], cfg)
            self.canvas.draw_idle()
            return

        mode = self.combo_preview_mode.currentText()
        if mode == "Ecological Diversity":
            ax = self.figure.add_subplot(111)
            div = self.last_result.diversity_df
            has_group = "Group" in div.columns
            groups = (
                list(dict.fromkeys(div["Group"].astype(str).tolist()))
                if has_group
                else ["All"]
            )
            colors = resolve_category_colors(groups, cfg)
            grouped = [
                (
                    g,
                    (
                        div.loc[div["Group"].astype(str) == g, "shannon_H"].to_numpy(dtype=float)
                        if has_group
                        else div["shannon_H"].to_numpy(dtype=float)
                    ),
                )
                for g in groups
            ]
            div_stats = self.last_result.metadata.get("diversity_stats_df")
            m_stats = (
                div_stats[div_stats["metric"] == "shannon_H"]
                if isinstance(div_stats, pd.DataFrame) and not div_stats.empty
                else pd.DataFrame()
            )
            _draw_box_with_replicates(
                ax,
                grouped,
                colors,
                cfg,
                ylabel="Shannon Entropy (H)",
                title="Sample Ecological Diversity",
                stats_df=m_stats,
            )
            apply_plot_style(self.figure, [ax], cfg)
        elif mode == "Grouped Box + Replicates":
            ax = self.figure.add_subplot(111)
            rep = self.last_result.replicates_df
            top_ct = (
                rep.groupby("cell_type", observed=True)["percentage"]
                .mean()
                .idxmax()
            )
            sub = rep[rep["cell_type"] == top_ct]
            sample_agg = sub.groupby(["Group", COL_SAMPLE], observed=True, as_index=False)[
                "percentage"
            ].mean()
            groups = list(dict.fromkeys(sample_agg["Group"].astype(str).tolist()))
            colors = resolve_category_colors(groups, cfg)
            grouped = [
                (
                    g,
                    sample_agg.loc[sample_agg["Group"].astype(str) == g, "percentage"].to_numpy(
                        dtype=float
                    ),
                )
                for g in groups
            ]
            sub_stats = (
                self.last_result.stats_df[self.last_result.stats_df["cell_type"] == top_ct]
                if not self.last_result.stats_df.empty
                else pd.DataFrame()
            )
            _draw_box_with_replicates(
                ax,
                grouped,
                colors,
                cfg,
                ylabel=f"{top_ct} (%)",
                title=f"Cohort Comparison: {top_ct}",
                stats_df=sub_stats,
            )
            apply_plot_style(self.figure, [ax], cfg)
        else:
            ax = self.figure.add_subplot(111)
            _draw_stacked_composition(ax, self.last_result, cfg, by_group=False)
            apply_plot_style(self.figure, [ax], cfg)

        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

    def open_popup_window(self) -> Optional[CellProportionStudioDialog]:
        """Open or focus the interactive ``CellProportionStudioDialog`` Figure Studio window."""
        if self.last_result is None and self.df is not None and not self.df.empty:
            self.compute_proportions()
        if self.last_result is None:
            QMessageBox.information(
                self,
                "No Proportion Data",
                "Load single-cell data and run Compute Proportions first.",
            )
            return None

        if self.popup_dialog is None:
            self.popup_dialog = CellProportionStudioDialog(
                self.last_result,
                parent=self.window(),
            )
        else:
            self.popup_dialog.update_data(self.last_result)

        self.popup_dialog.show()
        self.popup_dialog.raise_()
        self.popup_dialog.activateWindow()
        return self.popup_dialog

    def closeEvent(self, event: Any) -> None:
        """Close child Figure Studio dialog and clear Matplotlib figure resources."""
        if self.popup_dialog is not None:
            self.popup_dialog.close()
        self.figure.clear()
        super().closeEvent(event)


__all__ = [
    "DENOMINATOR_MODE_CHOICES",
    "PREVIEW_PLOT_MODES",
    "CellProportionStudioDialog",
    "CellProportionWidget",
]
