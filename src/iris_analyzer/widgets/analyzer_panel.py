"""
IRIS Analyzer Unified Dockable Panel Widget
===========================================
Combines Spatial Cell Density, Compositional Proportions & Diversity,
Pairwise Spatial Interactions, and Cellular Neighborhoods / Spatial-LDA
behind a compact segmented navigation bar, shared sample scope selector,
and dataset summary header.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from iris_analyzer.widgets.density_widget import CellDensityWidget
from iris_analyzer.widgets.interactions_widget import PairwiseInteractionsWidget
from iris_analyzer.widgets.neighborhoods_widget import CellularNeighborhoodsWidget
from iris_analyzer.widgets.proportion_widget import CellProportionWidget
from iris_common.logging import iris_log
from iris_common.schema import find_sample_column
from iris_common.widgets import SCOPE_ACTIVE, SCOPE_ALL, SCOPE_SELECTED, SampleScopeBar
from iris_common.widgets.theme import (
    SECTION_SUBTITLE_STYLE,
    SHARED_DARK_THEME,
    create_section_card,
)


class IrisAnalyzerPanelWidget(QWidget):
    """Unified dockable Spatial Analyzer combining 4 downstream analytics tabs."""

    subPanelChanged = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None, ctx: Any = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(SHARED_DARK_THEME)
        self._ctx: Any = ctx
        self._full_df: Optional[pd.DataFrame] = None
        self._scoped_df: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None
        self._all_samples: List[str] = []
        self.markers: List[str] = []
        self.celltype_col: str = "cell_type"
        self.scale_factor: float = 1.0
        self.group_map: Dict[str, str] = {}

        self._build_ui()
        self.sync_sample_scope(emit=False)
        self._update_summary_header()

    @property
    def ctx(self) -> Any:
        return self._ctx

    @ctx.setter
    def ctx(self, value: Any) -> None:
        self._ctx = value
        for child in self._child_widgets():
            child.ctx = value
        self.sync_sample_scope(emit=False)
        if self._full_df is not None:
            self._apply_current_scope(log_change=False)
        else:
            self._update_summary_header()

    def _child_widgets(self) -> tuple:
        return (
            self.density_widget,
            self.proportion_widget,
            self.interactions_widget,
            self.neighborhoods_widget,
        )

    def _build_ui(self) -> None:
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top Navigation, Sample Scope & Dataset Summary Header (Control-aligned outer margins)
        top_wrap = QWidget(self)
        top_layout = QVBoxLayout(top_wrap)
        top_layout.setContentsMargins(10, 10, 10, 2)
        top_layout.setSpacing(0)

        header_box, hb = create_section_card(
            "Spatial Analyzer",
            margins=(10, 6, 10, 10),
            spacing=8,
        )

        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)
        self.btn_nav_density = QPushButton("Density")
        self.btn_nav_proportions = QPushButton("Proportions")
        self.btn_nav_interactions = QPushButton("Interactions")
        self.btn_nav_neighborhoods = QPushButton("Neighborhoods")

        self.nav_buttons = [
            self.btn_nav_density,
            self.btn_nav_proportions,
            self.btn_nav_interactions,
            self.btn_nav_neighborhoods,
        ]
        for idx, btn in enumerate(self.nav_buttons):
            btn.setCheckable(True)
            btn.setMinimumHeight(28)
            btn.clicked.connect(lambda _checked, i=idx: self._switch_sub_panel(i))
            nav_row.addWidget(btn)
        self.btn_nav_density.setChecked(True)
        hb.addLayout(nav_row)

        self.sample_scope_bar = SampleScopeBar(
            self,
            default_mode=SCOPE_ALL,
            label="Sample Scope:",
            compact=True,
        )
        self.sample_scope_bar.scopeChanged.connect(self._on_scope_changed)
        hb.addWidget(self.sample_scope_bar)

        self.lbl_summary = QLabel(
            "Scope: All (0/0 samples) | No cells loaded | Scale: 1 \u00b5m/px | Cohorts: Ungrouped"
        )
        self.lbl_summary.setStyleSheet(SECTION_SUBTITLE_STYLE)
        self.lbl_summary.setWordWrap(True)
        hb.addWidget(self.lbl_summary)
        top_layout.addWidget(header_box)
        main_layout.addWidget(top_wrap)

        # Stacked Analytics Sub-panels
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Index 0: Density
        self.density_widget = CellDensityWidget(self)
        self.density_widget.ctx = self._ctx
        self.density_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stack.addWidget(self.density_widget)

        # Index 1: Proportions
        self.proportion_widget = CellProportionWidget(self)
        self.proportion_widget.ctx = self._ctx
        self.proportion_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stack.addWidget(self.proportion_widget)

        # Index 2: Interactions
        self.interactions_widget = PairwiseInteractionsWidget(self, ctx=self._ctx)
        self.interactions_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stack.addWidget(self.interactions_widget)

        # Index 3: Neighborhoods
        self.neighborhoods_widget = CellularNeighborhoodsWidget(self, ctx=self._ctx)
        self.neighborhoods_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stack.addWidget(self.neighborhoods_widget)

        main_layout.addWidget(self.stack, 1)
        self._update_nav_styles(0)

    def _switch_sub_panel(self, idx: int) -> None:
        if idx < 0 or idx >= self.stack.count():
            return
        self.stack.setCurrentIndex(idx)
        self._update_nav_styles(idx)
        if idx == 1 and hasattr(self.proportion_widget, "sync_rois_from_context"):
            self.proportion_widget.sync_rois_from_context()
        self.subPanelChanged.emit(idx)

    def switch_tab(self, idx: int) -> None:
        """Programmatically switch to sub-panel *idx* (0..3)."""
        self._switch_sub_panel(idx)

    def _update_nav_styles(self, active_idx: int) -> None:
        for i, btn in enumerate(self.nav_buttons):
            btn.blockSignals(True)
            btn.setChecked(i == active_idx)
            btn.blockSignals(False)
            if i == active_idx:
                btn.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #3A6A9A; color: white; font-weight: bold;
                        border: 1px solid #4B82B5; border-radius: 4px;
                        padding: 4px 6px; font-size: 11px;
                    }
                    """
                )
            else:
                btn.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #2A2E39; color: #9AA4B2; font-weight: bold;
                        border: 1px solid #3c4352; border-radius: 4px;
                        padding: 4px 6px; font-size: 11px;
                    }
                    QPushButton:hover { background-color: #343A48; color: #F0F2F5; }
                    """
                )

    def _collect_samples_from_data_and_ctx(self) -> Tuple[List[str], List[str]]:
        """Collect total available sample IDs and active canvas sample IDs."""
        all_candidates: List[str] = []
        if self._full_df is not None and not self._full_df.empty:
            s_col = find_sample_column(self._full_df)
            if s_col and s_col in self._full_df.columns:
                all_candidates.extend(
                    str(v).strip()
                    for v in self._full_df[s_col].dropna().unique()
                    if str(v).strip()
                )

        state = getattr(self._ctx, "state", None) if self._ctx is not None else None
        active_samples: List[str] = []
        if state is not None:
            file_map = getattr(state, "file_map", None)
            if isinstance(file_map, dict):
                all_candidates.extend(
                    str(k).strip() for k in file_map.keys() if str(k).strip()
                )

            layers_map = getattr(state, "sample_layers_map", None)
            if isinstance(layers_map, dict):
                layer_keys = [str(k).strip() for k in layers_map.keys() if str(k).strip()]
                all_candidates.extend(layer_keys)
                if len(layer_keys) > 1:
                    active_samples = layer_keys

            if not active_samples:
                cur_name = getattr(state, "current_sample_name", None)
                if cur_name and str(cur_name).strip():
                    active_samples = [str(cur_name).strip()]
                elif isinstance(layers_map, dict) and len(layers_map) == 1:
                    active_samples = [
                        str(k).strip() for k in layers_map.keys() if str(k).strip()
                    ]
                else:
                    multi = getattr(state, "multi_sample_images", None)
                    if isinstance(multi, dict) and multi:
                        active_samples = [
                            str(k).strip() for k in multi.keys() if str(k).strip()
                        ]

        all_samples = list(dict.fromkeys(all_candidates))
        return all_samples, active_samples

    def sync_sample_scope(self, *, emit: bool = False) -> None:
        """Refresh sample scope options from ``self._full_df`` and ``self._ctx``."""
        all_samples, active_samples = self._collect_samples_from_data_and_ctx()
        self._all_samples = all_samples
        self.sample_scope_bar.set_samples(all_samples, active_samples, emit=emit)

    def _on_scope_changed(self, _mode: str, _resolved: list) -> None:
        self._apply_current_scope(log_change=True, markers_updated=False)

    def _apply_current_scope(
        self,
        *,
        log_change: bool = True,
        markers_updated: bool = False,
    ) -> None:
        """Filter ``_full_df`` by current scope and propagate to all 4 sub-panels."""
        self._scoped_df = self.sample_scope_bar.filter_dataframe(self._full_df)
        self.df = self._scoped_df

        mode = self.sample_scope_bar.get_scope_mode()
        resolved = self.sample_scope_bar.resolve_samples()

        for child in self._child_widgets():
            if hasattr(child, "set_sample_scope"):
                child.set_sample_scope(mode, resolved, self._all_samples)
            child.set_data(
                self._scoped_df,
                markers=self.markers if markers_updated else None,
                celltype_col=self.celltype_col,
                scale_factor=self.scale_factor,
                ctx=self._ctx,
            )

        self._update_summary_header()

        if log_change and (self._full_df is not None or self._all_samples):
            n_cells = len(self._scoped_df) if self._scoped_df is not None else 0
            n_total = len(self._all_samples)
            iris_log(
                "IRIS-Analyzer",
                "Scope",
                f"{mode}: {len(resolved)}/{n_total} sample(s), {n_cells:,} cells.",
                flush=True,
            )

    def set_data(
        self,
        df: Optional[pd.DataFrame],
        markers: Optional[Sequence[str]] = None,
        celltype_col: str = "cell_type",
        scale_factor: float = 1.0,
        ctx: Any = None,
    ) -> None:
        """Propagate dataset, markers, phenotype column, scale, and context to all 4 sub-panels."""
        if ctx is not None:
            self._ctx = ctx
        if df is None and self._ctx is not None and getattr(self._ctx, "state", None) is not None:
            state = self._ctx.state
            df = (
                state.current_mapped_df
                if getattr(state, "current_mapped_df", None) is not None
                else getattr(state, "master_df", None)
            )

        self._full_df = df
        if markers is not None:
            self.markers = list(markers)
        self.celltype_col = celltype_col or "cell_type"
        self.scale_factor = float(scale_factor) if scale_factor and scale_factor > 0 else 1.0

        self.sync_sample_scope(emit=False)
        is_scoped = self.sample_scope_bar.get_scope_mode() != SCOPE_ALL
        self._apply_current_scope(
            log_change=bool(is_scoped and self._full_df is not None and not self._full_df.empty),
            markers_updated=(markers is not None),
        )

    def set_group_map(self, group_map: Optional[Dict[str, str]]) -> None:
        """Assign a shared sample -> cohort group mapping across all 4 sub-panels."""
        self.group_map = {
            str(k): str(v) for k, v in (group_map or {}).items() if str(v).strip()
        }
        for child in self._child_widgets():
            if hasattr(child, "set_group_map"):
                child.set_group_map(self.group_map)
        self._update_summary_header()

    def open_popup_window(self) -> Any:
        """Open the Figure Studio pop-up dialog for the active analytics sub-panel."""
        cur = self.stack.currentWidget()
        if cur is not None and hasattr(cur, "open_popup_window"):
            return cur.open_popup_window()
        return None

    def _update_summary_header(self) -> None:
        mode = self.sample_scope_bar.get_scope_mode()
        mode_short = {
            SCOPE_ALL: "All",
            SCOPE_ACTIVE: "Active",
            SCOPE_SELECTED: "Selected",
        }.get(mode, mode)
        resolved = self.sample_scope_bar.resolve_samples()
        n_total = len(self._all_samples)
        n_cells = len(self._scoped_df) if self._scoped_df is not None else 0

        if self._scoped_df is not None and not self._scoped_df.empty:
            s_col = find_sample_column(self._scoped_df)
            n_df_samples = (
                int(self._scoped_df[s_col].nunique())
                if s_col and s_col in self._scoped_df.columns
                else 1
            )
            n_scoped = len(resolved) if resolved else n_df_samples
            n_denom = max(n_total, n_scoped)
            scope_txt = f"Scope: {mode_short} ({n_scoped}/{n_denom} samples) | {n_cells:,} cells"
        else:
            n_scoped = len(resolved)
            scope_txt = (
                f"Scope: {mode_short} ({n_scoped}/{n_total} samples) | No cells loaded"
                if n_total > 0
                else f"Scope: {mode_short} (0/0 samples) | No cells loaded"
            )

        effective_groups = self.group_map
        if not effective_groups:
            for child in self._child_widgets():
                gmap = getattr(child, "group_map", None)
                if gmap:
                    effective_groups = gmap
                    break

        if effective_groups:
            n_grp = len(set(effective_groups.values()))
            cohort_txt = f"{len(effective_groups)} samples / {n_grp} groups"
        else:
            cohort_txt = "Ungrouped"

        self.lbl_summary.setText(
            f"{scope_txt} | Type: {self.celltype_col} | "
            f"Scale: {self.scale_factor:.4g} \u00b5m/px | Cohorts: {cohort_txt}"
        )

    def closeEvent(self, event: Any) -> None:
        """Ensure any open Figure Studio popups and child figures close cleanly."""
        for child in self._child_widgets():
            popup = getattr(child, "popup_dialog", None)
            if popup is not None:
                try:
                    popup.close()
                except Exception:
                    pass
            try:
                child.close()
            except Exception:
                pass
        super().closeEvent(event)


__all__ = ["IrisAnalyzerPanelWidget"]
