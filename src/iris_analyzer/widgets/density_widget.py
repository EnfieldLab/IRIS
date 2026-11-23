"""
IRIS Cell Density Dock Widget & Figure Studio Dialog
====================================================
Interactive spatial ROI/hole definition (napari Shapes integration, concave hull,
and intensity-based tissue/lumen masking), sample cohort assignment, calibrated
cell density computation, and figure/workbook export.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.axes
import numpy as np
import openpyxl
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

try:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    try:
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    except ImportError:
        NavigationToolbar = None

from matplotlib.figure import Figure
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
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

from iris_analyzer.density import (
    DensityResult,
    ROISpec,
    build_shapely_region,
    compute_cell_density,
    generate_concave_hull_polygon,
    generate_intensity_tissue_polygons,
)
from iris_common.logging import iris_log
from iris_common.schema import COL_SAMPLE, ensure_canonical_dataframe, find_sample_column
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
from iris_common.widgets.sample_scope import SCOPE_ACTIVE, SCOPE_ALL, SCOPE_SELECTED
from iris_common.widgets.theme import (
    BUTTON_CTA,
    POPOUT_BUTTON_STYLE,
    SHARED_DARK_THEME,
    configure_tab_widget,
    create_section_card,
    style_section_header,
)

_SAMPLE_EXT_RE = re.compile(r"(?:\.ome)?\.tiff?$", re.IGNORECASE)
_SAMPLE_ROLE_RE = re.compile(r"(?:_image|_mask)$", re.IGNORECASE)


def normalize_sample_key(value: object) -> str:
    """Strip whitespace, TIFF extensions, and ``_image``/``_mask`` suffixes."""
    if value is None:
        return ""
    text = str(value).strip()
    text = _SAMPLE_EXT_RE.sub("", text)
    text = _SAMPLE_ROLE_RE.sub("", text)
    return text


def resolve_scoped_samples(
    df: Optional[pd.DataFrame],
    scoped_samples: Sequence[str] = (),
    ctx: Any = None,
) -> List[str]:
    """Resolve active sample IDs from scoped DataFrame, scope bar selection, or context."""
    if df is not None and not df.empty:
        s_col = find_sample_column(df) or COL_SAMPLE
        if s_col in df.columns:
            vals = [
                str(v).strip()
                for v in df[s_col].dropna().unique().tolist()
                if str(v).strip()
            ]
            if vals:
                return vals
    if df is not None and df.empty:
        return [str(s).strip() for s in scoped_samples if str(s).strip()]
    if scoped_samples:
        return [str(s).strip() for s in scoped_samples if str(s).strip()]

    state = getattr(ctx, "state", None) if ctx is not None else None
    if state is not None:
        layers_map = getattr(state, "sample_layers_map", None)
        if isinstance(layers_map, dict) and len(layers_map) > 1:
            return [str(k).strip() for k in layers_map.keys() if str(k).strip()]
        cur = getattr(state, "current_sample_name", None)
        if cur and str(cur).strip():
            return [str(cur).strip()]
        if isinstance(layers_map, dict) and len(layers_map) == 1:
            return [str(k).strip() for k in layers_map.keys() if str(k).strip()]
    return []


def filter_rois_for_samples(
    rois: Sequence[ROISpec],
    active_samples: Sequence[str],
) -> List[ROISpec]:
    """Filter *rois* to entries matching *active_samples* or generic ``sample=None``."""
    if not rois:
        return []
    if not active_samples:
        return list(rois)
    exact = {str(s).strip() for s in active_samples if str(s).strip()}
    norm = {normalize_sample_key(s) for s in active_samples if normalize_sample_key(s)}
    filtered: List[ROISpec] = []
    for roi in rois:
        if roi.sample is None or not str(roi.sample).strip():
            filtered.append(roi)
            continue
        raw = str(roi.sample).strip()
        if raw in exact or normalize_sample_key(raw) in norm:
            filtered.append(roi)
    return filtered

STAT_TEST_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("Mann-Whitney U", "mann_whitney"),
    ("Welch t-test", "welch_t"),
    ("Student t-test", "student_t"),
    ("Kruskal-Wallis", "kruskal"),
    ("One-way ANOVA", "anova_oneway"),
)

DENSITY_UNIT_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("cells/mm²", "mm2"),
    ("cells/10^4 µm²", "10^4 um2"),
    ("cells/µm²", "um2"),
)


def stat_test_code_from_label(label: str) -> str:
    """Translate UI statistical test display name into engine test identifier."""
    mapping = dict(STAT_TEST_CHOICES)
    return mapping.get(label, "mann_whitney")


def extract_viewer_roi_polygons(
    viewer: Any,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Safely extract inclusion and exclusion polygons from viewer Shapes ROI layers."""
    incl_polys: List[np.ndarray] = []
    excl_polys: List[np.ndarray] = []
    if viewer is None:
        return incl_polys, excl_polys

    for lyr in getattr(viewer, "layers", []):
        lname = getattr(lyr, "name", "")
        if lname not in ("Density Inclusion ROIs", "Density Exclusion Holes"):
            continue
        data = getattr(lyr, "data", None)
        if data is None or len(data) == 0:
            continue
        polys = [
            np.asarray(p, dtype=float)[:, :2]
            for p in data
            if p is not None and np.asarray(p).ndim == 2 and len(p) >= 3
        ]
        if lname == "Density Inclusion ROIs":
            incl_polys = polys
        else:
            excl_polys = polys
    return incl_polys, excl_polys


def density_unit_code_from_label(label: str) -> str:
    """Translate UI density unit display label into engine unit identifier."""
    mapping = dict(DENSITY_UNIT_CHOICES)
    return mapping.get(label, "mm2")


def populate_table_widget(
    table: QTableWidget,
    df: Optional[pd.DataFrame],
    empty_note: str = "No data available.",
) -> None:
    """Populate a ``QTableWidget`` from a DataFrame with clean numeric formatting."""
    table.clear()
    if df is None or df.empty:
        table.setRowCount(1)
        table.setColumnCount(1)
        table.setHorizontalHeaderLabels(["Status"])
        table.setItem(0, 0, QTableWidgetItem(empty_note))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        return

    work = df.copy()
    if not isinstance(work.index, pd.RangeIndex) or work.index.name is not None:
        idx_name = str(work.index.name) if work.index.name is not None else "Index"
        if idx_name not in work.columns:
            work = work.reset_index().rename(columns={"index": idx_name})

    table.setRowCount(len(work))
    table.setColumnCount(len(work.columns))
    table.setHorizontalHeaderLabels([str(c) for c in work.columns])

    for r_idx, row_vals in enumerate(work.itertuples(index=False, name=None)):
        for c_idx, val in enumerate(row_vals):
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                text = ""
            elif isinstance(val, (bool, np.bool_)):
                text = str(bool(val))
            elif isinstance(val, (int, np.integer)):
                text = f"{int(val)}"
            elif isinstance(val, (float, np.floating)):
                text = f"{float(val):.4g}"
            else:
                text = str(val)
            table.setItem(r_idx, c_idx, QTableWidgetItem(text))

    header = table.horizontalHeader()
    mode = QHeaderView.Stretch if len(work.columns) <= 6 else QHeaderView.Interactive
    header.setSectionResizeMode(mode)


def append_excel_sheets(
    xlsx_path: Path,
    extra_sheets: Dict[str, Optional[pd.DataFrame]],
) -> Path:
    """Append additional formatted DataFrame worksheets to a Source Data ``.xlsx`` file."""
    valid_sheets = {
        name: df for name, df in extra_sheets.items() if df is not None and not df.empty
    }
    if not valid_sheets:
        return xlsx_path

    wb = openpyxl.load_workbook(xlsx_path)
    header_fill = PatternFill(start_color="E8EEF5", end_color="E8EEF5", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="111111")
    body_font = Font(name="Calibri", size=10, color="111111")
    header_border = Border(bottom=Side(style="thin", color="444444"))

    for sheet_name, df in valid_sheets.items():
        clean_name = sheet_name[:31]
        if clean_name in wb.sheetnames:
            del wb[clean_name]
        ws = wb.create_sheet(clean_name)

        clean_df = df.copy()
        if not isinstance(clean_df.index, pd.RangeIndex) or clean_df.index.name is not None:
            idx_name = str(clean_df.index.name) if clean_df.index.name is not None else "Category"
            if idx_name not in clean_df.columns:
                clean_df = clean_df.reset_index().rename(columns={"index": idx_name})

        headers = [str(c) for c in clean_df.columns]
        ws.append(headers)
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.border = header_border
            cell.alignment = Alignment(horizontal="left", vertical="center")

        for row_vals in clean_df.itertuples(index=False, name=None):
            sanitized = []
            for v in row_vals:
                if v is None:
                    sanitized.append("")
                elif isinstance(v, (float, np.floating)):
                    sanitized.append("" if not np.isfinite(float(v)) else float(v))
                elif isinstance(v, (bool, np.bool_)):
                    sanitized.append(bool(v))
                elif isinstance(v, (int, np.integer)):
                    sanitized.append(int(v))
                else:
                    sanitized.append(str(v))
            ws.append(sanitized)

        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=len(headers)):
            for cell in row:
                cell.font = body_font

        ws.freeze_panes = "A2"
        for col_idx, col_name in enumerate(headers, start=1):
            col_letter = get_column_letter(col_idx)
            max_len = len(col_name)
            for r_idx in range(2, min(ws.max_row, 200) + 1):
                val = ws.cell(row=r_idx, column=col_idx).value
                if val is not None:
                    max_len = max(max_len, len(str(val)))
            ws.column_dimensions[col_letter].width = float(min(max(max_len + 4, 12), 48))

    wb.save(str(xlsx_path))
    return xlsx_path


class SampleGroupAssignmentDialog(QDialog):
    """Interactive dialog for assigning samples/FOVs to experimental cohort groups."""

    def __init__(
        self,
        samples: Sequence[str],
        group_map: Optional[Dict[str, str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Assign Sample Cohort Groups")
        self.resize(520, 420)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setStyleSheet(STUDIO_DARK_STYLESHEET)

        self._samples = list(dict.fromkeys(str(s) for s in samples))
        self._group_map = dict(group_map or {})
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        hint = QLabel(
            "Assign each sample to an experimental group (e.g. Control, Treated) "
            "for cohort-level statistical comparison."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #BBBBBB;")
        layout.addWidget(hint)

        quick_row = QHBoxLayout()
        self.edit_prefix_sep = QLineEdit("_")
        self.edit_prefix_sep.setFixedWidth(50)
        self.edit_prefix_sep.setToolTip("Delimiter used to infer group name from sample prefix")
        btn_infer = QPushButton("Infer from Prefix")
        btn_infer.clicked.connect(self._infer_from_prefix)
        btn_clear = QPushButton("Clear Groups")
        btn_clear.clicked.connect(self._clear_groups)
        quick_row.addWidget(QLabel("Delimiter:"))
        quick_row.addWidget(self.edit_prefix_sep)
        quick_row.addWidget(btn_infer)
        quick_row.addWidget(btn_clear)
        quick_row.addStretch(1)
        layout.addLayout(quick_row)

        self.table = QTableWidget(len(self._samples), 2)
        self.table.setHorizontalHeaderLabels(["Sample", "Cohort Group"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for r_idx, sample in enumerate(self._samples):
            s_item = QTableWidgetItem(sample)
            s_item.setFlags(s_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r_idx, 0, s_item)
            g_item = QTableWidgetItem(self._group_map.get(sample, ""))
            self.table.setItem(r_idx, 1, g_item)

        layout.addWidget(self.table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _infer_from_prefix(self) -> None:
        sep = self.edit_prefix_sep.text() or "_"
        for r_idx, sample in enumerate(self._samples):
            grp = sample.split(sep)[0] if sep in sample else sample
            self.table.item(r_idx, 1).setText(grp)

    def _clear_groups(self) -> None:
        for r_idx in range(self.table.rowCount()):
            self.table.item(r_idx, 1).setText("")

    def get_group_map(self) -> Dict[str, str]:
        """Return ``{sample: group}`` for all non-empty group assignments."""
        out: Dict[str, str] = {}
        for r_idx in range(self.table.rowCount()):
            sample = self.table.item(r_idx, 0).text().strip()
            grp = self.table.item(r_idx, 1).text().strip()
            if sample and grp:
                out[sample] = grp
        return out


class TabbedPublicationStudioDialog(QDialog):
    """Reusable multi-tab Figure Studio window with live ``PublicationStyleSidebar``."""

    def __init__(
        self,
        title: str,
        *,
        categories: Optional[Sequence[str]] = None,
        config: Optional[PlotStyleConfig] = None,
        log_submodule: str = "Figure Studio",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1320, 840)
        self.setMinimumSize(940, 580)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setStyleSheet(STUDIO_DARK_STYLESHEET)

        self._log_submodule: str = log_submodule
        self.config: PlotStyleConfig = config.copy() if config is not None else PlotStyleConfig()
        self._categories: List[str] = list(categories or [])
        self._plot_tabs: Dict[int, Dict[str, Any]] = {}
        self._summary_df: pd.DataFrame = pd.DataFrame()
        self._replicates_df: Optional[pd.DataFrame] = None
        self._stats_df: Optional[pd.DataFrame] = None
        self._metadata: Dict[str, Any] = {}
        self._extra_excel_sheets: Dict[str, Optional[pd.DataFrame]] = {}

        self._build_shell()

    def _build_shell(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        self.splitter = QSplitter(Qt.Horizontal)
        self.tabs = QTabWidget()
        configure_tab_widget(self.tabs)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.splitter.addWidget(self.tabs)

        self.sidebar = PublicationStyleSidebar(
            config=self.config,
            categories=self._categories,
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
        root.addWidget(self.splitter, 1)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #9AA4B2; font-size: 11px; padding: 2px 4px;")
        root.addWidget(self.lbl_status)

    def add_plot_tab(
        self,
        title: str,
        render_fn: Callable[[Figure, PlotStyleConfig], Optional[Sequence[matplotlib.axes.Axes]]],
        header_widget: Optional[QWidget] = None,
    ) -> int:
        """Create a Matplotlib plot tab managed by the Figure Style Sidebar."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        if header_widget is not None:
            layout.addWidget(header_widget)

        fig = Figure(figsize=self.config.size_inches, dpi=100)
        preset = THEME_PRESETS.get(self.config.theme, THEME_PRESETS["IRIS Dark"])
        fig.patch.set_facecolor(preset["figure_facecolor"])

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        if NavigationToolbar is not None:
            toolbar = NavigationToolbar(canvas, tab)
            toolbar.setStyleSheet(
                "QToolBar { background-color: #1f232b; border: 1px solid #3c4352; "
                "spacing: 4px; padding: 2px; } "
                "QToolButton { background-color: transparent; border: 1px solid transparent; "
                "border-radius: 3px; padding: 3px; } "
                "QToolButton:hover { background-color: #3A6A9A; border: 1px solid #4B82B5; }"
            )
            layout.addWidget(toolbar)

        layout.addWidget(canvas, 1)
        idx = self.tabs.addTab(tab, title)
        self._plot_tabs[idx] = {
            "figure": fig,
            "canvas": canvas,
            "render_fn": render_fn,
        }
        return idx

    def add_table_tab(self, title: str, widget: QWidget) -> int:
        """Append a table/inspection tab to the studio window."""
        return self.tabs.addTab(widget, title)

    def set_export_payload(
        self,
        *,
        summary_df: pd.DataFrame,
        replicates_df: Optional[pd.DataFrame] = None,
        stats_df: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extra_sheets: Optional[Dict[str, Optional[pd.DataFrame]]] = None,
        categories: Optional[Sequence[str]] = None,
    ) -> None:
        """Update Source Data workbook tables and sidebar categories."""
        self._summary_df = summary_df
        self._replicates_df = replicates_df
        self._stats_df = stats_df
        self._metadata = dict(metadata or {})
        self._extra_excel_sheets = dict(extra_sheets or {})
        if categories is not None:
            self._categories = list(dict.fromkeys(str(c) for c in categories))
            self.sidebar.set_categories(self._categories)

    def active_figure(self) -> Figure:
        """Return the active tab's Matplotlib ``Figure`` (or first plot tab's figure)."""
        idx = self.tabs.currentIndex()
        if idx in self._plot_tabs:
            return self._plot_tabs[idx]["figure"]
        if self._plot_tabs:
            first_idx = min(self._plot_tabs.keys())
            return self._plot_tabs[first_idx]["figure"]
        return Figure()

    def _render_tab_index(self, idx: int) -> None:
        info = self._plot_tabs.get(idx)
        if info is None:
            return
        fig: Figure = info["figure"]
        canvas: FigureCanvas = info["canvas"]
        render_fn = info["render_fn"]

        fig.clear()
        styled_axes = render_fn(fig, self.config)
        apply_plot_style(
            fig,
            styled_axes if styled_axes is not None else fig.axes,
            self.config,
        )
        if fig.axes and not fig.get_constrained_layout():
            try:
                fig.tight_layout()
            except Exception:
                pass
        canvas.draw_idle()

    def render_all_tabs(self) -> None:
        """Re-render all figure tabs with the current ``PlotStyleConfig``."""
        for idx in sorted(self._plot_tabs.keys()):
            self._render_tab_index(idx)
        self.lbl_status.setText(
            f"{self.config.width_mm:.1f} x {self.config.height_mm:.1f} mm | "
            f"{self.config.dpi} DPI | Theme: {self.config.theme} | "
            f"Palette: {self.config.palette_name}"
        )

    def _on_tab_changed(self, idx: int) -> None:
        if idx in self._plot_tabs:
            self._render_tab_index(idx)

    def _on_style_changed(self, new_config: PlotStyleConfig) -> None:
        self.config = new_config.copy()
        self.render_all_tabs()

    def copy_to_clipboard(self) -> bool:
        """Copy the active styled figure to the system clipboard."""
        ok = copy_figure_to_clipboard(
            self.active_figure(),
            config=self.config,
            dpi=self.config.dpi,
        )
        if ok:
            self.lbl_status.setText(f"Copied figure to clipboard at {self.config.dpi} DPI.")
        else:
            self.lbl_status.setText("Clipboard unavailable in headless environment.")
        return ok

    def export_figure_to_path(self, filepath: Union[str, Path]) -> Path:
        """Export the active styled figure directly to *filepath*."""
        out = export_publication_figure(self.active_figure(), filepath, self.config)
        self.lbl_status.setText(f"Exported figure to {out.name}.")
        iris_log("IRIS-Analyzer", self._log_submodule, f"Exported figure to {out}.", flush=True)
        return out

    def export_figure(self) -> Optional[Path]:
        """Prompt user for a path and export the active styled figure."""
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
        """Export multi-sheet Source Data workbook directly to *filepath*."""
        meta = dict(self._metadata)
        meta.update({f"style.{k}": v for k, v in self.config.to_dict().items()})
        out = export_source_data_excel(
            filepath,
            summary_df=self._summary_df,
            replicates_df=self._replicates_df,
            stats_df=self._stats_df,
            metadata=meta,
            include_empty_sheets=True,
        )
        append_excel_sheets(out, self._extra_excel_sheets)
        self.lbl_status.setText(f"Exported Source Data workbook to {out.name}.")
        iris_log(
            "IRIS-Analyzer",
            self._log_submodule,
            f"Exported source data workbook to {out}.",
            flush=True,
        )
        return out

    def export_source_data(self) -> Optional[Path]:
        """Prompt user for a path and export Source Data Excel workbook."""
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
                self,
                "Export Complete",
                f"Source Data workbook saved to {out.name}.",
            )
            return out
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Could not export Source Data: {exc}")
            return None

    def closeEvent(self, event: Any) -> None:
        """Clear matplotlib figures cleanly on close."""
        for info in self._plot_tabs.values():
            info["figure"].clear()
        super().closeEvent(event)


def _draw_grouped_bar_with_replicates(
    ax: matplotlib.axes.Axes,
    rep_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    *,
    value_col: str,
    ylabel: str,
    title: str,
    config: PlotStyleConfig,
    seed: int = 42,
) -> None:
    """Plot cohort bar chart (mean +- SEM) with sample replicate dots and stat brackets."""
    if rep_df.empty:
        ax.text(0.5, 0.5, "No density measurements available.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    sample_df = (
        rep_df.groupby(["Group", COL_SAMPLE], observed=True, as_index=False)[value_col]
        .mean()
    )
    groups = list(dict.fromkeys(sample_df["Group"].astype(str).tolist()))
    colors = resolve_category_colors(groups, config)
    preset = THEME_PRESETS.get(config.theme, THEME_PRESETS["IRIS Dark"])
    edge_col = "#FFFFFF" if preset["is_dark"] else "#000000"

    x_positions: Dict[str, float] = {}
    y_maxes: Dict[str, float] = {}
    rng = np.random.default_rng(seed)

    for idx, grp in enumerate(groups):
        vals = sample_df.loc[sample_df["Group"].astype(str) == grp, value_col].to_numpy(dtype=float)
        mean_v = float(np.mean(vals)) if vals.size else 0.0
        sem_v = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        x_positions[grp] = float(idx)

        ax.bar(
            [idx],
            [mean_v],
            width=0.58,
            yerr=[sem_v],
            capsize=4,
            color=colors[grp],
            edgecolor=edge_col,
            linewidth=0.8,
            alpha=0.85,
            label=grp,
            error_kw={"ecolor": edge_col, "elinewidth": 1.0},
            zorder=2,
        )

        if vals.size:
            jitter = rng.uniform(-0.14, 0.14, size=vals.size) if vals.size > 1 else np.zeros(1)
            ax.scatter(
                np.full(vals.size, float(idx)) + jitter,
                vals,
                color=colors[grp],
                edgecolor=edge_col,
                linewidth=0.9,
                s=34,
                zorder=4,
            )
            y_maxes[grp] = float(max(np.max(vals), mean_v + sem_v))
        else:
            y_maxes[grp] = mean_v

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    max_val = max(y_maxes.values()) if y_maxes else 1.0
    ax.set_ylim(0.0, max(max_val * 1.18, 1e-6))

    if stats_df is not None and not stats_df.empty:
        valid_pairs = stats_df[
            stats_df["group1"].astype(str).isin(x_positions)
            & stats_df["group2"].astype(str).isin(x_positions)
        ]
        if not valid_pairs.empty:
            draw_significance_brackets(ax, valid_pairs, x_positions, y_maxes, config=config)


class CellDensityStudioDialog(TabbedPublicationStudioDialog):
    """Multi-tab Figure Studio window for calibrated cell density analysis."""

    def __init__(
        self,
        result: DensityResult,
        *,
        config: Optional[PlotStyleConfig] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        categories = (
            list(dict.fromkeys(result.replicates_df["Group"].astype(str).tolist()))
            if not result.replicates_df.empty and "Group" in result.replicates_df.columns
            else []
        )
        super().__init__(
            "Cell Density \u2014 Figure Studio",
            categories=categories,
            config=config,
            log_submodule="Cell Density",
            parent=parent,
        )
        self.result: DensityResult = result

        # Tab 0: Grouped Bar + Replicate Overlay + Significance Brackets
        tab0_header = QWidget()
        h0_layout = QHBoxLayout(tab0_header)
        h0_layout.setContentsMargins(0, 0, 0, 0)
        h0_layout.addWidget(QLabel("Cell Population:"))
        self.combo_focus_type = QComboBox()
        self.combo_focus_type.currentTextChanged.connect(lambda _: self._render_tab_index(0))
        h0_layout.addWidget(self.combo_focus_type)
        h0_layout.addStretch(1)
        self.add_plot_tab(
            "Grouped Bar + Replicate Overlay + Significance Brackets",
            self._plot_grouped_bar,
            header_widget=tab0_header,
        )

        # Tab 1: Cell-Type Density Comparison
        self.add_plot_tab(
            "Cell-Type Density Comparison",
            self._plot_celltype_comparison,
        )

        # Tab 2: ROI Area & Summary Table
        tab2_widget = QWidget()
        t2_layout = QVBoxLayout(tab2_widget)
        t2_layout.setContentsMargins(6, 6, 6, 6)
        t2_layout.setSpacing(6)
        t2_layout.addWidget(QLabel("<b>ROI Areas & Cell Counts</b>"))
        self.table_roi_areas = QTableWidget(0, 0)
        self.table_roi_areas.setAlternatingRowColors(True)
        self.table_roi_areas.setEditTriggers(QTableWidget.NoEditTriggers)
        t2_layout.addWidget(self.table_roi_areas, 1)
        t2_layout.addWidget(QLabel("<b>Cohort Density Summary</b>"))
        self.table_summary = QTableWidget(0, 0)
        self.table_summary.setAlternatingRowColors(True)
        self.table_summary.setEditTriggers(QTableWidget.NoEditTriggers)
        t2_layout.addWidget(self.table_summary, 1)
        self.add_table_tab("ROI Area & Summary Table", tab2_widget)

        # Tab 3: Statistical Tests Table
        tab3_widget = QWidget()
        t3_layout = QVBoxLayout(tab3_widget)
        t3_layout.setContentsMargins(6, 6, 6, 6)
        self.table_stats = QTableWidget(0, 0)
        self.table_stats.setAlternatingRowColors(True)
        self.table_stats.setEditTriggers(QTableWidget.NoEditTriggers)
        t3_layout.addWidget(self.table_stats)
        self.add_table_tab("Statistical Tests", tab3_widget)

        self.update_data(result)

    def update_data(self, result: DensityResult) -> None:
        """Refresh studio plots, tables, and Source Data export payload."""
        self.result = result

        prev_focus = self.combo_focus_type.currentText()
        self.combo_focus_type.blockSignals(True)
        self.combo_focus_type.clear()
        items = ["All Cells"] + list(result.cell_types)
        self.combo_focus_type.addItems(items)
        if prev_focus in items:
            self.combo_focus_type.setCurrentText(prev_focus)
        self.combo_focus_type.blockSignals(False)

        populate_table_widget(
            self.table_roi_areas,
            result.roi_areas_df,
            empty_note="No ROI areas recorded.",
        )
        populate_table_widget(
            self.table_summary,
            result.summary_df,
            empty_note="No summary rows recorded.",
        )
        populate_table_widget(
            self.table_stats,
            result.stats_df,
            empty_note="Assign >= 2 sample groups to compute cohort statistics.",
        )

        groups = (
            list(dict.fromkeys(result.replicates_df["Group"].astype(str).tolist()))
            if not result.replicates_df.empty and "Group" in result.replicates_df.columns
            else []
        )
        self.set_export_payload(
            summary_df=result.summary_df,
            replicates_df=result.replicates_df,
            stats_df=result.stats_df,
            metadata=result.metadata,
            extra_sheets={"ROI_Areas": result.roi_areas_df},
            categories=groups,
        )
        self.render_all_tabs()

    def _plot_grouped_bar(self, fig: Figure, cfg: PlotStyleConfig) -> List[matplotlib.axes.Axes]:
        ax = fig.add_subplot(111)
        focus = self.combo_focus_type.currentText() or "All Cells"
        rep = self.result.replicates_df
        sub_rep = rep[rep["cell_type"] == focus] if not rep.empty else rep
        sub_stats = (
            self.result.stats_df[self.result.stats_df["cell_type"] == focus]
            if not self.result.stats_df.empty
            else pd.DataFrame()
        )
        _draw_grouped_bar_with_replicates(
            ax,
            sub_rep,
            sub_stats,
            value_col="Density",
            ylabel=f"Cell Density ({self.result.unit_label})",
            title=f"{focus} Spatial Density",
            config=cfg,
        )
        return [ax]

    def _plot_celltype_comparison(
        self,
        fig: Figure,
        cfg: PlotStyleConfig,
    ) -> List[matplotlib.axes.Axes]:
        ax = fig.add_subplot(111)
        rep = self.result.replicates_df
        if rep.empty or not self.result.cell_types:
            ax.text(0.5, 0.5, "No cell-type density data available.", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            return [ax]

        sub = rep[rep["cell_type"].isin(self.result.cell_types)]
        sample_df = (
            sub.groupby(["Group", COL_SAMPLE, "cell_type"], observed=True, as_index=False)[
                "Density"
            ]
            .mean()
        )
        groups = list(dict.fromkeys(sample_df["Group"].astype(str).tolist()))
        ctypes = list(self.result.cell_types)
        colors = resolve_category_colors(groups, cfg)
        preset = THEME_PRESETS.get(cfg.theme, THEME_PRESETS["IRIS Dark"])
        edge_col = "#FFFFFF" if preset["is_dark"] else "#000000"

        n_groups = max(len(groups), 1)
        bar_w = min(0.75 / n_groups, 0.36)
        x_base = np.arange(len(ctypes), dtype=float)
        rng = np.random.default_rng(17)
        cell_max_y = np.zeros(len(ctypes), dtype=float)

        for g_idx, grp in enumerate(groups):
            offset = (g_idx - (n_groups - 1) / 2.0) * bar_w
            means = []
            sems = []
            xs = x_base + offset
            for c_idx, ct in enumerate(ctypes):
                vals = sample_df.loc[
                    (sample_df["Group"].astype(str) == grp) & (sample_df["cell_type"] == ct),
                    "Density",
                ].to_numpy(dtype=float)
                m = float(np.mean(vals)) if vals.size else 0.0
                s = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
                means.append(m)
                sems.append(s)
                if vals.size:
                    jitter = rng.uniform(-bar_w * 0.22, bar_w * 0.22, size=vals.size)
                    ax.scatter(
                        np.full(vals.size, xs[c_idx]) + jitter,
                        vals,
                        color=colors[grp],
                        edgecolor=edge_col,
                        linewidth=0.7,
                        s=20,
                        zorder=4,
                    )
                    cell_max_y[c_idx] = max(cell_max_y[c_idx], float(np.max(vals)), m + s)

            ax.bar(
                xs,
                means,
                width=bar_w * 0.92,
                yerr=sems,
                capsize=2.5,
                color=colors[grp],
                edgecolor=edge_col,
                linewidth=0.6,
                alpha=0.85,
                label=grp,
                error_kw={"ecolor": edge_col, "elinewidth": 0.9},
                zorder=2,
            )

        y_top = float(np.max(cell_max_y)) if cell_max_y.size else 1.0
        if not self.result.stats_df.empty:
            for c_idx, ct in enumerate(ctypes):
                ct_stats = self.result.stats_df[self.result.stats_df["cell_type"] == ct]
                if ct_stats.empty:
                    continue
                sig_rows = ct_stats[ct_stats["significant"].astype(bool)]
                if sig_rows.empty:
                    continue
                best_row = sig_rows.sort_values("q_value").iloc[0]
                lbl = str(best_row["significance_label"])
                if lbl and lbl != "ns":
                    ax.text(
                        x_base[c_idx],
                        cell_max_y[c_idx] + max(y_top * 0.04, 1e-6),
                        lbl,
                        ha="center",
                        va="bottom",
                        color=preset["bracket_color"],
                        fontsize=float(cfg.stat_size),
                        fontweight="bold",
                    )

        ax.set_xticks(x_base)
        ax.set_xticklabels(ctypes, rotation=30 if len(ctypes) > 4 else 0, ha="right")
        ax.set_ylabel(f"Cell Density ({self.result.unit_label})")
        ax.set_title("Cell-Type Density Comparison Across Cohorts")
        ax.set_ylim(0.0, max(y_top * 1.18, 1e-6))
        if len(groups) > 1 or groups != ["All"]:
            ax.legend(loc="best", frameon=True)
        return [ax]


class CellDensityWidget(QWidget):
    """Dockable widget for ROI/tissue masking and calibrated cell density analysis."""

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
        self.last_result: Optional[DensityResult] = None
        self.popup_dialog: Optional[CellDensityStudioDialog] = None
        self._bound_layers: set[int] = set()
        self._syncing_viewer: bool = False
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

        # ROI / Mask Generation Box
        roi_box, roi_layout = create_section_card("Mask & ROI Generation")

        draw_row = QHBoxLayout()
        draw_row.setSpacing(8)
        self.btn_draw_inclusion = QPushButton("Draw Inclusion Polygon")
        self.btn_draw_inclusion.clicked.connect(self.draw_inclusion_polygon)
        draw_row.addWidget(self.btn_draw_inclusion)

        self.btn_draw_exclusion = QPushButton("Draw Exclusion Hole")
        self.btn_draw_exclusion.clicked.connect(self.draw_exclusion_hole)
        draw_row.addWidget(self.btn_draw_exclusion)
        roi_layout.addLayout(draw_row)

        hull_row = QHBoxLayout()
        hull_row.setSpacing(8)
        self.btn_auto_hull = QPushButton("Auto Concave Hull")
        self.btn_auto_hull.clicked.connect(self.run_auto_concave_hull)
        hull_row.addWidget(self.btn_auto_hull, 1)

        hull_row.addWidget(QLabel("alpha:"))
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.0, 1.0)
        self.spin_alpha.setSingleStep(0.05)
        self.spin_alpha.setDecimals(2)
        self.spin_alpha.setValue(0.30)
        hull_row.addWidget(self.spin_alpha)
        roi_layout.addLayout(hull_row)

        intensity_row = QHBoxLayout()
        intensity_row.setSpacing(8)
        self.btn_auto_tissue = QPushButton("Auto Tissue Mask (Intensity)")
        self.btn_auto_tissue.clicked.connect(self.run_auto_tissue_mask)
        intensity_row.addWidget(self.btn_auto_tissue, 1)

        self.combo_thresh = QComboBox()
        self.combo_thresh.addItems(["otsu", "triangle", "li"])
        intensity_row.addWidget(self.combo_thresh)

        self.spin_channel = QSpinBox()
        self.spin_channel.setRange(0, 64)
        self.spin_channel.setValue(0)
        self.spin_channel.setPrefix("Ch ")
        intensity_row.addWidget(self.spin_channel)
        roi_layout.addLayout(intensity_row)

        json_row = QHBoxLayout()
        json_row.setSpacing(8)
        self.btn_import_json = QPushButton("Import Regions JSON")
        self.btn_import_json.clicked.connect(self._prompt_import_json)
        json_row.addWidget(self.btn_import_json, 1)

        self.btn_clear_rois = QPushButton("Clear ROIs")
        self.btn_clear_rois.clicked.connect(self.clear_rois)
        json_row.addWidget(self.btn_clear_rois)
        roi_layout.addLayout(json_row)

        self.lbl_roi_status = QLabel("Active ROIs: Auto concave hull per sample")
        style_section_header(self.lbl_roi_status, subtitle=True)
        self.lbl_roi_status.setWordWrap(True)
        roi_layout.addWidget(self.lbl_roi_status)
        layout.addWidget(roi_box)

        # Analysis Settings Box
        cfg_box, cfg_layout = create_section_card("Density Parameters & Cohort")
        form = QFormLayout()
        form.setSpacing(6)

        group_row = QHBoxLayout()
        group_row.setSpacing(8)
        self.btn_assign_groups = QPushButton("Assign Cohort Groups...")
        self.btn_assign_groups.clicked.connect(self.open_group_assignment_dialog)
        group_row.addWidget(self.btn_assign_groups)
        self.lbl_groups_status = QLabel("Ungrouped")
        style_section_header(self.lbl_groups_status, subtitle=True)
        group_row.addWidget(self.lbl_groups_status, 1)
        form.addRow("Cohorts:", group_row)

        self.combo_unit = QComboBox()
        self.combo_unit.addItems([label for label, _ in DENSITY_UNIT_CHOICES])
        form.addRow("Density Unit:", self.combo_unit)

        self.combo_stat_test = QComboBox()
        self.combo_stat_test.addItems([label for label, _ in STAT_TEST_CHOICES])
        form.addRow("Statistical Test:", self.combo_stat_test)
        cfg_layout.addLayout(form)

        self.btn_compute = QPushButton("Compute Density")
        self.btn_compute.setStyleSheet(BUTTON_CTA)
        self.btn_compute.clicked.connect(self.compute_density)
        cfg_layout.addWidget(self.btn_compute)
        layout.addWidget(cfg_box)

        for btn in (
            self.btn_draw_inclusion,
            self.btn_draw_exclusion,
            self.btn_auto_hull,
            self.btn_auto_tissue,
            self.btn_import_json,
            self.btn_clear_rois,
            self.btn_assign_groups,
            self.btn_compute,
        ):
            btn.setMinimumHeight(28)

        # Plot Preview Box
        plot_box, pb_layout = create_section_card(margins=(10, 6, 10, 10), spacing=8)

        hdr = QHBoxLayout()
        lbl_plot_title = QLabel("Sample-Level Cell Density")
        style_section_header(lbl_plot_title, subtitle=False)
        hdr.addWidget(lbl_plot_title)
        hdr.addStretch(1)
        self.btn_popout = QPushButton("\u2922 Pop-up Window")
        self.btn_popout.setStyleSheet(POPOUT_BUTTON_STYLE)
        self.btn_popout.setToolTip("Open Cell Density Figure Studio")
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
        self._update_roi_status_label()

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

    def set_data(
        self,
        df: Optional[pd.DataFrame],
        markers: Optional[Sequence[str]] = None,
        celltype_col: str = "cell_type",
        scale_factor: float = 1.0,
        ctx: Any = None,
    ) -> None:
        """Bind single-cell table, calibration scale (µm/px), and optional viewer context."""
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
        self._update_roi_status_label()
        self._render_preview()

    def set_rois(self, rois: Sequence[ROISpec]) -> None:
        """Programmatically set active ``ROISpec`` definitions."""
        self.rois = list(rois)
        if self.ctx is not None:
            self.ctx.density_rois = list(self.rois)
        self._update_roi_status_label()

    def clear_rois(self) -> None:
        """Clear explicit ROIs and revert to auto concave-hull tissue area estimation."""
        self.set_rois([])
        iris_log("IRIS-Analyzer", "Cell Density", "Cleared explicit ROIs.", flush=True)

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

    def _update_roi_status_label(self) -> None:
        if not self.rois:
            self.lbl_roi_status.setText("Active ROIs: Auto concave hull per sample")
            return
        scoped_rois = self._filter_rois_for_scope(self.rois)
        rois_for_area = scoped_rois if scoped_rois else self.rois
        total_um2 = 0.0
        for roi in rois_for_area:
            _, px_area = build_shapely_region(roi.inclusion_polygons, roi.exclusion_polygons)
            sc = roi.microns_per_pixel if roi.microns_per_pixel != 1.0 else self.scale_factor
            total_um2 += px_area * (sc ** 2)
        if len(scoped_rois) != len(self.rois):
            self.lbl_roi_status.setText(
                f"Active ROIs: {len(scoped_rois)}/{len(self.rois)} region(s) in scope, "
                f"total area {total_um2 / 1e6:.4f} mm²"
            )
        else:
            self.lbl_roi_status.setText(
                f"Active ROIs: {len(self.rois)} region(s), total area {total_um2 / 1e6:.4f} mm²"
            )

    def _get_or_create_shapes_layer(
        self,
        layer_name: str,
        edge_color: str,
        face_color: Any,
    ) -> Any:
        if self.ctx is None or getattr(self.ctx, "viewer", None) is None:
            return None
        viewer = self.ctx.viewer
        layer = None
        for lyr in getattr(viewer, "layers", []):
            if getattr(lyr, "name", None) == layer_name:
                layer = lyr
                break
        if layer is None:
            layer = viewer.add_shapes(
                [],
                shape_type="polygon",
                name=layer_name,
                edge_color=edge_color,
                face_color=face_color,
                edge_width=2,
                scale=(self.scale_factor, self.scale_factor),
            )
        self._bind_layer_events(layer)
        return layer

    def _bind_layer_events(self, layer: Any) -> None:
        if layer is None or id(layer) in self._bound_layers:
            return
        events = getattr(layer, "events", None)
        data_ev = getattr(events, "data", None)
        if data_ev is not None and hasattr(data_ev, "connect"):
            data_ev.connect(self._on_viewer_shapes_changed)
            self._bound_layers.add(id(layer))

    def _on_viewer_shapes_changed(self, _event: Any = None) -> None:
        if self._syncing_viewer:
            return
        self._sync_rois_from_viewer()

    def _sync_rois_from_viewer(self) -> None:
        if self.ctx is None or getattr(self.ctx, "viewer", None) is None:
            return
        incl_polys, excl_polys = extract_viewer_roi_polygons(self.ctx.viewer)

        if not incl_polys:
            self.set_rois([])
            return

        active_samples = self._active_scope_samples()
        target_sample: Optional[str] = None
        cur_sample = (
            getattr(self.ctx.state, "current_sample_name", None)
            if getattr(self.ctx, "state", None) is not None
            else None
        )
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
                    name="Drawn ROI",
                    sample=target_sample,
                    inclusion_polygons=incl_polys,
                    exclusion_polygons=excl_polys,
                    microns_per_pixel=self.scale_factor,
                )
            ]
        )

    def draw_inclusion_polygon(self) -> Any:
        """Activate polygon drawing on the ``Density Inclusion ROIs`` napari shapes layer."""
        layer = self._get_or_create_shapes_layer(
            "Density Inclusion ROIs",
            edge_color="#00E5FF",
            face_color=[0.0, 0.9, 1.0, 0.15],
        )
        if layer is None:
            self.lbl_roi_status.setText(
                "No napari viewer connected; use Auto Concave Hull or Import Regions JSON."
            )
            return None
        try:
            layer.mode = "add_polygon"
        except Exception:
            pass
        return layer

    def draw_exclusion_hole(self) -> Any:
        """Activate polygon drawing on the ``Density Exclusion Holes`` napari shapes layer."""
        layer = self._get_or_create_shapes_layer(
            "Density Exclusion Holes",
            edge_color="#FF5252",
            face_color=[1.0, 0.32, 0.32, 0.20],
        )
        if layer is None:
            self.lbl_roi_status.setText(
                "No napari viewer connected; use Auto Tissue Mask or Import Regions JSON."
            )
            return None
        try:
            layer.mode = "add_polygon"
        except Exception:
            pass
        return layer

    def _push_rois_to_viewer(self, rois: Sequence[ROISpec]) -> None:
        if self.ctx is None or getattr(self.ctx, "viewer", None) is None:
            return
        incl_all = [p for r in rois for p in r.inclusion_polygons]
        excl_all = [p for r in rois for p in r.exclusion_polygons]
        self._syncing_viewer = True
        try:
            if incl_all:
                incl_lyr = self._get_or_create_shapes_layer(
                    "Density Inclusion ROIs",
                    edge_color="#00E5FF",
                    face_color=[0.0, 0.9, 1.0, 0.15],
                )
                if incl_lyr is not None:
                    incl_lyr.data = incl_all
            if excl_all:
                excl_lyr = self._get_or_create_shapes_layer(
                    "Density Exclusion Holes",
                    edge_color="#FF5252",
                    face_color=[1.0, 0.32, 0.32, 0.20],
                )
                if excl_lyr is not None:
                    excl_lyr.data = excl_all
            else:
                for lyr in getattr(self.ctx.viewer, "layers", []):
                    if getattr(lyr, "name", "") == "Density Exclusion Holes":
                        lyr.data = []
        finally:
            self._syncing_viewer = False

    def run_auto_concave_hull(self) -> List[ROISpec]:
        """Generate concave-hull inclusion ROIs per sample from cell coordinates."""
        if self.df is None or self.df.empty:
            return []
        s_col = find_sample_column(self.df) or COL_SAMPLE
        alpha = float(self.spin_alpha.value())
        new_rois: List[ROISpec] = []
        if s_col in self.df.columns:
            for sample_name, sub in self.df.groupby(s_col, sort=False):
                hull = generate_concave_hull_polygon(sub, alpha_ratio=alpha)
                if len(hull) >= 3:
                    new_rois.append(
                        ROISpec(
                            name=f"{sample_name} Concave Hull",
                            sample=str(sample_name),
                            inclusion_polygons=[hull],
                            exclusion_polygons=[],
                            microns_per_pixel=self.scale_factor,
                        )
                    )
        else:
            hull = generate_concave_hull_polygon(self.df, alpha_ratio=alpha)
            if len(hull) >= 3:
                new_rois.append(
                    ROISpec(
                        name="Tissue Concave Hull",
                        sample=None,
                        inclusion_polygons=[hull],
                        exclusion_polygons=[],
                        microns_per_pixel=self.scale_factor,
                    )
                )
        self.set_rois(new_rois)
        self._push_rois_to_viewer(new_rois)
        iris_log(
            "IRIS-Analyzer",
            "Cell Density",
            f"Generated {len(new_rois)} concave-hull ROI(s) (alpha={alpha:.2f}).",
            flush=True,
        )
        return new_rois

    def run_auto_tissue_mask(self) -> List[ROISpec]:
        """Generate intensity-based tissue inclusion and lumen hole ROIs."""
        state = getattr(self.ctx, "state", None) if self.ctx is not None else None
        images: Dict[str, np.ndarray] = {}
        if state is not None:
            multi = getattr(state, "multi_sample_images", None) or {}
            if multi:
                images = {str(k): v for k, v in multi.items()}
            elif getattr(state, "raw_image", None) is not None:
                s_name = getattr(state, "current_sample_name", None) or "Sample_1"
                images = {str(s_name): state.raw_image}

        if images and (
            self.df is not None or self._scope_mode in (SCOPE_ACTIVE, SCOPE_SELECTED)
        ):
            if self.df is not None and self.df.empty:
                images = {}
            elif self._scope_mode in (SCOPE_ACTIVE, SCOPE_SELECTED) and not self._scoped_samples:
                images = {}
            else:
                active = self._active_scope_samples()
                if active:
                    exact = set(active)
                    norm = {normalize_sample_key(s) for s in active if normalize_sample_key(s)}
                    images = {
                        k: v
                        for k, v in images.items()
                        if k in exact or normalize_sample_key(k) in norm
                    }

        if not images:
            self.lbl_roi_status.setText(
                "No raw image in state; use Auto Concave Hull or Import Regions JSON."
            )
            return []

        method = self.combo_thresh.currentText()
        ch_idx = int(self.spin_channel.value())
        new_rois: List[ROISpec] = []
        for sample_name, img in images.items():
            incl, excl = generate_intensity_tissue_polygons(
                img,
                channel_idx=ch_idx,
                threshold_method=method,
            )
            if incl:
                new_rois.append(
                    ROISpec(
                        name=f"{sample_name} Tissue Mask",
                        sample=sample_name,
                        inclusion_polygons=incl,
                        exclusion_polygons=excl,
                        microns_per_pixel=self.scale_factor,
                    )
                )
        self.set_rois(new_rois)
        self._push_rois_to_viewer(new_rois)
        iris_log(
            "IRIS-Analyzer",
            "Cell Density",
            f"Generated {len(new_rois)} intensity tissue ROI(s) ({method}, ch={ch_idx}).",
            flush=True,
        )
        return new_rois

    def _prompt_import_json(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Import ROI Polygons JSON",
            "",
            "JSON Files (*.json)",
        )
        if filepath:
            self.import_regions_from_path(filepath)

    def import_regions_from_path(self, filepath: Union[str, Path]) -> List[ROISpec]:
        """Load ROI inclusion/exclusion polygons from a JSON file."""
        with open(Path(filepath), "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        loaded: List[ROISpec] = []
        if isinstance(payload, dict) and "rois" in payload:
            items = payload["rois"]
        elif isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and "vertices" in payload:
            items = [
                {
                    "name": f"{payload.get('sample_name', 'ROI')} Boundary",
                    "sample": payload.get("sample_name"),
                    "inclusion_polygons": [payload["vertices"]],
                    "exclusion_polygons": payload.get("exclusion_polygons", []),
                    "microns_per_pixel": payload.get("scale", self.scale_factor),
                }
            ]
        else:
            items = []

        for idx, item in enumerate(items):
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
        self._push_rois_to_viewer(loaded)
        iris_log(
            "IRIS-Analyzer",
            "Cell Density",
            f"Imported {len(loaded)} ROI(s) from {Path(filepath).name}.",
            flush=True,
        )
        return loaded

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

    def compute_density(self) -> Optional[DensityResult]:
        """Execute calibrated spatial density calculation and update preview/studio."""
        if self.df is None or self.df.empty:
            self._render_preview()
            return None

        unit_code = density_unit_code_from_label(self.combo_unit.currentText())
        test_code = stat_test_code_from_label(self.combo_stat_test.currentText())
        effective_rois = self._filter_rois_for_scope(self.rois)

        image_shapes: Optional[Dict[str, Tuple[int, int]]] = None
        state = getattr(self.ctx, "state", None) if self.ctx is not None else None
        if not effective_rois and state is not None:
            multi = getattr(state, "multi_sample_images", None) or {}
            if multi:
                image_shapes = {str(k): (v.shape[-2], v.shape[-1]) for k, v in multi.items()}
            elif getattr(state, "raw_image", None) is not None:
                s_name = getattr(state, "current_sample_name", None) or "Sample_1"
                raw = state.raw_image
                image_shapes = {str(s_name): (raw.shape[-2], raw.shape[-1])}

        self.last_result = compute_cell_density(
            self.df,
            cell_type_col=self.celltype_col,
            scale_factor=self.scale_factor,
            rois=effective_rois if effective_rois else None,
            image_shapes=image_shapes,
            group_map=self.group_map if self.group_map else None,
            unit=unit_code,
            stat_test=test_code,
        )
        n_rois = int(self.last_result.metadata.get("n_rois", 0))
        n_cells = int(self.last_result.metadata.get("n_cells", 0))
        n_samples = int(self.last_result.metadata.get("n_samples", 0))
        iris_log(
            "IRIS-Analyzer",
            "Cell Density",
            f"Computed density ({self.last_result.unit_label}, test={test_code}): "
            f"{n_rois} ROI(s), {n_samples} sample(s), {n_cells:,} cells.",
            flush=True,
        )
        self._render_preview()
        if self.popup_dialog is not None and self.popup_dialog.isVisible():
            self.popup_dialog.update_data(self.last_result)
        return self.last_result

    def _render_preview(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        cfg = PlotStyleConfig(theme="IRIS Dark")
        if self.last_result is None or self.last_result.replicates_df.empty:
            ax.set_facecolor("#14161d")
            ax.text(
                0.5,
                0.5,
                "Load data and click 'Compute Density' to view densities.",
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

        rep = self.last_result.replicates_df
        has_multi_groups = (
            "Group" in rep.columns
            and rep["Group"].nunique() >= 2
            and not self.last_result.stats_df.empty
        )
        if has_multi_groups:
            sub_rep = rep[rep["cell_type"] == "All Cells"]
            sub_stats = self.last_result.stats_df[
                self.last_result.stats_df["cell_type"] == "All Cells"
            ]
            _draw_grouped_bar_with_replicates(
                ax,
                sub_rep,
                sub_stats,
                value_col="Density",
                ylabel=self.last_result.unit_label,
                title="Total Cell Density by Cohort",
                config=cfg,
            )
        else:
            sub_rep = rep[rep["cell_type"].isin(self.last_result.cell_types)]
            means = (
                sub_rep.groupby("cell_type", observed=True)["Density"]
                .mean()
                .sort_values(ascending=False)
                .head(8)
            )
            colors = resolve_category_colors(list(means.index), cfg)
            ax.bar(
                list(means.index),
                means.to_numpy(dtype=float),
                color=[colors[c] for c in means.index],
                edgecolor="#FFFFFF",
                linewidth=0.7,
            )
            ax.set_xticks(range(len(means)))
            ax.set_xticklabels(list(means.index), rotation=30, ha="right")
            ax.set_ylabel(self.last_result.unit_label)
            ax.set_title("Mean Cell-Type Density")

        apply_plot_style(self.figure, [ax], cfg)
        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

    def open_popup_window(self) -> Optional[CellDensityStudioDialog]:
        """Open or focus the interactive ``CellDensityStudioDialog`` Figure Studio window."""
        if self.last_result is None and self.df is not None and not self.df.empty:
            self.compute_density()
        if self.last_result is None:
            QMessageBox.information(
                self,
                "No Density Data",
                "Load single-cell data and run Compute Density first.",
            )
            return None

        if self.popup_dialog is None:
            self.popup_dialog = CellDensityStudioDialog(
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
    "STAT_TEST_CHOICES",
    "DENSITY_UNIT_CHOICES",
    "normalize_sample_key",
    "resolve_scoped_samples",
    "filter_rois_for_samples",
    "SampleGroupAssignmentDialog",
    "TabbedPublicationStudioDialog",
    "CellDensityStudioDialog",
    "CellDensityWidget",
]
