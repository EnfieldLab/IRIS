"""
IRIS Figure Studio and Source Data Exporter.

Provides reusable publication-grade Matplotlib styling, collision-free
significance bracket rendering, Illustrator-compatible vector/raster export,
high-DPI clipboard rendering, multi-sheet Source Data Excel export, and Qt
widgets for interactive customization across IRIS modules.
"""
from __future__ import annotations

import copy
import io
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.axes
import matplotlib.category as mcategory
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import openpyxl
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QImage
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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

try:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    try:
        from matplotlib.backends.backend_qt5agg import (
            NavigationToolbar2QT as NavigationToolbar,
        )
    except ImportError:
        NavigationToolbar = None

from iris_common.widgets.theme import (
    BUTTON_CTA,
    BUTTON_STUDIO,
    DIALOG_DARK_THEME,
    configure_tab_widget,
    create_section_card,
    style_section_header,
)

logger = logging.getLogger(__name__)

ThemeType = Literal[
    "IRIS Dark",
    "Classic White",
    "Boxed White",
    "Minimal Grid",
]
GridModeType = Literal["None", "Y-Axis", "X-Axis", "Both"]
PDisplayModeType = Literal["stars", "p_value", "q_value"]

_LEGACY_THEME_ALIASES: Dict[str, str] = {
    "Publication White (Nature)": "Classic White",
    "Cell Press Boxed": "Boxed White",
}

_LEGACY_PALETTE_ALIASES: Dict[str, str] = {
    "Nature NPG": "Vibrant",
    "Lancet": "Clinical Blue",
    "JCO": "Clinical Blue",
    "Viridis": "viridis",
}

FONT_FAMILIES: Tuple[str, ...] = (
    "Arial",
    "Helvetica",
    "DejaVu Sans",
    "Times New Roman",
)
DPI_OPTIONS: Tuple[int, ...] = (150, 300, 600, 1200)
GRID_MODES: Tuple[GridModeType, ...] = ("None", "Y-Axis", "X-Axis", "Both")
GRID_LINESTYLES: Tuple[str, ...] = ("--", "-", ":", "-.")
P_DISPLAY_MODES: Tuple[PDisplayModeType, ...] = ("stars", "p_value", "q_value")

SIZE_PRESETS: Dict[str, Optional[Tuple[float, float]]] = {
    "Single Column (89 mm)": (89.0, 70.0),
    "1.5 Column (136 mm)": (136.0, 95.0),
    "Double Column (183 mm)": (183.0, 120.0),
    "Full Page (183 x 247 mm)": (183.0, 247.0),
    "Custom": None,
}


def _sample_cmap_hex(cmap_name: str, n: int = 8) -> List[str]:
    """Sample *n* evenly spaced hex colors from a Matplotlib colormap."""
    cmap = plt.colormaps.get_cmap(cmap_name)
    low, high = (0.08, 0.92) if cmap_name == "viridis" else (0.05, 0.95)
    return [mcolors.to_hex(cmap(pos)) for pos in np.linspace(low, high, n)]


def _discrete_cmap_hex(cmap_name: str) -> List[str]:
    """Extract discrete hex colors from a Matplotlib qualitative colormap."""
    cmap = plt.colormaps.get_cmap(cmap_name)
    colors = getattr(cmap, "colors", None)
    if colors is not None:
        return [mcolors.to_hex(c) for c in colors]
    return _sample_cmap_hex(cmap_name, 8)


SCIENTIFIC_PALETTES: Dict[str, List[str]] = {
    "Okabe-Ito": [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#000000",
    ],
    "Vibrant": [
        "#E64B35",
        "#4DBBD5",
        "#00A087",
        "#3C5488",
        "#F39B7F",
        "#8491B4",
        "#91D1C2",
        "#DC0000",
        "#7E6148",
        "#B09C85",
    ],
    "Clinical Blue": [
        "#00468B",
        "#ED0000",
        "#42B540",
        "#0099B4",
        "#925E9F",
        "#FDAF91",
        "#AD002A",
        "#ADB6B6",
        "#1B1919",
    ],
    "Set2": _discrete_cmap_hex("Set2"),
    "Paired": _discrete_cmap_hex("Paired"),
    "viridis": _sample_cmap_hex("viridis", 8),
    "tab10": _discrete_cmap_hex("tab10"),
}

THEME_PRESETS: Dict[str, Dict[str, Any]] = {
    "IRIS Dark": {
        "figure_facecolor": "#14161d",
        "axes_facecolor": "#14161d",
        "text_color": "#E0E0E0",
        "tick_color": "#D0D4DE",
        "spine_color": "#4A5162",
        "grid_color": "#2E3442",
        "legend_facecolor": "#1f232b",
        "legend_edgecolor": "#4A5162",
        "bracket_color": "#E0E0E0",
        "despine_top": True,
        "despine_right": True,
        "spine_linewidth": 1.0,
        "grid_mode": "Y-Axis",
        "grid_linestyle": "--",
        "grid_alpha": 0.45,
        "font_family": "Arial",
        "is_dark": True,
    },
    "Classic White": {
        "figure_facecolor": "#FFFFFF",
        "axes_facecolor": "#FFFFFF",
        "text_color": "#000000",
        "tick_color": "#000000",
        "spine_color": "#000000",
        "grid_color": "#CCCCCC",
        "legend_facecolor": "#FFFFFF",
        "legend_edgecolor": "none",
        "bracket_color": "#000000",
        "despine_top": True,
        "despine_right": True,
        "spine_linewidth": 0.8,
        "grid_mode": "None",
        "grid_linestyle": ":",
        "grid_alpha": 0.5,
        "font_family": "Arial",
        "is_dark": False,
    },
    "Boxed White": {
        "figure_facecolor": "#FFFFFF",
        "axes_facecolor": "#FFFFFF",
        "text_color": "#111111",
        "tick_color": "#111111",
        "spine_color": "#000000",
        "grid_color": "#DDDDDD",
        "legend_facecolor": "#FFFFFF",
        "legend_edgecolor": "#888888",
        "bracket_color": "#000000",
        "despine_top": False,
        "despine_right": False,
        "spine_linewidth": 1.0,
        "grid_mode": "None",
        "grid_linestyle": "-",
        "grid_alpha": 0.35,
        "font_family": "Helvetica",
        "is_dark": False,
    },
    "Minimal Grid": {
        "figure_facecolor": "#FFFFFF",
        "axes_facecolor": "#FAFAFA",
        "text_color": "#1F2430",
        "tick_color": "#2A303D",
        "spine_color": "#9AA3B2",
        "grid_color": "#E2E6EE",
        "legend_facecolor": "#FFFFFF",
        "legend_edgecolor": "#D4DAE4",
        "bracket_color": "#1F2430",
        "despine_top": True,
        "despine_right": True,
        "spine_linewidth": 0.75,
        "grid_mode": "Both",
        "grid_linestyle": "-",
        "grid_alpha": 0.7,
        "font_family": "DejaVu Sans",
        "is_dark": False,
    },
}

STUDIO_DARK_STYLESHEET = DIALOG_DARK_THEME


@dataclass
class PlotStyleConfig:
    """Configuration controlling typography, palette, axes, dimensions, and export."""

    theme: ThemeType = "IRIS Dark"
    font_family: str = "Arial"
    title_size: float = 12.0
    label_size: float = 11.0
    tick_size: float = 10.0
    legend_size: float = 9.0
    stat_size: float = 9.0
    bold_title: bool = True
    bold_labels: bool = False
    tick_rotation: int = 0
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    legend_title: str = ""
    custom_labels: Dict[str, str] = field(default_factory=dict)
    custom_colors: Dict[str, str] = field(default_factory=dict)
    palette_name: str = "Okabe-Ito"
    despine_top: bool = True
    despine_right: bool = True
    spine_linewidth: float = 1.0
    grid_mode: GridModeType = "Y-Axis"
    grid_linestyle: str = "--"
    grid_alpha: float = 0.35
    width_mm: float = 183.0
    height_mm: float = 120.0
    size_preset: str = "Double Column (183 mm)"
    dpi: int = 300
    transparent: bool = False
    hide_ns: bool = True
    p_display_mode: PDisplayModeType = "stars"

    def __post_init__(self) -> None:
        if self.theme in _LEGACY_THEME_ALIASES:
            self.theme = _LEGACY_THEME_ALIASES[self.theme]  # type: ignore[assignment]
        if self.palette_name in _LEGACY_PALETTE_ALIASES:
            self.palette_name = _LEGACY_PALETTE_ALIASES[self.palette_name]

    @property
    def size_inches(self) -> Tuple[float, float]:
        """Return target physical figure size in inches."""
        return (float(self.width_mm) / 25.4, float(self.height_mm) / 25.4)

    def copy(self) -> PlotStyleConfig:
        """Return an independent deep copy of this configuration."""
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration fields into a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_theme(cls, theme: str, **overrides: Any) -> PlotStyleConfig:
        """Construct a configuration initialized with defaults from *theme*."""
        canonical_theme = _LEGACY_THEME_ALIASES.get(theme, theme)
        preset = THEME_PRESETS.get(canonical_theme, THEME_PRESETS["IRIS Dark"])
        kwargs: Dict[str, Any] = {
            "theme": canonical_theme,
            "font_family": preset["font_family"],
            "despine_top": preset["despine_top"],
            "despine_right": preset["despine_right"],
            "spine_linewidth": preset["spine_linewidth"],
            "grid_mode": preset["grid_mode"],
            "grid_linestyle": preset["grid_linestyle"],
            "grid_alpha": preset["grid_alpha"],
        }
        kwargs.update(overrides)
        return cls(**kwargs)


def get_palette_colors(palette_name: str, n: Optional[int] = None) -> List[str]:
    """Return hex colors for *palette_name*, cycling or sampling to length *n*."""
    canonical = _LEGACY_PALETTE_ALIASES.get(palette_name, palette_name)
    cmap_map = {"viridis": "viridis", "RdBu_r": "RdBu_r", "coolwarm": "coolwarm"}
    if n is not None and n > 0 and canonical in cmap_map:
        return _sample_cmap_hex(cmap_map[canonical], n)

    base = SCIENTIFIC_PALETTES.get(canonical)
    if base is None:
        if canonical in plt.colormaps:
            return _sample_cmap_hex(canonical, n or 8)
        base = SCIENTIFIC_PALETTES["Okabe-Ito"]

    if n is None:
        return list(base)
    if n <= 0:
        return []
    return [base[i % len(base)] for i in range(n)]


def resolve_category_colors(
    categories: Sequence[str],
    config: PlotStyleConfig,
) -> Dict[str, str]:
    """Map each category name to its custom override color or palette color."""
    palette = get_palette_colors(config.palette_name, len(categories))
    resolved: Dict[str, str] = {}
    for idx, cat in enumerate(categories):
        resolved[cat] = config.custom_colors.get(cat, palette[idx])
    return resolved


def _build_rc_params(config: PlotStyleConfig) -> Dict[str, Any]:
    """Build Matplotlib rcParams context dictionary for Illustrator-safe export."""
    sans_list = [config.font_family]
    if "DejaVu Sans" not in sans_list:
        sans_list.append("DejaVu Sans")
    rc_params: Dict[str, Any] = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.sans-serif": sans_list,
    }
    if config.font_family == "Times New Roman":
        rc_params["font.family"] = "serif"
        rc_params["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    else:
        rc_params["font.family"] = "sans-serif"
    return rc_params


def _normalize_axes(
    fig: Figure,
    ax: Union[matplotlib.axes.Axes, Iterable[matplotlib.axes.Axes], None],
) -> List[matplotlib.axes.Axes]:
    """Normalize an Axes object, iterable of Axes, or None to a flat list."""
    if ax is None:
        return list(fig.axes)
    if isinstance(ax, matplotlib.axes.Axes):
        return [ax]
    if isinstance(ax, np.ndarray):
        return [a for a in ax.ravel() if isinstance(a, matplotlib.axes.Axes)]
    return [a for a in ax if isinstance(a, matplotlib.axes.Axes)]


class _CustomLabelFormatter(mticker.Formatter):
    """Wrap an existing Matplotlib formatter to apply reversible custom label overrides."""

    def __init__(self, base_formatter: mticker.Formatter, label_map: Dict[str, str]) -> None:
        self.base_formatter = base_formatter
        self.label_map = dict(label_map)

    def set_axis(self, axis: Any) -> None:
        super().set_axis(axis)
        self.base_formatter.set_axis(axis)

    def set_locs(self, locs: Sequence[float]) -> None:
        super().set_locs(locs)
        self.base_formatter.set_locs(locs)

    def get_offset(self) -> str:
        return self.base_formatter.get_offset()

    def __call__(self, x: float, pos: Optional[int] = None) -> str:
        raw = self.base_formatter(x, pos)
        return self.label_map.get(raw, raw)

    def format_ticks(self, values: Sequence[float]) -> List[str]:
        raw_list = self.base_formatter.format_ticks(values)
        return [self.label_map.get(raw, raw) for raw in raw_list]


def _sync_axis_formatter_labels(
    axis_obj: Any,
    custom_labels: Dict[str, str],
) -> List[str]:
    """Apply or revert *custom_labels* via axis formatter and return original tick strings."""
    cur_fmt = axis_obj.get_major_formatter()
    if isinstance(cur_fmt, _CustomLabelFormatter):
        base_fmt = cur_fmt.base_formatter
    else:
        base_fmt = cur_fmt

    locs = axis_obj.get_majorticklocs()
    orig_labels = list(base_fmt.format_ticks(locs)) if len(locs) else []

    if custom_labels:
        if isinstance(cur_fmt, _CustomLabelFormatter):
            cur_fmt.label_map = dict(custom_labels)
        else:
            axis_obj.set_major_formatter(_CustomLabelFormatter(base_fmt, custom_labels))
    elif isinstance(cur_fmt, _CustomLabelFormatter):
        axis_obj.set_major_formatter(base_fmt)

    return orig_labels


def _apply_custom_label_text(
    text_obj: matplotlib.text.Text,
    custom_labels: Dict[str, str],
) -> str:
    """Apply *custom_labels* to a Matplotlib Text artist, tracking original text."""
    if not hasattr(text_obj, "_iris_orig_text"):
        setattr(text_obj, "_iris_orig_text", text_obj.get_text())
    orig_text = str(getattr(text_obj, "_iris_orig_text"))
    new_text = custom_labels.get(orig_text, orig_text)
    if text_obj.get_text() != new_text:
        text_obj.set_text(new_text)
    return orig_text


def _apply_custom_colors_to_axis(
    axis: matplotlib.axes.Axes,
    custom_colors: Dict[str, str],
    orig_xtick_labels: List[str],
) -> None:
    """Update artist colors matching category names in *custom_colors*."""
    if not custom_colors:
        return

    for container in axis.containers:
        label = getattr(container, "_iris_orig_label", None)
        if label is None:
            label = container.get_label()
            setattr(container, "_iris_orig_label", label)
        if label in custom_colors:
            color = custom_colors[label]
            for patch in container:
                if hasattr(patch, "set_facecolor"):
                    patch.set_facecolor(color)
        elif len(container) == len(orig_xtick_labels) and orig_xtick_labels:
            for idx, patch in enumerate(container):
                cat = orig_xtick_labels[idx]
                if cat in custom_colors and hasattr(patch, "set_facecolor"):
                    patch.set_facecolor(custom_colors[cat])

    for line in axis.lines:
        if getattr(line, "_iris_bracket_line", False):
            continue
        label = getattr(line, "_iris_orig_label", line.get_label())
        setattr(line, "_iris_orig_label", label)
        if label in custom_colors:
            line.set_color(custom_colors[label])

    for coll in axis.collections:
        label = getattr(coll, "_iris_orig_label", coll.get_label())
        setattr(coll, "_iris_orig_label", label)
        if label in custom_colors and hasattr(coll, "set_facecolor"):
            coll.set_facecolor(custom_colors[label])


def _apply_axis_style(
    axis: matplotlib.axes.Axes,
    config: PlotStyleConfig,
    preset: Dict[str, Any],
) -> None:
    """Apply theme, typography, spines, grid, labels, and legend to one Axes."""
    axis.set_facecolor(preset["axes_facecolor"])
    axis.patch.set_alpha(0.0 if config.transparent else 1.0)

    spine_lw = max(0.2, float(config.spine_linewidth))
    spine_visibility = {
        "top": not config.despine_top,
        "right": not config.despine_right,
        "bottom": True,
        "left": True,
    }
    for name, visible in spine_visibility.items():
        if name in axis.spines:
            spine = axis.spines[name]
            spine.set_visible(visible)
            spine.set_color(preset["spine_color"])
            spine.set_linewidth(spine_lw)

    axis.tick_params(
        axis="both",
        which="both",
        colors=preset["tick_color"],
        labelsize=float(config.tick_size),
        width=max(0.5, spine_lw),
    )
    axis.tick_params(axis="x", labelrotation=int(config.tick_rotation))

    orig_xtick_labels = _sync_axis_formatter_labels(axis.xaxis, config.custom_labels)
    _sync_axis_formatter_labels(axis.yaxis, config.custom_labels)

    for lbl in axis.get_xticklabels():
        lbl.set_fontfamily(config.font_family)
        lbl.set_fontsize(float(config.tick_size))
        lbl.set_color(preset["tick_color"])
        lbl.set_rotation(int(config.tick_rotation))
        if 0 < int(config.tick_rotation) < 90:
            lbl.set_horizontalalignment("right")
            lbl.set_rotation_mode("anchor")
        else:
            lbl.set_horizontalalignment("center")

    for lbl in axis.get_yticklabels():
        lbl.set_fontfamily(config.font_family)
        lbl.set_fontsize(float(config.tick_size))
        lbl.set_color(preset["tick_color"])

    title_weight = "bold" if config.bold_title else "normal"
    label_weight = "bold" if config.bold_labels else "normal"

    if config.title:
        axis.set_title(
            config.title,
            fontsize=float(config.title_size),
            fontweight=title_weight,
            fontfamily=config.font_family,
            color=preset["text_color"],
        )
    else:
        axis.title.set_fontsize(float(config.title_size))
        axis.title.set_fontweight(title_weight)
        axis.title.set_fontfamily(config.font_family)
        axis.title.set_color(preset["text_color"])

    if config.xlabel:
        axis.set_xlabel(
            config.xlabel,
            fontsize=float(config.label_size),
            fontweight=label_weight,
            fontfamily=config.font_family,
            color=preset["text_color"],
        )
    else:
        axis.xaxis.label.set_fontsize(float(config.label_size))
        axis.xaxis.label.set_fontweight(label_weight)
        axis.xaxis.label.set_fontfamily(config.font_family)
        axis.xaxis.label.set_color(preset["text_color"])

    if config.ylabel:
        axis.set_ylabel(
            config.ylabel,
            fontsize=float(config.label_size),
            fontweight=label_weight,
            fontfamily=config.font_family,
            color=preset["text_color"],
        )
    else:
        axis.yaxis.label.set_fontsize(float(config.label_size))
        axis.yaxis.label.set_fontweight(label_weight)
        axis.yaxis.label.set_fontfamily(config.font_family)
        axis.yaxis.label.set_color(preset["text_color"])

    axis.set_axisbelow(True)
    axis.grid(False)
    grid_axis_map = {"Y-Axis": "y", "X-Axis": "x", "Both": "both"}
    if config.grid_mode in grid_axis_map:
        axis.grid(
            True,
            axis=grid_axis_map[config.grid_mode],
            color=preset["grid_color"],
            linestyle=config.grid_linestyle,
            alpha=float(config.grid_alpha),
            linewidth=0.6,
        )

    for line in axis.lines:
        if getattr(line, "_iris_bracket_line", False):
            line.set_color(preset["bracket_color"])
            line.set_linewidth(max(0.8, spine_lw))

    for txt in axis.texts:
        txt.set_fontfamily(config.font_family)
        if getattr(txt, "_iris_bracket_text", False):
            txt.set_color(preset["bracket_color"])
            txt.set_fontsize(float(config.stat_size))

    _apply_custom_colors_to_axis(axis, config.custom_colors, orig_xtick_labels)

    leg = axis.get_legend()
    if leg is not None:
        if config.legend_title:
            leg.set_title(config.legend_title)
        leg_title = leg.get_title()
        if leg_title is not None:
            leg_title.set_fontsize(float(config.legend_size))
            leg_title.set_fontfamily(config.font_family)
            leg_title.set_fontweight(label_weight)
            leg_title.set_color(preset["text_color"])

        handles = getattr(leg, "legend_handles", None)
        if handles is None:
            handles = getattr(leg, "legendHandles", [])
        for idx, txt in enumerate(leg.get_texts()):
            orig_lbl = _apply_custom_label_text(txt, config.custom_labels)
            txt.set_fontsize(float(config.legend_size))
            txt.set_fontfamily(config.font_family)
            txt.set_color(preset["text_color"])
            if orig_lbl in config.custom_colors and idx < len(handles):
                handle = handles[idx]
                c = config.custom_colors[orig_lbl]
                if hasattr(handle, "set_facecolor"):
                    handle.set_facecolor(c)
                if hasattr(handle, "set_color"):
                    handle.set_color(c)

        if preset["legend_edgecolor"] == "none":
            leg.set_frame_on(False)
        else:
            leg.set_frame_on(True)
            frame = leg.get_frame()
            frame.set_facecolor(preset["legend_facecolor"])
            frame.set_edgecolor(preset["legend_edgecolor"])


def apply_plot_style(
    fig: Figure,
    ax: Union[matplotlib.axes.Axes, Iterable[matplotlib.axes.Axes], None],
    config: PlotStyleConfig,
) -> None:
    """Style figure and axes backgrounds, spines, ticks, fonts, grid, and legends."""
    canonical_theme = _LEGACY_THEME_ALIASES.get(config.theme, config.theme)
    preset = THEME_PRESETS.get(canonical_theme, THEME_PRESETS["IRIS Dark"])
    fig.patch.set_facecolor(preset["figure_facecolor"])
    fig.patch.set_alpha(0.0 if config.transparent else 1.0)

    for axis in _normalize_axes(fig, ax):
        _apply_axis_style(axis, config, preset)


def format_significance_label(
    p_value: Optional[float],
    q_value: Optional[float] = None,
    mode: PDisplayModeType = "stars",
) -> str:
    """Format a p-value or q-value into significance stars or numeric text."""
    if mode == "q_value":
        val = q_value if q_value is not None else p_value
        prefix = "q"
    else:
        val = p_value if p_value is not None else q_value
        prefix = "p"

    if val is None or not np.isfinite(float(val)):
        return "ns"

    v = float(val)
    if mode == "stars":
        if v < 0.0001:
            return "****"
        if v < 0.001:
            return "***"
        if v < 0.01:
            return "**"
        if v < 0.05:
            return "*"
        return "ns"

    if v < 0.0001:
        return f"{prefix} < 0.0001"
    if v < 0.001:
        return f"{prefix} = {v:.4f}"
    return f"{prefix} = {v:.3f}"


def _resolve_group_lookup(
    mapping: Union[Dict[Any, float], Sequence[float], float],
    key: Any,
    fallback: Optional[float] = None,
) -> float:
    """Resolve group coordinate or ceiling from a dict, sequence, or scalar."""
    if isinstance(mapping, dict):
        if key in mapping:
            return float(mapping[key])
        key_str = str(key)
        if key_str in mapping:
            return float(mapping[key_str])
        if fallback is not None:
            return float(fallback)
        raise KeyError(f"Group key {key!r} not found in mapping.")
    if isinstance(mapping, (list, tuple, np.ndarray, pd.Series)):
        idx = int(key)
        return float(mapping[idx])
    return float(mapping)


def _extract_comparisons(
    comparisons: Union[Sequence[Any], pd.DataFrame],
) -> List[Tuple[Any, Any, Optional[float], Optional[float], Optional[str]]]:
    """Normalize tuple, dict, or DataFrame comparison specs into standard tuples."""
    records: List[Dict[str, Any]] = []
    if isinstance(comparisons, pd.DataFrame):
        records = comparisons.to_dict(orient="records")
    else:
        for item in comparisons:
            if isinstance(item, dict):
                records.append(item)
            elif isinstance(item, (tuple, list)):
                if len(item) < 3:
                    raise ValueError(
                        "Tuple comparison must have at least 3 elements: (group_a, group_b, p_val)."
                    )
                ga, gb, third = item[0], item[1], item[2]
                fourth = item[3] if len(item) >= 4 else None
                if isinstance(third, str):
                    yield_item = (ga, gb, None, _to_float_or_none(fourth), third)
                else:
                    yield_item = (ga, gb, _to_float_or_none(third), _to_float_or_none(fourth), None)
                records.append({"__normalized__": yield_item})
            else:
                raise TypeError(f"Unsupported comparison entry type: {type(item).__name__}")

    normalized: List[Tuple[Any, Any, Optional[float], Optional[float], Optional[str]]] = []
    for rec in records:
        if "__normalized__" in rec:
            normalized.append(rec["__normalized__"])
            continue
        ga = _first_present(rec, ("group_a", "group1", "a", "Group_A", "Group1", "source"))
        gb = _first_present(rec, ("group_b", "group2", "b", "Group_B", "Group2", "target"))
        p_val = _to_float_or_none(
            _first_present(rec, ("p_value", "pvalue", "p_val", "p", "P_Value"))
        )
        q_val = _to_float_or_none(
            _first_present(
                rec,
                ("q_value", "qvalue", "q_val", "q", "p_adj", "padj", "fdr", "adj_p"),
            )
        )
        custom_str = rec.get("label") if isinstance(rec.get("label"), str) else None
        if ga is None or gb is None:
            raise ValueError(f"Comparison record missing group keys: {rec}")
        normalized.append((ga, gb, p_val, q_val, custom_str))
    return normalized


def _first_present(mapping: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    """Return the value for the first matching key present in *mapping*."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _to_float_or_none(value: Any) -> Optional[float]:
    """Convert numeric values to float, returning None for missing or NaN inputs."""
    if value is None:
        return None
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    return fval if np.isfinite(fval) else None


def draw_significance_brackets(
    ax: matplotlib.axes.Axes,
    comparisons: Union[Sequence[Any], pd.DataFrame],
    x_positions: Union[Dict[Any, float], Sequence[float]],
    y_maxes: Union[Dict[Any, float], Sequence[float], float],
    config: Optional[PlotStyleConfig] = None,
    *,
    color: Optional[str] = None,
    linewidth: Optional[float] = None,
    step_fraction: float = 0.08,
    tip_fraction: float = 0.025,
    pad_fraction: float = 0.04,
    hide_ns: Optional[bool] = None,
    p_display_mode: Optional[PDisplayModeType] = None,
) -> List[Dict[str, Any]]:
    """
    Draw collision-free stacked significance brackets above compared groups.

    Supports pairwise comparisons as tuples `(group_a, group_b, p_val[, q_val])`,
    dicts, or a DataFrame, and updates the y-axis upper limit for clean headroom.
    """
    cfg = config or PlotStyleConfig()
    effective_hide_ns = cfg.hide_ns if hide_ns is None else bool(hide_ns)
    effective_mode = cfg.p_display_mode if p_display_mode is None else p_display_mode

    canonical_theme = _LEGACY_THEME_ALIASES.get(cfg.theme, cfg.theme)
    preset = THEME_PRESETS.get(canonical_theme, THEME_PRESETS["IRIS Dark"])
    if color is not None:
        bracket_color = color
    elif config is not None:
        bracket_color = preset["bracket_color"]
    else:
        bg_rgb = mcolors.to_rgb(ax.get_facecolor())
        lum = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
        bracket_color = "#E0E0E0" if lum < 0.45 else "#000000"

    line_w = float(linewidth) if linewidth is not None else max(0.8, float(cfg.spine_linewidth))

    parsed = _extract_comparisons(comparisons)
    if not parsed:
        return []

    if isinstance(x_positions, dict):
        all_group_xs = {k: float(v) for k, v in x_positions.items()}
    else:
        all_group_xs = {idx: float(v) for idx, v in enumerate(x_positions)}

    if isinstance(y_maxes, dict):
        all_group_ys = {k: float(v) for k, v in y_maxes.items()}
    elif isinstance(y_maxes, (list, tuple, np.ndarray, pd.Series)):
        all_group_ys = {idx: float(v) for idx, v in enumerate(y_maxes)}
    else:
        all_group_ys = {k: float(y_maxes) for k in all_group_xs}

    candidates: List[Dict[str, Any]] = []
    for ga, gb, p_val, q_val, custom_str in parsed:
        stat_for_ns = q_val if (effective_mode == "q_value" and q_val is not None) else p_val
        if effective_hide_ns and stat_for_ns is not None and stat_for_ns >= 0.05:
            continue

        label = custom_str or format_significance_label(p_val, q_val, mode=effective_mode)
        if effective_hide_ns and label == "ns":
            continue

        x1 = _resolve_group_lookup(x_positions, ga)
        x2 = _resolve_group_lookup(x_positions, gb)
        x_left, x_right = min(x1, x2), max(x1, x2)
        if x_right <= x_left:
            continue

        candidates.append(
            {
                "group_a": ga,
                "group_b": gb,
                "x_left": x_left,
                "x_right": x_right,
                "span": x_right - x_left,
                "p_value": p_val,
                "q_value": q_val,
                "label": label,
            }
        )

    if not candidates:
        return []

    cur_ymin, cur_ymax = ax.get_ylim()
    y_vals = list(all_group_ys.values()) or [cur_ymax]
    y_data_max = max(y_vals)
    y_span = max(y_data_max - min(cur_ymin, min(y_vals)), 1e-6)
    step = y_span * float(step_fraction)
    tip = y_span * float(tip_fraction)
    base_pad = y_span * float(pad_fraction)

    # Sort narrow spans first so nested brackets sit cleanly inside wider ones.
    candidates.sort(key=lambda item: (item["span"], item["x_left"], item["x_right"]))

    placed: List[Tuple[float, float, float]] = []
    results: List[Dict[str, Any]] = []
    eps = 1e-6

    for cand in candidates:
        xl, xr = cand["x_left"], cand["x_right"]
        span_maxes = [
            all_group_ys[k]
            for k, gx in all_group_xs.items()
            if (xl - eps) <= gx <= (xr + eps) and k in all_group_ys
        ]
        local_ceiling = max(span_maxes) if span_maxes else y_data_max
        y_base = local_ceiling + base_pad

        overlapping_ys = [
            py for (pxl, pxr, py) in placed if not (xr < pxl - eps or xl > pxr + eps)
        ]
        while any(abs(y_base - py) < step * 0.99 for py in overlapping_ys):
            y_base += step

        placed.append((xl, xr, y_base))

        (line_artist,) = ax.plot(
            [xl, xl, xr, xr],
            [y_base - tip, y_base, y_base, y_base - tip],
            color=bracket_color,
            linewidth=line_w,
            clip_on=False,
        )
        setattr(line_artist, "_iris_bracket_line", True)

        text_artist = ax.text(
            0.5 * (xl + xr),
            y_base + tip * 0.35,
            cand["label"],
            ha="center",
            va="bottom",
            fontsize=float(cfg.stat_size),
            fontfamily=cfg.font_family,
            color=bracket_color,
        )
        setattr(text_artist, "_iris_bracket_text", True)

        results.append(
            {
                "group_a": cand["group_a"],
                "group_b": cand["group_b"],
                "x_left": xl,
                "x_right": xr,
                "y": y_base,
                "label": cand["label"],
                "p_value": cand["p_value"],
                "q_value": cand["q_value"],
                "line": line_artist,
                "text": text_artist,
            }
        )

    highest_y = max(py for _, _, py in placed)
    required_ymax = highest_y + step * 1.45
    if required_ymax > cur_ymax:
        ax.set_ylim(cur_ymin, required_ymax)

    return results


def export_publication_figure(
    figure: Figure,
    filepath: Union[str, Path],
    config: Optional[PlotStyleConfig] = None,
    *,
    bbox_inches: Optional[str] = "tight",
    pad_inches: float = 0.06,
) -> Path:
    """
    Export *figure* at publication dimensions with embedded editable fonts.

    Temporarily applies TrueType/editable SVG rcParams and target physical inches
    `(width_mm / 25.4, height_mm / 25.4)`, writes `.pdf`, `.svg`, `.eps`, `.tiff`
    (LZW compression), or `.png`, and restores original figure size and DPI.
    """
    cfg = config or PlotStyleConfig()
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if not suffix:
        raise ValueError(f"Output filepath must include a file extension: {out_path}")

    format_map = {
        ".pdf": "pdf",
        ".svg": "svg",
        ".eps": "eps",
        ".ps": "ps",
        ".tif": "tiff",
        ".tiff": "tiff",
        ".png": "png",
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
    }
    fmt = format_map.get(suffix)
    if fmt is None:
        raise ValueError(f"Unsupported figure export format: {suffix}")

    orig_size = tuple(float(v) for v in figure.get_size_inches())
    orig_dpi = float(figure.get_dpi())
    target_w, target_h = cfg.size_inches
    rc_params = _build_rc_params(cfg)
    transparent = bool(cfg.transparent) and fmt not in {"eps", "ps", "jpeg"}

    try:
        figure.set_size_inches(target_w, target_h, forward=False)
        figure.set_dpi(int(cfg.dpi))
        with matplotlib.rc_context(rc_params):
            save_kwargs: Dict[str, Any] = {
                "format": fmt,
                "dpi": int(cfg.dpi),
                "transparent": transparent,
            }
            if bbox_inches is not None:
                save_kwargs["bbox_inches"] = bbox_inches
                save_kwargs["pad_inches"] = pad_inches
            if not transparent:
                save_kwargs["facecolor"] = figure.get_facecolor()
            if fmt == "tiff":
                save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}

            try:
                figure.savefig(str(out_path), **save_kwargs)
            except TypeError:
                if fmt == "tiff":
                    _save_tiff_lzw_fallback(figure, out_path, cfg.dpi, save_kwargs)
                else:
                    raise
    finally:
        figure.set_size_inches(orig_size[0], orig_size[1], forward=False)
        figure.set_dpi(orig_dpi)

    return out_path


def _save_tiff_lzw_fallback(
    figure: Figure,
    out_path: Path,
    dpi: int,
    save_kwargs: Dict[str, Any],
) -> None:
    """Fallback TIFF LZW writer via PNG buffer and Pillow/tifffile."""
    png_kwargs = {k: v for k, v in save_kwargs.items() if k != "pil_kwargs"}
    png_kwargs["format"] = "png"
    buf = io.BytesIO()
    figure.savefig(buf, **png_kwargs)
    buf.seek(0)
    try:
        from PIL import Image

        with Image.open(buf) as img:
            img.save(str(out_path), format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
    except Exception:
        import tifffile

        arr = plt.imread(buf, format="png")
        arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        tifffile.imwrite(str(out_path), arr_u8, compression="lzw")


def render_figure_to_png_bytes(
    figure: Figure,
    config: Optional[PlotStyleConfig] = None,
    dpi: Optional[int] = None,
) -> bytes:
    """Render *figure* into PNG bytes at target DPI, restoring figure dimensions."""
    target_dpi = int(dpi) if dpi is not None else (int(config.dpi) if config is not None else 300)
    orig_size = tuple(float(v) for v in figure.get_size_inches())
    orig_dpi = float(figure.get_dpi())
    rc_params = _build_rc_params(config) if config is not None else {}
    transparent = bool(config.transparent) if config is not None else False

    buf = io.BytesIO()
    try:
        if config is not None:
            w_in, h_in = config.size_inches
            figure.set_size_inches(w_in, h_in, forward=False)
        figure.set_dpi(target_dpi)
        with matplotlib.rc_context(rc_params):
            save_kwargs: Dict[str, Any] = {
                "format": "png",
                "dpi": target_dpi,
                "bbox_inches": "tight",
                "pad_inches": 0.06,
                "transparent": transparent,
            }
            if not transparent:
                save_kwargs["facecolor"] = figure.get_facecolor()
            figure.savefig(buf, **save_kwargs)
    finally:
        figure.set_size_inches(orig_size[0], orig_size[1], forward=False)
        figure.set_dpi(orig_dpi)

    return buf.getvalue()


def copy_figure_to_clipboard(
    figure: Figure,
    config: Optional[PlotStyleConfig] = None,
    dpi: Optional[int] = None,
) -> bool:
    """
    Render *figure* at target DPI and copy the image to the Qt clipboard.

    Returns False safely when no QApplication instance or clipboard is active.
    """
    png_bytes = render_figure_to_png_bytes(figure, config=config, dpi=dpi)
    app = QApplication.instance()
    if app is None:
        logger.warning("No active QApplication; skipping clipboard copy in headless mode.")
        return False
    clipboard = app.clipboard()
    if clipboard is None:
        return False

    qimg = QImage.fromData(png_bytes, "PNG")
    if qimg.isNull():
        return False
    clipboard.setImage(qimg)
    return True


_ILLEGAL_XLSX_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_excel_value(val: Any) -> Any:
    """Convert pandas/NumPy scalars into clean openpyxl-compatible Python values."""
    if val is None:
        return ""
    if isinstance(val, (float, np.floating)):
        fval = float(val)
        return "" if not np.isfinite(fval) else fval
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(val, (dict, list, tuple)):
        text = json.dumps(val, default=str)
    else:
        text = str(val)
    return _ILLEGAL_XLSX_RE.sub("", text)


def _prepare_export_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Preserve meaningful DataFrame indices as a column while dropping default RangeIndex."""
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    if not isinstance(out.index, pd.RangeIndex) or out.index.name is not None:
        idx_name = str(out.index.name) if out.index.name is not None else "Category"
        if idx_name not in out.columns:
            out = out.reset_index().rename(columns={"index": idx_name})
    return out


def _write_dataframe_sheet(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    df: pd.DataFrame,
    empty_note: str,
) -> None:
    """Write a DataFrame to an openpyxl worksheet with Source Data formatting."""
    header_fill = PatternFill(start_color="E8EEF5", end_color="E8EEF5", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="111111")
    body_font = Font(name="Calibri", size=10, color="111111")
    thin_side = Side(style="thin", color="444444")
    header_border = Border(bottom=thin_side)

    clean_df = _prepare_export_dataframe(df)
    if clean_df.empty and len(clean_df.columns) == 0:
        ws.append(["Status"])
        ws.append([empty_note])
        ws["A1"].font = header_font
        ws["A1"].fill = header_fill
        ws["A1"].border = header_border
        ws["A2"].font = body_font
        ws.column_dimensions["A"].width = max(24, len(empty_note) + 4)
        return

    headers = [str(col) for col in clean_df.columns]
    ws.append(headers)
    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = header_border
        cell.alignment = Alignment(horizontal="left", vertical="center")

    for row_vals in clean_df.itertuples(index=False, name=None):
        ws.append([_sanitize_excel_value(v) for v in row_vals])

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=len(headers)):
        for cell in row:
            cell.font = body_font

    ws.freeze_panes = "A2"
    for col_idx, col_name in enumerate(headers, start=1):
        col_letter = get_column_letter(col_idx)
        max_len = len(col_name)
        for row_idx in range(2, min(ws.max_row, 200) + 1):
            val = ws.cell(row=row_idx, column=col_idx).value
            if val is not None:
                max_len = max(max_len, len(str(val)))
        ws.column_dimensions[col_letter].width = float(min(max(max_len + 4, 12), 48))


def _build_metadata_dataframe(
    metadata: Optional[Union[PlotStyleConfig, Dict[str, Any]]],
) -> pd.DataFrame:
    """Convert metadata dict or PlotStyleConfig into a Parameter/Value DataFrame."""
    rows: List[Tuple[str, Any]] = []
    if isinstance(metadata, PlotStyleConfig):
        meta_dict = metadata.to_dict()
    elif isinstance(metadata, dict):
        meta_dict = dict(metadata)
    else:
        meta_dict = {}

    for key, value in meta_dict.items():
        if isinstance(value, PlotStyleConfig):
            for sub_k, sub_v in value.to_dict().items():
                rows.append((f"{key}.{sub_k}", _sanitize_excel_value(sub_v)))
        else:
            rows.append((str(key), _sanitize_excel_value(value)))

    existing_keys = {k for k, _ in rows}
    if "Export_Timestamp_UTC" not in existing_keys:
        rows.append(
            ("Export_Timestamp_UTC", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
        )
    if "Format" not in existing_keys:
        rows.append(("Format", "Source Data Workbook"))

    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def export_source_data_excel(
    filepath: Union[str, Path],
    summary_df: pd.DataFrame,
    replicates_df: Optional[pd.DataFrame] = None,
    stats_df: Optional[pd.DataFrame] = None,
    metadata: Optional[Union[PlotStyleConfig, Dict[str, Any]]] = None,
    *,
    include_empty_sheets: bool = False,
) -> Path:
    """
    Write a multi-sheet Source Data workbook (.xlsx) via openpyxl.

    Sheets written:
    - `Summary_Data` (always)
    - `Sample_Replicates` (when `replicates_df` is provided or `include_empty_sheets=True`)
    - `Statistical_Tests` (when `stats_df` is provided or `include_empty_sheets=True`)
    - `Figure_Parameters` (always)
    """
    out_path = Path(filepath)
    if out_path.suffix.lower() != ".xlsx":
        out_path = out_path.with_suffix(".xlsx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    ws_summary = wb.active
    ws_summary.title = "Summary_Data"
    _write_dataframe_sheet(ws_summary, summary_df, "No summary data available.")

    if replicates_df is not None or include_empty_sheets:
        ws_rep = wb.create_sheet("Sample_Replicates")
        _write_dataframe_sheet(ws_rep, replicates_df, "No replicate-level data provided.")

    if stats_df is not None or include_empty_sheets:
        ws_stats = wb.create_sheet("Statistical_Tests")
        _write_dataframe_sheet(ws_stats, stats_df, "No statistical test results provided.")

    ws_meta = wb.create_sheet("Figure_Parameters")
    meta_df = _build_metadata_dataframe(metadata)
    _write_dataframe_sheet(ws_meta, meta_df, "No figure parameters recorded.")

    wb.save(str(out_path))
    return out_path


class PublicationStyleSidebar(QWidget):
    """
    Reusable sidebar widget for interactive publication figure styling and export.

    Signals
    -------
    styleChanged : Signal(PlotStyleConfig)
        Emitted whenever any visual or export styling option changes.
    exportFigureRequested : Signal()
        Emitted when the user requests exporting the figure.
    exportSourceDataRequested : Signal()
        Emitted when the user requests exporting Source Data (.xlsx).
    copyClipboardRequested : Signal()
        Emitted when the user requests copying the figure to clipboard.
    """

    styleChanged = Signal(PlotStyleConfig)
    exportFigureRequested = Signal()
    exportSourceDataRequested = Signal()
    copyClipboardRequested = Signal()

    def __init__(
        self,
        config: Optional[PlotStyleConfig] = None,
        categories: Optional[Sequence[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config: PlotStyleConfig = config.copy() if config is not None else PlotStyleConfig()
        self._categories: List[str] = list(categories) if categories is not None else []
        self._default_colors: Dict[str, str] = {}
        self._category_rows: Dict[str, Dict[str, Any]] = {}
        self._updating: bool = False

        self.setStyleSheet(STUDIO_DARK_STYLESHEET)
        self._build_ui()
        self.set_config(self._config, emit=False)
        self.set_categories(self._categories)

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self._form_container = QVBoxLayout(content)
        self._form_container.setContentsMargins(4, 4, 4, 4)
        self._form_container.setSpacing(8)

        self._build_theme_group()
        self._build_typography_group()
        self._build_labels_group()
        self._build_axes_group()
        self._build_stats_group()
        self._build_categories_group()
        self._build_dimensions_group()
        self._form_container.addStretch(1)

        scroll.setWidget(content)
        root_layout.addWidget(scroll, 1)
        self._build_action_buttons(root_layout)

    def _build_theme_group(self) -> None:
        group, card_layout = create_section_card("Theme & Palette")
        form = QFormLayout()
        form.setSpacing(6)

        self.combo_theme = QComboBox()
        self.combo_theme.addItems(list(THEME_PRESETS.keys()))
        self.combo_theme.currentTextChanged.connect(self._on_theme_selected)
        form.addRow("Theme:", self.combo_theme)

        self.combo_palette = QComboBox()
        self.combo_palette.addItems(list(SCIENTIFIC_PALETTES.keys()))
        self.combo_palette.currentTextChanged.connect(self._on_palette_changed)
        form.addRow("Palette:", self.combo_palette)

        card_layout.addLayout(form)
        self._form_container.addWidget(group)

    def _build_typography_group(self) -> None:
        group, card_layout = create_section_card("Typography")
        form = QFormLayout()
        form.setSpacing(6)

        self.combo_font = QComboBox()
        self.combo_font.addItems(list(FONT_FAMILIES))
        self.combo_font.currentTextChanged.connect(self._on_control_changed)
        form.addRow("Font Family:", self.combo_font)

        self.spin_title_size = self._make_double_spin(6.0, 36.0, 0.5, 1, 12.0)
        self.spin_label_size = self._make_double_spin(6.0, 28.0, 0.5, 1, 11.0)
        self.spin_tick_size = self._make_double_spin(5.0, 24.0, 0.5, 1, 10.0)
        self.spin_legend_size = self._make_double_spin(5.0, 24.0, 0.5, 1, 9.0)
        self.spin_stat_size = self._make_double_spin(5.0, 24.0, 0.5, 1, 9.0)

        form.addRow("Title Size (pt):", self.spin_title_size)
        form.addRow("Label Size (pt):", self.spin_label_size)
        form.addRow("Tick Size (pt):", self.spin_tick_size)
        form.addRow("Legend Size (pt):", self.spin_legend_size)
        form.addRow("Stat Size (pt):", self.spin_stat_size)

        self.chk_bold_title = QCheckBox("Bold Title")
        self.chk_bold_title.toggled.connect(self._on_control_changed)
        form.addRow("", self.chk_bold_title)

        self.chk_bold_labels = QCheckBox("Bold Axis Labels")
        self.chk_bold_labels.toggled.connect(self._on_control_changed)
        form.addRow("", self.chk_bold_labels)

        self.spin_tick_rotation = QSpinBox()
        self.spin_tick_rotation.setRange(0, 90)
        self.spin_tick_rotation.setSingleStep(15)
        self.spin_tick_rotation.setSuffix(" deg")
        self.spin_tick_rotation.valueChanged.connect(self._on_control_changed)
        form.addRow("X-Tick Angle:", self.spin_tick_rotation)

        card_layout.addLayout(form)
        self._form_container.addWidget(group)

    def _build_labels_group(self) -> None:
        group, card_layout = create_section_card("Labels & Titles")
        form = QFormLayout()
        form.setSpacing(6)

        self.edit_title = self._make_line_edit("Auto / plot default")
        self.edit_xlabel = self._make_line_edit("Auto / plot default")
        self.edit_ylabel = self._make_line_edit("Auto / plot default")
        self.edit_legend_title = self._make_line_edit("Auto / plot default")

        form.addRow("Title:", self.edit_title)
        form.addRow("X Label:", self.edit_xlabel)
        form.addRow("Y Label:", self.edit_ylabel)
        form.addRow("Legend Title:", self.edit_legend_title)

        card_layout.addLayout(form)
        self._form_container.addWidget(group)

    def _build_axes_group(self) -> None:
        group, card_layout = create_section_card("Axes, Spines & Grid")
        form = QFormLayout()
        form.setSpacing(6)

        self.chk_despine_top = QCheckBox("Hide Top Spine")
        self.chk_despine_top.toggled.connect(self._on_control_changed)
        form.addRow("", self.chk_despine_top)

        self.chk_despine_right = QCheckBox("Hide Right Spine")
        self.chk_despine_right.toggled.connect(self._on_control_changed)
        form.addRow("", self.chk_despine_right)

        self.spin_spine_width = self._make_double_spin(0.2, 4.0, 0.2, 2, 1.0)
        form.addRow("Spine Width:", self.spin_spine_width)

        self.combo_grid = QComboBox()
        self.combo_grid.addItems(list(GRID_MODES))
        self.combo_grid.currentTextChanged.connect(self._on_control_changed)
        form.addRow("Grid Mode:", self.combo_grid)

        self.combo_grid_style = QComboBox()
        self.combo_grid_style.addItems(list(GRID_LINESTYLES))
        self.combo_grid_style.currentTextChanged.connect(self._on_control_changed)
        form.addRow("Grid Line:", self.combo_grid_style)

        self.spin_grid_alpha = self._make_double_spin(0.05, 1.0, 0.05, 2, 0.35)
        form.addRow("Grid Alpha:", self.spin_grid_alpha)

        card_layout.addLayout(form)
        self._form_container.addWidget(group)

    def _build_stats_group(self) -> None:
        group, card_layout = create_section_card("Statistics Annotations")
        form = QFormLayout()
        form.setSpacing(6)

        self.chk_hide_ns = QCheckBox("Hide Non-Significant (ns)")
        self.chk_hide_ns.toggled.connect(self._on_control_changed)
        form.addRow("", self.chk_hide_ns)

        self.combo_p_mode = QComboBox()
        self.combo_p_mode.addItems(list(P_DISPLAY_MODES))
        self.combo_p_mode.currentTextChanged.connect(self._on_control_changed)
        form.addRow("Display Mode:", self.combo_p_mode)

        card_layout.addLayout(form)
        self._form_container.addWidget(group)

    def _build_categories_group(self) -> None:
        self.group_categories, outer = create_section_card("Category Colors & Labels")

        self._cat_rows_layout = QVBoxLayout()
        self._cat_rows_layout.setSpacing(4)
        outer.addLayout(self._cat_rows_layout)

        self.btn_reset_categories = QPushButton("Reset Category Overrides")
        self.btn_reset_categories.clicked.connect(self._reset_category_overrides)
        outer.addWidget(self.btn_reset_categories)

        self._form_container.addWidget(self.group_categories)

    def _build_dimensions_group(self) -> None:
        group, card_layout = create_section_card("Page Size & Export")
        form = QFormLayout()
        form.setSpacing(6)

        self.combo_size_preset = QComboBox()
        self.combo_size_preset.addItems(list(SIZE_PRESETS.keys()))
        self.combo_size_preset.currentTextChanged.connect(self._on_size_preset_changed)
        form.addRow("Preset:", self.combo_size_preset)

        self.spin_width_mm = self._make_double_spin(
            20.0, 500.0, 1.0, 1, 183.0, on_change=self._on_dimension_changed
        )
        self.spin_width_mm.setSuffix(" mm")
        form.addRow("Width:", self.spin_width_mm)

        self.spin_height_mm = self._make_double_spin(
            20.0, 500.0, 1.0, 1, 120.0, on_change=self._on_dimension_changed
        )
        self.spin_height_mm.setSuffix(" mm")
        form.addRow("Height:", self.spin_height_mm)

        self.combo_dpi = QComboBox()
        self.combo_dpi.addItems([str(d) for d in DPI_OPTIONS])
        self.combo_dpi.currentTextChanged.connect(self._on_control_changed)
        form.addRow("Resolution (DPI):", self.combo_dpi)

        self.chk_transparent = QCheckBox("Transparent Background")
        self.chk_transparent.toggled.connect(self._on_control_changed)
        form.addRow("", self.chk_transparent)

        card_layout.addLayout(form)
        self._form_container.addWidget(group)

    def _build_action_buttons(self, root_layout: QVBoxLayout) -> None:
        btn_box = QVBoxLayout()
        btn_box.setSpacing(5)

        self.btn_copy_clipboard = QPushButton("Copy to Clipboard")
        self.btn_copy_clipboard.setStyleSheet(BUTTON_STUDIO)
        self.btn_copy_clipboard.clicked.connect(self.copyClipboardRequested.emit)
        btn_box.addWidget(self.btn_copy_clipboard)

        self.btn_export_figure = QPushButton("Export Figure")
        self.btn_export_figure.setStyleSheet(BUTTON_CTA)
        self.btn_export_figure.clicked.connect(self.exportFigureRequested.emit)
        btn_box.addWidget(self.btn_export_figure)

        self.btn_export_source_data = QPushButton("Export Source Data (.xlsx)")
        self.btn_export_source_data.setStyleSheet(BUTTON_CTA)
        self.btn_export_source_data.clicked.connect(self.exportSourceDataRequested.emit)
        btn_box.addWidget(self.btn_export_source_data)

        root_layout.addLayout(btn_box)

    def _make_double_spin(
        self,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
        value: float,
        on_change: Optional[Callable[..., None]] = None,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.valueChanged.connect(on_change or self._on_control_changed)
        return spin

    def _make_line_edit(self, placeholder: str) -> QLineEdit:
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.textChanged.connect(self._on_control_changed)
        return edit

    def _on_theme_selected(self, theme_name: str) -> None:
        if self._updating:
            return
        canonical_theme = _LEGACY_THEME_ALIASES.get(theme_name, theme_name)
        preset = THEME_PRESETS.get(canonical_theme)
        if preset is None:
            return
        self._updating = True
        try:
            self.combo_font.setCurrentText(preset["font_family"])
            self.chk_despine_top.setChecked(bool(preset["despine_top"]))
            self.chk_despine_right.setChecked(bool(preset["despine_right"]))
            self.spin_spine_width.setValue(float(preset["spine_linewidth"]))
            self.combo_grid.setCurrentText(preset["grid_mode"])
            self.combo_grid_style.setCurrentText(preset["grid_linestyle"])
            self.spin_grid_alpha.setValue(float(preset["grid_alpha"]))
        finally:
            self._updating = False
        self._emit_style_changed()

    def _on_palette_changed(self, _palette_name: str) -> None:
        if self._updating:
            return
        self._refresh_category_swatches()
        self._emit_style_changed()

    def _on_size_preset_changed(self, preset_name: str) -> None:
        if self._updating:
            return
        dims = SIZE_PRESETS.get(preset_name)
        if dims is not None:
            self._updating = True
            try:
                self.spin_width_mm.setValue(dims[0])
                self.spin_height_mm.setValue(dims[1])
            finally:
                self._updating = False
        self._emit_style_changed()

    def _on_dimension_changed(self, _value: float) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self.combo_size_preset.setCurrentText("Custom")
        finally:
            self._updating = False
        self._emit_style_changed()

    def _on_control_changed(self, *_args: Any) -> None:
        if self._updating:
            return
        self._emit_style_changed()

    def _emit_style_changed(self) -> None:
        self._config = self.get_config()
        self.styleChanged.emit(self._config.copy())

    def get_config(self) -> PlotStyleConfig:
        """Read current UI control states into a new PlotStyleConfig instance."""
        custom_labels: Dict[str, str] = {}
        for cat, row_info in self._category_rows.items():
            txt = row_info["line_edit"].text().strip()
            if txt and txt != cat:
                custom_labels[cat] = txt

        return PlotStyleConfig(
            theme=self.combo_theme.currentText(),  # type: ignore[arg-type]
            font_family=self.combo_font.currentText(),
            title_size=float(self.spin_title_size.value()),
            label_size=float(self.spin_label_size.value()),
            tick_size=float(self.spin_tick_size.value()),
            legend_size=float(self.spin_legend_size.value()),
            stat_size=float(self.spin_stat_size.value()),
            bold_title=self.chk_bold_title.isChecked(),
            bold_labels=self.chk_bold_labels.isChecked(),
            tick_rotation=int(self.spin_tick_rotation.value()),
            title=self.edit_title.text().strip(),
            xlabel=self.edit_xlabel.text().strip(),
            ylabel=self.edit_ylabel.text().strip(),
            legend_title=self.edit_legend_title.text().strip(),
            custom_labels=custom_labels,
            custom_colors=dict(self._config.custom_colors),
            palette_name=self.combo_palette.currentText(),
            despine_top=self.chk_despine_top.isChecked(),
            despine_right=self.chk_despine_right.isChecked(),
            spine_linewidth=float(self.spin_spine_width.value()),
            grid_mode=self.combo_grid.currentText(),  # type: ignore[arg-type]
            grid_linestyle=self.combo_grid_style.currentText(),
            grid_alpha=float(self.spin_grid_alpha.value()),
            width_mm=float(self.spin_width_mm.value()),
            height_mm=float(self.spin_height_mm.value()),
            size_preset=self.combo_size_preset.currentText(),
            dpi=int(self.combo_dpi.currentText()),
            transparent=self.chk_transparent.isChecked(),
            hide_ns=self.chk_hide_ns.isChecked(),
            p_display_mode=self.combo_p_mode.currentText(),  # type: ignore[arg-type]
        )

    def set_config(self, config: PlotStyleConfig, emit: bool = False) -> None:
        """Synchronize sidebar widgets from *config*."""
        cfg = config.copy()
        if cfg.theme in _LEGACY_THEME_ALIASES:
            cfg.theme = _LEGACY_THEME_ALIASES[cfg.theme]  # type: ignore[assignment]
        if cfg.palette_name in _LEGACY_PALETTE_ALIASES:
            cfg.palette_name = _LEGACY_PALETTE_ALIASES[cfg.palette_name]
        self._config = cfg
        self._updating = True
        try:
            self.combo_theme.setCurrentText(cfg.theme)
            self.combo_palette.setCurrentText(cfg.palette_name)
            self.combo_font.setCurrentText(cfg.font_family)
            self.spin_title_size.setValue(float(cfg.title_size))
            self.spin_label_size.setValue(float(cfg.label_size))
            self.spin_tick_size.setValue(float(cfg.tick_size))
            self.spin_legend_size.setValue(float(cfg.legend_size))
            self.spin_stat_size.setValue(float(cfg.stat_size))
            self.chk_bold_title.setChecked(bool(cfg.bold_title))
            self.chk_bold_labels.setChecked(bool(cfg.bold_labels))
            self.spin_tick_rotation.setValue(int(cfg.tick_rotation))
            self.edit_title.setText(cfg.title)
            self.edit_xlabel.setText(cfg.xlabel)
            self.edit_ylabel.setText(cfg.ylabel)
            self.edit_legend_title.setText(cfg.legend_title)
            self.chk_despine_top.setChecked(bool(cfg.despine_top))
            self.chk_despine_right.setChecked(bool(cfg.despine_right))
            self.spin_spine_width.setValue(float(cfg.spine_linewidth))
            self.combo_grid.setCurrentText(cfg.grid_mode)
            self.combo_grid_style.setCurrentText(cfg.grid_linestyle)
            self.spin_grid_alpha.setValue(float(cfg.grid_alpha))
            self.combo_size_preset.setCurrentText(cfg.size_preset)
            self.spin_width_mm.setValue(float(cfg.width_mm))
            self.spin_height_mm.setValue(float(cfg.height_mm))
            self.combo_dpi.setCurrentText(str(int(cfg.dpi)))
            self.chk_transparent.setChecked(bool(cfg.transparent))
            self.chk_hide_ns.setChecked(bool(cfg.hide_ns))
            self.combo_p_mode.setCurrentText(cfg.p_display_mode)
            self._sync_category_widgets_from_config()
        finally:
            self._updating = False

        if emit:
            self._emit_style_changed()

    def set_categories(
        self,
        categories: Sequence[str],
        default_colors: Optional[Dict[str, str]] = None,
    ) -> None:
        """Populate per-category color swatch pickers and inline label editors."""
        self._categories = list(dict.fromkeys(str(c) for c in categories))
        self._default_colors = dict(default_colors or {})
        self._rebuild_category_rows()

    def _rebuild_category_rows(self) -> None:
        self._category_rows.clear()
        while self._cat_rows_layout.count():
            item = self._cat_rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not self._categories:
            hint = QLabel("No categories detected for current plot.")
            style_section_header(hint, subtitle=True)
            self._cat_rows_layout.addWidget(hint)
            self.btn_reset_categories.setEnabled(False)
            return

        self.btn_reset_categories.setEnabled(True)
        palette = get_palette_colors(self.combo_palette.currentText(), len(self._categories))
        for idx, cat in enumerate(self._categories):
            row_widget = QWidget()
            row_widget.setObjectName("irisCardRow")
            row_widget.setStyleSheet("QWidget#irisCardRow { background-color: transparent; }")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            base_color = self._default_colors.get(cat, palette[idx])
            active_color = self._config.custom_colors.get(cat, base_color)

            swatch = QPushButton()
            swatch.setFixedSize(24, 22)
            swatch.setToolTip(f"Pick color for {cat}")
            self._style_swatch_button(swatch, active_color)
            swatch.clicked.connect(lambda _checked=False, c=cat: self._pick_category_color(c))

            lbl_name = QLabel(cat)
            lbl_name.setMinimumWidth(65)
            lbl_name.setMaximumWidth(100)
            lbl_name.setToolTip(cat)

            edit_lbl = QLineEdit()
            edit_lbl.setPlaceholderText(cat)
            if cat in self._config.custom_labels:
                edit_lbl.setText(self._config.custom_labels[cat])
            edit_lbl.textChanged.connect(self._on_control_changed)

            row_layout.addWidget(swatch)
            row_layout.addWidget(lbl_name)
            row_layout.addWidget(edit_lbl, 1)
            self._cat_rows_layout.addWidget(row_widget)
            self._category_rows[cat] = {
                "button": swatch,
                "line_edit": edit_lbl,
                "color": active_color,
            }

    def _refresh_category_swatches(self) -> None:
        if not self._categories:
            return
        palette = get_palette_colors(self.combo_palette.currentText(), len(self._categories))
        for idx, cat in enumerate(self._categories):
            row_info = self._category_rows.get(cat)
            if row_info is None:
                continue
            base_color = self._default_colors.get(cat, palette[idx])
            active_color = self._config.custom_colors.get(cat, base_color)
            row_info["color"] = active_color
            self._style_swatch_button(row_info["button"], active_color)

    def _sync_category_widgets_from_config(self) -> None:
        for cat, row_info in self._category_rows.items():
            edit = row_info["line_edit"]
            edit.setText(self._config.custom_labels.get(cat, ""))
        self._refresh_category_swatches()

    @staticmethod
    def _style_swatch_button(button: QPushButton, hex_color: str) -> None:
        button.setStyleSheet(
            f"background-color: {hex_color}; border: 1px solid #888888; border-radius: 3px;"
        )

    def _pick_category_color(self, category: str) -> None:
        row_info = self._category_rows.get(category)
        current_hex = row_info["color"] if row_info else "#3A6A9A"
        color = QColorDialog.getColor(
            QColor(current_hex),
            self,
            f"Select Color: {category}",
        )
        if color.isValid():
            self.set_category_color(category, color.name())

    def set_category_color(self, category: str, hex_color: str) -> None:
        """Programmatically set a custom color override for *category*."""
        self._config.custom_colors[category] = hex_color
        row_info = self._category_rows.get(category)
        if row_info is not None:
            row_info["color"] = hex_color
            self._style_swatch_button(row_info["button"], hex_color)
        if not self._updating:
            self._emit_style_changed()

    def set_category_label(self, category: str, label: str) -> None:
        """Programmatically set or clear an inline label override for *category*."""
        row_info = self._category_rows.get(category)
        if row_info is not None:
            row_info["line_edit"].setText(label)
            return
        if label.strip() and label.strip() != category:
            self._config.custom_labels[category] = label.strip()
        else:
            self._config.custom_labels.pop(category, None)
        if not self._updating:
            self._emit_style_changed()

    def _reset_category_overrides(self) -> None:
        self._config.custom_colors.clear()
        self._config.custom_labels.clear()
        self._updating = True
        try:
            for row_info in self._category_rows.values():
                row_info["line_edit"].clear()
            self._refresh_category_swatches()
        finally:
            self._updating = False
        self._emit_style_changed()


class PublicationPlotStudioDialog(QDialog):
    """
    Standalone publication figure studio window combining Matplotlib canvas,
    NavigationToolbar2QT, PublicationStyleSidebar, and optional stats table tab.
    """

    def __init__(
        self,
        plot_fn: Optional[Callable[[Figure, PlotStyleConfig], Any]] = None,
        *,
        summary_df: Optional[pd.DataFrame] = None,
        replicates_df: Optional[pd.DataFrame] = None,
        stats_df: Optional[pd.DataFrame] = None,
        categories: Optional[Sequence[str]] = None,
        config: Optional[PlotStyleConfig] = None,
        title: str = "IRIS Figure Studio",
        metadata: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1280, 820)
        self.setMinimumSize(900, 560)
        self.setWindowFlags(self.windowFlags() | Qt.Window)

        self.plot_fn = plot_fn
        self.summary_df = summary_df
        self.replicates_df = replicates_df
        self.stats_df = stats_df
        self.metadata = dict(metadata or {})
        self.config: PlotStyleConfig = config.copy() if config is not None else PlotStyleConfig()
        self._categories_explicit = categories is not None
        self._categories: List[str] = list(categories) if categories is not None else []
        self._stats_tab_added = False

        self.setStyleSheet(STUDIO_DARK_STYLESHEET)
        self._build_ui()
        self._populate_stats_table()
        self.render_plot()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(6)

        self.splitter = QSplitter(Qt.Horizontal)

        # Left/Center Pane: Figure Preview + Optional Statistics Table
        self.tabs = QTabWidget()
        configure_tab_widget(self.tabs)

        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        preview_layout.setContentsMargins(6, 6, 6, 6)
        preview_layout.setSpacing(4)

        self.figure = Figure(figsize=self.config.size_inches, dpi=100)
        canonical_theme = _LEGACY_THEME_ALIASES.get(self.config.theme, self.config.theme)
        preset = THEME_PRESETS.get(canonical_theme, THEME_PRESETS["IRIS Dark"])
        self.figure.patch.set_facecolor(preset["figure_facecolor"])

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.toolbar = None
        if NavigationToolbar is not None:
            self.toolbar = NavigationToolbar(self.canvas, preview_tab)
            self.toolbar.setStyleSheet(
                "QToolBar { background-color: #1f232b; border: 1px solid #3c4352; "
                "spacing: 4px; padding: 2px; } "
                "QToolButton { background-color: transparent; border: 1px solid transparent; "
                "border-radius: 3px; padding: 3px; } "
                "QToolButton:hover { background-color: #3A6A9A; border: 1px solid #4B82B5; }"
            )
            preview_layout.addWidget(self.toolbar)

        preview_layout.addWidget(self.canvas, 1)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #9AA4B2; font-size: 11px; padding: 2px 4px;")
        preview_layout.addWidget(self.lbl_status)

        self.tabs.addTab(preview_tab, "Figure Preview")

        # Optional Statistics Table Tab
        self.stats_tab = QWidget()
        stats_layout = QVBoxLayout(self.stats_tab)
        stats_layout.setContentsMargins(6, 6, 6, 6)
        self.stats_table = QTableWidget(0, 0)
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        stats_layout.addWidget(self.stats_table)

        self.splitter.addWidget(self.tabs)

        # Right Pane: Style & Export Sidebar
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

        main_layout.addWidget(self.splitter, 1)

    def _populate_stats_table(self) -> None:
        if self.stats_df is None:
            if self._stats_tab_added:
                idx = self.tabs.indexOf(self.stats_tab)
                if idx >= 0:
                    self.tabs.removeTab(idx)
                self._stats_tab_added = False
            return

        if not self._stats_tab_added:
            self.tabs.addTab(self.stats_tab, "Statistical Tests")
            self._stats_tab_added = True

        df = _prepare_export_dataframe(self.stats_df)
        self.stats_table.setRowCount(len(df))
        self.stats_table.setColumnCount(len(df.columns))
        self.stats_table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for r_idx, row_vals in enumerate(df.itertuples(index=False, name=None)):
            for c_idx, val in enumerate(row_vals):
                if isinstance(val, (float, np.floating)) and np.isfinite(float(val)):
                    text = f"{float(val):.4g}"
                else:
                    text = str(_sanitize_excel_value(val))
                self.stats_table.setItem(r_idx, c_idx, QTableWidgetItem(text))

        header = self.stats_table.horizontalHeader()
        resize_mode = (
            QHeaderView.Stretch if len(df.columns) <= 6 else QHeaderView.Interactive
        )
        header.setSectionResizeMode(resize_mode)

    def _on_style_changed(self, new_config: PlotStyleConfig) -> None:
        self.config = new_config.copy()
        self.render_plot()

    def render_plot(self) -> None:
        """Render or re-style the Matplotlib figure and update the Qt canvas."""
        if self.plot_fn is not None:
            self.figure.clear()
            self.plot_fn(self.figure, self.config)

        apply_plot_style(self.figure, self.figure.axes, self.config)

        if not self._categories_explicit and not self._categories and self.figure.axes:
            detected = self._detect_categories_from_axes(self.figure.axes[0])
            if detected:
                self._categories = detected
                self.sidebar.set_categories(detected)

        if self.figure.axes and not self.figure.get_constrained_layout():
            try:
                self.figure.tight_layout()
            except Exception:
                pass

        self.canvas.draw_idle()
        self.lbl_status.setText(
            f"{self.config.width_mm:.1f} x {self.config.height_mm:.1f} mm | "
            f"{self.config.dpi} DPI | Theme: {self.config.theme} | "
            f"Palette: {self.config.palette_name}"
        )

    @staticmethod
    def _detect_categories_from_axes(axis: matplotlib.axes.Axes) -> List[str]:
        leg = axis.get_legend()
        if leg is not None:
            labels = [
                str(getattr(t, "_iris_orig_text", t.get_text())).strip()
                for t in leg.get_texts()
            ]
            labels = [lbl for lbl in labels if lbl]
            if labels:
                return list(dict.fromkeys(labels))

        fmt = axis.xaxis.get_major_formatter()
        if isinstance(fmt, _CustomLabelFormatter):
            fmt = fmt.base_formatter
        if isinstance(fmt, mcategory.StrCategoryFormatter) and hasattr(fmt, "_units"):
            return [str(k) for k in fmt._units.keys()]
        return []

    def update_data(
        self,
        *,
        plot_fn: Optional[Callable[[Figure, PlotStyleConfig], Any]] = None,
        summary_df: Optional[pd.DataFrame] = None,
        replicates_df: Optional[pd.DataFrame] = None,
        stats_df: Optional[pd.DataFrame] = None,
        categories: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update plot callback or Source Data tables and re-render."""
        if plot_fn is not None:
            self.plot_fn = plot_fn
        if summary_df is not None:
            self.summary_df = summary_df
        if replicates_df is not None:
            self.replicates_df = replicates_df
        if stats_df is not None:
            self.stats_df = stats_df
            self._populate_stats_table()
        if categories is not None:
            self._categories_explicit = True
            self._categories = list(categories)
            self.sidebar.set_categories(self._categories)
        if metadata is not None:
            self.metadata = dict(metadata)
        self.render_plot()

    def copy_to_clipboard(self) -> bool:
        """Copy the styled figure at configured DPI into the system clipboard."""
        ok = copy_figure_to_clipboard(self.figure, config=self.config, dpi=self.config.dpi)
        if ok:
            self.lbl_status.setText(f"Copied figure to clipboard at {self.config.dpi} DPI.")
        else:
            self.lbl_status.setText("Clipboard unavailable in current headless environment.")
        return ok

    def export_figure_to_path(self, filepath: Union[str, Path]) -> Path:
        """Export the styled figure directly to *filepath* without prompting."""
        out = export_publication_figure(self.figure, filepath, self.config)
        self.lbl_status.setText(f"Exported figure to {out.name}.")
        return out

    def export_figure(self) -> Optional[Path]:
        """Prompt for an output path and export the styled figure."""
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
        """Export Source Data Excel workbook directly to *filepath*."""
        summary = self.summary_df if self.summary_df is not None else pd.DataFrame()
        meta: Dict[str, Any] = dict(self.metadata)
        meta.update({f"style.{k}": v for k, v in self.config.to_dict().items()})
        out = export_source_data_excel(
            filepath,
            summary_df=summary,
            replicates_df=self.replicates_df,
            stats_df=self.stats_df,
            metadata=meta,
        )
        self.lbl_status.setText(f"Exported Source Data workbook to {out.name}.")
        return out

    def export_source_data(self) -> Optional[Path]:
        """Prompt for an Excel path and export Source Data."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Source Data (Excel)",
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


FigureStudioDialog = PublicationPlotStudioDialog
FigureStyleSidebar = PublicationStyleSidebar


__all__ = [
    "DPI_OPTIONS",
    "FONT_FAMILIES",
    "FigureStudioDialog",
    "FigureStyleSidebar",
    "GRID_LINESTYLES",
    "GRID_MODES",
    "P_DISPLAY_MODES",
    "PlotStyleConfig",
    "PublicationPlotStudioDialog",
    "PublicationStyleSidebar",
    "SCIENTIFIC_PALETTES",
    "SIZE_PRESETS",
    "STUDIO_DARK_STYLESHEET",
    "THEME_PRESETS",
    "apply_plot_style",
    "copy_figure_to_clipboard",
    "draw_significance_brackets",
    "export_publication_figure",
    "export_source_data_excel",
    "format_significance_label",
    "get_palette_colors",
    "render_figure_to_png_bytes",
    "resolve_category_colors",
]
