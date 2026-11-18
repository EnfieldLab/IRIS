"""
IRIS Common Shared Themes & Publication Presets
===============================================
Consistent dark UI tokens, independent window configuration helpers, and
journal-grade Matplotlib rcParams presets shared across IRIS widgets.
"""
from __future__ import annotations

from typing import Any, Dict

SHARED_DARK_THEME = """
    QWidget {
        background-color: #1f232b;
        color: #E0E0E0;
        font-size: 12px;
    }
    QGroupBox {
        border: 1px solid #444444;
        border-radius: 8px;
        margin-top: 6px;
        padding-top: 6px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        color: #DDD;
        font-size: 12px;
        font-weight: bold;
        padding-left: 8px;
        padding-right: 8px;
    }
    QLabel {
        color: #DDD;
        font-size: 12px;
        background-color: transparent;
    }
    QPushButton {
        background-color: #3A6A9A;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        padding: 5px 10px;
        border: none;
        font-size: 12px;
    }
    QPushButton:hover {
        background-color: #4A7FBA;
    }
    QPushButton:disabled {
        background-color: #2A3A4A;
        color: #777;
    }
    QPushButton:pressed {
        background-color: #2A5A7A;
    }
    QComboBox {
        background-color: #2A2A2A;
        color: white;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 3px 6px;
        height: 22px;
    }
    QComboBox::drop-down {
        border: none;
    }
    QSpinBox, QDoubleSpinBox {
        background-color: #2A2A2A;
        color: white;
        border: 1px solid #555;
        border-radius: 4px;
        height: 22px;
        padding: 1px 4px;
    }
    QLineEdit {
        background-color: #2A2A2A;
        color: white;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 3px 6px;
        height: 22px;
    }
    QCheckBox {
        color: #DDD;
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
        border-radius: 3px;
    }
    QCheckBox::indicator:unchecked {
        border: 1px solid #555;
        background-color: #2A2A2A;
    }
    QCheckBox::indicator:checked {
        border: 1px solid #3A6A9A;
        background-color: #3A6A9A;
    }
    QSlider::groove:horizontal {
        border: 1px solid #444;
        height: 6px;
        background: #2A2A2A;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #3A6A9A;
        border: 1px solid #555;
        width: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }
    QSlider::handle:horizontal:hover {
        background: #4A7FBA;
    }
    QProgressBar {
        border: 1px solid #555;
        border-radius: 4px;
        text-align: center;
        background-color: #2A2A2A;
        color: white;
        height: 18px;
    }
    QProgressBar::chunk {
        background-color: #3A6A9A;
        border-radius: 3px;
    }
    QScrollArea {
        border: none;
        background-color: #1f232b;
    }
    QTableWidget {
        background-color: #1e2128;
        alternate-background-color: #262a33;
        color: #E0E0E0;
        gridline-color: #333842;
        border: 1px solid #444;
        border-radius: 6px;
    }
    QTableWidget::item {
        color: #E0E0E0;
        background-color: #1e2128;
        padding: 4px;
    }
    QTableWidget::item:alternate {
        background-color: #262a33;
        color: #E0E0E0;
    }
    QTableWidget::item:selected {
        background-color: #3A6A9A;
        color: white;
    }
    QHeaderView::section {
        background-color: #2A2E39;
        color: white;
        padding: 4px;
        border: 1px solid #3c424f;
        font-weight: bold;
    }
    QListWidget {
        background-color: #2A2A2A;
        color: white;
        border: 1px solid #555;
        border-radius: 6px;
        font-size: 12px;
    }
    QListWidget::item {
        padding: 4px;
        border-bottom: 1px solid #333;
    }
    QListWidget::item:hover {
        background-color: #333842;
    }
    QListWidget::item:selected {
        background-color: #3A6A9A;
        color: white;
    }
    QFrame[frameShape="4"] {
        color: #555555;
        background-color: #555555;
        max-height: 1px;
    }
"""

DIALOG_DARK_THEME = """
    QDialog {
        background-color: #1a1d24;
        color: #E0E0E0;
        font-size: 12px;
    }
    QGroupBox {
        border: 1px solid #444;
        border-radius: 6px;
        margin-top: 6px;
        padding-top: 6px;
        font-weight: bold;
        color: #DDD;
    }
    QPushButton {
        background-color: #2D5F8B;
        color: white;
        font-weight: bold;
        border: 1px solid #4B82B5;
        border-radius: 3px;
        padding: 5px 12px;
        font-size: 12px;
    }
    QPushButton:hover {
        background-color: #3D75A8;
    }
    QPushButton:disabled {
        background-color: #2A3A4A;
        color: #777;
    }
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
        background-color: #2A2A2A;
        color: white;
        border: 1px solid #555;
        border-radius: 3px;
        padding: 3px 6px;
        font-size: 12px;
    }
    QLabel {
        color: #DDD;
        font-size: 12px;
    }
    QCheckBox {
        color: #DDD;
        font-size: 12px;
    }
    QListWidget {
        background-color: #14161d;
        color: white;
        border: 1px solid #444;
        border-radius: 3px;
        font-size: 12px;
    }
    QListWidget::item:selected {
        background-color: #3A6A9A;
    }
"""

POPOUT_BUTTON_STYLE = """
    QPushButton {
        background-color: #2D5F8B;
        color: white;
        font-weight: bold;
        border: 1px solid #4B82B5;
        border-radius: 3px;
        padding: 3px 8px;
        font-size: 11px;
    }
    QPushButton:hover {
        background-color: #3D75A8;
    }
"""

COLOR_PRIMARY = "#3A6A9A"
COLOR_PRIMARY_HOVER = "#4A7FBA"
COLOR_DANGER = "#CC4444"
COLOR_SUCCESS = "#44AA66"
COLOR_WARNING = "#CCAA44"
COLOR_MUTED = "#888888"

CONFIDENCE_HIGH_COLOR = "#44AA66"
CONFIDENCE_MEDIUM_COLOR = "#CCAA44"
CONFIDENCE_LOW_COLOR = "#CC4444"

STATUS_STYLES: Dict[str, str] = {
    "Idle": "color: #888; font-style: italic;",
    "Queued": "color: #CCAA44; font-weight: bold;",
    "Running": "color: #44AAFF; font-weight: bold;",
    "Done": "color: #44AA66; font-weight: bold;",
    "Failed": "color: #CC4444; font-weight: bold;",
}

BUTTON_PRIMARY = """
    QPushButton {
        background-color: #3A6A9A;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        padding: 5px 10px;
        border: none;
    }
    QPushButton:hover { background-color: #4A7FBA; }
"""

BUTTON_SECONDARY = """
    QPushButton {
        background-color: #2D333F;
        color: #E0E0E0;
        font-weight: bold;
        border-radius: 4px;
        padding: 5px 10px;
        border: 1px solid #444C5C;
    }
    QPushButton:hover { background-color: #3D4454; }
"""

BUTTON_DANGER = """
    QPushButton {
        background-color: #CC4444;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        padding: 5px 10px;
        border: none;
    }
    QPushButton:hover { background-color: #DD5555; }
"""

BUTTON_SUCCESS = """
    QPushButton {
        background-color: #44AA66;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        padding: 5px 10px;
        border: none;
    }
    QPushButton:hover { background-color: #55BB77; }
"""

DEFAULT_MARKER_COLORS: Dict[str, str] = {
    "DAPI": "blue",
    "HOECHST": "blue",
    "ASMA": "green",
    "SMA": "green",
    "ACTA2": "green",
    "CD45": "purple",
    "PTPRC": "purple",
    "CK": "red",
    "PANCK": "red",
    "KRT": "red",
    "CD3": "cyan",
    "CD4": "magenta",
    "CD8": "yellow",
    "CD20": "orange",
    "CD68": "bop orange",
    "CD163": "bop purple",
    "PDL1": "bop blue",
    "KI67": "gray",
    "MKI67": "gray",
}

DEFAULT_CELLTYPE_PALETTE: list[str] = [
    "#4477EE",
    "#44AA44",
    "#EE4444",
    "#FFAA22",
    "#AA44EE",
    "#22AACC",
    "#FF6688",
    "#88CC44",
    "#AA8822",
    "#44CCAA",
    "#CC4488",
    "#6688FF",
    "#FFCC44",
    "#44DDFF",
    "#FF8844",
]

PUBLICATION_RC_PRESETS: Dict[str, Dict[str, Any]] = {
    "IRIS Dark": {
        "figure.facecolor": "#1f232b",
        "axes.facecolor": "#1a1d24",
        "axes.edgecolor": "#777777",
        "axes.labelcolor": "#E0E0E0",
        "axes.titlecolor": "#FFFFFF",
        "xtick.color": "#CCCCCC",
        "ytick.color": "#CCCCCC",
        "text.color": "#E0E0E0",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
    },
    "Publication White (Nature)": {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#000000",
        "axes.linewidth": 0.8,
        "axes.labelcolor": "#000000",
        "axes.titlecolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "text.color": "#000000",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    },
    "Cell Press Boxed": {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#111111",
        "axes.linewidth": 0.9,
        "axes.labelcolor": "#111111",
        "axes.titlecolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "text.color": "#111111",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    },
    "Minimal Grid": {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.75,
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
    },
}


def configure_independent_window(dialog: Any) -> Any:
    """Apply ``Qt.Window`` flag lazily so pop-out dialogs behave as independent windows."""
    from qtpy.QtCore import Qt

    dialog.setWindowFlags(dialog.windowFlags() | Qt.Window)
    return dialog


__all__ = [
    "SHARED_DARK_THEME",
    "DIALOG_DARK_THEME",
    "POPOUT_BUTTON_STYLE",
    "PUBLICATION_RC_PRESETS",
    "COLOR_PRIMARY",
    "COLOR_PRIMARY_HOVER",
    "COLOR_DANGER",
    "COLOR_SUCCESS",
    "COLOR_WARNING",
    "COLOR_MUTED",
    "CONFIDENCE_HIGH_COLOR",
    "CONFIDENCE_MEDIUM_COLOR",
    "CONFIDENCE_LOW_COLOR",
    "STATUS_STYLES",
    "BUTTON_PRIMARY",
    "BUTTON_SECONDARY",
    "BUTTON_DANGER",
    "BUTTON_SUCCESS",
    "DEFAULT_MARKER_COLORS",
    "DEFAULT_CELLTYPE_PALETTE",
    "configure_independent_window",
]
