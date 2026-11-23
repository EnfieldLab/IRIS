"""
IRIS Common Shared Themes & Publication Presets
===============================================
Consistent dark UI tokens matching the Control panel reference geometry,
independent window configuration helpers, and Matplotlib rcParams presets
shared across IRIS widgets.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

_DARK_THEME_BODY = """
    QDialog, QWidget {
        background-color: #1f232b;
        color: #E0E0E0;
        font-size: 12px;
    }
    QGroupBox {
        background-color: #232832;
        border: 1px solid #3c4352;
        border-radius: 8px;
        margin-top: 2px;
        padding: 6px;
        color: #E0E0E0;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        background-color: #232832;
        color: #F0F2F5;
        font-size: 12px;
        font-weight: bold;
        padding: 0px 6px;
        left: 10px;
    }
    QLabel {
        color: #E0E0E0;
        font-size: 12px;
        background-color: transparent;
    }
    QStackedWidget {
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
        color: #777777;
    }
    QPushButton:pressed {
        background-color: #2A5A7A;
    }
    QComboBox {
        background-color: #2A2E39;
        color: #F0F2F5;
        border: 1px solid #4B5364;
        border-radius: 4px;
        padding: 3px 6px;
        min-height: 22px;
        font-size: 12px;
    }
    QComboBox::drop-down {
        border: none;
    }
    QComboBox QAbstractItemView {
        background-color: #2A2E39;
        color: #F0F2F5;
        border: 1px solid #4B5364;
        selection-background-color: #3A6A9A;
        selection-color: #FFFFFF;
    }
    QSpinBox, QDoubleSpinBox {
        background-color: #2A2E39;
        color: #F0F2F5;
        border: 1px solid #4B5364;
        border-radius: 4px;
        min-height: 22px;
        padding: 1px 4px;
        font-size: 12px;
    }
    QLineEdit {
        background-color: #2A2E39;
        color: #F0F2F5;
        border: 1px solid #4B5364;
        border-radius: 4px;
        padding: 3px 6px;
        min-height: 22px;
        font-size: 12px;
    }
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
        border: 1px solid #4B82B5;
    }
    QCheckBox {
        color: #E0E0E0;
        font-size: 12px;
        spacing: 5px;
        background-color: transparent;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
        border-radius: 3px;
    }
    QCheckBox::indicator:unchecked {
        border: 1px solid #4B5364;
        background-color: #2A2E39;
    }
    QCheckBox::indicator:checked {
        border: 1px solid #3A6A9A;
        background-color: #3A6A9A;
    }
    QSlider::groove:horizontal {
        border: 1px solid #3c4352;
        height: 6px;
        background: #2A2E39;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #3A6A9A;
        border: 1px solid #4B82B5;
        width: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }
    QSlider::handle:horizontal:hover {
        background: #4A7FBA;
    }
    QProgressBar {
        border: 1px solid #4B5364;
        border-radius: 4px;
        text-align: center;
        background-color: #14161d;
        color: #F0F2F5;
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
    QTabWidget::pane {
        border: 1px solid #3c4352;
        border-radius: 4px;
        background-color: #14161d;
    }
    QTabBar::tab {
        background-color: #2A2E39;
        color: #C8D0DC;
        padding: 6px 14px;
        border: 1px solid #3c4352;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        font-weight: bold;
    }
    QTabBar::tab:hover {
        background-color: #343A48;
        color: #F0F2F5;
    }
    QTabBar::tab:selected {
        background-color: #14161d;
        color: #FFFFFF;
        border-top: 2px solid #3A6A9A;
    }
    QTabBar::scroller {
        width: 24px;
    }
    QTableWidget, QTreeWidget {
        background-color: #1b1f27;
        alternate-background-color: #222732;
        color: #E0E0E0;
        gridline-color: #333842;
        border: 1px solid #3c4352;
        border-radius: 6px;
    }
    QTableWidget::item {
        color: #E0E0E0;
        background-color: #1b1f27;
        padding: 4px;
    }
    QTableWidget::item:alternate {
        background-color: #222732;
        color: #E0E0E0;
    }
    QTableWidget::item:selected, QTreeWidget::item:selected {
        background-color: #3A6A9A;
        color: #FFFFFF;
    }
    QHeaderView::section {
        background-color: #2A2E39;
        color: #F0F2F5;
        padding: 4px 6px;
        border: 1px solid #3c4352;
        font-weight: bold;
    }
    QListWidget {
        background-color: #1b1f27;
        color: #E0E0E0;
        border: 1px solid #3c4352;
        border-radius: 6px;
        font-size: 12px;
    }
    QListWidget::item {
        padding: 4px;
        border-bottom: 1px solid #2A2E39;
    }
    QListWidget::item:hover {
        background-color: #2A2E39;
    }
    QListWidget::item:selected {
        background-color: #3A6A9A;
        color: #FFFFFF;
    }
    QMenu {
        background-color: #232832;
        color: #F0F2F5;
        border: 1px solid #4B5364;
        border-radius: 4px;
        padding: 4px;
    }
    QMenu::item {
        padding: 5px 18px 5px 8px;
        border-radius: 3px;
    }
    QMenu::item:selected {
        background-color: #3A6A9A;
        color: #FFFFFF;
    }
    QMenu::separator {
        height: 1px;
        background-color: #3c4352;
        margin: 4px 6px;
    }
    QFrame[frameShape="4"] {
        color: #3c4352;
        background-color: #3c4352;
        max-height: 1px;
    }
    QSplitter::handle {
        background-color: #2A2E39;
    }
"""

SHARED_DARK_THEME = _DARK_THEME_BODY

DIALOG_DARK_THEME = (
    "\n    /* IRIS unified dialog palette (canonicalized from legacy #1a1d24 to #1f232b) */"
    + _DARK_THEME_BODY
)

POPOUT_BUTTON_STYLE = """
    QPushButton {
        background-color: #2D5F8B;
        color: white;
        font-weight: bold;
        border: 1px solid #4B82B5;
        border-radius: 4px;
        padding: 3px 8px;
        font-size: 11px;
    }
    QPushButton:hover {
        background-color: #3D75A8;
    }
    QPushButton:disabled {
        background-color: #2A3A4A;
        color: #777777;
        border-color: #3c4352;
    }
"""

COLOR_PRIMARY = "#3A6A9A"
COLOR_PRIMARY_HOVER = "#4A7FBA"
COLOR_STUDIO = "#2D5F8B"
COLOR_STUDIO_BORDER = "#4B82B5"
COLOR_STUDIO_HOVER = "#3D75A8"
COLOR_CTA = "#2D6A4F"
COLOR_CTA_BORDER = "#40916C"
COLOR_CTA_HOVER = "#40916C"
COLOR_DANGER = "#CC4444"
COLOR_DANGER_BASE = "#8B3A3A"
COLOR_SUCCESS = "#2D6A4F"
COLOR_WARNING = "#CCAA44"
COLOR_MUTED = "#9AA4B2"
COLOR_CARD_BG = "#232832"
COLOR_CARD_BORDER = "#3c4352"
COLOR_WELL_BG = "#14161d"
COLOR_INPUT_BG = "#2A2E39"
COLOR_INPUT_BORDER = "#4B5364"
COLOR_TEXT_HEADER = "#E0E0E0"

CONFIDENCE_HIGH_COLOR = "#44AA66"
CONFIDENCE_MEDIUM_COLOR = "#CCAA44"
CONFIDENCE_LOW_COLOR = "#CC4444"

STATUS_STYLES: Dict[str, str] = {
    "Idle": "color: #9AA4B2; font-style: italic;",
    "Queued": "color: #CCAA44; font-weight: bold;",
    "Running": "color: #44AAFF; font-weight: bold;",
    "Done": "color: #44AA66; font-weight: bold;",
    "Failed": "color: #CC4444; font-weight: bold;",
}

SECTION_TITLE_STYLE = (
    "color: #E0E0E0; font-size: 14px; font-weight: bold; "
    "margin: 0px; padding: 0px; background-color: transparent;"
)

SECTION_SUBTITLE_STYLE = (
    "color: #9AA4B2; font-size: 11px; font-style: italic; "
    "margin: 0px; padding: 0px; background-color: transparent;"
)

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
    QPushButton:disabled { background-color: #2A3A4A; color: #777777; }
"""

BUTTON_STUDIO = """
    QPushButton {
        background-color: #2D5F8B;
        color: white;
        font-weight: bold;
        border: 1px solid #4B82B5;
        border-radius: 4px;
        padding: 5px 12px;
    }
    QPushButton:hover { background-color: #3D75A8; }
    QPushButton:disabled { background-color: #2A3A4A; color: #777777; border-color: #3c4352; }
"""

BUTTON_CTA = """
    QPushButton {
        background-color: #2D6A4F;
        color: white;
        font-weight: bold;
        border: 1px solid #40916C;
        border-radius: 4px;
        padding: 5px 12px;
    }
    QPushButton:hover { background-color: #40916C; }
    QPushButton:disabled { background-color: #2A3A4A; color: #777777; border-color: #3c4352; }
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
    QPushButton:disabled { background-color: #2A3A4A; color: #777777; border-color: #3c4352; }
"""

BUTTON_DANGER = """
    QPushButton {
        background-color: #8B3A3A;
        color: white;
        font-weight: bold;
        border: 1px solid #CC4444;
        border-radius: 4px;
        padding: 5px 10px;
    }
    QPushButton:hover { background-color: #CC4444; }
    QPushButton:disabled { background-color: #2A3A4A; color: #777777; border-color: #3c4352; }
"""

BUTTON_SUCCESS = """
    QPushButton {
        background-color: #2D6A4F;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        padding: 5px 10px;
        border: none;
    }
    QPushButton:hover { background-color: #40916C; }
    QPushButton:disabled { background-color: #2A3A4A; color: #777777; }
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
    "Classic White": {
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
    "Boxed White": {
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


def configure_tab_widget(tab_widget: Any) -> Any:
    """Prevent QTabBar word truncation/elision and sync bold font metrics."""
    from qtpy.QtCore import Qt

    bar = tab_widget.tabBar()
    if bar is not None:
        bar.setElideMode(Qt.ElideNone)
        bar.setExpanding(False)
        bar.setUsesScrollButtons(True)
        font = bar.font()
        font.setBold(True)
        bar.setFont(font)
    return tab_widget


def style_section_header(label: Any, *, subtitle: bool = False) -> Any:
    """Apply canonical IRIS section-card header or muted subtitle styling to a QLabel."""
    label.setStyleSheet(SECTION_SUBTITLE_STYLE if subtitle else SECTION_TITLE_STYLE)
    return label


def create_section_card(
    title: str = "",
    subtitle: str = "",
    parent: Optional[Any] = None,
    *,
    margins: Tuple[int, int, int, int] = (10, 6, 10, 10),
    spacing: int = 8,
) -> Tuple[Any, Any]:
    """Create an untitled ``QGroupBox`` card with Control-reference spacing and headers."""
    from qtpy.QtWidgets import QGroupBox, QLabel, QVBoxLayout

    card = QGroupBox(parent)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)

    if title:
        lbl_title = QLabel(title, card)
        style_section_header(lbl_title, subtitle=False)
        layout.addWidget(lbl_title)
    if subtitle:
        lbl_sub = QLabel(subtitle, card)
        style_section_header(lbl_sub, subtitle=True)
        lbl_sub.setWordWrap(True)
        layout.addWidget(lbl_sub)

    return card, layout


__all__ = [
    "SHARED_DARK_THEME",
    "DIALOG_DARK_THEME",
    "POPOUT_BUTTON_STYLE",
    "PUBLICATION_RC_PRESETS",
    "SECTION_TITLE_STYLE",
    "SECTION_SUBTITLE_STYLE",
    "COLOR_PRIMARY",
    "COLOR_PRIMARY_HOVER",
    "COLOR_STUDIO",
    "COLOR_STUDIO_BORDER",
    "COLOR_STUDIO_HOVER",
    "COLOR_CTA",
    "COLOR_CTA_BORDER",
    "COLOR_CTA_HOVER",
    "COLOR_DANGER",
    "COLOR_DANGER_BASE",
    "COLOR_SUCCESS",
    "COLOR_WARNING",
    "COLOR_MUTED",
    "COLOR_CARD_BG",
    "COLOR_CARD_BORDER",
    "COLOR_WELL_BG",
    "COLOR_INPUT_BG",
    "COLOR_INPUT_BORDER",
    "COLOR_TEXT_HEADER",
    "CONFIDENCE_HIGH_COLOR",
    "CONFIDENCE_MEDIUM_COLOR",
    "CONFIDENCE_LOW_COLOR",
    "STATUS_STYLES",
    "BUTTON_PRIMARY",
    "BUTTON_STUDIO",
    "BUTTON_CTA",
    "BUTTON_SECONDARY",
    "BUTTON_DANGER",
    "BUTTON_SUCCESS",
    "DEFAULT_MARKER_COLORS",
    "DEFAULT_CELLTYPE_PALETTE",
    "configure_independent_window",
    "configure_tab_widget",
    "style_section_header",
    "create_section_card",
]
