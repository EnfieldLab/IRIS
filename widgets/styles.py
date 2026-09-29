"""
IRIS-Viewer Shared Styles
Consistent dark theme tokens and stylesheet fragments, extended from the IRIS design language.
All widgets in IRIS-Viewer should apply these stylesheets for visual consistency.
"""

SHARED_DARK_THEME = """
    QWidget {
        background-color: #1f232b;
        color: #E0E0E0;
        font-size: 12px;
        border-radius: 0px;
    }
    QGroupBox {
        border: 1px solid #444444;
        margin-top: 8px;
        border-radius: 0px;
        padding-top: 8px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        color: #DDD;
        font-size: 12px;
        font-weight: bold;
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
        border-radius: 0px;
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
        border-radius: 0px;
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
        border-radius: 0px;
        height: 22px;
        padding: 1px 4px;
    }
    QCheckBox {
        color: #DDD;
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
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
        border-radius: 0px;
        text-align: center;
        background-color: #2A2A2A;
        color: white;
        height: 18px;
    }
    QProgressBar::chunk {
        background-color: #3A6A9A;
    }
    QScrollArea {
        border: none;
        background-color: #1f232b;
    }
    QTableWidget {
        background-color: #2A2A2A;
        color: white;
        gridline-color: #444;
        border: 1px solid #555;
    }
    QHeaderView::section {
        background-color: #333;
        color: white;
        padding: 4px;
        border: 1px solid #444;
        font-weight: bold;
    }
    QListWidget {
        background-color: #2A2A2A;
        color: white;
        border: 1px solid #555;
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

# ── Accent Colors ────────────────────────────────────────────────────────────

COLOR_PRIMARY = "#3A6A9A"
COLOR_PRIMARY_HOVER = "#4A7FBA"
COLOR_DANGER = "#CC4444"
COLOR_SUCCESS = "#44AA66"
COLOR_WARNING = "#CCAA44"
COLOR_MUTED = "#888888"

CONFIDENCE_HIGH_COLOR = "#44AA66"     # Green
CONFIDENCE_MEDIUM_COLOR = "#CCAA44"   # Yellow/Amber
CONFIDENCE_LOW_COLOR = "#CC4444"      # Red

# ── Status Badge Styles ─────────────────────────────────────────────────────

STATUS_STYLES = {
    'Idle':     "color: #888; font-style: italic;",
    'Queued':   "color: #CCAA44; font-weight: bold;",
    'Running':  "color: #44AAFF; font-weight: bold;",
    'Done':     "color: #44AA66; font-weight: bold;",
    'Failed':   "color: #CC4444; font-weight: bold;",
}

# ── Button Variant Styles ───────────────────────────────────────────────────

BUTTON_DANGER = """
    QPushButton {
        background-color: #CC4444;
        color: white;
        font-weight: bold;
        border-radius: 0px;
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
        border-radius: 0px;
        padding: 5px 10px;
        border: none;
    }
    QPushButton:hover { background-color: #55BB77; }
"""
