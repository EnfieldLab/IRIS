"""
IRIS-Viewer Feature Extraction Panel Widget
Controls for selecting which morphological and intensity features to compute,
previewing extracted data, and exporting the results.
"""
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QGroupBox, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QScrollArea, QSizePolicy
)
from qtpy.QtCore import Qt, Signal
import pandas as pd
from widgets.styles import SHARED_DARK_THEME, BUTTON_SUCCESS
from engine.feature_extraction import MORPHOLOGY_FEATURES, INTENSITY_FEATURES


class FeaturePanelWidget(QWidget):
    """Dock widget for cell feature extraction configuration and data preview."""

    # ── Signals ──────────────────────────────────────────────────────
    extractTriggered = Signal(list, list)   # selected_morphology, selected_intensity
    exportTriggered = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(SHARED_DARK_THEME)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        
        self.content_widget = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(6)
        self.content_widget.setLayout(self.main_layout)
        
        self.scroll_area.setWidget(self.content_widget)
        
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.scroll_area)
        self.setLayout(outer_layout)

        self._build_morphology_group()
        self._build_intensity_group()
        self._build_run_controls()
        self.main_layout.addStretch()

    # ── Morphology Features ──────────────────────────────────────────
    def _build_morphology_group(self):
        group = QGroupBox()
        gl = QVBoxLayout()
        gl.setContentsMargins(5, 5, 5, 5)
        gl.setSpacing(4)
        group.setLayout(gl)

        title = QLabel("Morphological Features")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        gl.addWidget(title)

        self._add_separator(gl)

        # Select All / None
        btn_row = QHBoxLayout()
        btn_sel_all = QPushButton("All")
        btn_sel_all.setFixedWidth(50)
        btn_sel_all.clicked.connect(lambda: self._toggle_morph_checks(True))
        btn_sel_none = QPushButton("None")
        btn_sel_none.setFixedWidth(50)
        btn_sel_none.clicked.connect(lambda: self._toggle_morph_checks(False))
        btn_row.addWidget(btn_sel_all)
        btn_row.addWidget(btn_sel_none)
        btn_row.addStretch()
        gl.addLayout(btn_row)

        self.morph_checkboxes = {}
        for key, display_name in MORPHOLOGY_FEATURES.items():
            chk = QCheckBox(display_name)
            chk.setChecked(True)
            chk.setProperty("feature_key", key)
            self.morph_checkboxes[key] = chk
            gl.addWidget(chk)

        self.main_layout.addWidget(group)

    # ── Intensity Features ───────────────────────────────────────────
    def _build_intensity_group(self):
        group = QGroupBox()
        gl = QVBoxLayout()
        gl.setContentsMargins(5, 5, 5, 5)
        gl.setSpacing(4)
        group.setLayout(gl)

        title = QLabel("Marker Intensity Features")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        gl.addWidget(title)

        self._add_separator(gl)

        info = QLabel("Select which intensity statistics to compute per channel.\n"
                       "Each selected stat generates one column per marker.")
        info.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        info.setWordWrap(True)
        gl.addWidget(info)

        # Select All / None
        btn_row = QHBoxLayout()
        btn_sel_all = QPushButton("All")
        btn_sel_all.setFixedWidth(50)
        btn_sel_all.clicked.connect(lambda: self._toggle_intensity_checks(True))
        btn_sel_none = QPushButton("None")
        btn_sel_none.setFixedWidth(50)
        btn_sel_none.clicked.connect(lambda: self._toggle_intensity_checks(False))
        btn_row.addWidget(btn_sel_all)
        btn_row.addWidget(btn_sel_none)
        btn_row.addStretch()
        gl.addLayout(btn_row)

        self.intensity_checkboxes = {}
        intensity_display = {
            'mean': 'Mean Intensity',
            'median': 'Median Intensity',
            'min': 'Min Intensity',
            'max': 'Max Intensity',
            'sum': 'Sum Intensity',
        }
        for key in INTENSITY_FEATURES:
            display = intensity_display.get(key, key.title())
            chk = QCheckBox(display)
            chk.setChecked(True)
            self.intensity_checkboxes[key] = chk
            gl.addWidget(chk)

        self.main_layout.addWidget(group)

    # ── Run Controls ─────────────────────────────────────────────────
    def _build_run_controls(self):
        row = QHBoxLayout()

        self.btn_extract = QPushButton("Extract Features")
        self.btn_extract.setStyleSheet(
            "QPushButton { background-color: #3A6A9A; color: white; font-weight: bold; "
            "padding: 8px; font-size: 13px; } QPushButton:hover { background-color: #4A7FBA; }"
        )
        self.btn_extract.clicked.connect(self._emit_extract_signal)
        row.addWidget(self.btn_extract)

        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setStyleSheet(BUTTON_SUCCESS)
        self.btn_export.clicked.connect(lambda: self.exportTriggered.emit())
        self.btn_export.setEnabled(False)
        row.addWidget(self.btn_export)

        self.main_layout.addLayout(row)

        self.lbl_extract_status = QLabel("No features extracted yet")
        self.lbl_extract_status.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        self.main_layout.addWidget(self.lbl_extract_status)



    # ── Utility ──────────────────────────────────────────────────────
    def _add_separator(self, layout):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #555555; background-color: #555555; max-height: 1px; margin-bottom: 4px;")
        layout.addWidget(line)

    def _toggle_morph_checks(self, state: bool):
        for chk in self.morph_checkboxes.values():
            chk.setChecked(state)

    def _toggle_intensity_checks(self, state: bool):
        for chk in self.intensity_checkboxes.values():
            chk.setChecked(state)

    def _emit_extract_signal(self):
        morph = [k for k, chk in self.morph_checkboxes.items() if chk.isChecked()]
        intensity = [k for k, chk in self.intensity_checkboxes.items() if chk.isChecked()]
        self.extractTriggered.emit(morph, intensity)

    def populate_preview(self, df: pd.DataFrame, max_rows: int = 50):
        """Update the extraction status without showing a preview table."""
        if df is None or df.empty:
            self.lbl_extract_status.setText("No data extracted")
            return

        n_rows, n_cols = df.shape

        self.lbl_extract_status.setText(f"Extracted {len(df):,} cells × {n_cols} features")
        self.lbl_extract_status.setStyleSheet("color: #44AA66; font-weight: bold; font-size: 11px;")
        self.btn_export.setEnabled(True)

    def set_extracting_state(self, is_extracting: bool):
        """Toggle between extracting and idle UI states."""
        self.btn_extract.setEnabled(not is_extracting)
        if is_extracting:
            self.lbl_extract_status.setText("Extracting features...")
            self.lbl_extract_status.setStyleSheet("color: #44AAFF; font-weight: bold; font-size: 11px;")
