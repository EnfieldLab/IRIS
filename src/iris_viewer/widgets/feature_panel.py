"""
IRIS-Viewer Feature Extraction Panel Widget
Controls for selecting mask source, sample scope, cellular compartments,
morphological and intensity features to compute, previewing extracted data,
and exporting results.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from iris_common.widgets.sample_scope import SCOPE_ACTIVE, SampleScopeBar
from iris_viewer.engine.feature_extraction import INTENSITY_FEATURES, MORPHOLOGY_FEATURES
from iris_viewer.widgets.styles import (
    BUTTON_CTA,
    BUTTON_PRIMARY,
    BUTTON_SECONDARY,
    SECTION_SUBTITLE_STYLE,
    SHARED_DARK_THEME,
    create_section_card,
)

COMPARTMENT_OPTIONS: Dict[str, str] = {
    "whole_cell": "Whole-Cell",
    "nuclear": "Nuclear",
    "cytoplasm": "Cytoplasmic Ring",
    "membrane": "Membrane Ring",
    "nc_ratio": "N:C Ratio",
}


class FeaturePanelWidget(QWidget):
    """Dock widget for cell feature extraction configuration and data preview."""

    # Signals
    extractTriggered = Signal(list, list)   # selected_morphology, selected_intensity
    exportTriggered = Signal()
    dataExtracted = Signal(object)          # df
    maskSourceChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(SHARED_DARK_THEME)
        self.cached_df = None

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        self.content_widget = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        self.content_widget.setLayout(self.main_layout)

        self.scroll_area.setWidget(self.content_widget)

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.scroll_area)
        self.setLayout(outer_layout)

        self._build_mask_and_compartments_group()
        self._build_morphology_group()
        self._build_intensity_group()
        self._build_run_controls()
        self.main_layout.addStretch()

    def _build_mask_and_compartments_group(self) -> None:
        group, gl = create_section_card("Mask Source & Compartments")

        row_mask = QHBoxLayout()
        row_mask.setSpacing(8)
        row_mask.addWidget(QLabel("Mask Source:"))
        self.combo_active_mask = QComboBox()
        self.combo_active_mask.currentTextChanged.connect(
            lambda txt: self.maskSourceChanged.emit(txt) if txt else None
        )
        row_mask.addWidget(self.combo_active_mask, stretch=1)
        gl.addLayout(row_mask)

        self.sample_scope_bar = SampleScopeBar(
            default_mode=SCOPE_ACTIVE,
            label="Scope:",
            compact=True,
        )
        gl.addWidget(self.sample_scope_bar)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_all = QPushButton("All")
        btn_all.setFixedWidth(56)
        btn_all.setStyleSheet(BUTTON_SECONDARY)
        btn_all.clicked.connect(lambda: self._toggle_compartment_checks(True))
        btn_none = QPushButton("None")
        btn_none.setFixedWidth(56)
        btn_none.setStyleSheet(BUTTON_SECONDARY)
        btn_none.clicked.connect(lambda: self._toggle_compartment_checks(False))
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addStretch()
        gl.addLayout(btn_row)

        self.compartment_checkboxes: Dict[str, QCheckBox] = {}
        for key, display_name in COMPARTMENT_OPTIONS.items():
            chk = QCheckBox(display_name)
            chk.setChecked(key == "whole_cell")
            chk.setProperty("compartment_key", key)
            self.compartment_checkboxes[key] = chk
            gl.addWidget(chk)

        self.main_layout.addWidget(group)

    # Morphology Features
    def _build_morphology_group(self) -> None:
        group, gl = create_section_card("Morphological Features")

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_sel_all = QPushButton("All")
        btn_sel_all.setFixedWidth(56)
        btn_sel_all.setStyleSheet(BUTTON_SECONDARY)
        btn_sel_all.clicked.connect(lambda: self._toggle_morph_checks(True))
        btn_sel_none = QPushButton("None")
        btn_sel_none.setFixedWidth(56)
        btn_sel_none.setStyleSheet(BUTTON_SECONDARY)
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

    # Intensity Features
    def _build_intensity_group(self) -> None:
        group, gl = create_section_card(
            "Marker Intensity Features",
            "Select which intensity statistics to compute per channel. "
            "Each selected stat generates one column per marker.",
        )

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_sel_all = QPushButton("All")
        btn_sel_all.setFixedWidth(56)
        btn_sel_all.setStyleSheet(BUTTON_SECONDARY)
        btn_sel_all.clicked.connect(lambda: self._toggle_intensity_checks(True))
        btn_sel_none = QPushButton("None")
        btn_sel_none.setFixedWidth(56)
        btn_sel_none.setStyleSheet(BUTTON_SECONDARY)
        btn_sel_none.clicked.connect(lambda: self._toggle_intensity_checks(False))
        btn_row.addWidget(btn_sel_all)
        btn_row.addWidget(btn_sel_none)
        btn_row.addStretch()
        gl.addLayout(btn_row)

        self.intensity_checkboxes = {}
        intensity_display = {
            "mean": "Mean Intensity",
            "median": "Median Intensity",
            "min": "Min Intensity",
            "max": "Max Intensity",
            "sum": "Sum Intensity",
        }
        for key in INTENSITY_FEATURES:
            display = intensity_display.get(key, key.title())
            chk = QCheckBox(display)
            chk.setChecked(True)
            self.intensity_checkboxes[key] = chk
            gl.addWidget(chk)

        self.main_layout.addWidget(group)

    # Run Controls
    def _build_run_controls(self) -> None:
        row = QHBoxLayout()
        row.setSpacing(8)

        self.btn_extract = QPushButton("Extract Features")
        self.btn_extract.setMinimumHeight(30)
        self.btn_extract.setStyleSheet(BUTTON_CTA)
        self.btn_extract.clicked.connect(self._emit_extract_signal)
        row.addWidget(self.btn_extract)

        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setMinimumHeight(30)
        self.btn_export.setStyleSheet(BUTTON_PRIMARY)
        self.btn_export.clicked.connect(lambda: self.exportTriggered.emit())
        self.btn_export.setEnabled(False)
        row.addWidget(self.btn_export)

        self.main_layout.addLayout(row)

        self.lbl_extract_status = QLabel("No features extracted yet")
        self.lbl_extract_status.setStyleSheet(SECTION_SUBTITLE_STYLE)
        self.main_layout.addWidget(self.lbl_extract_status)

    # Utility
    def _add_separator(self, layout):
        pass

    def _toggle_compartment_checks(self, state: bool) -> None:
        for chk in self.compartment_checkboxes.values():
            chk.setChecked(state)

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

    def set_samples(
        self,
        all_samples: Sequence[str],
        active_samples: Optional[Sequence[str]] = None,
    ) -> None:
        """Synchronize available and active samples on the Feature Scope selector."""
        self.sample_scope_bar.set_samples(all_samples, active_samples)

    def get_target_samples(
        self,
        all_samples: Optional[Sequence[str]] = None,
        active_samples: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Sync optional sample lists and resolve target sample names for feature extraction."""
        if all_samples is not None:
            self.sample_scope_bar.set_samples(all_samples, active_samples)
        elif active_samples is not None:
            self.sample_scope_bar.set_active_samples(active_samples)
        return self.sample_scope_bar.resolve_samples()

    def get_selected_compartments(self) -> List[str]:
        """Return list of checked compartment identifiers."""
        return [k for k, chk in self.compartment_checkboxes.items() if chk.isChecked()]

    def get_selected_mask_source(self) -> str:
        """Return currently selected mask source name."""
        return self.combo_active_mask.currentText().strip()

    def update_mask_sources(
        self, methods: List[str], active: Optional[str] = None
    ) -> None:
        """Synchronize Mask Source dropdown options with available segmentation methods."""
        clean = [str(m) for m in methods if str(m).strip()]
        prev = active or self.combo_active_mask.currentText()
        self.combo_active_mask.blockSignals(True)
        self.combo_active_mask.clear()
        self.combo_active_mask.addItems(clean)
        if prev in clean:
            self.combo_active_mask.setCurrentText(prev)
        elif clean:
            self.combo_active_mask.setCurrentIndex(0)
        self.combo_active_mask.blockSignals(False)

    def populate_preview(self, df: pd.DataFrame, max_rows: int = 50) -> None:
        """Update extraction status and cache extracted single-cell features.

        Args:
            df: Single-cell morphological and intensity feature DataFrame.
            max_rows: Maximum rows for preview table display.
        """
        self.cached_df = df
        if df is None or df.empty:
            self.lbl_extract_status.setText("No data extracted")
            return

        _, n_cols = df.shape
        n_samples = int(df["Image"].nunique()) if "Image" in df.columns else 1
        sample_txt = f" across {n_samples} sample(s)" if n_samples > 1 else ""
        self.lbl_extract_status.setText(
            f"Extracted {len(df):,} cells{sample_txt} x {n_cols} features"
        )
        self.lbl_extract_status.setStyleSheet(
            "color: #44AA66; font-weight: bold; font-size: 11px; "
            "background-color: transparent;"
        )
        self.btn_export.setEnabled(True)
        self.dataExtracted.emit(df)

    def set_extracting_state(self, is_extracting: bool) -> None:
        """Toggle UI between active computation and idle states.

        Args:
            is_extracting: True while feature extraction worker thread is running.
        """
        self.btn_extract.setEnabled(not is_extracting)
        self.sample_scope_bar.setEnabled(not is_extracting)
        if is_extracting:
            self.lbl_extract_status.setText("Extracting features...")
            self.lbl_extract_status.setStyleSheet(
                "color: #44AAFF; font-weight: bold; font-size: 11px; "
                "background-color: transparent;"
            )
