"""
IRIS-Viewer Consensus Panel Widget
Manages the confidence overlay visualization, threshold adjustment, and
interactive review of low-confidence cells where the user can pick the
preferred segmentation method on a per-cell basis.
"""
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QCheckBox, QPushButton, QGroupBox, QFrame, QSizePolicy, QScrollArea
)
from qtpy.QtCore import Qt, Signal
from widgets.styles import (
    SHARED_DARK_THEME, CONFIDENCE_HIGH_COLOR,
    CONFIDENCE_MEDIUM_COLOR, CONFIDENCE_LOW_COLOR
)


class ConsensusPanelWidget(QWidget):
    """Dock widget for consensus mask confidence controls and interactive cell review."""

    # Signals
    generateConsensusRequested = Signal()
    thresholdChanged = Signal(float)
    showConsensusToggled = Signal(bool)
    showIndividualToggled = Signal(str, bool) # method_name, visible
    reviewModeToggled = Signal(bool)
    cellOverrideRequested = Signal(int, str) # cell_id, preferred_method

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

        self._build_generation_controls()
        self._build_threshold_controls()
        self._build_overlay_controls()
        self._build_color_legend()
        self._build_review_controls()
        self._build_statistics()
        self.main_layout.addStretch()

    def _build_generation_controls(self):
        group = QGroupBox()
        gl = QVBoxLayout()
        gl.setContentsMargins(5, 5, 5, 5)
        group.setLayout(gl)

        self.btn_generate = QPushButton("Generate Consensus Map")
        self.btn_generate.setStyleSheet(
            "QPushButton { background-color: #4A7FBA; color: white; font-weight: bold; padding: 8px; font-size: 13px; } "
            "QPushButton:hover { background-color: #5A8FCA; }"
        )
        self.btn_generate.clicked.connect(lambda: self.generateConsensusRequested.emit())
        gl.addWidget(self.btn_generate)
        
        self.lbl_gen_status = QLabel("Pending Masks")
        self.lbl_gen_status.setStyleSheet("color: #AAA; font-size: 11px;")
        gl.addWidget(self.lbl_gen_status)
        
        self.main_layout.addWidget(group)

    def set_generating_state(self, is_generating: bool):
        """Toggle UI state to prevent clicking during consensus generation."""
        self.btn_generate.setEnabled(not is_generating)
        if is_generating:
            self.lbl_gen_status.setText("Merging Masks... (This may take a moment)")
            self.lbl_gen_status.setStyleSheet("color: #44AAFF; font-weight: bold; font-size: 11px;")
        else:
            self.lbl_gen_status.setText("Consensus Generated")
            self.lbl_gen_status.setStyleSheet("color: #44AA66; font-weight: bold; font-size: 11px;")
    
    # Confidence Threshold
    def _build_threshold_controls(self):
        group = QGroupBox()
        gl = QVBoxLayout()
        gl.setContentsMargins(5, 5, 5, 5)
        gl.setSpacing(4)
        group.setLayout(gl)

        title = QLabel("Confidence Threshold")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        gl.addWidget(title)

        self._add_separator(gl)

        info = QLabel("Cells below this threshold are highlighted for review.\n"
                       "Confidence = fraction of methods that agree on the cell.")
        info.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        info.setWordWrap(True)
        gl.addWidget(info)

        row = QHBoxLayout()
        row.addWidget(QLabel("Threshold:"))
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100)
        self.slider_threshold.setValue(66)
        self.slider_threshold.setTickPosition(QSlider.TicksBelow)
        self.slider_threshold.setTickInterval(10)
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)
        row.addWidget(self.slider_threshold)

        self.lbl_threshold_value = QLabel("0.66")
        self.lbl_threshold_value.setFixedWidth(40)
        self.lbl_threshold_value.setStyleSheet("color: #FFF; font-weight: bold;")
        row.addWidget(self.lbl_threshold_value)
        gl.addLayout(row)

        self.main_layout.addWidget(group)

    # ── Overlay Controls ─────────────────────────────────────────────
    def _build_overlay_controls(self):
        group = QGroupBox()
        gl = QVBoxLayout()
        gl.setContentsMargins(5, 5, 5, 5)
        gl.setSpacing(4)
        group.setLayout(gl)

        title = QLabel("Overlay Controls")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        gl.addWidget(title)

        self._add_separator(gl)

        self.chk_show_consensus = QCheckBox("Show Consensus Mask")
        self.chk_show_consensus.setChecked(True)
        self.chk_show_consensus.toggled.connect(
            lambda v: self.showConsensusToggled.emit(v)
        )
        gl.addWidget(self.chk_show_consensus)

        self.chk_show_confidence = QCheckBox("Show Confidence Contours")
        self.chk_show_confidence.setChecked(True)
        gl.addWidget(self.chk_show_confidence)

        self._add_separator(gl)

        lbl_individual = QLabel("Individual Method Masks:")
        lbl_individual.setStyleSheet("color: #BBB; font-size: 11px;")
        gl.addWidget(lbl_individual)

        self.method_visibility_checks = {}
        for method in ["Cellpose", "StarDist", "CellSAM", "InstanSeg"]:
            chk = QCheckBox(method)
            chk.setChecked(False)
            chk.toggled.connect(
                lambda checked, m=method: self.showIndividualToggled.emit(m, checked)
            )
            self.method_visibility_checks[method] = chk
            gl.addWidget(chk)

        self.main_layout.addWidget(group)

    # ── Color Legend ─────────────────────────────────────────────────
    def _build_color_legend(self):
        group = QGroupBox()
        gl = QVBoxLayout()
        gl.setContentsMargins(5, 5, 5, 5)
        gl.setSpacing(4)
        group.setLayout(gl)

        title = QLabel("Confidence Legend")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        gl.addWidget(title)

        self._add_separator(gl)

        for label_text, color in [
            ("High Confidence (≥ threshold)", CONFIDENCE_HIGH_COLOR),
            ("Medium Confidence (≥ 0.5 × threshold)", CONFIDENCE_MEDIUM_COLOR),
            ("Low Confidence (< 0.5 × threshold)", CONFIDENCE_LOW_COLOR),
        ]:
            row = QHBoxLayout()
            swatch = QLabel()
            swatch.setFixedSize(16, 16)
            swatch.setStyleSheet(f"background-color: {color}; border: 1px solid #888;")
            row.addWidget(swatch)
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #CCC; font-size: 11px;")
            row.addWidget(lbl)
            row.addStretch()
            gl.addLayout(row)

        self.main_layout.addWidget(group)

    # ── Review Controls ──────────────────────────────────────────────
    def _build_review_controls(self):
        group = QGroupBox()
        gl = QVBoxLayout()
        gl.setContentsMargins(5, 5, 5, 5)
        gl.setSpacing(4)
        group.setLayout(gl)

        title = QLabel("Interactive Review")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        gl.addWidget(title)

        self._add_separator(gl)

        info = QLabel("Enable review mode, then click on low-confidence cells\n"
                       "to inspect each method's segmentation and pick the best one.")
        info.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        info.setWordWrap(True)
        gl.addWidget(info)

        self.chk_review_mode = QCheckBox("Enable Review Mode")
        self.chk_review_mode.toggled.connect(
            lambda v: self.reviewModeToggled.emit(v)
        )
        gl.addWidget(self.chk_review_mode)

        self._add_separator(gl)

        # Cell review info (populated dynamically when user clicks a cell)
        self.lbl_review_cell = QLabel("No cell selected")
        self.lbl_review_cell.setStyleSheet("color: #AAA; font-size: 11px;")
        gl.addWidget(self.lbl_review_cell)

        self.lbl_review_confidence = QLabel("Confidence: —")
        self.lbl_review_confidence.setStyleSheet("color: #AAA; font-size: 11px;")
        gl.addWidget(self.lbl_review_confidence)

        self.lbl_review_methods = QLabel("Methods: —")
        self.lbl_review_methods.setStyleSheet("color: #AAA; font-size: 11px;")
        self.lbl_review_methods.setWordWrap(True)
        gl.addWidget(self.lbl_review_methods)

        # Method override buttons (populated dynamically)
        self.override_buttons_layout = QVBoxLayout()
        self.override_buttons_layout.setContentsMargins(0, 4, 0, 0)
        self.override_buttons_layout.setSpacing(3)
        gl.addLayout(self.override_buttons_layout)

        self.main_layout.addWidget(group)

    # ── Statistics ───────────────────────────────────────────────────
    def _build_statistics(self):
        group = QGroupBox()
        gl = QVBoxLayout()
        gl.setContentsMargins(5, 5, 5, 5)
        gl.setSpacing(3)
        group.setLayout(gl)

        title = QLabel("Agreement Statistics")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        gl.addWidget(title)

        self._add_separator(gl)

        self.lbl_total_cells = QLabel("Total Cells: —")
        self.lbl_total_cells.setStyleSheet("color: #AAA; font-size: 11px;")
        gl.addWidget(self.lbl_total_cells)

        self.lbl_high_conf = QLabel("High Confidence: —")
        self.lbl_high_conf.setStyleSheet(f"color: {CONFIDENCE_HIGH_COLOR}; font-size: 11px; font-weight: bold;")
        gl.addWidget(self.lbl_high_conf)

        self.lbl_med_conf = QLabel("Medium Confidence: —")
        self.lbl_med_conf.setStyleSheet(f"color: {CONFIDENCE_MEDIUM_COLOR}; font-size: 11px; font-weight: bold;")
        gl.addWidget(self.lbl_med_conf)

        self.lbl_low_conf = QLabel("Low Confidence: —")
        self.lbl_low_conf.setStyleSheet(f"color: {CONFIDENCE_LOW_COLOR}; font-size: 11px; font-weight: bold;")
        gl.addWidget(self.lbl_low_conf)

        self.lbl_mean_conf = QLabel("Mean Confidence: —")
        self.lbl_mean_conf.setStyleSheet("color: #AAA; font-size: 11px;")
        gl.addWidget(self.lbl_mean_conf)

        self.main_layout.addWidget(group)

    # ── Utility ──────────────────────────────────────────────────────
    def _add_separator(self, layout):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #555555; background-color: #555555; max-height: 1px; margin-bottom: 4px;")
        layout.addWidget(line)

    def _on_threshold_changed(self, int_value):
        val = int_value / 100.0
        self.lbl_threshold_value.setText(f"{val:.2f}")
        self.thresholdChanged.emit(val)

    # ── Public API ───────────────────────────────────────────────────
    def get_threshold(self) -> float:
        return self.slider_threshold.value() / 100.0

    def update_statistics(
        self,
        total: int = 0,
        high: int = 0,
        medium: int = 0,
        low: int = 0,
        mean_conf: float = 0.0,
    ):
        """Update the agreement statistics display."""
        self.lbl_total_cells.setText(f"Total Cells: {total:,}")
        self.lbl_high_conf.setText(f"High Confidence: {high:,} ({high/total*100:.1f}%)" if total > 0 else "High Confidence: —")
        self.lbl_med_conf.setText(f"Medium Confidence: {medium:,} ({medium/total*100:.1f}%)" if total > 0 else "Medium Confidence: —")
        self.lbl_low_conf.setText(f"Low Confidence: {low:,} ({low/total*100:.1f}%)" if total > 0 else "Low Confidence: —")
        self.lbl_mean_conf.setText(f"Mean Confidence: {mean_conf:.3f}")

    def show_cell_review(self, cell_id: int, confidence: float, method_map: dict):
        """
        Populate the review section when a user clicks on a low-confidence cell.
        """
        self.lbl_review_cell.setText(f"Cell ID: {cell_id}")
        self.lbl_review_confidence.setText(f"Confidence: {confidence:.3f}")
        self.lbl_review_methods.setText(
            f"Detected by: {', '.join(method_map.keys())} ({len(method_map)} method(s))"
        )

        while self.override_buttons_layout.count():
            child = self.override_buttons_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Color palette matching the visual isolation layer
        palette_hex = ["#FF4444", "#44FF44", "#4488FF", "#FF44FF"]
        palette_hover = ["#FF6666", "#66FF66", "#66AAFF", "#FF66FF"]

        # Sort the methods to ensure consistent color assignment
        for i, method_name in enumerate(sorted(method_map.keys())):
            bg_color = palette_hex[i % len(palette_hex)]
            hover_color = palette_hover[i % len(palette_hover)]
            
            btn = QPushButton(f"Use {method_name} mask")
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {bg_color}; color: black; padding: 4px; font-size: 11px; font-weight: bold; border-radius: 2px; }}"
                f"QPushButton:hover {{ background-color: {hover_color}; }}"
            )
            btn.clicked.connect(
                lambda checked, m=method_name, cid=cell_id: self.cellOverrideRequested.emit(cid, m)
            )
            self.override_buttons_layout.addWidget(btn)

        # Add the Remove Mask button
        btn_remove = QPushButton("Remove Cell entirely")
        btn_remove.setStyleSheet(
            "QPushButton { background-color: #333333; color: white; padding: 4px; font-size: 11px; margin-top: 4px; border: 1px solid #555; }"
            "QPushButton:hover { background-color: #555555; }"
        )
        btn_remove.clicked.connect(
            lambda checked, cid=cell_id: self.cellOverrideRequested.emit(cid, "Remove")
        )
        self.override_buttons_layout.addWidget(btn_remove)

    def clear_cell_review(self):
        """Reset the review section to idle state."""
        self.lbl_review_cell.setText("No cell selected")
        self.lbl_review_confidence.setText("Confidence: —")
        self.lbl_review_methods.setText("Methods: —")
        while self.override_buttons_layout.count():
            child = self.override_buttons_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
