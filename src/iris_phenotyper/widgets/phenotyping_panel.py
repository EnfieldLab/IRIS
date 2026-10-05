"""
IRIS-Phenotyper Phenotyping Panel
Mirrors the Viewer's SegmentationPanelWidget — tabs for each active backend,
with method-specific parameter configuration and run/cancel controls.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSpinBox, QTabWidget, QVBoxLayout,
    QWidget, QFrame, QComboBox,
)


_DARK = """
QWidget { background: #1A1A2E; color: #E0E0E0; font-family: 'Inter', sans-serif; }
QGroupBox { border: 1px solid #3A3A5A; border-radius: 6px; margin-top: 8px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #8888FF; font-weight: bold; }
QPushButton { background: #2D2D4E; color: #DDD; border: 1px solid #444; border-radius: 4px; padding: 5px 10px; }
QPushButton:hover { background: #3D3D6E; }
QLabel { color: #CCC; }
QSpinBox, QDoubleSpinBox, QComboBox { background: #252540; border: 1px solid #444; border-radius: 3px; padding: 2px; }
"""

_AVAIL_METHODS = ["ASTIR", "CELESTA", "STELLAR", "CellSighter"]


class MethodConfigWidget(QFrame):
    """Per-method configuration panel shown as a tab."""

    def __init__(self, method_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.method_name = method_name
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { background: #1E1E3A; border-radius: 5px; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Method-specific parameter widgets
        self._params: Dict = {}
        self._build_params(layout)

    def _build_params(self, layout: QVBoxLayout) -> None:
        if self.method_name == "ASTIR":
            layout.addWidget(QLabel("Max Epochs (VI optimisation):"))
            self._params["max_epochs"] = QSpinBox()
            self._params["max_epochs"].setRange(50, 5000)
            self._params["max_epochs"].setValue(500)
            self._params["max_epochs"].setSingleStep(50)
            layout.addWidget(self._params["max_epochs"])

            layout.addWidget(QLabel("Learning Rate:"))
            self._params["learning_rate"] = QDoubleSpinBox()
            self._params["learning_rate"].setRange(1e-5, 1e-1)
            self._params["learning_rate"].setValue(1e-3)
            self._params["learning_rate"].setDecimals(6)
            self._params["learning_rate"].setSingleStep(1e-4)
            layout.addWidget(self._params["learning_rate"])

        elif self.method_name == "CELESTA":
            layout.addWidget(QLabel("EM Iterations:"))
            self._params["n_iter"] = QSpinBox()
            self._params["n_iter"].setRange(1, 50)
            self._params["n_iter"].setValue(10)
            layout.addWidget(self._params["n_iter"])

            layout.addWidget(QLabel("High Expression Threshold:"))
            self._params["high_thresh"] = QDoubleSpinBox()
            self._params["high_thresh"].setRange(0.5, 0.99)
            self._params["high_thresh"].setValue(0.9)
            self._params["high_thresh"].setSingleStep(0.05)
            layout.addWidget(self._params["high_thresh"])

            layout.addWidget(QLabel("Low Expression Threshold:"))
            self._params["low_thresh"] = QDoubleSpinBox()
            self._params["low_thresh"].setRange(0.01, 0.5)
            self._params["low_thresh"].setValue(0.4)
            self._params["low_thresh"].setSingleStep(0.05)
            layout.addWidget(self._params["low_thresh"])

            # Fallback notice
            note = QLabel(
                "⚠ CELESTA requires: pip install git+https://github.com/plevritis-lab/CELESTA.git\n"
                "If unavailable, a rule-based fallback is used."
            )
            note.setWordWrap(True)
            note.setStyleSheet("color: #FFAA44; font-size: 10px;")
            layout.addWidget(note)

        elif self.method_name == "STELLAR":
            note = QLabel("STELLAR is implemented in Phase 4.\nRequires a reference AnnData file.")
            note.setWordWrap(True)
            note.setStyleSheet("color: #AAAAFF; font-size: 11px;")
            layout.addWidget(note)

        elif self.method_name == "CellSighter":
            note = QLabel("CellSighter is implemented in Phase 4.\nRequires manual cell annotations for training.")
            note.setWordWrap(True)
            note.setStyleSheet("color: #AAAAFF; font-size: 11px;")
            layout.addWidget(note)

        layout.addStretch()

    def get_params(self) -> Dict:
        return {
            k: (w.value() if hasattr(w, "value") else w.currentText())
            for k, w in self._params.items()
        }

    def set_status(self, status: str) -> None:
        colours = {
            "Queued": "#888", "Running": "#44AAEE",
            "Done": "#44AA44", "Failed": "#DD4444",
        }
        self._status_lbl.setText(status) if hasattr(self, "_status_lbl") else None


class PhenotypingPanelWidget(QWidget):
    """
    Main phenotyping control panel.

    Signals
    -------
    runRequested : dict
        Emitted with {method_name: params_dict} when the user clicks Run.
    cancelRequested
        Emitted when Cancel is clicked.
    """

    runRequested = Signal(dict)
    cancelRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(_DARK)
        self._method_widgets: Dict[str, MethodConfigWidget] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Header
        hdr = QLabel("Phenotyping Methods")
        hdr.setStyleSheet("font-size: 14px; font-weight: bold; color: #8888FF;")
        layout.addWidget(hdr)

        info = QLabel(
            "Enable methods below and configure parameters.  "
            "Multiple methods will be merged via ensemble voting."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #AAA; font-size: 11px;")
        layout.addWidget(info)

        # Method tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(
            "QTabBar::tab { background: #1E1E3A; color: #AAA; padding: 5px 12px; }"
            "QTabBar::tab:selected { background: #2D2D6E; color: #FFF; }"
        )

        for method in _AVAIL_METHODS:
            w = MethodConfigWidget(method)
            self._method_widgets[method] = w
            self.tab_widget.addTab(w, method)

        layout.addWidget(self.tab_widget)

        # ── Ensemble Strategy ─────────────────────────────────────────────────
        ens_box = QGroupBox("Ensemble Voting")
        ens_layout = QVBoxLayout(ens_box)

        row_strat = QHBoxLayout()
        row_strat.addWidget(QLabel("Strategy:"))
        self.combo_strategy = QComboBox()
        self.combo_strategy.addItems(["soft_vote", "weighted_majority", "borda_count"])
        row_strat.addWidget(self.combo_strategy)
        ens_layout.addLayout(row_strat)

        row_thresh = QHBoxLayout()
        row_thresh.addWidget(QLabel("Min Confidence:"))
        self.spin_min_conf = QDoubleSpinBox()
        self.spin_min_conf.setRange(0.0, 1.0)
        self.spin_min_conf.setValue(0.3)
        self.spin_min_conf.setSingleStep(0.05)
        row_thresh.addWidget(self.spin_min_conf)
        ens_layout.addLayout(row_thresh)

        layout.addWidget(ens_box)

        # ── Run / Cancel ──────────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  Run Phenotyping")
        self.btn_run.setStyleSheet(
            "QPushButton { background: #3A5A9A; color: white; font-weight: bold;"
            "padding: 8px; font-size: 13px; border-radius: 5px; }"
            "QPushButton:hover { background: #4A7FBA; }"
        )
        self.btn_run.clicked.connect(self._emit_run)
        layout.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("■  Cancel")
        self.btn_cancel.setStyleSheet(
            "QPushButton { background: #7A2A2A; color: white; font-weight: bold;"
            "padding: 6px; border-radius: 5px; }"
            "QPushButton:hover { background: #AA3333; }"
        )
        self.btn_cancel.clicked.connect(self.cancelRequested.emit)
        self.btn_cancel.hide()
        layout.addWidget(self.btn_cancel)

        self.lbl_status = QLabel("Status: Ready")
        self.lbl_status.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.lbl_status)

    def _emit_run(self) -> None:
        configs: Dict = {}
        for method, w in self._method_widgets.items():
            if method not in ("STELLAR", "CellSighter"):  # Phase 4
                configs[method] = w.get_params()
        configs["_ensemble"] = {
            "strategy": self.combo_strategy.currentText(),
            "min_probability_threshold": self.spin_min_conf.value(),
        }
        self.runRequested.emit(configs)

    def set_running_state(self, running: bool) -> None:
        self.btn_run.setVisible(not running)
        self.btn_cancel.setVisible(running)
        self.tab_widget.setEnabled(not running)
        if running:
            self.lbl_status.setText("Status: Running…")
            self.lbl_status.setStyleSheet("color: #44AAEE; font-size: 11px;")
        else:
            self.lbl_status.setStyleSheet("color: #888; font-size: 11px;")

    def update_method_status(self, method_name: str, status: str) -> None:
        w = self._method_widgets.get(method_name)
        if w:
            w.set_status(status)
        statuses = {"Running": "44AAEE", "Done": "44AA44", "Failed": "DD4444"}
        color = statuses.get(status, "888")
        self.lbl_status.setText(f"{method_name}: {status}")
        self.lbl_status.setStyleSheet(f"color: #{color}; font-size: 11px;")
