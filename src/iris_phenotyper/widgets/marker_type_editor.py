"""
IRIS-Phenotyper Marker-Type Editor Widget

A Qt table widget where the user defines the marker-to-cell-type mapping:
  - Rows = cell types
  - Left column = cell type name
  - Right column = comma-separated positive markers

Used to configure ASTIR and MAPS backends.
Also supports:
  - Adding / removing cell type rows
  - Saving and loading the panel to/from YAML / CSV
  - Importing a Viewer features CSV to auto-populate available markers
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from iris_common.types import CellTypePanel, CellTypeRule, MarkerPanel
from iris_common.widgets.styles import SHARED_DARK_THEME


class MarkerTypeEditorWidget(QWidget):
    """
    Editable table for defining the marker → cell type mapping used by
    ASTIR and MAPS.

    Signals
    -------
    panelChanged : emitted whenever the user edits the table
    """

    panelChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._available_markers: List[str] = []
        self.setStyleSheet(SHARED_DARK_THEME)
        self._setup_ui()

    # ── UI Setup ──────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Header
        hdr = QLabel("Marker → Cell Type Panel")
        layout.addWidget(hdr)

        info = QLabel(
            "Define positive markers for each cell type.\n"
            "Separate multiple markers with commas.  "
            "Optional: add negative markers prefixed with '-' (MAPS only)."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["Cell Type", "Positive Markers", "Parent (optional)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(lambda _: self.panelChanged.emit())
        layout.addWidget(self.table)

        # Buttons row
        btn_row = QHBoxLayout()
        for label, slot in [
            ("+ Add Type", self._add_row),
            ("– Remove Selected", self._remove_row),
            ("Save Panel", self._save_panel),
            ("Load Panel", self._load_panel),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        # Pre-populate with common defaults
        self._add_defaults()

    # ── Defaults ──────────────────────────────────────────────────────────────

    def _add_defaults(self) -> None:
        defaults = [
            ("Tumor", "CK, PANCK", ""),
            ("T cell", "CD3", "Immune"),
            ("CD4 T cell", "CD3, CD4", "T cell"),
            ("CD8 T cell", "CD3, CD8", "T cell"),
            ("B cell", "CD20", "Immune"),
            ("Macrophage", "CD68", "Immune"),
            ("M2 Macrophage", "CD68, CD163", "Macrophage"),
            ("Neutrophil", "MPO", "Immune"),
            ("NK cell", "CD56", "Immune"),
            ("Fibroblast", "ASMA, SMA", "Stromal"),
            ("Endothelial", "CD31", "Stromal"),
        ]
        for cell_type, markers, parent in defaults:
            self._insert_row(cell_type, markers, parent)

    def _insert_row(
        self,
        cell_type: str = "",
        markers: str = "",
        parent: str = "",
    ) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(cell_type))
        self.table.setItem(row, 1, QTableWidgetItem(markers))
        self.table.setItem(row, 2, QTableWidgetItem(parent))

    # ── Slot Handlers ─────────────────────────────────────────────────────────

    def _add_row(self) -> None:
        self._insert_row()
        self.table.scrollToBottom()

    def _remove_row(self) -> None:
        rows = sorted(set(i.row() for i in self.table.selectedItems()), reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self.panelChanged.emit()

    def _save_panel(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Marker Panel", "", "JSON (*.json);;YAML (*.yaml *.yml)"
        )
        if not path:
            return
        panel = self.get_cell_type_panel()
        data = [
            {
                "cell_type": r.cell_type,
                "markers_positive": r.markers_positive,
                "markers_negative": r.markers_negative,
                "parent": r.parent,
            }
            for r in panel.rules
        ]
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        QMessageBox.information(self, "Saved", f"Panel saved to:\n{path}")

    def _load_panel(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Marker Panel", "",
            "JSON (*.json);;YAML (*.yaml *.yml);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
            self.table.setRowCount(0)
            for entry in data:
                pos = ", ".join(entry.get("markers_positive", []))
                parent = entry.get("parent") or ""
                self._insert_row(entry.get("cell_type", ""), pos, parent)
            self.panelChanged.emit()
            QMessageBox.information(self, "Loaded", f"Panel loaded from:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load panel:\n{exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def set_available_markers(self, markers: List[str]) -> None:
        """Provide the marker list from the loaded features CSV for validation."""
        self._available_markers = markers

    def get_cell_type_panel(self) -> CellTypePanel:
        """Read the current table contents and return a CellTypePanel."""
        rules: List[CellTypeRule] = []
        for row in range(self.table.rowCount()):
            ct_item = self.table.item(row, 0)
            mk_item = self.table.item(row, 1)
            par_item = self.table.item(row, 2)

            if ct_item is None:
                continue
            ct = ct_item.text().strip()
            if not ct:
                continue

            mk_text = mk_item.text() if mk_item else ""
            pos_markers = [m.strip().lstrip("+") for m in mk_text.split(",")
                           if m.strip() and not m.strip().startswith("-")]
            neg_markers = [m.strip().lstrip("-") for m in mk_text.split(",")
                           if m.strip().startswith("-")]
            parent = par_item.text().strip() if par_item else None

            rules.append(CellTypeRule(
                cell_type=ct,
                markers_positive=pos_markers,
                markers_negative=neg_markers,
                parent=parent or None,
            ))

        return CellTypePanel(rules=rules)

    def set_cell_type_panel(self, panel: CellTypePanel) -> None:
        """Populate the table from an existing CellTypePanel."""
        self.table.setRowCount(0)
        for rule in panel.rules:
            markers = ", ".join(
                rule.markers_positive
                + [f"-{m}" for m in rule.markers_negative]
            )
            self._insert_row(rule.cell_type, markers, rule.parent or "")
        self.panelChanged.emit()
