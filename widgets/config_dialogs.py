"""
IRIS-Viewer Panel Configuration Dialog
Direct adaptation of IRIS/widgets/config_dialogs.py for mapping 
multiplex image channel indices to biological marker labels.
"""
import pandas as pd
from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QTableWidget,
                             QTableWidgetItem, QFileDialog, QMessageBox, QHeaderView)
from qtpy.QtCore import Qt


class ViewerPanelConfigDialog(QDialog):
    """
    Dialog to map multiplexed image channel indices to custom biological markers
    manually or automatically via design sheet layout matrices.
    """
    def __init__(self, num_channels: int, parent=None):
        super().__init__(parent)
        self.num_channels = num_channels
        self.marker_panel = []

        self.setWindowTitle("IRIS-Viewer — Marker Panel Configuration")
        self.resize(480, 550)
        self.setModal(True)

        self.setStyleSheet("""
            QDialog { background-color: #1f232b; color: #E0E0E0; }
            QLabel { color: #DDD; font-size: 12px; font-weight: bold; }
            QPushButton { background-color: #3A6A9A; color: white; font-weight: bold; font-size: 12px; border: none; padding: 6px; }
            QPushButton:hover { background-color: #4A7FBA; }
            QTableWidget { background-color: #2A2A2A; color: white; gridline-color: #444; border: 1px solid #555; }
            QHeaderView::section { background-color: #333; color: white; padding: 4px; border: 1px solid #444; }
            QLineEdit { background-color: #14161d; color: white; border: 1px solid #555; padding: 3px; }
        """)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)

        info_lbl = QLabel(f"Detected Dimension: {self.num_channels} Channels.\n"
                          "Assign marker labels manually or upload a channel-to-marker mapping spreadsheet.")
        info_lbl.setStyleSheet("color: #AAA; font-style: italic; font-weight: normal;")
        layout.addWidget(info_lbl)

        io_row = QHBoxLayout()
        self.btn_load_sheet = QPushButton("Import Mapping Sheet (.csv / .xlsx)")
        self.btn_load_sheet.clicked.connect(self._import_panel_sheet)
        io_row.addWidget(self.btn_load_sheet)
        layout.addLayout(io_row)

        self.table = QTableWidget(self.num_channels, 2)
        self.table.setHorizontalHeaderLabels(["Channels", "Marker Labels"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)

        for idx in range(self.num_channels):
            chan_item = QTableWidgetItem(f"Channel {idx + 1}")
            chan_item.setFlags(chan_item.flags() & ~Qt.ItemIsEditable)
            chan_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(idx, 0, chan_item)

            le = QLineEdit()
            le.setPlaceholderText(f"Marker label for Channel {idx + 1}")
            if idx == 0:
                le.setText("DAPI")
            self.table.setCellWidget(idx, 1, le)

        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm Channel to Marker Mapping")
        self.btn_confirm.clicked.connect(self._validate_and_accept)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_confirm)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    def _import_panel_sheet(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Sheet", "", "File Extensions (*.csv *.xlsx *.xls)")
        if not filepath:
            return
        try:
            df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
            df.columns = [c.lower().strip() for c in df.columns]
            if 'channels' not in df.columns or 'markers' not in df.columns:
                raise KeyError("Columns must have header names designated explicitly: 'channels' and 'markers'.")

            for _, row in df.iterrows():
                try:
                    chan_idx = int(row['channels']) - 1
                    if 0 <= chan_idx < self.num_channels:
                        le = self.table.cellWidget(chan_idx, 1)
                        if isinstance(le, QLineEdit):
                            le.setText(str(row['markers']).strip())
                except Exception:
                    continue
            QMessageBox.information(self, "Success", "Channel-to-marker mapping template parsed successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Parsing Discrepancy", f"Channel-to-marker mapping parsing exception: {str(e)}")

    def _validate_and_accept(self):
        panel_labels = []
        for idx in range(self.num_channels):
            le = self.table.cellWidget(idx, 1)
            text = le.text().strip() if isinstance(le, QLineEdit) else f"Ch_{idx + 1}"
            panel_labels.append(text if text else f"Ch_{idx + 1}")
        self.marker_panel = panel_labels
        self.accept()
