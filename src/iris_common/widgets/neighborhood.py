"""
IRIS Cellular Neighborhoods Profiler Widget
============================================
Visualizes and analyzes local cellular niches (neighborhoods) with stacked
bar composition charts, custom metadata curation, and high-resolution export.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
try:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    try:
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    except ImportError:
        NavigationToolbar = None
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from iris_common.io.exporters import export_neighborhood_composition_matrix


class CellProportionDialog(QDialog):
    """
    Interactive pop-up window for inspecting, exploring, and exporting
    cellular neighborhood composition plots.
    """

    def __init__(
        self,
        centers: np.ndarray,
        type_labels: List[str],
        niche_metadata: Dict[int, Dict[str, str]],
        phenotype_color_map: Dict[str, str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cell Phenotype Proportions per Neighborhood")
        self.resize(1020, 720)
        self.setMinimumSize(680, 480)
        self.setWindowFlags(self.windowFlags() | Qt.Window)

        self.centers = centers
        self.type_labels = type_labels
        self.niche_metadata = niche_metadata
        self.phenotype_color_map = phenotype_color_map
        self.bar_data = []

        self.setStyleSheet("""
            QDialog { background-color: #1a1d24; color: #E0E0E0; font-size: 12px; }
            QGroupBox { border: 1px solid #444; margin-top: 6px; font-weight: bold; color: #DDD; }
            QPushButton {
                background-color: #2D5F8B; color: white; font-weight: bold;
                border: 1px solid #4B82B5; border-radius: 3px; padding: 4px 10px;
            }
            QPushButton:hover { background-color: #3D75A8; }
            QComboBox, QSpinBox {
                background-color: #2A2A2A; color: white; border: 1px solid #555;
                border-radius: 3px; padding: 2px 6px;
            }
            QLabel { color: #DDD; }
            QCheckBox { color: #DDD; }
        """)

        self._build_ui()
        self.render_plot()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ── Top Toolbar Controls ──────────────────────────────────────
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

        controls_layout.addWidget(QLabel("Plot Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "Stacked Proportions (0.0 – 1.0)",
            "Stacked Percentages (0% – 100%)",
            "Grouped by Neighborhood",
            "Grouped by Cell Type",
        ])
        self.combo_mode.currentIndexChanged.connect(self.render_plot)
        controls_layout.addWidget(self.combo_mode)

        self.chk_show_labels = QCheckBox("Show % on Bars")
        self.chk_show_labels.setChecked(False)
        self.chk_show_labels.toggled.connect(self.render_plot)
        controls_layout.addWidget(self.chk_show_labels)

        controls_layout.addWidget(QLabel("Angle:"))
        self.spin_angle = QSpinBox()
        self.spin_angle.setRange(0, 90)
        self.spin_angle.setValue(25)
        self.spin_angle.valueChanged.connect(self.render_plot)
        controls_layout.addWidget(self.spin_angle)

        controls_layout.addWidget(QLabel("Font Size:"))
        self.spin_font = QSpinBox()
        self.spin_font.setRange(8, 22)
        self.spin_font.setValue(11)
        self.spin_font.valueChanged.connect(self.render_plot)
        controls_layout.addWidget(self.spin_font)

        controls_layout.addStretch(1)

        self.btn_copy = QPushButton("Copy to Clipboard")
        self.btn_copy.setToolTip("Copy plot as image to clipboard")
        self.btn_copy.clicked.connect(self.copy_to_clipboard)
        controls_layout.addWidget(self.btn_copy)

        self.btn_save = QPushButton("Export Figure...")
        self.btn_save.setToolTip("Export plot as high-resolution PDF, SVG, or PNG")
        self.btn_save.clicked.connect(self.export_figure)
        controls_layout.addWidget(self.btn_save)

        layout.addLayout(controls_layout)

        # ── Hover Details Status Bar ─────────────────────────────────
        self.lbl_hover = QLabel("Hover over any bar segment to inspect exact proportions.")
        self.lbl_hover.setStyleSheet("color: #AAA; font-size: 12px; font-style: italic; padding: 2px 4px;")
        layout.addWidget(self.lbl_hover)

        # ── Matplotlib Canvas & Toolbar ──────────────────────────────
        self.figure = Figure(figsize=(10, 6.5), dpi=100)
        self.figure.patch.set_facecolor('#14161d')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        if NavigationToolbar is not None:
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setStyleSheet("""
                QToolBar { background-color: #1f232b; border: 1px solid #333; spacing: 4px; padding: 2px; }
                QToolButton { background-color: transparent; border: 1px solid transparent; border-radius: 3px; padding: 3px; }
                QToolButton:hover { background-color: #3A6A9A; border: 1px solid #558; }
            """)
            layout.addWidget(self.toolbar)

        layout.addWidget(self.canvas, 1)

        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def render_plot(self) -> None:
        if self.centers is None or self.type_labels is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#14161d')

        mode = self.combo_mode.currentText()
        font_sz = self.spin_font.value()
        angle = self.spin_angle.value()
        show_labels = self.chk_show_labels.isChecked()

        num_niches = self.centers.shape[0]
        num_types = len(self.type_labels)
        niches_axis_labels = [
            self.niche_metadata.get(i, {}).get("name", f"Niche {i}")
            for i in range(num_niches)
        ]

        tab20_cmap = plt.colormaps.get_cmap("tab20")
        self.bar_data = []

        if "Stacked" in mode:
            is_pct = "Percentages" in mode
            scale_mult = 100.0 if is_pct else 1.0
            bottom = np.zeros(num_niches)

            for idx, label in enumerate(self.type_labels):
                fractions = self.centers[:, idx] * scale_mult
                assigned_color = (
                    self.phenotype_color_map.get(label)
                    or self.phenotype_color_map.get(label.lower())
                    or tab20_cmap(idx % 20)
                )

                bars = ax.bar(
                    niches_axis_labels,
                    fractions,
                    bottom=bottom,
                    label=label,
                    color=assigned_color,
                    edgecolor='black',
                    linewidth=0.5,
                )

                for b_idx, bar in enumerate(bars):
                    val = fractions[b_idx]
                    pct_val = (self.centers[b_idx, idx]) * 100.0
                    pct_str = f"{pct_val:.1f}%"
                    self.bar_data.append((
                        bar,
                        niches_axis_labels[b_idx],
                        label,
                        self.centers[b_idx, idx],
                        pct_str,
                    ))

                    if show_labels and pct_val >= 5.0:
                        cy = bottom[b_idx] + val / 2.0
                        cx = bar.get_x() + bar.get_width() / 2.0
                        try:
                            rgb = mcolors.to_rgb(assigned_color)
                            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                            txt_color = "black" if lum > 0.55 else "white"
                        except Exception:
                            txt_color = "white"
                        ax.text(
                            cx, cy, f"{pct_val:.0f}%",
                            ha='center', va='center',
                            fontsize=max(7, font_sz - 3),
                            fontweight='bold', color=txt_color,
                        )

                bottom += fractions

            ax.set_ylim(0, 100 if is_pct else 1.0)
            ax.set_ylabel("Proportion (%)" if is_pct else "Cell Type Proportion", color='white', fontsize=font_sz)

        elif mode == "Grouped by Neighborhood":
            x = np.arange(num_niches)
            width = 0.8 / max(1, num_types)
            for idx, label in enumerate(self.type_labels):
                fractions = self.centers[:, idx]
                assigned_color = (
                    self.phenotype_color_map.get(label)
                    or self.phenotype_color_map.get(label.lower())
                    or tab20_cmap(idx % 20)
                )
                bars = ax.bar(
                    x + (idx - num_types / 2) * width + width / 2,
                    fractions,
                    width,
                    label=label,
                    color=assigned_color,
                    edgecolor='black',
                    linewidth=0.5,
                )
                for b_idx, bar in enumerate(bars):
                    pct_val = fractions[b_idx] * 100.0
                    self.bar_data.append((
                        bar,
                        niches_axis_labels[b_idx],
                        label,
                        fractions[b_idx],
                        f"{pct_val:.1f}%",
                    ))
            ax.set_xticks(x)
            ax.set_xticklabels(niches_axis_labels, rotation=angle, ha='right')
            ax.set_ylabel("Proportion", color='white', fontsize=font_sz)

        elif mode == "Grouped by Cell Type":
            x = np.arange(num_types)
            width = 0.8 / max(1, num_niches)
            palette = IrisCellularNeighborhoodsWidget.DEFAULT_PALETTE if hasattr(IrisCellularNeighborhoodsWidget, 'DEFAULT_PALETTE') else [
                "#FF0055", "#00FFaa", "#aa00FF", "#FFFF00", "#00FFFF",
            ]
            for n_idx in range(num_niches):
                fractions = self.centers[n_idx, :]
                n_name = niches_axis_labels[n_idx]
                n_col = self.niche_metadata.get(n_idx, {}).get("color", palette[n_idx % len(palette)])
                bars = ax.bar(
                    x + (n_idx - num_niches / 2) * width + width / 2,
                    fractions,
                    width,
                    label=n_name,
                    color=n_col,
                    edgecolor='black',
                    linewidth=0.5,
                )
                for t_idx, bar in enumerate(bars):
                    pct_val = fractions[t_idx] * 100.0
                    self.bar_data.append((
                        bar,
                        n_name,
                        self.type_labels[t_idx],
                        fractions[t_idx],
                        f"{pct_val:.1f}%",
                    ))
            ax.set_xticks(x)
            ax.set_xticklabels(self.type_labels, rotation=angle, ha='right')
            ax.set_ylabel("Proportion", color='white', fontsize=font_sz)

        ax.tick_params(colors='white', labelsize=font_sz)
        if "Grouped by Cell Type" not in mode:
            ax.set_xticks(range(num_niches))
            ax.set_xticklabels(niches_axis_labels, rotation=angle, ha='right')

        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            fontsize=max(8, font_sz - 2),
            facecolor='#2A2A2A',
            edgecolor='#444444',
            labelcolor='white',
        )
        for spine in ax.spines.values():
            spine.set_color('#444444')
        ax.grid(axis='y', color='#333333', linestyle='--', alpha=0.5)

        self.figure.tight_layout()
        self.canvas.draw()

    def _on_mouse_move(self, event) -> None:
        if event.inaxes is None or not hasattr(self, "bar_data"):
            return

        for bar, niche_lbl, ct, val, pct_str in self.bar_data:
            contains, _ = bar.contains(event)
            if contains:
                self.lbl_hover.setText(
                    f"<b>{niche_lbl}</b> &nbsp;→&nbsp; <b>{ct}</b>: "
                    f"<span style='color:#00E5FF; font-weight:bold;'>{pct_str}</span> (raw: {val:.4f})"
                )
                return

        self.lbl_hover.setText("Hover over any bar segment to inspect exact proportions.")

    def copy_to_clipboard(self) -> None:
        try:
            pixmap = self.canvas.grab()
            QApplication.clipboard().setPixmap(pixmap)
            self.lbl_hover.setText("Copied figure to clipboard! (You can paste directly into slides or docs)")
            self.lbl_hover.setStyleSheet("color: #76FF03; font-size: 12px; font-weight: bold; padding: 2px 4px;")
        except Exception as exc:
            QMessageBox.warning(self, "Copy Failed", f"Could not copy to clipboard: {exc}")

    def export_figure(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Figure",
            "cellular_proportions_figure.pdf",
            "Vector PDF (*.pdf);;Scalable Vector Graphics (*.svg);;High-Res PNG (*.png)",
        )
        if filepath:
            self.figure.savefig(
                filepath,
                dpi=300,
                bbox_inches='tight',
                facecolor=self.figure.get_facecolor(),
            )
            QMessageBox.information(self, "Export Complete", f"Figure saved to {Path(filepath).name}.")

    def update_plot(
        self,
        centers: np.ndarray,
        type_labels: List[str],
        niche_metadata: Dict[int, Dict[str, str]],
        phenotype_color_map: Dict[str, str],
    ) -> None:
        self.centers = centers
        self.type_labels = type_labels
        self.niche_metadata = niche_metadata
        self.phenotype_color_map = phenotype_color_map
        self.render_plot()


class IrisCellularNeighborhoodsWidget(QWidget):
    """
    Dock widget for spatial cellular microenvironment / neighborhood analysis.

    Signals
    -------
    runNicheAnalysis : Signal(int, int, str, float)
        Emitted when the user triggers computation:
        (k_neighbors, n_niches, method, resolution).
    nicheMetadataChanged : Signal(dict)
        Emitted when niche names or colors are updated:
        {niche_id: {"name": str, "color": str}}.
    """

    runNicheAnalysis = Signal(int, int, str, float)
    nicheMetadataChanged = Signal(dict)

    DEFAULT_PALETTE = [
        "#FF0055", "#00FFaa", "#aa00FF", "#FFFF00", "#00FFFF",
        "#FFA500", "#FF00FF", "#00E5FF", "#76FF03", "#FFD600",
        "#E040FB", "#FF6D00", "#1DE9B6", "#FF5252", "#69F0AE",
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.cached_centers: Optional[np.ndarray] = None
        self.cached_labels: Optional[List[str]] = None
        self.cached_df: Optional[pd.DataFrame] = None
        self.cached_phenotype_colors: Dict[str, str] = {}
        self.popup_dialog: Optional[CellProportionDialog] = None
        self.niche_metadata: Dict[int, Dict[str, str]] = {}
        self.legend_widget_ref = None

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)
        self.setLayout(self.main_layout)

        self.setStyleSheet("""
            QWidget { background-color: #1f232b; color: #E0E0E0; font-size: 12px; }
            QGroupBox { border: 1px solid #444; border-radius: 8px; margin-top: 6px; padding-top: 6px; }
            QLabel { color: #DDD; }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2A2A2A; color: white; border: 1px solid #555;
                border-radius: 4px; padding: 2px 4px; height: 22px;
            }
            QPushButton {
                background-color: #3A6A9A; color: white; font-weight: bold;
                border: 1px solid #558; border-radius: 4px; padding: 5px;
            }
            QPushButton:hover { background-color: #4A7FBA; }
            QPushButton:disabled { background-color: #333333; color: #777777; border-color: #444; }
            QTableWidget {
                background-color: #2A2A2A; color: white; gridline-color: #444;
                border: 1px solid #555; selection-background-color: #3A6A9A;
            }
            QHeaderView::section {
                background-color: #333; color: white; padding: 4px;
                border: 1px solid #444; font-weight: bold;
            }
        """)

        self._build_ui()

    def _build_ui(self) -> None:
        # ── Configuration Box ─────────────────────────────────────────────
        config_box = QGroupBox()
        cb = QVBoxLayout()
        cb.setContentsMargins(10, 3, 10, 10)
        cb.setSpacing(6)
        config_box.setLayout(cb)

        title = QLabel("Cellular Neighborhood Analysis")
        title.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold; margin-top: 0px; padding-top: 0px;")
        cb.addWidget(title)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Algorithm:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(["KMeans", "Leiden"])
        self.combo_method.currentTextChanged.connect(self._toggle_method_visibility)
        row1.addWidget(self.combo_method)

        row1.addWidget(QLabel("k-Neighbors:"))
        self.spin_k = QSpinBox()
        self.spin_k.setRange(3, 200)
        self.spin_k.setValue(5)
        row1.addWidget(self.spin_k)
        cb.addLayout(row1)

        self.param_stack_layout = QHBoxLayout()
        self.lbl_clusters = QLabel("Target Niches (N):")
        self.spin_niches = QSpinBox()
        self.spin_niches.setRange(2, 50)
        self.spin_niches.setValue(5)
        self.param_stack_layout.addWidget(self.lbl_clusters)
        self.param_stack_layout.addWidget(self.spin_niches)

        self.lbl_res = QLabel("Resolution:")
        self.lbl_res.hide()
        self.spin_res = QDoubleSpinBox()
        self.spin_res.setRange(0.01, 10.0)
        self.spin_res.setValue(0.5)
        self.spin_res.setDecimals(2)
        self.spin_res.hide()
        self.param_stack_layout.addWidget(self.lbl_res)
        self.param_stack_layout.addWidget(self.spin_res)
        cb.addLayout(self.param_stack_layout)

        self.btn_niche = QPushButton("Extract Spatial Neighborhoods")
        self.btn_niche.clicked.connect(self._emit_analysis_signal)
        cb.addWidget(self.btn_niche)
        self.main_layout.addWidget(config_box, 0)

        # ── Plot & Metadata Group ─────────────────────────────────────────
        self.plot_group = QGroupBox()
        pg = QVBoxLayout()
        pg.setContentsMargins(6, 6, 6, 6)
        pg.setSpacing(6)
        self.plot_group.setLayout(pg)

        plot_hdr = QHBoxLayout()
        lbl_plot_title = QLabel("Cell Type Proportions")
        lbl_plot_title.setStyleSheet("font-weight: bold; color: #EEE; font-size: 13px;")
        plot_hdr.addWidget(lbl_plot_title)
        plot_hdr.addStretch(1)

        self.btn_popout_header = QPushButton("⤢ Pop-up Window")
        self.btn_popout_header.setToolTip("Open full-size interactive window for detailed inspection")
        self.btn_popout_header.setStyleSheet("""
            QPushButton {
                background-color: #2D5F8B; color: white; font-weight: bold;
                border: 1px solid #4B82B5; border-radius: 3px; padding: 3px 8px; font-size: 11px;
            }
            QPushButton:hover { background-color: #3D75A8; }
        """)
        self.btn_popout_header.clicked.connect(self.open_popup_window)
        plot_hdr.addWidget(self.btn_popout_header)
        pg.addLayout(plot_hdr)

        self.figure = Figure(figsize=(6, 5))
        self.figure.patch.set_facecolor('#1f232b')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(300)
        self.canvas.setToolTip("Double-click plot to open large pop-up window")
        self.canvas.mouseDoubleClickEvent = lambda event: self.open_popup_window()
        pg.addWidget(self.canvas, 3)

        self.table_meta = QTableWidget(0, 2)
        self.table_meta.setHorizontalHeaderLabels(["CN ID", "Custom Description / Name"])
        self.table_meta.verticalHeader().setVisible(False)
        self.table_meta.setMinimumHeight(140)
        self.table_meta.itemChanged.connect(self._handle_table_item_changed)

        header = self.table_meta.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_meta.setColumnWidth(0, 65)
        pg.addWidget(self.table_meta, 2)

        self.main_layout.addWidget(self.plot_group, 2)

    def _toggle_method_visibility(self, method: str) -> None:
        is_leiden = (method == "Leiden")
        self.lbl_res.setVisible(is_leiden)
        self.spin_res.setVisible(is_leiden)
        self.lbl_clusters.setVisible(not is_leiden)
        self.spin_niches.setVisible(not is_leiden)

    def _emit_analysis_signal(self) -> None:
        self.runNicheAnalysis.emit(
            self.spin_k.value(),
            self.spin_niches.value(),
            self.combo_method.currentText(),
            self.spin_res.value(),
        )

    def export_publication_plot(self) -> None:
        """Saves current composition bar chart to PDF, SVG, or high-res PNG."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Figure",
            "iris_neighborhoods_figure.pdf",
            "Vector PDF (*.pdf);;Scalable Vector Graphics (*.svg);;High-Res PNG (*.png)",
        )
        if filepath:
            self.figure.savefig(
                filepath,
                dpi=300,
                bbox_inches='tight',
                facecolor=self.figure.get_facecolor(),
            )
            QMessageBox.information(self, "Export Complete", f"Figure saved to {Path(filepath).name}.")

    def export_neighborhood_profiles_csv(self) -> None:
        """Saves composition profile matrix and cell assignment mapping to CSV."""
        if self.cached_centers is None or self.cached_labels is None:
            QMessageBox.warning(self, "No Data", "Run neighborhood computations first.")
            return

        filepath_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export Composition Matrix",
            "cellular_neighborhood_profiles.csv",
            "Comma Document (*.csv)",
        )
        if not filepath_str:
            return
        try:
            export_neighborhood_composition_matrix(
                Path(filepath_str),
                self.cached_centers,
                self.cached_labels,
                self.niche_metadata,
                self.cached_df,
            )
            QMessageBox.information(
                self,
                "Export Complete",
                "Neighborhood composition profile matrix successfully saved.",
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Export Error", f"Failed to save profile matrix: {str(exc)}"
            )

    def initialize_metadata_table(self, num_niches: int) -> None:
        """Populates the table with initial niche IDs, default names, and colors."""
        self.table_meta.blockSignals(True)
        self.table_meta.setRowCount(num_niches)
        palette = self.DEFAULT_PALETTE

        self.niche_metadata.clear()
        for i in range(num_niches):
            name_str = f"Niche {i}"
            hex_color = palette[i % len(palette)]
            self.niche_metadata[i] = {"name": name_str, "color": hex_color}

            id_item = QTableWidgetItem(f"{i}")
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            id_item.setTextAlignment(Qt.AlignCenter)
            id_item.setForeground(QColor(hex_color))
            self.table_meta.setItem(i, 0, id_item)

            name_item = QTableWidgetItem(name_str)
            self.table_meta.setItem(i, 1, name_item)

        self.table_meta.blockSignals(False)
        self.nicheMetadataChanged.emit(self.niche_metadata)

    def _handle_table_item_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        col = item.column()
        if col == 1:
            new_name = item.text().strip()
            if new_name and row in self.niche_metadata:
                self.niche_metadata[row]["name"] = new_name
                self.nicheMetadataChanged.emit(self.niche_metadata)
                if self.cached_centers is not None and self.cached_labels is not None:
                    self.display_niche_compositions(self.cached_centers, self.cached_labels)

    def _get_phenotype_color_map(self) -> Dict[str, str]:
        """
        Extracts and caches cell phenotype colors from the Legend widget.
        Ensures consistent coloration between canvas phenotypes and proportion charts.
        """
        if self.legend_widget_ref:
            rows_to_check = getattr(
                self.legend_widget_ref, "ref_ct_rows", []
            ) or getattr(self.legend_widget_ref, "seg_rows", [])
            for row in rows_to_check:
                clean_name = row.name.split(" [")[0] if " [" in row.name else row.name
                clean_name = clean_name.strip()
                if hasattr(row, "hex_color"):
                    self.cached_phenotype_colors[clean_name] = row.hex_color
                    self.cached_phenotype_colors[clean_name.lower()] = row.hex_color

        return self.cached_phenotype_colors

    def open_popup_window(self) -> None:
        """Opens or brings to front an interactive pop-up window for cell proportions."""
        if self.cached_centers is None or self.cached_labels is None:
            QMessageBox.information(
                self,
                "No Neighborhood Data",
                "Please run spatial neighborhood analysis first to display cell proportions.",
            )
            return

        pheno_colors = self._get_phenotype_color_map()
        if self.popup_dialog is None:
            self.popup_dialog = CellProportionDialog(
                centers=self.cached_centers,
                type_labels=self.cached_labels,
                niche_metadata=self.niche_metadata,
                phenotype_color_map=pheno_colors,
                parent=self.window(),
            )
        else:
            self.popup_dialog.update_plot(
                centers=self.cached_centers,
                type_labels=self.cached_labels,
                niche_metadata=self.niche_metadata,
                phenotype_color_map=pheno_colors,
            )

        self.popup_dialog.show()
        self.popup_dialog.raise_()
        self.popup_dialog.activateWindow()

    def _trigger_overlay_refresh(self) -> None:
        """Forces refresh and re-rendering of spatial neighborhood coloration on canvas."""
        if not self.niche_metadata:
            QMessageBox.information(self, "No Niches", "Please extract spatial neighborhoods first.")
            return
        self.nicheMetadataChanged.emit(self.niche_metadata)

    def toggle_phenotypes_visibility(self) -> None:
        """Toggles visibility of cell phenotype layers for easy microenvironment comparison."""
        if self.legend_widget_ref and hasattr(self.legend_widget_ref, "ref_ct_rows"):
            rows = self.legend_widget_ref.ref_ct_rows
            if rows:
                any_vis = any(r.checkbox.isChecked() for r in rows)
                new_state = not any_vis
                for r in rows:
                    r.checkbox.setChecked(new_state)

    def display_niche_compositions(
        self,
        centers: np.ndarray,
        type_labels: List[str],
        updated_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Renders stacked bar plot of cell type proportions per niche."""
        self.cached_centers = centers
        self.cached_labels = type_labels
        if updated_df is not None:
            self.cached_df = updated_df

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#14161d')

        num_niches = centers.shape[0]
        bottom = np.zeros(num_niches)
        niches_axis_labels = [
            self.niche_metadata.get(i, {}).get("name", f"Niche {i}")
            for i in range(num_niches)
        ]

        phenotype_color_map = self._get_phenotype_color_map()
        tab20_cmap = plt.colormaps.get_cmap("tab20")
        for idx, label in enumerate(type_labels):
            fractions = centers[:, idx]
            assigned_color = (
                phenotype_color_map.get(label)
                or phenotype_color_map.get(label.lower())
                or phenotype_color_map.get(label.strip())
                or tab20_cmap(idx % 20)
            )
            ax.bar(
                niches_axis_labels,
                fractions,
                bottom=bottom,
                label=label,
                color=assigned_color,
                edgecolor='black',
                linewidth=0.5,
            )
            bottom += fractions

        ax.tick_params(colors='white', labelsize=10)
        ax.set_ylabel("Cell Type Proportion", color='white', fontsize=11)
        ax.set_xticks(range(num_niches))
        ax.set_xticklabels(niches_axis_labels, rotation=25, ha='right')
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            fontsize=9,
            facecolor='#2A2A2A',
            edgecolor='#444444',
            labelcolor='white',
        )
        for spine in ax.spines.values():
            spine.set_color('#444444')

        self.figure.tight_layout()
        self.canvas.draw()

        # Synchronize pop-up window if open
        if self.popup_dialog is not None and self.popup_dialog.isVisible():
            self.popup_dialog.update_plot(
                centers=self.cached_centers,
                type_labels=self.cached_labels,
                niche_metadata=self.niche_metadata,
                phenotype_color_map=phenotype_color_map,
            )
