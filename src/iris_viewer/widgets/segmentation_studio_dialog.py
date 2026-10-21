"""
iris_viewer.widgets.segmentation_studio_dialog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interactive full-size studio window for inspecting segmentation results,
adjusting consensus overlap thresholds with live histogram feedback,
and interactively navigating disputed cell masks on canvas.
"""
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class SegmentationStudioDialog(QDialog):
    """
    Interactive full-size studio window for inspecting segmentation results,
    adjusting consensus overlap thresholds with live histogram feedback,
    and interactively navigating disputed cell masks on canvas.
    """
    def __init__(self, parent=None, ctx=None, consensus_panel=None, feature_panel=None):
        super().__init__(parent)
        self.ctx = ctx
        self.consensus_panel = consensus_panel
        self.feature_panel = feature_panel
        self.setWindowTitle("Segmentation Module")
        self.resize(1150, 750)

        self.setStyleSheet("""
            QDialog { background-color: #1a1d24; color: #E0E0E0; font-size: 12px; }
            QGroupBox { border: 1px solid #444; border-radius: 8px; margin-top: 6px; padding-top: 6px; font-weight: bold; color: #DDD; }
            QPushButton {
                background-color: #2D5F8B; color: white; font-weight: bold;
                border: 1px solid #4B82B5; border-radius: 4px; padding: 5px 12px; font-size: 12px;
            }
            QPushButton:hover { background-color: #3D75A8; }
            QComboBox, QSpinBox, QSlider {
                background-color: #2A2A2A; color: white; border: 1px solid #555;
                border-radius: 4px; padding: 3px 6px; font-size: 12px;
            }
            QLabel { color: #DDD; font-size: 12px; }
            QTableWidget {
                background-color: #14161d; color: white; gridline-color: #333;
                border: 1px solid #444; selection-background-color: #3A6A9A; font-size: 11px;
            }
            QHeaderView::section {
                background-color: #222630; color: white; padding: 5px;
                border: 1px solid #333; font-weight: bold; font-size: 11px;
            }
            QTabWidget::pane { border: 1px solid #444; background: #14161d; }
            QTabBar::tab {
                background: #2A2A2A; color: #BBB; padding: 6px 14px; font-weight: bold;
                border: 1px solid #444; border-bottom: none; border-top-left-radius: 3px; border-top-right-radius: 3px;
            }
            QTabBar::tab:selected { background: #3A6A9A; color: white; }
        """)

        self._build_ui()
        self._wire_signals()
        self.refresh_data()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Top Control & KPI Bar
        top_bar = QHBoxLayout()
        top_bar.setSpacing(12)

        top_bar.addWidget(QLabel("Compare:"))
        self.combo_compare = QComboBox()
        self.combo_compare.addItems(["Nuclear Masks", "Cytoplasm Masks", "Both"])
        if self.consensus_panel is not None:
            self.combo_compare.setCurrentText(self.consensus_panel.combo_compare.currentText())
        top_bar.addWidget(self.combo_compare)

        top_bar.addWidget(QLabel("IoU Threshold:"))
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100)
        init_val = 66
        if self.consensus_panel is not None:
            init_val = self.consensus_panel.slider_threshold.value()
        self.slider_threshold.setValue(init_val)
        self.slider_threshold.setFixedWidth(130)
        self.slider_threshold.valueChanged.connect(self._on_slider_changed)
        top_bar.addWidget(self.slider_threshold)

        self.lbl_threshold_val = QLabel(f"{init_val / 100.0:.2f}")
        self.lbl_threshold_val.setStyleSheet("color: white; font-weight: bold; font-size: 12px;")
        self.lbl_threshold_val.setFixedWidth(36)
        top_bar.addWidget(self.lbl_threshold_val)

        self.btn_recompute = QPushButton("Recompute Consensus")
        self.btn_recompute.clicked.connect(self._recompute_consensus)
        top_bar.addWidget(self.btn_recompute)

        top_bar.addStretch(1)

        # KPI Badges
        kpi_box = QHBoxLayout()
        kpi_box.setSpacing(8)
        self.lbl_kpi_total = QLabel("Total: —")
        self.lbl_kpi_agreed = QLabel("Agreed: —")
        self.lbl_kpi_flagged = QLabel("Flagged: —")
        self.lbl_kpi_mean = QLabel("Mean IoU: —")

        badge_style = "background-color: #222630; padding: 4px 8px; border-radius: 4px; border: 1px solid #444; font-size: 11px;"
        for lbl in (self.lbl_kpi_total, self.lbl_kpi_agreed, self.lbl_kpi_flagged, self.lbl_kpi_mean):
            lbl.setStyleSheet(badge_style)
            kpi_box.addWidget(lbl)
        top_bar.addLayout(kpi_box)

        main_layout.addLayout(top_bar)

        # Main Splitter: Interactive Review Table & Visual Analytics
        splitter = QSplitter(Qt.Horizontal)

        # Left: Interactive Table & Issue Jump Navigation
        left_widget = QWidget()
        ll = QVBoxLayout(left_widget)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)

        grp_table = QGroupBox("Disputed Cells (Below IoU Overlap)")
        gl = QVBoxLayout(grp_table)
        gl.setContentsMargins(8, 8, 8, 8)
        gl.setSpacing(6)

        lbl_hint = QLabel("Click any cell to jump canvas camera to cell with bounding box and competing masks.")
        lbl_hint.setStyleSheet("color: #9E9E9E; font-size: 11px; font-style: italic;")
        lbl_hint.setWordWrap(True)
        gl.addWidget(lbl_hint)

        nav_row = QHBoxLayout()
        self.btn_prev_issue = QPushButton("Prev Issue")
        self.btn_prev_issue.clicked.connect(self._on_prev_issue)
        self.btn_next_issue = QPushButton("Next Issue")
        self.btn_next_issue.clicked.connect(self._on_next_issue)
        nav_row.addWidget(self.btn_prev_issue)
        nav_row.addWidget(self.btn_next_issue)
        self.lbl_table_count = QLabel("0 issues")
        self.lbl_table_count.setStyleSheet("color: #FFAA00; font-weight: bold; font-size: 11px;")
        nav_row.addStretch(1)
        nav_row.addWidget(self.lbl_table_count)
        gl.addLayout(nav_row)

        self.table_issues = QTableWidget(0, 5)
        self.table_issues.setHorizontalHeaderLabels(["#", "Cell ID", "IoU Overlap", "Competing Methods", "Decision"])
        self.table_issues.verticalHeader().setVisible(False)
        self.table_issues.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_issues.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self.table_issues.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_issues.itemSelectionChanged.connect(self._on_table_cell_selected)
        gl.addWidget(self.table_issues, 1)

        # Selected Cell Decision & Override Card
        self.grp_override = QGroupBox("Active Cell Decision")
        ol = QVBoxLayout(self.grp_override)
        ol.setContentsMargins(6, 6, 6, 6)
        ol.setSpacing(4)
        self.lbl_sel_cell_info = QLabel("No cell selected. Click an issue in the table above.")
        self.lbl_sel_cell_info.setStyleSheet("color: #AAA; font-size: 11px;")
        ol.addWidget(self.lbl_sel_cell_info)

        self.override_buttons_layout = QVBoxLayout()
        self.override_buttons_layout.setSpacing(3)
        ol.addLayout(self.override_buttons_layout)
        gl.addWidget(self.grp_override)

        ll.addWidget(grp_table)
        splitter.addWidget(left_widget)

        # Right: Analytics & Feature Exploration
        right_tabs = QTabWidget()

        # Tab 0: IoU Agreement Distribution Plot
        tab_plot = QWidget()
        tpl = QVBoxLayout(tab_plot)
        tpl.setContentsMargins(8, 8, 8, 8)
        tpl.setSpacing(6)

        lbl_plot_desc = QLabel("Distribution of cell IoU overlaps across competing algorithms. Green = Agreed, Coral = Disputed.")
        lbl_plot_desc.setStyleSheet("color: #9E9E9E; font-size: 11px; font-style: italic;")
        tpl.addWidget(lbl_plot_desc)

        self.fig_hist = Figure(figsize=(5.5, 4.5), dpi=100)
        self.fig_hist.patch.set_facecolor('#14161d')
        self.canvas_hist = FigureCanvas(self.fig_hist)
        self.canvas_hist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tpl.addWidget(self.canvas_hist, 1)
        right_tabs.addTab(tab_plot, "IoU Agreement Distribution")

        # Tab 1: Extracted Features Preview
        tab_feat = QWidget()
        tfl = QVBoxLayout(tab_feat)
        tfl.setContentsMargins(8, 8, 8, 8)
        tfl.setSpacing(6)

        self.lbl_feat_summary = QLabel("Preview of extracted morphological & intensity features per cell.")
        self.lbl_feat_summary.setStyleSheet("color: #AAA; font-size: 11px;")
        tfl.addWidget(self.lbl_feat_summary)

        self.table_features = QTableWidget(0, 0)
        self.table_features.verticalHeader().setVisible(False)
        tfl.addWidget(self.table_features, 1)

        f_btn_row = QHBoxLayout()
        self.btn_extract_feat = QPushButton("Extract Features Now")
        self.btn_extract_feat.clicked.connect(self._on_extract_features_clicked)
        f_btn_row.addWidget(self.btn_extract_feat)
        self.btn_export_feat = QPushButton("Export CSV...")
        self.btn_export_feat.clicked.connect(self._on_export_features_clicked)
        f_btn_row.addWidget(self.btn_export_feat)
        f_btn_row.addStretch(1)
        tfl.addLayout(f_btn_row)

        right_tabs.addTab(tab_feat, "Extracted Features")

        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter, 1)

    def _wire_signals(self):
        if self.consensus_panel is not None:
            self.consensus_panel.statisticsUpdated.connect(self._on_stats_updated)
            self.consensus_panel.cellReviewShown.connect(self._on_cell_review_shown)
        if self.feature_panel is not None:
            self.feature_panel.dataExtracted.connect(self._populate_feature_table)

    def _on_slider_changed(self, int_val: int):
        val = int_val / 100.0
        self.lbl_threshold_val.setText(f"{val:.2f}")
        if self.consensus_panel is not None:
            self.consensus_panel.slider_threshold.blockSignals(True)
            self.consensus_panel.slider_threshold.setValue(int_val)
            self.consensus_panel.lbl_threshold_value.setText(f"{val:.2f}")
            self.consensus_panel.slider_threshold.blockSignals(False)
            self.consensus_panel.thresholdChanged.emit(val)
        self._draw_iou_histogram()
        self._populate_issues_table()

    def _recompute_consensus(self):
        if self.consensus_panel is not None:
            self.consensus_panel.combo_compare.setCurrentText(self.combo_compare.currentText())
            self.consensus_panel.generateConsensusRequested.emit()

    def _on_stats_updated(self, total: int, agreed: int, flagged: int, mean_conf: float):
        self.refresh_data()

    def _on_cell_review_shown(self, cell_id: int, confidence: float, method_map: dict):
        for r in range(self.table_issues.rowCount()):
            item_cid = self.table_issues.item(r, 1)
            if item_cid and item_cid.text() == str(cell_id):
                self.table_issues.blockSignals(True)
                self.table_issues.selectRow(r)
                self.table_issues.blockSignals(False)
                break
        self._populate_override_card(cell_id, confidence, method_map)

    def _on_prev_issue(self):
        if self.ctx and hasattr(self.ctx, "review_svc"):
            self.ctx.review_svc.prev_issue()

    def _on_next_issue(self):
        if self.ctx and hasattr(self.ctx, "review_svc"):
            self.ctx.review_svc.next_issue()

    def _on_table_cell_selected(self):
        curr_row = self.table_issues.currentRow()
        if curr_row < 0:
            return
        item_cid = self.table_issues.item(curr_row, 1)
        if not item_cid:
            return
        try:
            cell_id = int(item_cid.text())
        except ValueError:
            return

        if self.ctx and hasattr(self.ctx, "review_svc"):
            self.ctx.review_svc.jump_to_issue(curr_row)

        conf = 0.0
        m_map = {}
        if self.ctx and hasattr(self.ctx, "state"):
            conf = self.ctx.state.per_cell_confidence.get(cell_id, 0.0)
            m_map = self.ctx.state.cell_method_map.get(cell_id, {})
        self._populate_override_card(cell_id, conf, m_map)

    def _populate_override_card(self, cell_id: int, confidence: float, method_map: dict):
        while self.override_buttons_layout.count():
            child = self.override_buttons_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        methods_txt = ", ".join(method_map.keys()) if method_map else "None"
        self.lbl_sel_cell_info.setText(
            f"<b>Cell ID:</b> {cell_id} &nbsp;|&nbsp; <b>IoU:</b> {confidence:.3f} &nbsp;|&nbsp; <b>Detected by:</b> {methods_txt}"
        )

        palette_hex = ["#FF4444", "#44FF44", "#4488FF", "#FF44FF"]
        palette_hover = ["#FF6666", "#66FF66", "#66AAFF", "#FF66FF"]

        for i, m_name in enumerate(sorted(method_map.keys())):
            bg = palette_hex[i % len(palette_hex)]
            hov = palette_hover[i % len(palette_hover)]
            btn = QPushButton(f"Use {m_name} mask")
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {bg}; color: black; padding: 5px; font-size: 11px; font-weight: bold; border-radius: 2px; }}"
                f"QPushButton:hover {{ background-color: {hov}; }}"
            )
            btn.clicked.connect(lambda chk, m=m_name, cid=cell_id: self._apply_override(cid, m))
            self.override_buttons_layout.addWidget(btn)

        btn_rem = QPushButton("Remove Cell entirely")
        btn_rem.setStyleSheet(
            "QPushButton {{ background-color: #333333; color: white; padding: 5px; font-size: 11px; margin-top: 4px; border: 1px solid #555; }}"
            "QPushButton:hover {{ background-color: #555555; }}"
        )
        btn_rem.clicked.connect(lambda chk, cid=cell_id: self._apply_override(cid, "Remove"))
        self.override_buttons_layout.addWidget(btn_rem)

    def _apply_override(self, cell_id: int, method: str):
        if self.ctx and hasattr(self.ctx, "review_svc"):
            self.ctx.review_svc.apply_manual_override(cell_id, method)
        curr_row = self.table_issues.currentRow()
        if 0 <= curr_row < self.table_issues.rowCount():
            item = QTableWidgetItem(f"Override: {method}")
            item.setForeground(Qt.green if method != "Remove" else Qt.gray)
            self.table_issues.setItem(curr_row, 4, item)

    def refresh_data(self):
        """Refreshes KPI metrics, flagged issues table, and IoU histogram."""
        if self.consensus_panel is not None and hasattr(self.consensus_panel, "cached_stats"):
            stats = self.consensus_panel.cached_stats
            tot = stats.get("total", 0)
            agr = stats.get("agreed", 0)
            flg = stats.get("flagged", 0)
            mn = stats.get("mean_conf", 0.0)
            pct_agr = (agr / tot * 100) if tot > 0 else 0
            pct_flg = (flg / tot * 100) if tot > 0 else 0
            self.lbl_kpi_total.setText(f"Total: {tot:,}")
            self.lbl_kpi_agreed.setText(f"Agreed: {agr:,} ({pct_agr:.1f}%)")
            self.lbl_kpi_flagged.setText(f"Flagged: {flg:,} ({pct_flg:.1f}%)")
            self.lbl_kpi_mean.setText(f"Mean IoU: {mn:.3f}")

        self._populate_issues_table()
        self._draw_iou_histogram()
        if self.feature_panel is not None and getattr(self.feature_panel, "cached_df", None) is not None:
            self._populate_feature_table(self.feature_panel.cached_df)

    def _populate_issues_table(self):
        self.table_issues.blockSignals(True)
        self.table_issues.setRowCount(0)

        flagged = []
        if self.ctx and hasattr(self.ctx, "flagged_cells"):
            flagged = self.ctx.flagged_cells

        self.lbl_table_count.setText(f"{len(flagged)} issues found")
        self.table_issues.setRowCount(len(flagged))

        for row_idx, cid in enumerate(flagged):
            conf = 0.0
            m_map = {}
            override_res = "Pending Review"
            if self.ctx and hasattr(self.ctx, "state"):
                conf = self.ctx.state.per_cell_confidence.get(cid, 0.0)
                m_map = self.ctx.state.cell_method_map.get(cid, {})
                if hasattr(self.ctx.state, "manual_overrides") and cid in self.ctx.state.manual_overrides:
                    override_res = f"Override: {self.ctx.state.manual_overrides[cid]}"

            item_idx = QTableWidgetItem(str(row_idx + 1))
            item_cid = QTableWidgetItem(str(cid))
            item_conf = QTableWidgetItem(f"{conf:.3f}")
            item_methods = QTableWidgetItem(", ".join(m_map.keys()))
            item_res = QTableWidgetItem(override_res)

            item_idx.setTextAlignment(Qt.AlignCenter)
            item_cid.setTextAlignment(Qt.AlignCenter)
            item_conf.setTextAlignment(Qt.AlignCenter)

            if "Override" in override_res:
                item_res.setForeground(Qt.green)

            self.table_issues.setItem(row_idx, 0, item_idx)
            self.table_issues.setItem(row_idx, 1, item_cid)
            self.table_issues.setItem(row_idx, 2, item_conf)
            self.table_issues.setItem(row_idx, 3, item_methods)
            self.table_issues.setItem(row_idx, 4, item_res)

        self.table_issues.blockSignals(False)

    def _draw_iou_histogram(self):
        self.fig_hist.clear()
        ax = self.fig_hist.add_subplot(111)
        ax.set_facecolor('#14161d')

        confs = []
        if self.ctx and hasattr(self.ctx, "state") and getattr(self.ctx.state, "per_cell_confidence", None):
            confs = list(self.ctx.state.per_cell_confidence.values())

        if not confs:
            ax.text(0.5, 0.5, "No consensus generated yet.\nRun pipelines and click 'Generate Consensus Map'.",
                    color='#888', ha='center', va='center', fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#333')
            self.canvas_hist.draw()
            return

        confs_arr = np.array(confs, dtype=float)
        thresh = self.slider_threshold.value() / 100.0

        bins = np.linspace(0.0, 1.0, 21)
        counts, edges = np.histogram(confs_arr, bins=bins)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        width = 0.042

        for center, count in zip(bin_centers, counts):
            color = '#44AA66' if center >= thresh else '#E57373'
            ax.bar(center, count, width=width, color=color, edgecolor='#222', alpha=0.85)

        ax.axvline(thresh, color='#FF5252', linestyle='--', linewidth=2, label=f'Threshold ({thresh:.2f})')
        ax.set_xlabel("Intersection-over-Union (IoU) Overlap", color='#DDD', fontsize=10, fontweight='bold')
        ax.set_ylabel("Number of Cells", color='#DDD', fontsize=10, fontweight='bold')
        ax.set_title(f"Cell Agreement Distribution ({len(confs_arr):,} cells)", color='#FFF', fontsize=11, fontweight='bold')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444')
        ax.legend(facecolor='#222630', edgecolor='#444', labelcolor='white')
        self.fig_hist.tight_layout()
        self.canvas_hist.draw()

    def _populate_feature_table(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        preview_df = df.head(50)
        n_rows, n_cols = preview_df.shape
        self.table_features.setRowCount(n_rows)
        self.table_features.setColumnCount(n_cols)
        self.table_features.setHorizontalHeaderLabels([str(c) for c in preview_df.columns])

        for r in range(n_rows):
            for c in range(n_cols):
                val = preview_df.iloc[r, c]
                txt = f"{val:.3f}" if isinstance(val, (float, np.floating)) else str(val)
                item = QTableWidgetItem(txt)
                self.table_features.setItem(r, c, item)

        self.lbl_feat_summary.setText(f"Showing first 50 rows of {len(df):,} extracted cells × {len(df.columns)} features.")

    def _on_extract_features_clicked(self):
        if self.feature_panel is not None:
            self.feature_panel._emit_extract_signal()

    def _on_export_features_clicked(self):
        if self.feature_panel is not None:
            self.feature_panel.exportTriggered.emit()


