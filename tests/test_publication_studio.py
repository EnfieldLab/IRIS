"""Unit and widget tests for Publication-Ready Figure Studio and Source Data Exporter."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import openpyxl
import pandas as pd
import pytest
from matplotlib.figure import Figure
from PIL import Image
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QApplication, QColorDialog

from iris_common.widgets.publication_studio import (
    DPI_OPTIONS,
    FONT_FAMILIES,
    SCIENTIFIC_PALETTES,
    SIZE_PRESETS,
    THEME_PRESETS,
    PlotStyleConfig,
    PublicationPlotStudioDialog,
    PublicationStyleSidebar,
    apply_plot_style,
    copy_figure_to_clipboard,
    draw_significance_brackets,
    export_publication_figure,
    export_source_data_excel,
    format_significance_label,
    get_palette_colors,
    resolve_category_colors,
)


def test_presets_and_config_defaults():
    """Verify required themes, palettes, presets, and PlotStyleConfig helpers."""
    assert set(THEME_PRESETS.keys()) == {
        "IRIS Dark",
        "Publication White (Nature)",
        "Cell Press Boxed",
        "Minimal Grid",
    }
    assert set(SCIENTIFIC_PALETTES.keys()) == {
        "Okabe-Ito",
        "Nature NPG",
        "Lancet",
        "JCO",
        "Viridis",
        "RdBu_r",
        "coolwarm",
    }
    assert set(SIZE_PRESETS.keys()) == {
        "Single Column (89 mm)",
        "1.5 Column (136 mm)",
        "Double Column (183 mm)",
        "Full Page (183 x 247 mm)",
        "Custom",
    }
    assert set(FONT_FAMILIES) == {"Arial", "Helvetica", "DejaVu Sans", "Times New Roman"}
    assert set(DPI_OPTIONS) == {150, 300, 600, 1200}

    cfg = PlotStyleConfig.from_theme(
        "Publication White (Nature)",
        width_mm=89.0,
        height_mm=70.0,
        dpi=600,
    )
    assert cfg.theme == "Publication White (Nature)"
    assert cfg.grid_mode == "None"
    assert cfg.despine_top is True
    assert np.isclose(cfg.size_inches[0], 89.0 / 25.4)
    assert np.isclose(cfg.size_inches[1], 70.0 / 25.4)

    cloned = cfg.copy()
    cloned.custom_colors["T cell"] = "#112233"
    assert "T cell" not in cfg.custom_colors

    colors = resolve_category_colors(
        ["A", "B", "C"],
        PlotStyleConfig(palette_name="Nature NPG", custom_colors={"B": "#ABCDEF"}),
    )
    assert colors["B"] == "#ABCDEF"
    assert colors["A"] == SCIENTIFIC_PALETTES["Nature NPG"][0]
    assert len(get_palette_colors("Viridis", 12)) == 12


def test_apply_plot_style_dark_and_publication_white():
    """Verify apply_plot_style applies colors, spines, ticks, fonts, grid, and overrides."""
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    bars = ax.bar(["CD4_T", "B_Cell"], [0.45, 0.55], color=["#336699", "#993366"], label="Cohort")
    ax.set_title("Original Title")
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Fraction")
    ax.legend()

    dark_cfg = PlotStyleConfig(
        theme="IRIS Dark",
        grid_mode="Y-Axis",
        tick_rotation=45,
        bold_title=True,
    )
    apply_plot_style(fig, ax, dark_cfg)
    assert fig.patch.get_facecolor()[:3] == pytest.approx(
        matplotlib.colors.to_rgb(THEME_PRESETS["IRIS Dark"]["figure_facecolor"])
    )
    assert ax.spines["top"].get_visible() is False
    assert ax.spines["right"].get_visible() is False
    assert ax.get_xticklabels()[0].get_rotation() == 45
    assert ax.title.get_fontweight() == "bold"

    white_cfg = PlotStyleConfig(
        theme="Publication White (Nature)",
        font_family="DejaVu Sans",
        title="Nature Panel A",
        xlabel="Subset",
        ylabel="Proportion (%)",
        legend_title="Group",
        custom_labels={"CD4_T": "CD4+ T cells"},
        custom_colors={"CD4_T": "#E64B35"},
        despine_top=True,
        despine_right=True,
        grid_mode="None",
    )
    apply_plot_style(fig, ax, white_cfg)
    assert fig.patch.get_facecolor()[:3] == pytest.approx((1.0, 1.0, 1.0))
    assert ax.patch.get_facecolor()[:3] == pytest.approx((1.0, 1.0, 1.0))
    assert ax.get_title() == "Nature Panel A"
    assert ax.get_xlabel() == "Subset"
    assert ax.get_ylabel() == "Proportion (%)"
    assert ax.get_xticklabels()[0].get_text() == "CD4+ T cells"
    assert bars[0].get_facecolor()[:3] == pytest.approx(matplotlib.colors.to_rgb("#E64B35"))
    assert ax.get_legend().get_title().get_text() == "Group"
    assert ax.get_legend().get_frame_on() is False

    # Reapplying without custom label restores original tick label cleanly
    white_cfg_cleared = PlotStyleConfig(theme="Publication White (Nature)")
    apply_plot_style(fig, ax, white_cfg_cleared)
    assert ax.get_xticklabels()[0].get_text() == "CD4_T"

    # Cell Press Boxed restores top/right spines; transparent sets patch alpha to 0
    boxed_cfg = PlotStyleConfig.from_theme("Cell Press Boxed", transparent=True)
    apply_plot_style(fig, ax, boxed_cfg)
    assert ax.spines["top"].get_visible() is True
    assert ax.spines["right"].get_visible() is True
    assert fig.patch.get_alpha() == 0.0
    assert ax.patch.get_alpha() == 0.0


def test_draw_significance_brackets_stacking_and_modes():
    """Verify collision-free bracket stacking, tall intermediate clearance, and hide_ns."""
    fig = Figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    y_maxes = [10.0, 22.0, 12.0, 8.0]
    x_pos = [0.0, 1.0, 2.0, 3.0]
    ax.bar(x_pos, y_maxes)
    ax.set_ylim(0, 24.0)

    comparisons = [
        (0, 1, 0.012, 0.024),
        (1, 2, 0.0008, 0.0024),
        (0, 2, 0.00005, 0.00015),
        (2, 3, 0.35, 0.42),  # ns
    ]

    cfg = PlotStyleConfig(theme="Publication White (Nature)", hide_ns=True, p_display_mode="stars")
    drawn = draw_significance_brackets(ax, comparisons, x_pos, y_maxes, config=cfg)

    assert len(drawn) == 3
    labels = [d["label"] for d in drawn]
    assert labels == ["*", "***", "****"]

    # Overlapping pairs must occupy distinct vertical levels and clear intermediate peak 22.0
    ys = [d["y"] for d in drawn]
    assert len(set(round(y, 6) for y in ys)) == 3
    assert min(ys) > 22.0
    # Wide span (0, 2) must sit above narrow overlapping spans (0, 1) and (1, 2)
    by_pair = {(d["group_a"], d["group_b"]): d for d in drawn}
    assert by_pair[(0, 2)]["y"] > by_pair[(0, 1)]["y"]
    assert by_pair[(0, 2)]["y"] > by_pair[(1, 2)]["y"]
    # Y-axis upper limit expanded cleanly above top bracket
    assert ax.get_ylim()[1] > max(ys)

    # With hide_ns=False and q_value mode using dict/DataFrame inputs
    fig2 = Figure()
    ax2 = fig2.add_subplot(111)
    ax2.bar([0, 1, 2], [4.0, 6.0, 5.0])
    ax2.set_ylim(0, 7.0)
    stats_df = pd.DataFrame(
        [
            {"group_a": "Ctrl", "group_b": "Low", "p_value": 0.02, "q_value": 0.04},
            {"group_a": "Low", "group_b": "High", "p_value": 0.25, "q_value": 0.30},
        ]
    )
    drawn2 = draw_significance_brackets(
        ax2,
        stats_df,
        {"Ctrl": 0.0, "Low": 1.0, "High": 2.0},
        {"Ctrl": 4.0, "Low": 6.0, "High": 5.0},
        config=PlotStyleConfig(hide_ns=False, p_display_mode="q_value"),
    )
    assert len(drawn2) == 2
    assert drawn2[0]["label"] == "q = 0.040"
    assert drawn2[1]["label"] == "q = 0.300"
    assert format_significance_label(0.00001, mode="p_value") == "p < 0.0001"


def test_export_publication_figure_formats_and_restore(tmp_path: Path):
    """Verify PDF (fonttype 42), SVG (fonttype none), EPS, LZW TIFF, PNG, and restore."""
    fig = Figure(figsize=(6.0, 4.5), dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(["Control", "Treated"], [12.0, 19.5], color=["#0072B2", "#D55E00"])
    ax.set_title("Publication Export Test")
    ax.set_ylabel("Density (cells/mm2)")

    cfg = PlotStyleConfig.from_theme(
        "Publication White (Nature)",
        width_mm=89.0,
        height_mm=70.0,
        dpi=150,
    )
    apply_plot_style(fig, ax, cfg)

    orig_pdf_ft = matplotlib.rcParams["pdf.fonttype"]
    orig_svg_ft = matplotlib.rcParams["svg.fonttype"]

    pdf_path = export_publication_figure(fig, tmp_path / "fig.pdf", cfg)
    svg_path = export_publication_figure(fig, tmp_path / "fig.svg", cfg)
    eps_path = export_publication_figure(fig, tmp_path / "fig.eps", cfg)
    tiff_path = export_publication_figure(fig, tmp_path / "fig.tiff", cfg)
    png_path = export_publication_figure(fig, tmp_path / "fig.png", cfg)

    # Global rcParams and original figure dimensions must be restored
    assert matplotlib.rcParams["pdf.fonttype"] == orig_pdf_ft
    assert matplotlib.rcParams["svg.fonttype"] == orig_svg_ft
    assert tuple(fig.get_size_inches()) == pytest.approx((6.0, 4.5))
    assert fig.get_dpi() == pytest.approx(100.0)

    pdf_bytes = pdf_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    assert b"/Type3" not in pdf_bytes
    assert (b"/TrueType" in pdf_bytes) or (b"/CIDFontType2" in pdf_bytes)

    svg_text = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "<text" in svg_text

    eps_bytes = eps_path.read_bytes()
    assert eps_bytes.startswith(b"%!PS-Adobe")

    with Image.open(tiff_path) as tiff_img:
        assert tiff_img.format == "TIFF"
        assert tiff_img.info.get("compression") == "tiff_lzw"
        assert tiff_img.info.get("dpi") == pytest.approx((150.0, 150.0), abs=1.0)

    with Image.open(png_path) as png_img:
        assert png_img.format == "PNG"
        assert png_img.size[0] > 0 and png_img.size[1] > 0


def test_copy_figure_to_clipboard_headless_and_gui(monkeypatch, qapp):
    """Verify headless safety when QApplication.instance() is None and GUI copy with qapp."""
    fig = Figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot([0, 1, 2], [1, 3, 2])
    cfg = PlotStyleConfig(theme="Publication White (Nature)", dpi=150)

    # Headless safety: QApplication.instance() returning None must not crash
    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))
    assert copy_figure_to_clipboard(fig, config=cfg, dpi=150) is False

    # Restore real qapp instance and verify clipboard image populated
    monkeypatch.undo()
    assert copy_figure_to_clipboard(fig, config=cfg, dpi=150) is True
    qapp.processEvents()
    clipboard_img = QApplication.clipboard().image()
    assert not clipboard_img.isNull()
    assert clipboard_img.width() > 0
    assert clipboard_img.height() > 0


def test_export_source_data_excel_multisheet(tmp_path: Path):
    """Verify multi-sheet Nature Source Data workbook structure and contents."""
    summary_df = pd.DataFrame(
        {
            "Group": ["Control", "Responder"],
            "Mean": [12.5, 24.75],
            "SEM": [1.1, 2.3],
        }
    )
    replicates_df = pd.DataFrame(
        {
            "Image": ["FOV_01", "FOV_02", "FOV_03", "FOV_04"],
            "Group": ["Control", "Control", "Responder", "Responder"],
            "Value": [11.4, 13.6, 22.45, 27.05],
        }
    )
    stats_df = pd.DataFrame(
        {
            "group_a": ["Control"],
            "group_b": ["Responder"],
            "test": ["Mann-Whitney U"],
            "p_value": [0.0042],
            "q_value": [0.0084],
        }
    )
    cfg = PlotStyleConfig(theme="Publication White (Nature)", dpi=600, title="CD8 Density")

    xlsx_path = export_source_data_excel(
        tmp_path / "source_data.xlsx",
        summary_df=summary_df,
        replicates_df=replicates_df,
        stats_df=stats_df,
        metadata=cfg,
    )
    assert xlsx_path.exists()

    wb = openpyxl.load_workbook(xlsx_path)
    assert wb.sheetnames == [
        "Summary_Data",
        "Sample_Replicates",
        "Statistical_Tests",
        "Figure_Parameters",
    ]

    ws_sum = wb["Summary_Data"]
    sum_rows = list(ws_sum.iter_rows(values_only=True))
    assert sum_rows[0] == ("Group", "Mean", "SEM")
    assert sum_rows[1] == ("Control", 12.5, 1.1)
    assert sum_rows[2] == ("Responder", 24.75, 2.3)

    ws_rep = wb["Sample_Replicates"]
    assert ws_rep.max_row == 5
    assert ws_rep.cell(2, 1).value == "FOV_01"

    ws_stats = wb["Statistical_Tests"]
    assert ws_stats.cell(1, 4).value == "p_value"
    assert ws_stats.cell(2, 4).value == pytest.approx(0.0042)

    ws_meta = wb["Figure_Parameters"]
    params = {r[0]: r[1] for r in ws_meta.iter_rows(min_row=2, values_only=True)}
    assert params["theme"] == "Publication White (Nature)"
    assert params["dpi"] == 600
    assert params["Format"] == "Nature Source Data Workbook"

    # Minimal call without optional DataFrames still writes clean workbook
    min_path = export_source_data_excel(
        tmp_path / "minimal_source_data.xlsx",
        summary_df=summary_df,
    )
    wb_min = openpyxl.load_workbook(min_path)
    assert wb_min.sheetnames == ["Summary_Data", "Figure_Parameters"]

    # Named index matrix + NaN handling preserves row labels cleanly
    matrix_df = pd.DataFrame(
        {"CD3": [1.2, np.nan], "CD20": [0.4, 3.8]},
        index=pd.Index(["T_Cell", "B_Cell"], name="Cell_Type"),
    )
    matrix_path = export_source_data_excel(tmp_path / "matrix_source.xlsx", summary_df=matrix_df)
    wb_mat = openpyxl.load_workbook(matrix_path)
    ws_mat = wb_mat["Summary_Data"]
    assert [ws_mat.cell(1, c).value for c in range(1, 4)] == ["Cell_Type", "CD3", "CD20"]
    assert ws_mat.cell(2, 1).value == "T_Cell"
    assert ws_mat.cell(3, 2).value is None or ws_mat.cell(3, 2).value == ""


def test_publication_style_sidebar_interactions(monkeypatch, qapp):
    """Verify sidebar signals, size presets, QColorDialog swatches, and label overrides."""
    sidebar = PublicationStyleSidebar(categories=["CD4_T", "Macrophage"])
    qapp.processEvents()

    emitted_configs: list[PlotStyleConfig] = []
    signals_fired: list[str] = []
    sidebar.styleChanged.connect(emitted_configs.append)
    sidebar.copyClipboardRequested.connect(lambda: signals_fired.append("copy"))
    sidebar.exportFigureRequested.connect(lambda: signals_fired.append("export_fig"))
    sidebar.exportSourceDataRequested.connect(lambda: signals_fired.append("export_xlsx"))

    sidebar.combo_theme.setCurrentText("Publication White (Nature)")
    sidebar.combo_size_preset.setCurrentText("Single Column (89 mm)")
    assert sidebar.spin_width_mm.value() == pytest.approx(89.0)

    # Manual width edit switches size preset to Custom
    sidebar.spin_width_mm.setValue(110.0)
    assert sidebar.combo_size_preset.currentText() == "Custom"

    # Inline label override
    cat_edit = sidebar._category_rows["CD4_T"]["line_edit"]
    cat_edit.setText("CD4+ T cells")

    # Per-category QColorDialog swatch button
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *a, **kw: QColor("#FF2244")))
    swatch_btn = sidebar._category_rows["Macrophage"]["button"]
    swatch_btn.click()

    qapp.processEvents()
    assert emitted_configs
    latest = emitted_configs[-1]
    assert latest.theme == "Publication White (Nature)"
    assert latest.width_mm == pytest.approx(110.0)
    assert latest.custom_labels == {"CD4_T": "CD4+ T cells"}
    assert latest.custom_colors["Macrophage"] == "#ff2244"

    sidebar.btn_copy_clipboard.click()
    sidebar.btn_export_figure.click()
    sidebar.btn_export_source_data.click()
    assert signals_fired == ["copy", "export_fig", "export_xlsx"]


def test_publication_plot_studio_dialog_end_to_end(tmp_path: Path, qapp):
    """Verify PublicationPlotStudioDialog Qt.Window flag, stats tab, live restyle, and export."""
    summary_df = pd.DataFrame({"Group": ["A", "B"], "Count": [15, 28]})
    stats_df = pd.DataFrame(
        [{"group_a": "A", "group_b": "B", "p_value": 0.003, "q_value": 0.003}]
    )

    def _plot(fig: Figure, cfg: PlotStyleConfig) -> None:
        ax = fig.add_subplot(111)
        colors = resolve_category_colors(["A", "B"], cfg)
        ax.bar(["A", "B"], [15, 28], color=[colors["A"], colors["B"]])
        ax.set_ylabel("Cell Count")
        draw_significance_brackets(ax, [("A", "B", 0.003)], {"A": 0.0, "B": 1.0}, [15, 28], cfg)

    dlg = PublicationPlotStudioDialog(
        plot_fn=_plot,
        summary_df=summary_df,
        stats_df=stats_df,
        categories=["A", "B"],
    )
    qapp.processEvents()

    assert bool(dlg.windowFlags() & Qt.Window) is True
    assert dlg.tabs.count() == 2
    assert dlg.tabs.tabText(1) == "Statistical Tests"
    assert dlg.stats_table.rowCount() == 1

    # Switch to white theme via sidebar and verify live figure update
    dlg.sidebar.combo_theme.setCurrentText("Publication White (Nature)")
    qapp.processEvents()
    assert dlg.figure.patch.get_facecolor()[:3] == pytest.approx((1.0, 1.0, 1.0))

    pdf_out = dlg.export_figure_to_path(tmp_path / "dialog_export.pdf")
    xlsx_out = dlg.export_source_data_to_path(tmp_path / "dialog_source.xlsx")
    assert pdf_out.exists() and pdf_out.stat().st_size > 0
    assert xlsx_out.exists() and xlsx_out.stat().st_size > 0
    assert dlg.copy_to_clipboard() is True

    # Dialog without stats_df omits optional statistics tab and auto-detects categories
    def _auto_cat_plot(fig: Figure, cfg: PlotStyleConfig) -> None:
        ax = fig.add_subplot(111)
        ax.bar(["Niche_0", "Niche_1"], [0.4, 0.6])

    empty_dlg = PublicationPlotStudioDialog(plot_fn=_auto_cat_plot)
    qapp.processEvents()
    assert bool(empty_dlg.windowFlags() & Qt.Window) is True
    assert empty_dlg.tabs.count() == 1
    assert set(empty_dlg.sidebar._category_rows.keys()) == {"Niche_0", "Niche_1"}

    # Reset category overrides clears custom labels and colors
    dlg.sidebar.set_category_label("A", "Group Alpha")
    dlg.sidebar.set_category_color("B", "#1188CC")
    assert dlg.config.custom_labels == {"A": "Group Alpha"}
    dlg.sidebar.btn_reset_categories.click()
    qapp.processEvents()
    assert dlg.config.custom_labels == {}
    assert dlg.config.custom_colors == {}

    # Transparent export on PNG and EPS executes cleanly
    trans_cfg = PlotStyleConfig(theme="Publication White (Nature)", transparent=True, dpi=150)
    trans_png = export_publication_figure(dlg.figure, tmp_path / "trans.png", trans_cfg)
    trans_eps = export_publication_figure(dlg.figure, tmp_path / "trans.eps", trans_cfg)
    with Image.open(trans_png) as img_rgba:
        assert img_rgba.mode == "RGBA"
    assert trans_eps.exists() and trans_eps.stat().st_size > 0

    dlg.close()
    empty_dlg.close()
