"""GUI and end-to-end integration tests for Phase 3 Phenotyping UI & Studio dialogs."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QComboBox

from iris_viewer.core.context import AppContext
from iris_viewer.core.state import ViewerAppState
from iris_viewer.core.types import CellTypePanel, CellTypeRule
from iris_viewer.engine.phenotyping.astir_backend import ASTIRBackend
from iris_viewer.engine.phenotyping.maps_backend import MAPSBackend
from iris_viewer.widgets.marker_expression.dialog import MarkerExpressionDialog
from iris_viewer.widgets.phenotyping.diagnostics_dialog import PhenotyperDiagnosticDialog
from iris_viewer.widgets.phenotyping.panel import PhenotyperPanelWidget


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QApplication instance is active for Qt tests."""
    return QApplication.instance() or QApplication([])


def _make_synthetic_cohort():
    """Create a 2-sample single-cell DataFrame plus a (4, 32, 32) image & label mask."""
    rng = np.random.default_rng(7)
    n_per_sample = 12
    n_total = n_per_sample * 2

    # Build 12 distinct cells with clear T / B / Stromal marker profiles per sample
    types_a = ["T cells"] * 4 + ["B cells"] * 4 + ["Stromal"] * 4
    types_b = ["Unassigned"] * n_per_sample

    cd3 = np.concatenate([
        rng.normal(8.0, 0.4, 4), rng.normal(0.2, 0.1, 4), rng.normal(0.2, 0.1, 4),
        rng.normal(8.0, 0.4, 4), rng.normal(0.2, 0.1, 4), rng.normal(0.2, 0.1, 4),
    ]).clip(0.05)
    cd8 = np.concatenate([
        rng.normal(7.5, 0.4, 4), rng.normal(0.2, 0.1, 4), rng.normal(0.2, 0.1, 4),
        rng.normal(7.5, 0.4, 4), rng.normal(0.2, 0.1, 4), rng.normal(0.2, 0.1, 4),
    ]).clip(0.05)
    cd20 = np.concatenate([
        rng.normal(0.2, 0.1, 4), rng.normal(8.5, 0.4, 4), rng.normal(0.2, 0.1, 4),
        rng.normal(0.2, 0.1, 4), rng.normal(8.5, 0.4, 4), rng.normal(0.2, 0.1, 4),
    ]).clip(0.05)
    vim = np.concatenate([
        rng.normal(0.2, 0.1, 4), rng.normal(0.2, 0.1, 4), rng.normal(9.0, 0.4, 4),
        rng.normal(0.2, 0.1, 4), rng.normal(0.2, 0.1, 4), rng.normal(9.0, 0.4, 4),
    ]).clip(0.05)

    # Spatial grids inside 50 um neighborhood radius
    xs = np.tile(np.array([10.0, 20.0, 30.0, 40.0]), 6)
    ys = np.repeat(np.array([10.0, 25.0, 40.0]), 8)

    df = pd.DataFrame({
        "ObjectNumber": np.arange(1, n_total + 1, dtype=int),
        "Image": ["Sample_A"] * n_per_sample + ["Sample_B"] * n_per_sample,
        "Pos_X": xs,
        "Pos_Y": ys,
        "CD3_mean": cd3,
        "CD8_mean": cd8,
        "CD20_mean": cd20,
        "Vimentin_mean": vim,
        "Ref_CellType": types_a + types_b,
        "True_CellType": types_a * 2,
    })

    # Synthetic (4, 64, 64) image + 6-cell label mask (ObjectNumbers 1..6)
    img = rng.uniform(0.0, 1.0, size=(4, 64, 64)).astype(np.float32)
    mask = np.zeros((64, 64), dtype=np.int32)
    for i in range(6):
        r = 6 + (i // 3) * 26
        c = 6 + (i % 3) * 18
        mask[r : r + 12, c : c + 12] = i + 1
        if i < 3:
            img[0, r : r + 12, c : c + 12] = 10.0  # CD3
            img[1, r : r + 12, c : c + 12] = 9.0   # CD8
        else:
            img[2, r : r + 12, c : c + 12] = 11.0  # CD20

    markers = ["CD3", "CD8", "CD20", "Vimentin"]
    return df, markers, img, mask


def _make_ctx(df: pd.DataFrame, markers: list[str], img: np.ndarray, mask: np.ndarray) -> AppContext:
    state = ViewerAppState()
    state.features_df = df
    state.master_df = df.copy()
    state.current_mapped_df = df.copy()
    state.resolved_dfs = {
        "Sample_A": df[df["Image"] == "Sample_A"].copy(),
        "Sample_B": df[df["Image"] == "Sample_B"].copy(),
    }
    state.markers = list(markers)
    state.raw_image = img
    state.current_mask = mask
    state.current_sample_name = "Sample_A"

    combo_ref = QComboBox()
    combo_ref.addItems(["Ref_CellType", "True_CellType"])
    cw = SimpleNamespace(combo_ref=combo_ref)
    return AppContext(state=state, control_widget=cw)


def test_phenotyper_panel_seven_tabs_and_header_popout(qapp):
    """Verify 2-row 7-tab navigation bar and header Pop-up Window diagnostics launch."""
    df, markers, img, mask = _make_synthetic_cohort()
    ctx = _make_ctx(df, markers, img, mask)
    panel = PhenotyperPanelWidget(ctx=ctx)
    panel.set_data(df=df, markers=markers)

    assert [b.text() for b in panel.nav_buttons] == [
        "Manual Gating",
        "Clustering",
        "ASTIR",
        "MAPS",
    ]
    assert [b.text() for b in panel.dl_nav_buttons] == [
        "STELLAR",
        "CellSighter",
        "Nimbus",
    ]
    assert len(panel.all_nav_buttons) == 7
    assert panel.stack.count() == 7
    assert panel.btn_diagnostics_popout.text() == "⤢ Pop-up Window"

    for idx, btn in enumerate(panel.all_nav_buttons):
        btn.click()
        qapp.processEvents()
        assert panel.stack.currentIndex() == idx
        assert btn.isChecked() is True

    panel.btn_diagnostics_popout.click()
    qapp.processEvents()
    assert panel.diagnostics_dialog is not None
    assert isinstance(panel.diagnostics_dialog, PhenotyperDiagnosticDialog)
    assert bool(panel.diagnostics_dialog.windowFlags() & Qt.Window)

    panel.diagnostics_dialog.close()
    panel.close()
    qapp.processEvents()


def test_manual_gating_and_clustering_tabs(qapp):
    """Verify Manual Gating (Tab 0) and Clustering (Tab 1) work and sync reference columns."""
    df, markers, img, mask = _make_synthetic_cohort()
    ctx = _make_ctx(df, markers, img, mask)
    panel = PhenotyperPanelWidget(ctx=ctx)
    panel.set_data(df=df, markers=markers)

    # Manual gating
    mg = panel.manual_gating
    mg.combo_x.setCurrentText("CD3")
    mg.combo_y.setCurrentText("CD8")
    mg.combo_scale.setCurrentText("Linear (Raw)")
    mg.spin_tx.setValue(3.0)
    mg.spin_ty.setValue(3.0)
    mg._apply_gate()
    assert "Ref_CellType" in df.columns

    # Clustering
    panel.combo_cluster_method.setCurrentText("KMeans")
    panel.spin_kmeans_k.setValue(3)
    panel._run_clustering()
    assert "Cluster" in df.columns
    assert panel.list_clusters.count() == 3
    assert "Cluster" in [
        panel.combo_maps_ref_col.itemText(i) for i in range(panel.combo_maps_ref_col.count())
    ]

    panel.close()
    qapp.processEvents()


def test_astir_tab_yaml_load_and_vi_inference(qapp, tmp_path):
    """Verify ASTIR YAML loading (jackson-2020 format), VI bridge, and column propagation."""
    df, markers, img, mask = _make_synthetic_cohort()
    ctx = _make_ctx(df, markers, img, mask)
    panel = PhenotyperPanelWidget(ctx=ctx)
    panel.set_data(df=df, markers=markers)

    yaml_file = tmp_path / "test_markers.yml"
    yaml_file.write_text(
        """
cell_states:
  ACTIVATED:
    - CD8
cell_types:
  T cells:
    - CD3
    - CD8
  B cells:
    - CD20
  Stromal:
    - Vimentin
hierarchy:
  Lymphoid:
    - T cells
    - B cells
""".strip(),
        encoding="utf-8",
    )

    assert panel.btn_load_astir_yaml.text() == "Load Marker YAML..."
    loaded_panel = panel.load_astir_yaml(str(yaml_file))
    assert len(loaded_panel.rules) == 3
    assert "3 cell types" in panel.lbl_astir_yaml_summary.text()

    panel.spin_astir_epochs.setValue(10)
    panel._run_astir()

    assert "ASTIR_CellType" in df.columns
    assert "ASTIR_CellType_Confidence" in df.columns
    assert df["ASTIR_CellType_Confidence"].between(0.0, 1.0).all()
    assert "ASTIR_CellType" in ctx.state.master_df.columns
    assert "ASTIR_CellType" in ctx.state.resolved_dfs["Sample_A"].columns
    assert "ASTIR_CellType" in ctx.state.current_mapped_df.columns
    assert ctx.control_widget.combo_ref.findText("ASTIR_CellType") >= 0

    # Also test direct ASTIRBackend fit/predict bridge & auto-refit
    backend = ASTIRBackend()
    ct_panel = CellTypePanel(
        rules=[
            CellTypeRule("T cells", markers_positive=["CD3", "CD8"]),
            CellTypeRule("B cells", markers_positive=["CD20"]),
        ]
    )
    preds1 = backend.predict(df, cell_type_panel=ct_panel, max_epochs=5)
    assert len(preds1) == len(df)
    df_shifted = df.copy()
    df_shifted["CD3_mean"] = df["CD20_mean"]
    df_shifted["CD20_mean"] = df["CD3_mean"]
    preds2 = backend.predict(df_shifted, cell_type_panel=ct_panel, max_epochs=5)
    assert len(preds2) == len(df_shifted)

    panel.close()
    qapp.processEvents()


def test_maps_tab_trained_checkpoint_and_fallback(qapp, tmp_path):
    """Verify true MAPS MLP UI controls plus Fallback, Trained, and Checkpoint modes."""
    df, markers, img, mask = _make_synthetic_cohort()
    ctx = _make_ctx(df, markers, img, mask)
    panel = PhenotyperPanelWidget(ctx=ctx)
    panel.set_data(df=df, markers=markers)

    # Confirm GAT/GCN/radius inaccurate wording was removed
    maps_texts = " ".join(lbl.text() for lbl in panel.maps_widget.findChildren(type(panel.lbl_maps_status)))
    assert "GAT" not in maps_texts and "GCN" not in maps_texts
    assert [
        panel.combo_maps_mode.itemText(i) for i in range(panel.combo_maps_mode.count())
    ] == ["Train from Reference Column", "Load Checkpoint (.pt)", "Marker-Rule Fallback"]

    # 1. Marker-Rule Fallback
    panel.combo_maps_mode.setCurrentText("Marker-Rule Fallback")
    panel.txt_maps_col.setText("MAPS_Fallback")
    panel._run_maps()
    assert "MAPS_Fallback" in df.columns
    assert "MAPS_Fallback_Confidence" in df.columns

    # 2. Train from Reference Column
    panel.combo_maps_mode.setCurrentText("Train from Reference Column")
    panel.combo_maps_ref_col.setCurrentText("True_CellType")
    panel.spin_maps_hidden.setValue(32)
    panel.spin_maps_epochs.setValue(5)
    panel.txt_maps_col.setText("MAPS_CellType")
    panel._run_maps()
    assert "MAPS_CellType" in df.columns
    assert "MAPS_CellType_Confidence" in df.columns
    assert set(df["MAPS_CellType"].unique()).issubset({"T cells", "B cells", "Stromal"})

    # 3. Load Checkpoint (.pt)
    from iris_phenotyper.maps import MAPSPhenotyper

    ckpt_path = tmp_path / "maps_ckpt.pt"
    maps_model = MAPSPhenotyper(
        hidden_dim=32,
        max_epochs=5,
        min_epochs=2,
        patience=3,
        val_fraction=0.0,
        device="cpu",
    )
    maps_model.fit(
        df,
        label_col="True_CellType",
        feature_cols=["CD3_mean", "CD8_mean", "CD20_mean", "Vimentin_mean"],
    )
    maps_model.save(ckpt_path)

    panel.combo_maps_mode.setCurrentText("Load Checkpoint (.pt)")
    panel.edit_maps_ckpt.setText(str(ckpt_path))
    panel.txt_maps_col.setText("MAPS_Ckpt")
    panel._run_maps()
    assert "MAPS_Ckpt" in df.columns
    assert "MAPS_Ckpt_Confidence" in df.columns

    # Also test MAPSBackend direct fit/predict
    mb = MAPSBackend()
    mb.fit(
        df,
        label_col="True_CellType",
        feature_cols=["CD3_mean", "CD8_mean", "CD20_mean", "Vimentin_mean"],
        hidden_dim=32,
        max_epochs=4,
        min_epochs=2,
        val_fraction=0.0,
        device="cpu",
    )
    out = mb.predict(df)
    assert len(out) == len(df)

    panel.close()
    qapp.processEvents()


def test_stellar_tab_transfer_and_novelty(qapp):
    """Verify STELLAR reference -> target transfer, IsNovel flag, and canvas gate overlay."""
    df, markers, img, mask = _make_synthetic_cohort()
    ctx = _make_ctx(df, markers, img, mask)
    panel = PhenotyperPanelWidget(ctx=ctx)
    panel.set_data(df=df, markers=markers)

    gate_calls = []
    orig_on_gate = panel.manual_gating.on_gate_applied
    panel.manual_gating.on_gate_applied = lambda info: (
        gate_calls.append(info),
        orig_on_gate(info),
    )

    panel.combo_stellar_ref_col.setCurrentText("Ref_CellType")
    panel.spin_stellar_dist.setValue(50.0)
    panel.spin_stellar_epochs.setValue(3)
    panel.spin_stellar_seed_classes.setValue(2)
    panel.spin_stellar_novel_thresh.setValue(0.80)
    panel._run_stellar()

    assert "STELLAR_CellType" in df.columns
    assert "STELLAR_CellType_Confidence" in df.columns
    assert "STELLAR_CellType_IsNovel" in df.columns
    assert df["STELLAR_CellType_IsNovel"].dtype == bool
    # Target cells (Sample_B) received predictions
    tgt_labels = df.loc[df["Image"] == "Sample_B", "STELLAR_CellType"]
    assert (tgt_labels != "Unassigned").all()
    assert len(gate_calls) >= 1
    assert "STELLAR_CellType_IsNovel" in ctx.state.master_df.columns

    panel.close()
    qapp.processEvents()


def test_cellsighter_tab_train_and_predict_fast(qapp, tmp_path):
    """Verify CellSighter tab trains & predicts quickly on active image + mask."""
    df, markers, img, mask = _make_synthetic_cohort()
    # Align first 6 rows of df to Sample_A mask IDs 1..6
    df_cs = df.iloc[:6].copy()
    df_cs["Image"] = "Sample_A"
    df_cs["ObjectNumber"] = [1, 2, 3, 4, 5, 6]
    df_cs["Ref_CellType"] = ["T cells", "T cells", "T cells", "B cells", "B cells", "B cells"]

    ctx = _make_ctx(df_cs, markers, img, mask)
    ctx.state.get_active_masks = MagicMock(return_value=(mask, None))

    panel = PhenotyperPanelWidget(ctx=ctx)
    panel.set_data(df=df_cs, markers=markers)

    panel.combo_cs_mode.setCurrentText("Train on Reference Column")
    panel.combo_cs_ref_col.setCurrentText("Ref_CellType")
    panel.spin_cs_crop_size.setValue(16)
    panel.spin_cs_crop_input_size.setValue(12)
    panel.combo_cs_norm.setCurrentText("quantile")
    panel.spin_cs_epochs.setValue(1)
    panel.spin_cs_lr.setValue(1e-3)
    panel._run_cellsighter()

    ctx.state.get_active_masks.assert_called()
    assert "CellSighter_CellType" in df_cs.columns
    assert "CellSighter_CellType_Confidence" in df_cs.columns
    assert df_cs["CellSighter_CellType_Confidence"].between(0.0, 1.0).all()
    assert set(df_cs["CellSighter_CellType"].unique()).issubset({"T cells", "B cells"})

    panel.close()
    qapp.processEvents()


def test_nimbus_tab_positivity_columns_and_default_ckpt(qapp, monkeypatch):
    """Verify Nimbus default V1.pt path, U-Net inference, and positivity columns."""
    import torch

    from iris_phenotyper.nimbus.pipeline import _get_unet_classes

    df, markers, img, mask = _make_synthetic_cohort()
    df_nb = df.iloc[:6].copy()
    df_nb["Image"] = "Sample_A"
    df_nb["ObjectNumber"] = [1, 2, 3, 4, 5, 6]

    ctx = _make_ctx(df_nb, markers, img, mask)
    panel = PhenotyperPanelWidget(ctx=ctx)
    panel.set_data(df=df_nb, markers=markers)

    expected_v1 = Path(
        "/Users/wwu/Documents/IRIS_Benchmarks/nimbus/upstream_repo/"
        "src/nimbus_inference/assets/V1.pt"
    )
    if expected_v1.is_file():
        assert panel.edit_nimbus_ckpt.text() == str(expected_v1)

    # Swap heavy 142MB V1.pt load with a lightweight fresh NimbusUNet for fast unit testing
    tiny_unet = _get_unet_classes()["NimbusUNet"](filters_root=8, padding="reflect").eval()
    monkeypatch.setattr(
        "iris_phenotyper.nimbus.pipeline.load_nimbus_model",
        lambda **_kwargs: (tiny_unet, torch.device("cpu"), None),
    )

    panel.chk_nimbus_tta.setChecked(False)
    panel.spin_nimbus_quantile.setValue(0.999)
    panel.spin_nimbus_pos_thresh.setValue(0.50)
    panel.combo_nimbus_mode.setCurrentText("Panel Marker Rules")
    panel._run_nimbus()

    assert not panel.lbl_nimbus_status.text().startswith("Nimbus error"), panel.lbl_nimbus_status.text()
    assert "Nimbus_CellType" in df_nb.columns
    assert "Nimbus_CellType_Confidence" in df_nb.columns
    for m in markers:
        pos_col = f"{m}_nimbus_positivity"
        assert pos_col in df_nb.columns
        assert df_nb[pos_col].between(0.0, 1.0).all()
        assert pos_col in ctx.state.master_df.columns

    panel.close()
    qapp.processEvents()


def test_diagnostics_dialog_all_tabs_and_publication_studio(qapp):
    """Verify PhenotyperDiagnosticDialog tabs, concordance math, and Publication Studio."""
    df, markers, _, _ = _make_synthetic_cohort()
    df["Method_A_CellType"] = ["T cells"] * 8 + ["B cells"] * 8 + ["Stromal"] * 8
    df["Method_B_CellType"] = ["T cells"] * 6 + ["B cells"] * 10 + ["Stromal"] * 8
    df["Method_B_CellType_Confidence"] = np.linspace(0.25, 0.98, len(df))
    df["STELLAR_CellType"] = ["T cells"] * 8 + ["B cells"] * 8 + ["Novel_1"] * 8
    df["STELLAR_CellType_Confidence"] = np.linspace(0.30, 0.95, len(df))
    df["STELLAR_CellType_IsNovel"] = [False] * 16 + [True] * 8
    for m in markers:
        df[f"{m}_nimbus_positivity"] = np.linspace(0.1, 0.9, len(df))

    dlg = PhenotyperDiagnosticDialog(df=df)
    assert bool(dlg.windowFlags() & Qt.Window)

    # Tab 0: Confidence & Novelty
    dlg.tabs.setCurrentIndex(0)
    dlg.combo_conf_col.setCurrentText("STELLAR_CellType_Confidence")
    dlg.combo_novel_col.setCurrentText("STELLAR_CellType_IsNovel")
    dlg._render_confidence_tab()
    assert "8 / 24" in dlg.lbl_conf_summary.text()

    studio0 = dlg.open_publication_studio()
    assert studio0 is not None
    assert bool(studio0.windowFlags() & Qt.Window)
    studio0.close()

    # Tab 1: Nimbus Positivity Heatmap
    dlg.tabs.setCurrentIndex(1)
    dlg.combo_nimbus_group.setCurrentText("Method_A_CellType")
    dlg._render_nimbus_tab()
    assert dlg._cached_nimbus_df is not None
    assert dlg._cached_nimbus_df.shape == (3, len(markers))

    studio1 = dlg.open_publication_studio()
    assert studio1 is not None
    studio1.close()

    # Tab 2: Method Concordance Matrix
    dlg.tabs.setCurrentIndex(2)
    dlg.combo_conc_a.setCurrentText("Method_A_CellType")
    dlg.combo_conc_b.setCurrentText("Method_B_CellType")
    dlg._render_concordance_tab()

    conc = dlg.compute_concordance("Method_A_CellType", "Method_B_CellType")
    assert conc["n_cells"] == 24
    assert conc["accuracy"] == pytest.approx(22.0 / 24.0)
    assert conc["agreement_rate"] == pytest.approx(22.0 / 24.0)
    assert 0.0 < conc["cohens_kappa"] < 1.0
    assert "Cohen's Kappa" in dlg.lbl_conc_metrics.text()

    studio2 = dlg.open_publication_studio()
    assert studio2 is not None
    assert studio2.stats_df is not None
    studio2.close()

    dlg.close()
    qapp.processEvents()


def test_marker_expression_dialog_publication_studio(qapp):
    """Verify MarkerExpressionDialog Publication Studio launch & live re-rendering."""
    df, markers, _, _ = _make_synthetic_cohort()
    dlg = MarkerExpressionDialog(
        df=df,
        markers=markers,
        celltype_col="True_CellType",
        current_dataset="All Samples",
    )
    assert dlg.btn_pub_studio.text() == "Figure Studio"

    studio = dlg.open_publication_studio()
    assert studio is not None
    assert bool(studio.windowFlags() & Qt.Window)
    assert studio.summary_df is not None
    assert list(studio.summary_df.columns) == markers
    assert len(studio.figure.axes) >= 1

    # Switch theme and DPI via sidebar to verify plot_fn re-renders cleanly
    studio.sidebar.combo_theme.setCurrentText("Classic White")
    studio.sidebar.combo_dpi.setCurrentText("600")
    qapp.processEvents()
    assert studio.config.theme == "Classic White"
    assert studio.config.dpi == 600

    studio.close()
    dlg.close()
    qapp.processEvents()
