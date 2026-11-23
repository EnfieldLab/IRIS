"""Comprehensive Round 6 verification tests for Pipeline Builder & Interactive Hotspot Curation:

1. Item 1: `StepWidget.btn_toggle_advanced` ("More Native Parameters") toggles `advanced_container`.
2. Item 2: `PipelineGroupWidget.chk_auto_complete_cyto` ("Auto-dilate unmatched nuclei"), 2-mode
   `StepWidget.STEP_TYPES` (`["Nuclear (Base)", "Cytoplasm (Sequential)"]`), and `watershed` mode
   revealing `combo_fallback_mem` and passing `fallback_membrane_channels` into
   `_dilate_remaining_cytoplasm`.
3. Item 3:
   - Visible `"Base Pipeline:"` / `"Override With:"` labels removed; agreed cells auto-seed from
     `Reference`.
   - Dynamic `[idx+1] Keep <Method> (N)` buttons in `SegmentationStudioDialog` and
     `DisagreementPanelWidget` sync live with checked methods in `Method Color Key`.
   - Interactive Keep splices chosen model into `Final Curated Mask`, sets `table_issues` column 7
     (`"Picked"`), and auto-advances `current_issue_idx` to the next hotspot.
   - `[0] No Cell / Artifact` removes false-positive cells cleanly with `N_nuc == N_cyto`.
   - Revisiting an already-curated hotspot and switching methods cleanly replaces prior spliced
     cells without inflating `curated_regions_count`.
   - `auto_adjudicate_consensus_mask` resolves uncurated hotspots while preserving manual picks.
   - `Keep in Selected Rows`, `Keep in ROI`, and `Reset to Reference` work end-to-end with zero
     QSS parse warnings, zero emojis in `src/`, and zero top-level Qt/Napari imports in `engine/`.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import re
import sys

import numpy as np
import pytest
from qtpy.QtCore import QItemSelectionModel, Qt, QtMsgType, qInstallMessageHandler
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QLabel

from iris_viewer.app.controller import ViewerController
from iris_viewer.engine.curation import (
    FINAL_CURATED_MASK_NAME,
    NO_CELL_CHOICE,
    auto_adjudicate_consensus_mask,
    initialize_curated_mask,
    splice_method_into_regions,
)
from iris_viewer.engine.disagreement import compute_model_disagreement
from iris_viewer.engine.pipeline import (
    SegmentationSequence,
    SegmentationStep,
    execute_group_sync,
)
from iris_viewer.widgets.segmentation_pipeline_builder import (
    PipelineBuilderWidget,
    PipelineGroupWidget,
    StepWidget,
)
from iris_viewer.widgets.segmentation_studio_dialog import SegmentationStudioDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _close_controller(ctrl: ViewerController, qapp: QApplication) -> None:
    sp = getattr(ctrl, "segmentation_panel", None)
    if sp is not None and getattr(sp, "popup_dialog", None) is not None:
        try:
            sp.popup_dialog.close()
        except Exception:
            pass
    try:
        ctrl.viewer.close()
    except Exception:
        pass
    qapp.processEvents()


def _build_three_model_curation_fixture(ctrl: ViewerController) -> None:
    """Populate *ctrl* with a 3-model synthetic dataset containing 1 agreed + 3 disputed hotspots.

    - Agreed cell (shared identically by ModelA, ModelB, ModelC) at y=90:104, x=90:104
    - Hotspot #1 (Split, y=10:24, x=10:36):
        ModelA = 1 merged cell; ModelB = 2 split cells; ModelC = 2 split cells
    - Hotspot #2 (Boundary Shift / Split, y=45:60, x=10:26):
        ModelA = 1 small cell; ModelB = 1 shifted cell; ModelC = 1 shifted cell matching ModelB
    - Hotspot #3 (False-positive Artifact exclusive to ModelA, y=10:24, x=80:94):
        ModelA = 1 artifact cell; ModelB = 0 cells; ModelC = 0 cells
    """
    h, w = 120, 120
    img = np.zeros((2, h, w), dtype=np.float32)
    img[0, 12:22, 12:20] = 900.0
    img[0, 12:22, 26:34] = 900.0
    img[0, 46:58, 12:24] = 850.0
    img[0, 92:102, 92:102] = 950.0
    img[1] = 200.0

    nuc_a = np.zeros((h, w), dtype=np.uint32)
    cyto_a = np.zeros((h, w), dtype=np.uint32)
    # 1: Merged top-left
    nuc_a[10:24, 10:36] = 1
    cyto_a[8:26, 8:38] = 1
    # 2: Mid-left cell
    nuc_a[46:58, 10:20] = 2
    cyto_a[44:60, 8:22] = 2
    # 3: False-positive artifact top-right
    nuc_a[10:24, 80:94] = 3
    cyto_a[8:26, 78:96] = 3
    # 4: Agreed cell bottom-right
    nuc_a[90:104, 90:104] = 4
    cyto_a[88:106, 88:106] = 4

    nuc_b = np.zeros((h, w), dtype=np.uint32)
    cyto_b = np.zeros((h, w), dtype=np.uint32)
    # Split top-left into 2 cells
    nuc_b[10:24, 10:22] = 1
    cyto_b[8:26, 8:22] = 1
    nuc_b[10:24, 24:36] = 2
    cyto_b[8:26, 23:38] = 2
    # Shifted mid-left cell (low IoU with ModelA's cell 2)
    nuc_b[46:58, 16:28] = 3
    cyto_b[44:60, 15:30] = 3
    # Agreed cell bottom-right
    nuc_b[90:104, 90:104] = 4
    cyto_b[88:106, 88:106] = 4

    nuc_c = nuc_b.copy()
    cyto_c = cyto_b.copy()

    ctrl.state.raw_image = img
    ctrl.state.markers = ["DAPI", "Mem"]
    ctrl.state.current_sample_name = "SampleR6"
    ctrl.state.segmentation_results = {
        "ModelA": {"nuclear": nuc_a, "cytoplasm": cyto_a},
        "ModelB": {"nuclear": nuc_b, "cytoplasm": cyto_b},
        "ModelC": {"nuclear": nuc_c, "cytoplasm": cyto_c},
    }
    ctrl.disagreement_panel.update_available_methods(
        ["ModelA", "ModelB", "ModelC"], active_method="ModelA"
    )
    ctrl.disagreement_panel.combo_ref_method.setCurrentText("ModelA")
    report = compute_model_disagreement(
        ctrl.state.segmentation_results,
        compare_mode="Nuclear Masks",
        reference_method="ModelA",
        iou_threshold=0.75,
    )
    ctrl.state.disagreement_report = report
    ctrl.review_svc.render_disagreement_overlays(report)


# ---------------------------------------------------------------------------
# Item 1 & Item 2: Pipeline Builder 2-Mode StepWidget & Watershed Membrane
# ---------------------------------------------------------------------------


def test_round6_step_widget_two_modes_and_more_native_parameters(qapp):
    """Verify 2-mode StepWidget dropdown, 'More Native Parameters', & membrane visibility."""
    assert StepWidget.STEP_TYPES == ["Nuclear (Base)", "Cytoplasm (Sequential)"]
    markers = ["DAPI", "PanCK", "CD45"]
    sw = StepWidget(["Cellpose", "StarDist", "InstanSeg", "CellSAM"], markers)
    try:
        sw.show()
        qapp.processEvents()

        assert [sw.combo_type.itemText(i) for i in range(sw.combo_type.count())] == [
            "Nuclear (Base)",
            "Cytoplasm (Sequential)",
        ]
        assert sw.btn_toggle_advanced.text() == "More Native Parameters"
        assert sw.btn_toggle_advanced.isChecked() is False
        assert sw.advanced_container.isVisible() is False

        # Nuclear (Base) hides membrane controls across all backends
        for method in ("Cellpose", "StarDist", "InstanSeg", "CellSAM"):
            sw.combo_method.setCurrentText(method)
            qapp.processEvents()
            assert sw.lbl_mem.isVisible() is False
            assert sw.combo_mem.isVisible() is False

        # Toggling 'More Native Parameters' opens advanced_container in Nuclear (Base)
        sw.btn_toggle_advanced.click()
        qapp.processEvents()
        assert sw.btn_toggle_advanced.isChecked() is True
        assert sw.advanced_container.isVisible() is True

        # Cytoplasm (Sequential) keeps btn_toggle_advanced visible and advanced_container synced
        sw.combo_type.setCurrentText("Cytoplasm (Sequential)")
        qapp.processEvents()
        assert sw.btn_toggle_advanced.isHidden() is False
        assert sw.advanced_container.isVisible() is True

        for method in ("Cellpose", "InstanSeg", "CellSAM"):
            sw.combo_method.setCurrentText(method)
            qapp.processEvents()
            assert sw.lbl_mem.isVisible() is True
            assert sw.combo_mem.isVisible() is True

        # StarDist hides membrane channels even in Cytoplasm (Sequential)
        sw.combo_method.setCurrentText("StarDist")
        qapp.processEvents()
        assert sw.lbl_mem.isVisible() is False
        assert sw.combo_mem.isVisible() is False

        # Clicking again collapses advanced_container
        sw.btn_toggle_advanced.click()
        qapp.processEvents()
        assert sw.btn_toggle_advanced.isChecked() is False
        assert sw.advanced_container.isVisible() is False
    finally:
        sw.close()
        qapp.processEvents()


def test_round6_pipeline_group_watershed_fallback_membrane_end_to_end(qapp):
    """Verify 'Auto-dilate unmatched nuclei', combo_fallback_mem, & watershed engine flow."""
    pb = PipelineBuilderWidget()
    try:
        pb.update_channel_dropdowns(["DAPI", "PanCK", "CD45"])
        pb.show()
        qapp.processEvents()

        group: PipelineGroupWidget = pb.tab_widget.widget(0)
        assert group.chk_auto_complete_cyto.text() == "Auto-dilate unmatched nuclei"
        assert "(N_nuc" not in group.chk_auto_complete_cyto.text()

        # Default mode is distance -> fallback membrane selector hidden
        assert group.combo_fallback_mode.currentText() == "distance"
        assert group.lbl_fallback_mem.isVisible() is False
        assert group.combo_fallback_mem.isVisible() is False

        # Switching to watershed reveals lbl_fallback_mem and combo_fallback_mem
        group.combo_fallback_mode.setCurrentText("watershed")
        qapp.processEvents()
        assert group.lbl_fallback_mem.isVisible() is True
        assert group.combo_fallback_mem.isVisible() is True

        # Toggling Auto-dilate unmatched nuclei enables/disables all completion controls
        group.chk_auto_complete_cyto.setChecked(False)
        qapp.processEvents()
        assert group.spin_fallback_dilate.isEnabled() is False
        assert group.combo_fallback_mode.isEnabled() is False
        assert group.lbl_fallback_mem.isEnabled() is False
        assert group.combo_fallback_mem.isEnabled() is False
        group.chk_auto_complete_cyto.setChecked(True)
        qapp.processEvents()
        assert group.combo_fallback_mem.isEnabled() is True

        # Select CD45 (channel 2), emit groupParamsChanged, and preserve on update_markers
        changed_count = []
        group.groupParamsChanged.connect(lambda: changed_count.append(1))
        act_cd45 = next(
            act for act in group.combo_fallback_mem.menu.actions() if act.text() == "CD45"
        )
        act_cd45.trigger()
        qapp.processEvents()
        assert len(changed_count) >= 1
        assert group.combo_fallback_mem.get_checked_items() == ["CD45"]

        pb.update_channel_dropdowns(["DAPI", "PanCK", "CD45", "CD3"])
        assert group.combo_fallback_mem.get_checked_items() == ["CD45"]
        assert [a.text() for a in group.combo_fallback_mem.menu.actions()] == [
            "DAPI",
            "PanCK",
            "CD45",
            "CD3",
        ]

        group.spin_fallback_dilate.setValue(20)
        steps = group.get_steps()
        assert steps[0].params["fallback_membrane_channels"] == [2]
        seq_ui = group.get_sequence()
        assert seq_ui.fallback_dilation_mode == "watershed"
        assert seq_ui.fallback_dilation_radius == 20
        assert seq_ui.fallback_membrane_channels == [2]

        # Engine verification: channel 1 has cleft at x=36..38, channel 2 has cleft at x=22..24
        # Point (20, 28) is assigned to cell 1 under ch1 watershed vs cell 2 under ch2 watershed!
        nuc_mask = np.zeros((40, 60), dtype=np.uint32)
        nuc_mask[18:22, 13:17] = 1
        nuc_mask[18:22, 43:47] = 2

        mem_ch1 = np.zeros((40, 60), dtype=np.float32)
        mem_ch1[10:30, 8:36] = 500.0
        mem_ch1[10:30, 38:52] = 500.0

        mem_ch2 = np.zeros((40, 60), dtype=np.float32)
        mem_ch2[10:30, 8:22] = 500.0
        mem_ch2[10:30, 24:52] = 500.0

        img = np.stack([(nuc_mask > 0).astype(np.float32) * 500.0, mem_ch1, mem_ch2], axis=0)

        class _FixedNuc:
            def segment(self, **_kw):
                return {"nuclear": nuc_mask.copy(), "cytoplasm": None}

        class _EmptyCyto:
            def segment(self, **_kw):
                return {"nuclear": None, "cytoplasm": np.zeros((40, 60), dtype=np.uint32)}

        backends = {"FixedNuc": _FixedNuc(), "EmptyCyto": _EmptyCyto()}

        # Explicit fallback_membrane_channels=[2] overrides step's membrane_channels=[1]
        seq_ch2 = SegmentationSequence(
            steps=[
                SegmentationStep("FixedNuc", {"nuclear_channel": 0}, rule="Nuclear (Base)"),
                SegmentationStep(
                    "EmptyCyto",
                    {"nuclear_channel": 0, "membrane_channels": [1]},
                    rule="Cytoplasm (Sequential)",
                ),
            ],
            auto_complete_cytoplasm=True,
            fallback_dilation_radius=20,
            fallback_dilation_mode="watershed",
            fallback_membrane_channels=seq_ui.fallback_membrane_channels,
        )
        res_ch2 = execute_group_sync("fb_ch2", seq_ch2, img, backends, device="cpu")
        assert res_ch2["cytoplasm"][20, 28] == 2
        assert (set(np.unique(res_ch2["nuclear"]).tolist()) - {0}) == {1, 2}
        assert (set(np.unique(res_ch2["cytoplasm"]).tolist()) - {0}) == {1, 2}

        # Empty fallback_membrane_channels=[] falls back to step's membrane_channels=[1]
        seq_fallback = SegmentationSequence(
            steps=seq_ch2.steps,
            auto_complete_cytoplasm=True,
            fallback_dilation_radius=20,
            fallback_dilation_mode="watershed",
            fallback_membrane_channels=[],
        )
        res_ch1 = execute_group_sync("fb_ch1", seq_fallback, img, backends, device="cpu")
        assert res_ch1["cytoplasm"][20, 28] == 1
    finally:
        pb.close()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# Item 3: Interactive Hotspot Curation & Legacy Simplification
# ---------------------------------------------------------------------------


def test_round6_curation_ui_labels_hidden_legacy_and_auto_seed_reference(qapp):
    """Confusing Base/Override labels are gone and agreed cells auto-seed from Reference."""
    ctrl = ViewerController()
    try:
        _build_three_model_curation_fixture(ctrl)
        dp = ctrl.disagreement_panel
        # Set active_segmentation_method != Reference to verify Reference takes precedence
        ctrl.state.set_active_segmentation_method("ModelC")
        dp.combo_ref_method.setCurrentText("ModelB")
        ctrl.state.disagreement_report.reference_method = "ModelB"

        studio = SegmentationStudioDialog(
            ctx=ctrl._ctx,
            disagreement_panel=dp,
            feature_panel=ctrl.feature_panel,
        )
        try:
            dp.show()
            studio.show()
            qapp.processEvents()

            # Legacy confusing labels removed; legacy programmatic attributes kept hidden
            for w in (dp, studio):
                all_label_texts = [lbl.text() for lbl in w.findChildren(QLabel)]
                assert all("Base Pipeline:" not in txt for txt in all_label_texts)
                assert all("Override With:" not in txt for txt in all_label_texts)
                assert any("Batch / ROI Method:" in txt for txt in all_label_texts)
                assert w.combo_active_method.isVisible() is False

            assert dp.btn_use_single_model.isVisible() is False
            assert studio.btn_set_active_mask.isVisible() is False

            # Auto-seed from Reference ('ModelB'), NOT active_segmentation_method ('ModelC')
            seeded = ctrl.review_svc._get_or_init_curated_state()
            assert seeded is not None
            assert seeded.base_method == "ModelB"
        finally:
            studio.close()
            qapp.processEvents()
    finally:
        _close_controller(ctrl, qapp)


def test_round6_dynamic_keep_buttons_sync_with_method_color_key(qapp):
    """Dynamic [Keep <Method> (N)] buttons & Batch combo sync live with Method Color Key."""
    ctrl = ViewerController()
    try:
        _build_three_model_curation_fixture(ctrl)
        dp = ctrl.disagreement_panel
        studio = SegmentationStudioDialog(
            ctx=ctrl._ctx,
            disagreement_panel=dp,
            feature_panel=ctrl.feature_panel,
        )
        try:
            ctrl.review_svc.jump_to_issue(0)
            studio.show()
            qapp.processEvents()

            expected = ["ModelA", "ModelB", "ModelC"]
            assert list(studio.hotspot_keep_buttons.keys()) == expected
            assert list(dp.hotspot_keep_buttons.keys()) == expected
            assert studio.hotspot_keep_buttons["ModelA"].text().startswith("[1] Keep ModelA (")
            assert studio.hotspot_keep_buttons["ModelB"].text().startswith("[2] Keep ModelB (")
            assert studio.hotspot_keep_buttons["ModelC"].text().startswith("[3] Keep ModelC (")
            assert studio.btn_keep_no_cell.text() == "[0] No Cell / Artifact"
            assert dp.btn_keep_no_cell.text() == "[0] No Cell / Artifact"

            # Uncheck ModelB in Method Color Key -> button & batch option disappear live
            dp._method_row_widgets["ModelB"]["checkbox"].setChecked(False)
            qapp.processEvents()

            assert list(studio.hotspot_keep_buttons.keys()) == ["ModelA", "ModelC"]
            assert list(dp.hotspot_keep_buttons.keys()) == ["ModelA", "ModelC"]
            assert studio.hotspot_keep_buttons["ModelC"].text().startswith("[2] Keep ModelC (")
            studio_override_items = [
                studio.combo_override_method.itemText(i)
                for i in range(studio.combo_override_method.count())
            ]
            assert "ModelB" not in studio_override_items
            assert NO_CELL_CHOICE in studio_override_items

            # Re-check ModelB -> button & batch option restore live
            dp._method_row_widgets["ModelB"]["checkbox"].setChecked(True)
            qapp.processEvents()

            assert list(studio.hotspot_keep_buttons.keys()) == expected
            assert list(dp.hotspot_keep_buttons.keys()) == expected
            studio_override_items_restored = [
                studio.combo_override_method.itemText(i)
                for i in range(studio.combo_override_method.count())
            ]
            assert "ModelB" in studio_override_items_restored
        finally:
            studio.close()
            qapp.processEvents()
    finally:
        _close_controller(ctrl, qapp)


def test_round6_interactive_hotspot_keep_auto_advance_and_revisit(qapp):
    """Keep splices model, updates 'Picked', auto-advances, & replaces cleanly on revisit."""
    ctrl = ViewerController()
    try:
        _build_three_model_curation_fixture(ctrl)
        dp = ctrl.disagreement_panel
        studio = SegmentationStudioDialog(
            ctx=ctrl._ctx,
            disagreement_panel=dp,
            feature_panel=ctrl.feature_panel,
        )
        ctrl.segmentation_panel.popup_dialog = studio
        try:
            studio.show()
            qapp.processEvents()

            assert studio.table_issues.columnCount() == 8
            assert studio.table_issues.horizontalHeaderItem(7).text() == "Picked"
            assert studio.chk_auto_advance.isChecked() is True
            assert len(ctrl._ctx.flagged_cells) == 3

            # Jump to the Split hotspot (ModelA=1 merged cell, ModelB=2 split cells)
            split_idx = next(
                i for i, it in enumerate(ctrl._ctx.flagged_cells) if it.category == "Split"
            )
            ctrl.review_svc.jump_to_issue(split_idx)
            qapp.processEvents()
            assert ctrl._ctx.current_issue_idx == split_idx

            # Click [2] Keep ModelB (...) -> splices ModelB (2 cells), marks Picked='ModelB',
            # and auto-advances current_issue_idx to the next hotspot
            studio.hotspot_keep_buttons["ModelB"].click()
            qapp.processEvents()

            assert FINAL_CURATED_MASK_NAME in ctrl.state.segmentation_results
            assert FINAL_CURATED_MASK_NAME not in studio.hotspot_keep_buttons
            curated = ctrl.state.curated_mask_state
            assert curated is not None
            assert curated.curated_regions_count == 1
            # ModelA had 4 cells (1 merged + 3 others); ModelB split cell 1 into 2 -> 5 cells total
            nuc_ids = set(np.unique(curated.nuclear).tolist()) - {0}
            cyto_ids = set(np.unique(curated.cytoplasm).tolist()) - {0}
            assert nuc_ids == {1, 2, 3, 4, 5}
            assert cyto_ids == nuc_ids
            assert studio.table_issues.item(split_idx, 7).text() == "ModelB"
            expected_next = (split_idx + 1) % len(ctrl._ctx.flagged_cells)
            assert ctrl._ctx.current_issue_idx == expected_next
            assert studio.lbl_curation_progress.text() == "Curated: 1 / 3"

            # Skip -> advances without changing curation counts
            studio.btn_skip_hotspot.click()
            qapp.processEvents()
            assert ctrl._ctx.current_issue_idx == (expected_next + 1) % len(ctrl._ctx.flagged_cells)
            assert ctrl.state.curated_mask_state.curated_regions_count == 1

            # Revisit the Split hotspot, turn off auto-advance, verify [Picked], & switch to ModelA
            ctrl.review_svc.jump_to_issue(split_idx)
            qapp.processEvents()
            assert "[Picked]" in studio.hotspot_keep_buttons["ModelB"].text()
            studio.chk_auto_advance.setChecked(False)
            studio.hotspot_keep_buttons["ModelA"].click()
            qapp.processEvents()

            revisited = ctrl.state.curated_mask_state
            assert revisited.curated_regions_count == 1  # Never inflated on revisit
            nuc_ids_after = set(np.unique(revisited.nuclear).tolist()) - {0}
            cyto_ids_after = set(np.unique(revisited.cytoplasm).tolist()) - {0}
            assert nuc_ids_after == {1, 2, 3, 4}
            assert cyto_ids_after == nuc_ids_after
            assert studio.table_issues.item(split_idx, 7).text() == "ModelA"
            assert ctrl._ctx.current_issue_idx == split_idx  # Stays put when unchecked
            assert "[Picked]" in studio.hotspot_keep_buttons["ModelA"].text()
        finally:
            ctrl.segmentation_panel.popup_dialog = None
            studio.close()
            qapp.processEvents()
    finally:
        _close_controller(ctrl, qapp)


def test_round6_no_cell_artifact_removal_and_keyboard_shortcuts(qapp):
    """[0] No Cell / Artifact removes false positives cleanly; keyboard 1..9, 0, Space work."""
    ctrl = ViewerController()
    try:
        _build_three_model_curation_fixture(ctrl)
        dp = ctrl.disagreement_panel
        studio = SegmentationStudioDialog(
            ctx=ctrl._ctx,
            disagreement_panel=dp,
            feature_panel=ctrl.feature_panel,
        )
        ctrl.segmentation_panel.popup_dialog = studio
        try:
            studio.show()
            studio.right_tabs.setCurrentIndex(1)  # Hotspot Patch Inspector tab
            qapp.processEvents()

            # Locate the Model-Exclusive false-positive artifact row (y=10:24, x=80:94)
            art_row = next(
                i
                for i, it in enumerate(ctrl._ctx.flagged_cells)
                if it.category == "Model-Exclusive"
            )
            ctrl.review_svc.jump_to_issue(art_row)
            qapp.processEvents()

            # Press Key_0 -> triggers [0] No Cell / Artifact and removes artifact cell
            studio.chk_auto_advance.setChecked(False)
            QTest.keyClick(studio, Qt.Key_0)
            qapp.processEvents()

            assert FINAL_CURATED_MASK_NAME in ctrl.state.segmentation_results
            assert FINAL_CURATED_MASK_NAME not in studio.hotspot_keep_buttons
            curated = ctrl.state.curated_mask_state
            assert curated is not None
            assert int(np.max(curated.nuclear[10:24, 80:94])) == 0
            assert int(np.max(curated.cytoplasm[8:26, 78:96])) == 0
            # Agreed cell at y=90:104, x=90:104 remains untouched and N_nuc == N_cyto
            assert int(np.max(curated.nuclear[90:104, 90:104])) > 0
            nuc_ids = set(np.unique(curated.nuclear).tolist()) - {0}
            cyto_ids = set(np.unique(curated.cytoplasm).tolist()) - {0}
            assert nuc_ids == {1, 2, 3}
            assert cyto_ids == nuc_ids
            assert studio.table_issues.item(art_row, 7).text() == NO_CELL_CHOICE
            assert "[Picked]" in studio.btn_keep_no_cell.text()

            # Verify Space skips without editing and Key_2 keeps ModelB via dock or dialog
            before_idx = ctrl._ctx.current_issue_idx
            QTest.keyClick(studio, Qt.Key_Space)
            qapp.processEvents()
            assert ctrl._ctx.current_issue_idx == (before_idx + 1) % len(ctrl._ctx.flagged_cells)

            target_idx = ctrl._ctx.current_issue_idx
            QTest.keyClick(studio, Qt.Key_2)
            qapp.processEvents()
            assert studio.table_issues.item(target_idx, 7).text() == "ModelB"

            # Verify dock widget's own Keep button also routes cleanly via hotspotKeepRequested
            dp.chk_auto_advance.setChecked(False)
            dp.hotspot_keep_buttons["ModelC"].click()
            qapp.processEvents()
            cur_idx = ctrl._ctx.current_issue_idx
            assert studio.table_issues.item(cur_idx, 7).text() == "ModelC"
        finally:
            ctrl.segmentation_panel.popup_dialog = None
            studio.close()
            qapp.processEvents()
    finally:
        _close_controller(ctrl, qapp)


def test_round6_auto_consensus_preserves_manual_and_batch_roi_reset(qapp):
    """Auto-consensus preserves manual picks; Selected Rows, ROI, & Reset work cleanly."""
    ctrl = ViewerController()
    try:
        _build_three_model_curation_fixture(ctrl)
        dp = ctrl.disagreement_panel
        studio = SegmentationStudioDialog(
            ctx=ctrl._ctx,
            disagreement_panel=dp,
            feature_panel=ctrl.feature_panel,
        )
        ctrl.segmentation_panel.popup_dialog = studio
        try:
            studio.show()
            qapp.processEvents()

            # 1. Headless engine verification: manual pick on Split hotspot preserved on consensus
            report = ctrl.state.disagreement_report
            seg = ctrl.state.segmentation_results
            split_item = next(it for it in report.items if it.category == "Split")
            shift_item = next(it for it in report.items if it.category == "Boundary Shift")

            manual_state = initialize_curated_mask(seg, "ModelA")
            # Manually pick No Cell on Split hotspot (even though consensus would pick ModelB)
            manual_state = splice_method_into_regions(
                manual_state, seg, NO_CELL_CHOICE, items=[split_item]
            )
            cons = auto_adjudicate_consensus_mask(seg, report, existing_curated=manual_state)
            assert cons.region_decisions[split_item.region_id] == NO_CELL_CHOICE
            assert int(np.max(cons.nuclear[10:24, 10:36])) == 0
            # Uncurated Boundary Shift hotspot was resolved by peer consensus (ModelB/ModelC)
            assert cons.region_decisions[shift_item.region_id].startswith("Consensus: ")
            assert int(np.max(cons.nuclear[46:58, 16:28])) > 0

            # 2. UI flow: manual pick on row 0 -> Auto-Resolve Uncurated preserves row 0
            ctrl.review_svc.jump_to_issue(0)
            studio.chk_auto_advance.setChecked(False)
            rid0 = ctrl._ctx.flagged_cells[0].region_id
            studio.btn_keep_no_cell.click()
            qapp.processEvents()
            assert studio.table_issues.item(0, 7).text() == NO_CELL_CHOICE

            studio.btn_auto_consensus.click()
            qapp.processEvents()
            assert ctrl.state.curated_mask_state.region_decisions[rid0] == NO_CELL_CHOICE
            assert studio.table_issues.item(0, 7).text() == NO_CELL_CHOICE
            other_picked = [
                studio.table_issues.item(r, 7).text()
                for r in range(1, studio.table_issues.rowCount())
            ]
            assert any(txt.startswith("Consensus:") for txt in other_picked)

            # 3. Reset to Reference clears decisions back to Reference ('ModelA')
            studio.btn_reset_curation.click()
            qapp.processEvents()
            reset_state = ctrl.state.curated_mask_state
            assert reset_state.base_method == "ModelA"
            assert reset_state.region_decisions == {}
            assert reset_state.curated_regions_count == 0
            assert studio.lbl_curation_progress.text() == "Curated: 0 / 3"
            for r in range(studio.table_issues.rowCount()):
                assert studio.table_issues.item(r, 7).text() == "Ref"

            # 4. Keep in Selected Rows (multi-row selection)
            sel = studio.table_issues.selectionModel()
            sel.clearSelection()
            for r in range(2):
                sel.select(
                    studio.table_issues.model().index(r, 0),
                    QItemSelectionModel.Select | QItemSelectionModel.Rows,
                )
            studio.combo_override_method.setCurrentText("ModelB")
            studio.btn_accept_selected_rows.click()
            qapp.processEvents()
            assert studio.table_issues.item(0, 7).text() == "ModelB"
            assert studio.table_issues.item(1, 7).text() == "ModelB"

            # 5. Draw ROI + Keep in ROI with No Cell clears cells inside drawn ROI
            studio.btn_draw_curation_roi.click()
            roi_lyr = ctrl.viewer.layers["Curation Selection ROI"]
            assert roi_lyr is not None
            # Polygon covering the agreed cell at y=90:104, x=90:104
            roi_lyr.data = [np.array([[85, 85], [85, 110], [110, 110], [110, 85]], dtype=float)]
            studio.combo_override_method.setCurrentText(NO_CELL_CHOICE)
            studio.btn_apply_curation_roi.click()
            qapp.processEvents()
            cur_after_roi = ctrl.state.curated_mask_state
            assert int(np.max(cur_after_roi.nuclear[90:104, 90:104])) == 0
            assert set(np.unique(cur_after_roi.nuclear).tolist()) == set(
                np.unique(cur_after_roi.cytoplasm).tolist()
            )
        finally:
            ctrl.segmentation_panel.popup_dialog = None
            studio.close()
            qapp.processEvents()
    finally:
        _close_controller(ctrl, qapp)


def test_round6_static_hygiene_zero_qss_warnings_and_headless_engines(qapp):
    """Verify zero emojis in src/, zero Qt QSS parse warnings, & Qt-free engine imports."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    emoji_re = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")

    emoji_hits = []
    top_level_qt_hits = []
    for py_path in sorted(src_root.rglob("*.py")):
        text = py_path.read_text(encoding="utf-8")
        if emoji_re.search(text):
            emoji_hits.append(str(py_path.relative_to(repo_root)))
        if "engine" in py_path.parts:
            tree = ast.parse(text, filename=str(py_path))
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(
                            tok in alias.name.lower()
                            for tok in ("qtpy", "pyqt", "pyside", "napari")
                        ):
                            top_level_qt_hits.append(f"{py_path.name}:{alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if any(
                        tok in node.module.lower()
                        for tok in ("qtpy", "pyqt", "pyside", "napari")
                    ):
                        top_level_qt_hits.append(f"{py_path.name}:{node.module}")

    assert emoji_hits == []
    assert top_level_qt_hits == []

    # Headless import check on curation.py and pipeline.py with qtpy/napari blocked
    saved_modules = {k: sys.modules.get(k) for k in ("qtpy", "napari") if k in sys.modules}
    sys.modules["qtpy"] = None  # type: ignore[assignment]
    sys.modules["napari"] = None  # type: ignore[assignment]
    try:
        for mod_name, rel_path in (
            ("_headless_r6_curation", src_root / "iris_viewer" / "engine" / "curation.py"),
            ("_headless_r6_pipeline", src_root / "iris_viewer" / "engine" / "pipeline.py"),
        ):
            spec = importlib.util.spec_from_file_location(mod_name, rel_path)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            sys.modules.pop(spec.name, None)
    finally:
        for k in ("qtpy", "napari"):
            if k in saved_modules:
                sys.modules[k] = saved_modules[k]
            else:
                sys.modules.pop(k, None)

    # Zero Qt stylesheet warnings while exercising Round 6 UI widgets
    qss_warnings = []

    def _msg_handler(mode, _ctx, message):
        if mode in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg):
            lower = str(message).lower()
            if "could not parse stylesheet" in lower or "stylesheet" in lower:
                qss_warnings.append(str(message))

    prev_handler = qInstallMessageHandler(_msg_handler)
    ctrl = ViewerController()
    pb = PipelineBuilderWidget()
    try:
        _build_three_model_curation_fixture(ctrl)
        dp = ctrl.disagreement_panel
        studio = SegmentationStudioDialog(
            ctx=ctrl._ctx,
            disagreement_panel=dp,
            feature_panel=ctrl.feature_panel,
        )
        ctrl.segmentation_panel.popup_dialog = studio
        try:
            pb.update_channel_dropdowns(["DAPI", "PanCK"])
            pb.show()
            grp = pb.tab_widget.widget(0)
            grp.combo_fallback_mode.setCurrentText("watershed")
            grp.step_widgets[0].btn_toggle_advanced.click()

            studio.show()
            ctrl.review_svc.jump_to_issue(0)
            studio.hotspot_keep_buttons["ModelB"].click()
            studio.btn_keep_no_cell.click()
            dp._method_row_widgets["ModelC"]["checkbox"].setChecked(False)
            studio.btn_auto_consensus.click()
            studio.btn_reset_curation.click()
            qapp.processEvents()
        finally:
            ctrl.segmentation_panel.popup_dialog = None
            studio.close()
            qapp.processEvents()
    finally:
        pb.close()
        _close_controller(ctrl, qapp)
        qInstallMessageHandler(prev_handler)

    assert qss_warnings == []
