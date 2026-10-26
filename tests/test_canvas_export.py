"""Unit tests for canvas viewing size adjustment and napari screenshot/clipboard integration.

Verifies:
- Viewing window dimension presets and custom size locking.
- Resetting canvas to dynamic auto-fit.
- Compositing QuPath-styled scale bar into napari's in-built screenshot and clipboard methods.
- Toggling scale bar inclusion on and off for exports.
- Direct copy and save action buttons.
"""

from __future__ import annotations

import napari
import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from iris_viewer.widgets.image_control import ImageControlWidget


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QApplication instance is active for Qt tests."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def test_viewer(qapp):
    """Create a headless napari viewer with test image."""
    viewer = napari.Viewer(show=False)
    img = np.ones((500, 500), dtype=np.uint8) * 30
    viewer.add_image(img, name="TestImage")
    qapp.processEvents()
    yield viewer
    viewer.close()


def test_image_control_widget_initialization_without_viewer(qapp):
    """Verify ImageControlWidget initializes cleanly without a viewer."""
    widget = ImageControlWidget(viewer=None)
    assert widget.viewer is None
    assert widget.combo_canvas_size.currentText() == "Auto (Fit Window)"
    assert widget.chk_show_scalebar.isChecked() is True
    assert widget.chk_scalebar_in_export.isChecked() is True


def test_image_control_widget_canvas_size_presets_and_reset(test_viewer, qapp):
    """Verify canvas sizing presets lock dimensions and reset unlocks to auto-fit."""
    widget = ImageControlWidget(viewer=test_viewer)
    qapp.processEvents()

    qt_viewer = test_viewer.window._qt_viewer
    canvas_native = qt_viewer.canvas.native

    # 1. Test 800x800 square preset
    widget.combo_canvas_size.setCurrentText("800 × 800 px (Square)")
    qapp.processEvents()
    assert canvas_native.width() == 800
    assert canvas_native.height() == 800
    assert "800 × 800 px" in widget.lbl_canvas_size_status.text()

    # 2. Test 1200x1200 square preset
    widget.combo_canvas_size.setCurrentText("1200 × 1200 px (Square)")
    qapp.processEvents()
    assert canvas_native.width() == 1200
    assert canvas_native.height() == 1200
    assert "1200 × 1200 px" in widget.lbl_canvas_size_status.text()

    # 3. Test custom size application
    widget.spin_canvas_w.setValue(950)
    widget.spin_canvas_h.setValue(750)
    widget.btn_apply_canvas_size.click()
    qapp.processEvents()
    assert canvas_native.width() == 950
    assert canvas_native.height() == 750
    assert "950 × 750 px" in widget.lbl_canvas_size_status.text()

    # 4. Test reset to auto-fit
    widget.btn_reset_canvas_size.click()
    qapp.processEvents()
    assert widget.combo_canvas_size.currentText() == "Auto (Fit Window)"
    assert "Auto" in widget.lbl_canvas_size_status.text()
    assert canvas_native.maximumWidth() > 1000000


def test_canvas_scalebar_screenshot_overlay_integration(test_viewer, qapp):
    """Verify QuPath scale bar is composited into napari screenshot when enabled."""
    widget = ImageControlWidget(viewer=test_viewer)
    qapp.processEvents()

    # Ensure hook is installed
    assert getattr(test_viewer.window._qt_viewer, "_iris_screenshot_hooked", False) is True

    # Take screenshot with scale bar enabled
    widget.chk_scalebar_in_export.setChecked(True)
    shot_with_sb = test_viewer.screenshot(canvas_only=True)
    white_pixels = np.sum(
        (shot_with_sb[:, :, 0] > 240) & (shot_with_sb[:, :, 1] > 240) & (shot_with_sb[:, :, 2] > 240)
    )
    assert white_pixels > 0, "QuPath scale bar white pixels missing from screenshot"

    # Toggle scale bar inclusion off
    widget.chk_scalebar_in_export.setChecked(False)
    shot_without_sb = test_viewer.screenshot(canvas_only=True)
    white_pixels_disabled = np.sum(
        (shot_without_sb[:, :, 0] > 240) & (shot_without_sb[:, :, 1] > 240) & (shot_without_sb[:, :, 2] > 240)
    )
    assert white_pixels_disabled == 0, "Scale bar should not appear when chk_scalebar_in_export is False"


def test_image_control_widget_clipboard_copy_with_scalebar(test_viewer, qapp):
    """Verify copying canvas to clipboard via in-built napari method includes scale bar."""
    widget = ImageControlWidget(viewer=test_viewer)
    qapp.processEvents()

    widget.chk_scalebar_in_export.setChecked(True)
    widget.btn_copy_canvas.click()
    qapp.processEvents()

    clipboard = QApplication.clipboard()
    copied_img = clipboard.image()
    assert not copied_img.isNull()
    assert copied_img.width() > 0
    assert copied_img.height() > 0


def test_image_control_widget_set_viewer_late_binding(qapp):
    """Verify set_viewer binds overlays and hooks when viewer is assigned after init."""
    widget = ImageControlWidget(viewer=None)
    viewer = napari.Viewer(show=False)
    viewer.add_image(np.ones((200, 200), dtype=np.uint8))
    qapp.processEvents()

    widget.set_viewer(viewer)
    qapp.processEvents()

    assert widget.viewer is viewer
    assert widget.custom_scale_bar is not None
    assert getattr(viewer.window._qt_viewer, "_iris_screenshot_hooked", False) is True

    viewer.close()


def test_image_control_widget_ui_labels_and_button_formatting(test_viewer, qapp):
    """Verify updated label names, button styling, and absence of ellipsis on buttons."""
    from qtpy.QtWidgets import QPushButton
    widget = ImageControlWidget(viewer=test_viewer)
    qapp.processEvents()

    assert widget.chk_lock_camera.text() == "Lock Camera View"
    assert widget.chk_scalebar_in_export.text() == "Include Scale Bar in Screenshot"
    assert widget.spin_legend_opacity.value() == 0.0
    assert widget.spin_sb_opacity.value() == 0.0

    # Verify no button contains ellipsis
    for btn in widget.findChildren(QPushButton):
        txt = btn.text()
        assert "…" not in txt and "..." not in txt, f"Button '{txt}' still contains ellipsis"

    # Verify Copy Canvas button has height 28 and no custom blue stylesheet override
    assert widget.btn_copy_canvas.minimumHeight() == 28
    assert "#1E4D6B" not in widget.btn_copy_canvas.styleSheet()


def test_canvas_scalebar_background_opacity_rendering(test_viewer, qapp):
    """Verify scale bar background is transparent at 0.0 and pure black with configurable opacity."""
    from napari._qt.qt_viewer import QImg2array

    widget = ImageControlWidget(viewer=test_viewer)
    qapp.processEvents()
    sb = widget.custom_scale_bar

    # At 0.0 opacity, corner has 0 alpha
    widget.spin_sb_opacity.setValue(0.0)
    qapp.processEvents()
    img0 = sb.grab().toImage()
    arr0 = QImg2array(img0)
    assert arr0[0, 0, 3] == 0, "Background should be transparent when opacity is 0.0"

    # At 0.5 opacity, interior pixels should be pure black (R=0, G=0, B=0) with alpha ~127
    widget.spin_sb_opacity.setValue(0.5)
    qapp.processEvents()
    assert sb._bg_opacity == 0.5
    img5 = sb.grab().toImage()
    arr5 = QImg2array(img5)
    interior_pixel = arr5[6, 6]
    assert interior_pixel[0] == 0 and interior_pixel[1] == 0 and interior_pixel[2] == 0
    assert 115 <= interior_pixel[3] <= 140

