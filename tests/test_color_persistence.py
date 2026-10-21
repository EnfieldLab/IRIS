"""
Tests for marker and cell-type color persistence across single-sample and multi-sample modes.
"""
from __future__ import annotations

import types
from pathlib import Path

from napari.utils.colormaps import Colormap

from iris_viewer.app.services.color_service import ColorService
from iris_viewer.app.services.image_service import ImageService
from iris_viewer.app.services.multi_sample_service import MultiSampleService
from iris_viewer.core.state import ViewerAppState


def test_state_color_persistence_roundtrip():
    """State should serialize and deserialize channel and celltype colors."""
    state = ViewerAppState()
    state.channel_colors = {"dapi": "#00ffff", "cd45": "#ff00ff"}
    state.celltype_colors = {"t cell": "#ff5500", "b cell": "#00aa55"}

    data = state.to_session_dict()
    assert "channel_colors" in data
    assert "celltype_colors" in data
    assert data["channel_colors"]["dapi"] == "#00ffff"
    assert data["celltype_colors"]["t cell"] == "#ff5500"

    restored = ViewerAppState.from_session_dict(data)
    assert restored.channel_colors["dapi"] == "#00ffff"
    assert restored.celltype_colors["t cell"] == "#ff5500"


def test_state_reset_sample_preserves_colors():
    """reset_sample_state should preserve channel_colors and celltype_colors."""
    state = ViewerAppState()
    state.channel_colors = {"ck": "#ffff00"}
    state.celltype_colors = {"tumor": "#e6194B"}
    state.persistent_settings = {"CK": {"colormap": "custom"}}

    state.reset_sample_state()
    assert state.channel_colors == {"ck": "#ffff00"}
    assert state.celltype_colors == {"tumor": "#e6194B"}
    assert "CK" in state.persistent_settings


def test_color_service_load_channel_colors(tmp_path: Path):
    """Loading CSV marker colors should update default_colors, state, and persistent_settings."""
    csv_file = tmp_path / "marker_colors.csv"
    csv_file.write_text("marker,color\nPanCK,#FF5733\nCD3,#33FF57\n", encoding="utf-8")

    state = ViewerAppState()
    state.markers = ["PanCK", "CD3"]

    ctx = types.SimpleNamespace(
        state=state,
        viewer=types.SimpleNamespace(layers=[]),
        control_widget=None,
        legend_widget=None,
    )

    default_colors: dict = {}
    celltype_colors: dict = {}
    svc = ColorService(ctx, default_colors, celltype_colors)

    svc.load_channel_colors(str(csv_file), show_dialog=False)

    # Verify updated in shared dict
    assert default_colors.get("panck") == "#ff5733"
    assert default_colors.get("PanCK") == "#ff5733"
    assert default_colors.get("cd3") == "#33ff57"

    # Verify updated in state
    assert state.channel_colors.get("panck") == "#ff5733"
    assert state.channel_colors.get("cd3") == "#33ff57"

    # Verify colormap created in persistent_settings
    assert "panck" in state.persistent_settings
    assert isinstance(state.persistent_settings["panck"]["colormap"], Colormap)


def test_color_service_load_celltype_colors(tmp_path: Path):
    """Loading CSV celltype colors should update celltype_colors and state."""
    csv_file = tmp_path / "celltype_colors.csv"
    csv_file.write_text("cell_type,color\nMacrophage,#800080\nEndothelial,#008080\n", encoding="utf-8")

    state = ViewerAppState()
    ctx = types.SimpleNamespace(
        state=state,
        viewer=types.SimpleNamespace(layers=[]),
        control_widget=None,
        legend_widget=None,
    )

    default_colors: dict = {}
    celltype_colors: dict = {}
    svc = ColorService(ctx, default_colors, celltype_colors)

    svc.load_celltype_colors(str(csv_file), show_dialog=False)

    assert celltype_colors.get("macrophage") == "#800080"
    assert celltype_colors.get("endothelial") == "#008080"
    assert state.celltype_colors.get("macrophage") == "#800080"


def test_multi_sample_service_marker_color_resolution():
    """MultiSampleService.get_marker_color should resolve case-insensitively with overrides."""
    state = ViewerAppState()
    ctx = types.SimpleNamespace(
        state=state,
        viewer=types.SimpleNamespace(layers=[]),
        legend_widget=None,
        control_widget=None,
    )

    svc = MultiSampleService(ctx)

    # Initially default color for DAPI
    assert svc.get_marker_color("DAPI").lower() == "#0000ff" or svc.get_marker_color("DAPI") == "blue"

    # Override via state
    state.channel_colors["dapi"] = "#00ffff"
    assert svc.get_marker_color("DAPI") == "#00ffff"
    assert svc.get_marker_color("dapi") == "#00ffff"
    assert svc.get_marker_color("Dapi") == "#00ffff"


def test_image_service_marker_color_resolution():
    """ImageService.get_marker_color should query multi_sample_svc and state."""
    state = ViewerAppState()
    ctx = types.SimpleNamespace(
        state=state,
        viewer=types.SimpleNamespace(layers=[]),
        legend_widget=None,
        control_widget=None,
    )
    multi_svc = MultiSampleService(ctx)
    ctx.multi_sample_svc = multi_svc

    img_svc = ImageService(ctx)

    # Override marker color
    state.channel_colors["cd45"] = "#ff0077"
    assert img_svc.get_marker_color("CD45") == "#ff0077"
    assert img_svc.get_marker_color("cd45") == "#ff0077"
