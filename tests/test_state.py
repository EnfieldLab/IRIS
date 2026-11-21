"""
Tests for core/state.py — session persistence, active method selection, and reset behaviour.
No Qt or napari required.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from iris_viewer.core.state import ViewerAppState


def _make_populated_state() -> ViewerAppState:
    st = ViewerAppState()
    st.image_dir = Path("/some/dir")
    st.markers = ["DAPI", "CD45", "CK"]
    st.current_sample_name = "sample_01"
    nuc_cp = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.uint32)
    cyto_cp = np.array([[1, 1, 2], [1, 2, 2]], dtype=np.uint32)
    nuc_sd = np.array([[0, 1, 0], [1, 0, 3]], dtype=np.uint32)
    st.segmentation_results = {
        "Cellpose": {"nuclear": nuc_cp, "cytoplasm": cyto_cp},
        "StarDist": {"nuclear": nuc_sd, "cytoplasm": None},
    }
    st.set_active_segmentation_method("Cellpose")
    st.raw_image = np.ones((3, 2, 3), dtype=np.float32)
    st.disagreement_heatmap = np.array([[0.0, 0.5, 0.8], [0.1, 0.4, 0.9]], dtype=np.float32)
    st.disagreement_report = {"flagged_regions": 2}
    st.preview_roi_bbox = (0, 0, 2, 3)
    st.persistent_settings = {"foo": "bar"}
    st.channel_colors = {"DAPI": "#0000FF"}
    st.celltype_colors = {"Tumor": "#FF0000"}
    return st


class TestSessionPersistence:

    def test_viewer_app_state_to_session_dict_fields(self):
        st = _make_populated_state()
        d = st.to_session_dict(disagreement_threshold=0.75)

        assert d["version"] == 2
        assert d["image_dir"] == "/some/dir"
        assert d["markers"] == ["DAPI", "CD45", "CK"]
        assert d["current_sample_name"] == "sample_01"
        assert d["active_segmentation_method"] == "Cellpose"
        assert d["disagreement_threshold"] == pytest.approx(0.75)
        assert d["persistent_settings"] == {"foo": "bar"}
        assert d["channel_colors"] == {"DAPI": "#0000FF"}
        assert d["celltype_colors"] == {"Tumor": "#FF0000"}

    def test_viewer_app_state_active_method_roundtrip(self):
        st = _make_populated_state()
        d = st.to_session_dict()
        restored = ViewerAppState.from_session_dict(d)

        assert restored.active_segmentation_method == "Cellpose"
        assert restored.current_sample_name == "sample_01"

    def test_viewer_app_state_numpy_arrays_not_serialised(self):
        """Large numpy arrays and report objects must NOT end up in session JSON."""
        st = _make_populated_state()
        d = st.to_session_dict()

        for excluded in (
            "raw_image",
            "current_mask",
            "cytoplasm_mask",
            "segmentation_results",
            "disagreement_heatmap",
            "disagreement_report",
            "preview_roi_bbox",
        ):
            assert excluded not in d

    def test_viewer_app_state_save_and_load_roundtrip(self):
        st = _make_populated_state()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            st.save_session(path, disagreement_threshold=0.6)

            assert path.is_file()

            restored, raw = ViewerAppState.load_session(path)
            assert restored.markers == ["DAPI", "CD45", "CK"]
            assert restored.active_segmentation_method == "Cellpose"
            assert raw["disagreement_threshold"] == pytest.approx(0.6)

    def test_viewer_app_state_load_with_missing_image_dir_does_not_raise(self):
        """Non-existent image_dir should not raise — it is a Path, not validated here."""
        data = {
            "version": 2,
            "image_dir": "/nonexistent/path",
            "markers": [],
            "current_sample_name": None,
            "active_segmentation_method": None,
            "persistent_settings": {},
        }
        restored = ViewerAppState.from_session_dict(data)
        assert restored.image_dir == Path("/nonexistent/path")
        assert restored.active_segmentation_method is None

    def test_viewer_app_state_empty_state_session_roundtrip(self):
        st = ViewerAppState()
        d = st.to_session_dict()

        assert d["image_dir"] is None
        assert d["markers"] == []
        assert d["active_segmentation_method"] is None
        assert d["disagreement_threshold"] == pytest.approx(0.65)

        restored = ViewerAppState.from_session_dict(d)
        assert restored.markers == []
        assert restored.active_segmentation_method is None


class TestActiveMethodSelection:

    def test_legacy_consensus_attributes_removed(self):
        st = ViewerAppState()
        for legacy_attr in (
            "consensus_mask",
            "confidence_map",
            "per_cell_confidence",
            "cell_method_map",
            "manual_overrides",
            "override_history",
            "snapshot_override_state",
            "undo_override",
        ):
            assert not hasattr(st, legacy_attr)

    def test_set_and_get_active_segmentation_method(self):
        st = _make_populated_state()
        nuc, cyto = st.get_active_masks()
        np.testing.assert_array_equal(nuc, st.segmentation_results["Cellpose"]["nuclear"])
        np.testing.assert_array_equal(cyto, st.segmentation_results["Cellpose"]["cytoplasm"])
        assert st.current_mask is nuc
        assert st.cytoplasm_mask is cyto

        nuc_sd, cyto_sd = st.set_active_segmentation_method("StarDist")
        assert st.active_segmentation_method == "StarDist"
        np.testing.assert_array_equal(nuc_sd, st.segmentation_results["StarDist"]["nuclear"])
        assert cyto_sd is None
        assert st.current_mask is nuc_sd
        assert st.cytoplasm_mask is None

    def test_set_active_method_missing_or_none_clears_masks(self):
        st = _make_populated_state()
        nuc, cyto = st.set_active_segmentation_method("NonExistent")
        assert (nuc, cyto) == (None, None)
        assert st.active_segmentation_method is None
        assert st.current_mask is None
        assert st.cytoplasm_mask is None

        st.set_active_segmentation_method("Cellpose")
        nuc2, cyto2 = st.set_active_segmentation_method(None)
        assert (nuc2, cyto2) == (None, None)
        assert st.active_segmentation_method is None
        assert st.current_mask is None
        assert st.cytoplasm_mask is None

    def test_get_active_masks_falls_back_to_loaded_masks_when_no_active_method(self):
        st = ViewerAppState()
        loaded_nuc = np.array([[0, 1], [1, 0]], dtype=np.uint32)
        loaded_cyto = np.array([[1, 1], [1, 0]], dtype=np.uint32)
        st.current_mask = loaded_nuc
        st.cytoplasm_mask = loaded_cyto

        nuc, cyto = st.get_active_masks()
        assert nuc is loaded_nuc
        assert cyto is loaded_cyto


class TestStateReset:

    def test_viewer_app_state_reset_sample_state_clears_volatile(self):
        st = _make_populated_state()
        st.reset_sample_state()

        assert st.raw_image is None
        assert st.segmentation_results == {}
        assert st.segmentation_status == {}
        assert st.active_segmentation_method is None
        assert st.disagreement_report is None
        assert st.disagreement_heatmap is None
        assert st.preview_roi_bbox is None
        assert st.current_mask is None
        assert st.cytoplasm_mask is None
        assert st.features_df is None
        # Global configuration preserved
        assert st.image_dir == Path("/some/dir")
        assert st.markers == ["DAPI", "CD45", "CK"]
        assert st.channel_colors == {"DAPI": "#0000FF"}
        assert st.celltype_colors == {"Tumor": "#FF0000"}

    def test_viewer_app_state_reset_all_clears_everything(self):
        st = _make_populated_state()
        st.reset_all()

        assert st.image_dir is None
        assert st.markers == []
        assert st.current_sample_name is None
        assert st.active_segmentation_method is None
        assert st.disagreement_report is None
        assert st.disagreement_heatmap is None
        assert st.preview_roi_bbox is None

    def test_viewer_app_state_channel_contrast_limits_cleared_on_reset(self):
        st = ViewerAppState()
        st.channel_contrast_limits["DAPI"] = [0.0, 100.0]
        st.reset_sample_state()

        assert st.channel_contrast_limits == {}
