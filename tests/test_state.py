"""
Tests for core/state.py — session persistence and override undo stack.
No Qt or napari required.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from iris_viewer.core.state import ViewerAppState

# Helpers

def _make_populated_state() -> ViewerAppState:
    st = ViewerAppState()
    st.image_dir = Path("/some/dir")
    st.markers = ["DAPI", "CD45", "CK"]
    st.current_sample_name = "sample_01"
    st.per_cell_confidence = {1: 0.9, 2: 0.4, 3: 0.75}
    st.cell_method_map = {1: {"Cellpose": 10}, 2: {"StarDist": 12, "Cellpose": 14}}
    st.manual_overrides = {2: "Cellpose"}
    st.consensus_mask = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    st.confidence_map = np.array([[0.0, 0.9, 0.4], [0.9, 0.4, 0.75]], dtype=np.float32)
    st.persistent_settings = {"foo": "bar"}
    return st


# Session Serialisation

class TestSessionPersistence:

    def test_to_session_dict_fields(self):
        st = _make_populated_state()
        d = st.to_session_dict(consensus_threshold=0.75)

        assert d["version"] == 1
        assert d["image_dir"] == "/some/dir"
        assert d["markers"] == ["DAPI", "CD45", "CK"]
        assert d["current_sample_name"] == "sample_01"
        assert d["consensus_threshold"] == 0.75
        assert d["persistent_settings"] == {"foo": "bar"}

    def test_per_cell_confidence_roundtrip(self):
        st = _make_populated_state()
        d = st.to_session_dict()
        restored = ViewerAppState.from_session_dict(d)

        assert restored.per_cell_confidence == {1: 0.9, 2: 0.4, 3: 0.75}

    def test_cell_method_map_roundtrip(self):
        st = _make_populated_state()
        d = st.to_session_dict()
        restored = ViewerAppState.from_session_dict(d)

        assert restored.cell_method_map == {1: {"Cellpose": 10}, 2: {"StarDist": 12, "Cellpose": 14}}

    def test_manual_overrides_roundtrip(self):
        st = _make_populated_state()
        d = st.to_session_dict()
        restored = ViewerAppState.from_session_dict(d)

        assert restored.manual_overrides == {2: "Cellpose"}

    def test_numpy_arrays_not_serialised(self):
        """Arrays must NOT end up in the session JSON."""
        st = _make_populated_state()
        d = st.to_session_dict()

        assert "consensus_mask" not in d
        assert "raw_image" not in d
        assert "confidence_map" not in d

    def test_save_and_load_roundtrip(self):
        st = _make_populated_state()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            st.save_session(path, consensus_threshold=0.6)

            assert path.is_file()

            restored, raw = ViewerAppState.load_session(path)
            assert restored.markers == ["DAPI", "CD45", "CK"]
            assert restored.per_cell_confidence == {1: 0.9, 2: 0.4, 3: 0.75}
            assert raw["consensus_threshold"] == pytest.approx(0.6)

    def test_load_with_missing_image_dir_does_not_raise(self):
        """Non-existent image_dir should not raise — it is a Path, not validated here."""
        data = {
            "version": 1,
            "image_dir": "/nonexistent/path",
            "markers": [],
            "current_sample_name": None,
            "per_cell_confidence": {},
            "manual_overrides": {},
            "persistent_settings": {},
        }
        restored = ViewerAppState.from_session_dict(data)
        assert restored.image_dir == Path("/nonexistent/path")

    def test_empty_state_session_roundtrip(self):
        st = ViewerAppState()
        d = st.to_session_dict()

        assert d["image_dir"] is None
        assert d["markers"] == []

        restored = ViewerAppState.from_session_dict(d)
        assert restored.markers == []
        assert restored.per_cell_confidence == {}


# Override Undo Stack

class TestOverrideHistory:

    def test_snapshot_and_undo(self):
        st = _make_populated_state()
        orig_mask = st.consensus_mask.copy()

        st.snapshot_override_state()

        # Mutate state as an override would
        st.consensus_mask = np.zeros_like(orig_mask)
        st.per_cell_confidence = {}
        st.manual_overrides = {}
        st.cell_method_map = {}

        success = st.undo_override()
        assert success is True
        np.testing.assert_array_equal(st.consensus_mask, orig_mask)
        assert st.per_cell_confidence == {1: 0.9, 2: 0.4, 3: 0.75}
        assert st.cell_method_map == {1: {"Cellpose": 10}, 2: {"StarDist": 12, "Cellpose": 14}}

    def test_undo_when_empty_returns_false(self):
        st = ViewerAppState()
        assert st.undo_override() is False

    def test_maxlen_keeps_only_50_snapshots(self):
        st = _make_populated_state()
        for _ in range(60):
            st.snapshot_override_state()

        assert len(st.override_history) == 50

    def test_reset_clears_history(self):
        st = _make_populated_state()
        st.snapshot_override_state()
        st.reset_sample_state()

        assert len(st.override_history) == 0
        assert st.undo_override() is False


# Reset Behaviour

class TestStateReset:

    def test_reset_sample_state_clears_volatile(self):
        st = _make_populated_state()
        st.reset_sample_state()

        assert st.raw_image is None
        assert st.segmentation_results == {}
        assert st.consensus_mask is None
        assert st.per_cell_confidence == {}
        assert st.features_df is None
        # image_dir and markers are preserved
        assert st.image_dir == Path("/some/dir")
        assert st.markers == ["DAPI", "CD45", "CK"]

    def test_reset_all_clears_everything(self):
        st = _make_populated_state()
        st.reset_all()

        assert st.image_dir is None
        assert st.markers == []
        assert st.current_sample_name is None

    def test_channel_contrast_limits_cleared_on_reset(self):
        st = ViewerAppState()
        st.channel_contrast_limits["DAPI"] = [0.0, 100.0]
        st.reset_sample_state()

        assert st.channel_contrast_limits == {}
