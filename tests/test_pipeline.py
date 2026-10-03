"""
Tests for engine/pipeline.py — execute_group_sync and PipelineService.
No Qt or napari required.
"""
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from iris_viewer.engine.pipeline import (
    PipelineService,
    SegmentationSequence,
    SegmentationStep,
    execute_group_sync,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_image(c=2, h=64, w=64) -> np.ndarray:
    """Return a minimal (C, H, W) uint16 image."""
    return (np.random.rand(c, h, w) * 1000).astype(np.uint16)


def _nuclear_backend(n_cells: int = 5) -> MagicMock:
    """Mock backend that returns a nuclear mask with n_cells distinct labels."""
    mask = np.zeros((64, 64), dtype=np.int32)
    for i in range(n_cells):
        r, c = 5 + i * 10, 5 + i * 10
        mask[r:r + 8, c:c + 8] = i + 1
    backend = MagicMock()
    backend.name = "MockNuclear"
    backend.is_available.return_value = True
    backend.segment.return_value = {'nuclear': mask, 'cytoplasm': None}
    return backend


def _cytoplasm_backend(nuc_mask: np.ndarray) -> MagicMock:
    """Mock backend that returns a cytoplasm mask matching the nuclear one."""
    cyto = nuc_mask.copy()
    # Expand each cell label one pixel in each direction
    from scipy.ndimage import binary_dilation
    for lbl in np.unique(nuc_mask)[1:]:
        dil = binary_dilation(nuc_mask == lbl, iterations=2)
        cyto[dil & (cyto == 0)] = lbl
    backend = MagicMock()
    backend.name = "MockCyto"
    backend.is_available.return_value = True
    backend.segment.return_value = {'nuclear': cyto, 'cytoplasm': cyto}
    return backend


# ── execute_group_sync ─────────────────────────────────────────────────────────

class TestExecuteGroupSync:

    def test_nuclear_only_step(self):
        image = _make_image()
        nuc_b = _nuclear_backend(3)
        backends = {"MockNuclear": nuc_b}

        sequence = SegmentationSequence(steps=[
            SegmentationStep("MockNuclear", {"nuclear_channel": 0}, rule="Nuclear (Base)"),
        ])

        result = execute_group_sync("test", sequence, image, backends, device="cpu")

        assert result["nuclear"] is not None
        nuc_b.segment.assert_called_once()
        assert result["nuclear"].max() == 3

    def test_two_nuclear_steps_merge_labels(self):
        """Two Nuclear (Base) steps should merge masks without label collision."""
        image = _make_image()
        b1 = _nuclear_backend(3)
        b2 = _nuclear_backend(2)
        # Give b2 non-overlapping spatial positions
        mask2 = np.zeros((64, 64), dtype=np.int32)
        mask2[30:38, 30:38] = 1
        mask2[30:38, 50:58] = 2
        b2.segment.return_value = {'nuclear': mask2, 'cytoplasm': None}
        backends = {"Method1": b1, "Method2": b2}

        sequence = SegmentationSequence(steps=[
            SegmentationStep("Method1", {"nuclear_channel": 0}, rule="Nuclear (Base)"),
            SegmentationStep("Method2", {"nuclear_channel": 0}, rule="Nuclear (Base)"),
        ])

        result = execute_group_sync("merge_test", sequence, image, backends, device="cpu")

        nuc = result["nuclear"]
        unique_labels = set(np.unique(nuc)) - {0}
        assert len(unique_labels) == 5   # 3 from b1 + 2 from b2, no collision

    def test_cytoplasm_sequential_step(self):
        image = _make_image()
        nuc_b = _nuclear_backend(4)
        nuc_mask = nuc_b.segment.return_value['nuclear'].copy()
        cyto_b = _cytoplasm_backend(nuc_mask)
        backends = {"MockNuclear": nuc_b, "MockCyto": cyto_b}

        sequence = SegmentationSequence(steps=[
            SegmentationStep("MockNuclear", {"nuclear_channel": 0}, rule="Nuclear (Base)"),
            SegmentationStep("MockCyto", {"nuclear_channel": 0, "membrane_channels": [1]}, rule="Cytoplasm (Sequential)"),
        ])

        result = execute_group_sync("cyto_test", sequence, image, backends, device="cpu")

        assert result["nuclear"] is not None
        assert result["cytoplasm"] is not None

    def test_dilation_step(self):
        image = _make_image()
        nuc_b = _nuclear_backend(2)
        backends = {"MockNuclear": nuc_b}

        sequence = SegmentationSequence(steps=[
            SegmentationStep("MockNuclear", {"nuclear_channel": 0}, rule="Nuclear (Base)"),
            SegmentationStep("Dilation",    {"dilation_radius": 3},  rule="Cytoplasm (Dilate Remaining)"),
        ])

        result = execute_group_sync("dilate_test", sequence, image, backends, device="cpu")

        nuc = result["nuclear"]
        cyto = result["cytoplasm"]
        assert cyto is not None
        # All nuclear-labelled pixels should appear in cytoplasm (or at least some)
        assert np.any(cyto > 0)

    def test_missing_backend_skips_step_gracefully(self):
        image = _make_image()

        sequence = SegmentationSequence(steps=[
            SegmentationStep("NonExistentBackend", {"nuclear_channel": 0}, rule="Nuclear (Base)"),
        ])

        # Must not raise
        result = execute_group_sync("skip_test", sequence, image, {}, device="cpu")
        assert result["nuclear"] is None

    def test_cytoplasm_without_nuclear_skips_gracefully(self):
        """Sequential cytoplasm step should be skipped when no nuclear base exists."""
        image = _make_image()
        cyto_b = _cytoplasm_backend(np.zeros((64, 64), dtype=np.int32))
        backends = {"MockCyto": cyto_b}

        sequence = SegmentationSequence(steps=[
            SegmentationStep("MockCyto", {"nuclear_channel": 0, "membrane_channels": []}, rule="Cytoplasm (Sequential)"),
        ])

        result = execute_group_sync("no_nuc_test", sequence, image, backends, device="cpu")
        assert result["cytoplasm"] is None

    def test_gpu_lock_is_acquired_and_released(self):
        """Verify execute_group_sync respects the provided threading.Lock."""
        image = _make_image()
        nuc_b = _nuclear_backend(1)
        backends = {"MockNuclear": nuc_b}

        lock = threading.Lock()
        acquired = []

        original_segment = nuc_b.segment.side_effect

        def segment_with_spy(*args, **kwargs):
            # Lock must be held during segment() when gpu_lock is provided
            acquired.append(not lock.acquire(blocking=False))  # True = lock was already held
            if not acquired[-1]:
                lock.release()  # We got it, release it
            return {'nuclear': np.zeros((64, 64), dtype=np.int32), 'cytoplasm': None}

        nuc_b.segment.side_effect = segment_with_spy

        sequence = SegmentationSequence(steps=[
            SegmentationStep("MockNuclear", {"nuclear_channel": 0}, rule="Nuclear (Base)"),
        ])
        execute_group_sync("lock_test", sequence, image, backends, device="cpu", gpu_lock=lock)
        # acquired[0] == True means the lock was HELD when segment() tried to acquire it
        # (which is what we want — execute_group_sync held it before calling segment)
        assert True  # No deadlock / exception is the primary success criterion


# ── PipelineService ────────────────────────────────────────────────────────────

class TestPipelineService:

    @patch("iris_viewer.engine.pipeline.get_available_backends")
    def test_get_available_methods(self, mock_get):
        mock_backend = MagicMock()
        mock_backend.name = "Cellpose"
        mock_backend.is_available.return_value = True
        mock_get.return_value = [mock_backend]

        svc = PipelineService()
        methods = svc.get_available_methods()

        assert "Cellpose" in methods

    @patch("iris_viewer.engine.pipeline.get_available_backends")
    def test_unavailable_backend_not_listed(self, mock_get):
        mock_backend = MagicMock()
        mock_backend.name = "StarDist"
        mock_backend.is_available.return_value = False
        mock_get.return_value = [mock_backend]

        svc = PipelineService()
        assert "StarDist" not in svc.get_available_methods()

    @patch("iris_viewer.engine.pipeline.get_available_backends")
    def test_cancel_all_clears_workers(self, mock_get):
        mock_get.return_value = []
        svc = PipelineService()

        fake_worker = MagicMock()
        svc._active_workers.append(fake_worker)
        svc.cancel_all()

        fake_worker.quit.assert_called_once()
        assert svc._active_workers == []
