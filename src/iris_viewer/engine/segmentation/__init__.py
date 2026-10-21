from .base import SegmentationBackend
from .cellpose_backend import CellposeBackend
from .cellsam_backend import CellSAMBackend
from .instanseg_backend import InstanSegBackend
from .stardist_backend import StardistBackend

AVAILABLE_BACKENDS = [
    CellposeBackend,
    StardistBackend,
    CellSAMBackend,
    InstanSegBackend,
]


__all__ = [
    "SegmentationBackend",
    "CellposeBackend",
    "CellSAMBackend",
    "InstanSegBackend",
    "StardistBackend",
    "AVAILABLE_BACKENDS",
    "get_available_backends",
]


def get_available_backends():
    """Returns instantiated backends that have their dependencies installed."""
    backends = []
    for cls in AVAILABLE_BACKENDS:
        backend = cls()
        backends.append(backend)
    return backends


