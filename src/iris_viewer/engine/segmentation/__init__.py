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


def get_available_backends(only_available: bool = True):
    """Return instantiated segmentation backends.

    Args:
        only_available: If True (default), include only backends whose runtime
            dependencies are installed. If False, return all registered backends.
    """
    backends = [cls() for cls in AVAILABLE_BACKENDS]
    if only_available:
        return [b for b in backends if b.is_available()]
    return backends


