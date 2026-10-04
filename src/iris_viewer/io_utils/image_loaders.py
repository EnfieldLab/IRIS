"""
IRIS-Viewer Image Loaders — re-exports from iris-common.

iris-common is now the canonical source of these functions.
This module keeps backwards compatibility for any code that
imports directly from iris_viewer.io_utils.image_loaders.
"""
# Re-export everything from the shared library
from iris_common.io.loaders import (                    # noqa: F401
    discover_image_files,
    load_multichannel_tiff,
    load_existing_mask,
    probe_channel_count,
    probe_ome_channel_names,
)

__all__ = [
    "discover_image_files",
    "load_multichannel_tiff",
    "load_existing_mask",
    "probe_channel_count",
    "probe_ome_channel_names",
]