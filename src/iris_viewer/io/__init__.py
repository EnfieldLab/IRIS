"""
iris_viewer.io — Image loading and export helpers.

All core implementations live in ``iris_common.io``.  This package
re-exports them under the ``iris_viewer.io`` namespace so application
code stays decoupled from the shared library path.
"""
from iris_common.io.loaders import (       # noqa: F401
    discover_image_files,
    load_multichannel_tiff,
    load_existing_mask,
    probe_channel_count,
    probe_ome_channel_names,
)
from iris_common.io.exporters import (     # noqa: F401
    export_consensus_mask,
    export_features_csv,
    export_confidence_map,
    export_batch_summary,
)

__all__ = [
    "discover_image_files",
    "load_multichannel_tiff",
    "load_existing_mask",
    "probe_channel_count",
    "probe_ome_channel_names",
    "export_consensus_mask",
    "export_features_csv",
    "export_confidence_map",
    "export_batch_summary",
]
