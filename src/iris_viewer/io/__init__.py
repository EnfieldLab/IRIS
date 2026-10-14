"""
IRIS-Viewer I/O Engine

Image loading, spatial data ingestion (CSV, Parquet, AnnData), and multi-format exporters.
"""
from __future__ import annotations

from .loaders import (
    discover_image_files,
    load_multichannel_tiff,
    load_existing_mask,
    probe_channel_count,
    probe_ome_channel_names,
)
from .exporters import (
    export_consensus_mask,
    export_features_csv,
    export_confidence_map,
    export_batch_summary,
)
from .data_adapter import (
    IrisDataAdapter,
    IrisPhenotypeMapper,
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
    "IrisDataAdapter",
    "IrisPhenotypeMapper",
]
