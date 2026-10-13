"""iris_common.io — shared image loaders and exporters."""
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
    export_phenotyping_results,
    export_neighborhood_composition_matrix,
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
    "export_phenotyping_results",
    "export_neighborhood_composition_matrix",
]
