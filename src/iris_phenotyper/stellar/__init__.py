"""
STELLAR Spatial Single-Cell Phenotyping Package
===============================================
Provides the faithful STELLAR GraphSAGE + adaptive-margin + novel-discovery
pipeline and IRIS :class:`PhenotyperMethodBase` adapter.
"""
from __future__ import annotations

from iris_phenotyper.stellar.adapter import STELLARPhenotyper
from iris_phenotyper.stellar.pipeline import (
    STELLARConfig,
    STELLARGraph,
    STELLARPipeline,
    STELLARResult,
    build_spatial_graph,
    is_pyg_available,
    is_pyg_cluster_available,
    partition_graph,
    resolve_device,
)

__all__ = [
    "STELLARConfig",
    "STELLARGraph",
    "STELLARPhenotyper",
    "STELLARPipeline",
    "STELLARResult",
    "build_spatial_graph",
    "is_pyg_available",
    "is_pyg_cluster_available",
    "partition_graph",
    "resolve_device",
]
