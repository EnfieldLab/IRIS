"""
IRIS-Viewer Computational Engines

Segmentation pipelines, consensus ensemble voting, feature extraction,
spatial geometry, and cellular neighborhood clustering.
"""
from __future__ import annotations

from .consensus import apply_manual_override, compute_consensus_mask
from .feature_extraction import extract_cell_features
from .geometry import (
    compute_tissue_boundary_mask,
    ensure_spatial_coordinates,
    recompute_area_from_vertices,
)
from .neighborhood import identify_cellular_neighborhoods
from .pipeline import PipelineService, SegmentationSequence, SegmentationStep

__all__ = [
    "compute_consensus_mask",
    "apply_manual_override",
    "extract_cell_features",
    "PipelineService",
    "SegmentationSequence",
    "SegmentationStep",
    "compute_tissue_boundary_mask",
    "recompute_area_from_vertices",
    "ensure_spatial_coordinates",
    "identify_cellular_neighborhoods",
]
