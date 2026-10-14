"""
IRIS-Viewer Computational Engines

Segmentation pipelines, consensus ensemble voting, feature extraction,
spatial geometry, and cellular neighborhood clustering.
"""
from __future__ import annotations

from .consensus import compute_consensus_mask, apply_manual_override
from .feature_extraction import extract_cell_features
from .pipeline import PipelineService, SegmentationSequence, SegmentationStep
from .geometry import compute_tissue_boundary_mask, recompute_area_from_vertices
from .neighborhood import identify_cellular_neighborhoods

__all__ = [
    "compute_consensus_mask",
    "apply_manual_override",
    "extract_cell_features",
    "PipelineService",
    "SegmentationSequence",
    "SegmentationStep",
    "compute_tissue_boundary_mask",
    "recompute_area_from_vertices",
    "identify_cellular_neighborhoods",
]
