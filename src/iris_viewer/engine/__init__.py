"""
IRIS-Viewer Computational Engines

Segmentation pipelines, multi-model disagreement QC, feature extraction,
spatial geometry, and cellular neighborhood clustering.
"""
from __future__ import annotations

from .disagreement import (
    DisagreementItem,
    DisagreementReport,
    compute_model_disagreement,
    render_disagreement_contours_rgba,
)
from .feature_extraction import extract_cell_features
from .geometry import (
    compute_tissue_boundary_mask,
    ensure_spatial_coordinates,
    recompute_area_from_vertices,
)
from .neighborhood import identify_cellular_neighborhoods
from .pipeline import (
    PipelineService,
    SegmentationSequence,
    SegmentationStep,
    crop_roi_and_segment_sync,
)

__all__ = [
    "DisagreementItem",
    "DisagreementReport",
    "compute_model_disagreement",
    "render_disagreement_contours_rgba",
    "extract_cell_features",
    "PipelineService",
    "SegmentationSequence",
    "SegmentationStep",
    "crop_roi_and_segment_sync",
    "compute_tissue_boundary_mask",
    "recompute_area_from_vertices",
    "ensure_spatial_coordinates",
    "identify_cellular_neighborhoods",
]
