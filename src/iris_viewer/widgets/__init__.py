"""
IRIS-Viewer UI Components and Interactive Docks
"""
from __future__ import annotations

from .canvas_legend import CanvasLegendOverlay
from .canvas_scalebar import CanvasScaleBarOverlay
from .consensus_panel import ConsensusPanelWidget
from .feature_panel import FeaturePanelWidget
from .image_control import ImageControlWidget
from .legend import ViewerLegendWidget
from .marker_expression import IrisMarkerExpressionWidget
from .multi_sample import MultiSampleComparisonWidget
from .neighborhood import IrisCellularNeighborhoodsWidget
from .phenotyping_panel import (
    GatingStudioDialog,
    ManualGatingWidget,
    PhenotyperPanelWidget,
    SampleImpactDialog,
)
from .segmentation_panel import (
    IrisSegmentationWidget,
    SegmentationPanelWidget,
    SegmentationStudioDialog,
)
from .styles import SHARED_DARK_THEME

__all__ = [
    "ImageControlWidget",
    "ViewerLegendWidget",
    "MultiSampleComparisonWidget",
    "IrisCellularNeighborhoodsWidget",
    "IrisMarkerExpressionWidget",
    "ConsensusPanelWidget",
    "FeaturePanelWidget",
    "SegmentationPanelWidget",
    "IrisSegmentationWidget",
    "SegmentationStudioDialog",
    "PhenotyperPanelWidget",
    "ManualGatingWidget",
    "GatingStudioDialog",
    "SampleImpactDialog",
    "CanvasLegendOverlay",
    "CanvasScaleBarOverlay",
    "SHARED_DARK_THEME",
]
