"""iris_viewer.widgets — interactive panels for IRIS-Viewer."""
from .consensus_panel import ConsensusPanelWidget
from .feature_panel import FeaturePanelWidget
from .segmentation_panel import SegmentationPanelWidget
from iris_common.widgets.neighborhood import IrisCellularNeighborhoodsWidget

__all__ = [
    "ConsensusPanelWidget",
    "FeaturePanelWidget",
    "SegmentationPanelWidget",
    "IrisCellularNeighborhoodsWidget",
]
