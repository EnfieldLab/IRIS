"""iris_common.widgets — shared UI widgets for IRIS suite."""
from .neighborhood import IrisCellularNeighborhoodsWidget
from .marker_expression import IrisMarkerExpressionWidget, MarkerExpressionDialog
from .image_control import ImageControlWidget
from .legend import ViewerLegendWidget
from .multi_sample import MultiSampleComparisonWidget

__all__ = [
    "IrisCellularNeighborhoodsWidget",
    "IrisMarkerExpressionWidget",
    "MarkerExpressionDialog",
    "ImageControlWidget",
    "ViewerLegendWidget",
    "MultiSampleComparisonWidget",
]

