"""iris_common.widgets — shared UI widgets for IRIS suite."""
from .neighborhood import IrisCellularNeighborhoodsWidget
from .image_control import ImageControlWidget
from .legend import ViewerLegendWidget
from .multi_sample import MultiSampleComparisonWidget

__all__ = [
    "IrisCellularNeighborhoodsWidget",
    "ImageControlWidget",
    "ViewerLegendWidget",
    "MultiSampleComparisonWidget",
]
