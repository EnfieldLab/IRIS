"""
iris_viewer.widgets.marker_expression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subpackage for single-cell marker expression profiling, providing:
  - subset_dialog: Modal dialog for cohort sample subsetting
  - dialog: Interactive full-resolution MarkerExpressionDialog with dendrograms & active row studio
  - widget: Dockable IrisMarkerExpressionWidget for viewer integration
"""

from iris_viewer.widgets.marker_expression.dialog import MarkerExpressionDialog
from iris_viewer.widgets.marker_expression.subset_dialog import SampleSubsetDialog
from iris_viewer.widgets.marker_expression.widget import IrisMarkerExpressionWidget

__all__ = [
    "SampleSubsetDialog",
    "MarkerExpressionDialog",
    "IrisMarkerExpressionWidget",
]
