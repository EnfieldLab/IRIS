"""
iris_viewer.widgets.phenotyping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modular subpackage for single-cell multiplex spatial phenotyping, containing:
  - thresholding: Mathematical transforms & automated Otsu/valley thresholding
  - hierarchy: Hierarchical gating tree node representation
  - sample_impact_dialog: Cohort-level sample impact and cross-core distribution analysis
  - gating_studio_dialog: Interactive multi-dimensional gating studio dialog
  - manual_gating: Dock subpanel widget for manual gating
  - panel: Main PhenotyperPanelWidget coordinating gating, clustering, ASTIR, and MAPS
"""

from iris_viewer.widgets.phenotyping.gating_studio_dialog import GatingStudioDialog
from iris_viewer.widgets.phenotyping.hierarchy import GatingHierarchyNode
from iris_viewer.widgets.phenotyping.manual_gating import ManualGatingWidget
from iris_viewer.widgets.phenotyping.panel import PhenotyperPanelWidget
from iris_viewer.widgets.phenotyping.sample_impact_dialog import SampleImpactDialog
from iris_viewer.widgets.phenotyping.thresholding import (
    arcsinh_transform,
    assess_gating_quality,
    compute_optimal_threshold,
    find_sample_column,
    log10_transform,
)

__all__ = [
    "arcsinh_transform",
    "log10_transform",
    "compute_optimal_threshold",
    "assess_gating_quality",
    "find_sample_column",
    "GatingHierarchyNode",
    "SampleImpactDialog",
    "GatingStudioDialog",
    "ManualGatingWidget",
    "PhenotyperPanelWidget",
]
