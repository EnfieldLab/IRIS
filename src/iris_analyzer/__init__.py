"""
IRIS Analyzer
=============
Downstream spatial statistics, microenvironment analysis, and cohort-level
biomarker analytics package for the IRIS suite.

Exposes the headless computational engines eagerly (zero Qt/Napari imports)
and loads Qt dock widgets / Publication Studio dialogs lazily via PEP 562
``__getattr__``.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .density import (
    DensityResult,
    ROISpec,
    build_shapely_region,
    compute_cell_density,
    compute_polygon_area_and_filter,
    generate_concave_hull_polygon,
    generate_intensity_tissue_polygons,
)
from .graph import (
    SpatialGraphResult,
    build_spatial_graph,
    extract_pair_edges_yx,
    normalize_graph_mode,
)
from .interactions import (
    InteractionResult,
    run_pairwise_interactions,
)
from .neighborhoods import (
    MovingWindowCNResult,
    SpatialLDAResult,
    compute_cn_cohort_stats,
    run_moving_window_cn,
    run_spatial_lda,
)
from .proportion import (
    ProportionResult,
    compute_cell_proportions,
    compute_sample_diversity,
)

if TYPE_CHECKING:
    from .widgets import (
        CellDensityStudioDialog,
        CellDensityWidget,
        CellProportionStudioDialog,
        CellProportionWidget,
        CellularNeighborhoodStudioDialog,
        CellularNeighborhoodsWidget,
        IrisAnalyzerPanelWidget,
        PairwiseInteractionStudioDialog,
        PairwiseInteractionsWidget,
    )

__version__ = "0.2.0"

_LAZY_WIDGET_EXPORTS = {
    "IrisAnalyzerPanelWidget": "iris_analyzer.widgets.analyzer_panel",
    "CellDensityWidget": "iris_analyzer.widgets.density_widget",
    "CellDensityStudioDialog": "iris_analyzer.widgets.density_widget",
    "CellProportionWidget": "iris_analyzer.widgets.proportion_widget",
    "CellProportionStudioDialog": "iris_analyzer.widgets.proportion_widget",
    "PairwiseInteractionsWidget": "iris_analyzer.widgets.interactions_widget",
    "PairwiseInteractionStudioDialog": "iris_analyzer.widgets.interactions_widget",
    "CellularNeighborhoodsWidget": "iris_analyzer.widgets.neighborhoods_widget",
    "CellularNeighborhoodStudioDialog": "iris_analyzer.widgets.neighborhoods_widget",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_WIDGET_EXPORTS:
        module = importlib.import_module(_LAZY_WIDGET_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "__version__",
    # Density & ROI geometry
    "ROISpec",
    "DensityResult",
    "build_shapely_region",
    "compute_polygon_area_and_filter",
    "generate_concave_hull_polygon",
    "generate_intensity_tissue_polygons",
    "compute_cell_density",
    # Proportions & diversity
    "ProportionResult",
    "compute_sample_diversity",
    "compute_cell_proportions",
    # Spatial graphs & pairwise interactions
    "SpatialGraphResult",
    "build_spatial_graph",
    "extract_pair_edges_yx",
    "normalize_graph_mode",
    "InteractionResult",
    "run_pairwise_interactions",
    # Cellular neighborhoods & Spatial-LDA
    "SpatialLDAResult",
    "MovingWindowCNResult",
    "run_spatial_lda",
    "run_moving_window_cn",
    "compute_cn_cohort_stats",
    # Lazy Qt widgets & studio dialogs
    "IrisAnalyzerPanelWidget",
    "CellDensityWidget",
    "CellDensityStudioDialog",
    "CellProportionWidget",
    "CellProportionStudioDialog",
    "PairwiseInteractionsWidget",
    "PairwiseInteractionStudioDialog",
    "CellularNeighborhoodsWidget",
    "CellularNeighborhoodStudioDialog",
]
