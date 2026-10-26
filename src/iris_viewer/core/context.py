"""
iris_viewer.core.context
========================
Typed application context namespace coordinating state, viewer, widgets, and services.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Set

from iris_viewer.core.state import ViewerAppState


@dataclass
class AppContext(SimpleNamespace):
    """Central application context namespace.

    Provides typed access to viewer, state, control panels, widgets, and services,
    while subclassing SimpleNamespace for 100% backward compatibility with dynamic
    attribute assignment and hasattr checks across services and tests.
    """
    viewer: Any = None
    state: Optional[ViewerAppState] = None
    pipeline_service: Any = None
    control_widget: Any = None
    segmentation_panel: Any = None
    consensus_panel: Any = None
    feature_panel: Any = None
    phenotyper_panel: Any = None
    legend_widget: Any = None
    comparison_widget: Any = None
    neighborhood_widget: Any = None
    marker_widget: Any = None
    expression_widget: Any = None

    # Shared collections & tracking
    added_legend_entries: Set[Any] = field(default_factory=set)
    active_workers: List[Any] = field(default_factory=list)
    flagged_cells: List[Any] = field(default_factory=list)
    current_issue_idx: int = -1
    track_worker: Optional[Callable] = None

    # Services
    color_svc: Any = None
    annotation_svc: Any = None
    image_svc: Any = None
    batch_svc: Any = None
    review_svc: Any = None
    multi_sample_svc: Any = None
    curation_svc: Any = None
    boundary_svc: Any = None
    neighborhood_svc: Any = None

    @property
    def _active_workers(self) -> List[Any]:
        """Backward-compatibility alias for active_workers."""
        return self.active_workers

    @_active_workers.setter
    def _active_workers(self, value: List[Any]) -> None:
        self.active_workers = value
