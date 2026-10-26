"""
Unit tests for iris_viewer.core.context module:
- AppContext: Typed context dataclass initialization, typed defaults, and dynamic attributes
- AppContext._active_workers: Backward-compatibility alias for active_workers list
"""
from __future__ import annotations

import types

from iris_viewer.core.context import AppContext
from iris_viewer.core.state import ViewerAppState


def test_app_context_initialization_defaults_and_dynamic_attributes():
    """Verify AppContext initializes with typed fields, proper defaults, and supports dynamic attributes."""
    state = ViewerAppState()
    ctx = AppContext(state=state)

    # Initial defaults
    assert ctx.state is state
    assert ctx.active_workers == []
    assert ctx.flagged_cells == []
    assert ctx.current_issue_idx == -1
    assert ctx.added_legend_entries == set()
    assert isinstance(ctx, types.SimpleNamespace)

    # Dynamic attribute attachment
    ctx.custom_field = "dynamic_property"
    assert hasattr(ctx, "custom_field")
    assert ctx.custom_field == "dynamic_property"


def test_app_context_active_workers_backward_compatibility_alias():
    """Verify AppContext._active_workers getter and setter alias keeps active_workers synchronized."""
    ctx = AppContext()

    # Append via alias
    dummy_worker_1 = object()
    ctx._active_workers.append(dummy_worker_1)
    assert dummy_worker_1 in ctx.active_workers
    assert ctx._active_workers == ctx.active_workers

    # Reassignment via alias
    new_worker_list = [object(), object()]
    ctx._active_workers = new_worker_list
    assert ctx.active_workers is new_worker_list
    assert len(ctx.active_workers) == 2
