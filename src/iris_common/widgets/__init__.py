"""
IRIS Common Shared Widgets & Theme Utilities
============================================
Headless-safe exports for shared themes and lazy access to publication studio
components. Never imports ``napari`` at module import time.
"""
from __future__ import annotations

from typing import Any

from .theme import (
    BUTTON_DANGER,
    BUTTON_PRIMARY,
    BUTTON_SECONDARY,
    BUTTON_SUCCESS,
    COLOR_DANGER,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    CONFIDENCE_HIGH_COLOR,
    CONFIDENCE_LOW_COLOR,
    CONFIDENCE_MEDIUM_COLOR,
    DEFAULT_CELLTYPE_PALETTE,
    DEFAULT_MARKER_COLORS,
    DIALOG_DARK_THEME,
    POPOUT_BUTTON_STYLE,
    PUBLICATION_RC_PRESETS,
    SHARED_DARK_THEME,
    STATUS_STYLES,
    configure_independent_window,
)

_PUBLICATION_STUDIO_EXPORTS = {
    "PlotStyleConfig",
    "PublicationPlotStudioDialog",
    "PublicationStyleSidebar",
    "SCIENTIFIC_PALETTES",
    "THEME_PRESETS",
    "apply_plot_style",
    "copy_figure_to_clipboard",
    "draw_significance_brackets",
    "export_publication_figure",
    "export_source_data_excel",
}


def __getattr__(name: str) -> Any:
    if name in _PUBLICATION_STUDIO_EXPORTS:
        from . import publication_studio

        return getattr(publication_studio, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SHARED_DARK_THEME",
    "DIALOG_DARK_THEME",
    "POPOUT_BUTTON_STYLE",
    "PUBLICATION_RC_PRESETS",
    "COLOR_PRIMARY",
    "COLOR_PRIMARY_HOVER",
    "COLOR_DANGER",
    "COLOR_SUCCESS",
    "COLOR_WARNING",
    "COLOR_MUTED",
    "CONFIDENCE_HIGH_COLOR",
    "CONFIDENCE_MEDIUM_COLOR",
    "CONFIDENCE_LOW_COLOR",
    "STATUS_STYLES",
    "BUTTON_PRIMARY",
    "BUTTON_SECONDARY",
    "BUTTON_DANGER",
    "BUTTON_SUCCESS",
    "DEFAULT_MARKER_COLORS",
    "DEFAULT_CELLTYPE_PALETTE",
    "configure_independent_window",
    *sorted(_PUBLICATION_STUDIO_EXPORTS),
]
