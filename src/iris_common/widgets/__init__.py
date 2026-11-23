"""
IRIS Common Shared Widgets & Theme Utilities
============================================
Headless-safe exports for shared themes and lazy access to Qt widgets
(``SampleScopeBar``, ``FigureStudioDialog``, ``FigureStyleSidebar``).
Never imports ``qtpy`` or ``napari`` at module import time.
"""
from __future__ import annotations

from typing import Any

from .theme import (
    BUTTON_CTA,
    BUTTON_DANGER,
    BUTTON_PRIMARY,
    BUTTON_SECONDARY,
    BUTTON_STUDIO,
    BUTTON_SUCCESS,
    COLOR_CARD_BG,
    COLOR_CARD_BORDER,
    COLOR_CTA,
    COLOR_CTA_BORDER,
    COLOR_CTA_HOVER,
    COLOR_DANGER,
    COLOR_DANGER_BASE,
    COLOR_INPUT_BG,
    COLOR_INPUT_BORDER,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_STUDIO,
    COLOR_STUDIO_BORDER,
    COLOR_STUDIO_HOVER,
    COLOR_SUCCESS,
    COLOR_TEXT_HEADER,
    COLOR_WARNING,
    COLOR_WELL_BG,
    CONFIDENCE_HIGH_COLOR,
    CONFIDENCE_LOW_COLOR,
    CONFIDENCE_MEDIUM_COLOR,
    DEFAULT_CELLTYPE_PALETTE,
    DEFAULT_MARKER_COLORS,
    DIALOG_DARK_THEME,
    POPOUT_BUTTON_STYLE,
    PUBLICATION_RC_PRESETS,
    SECTION_SUBTITLE_STYLE,
    SECTION_TITLE_STYLE,
    SHARED_DARK_THEME,
    STATUS_STYLES,
    configure_independent_window,
    configure_tab_widget,
    create_section_card,
    style_section_header,
)

_PUBLICATION_STUDIO_EXPORTS = {
    "FigureStudioDialog",
    "FigureStyleSidebar",
    "PlotStyleConfig",
    "PublicationPlotStudioDialog",
    "PublicationStyleSidebar",
    "SCIENTIFIC_PALETTES",
    "STUDIO_DARK_STYLESHEET",
    "THEME_PRESETS",
    "apply_plot_style",
    "copy_figure_to_clipboard",
    "draw_significance_brackets",
    "export_publication_figure",
    "export_source_data_excel",
    "get_palette_colors",
    "resolve_category_colors",
}

_SAMPLE_SCOPE_EXPORTS = {
    "SCOPE_ACTIVE",
    "SCOPE_ALL",
    "SCOPE_MODES",
    "SCOPE_SELECTED",
    "SampleScopeBar",
}


def __getattr__(name: str) -> Any:
    if name in _PUBLICATION_STUDIO_EXPORTS:
        from . import publication_studio

        return getattr(publication_studio, name)
    if name in _SAMPLE_SCOPE_EXPORTS:
        from . import sample_scope

        return getattr(sample_scope, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SHARED_DARK_THEME",
    "DIALOG_DARK_THEME",
    "POPOUT_BUTTON_STYLE",
    "PUBLICATION_RC_PRESETS",
    "SECTION_TITLE_STYLE",
    "SECTION_SUBTITLE_STYLE",
    "COLOR_PRIMARY",
    "COLOR_PRIMARY_HOVER",
    "COLOR_STUDIO",
    "COLOR_STUDIO_BORDER",
    "COLOR_STUDIO_HOVER",
    "COLOR_CTA",
    "COLOR_CTA_BORDER",
    "COLOR_CTA_HOVER",
    "COLOR_DANGER",
    "COLOR_DANGER_BASE",
    "COLOR_SUCCESS",
    "COLOR_WARNING",
    "COLOR_MUTED",
    "COLOR_CARD_BG",
    "COLOR_CARD_BORDER",
    "COLOR_WELL_BG",
    "COLOR_INPUT_BG",
    "COLOR_INPUT_BORDER",
    "COLOR_TEXT_HEADER",
    "CONFIDENCE_HIGH_COLOR",
    "CONFIDENCE_MEDIUM_COLOR",
    "CONFIDENCE_LOW_COLOR",
    "STATUS_STYLES",
    "BUTTON_PRIMARY",
    "BUTTON_STUDIO",
    "BUTTON_CTA",
    "BUTTON_SECONDARY",
    "BUTTON_DANGER",
    "BUTTON_SUCCESS",
    "DEFAULT_MARKER_COLORS",
    "DEFAULT_CELLTYPE_PALETTE",
    "configure_independent_window",
    "configure_tab_widget",
    "style_section_header",
    "create_section_card",
    *sorted(_PUBLICATION_STUDIO_EXPORTS),
    *sorted(_SAMPLE_SCOPE_EXPORTS),
]
