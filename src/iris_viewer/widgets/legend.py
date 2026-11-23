"""
IRIS-Viewer Legend Widget
Layer visibility and color management, using the unified IRIS dark theme
and Control-reference section cards.
Extended with Segmentation Masks and Disagreement & QC (including per-method color rows).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import matplotlib.colors as mcolors
import numpy as np
from napari.utils.colormaps import Colormap
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from iris_viewer.core.layers import get_layer_base_name
from iris_viewer.widgets.styles import (
    BUTTON_SECONDARY,
    SECTION_SUBTITLE_STYLE,
    SECTION_TITLE_STYLE,
    SHARED_DARK_THEME,
    create_section_card,
)


class ViewerLegendRow(QWidget):
    """Single row in the legend panel with color swatch and visibility toggle."""

    colorChanged = Signal(str, str)
    visibilityChanged = Signal(str, bool)
    layerClicked = Signal(str)
    layerDoubleClicked = Signal(str)

    def __init__(
        self,
        name: str,
        hex_color: str,
        initial_visible: bool = True,
        is_clickable: bool = True,
    ):
        super().__init__()
        self.name = name
        self.hex_color = "#FFFFFF"
        self._is_qc_method_row = False
        self._bound_layer = None
        self._vis_callback = None
        self._layer_type = "image"
        self._shapes_click_wired = False

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(8)
        self.setLayout(self.layout)
        self.setStyleSheet("QWidget { background-color: transparent; color: #E0E0E0; }")

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(initial_visible)
        self.checkbox.toggled.connect(self._on_toggle)
        self.layout.addWidget(self.checkbox)

        self.swatch = QLabel()
        self.swatch.setFixedSize(18, 18)
        self.update_swatch_display(hex_color)

        if is_clickable:
            self.swatch.mousePressEvent = self._on_swatch_click
            self.swatch.setCursor(Qt.PointingHandCursor)
        self.layout.addWidget(self.swatch)

        self.label = QLabel(name)
        self.label.setStyleSheet(
            "color: #E0E0E0; font-size: 12px; font-weight: bold; background-color: transparent;"
        )
        self.label.mousePressEvent = self._on_label_click
        self.label.mouseDoubleClickEvent = self._on_double_click
        self.layout.addWidget(self.label)
        self.layout.addStretch()

    def _on_label_click(self, event):
        if event.button() == Qt.LeftButton:
            self.layerClicked.emit(self.name)
        QLabel.mousePressEvent(self.label, event)

    def _on_double_click(self, event):
        if event.button() == Qt.LeftButton:
            self.layerDoubleClicked.emit(self.name)
        QLabel.mouseDoubleClickEvent(self.label, event)

    def update_swatch_display(self, color_val):
        """Update the color swatch widget dynamically."""
        is_transparent = False

        if color_val is None:
            is_transparent = True
        elif isinstance(color_val, str):
            if color_val == "transparent":
                is_transparent = True
            else:
                self.hex_color = color_val
        else:
            try:
                self.hex_color = mcolors.to_hex(color_val)
            except (ValueError, TypeError):
                self.hex_color = "#FFFFFF"

        css_color = "transparent" if is_transparent else self.hex_color
        border = "1px dashed #666" if is_transparent else "1px solid #C8D0DC"
        self.swatch.setStyleSheet(
            f"background-color: {css_color}; border: {border}; border-radius: 3px;"
        )

    def _on_toggle(self, checked: bool):
        self.visibilityChanged.emit(self.name, checked)

    def _on_swatch_click(self, event):
        from qtpy.QtGui import QColor

        new_color = QColorDialog.getColor(
            QColor(self.hex_color), self, f"Select Color: {self.name}"
        )
        if new_color.isValid():
            self.update_swatch_display(new_color.name())
            self.colorChanged.emit(self.name, new_color.name())


class ViewerLegendWidget(QWidget):
    """
    Layer legend panel for IRIS-Viewer with groups for:
    - Channels (fluorescence channels)
    - Segmentation Masks (one row per method)
    - Disagreement & QC (disagreement heatmap, contours, and per-method comparison colors)
    - Phenotypes & Cellular Neighborhoods
    """

    layerSelected = Signal(str)

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setStyleSheet(SHARED_DARK_THEME)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(8)
        self.setLayout(self.main_layout)

        info_lbl = QLabel("Layer visibility and color controls.")
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet(SECTION_SUBTITLE_STYLE)
        self.main_layout.addWidget(info_lbl)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(10)
        self.content_layout.setAlignment(Qt.AlignTop)
        self.content_widget.setLayout(self.content_layout)
        self.scroll.setWidget(self.content_widget)
        self.main_layout.addWidget(self.scroll)

        # Initialize Groups
        self.channel_group, self.channel_layout, self.channel_rows = self._create_group("Channels")
        self.seg_group, self.seg_layout, self.seg_rows = self._create_group("Segmentation Masks")
        (
            self.disagreement_group,
            self.disagreement_layout,
            self.disagreement_rows,
        ) = self._create_group("Disagreement & QC")
        self.disagreement_method_rows = []
        self.ref_ct_group, self.ref_ct_layout, self.ref_ct_rows = self._create_group("Phenotypes")
        self.cn_group, self.cn_layout, self.cn_rows = self._create_group(
            "Cellular Neighborhoods"
        )

    def _create_group(self, title_text: str):
        """Create a Control-styled card group with Show/Hide All buttons."""
        group, vbox = create_section_card(margins=(10, 6, 10, 10), spacing=8)

        header = QHBoxLayout()
        header.setSpacing(6)
        label = QLabel(title_text)
        label.setStyleSheet(SECTION_TITLE_STYLE)
        header.addWidget(label)
        header.addStretch()

        btn_show = QPushButton("Show All")
        btn_hide = QPushButton("Hide All")
        btn_show.setStyleSheet(BUTTON_SECONDARY)
        btn_hide.setStyleSheet(BUTTON_SECONDARY)
        btn_show.setFixedHeight(24)
        btn_hide.setFixedHeight(24)
        header.addWidget(btn_show)
        header.addWidget(btn_hide)
        vbox.addLayout(header)

        row_container = QVBoxLayout()
        row_container.setContentsMargins(0, 2, 0, 0)
        row_container.setSpacing(3)
        vbox.addLayout(row_container)

        self.content_layout.addWidget(group)
        group.hide()
        row_list = []

        btn_show.clicked.connect(lambda: self._toggle_group(row_list, True))
        btn_hide.clicked.connect(lambda: self._toggle_group(row_list, False))

        return group, row_container, row_list

    def _toggle_group(self, row_list: list, state: bool):
        for row in row_list:
            row.checkbox.setChecked(state)
        if row_list is self.disagreement_rows:
            for row in self.disagreement_method_rows:
                row.checkbox.setChecked(state)

    def _unbind_layer_visibility(self, row: ViewerLegendRow) -> None:
        old_layer = getattr(row, "_bound_layer", None)
        old_cb = getattr(row, "_vis_callback", None)
        if old_layer is not None and old_cb is not None:
            events = getattr(old_layer, "events", None)
            vis_ev = getattr(events, "visible", None)
            if vis_ev is not None and hasattr(vis_ev, "disconnect"):
                try:
                    vis_ev.disconnect(old_cb)
                except Exception:
                    pass
        row._bound_layer = None
        row._vis_callback = None

    def _bind_layer_visibility(self, row: ViewerLegendRow, layer) -> None:
        if getattr(row, "_bound_layer", None) is layer and getattr(row, "_vis_callback", None):
            return
        self._unbind_layer_visibility(row)
        row._bound_layer = layer
        if layer is not None and hasattr(layer, "events") and hasattr(layer.events, "visible"):
            def on_layer_vis_change(_event=None, target_lyr=layer, target_row=row) -> None:
                if getattr(target_row, "_bound_layer", None) is not target_lyr:
                    return
                if hasattr(target_lyr, "visible"):
                    vis = bool(target_lyr.visible)
                    if target_row.checkbox.isChecked() != vis:
                        target_row.checkbox.blockSignals(True)
                        target_row.checkbox.setChecked(vis)
                        target_row.checkbox.blockSignals(False)

            layer.events.visible.connect(on_layer_vis_change)
            row._vis_callback = on_layer_vis_change

    def _wire_shapes_click(self, row: ViewerLegendRow) -> None:
        if not getattr(row, "_shapes_click_wired", False):
            row.layerClicked.connect(self.layerSelected.emit)
            row._shapes_click_wired = True
            row.label.setCursor(Qt.PointingHandCursor)
            row.label.setToolTip("Click to select and edit ROI on canvas")

    def _add_layer_row(
        self,
        layer,
        name: str,
        initial_color: str,
        target_group: QGroupBox,
        target_layout: QVBoxLayout,
        target_list: list,
        layer_type: str = "image",
    ) -> None:
        """Create and wire an interactive legend row for a napari layer."""
        target_group.show()

        for existing in target_list:
            if existing.name == name:
                existing._layer_type = layer_type
                if initial_color:
                    existing.update_swatch_display(initial_color)
                if layer is not None and hasattr(layer, "visible"):
                    vis = bool(layer.visible)
                    existing.checkbox.blockSignals(True)
                    existing.checkbox.setChecked(vis)
                    existing.checkbox.blockSignals(False)
                self._bind_layer_visibility(existing, layer)
                if layer_type == "shapes":
                    self._wire_shapes_click(existing)
                return

        initial_vis = (
            bool(layer.visible) if (layer is not None and hasattr(layer, "visible")) else True
        )
        row = ViewerLegendRow(name, initial_color, initial_visible=initial_vis)
        row._layer_type = layer_type
        row.layerDoubleClicked.connect(self.layerSelected.emit)
        if layer_type == "shapes":
            self._wire_shapes_click(row)

        def on_ui_vis_change(n: str, is_vis: bool) -> None:
            if not self.viewer:
                return
            for lyr in self.viewer.layers:
                l_base = get_layer_base_name(lyr)
                if l_base == n:
                    lyr.visible = is_vis

        def on_ui_col_change(n: str, hex_c: str) -> None:
            if not self.viewer:
                return
            cur_type = getattr(row, "_layer_type", layer_type)
            for target_layer in self.viewer.layers:
                l_base = get_layer_base_name(target_layer)
                if l_base == n:
                    if cur_type == "image":
                        colors = np.array([[0, 0, 0, 0], list(mcolors.to_rgb(hex_c)) + [1]])
                        target_layer.colormap = Colormap(
                            colors, name=f"custom_{target_layer.name}"
                        )
                    elif cur_type == "labels":
                        target_layer.refresh()
                    elif cur_type == "shapes":
                        target_layer.edge_color = hex_c
                        target_layer.face_color = list(mcolors.to_rgb(hex_c)) + [0.08]

            ctx = getattr(self, "ctx", None)
            if ctx:
                if cur_type == "image":
                    if hasattr(ctx, "state"):
                        if hasattr(ctx.state, "channel_colors"):
                            ctx.state.channel_colors[n] = hex_c
                            ctx.state.channel_colors[n.lower()] = hex_c
                            ctx.state.channel_colors[n.upper()] = hex_c
                        if hasattr(ctx.state, "persistent_settings"):
                            colors = np.array([[0, 0, 0, 0], list(mcolors.to_rgb(hex_c)) + [1]])
                            cmap = Colormap(colors, name=f"custom_{n}")
                            ctx.state.persistent_settings.setdefault(n, {})["colormap"] = cmap
                            ctx.state.persistent_settings.setdefault(n.lower(), {})[
                                "colormap"
                            ] = cmap
                            ctx.state.persistent_settings.setdefault(n.upper(), {})[
                                "colormap"
                            ] = cmap
                    if hasattr(ctx, "color_svc"):
                        ctx.color_svc.default_colors[n] = hex_c
                        ctx.color_svc.default_colors[n.lower()] = hex_c
                        ctx.color_svc.default_colors[n.upper()] = hex_c
                    if hasattr(ctx, "multi_sample_svc"):
                        ctx.multi_sample_svc.DEFAULT_COLORS[n] = hex_c
                        ctx.multi_sample_svc.DEFAULT_COLORS[n.lower()] = hex_c
                        ctx.multi_sample_svc.DEFAULT_COLORS[n.upper()] = hex_c
                elif cur_type != "shapes":
                    if hasattr(ctx, "state") and hasattr(ctx.state, "celltype_colors"):
                        ctx.state.celltype_colors[n] = hex_c
                        ctx.state.celltype_colors[n.lower()] = hex_c
                    if hasattr(ctx, "color_svc"):
                        ctx.color_svc.celltype_colors[n.lower()] = hex_c
                    if hasattr(ctx, "multi_sample_svc"):
                        ctx.multi_sample_svc.CELLTYPE_COLORS[n.lower()] = hex_c

        row.visibilityChanged.connect(on_ui_vis_change)
        row.colorChanged.connect(on_ui_col_change)
        self._bind_layer_visibility(row, layer)
        target_layout.addWidget(row)
        target_list.append(row)

    # Public API
    def add_channel_row(self, layer, name: str, initial_color: str) -> None:
        """Add an imaging channel entry to the channels section."""
        self._add_layer_row(
            layer, name, initial_color, self.channel_group, self.channel_layout, self.channel_rows
        )

    def add_segmentation_row(
        self,
        layer,
        name: str,
        initial_color: str,
        layer_type: str = "labels",
    ) -> None:
        """Add a single-cell segmentation mask or ROI entry to the segmentation section."""
        self._add_layer_row(
            layer,
            name,
            initial_color,
            self.seg_group,
            self.seg_layout,
            self.seg_rows,
            layer_type=layer_type,
        )

    def add_disagreement_row(
        self, layer, name: str, initial_color: str, layer_type: str = "image"
    ) -> None:
        """Add a disagreement heatmap, contour, or isolation entry to Disagreement & QC."""
        self._add_layer_row(
            layer,
            name,
            initial_color,
            self.disagreement_group,
            self.disagreement_layout,
            self.disagreement_rows,
            layer_type=layer_type,
        )

    def sync_disagreement_method_rows(
        self,
        layer: Optional[object],
        method_colors: Dict[str, str],
        *,
        checked_methods: Optional[Sequence[str]] = None,
        on_color_changed: Optional[Callable[[str, str], None]] = None,
        on_visibility_changed: Optional[Callable[[str, bool], None]] = None,
    ) -> None:
        """Synchronize per-method QC color rows in the Disagreement & QC group."""
        checked_set = (
            set(checked_methods) if checked_methods is not None else set(method_colors.keys())
        )

        # Remove stale QC method rows not in method_colors
        existing_qc: Dict[str, ViewerLegendRow] = {}
        for row in list(self.disagreement_method_rows):
            if row.name not in method_colors:
                self.disagreement_method_rows.remove(row)
                self.disagreement_layout.removeWidget(row)
                row.deleteLater()
            else:
                existing_qc[row.name] = row

        if not method_colors:
            if not self.disagreement_rows and not self.disagreement_method_rows:
                self.disagreement_group.hide()
            return

        self.disagreement_group.show()
        for method_name, hex_color in method_colors.items():
            is_checked = method_name in checked_set
            if method_name in existing_qc:
                row = existing_qc[method_name]
                row.update_swatch_display(hex_color)
                row.checkbox.blockSignals(True)
                row.checkbox.setChecked(is_checked)
                row.checkbox.blockSignals(False)
                continue

            row = ViewerLegendRow(
                method_name,
                hex_color,
                initial_visible=is_checked,
                is_clickable=True,
            )
            row._is_qc_method_row = True

            if on_color_changed is not None:
                row.colorChanged.connect(on_color_changed)
            if on_visibility_changed is not None:
                row.visibilityChanged.connect(on_visibility_changed)
            elif layer is not None and hasattr(layer, "visible"):
                row.visibilityChanged.connect(
                    lambda _n, vis, lyr=layer: setattr(lyr, "visible", bool(vis))
                )

            self.disagreement_layout.addWidget(row)
            self.disagreement_method_rows.append(row)

    def add_cn_row(self, layer, name: str, initial_color: str, layer_type: str = "image") -> None:
        """Add a cellular neighborhood cluster entry to the neighborhood section."""
        self._add_layer_row(
            layer,
            name,
            initial_color,
            self.cn_group,
            self.cn_layout,
            self.cn_rows,
            layer_type=layer_type,
        )

    # Aliases for Services
    def add_grouped_channel_row(
        self, layer, name: str, initial_color: str, group_name: str = "Channels"
    ):
        self.add_channel_row(layer, name, initial_color)

    def add_ref_row(self, layer, name: str, initial_color: str, layer_type: str = "labels"):
        self._add_layer_row(
            layer,
            name,
            initial_color,
            self.seg_group,
            self.seg_layout,
            self.seg_rows,
            layer_type=layer_type,
        )

    def add_ref_ct_row(self, layer, name: str, initial_color: str):
        if layer is None and self.viewer:
            for lyr in self.viewer.layers:
                l_base = get_layer_base_name(lyr)
                if l_base == name or lyr.name == name:
                    layer = lyr
                    break
        self._add_layer_row(
            layer,
            name,
            initial_color,
            self.ref_ct_group,
            self.ref_ct_layout,
            self.ref_ct_rows,
            layer_type="image",
        )

    def remove_row(self, layer_name: str):
        """Remove a specific row by its layer name."""
        for group, layout, row_list in [
            (self.channel_group, self.channel_layout, self.channel_rows),
            (self.seg_group, self.seg_layout, self.seg_rows),
            (self.disagreement_group, self.disagreement_layout, self.disagreement_rows),
            (self.ref_ct_group, self.ref_ct_layout, self.ref_ct_rows),
            (self.cn_group, self.cn_layout, self.cn_rows),
        ]:
            for row in list(row_list):
                if row.name == layer_name:
                    self._unbind_layer_visibility(row)
                    row_list.remove(row)
                    layout.removeWidget(row)
                    row.deleteLater()
            if group is self.disagreement_group:
                for row in list(self.disagreement_method_rows):
                    if row.name == layer_name:
                        self._unbind_layer_visibility(row)
                        self.disagreement_method_rows.remove(row)
                        layout.removeWidget(row)
                        row.deleteLater()
                if not self.disagreement_rows and not self.disagreement_method_rows:
                    group.hide()
            elif not row_list:
                group.hide()

        ctx = getattr(self, "ctx", None)
        if ctx:
            if hasattr(ctx, "added_legend_entries") and isinstance(ctx.added_legend_entries, set):
                ctx.added_legend_entries.discard(layer_name)
            if (
                hasattr(ctx, "multi_sample_svc")
                and hasattr(ctx.multi_sample_svc, "_added_master_legends")
            ):
                ctx.multi_sample_svc._added_master_legends.discard(layer_name)

    def clear(self):
        """Remove all rows from all groups."""
        for group, layout, row_list in [
            (self.channel_group, self.channel_layout, self.channel_rows),
            (self.seg_group, self.seg_layout, self.seg_rows),
            (self.disagreement_group, self.disagreement_layout, self.disagreement_rows),
            (self.ref_ct_group, self.ref_ct_layout, self.ref_ct_rows),
            (self.cn_group, self.cn_layout, self.cn_rows),
        ]:
            for row in row_list:
                self._unbind_layer_visibility(row)
            row_list.clear()
            if group is self.disagreement_group:
                for row in self.disagreement_method_rows:
                    self._unbind_layer_visibility(row)
                self.disagreement_method_rows.clear()
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            group.hide()

        ctx = getattr(self, "ctx", None)
        if ctx:
            if hasattr(ctx, "added_legend_entries") and isinstance(ctx.added_legend_entries, set):
                ctx.added_legend_entries.clear()
            if (
                hasattr(ctx, "multi_sample_svc")
                and hasattr(ctx.multi_sample_svc, "_added_master_legends")
            ):
                ctx.multi_sample_svc._added_master_legends.clear()
