"""
IRIS-Viewer Legend Widget
Layer visibility and color management, adapted from IRIS/widgets/legend.py.
Extended with segmentation-specific groups: individual method masks and consensus.
"""
import matplotlib.colors as mcolors
import numpy as np
from napari.utils.colormaps import Colormap
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class ViewerLegendRow(QWidget):
    """Single row in the legend panel representing a named layer with color swatch and visibility toggle."""

    colorChanged = Signal(str, str)
    visibilityChanged = Signal(str, bool)
    layerDoubleClicked = Signal(str)

    def __init__(self, name: str, hex_color: str, initial_visible: bool = True, is_clickable: bool = True):
        super().__init__()
        self.name = name
        self.hex_color = "#FFFFFF"

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(8)
        self.setLayout(self.layout)
        self.setStyleSheet("QWidget { background-color: #1f232b; color: white; border-radius: 0px; padding: 0px; }")

        # Visibility checkbox
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(initial_visible)
        self.checkbox.toggled.connect(self._on_toggle)
        self.layout.addWidget(self.checkbox)

        # Color swatch
        self.swatch = QLabel()
        self.swatch.setFixedSize(20, 20)
        self.update_swatch_display(hex_color)

        if is_clickable:
            self.swatch.mousePressEvent = self._on_swatch_click
            self.swatch.setCursor(Qt.PointingHandCursor)
        self.layout.addWidget(self.swatch)

        # Layer name label
        self.label = QLabel(name)
        self.label.setStyleSheet("color: white; font-size: 12px; font-weight: bold;")
        self.label.mouseDoubleClickEvent = self._on_double_click
        self.layout.addWidget(self.label)
        self.layout.addStretch()

    def _on_double_click(self, event):
        if event.button() == Qt.LeftButton:
            self.layerDoubleClicked.emit(self.name)
        super().mouseDoubleClickEvent(event)

    def update_swatch_display(self, color_val):
        """Updates the color swatch widget dynamically."""
        is_transparent = False

        if color_val is None:
            is_transparent = True
        elif isinstance(color_val, str):
            if color_val == 'transparent':
                is_transparent = True
            else:
                self.hex_color = color_val
        else:
            try:
                self.hex_color = mcolors.to_hex(color_val)
            except (ValueError, TypeError):
                self.hex_color = "#FFFFFF"

        css_color = 'transparent' if is_transparent else self.hex_color
        border = "1px dashed #666" if is_transparent else "1px solid #888"
        self.swatch.setStyleSheet(f"background-color: {css_color}; border: {border}; border-radius: 0px;")

    def _on_toggle(self, checked: bool):
        self.visibilityChanged.emit(self.name, checked)

    def _on_swatch_click(self, event):
        from qtpy.QtGui import QColor
        new_color = QColorDialog.getColor(QColor(self.hex_color), self, f"Select Color: {self.name}")
        if new_color.isValid():
            self.update_swatch_display(new_color.name())
            self.colorChanged.emit(self.name, new_color.name())


class ViewerLegendWidget(QWidget):
    """
    Layer legend panel for IRIS-Viewer with groups for:
    - Channels (fluorescence channels)
    - Segmentation Masks (one row per method)
    - Consensus (consensus mask + confidence overlay)
    """

    layerSelected = Signal(str)

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)
        self.setLayout(self.main_layout)
        self.setStyleSheet(
            "QWidget { background-color: #1f232b; color: #E0E0E0; border-radius: 0px; } "
            "QGroupBox { border: 1px solid #444; margin-top: 8px; } "
            "QPushButton { font-size: 12px; font-weight: bold; padding: 2px 6px; background: #333; color: white; border: 1px solid #555; }"
        )

        info_lbl = QLabel("Layer visibility and color controls.")
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color: #AAA; font-size: 12px; font-style: italic;")
        self.main_layout.addWidget(info_lbl)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; background-color: #1f232b; }")
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setAlignment(Qt.AlignTop)
        self.content_widget.setLayout(self.content_layout)
        self.scroll.setWidget(self.content_widget)
        self.main_layout.addWidget(self.scroll)

        # Initialize Groups
        self.channel_group, self.channel_layout, self.channel_rows = self._create_group("Channels")
        self.seg_group, self.seg_layout, self.seg_rows = self._create_group("Segmentation Masks")
        self.consensus_group, self.consensus_layout, self.consensus_rows = self._create_group("Consensus")
        self.ref_ct_group, self.ref_ct_layout, self.ref_ct_rows = self._create_group("Phenotypes")
        self.cn_group, self.cn_layout, self.cn_rows = self._create_group("Cellular Neighborhoods")

    def _create_group(self, title_text: str):
        """Create a collapsible group with Show/Hide All buttons."""
        group = QGroupBox()
        vbox = QVBoxLayout()
        vbox.setSpacing(4)
        vbox.setContentsMargins(5, 5, 5, 5)
        group.setLayout(vbox)

        header = QHBoxLayout()
        label = QLabel(title_text)
        label.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")
        header.addWidget(label)
        header.addStretch()

        btn_show = QPushButton("Show All")
        btn_hide = QPushButton("Hide All")
        header.addWidget(btn_show)
        header.addWidget(btn_hide)
        vbox.addLayout(header)

        row_container = QVBoxLayout()
        row_container.setContentsMargins(0, 2, 0, 0)
        row_container.setSpacing(2)
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

    def _add_layer_row(
        self,
        layer,
        name: str,
        initial_color: str,
        target_group: QGroupBox,
        target_layout: QVBoxLayout,
        target_list: list,
        layer_type: str = 'image'
    ) -> None:
        """Create and wire an interactive legend row for a napari layer.

        Args:
            layer: Associated napari layer, or None if dynamically resolved.
            name: Display name for the channel or phenotype layer.
            initial_color: Hex color string or matplotlib color spec.
            target_group: QGroupBox container holding this layer category.
            target_layout: QVBoxLayout to which the row widget is added.
            target_list: List tracking active row instances in this category.
            layer_type: Type of napari layer ('image' or 'labels').
        """
        target_group.show()

        # Deduplicate: if an entry with this name already exists in target_list, update color if needed and return
        for existing in target_list:
            if existing.name == name:
                if initial_color:
                    existing.update_swatch_display(initial_color)
                return

        initial_vis = layer.visible if (layer is not None and hasattr(layer, 'visible')) else True
        row = ViewerLegendRow(name, initial_color, initial_visible=initial_vis)
        row.layerDoubleClicked.connect(self.layerSelected.emit)

        def on_ui_vis_change(n: str, is_vis: bool) -> None:
            if not self.viewer:
                return
            for l in self.viewer.layers:
                l_base = l.name.split(" [")[0] if " [" in l.name else l.name
                if l_base == n:
                    l.visible = is_vis

        def on_ui_col_change(n: str, hex_c: str) -> None:
            if not self.viewer:
                return
            for target_layer in self.viewer.layers:
                l_base = target_layer.name.split(" [")[0] if " [" in target_layer.name else target_layer.name
                if l_base == n:
                    if layer_type == 'image':
                        colors = np.array([[0, 0, 0, 0], list(mcolors.to_rgb(hex_c)) + [1]])
                        target_layer.colormap = Colormap(colors, name=f'custom_{target_layer.name}')
                    elif layer_type == 'labels':
                        target_layer.refresh()

            ctx = getattr(self, "ctx", None)
            if ctx:
                if layer_type == 'image':
                    if hasattr(ctx, "state"):
                        if hasattr(ctx.state, "channel_colors"):
                            ctx.state.channel_colors[n] = hex_c
                            ctx.state.channel_colors[n.lower()] = hex_c
                            ctx.state.channel_colors[n.upper()] = hex_c
                        if hasattr(ctx.state, "persistent_settings"):
                            colors = np.array([[0, 0, 0, 0], list(mcolors.to_rgb(hex_c)) + [1]])
                            cmap = Colormap(colors, name=f'custom_{n}')
                            ctx.state.persistent_settings.setdefault(n, {})["colormap"] = cmap
                            ctx.state.persistent_settings.setdefault(n.lower(), {})["colormap"] = cmap
                            ctx.state.persistent_settings.setdefault(n.upper(), {})["colormap"] = cmap
                    if hasattr(ctx, "color_svc"):
                        ctx.color_svc.default_colors[n] = hex_c
                        ctx.color_svc.default_colors[n.lower()] = hex_c
                        ctx.color_svc.default_colors[n.upper()] = hex_c
                    if hasattr(ctx, "multi_sample_svc"):
                        ctx.multi_sample_svc.DEFAULT_COLORS[n] = hex_c
                        ctx.multi_sample_svc.DEFAULT_COLORS[n.lower()] = hex_c
                        ctx.multi_sample_svc.DEFAULT_COLORS[n.upper()] = hex_c
                else:
                    if hasattr(ctx, "state") and hasattr(ctx.state, "celltype_colors"):
                        ctx.state.celltype_colors[n] = hex_c
                        ctx.state.celltype_colors[n.lower()] = hex_c
                    if hasattr(ctx, "color_svc"):
                        ctx.color_svc.celltype_colors[n.lower()] = hex_c
                    if hasattr(ctx, "multi_sample_svc"):
                        ctx.multi_sample_svc.CELLTYPE_COLORS[n.lower()] = hex_c

        def on_layer_vis_change(event) -> None:
            if layer is not None and hasattr(layer, 'visible') and row.checkbox.isChecked() != layer.visible:
                row.checkbox.blockSignals(True)
                row.checkbox.setChecked(layer.visible)
                row.checkbox.blockSignals(False)

        row.visibilityChanged.connect(on_ui_vis_change)
        row.colorChanged.connect(on_ui_col_change)
        if layer is not None and hasattr(layer, 'events') and hasattr(layer.events, 'visible'):
            layer.events.visible.connect(on_layer_vis_change)
        target_layout.addWidget(row)
        target_list.append(row)

    # Public API
    def add_channel_row(self, layer, name: str, initial_color: str) -> None:
        """Add an imaging channel entry to the channels section."""
        self._add_layer_row(layer, name, initial_color, self.channel_group, self.channel_layout, self.channel_rows)

    def add_segmentation_row(self, layer, name: str, initial_color: str) -> None:
        """Add a single-cell segmentation mask entry to the segmentation section."""
        self._add_layer_row(layer, name, initial_color, self.seg_group, self.seg_layout, self.seg_rows, layer_type='labels')

    def add_consensus_row(self, layer, name: str, initial_color: str) -> None:
        """Add a consensus ensemble mask entry to the consensus section."""
        self._add_layer_row(layer, name, initial_color, self.consensus_group, self.consensus_layout, self.consensus_rows, layer_type='labels')

    def add_cn_row(self, layer, name: str, initial_color: str, layer_type: str = 'image') -> None:
        """Add a cellular neighborhood cluster entry to the neighborhood section."""
        self._add_layer_row(layer, name, initial_color, self.cn_group, self.cn_layout, self.cn_rows, layer_type=layer_type)

    # Aliases for Services
    def add_grouped_channel_row(self, layer, name: str, initial_color: str, group_name: str="Channels"):
        self.add_channel_row(layer, name, initial_color)

    def add_ref_row(self, layer, name: str, initial_color: str, layer_type: str="labels"):
        self._add_layer_row(layer, name, initial_color, self.seg_group, self.seg_layout, self.seg_rows, layer_type=layer_type)

    def add_ref_ct_row(self, layer, name: str, initial_color: str):
        if layer is None and self.viewer:
            for l in self.viewer.layers:
                l_base = l.name.split(" [")[0] if " [" in l.name else l.name
                if l_base == name or l.name == name:
                    layer = l
                    break
        self._add_layer_row(layer, name, initial_color, self.ref_ct_group, self.ref_ct_layout, self.ref_ct_rows, layer_type='image')

    def remove_row(self, layer_name: str):
        """Remove a specific row by its layer name."""
        for group, layout, row_list in [
            (self.channel_group, self.channel_layout, self.channel_rows),
            (self.seg_group, self.seg_layout, self.seg_rows),
            (self.consensus_group, self.consensus_layout, self.consensus_rows),
            (self.ref_ct_group, self.ref_ct_layout, self.ref_ct_rows),
            (self.cn_group, self.cn_layout, self.cn_rows),
        ]:
            for row in list(row_list):
                if row.name == layer_name:
                    row_list.remove(row)
                    layout.removeWidget(row)
                    row.deleteLater()
            if not row_list:
                group.hide()

        ctx = getattr(self, "ctx", None)
        if ctx:
            if hasattr(ctx, "added_legend_entries") and isinstance(ctx.added_legend_entries, set):
                ctx.added_legend_entries.discard(layer_name)
            if hasattr(ctx, "multi_sample_svc") and hasattr(ctx.multi_sample_svc, "_added_master_legends"):
                ctx.multi_sample_svc._added_master_legends.discard(layer_name)

    def clear(self):
        """Remove all rows from all groups."""
        for group, layout, row_list in [
            (self.channel_group, self.channel_layout, self.channel_rows),
            (self.seg_group, self.seg_layout, self.seg_rows),
            (self.consensus_group, self.consensus_layout, self.consensus_rows),
            (self.ref_ct_group, self.ref_ct_layout, self.ref_ct_rows),
            (self.cn_group, self.cn_layout, self.cn_rows),
        ]:
            row_list.clear()
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            group.hide()

        ctx = getattr(self, "ctx", None)
        if ctx:
            if hasattr(ctx, "added_legend_entries") and isinstance(ctx.added_legend_entries, set):
                ctx.added_legend_entries.clear()
            if hasattr(ctx, "multi_sample_svc") and hasattr(ctx.multi_sample_svc, "_added_master_legends"):
                ctx.multi_sample_svc._added_master_legends.clear()
