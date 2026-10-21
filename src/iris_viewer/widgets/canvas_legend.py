import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


class CanvasLegendOverlay(QWidget):
    def __init__(self, viewer: napari.Viewer):
        qt_viewer = getattr(viewer.window, '_qt_viewer', None)
        canvas_native = getattr(qt_viewer.canvas, 'native', None) if qt_viewer and hasattr(qt_viewer, 'canvas') else None
        super().__init__(canvas_native)

        self.viewer = viewer
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # Let clicks pass through to canvas
        self.setAttribute(Qt.WA_StyledBackground, True)

        # UI Layout
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(2)
        self.layout.setSizeConstraint(QVBoxLayout.SetFixedSize)
        self.setLayout(self.layout)

        self._opacity = 0.0
        self._font_size = 16
        self.update_style()

        # We need to position it manually on resize events of the viewer window
        if self.parent():
            self.parent().installEventFilter(self)

        # Connect to viewer events
        self.viewer.layers.events.inserted.connect(self._on_layers_changed)
        self.viewer.layers.events.removed.connect(self._on_layers_changed)
        self.viewer.layers.events.reordered.connect(self._on_layers_changed)

        self.hide() # Hidden by default
        self._on_layers_changed(None)

    def eventFilter(self, obj, event):
        if obj == self.parent() and event.type() in (event.Resize, event.Move):
            self._reposition()
        return super().eventFilter(obj, event)

    def _reposition(self):
        if not self.parent(): return
        self.move(20, 20)
        self.raise_()

    def update_style(self):
        # Dynamic style based on opacity
        alpha = int(self._opacity * 255)
        self.setObjectName("CanvasLegendOverlay")
        if alpha == 0:
            self.setStyleSheet("""
                QWidget#CanvasLegendOverlay {
                    background-color: transparent;
                    border: none;
                }
            """)
        else:
            self.setStyleSheet(f"""
                QWidget#CanvasLegendOverlay {{
                    background-color: rgba(30, 30, 40, {alpha});
                    border-radius: 5px;
                    border: 1px solid rgba(100, 100, 100, {alpha});
                }}
            """)

    def set_font_size(self, size: int):
        self._font_size = size
        self._update_legend()

    def set_opacity(self, opacity: float):
        self._opacity = opacity
        self.update_style()

    def set_visible(self, visible: bool):
        self.setVisible(visible)
        if visible:
            self.raise_()
            self._update_legend()

    def _on_layers_changed(self, event):
        # We need to bind to visibility/colormap changes for new layers
        for layer in self.viewer.layers:
            if not hasattr(layer, '_legend_bound'):
                layer.events.visible.connect(self._update_legend)
                if hasattr(layer, 'events'):
                    if hasattr(layer.events, 'colormap'):
                        layer.events.colormap.connect(self._update_legend)
                    if hasattr(layer.events, 'color'):
                        layer.events.color.connect(self._update_legend)
                    if hasattr(layer.events, 'name'):
                        layer.events.name.connect(self._update_legend)
                layer._legend_bound = True
        self._update_legend()

    def _update_legend(self, *args):
        if not self.isVisible():
            return

        # Clear layout
        def clear_layout(layout):
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout():
                    clear_layout(child.layout())
                    child.layout().deleteLater()
        clear_layout(self.layout)

        title = QLabel("Active Legend")
        title.setStyleSheet("color: white; font-weight: bold; font-size: 16px;")
        self.layout.addWidget(title)

        has_items = False

        # Add visible layers
        added_channels = set()
        import re

        for layer in self.viewer.layers:
            if not layer.visible:
                continue

            # Skip bounding boxes or utility layers if desired, but we can just show them
            if "Bounds" in layer.name:
                continue

            base_name = re.sub(r' \[\d+\]$', '', layer.name)

            color_hex = "#FFFFFF"
            if hasattr(layer, 'colormap'):
                # Extract color from colormap (usually custom colormaps for markers end with the color)
                # Napari colormaps have colors attribute
                try:
                    colors = layer.colormap.colors
                    # Get the most intense color (usually the last one in the map)
                    c = colors[-1]
                    r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
                    color_hex = f"#{r:02x}{g:02x}{b:02x}"
                except Exception:
                    pass
            elif hasattr(layer, 'color'):
                # Labels / Points layer
                try:
                    if isinstance(layer.color, dict) and 1 in layer.color:
                        c = layer.color[1]
                        r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
                        color_hex = f"#{r:02x}{g:02x}{b:02x}"
                except Exception:
                    pass

            if (base_name, color_hex) in added_channels:
                continue
            added_channels.add((base_name, color_hex))

            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)

            color_box = QLabel()
            color_box.setFixedSize(12, 12)
            color_box.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #AAA; border-radius: 2px;")

            name_lbl = QLabel(base_name)
            name_lbl.setStyleSheet(f"color: {color_hex}; font-size: {self._font_size}px; font-weight: bold;")

            row.addWidget(color_box)
            row.addWidget(name_lbl)
            row.addStretch()

            self.layout.addLayout(row)
            has_items = True

        if not has_items:
            empty = QLabel("No active layers")
            empty.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
            self.layout.addWidget(empty)


