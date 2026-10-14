from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from qtpy.QtCore import Qt, QPoint
import napari

class CanvasScaleBarOverlay(QWidget):
    """A QuPath-style thick scale bar overlay for the Napari canvas."""
    def __init__(self, viewer: napari.Viewer):
        qt_viewer = getattr(viewer.window, '_qt_viewer', None)
        canvas_native = getattr(qt_viewer.canvas, 'native', None) if qt_viewer and hasattr(qt_viewer, 'canvas') else None
        super().__init__(canvas_native)
        
        self.viewer = viewer
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)
        self.layout.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        self.setLayout(self.layout)
        
        # Text label (e.g. "100 µm")
        self.lbl_text = QLabel("100 µm")
        self.lbl_text.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.lbl_text)
        
        # Thick bar
        self.lbl_bar = QLabel()
        self.lbl_bar.setStyleSheet("background-color: white; border: 1px solid black;")
        self.layout.addWidget(self.lbl_bar)
        
        self._scale_val = 1.0
        self._unit = "px"
        self._position = "bottom_left"
        self._font_size = 26
        self._thickness = 10
        self._update_style()
        
        # Hook into viewer zoom changes
        self.viewer.camera.events.zoom.connect(self._update_scale_bar)
        
        # We need to position it manually on resize events of the parent
        if self.parent():
            self.parent().installEventFilter(self)
            
        self.show() # Always shown
        self._update_scale_bar()

    def _update_style(self):
        self.lbl_text.setStyleSheet(f"color: white; font-weight: bold; font-size: {self._font_size}px; background: transparent;")
        self.lbl_bar.setFixedHeight(self._thickness)
        self.adjustSize()
        self._reposition()

    def set_font_size(self, size: int):
        self._font_size = size
        self._update_style()

    def set_thickness(self, thickness: int):
        self._thickness = thickness
        self._update_style()

    def set_position(self, pos: str):
        self._position = pos
        self._reposition()

    def set_physical_scale(self, scale_val: float):
        """Called when the user changes physical scale in the UI."""
        self._scale_val = scale_val
        self._unit = "µm" if scale_val != 1.0 else "px"
        self._update_scale_bar()

    def eventFilter(self, obj, event):
        if obj == self.parent() and event.type() in (event.Resize, event.Move):
            self._reposition()
        return super().eventFilter(obj, event)
        
    def _reposition(self):
        if not self.parent(): return
        
        pw = self.parent().width()
        ph = self.parent().height()
        
        margin = 30
        if self._position == "bottom_right":
            self.move(pw - self.width() - margin, ph - self.height() - margin)
        elif self._position == "bottom_left":
            self.move(margin, ph - self.height() - margin)
        elif self._position == "top_right":
            self.move(pw - self.width() - margin, margin)
        elif self._position == "top_left":
            self.move(margin, margin)
            
        self.raise_()

    def _update_scale_bar(self, *args):
        if not self.isVisible():
            return
            
        zoom = self.viewer.camera.zoom
        if zoom <= 0: return
        
        # Determine target length in pixels (e.g., around 150 pixels on screen)
        target_px = 150
        
        # Calculate physical distance that matches target_px
        # In Napari, camera zoom maps directly from screen pixels to world units.
        physical_target = target_px / zoom
        
        # Find the nearest "nice" number (1, 2, 5, 10, 20, 50, 100...)
        import math
        if physical_target <= 0: return
        
        magnitude = 10 ** math.floor(math.log10(physical_target))
        residual = physical_target / magnitude
        
        if residual < 2:
            nice_val = 1 * magnitude
        elif residual < 5:
            nice_val = 2 * magnitude
        elif residual < 10:
            nice_val = 5 * magnitude
        else:
            nice_val = 10 * magnitude
            
        # Calculate actual pixels for the nice value
        actual_px = nice_val * zoom
        
        self.lbl_bar.setFixedWidth(int(actual_px))
        
        # Format text
        if isinstance(nice_val, int) or nice_val.is_integer() or nice_val >= 1:
            val_str = f"{int(nice_val)}"
        else:
            val_str = f"{nice_val:.2g}"
            
        self.lbl_text.setText(f"{val_str} {self._unit}")
        
        self.adjustSize()
        self._reposition()
