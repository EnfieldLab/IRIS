"""
Canvas drag handler for grouped sample layer translation.
Reused from IRIS core with no modifications.
"""
import numpy as np


def handle_global_canvas_drag(viewer, event, sample_layers_map):
    active_layer = viewer.layers.selection.active
    if active_layer is None or not active_layer.name.startswith("Bounds - "):
        return

    sample_name = active_layer.name.replace("Bounds - ", "")
    layers_to_move = sample_layers_map.get(sample_name, [])
    if not layers_to_move:
        return

    old_pan = viewer.camera.mouse_pan
    viewer.camera.mouse_pan = False

    start_pos = np.array(event.position)
    start_translates = {l: np.array(l.translate) for l in layers_to_move if l in viewer.layers}

    yield

    while event.type == 'mouse_move':
        current_pos = np.array(event.position)
        delta = current_pos - start_pos
        for l in layers_to_move:
            if l in viewer.layers:
                l.translate = tuple(start_translates[l] + delta)
        yield

    viewer.camera.mouse_pan = old_pan
