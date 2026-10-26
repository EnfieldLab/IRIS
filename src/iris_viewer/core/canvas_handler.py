"""
Canvas drag handler for grouped sample layer translation.
Reused from IRIS core with no modifications.
"""
from __future__ import annotations

import numpy as np

from iris_viewer.core.layers import get_layer_sample_name, get_layer_type


def handle_global_canvas_drag(viewer, event, sample_layers_map):
    active_layer = viewer.layers.selection.active
    if active_layer is None:
        return

    l_type = get_layer_type(active_layer)
    if l_type == "bounds" or active_layer.name.startswith("Bounds - "):
        sample_name = get_layer_sample_name(active_layer) or active_layer.name.replace("Bounds - ", "")
    else:
        return

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
