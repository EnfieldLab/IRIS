"""
IRIS-Viewer Review Service
Handles confidence overlay rendering, flagged-cell navigation, competing-mask
isolation display, interactive mouse-click cell selection, and manual override
with undo support.
"""
from __future__ import annotations

import types

import numpy as np
import matplotlib.colors as mcolors
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from iris_viewer.engine.consensus import apply_manual_override
from iris_common.widgets.styles import (
    CONFIDENCE_HIGH_COLOR,
    CONFIDENCE_LOW_COLOR,
    CONFIDENCE_MEDIUM_COLOR,
)

# Colour palette used for isolating competing method masks side-by-side
_ISOLATION_PALETTE = [
    (1.0, 0.27, 0.27),   # Red
    (0.27, 1.0, 0.27),   # Green
    (0.27, 0.53, 1.0),   # Blue
    (1.0, 0.27, 1.0),    # Magenta
]


class ReviewService:
    """
    Manages the interactive review workflow:
    - Confidence contour overlay (vectorised, no Python pixel loop)
    - Flagged-cell navigation (Next / Prev issue)
    - Competing-mask isolation view
    - Manual cell override with undo stack
    - Mouse-click cell selection
    """

    def __init__(self, ctx: types.SimpleNamespace) -> None:
        self.ctx = ctx

    # ── Confidence Overlay ─────────────────────────────────────────────────

    def update_confidence_overlay(self, threshold: float) -> None:
        """
        Render colour-coded contours on the consensus mask boundaries.

        Green  = IoU >= threshold (agreed)
        Amber  = IoU >= threshold/2 (borderline)
        Red    = IoU <  threshold/2 (disputed)

        Uses vectorised NumPy indexing — no Python loop over pixels.
        """
        ctx = self.ctx
        if ctx.state.consensus_mask is None or not ctx.state.per_cell_confidence:
            return

        h, w = ctx.state.consensus_mask.shape
        contour_rgba = np.zeros((h, w, 4), dtype=np.float32)
        half_thresh = threshold * 0.5

        boundary_mask = find_boundaries(ctx.state.consensus_mask, mode='inner')
        max_id = int(ctx.state.consensus_mask.max())

        if max_id > 0:
            color_lookup = np.zeros((max_id + 1, 4), dtype=np.float32)
            c_high = np.array(mcolors.to_rgba(CONFIDENCE_HIGH_COLOR), dtype=np.float32)
            c_med  = np.array(mcolors.to_rgba(CONFIDENCE_MEDIUM_COLOR), dtype=np.float32)
            c_low  = np.array(mcolors.to_rgba(CONFIDENCE_LOW_COLOR), dtype=np.float32)

            for cell_id, conf in ctx.state.per_cell_confidence.items():
                if 0 < cell_id <= max_id:
                    if conf >= threshold:
                        color_lookup[cell_id] = c_high
                    elif conf >= half_thresh:
                        color_lookup[cell_id] = c_med
                    else:
                        color_lookup[cell_id] = c_low

            # Vectorised: look up boundary pixel IDs, map to colours in one shot
            ids_at_bounds = ctx.state.consensus_mask[boundary_mask]
            contour_rgba[boundary_mask] = color_lookup[ids_at_bounds]

        layer_name = "Confidence Contours"
        if layer_name in ctx.viewer.layers:
            ctx.viewer.layers[layer_name].data = contour_rgba
        else:
            scale_val = ctx.control_widget.spin_scale.value()
            lyr = ctx.viewer.add_image(
                contour_rgba, name=layer_name, rgb=True,
                blending='additive', visible=True,
                scale=(scale_val, scale_val),
            )
            if layer_name not in ctx.added_legend_entries:
                ctx.legend_widget.add_consensus_row(lyr, layer_name, '#FFFFFF')
                ctx.added_legend_entries.add(layer_name)

        self.update_confidence_stats(threshold)

    def update_confidence_stats(self, threshold: float | None = None) -> None:
        """Recompute flagged-cell list and push statistics to the consensus panel."""
        ctx = self.ctx
        if not ctx.state.per_cell_confidence:
            return

        if threshold is None:
            threshold = ctx.consensus_panel.get_threshold()

        confs = ctx.state.per_cell_confidence
        ctx.flagged_cells = [cid for cid, c in confs.items() if c < threshold]
        ctx.current_issue_idx = -1

        total   = len(confs)
        flagged = len(ctx.flagged_cells)
        agreed  = total - flagged
        mean_c  = float(np.mean(list(confs.values()))) if confs else 0.0

        ctx.consensus_panel.update_statistics(total, agreed, flagged, mean_c)
        ctx.control_widget.update_dashboard(confidence=f"{mean_c:.3f} (mean IoU)")

        # Hide lingering review overlay when stats are refreshed
        if "Active Review Isolation" in ctx.viewer.layers:
            ctx.viewer.layers["Active Review Isolation"].visible = False
        if "Review BBox" in ctx.viewer.layers:
            try:
                ctx.viewer.layers["Review BBox"].data = []
            except Exception:
                pass

    # ── Issue Navigation ────────────────────────────────────────────────────

    def next_issue(self) -> None:
        self.jump_to_issue(self.ctx.current_issue_idx + 1)

    def prev_issue(self) -> None:
        n = len(self.ctx.flagged_cells)
        if n == 0:
            return
        idx = (self.ctx.current_issue_idx - 1) % n
        self.jump_to_issue(idx)

    def jump_to_issue(self, idx: int) -> None:
        """Navigate to a flagged cell by index; centres the camera and opens the review pane."""
        ctx = self.ctx
        if not ctx.flagged_cells:
            ctx.consensus_panel.lbl_issue_count.setText("No issues found")
            return

        ctx.current_issue_idx = idx % len(ctx.flagged_cells)
        cell_id = ctx.flagged_cells[ctx.current_issue_idx]
        ctx.consensus_panel.lbl_issue_count.setText(
            f"Issue {ctx.current_issue_idx + 1} of {len(ctx.flagged_cells)}"
        )

        method_map = ctx.state.cell_method_map.get(cell_id, {})

        # Compute a bounding box that covers all competing method masks
        min_r = min_c = float('inf')
        max_r = max_c = 0.0

        for m_name, orig_id in method_map.items():
            if m_name not in ctx.state.segmentation_results:
                continue
            nuc  = ctx.state.segmentation_results[m_name].get('nuclear')
            cyto = ctx.state.segmentation_results[m_name].get('cytoplasm')
            combined = np.maximum(nuc, cyto) if (nuc is not None and cyto is not None) else (nuc or cyto)
            if combined is None:
                continue
            ys, xs = np.where(combined == orig_id)
            if len(ys) == 0:
                continue
            min_r = min(min_r, float(ys.min()))
            min_c = min(min_c, float(xs.min()))
            max_r = max(max_r, float(ys.max()))
            max_c = max(max_c, float(xs.max()))

        # Fallback: use consensus footprint
        if min_r == float('inf'):
            binary = (ctx.state.consensus_mask == cell_id).astype(np.uint8)
            props = regionprops(binary)
            if not props:
                return
            min_r, min_c, max_r, max_c = props[0].bbox

        PAD = 20
        min_r = max(0, int(min_r) - PAD)
        min_c = max(0, int(min_c) - PAD)
        max_r = min(ctx.state.consensus_mask.shape[0], int(max_r) + PAD)
        max_c = min(ctx.state.consensus_mask.shape[1], int(max_c) + PAD)

        ctx.viewer.camera.center = ((min_r + max_r) / 2.0, (min_c + max_c) / 2.0)
        ctx.viewer.camera.zoom = max(4.0, 280.0 / max(max_r - min_r, 1))

        bbox = np.array([
            [min_r, min_c], [min_r, max_c],
            [max_r, max_c], [max_r, min_c],
        ], dtype=float)

        if "Review BBox" in ctx.viewer.layers:
            ctx.viewer.layers["Review BBox"].data = [bbox]
        else:
            scale_val = ctx.control_widget.spin_scale.value()
            ctx.viewer.add_shapes(
                [bbox], shape_type='polygon',
                edge_color='red', face_color='transparent',
                edge_width=3, name="Review BBox",
                scale=(scale_val, scale_val),
            )

        conf = ctx.state.per_cell_confidence.get(cell_id, 0.0)
        ctx.consensus_panel.show_cell_review(cell_id, conf, method_map)
        self.show_competing_masks(method_map)

    # ── Competing Mask Isolation ────────────────────────────────────────────

    def show_competing_masks(self, method_map: dict) -> None:
        """Render each method's cell footprint in a distinct colour for side-by-side comparison."""
        ctx = self.ctx
        h, w = ctx.state.consensus_mask.shape
        iso = np.zeros((h, w, 4), dtype=np.float32)

        for i, m_name in enumerate(sorted(method_map.keys())):
            orig_id = method_map[m_name]
            if m_name not in ctx.state.segmentation_results:
                continue
            nuc  = ctx.state.segmentation_results[m_name].get('nuclear')
            cyto = ctx.state.segmentation_results[m_name].get('cytoplasm')
            combined = np.maximum(nuc, cyto) if (nuc is not None and cyto is not None) else (nuc or cyto)
            if combined is None:
                continue

            r, g, b = _ISOLATION_PALETTE[i % len(_ISOLATION_PALETTE)]
            cell_px = combined == orig_id
            boundary = find_boundaries(cell_px.astype(np.uint8), mode='inner')

            # Fill with translucent colour
            iso[cell_px, 0] = np.clip(iso[cell_px, 0] + r * 0.35, 0, 1)
            iso[cell_px, 1] = np.clip(iso[cell_px, 1] + g * 0.35, 0, 1)
            iso[cell_px, 2] = np.clip(iso[cell_px, 2] + b * 0.35, 0, 1)
            iso[cell_px, 3] = np.maximum(iso[cell_px, 3], 0.25)

            # Bright outline
            iso[boundary, :3] = np.array([r, g, b])
            iso[boundary, 3] = 0.9

        layer_name = "Active Review Isolation"
        if layer_name in ctx.viewer.layers:
            ctx.viewer.layers[layer_name].data = iso
            ctx.viewer.layers[layer_name].visible = True
        else:
            scale_val = ctx.control_widget.spin_scale.value()
            lyr = ctx.viewer.add_image(
                iso, name=layer_name, rgb=True,
                opacity=1.0, blending='translucent', visible=True,
                scale=(scale_val, scale_val),
            )
            if layer_name not in ctx.added_legend_entries:
                ctx.legend_widget.add_consensus_row(lyr, layer_name, '#FFFFFF')
                ctx.added_legend_entries.add(layer_name)

    # ── Manual Override ─────────────────────────────────────────────────────

    def apply_manual_override(self, cell_id: int, preferred_method: str) -> None:
        """
        Apply a user-chosen override for a cell.  Snapshots state first so the
        action can be undone.
        """
        ctx = self.ctx
        if ctx.state.consensus_mask is None:
            return

        # Snapshot BEFORE mutation — enables undo
        ctx.state.snapshot_override_state()
        ctx.state.manual_overrides[cell_id] = preferred_method

        # Build the per-method combined (nuclear + cytoplasm) mask lookup
        masks = {}
        for m_name, seg in ctx.state.segmentation_results.items():
            nuc  = seg.get('nuclear')
            cyto = seg.get('cytoplasm')
            if nuc is not None:
                masks[m_name] = np.maximum(nuc, cyto) if cyto is not None else nuc

        (ctx.state.consensus_mask,
         ctx.state.confidence_map,
         ctx.state.per_cell_confidence) = apply_manual_override(
            ctx.state.consensus_mask,
            ctx.state.confidence_map,
            ctx.state.per_cell_confidence,
            ctx.state.cell_method_map,
            cell_id,
            preferred_method,
            masks,
        )

        if "Consensus Mask" in ctx.viewer.layers:
            ctx.viewer.layers["Consensus Mask"].data = ctx.state.consensus_mask

        self.update_confidence_overlay(ctx.consensus_panel.get_threshold())
        ctx.consensus_panel.clear_cell_review()

        if "Active Review Isolation" in ctx.viewer.layers:
            ctx.viewer.layers["Active Review Isolation"].visible = False

    def undo_override(self) -> None:
        """Revert the most recent manual override."""
        ctx = self.ctx
        if ctx.state.undo_override():
            if "Consensus Mask" in ctx.viewer.layers:
                ctx.viewer.layers["Consensus Mask"].data = ctx.state.consensus_mask
            self.update_confidence_overlay(ctx.consensus_panel.get_threshold())
            ctx.consensus_panel.clear_cell_review()

    # ── Mouse-click Cell Selection ──────────────────────────────────────────

    def on_mouse_click(self, viewer, event) -> None:
        """napari mouse_drag_callback: select a cell in the consensus mask by clicking."""
        ctx = self.ctx
        if ctx.state.consensus_mask is None:
            return

        consensus_layer = next(
            (l for l in ctx.viewer.layers if l.name == "Consensus Mask"), None
        )
        if consensus_layer is None:
            return

        data_coord = consensus_layer.world_to_data(event.position)
        if len(data_coord) < 2:
            return

        y, x = int(data_coord[-2]), int(data_coord[-1])
        mask = ctx.state.consensus_mask

        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            cell_id = int(mask[y, x])
            if cell_id > 0 and ctx.state.cell_method_map:
                conf = ctx.state.per_cell_confidence.get(cell_id, 0.0)
                method_map = ctx.state.cell_method_map.get(cell_id, {})
                ctx.consensus_panel.show_cell_review(cell_id, conf, method_map)
                self.show_competing_masks(method_map)
