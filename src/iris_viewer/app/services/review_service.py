"""
IRIS-Viewer Review Service
Handles multi-model disagreement overlay rendering ("Disagreement Heatmap" and
"Disagreement Contours"), hotspot navigation with "Review BBox", multi-model
isolation ("Active Disagreement Isolation"), category filtering, click-to-inspect,
and two-model visibility flickering.
"""
from __future__ import annotations

import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skimage.segmentation import find_boundaries

from iris_common.logging import iris_log
from iris_viewer.engine.disagreement import (
    METHOD_QC_COLORS,
    DisagreementItem,
    DisagreementReport,
    _hex_to_rgb_float,
    compute_model_disagreement,
    render_disagreement_contours_rgba,
    render_method_comparison_rgba,
)


def _make_disagreement_colormap():
    """Build a transparent -> amber -> red colormap for Disagreement Heatmap."""
    from napari.utils.colormaps import Colormap

    return Colormap(
        colors=np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.8, 0.0, 0.65],
            [1.0, 0.23, 0.19, 0.95],
        ]),
        controls=np.array([0.0, 0.35, 1.0]),
        name="disagreement_heat",
    )


class ReviewService:
    """Manages multi-model disagreement overlays, hotspot navigation, and flickering."""

    def __init__(self, ctx: types.SimpleNamespace) -> None:
        self.ctx = ctx
        self._active_flicker_method: Optional[str] = None

    def _get_scale(self) -> float:
        cw = getattr(self.ctx, "control_widget", None)
        if cw is not None and hasattr(cw, "spin_scale"):
            try:
                return float(cw.spin_scale.value())
            except Exception:
                pass
        return 1.0

    def _get_image_shape(self) -> Optional[Tuple[int, int]]:
        state = getattr(self.ctx, "state", None)
        if state is None:
            return None
        if getattr(state, "disagreement_heatmap", None) is not None:
            return state.disagreement_heatmap.shape[:2]
        if getattr(state, "disagreement_report", None) is not None:
            hm = getattr(state.disagreement_report, "disagreement_heatmap", None)
            if hm is not None:
                return hm.shape[:2]
        for seg in getattr(state, "segmentation_results", {}).values():
            for key in ("nuclear", "cytoplasm"):
                arr = seg.get(key)
                if arr is not None:
                    return arr.shape[:2]
        if getattr(state, "current_mask", None) is not None:
            return state.current_mask.shape[:2]
        return None

    def _add_legend_disagreement_row(
        self, layer: Any, name: str, color_hex: str, layer_type: str = "image"
    ) -> None:
        legend = getattr(self.ctx, "legend_widget", None)
        entries = getattr(self.ctx, "added_legend_entries", None)
        if legend is None:
            return
        if entries is not None and name in entries:
            return
        if hasattr(legend, "add_disagreement_row"):
            legend.add_disagreement_row(layer, name, color_hex, layer_type=layer_type)
        if isinstance(entries, set):
            entries.add(name)

    # Disagreement Overlay Rendering

    def update_disagreement_overlay(self, threshold: float) -> Optional[DisagreementReport]:
        """Recompute multi-model disagreement at *threshold* and refresh viewer layers."""
        ctx = self.ctx
        state = getattr(ctx, "state", None)
        if state is None or not getattr(state, "segmentation_results", None):
            return None

        dp = getattr(ctx, "disagreement_panel", None)
        compare_mode = (
            dp.combo_compare.currentText()
            if dp is not None and hasattr(dp, "combo_compare")
            else "Nuclear Masks"
        )
        ref_method = (
            dp.get_reference_method()
            if dp is not None and hasattr(dp, "get_reference_method")
            else None
        ) or None
        selected_methods: Optional[List[str]] = None
        if dp is not None and hasattr(dp, "get_selected_methods") and getattr(dp, "_methods", []):
            selected_methods = dp.get_selected_methods()
            if not selected_methods:
                return None

        seg_results: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        for name, res in state.segmentation_results.items():
            if selected_methods is not None and name not in selected_methods:
                continue
            if compare_mode == "Nuclear Masks" and res.get("nuclear") is not None:
                seg_results[name] = {"nuclear": res["nuclear"], "cytoplasm": None}
            elif compare_mode == "Cytoplasm Masks" and res.get("cytoplasm") is not None:
                seg_results[name] = {"nuclear": None, "cytoplasm": res["cytoplasm"]}
            elif compare_mode == "Both" and (
                res.get("nuclear") is not None or res.get("cytoplasm") is not None
            ):
                seg_results[name] = res

        if not seg_results:
            return None

        if ref_method not in seg_results:
            ref_method = None

        report = compute_model_disagreement(
            seg_results,
            compare_mode=compare_mode,
            reference_method=ref_method,
            iou_threshold=float(threshold),
            selected_methods=selected_methods,
        )
        self.render_disagreement_overlays(report)
        return report

    def _refresh_canvas_legend(self) -> None:
        cw = getattr(self.ctx, "control_widget", None)
        if cw is not None and getattr(cw, "canvas_legend", None) is not None:
            if hasattr(cw.canvas_legend, "refresh"):
                cw.canvas_legend.refresh()
            elif hasattr(cw.canvas_legend, "_update_legend"):
                cw.canvas_legend._update_legend()

    def _sync_method_comparison_overlay(
        self,
        compare_mode: str,
        show_contours: bool,
        by_method: bool,
        fallback_methods: Optional[List[str]] = None,
    ) -> None:
        ctx = self.ctx
        state = getattr(ctx, "state", None)
        viewer = getattr(ctx, "viewer", None)
        if state is None or viewer is None or not getattr(state, "segmentation_results", None):
            return

        dp = getattr(ctx, "disagreement_panel", None)
        selected_methods: Optional[List[str]] = None
        if dp is not None and hasattr(dp, "get_selected_methods") and getattr(dp, "_methods", []):
            selected_methods = dp.get_selected_methods()
        elif fallback_methods:
            selected_methods = list(fallback_methods)

        method_colors = (
            dp.get_method_colors()
            if dp is not None and hasattr(dp, "get_method_colors")
            else None
        )

        try:
            method_rgba, resolved_colors = render_method_comparison_rgba(
                state.segmentation_results,
                compare_mode=compare_mode,
                selected_methods=selected_methods,
                method_colors=method_colors,
                thickness=2,
                fill_alpha=0.18,
            )
        except ValueError:
            return

        scale_val = self._get_scale()
        mc_name = "Method Comparison Overlay"
        mc_vis = bool(show_contours) and bool(by_method)
        mc_meta = {
            "layer_type": "disagreement_method_overlay",
            "method_colors": dict(resolved_colors),
        }

        if mc_name in viewer.layers:
            mc_lyr = viewer.layers[mc_name]
            mc_lyr.data = method_rgba
            mc_lyr.visible = mc_vis
            mc_lyr.metadata = mc_meta
            mc_lyr.scale = (scale_val, scale_val)
        else:
            mc_lyr = viewer.add_image(
                method_rgba,
                name=mc_name,
                rgb=True,
                blending="translucent",
                visible=mc_vis,
                scale=(scale_val, scale_val),
                metadata=mc_meta,
            )

        legend = getattr(ctx, "legend_widget", None)
        if legend is not None and hasattr(legend, "sync_disagreement_method_rows"):
            if by_method:
                all_colors = dict(method_colors or {})
                all_colors.update(resolved_colors)
                checked_list = (
                    list(selected_methods)
                    if selected_methods is not None
                    else list(all_colors.keys())
                )

                def _on_legend_color(m_name: str, hex_c: str) -> None:
                    if dp is not None and hasattr(dp, "set_method_color"):
                        dp.set_method_color(m_name, hex_c, emit=True)
                    else:
                        self.render_disagreement_overlays()

                def _on_legend_vis(m_name: str, is_vis: bool) -> None:
                    if dp is not None and hasattr(dp, "set_method_checked"):
                        dp.set_method_checked(m_name, is_vis, emit=True)

                legend.sync_disagreement_method_rows(
                    mc_lyr,
                    all_colors,
                    checked_methods=checked_list,
                    on_color_changed=_on_legend_color,
                    on_visibility_changed=_on_legend_vis,
                )
            else:
                legend.sync_disagreement_method_rows(None, {})

        self._refresh_canvas_legend()

    def render_disagreement_overlays(
        self,
        report: Optional[DisagreementReport] = None,
        show_heatmap: Optional[bool] = None,
        show_contours: Optional[bool] = None,
    ) -> None:
        """Render disagreement heatmap, contours, and per-method comparison overlay in Napari."""
        ctx = self.ctx
        state = getattr(ctx, "state", None)
        if state is None:
            return

        dp = getattr(ctx, "disagreement_panel", None)
        if show_heatmap is None:
            show_heatmap = (
                bool(dp.chk_show_heatmap.isChecked())
                if dp is not None and hasattr(dp, "chk_show_heatmap")
                else True
            )
        if show_contours is None:
            show_contours = (
                bool(dp.chk_show_contours.isChecked())
                if dp is not None and hasattr(dp, "chk_show_contours")
                else True
            )
        overlay_mode = (
            dp.get_overlay_color_mode()
            if dp is not None and hasattr(dp, "get_overlay_color_mode")
            else "Color by Method"
        )
        by_method = overlay_mode == "Color by Method"

        if report is None:
            report = getattr(state, "disagreement_report", None)
        if report is None:
            if getattr(state, "segmentation_results", None):
                compare_mode = (
                    dp.combo_compare.currentText()
                    if dp is not None and hasattr(dp, "combo_compare")
                    else "Nuclear Masks"
                )
                self._sync_method_comparison_overlay(compare_mode, show_contours, by_method)
            return

        state.disagreement_report = report
        state.disagreement_heatmap = report.disagreement_heatmap

        viewer = getattr(ctx, "viewer", None)
        if viewer is not None:
            scale_val = self._get_scale()

            # 1. Disagreement Heatmap
            hm_name = "Disagreement Heatmap"
            if hm_name in viewer.layers:
                viewer.layers[hm_name].data = report.disagreement_heatmap
                viewer.layers[hm_name].visible = bool(show_heatmap)
            else:
                hm_lyr = viewer.add_image(
                    report.disagreement_heatmap,
                    name=hm_name,
                    colormap=_make_disagreement_colormap(),
                    blending="additive",
                    visible=bool(show_heatmap),
                    scale=(scale_val, scale_val),
                    contrast_limits=[0.0, 1.0],
                )
                self._add_legend_disagreement_row(hm_lyr, hm_name, "#FF3B30", layer_type="image")

            # 2. Disagreement Contours RGBA (visible when "Color by Issue Type")
            contour_rgba = render_disagreement_contours_rgba(
                state.segmentation_results,
                report,
                show_agreed=False,
            )
            contour_name = "Disagreement Contours"
            contour_vis = bool(show_contours) and not by_method
            if contour_name in viewer.layers:
                viewer.layers[contour_name].data = contour_rgba
                viewer.layers[contour_name].visible = contour_vis
            else:
                contour_lyr = viewer.add_image(
                    contour_rgba,
                    name=contour_name,
                    rgb=True,
                    blending="translucent",
                    visible=contour_vis,
                    scale=(scale_val, scale_val),
                )
                self._add_legend_disagreement_row(
                    contour_lyr, contour_name, "#FFCC00", layer_type="image"
                )

            # 3. Per-Method Comparison Overlay RGBA (visible when "Color by Method")
            self._sync_method_comparison_overlay(
                report.compare_mode,
                show_contours,
                by_method,
                fallback_methods=list(report.methods),
            )

        self.update_disagreement_stats(report)

    def toggle_overlay_visibility(self, show_heatmap: bool, show_contours: bool) -> None:
        """Toggle visibility of disagreement heatmap, contour, and method comparison layers."""
        viewer = getattr(self.ctx, "viewer", None)
        if viewer is None:
            return
        dp = getattr(self.ctx, "disagreement_panel", None)
        overlay_mode = (
            dp.get_overlay_color_mode()
            if dp is not None and hasattr(dp, "get_overlay_color_mode")
            else "Color by Method"
        )
        by_method = overlay_mode == "Color by Method"
        if "Disagreement Heatmap" in viewer.layers:
            viewer.layers["Disagreement Heatmap"].visible = bool(show_heatmap)
        if "Disagreement Contours" in viewer.layers:
            viewer.layers["Disagreement Contours"].visible = bool(show_contours) and not by_method
        if "Method Comparison Overlay" in viewer.layers:
            viewer.layers["Method Comparison Overlay"].visible = bool(show_contours) and by_method
        self._refresh_canvas_legend()

    def update_disagreement_stats(
        self,
        report: Optional[DisagreementReport] = None,
        category_filter: Optional[str] = None,
    ) -> None:
        """Filter disputed regions by category and push KPI stats to UI."""
        ctx = self.ctx
        state = getattr(ctx, "state", None)
        if report is None and state is not None:
            report = getattr(state, "disagreement_report", None)
        if report is None:
            return

        dp = getattr(ctx, "disagreement_panel", None)
        if category_filter is None and dp is not None and hasattr(dp, "get_category_filter"):
            category_filter = dp.get_category_filter()

        if not category_filter or category_filter == "All Disagreements":
            filtered_items = list(report.items)
        else:
            filtered_items = [it for it in report.items if it.category == category_filter]

        ctx.flagged_cells = filtered_items
        ctx.current_issue_idx = -1

        if dp is not None and hasattr(dp, "update_statistics"):
            dp.update_statistics(
                total=report.total_ref_cells,
                agreed=report.agreed_cells,
                flagged=len(filtered_items),
                mean_iou=report.mean_iou,
                category_counts=report.category_counts,
            )

        cw = getattr(ctx, "control_widget", None)
        if cw is not None and hasattr(cw, "update_dashboard"):
            cw.update_dashboard(confidence=f"{report.mean_iou:.3f} (mean IoU)")

        viewer = getattr(ctx, "viewer", None)
        if viewer is not None:
            if "Active Disagreement Isolation" in viewer.layers:
                viewer.layers["Active Disagreement Isolation"].visible = False
            if "Review BBox" in viewer.layers:
                try:
                    viewer.layers["Review BBox"].data = []
                except Exception:
                    pass

    def set_category_filter(self, category: str) -> None:
        """Update active category filter and refresh flagged hotspot list."""
        self.update_disagreement_stats(category_filter=category)

    # Hotspot Navigation

    def next_issue(self) -> None:
        self.jump_to_issue(self.ctx.current_issue_idx + 1)

    def prev_issue(self) -> None:
        flagged = getattr(self.ctx, "flagged_cells", [])
        n = len(flagged)
        if n == 0:
            return
        idx = (self.ctx.current_issue_idx - 1) % n
        self.jump_to_issue(idx)

    def jump_to_region_id(self, region_id: int) -> None:
        """Navigate directly to a hotspot matching *region_id*."""
        ctx = self.ctx
        flagged = getattr(ctx, "flagged_cells", [])
        for idx, item in enumerate(flagged):
            if getattr(item, "region_id", None) == region_id:
                self.jump_to_issue(idx)
                return

        report = getattr(getattr(ctx, "state", None), "disagreement_report", None)
        if report is not None:
            for item in report.items:
                if item.region_id == region_id:
                    self._focus_item(item, move_camera=True)
                    return

    def jump_to_issue(self, idx: int) -> None:
        """Navigate to a flagged hotspot by index; zoom camera and isolate masks."""
        ctx = self.ctx
        dp = getattr(ctx, "disagreement_panel", None)
        flagged = getattr(ctx, "flagged_cells", [])

        if not flagged:
            if dp is not None and hasattr(dp, "lbl_issue_count"):
                dp.lbl_issue_count.setText("No flagged regions")
            return

        ctx.current_issue_idx = idx % len(flagged)
        item: DisagreementItem = flagged[ctx.current_issue_idx]

        if dp is not None and hasattr(dp, "lbl_issue_count"):
            dp.lbl_issue_count.setText(
                f"Hotspot {ctx.current_issue_idx + 1} of {len(flagged)}"
            )

        self._focus_item(item, move_camera=True)

    def _focus_item(self, item: DisagreementItem, move_camera: bool = True) -> None:
        ctx = self.ctx
        dp = getattr(ctx, "disagreement_panel", None)
        shape = self._get_image_shape()
        min_r, min_c, max_r, max_c = item.bbox

        pad = 20
        min_r = max(0, int(min_r) - pad)
        min_c = max(0, int(min_c) - pad)
        if shape is not None:
            max_r = min(shape[0], int(max_r) + pad)
            max_c = min(shape[1], int(max_c) + pad)
        else:
            max_r = int(max_r) + pad
            max_c = int(max_c) + pad

        viewer = getattr(ctx, "viewer", None)
        if viewer is not None:
            scale_val = self._get_scale()
            if move_camera:
                viewer.camera.center = (
                    ((min_r + max_r) / 2.0) * scale_val,
                    ((min_c + max_c) / 2.0) * scale_val,
                )
                viewer.camera.zoom = max(4.0, 280.0 / max(max_r - min_r, 1))

            bbox_poly = np.array(
                [
                    [min_r, min_c],
                    [min_r, max_c],
                    [max_r, max_c],
                    [max_r, min_c],
                ],
                dtype=float,
            )

            if "Review BBox" in viewer.layers:
                viewer.layers["Review BBox"].data = [bbox_poly]
                viewer.layers["Review BBox"].visible = True
            else:
                viewer.add_shapes(
                    [bbox_poly],
                    shape_type="polygon",
                    edge_color="red",
                    face_color="transparent",
                    edge_width=3,
                    name="Review BBox",
                    scale=(scale_val, scale_val),
                )

        if dp is not None and hasattr(dp, "show_region_review"):
            dp.show_region_review(item)

        self.show_competing_masks(item.method_cell_ids)

    # Multi-Model Isolation View

    def show_competing_masks(
        self,
        method_cell_ids: Dict[str, List[int]],
    ) -> None:
        """Render each model's masks inside the hotspot in a distinct color."""
        ctx = self.ctx
        state = getattr(ctx, "state", None)
        shape = self._get_image_shape()
        if state is None or shape is None:
            return

        h, w = shape
        iso = np.zeros((h, w, 4), dtype=np.float32)

        dp = getattr(ctx, "disagreement_panel", None)
        compare_mode = (
            dp.combo_compare.currentText()
            if dp is not None and hasattr(dp, "combo_compare")
            else "Nuclear Masks"
        )
        method_colors_map = (
            dp.get_method_colors()
            if dp is not None and hasattr(dp, "get_method_colors")
            else {}
        )

        # In "Both" mode, use cytoplasm only if all participating methods provide it
        participating = [
            m_name
            for m_name in method_cell_ids
            if m_name in state.segmentation_results
            and (
                state.segmentation_results[m_name].get("nuclear") is not None
                or state.segmentation_results[m_name].get("cytoplasm") is not None
            )
        ]
        all_have_cyto = bool(participating) and all(
            state.segmentation_results[m_name].get("cytoplasm") is not None
            for m_name in participating
        )

        used_colors: Dict[str, str] = {}
        for i, m_name in enumerate(sorted(method_cell_ids.keys())):
            cell_ids = method_cell_ids[m_name]
            if not cell_ids or m_name not in state.segmentation_results:
                continue

            seg = state.segmentation_results[m_name]
            nuc = seg.get("nuclear")
            cyto = seg.get("cytoplasm")

            if compare_mode == "Nuclear Masks":
                combined = nuc
            elif compare_mode == "Cytoplasm Masks":
                combined = cyto
            else:
                if all_have_cyto:
                    combined = np.where(cyto > 0, cyto, nuc) if nuc is not None else cyto
                else:
                    combined = nuc

            if combined is None:
                continue

            hex_c = method_colors_map.get(m_name) or METHOD_QC_COLORS[i % len(METHOD_QC_COLORS)]
            r, g, b = _hex_to_rgb_float(hex_c)
            used_colors[m_name] = hex_c

            cell_px = np.isin(combined, cell_ids)
            if not np.any(cell_px):
                continue

            boundary = find_boundaries(cell_px.astype(np.uint8), mode="inner")

            iso[cell_px, 0] = np.clip(iso[cell_px, 0] + r * 0.35, 0.0, 1.0)
            iso[cell_px, 1] = np.clip(iso[cell_px, 1] + g * 0.35, 0.0, 1.0)
            iso[cell_px, 2] = np.clip(iso[cell_px, 2] + b * 0.35, 0.0, 1.0)
            iso[cell_px, 3] = np.maximum(iso[cell_px, 3], 0.25)

            iso[boundary, 0] = r
            iso[boundary, 1] = g
            iso[boundary, 2] = b
            iso[boundary, 3] = 0.9

        viewer = getattr(ctx, "viewer", None)
        if viewer is None:
            return

        layer_name = "Active Disagreement Isolation"
        iso_meta = {"layer_type": "disagreement_isolation", "method_colors": dict(used_colors)}
        if layer_name in viewer.layers:
            viewer.layers[layer_name].data = iso
            viewer.layers[layer_name].visible = True
            viewer.layers[layer_name].metadata = iso_meta
        else:
            scale_val = self._get_scale()
            lyr = viewer.add_image(
                iso,
                name=layer_name,
                rgb=True,
                opacity=1.0,
                blending="translucent",
                visible=True,
                scale=(scale_val, scale_val),
                metadata=iso_meta,
            )
            self._add_legend_disagreement_row(lyr, layer_name, "#FFFFFF", layer_type="image")
        self._refresh_canvas_legend()

    # Model Flickering & Active Mask Selection

    def flicker_models(self, method_a: str, method_b: str) -> Optional[str]:
        """Alternate canvas visibility between *method_a* and *method_b* mask layers."""
        ctx = self.ctx
        viewer = getattr(ctx, "viewer", None)
        state = getattr(ctx, "state", None)
        if viewer is None or state is None:
            return None

        available = list(getattr(state, "segmentation_results", {}).keys())
        if not method_a and not method_b:
            return None
        if not method_b:
            method_b = method_a
        if not method_a:
            method_a = method_b

        if method_a == method_b:
            shown = method_a
            hidden = None
        else:
            if self._active_flicker_method == method_a:
                shown, hidden = method_b, method_a
            else:
                shown, hidden = method_a, method_b

        # Ensure layers exist for shown/hidden if in segmentation_results
        from iris_viewer.core.layers import get_layer_base_name

        scale_val = self._get_scale()
        for m_name in (shown, hidden):
            if not m_name or m_name not in state.segmentation_results:
                continue
            seg = state.segmentation_results[m_name]
            for key, suffix in (("nuclear", "Nuclear Mask"), ("cytoplasm", "Cyto Mask")):
                arr = seg.get(key)
                if arr is None:
                    continue
                lname = f"{m_name} {suffix}"
                layer_opacity = 0.5 if key == "cytoplasm" else 0.8
                has_layer = any(get_layer_base_name(lyr) == lname for lyr in viewer.layers)
                if not has_layer:
                    lyr = viewer.add_labels(
                        arr,
                        name=lname,
                        opacity=layer_opacity,
                        visible=(m_name == shown),
                        scale=(scale_val, scale_val),
                    )
                    try:
                        lyr.contour = 0
                    except Exception:
                        pass

        # Toggle layer visibility cleanly across method layers (single & multi-sample)
        target_vis: Dict[str, bool] = {}
        for m_name in available:
            is_target = m_name == shown
            for suffix in ("Nuclear Mask", "Cyto Mask"):
                target_vis[f"{m_name} {suffix}"] = is_target

        for lyr in viewer.layers:
            base = get_layer_base_name(lyr)
            if base in target_vis:
                lyr.visible = target_vis[base]
                lyr.opacity = 0.5 if "cyto" in base.lower() else 0.8
                try:
                    lyr.contour = 0
                except Exception:
                    pass

        self._active_flicker_method = shown

        dp = getattr(ctx, "disagreement_panel", None)
        if dp is not None and hasattr(dp, "lbl_compute_status"):
            dp.lbl_compute_status.setText(f"Flicker: showing '{shown}'")
        iris_log("IRIS-Viewer", "Disagreement QC", f"Flicker showing '{shown}'.")

        return shown

    def set_active_mask_layers(
        self, method_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Sync state's active mask pointers and show corresponding method layers."""
        state = getattr(self.ctx, "state", None)
        if state is None:
            return None, None
        nuc, cyto = state.set_active_segmentation_method(method_name)

        viewer = getattr(self.ctx, "viewer", None)
        if viewer is not None:
            from iris_viewer.core.layers import get_layer_base_name

            target_vis: Dict[str, bool] = {}
            for m_name in state.segmentation_results.keys():
                is_active = m_name == method_name
                for suffix in ("Nuclear Mask", "Cyto Mask"):
                    target_vis[f"{m_name} {suffix}"] = is_active

            for lyr in viewer.layers:
                base = get_layer_base_name(lyr)
                if base in target_vis:
                    lyr.visible = target_vis[base]
                    lyr.opacity = 0.5 if "cyto" in base.lower() else 0.8
                    try:
                        lyr.contour = 0
                    except Exception:
                        pass

        return nuc, cyto

    # Mouse-Click Region Inspection

    def on_mouse_click(self, viewer: Any, event: Any) -> None:
        """Napari mouse_drag_callback: inspect the disagreement region under cursor."""
        ctx = self.ctx
        state = getattr(ctx, "state", None)
        if state is None or getattr(state, "disagreement_report", None) is None:
            return

        report: DisagreementReport = state.disagreement_report
        if not report.items:
            return

        ref_layer = None
        for cand_name in (
            "Disagreement Contours",
            "Disagreement Heatmap",
            "Active Disagreement Isolation",
        ):
            if cand_name in viewer.layers:
                ref_layer = viewer.layers[cand_name]
                break
        if ref_layer is None:
            for lyr in viewer.layers:
                if hasattr(lyr, "world_to_data"):
                    ref_layer = lyr
                    break
        if ref_layer is None:
            return

        data_coord = ref_layer.world_to_data(event.position)
        if len(data_coord) < 2:
            return

        y, x = int(data_coord[-2]), int(data_coord[-1])

        candidates: List[DisagreementItem] = []
        pool = list(getattr(ctx, "flagged_cells", [])) or list(report.items)
        for item in pool:
            if not isinstance(item, DisagreementItem):
                continue
            r0, c0, r1, c1 = item.bbox
            if r0 <= y < r1 and c0 <= x < c1:
                candidates.append(item)

        if not candidates and pool is not report.items:
            for item in report.items:
                r0, c0, r1, c1 = item.bbox
                if r0 <= y < r1 and c0 <= x < c1:
                    candidates.append(item)

        if not candidates:
            return

        # Prefer items where pixel (y, x) hits one of the participating masks
        pixel_hits: List[DisagreementItem] = []
        for item in candidates:
            hit = False
            for m_name, cids in item.method_cell_ids.items():
                seg = state.segmentation_results.get(m_name, {})
                for key in ("nuclear", "cytoplasm"):
                    mask = seg.get(key)
                    if (
                        mask is not None
                        and 0 <= y < mask.shape[0]
                        and 0 <= x < mask.shape[1]
                        and int(mask[y, x]) in cids
                    ):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                pixel_hits.append(item)

        chosen_pool = pixel_hits if pixel_hits else candidates
        chosen = max(
            chosen_pool,
            key=lambda it: (
                it.severity,
                -max(1, (it.bbox[2] - it.bbox[0]) * (it.bbox[3] - it.bbox[1])),
            ),
        )

        flagged = getattr(ctx, "flagged_cells", [])
        for idx, it in enumerate(flagged):
            if getattr(it, "region_id", None) == chosen.region_id:
                ctx.current_issue_idx = idx
                dp = getattr(ctx, "disagreement_panel", None)
                if dp is not None and hasattr(dp, "lbl_issue_count"):
                    dp.lbl_issue_count.setText(f"Hotspot {idx + 1} of {len(flagged)}")
                break

        self._focus_item(chosen, move_camera=False)
