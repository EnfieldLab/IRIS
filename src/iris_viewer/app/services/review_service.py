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
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries

from iris_common.logging import iris_log
from iris_viewer.engine.curation import (
    FINAL_CURATED_MASK_NAME,
    NO_CELL_CHOICE,
    CuratedMaskState,
    auto_adjudicate_consensus_mask,
    initialize_curated_mask,
    splice_method_into_regions,
)
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
        self._curated_states: Dict[str, CuratedMaskState] = {}

    def _get_scale(self) -> float:
        cw = getattr(self.ctx, "control_widget", None)
        if cw is not None and hasattr(cw, "spin_scale"):
            try:
                return float(cw.spin_scale.value())
            except Exception:
                pass
        return 1.0

    def _get_sample_translate(self) -> Tuple[float, float]:
        """Return ``(ty, tx)`` world translation for the active TMA sample."""
        ctx = self.ctx
        viewer = getattr(ctx, "viewer", None)
        state = getattr(ctx, "state", None)
        if viewer is None or state is None:
            return (0.0, 0.0)
        cur_sample = getattr(state, "current_sample_name", None)
        if not cur_sample:
            return (0.0, 0.0)
        multi_imgs = getattr(state, "multi_sample_images", None) or {}
        sample_layers = getattr(state, "sample_layers_map", None) or {}
        if not multi_imgs and len(sample_layers) <= 1:
            return (0.0, 0.0)

        from iris_viewer.core.layers import get_layer_sample_name, get_layer_type

        bounds_name = f"Bounds - {cur_sample}"
        for lyr in getattr(viewer, "layers", []):
            if getattr(lyr, "name", None) == bounds_name or (
                get_layer_type(lyr) == "bounds" and get_layer_sample_name(lyr) == cur_sample
            ):
                t = getattr(lyr, "translate", None)
                if isinstance(t, (tuple, list, np.ndarray)) and len(t) >= 2:
                    return (float(t[-2]), float(t[-1]))

        for lyr in sample_layers.get(cur_sample, []):
            t = getattr(lyr, "translate", None)
            if isinstance(t, (tuple, list, np.ndarray)) and len(t) >= 2:
                return (float(t[-2]), float(t[-1]))
        return (0.0, 0.0)

    @staticmethod
    def _apply_layer_position(
        layer: Any, scale_val: float, translate: Tuple[float, float]
    ) -> None:
        try:
            layer.scale = (scale_val, scale_val)
            layer.translate = translate
        except Exception:
            pass

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
            if name == FINAL_CURATED_MASK_NAME and (
                selected_methods is None or name not in selected_methods
            ):
                continue
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
            selected_methods = [m for m in fallback_methods if m != FINAL_CURATED_MASK_NAME]
        else:
            selected_methods = [
                m for m in state.segmentation_results.keys() if m != FINAL_CURATED_MASK_NAME
            ]

        method_colors = (
            dp.get_method_colors()
            if dp is not None and hasattr(dp, "get_method_colors")
            else None
        )
        thickness = (
            int(dp.get_contour_thickness())
            if dp is not None and hasattr(dp, "get_contour_thickness")
            else 2
        )
        fill_alpha = (
            float(dp.get_fill_alpha())
            if dp is not None and hasattr(dp, "get_fill_alpha")
            else 0.0
        )

        try:
            method_rgba, resolved_colors = render_method_comparison_rgba(
                state.segmentation_results,
                compare_mode=compare_mode,
                selected_methods=selected_methods,
                method_colors=method_colors,
                thickness=thickness,
                fill_alpha=fill_alpha,
            )
        except ValueError:
            return

        scale_val = self._get_scale()
        sample_translate = self._get_sample_translate()
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
            self._apply_layer_position(mc_lyr, scale_val, sample_translate)
        else:
            mc_lyr = viewer.add_image(
                method_rgba,
                name=mc_name,
                rgb=True,
                blending="translucent",
                visible=mc_vis,
                scale=(scale_val, scale_val),
                translate=sample_translate,
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
                else False
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
                self._sync_filled_mask_visibility_for_qc(
                    bool(show_contours) or bool(show_heatmap)
                )
            return

        state.disagreement_report = report
        state.disagreement_heatmap = report.disagreement_heatmap

        viewer = getattr(ctx, "viewer", None)
        if viewer is not None:
            scale_val = self._get_scale()
            sample_translate = self._get_sample_translate()

            # 1. Disagreement Heatmap
            hm_name = "Disagreement Heatmap"
            if hm_name in viewer.layers:
                viewer.layers[hm_name].data = report.disagreement_heatmap
                viewer.layers[hm_name].visible = bool(show_heatmap)
                self._apply_layer_position(viewer.layers[hm_name], scale_val, sample_translate)
            else:
                hm_lyr = viewer.add_image(
                    report.disagreement_heatmap,
                    name=hm_name,
                    colormap=_make_disagreement_colormap(),
                    blending="additive",
                    visible=bool(show_heatmap),
                    scale=(scale_val, scale_val),
                    translate=sample_translate,
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
                self._apply_layer_position(
                    viewer.layers[contour_name], scale_val, sample_translate
                )
            else:
                contour_lyr = viewer.add_image(
                    contour_rgba,
                    name=contour_name,
                    rgb=True,
                    blending="translucent",
                    visible=contour_vis,
                    scale=(scale_val, scale_val),
                    translate=sample_translate,
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
            self._sync_filled_mask_visibility_for_qc(
                bool(show_contours) or bool(show_heatmap)
            )

        self.update_disagreement_stats(report)

    def _sync_filled_mask_visibility_for_qc(self, qc_overlay_active: bool) -> None:
        """Hide opaque filled Labels layers while QC contours are shown so channels stay visible."""
        ctx = self.ctx
        viewer = getattr(ctx, "viewer", None)
        state = getattr(ctx, "state", None)
        if viewer is None or state is None:
            return

        dp = getattr(ctx, "disagreement_panel", None)
        keep_filled = (
            bool(dp.get_keep_filled_mask())
            if dp is not None and hasattr(dp, "get_keep_filled_mask")
            else False
        )

        from iris_viewer.core.layers import get_layer_base_name

        mask_bases = {
            "Nuclear Segmentation",
            "Cytoplasm Segmentation",
            "ROI Preview - Nuclear",
            "ROI Preview - Cytoplasm",
        }
        for m_name in getattr(state, "segmentation_results", {}).keys():
            mask_bases.add(f"{m_name} Nuclear Mask")
            mask_bases.add(f"{m_name} Cyto Mask")

        if qc_overlay_active and not keep_filled:
            for lyr in viewer.layers:
                if get_layer_base_name(lyr) in mask_bases:
                    lyr.visible = False
        else:
            active = getattr(state, "active_segmentation_method", None)
            if active and active in getattr(state, "segmentation_results", {}):
                self.set_active_mask_layers(active)
            else:
                for lyr in viewer.layers:
                    if get_layer_base_name(lyr) in (
                        "Nuclear Segmentation",
                        "Cytoplasm Segmentation",
                    ):
                        lyr.visible = True

    def refresh_contour_style(self) -> None:
        """Re-render hollow QC contours and sync filled-mask visibility on style changes."""
        ctx = self.ctx
        dp = getattr(ctx, "disagreement_panel", None)
        state = getattr(ctx, "state", None)
        if dp is None or state is None:
            return

        show_heatmap = (
            bool(dp.chk_show_heatmap.isChecked())
            if hasattr(dp, "chk_show_heatmap")
            else False
        )
        show_contours = (
            bool(dp.chk_show_contours.isChecked())
            if hasattr(dp, "chk_show_contours")
            else True
        )
        overlay_mode = (
            dp.get_overlay_color_mode()
            if hasattr(dp, "get_overlay_color_mode")
            else "Color by Method"
        )
        by_method = overlay_mode == "Color by Method"
        compare_mode = (
            dp.combo_compare.currentText()
            if hasattr(dp, "combo_compare")
            else "Nuclear Masks"
        )
        report = getattr(state, "disagreement_report", None)
        fallback = list(report.methods) if report is not None else None
        if getattr(state, "segmentation_results", None):
            self._sync_method_comparison_overlay(
                compare_mode,
                show_contours,
                by_method,
                fallback_methods=fallback,
            )

        flagged = getattr(ctx, "flagged_cells", [])
        idx = getattr(ctx, "current_issue_idx", -1)
        viewer = getattr(ctx, "viewer", None)
        if (
            viewer is not None
            and "Active Disagreement Isolation" in viewer.layers
            and bool(getattr(viewer.layers["Active Disagreement Isolation"], "visible", False))
            and 0 <= idx < len(flagged)
        ):
            self.show_competing_masks(flagged[idx].method_cell_ids)

        self._sync_filled_mask_visibility_for_qc(bool(show_contours) or bool(show_heatmap))
        self._refresh_canvas_legend()

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
        self._sync_filled_mask_visibility_for_qc(bool(show_contours) or bool(show_heatmap))
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
            sample_translate = self._get_sample_translate()
            if move_camera:
                viewer.camera.center = (
                    ((min_r + max_r) / 2.0) * scale_val + float(sample_translate[0]),
                    ((min_c + max_c) / 2.0) * scale_val + float(sample_translate[1]),
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
                self._apply_layer_position(
                    viewer.layers["Review BBox"], scale_val, sample_translate
                )
            else:
                viewer.add_shapes(
                    [bbox_poly],
                    shape_type="polygon",
                    edge_color="red",
                    face_color="transparent",
                    edge_width=3,
                    name="Review BBox",
                    scale=(scale_val, scale_val),
                    translate=sample_translate,
                )

        if dp is not None and hasattr(dp, "show_region_review"):
            if getattr(dp, "ctx", None) is None:
                dp.ctx = ctx
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

        thickness = (
            int(dp.get_contour_thickness())
            if dp is not None and hasattr(dp, "get_contour_thickness")
            else 2
        )
        keep_filled = (
            bool(dp.get_keep_filled_mask())
            if dp is not None and hasattr(dp, "get_keep_filled_mask")
            else False
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
            if thickness > 1 and np.any(boundary):
                boundary = binary_dilation(boundary, iterations=thickness - 1)

            if keep_filled:
                iso[cell_px, 0] = np.clip(iso[cell_px, 0] + r * 0.35, 0.0, 1.0)
                iso[cell_px, 1] = np.clip(iso[cell_px, 1] + g * 0.35, 0.0, 1.0)
                iso[cell_px, 2] = np.clip(iso[cell_px, 2] + b * 0.35, 0.0, 1.0)
                iso[cell_px, 3] = np.maximum(iso[cell_px, 3], 0.25)

            iso[boundary, 0] = r
            iso[boundary, 1] = g
            iso[boundary, 2] = b
            iso[boundary, 3] = 1.0

        viewer = getattr(ctx, "viewer", None)
        if viewer is None:
            return

        layer_name = "Active Disagreement Isolation"
        scale_val = self._get_scale()
        sample_translate = self._get_sample_translate()
        iso_meta = {"layer_type": "disagreement_isolation", "method_colors": dict(used_colors)}
        if layer_name in viewer.layers:
            viewer.layers[layer_name].data = iso
            viewer.layers[layer_name].visible = True
            viewer.layers[layer_name].metadata = iso_meta
            self._apply_layer_position(viewer.layers[layer_name], scale_val, sample_translate)
        else:
            lyr = viewer.add_image(
                iso,
                name=layer_name,
                rgb=True,
                opacity=1.0,
                blending="translucent",
                visible=True,
                scale=(scale_val, scale_val),
                translate=sample_translate,
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
            dp.lbl_compute_status.setText(f"Blink Compare: showing '{shown}'")
        iris_log("IRIS-Viewer", "Disagreement QC", f"Blink compare showing '{shown}'.")

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

    # Final Mask Curation & Adjudication

    def _upsert_curated_labels_layer(
        self,
        data: np.ndarray,
        layer_name: str,
        *,
        opacity: float,
        scale_val: float,
        translate: Tuple[float, float],
        legend_color: str,
    ) -> Any:
        ctx = self.ctx
        viewer = getattr(ctx, "viewer", None)
        state = getattr(ctx, "state", None)
        if viewer is None:
            return None

        from iris_viewer.core.layers import get_layer_base_name, get_layer_sample_name

        cur_sample = getattr(state, "current_sample_name", None) if state is not None else None
        existing_lyr = None
        for lyr in viewer.layers:
            if get_layer_base_name(lyr) == layer_name and (
                cur_sample is None or get_layer_sample_name(lyr) in (cur_sample, None)
            ):
                existing_lyr = lyr
                break

        if existing_lyr is not None:
            lyr = existing_lyr
            lyr.data = data
            lyr.opacity = float(opacity)
            lyr.visible = True
            self._apply_layer_position(lyr, scale_val, translate)
        else:
            meta: Dict[str, Any] = {
                "layer_type": "segmentation",
                "base_name": layer_name,
                "segmentation_name": layer_name,
            }
            if cur_sample:
                meta["sample_name"] = cur_sample
            lyr = viewer.add_labels(
                data,
                name=layer_name,
                opacity=float(opacity),
                visible=True,
                scale=(scale_val, scale_val),
                translate=translate,
                metadata=meta,
            )

        try:
            lyr.contour = 0
        except Exception:
            pass

        if hasattr(viewer.layers, "move"):
            try:
                idx = list(viewer.layers).index(lyr)
                top = len(viewer.layers) - 1
                if 0 <= idx < top:
                    viewer.layers.move(idx, top)
            except Exception:
                pass

        if state is not None and cur_sample:
            sl_map = getattr(state, "sample_layers_map", None)
            if isinstance(sl_map, dict):
                sl = sl_map.setdefault(cur_sample, [])
                if lyr not in sl:
                    sl.append(lyr)

        legend = getattr(ctx, "legend_widget", None)
        entries = getattr(ctx, "added_legend_entries", None)
        if legend is not None and hasattr(legend, "add_segmentation_row"):
            legend.add_segmentation_row(lyr, layer_name, legend_color)
            if isinstance(entries, set):
                entries.add(layer_name)

        return lyr

    def register_final_curated_mask(
        self,
        curated: CuratedMaskState,
        refresh_dialog_table: bool = True,
    ) -> CuratedMaskState:
        """Persist *curated* into state, Napari layers, Legend, and Feature panel."""
        ctx = self.ctx
        state = getattr(ctx, "state", None)
        if state is None:
            return curated

        cur_sample = getattr(state, "current_sample_name", None)
        seg_entry = {
            "nuclear": curated.nuclear,
            "cytoplasm": curated.cytoplasm,
        }
        state.segmentation_results[FINAL_CURATED_MASK_NAME] = seg_entry
        state.curated_mask_state = curated
        state._curated_sample_name = cur_sample
        self._curated_states[cur_sample or "__default__"] = curated

        if cur_sample:
            batch_res = getattr(state, "batch_results", None)
            if isinstance(batch_res, dict) and cur_sample in batch_res and isinstance(
                batch_res[cur_sample], dict
            ):
                batch_res[cur_sample][FINAL_CURATED_MASK_NAME] = dict(seg_entry)

            sample_seg_res = getattr(state, "_sample_seg_results", None)
            if isinstance(sample_seg_res, dict):
                sample_seg_res.setdefault(cur_sample, {})[FINAL_CURATED_MASK_NAME] = dict(
                    seg_entry
                )

            masks_map = getattr(state, "masks_map", None)
            if isinstance(masks_map, dict):
                masks_map[cur_sample] = curated.nuclear

        viewer = getattr(ctx, "viewer", None)
        if viewer is not None:
            scale_val = self._get_scale()
            sample_translate = self._get_sample_translate()
            self._upsert_curated_labels_layer(
                curated.nuclear,
                f"{FINAL_CURATED_MASK_NAME} Nuclear Mask",
                opacity=0.8,
                scale_val=scale_val,
                translate=sample_translate,
                legend_color="#34C759",
            )
            cyto_layer_name = f"{FINAL_CURATED_MASK_NAME} Cyto Mask"
            if curated.cytoplasm is not None:
                self._upsert_curated_labels_layer(
                    curated.cytoplasm,
                    cyto_layer_name,
                    opacity=0.5,
                    scale_val=scale_val,
                    translate=sample_translate,
                    legend_color="#4CC9F0",
                )
            elif cyto_layer_name in viewer.layers:
                viewer.layers[cyto_layer_name].visible = False

        self.set_active_mask_layers(FINAL_CURATED_MASK_NAME)

        all_methods = [
            m
            for m, r in state.segmentation_results.items()
            if isinstance(r, dict)
            and (r.get("nuclear") is not None or r.get("cytoplasm") is not None)
        ]

        fp = getattr(ctx, "feature_panel", None)
        if fp is not None and hasattr(fp, "update_mask_sources"):
            fp.update_mask_sources(all_methods, active=FINAL_CURATED_MASK_NAME)

        n_cells = int(np.sum(np.unique(curated.nuclear) > 0)) if curated.nuclear.size > 0 else 0
        status_txt = (
            f"Final Mask: {FINAL_CURATED_MASK_NAME} "
            f"({n_cells:,} cells | base={curated.base_method} | "
            f"{curated.curated_regions_count} region(s) curated)"
        )

        dp = getattr(ctx, "disagreement_panel", None)
        if dp is not None:
            if hasattr(dp, "set_final_mask_status"):
                dp.set_final_mask_status(status_txt, active=True)
            elif hasattr(dp, "lbl_final_mask_status"):
                dp.lbl_final_mask_status.setText(status_txt)
            if hasattr(dp, "update_available_methods"):
                dp.update_available_methods(all_methods, active_method=FINAL_CURATED_MASK_NAME)
            if hasattr(dp, "_refresh_hotspot_curation_bar"):
                dp._refresh_hotspot_curation_bar()
            if hasattr(dp, "curationStateChanged"):
                dp.curationStateChanged.emit()

        cw = getattr(ctx, "control_widget", None)
        if cw is not None and hasattr(cw, "update_dashboard"):
            cw.update_dashboard(total_cells=f"{n_cells:,}")

        sp = getattr(ctx, "segmentation_panel", None)
        if sp is not None and getattr(sp, "popup_dialog", None) is not None:
            dlg = sp.popup_dialog
            if hasattr(dlg, "lbl_final_mask_status"):
                dlg.lbl_final_mask_status.setText(status_txt)
            if refresh_dialog_table:
                if hasattr(dlg, "isVisible") and dlg.isVisible() and hasattr(dlg, "refresh_data"):
                    dlg.refresh_data()
            else:
                if hasattr(dlg, "_update_table_picked_column"):
                    dlg._update_table_picked_column()
                if hasattr(dlg, "_refresh_hotspot_curation_bar"):
                    dlg._refresh_hotspot_curation_bar()
                if hasattr(dlg, "_sync_final_mask_status"):
                    dlg._sync_final_mask_status()

        self._refresh_canvas_legend()
        iris_log(
            "IRIS-Viewer",
            "Disagreement QC",
            f"Registered '{FINAL_CURATED_MASK_NAME}': {n_cells:,} cells "
            f"(base={curated.base_method}, curated_regions={curated.curated_regions_count}).",
        )
        return curated

    def _resolve_reference_base_method(self) -> str:
        """Prefer Reference method from report/UI when seeding a fresh curated mask."""
        state = getattr(self.ctx, "state", None)
        if state is None or not getattr(state, "segmentation_results", None):
            return ""
        seg = state.segmentation_results

        def _is_valid(cand: Optional[str]) -> bool:
            if not cand or cand == FINAL_CURATED_MASK_NAME or cand not in seg:
                return False
            entry = seg[cand]
            return isinstance(entry, dict) and entry.get("nuclear") is not None

        report = getattr(state, "disagreement_report", None)
        rep_ref = getattr(report, "reference_method", None) if report is not None else None
        if _is_valid(rep_ref):
            return str(rep_ref)

        dp = getattr(self.ctx, "disagreement_panel", None)
        if dp is not None and hasattr(dp, "get_reference_method"):
            dp_ref = dp.get_reference_method()
            if _is_valid(dp_ref):
                return str(dp_ref)

        active = getattr(state, "active_segmentation_method", None)
        if _is_valid(active):
            return str(active)

        if dp is not None and hasattr(dp, "get_active_method"):
            dp_active = dp.get_active_method()
            if _is_valid(dp_active):
                return str(dp_active)

        for m in seg.keys():
            if _is_valid(m):
                return str(m)
        return ""

    def _get_existing_curated_state(self) -> Optional[CuratedMaskState]:
        state = getattr(self.ctx, "state", None)
        if state is None or not getattr(state, "segmentation_results", None):
            return None

        cur_sample = getattr(state, "current_sample_name", None)
        key = cur_sample or "__default__"
        existing = self._curated_states.get(key)
        if existing is None:
            cand_state = getattr(state, "curated_mask_state", None)
            cand_sample = getattr(state, "_curated_sample_name", cur_sample)
            if cand_state is not None and cand_sample == cur_sample:
                existing = cand_state

        if (
            existing is not None
            and FINAL_CURATED_MASK_NAME in state.segmentation_results
            and existing.base_method in state.segmentation_results
        ):
            cur_nuc = state.segmentation_results[FINAL_CURATED_MASK_NAME].get("nuclear")
            if cur_nuc is not None and cur_nuc.shape == existing.nuclear.shape:
                return existing
        return None

    def _get_or_init_curated_state(self) -> Optional[CuratedMaskState]:
        state = getattr(self.ctx, "state", None)
        if state is None or not getattr(state, "segmentation_results", None):
            return None

        existing = self._get_existing_curated_state()
        if existing is not None:
            return existing

        base_method = self._resolve_reference_base_method()
        if not base_method:
            return None
        return initialize_curated_mask(state.segmentation_results, base_method)

    def reset_curated_mask_to_reference(self) -> Optional[CuratedMaskState]:
        """Reset 'Final Curated Mask' cleanly back to the Reference pipeline."""
        state = getattr(self.ctx, "state", None)
        if state is None or not getattr(state, "segmentation_results", None):
            return None

        ref_method = self._resolve_reference_base_method()
        if not ref_method:
            return None
        curated = initialize_curated_mask(state.segmentation_results, ref_method)
        return self.register_final_curated_mask(curated, refresh_dialog_table=True)

    def use_pipeline_as_final_mask(self, method_name: str) -> Optional[CuratedMaskState]:
        """Use a single selected pipeline's paired masks as the Final Curated Mask."""
        state = getattr(self.ctx, "state", None)
        if state is None or not method_name or method_name not in state.segmentation_results:
            return None
        if method_name == FINAL_CURATED_MASK_NAME:
            return self._get_or_init_curated_state()
        curated = initialize_curated_mask(state.segmentation_results, method_name)
        return self.register_final_curated_mask(curated)

    def auto_adjudicate_final_mask(self) -> Optional[CuratedMaskState]:
        """Auto-resolve uncurated disputed regions via >= 3 model peer consensus."""
        state = getattr(self.ctx, "state", None)
        if state is None or not getattr(state, "segmentation_results", None):
            return None

        dp = getattr(self.ctx, "disagreement_panel", None)
        threshold = (
            float(dp.get_threshold())
            if dp is not None and hasattr(dp, "get_threshold")
            else 0.65
        )
        report = getattr(state, "disagreement_report", None)
        if report is None:
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
            selected_methods = (
                dp.get_selected_methods()
                if (
                    dp is not None
                    and hasattr(dp, "get_selected_methods")
                    and getattr(dp, "_methods", [])
                )
                else None
            )
            seg_inputs = {
                m: r
                for m, r in state.segmentation_results.items()
                if m != FINAL_CURATED_MASK_NAME
                and (selected_methods is None or m in selected_methods)
            }
            if not seg_inputs:
                return None
            if ref_method not in seg_inputs:
                ref_method = None
            report = compute_model_disagreement(
                seg_inputs,
                compare_mode=compare_mode,
                reference_method=ref_method,
                iou_threshold=threshold,
                selected_methods=selected_methods,
            )
            state.disagreement_report = report
            state.disagreement_heatmap = report.disagreement_heatmap
            if getattr(self.ctx, "viewer", None) is not None:
                self.render_disagreement_overlays(report)

        existing = self._get_existing_curated_state()
        curated = auto_adjudicate_consensus_mask(
            state.segmentation_results,
            report,
            existing_curated=existing,
        )
        return self.register_final_curated_mask(curated)

    @staticmethod
    def _validate_curation_source(
        state: Any, source_method: Optional[str]
    ) -> Optional[str]:
        if state is None or not getattr(state, "segmentation_results", None):
            return None
        clean = "" if source_method is None else str(source_method).strip()
        if clean in ("", NO_CELL_CHOICE):
            return NO_CELL_CHOICE
        if clean == FINAL_CURATED_MASK_NAME or clean not in state.segmentation_results:
            return None
        return clean

    def curate_hotspot_and_advance(
        self,
        source_method: str,
        item: Optional[Any] = None,
        auto_advance: bool = True,
    ) -> Optional[CuratedMaskState]:
        """Keep *source_method* (or NO_CELL_CHOICE) for one hotspot and optionally advance."""
        state = getattr(self.ctx, "state", None)
        resolved_source = self._validate_curation_source(state, source_method)
        if resolved_source is None or state is None:
            return None

        flagged = list(getattr(self.ctx, "flagged_cells", []) or [])
        if not flagged:
            report = getattr(state, "disagreement_report", None)
            rep_items = list(getattr(report, "items", []) or []) if report is not None else []
            if rep_items:
                flagged = rep_items
                self.ctx.flagged_cells = flagged

        target_item = item
        if target_item is None:
            if not flagged:
                return None
            idx = int(getattr(self.ctx, "current_issue_idx", -1))
            if idx < 0 or idx >= len(flagged):
                idx = 0
                self.ctx.current_issue_idx = 0
            target_item = flagged[idx]
        elif flagged:
            target_rid = (
                target_item.get("region_id")
                if isinstance(target_item, dict)
                else getattr(target_item, "region_id", None)
            )
            if target_rid is not None:
                for i, f_it in enumerate(flagged):
                    f_rid = (
                        f_it.get("region_id")
                        if isinstance(f_it, dict)
                        else getattr(f_it, "region_id", None)
                    )
                    if f_rid == target_rid:
                        self.ctx.current_issue_idx = i
                        break

        curated = self._get_or_init_curated_state()
        if curated is None:
            return None

        updated = splice_method_into_regions(
            curated,
            state.segmentation_results,
            resolved_source,
            items=[target_item],
        )
        registered = self.register_final_curated_mask(updated, refresh_dialog_table=False)

        if auto_advance and len(flagged) > 1:
            self.next_issue()
        else:
            dp = getattr(self.ctx, "disagreement_panel", None)
            if dp is not None and hasattr(dp, "show_region_review"):
                if getattr(dp, "ctx", None) is None:
                    dp.ctx = self.ctx
                dp.show_region_review(target_item)
            sp = getattr(self.ctx, "segmentation_panel", None)
            if sp is not None and getattr(sp, "popup_dialog", None) is not None:
                dlg = sp.popup_dialog
                if hasattr(dlg, "_update_region_card"):
                    dlg._update_region_card(target_item)
                if hasattr(dlg, "_draw_hotspot_patch"):
                    dlg._draw_hotspot_patch(target_item)
                if hasattr(dlg, "_refresh_hotspot_curation_bar"):
                    dlg._refresh_hotspot_curation_bar()

        return registered

    def apply_hotspot_curation(
        self,
        source_method: str,
        items: Optional[List[Any]] = None,
        use_filtered_category: bool = False,
        refresh_dialog_table: bool = True,
    ) -> Optional[CuratedMaskState]:
        """Splice *source_method* into selected hotspot item(s) or filtered category."""
        state = getattr(self.ctx, "state", None)
        resolved_source = self._validate_curation_source(state, source_method)
        if resolved_source is None or state is None:
            return None

        target_items: List[Any] = []
        if use_filtered_category:
            if items:
                target_items = list(items)
            else:
                flagged = list(getattr(self.ctx, "flagged_cells", []) or [])
                if flagged:
                    target_items = flagged
                else:
                    report = getattr(state, "disagreement_report", None)
                    dp = getattr(self.ctx, "disagreement_panel", None)
                    cat = (
                        dp.get_category_filter()
                        if dp is not None and hasattr(dp, "get_category_filter")
                        else "All Disagreements"
                    )
                    rep_items = list(getattr(report, "items", []) or [])
                    if cat and cat != "All Disagreements":
                        target_items = [
                            it
                            for it in rep_items
                            if getattr(it, "category", None) == cat
                        ]
                    else:
                        target_items = rep_items
        else:
            if items:
                target_items = list(items)
            else:
                flagged = list(getattr(self.ctx, "flagged_cells", []) or [])
                idx = getattr(self.ctx, "current_issue_idx", -1)
                if 0 <= idx < len(flagged):
                    target_items = [flagged[idx]]

        if not target_items:
            return None

        curated = self._get_or_init_curated_state()
        if curated is None:
            return None

        updated = splice_method_into_regions(
            curated,
            state.segmentation_results,
            resolved_source,
            items=target_items,
        )
        return self.register_final_curated_mask(
            updated, refresh_dialog_table=refresh_dialog_table
        )

    def draw_curation_roi(self) -> Any:
        """Create or focus editable 'Curation Selection ROI' Shapes layer in Napari."""
        ctx = self.ctx
        viewer = getattr(ctx, "viewer", None)
        if viewer is None:
            return None

        shape = self._get_image_shape()
        h, w = shape if shape is not None else (512, 512)
        scale_val = self._get_scale()
        translate = self._get_sample_translate()

        box_h = min(128, max(16, int(h // 2)))
        box_w = min(128, max(16, int(w // 2)))
        cy_px, cx_px = float(h) / 2.0, float(w) / 2.0

        cam = getattr(viewer, "camera", None)
        center = getattr(cam, "center", None)
        if center is not None and len(center) >= 2:
            cand_y = float(center[-2]) / max(scale_val, 1e-6) - float(translate[0]) / max(
                scale_val, 1e-6
            )
            cand_x = float(center[-1]) / max(scale_val, 1e-6) - float(translate[1]) / max(
                scale_val, 1e-6
            )
            if (
                np.isfinite(cand_y)
                and np.isfinite(cand_x)
                and (float(center[-2]) != 0.0 or float(center[-1]) != 0.0)
                and 0.0 <= cand_y < float(h)
                and 0.0 <= cand_x < float(w)
            ):
                cy_px, cx_px = cand_y, cand_x

        y0 = max(0, min(int(round(cy_px - box_h / 2.0)), max(0, h - box_h)))
        x0 = max(0, min(int(round(cx_px - box_w / 2.0)), max(0, w - box_w)))
        y1 = min(h, y0 + box_h)
        x1 = min(w, x0 + box_w)
        rect = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)

        roi_name = "Curation Selection ROI"
        if roi_name in viewer.layers:
            lyr = viewer.layers[roi_name]
            if not getattr(lyr, "data", None):
                lyr.data = [rect]
            lyr.visible = True
            self._apply_layer_position(lyr, scale_val, translate)
        else:
            lyr = viewer.add_shapes(
                [rect],
                shape_type="polygon",
                name=roi_name,
                edge_color="#FFD60A",
                face_color=[1.0, 0.84, 0.04, 0.10],
                edge_width=2,
                scale=(scale_val, scale_val),
                translate=translate,
            )

        try:
            lyr.mode = "select"
        except Exception:
            pass

        try:
            viewer.layers.selection.clear()
            viewer.layers.selection.add(lyr)
        except Exception:
            pass

        legend = getattr(ctx, "legend_widget", None)
        entries = getattr(ctx, "added_legend_entries", None)
        if legend is not None and hasattr(legend, "add_segmentation_row"):
            try:
                legend.add_segmentation_row(lyr, roi_name, "#FFD60A", layer_type="shapes")
            except TypeError:
                legend.add_segmentation_row(lyr, roi_name, "#FFD60A")
            if isinstance(entries, set):
                entries.add(roi_name)

        iris_log("IRIS-Viewer", "Disagreement QC", f"Focused '{roi_name}' on canvas.")
        return lyr

    def apply_roi_curation(self, source_method: str) -> Optional[CuratedMaskState]:
        """Splice *source_method* (or clear for NO_CELL_CHOICE) inside ROI polygons."""
        ctx = self.ctx
        viewer = getattr(ctx, "viewer", None)
        state = getattr(ctx, "state", None)
        resolved_source = self._validate_curation_source(state, source_method)
        if viewer is None or state is None or resolved_source is None:
            return None

        roi_name = "Curation Selection ROI"
        if roi_name not in viewer.layers:
            return None
        data = getattr(viewer.layers[roi_name], "data", None)
        if not data:
            return None

        polys: List[np.ndarray] = []
        for shape_pts in data:
            arr = np.asarray(shape_pts, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 2:
                polys.append(arr[:, -2:])
        if not polys:
            return None

        curated = self._get_or_init_curated_state()
        if curated is None:
            return None

        updated = splice_method_into_regions(
            curated,
            state.segmentation_results,
            resolved_source,
            polygon_yx=polys,
        )
        return self.register_final_curated_mask(updated)
