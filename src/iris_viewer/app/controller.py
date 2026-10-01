""" IRIS-Viewer Controller """
import os
import threading
import sys
from pathlib import Path

local_data_dir = Path(__file__).parent.parent / "data"
local_data_dir.mkdir(exist_ok=True)

os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(local_data_dir / "cellpose")
os.environ["INSTANSEG_MODEL_PATH"] = str(local_data_dir / "instanseg_models")
os.environ["INSTANSEG_BIOIMAGEIO_PATH"] = str(local_data_dir / "bioimageio")
os.environ["TORCH_HOME"] = str(local_data_dir / "torch")
os.environ["HF_HOME"] = str(local_data_dir / "huggingface")

import numpy as np
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional
import napari
from napari.qt import thread_worker
from napari.utils.colormaps import Colormap
from qtpy.QtWidgets import QFileDialog, QMessageBox, QMenu

from iris_viewer.core.state import ViewerAppState
from iris_viewer.engine.segmentation import get_available_backends
from iris_viewer.engine.pipeline import PipelineService, SegmentationSequence, SegmentationStep
from iris_viewer.engine.consensus import compute_consensus_mask, apply_manual_override
from iris_viewer.engine.feature_extraction import extract_cell_features
from iris_viewer.io_utils.image_loaders import (
    discover_image_files, load_multichannel_tiff,
    load_existing_mask, probe_channel_count
)
from iris_viewer.io_utils.exporters import (
    export_consensus_mask, export_features_csv,
    export_confidence_map, export_batch_summary
)

from iris_viewer.widgets.control import ViewerControlWidget
from iris_viewer.widgets.segmentation_panel import SegmentationPanelWidget
from iris_viewer.widgets.consensus_panel import ConsensusPanelWidget
from iris_viewer.widgets.feature_panel import FeaturePanelWidget
from iris_viewer.widgets.legend import ViewerLegendWidget
from iris_viewer.widgets.config_dialogs import ViewerPanelConfigDialog
from iris_viewer.widgets.styles import (
    CONFIDENCE_HIGH_COLOR, CONFIDENCE_MEDIUM_COLOR, CONFIDENCE_LOW_COLOR
)

DEFAULT_COLORS = {
    'DAPI': 'white', 
    'ASMA': 'green', 'SMA': 'green',
    'CD45': 'purple',
    'CK': 'red', 'PANCK': 'red',
}

class ViewerController:
    def __init__(self):
        self.state = ViewerAppState()
        self.backends = get_available_backends()
        self._pending_seg_count = 0
        self._completed_seg_count = 0
        self.added_legend_entries = set()
        self.pipeline_service = PipelineService()
        
        self.viewer = napari.Viewer(title="IRIS-Viewer")
        try:
            self.viewer.scale_bar.visible = True
            self.viewer.scale_bar.ticks = False
            self.viewer.scale_bar.box = False
            self.viewer.scale_bar.font_size = 26.0
            self.viewer.scale_bar.position = "bottom_left"
        except AttributeError:
            pass

        self.control_widget = ViewerControlWidget(viewer=self.viewer)
        self.segmentation_panel = SegmentationPanelWidget()
        self.consensus_panel = ConsensusPanelWidget()
        self.feature_panel = FeaturePanelWidget()
        self.legend_widget = ViewerLegendWidget(self.viewer)

        self.viewer.window.add_dock_widget(self.control_widget, name="Control", area="right")
        self.viewer.window.add_dock_widget(self.segmentation_panel, name="Segmentation", area="right")
        self.viewer.window.add_dock_widget(self.consensus_panel, name="Consensus", area="right")
        self.viewer.window.add_dock_widget(self.feature_panel, name="Features", area="right")
        self.viewer.window.add_dock_widget(self.legend_widget, name="Legend", area="right")

        self.viewer.window._qt_window.tabifyDockWidget(
            self.viewer.window.dock_widgets["Control"].parent(),
            self.viewer.window.dock_widgets["Segmentation"].parent(),
        )
        self.viewer.window._qt_window.tabifyDockWidget(
            self.viewer.window.dock_widgets["Segmentation"].parent(),
            self.viewer.window.dock_widgets["Consensus"].parent(),
        )
        self.viewer.window._qt_window.tabifyDockWidget(
            self.viewer.window.dock_widgets["Consensus"].parent(),
            self.viewer.window.dock_widgets["Features"].parent(),
        )
        self.viewer.window._qt_window.tabifyDockWidget(
            self.viewer.window.dock_widgets["Features"].parent(),
            self.viewer.window.dock_widgets["Legend"].parent(),
        )
        self.viewer.window.dock_widgets["Control"].parent().raise_()

        self._configure_backend_availability()
        self._bind_control_signals()
        self._bind_segmentation_signals()
        self._bind_consensus_signals()
        self._bind_feature_signals()
        self._bind_viewer_events()

    def _configure_backend_availability(self):
        self.segmentation_panel.update_available_methods(self.pipeline_service.get_available_methods())

    def _bind_control_signals(self):
        self.control_widget.loadDirectoryTriggered.connect(self._on_load_directory)
        self.control_widget.sampleSelected.connect(self._on_sample_selected)
        self.control_widget.loadExistingMaskTriggered.connect(self._on_load_existing_mask)
        self.control_widget.exportMaskTriggered.connect(self._export_mask)
        self.control_widget.exportFeaturesTriggered.connect(self._export_features)
        self.control_widget.exportConfidenceTriggered.connect(self._export_confidence)
        self.control_widget.batchProcessTriggered.connect(self._run_batch_processing)
        self.control_widget.loadChannelColorTriggered.connect(self._load_channel_colors)
        self.control_widget.viewLockToggled.connect(self._toggle_view_lock)

    def _bind_segmentation_signals(self):
        self.segmentation_panel.runSequence.connect(self._run_segmentation_pipeline)
        self.segmentation_panel.cancelSequence.connect(self._cancel_segmentation)

    def _bind_consensus_signals(self):
        self.consensus_panel.generateConsensusRequested.connect(self._generate_consensus)
        self.consensus_panel.thresholdChanged.connect(self._update_confidence_overlay)
        self.consensus_panel.cellOverrideRequested.connect(self._apply_manual_override)
        self.consensus_panel.showConsensusToggled.connect(self._toggle_consensus_visibility)
        self.consensus_panel.showIndividualToggled.connect(self._toggle_method_mask_visibility)
        self.consensus_panel.reviewModeToggled.connect(self._toggle_review_mode)
        
        self.legend_widget.layerSelected.connect(self._select_layer)
        self.control_widget.scaleFactorChanged.connect(self._update_scale)

    def _select_layer(self, layer_name: str):
        self.viewer.layers.selection.clear()
        for layer in self.viewer.layers:
            l_base = layer.name.split(" [")[0] if " [" in layer.name else layer.name
            if l_base == layer_name:
                self.viewer.layers.selection.add(layer)

    def _update_scale(self, scale_val: float):
        try:
            self.viewer.scale_bar.visible = True
            self.viewer.scale_bar.unit = "um" if scale_val != 1.0 else "px"
        except AttributeError:
            pass
            
        for layer in self.viewer.layers:
            if hasattr(layer, 'scale'):
                layer.scale = (scale_val, scale_val)

    def _bind_feature_signals(self):
        self.feature_panel.extractTriggered.connect(self._run_feature_extraction)
        self.feature_panel.exportTriggered.connect(self._export_features)

    def _bind_viewer_events(self):
        self.viewer.mouse_drag_callbacks.append(self._on_mouse_click)

    def _toggle_view_lock(self, is_locked: bool):
        self.viewer.camera.mouse_pan = not is_locked
        self.viewer.camera.mouse_zoom = not is_locked

    def _load_channel_colors(self, filepath: str):
        try:
            import pandas as pd
            import matplotlib.colors as mcolors
            from pathlib import Path
            path_obj = Path(filepath)
            df = pd.read_excel(filepath) if path_obj.suffix.lower() == '.xlsx' else pd.read_csv(filepath)
            
            df.columns = [str(c).lower().strip() for c in df.columns]
            if 'markers' not in df.columns or 'colors' not in df.columns:
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.warning(self.control_widget, "Invalid Format", "File must contain 'markers' and 'colors' headers.")
                return
                
            color_map = {}
            failed_markers = []
            for _, row in df.iterrows():
                marker = str(row['markers']).strip().lower()
                color_val = str(row['colors']).strip()
                if pd.notna(row['markers']) and pd.notna(row['colors']) and mcolors.is_color_like(color_val):
                    color_map[marker] = mcolors.to_hex(color_val)
                else:
                    failed_markers.append(marker)
                    
            updated_count = 0
            for m in self.state.markers:
                if m.lower() in color_map:
                    new_color = color_map[m.lower()]
                    for row in self.legend_widget.channel_rows:
                        if row.name.strip().lower() == m.lower():
                            row.update_swatch_display(new_color)
                            row.colorChanged.emit(row.name, new_color)
                            updated_count += 1
                            break
                            
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.information(self.control_widget, "Success", f"Updated {updated_count} marker colors.")
        except Exception as e:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.critical(self.control_widget, "Error", f"Failed to load marker colors: {str(e)}")

    def _on_load_directory(self, dir_path: str):
        self.state.image_dir = Path(dir_path)
        self.state.file_map = discover_image_files(self.state.image_dir)
        
        if not self.state.file_map:
            QMessageBox.warning(self.control_widget, "No Images Found", f"No TIFF images found in:\n{dir_path}")
            return
            
        sample_names = sorted(self.state.file_map.keys())
        first_img_path = self.state.file_map[sample_names[0]]
        
        if not self.state.markers:
            num_channels = probe_channel_count(first_img_path)
            dialog = ViewerPanelConfigDialog(num_channels, parent=self.control_widget)
            if not dialog.exec_():
                QMessageBox.warning(self.control_widget, "Cancelled", "Channel assignment cancelled.")
                return
                
            self.state.markers = dialog.marker_panel
            self.segmentation_panel.update_channel_dropdowns(self.state.markers)
            
            QMessageBox.information(self.control_widget, "Channels Assigned", f"Successfully assigned {len(self.state.markers)} markers.")
            
        self.control_widget.update_sample_list(sample_names)
        if sample_names:
            self.control_widget.combo_sample.setCurrentIndex(0)
            self._on_sample_selected(sample_names[0])

    def _on_sample_selected(self, sample_name: str):
        if not sample_name or sample_name not in self.state.file_map:
            return
            
        self.viewer.layers.clear()
        self.legend_widget.clear()
        self.added_legend_entries.clear()
        
        self.state.reset_sample_state()
        self.state.current_sample_name = sample_name
        img_path = self.state.file_map[sample_name]
        
        @thread_worker
        def _load_image():
            raw_image, n_ch = load_multichannel_tiff(img_path)
            return raw_image, n_ch
            
        worker = _load_image()
        worker.returned.connect(lambda res: self._on_image_loaded(res, sample_name))
        worker.start()

    def _on_image_loaded(self, result, sample_name: str):
        raw_image, n_channels = result
        self.state.raw_image = raw_image
        scale_val = self.control_widget.spin_scale.value()
        
        for i, marker in enumerate(self.state.markers):
            if i >= raw_image.shape[0]:
                break
            chan_data = raw_image[i]
            # Fast approximation of contrast limits to avoid computing full dask array
            step_h = max(1, chan_data.shape[0] // 500)
            step_w = max(1, chan_data.shape[1] // 500)
            small_chan = chan_data[::step_h, ::step_w]
            if hasattr(small_chan, 'compute'):
                small_chan = small_chan.compute()
                
            clims = [float(np.percentile(small_chan, 1)), float(np.percentile(small_chan, 99.9))]
            if clims[0] >= clims[1]:
                clims[1] = clims[0] + 1.0
                
            cname = DEFAULT_COLORS.get(marker.upper(), "white")
            is_vis = marker.upper() in ("DAPI", "CK", "ASMA", "CD45")
            
            layer = self.viewer.add_image(
                chan_data, name=marker, colormap=cname,
                blending='additive', visible=is_vis,
                contrast_limits=clims, scale=(scale_val, scale_val),
            )
            if marker not in self.added_legend_entries:
                try:
                    init_color = layer.colormap.map([1.0])[0]
                except Exception:
                    init_color = '#FFFFFF'
                self.legend_widget.add_channel_row(layer, marker, init_color)
                self.added_legend_entries.add(marker)
                
        img_path = self.state.file_map[sample_name]
        mask_path = img_path.with_name(f"{sample_name}_mask.tiff")
        if mask_path.exists():
            self._load_and_display_mask(mask_path, sample_name, scale_val)
            
        h, w = raw_image.shape[-2], raw_image.shape[-1]
        self.control_widget.update_dashboard(image_dims=f"{w} x {h}", num_channels=str(n_channels))
        
        QMessageBox.information(
            self.control_widget, "Image Loaded",
            f"Successfully loaded {sample_name} ({n_channels} channels, {w} x {h})."
        )

    def _on_load_existing_mask(self, filepath: str):
        mask_path = Path(filepath)
        scale_val = self.control_widget.spin_scale.value()
        sample_name = self.state.current_sample_name or "External"
        self._load_and_display_mask(mask_path, sample_name, scale_val)

    def _load_and_display_mask(self, mask_path: Path, sample_name: str, scale_val: float):
        mask_data = load_existing_mask(mask_path)
        if mask_data is None:
            return
            
        if mask_data.ndim == 3:
            for i in range(mask_data.shape[0]):
                layer_name = f"Loaded Mask Ch{i}" if mask_data.shape[0] > 1 else "Loaded Mask"
                try:
                    seg_layer = self.viewer.add_labels(
                        mask_data[i], name=layer_name, opacity=1.0,
                        visible=(i == 0), contour=1, scale=(scale_val, scale_val),
                    )
                except TypeError:
                    seg_layer = self.viewer.add_labels(
                        mask_data[i], name=layer_name, opacity=1.0,
                        visible=(i == 0), scale=(scale_val, scale_val),
                    )
                if layer_name not in self.added_legend_entries:
                    self.legend_widget.add_segmentation_row(seg_layer, layer_name, "#888888")
                    self.added_legend_entries.add(layer_name)
                    
            self.state.segmentation_results["Loaded"] = {
                'nuclear': mask_data[0],
                'cytoplasm': mask_data[1] if mask_data.shape[0] > 1 else None,
            }
        else:
            layer_name = "Loaded Mask"
            try:
                seg_layer = self.viewer.add_labels(
                    mask_data, name=layer_name, opacity=1.0,
                    visible=True, contour=1, scale=(scale_val, scale_val),
                )
            except TypeError:
                seg_layer = self.viewer.add_labels(
                    mask_data, name=layer_name, opacity=1.0,
                    visible=True, scale=(scale_val, scale_val),
                )
            if layer_name not in self.added_legend_entries:
                self.legend_widget.add_segmentation_row(seg_layer, layer_name, "#888888")
                self.added_legend_entries.add(layer_name)
                
            self.state.segmentation_results["Loaded"] = {'nuclear': mask_data, 'cytoplasm': None}
            
        n_cells = len(np.unique(mask_data)) - 1
        self.control_widget.update_dashboard(seg_status="Loaded from file", total_cells=f"{n_cells:,}")

    def _run_segmentation_pipeline(self, steps: list):
        if self.state.raw_image is None:
            QMessageBox.warning(self.control_widget, "No Image", "Load an image before running segmentation.")
            return

        self.state.segmentation_results.clear()
        self.state.segmentation_status.clear()
        
        for idx, step in enumerate(steps):
            step_key = f"{step.method_name}_{idx}"
            self.state.segmentation_status[step_key] = "Queued"
            self.segmentation_panel.update_step_status(idx, "Queued")
            
        self.segmentation_panel.set_running_state(True)
        
        sequence = SegmentationSequence(steps=steps)
        self.pipeline_service.execute_sequence(
            sequence=sequence,
            image=self.state.raw_image,
            on_step_complete=self._on_single_segmentation_done,
            on_sequence_complete=self._on_all_segmentations_complete,
            on_error=self._on_segmentation_error
        )

    def _on_segmentation_error(self, step_idx: int, method_name: str, error_msg: str):
        step_key = f"{method_name}_{step_idx}"
        self.state.segmentation_status[step_key] = "Failed"
        self.segmentation_panel.update_step_status(step_idx, "Failed")
        print(f"[Pipeline Error] {method_name} (Step {step_idx}): {error_msg}")

    def _on_single_segmentation_done(self, step_idx: int, method_name: str, seg_result: dict):
        step_key = f"{method_name}_{step_idx}"
        self.state.segmentation_results[step_key] = seg_result
        self.state.segmentation_status[step_key] = "Done"
        self.segmentation_panel.update_step_status(step_idx, "Done")
        
        scale_val = self.control_widget.spin_scale.value()
        
        def _add_mask_layer(mask_data, name_suffix):
            if mask_data is None: return
            layer_name = f"{method_name} (S{step_idx}) {name_suffix}"
            try:
                seg_layer = self.viewer.add_labels(
                    mask_data, name=layer_name, opacity=0.3,
                    visible=False, contour=1, scale=(scale_val, scale_val),
                )
            except TypeError:
                seg_layer = self.viewer.add_labels(
                    mask_data, name=layer_name, opacity=0.3,
                    visible=False, scale=(scale_val, scale_val),
                )
            if layer_name not in self.added_legend_entries:
                self.legend_widget.add_segmentation_row(seg_layer, layer_name, "#888888")
                self.added_legend_entries.add(layer_name)
                
        _add_mask_layer(seg_result.get('nuclear'), "Nuclear Mask")
        _add_mask_layer(seg_result.get('cytoplasm'), "Cyto Mask")

    def _on_all_segmentations_complete(self):
        self.segmentation_panel.set_running_state(False)
        masks = {m: res.get('nuclear') for m, res in self.state.segmentation_results.items() if res.get('nuclear') is not None}
        
        if not masks:
            QMessageBox.warning(self.control_widget, "No Results", "All segmentation methods failed. Check dependencies and parameters.")
            return
            
        n_methods = len(masks)
        self.control_widget.update_dashboard(seg_status=f"{n_methods} method(s) applied")
        self.segmentation_panel.lbl_consensus_status.setText(f"Ready for consensus: {n_methods} method(s)")
        
        QMessageBox.information(
            self.control_widget, "Segmentation Complete", 
            "Segmentation masks generated successfully.\n\nGo to the 'Consensus' tab to merge them into a unified confidence map."
        )

    def _generate_consensus(self):
        masks = {m: res.get('nuclear') for m, res in self.state.segmentation_results.items() if res.get('nuclear') is not None}
        
        if not masks:
            QMessageBox.warning(self.control_widget, "No Masks", "Run segmentation methods first before generating consensus.")
            return
            
        self.consensus_panel.set_generating_state(True)
        
        @thread_worker
        def _run_consensus(masks_dict=masks):
            return compute_consensus_mask(masks_dict)
            
        worker = _run_consensus(masks)
        worker.returned.connect(self._on_consensus_complete)
        worker.start()

    def _on_consensus_complete(self, result):
        self.consensus_panel.set_generating_state(False)
        consensus_mask, confidence_map, per_cell_conf, cell_method_map = result
        
        self.state.consensus_mask = consensus_mask
        self.state.confidence_map = confidence_map
        self.state.per_cell_confidence = per_cell_conf
        self.state._cell_method_map = cell_method_map
        scale_val = self.control_widget.spin_scale.value()
        
        if "Consensus Mask" in self.viewer.layers:
            self.viewer.layers["Consensus Mask"].data = consensus_mask
        else:
            try:
                consensus_layer = self.viewer.add_labels(consensus_mask, name="Consensus Mask", opacity=1.0, visible=True, contour=1, scale=(scale_val, scale_val))
            except TypeError:
                consensus_layer = self.viewer.add_labels(consensus_mask, name="Consensus Mask", opacity=1.0, visible=True, scale=(scale_val, scale_val))
            if "Consensus Mask" not in self.added_legend_entries:
                self.legend_widget.add_consensus_row(consensus_layer, "Consensus Mask", "#44AA66")
                self.added_legend_entries.add("Consensus Mask")
                
        conf_cmap = Colormap(
            colors=np.array([[0.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.2, 1.0], [0.8, 0.8, 0.2, 1.0], [0.2, 0.8, 0.2, 1.0]]),
            controls=np.array([0.0, 0.001, 0.5, 1.0]),
            name='confidence_ryg'
        )
        if "Confidence Map" in self.viewer.layers:
            self.viewer.layers["Confidence Map"].data = confidence_map
        else:
            conf_layer = self.viewer.add_image(confidence_map, name="Confidence Map", colormap=conf_cmap, blending='additive', visible=False, scale=(scale_val, scale_val), contrast_limits=[0, 1])
            if "Confidence Map" not in self.added_legend_entries:
                self.legend_widget.add_consensus_row(conf_layer, "Confidence Map", "#FFAA00")
                self.added_legend_entries.add("Confidence Map")
                
        self._update_confidence_stats()
        n_cells = len(per_cell_conf)
        n_methods = sum(1 for res in self.state.segmentation_results.values() if res.get('nuclear') is not None)
        
        self.control_widget.update_dashboard(total_cells=f"{n_cells:,}")
        self.segmentation_panel.lbl_consensus_status.setText(f"Consensus: {n_cells:,} cells from {n_methods} method(s)")
        
        self._update_confidence_overlay(self.consensus_panel.get_threshold())
        QMessageBox.information(self.control_widget, "Success", "Consensus Map and Confidence Contours generated successfully!")

    def _cancel_segmentation(self):
        self.pipeline_service.cancel_all()
        self.segmentation_panel.set_running_state(False)
        for key in list(self.state.segmentation_status.keys()):
            if self.state.segmentation_status[key] in ("Running", "Queued"):
                self.state.segmentation_status[key] = "Cancelled"
                
                # Extract step index from key 'Method_Idx'
                try:
                    idx = int(key.split('_')[-1])
                    self.segmentation_panel.update_step_status(idx, "Cancelled")
                except ValueError:
                    pass
                
        QMessageBox.information(
            self.control_widget, "Cancellation Sent", 
            "Segmentation cancellation requested. \n\nNote: Background deep-learning processes must complete their current tensor operation before the thread truly closes."
        )
        self.state.batch_queue.clear()
        self.control_widget.btn_batch.setText("Run Batch Selected")
        self.control_widget.btn_batch.setEnabled(True)

    def _update_confidence_overlay(self, threshold: float):
        if self.state.consensus_mask is None or self.state.per_cell_confidence is None:
            return

        h, w = self.state.consensus_mask.shape
        contour_rgb = np.zeros((h, w, 4), dtype=np.float32)
        half_thresh = threshold * 0.5
        
        from skimage.segmentation import find_boundaries
        
        # FIX: Utilize array vectorization and NumPy indexing rather than looping via native Python loops
        boundary_mask = find_boundaries(self.state.consensus_mask, mode='inner')
        max_id = int(self.state.consensus_mask.max())
        
        if max_id > 0:
            color_lookup = np.zeros((max_id + 1, 4), dtype=np.float32)
            high_color = mcolors.to_rgba(CONFIDENCE_HIGH_COLOR)
            med_color = mcolors.to_rgba(CONFIDENCE_MEDIUM_COLOR)
            low_color = mcolors.to_rgba(CONFIDENCE_LOW_COLOR)
            
            for cell_id, conf in self.state.per_cell_confidence.items():
                if cell_id <= max_id:
                    if conf >= threshold:
                        color_lookup[cell_id] = high_color
                    elif conf >= half_thresh:
                        color_lookup[cell_id] = med_color
                    else:
                        color_lookup[cell_id] = low_color
            
            cell_ids_at_boundaries = self.state.consensus_mask[boundary_mask]
            contour_rgb[boundary_mask] = color_lookup[cell_ids_at_boundaries]

        layer_name = "Confidence Contours"
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = contour_rgb
        else:
            scale_val = self.control_widget.spin_scale.value()
            contour_layer = self.viewer.add_image(
                contour_rgb, name=layer_name,
                rgb=True, blending='additive',
                visible=self.consensus_panel.chk_show_confidence.isChecked(),
                scale=(scale_val, scale_val),
            )
            if layer_name not in self.added_legend_entries:
                self.legend_widget.add_consensus_row(contour_layer, layer_name, "#FFFFFF")
                self.added_legend_entries.add(layer_name)
                
        self._update_confidence_stats()

    def _update_confidence_stats(self):
        if not self.state.per_cell_confidence: return
        
        threshold = self.consensus_panel.get_threshold()
        half_thresh = threshold * 0.5
        confs = list(self.state.per_cell_confidence.values())
        
        total = len(confs)
        high = sum(1 for c in confs if c >= threshold)
        low = sum(1 for c in confs if c < half_thresh)
        medium = total - high - low
        mean_conf = float(np.mean(confs)) if confs else 0.0
        
        self.consensus_panel.update_statistics(total, high, medium, low, mean_conf)
        self.control_widget.update_dashboard(confidence=f"{mean_conf:.3f} (mean)")

    def _toggle_consensus_visibility(self, visible: bool):
        if "Consensus Mask" in [l.name for l in self.viewer.layers]:
            self.viewer.layers["Consensus Mask"].visible = visible

    def _toggle_method_mask_visibility(self, method_name: str, visible: bool):
        for suffix in ["Nuclear Mask", "Cyto Mask"]:
            layer_name = f"{method_name} {suffix}"
            if layer_name in [l.name for l in self.viewer.layers]:
                self.viewer.layers[layer_name].visible = visible

    def _toggle_review_mode(self, enabled: bool):
        if not enabled: 
            self.consensus_panel.clear_cell_review()
            if "Active Review Isolation" in self.viewer.layers:
                self.viewer.layers["Active Review Isolation"].visible = False

    def _apply_manual_override(self, cell_id: int, preferred_method: str):
        if self.state.consensus_mask is None or not hasattr(self.state, '_cell_method_map'): return
        
        masks = {}
        for method_name, seg_result in self.state.segmentation_results.items():
            nuc = seg_result.get('nuclear')
            if nuc is not None: masks[method_name] = nuc
            
        self.state.consensus_mask, self.state.confidence_map, self.state.per_cell_confidence = \
            apply_manual_override(
                self.state.consensus_mask,
                self.state.confidence_map,
                self.state.per_cell_confidence,
                self.state._cell_method_map,
                cell_id,
                preferred_method,
                masks,
            )
            
        if "Consensus Mask" in [l.name for l in self.viewer.layers]:
            self.viewer.layers["Consensus Mask"].data = self.state.consensus_mask
            
        self._update_confidence_overlay(self.consensus_panel.get_threshold())
        self.consensus_panel.clear_cell_review()
        
        # Hide the isolation layer once a decision is made
        if "Active Review Isolation" in self.viewer.layers:
            self.viewer.layers["Active Review Isolation"].visible = False

    def _on_mouse_click(self, viewer, event):
        if not self.consensus_panel.chk_review_mode.isChecked(): return
        if self.state.consensus_mask is None: return
        
        consensus_layer = next((l for l in self.viewer.layers if l.name == "Consensus Mask"), None)
        if consensus_layer is None: return
        
        data_coord = consensus_layer.world_to_data(event.position)
        if len(data_coord) >= 2:
            y, x = int(data_coord[-2]), int(data_coord[-1])
            mask = self.state.consensus_mask
            
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                cell_id = int(mask[y, x])
                if cell_id > 0 and hasattr(self.state, '_cell_method_map'):
                    conf = self.state.per_cell_confidence.get(cell_id, 0.0)
                    method_map = self.state._cell_method_map.get(cell_id, {})
                    self.consensus_panel.show_cell_review(cell_id, conf, method_map)
                    
                    # Create an isolated, RGBA blending view of the competing masks to prevent "squishing"
                    h, w = self.state.consensus_mask.shape
                    isolation_rgba = np.zeros((h, w, 4), dtype=np.float32)
                    
                    # Same color order as the UI buttons (Red, Green, Blue, Magenta)
                    palette = [
                        (1.0, 0.2, 0.2), 
                        (0.2, 1.0, 0.2), 
                        (0.2, 0.5, 1.0), 
                        (1.0, 0.2, 1.0),
                    ]
                    
                    from skimage.segmentation import find_boundaries
                    
                    # Sort keys to match the UI button assignment identically
                    for i, m_name in enumerate(sorted(method_map.keys())):
                        orig_id = method_map[m_name]
                        if m_name in self.state.segmentation_results:
                            m_mask = self.state.segmentation_results[m_name].get('nuclear')
                            if m_mask is not None:
                                color = palette[i % len(palette)]
                                cell_pixels = (m_mask == orig_id)
                                
                                # Additive blending for translucent fills (30% opacity)
                                isolation_rgba[cell_pixels, 0] = np.clip(isolation_rgba[cell_pixels, 0] + color[0], 0, 1)
                                isolation_rgba[cell_pixels, 1] = np.clip(isolation_rgba[cell_pixels, 1] + color[1], 0, 1)
                                isolation_rgba[cell_pixels, 2] = np.clip(isolation_rgba[cell_pixels, 2] + color[2], 0, 1)
                                isolation_rgba[cell_pixels, 3] = np.maximum(isolation_rgba[cell_pixels, 3], 0.3)
                                
                                # Solid boundaries (80% opacity)
                                boundary = find_boundaries(cell_pixels, mode='inner')
                                isolation_rgba[boundary, 0] = color[0]
                                isolation_rgba[boundary, 1] = color[1]
                                isolation_rgba[boundary, 2] = color[2]
                                isolation_rgba[boundary, 3] = 0.8
                                
                    layer_name = "Active Review Isolation"
                    if layer_name in self.viewer.layers:
                        self.viewer.layers[layer_name].data = isolation_rgba
                        self.viewer.layers[layer_name].visible = True
                    else:
                        scale_val = self.control_widget.spin_scale.value()
                        # Added as an RGBA Image layer to avoid the contour TypeError and natively support alpha blending
                        iso_layer = self.viewer.add_image(
                            isolation_rgba, name=layer_name, rgb=True, opacity=1.0, 
                            blending='translucent', visible=True, scale=(scale_val, scale_val)
                        )
                        if layer_name not in self.added_legend_entries:
                            self.legend_widget.add_consensus_row(iso_layer, layer_name, "#FFFFFF")
                            self.added_legend_entries.add(layer_name)

    def _run_feature_extraction(self, selected_morphology: list, selected_intensity: list):
        if self.state.consensus_mask is None:
            QMessageBox.warning(self.control_widget, "No Mask", "Run segmentation or load a mask before extracting features.")
            return
        if self.state.raw_image is None:
            QMessageBox.warning(self.control_widget, "No Image", "Load an image before extracting features.")
            return
            
        self.feature_panel.set_extracting_state(True)
        scale_val = self.control_widget.spin_scale.value()
        
        @thread_worker
        def _extract():
            return extract_cell_features(
                mask=self.state.consensus_mask,
                raw_image=self.state.raw_image,
                markers=self.state.markers,
                scale_factor=scale_val,
                selected_morphology=selected_morphology,
                selected_intensity=selected_intensity,
            )
            
        worker = _extract()
        worker.returned.connect(self._on_features_extracted)
        worker.start()

    def _on_features_extracted(self, df):
        self.state.features_df = df
        self.feature_panel.set_extracting_state(False)
        self.feature_panel.populate_preview(df)

    def _export_mask(self):
        if self.state.consensus_mask is None:
            QMessageBox.warning(self.control_widget, "No Mask", "No consensus mask to export.")
            return
            
        sample = self.state.current_sample_name or "sample"
        default_name = f"{sample}_mask.tiff"
        filepath, _ = QFileDialog.getSaveFileName(self.control_widget, "Export Consensus Mask", default_name, "TIFF (*.tiff)")
        
        if filepath:
            cyto = None
            for seg_result in self.state.segmentation_results.values():
                if seg_result.get('cytoplasm') is not None:
                    cyto = seg_result['cytoplasm']
                    break
            export_consensus_mask(Path(filepath), self.state.consensus_mask, cyto)
            QMessageBox.information(self.control_widget, "Export Complete", "Consensus mask saved.")

    def _export_features(self):
        if self.state.features_df is None or self.state.features_df.empty:
            QMessageBox.warning(self.control_widget, "No Features", "Extract features first.")
            return
            
        sample = self.state.current_sample_name or "sample"
        default_name = f"{sample}_features.csv"
        filepath, _ = QFileDialog.getSaveFileName(self.control_widget, "Export Features CSV", default_name, "CSV (*.csv)")
        
        if filepath:
            export_features_csv(Path(filepath), self.state.features_df, sample)
            QMessageBox.information(self.control_widget, "Export Complete", "Features CSV saved.")

    def _export_confidence(self):
        if self.state.confidence_map is None:
            QMessageBox.warning(self.control_widget, "No Confidence Map", "Run segmentation first.")
            return
            
        sample = self.state.current_sample_name or "sample"
        default_name = f"{sample}_confidence.tiff"
        filepath, _ = QFileDialog.getSaveFileName(self.control_widget, "Export Confidence Map", default_name, "TIFF (*.tiff)")
        
        if filepath:
            export_confidence_map(Path(filepath), self.state.confidence_map)
            QMessageBox.information(self.control_widget, "Export Complete", "Confidence map saved.")

    def _run_batch_processing(self):
        if not self.state.file_map:
            QMessageBox.warning(self.control_widget, "No Directory", "Load an image directory first.")
            return
            
        selected_methods = [name for name, chk in self.segmentation_panel.method_checkboxes.items() if chk.isChecked()]
        if not selected_methods:
            QMessageBox.warning(self.control_widget, "No Methods", "Select at least one segmentation method.")
            return
            
        sample_names = sorted(self.state.file_map.keys())
        self.state.batch_queue = sample_names.copy()
        self.state.batch_results.clear()
        self.control_widget.btn_batch.setText(f"Processing 0/{len(sample_names)}...")
        self.control_widget.btn_batch.setEnabled(False)
        self._batch_process_next()

    def _batch_process_next(self):
        if not self.state.batch_queue:
            self._on_batch_complete()
            return
            
        sample_name = self.state.batch_queue.pop(0)
        img_path = self.state.file_map[sample_name]
        
        ui_state = {
            'nuc_name': self.segmentation_panel.combo_cp_nuc.currentText(),
            'mem_name': self.segmentation_panel.combo_cp_mem.currentText(),
            'scale_val': self.control_widget.spin_scale.value(),
            'methods_checked': {n: c.isChecked() for n, c in self.segmentation_panel.method_checkboxes.items()},
        }
        
        @thread_worker(progress={'total': 0, 'desc': f"Batch: {sample_name}"})
        def _batch_single():
            raw_image, _ = load_multichannel_tiff(img_path)
            nuc_name = ui_state['nuc_name']
            mem_name = ui_state['mem_name']
            nuc_idx = self.state.markers.index(nuc_name) if nuc_name in self.state.markers else 0
            mem_idx = self.state.markers.index(mem_name) if mem_name in self.state.markers and mem_name != 'None' else None
            
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
            except ImportError:
                device = "cpu"
                
            expand_radius = 0
            
            masks = {}
            batch_seg_results = {}
            methods_used = []
            
            for method_name in [n for n, c in self.segmentation_panel.method_checkboxes.items() if c.isChecked()]:
                backend = next((b for b in self.backends if b.name == method_name and b.is_available()), None)
                if backend is None: continue
                
                method_kwargs = {}
                if method_name == "Cellpose":
                    method_kwargs = {
                        'model_type': self.segmentation_panel.combo_cp_model.currentText(),
                        'flow_threshold': self.segmentation_panel.spin_cp_flow.value(),
                        'cellprob_threshold': self.segmentation_panel.spin_cp_cellprob.value(),
                    }
                    diameter = self.segmentation_panel.spin_cp_diameter.value()
                    diameter = diameter if diameter > 0 else None
                elif method_name == "StarDist":
                    method_kwargs = {
                        'model_name': self.segmentation_panel.combo_sd_model.currentText(),
                        'prob_thresh': self.segmentation_panel.spin_sd_prob.value(),
                        'nms_thresh': self.segmentation_panel.spin_sd_nms.value(),
                    }
                    diameter = None
                elif method_name == "InstanSeg":
                    method_kwargs = {
                        'model_name': self.segmentation_panel.combo_is_model.currentText(),
                        'pixel_size': self.segmentation_panel.spin_is_pixelsize.value(),
                    }
                    diameter = None
                else:
                    diameter = None
                    
                try:
                    result = backend.segment(
                        image=raw_image, nuclear_channel=nuc_idx,
                        membrane_channel=mem_idx, diameter=diameter,
                        device=device, expand_radius=expand_radius,
                        **method_kwargs,
                    )
                    nuc = result.get('nuclear')
                    if nuc is not None:
                        masks[method_name] = nuc
                        batch_seg_results[method_name] = result
                        methods_used.append(method_name)
                except Exception as e:
                    print(f"[IRIS-Viewer] [Batch] {method_name} failed on {sample_name}: {e}")
                    
            if not masks:
                return sample_name, None
                
            consensus_mask, confidence_map, per_cell_conf, _ = compute_consensus_mask(masks)
            
            scale_val = self.control_widget.spin_scale.value()
            features_df = extract_cell_features(
                mask=consensus_mask,
                raw_image=raw_image,
                markers=self.state.markers,
                scale_factor=scale_val,
            )
            
            out_dir = img_path.parent
            mask_out = out_dir / f"{sample_name}_consensus_mask.tiff"
            feat_out = out_dir / f"{sample_name}_features.csv"
            
            # FIX: Ensure cytoplasm arrays are retrieved from the backend results and exported correctly
            cyto = None
            for mn in methods_used:
                if batch_seg_results[mn].get('cytoplasm') is not None:
                    cyto = batch_seg_results[mn]['cytoplasm']
                    break
                    
            export_consensus_mask(mask_out, consensus_mask, cyto)
            export_features_csv(feat_out, features_df, sample_name)
            
            confs = list(per_cell_conf.values())
            threshold = 0.66
            return sample_name, {
                'total_cells': len(per_cell_conf),
                'high_conf': sum(1 for c in confs if c >= threshold),
                'med_conf': sum(1 for c in confs if threshold > c >= threshold * 0.5),
                'low_conf': sum(1 for c in confs if c < threshold * 0.5),
                'mean_confidence': float(np.mean(confs)) if confs else 0.0,
                'methods': methods_used,
            }
            
        worker = _batch_single()
        worker.returned.connect(self._on_batch_sample_done)
        
        if not hasattr(self, '_active_workers'):
            self._active_workers = []
        self._active_workers.append(worker)
        
        worker.start()

    def _on_batch_sample_done(self, result):
        sample_name, stats = result
        if stats is not None:
            self.state.batch_results[sample_name] = stats
        else:
            self.state.batch_results[sample_name] = {
                'total_cells': 0, 'high_conf': 0, 'med_conf': 0,
                'low_conf': 0, 'mean_confidence': 0.0, 'methods': [],
            }
        self._batch_process_next()

    def _on_batch_complete(self):
        self.control_widget.btn_batch.setText("Batch Process All Samples")
        self.control_widget.btn_batch.setEnabled(True)
        if self.state.batch_results and self.state.image_dir:
            summary_path = self.state.image_dir / "batch_summary.csv"
            export_batch_summary(summary_path, self.state.batch_results)
            
        n_samples = len(self.state.batch_results)
        n_success = sum(1 for s in self.state.batch_results.values() if s['total_cells'] > 0)
        
        QMessageBox.information(
            self.control_widget, "Batch Complete",
            f"Processed {n_samples} samples ({n_success} successful).\n"
            f"Results saved alongside source images.\n"
            f"Summary: {self.state.image_dir / 'batch_summary.csv'}"
        )