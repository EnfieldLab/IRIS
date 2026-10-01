""" Cellpose Segmentation Backend """
import numpy as np
from typing import Dict, Optional, Any
from .base import SegmentationBackend
import warnings

warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")
warnings.filterwarnings("ignore", message=".*channels argument is deprecated.*")
warnings.filterwarnings("ignore", message=".*model_type argument is not used.*")

class CellposeBackend(SegmentationBackend):
    name = "Cellpose"
    supports_cytoplasm = True

    def __init__(self):
        self._cached_model_cyto = None
        self._cached_model_nuc = None
        self._cached_device = None
        self._cached_cyto_type = None

    def is_available(self) -> bool:
        try:
            import cellpose
            return True
        except ImportError:
            return False

    def get_default_params(self) -> Dict[str, Any]:
        return {'model_type': 'cyto3', 'flow_threshold': 0.4, 'cellprob_threshold': 0.0}

    def segment(self, image: np.ndarray, nuclear_channel: int, membrane_channel: Optional[int], diameter: Optional[float], device: str, expand_radius: int = 0, **kwargs) -> Dict[str, Optional[np.ndarray]]:
        from cellpose import models
        import cellpose
        import torch
        import gc
        import inspect

        model_type = kwargs.get('model_type', 'cyto3')
        flow_threshold = kwargs.get('flow_threshold', 0.4)
        cellprob_threshold = kwargs.get('cellprob_threshold', 0.0)

        if device == 'mps':
            torch_device = torch.device('mps')
            use_gpu = True
        elif device == 'cuda':
            torch_device = torch.device('cuda')
            use_gpu = True
        else:
            torch_device = torch.device('cpu')
            use_gpu = False

        sig = inspect.signature(models.CellposeModel.eval)
        is_v4 = 'channel_axis' in sig.parameters
        run_two_pass = (membrane_channel is not None and membrane_channel != nuclear_channel)

        # Cache Cyto Model 
        if self._cached_model_cyto is None or self._cached_cyto_type != model_type or self._cached_device != device:
            if is_v4:
                self._cached_model_cyto = models.CellposeModel(gpu=use_gpu, device=torch_device, pretrained_model=model_type)
            else:
                self._cached_model_cyto = models.CellposeModel(gpu=use_gpu, device=torch_device, model_type=model_type)
            self._cached_cyto_type = model_type
            self._cached_device = device

        try:
            if membrane_channel is not None and membrane_channel != nuclear_channel:
                mem_img = np.asarray(image[membrane_channel])
                nuc_img = np.asarray(image[nuclear_channel])
                input_img = np.stack([mem_img, nuc_img], axis=0 if is_v4 else -1)
                
                kwargs_eval = {'diameter': diameter, 'flow_threshold': flow_threshold, 'cellprob_threshold': cellprob_threshold, 'batch_size': 4, 'tile_overlap': 0.1}
                if is_v4: kwargs_eval['channel_axis'] = 0
                else: kwargs_eval['channels'] = [1, 2] # 1=membrane, 2=nuclear
            else:
                input_img = np.asarray(image[nuclear_channel] if image.ndim == 3 else image)
                kwargs_eval = {'diameter': diameter, 'flow_threshold': flow_threshold, 'cellprob_threshold': cellprob_threshold, 'batch_size': 4, 'tile_overlap': 0.1}
                if not is_v4: kwargs_eval['channels'] = [0, 0]

            print(f"[IRIS-Viewer] Cellpose Execution: Running '{model_type}'...")
            res = self._cached_model_cyto.eval(input_img, **kwargs_eval)
            masks = res[0]
            
            result = {}
            if model_type == 'nuclei':
                result['nuclear'] = masks.astype(np.uint32)
                result['cytoplasm'] = None
            else:
                # If a cyto model is run, it segments the whole cell. We assign it to both.
                result['nuclear'] = masks.astype(np.uint32)
                result['cytoplasm'] = masks.astype(np.uint32)

        finally:
            if use_gpu:
                if device == 'mps': torch.mps.empty_cache()
                else: torch.cuda.empty_cache()
            gc.collect()

        return result