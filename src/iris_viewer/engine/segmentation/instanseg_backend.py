""" InstanSeg Segmentation Backend """
import numpy as np
from typing import Dict, Optional, Any
from .base import SegmentationBackend

class InstanSegBackend(SegmentationBackend):
    name = "InstanSeg"
    supports_cytoplasm = True

    def __init__(self):
        self._cached_model = None
        self._cached_model_name = None
        self._cached_device = None

    def is_available(self) -> bool:
        try:
            import instanseg
            from instanseg import InstanSeg
            return True
        except ImportError:
            return False

    def get_default_params(self) -> Dict[str, Any]:
        return {'model_name': 'fluorescence_nuclei_and_cells', 'pixel_size': 0.5}

    def segment(self, image: np.ndarray, nuclear_channel: int, membrane_channel: Optional[int], diameter: Optional[float], device: str, expand_radius: int = 0, **kwargs) -> Dict[str, Optional[np.ndarray]]:
        from instanseg import InstanSeg
        import torch
        import gc
        
        model_name = kwargs.get('model_name', 'fluorescence_nuclei_and_cells')
        pixel_size = kwargs.get('pixel_size', 0.5)
        if pixel_size <= 0:
            pixel_size = None

        if device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            device = 'cpu'

        if self._cached_model is None or self._cached_model_name != model_name or self._cached_device != device:
            self._cached_model = InstanSeg(model_name, device=device, verbosity=0)
            self._cached_model_name = model_name
            self._cached_device = device

        if image.ndim == 3 and image.shape[0] > 1:
            if membrane_channel is not None and membrane_channel != nuclear_channel:
                input_img = np.stack([image[nuclear_channel], image[membrane_channel]], axis=0).astype(np.float32)
            else:
                input_img = image[nuclear_channel:nuclear_channel + 1].astype(np.float32)
        else:
            input_img = (image[np.newaxis] if image.ndim == 2 else image).astype(np.float32)

        # FIX: Implement per-channel percentile normalization to prevent model hallucination on empty spaces
        normalized_img = np.zeros_like(input_img, dtype=np.float32)
        for c in range(input_img.shape[0]):
            ch = input_img[c]
            p_low, p_high = np.percentile(ch, [1, 99.9])
            if p_high <= p_low:
                p_high = p_low + 1.0
            normalized_img[c] = np.clip((ch - p_low) / (p_high - p_low), 0, 1)
        
        input_tensor = torch.from_numpy(normalized_img).unsqueeze(0)

        original_quantile = torch.quantile
        def safe_quantile(input, q, dim=None, keepdim=False, *, interpolation='linear', out=None):
            if input.dtype == torch.float32 and input.numel() > 16000000:
                cpu_tensor = input.cpu().to(torch.float64)
                q_cast = q.to(device=cpu_tensor.device, dtype=torch.float64) if isinstance(q, torch.Tensor) else float(q)
                res = original_quantile(cpu_tensor, q_cast, dim=dim, keepdim=keepdim, interpolation=interpolation, out=out)
                return res.to(torch.float32).to(input.device)
            return original_quantile(input, q, dim=dim, keepdim=keepdim, interpolation=interpolation, out=out)
        
        torch.quantile = safe_quantile

        try:
            eval_result = self._cached_model.eval_medium_image(
                input_tensor, 
                pixel_size=pixel_size, 
                resolve_cell_and_nucleus=True, 
                cleanup_fragments=True
            )
            labeled_output = eval_result[0]
        finally:
            torch.quantile = original_quantile
            if device == 'mps': torch.mps.empty_cache()
            elif device == 'cuda': torch.cuda.empty_cache()
            gc.collect()

        if isinstance(labeled_output, torch.Tensor):
            labeled_output = labeled_output.cpu().numpy()
            
        if labeled_output.ndim == 4:
            labeled_output = labeled_output[0]

        orig_h, orig_w = image.shape[-2:]
        mask_h, mask_w = labeled_output.shape[-2:]

        # FIX: Replace destructive tensor interpolation with mathematically safe zero-padding and cropping 
        if (mask_h, mask_w) != (orig_h, orig_w):
            if mask_h > orig_h: labeled_output = labeled_output[..., :orig_h, :]
            if mask_w > orig_w: labeled_output = labeled_output[..., :, :orig_w]
            
            pad_h = max(0, orig_h - labeled_output.shape[-2])
            pad_w = max(0, orig_w - labeled_output.shape[-1])
            
            if pad_h > 0 or pad_w > 0:
                pad_width = [(0, 0)] * labeled_output.ndim
                pad_width[-2] = (0, pad_h)
                pad_width[-1] = (0, pad_w)
                labeled_output = np.pad(labeled_output, pad_width, mode='constant', constant_values=0)

        if labeled_output.shape[0] >= 2:
            nuclear_mask = labeled_output[0].astype(np.uint32)
            cyto_mask = labeled_output[1].astype(np.uint32)
        else:
            nuclear_mask = labeled_output[0].astype(np.uint32)
            cyto_mask = self._expand_nuclear_to_cytoplasm(nuclear_mask, expand_radius) if expand_radius > 0 else None

        return {'nuclear': nuclear_mask, 'cytoplasm': cyto_mask}