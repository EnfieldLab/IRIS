"""
IRIS-Viewer Image Loaders
Multi-channel TIFF/OME-TIFF loader with automatic shape detection
and marker channel assignment via the panel configuration dialog.
"""
import tifffile
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def discover_image_files(directory: Path) -> Dict[str, Path]:
    """
    Discover all *_image.tiff files in a directory.
    Returns mapping of sample_name -> file_path.
    """
    file_map = {}
    if not directory.is_dir():
        return file_map

    for f in sorted(directory.iterdir()):
        if f.is_file() and f.name.endswith('_image.tiff'):
            sample_name = f.name.replace('_image.tiff', '')
            file_map[sample_name] = f

    # Also check for standalone .tiff files that don't follow the convention
    if not file_map:
        for f in sorted(directory.iterdir()):
            if f.is_file() and f.suffix.lower() in ('.tiff', '.tif'):
                sample_name = f.stem
                file_map[sample_name] = f

    return file_map


def load_multichannel_tiff(filepath: Path) -> Tuple[np.ndarray, int]:
    """
    Load a multi-channel TIFF image.
    
    Returns
    -------
    image : np.ndarray
        Array of shape (C, H, W) for multi-channel, or (1, H, W) for single-channel.
    num_channels : int
        Number of channels detected.
    """
    raw = tifffile.imread(str(filepath))

    if raw.ndim == 2:
        # Single 2D image -> (1, H, W)
        return raw[np.newaxis, ...], 1
    elif raw.ndim == 3:
        # Could be (C, H, W) or (H, W, 3/4) for RGB/RGBA
        if raw.shape[-1] in (3, 4) and raw.shape[0] > 4:
            # Likely (H, W, C) -> transpose to (C, H, W)
            return np.transpose(raw, (2, 0, 1)), raw.shape[-1]
        else:
            # Likely (C, H, W) already
            return raw, raw.shape[0]
    elif raw.ndim == 4:
        # (Z, C, H, W) or (T, C, H, W) -> take first frame/slice
        return raw[0], raw[0].shape[0]
    else:
        raise ValueError(f"Unexpected TIFF shape: {raw.shape}")


def load_existing_mask(filepath: Path) -> Optional[np.ndarray]:
    """
    Load an existing segmentation mask TIFF.
    Returns (C, H, W) for multi-channel masks or (H, W) for single channel.
    """
    if not filepath.exists():
        return None
    mask = tifffile.imread(str(filepath))
    return mask.astype(np.uint32)

def probe_channel_count(filepath: Path) -> int:
    """Read TIFF metadata safely utilizing OME-XML or axis tags."""
    try:
        with tifffile.TiffFile(str(filepath)) as tif:
            # Check for OME-XML metadata first
            if tif.is_ome:
                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(tif.ome_metadata)
                    pixels_elem = root.find('.//{*}Pixels')
                    if pixels_elem is not None and 'SizeC' in pixels_elem.attrib:
                        return int(pixels_elem.attrib['SizeC'])
                except Exception:
                    pass
            
            # Fallback to axis strings if provided
            series = tif.series[0]
            if 'C' in series.axes:
                return series.shape[series.axes.index('C')]
                
            # Fallback to basic dimension heuristics
            if len(series.shape) >= 3:
                return series.shape[0]
            return 1
    except Exception:
        return 1