"""
IRIS-Common Image Loaders
Multi-channel TIFF / OME-TIFF loader shared between Viewer and Phenotyper.

Key additions over the original viewer version:
- probe_ome_channel_names() — reads OME-XML Channel Name attributes to
  auto-populate marker panels without manual typing.
- Zarr / OME-Zarr stub (load_zarr_multichannel) — Phase 5 hook.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import numpy as np
import tifffile

# Discovery

def discover_image_files(directory: Path) -> Dict[str, Path]:
    """
    Discover all multiplexed image TIFFs in a directory.

    Naming priority:
      1. ``*_image.tiff``  (IRIS convention)
      2. Any other ``.tiff`` / ``.tif`` file

    Returns
    -------
    dict mapping ``sample_name → file_path``.
    """
    file_map: Dict[str, Path] = {}
    if not directory.is_dir():
        return file_map

    for f in sorted(directory.iterdir()):
        if f.is_file() and f.name.endswith('_image.tiff'):
            file_map[f.name.replace('_image.tiff', '')] = f

    if not file_map:
        for f in sorted(directory.iterdir()):
            if f.is_file() and f.suffix.lower() in ('.tiff', '.tif'):
                file_map[f.stem] = f

    return file_map


# Multi-channel TIFF Loading

def load_multichannel_tiff(filepath: Path) -> Tuple[Any, int]:
    """
    Load a multi-channel TIFF image lazily using Dask.

    Handles:
    - ``(C, H, W)`` — standard multi-channel TIFF
    - ``(H, W, C)`` — RGB/RGBA-style; transposed to ``(C, H, W)``
    - ``(Z, C, H, W)`` / ``(T, C, H, W)`` — first frame taken
    - ``(H, W)`` — single-channel; wrapped as ``(1, H, W)``

    Returns
    -------
    image : dask.array.Array of shape ``(C, H, W)``
    num_channels : int
    """
    store = tifffile.imread(str(filepath), aszarr=True)
    raw = da.from_zarr(store)

    if raw.ndim == 2:
        return raw[np.newaxis, ...], 1
    elif raw.ndim == 3:
        if raw.shape[-1] in (3, 4) and raw.shape[0] > 4:
            return da.transpose(raw, (2, 0, 1)), raw.shape[-1]
        return raw, raw.shape[0]
    elif raw.ndim == 4:
        return raw[0], raw[0].shape[0]
    else:
        raise ValueError(f"Unexpected TIFF shape: {raw.shape}. "
                         "Expected 2-D, 3-D, or 4-D array.")


def load_existing_mask(filepath: Path) -> Optional[np.ndarray]:
    """
    Load an existing segmentation mask TIFF.

    Returns
    -------
    np.ndarray of dtype ``uint32``:
      - ``(H, W)`` for single-channel masks
      - ``(C, H, W)`` for multi-channel masks (e.g. nuclear + cytoplasm)
    None if the file does not exist.
    """
    if not filepath.exists():
        return None
    return tifffile.imread(str(filepath)).astype(np.uint32)


# OME-TIFF Metadata

def probe_channel_count(filepath: Path) -> int:
    """
    Read TIFF metadata to determine the number of channels.
    Tries OME-XML → axis tags → dimension heuristics in order.
    """
    try:
        with tifffile.TiffFile(str(filepath)) as tif:
            if tif.is_ome:
                try:
                    root = ET.fromstring(tif.ome_metadata)
                    pixels = root.find('.//{*}Pixels')
                    if pixels is not None and 'SizeC' in pixels.attrib:
                        return int(pixels.attrib['SizeC'])
                except Exception:
                    pass

            series = tif.series[0]
            if 'C' in series.axes:
                return series.shape[series.axes.index('C')]

            if len(series.shape) >= 3:
                return series.shape[0]
            return 1
    except Exception:
        return 1


def probe_ome_channel_names(filepath: Path) -> List[str]:
    """
    Extract ``Channel Name`` attributes from OME-XML metadata.

    Returns an ordered list of marker names matching channel order.
    Returns an empty list if the file is not OME-TIFF or has no channel names.

    Example OME-XML snippet parsed::

        <Channel ID="Channel:0:0" Name="DAPI" .../>
        <Channel ID="Channel:0:1" Name="CD45" .../>
    """
    try:
        with tifffile.TiffFile(str(filepath)) as tif:
            if not tif.is_ome:
                return []
            root = ET.fromstring(tif.ome_metadata)
            channels = root.findall('.//{*}Channel')
            names = []
            for ch in channels:
                name = ch.attrib.get('Name') or ch.attrib.get('Fluor', '')
                names.append(name.strip())
            # Filter out empty names but preserve order
            return [n for n in names if n]
    except Exception:
        return []


# Zarr / OME-Zarr (Phase 5 hook)

def load_zarr_multichannel(path: Path) -> Tuple[Any, int]:
    """
    Load an OME-Zarr / NGFF dataset.

    .. note::
       Phase 5 placeholder.  Raises ``NotImplementedError`` until
       pyramid TIFF / OME-Zarr support is fully implemented.
    """
    raise NotImplementedError(
        "OME-Zarr loading is planned for Phase 5. "
        "Use load_multichannel_tiff() for standard TIFF files."
    )
