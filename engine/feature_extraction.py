""" IRIS-Viewer Feature Extraction Engine """
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from typing import List, Optional, Dict

MORPHOLOGY_FEATURES = {
    'area':               'Area (px)',
    'perimeter':          'Perimeter (px)',
    'eccentricity':       'Eccentricity',
    'solidity':           'Solidity',
    'major_axis_length':  'Major Axis Length',
    'minor_axis_length':  'Minor Axis Length',
    'orientation':        'Orientation (rad)',
    'extent':             'Extent',
    'equivalent_diameter_area': 'Equivalent Diameter',
}

INTENSITY_FEATURES = ['mean', 'median', 'min', 'max', 'sum']

# FIX: Implement memory-efficient extra_properties to replace native Python regionprops loops
def custom_sum_intensity(mask, intensity):
    return np.sum(intensity[mask])

def custom_median_intensity(mask, intensity):
    return np.median(intensity[mask]) if np.any(mask) else 0.0

def extract_cell_features(
    mask: np.ndarray,
    raw_image: np.ndarray,
    markers: List[str],
    scale_factor: float = 1.0,
    selected_morphology: Optional[List[str]] = None,
    selected_intensity: Optional[List[str]] = None,
) -> pd.DataFrame:
    
    if mask is None or np.max(mask) == 0:
        return pd.DataFrame()

    if selected_morphology is None:
        morph_keys = list(MORPHOLOGY_FEATURES.keys())
    else:
        morph_keys = [k for k in selected_morphology if k in MORPHOLOGY_FEATURES]
        
    if selected_intensity is None:
        intensity_stats = INTENSITY_FEATURES.copy()
    else:
        intensity_stats = [s for s in selected_intensity if s in INTENSITY_FEATURES]

    # Step 1: Morphological features via regionprops table
    base_props = ['label', 'centroid']
    regionprops_keys = base_props + morph_keys
    
    try:
        props_table = regionprops_table(mask, properties=regionprops_keys)
    except Exception:
        props_table = regionprops_table(mask, properties=base_props)
        
    df = pd.DataFrame(props_table)
    
    rename_map = {
        'label': 'ObjectNumber',
        'centroid-1': 'Pos_X',
        'centroid-0': 'Pos_Y',
    }
    df.rename(columns=rename_map, inplace=True)

    if scale_factor != 1.0:
        if 'area' in df.columns:
            df['area'] = df['area'] * (scale_factor ** 2)
        if 'perimeter' in df.columns:
            df['perimeter'] = df['perimeter'] * scale_factor
        if 'major_axis_length' in df.columns:
            df['major_axis_length'] = df['major_axis_length'] * scale_factor
        if 'minor_axis_length' in df.columns:
            df['minor_axis_length'] = df['minor_axis_length'] * scale_factor
        if 'equivalent_diameter_area' in df.columns:
            df['equivalent_diameter_area'] = df['equivalent_diameter_area'] * scale_factor

    # Step 2: Per-channel intensity features
    if raw_image is not None and len(markers) > 0 and len(intensity_stats) > 0:
        n_channels = min(raw_image.shape[0], len(markers))
        
        intensity_prop_map = {
            'mean': 'intensity_mean',
            'max': 'intensity_max',
            'min': 'intensity_min',
        }
        
        for ch_idx in range(n_channels):
            marker_name = markers[ch_idx]
            ch_data = raw_image[ch_idx]
            
            skimage_props = [intensity_prop_map[s] for s in intensity_stats if s in intensity_prop_map]
            
            # Use the robust callbacks for median/sum calculation
            extra_props = []
            if 'sum' in intensity_stats: extra_props.append(custom_sum_intensity)
            if 'median' in intensity_stats: extra_props.append(custom_median_intensity)
            
            if skimage_props or extra_props:
                ch_table = regionprops_table(
                    mask, 
                    intensity_image=ch_data, 
                    properties=skimage_props,
                    extra_properties=extra_props
                )
                
                for req_stat, ski_key in intensity_prop_map.items():
                    if req_stat in intensity_stats:
                        df[f"{marker_name}_{req_stat}"] = ch_table[ski_key]
                        
                if 'sum' in intensity_stats:
                    df[f"{marker_name}_sum"] = ch_table['custom_sum_intensity']
                if 'median' in intensity_stats:
                    df[f"{marker_name}_median"] = ch_table['custom_median_intensity']

    return df