"""
IRIS-Viewer Exporters
Handles exporting consensus masks and extracted feature tables to disk,
following the IRIS convention: *_mask.tiff + *_features.csv
"""
import tifffile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def export_consensus_mask(
    filepath: Path,
    nuclear_mask: np.ndarray,
    cytoplasm_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Export the consensus mask as a multi-channel TIFF.
    Channel 0: nuclear mask
    Channel 1: cytoplasm mask (if available)
    
    This follows IRIS convention where *_mask.tiff contains stacked label arrays.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if cytoplasm_mask is not None:
        if nuclear_mask.shape != cytoplasm_mask.shape:
            raise ValueError(
                f"Export failed: Mask shape mismatch. "
                f"Nuclear {nuclear_mask.shape} vs Cytoplasm {cytoplasm_mask.shape}."
            )
        output = np.stack([nuclear_mask, cytoplasm_mask], axis=0).astype(np.uint32)
    else:
        output = nuclear_mask.astype(np.uint32)
        
    tifffile.imwrite(str(filepath), output, compression='zlib')

def export_features_csv(
    filepath: Path,
    df: pd.DataFrame,
    sample_name: str = "",
) -> None:
    """
    Export the extracted features DataFrame to CSV.
    Adds an 'Image' column with the sample name if not already present.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    export_df = df.copy()

    if 'Image' not in export_df.columns and sample_name:
        export_df.insert(0, 'Image', sample_name)

    export_df.to_csv(str(filepath), index=False)


def export_confidence_map(
    filepath: Path,
    confidence_map: np.ndarray,
) -> None:
    """
    Export the per-pixel confidence map as a float32 TIFF for downstream analysis.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(filepath), confidence_map.astype(np.float32), compression='zlib')


def export_batch_summary(
    filepath: Path,
    batch_results: dict,
) -> None:
    """
    Export a summary CSV for batch processing results.
    One row per sample with cell count, method agreement statistics.
    """
    rows = []
    for sample_name, results in batch_results.items():
        row = {
            'Sample': sample_name,
            'Total_Cells': results.get('total_cells', 0),
            'High_Confidence_Cells': results.get('high_conf', 0),
            'Medium_Confidence_Cells': results.get('med_conf', 0),
            'Low_Confidence_Cells': results.get('low_conf', 0),
            'Mean_Confidence': results.get('mean_confidence', 0.0),
            'Methods_Used': ', '.join(results.get('methods', [])),
        }
        rows.append(row)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(str(filepath), index=False)
