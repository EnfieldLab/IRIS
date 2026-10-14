"""
IRIS-Common Exporters
Shared export functions for both Viewer and Phenotyper outputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile


def export_consensus_mask(
    filepath: Path,
    nuclear_mask: np.ndarray,
    cytoplasm_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Export the consensus mask in multiple formats (TIFF, OME-TIFF, PNG, NumPy NPY/NPZ).

    Channel 0: nuclear label mask (required)
    Channel 1: cytoplasm label mask (optional)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if cytoplasm_mask is not None:
        if nuclear_mask.shape != cytoplasm_mask.shape:
            raise ValueError(
                f"Mask shape mismatch: nuclear {nuclear_mask.shape} "
                f"vs cytoplasm {cytoplasm_mask.shape}."
            )
        output = np.stack([nuclear_mask, cytoplasm_mask], axis=0).astype(np.uint32)
    else:
        output = nuclear_mask.astype(np.uint32)

    suffix = filepath.suffix.lower()
    lower_name = filepath.name.lower()

    if suffix == '.npy':
        np.save(str(filepath), output)
    elif suffix == '.npz':
        np.savez_compressed(str(filepath), mask=output)
    elif suffix == '.png':
        try:
            from PIL import Image
            img_arr = output[0] if output.ndim == 3 else output
            if img_arr.max() > 255:
                # 16-bit PNG for high cell counts
                img = Image.fromarray(img_arr.astype(np.uint16))
            else:
                img = Image.fromarray(img_arr.astype(np.uint8))
            img.save(str(filepath))
        except Exception:
            tifffile.imwrite(str(filepath), output)
    elif '.ome.tif' in lower_name:
        tifffile.imwrite(str(filepath), output, ome=True, compression='zlib')
    else:
        tifffile.imwrite(str(filepath), output, compression='zlib')


def export_features_csv(
    filepath: Path,
    df: pd.DataFrame,
    sample_name: str = "",
) -> None:
    """
    Export a feature DataFrame in multiple formats (CSV, Excel XLSX, Parquet, TSV, JSON).
    Inserts an ``Image`` column with *sample_name* if not already present.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if 'Image' not in out.columns and sample_name:
        out.insert(0, 'Image', sample_name)

    suffix = filepath.suffix.lower()
    if suffix == '.xlsx':
        try:
            out.to_excel(str(filepath), index=False)
        except Exception as e:
            # Fallback to CSV if openpyxl is not installed
            fallback = filepath.with_suffix('.csv')
            out.to_csv(str(fallback), index=False)
            raise RuntimeError(f"Could not write Excel file ({e}). Saved as CSV: {fallback}") from e
    elif suffix == '.parquet':
        try:
            out.to_parquet(str(filepath), index=False)
        except Exception as e:
            fallback = filepath.with_suffix('.csv')
            out.to_csv(str(fallback), index=False)
            raise RuntimeError(f"Could not write Parquet file ({e}). Saved as CSV: {fallback}") from e
    elif suffix in ('.tsv', '.txt'):
        out.to_csv(str(filepath), sep='\t', index=False)
    elif suffix == '.json':
        out.to_json(str(filepath), orient='records', indent=2)
    else:
        out.to_csv(str(filepath), index=False)


def export_confidence_map(filepath: Path, confidence_map: np.ndarray) -> None:
    """
    Export a per-pixel confidence map in multiple formats (float32 TIFF, PNG heatmap, NumPy NPY/NPZ).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    suffix = filepath.suffix.lower()

    if suffix == '.npy':
        np.save(str(filepath), confidence_map.astype(np.float32))
    elif suffix == '.npz':
        np.savez_compressed(str(filepath), confidence=confidence_map.astype(np.float32))
    elif suffix == '.png':
        norm = np.clip(confidence_map * 255.0, 0, 255).astype(np.uint8)
        try:
            from PIL import Image
            Image.fromarray(norm).save(str(filepath))
        except Exception:
            tifffile.imwrite(str(filepath), norm)
    else:
        tifffile.imwrite(str(filepath), confidence_map.astype(np.float32), compression='zlib')


def export_batch_summary(filepath: Path, batch_results: dict) -> None:
    """Export a batch-processing summary CSV (one row per sample)."""
    rows = []
    for sample_name, res in batch_results.items():
        rows.append({
            'Sample':                  sample_name,
            'Total_Cells':             res.get('total_cells', 0),
            'High_Confidence_Cells':   res.get('high_conf', 0),
            'Medium_Confidence_Cells': res.get('med_conf', 0),
            'Low_Confidence_Cells':    res.get('low_conf', 0),
            'Mean_Confidence':         res.get('mean_confidence', 0.0),
            'Methods_Used':            ', '.join(res.get('methods', [])),
        })
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(str(filepath), index=False)


def export_phenotyping_results(
    filepath: Path,
    df: pd.DataFrame,
    sample_name: str = "",
) -> None:
    """
    Export phenotyping results as CSV.
    Expects *df* to contain at minimum:
      ObjectNumber, cell_type, cell_type_confidence
    and optionally per-method probability columns.
    """
    export_features_csv(filepath, df, sample_name)


def export_neighborhood_composition_matrix(
    filepath: Path,
    centers: np.ndarray,
    type_labels: list,
    niche_metadata: dict,
    cell_data: Optional[pd.DataFrame] = None,
) -> None:
    """
    Standardizes and writes a Cell-Type Proportion vs Niche Matrix to disk.
    Optionally exports the secondary per-cell assignment CSV.
    """
    if centers is None or type_labels is None:
        raise ValueError("Cannot serialize an empty or uninitialized composition matrix.")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    row_names = [niche_metadata.get(i, {}).get("name", f"Niche {i}") for i in range(centers.shape[0])]
    df_export = pd.DataFrame(centers, index=row_names, columns=type_labels)
    df_export.index.name = "Neighborhood_Label"
    df_export.to_csv(str(filepath))

    # Save secondary per-cell mapping table if cell_data is provided
    if cell_data is not None and not cell_data.empty:
        cell_filepath = filepath.with_name(filepath.stem + "_cell_assignments" + filepath.suffix)
        base_cols = ['Image', 'ObjectNumber', 'Pos_X', 'Pos_Y']
        pheno_col = next((col for col in ['cell_type', 'Verified_CellType', 'CellType'] if col in cell_data.columns), None)
        if pheno_col and pheno_col not in base_cols:
            base_cols.append(pheno_col)

        export_cols = [c for c in base_cols if c in cell_data.columns]
        if 'Cellular_Neighborhood' in cell_data.columns:
            export_cols.append('Cellular_Neighborhood')

        export_df = cell_data[export_cols].copy()
        if 'Cellular_Neighborhood' in export_df.columns:
            def _map_name(x):
                if x in niche_metadata:
                    return niche_metadata[x].get("name", f"Niche {x}")
                if isinstance(x, str) and x.startswith("Niche "):
                    try:
                        idx = int(x.replace("Niche ", "").strip())
                        if idx in niche_metadata:
                            return niche_metadata[idx].get("name", x)
                    except ValueError:
                        pass
                try:
                    idx = int(x)
                    if idx in niche_metadata:
                        return niche_metadata[idx].get("name", f"Niche {idx}")
                except (ValueError, TypeError):
                    pass
                return str(x)
            export_df['Neighborhood_Name'] = export_df['Cellular_Neighborhood'].map(_map_name)
        export_df.to_csv(str(cell_filepath), index=False)

