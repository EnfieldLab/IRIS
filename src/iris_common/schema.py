"""
IRIS Canonical DataFrame Schema & Column Resolution
=================================================
Centralizes the IRIS canonical single-cell schema constants and resolution
helpers used across Viewer, Phenotyper, and Analyzer packages.

Canonical columns:
- Sample identifier : ``Image``
- Cell identifier   : ``ObjectNumber`` (1-indexed positive integers; 0 = background)
- Spatial X         : ``Pos_X``
- Spatial Y         : ``Pos_Y``
- Default phenotype : ``cell_type``
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

COL_SAMPLE: str = "Image"
COL_CELL_ID: str = "ObjectNumber"
COL_X: str = "Pos_X"
COL_Y: str = "Pos_Y"
COL_CELL_TYPE: str = "cell_type"

INTENSITY_STAT_PRIORITY: Tuple[str, ...] = ("mean", "median", "sum", "max", "min", "std")

SAMPLE_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "Image",
    "Sample",
    "sample_id",
    "Sample_ID",
    "Sample_Name",
    "sample_name",
    "Core",
    "core",
    "TMA_Core",
    "tma_core",
    "sample",
    "image",
    "filename",
    "file",
    "slide",
    "FOV",
    "fov",
    "region",
    "roi",
)

CELL_ID_ALIASES: Tuple[str, ...] = (
    "ObjectNumber",
    "object_number",
    "objectnumber",
    "Cell_ID",
    "cell_id",
    "label_id",
    "label",
    "id",
)

X_COORD_ALIASES: Tuple[str, ...] = (
    "Pos_X",
    "pos_x",
    "voron_x",
    "Centroid_X",
    "centroid_x",
    "x_centroid",
    "center_x",
    "centroid-1",
    "X_centroid",
    "X",
    "x",
)

Y_COORD_ALIASES: Tuple[str, ...] = (
    "Pos_Y",
    "pos_y",
    "voron_y",
    "Centroid_Y",
    "centroid_y",
    "y_centroid",
    "center_y",
    "centroid-0",
    "Y_centroid",
    "Y",
    "y",
)

CELL_TYPE_ALIASES: Tuple[str, ...] = (
    "cell_type",
    "CellType",
    "celltype",
    "Verified_CellType",
    "verified_celltype",
    "Phenotype",
    "phenotype",
    "GROUP",
    "group",
    "manual_label",
    "class",
    "label",
)


def find_sample_column(df: Optional[pd.DataFrame]) -> Optional[str]:
    """Identify the column representing sample, FOV, or TMA core identity."""
    if df is None or len(df.columns) == 0:
        return None

    cols = list(df.columns)
    for candidate in SAMPLE_COLUMN_CANDIDATES:
        if candidate in cols:
            return candidate

    lower_map = {str(c).strip().lower(): str(c) for c in cols}
    for candidate in SAMPLE_COLUMN_CANDIDATES:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def resolve_marker_column(
    df: Optional[pd.DataFrame],
    marker: str,
    preferred_stat: str = "mean",
) -> Optional[str]:
    """Resolve a marker name to its matching intensity column in ``df``.

    Resolution order:
    1. Exact match (``marker``)
    2. Preferred stat suffix (``f"{marker}_{preferred_stat}"``)
    3. Case-insensitive exact match
    4. Case-insensitive preferred stat match
    5. Fallback stat suffixes (``mean``, ``median``, ``sum``, ``max``, ``min``, ``std``)
    6. Base marker name if ``marker`` already carries a stat suffix (e.g., ``CD3_mean`` -> ``CD3``)
    """
    if df is None or not marker or len(df.columns) == 0:
        return None

    cols = [str(c) for c in df.columns]
    col_set = set(cols)
    if marker in col_set:
        return marker

    pref_col = f"{marker}_{preferred_stat}"
    if pref_col in col_set:
        return pref_col

    lower_map = {c.strip().lower(): c for c in cols}
    marker_lower = marker.strip().lower()
    if marker_lower in lower_map:
        return lower_map[marker_lower]

    pref_lower = f"{marker_lower}_{preferred_stat.strip().lower()}"
    if pref_lower in lower_map:
        return lower_map[pref_lower]

    for stat in INTENSITY_STAT_PRIORITY:
        cand = f"{marker}_{stat}"
        if cand in col_set:
            return cand
        cand_lower = f"{marker_lower}_{stat}"
        if cand_lower in lower_map:
            return lower_map[cand_lower]

    for stat in INTENSITY_STAT_PRIORITY:
        suffix = f"_{stat}"
        if marker_lower.endswith(suffix):
            base = marker[: len(marker) - len(suffix)]
            if base in col_set:
                return base
            base_lower = base.strip().lower()
            if base_lower in lower_map:
                return lower_map[base_lower]

    return None


def _find_matching_column(
    df: pd.DataFrame,
    aliases: Tuple[str, ...],
    *,
    numeric_only: Optional[bool] = None,
) -> Optional[str]:
    """Find the first column matching *aliases* (exact, then case-insensitive)."""
    cols = list(df.columns)
    lower_map = {str(c).strip().lower(): str(c) for c in cols}

    for alias in aliases:
        matched: Optional[str] = None
        if alias in df.columns:
            matched = alias
        elif alias.lower() in lower_map:
            matched = lower_map[alias.lower()]

        if matched is None:
            continue

        if numeric_only is True and not pd.api.types.is_numeric_dtype(df[matched]):
            continue
        if numeric_only is False and pd.api.types.is_numeric_dtype(df[matched]):
            continue
        return matched
    return None


def ensure_canonical_dataframe(
    df: pd.DataFrame,
    *,
    inplace: bool = False,
    default_sample: Optional[str] = None,
    clean_sample_names: bool = True,
    require_coordinates: bool = False,
) -> pd.DataFrame:
    """Normalize a single-cell DataFrame into IRIS canonical column conventions.

    Preserves all existing columns while ensuring canonical columns are populated
    when matching aliases or defaults are available:
    - ``Image``: sample/core identifier
    - ``ObjectNumber``: 1-indexed positive integer cell IDs
    - ``Pos_X`` / ``Pos_Y``: float spatial coordinates
    - ``cell_type``: string phenotype label column
    """
    out = df if inplace else df.copy()

    # 1. Sample column ('Image')
    if COL_SAMPLE not in out.columns:
        sample_src = find_sample_column(out)
        if sample_src is not None:
            series = out[sample_src].astype("string").str.strip()
            if clean_sample_names:
                series = series.str.replace(r"_image\.tiff?$", "", regex=True).str.replace(
                    r"\.tiff?$", "", regex=True
                )
            out[COL_SAMPLE] = series.astype(object).where(out[sample_src].notna(), other=np.nan)
        elif default_sample is not None:
            out[COL_SAMPLE] = default_sample

    # 2. Cell ID column ('ObjectNumber')
    if COL_CELL_ID in out.columns:
        numeric_ids = pd.to_numeric(out[COL_CELL_ID], errors="coerce")
        if len(out) > 0 and numeric_ids.notna().all():
            out[COL_CELL_ID] = numeric_ids.astype(np.int64)
        elif len(out) == 0:
            out[COL_CELL_ID] = pd.Series(dtype=np.int64)
        else:
            fallback = pd.Series(np.arange(1, len(out) + 1, dtype=np.int64), index=out.index)
            out[COL_CELL_ID] = numeric_ids.fillna(fallback).astype(np.int64)
    else:
        id_src = _find_matching_column(out, CELL_ID_ALIASES, numeric_only=True)
        if id_src is not None:
            numeric_ids = pd.to_numeric(out[id_src], errors="coerce")
            fallback = pd.Series(np.arange(1, len(out) + 1, dtype=np.int64), index=out.index)
            out[COL_CELL_ID] = numeric_ids.fillna(fallback).astype(np.int64)
        else:
            out[COL_CELL_ID] = np.arange(1, len(out) + 1, dtype=np.int64)

    # 3. Spatial coordinates ('Pos_X', 'Pos_Y')
    if COL_X not in out.columns:
        x_src = _find_matching_column(out, X_COORD_ALIASES, numeric_only=True)
        if x_src is not None:
            out[COL_X] = pd.to_numeric(out[x_src], errors="coerce").astype(float)
    if COL_Y not in out.columns:
        y_src = _find_matching_column(out, Y_COORD_ALIASES, numeric_only=True)
        if y_src is not None:
            out[COL_Y] = pd.to_numeric(out[y_src], errors="coerce").astype(float)

    if require_coordinates and (COL_X not in out.columns or COL_Y not in out.columns):
        raise ValueError(
            f"DataFrame missing canonical spatial coordinates ({COL_X!r}, {COL_Y!r})."
        )

    # 4. Phenotype column ('cell_type')
    if COL_CELL_TYPE not in out.columns:
        ct_src = _find_matching_column(out, CELL_TYPE_ALIASES, numeric_only=False)
        if ct_src is not None:
            raw_ct = out[ct_src]
            cleaned = raw_ct.astype("string").str.strip()
            out[COL_CELL_TYPE] = cleaned.astype(object).where(raw_ct.notna(), other=np.nan)

    return out


__all__ = [
    "COL_SAMPLE",
    "COL_CELL_ID",
    "COL_X",
    "COL_Y",
    "COL_CELL_TYPE",
    "INTENSITY_STAT_PRIORITY",
    "SAMPLE_COLUMN_CANDIDATES",
    "find_sample_column",
    "resolve_marker_column",
    "ensure_canonical_dataframe",
]
