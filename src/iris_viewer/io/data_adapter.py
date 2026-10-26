"""
IRIS-Suite Data Adapter
Normalises external data formats into the IRIS-Suite internal schema.

Supported input formats
-----------------------
XLSX / CSV with the following columns (as exported from common mIF analysis pipelines):

  - First column (any name)    : sample identifier — typically a TMA core label
                                 like "r1c3" or "Sample_001_image"
  - voron_x / voron_y          : spatial coordinates (Voronoi cell centroids)
  - GROUP                      : cell type label from manual thresholding
  - DAPI … Ki67                : marker expression columns (no suffix)
  - area, elongation, etc.     : morphology columns

IRIS internal schema (output)
------------------------------
  - Image        : sample name (cleaned: _image/.tiff stripped)
  - ObjectNumber : integer cell ID (1-indexed per sample if not present)
  - Pos_X        : float x-coordinate (from voron_x)
  - Pos_Y        : float y-coordinate (from voron_y)
  - cell_type    : cell type label (from GROUP or equivalent column)
  - <Marker>_mean: per-marker intensity (renamed from raw marker columns)
  - <morph_col>  : morphology columns (area, perimeter, etc.) passed through
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Known column aliases

_COORD_X_ALIASES = {"voron_x", "pos_x", "x", "centroid_x", "x_centroid", "center_x"}
_COORD_Y_ALIASES = {"voron_y", "pos_y", "y", "centroid_y", "y_centroid", "center_y"}
_CELLTYPE_ALIASES = {"group", "cell_type", "celltype", "phenotype", "label",
                     "verified_celltype", "manual_label", "class"}
_ID_ALIASES = {"objectnumber", "object_number", "cell_id", "id", "label_id"}
_SAMPLE_ALIASES = {"image", "sample", "core", "filename", "tma_core", "slide"}

# Morphology columns to pass through as-is
_MORPH_COLS = {
    "area", "perimeter", "eccentricity", "solidity",
    "major_axis_length", "minor_axis_length", "orientation",
    "extent", "equivalent_diameter_area", "elongation",
    "circularity", "compactness",
}

# Columns to always exclude from marker detection
_NON_MARKER_COLS = (
    _COORD_X_ALIASES | _COORD_Y_ALIASES | _CELLTYPE_ALIASES |
    _ID_ALIASES | _SAMPLE_ALIASES | _MORPH_COLS |
    {"unnamed: 0", "index", "file", "row", "col", "notes", "batch"}
)


class IrisDataAdapter:
    """
    Converts an external expression/annotation DataFrame into the IRIS
    internal schema expected by both the Viewer and Phenotyper.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data as loaded from xlsx or CSV.
    sample_col : str, optional
        Column to use as the sample identifier.  Auto-detected if None.
    celltype_col : str, optional
        Column holding cell type labels.  Auto-detected if None.
    intensity_stat : str
        Suffix to append to detected marker columns.  Default: ``"mean"``.

    Attributes
    ----------
    normalised : pd.DataFrame
        The normalised DataFrame in IRIS internal format.
    markers : list[str]
        Detected marker names (without suffix).
    morph_cols : list[str]
        Detected morphology column names.
    sample_col_original : str
        The original column name used as sample identifier.
    celltype_col_original : str
        The original column name used as cell type.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sample_col: Optional[str] = None,
        celltype_col: Optional[str] = None,
        intensity_stat: str = "mean",
    ) -> None:
        self._raw = df.copy()
        self._intensity_stat = intensity_stat

        # Normalise column names for detection (lowercase, strip)
        self._col_lower: Dict[str, str] = {
            c.lower().strip(): c for c in df.columns
        }

        self.sample_col_original = self._detect_col(sample_col, _SAMPLE_ALIASES, required=False)
        self.celltype_col_original = self._detect_col(celltype_col, _CELLTYPE_ALIASES, required=False)
        self._x_col = self._detect_col(None, _COORD_X_ALIASES, required=True)
        self._y_col = self._detect_col(None, _COORD_Y_ALIASES, required=True)
        self._id_col = self._detect_col(None, _ID_ALIASES, required=False)

        self.markers: List[str] = self._detect_markers()
        self.morph_cols: List[str] = self._detect_morph_cols()

        self.normalised: pd.DataFrame = self._build_normalised()

    # Detection helpers

    def _detect_col(
        self, override: Optional[str], aliases: set, required: bool
    ) -> Optional[str]:
        """Return the first column whose lowercased name appears in *aliases*."""
        if override is not None:
            return override
        for lower_name, orig_name in self._col_lower.items():
            if lower_name in aliases:
                return orig_name
        if required:
            raise ValueError(
                f"Could not auto-detect a column from aliases {aliases}. "
                f"Available columns: {list(self._raw.columns)}"
            )
        return None

    def _detect_markers(self) -> List[str]:
        """
        Return marker column names.

        A column is treated as a marker if:
        - Its lowercase name is NOT in ``_NON_MARKER_COLS``
        - Its lowercase name does not start with ``unnamed``
        - The column contains numeric data
        - It is not a morphology column
        """
        skip = {
            self.sample_col_original,
            self.celltype_col_original,
            self._x_col, self._y_col, self._id_col,
        }
        skip = {s.lower().strip() if s else None for s in skip}

        markers = []
        for col in self._raw.columns:
            cl = col.lower().strip()
            if cl in skip or cl in _NON_MARKER_COLS:
                continue
            if cl.startswith("unnamed"):
                continue
            if not pd.api.types.is_numeric_dtype(self._raw[col]):
                continue
            markers.append(col)
        return markers

    def _detect_morph_cols(self) -> List[str]:
        """Return morphology column names present in the DataFrame."""
        return [
            orig for lower, orig in self._col_lower.items()
            if lower in _MORPH_COLS
        ]

    # Normalisation

    def _build_normalised(self) -> pd.DataFrame:
        """Construct the IRIS internal DataFrame while preserving all original columns."""
        # Start with a full copy to preserve all user columns
        out = self._raw.copy()

        # Sample identifier
        if self.sample_col_original:
            out["Image"] = (
                self._raw[self.sample_col_original]
                .astype(str)
                .str.replace(r"_image\.tiff?$", "", regex=True)
                .str.replace(r"\.tiff?$", "", regex=True)
                .str.strip()
            )
        else:
            # Try to use the first column as the sample identifier
            first_col = self._raw.columns[0]
            out["Image"] = (
                self._raw[first_col]
                .astype(str)
                .str.replace(r"_image\.tiff?$", "", regex=True)
                .str.strip()
            )

        # Cell ID
        if self._id_col:
            out["ObjectNumber"] = self._raw[self._id_col].astype(int)

        # Coordinates
        out["Pos_X"] = self._raw[self._x_col].astype(float)
        out["Pos_Y"] = self._raw[self._y_col].astype(float)

        # Cell type
        if self.celltype_col_original and "cell_type" not in out.columns:
            out["cell_type"] = self._raw[self.celltype_col_original].astype(str).str.strip()

        return out

    # Public helpers

    def per_sample(self) -> Dict[str, pd.DataFrame]:
        """Return a dict mapping sample name → normalised DataFrame slice."""
        if "Image" not in self.normalised.columns:
            return {"all": self.normalised}
        return {
            name: grp.reset_index(drop=True)
            for name, grp in self.normalised.groupby("Image")
        }

    def intensity_matrix(self, stat: Optional[str] = None) -> pd.DataFrame:
        """
        Extract a cells × markers matrix.

        Parameters
        ----------
        stat : str, optional
            Column suffix to select.  Defaults to the adapter's
            ``intensity_stat`` (usually ``"mean"``).

        Returns
        -------
        pd.DataFrame indexed by ObjectNumber, columns = marker names.
        """
        s = stat or self._intensity_stat
        suffix = f"_{s}"
        cols = [c for c in self.normalised.columns if c.endswith(suffix)]

        if cols:
            marker_names = [c.replace(suffix, "") for c in cols]
            mat = self.normalised[cols].copy()
            mat.columns = marker_names
        else:
            # Fallback to base markers if no suffix is found
            cols = [m for m in self.markers if m in self.normalised.columns]
            mat = self.normalised[cols].copy()

        mat.index = self.normalised["ObjectNumber"].values
        return mat

    def cell_type_labels(self) -> Optional[pd.Series]:
        """Return the cell_type column (or None if absent)."""
        if "cell_type" in self.normalised.columns:
            return self.normalised.set_index("ObjectNumber")["cell_type"]
        return None

    # AnnData bridge

    def to_anndata(self, sample_name: Optional[str] = None):
        """
        Convert to an AnnData object for scanpy workflows.

        Parameters
        ----------
        sample_name : str, optional
            If given, only the cells from that sample are included.

        Returns
        -------
        anndata.AnnData with:
          .X          = intensity matrix (cells × markers)
          .obs        = ObjectNumber, Pos_X, Pos_Y, cell_type, Image
          .var_names  = marker names
        """
        import anndata as ad

        df = self.normalised
        if sample_name:
            df = df[df["Image"] == sample_name].reset_index(drop=True)

        suffix = f"_{self._intensity_stat}"
        marker_cols = [c for c in df.columns if c.endswith(suffix)]
        marker_names = [c.replace(suffix, "") for c in marker_cols]

        obs_cols = [c for c in ["ObjectNumber", "Pos_X", "Pos_Y", "cell_type", "Image"]
                    if c in df.columns]
        obs = df[obs_cols].copy()
        if "ObjectNumber" in obs.columns:
            obs.index = obs["ObjectNumber"].astype(str)

        X = df[marker_cols].values.astype(np.float32)

        adata = ad.AnnData(X=X, obs=obs)
        adata.var_names = marker_names
        return adata

    # Factory classmethod

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        sheet_name: int | str = 0,
        **kwargs,
    ) -> "IrisDataAdapter":
        """
        Load from an xlsx or csv file and construct the adapter.

        Parameters
        ----------
        path : str or Path
        sheet_name : str or int
            Excel sheet name/index.  Ignored for CSV.
        **kwargs
            Passed to ``IrisDataAdapter.__init__``.
        """
        path = Path(path)
        if path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
            df = pd.read_excel(path, sheet_name=sheet_name)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        return cls(df, **kwargs)


# IrisPhenotypeMapper

class IrisPhenotypeMapper:
    """
    Maps segmentation mask cell IDs to cell types from a DataFrame,
    and builds binary masks per cell type for napari rendering.

    Parameters
    ----------
    mask : np.ndarray (H, W), dtype uint32
        Nuclear segmentation mask (cell IDs, background = 0).
    df : pd.DataFrame
        IRIS normalised DataFrame with at minimum:
          ObjectNumber, Pos_X, Pos_Y, and a cell type column.
    target_col : str
        Column name to use as cell type labels.  Default: ``"cell_type"``.
    """

    # Default cell type colour palette (matplotlib named colours)
    DEFAULT_COLORS = [
        "#4477EE", "#44AA44", "#EE4444", "#FFAA22", "#AA44EE",
        "#22AACC", "#FF6688", "#88CC44", "#AA8822", "#44CCAA",
        "#CC4488", "#6688FF", "#FFCC44", "#44DDFF", "#FF8844",
    ]

    def __init__(
        self,
        mask: np.ndarray,
        df: pd.DataFrame,
        target_col: str = "cell_type",
    ) -> None:
        if mask.ndim > 2:
            mask = mask[0] if mask.shape[0] < mask.shape[-1] else mask[..., 0]
        self.mask = mask
        self.df = df.copy()
        self.target_col = target_col if target_col in df.columns else self._fallback_col(df)

    def _fallback_col(self, df: pd.DataFrame) -> str:
        for c in ("cell_type", "Verified_CellType", "CellType", "GROUP", "Phenotype", "Label"):
            if c in df.columns:
                return c

        # If still not found, return the last object/string column or just the first column
        for c in df.columns:
            if df[c].dtype == object or df[c].dtype.name == 'category':
                return c

        return df.columns[0] if len(df.columns) > 0 else "cell_type"

    def resolve_ids(self) -> pd.DataFrame:
        """
        Attempt to assign ObjectNumber from mask pixel lookup
        when the DataFrame has Pos_X / Pos_Y coordinates.
        Ensures ObjectNumber corresponds to non-zero segmented labels.
        """
        df = self.df.copy()
        h, w = self.mask.shape[-2], self.mask.shape[-1]

        # Check if df already has a valid non-zero ObjectNumber column
        has_valid_obj = (
            "ObjectNumber" in df.columns
            and pd.to_numeric(df["ObjectNumber"], errors="coerce").fillna(0).gt(0).any()
        )

        if {"Pos_X", "Pos_Y"}.issubset(df.columns):
            x_num = pd.to_numeric(df["Pos_X"], errors="coerce")
            y_num = pd.to_numeric(df["Pos_Y"], errors="coerce")
            valid = x_num.notna() & y_num.notna()

            xs = np.round(x_num.fillna(-1).values).astype(np.int64, copy=False)
            ys = np.round(y_num.fillna(-1).values).astype(np.int64, copy=False)

            in_bounds = valid.values & (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            lookup_ids = np.zeros(len(df), dtype=np.int64)

            if in_bounds.any():
                v_idx = np.where(in_bounds)[0]
                v_xs = xs[v_idx]
                v_ys = ys[v_idx]

                direct_vals = self.mask[v_ys, v_xs].astype(np.int64)
                lookup_ids[v_idx] = direct_vals

                # Centroid may be slightly outside the boundary of the mask; search a 3x3 local window
                zero_mask = direct_vals == 0
                if zero_mask.any():
                    from scipy.ndimage import maximum_filter

                    dilated = maximum_filter(self.mask, size=3)
                    fallback_vals = dilated[v_ys[zero_mask], v_xs[zero_mask]].astype(np.int64)
                    lookup_ids[v_idx[zero_mask]] = fallback_vals

            if has_valid_obj:
                df["ObjectNumber"] = np.where(
                    lookup_ids > 0,
                    lookup_ids,
                    pd.to_numeric(df["ObjectNumber"], errors="coerce").fillna(0).astype(int),
                )
            else:
                df["ObjectNumber"] = lookup_ids
        elif "ObjectNumber" not in df.columns:
            df["ObjectNumber"] = np.arange(1, len(df) + 1)

        self.df = df
        return df

    def get_masks_by_type(self) -> Dict[str, Tuple[np.ndarray, str]]:
        """
        Build a binary mask per unique cell type.

        Returns
        -------
        dict mapping cell_type → (binary_mask, hex_colour)
        where binary_mask is uint8 array (H, W), 1 = cell of that type, 0 = background.
        """
        if self.target_col not in self.df.columns:
            return {}

        self.resolve_ids()

        unique_types = sorted(
            [str(ct) for ct in self.df[self.target_col].dropna().unique()]
        )
        palette = {
            ct: self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]
            for i, ct in enumerate(unique_types)
        }

        # Filter strictly for cells with positive ObjectNumber (oid > 0)
        # Background is ALWAYS 0 in segmentation masks and must NEVER be mapped as a cell!
        valid_df = self.df[pd.to_numeric(self.df["ObjectNumber"], errors="coerce").fillna(0) > 0]

        result: Dict[str, Tuple[np.ndarray, str]] = {}
        h, w = self.mask.shape[-2], self.mask.shape[-1]

        max_id = int(self.mask.max()) if self.mask.size > 0 else 0
        # Fast path: Vectorized categorical LUT indexing (single pass over the image array)
        if 0 < max_id < 2_000_000 and len(valid_df) > 0:
            lut = np.zeros(max_id + 1, dtype=np.uint16)
            for type_idx, cell_type in enumerate(unique_types, start=1):
                t_ids = (
                    valid_df.loc[
                        valid_df[self.target_col].astype(str) == cell_type,
                        "ObjectNumber",
                    ]
                    .astype(np.int64)
                    .values
                )
                valid_t_ids = t_ids[(t_ids > 0) & (t_ids <= max_id)]
                if len(valid_t_ids) > 0:
                    lut[valid_t_ids] = type_idx

            cat_mask = lut[self.mask]
            for type_idx, cell_type in enumerate(unique_types, start=1):
                binary = (cat_mask == type_idx).astype(np.uint8)
                result[cell_type] = (binary, palette[cell_type])
        else:
            # Fallback path for empty masks or sparse large IDs
            for cell_type in unique_types:
                type_rows = valid_df[valid_df[self.target_col].astype(str) == cell_type]
                target_ids = type_rows["ObjectNumber"].astype(np.int64).unique()
                target_ids = target_ids[target_ids > 0]
                if len(target_ids) == 0:
                    binary = np.zeros((h, w), dtype=np.uint8)
                else:
                    binary = np.isin(self.mask, target_ids).astype(np.uint8)
                result[cell_type] = (binary, palette[cell_type])

        return result
