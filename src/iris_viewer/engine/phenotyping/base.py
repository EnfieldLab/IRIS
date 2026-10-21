"""
iris-common PhenotypeBackend ABC
The canonical interface all cell phenotyping methods must implement.
Mirrors SegmentationBackend in design.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class PhenotypeBackend(ABC):
    """
    Abstract base class for cell phenotyping backends.

    Contract
    --------
    Every backend must provide:

    ``name``
        Human-readable method identifier shown in the UI.
    ``requires_training``
        True if ``fit()`` must be called before ``predict()``.
        (ASTIR, MAPS = False)
    ``requires_raw_image``
        True if the backend needs raw pixel patches in addition to
        the feature table.
    ``requires_spatial``
        True if the backend uses cell–cell spatial context (MAPS = True).

    ``is_available()``
        Runtime dependency check — import-guarded.
    ``fit()``
        Train the model.  Must be a no-op for ASTIR.
    ``predict()``
        Return per-cell probabilities over all cell types.

    Output format
    -------------
    ``predict()`` must return::

        Dict[int, Dict[str, float]]
            cell_id → {cell_type → probability}

    Probabilities must sum to ~1.0 per cell (but need not be exactly 1 —
    the ensemble engine will re-normalise).
    """

    name: str = "BasePhenotypeBackend"
    requires_training: bool = False
    requires_raw_image: bool = False
    requires_spatial: bool = False

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend's Python dependencies are importable."""
        ...

    @abstractmethod
    def fit(self, features: pd.DataFrame, cell_type_panel: Optional[Any] = None, **kwargs) -> None:
        """
        Train the model.

        Parameters
        ----------
        features : DataFrame
            Cell-level feature table as exported by the Viewer
            (rows = cells, columns include ``ObjectNumber``, ``Pos_X``,
            ``Pos_Y``, and ``<Marker>_mean`` intensity columns).
        cell_type_panel : CellTypePanel, optional
            Marker-to-type mapping schema.
        **kwargs
            Backend-specific keyword arguments (e.g. ``reference_adata``,
            ``annotations_csv``).
        """
        ...

    @abstractmethod
    def predict(
        self,
        features: pd.DataFrame,
        cell_type_panel: Optional[Any] = None,
        **kwargs,
    ) -> Dict[int, Dict[str, float]]:
        """
        Assign cell type probabilities.

        Parameters
        ----------
        features : DataFrame
            Same schema as for ``fit()``.
        **kwargs
            Backend-specific keyword arguments.

        Returns
        -------
        Dict[int, Dict[str, float]]
            Mapping ``cell_id → {cell_type: probability}``.
            ``cell_id`` must match the ``ObjectNumber`` column in *features*.
        """
        ...

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Return a dict of default parameters for UI population."""
        ...

    # Utility

    @staticmethod
    def _extract_intensity_matrix(
        features: pd.DataFrame,
        stat: str = "mean",
    ) -> pd.DataFrame:
        """
        Extract a cells × markers intensity matrix from a features DataFrame.

        Parameters
        ----------
        features : DataFrame with columns like ``DAPI_mean``, ``CD45_mean``, etc.
        stat : one of ``'mean'``, ``'median'``, ``'sum'``, ``'max'``

        Returns
        -------
        DataFrame indexed by ``ObjectNumber``, columns = marker names.
        """
        suffix = f"_{stat}"
        cols = [c for c in features.columns if c.endswith(suffix)]

        # If no suffix found, assume the user provided raw marker columns without suffixes
        if not cols:
            # We filter out typical non-marker numeric columns or metadata
            exclude = {"file", "cell_id", "objectnumber", "pos_x", "pos_y", "x", "y", "centroid-0", "centroid-1", "area", "perimeter", "eccentricity", "voron_x", "voron_y", "group"}
            import pandas.api.types as ptypes
            cols = [c for c in features.columns if c.lower() not in exclude and ptypes.is_numeric_dtype(features[c])]
            markers = cols
        else:
            markers = [c.replace(suffix, '') for c in cols]

        if not cols:
            raise ValueError(
                f"No suitable intensity columns found in features DataFrame. "
                f"Available columns: {list(features.columns)[:20]}"
            )

        idx = features['ObjectNumber'].values if 'ObjectNumber' in features.columns else features.index
        df = features[cols].copy()
        df.index = idx
        df.columns = markers
        return df

    @staticmethod
    def _cell_ids(features: pd.DataFrame) -> List[int]:
        if 'ObjectNumber' in features.columns:
            return features['ObjectNumber'].astype(int).tolist()
        return list(range(len(features)))
