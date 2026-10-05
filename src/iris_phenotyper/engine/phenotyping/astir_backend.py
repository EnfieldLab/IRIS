"""
IRIS-Phenotyper ASTIR Backend

ASTIR (Automatic Single-cell Type Identification and Residualization) uses
Bayesian variational inference to assign cell types from a user-defined
marker-to-type dictionary.

Reference: Geuenich et al. 2021 (bioRxiv → Nature Methods).
Install: pip install astir

Input (from Viewer features CSV)
---------------------------------
- Intensity matrix: cells × markers (``<Marker>_mean`` columns)
- CellTypePanel: user-defined marker-to-type mapping

Output
------
Dict[cell_id → {cell_type: probability}]
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from iris_common.engine.phenotyping import PhenotypeBackend
from iris_common.types import CellTypePanel, MarkerPanel


class ASTIRBackend(PhenotypeBackend):
    """
    Phenotyping backend wrapping the ASTIR library.

    Attributes
    ----------
    name : str
        ``"ASTIR"``
    requires_training : bool
        False — ASTIR uses Bayesian VI, no explicit fit step needed.
    requires_raw_image : bool
        False — intensity matrix is sufficient.
    requires_spatial : bool
        False — ASTIR is a non-spatial method.
    """

    name = "ASTIR"
    requires_training = False
    requires_raw_image = False
    requires_spatial = False

    def __init__(self) -> None:
        self._model = None
        self._trained: bool = False
        self._cell_ids: Optional[List[int]] = None

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            import astir  # noqa: F401
            return True
        except ImportError:
            return False

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "max_epochs": 500,
            "learning_rate": 1e-3,
            "n_samples": 100,
            "intensity_col_suffix": "_mean",
        }

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, features: pd.DataFrame, **kwargs) -> None:
        """
        ASTIR uses variational inference internally; ``fit`` is the full training
        call (calls ``astir.Astir.fit()`` under the hood).

        Parameters
        ----------
        features : DataFrame
            Viewer feature table. Must contain ``<Marker>_mean`` columns.
        cell_type_panel : CellTypePanel
            Marker-to-type mapping.
        marker_panel : MarkerPanel, optional
            Full marker panel for resolving marker names.
        max_epochs : int
            VI optimisation epochs (default: 500).
        learning_rate : float
            Adam LR (default: 1e-3).
        """
        import astir

        cell_type_panel: CellTypePanel = kwargs["cell_type_panel"]
        marker_panel: Optional[MarkerPanel] = kwargs.get("marker_panel")
        max_epochs: int = kwargs.get("max_epochs", 500)
        lr: float = kwargs.get("learning_rate", 1e-3)
        suffix: str = kwargs.get("intensity_col_suffix", "_mean")

        # Build intensity matrix (cells × markers)
        expr = self._extract_intensity_matrix(features, stat=suffix.lstrip("_"))

        # Convert CellTypePanel to ASTIR marker_dict format
        if marker_panel is not None:
            type_dict = cell_type_panel.to_astir_dict(marker_panel)
        else:
            # Build a synthetic MarkerPanel from available columns
            mp = MarkerPanel(markers=list(expr.columns))
            type_dict = cell_type_panel.to_astir_dict(mp)

        if not type_dict:
            raise ValueError(
                "ASTIR marker dict is empty — check that marker names in the "
                "CellTypePanel match column names in the features DataFrame."
            )

        self._cell_ids = self._cell_ids_from_features(features)

        # Astir expects a DataFrame with cells as rows, markers as columns
        self._model = astir.Astir(expr, type_dict)
        self._model.fit_type(max_epochs=max_epochs, lr=lr)
        self._trained = True

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(
        self, features: pd.DataFrame, **kwargs
    ) -> Dict[int, Dict[str, float]]:
        """
        Return per-cell ASTIR posterior type probabilities.

        If ``fit()`` has not been called, it will be called automatically
        using kwargs.
        """
        if not self._trained or self._model is None:
            self.fit(features, **kwargs)

        # prob_df: (cells × cell_types) DataFrame of posterior probs
        prob_df: pd.DataFrame = self._model.get_celltypes(threshold=0.0)

        cell_ids = self._cell_ids_from_features(features)

        result: Dict[int, Dict[str, float]] = {}
        for i, cell_id in enumerate(cell_ids):
            if i < len(prob_df):
                row = prob_df.iloc[i]
                result[int(cell_id)] = {
                    str(ct): float(prob)
                    for ct, prob in row.items()
                }
            else:
                result[int(cell_id)] = {}

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cell_ids_from_features(features: pd.DataFrame) -> List[int]:
        if "ObjectNumber" in features.columns:
            return features["ObjectNumber"].astype(int).tolist()
        return list(range(len(features)))
