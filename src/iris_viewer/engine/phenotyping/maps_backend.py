"""
IRIS-Phenotyper MAPS Backend

MAPS (Multiplexed Analysis of Phenotypes Spatial)
Reference: https://github.com/mahmoodlab/MAPS
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from iris_viewer.core.types import CellTypePanel
from iris_viewer.engine.phenotyping.base import PhenotypeBackend


class MAPSBackend(PhenotypeBackend):
    """
    Phenotyping backend wrapping the MAPS library.

    Attributes
    ----------
    name : str
        ``"MAPS"``
    requires_training : bool
        False.
    requires_raw_image : bool
        False.
    requires_spatial : bool
        True — MAPS incorporates spatial graphs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "MAPS"
        self.requires_training = False
        self.requires_raw_image = False
        self.requires_spatial = True

    def is_available(self) -> bool:
        """
        Check if MAPS is importable.
        """
        # For now, we return False to force fallback, until MAPS is fully installed.
        return False

    def get_default_params(self) -> Dict[str, Any]:
        """
        Return default parameters for MAPS.
        """
        return {
            "epochs": 50
        }

    def fit(self, features: pd.DataFrame, **kwargs) -> None:
        """
        Fit the MAPS model.
        """
        raise NotImplementedError("MAPS backend fit is not yet implemented.")

    def predict(
        self,
        features: pd.DataFrame,
        cell_types: Optional[CellTypePanel] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[int, Dict[str, float]]:
        """Predict cell types using MAPS, falling back to rule-based scoring when unavailable.

        Args:
            features: Single-cell feature DataFrame.
            cell_types: CellTypePanel specifying marker rules.
            params: Optional model parameter overrides.
            **kwargs: Extra parameters passed by caller.

        Returns:
            Dictionary mapping cell ID to probability distribution over cell types.
        """
        panel = cell_types or kwargs.get("cell_type_panel") or kwargs.get("panel")
        if not self.is_available():
            return self.predict_fallback(features, cell_type_panel=panel)
        raise NotImplementedError("MAPS graph inference is not yet implemented.")

    def predict_fallback(
        self,
        features: pd.DataFrame,
        **kwargs: Any,
    ) -> Dict[int, Dict[str, float]]:
        """Fallback deterministic rule-based assignment when MAPS is unavailable."""
        cell_types = kwargs.get("cell_type_panel")
        if not cell_types or not cell_types.rules:
            return {}

        # Very basic fallback implementation
        predictions = {}

        if features.empty:
            return predictions

        for idx, row in features.iterrows():
            cell_id = int(row.get("ObjectNumber", row.get("Cell_ID", row.get("cell_id", idx))))
            best_type = "Unresolved"
            best_score = -1.0

            for rule in cell_types.rules:
                ct_name = rule.cell_type
                pos_markers = rule.markers_positive
                if not pos_markers:
                    continue

                score = 0.0
                valid = True
                for marker in pos_markers:
                    col = f"{marker}_mean"
                    if col not in features.columns:
                        col = marker
                    if col in features.columns:
                        val = row[col]
                        if val < 0.1:  # Simple threshold
                            valid = False
                            break
                        score += val

                if valid and score > best_score:
                    best_score = score
                    best_type = ct_name

            # Map deterministic winner
            prob_dict = {ct: (1.0 if ct == best_type else 0.0) for ct in cell_types.cell_types}
            prob_dict["Unresolved"] = 1.0 if best_type == "Unresolved" else 0.0
            predictions[cell_id] = prob_dict

        return predictions
