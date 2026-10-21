"""IRIS-Phenotyper ASTIR Backend.

ASTIR (Automated Single-cell Typing and Inference of States) uses Bayesian
variational inference to model single-cell protein expression and assign
calibrated posterior probabilities from a user-defined marker panel.

Reference: Geuenich et al., Nature Communications 2021.
Install: pip install astir
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from iris_viewer.core.types import CellTypePanel, MarkerPanel
from iris_viewer.engine.phenotyping.base import PhenotypeBackend


class ASTIRBackend(PhenotypeBackend):
    """Phenotyping backend wrapping the ASTIR variational Bayesian inference library."""

    name = "ASTIR"
    requires_training = False
    requires_raw_image = False
    requires_spatial = False

    def __init__(self) -> None:
        self._model = None
        self._trained: bool = False
        self._cell_ids: Optional[List[int]] = None

    # Availability

    def is_available(self) -> bool:
        """Check whether the astir Python package is installed and importable."""
        try:
            import astir  # noqa: F401
            return True
        except ImportError:
            return False

    def get_default_params(self) -> Dict[str, Any]:
        """Return default hyperparameters for ASTIR model training."""
        return {
            "max_epochs": 50,
            "learning_rate": 1e-3,
            "batch_size": 256,
            "arcsinh_cofactor": 5.0,
            "intensity_col_suffix": "_mean",
        }

    # Fit

    def fit(
        self,
        features: pd.DataFrame,
        cell_type_panel: Optional[CellTypePanel] = None,
        **kwargs: Any,
    ) -> None:
        """Fit ASTIR variational inference model using the feature matrix.

        Args:
            features: Cell feature table containing marker intensity columns.
            cell_type_panel: Marker-to-cell-type mapping schema.
            **kwargs: Hyperparameters including max_epochs, learning_rate,
                batch_size, arcsinh_cofactor, and intensity_col_suffix.
        """
        import astir

        panel = cell_type_panel or kwargs.get("cell_type_panel") or kwargs.get("panel")
        if panel is None:
            raise ValueError("cell_type_panel is required for ASTIR fitting.")

        marker_panel: Optional[MarkerPanel] = kwargs.get("marker_panel")
        max_epochs: int = kwargs.get("max_epochs", 50)
        learning_rate: float = kwargs.get("learning_rate", kwargs.get("lr", 1e-3))
        batch_size: int = kwargs.get("batch_size", 256)
        cofactor: float = kwargs.get("arcsinh_cofactor", 5.0)
        suffix: str = kwargs.get("intensity_col_suffix", "_mean")

        expr = self._extract_intensity_matrix(features, stat=suffix.lstrip("_"))
        if cofactor > 0:
            expr = np.arcsinh(expr / cofactor)

        mp = marker_panel if marker_panel is not None else MarkerPanel(markers=list(expr.columns))
        raw_dict = panel.to_astir_dict(mp)
        type_dict = raw_dict if "cell_types" in raw_dict else {"cell_types": raw_dict}

        ct_entries = type_dict.get("cell_types", {})
        if not ct_entries or len(ct_entries) < 2:
            raise ValueError("ASTIR requires at least two distinct cell types with defined positive markers.")

        self._cell_ids = self._cell_ids_from_features(features)
        self._model = astir.Astir(expr, type_dict)
        self._model.fit_type(
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            batch_size=min(batch_size, max(len(expr), 1)),
        )
        self._trained = True

    # Predict

    def predict(
        self,
        features: pd.DataFrame,
        cell_type_panel: Optional[CellTypePanel] = None,
        **kwargs: Any,
    ) -> Dict[int, Dict[str, float]]:
        """Return per-cell posterior probabilities across cell types.

        Args:
            features: Cell feature table with marker intensities.
            cell_type_panel: Marker-to-type mapping schema.
            **kwargs: Additional parameters passed to fit() if not trained.

        Returns:
            Dict mapping cell ID to dict of cell type probabilities.
        """
        panel = cell_type_panel or kwargs.get("cell_type_panel") or kwargs.get("panel")
        cell_ids = self._cell_ids_from_features(features)
        if features.empty or panel is None or not panel.rules:
            return {int(cid): {} for cid in cell_ids}

        if not self._trained or self._model is None:
            try:
                self.fit(features, cell_type_panel=panel, **kwargs)
            except Exception:
                return self._predict_fallback(features, panel)

        try:
            prob_df: pd.DataFrame = self._model.get_celltype_probabilities()
            result: Dict[int, Dict[str, float]] = {}
            for i, cell_id in enumerate(cell_ids):
                if i < len(prob_df):
                    row = prob_df.iloc[i]
                    result[int(cell_id)] = {str(ct): float(p) for ct, p in row.items()}
                else:
                    result[int(cell_id)] = {}
            return result
        except Exception:
            return self._predict_fallback(features, panel)

    # Helpers

    def _predict_fallback(
        self,
        features: pd.DataFrame,
        cell_type_panel: CellTypePanel,
    ) -> Dict[int, Dict[str, float]]:
        """Deterministic rule-based assignment fallback when ASTIR VI is infeasible."""
        cell_ids = self._cell_ids_from_features(features)
        results: Dict[int, Dict[str, float]] = {}
        if not cell_type_panel.rules:
            return {int(cid): {} for cid in cell_ids}

        for idx, row in features.iterrows():
            cid = int(row.get("ObjectNumber", row.get("Cell_ID", row.get("cell_id", idx))))
            scores: Dict[str, float] = {}
            for rule in cell_type_panel.rules:
                ct = rule.cell_type
                markers = rule.markers_positive
                if not markers:
                    continue
                vals = [
                    row[col] for m in markers
                    for col in (f"{m}_mean", m)
                    if col in features.columns
                ]
                scores[ct] = float(np.mean(vals)) if vals else 0.0

            total = sum(scores.values())
            if total > 0:
                probs = {k: v / total for k, v in scores.items()}
            else:
                uniform = 1.0 / len(cell_type_panel.rules) if cell_type_panel.rules else 0.0
                probs = {r.cell_type: uniform for r in cell_type_panel.rules}
            results[cid] = probs

        return results

    @staticmethod
    def _cell_ids_from_features(features: pd.DataFrame) -> List[int]:
        for col in ("ObjectNumber", "Cell_ID", "cell_id"):
            if col in features.columns:
                return features[col].astype(int).tolist()
        return list(range(len(features)))

