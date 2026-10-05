"""
IRIS-Phenotyper CELESTA Backend

CELESTA (CELl typE idenTification by hieArchical clustering) is a rule-based
EM algorithm that uses a user-defined hierarchical anchor marker matrix to
assign cell types in a defined priority order.

Reference: Zhang et al. 2022 (Nature Methods).
Install: pip install git+https://github.com/plevritis-lab/CELESTA.git

Input (from Viewer features CSV)
---------------------------------
- Intensity matrix: cells × markers (``<Marker>_mean`` columns)
- CellTypePanel: defines the anchor marker matrix + hierarchy order

Output
------
Dict[cell_id → {cell_type: probability}]

Notes
-----
CELESTA is not on PyPI.  The backend gracefully degrades to ``is_available()
= False`` if the package is absent.  A stub probability of 1.0 is used for
the assigned type (CELESTA is deterministic given the EM solution).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from iris_common.engine.phenotyping import PhenotypeBackend
from iris_common.types import CellTypePanel, MarkerPanel


class CELESTABackend(PhenotypeBackend):
    """
    Phenotyping backend wrapping the CELESTA library.

    CELESTA assigns cell types hierarchically — top-level categories (e.g.
    'Immune', 'Stromal', 'Tumor') are resolved first, then sub-types
    (e.g. 'T cell', 'B cell', 'Macrophage') within each.  The hierarchy
    is encoded in the ``CellTypePanel.rules[i].parent`` field.

    Attributes
    ----------
    name : str
        ``"CELESTA"``
    requires_training : bool
        False — CELESTA uses EM without user-labelled training data.
    """

    name = "CELESTA"
    requires_training = False
    requires_raw_image = False
    requires_spatial = False

    def __init__(self) -> None:
        self._assignments: Optional[pd.Series] = None
        self._cell_ids: Optional[List[int]] = None
        self._cell_types: Optional[List[str]] = None

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            import celesta  # noqa: F401
            return True
        except ImportError:
            return False

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "n_iter": 10,
            "n_cores": 1,
            "high_expression_threshold_anchor": 0.9,
            "low_expression_threshold_anchor": 0.4,
            "intensity_col_suffix": "_mean",
        }

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, features: pd.DataFrame, **kwargs) -> None:
        """
        Run CELESTA EM cell type assignment.

        Parameters
        ----------
        features : DataFrame
            Viewer feature table with ``<Marker>_mean`` columns.
        cell_type_panel : CellTypePanel
            Anchor marker definitions and hierarchy.
        marker_panel : MarkerPanel, optional
        n_iter : int
            Number of EM iterations (default 10).
        high_expression_threshold_anchor : float
        low_expression_threshold_anchor : float
        """
        import celesta

        cell_type_panel: CellTypePanel = kwargs["cell_type_panel"]
        marker_panel: Optional[MarkerPanel] = kwargs.get("marker_panel")
        n_iter: int = kwargs.get("n_iter", 10)
        high_thresh: float = kwargs.get("high_expression_threshold_anchor", 0.9)
        low_thresh: float = kwargs.get("low_expression_threshold_anchor", 0.4)
        suffix: str = kwargs.get("intensity_col_suffix", "_mean")

        expr = self._extract_intensity_matrix(features, stat=suffix.lstrip("_"))

        if marker_panel is not None:
            anchor_df = cell_type_panel.to_celesta_anchor_matrix(marker_panel)
        else:
            mp = MarkerPanel(markers=list(expr.columns))
            anchor_df = cell_type_panel.to_celesta_anchor_matrix(mp)

        # Remove marker columns not present in the intensity matrix
        common_markers = [m for m in anchor_df.columns if m in expr.columns]
        anchor_df = anchor_df[common_markers]
        expr_aligned = expr[common_markers]

        result = celesta.assign_celltypes(
            expr_aligned,
            anchor_df,
            n_iter=n_iter,
            high_expression_threshold=high_thresh,
            low_expression_threshold=low_thresh,
        )

        self._assignments = result
        self._cell_ids = self._cell_ids_from_features(features)
        self._cell_types = list(anchor_df.index)

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(
        self, features: pd.DataFrame, **kwargs
    ) -> Dict[int, Dict[str, float]]:
        """
        Return CELESTA assignments as stub probabilities.

        CELESTA is deterministic — the winning type gets 1.0, all others 0.0.
        The ensemble engine treats this as a hard vote.
        """
        if self._assignments is None:
            self.fit(features, **kwargs)

        result: Dict[int, Dict[str, float]] = {}
        cell_ids = self._cell_ids_from_features(features)

        all_types = self._cell_types or []

        for i, cell_id in enumerate(cell_ids):
            assigned = (
                str(self._assignments.iloc[i])
                if i < len(self._assignments)
                else "Unknown"
            )
            probs: Dict[str, float] = {ct: 0.0 for ct in all_types}
            if assigned in probs:
                probs[assigned] = 1.0
            elif assigned != "Unknown":
                probs[assigned] = 1.0
            result[int(cell_id)] = probs

        return result

    # ── Fallback: pure-Python EM when celesta is unavailable ─────────────────

    def predict_fallback(
        self,
        features: pd.DataFrame,
        cell_type_panel: CellTypePanel,
        marker_panel: Optional[MarkerPanel] = None,
        **kwargs,
    ) -> Dict[int, Dict[str, float]]:
        """
        Minimal rule-based classifier used when the CELESTA library is absent.

        Scores each cell against each CellTypeRule by computing the fraction
        of positive markers that exceed the 75th percentile expression in the
        dataset.  Returns normalised scores as proxy probabilities.

        This is NOT a replacement for CELESTA's EM algorithm — use it only
        when CELESTA is unavailable, e.g. on macOS with Apple Silicon.
        """
        suffix = kwargs.get("intensity_col_suffix", "_mean").lstrip("_")
        expr = self._extract_intensity_matrix(features, stat=suffix)

        # Compute per-marker 75th percentile thresholds
        thresholds = expr.quantile(0.75)

        all_types = [r.cell_type for r in cell_type_panel.rules]
        cell_ids = self._cell_ids_from_features(features)

        result: Dict[int, Dict[str, float]] = {}
        for i, cell_id in enumerate(cell_ids):
            row = expr.iloc[i]
            scores: Dict[str, float] = {}

            for rule in cell_type_panel.rules:
                pos_score = sum(
                    1.0
                    for m in rule.markers_positive
                    if m in row.index and row[m] >= thresholds.get(m, 0)
                )
                neg_score = sum(
                    1.0
                    for m in rule.markers_negative
                    if m in row.index and row[m] < thresholds.get(m, 0)
                )
                n_pos = max(1, len(rule.markers_positive))
                n_neg = max(1, len(rule.markers_negative))
                scores[rule.cell_type] = (
                    (pos_score / n_pos + neg_score / n_neg) / 2.0
                )

            total = sum(scores.values()) or 1.0
            result[int(cell_id)] = {ct: s / total for ct, s in scores.items()}

        return result

    @staticmethod
    def _cell_ids_from_features(features: pd.DataFrame) -> List[int]:
        if "ObjectNumber" in features.columns:
            return features["ObjectNumber"].astype(int).tolist()
        return list(range(len(features)))
