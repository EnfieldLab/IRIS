"""
IRIS-Phenotyper Ensemble Voting Engine

Merges per-cell type probability predictions from multiple backends using
three voting strategies:

- ``soft_vote``         — weighted average of posterior probabilities
- ``weighted_majority`` — user-assigned method weights, winner-take-all
- ``borda_count``       — rank-based aggregation (robust to calibration issues)

Output
------
For each cell, produces:
    final_cell_type      : str
    cell_type_confidence : float   (fraction of methods that agree)
    per_method_breakdown : dict    {method_name: assigned_type}

Analogous to the Viewer's ``compute_consensus_mask``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Type aliases ──────────────────────────────────────────────────────────────

# Per-method predictions:  method_name → {cell_id → {cell_type → probability}}
Predictions = Dict[str, Dict[int, Dict[str, float]]]

# Ensemble output tuple
EnsembleResult = Tuple[
    Dict[int, str],         # cell_id → final_cell_type
    Dict[int, float],       # cell_id → confidence (inter-method agreement)
    Dict[int, Dict],        # cell_id → {method_name: assigned_type}
]


class PhenotypingEnsemble:
    """
    Ensemble voter for cell type predictions.

    Parameters
    ----------
    method_weights : dict, optional
        Manual per-method weights for ``weighted_majority`` and ``soft_vote``.
        Keys are method names; values are positive floats.
        Absent methods default to 1.0 (equal weighting).
    min_probability_threshold : float
        Minimum per-cell winning probability for the assignment to be
        considered confident.  Cells below this threshold are labelled
        ``"Unresolved"`` and flagged for review.  Default: 0.3
    """

    UNRESOLVED_LABEL: str = "Unresolved"

    def __init__(
        self,
        method_weights: Optional[Dict[str, float]] = None,
        min_probability_threshold: float = 0.30,
    ) -> None:
        self.method_weights = method_weights or {}
        self.min_probability_threshold = min_probability_threshold

    # ── Main API ──────────────────────────────────────────────────────────────

    def vote(
        self,
        predictions: Predictions,
        strategy: str = "soft_vote",
    ) -> EnsembleResult:
        """
        Aggregate predictions from all backends into a final cell type call.

        Parameters
        ----------
        predictions : Predictions
            ``{method_name: {cell_id: {cell_type: probability}}}``
        strategy : str
            One of ``"soft_vote"``, ``"weighted_majority"``, ``"borda_count"``.

        Returns
        -------
        final_types : Dict[int, str]
            ``cell_id → final_cell_type``
        confidences : Dict[int, float]
            ``cell_id → confidence score in [0, 1]``
        per_method : Dict[int, Dict]
            ``cell_id → {method_name: assigned_type}``
        """
        if not predictions:
            raise ValueError("No predictions provided to the ensemble.")

        strategy_fn = {
            "soft_vote":         self._soft_vote,
            "weighted_majority": self._weighted_majority,
            "borda_count":       self._borda_count,
        }.get(strategy)

        if strategy_fn is None:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: soft_vote, weighted_majority, borda_count."
            )

        return strategy_fn(predictions)

    # ── Strategies ────────────────────────────────────────────────────────────

    def _soft_vote(self, predictions: Predictions) -> EnsembleResult:
        """
        Weighted average of posterior probabilities.

        P(type|cell) = Σ_m  w_m · P_m(type|cell)  / Σ_m w_m

        where w_m is the weight for method m (default 1.0).
        """
        all_cell_ids = self._all_cell_ids(predictions)
        method_names = list(predictions.keys())

        final_types: Dict[int, str] = {}
        confidences: Dict[int, float] = {}
        per_method: Dict[int, Dict] = {}

        for cell_id in all_cell_ids:
            all_types = self._all_cell_types_for_cell(predictions, cell_id)
            if not all_types:
                final_types[cell_id] = self.UNRESOLVED_LABEL
                confidences[cell_id] = 0.0
                per_method[cell_id] = {}
                continue

            # Weighted average over methods
            agg: Dict[str, float] = {t: 0.0 for t in all_types}
            total_weight = 0.0
            method_calls: Dict[str, str] = {}

            for m_name in method_names:
                cell_probs = predictions[m_name].get(cell_id, {})
                if not cell_probs:
                    continue
                w = self.method_weights.get(m_name, 1.0)
                for t, p in cell_probs.items():
                    agg[t] = agg.get(t, 0.0) + w * p
                total_weight += w

                # Record what this method called (argmax)
                method_calls[m_name] = max(cell_probs, key=cell_probs.get)

            if total_weight == 0:
                final_types[cell_id] = self.UNRESOLVED_LABEL
                confidences[cell_id] = 0.0
                per_method[cell_id] = {}
                continue

            for t in agg:
                agg[t] /= total_weight

            winner = max(agg, key=agg.get)
            winning_prob = agg[winner]

            if winning_prob < self.min_probability_threshold:
                winner = self.UNRESOLVED_LABEL

            # Confidence = fraction of methods agreeing with winner
            agreeing = sum(1 for mc in method_calls.values() if mc == winner)
            conf = agreeing / len(method_calls) if method_calls else 0.0

            final_types[cell_id] = winner
            confidences[cell_id] = float(conf)
            per_method[cell_id] = method_calls

        return final_types, confidences, per_method

    def _weighted_majority(self, predictions: Predictions) -> EnsembleResult:
        """
        Winner-take-all with user-assigned weights.

        Each method casts a weighted vote for its argmax type.
        Confidence = fraction of total weight held by the winning type.
        """
        all_cell_ids = self._all_cell_ids(predictions)
        method_names = list(predictions.keys())

        final_types: Dict[int, str] = {}
        confidences: Dict[int, float] = {}
        per_method: Dict[int, Dict] = {}

        for cell_id in all_cell_ids:
            vote_tally: Dict[str, float] = {}
            method_calls: Dict[str, str] = {}
            total_weight = 0.0

            for m_name in method_names:
                cell_probs = predictions[m_name].get(cell_id, {})
                if not cell_probs:
                    continue
                w = self.method_weights.get(m_name, 1.0)
                winner_type = max(cell_probs, key=cell_probs.get)
                vote_tally[winner_type] = vote_tally.get(winner_type, 0.0) + w
                method_calls[m_name] = winner_type
                total_weight += w

            if not vote_tally or total_weight == 0:
                final_types[cell_id] = self.UNRESOLVED_LABEL
                confidences[cell_id] = 0.0
                per_method[cell_id] = {}
                continue

            winner = max(vote_tally, key=vote_tally.get)
            conf = vote_tally[winner] / total_weight

            if conf < self.min_probability_threshold:
                winner = self.UNRESOLVED_LABEL

            final_types[cell_id] = winner
            confidences[cell_id] = float(conf)
            per_method[cell_id] = method_calls

        return final_types, confidences, per_method

    def _borda_count(self, predictions: Predictions) -> EnsembleResult:
        """
        Rank-based Borda count aggregation.

        Each method ranks cell types by descending probability.  A cell type
        receives ``(N_types - rank)`` Borda points per method.  The type with
        the highest total Borda points wins.

        Robust to probability miscalibration — useful when mixing ASTIR
        (which produces calibrated Bayesian posteriors) with CELESTA
        (which produces stub 0/1 probabilities).
        """
        all_cell_ids = self._all_cell_ids(predictions)
        method_names = list(predictions.keys())

        final_types: Dict[int, str] = {}
        confidences: Dict[int, float] = {}
        per_method: Dict[int, Dict] = {}

        for cell_id in all_cell_ids:
            all_types = self._all_cell_types_for_cell(predictions, cell_id)
            if not all_types:
                final_types[cell_id] = self.UNRESOLVED_LABEL
                confidences[cell_id] = 0.0
                per_method[cell_id] = {}
                continue

            n_types = len(all_types)
            borda_scores: Dict[str, float] = {t: 0.0 for t in all_types}
            method_calls: Dict[str, str] = {}

            for m_name in method_names:
                cell_probs = predictions[m_name].get(cell_id, {})
                if not cell_probs:
                    continue
                w = self.method_weights.get(m_name, 1.0)
                ranked = sorted(cell_probs, key=cell_probs.get, reverse=True)
                for rank, ct in enumerate(ranked):
                    borda_scores[ct] = borda_scores.get(ct, 0.0) + w * (n_types - rank)
                method_calls[m_name] = ranked[0]

            if not method_calls:
                final_types[cell_id] = self.UNRESOLVED_LABEL
                confidences[cell_id] = 0.0
                per_method[cell_id] = {}
                continue

            winner = max(borda_scores, key=borda_scores.get)
            total_borda = sum(borda_scores.values()) or 1.0
            conf_borda = borda_scores[winner] / total_borda

            # Map borda dominance to agreement-like confidence
            agreeing = sum(1 for mc in method_calls.values() if mc == winner)
            conf = agreeing / len(method_calls)

            final_types[cell_id] = winner
            confidences[cell_id] = float(conf)
            per_method[cell_id] = method_calls

        return final_types, confidences, per_method

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _all_cell_ids(predictions: Predictions) -> List[int]:
        ids: set = set()
        for method_preds in predictions.values():
            ids.update(method_preds.keys())
        return sorted(ids)

    @staticmethod
    def _all_cell_types_for_cell(
        predictions: Predictions, cell_id: int
    ) -> List[str]:
        types: set = set()
        for method_preds in predictions.values():
            types.update(method_preds.get(cell_id, {}).keys())
        return sorted(types)

    # ── Post-processing helpers ────────────────────────────────────────────────

    def results_to_dataframe(
        self,
        features: pd.DataFrame,
        final_types: Dict[int, str],
        confidences: Dict[int, float],
        per_method: Dict[int, Dict],
    ) -> pd.DataFrame:
        """
        Merge ensemble results back into the features DataFrame.

        Adds columns:
          ``cell_type``             — final assigned label
          ``cell_type_confidence``  — inter-method agreement [0, 1]
          ``<method>_cell_type``    — per-method call (one column per method)

        Returns a copy of *features* with the new columns appended.
        """
        df = features.copy()
        id_col = "ObjectNumber" if "ObjectNumber" in df.columns else df.index

        def _lookup(cell_id: int, d: dict, default):
            return d.get(int(cell_id), default)

        df["cell_type"] = [
            _lookup(cid, final_types, self.UNRESOLVED_LABEL)
            for cid in (df[id_col.name] if hasattr(id_col, "name") else df.index)
        ]
        df["cell_type_confidence"] = [
            _lookup(cid, confidences, 0.0)
            for cid in (df[id_col.name] if hasattr(id_col, "name") else df.index)
        ]

        # Per-method breakdown columns
        all_methods: set = set()
        for breakdown in per_method.values():
            all_methods.update(breakdown.keys())
        for m_name in sorted(all_methods):
            df[f"{m_name}_cell_type"] = [
                per_method.get(
                    int(cid) if hasattr(id_col, "name") else i, {}
                ).get(m_name, "")
                for i, cid in enumerate(
                    df[id_col.name] if hasattr(id_col, "name") else df.index
                )
            ]

        return df
