"""
IRIS Phenotyper Base Contract
=============================
Defines the canonical prediction result dataclass and abstract base class
for all single-cell phenotyping methods in ``iris_phenotyper``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class PhenotypingPredictionResult:
    """Standardized output returned by every ``PhenotyperMethodBase.predict()`` call.

    Attributes
    ----------
    method_name:
        Identifier of the phenotyping method (e.g., ``"MAPS"``, ``"STELLAR"``).
    labels:
        Predicted cell-type labels indexed by integer ``ObjectNumber``.
    probabilities:
        Optional per-class posterior/confidence matrix indexed by ``ObjectNumber``
        with columns corresponding to cell-type names and values in ``[0, 1]``.
    confidence:
        Per-cell confidence or maximum posterior score indexed by ``ObjectNumber``.
    is_novel:
        Optional boolean Series indexed by ``ObjectNumber`` indicating novel
        or out-of-distribution cell populations (e.g., STELLAR novel discovery).
    marker_positivity:
        Optional per-cell, per-marker positivity score table indexed by
        ``ObjectNumber`` (e.g., Nimbus marker positivity probabilities).
    metadata:
        Free-form dictionary for run metadata (timings, checkpoint path, device, etc.).
    """

    method_name: str
    labels: pd.Series
    probabilities: Optional[pd.DataFrame]
    confidence: pd.Series
    is_novel: Optional[pd.Series] = None
    marker_positivity: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate index alignment, value ranges, and structural invariants."""
        if not isinstance(self.labels, pd.Series):
            raise TypeError(f"labels must be a pd.Series, got {type(self.labels).__name__}")
        if not isinstance(self.confidence, pd.Series):
            raise TypeError(
                f"confidence must be a pd.Series, got {type(self.confidence).__name__}"
            )
        if not self.labels.index.equals(self.confidence.index):
            raise ValueError("labels and confidence indices must match exactly.")

        if self.probabilities is not None:
            if not isinstance(self.probabilities, pd.DataFrame):
                raise TypeError(
                    "probabilities must be a pd.DataFrame or None, "
                    f"got {type(self.probabilities).__name__}"
                )
            if not self.probabilities.index.equals(self.labels.index):
                raise ValueError("probabilities index must match labels index.")
            if not self.probabilities.empty:
                vals = self.probabilities.to_numpy(dtype=float, copy=False)
                if not np.all(np.isfinite(vals)):
                    raise ValueError("probabilities contains non-finite values.")
                if np.any(vals < -1e-6) or np.any(vals > 1.0 + 1e-6):
                    raise ValueError("probabilities values must lie in [0, 1].")

        if self.is_novel is not None:
            if not isinstance(self.is_novel, pd.Series):
                raise TypeError(
                    f"is_novel must be a pd.Series or None, got {type(self.is_novel).__name__}"
                )
            if not self.is_novel.index.equals(self.labels.index):
                raise ValueError("is_novel index must match labels index.")

        if self.marker_positivity is not None:
            if not isinstance(self.marker_positivity, pd.DataFrame):
                raise TypeError(
                    "marker_positivity must be a pd.DataFrame or None, "
                    f"got {type(self.marker_positivity).__name__}"
                )
            if not self.marker_positivity.index.equals(self.labels.index):
                raise ValueError("marker_positivity index must match labels index.")

    def to_dataframe(
        self,
        include_probabilities: bool = True,
        prob_prefix: str = "prob_",
    ) -> pd.DataFrame:
        """Convert the prediction result into a cell-level summary DataFrame."""
        out = pd.DataFrame(
            {
                "ObjectNumber": pd.Index(self.labels.index).astype(int),
                "cell_type": self.labels.astype(str).to_numpy(),
                "confidence": pd.to_numeric(self.confidence, errors="coerce").to_numpy(),
                "method_name": self.method_name,
            }
        )
        if self.is_novel is not None:
            out["is_novel"] = self.is_novel.astype(bool).to_numpy()
        if include_probabilities and self.probabilities is not None and not self.probabilities.empty:
            for col in self.probabilities.columns:
                out[f"{prob_prefix}{col}"] = self.probabilities[col].to_numpy(dtype=float)
        return out

    def to_legacy_dict(self) -> Dict[int, Dict[str, float]]:
        """Convert predictions to legacy ``{cell_id: {cell_type: probability}}`` mapping."""
        result: Dict[int, Dict[str, float]] = {}
        if self.probabilities is not None and not self.probabilities.empty:
            cols = [str(c) for c in self.probabilities.columns]
            for cid, row in zip(self.probabilities.index, self.probabilities.to_numpy(dtype=float)):
                result[int(cid)] = dict(zip(cols, (float(v) for v in row)))
            return result

        for cid, label, conf in zip(
            self.labels.index,
            self.labels.astype(str).to_numpy(),
            pd.to_numeric(self.confidence, errors="coerce").fillna(0.0).to_numpy(),
        ):
            result[int(cid)] = {label: float(conf)}
        return result


class PhenotyperMethodBase(ABC):
    """Abstract base class for all IRIS single-cell phenotyping methods."""

    VALID_MODALITIES: ClassVar[tuple[str, ...]] = (
        "tabular",
        "spatial_tabular",
        "image_and_mask",
    )

    name: str = "BasePhenotyper"
    input_modality: str = "tabular"
    requires_training: bool = False

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if all runtime dependencies for this method are available."""

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Return default hyperparameters and configuration options."""

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Fit or configure the phenotyper on training or reference data."""

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> PhenotypingPredictionResult:
        """Run inference and return a ``PhenotypingPredictionResult``."""

    @staticmethod
    def extract_object_ids(df: pd.DataFrame, id_col: str = "ObjectNumber") -> pd.Index:
        """Extract canonical 1-indexed integer ObjectNumber index from a cell table."""
        if id_col in df.columns:
            ids = pd.to_numeric(df[id_col], errors="coerce").fillna(0).astype(int)
            return pd.Index(ids, name="ObjectNumber")
        for alias in ("object_number", "Cell_ID", "cell_id", "id"):
            if alias in df.columns:
                ids = pd.to_numeric(df[alias], errors="coerce").fillna(0).astype(int)
                return pd.Index(ids, name="ObjectNumber")
        return pd.Index(np.arange(1, len(df) + 1, dtype=int), name="ObjectNumber")
