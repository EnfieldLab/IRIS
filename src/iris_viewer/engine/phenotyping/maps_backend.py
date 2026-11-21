"""IRIS-Phenotyper MAPS Backend.

MAPS (Machine learning for Analysis of Proteomics in Spatial biology; Shaban et al.,
Nature Communications 2024, https://github.com/mahmoodlab/MAPS) is a supervised
feed-forward neural network (4-layer MLP with ReLU and Dropout) trained with
z-score feature standardization and class-balanced sampling.

Bridges ``fit()`` and ``predict()`` to ``iris_phenotyper.maps.MAPSPhenotyper`` when
trained on a reference column or backed by a ``.pt`` checkpoint, while preserving
deterministic rule-based fallback when invoked untrained with only a ``CellTypePanel``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from iris_viewer.core.types import CellTypePanel
from iris_viewer.engine.phenotyping.base import PhenotypeBackend


class MAPSBackend(PhenotypeBackend):
    """Phenotyping backend bridging the mahmoodlab/MAPS feed-forward MLP pipeline.

    Attributes
    ----------
    name : str
        ``"MAPS"``
    requires_training : bool
        False — supports rule-based fallback without prior training, or MLP
        training/checkpoint inference via ``MAPSPhenotyper``.
    requires_raw_image : bool
        False.
    requires_spatial : bool
        True.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "MAPS"
        self.requires_training = False
        self.requires_raw_image = False
        self.requires_spatial = True
        self._phenotyper: Any = None
        self._checkpoint_path: Optional[str] = None
        self.last_result_: Any = None
        self.last_probabilities_: Optional[pd.DataFrame] = None

    def is_available(self) -> bool:
        """Check whether PyTorch and the IRIS MAPS MLP pipeline are importable."""
        try:
            from iris_phenotyper.maps import is_maps_available

            return bool(is_maps_available())
        except Exception:
            return False

    def get_default_params(self) -> Dict[str, Any]:
        """Return default hyperparameters for MAPS MLP training and inference."""
        return {
            "hidden_dim": 512,
            "dropout": 0.10,
            "max_epochs": 50,
            "min_epochs": 10,
            "patience": 20,
            "learning_rate": 1e-3,
            "batch_size": 128,
            "scale_divisor": 255.0,
            "device": "auto",
        }

    @staticmethod
    def _cell_ids_from_features(features: pd.DataFrame) -> List[int]:
        for col in ("ObjectNumber", "Cell_ID", "cell_id"):
            if col in features.columns:
                return features[col].astype(int).tolist()
        return list(range(len(features)))

    @staticmethod
    def _sanitize_feature_df(
        features: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        df = features.copy()
        cols = list(feature_cols) if feature_cols is not None else [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        for col in cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
                if not np.all(np.isfinite(arr)):
                    df[col] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return df

    def fit(
        self,
        features: pd.DataFrame,
        labels: Optional[Any] = None,
        cell_type_panel: Optional[CellTypePanel] = None,
        **kwargs: Any,
    ) -> None:
        """Fit the MAPS 4-layer feed-forward MLP on single-cell features and labels."""
        from iris_phenotyper.maps import MAPSPhenotyper

        if features.empty:
            raise ValueError("Cannot fit MAPSBackend on an empty DataFrame.")

        params = dict(kwargs.pop("params", {}) or {})
        merged = {**params, **kwargs}

        label_col = str(merged.pop("label_col", "cell_type"))
        hidden_dim = int(merged.pop("hidden_dim", 512))
        dropout = float(merged.pop("dropout", 0.10))
        max_epochs = int(merged.pop("max_epochs", merged.pop("epochs", 50)))
        min_epochs = int(merged.pop("min_epochs", min(max_epochs, 10)))
        patience = int(merged.pop("patience", max(10, max_epochs // 2)))
        learning_rate = float(merged.pop("learning_rate", merged.pop("lr", 1e-3)))
        batch_size = int(merged.pop("batch_size", 128))
        scale_divisor = float(merged.pop("scale_divisor", 255.0))
        device = str(merged.pop("device", "auto"))
        val_fraction = float(merged.pop("val_fraction", 0.2))
        feature_cols = merged.pop("feature_cols", None)
        class_names = merged.pop("class_names", None)
        include_cell_size = bool(merged.pop("include_cell_size", True))
        use_balanced = bool(merged.pop("use_class_balanced_sampler", True))

        df_clean = self._sanitize_feature_df(features, feature_cols=feature_cols)

        phenotyper = MAPSPhenotyper(
            feature_cols=feature_cols,
            class_names=class_names,
            include_cell_size=include_cell_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            patience=patience,
            scale_divisor=scale_divisor,
            device=device,
            use_class_balanced_sampler=use_balanced,
            val_fraction=val_fraction,
        )
        phenotyper.fit(
            df_clean,
            labels=labels,
            label_col=label_col,
            feature_cols=feature_cols,
            class_names=class_names,
        )
        self._phenotyper = phenotyper
        self._checkpoint_path = None
        self.last_result_ = None
        self.last_probabilities_ = None

    def predict(
        self,
        features: pd.DataFrame,
        cell_types: Optional[CellTypePanel] = None,
        params: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[int, Dict[str, float]]:
        """Predict cell types using fitted/checkpoint MAPS MLP or rule-based fallback.

        Args:
            features: Single-cell feature DataFrame.
            cell_types: CellTypePanel specifying marker rules for fallback mode.
            params: Optional model parameter overrides.
            checkpoint_path: Optional path to a saved MAPS ``.pt`` checkpoint.
            **kwargs: Extra parameters passed by caller.

        Returns:
            Dictionary mapping cell ID to probability distribution over cell types.
        """
        merged_params = dict(params or {})
        ckpt = checkpoint_path or kwargs.get("checkpoint_path") or merged_params.get("checkpoint_path")
        panel = cell_types or kwargs.get("cell_type_panel") or kwargs.get("panel")

        if ckpt:
            from iris_phenotyper.maps import MAPSPhenotyper

            ckpt_str = str(Path(ckpt))
            if self._phenotyper is None or self._checkpoint_path != ckpt_str:
                device = str(kwargs.get("device", merged_params.get("device", "auto")))
                batch_size = int(kwargs.get("batch_size", merged_params.get("batch_size", 128)))
                self._phenotyper = MAPSPhenotyper.from_checkpoint(
                    ckpt_str,
                    device=device,
                    batch_size=batch_size,
                )
                self._checkpoint_path = ckpt_str

        if self._phenotyper is not None:
            feat_cols = getattr(self._phenotyper, "feature_names_", None)
            df_clean = self._sanitize_feature_df(features, feature_cols=feat_cols)
            batch_size = int(
                kwargs.get(
                    "batch_size",
                    merged_params.get("batch_size", self._phenotyper.config.batch_size),
                )
            )
            res = self._phenotyper.predict(df_clean, batch_size=batch_size)
            self.last_result_ = res
            self.last_probabilities_ = res.probabilities
            cell_ids = self._cell_ids_from_features(features)
            probs_df = res.probabilities
            out: Dict[int, Dict[str, float]] = {}
            if probs_df is not None:
                for i, cid in enumerate(cell_ids):
                    if i < len(probs_df):
                        row = probs_df.iloc[i]
                        out[int(cid)] = {str(ct): float(p) for ct, p in row.items()}
                    else:
                        out[int(cid)] = {}
            return out

        self.last_result_ = None
        self.last_probabilities_ = None
        return self.predict_fallback(features, cell_type_panel=panel)

    def predict_fallback(
        self,
        features: pd.DataFrame,
        **kwargs: Any,
    ) -> Dict[int, Dict[str, float]]:
        """Fallback deterministic rule-based assignment when MAPS MLP is untrained."""
        cell_types = (
            kwargs.get("cell_type_panel")
            or kwargs.get("cell_types")
            or kwargs.get("panel")
        )
        if not cell_types or not cell_types.rules:
            return {}

        predictions: Dict[int, Dict[str, float]] = {}
        if features.empty:
            return predictions

        lower_cols = {str(c).lower(): c for c in features.columns}
        prob_rows: List[Dict[str, float]] = []
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
                    col = next(
                        (c for c in (f"{marker}_mean", marker) if c in features.columns),
                        lower_cols.get(f"{marker.lower()}_mean")
                        or lower_cols.get(marker.lower()),
                    )
                    if col is not None:
                        val = row[col]
                        if val < 0.1:
                            valid = False
                            break
                        score += float(val)

                if valid and score > best_score:
                    best_score = score
                    best_type = ct_name

            prob_dict = {ct: (1.0 if ct == best_type else 0.0) for ct in cell_types.cell_types}
            prob_dict["Unresolved"] = 1.0 if best_type == "Unresolved" else 0.0
            predictions[cell_id] = prob_dict
            prob_rows.append(prob_dict)

        self.last_probabilities_ = pd.DataFrame(prob_rows)
        return predictions
