"""
IRIS Phenotyper Adapter for MAPS
================================
Wraps ``MAPSTrainer`` behind ``PhenotyperMethodBase`` and emits canonical
``PhenotypingPredictionResult`` objects.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from iris_phenotyper.base import PhenotyperMethodBase, PhenotypingPredictionResult
from iris_phenotyper.maps.pipeline import (
    DEFAULT_SCALE_DIVISOR,
    DEFAULT_SEED,
    MAPSTrainer,
    MAPSTrainingConfig,
    is_maps_available,
)

_CELL_SIZE_ALIASES: tuple[str, ...] = ("cellSize", "cell_size", "area", "Area")

_NON_FEATURE_COLS_LOWER: frozenset[str] = frozenset(
    {
        "objectnumber",
        "object_number",
        "cell_id",
        "cellid",
        "celllabel",
        "id",
        "image",
        "image_name",
        "frame_name",
        "file",
        "spots",
        "pos_x",
        "pos_y",
        "x_cent",
        "y_cent",
        "x",
        "y",
        "centroid-0",
        "centroid-1",
        "centroid_x",
        "centroid_y",
        "cell_type",
        "cell_label",
        "annotation",
        "annotation_num",
        "phenotype_label",
        "clustername",
        "celltype",
        "tile_name",
        "pointnum",
        "group",
        "label",
        "perimeter",
        "eccentricity",
        "solidity",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "extent",
        "equivalent_diameter_area",
    }
)


class MAPSPhenotyper(PhenotyperMethodBase):
    """IRIS phenotyping adapter for mahmoodlab/MAPS MLP."""

    name: str = "MAPS"
    input_modality: str = "tabular"
    requires_training: bool = True

    def __init__(
        self,
        *,
        checkpoint_path: Optional[Union[str, Path]] = None,
        class_names: Optional[Sequence[str]] = None,
        feature_cols: Optional[Sequence[str]] = None,
        include_cell_size: bool = True,
        cell_size_col: Optional[str] = None,
        hidden_dim: int = 512,
        dropout: float = 0.10,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        max_epochs: int = 500,
        min_epochs: int = 250,
        patience: int = 100,
        scale_divisor: float = DEFAULT_SCALE_DIVISOR,
        seed: int = DEFAULT_SEED,
        num_workers: int = 0,
        device: str = "auto",
        dtype: str = "auto",
        use_class_balanced_sampler: bool = True,
        val_fraction: float = 0.2,
        verbose: int = 0,
        results_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.checkpoint_path: Optional[Path] = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self.include_cell_size = bool(include_cell_size)
        self.cell_size_col = cell_size_col
        self.results_dir = Path(results_dir) if results_dir is not None else None

        self.config = MAPSTrainingConfig(
            hidden_dim=hidden_dim,
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            patience=patience,
            scale_divisor=scale_divisor,
            seed=seed,
            num_workers=num_workers,
            device=device,
            dtype=dtype,
            use_class_balanced_sampler=use_class_balanced_sampler,
            val_fraction=val_fraction,
            verbose=verbose,
        )

        self.class_names_: Optional[List[str]] = (
            [str(c) for c in class_names] if class_names is not None else None
        )
        self.feature_names_: Optional[List[str]] = (
            [str(c) for c in feature_cols] if feature_cols is not None else None
        )
        self.trainer_: Optional[MAPSTrainer] = None
        self.fit_summary_: Dict[str, Any] = {}

        if self.checkpoint_path is not None:
            self._load_from_checkpoint(self.checkpoint_path)

    def is_available(self) -> bool:
        """Return True if PyTorch and required scientific libraries are importable."""
        return is_maps_available()

    def get_default_params(self) -> Dict[str, Any]:
        """Return default MAPS hyperparameters and adapter options."""
        params = self.config.to_dict()
        params["include_cell_size"] = self.include_cell_size
        params["cell_size_col"] = self.cell_size_col
        return params

    def _find_cell_size_col(self, df: pd.DataFrame) -> Optional[str]:
        if self.cell_size_col is not None:
            return self.cell_size_col if self.cell_size_col in df.columns else None
        col_lower = {c.lower(): c for c in df.columns}
        for alias in _CELL_SIZE_ALIASES:
            if alias in df.columns:
                return alias
            if alias.lower() in col_lower:
                return col_lower[alias.lower()]
        return None

    def _detect_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect marker intensity columns and optional cell size/area column."""
        mean_cols = [c for c in df.columns if c.endswith("_mean") and ptypes.is_numeric_dtype(df[c])]
        size_col = self._find_cell_size_col(df) if self.include_cell_size else None

        if mean_cols:
            cols = list(mean_cols)
            if size_col is not None and size_col not in cols:
                cols.append(size_col)
            return cols

        size_aliases_lower = {a.lower() for a in _CELL_SIZE_ALIASES}
        cols = []
        for col in df.columns:
            low = col.lower()
            if low in _NON_FEATURE_COLS_LOWER or low in size_aliases_lower:
                continue
            if ptypes.is_numeric_dtype(df[col]):
                cols.append(col)

        if size_col is not None and size_col not in cols:
            cols.append(size_col)

        if not cols:
            raise ValueError(
                "No suitable numeric marker/size feature columns found in DataFrame. "
                f"Columns: {list(df.columns)[:20]}"
            )
        return cols

    def _resolve_columns_for_df(self, df: pd.DataFrame, expected: Sequence[str]) -> List[str]:
        """Align expected feature names onto ``df``, bridging IRIS and MAPS column aliases."""
        df_cols = set(df.columns)
        size_aliases_lower = {a.lower() for a in _CELL_SIZE_ALIASES}
        resolved: List[str] = []
        missing: List[str] = []

        for feat in expected:
            if feat in df_cols:
                resolved.append(feat)
                continue
            if not feat.endswith("_mean") and f"{feat}_mean" in df_cols:
                resolved.append(f"{feat}_mean")
                continue
            if feat.endswith("_mean") and feat[:-5] in df_cols:
                resolved.append(feat[:-5])
                continue
            if feat.lower() in size_aliases_lower:
                size_match = self._find_cell_size_col(df)
                if size_match is not None:
                    resolved.append(size_match)
                    continue
            missing.append(feat)

        if missing:
            raise ValueError(
                f"Missing required feature columns for MAPS inference: {missing}. "
                f"Available columns: {list(df.columns)[:25]}"
            )
        return resolved

    def _extract_feature_matrix(
        self,
        df: pd.DataFrame,
        expected: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        if expected is not None:
            cols = self._resolve_columns_for_df(df, expected)
            canonical_names = list(expected)
        else:
            cols = self._detect_feature_columns(df)
            canonical_names = list(cols)

        X = df[cols].to_numpy(dtype=np.float64)
        if X.size > 0 and not np.all(np.isfinite(X)):
            raise ValueError("Feature matrix contains NaN or Inf values.")
        return X, canonical_names

    @staticmethod
    def _extract_label_array(
        features: pd.DataFrame,
        labels: Optional[Union[pd.Series, Sequence[Any], np.ndarray]],
        label_col: str,
    ) -> np.ndarray:
        if labels is not None:
            arr = np.asarray(labels, dtype=object).ravel()
            if len(arr) != len(features):
                raise ValueError(
                    f"Length mismatch: features has {len(features)} rows, labels has {len(arr)}."
                )
            return arr

        for col in (label_col, "cell_type", "cell_label"):
            if col in features.columns:
                return features[col].to_numpy(dtype=object)

        raise ValueError(
            f"No labels provided and none of ('{label_col}', 'cell_type', 'cell_label') "
            "exist in features DataFrame."
        )

    def _encode_labels(
        self,
        raw_labels: np.ndarray,
        class_names: Optional[Sequence[str]],
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Filter missing labels and map arbitrary labels to contiguous integers [0, K-1]."""
        valid_mask = ~pd.isna(raw_labels)
        if not np.any(valid_mask):
            raise ValueError("Training labels contain only NaN/missing values.")

        clean = raw_labels[valid_mask]
        is_int_like = all(
            isinstance(v, (int, np.integer)) or (isinstance(v, (float, np.floating)) and v.is_integer())
            for v in clean
        )

        if class_names is not None:
            ordered_names = [str(c) for c in class_names]
            if is_int_like:
                int_vals = np.asarray([int(v) for v in clean], dtype=np.int64)
                if np.min(int_vals) < 0 or np.max(int_vals) >= len(ordered_names):
                    raise ValueError(
                        f"Integer labels out of range for class_names of length {len(ordered_names)}."
                    )
                return int_vals, ordered_names, valid_mask
            name_to_id = {name: i for i, name in enumerate(ordered_names)}
            str_vals = [str(v) for v in clean]
            unknown = sorted(set(str_vals) - set(name_to_id))
            if unknown:
                raise ValueError(f"Training labels contain unknown classes: {unknown}")
            encoded = np.asarray([name_to_id[s] for s in str_vals], dtype=np.int64)
            return encoded, ordered_names, valid_mask

        if is_int_like:
            int_vals = np.asarray([int(v) for v in clean], dtype=np.int64)
            unique_ids = sorted(set(int_vals.tolist()))
            if unique_ids == list(range(len(unique_ids))):
                ordered_names = [str(i) for i in unique_ids]
                return int_vals, ordered_names, valid_mask
            id_map = {orig: idx for idx, orig in enumerate(unique_ids)}
            ordered_names = [str(orig) for orig in unique_ids]
            encoded = np.asarray([id_map[int(v)] for v in int_vals], dtype=np.int64)
            return encoded, ordered_names, valid_mask

        str_vals = [str(v) for v in clean]
        ordered_names = sorted(set(str_vals))
        name_to_id = {name: i for i, name in enumerate(ordered_names)}
        encoded = np.asarray([name_to_id[s] for s in str_vals], dtype=np.int64)
        return encoded, ordered_names, valid_mask

    def fit(
        self,
        features: pd.DataFrame,
        labels: Optional[Union[pd.Series, Sequence[Any], np.ndarray]] = None,
        *,
        label_col: str = "cell_type",
        valid_features: Optional[pd.DataFrame] = None,
        valid_labels: Optional[Union[pd.Series, Sequence[Any], np.ndarray]] = None,
        feature_cols: Optional[Sequence[str]] = None,
        class_names: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> "MAPSPhenotyper":
        """Fit the MAPS MLP classifier on a single-cell feature DataFrame.

        Parameters
        ----------
        features:
            Training cell table with marker expression columns and optional cell size.
        labels:
            Target cell-type labels aligned with ``features``. If omitted, reads
            ``label_col`` (or ``"cell_type"`` / ``"cell_label"``) from ``features``.
        valid_features:
            Optional explicit validation cell table. If omitted, performs a
            stratified validation split controlled by ``val_fraction``.
        valid_labels:
            Optional validation labels corresponding to ``valid_features``.
        feature_cols:
            Optional explicit ordered list of feature columns.
        class_names:
            Optional explicit ordered list of class names.
        """
        if features.empty:
            raise ValueError("Cannot fit MAPSPhenotyper on an empty DataFrame.")

        raw_train_labels = self._extract_label_array(features, labels, label_col)
        target_classes = class_names if class_names is not None else self.class_names_
        y_train, resolved_classes, valid_mask = self._encode_labels(raw_train_labels, target_classes)
        train_df = features.loc[valid_mask].reset_index(drop=True)

        expected_feats = feature_cols if feature_cols is not None else self.feature_names_
        X_train, resolved_feats = self._extract_feature_matrix(train_df, expected=expected_feats)

        X_valid: Optional[np.ndarray] = None
        y_valid: Optional[np.ndarray] = None
        if valid_features is not None:
            if valid_features.empty:
                raise ValueError("valid_features cannot be empty when provided.")
            raw_val_labels = self._extract_label_array(valid_features, valid_labels, label_col)
            y_valid, _, val_mask = self._encode_labels(raw_val_labels, resolved_classes)
            val_df = valid_features.loc[val_mask].reset_index(drop=True)
            X_valid, _ = self._extract_feature_matrix(val_df, expected=resolved_feats)

        self.class_names_ = resolved_classes
        self.feature_names_ = resolved_feats

        trainer = MAPSTrainer(
            num_features=len(resolved_feats),
            num_classes=len(resolved_classes),
            config=self.config,
            results_dir=self.results_dir,
        )
        self.fit_summary_ = trainer.fit(
            X_train,
            y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            class_names=resolved_classes,
            feature_names=resolved_feats,
        )
        self.trainer_ = trainer
        return self

    def predict(
        self,
        features: pd.DataFrame,
        **kwargs: Any,
    ) -> PhenotypingPredictionResult:
        """Predict cell types and posterior probabilities for a single-cell table."""
        if self.trainer_ is None:
            raise RuntimeError("MAPSPhenotyper must be fitted or loaded from checkpoint first.")

        object_ids = self.extract_object_ids(features)
        class_names = self.class_names_ or [str(i) for i in range(self.trainer_.num_classes)]

        if features.empty:
            empty_labels = pd.Series(pd.Index([], dtype=object), index=object_ids, name="cell_type")
            empty_probs = pd.DataFrame(
                np.empty((0, len(class_names)), dtype=float),
                index=object_ids,
                columns=class_names,
            )
            empty_conf = pd.Series(np.empty((0,), dtype=float), index=object_ids, name="confidence")
            res = PhenotypingPredictionResult(
                method_name=self.name,
                labels=empty_labels,
                probabilities=empty_probs,
                confidence=empty_conf,
                metadata={
                    "device": str(self.trainer_.device),
                    "dtype": str(self.trainer_.dtype),
                    "num_features": self.trainer_.num_features,
                    "num_classes": self.trainer_.num_classes,
                },
            )
            res.validate()
            return res

        t0 = time.time()
        X, used_features = self._extract_feature_matrix(features, expected=self.feature_names_)
        if X.shape[1] != self.trainer_.num_features:
            raise ValueError(
                f"Expected {self.trainer_.num_features} features ({self.feature_names_}), "
                f"got {X.shape[1]} ({used_features})."
            )

        batch_size = kwargs.get("batch_size", self.config.batch_size)
        pred_indices, pred_probs = self.trainer_.predict(X, batch_size=batch_size)
        elapsed = time.time() - t0

        pred_probs = np.clip(pred_probs, 0.0, 1.0)
        pred_labels_str = [class_names[int(idx)] for idx in pred_indices]
        max_conf = np.max(pred_probs, axis=1) if pred_probs.shape[1] > 0 else np.zeros(len(X))

        labels_series = pd.Series(pred_labels_str, index=object_ids, name="cell_type")
        probs_df = pd.DataFrame(pred_probs, index=object_ids, columns=class_names)
        conf_series = pd.Series(max_conf, index=object_ids, dtype=float, name="confidence")

        result = PhenotypingPredictionResult(
            method_name=self.name,
            labels=labels_series,
            probabilities=probs_df,
            confidence=conf_series,
            metadata={
                "device": str(self.trainer_.device),
                "dtype": str(self.trainer_.dtype),
                "num_features": self.trainer_.num_features,
                "num_classes": self.trainer_.num_classes,
                "feature_names": list(used_features),
                "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
                "inference_time_sec": elapsed,
            },
        )
        result.validate()
        return result

    def save(self, path: Union[str, Path]) -> Path:
        """Persist fitted model weights, normalization stats, and class/feature schemas."""
        if self.trainer_ is None:
            raise RuntimeError("Cannot save MAPSPhenotyper before fit() or load().")
        saved = self.trainer_.save_checkpoint(
            path,
            class_names=self.class_names_,
            feature_names=self.feature_names_,
        )
        self.checkpoint_path = saved
        return saved

    def _load_from_checkpoint(self, path: Union[str, Path]) -> None:
        trainer, meta = MAPSTrainer.load_checkpoint(
            path,
            device=self.config.device,
            dtype=self.config.dtype,
            batch_size=self.config.batch_size,
            dropout=self.config.dropout,
        )
        self.trainer_ = trainer
        self.config.hidden_dim = meta["hidden_dim"]
        self.config.dropout = trainer.config.dropout
        self.config.scale_divisor = meta["scale_divisor"]
        if self.class_names_ is None and meta.get("class_names") is not None:
            self.class_names_ = [str(c) for c in meta["class_names"]]
        if self.feature_names_ is None and meta.get("feature_names") is not None:
            self.feature_names_ = [str(c) for c in meta["feature_names"]]

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        *,
        class_names: Optional[Sequence[str]] = None,
        feature_cols: Optional[Sequence[str]] = None,
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 128,
    ) -> "MAPSPhenotyper":
        """Construct a ready-to-predict ``MAPSPhenotyper`` from a saved ``.pt`` checkpoint."""
        return cls(
            checkpoint_path=path,
            class_names=class_names,
            feature_cols=feature_cols,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
