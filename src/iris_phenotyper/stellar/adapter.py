"""
IRIS Phenotyper Adapter for STELLAR
===================================
Wraps :class:`STELLARPipeline` behind :class:`PhenotyperMethodBase` for
unified IRIS single-cell phenotyping, spatial label transfer, and novel
cell-type discovery.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from iris_common.schema import (
    CELL_ID_ALIASES,
    CELL_TYPE_ALIASES,
    COL_CELL_ID,
    COL_CELL_TYPE,
    COL_SAMPLE,
    COL_X,
    COL_Y,
    SAMPLE_COLUMN_CANDIDATES,
    X_COORD_ALIASES,
    Y_COORD_ALIASES,
    find_sample_column,
    resolve_marker_column,
)
from iris_phenotyper.base import PhenotyperMethodBase, PhenotypingPredictionResult
from iris_phenotyper.stellar.pipeline import (
    STELLARConfig,
    STELLARPipeline,
    STELLARResult,
    _import_torch,
)

_NON_FEATURE_NAMES = {
    s.lower()
    for s in (
        *CELL_ID_ALIASES,
        *X_COORD_ALIASES,
        *Y_COORD_ALIASES,
        *SAMPLE_COLUMN_CANDIDATES,
        *CELL_TYPE_ALIASES,
        "unique_region",
        "donor",
        "tissue",
        "cell_type_a",
        "unnamed: 0",
        "index",
        "level_0",
        "confidence",
        "is_novel",
        "method_name",
    )
}


class STELLARPhenotyper(PhenotyperMethodBase):
    """IRIS phenotyper adapter for STELLAR spatial graph label transfer.

    STELLAR is a transductive geometric deep learning method that jointly learns
    cell representations from an annotated spatial reference dataset and an
    unannotated target spatial dataset, transferring known labels and optionally
    discovering novel cell populations via high-entropy cluster seeding.
    """

    name: str = "STELLAR"
    input_modality: str = "spatial_tabular"
    requires_training: bool = True

    def __init__(
        self,
        *,
        distance_thres: float = 50.0,
        epochs: int = 20,
        seed_epochs: int = 20,
        lr: float = 1e-3,
        wd: float = 5e-2,
        hidden_dim: int = 128,
        num_heads: Optional[int] = None,
        num_seed_class: int = 0,
        num_parts: int = 100,
        entropy_weight: float = 0.3,
        margin_scale: float = 10.0,
        seed: int = 1,
        device: str = "auto",
        backend: str = "auto",
        novel_prefix: str = "Novel_",
        clustering_resolution: float = 1.0,
        standardize: bool = False,
        feature_columns: Optional[Sequence[str]] = None,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        region_col: Optional[str] = None,
        label_col: Optional[str] = None,
        id_col: str = COL_CELL_ID,
    ) -> None:
        self.distance_thres = float(distance_thres)
        self.epochs = int(epochs)
        self.seed_epochs = int(seed_epochs)
        self.lr = float(lr)
        self.wd = float(wd)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = num_heads
        self.num_seed_class = int(num_seed_class)
        self.num_parts = int(num_parts)
        self.entropy_weight = float(entropy_weight)
        self.margin_scale = float(margin_scale)
        self.seed = int(seed)
        self.device = str(device)
        self.backend = str(backend)
        self.novel_prefix = str(novel_prefix)
        self.clustering_resolution = float(clustering_resolution)
        self.standardize = bool(standardize)
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
        self.x_col = x_col
        self.y_col = y_col
        self.region_col = region_col
        self.label_col = label_col
        self.id_col = id_col

        self._pipeline: Optional[STELLARPipeline] = None
        self._train_df: Optional[pd.DataFrame] = None
        self._train_label_col: Optional[str] = None
        self._resolved_features: Optional[List[str]] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._cached_target_signature: Optional[Tuple[int, int, int]] = None
        self._cached_prediction: Optional[PhenotypingPredictionResult] = None
        self.last_result_: Optional[STELLARResult] = None

    def is_available(self) -> bool:
        """Return True if PyTorch and core scientific dependencies are importable."""
        try:
            _import_torch()
            import scipy.spatial  # noqa: F401
            import sklearn  # noqa: F401

            return True
        except ImportError:
            return False

    def get_default_params(self) -> Dict[str, Any]:
        """Return default STELLAR hyperparameters."""
        return {
            "distance_thres": 50.0,
            "epochs": 20,
            "seed_epochs": 20,
            "lr": 1e-3,
            "wd": 5e-2,
            "hidden_dim": 128,
            "num_heads": None,
            "num_seed_class": 0,
            "num_parts": 100,
            "entropy_weight": 0.3,
            "margin_scale": 10.0,
            "seed": 1,
            "device": "auto",
            "backend": "auto",
            "novel_prefix": "Novel_",
            "clustering_resolution": 1.0,
            "standardize": False,
        }

    def _build_config(self, overrides: Optional[Dict[str, Any]] = None) -> STELLARConfig:
        params: Dict[str, Any] = {
            "distance_thres": self.distance_thres,
            "epochs": self.epochs,
            "seed_epochs": self.seed_epochs,
            "lr": self.lr,
            "wd": self.wd,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_seed_class": self.num_seed_class,
            "num_parts": self.num_parts,
            "entropy_weight": self.entropy_weight,
            "margin_scale": self.margin_scale,
            "seed": self.seed,
            "device": self.device,
            "backend": self.backend,
            "novel_prefix": self.novel_prefix,
            "clustering_resolution": self.clustering_resolution,
        }
        if overrides:
            for k, v in overrides.items():
                if k in params:
                    params[k] = v
        return STELLARConfig(**params)

    def _resolve_coord_cols(self, df: pd.DataFrame) -> Tuple[str, str]:
        if self.x_col and self.y_col:
            if self.x_col not in df.columns or self.y_col not in df.columns:
                raise ValueError(
                    f"Specified coordinate columns ({self.x_col!r}, {self.y_col!r}) not in df."
                )
            return self.x_col, self.y_col

        lower_map = {str(c).strip().lower(): str(c) for c in df.columns}
        x_name: Optional[str] = None
        y_name: Optional[str] = None
        for cand in (COL_X, *X_COORD_ALIASES):
            if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
                x_name = cand
                break
            if cand.lower() in lower_map and pd.api.types.is_numeric_dtype(
                df[lower_map[cand.lower()]]
            ):
                x_name = lower_map[cand.lower()]
                break
        for cand in (COL_Y, *Y_COORD_ALIASES):
            if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
                y_name = cand
                break
            if cand.lower() in lower_map and pd.api.types.is_numeric_dtype(
                df[lower_map[cand.lower()]]
            ):
                y_name = lower_map[cand.lower()]
                break
        if x_name is None or y_name is None:
            raise ValueError(
                "Could not resolve spatial coordinate columns (expected Pos_X/Pos_Y or x/y)."
            )
        return x_name, y_name

    def _resolve_region_series(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if self.region_col is not None:
            if self.region_col not in df.columns:
                raise ValueError(f"Specified region_col {self.region_col!r} not in DataFrame.")
            return df[self.region_col].to_numpy()
        if "unique_region" in df.columns:
            return df["unique_region"].to_numpy()
        if COL_SAMPLE in df.columns:
            return df[COL_SAMPLE].to_numpy()
        sample_col = find_sample_column(df)
        if sample_col is not None:
            return df[sample_col].to_numpy()
        return None

    def _resolve_label_col(self, df: pd.DataFrame, label_col: Optional[str] = None) -> str:
        explicit = label_col or self.label_col
        if explicit is not None:
            if explicit not in df.columns:
                raise ValueError(f"Label column {explicit!r} not found in training DataFrame.")
            return explicit
        for cand in (COL_CELL_TYPE, "cell_type_A", *CELL_TYPE_ALIASES):
            if cand in df.columns:
                return cand
        lower_map = {str(c).strip().lower(): str(c) for c in df.columns}
        for cand in (COL_CELL_TYPE, "cell_type_a", *CELL_TYPE_ALIASES):
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        raise ValueError(
            "Could not locate cell-type label column in training DataFrame "
            "(expected 'cell_type', 'cell_type_A', or pass label_col=...)."
        )

    def _resolve_feature_cols(
        self,
        train_df: pd.DataFrame,
        target_df: Optional[pd.DataFrame] = None,
        feature_columns: Optional[Sequence[str]] = None,
    ) -> List[str]:
        requested = list(feature_columns) if feature_columns is not None else self.feature_columns
        if requested is not None:
            resolved: List[str] = []
            for marker in requested:
                col = resolve_marker_column(train_df, marker)
                if col is None:
                    raise ValueError(f"Feature column {marker!r} not found in train_df.")
                if target_df is not None and col not in target_df.columns:
                    t_col = resolve_marker_column(target_df, marker)
                    if t_col is None:
                        raise ValueError(f"Feature column {marker!r} not found in target_df.")
                resolved.append(col)
            return resolved

        target_cols = set(map(str, target_df.columns)) if target_df is not None else None
        feats: List[str] = []
        for col in train_df.columns:
            c_str = str(col)
            if c_str.lower() in _NON_FEATURE_NAMES or c_str.startswith("Unnamed:"):
                continue
            if not pd.api.types.is_numeric_dtype(train_df[col]):
                continue
            if target_cols is not None:
                if c_str not in target_cols or not pd.api.types.is_numeric_dtype(target_df[c_str]):
                    continue
            feats.append(c_str)
        if not feats:
            raise ValueError("No numeric marker feature columns found for STELLAR.")
        return feats

    @staticmethod
    def _df_signature(df: pd.DataFrame) -> Tuple[int, int, int]:
        return (id(df), len(df), len(df.columns))

    def _to_prediction_result(
        self,
        target_df: pd.DataFrame,
        raw: STELLARResult,
        feature_cols: List[str],
    ) -> PhenotypingPredictionResult:
        object_ids = self.extract_object_ids(target_df, id_col=self.id_col)
        labels = pd.Series(raw.pred_labels, index=object_ids, name=COL_CELL_TYPE, dtype=object)
        probabilities = pd.DataFrame(
            raw.probabilities,
            index=object_ids,
            columns=raw.class_names,
            dtype=float,
        )
        confidence = pd.Series(raw.confidence, index=object_ids, name="confidence", dtype=float)
        is_novel = pd.Series(raw.is_novel, index=object_ids, name="is_novel", dtype=bool)
        result = PhenotypingPredictionResult(
            method_name=self.name,
            labels=labels,
            probabilities=probabilities,
            confidence=confidence,
            is_novel=is_novel,
            metadata={
                "backend": raw.backend_used,
                "device": raw.device_used,
                "epochs": self.epochs,
                "distance_thres": self.distance_thres,
                "num_known_classes": len(raw.known_classes),
                "known_classes": list(raw.known_classes),
                "num_seed_class": self.num_seed_class,
                "novel_classes": list(raw.novel_classes),
                "n_novel_cells": int(np.sum(raw.is_novel)),
                "mean_uncertainty": float(raw.mean_uncertainty),
                "train_time_sec": float(raw.train_time_sec),
                "embeddings_shape": tuple(raw.embeddings.shape),
                "feature_columns": list(feature_cols),
                "history": dict(raw.history),
            },
        )
        result.validate()
        return result

    def _run_transductive(
        self,
        train_df: pd.DataFrame,
        target_df: pd.DataFrame,
        *,
        feature_columns: Optional[Sequence[str]] = None,
        label_col: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> PhenotypingPredictionResult:
        lbl_col = self._resolve_label_col(train_df, label_col=label_col)
        train_clean = train_df.loc[train_df[lbl_col].notna()].reset_index(drop=True)
        if train_clean.empty:
            raise ValueError("Training DataFrame contains no non-null cell-type labels.")

        feat_cols = self._resolve_feature_cols(
            train_clean, target_df=target_df, feature_columns=feature_columns
        )
        lx_col, ly_col = self._resolve_coord_cols(train_clean)
        tx_col, ty_col = self._resolve_coord_cols(target_df)

        lx_mat = train_clean[feat_cols].to_numpy(dtype=np.float32, copy=True)
        ux_mat = target_df[feat_cols].to_numpy(dtype=np.float32, copy=True)
        if not np.all(np.isfinite(lx_mat)) or not np.all(np.isfinite(ux_mat)):
            lx_mat = np.nan_to_num(lx_mat, nan=0.0, posinf=0.0, neginf=0.0)
            ux_mat = np.nan_to_num(ux_mat, nan=0.0, posinf=0.0, neginf=0.0)

        if self.standardize:
            mean = lx_mat.mean(axis=0, keepdims=True)
            std = lx_mat.std(axis=0, keepdims=True)
            std = np.where(std > 1e-6, std, 1.0)
            lx_mat = (lx_mat - mean) / std
            ux_mat = (ux_mat - mean) / std
            self._feature_mean = mean
            self._feature_std = std
        else:
            self._feature_mean = None
            self._feature_std = None

        l_coords = train_clean[[lx_col, ly_col]].to_numpy(dtype=np.float64)
        u_coords = target_df[[tx_col, ty_col]].to_numpy(dtype=np.float64)
        l_regions = self._resolve_region_series(train_clean)
        u_regions = self._resolve_region_series(target_df)
        l_labels = train_clean[lbl_col].astype(str).tolist()

        cfg = self._build_config(config_overrides)
        pipeline = STELLARPipeline(cfg)
        raw = pipeline.fit_predict(
            labeled_x=lx_mat,
            labeled_y=l_labels,
            unlabeled_x=ux_mat,
            labeled_coords=l_coords,
            unlabeled_coords=u_coords,
            labeled_regions=l_regions,
            unlabeled_regions=u_regions,
        )

        self._pipeline = pipeline
        self._resolved_features = feat_cols
        self.last_result_ = raw
        pred_result = self._to_prediction_result(target_df, raw, feat_cols)
        self._cached_target_signature = self._df_signature(target_df)
        self._cached_prediction = pred_result
        return pred_result

    def fit(
        self,
        train_df: pd.DataFrame,
        target_df: Optional[pd.DataFrame] = None,
        *,
        feature_columns: Optional[Sequence[str]] = None,
        label_col: Optional[str] = None,
        **kwargs: Any,
    ) -> "STELLARPhenotyper":
        """Store reference training data or run joint transductive STELLAR fit.

        Parameters
        ----------
        train_df:
            Annotated reference spatial single-cell DataFrame.
        target_df:
            Optional unannotated target spatial DataFrame. When supplied,
            STELLAR immediately performs joint transductive training on
            ``(train_df, target_df)``. When omitted, reference data is stored
            and transductive training runs upon ``predict(target_df)``.
        """
        self._train_df = train_df
        self._train_label_col = label_col or self.label_col
        self._resolved_features = self._resolve_feature_cols(
            train_df, target_df=target_df, feature_columns=feature_columns
        )
        self._cached_target_signature = None
        self._cached_prediction = None

        if target_df is not None:
            self._run_transductive(
                train_df,
                target_df,
                feature_columns=self._resolved_features,
                label_col=self._train_label_col,
                config_overrides=kwargs,
            )
        return self

    def predict(
        self,
        target_df: Optional[pd.DataFrame] = None,
        *,
        retrain: Optional[bool] = None,
        **kwargs: Any,
    ) -> PhenotypingPredictionResult:
        """Predict cell-type labels and novel discoveries for ``target_df``."""
        if target_df is None:
            if self._cached_prediction is not None:
                return self._cached_prediction
            raise ValueError("target_df must be provided when no target was supplied to fit().")

        if (
            not kwargs
            and retrain is not True
            and self._cached_prediction is not None
            and self._cached_target_signature == self._df_signature(target_df)
        ):
            return self._cached_prediction

        if retrain is False:
            if self._pipeline is None or self._resolved_features is None:
                raise RuntimeError("No fitted STELLAR encoder available for retrain=False.")
            feat_cols = self._resolved_features
            tx_col, ty_col = self._resolve_coord_cols(target_df)
            ux_mat = target_df[feat_cols].to_numpy(dtype=np.float32, copy=True)
            ux_mat = np.nan_to_num(ux_mat, nan=0.0, posinf=0.0, neginf=0.0)
            if self._feature_mean is not None and self._feature_std is not None:
                ux_mat = (ux_mat - self._feature_mean) / self._feature_std
            u_coords = target_df[[tx_col, ty_col]].to_numpy(dtype=np.float64)
            u_regions = self._resolve_region_series(target_df)
            raw = self._pipeline.predict_graph(
                unlabeled_x=ux_mat,
                unlabeled_coords=u_coords,
                unlabeled_regions=u_regions,
            )
            self.last_result_ = raw
            return self._to_prediction_result(target_df, raw, feat_cols)

        if self._train_df is None:
            raise RuntimeError("Call fit(train_df) before predict(target_df).")

        return self._run_transductive(
            self._train_df,
            target_df,
            feature_columns=self._resolved_features,
            label_col=self._train_label_col,
            config_overrides=kwargs,
        )

    def fit_predict(
        self,
        train_df: pd.DataFrame,
        target_df: pd.DataFrame,
        *,
        feature_columns: Optional[Sequence[str]] = None,
        label_col: Optional[str] = None,
        **kwargs: Any,
    ) -> PhenotypingPredictionResult:
        """Fit STELLAR jointly on ``train_df`` and ``target_df`` and return predictions."""
        self.fit(
            train_df,
            target_df=target_df,
            feature_columns=feature_columns,
            label_col=label_col,
            **kwargs,
        )
        return self.predict(target_df)


__all__ = ["STELLARPhenotyper"]
