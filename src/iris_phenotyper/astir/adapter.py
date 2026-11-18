"""
ASTIR Phenotyper Adapter
========================
Wraps ``ASTIRPipeline`` into the canonical ``PhenotyperMethodBase`` contract,
returning standardized ``PhenotypingPredictionResult`` outputs.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from iris_common.schema import CELL_ID_ALIASES, COL_CELL_ID
from iris_common.types import CellTypePanel, MarkerPanel
from iris_phenotyper.base import PhenotyperMethodBase, PhenotypingPredictionResult

from .pipeline import (
    ASTIRPipeline,
    is_astir_available,
)


def _extract_object_index_and_names(
    data: Any,
) -> Tuple[pd.Index, List[Any]]:
    """Extract canonical integer ``ObjectNumber`` index plus original cell names."""
    if hasattr(data, "obs") and hasattr(data, "obs_names"):
        obs = data.obs
        n_rows = len(obs)
        cell_names = list(obs["cell_id"]) if "cell_id" in obs.columns else list(data.obs_names)
        for col in (COL_CELL_ID, *CELL_ID_ALIASES):
            if col in obs.columns:
                num = pd.to_numeric(obs[col], errors="coerce")
                if num.notna().all() and num.nunique() == n_rows:
                    return pd.Index(num.astype(int), name=COL_CELL_ID), cell_names
        return pd.Index(np.arange(1, n_rows + 1, dtype=int), name=COL_CELL_ID), cell_names

    if isinstance(data, pd.DataFrame):
        n_rows = len(data)
        cell_names = list(data.index)
        for col in (COL_CELL_ID, *CELL_ID_ALIASES):
            if col in data.columns:
                num = pd.to_numeric(data[col], errors="coerce")
                if num.notna().all() and num.nunique() == n_rows:
                    return pd.Index(num.astype(int), name=COL_CELL_ID), list(data[col])
        return pd.Index(np.arange(1, n_rows + 1, dtype=int), name=COL_CELL_ID), cell_names

    if isinstance(data, tuple) and len(data) == 3:
        cell_names = list(data[2])
        n_rows = len(cell_names)
        return pd.Index(np.arange(1, n_rows + 1, dtype=int), name=COL_CELL_ID), cell_names

    raise TypeError(f"Unsupported input type for ASTIRPhenotyper: {type(data).__name__}")


def _compute_dataset_fingerprint(
    data: Any,
    object_ids: pd.Index,
) -> Tuple[Any, ...]:
    """Build a lightweight fingerprint to detect dataset changes across ``predict()`` calls."""
    if hasattr(data, "X") and hasattr(data, "var_names"):
        n_rows, n_cols = data.shape
        cols = tuple(str(c) for c in data.var_names)
    elif isinstance(data, pd.DataFrame):
        n_rows, n_cols = data.shape
        cols = tuple(str(c) for c in data.columns)
    elif isinstance(data, tuple) and len(data) == 3:
        arr = np.asarray(data[0])
        n_rows, n_cols = arr.shape
        cols = tuple(str(c) for c in data[1])
    else:
        return ()

    ids_list = object_ids.tolist()
    head_ids = tuple(ids_list[:5])
    tail_ids = tuple(ids_list[-5:])
    return (int(n_rows), int(n_cols), head_ids, tail_ids, cols)


class ASTIRPhenotyper(PhenotyperMethodBase):
    """IRIS phenotyping adapter for ASTIR Bayesian variational inference."""

    name: str = "ASTIR"
    input_modality: str = "tabular"
    requires_training: bool = False

    def __init__(
        self,
        marker_config: Optional[Union[str, Path, Mapping[str, Any], CellTypePanel]] = None,
        *,
        marker_panel: Optional[MarkerPanel] = None,
        max_epochs: int = 100,
        learning_rate: float = 2e-3,
        batch_size: int = 256,
        delta_loss: float = 1e-3,
        n_init: int = 5,
        n_init_epochs: int = 3,
        threshold: float = 0.7,
        assignment_type: Literal["threshold", "max"] = "threshold",
        arcsinh_cofactor: float = 5.0,
        apply_arcsinh: Union[bool, Literal["auto"]] = "auto",
        winsorize_percentiles: Optional[Tuple[float, float]] = None,
        intensity_stat: str = "mean",
        batch_col: Optional[str] = "batch",
        create_design_mat: bool = True,
        fit_states: bool = False,
        run_diagnostics: bool = True,
        random_seed: int = 1234,
        dtype: Optional[str] = None,
        device: Optional[str] = None,
        engine: Literal["auto", "iris", "official"] = "auto",
        auto_refit: bool = True,
    ) -> None:
        self.marker_config = marker_config
        self.marker_panel = marker_panel
        self.max_epochs = int(max_epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.delta_loss = float(delta_loss)
        self.n_init = int(n_init)
        self.n_init_epochs = int(n_init_epochs)
        self.threshold = float(threshold)
        self.assignment_type: Literal["threshold", "max"] = assignment_type
        self.arcsinh_cofactor = float(arcsinh_cofactor)
        self.apply_arcsinh: Union[bool, Literal["auto"]] = apply_arcsinh
        self.winsorize_percentiles = winsorize_percentiles
        self.intensity_stat = intensity_stat
        self.batch_col = batch_col
        self.create_design_mat = bool(create_design_mat)
        self.fit_states = bool(fit_states)
        self.run_diagnostics = bool(run_diagnostics)
        self.random_seed = int(random_seed)
        self.dtype = dtype
        self.device = device
        self.engine: Literal["auto", "iris", "official"] = engine
        self.auto_refit = bool(auto_refit)

        self._pipeline: Optional[ASTIRPipeline] = None
        self._fit_fingerprint: Optional[Tuple[Any, ...]] = None
        self._fit_elapsed_s: float = 0.0
        self._fit_object_ids: Optional[pd.Index] = None
        self._fit_cell_names: List[Any] = []

    def is_available(self) -> bool:
        """Return True if PyTorch (or upstream ``astir``) is available."""
        return is_astir_available()

    def get_default_params(self) -> Dict[str, Any]:
        """Return default ASTIR hyperparameters."""
        return {
            "max_epochs": 100,
            "learning_rate": 2e-3,
            "batch_size": 256,
            "delta_loss": 1e-3,
            "n_init": 5,
            "n_init_epochs": 3,
            "threshold": 0.7,
            "assignment_type": "threshold",
            "arcsinh_cofactor": 5.0,
            "apply_arcsinh": "auto",
            "winsorize_percentiles": None,
            "intensity_stat": "mean",
            "batch_col": "batch",
            "create_design_mat": True,
            "fit_states": False,
            "run_diagnostics": True,
            "random_seed": 1234,
            "engine": "auto",
        }

    def _resolve_marker_config(
        self,
        cell_type_panel: Optional[CellTypePanel] = None,
        marker_config: Optional[Union[str, Path, Mapping[str, Any], CellTypePanel]] = None,
    ) -> Union[str, Path, Mapping[str, Any], CellTypePanel]:
        cfg = marker_config or cell_type_panel or self.marker_config
        if cfg is None:
            raise ValueError(
                "ASTIR requires a marker_config (YAML path, dict, or CellTypePanel)."
            )
        return cfg

    def fit(
        self,
        features: Any,
        cell_type_panel: Optional[CellTypePanel] = None,
        *,
        marker_config: Optional[Union[str, Path, Mapping[str, Any], CellTypePanel]] = None,
        marker_panel: Optional[MarkerPanel] = None,
        design: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "ASTIRPhenotyper":
        """Fit ASTIR variational inference on single-cell expression features."""
        cfg = self._resolve_marker_config(
            cell_type_panel=cell_type_panel or kwargs.get("panel"),
            marker_config=marker_config or kwargs.get("marker_dict"),
        )
        m_panel = marker_panel or self.marker_panel

        max_epochs = int(kwargs.get("max_epochs", self.max_epochs))
        learning_rate = float(kwargs.get("learning_rate", kwargs.get("lr", self.learning_rate)))
        batch_size = int(kwargs.get("batch_size", self.batch_size))
        delta_loss = float(kwargs.get("delta_loss", self.delta_loss))
        n_init = int(kwargs.get("n_init", self.n_init))
        n_init_epochs = int(kwargs.get("n_init_epochs", self.n_init_epochs))
        cofactor = float(kwargs.get("arcsinh_cofactor", self.arcsinh_cofactor))
        apply_arcsinh = kwargs.get("apply_arcsinh", self.apply_arcsinh)
        winsorize = kwargs.get("winsorize_percentiles", self.winsorize_percentiles)
        stat = str(kwargs.get("intensity_stat", self.intensity_stat))
        batch_col = kwargs.get("batch_col", self.batch_col)
        create_design = bool(kwargs.get("create_design_mat", self.create_design_mat))
        fit_states = bool(kwargs.get("fit_states", self.fit_states))
        random_seed = int(kwargs.get("random_seed", self.random_seed))
        dtype = kwargs.get("dtype", self.dtype)
        device = kwargs.get("device", self.device)
        engine = kwargs.get("engine", self.engine)

        t0 = time.perf_counter()
        object_ids, cell_names = _extract_object_index_and_names(features)

        pipe = ASTIRPipeline(
            features,
            cfg,
            marker_panel=m_panel,
            design=design,
            batch_col=batch_col,
            create_design_mat=create_design,
            apply_arcsinh=apply_arcsinh,
            arcsinh_cofactor=cofactor,
            winsorize_percentiles=winsorize,
            preferred_stat=stat,
            random_seed=random_seed,
            dtype=dtype,
            device=device,
            engine=engine,
        )
        pipe.fit_type(
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            delta_loss=delta_loss,
            n_init=n_init,
            n_init_epochs=n_init_epochs,
        )
        if fit_states and pipe._state_dict:
            pipe.fit_state(
                max_epochs=int(kwargs.get("state_max_epochs", max_epochs)),
                learning_rate=float(kwargs.get("state_learning_rate", learning_rate)),
                batch_size=int(kwargs.get("state_batch_size", batch_size)),
                delta_loss=delta_loss,
                n_init=n_init,
                n_init_epochs=n_init_epochs,
            )

        self._pipeline = pipe
        self._fit_fingerprint = _compute_dataset_fingerprint(features, object_ids)
        self._fit_elapsed_s = float(time.perf_counter() - t0)
        self._fit_object_ids = object_ids
        self._fit_cell_names = cell_names
        return self

    def predict(
        self,
        features: Any,
        cell_type_panel: Optional[CellTypePanel] = None,
        *,
        marker_config: Optional[Union[str, Path, Mapping[str, Any], CellTypePanel]] = None,
        marker_panel: Optional[MarkerPanel] = None,
        threshold: Optional[float] = None,
        assignment_type: Optional[Literal["threshold", "max"]] = None,
        refit: Optional[bool] = None,
        **kwargs: Any,
    ) -> PhenotypingPredictionResult:
        """Run ASTIR inference and return a validated ``PhenotypingPredictionResult``."""
        object_ids, cell_names = _extract_object_index_and_names(features)
        fp = _compute_dataset_fingerprint(features, object_ids)

        if refit is True:
            should_fit = True
        elif refit is False:
            if self._pipeline is None:
                raise RuntimeError("Cannot run predict(..., refit=False) before fit().")
            should_fit = False
        else:
            if self._pipeline is None:
                should_fit = True
            elif fp == self._fit_fingerprint:
                should_fit = False
            else:
                should_fit = self.auto_refit

        t0 = time.perf_counter()
        if should_fit:
            self.fit(
                features,
                cell_type_panel=cell_type_panel,
                marker_config=marker_config,
                marker_panel=marker_panel,
                **kwargs,
            )

        assert self._pipeline is not None
        thresh = float(self.threshold if threshold is None else threshold)
        assign_mode: Literal["threshold", "max"] = (
            self.assignment_type if assignment_type is None else assignment_type
        )

        if should_fit or fp == self._fit_fingerprint:
            prob_df = self._pipeline.get_celltype_probabilities()
            label_df = self._pipeline.get_celltypes(
                threshold=thresh, assignment_type=assign_mode
            )
        else:
            label_df, prob_df = self._pipeline.predict_celltypes(
                features,
                threshold=thresh,
                assignment_type=assign_mode,
                apply_arcsinh=kwargs.get("apply_arcsinh", self.apply_arcsinh),
                arcsinh_cofactor=float(kwargs.get("arcsinh_cofactor", self.arcsinh_cofactor)),
            )

        prob_df = prob_df.copy()
        prob_df.index = object_ids
        prob_values = np.clip(prob_df.to_numpy(dtype=float), 0.0, 1.0)
        prob_df = pd.DataFrame(prob_values, index=object_ids, columns=prob_df.columns)

        labels = pd.Series(
            label_df["cell_type"].astype(str).to_numpy(),
            index=object_ids,
            name="cell_type",
        )
        confidence = pd.Series(
            prob_values.max(axis=1),
            index=object_ids,
            name="confidence",
            dtype=float,
        )

        metadata: Dict[str, Any] = {
            "engine_used": self._pipeline.engine_used,
            "did_arcsinh": self._pipeline.did_arcsinh,
            "threshold": thresh,
            "assignment_type": assign_mode,
            "unknown_count": int((labels == "Unknown").sum()),
            "other_count": int((labels == "Other").sum()),
            "n_epochs_type": int(len(self._pipeline.get_type_losses())),
            "final_elbo_loss": float(self._pipeline.get_type_losses()[-1]),
            "elapsed_s": float(self._fit_elapsed_s if should_fit else (time.perf_counter() - t0)),
            "cell_ids": cell_names,
        }

        if bool(kwargs.get("run_diagnostics", self.run_diagnostics)):
            try:
                diag_df = self._pipeline.diagnostics_celltype(threshold=thresh)
                metadata["diagnostics_celltype"] = diag_df
                metadata["diagnostics_issues_count"] = int(len(diag_df))
            except Exception as exc:
                metadata["diagnostics_error"] = str(exc)

        if self._pipeline._state_model is not None or (
            self._pipeline._official_astir is not None
            and self._pipeline._official_astir._state_ast is not None
        ):
            try:
                state_df = self._pipeline.get_cellstates()
                state_df.index = object_ids
                metadata["cell_states"] = state_df
            except Exception:
                pass

        result = PhenotypingPredictionResult(
            method_name=self.name,
            labels=labels,
            probabilities=prob_df,
            confidence=confidence,
            is_novel=None,
            marker_positivity=None,
            metadata=metadata,
        )
        result.validate()
        return result

    @property
    def pipeline(self) -> Optional[ASTIRPipeline]:
        """Return underlying ``ASTIRPipeline`` instance if fitted."""
        return self._pipeline
