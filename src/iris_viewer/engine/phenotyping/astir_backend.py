"""IRIS-Phenotyper ASTIR Backend.

ASTIR (Automated Single-cell Typing and Inference of States) uses Bayesian
variational inference to model single-cell protein expression and assign
calibrated posterior probabilities from a user-defined marker panel.

Bridges cleanly to ``iris_phenotyper.astir.ASTIRPhenotyper`` when the external
``astir`` package is unavailable, while preserving compatibility with official
``astir`` when installed.

Reference: Geuenich et al., Cell Systems 2021.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from iris_viewer.core.types import CellTypePanel, CellTypeRule, MarkerPanel
from iris_viewer.engine.phenotyping.base import PhenotypeBackend


class ASTIRBackend(PhenotypeBackend):
    """Phenotyping backend wrapping ASTIR Bayesian variational inference."""

    name = "ASTIR"
    requires_training = False
    requires_raw_image = False
    requires_spatial = False

    def __init__(self) -> None:
        self._model: Any = None
        self._trained: bool = False
        self._cell_ids: Optional[List[int]] = None
        self._dataset_fingerprint: Optional[tuple] = None
        self._last_result: Any = None
        self.last_probabilities_: Optional[pd.DataFrame] = None

    @staticmethod
    def _marker_config_sig(marker_config: Any) -> Any:
        if marker_config is None:
            return None
        if isinstance(marker_config, (str, Path)):
            p = Path(marker_config)
            try:
                mtime = p.stat().st_mtime if p.exists() else None
            except OSError:
                mtime = None
            return ("path", str(p), mtime)
        if isinstance(marker_config, Mapping):
            return ("mapping", json.dumps(marker_config, sort_keys=True, default=str))
        return ("other", repr(marker_config))

    @classmethod
    def _compute_dataset_fingerprint(
        cls,
        features: pd.DataFrame,
        cell_ids: List[int],
        panel: Optional[CellTypePanel] = None,
        marker_config: Any = None,
    ) -> tuple:
        """Compute a lightweight fingerprint to detect dataset or panel changes."""
        row_hash = int(pd.util.hash_pandas_object(features, index=True).sum())
        panel_sig = (
            tuple(
                (r.cell_type, tuple(r.markers_positive), tuple(r.markers_negative))
                for r in panel.rules
            )
            if panel is not None
            else ()
        )
        return (
            len(features),
            tuple(cell_ids[:5]),
            tuple(cell_ids[-5:]),
            tuple(sorted(str(c) for c in features.columns)),
            row_hash,
            panel_sig,
            cls._marker_config_sig(marker_config),
        )

    # Availability

    def is_available(self) -> bool:
        """Check whether the external ``astir`` Python package is importable."""
        try:
            import astir  # noqa: F401

            return True
        except ImportError:
            return False

    def is_vi_available(self) -> bool:
        """Return True if either external ``astir`` or PyTorch IRIS VI is available."""
        if self.is_available():
            return True
        try:
            from iris_phenotyper.astir import ASTIRPhenotyper

            return ASTIRPhenotyper().is_available()
        except Exception:
            return False

    def get_default_params(self) -> Dict[str, Any]:
        """Return default hyperparameters for ASTIR model training."""
        return {
            "max_epochs": 50,
            "learning_rate": 1e-3,
            "batch_size": 256,
            "arcsinh_cofactor": 5.0,
            "intensity_col_suffix": "_mean",
            "engine": "auto",
        }

    # Fit

    @staticmethod
    def _panel_from_marker_config(marker_config: Any) -> Optional[CellTypePanel]:
        if marker_config is None:
            return None
        try:
            from iris_phenotyper.astir.pipeline import normalize_marker_config

            ct_norm, _, h_norm = normalize_marker_config(marker_config)
            if not ct_norm:
                return None
            parent_lookup: Dict[str, str] = {}
            if isinstance(h_norm, Mapping):
                for parent_name, children in h_norm.items():
                    if isinstance(children, (list, tuple, set)):
                        for child in children:
                            parent_lookup[str(child)] = str(parent_name)
            rules = [
                CellTypeRule(
                    cell_type=str(ct),
                    markers_positive=[str(m) for m in markers],
                    parent=parent_lookup.get(str(ct)),
                )
                for ct, markers in ct_norm.items()
                if markers
            ]
            return CellTypePanel(rules=rules)
        except Exception:
            return None

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
                batch_size, arcsinh_cofactor, intensity_col_suffix, marker_config,
                and engine (``"auto"``, ``"iris"``, or ``"official"``).
        """
        panel = cell_type_panel or kwargs.get("cell_type_panel") or kwargs.get("panel")
        marker_config = kwargs.get("marker_config")
        if panel is None and marker_config is None:
            raise ValueError("cell_type_panel or marker_config is required for ASTIR fitting.")

        marker_panel: Optional[MarkerPanel] = kwargs.get("marker_panel")
        max_epochs: int = int(kwargs.get("max_epochs", 50))
        learning_rate: float = float(kwargs.get("learning_rate", kwargs.get("lr", 1e-3)))
        batch_size: int = int(kwargs.get("batch_size", 256))
        cofactor: float = float(kwargs.get("arcsinh_cofactor", 5.0))
        suffix: str = str(kwargs.get("intensity_col_suffix", "_mean"))
        engine: str = str(kwargs.get("engine", "auto"))

        expr = self._extract_intensity_matrix(features, stat=suffix.lstrip("_"))
        mp = marker_panel if marker_panel is not None else MarkerPanel(markers=list(expr.columns))

        if panel is not None:
            raw_dict = panel.to_astir_dict(mp)
            type_dict: Dict[str, Any] = (
                raw_dict if "cell_types" in raw_dict else {"cell_types": raw_dict}
            )
        else:
            from iris_phenotyper.astir.pipeline import normalize_marker_config

            ct_norm, st_norm, h_norm = normalize_marker_config(marker_config, marker_panel=mp)
            type_dict = {"cell_types": ct_norm or {}}
            if st_norm:
                type_dict["cell_states"] = st_norm
            if h_norm:
                type_dict["hierarchy"] = h_norm

        ct_entries = type_dict.get("cell_types", {})
        avail_lower = {str(c).lower(): str(c) for c in expr.columns}
        filtered_ct: Dict[str, List[str]] = {}
        for ct, markers in ct_entries.items():
            matched = [avail_lower[m.lower()] for m in markers if m.lower() in avail_lower]
            if matched:
                filtered_ct[str(ct)] = matched

        if len(filtered_ct) < 2:
            raise ValueError(
                "ASTIR requires at least two distinct cell types with defined positive markers."
            )
        type_dict["cell_types"] = filtered_ct

        self._cell_ids = self._cell_ids_from_features(features)
        use_official = (engine == "official") or (engine == "auto" and self.is_available())

        if use_official:
            import astir

            expr_scaled = np.arcsinh(expr / cofactor) if cofactor > 0 else expr
            self._model = astir.Astir(expr_scaled, type_dict)
            self._model.fit_type(
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                batch_size=min(batch_size, max(len(expr_scaled), 1)),
            )
            self._last_result = None
            self.last_probabilities_ = None
        else:
            from iris_phenotyper.astir import ASTIRPhenotyper

            expr_clean = expr.copy()
            expr_clean.index = pd.RangeIndex(len(expr_clean))
            n_init = int(kwargs.get("n_init", 2 if len(expr_clean) < 100 else 3))
            n_init_epochs = int(kwargs.get("n_init_epochs", min(3, max_epochs)))
            delta_loss = float(kwargs.get("delta_loss", 1e-3))
            threshold = float(kwargs.get("threshold", 0.5))
            assignment_type = str(kwargs.get("assignment_type", "threshold"))
            run_diagnostics = bool(kwargs.get("run_diagnostics", False))

            cfg_payload: Dict[str, Any] = dict(type_dict)
            if marker_config is not None and panel is not None:
                from iris_phenotyper.astir.pipeline import normalize_marker_config

                _, st_norm, h_norm = normalize_marker_config(marker_config, marker_panel=mp)
                if st_norm and "cell_states" not in cfg_payload:
                    cfg_payload["cell_states"] = st_norm
                if h_norm and "hierarchy" not in cfg_payload:
                    cfg_payload["hierarchy"] = h_norm

            phenotyper = ASTIRPhenotyper(
                marker_config=cfg_payload,
                marker_panel=mp,
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                batch_size=min(batch_size, max(len(expr_clean), 1)),
                delta_loss=delta_loss,
                n_init=n_init,
                n_init_epochs=n_init_epochs,
                threshold=threshold,
                assignment_type="max" if assignment_type == "max" else "threshold",
                arcsinh_cofactor=cofactor if cofactor > 0 else 5.0,
                apply_arcsinh=bool(cofactor > 0),
                create_design_mat=False,
                run_diagnostics=run_diagnostics,
                engine="iris",
            )
            phenotyper.fit(expr_clean, marker_config=cfg_payload, marker_panel=mp)
            res = phenotyper.predict(expr_clean, refit=False)
            self._model = phenotyper
            self._last_result = res
            self.last_probabilities_ = res.probabilities

        self._trained = True
        self._dataset_fingerprint = self._compute_dataset_fingerprint(
            features,
            self._cell_ids,
            panel,
            marker_config=marker_config,
        )

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
        marker_config = kwargs.get("marker_config")
        panel = cell_type_panel or kwargs.get("cell_type_panel") or kwargs.get("panel")
        if panel is None and marker_config is not None:
            panel = self._panel_from_marker_config(marker_config)

        cell_ids = self._cell_ids_from_features(features)
        if features.empty or panel is None or not panel.rules:
            self.last_probabilities_ = None
            return {int(cid): {} for cid in cell_ids}

        fingerprint = self._compute_dataset_fingerprint(
            features,
            cell_ids,
            panel,
            marker_config=marker_config,
        )
        if (
            not self._trained
            or self._model is None
            or self._dataset_fingerprint != fingerprint
        ):
            try:
                self.fit(features, cell_type_panel=panel, **kwargs)
            except Exception:
                self.last_probabilities_ = None
                return self._predict_fallback(features, panel)

        try:
            if hasattr(self._model, "get_celltype_probabilities"):
                prob_df: pd.DataFrame = self._model.get_celltype_probabilities()
            elif self._last_result is not None and self._last_result.probabilities is not None:
                prob_df = self._last_result.probabilities
            else:
                raise RuntimeError("No fitted ASTIR probability table available.")

            self.last_probabilities_ = prob_df
            result: Dict[int, Dict[str, float]] = {}
            for i, cell_id in enumerate(cell_ids):
                if i < len(prob_df):
                    row = prob_df.iloc[i]
                    result[int(cell_id)] = {str(ct): float(p) for ct, p in row.items()}
                else:
                    result[int(cell_id)] = {}
            return result
        except Exception:
            self.last_probabilities_ = None
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

        lower_cols = {str(c).lower(): c for c in features.columns}
        rows_list: List[Dict[str, float]] = []
        for idx, row in features.iterrows():
            cid = int(row.get("ObjectNumber", row.get("Cell_ID", row.get("cell_id", idx))))
            scores: Dict[str, float] = {}
            for rule in cell_type_panel.rules:
                ct = rule.cell_type
                markers = rule.markers_positive
                if not markers:
                    continue
                vals = []
                for m in markers:
                    col = next(
                        (c for c in (m, f"{m}_mean") if c in features.columns),
                        lower_cols.get(m.lower()) or lower_cols.get(f"{m.lower()}_mean"),
                    )
                    if col is not None:
                        vals.append(row[col])
                scores[ct] = float(np.mean(vals)) if vals else 0.0

            total = sum(scores.values())
            if total > 0:
                probs = {k: v / total for k, v in scores.items()}
            else:
                uniform = 1.0 / len(cell_type_panel.rules) if cell_type_panel.rules else 0.0
                probs = {r.cell_type: uniform for r in cell_type_panel.rules}
            results[cid] = probs
            rows_list.append(probs)

        self.last_probabilities_ = pd.DataFrame(rows_list)
        return results

    @staticmethod
    def _cell_ids_from_features(features: pd.DataFrame) -> List[int]:
        for col in ("ObjectNumber", "Cell_ID", "cell_id"):
            if col in features.columns:
                return features[col].astype(int).tolist()
        return list(range(len(features)))
