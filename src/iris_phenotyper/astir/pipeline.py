"""
ASTIR Core Pipeline
===================
Faithful PyTorch variational inference implementation of ASTIR (Automated
aSsignmenT sIngle-cell pRoteomics; Geuenich et al., Cell Systems 2021) compatible
with PyTorch 2.x / NumPy 2.x, alongside optional delegation to the upstream
``astir`` package when installed.
"""
from __future__ import annotations

import logging
import math
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from iris_common.logging import iris_log
from iris_common.schema import (
    CELL_ID_ALIASES,
    COL_CELL_ID,
    COL_SAMPLE,
    find_sample_column,
    resolve_marker_column,
)
from iris_common.types import CellTypePanel, MarkerPanel

logger = logging.getLogger(__name__)

_TYPE_KEY_RE = re.compile(r"cell[^a-zA-Z0-9]*type", re.IGNORECASE)
_STATE_KEY_RE = re.compile(r"cell[^a-zA-Z0-9]*state", re.IGNORECASE)
_HIERARCHY_KEY_RE = re.compile(r".*hierarch.*", re.IGNORECASE)

_METADATA_COL_LOWER = {
    "image",
    "sample",
    "sample_id",
    "sample_name",
    "core",
    "tma_core",
    "fov",
    "roi",
    "batch",
    "file",
    "filename",
    "slide",
    "objectnumber",
    "object_number",
    "cell_id",
    "label_id",
    "label",
    "id",
    "pos_x",
    "pos_y",
    "x",
    "y",
    "x_centroid",
    "y_centroid",
    "centroid_x",
    "centroid_y",
    "centroid-0",
    "centroid-1",
    "voron_x",
    "voron_y",
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "major_axis_length",
    "minor_axis_length",
    "cell_type",
    "celltype",
    "phenotype",
    "verified_celltype",
    "manual_label",
    "group",
    "cluster",
    "class",
}


def _import_torch() -> Any:
    """Import PyTorch cleanly, guarding duplicate macOS OpenMP runtimes."""
    if sys.platform == "darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    if sys.platform == "darwin":
        try:
            torch.set_num_threads(1)
        except RuntimeError:
            pass
    return torch


def is_official_astir_available() -> bool:
    """Return True if the upstream ``astir`` package can be imported."""
    try:
        import astir  # noqa: F401

        return True
    except ImportError:
        return False


def is_astir_available() -> bool:
    """Return True if either PyTorch (for IRIS ASTIR VI) or ``astir`` is available."""
    if is_official_astir_available():
        return True
    try:
        _import_torch()
        return True
    except ImportError:
        return False


def resolve_device(device: Optional[str] = None) -> str:
    """Resolve compute device string (``'cpu'``, ``'cuda'``, or ``'mps'``)."""
    torch = _import_torch()
    if device is not None and device != "auto":
        return str(device)
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_dtype(dtype: Optional[Union[str, Any]], device: str) -> Any:
    """Resolve torch floating dtype, ensuring MPS compatibility (float32)."""
    torch = _import_torch()
    if dtype is None:
        resolved = torch.float32 if str(device).startswith("mps") else torch.float64
    elif isinstance(dtype, str):
        mapping = {"float32": torch.float32, "float64": torch.float64}
        if dtype not in mapping:
            raise ValueError(f"Unsupported ASTIR dtype {dtype!r}; expected float32 or float64.")
        resolved = mapping[dtype]
    else:
        resolved = dtype

    if resolved not in (torch.float32, torch.float64):
        raise ValueError("ASTIR dtype must be torch.float32 or torch.float64.")
    if str(device).startswith("mps") and resolved == torch.float64:
        warnings.warn(
            "MPS backend does not support float64; falling back to torch.float32.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.float32
    return resolved


def normalize_marker_config(
    marker_config: Union[str, Path, Mapping[str, Any], CellTypePanel],
    *,
    marker_panel: Optional[MarkerPanel] = None,
) -> Tuple[Optional[Dict[str, List[str]]], Optional[Dict[str, List[str]]], Optional[Dict[str, Any]]]:
    """Parse a YAML path, dict, or ``CellTypePanel`` into ASTIR marker dictionaries.

    Returns ``(cell_types_dict, cell_states_dict, hierarchy_dict)``.
    """
    if isinstance(marker_config, (str, Path)):
        path = Path(marker_config)
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Marker YAML {path} did not parse to a mapping.")
        raw: Dict[str, Any] = dict(loaded)
    elif isinstance(marker_config, CellTypePanel):
        if marker_panel is not None:
            raw = marker_config.to_astir_dict(marker_panel)
        else:
            ct_map = {
                rule.cell_type: list(rule.markers_positive)
                for rule in marker_config.rules
                if rule.markers_positive
            }
            raw = {"cell_types": ct_map}
        hierarchy: Dict[str, List[str]] = {}
        for rule in marker_config.rules:
            if rule.parent:
                hierarchy.setdefault(rule.parent, []).append(rule.cell_type)
        if hierarchy:
            raw["hierarchy"] = hierarchy
    elif isinstance(marker_config, Mapping):
        raw = dict(marker_config)
    else:
        raise TypeError(
            "marker_config must be a YAML path, mapping, or CellTypePanel, "
            f"got {type(marker_config).__name__}."
        )

    type_dict: Optional[Dict[str, List[str]]] = None
    state_dict: Optional[Dict[str, List[str]]] = None
    hierarchy_dict: Optional[Dict[str, Any]] = None

    for key, val in raw.items():
        key_str = str(key)
        if _TYPE_KEY_RE.match(key_str) and isinstance(val, Mapping):
            type_dict = {str(k): [str(m) for m in v] for k, v in val.items() if v}
        elif _STATE_KEY_RE.match(key_str) and isinstance(val, Mapping):
            state_dict = {str(k): [str(m) for m in v] for k, v in val.items() if v}
        elif _HIERARCHY_KEY_RE.match(key_str) and isinstance(val, Mapping):
            hierarchy_dict = dict(val)

    # Support flat {cell_type: [markers...]} mappings
    if type_dict is None and state_dict is None and raw:
        if all(isinstance(v, (list, tuple, set)) for v in raw.values()):
            type_dict = {str(k): [str(m) for m in v] for k, v in raw.items() if v}

    return type_dict, state_dict, hierarchy_dict


def _extract_cell_names(df: pd.DataFrame) -> List[Any]:
    """Extract cell identifiers from index or canonical ID columns."""
    if df.index is not None and not isinstance(df.index, pd.RangeIndex):
        return list(df.index)
    for col in (COL_CELL_ID, *CELL_ID_ALIASES):
        if col in df.columns:
            return df[col].tolist()
    return list(df.index)


def _infer_marker_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Extract numeric marker columns from a DataFrame, normalizing stat suffixes."""
    mean_cols = [str(c) for c in df.columns if str(c).endswith("_mean")]
    if mean_cols:
        col_map = {c[: -len("_mean")]: c for c in mean_cols}
        mat = df[mean_cols].copy()
        mat.columns = list(col_map.keys())
        return mat, col_map

    cols = [
        str(c)
        for c in df.columns
        if str(c).strip().lower() not in _METADATA_COL_LOWER
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    mat = df[cols].copy()
    return mat, {c: c for c in cols}


def _resolve_requested_markers(
    df: pd.DataFrame,
    requested_markers: Sequence[str],
    preferred_stat: str = "mean",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Resolve requested marker names to columns in ``df``."""
    resolved_cols: List[str] = []
    resolved_names: List[str] = []
    mapping: Dict[str, str] = {}

    for marker in requested_markers:
        col = resolve_marker_column(df, marker, preferred_stat=preferred_stat)
        if col is not None:
            resolved_cols.append(col)
            resolved_names.append(marker)
            mapping[marker] = col

    if not resolved_cols:
        raise RuntimeError(
            "Classification failed. No overlap between requested markers and expression columns."
        )

    out = df[resolved_cols].copy()
    out.columns = resolved_names
    return out, mapping


def build_batch_design_matrix(batches: Sequence[Any]) -> np.ndarray:
    """Build ASTIR intercept + one-hot batch design matrix ``[1, B - 1]``."""
    arr = np.asarray(list(batches)).reshape(-1, 1)
    if len(np.unique(arr)) <= 1:
        return np.ones((len(arr), 1), dtype=np.float64)
    enc = OneHotEncoder(sparse_output=False)
    ohe = enc.fit_transform(arr)
    ohe_reduced = ohe[:, :-1]
    return np.concatenate([np.ones((len(arr), 1), dtype=np.float64), ohe_reduced], axis=1)


def extract_expression_and_design(
    expr_input: Union[
        pd.DataFrame,
        Any,
        Tuple[Union[np.ndarray, Any], Sequence[str], Sequence[Any]],
    ],
    *,
    requested_markers: Optional[Sequence[str]] = None,
    preferred_stat: str = "mean",
    design: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    batch_col: Optional[str] = "batch",
    create_design_mat: bool = True,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Extract cell-by-marker DataFrame and optional batch design matrix.

    Supports ``pd.DataFrame`` (canonical IRIS tables or raw marker tables),
    ``anndata.AnnData``, or ``(array, feature_names, cell_names)`` tuples.
    """
    extracted_design = design
    if isinstance(expr_input, tuple) and len(expr_input) == 3:
        arr, feature_names, cell_names = expr_input
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        df_raw = pd.DataFrame(
            np.asarray(arr, dtype=np.float64),
            index=list(cell_names),
            columns=[str(c) for c in feature_names],
        )
    elif hasattr(expr_input, "X") and hasattr(expr_input, "var_names"):
        x_val = expr_input.X.toarray() if hasattr(expr_input.X, "toarray") else np.asarray(
            expr_input.X
        )
        df_raw = pd.DataFrame(
            np.asarray(x_val, dtype=np.float64),
            index=list(expr_input.obs_names),
            columns=[str(c) for c in expr_input.var_names],
        )
        if extracted_design is None and create_design_mat and hasattr(expr_input, "obs"):
            b_col = batch_col if batch_col in expr_input.obs.columns else None
            if b_col is None:
                b_col = find_sample_column(expr_input.obs)
            if b_col is not None:
                extracted_design = build_batch_design_matrix(expr_input.obs[b_col].to_numpy())
    elif isinstance(expr_input, pd.DataFrame):
        df_raw = expr_input.copy()
        df_raw.index = _extract_cell_names(expr_input)
        if extracted_design is None and create_design_mat:
            b_col = batch_col if (batch_col and batch_col in expr_input.columns) else None
            if b_col is None and COL_SAMPLE in expr_input.columns:
                b_col = COL_SAMPLE
            if b_col is not None and expr_input[b_col].nunique(dropna=True) > 1:
                extracted_design = build_batch_design_matrix(expr_input[b_col].to_numpy())
    else:
        raise TypeError(
            "expr_input must be a pd.DataFrame, AnnData, or (array, features, cell_names) tuple."
        )

    if requested_markers is not None:
        expr_df, _ = _resolve_requested_markers(
            df_raw, requested_markers, preferred_stat=preferred_stat
        )
    else:
        expr_df, _ = _infer_marker_columns(df_raw)

    expr_df = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float64)

    if extracted_design is not None:
        if isinstance(extracted_design, pd.DataFrame):
            design_arr = extracted_design.to_numpy(dtype=np.float64)
        else:
            design_arr = np.asarray(extracted_design, dtype=np.float64)
    else:
        design_arr = None

    return expr_df, design_arr


def preprocess_expression(
    expr_df: pd.DataFrame,
    *,
    apply_arcsinh: Union[bool, Literal["auto"]] = False,
    arcsinh_cofactor: float = 5.0,
    winsorize_percentiles: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, bool]:
    """Apply optional ``arcsinh(x / cofactor)`` transform and percentile winsorization.

    Returns ``(processed_df, did_arcsinh)``.
    """
    arr = expr_df.to_numpy(dtype=np.float64, copy=True)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    did_arcsinh = False
    if apply_arcsinh == "auto":
        finite_max = float(np.max(arr)) if arr.size > 0 else 0.0
        do_arcsinh = finite_max > 10.0
    else:
        do_arcsinh = bool(apply_arcsinh)

    if do_arcsinh and arcsinh_cofactor > 0:
        arr = np.arcsinh(arr / float(arcsinh_cofactor))
        did_arcsinh = True

    if winsorize_percentiles is not None and arr.shape[0] > 0:
        low_p, high_p = winsorize_percentiles
        q_low = np.percentile(arr, low_p, axis=0)
        q_high = np.percentile(arr, high_p, axis=0)
        arr = np.clip(arr, q_low, q_high)

    return pd.DataFrame(arr, index=expr_df.index, columns=expr_df.columns), did_arcsinh


class ASTIRDataset:
    """Single-cell expression dataset container mirroring ``astir.data.SCDataset``."""

    def __init__(
        self,
        expr_df: pd.DataFrame,
        marker_dict: Dict[str, List[str]],
        *,
        include_other_column: bool,
        design: Optional[np.ndarray] = None,
        dtype: Any = None,
        device: str = "cpu",
    ) -> None:
        torch = _import_torch()
        self._device = torch.device(device)
        self._dtype = _resolve_dtype(dtype, device)
        self._classes = [str(k) for k in marker_dict.keys()]

        available_cols = set(str(c) for c in expr_df.columns)
        filtered_dict: Dict[str, List[str]] = {}
        for cls_name in self._classes:
            valid = [m for m in marker_dict[cls_name] if m in available_cols]
            if valid:
                filtered_dict[cls_name] = valid
        self._marker_dict = filtered_dict
        self._classes = list(filtered_dict.keys())

        if len(self._classes) <= 1 and include_other_column:
            raise RuntimeError(
                "Classification failed. At least two cell types with valid markers are required."
            )
        if len(self._classes) == 0:
            raise RuntimeError("No classes with matching markers were found in expression data.")

        self._m_features = sorted({m for markers in filtered_dict.values() for m in markers})
        self._cell_names = list(expr_df.index)

        if expr_df.shape[0] <= 0:
            raise RuntimeError("Classification failed. At least one cell is required.")

        y_np = expr_df[self._m_features].to_numpy(dtype=np.float64)
        self._exprs = torch.from_numpy(y_np).to(device=self._device, dtype=self._dtype)
        self._marker_mat = self._construct_marker_mat(include_other_column)
        self._design = self._sanitize_design(design)

        # Match SCDataset: unbiased std + epsilon guard
        self._exprs_mean = self._exprs.mean(0) + 1e-6
        std_raw = (
            self._exprs.std(0)
            if self._exprs.shape[0] > 1
            else torch.zeros(len(self._m_features), device=self._device, dtype=self._dtype)
        )
        self._exprs_std = std_raw + 1e-6
        self._exprs_x = (self._exprs - self._exprs_mean) / self._exprs_std

    def _construct_marker_mat(self, include_other_column: bool) -> Any:
        torch = _import_torch()
        n_feat = len(self._m_features)
        n_cls = len(self._classes)
        n_cols = n_cls + 1 if include_other_column else n_cls
        rho = torch.zeros((n_feat, n_cols), dtype=self._dtype, device=self._device)
        for g, feat in enumerate(self._m_features):
            for c, cls_name in enumerate(self._classes):
                if feat in self._marker_dict[cls_name]:
                    rho[g, c] = 1.0
        return rho

    def _sanitize_design(self, design: Optional[np.ndarray]) -> Any:
        torch = _import_torch()
        n_cells = self._exprs.shape[0]
        if design is None:
            return torch.ones((n_cells, 1), dtype=self._dtype, device=self._device)
        arr = np.asarray(design, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] != n_cells:
            raise RuntimeError(
                "Number of rows of design matrix must equal number of rows of expression data."
            )
        return torch.from_numpy(arr).to(device=self._device, dtype=self._dtype)

    def __len__(self) -> int:
        return int(self._exprs.shape[0])

    def get_exprs(self) -> Any:
        return self._exprs

    def get_standardized(self) -> Any:
        return self._exprs_x

    def standardize_external(self, expr_df: pd.DataFrame) -> Any:
        """Standardize external expression DataFrame using training mean and std."""
        torch = _import_torch()
        y_np = expr_df[self._m_features].to_numpy(dtype=np.float64)
        y_t = torch.from_numpy(y_np).to(device=self._device, dtype=self._dtype)
        return (y_t - self._exprs_mean) / self._exprs_std

    def get_exprs_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._exprs.detach().cpu().numpy(),
            index=self._cell_names,
            columns=self._m_features,
        )

    def get_marker_mat(self) -> Any:
        return self._marker_mat

    def get_mu(self) -> Any:
        return self._exprs_mean

    def get_sigma(self) -> Any:
        return self._exprs_std

    def get_design(self) -> Any:
        return self._design

    def get_n_classes(self) -> int:
        return len(self._classes)

    def get_n_features(self) -> int:
        return len(self._m_features)

    def get_features(self) -> List[str]:
        return list(self._m_features)

    def get_classes(self) -> List[str]:
        return list(self._classes)

    def get_cell_names(self) -> List[Any]:
        return list(self._cell_names)

    def get_mu_init(self, n_putative_cells: int = 10) -> np.ndarray:
        """Compute baseline non-expressing mu initialization per marker."""
        df_exprs = self.get_exprs_df()
        k = min(max(1, n_putative_cells), len(df_exprs))

        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(df_exprs.to_numpy(dtype=np.float64))
        df_scaled = pd.DataFrame(scaled_vals, index=df_exprs.index, columns=df_exprs.columns)

        putative_cells: Dict[str, pd.Index] = {}
        for cell_type, type_markers in self._marker_dict.items():
            scored = df_scaled[type_markers].mean(axis=1)
            putative_cells[cell_type] = scored.nlargest(k).index

        mean_inits: List[float] = []
        for feature in self._m_features:
            no_feat_types = [c for c in self._classes if feature not in self._marker_dict[c]]
            if no_feat_types:
                idx = putative_cells[no_feat_types[0]]
                for ct in no_feat_types[1:]:
                    idx = idx.union(putative_cells[ct])
                val = float(df_exprs.loc[idx, feature].mean())
            else:
                val = float(df_exprs[feature].mean())
            mean_inits.append(max(val, 0.0) + 1e-6)

        return np.asarray(mean_inits, dtype=np.float64)


def _build_type_recognition_net(
    n_classes: int,
    n_features: int,
    hidden_size: int = 20,
    dtype: Any = None,
    device: str = "cpu",
) -> Any:
    """Build ASTIR ``TypeRecognitionNet`` (G -> 20 -> LeakyReLU -> C + 1)."""
    torch = _import_torch()
    nn = torch.nn
    F = torch.nn.functional

    class TypeRecognitionNet(nn.Module):
        def __init__(self, c_dim: int, g_dim: int, h_dim: int = 20) -> None:
            super().__init__()
            self.hidden_1 = nn.Linear(g_dim, h_dim)
            self.hidden_2 = nn.Linear(h_dim, c_dim + 1)

        def forward(self, x: Any) -> Tuple[Any, Any]:
            h = F.leaky_relu(self.hidden_1(x))
            logits = self.hidden_2(h)
            return F.softmax(logits, dim=1), F.log_softmax(logits, dim=1)

    return TypeRecognitionNet(n_classes, n_features, hidden_size).to(
        device=torch.device(device), dtype=dtype
    )


def _build_state_recognition_net(
    n_states: int,
    n_features: int,
    const: int = 2,
    dropout_rate: float = 0.0,
    batch_norm: bool = False,
    dtype: Any = None,
    device: str = "cpu",
) -> Any:
    """Build ASTIR ``StateRecognitionNet`` predicting Gaussian variational params."""
    torch = _import_torch()
    nn = torch.nn
    F = torch.nn.functional

    class StateRecognitionNet(nn.Module):
        def __init__(
            self,
            c_dim: int,
            g_dim: int,
            const_mult: int = 2,
            drop: float = 0.0,
            use_bn: bool = False,
        ) -> None:
            super().__init__()
            self.batch_norm = use_bn
            hidden_dim = max(1, math.ceil(const_mult * c_dim))
            self.linear1 = nn.Linear(g_dim, hidden_dim)
            self.dropout1 = nn.Dropout(drop)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.dropout2 = nn.Dropout(drop)
            self.linear3_mu = nn.Linear(hidden_dim, c_dim)
            self.dropout_mu = nn.Dropout(drop)
            self.linear3_std = nn.Linear(hidden_dim, c_dim)
            self.dropout_std = nn.Dropout(drop)
            if self.batch_norm:
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.bn2 = nn.BatchNorm1d(hidden_dim)
                self.bn_out_mu = nn.BatchNorm1d(c_dim)
                self.bn_out_std = nn.BatchNorm1d(c_dim)

        def forward(self, x: Any) -> Tuple[Any, Any]:
            h = self.linear1(x)
            if self.batch_norm:
                h = self.bn1(h)
            h = self.dropout1(F.relu(h))

            h = self.linear2(h)
            if self.batch_norm:
                h = self.bn2(h)
            h = self.dropout2(F.relu(h))

            mu_z = self.linear3_mu(h)
            if self.batch_norm:
                mu_z = self.bn_out_mu(mu_z)
            mu_z = self.dropout_mu(mu_z)

            std_z = self.linear3_std(h)
            if self.batch_norm:
                std_z = self.bn_out_std(std_z)
            std_z = self.dropout_std(std_z)
            return mu_z, std_z

    return StateRecognitionNet(
        n_states,
        n_features,
        const_mult=const,
        drop=dropout_rate,
        use_bn=batch_norm,
    ).to(device=torch.device(device), dtype=dtype)


class ASTIRCellTypeModel:
    """Variational Bayes cell-type inference model matching ``astir.models.CellTypeModel``."""

    def __init__(
        self,
        dset: ASTIRDataset,
        *,
        random_seed: int = 1234,
        dtype: Any = None,
        device: str = "cpu",
        hidden_size: int = 20,
    ) -> None:
        torch = _import_torch()
        self._dset = dset
        self._device = torch.device(device)
        self._dtype = _resolve_dtype(dtype, device)
        self._random_seed = int(random_seed)
        self._hidden_size = int(hidden_size)

        torch.manual_seed(self._random_seed)
        np.random.seed(self._random_seed)

        self._is_converged = False
        self._losses: List[float] = []
        self._assignment: pd.DataFrame = pd.DataFrame()
        self._init_parameters()

    def _init_parameters(self) -> None:
        torch = _import_torch()
        g_dim = self._dset.get_n_features()
        c_dim = self._dset.get_n_classes()
        p_dim = int(self._dset.get_design().shape[1])

        self._recog = _build_type_recognition_net(
            c_dim, g_dim, hidden_size=self._hidden_size, dtype=self._dtype, device=str(self._device)
        )
        self._rho = self._dset.get_marker_mat().to(self._device)
        self._alpha_prior = torch.distributions.Dirichlet(
            torch.ones(c_dim + 1, dtype=self._dtype, device=self._device) * float(c_dim + 1)
        )

        # Sample log_delta on CPU for cross-device seed parity, then move to target device
        normal_init = torch.distributions.Normal(
            torch.tensor(0.0, dtype=self._dtype),
            torch.tensor(0.1, dtype=self._dtype),
        )
        log_delta_init = normal_init.sample((g_dim, c_dim + 1)).to(self._device)

        mu_base = torch.log(
            torch.tensor(self._dset.get_mu_init(), dtype=self._dtype, device=self._device)
        ).reshape(-1, 1)
        if p_dim > 1:
            mu_init = torch.cat(
                [mu_base, torch.zeros((g_dim, p_dim - 1), dtype=self._dtype, device=self._device)],
                dim=1,
            )
        else:
            mu_init = mu_base

        self._variables: Dict[str, Any] = {
            "mu": mu_init.clone().detach().requires_grad_(True),
            "log_sigma": torch.log(self._dset.get_sigma()).to(self._device).detach().requires_grad_(True),
            "log_delta": log_delta_init.clone().detach().requires_grad_(True),
            "p": torch.zeros((g_dim, c_dim + 1), dtype=self._dtype, device=self._device).requires_grad_(True),
            "alpha_logits": torch.ones(c_dim + 1, dtype=self._dtype, device=self._device).requires_grad_(True),
        }

    def _elbo_loss(self, y_batch: Any, x_batch: Any, design_batch: Any) -> Any:
        torch = _import_torch()
        F = torch.nn.functional
        g_dim = self._dset.get_n_features()
        c_dim = self._dset.get_n_classes()
        n_batch = y_batch.shape[0]

        y_spread = y_batch.view(n_batch, 1, g_dim).expand(n_batch, c_dim + 1, g_dim)

        delta_tilde = torch.exp(torch.clamp(self._variables["log_delta"], min=-20.0, max=20.0))
        mean = delta_tilde * self._rho
        mean_design = torch.mm(design_batch, self._variables["mu"].T)
        mean = mean + mean_design.view(n_batch, g_dim, 1).expand(n_batch, g_dim, c_dim + 1)
        loc = torch.exp(torch.clamp(mean, min=-30.0, max=20.0)).permute(0, 2, 1)

        p_sig = torch.sigmoid(self._variables["p"])
        sigma = torch.exp(torch.clamp(self._variables["log_sigma"], min=-20.0, max=20.0))
        rho_p = self._rho * p_sig

        v1 = (rho_p.T * sigma).view(1, c_dim + 1, g_dim, 1).expand(n_batch, c_dim + 1, g_dim, 1)
        v2 = (
            (torch.pow(sigma, 2) * (1.0 - torch.pow(rho_p, 2)).T)
            .view(1, c_dim + 1, g_dim)
            .expand(n_batch, c_dim + 1, g_dim)
            + 1e-6
        )

        dist = torch.distributions.LowRankMultivariateNormal(
            loc=loc,
            cov_factor=v1.contiguous(),
            cov_diag=v2.contiguous(),
        )
        log_p_y_on_c = dist.log_prob(y_spread)

        gamma, log_gamma = self._recog(x_batch)
        log_alpha = F.log_softmax(self._variables["alpha_logits"], dim=0)
        alpha = torch.clamp(F.softmax(self._variables["alpha_logits"], dim=0), min=1e-12)
        alpha = alpha / alpha.sum()
        mix_prior = self._alpha_prior.log_prob(alpha)

        elbo = (gamma * (log_p_y_on_c + log_alpha - log_gamma)).sum() + mix_prior
        return -elbo

    def fit(
        self,
        *,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        delta_loss: float = 1e-3,
    ) -> None:
        torch = _import_torch()
        y_all = self._dset.get_exprs()
        x_all = self._dset.get_standardized()
        d_all = self._dset.get_design()
        n_cells = len(self._dset)
        bs = max(1, min(int(batch_size), n_cells))

        opt_params = list(self._variables.values()) + list(self._recog.parameters())
        optimizer = torch.optim.Adam(opt_params, lr=float(learning_rate))

        run_losses: List[float] = []
        for _ in range(int(max_epochs)):
            perm = torch.randperm(n_cells, device=self._device)
            epoch_loss = 0.0
            for start in range(0, n_cells, bs):
                idx = perm[start : start + bs]
                optimizer.zero_grad()
                loss = self._elbo_loss(y_all[idx], x_all[idx], d_all[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().item())

            rel_change = 1.0
            if run_losses and abs(run_losses[-1]) > 1e-12:
                rel_change = abs((epoch_loss - run_losses[-1]) / run_losses[-1])
            run_losses.append(epoch_loss)
            self._losses.append(epoch_loss)

            if len(run_losses) > 1 and rel_change <= delta_loss:
                self._is_converged = True
                break

        with torch.no_grad():
            gamma, _ = self._recog(x_all)
            probs_np = gamma.detach().cpu().numpy()

        self._assignment = pd.DataFrame(
            probs_np,
            index=self._dset.get_cell_names(),
            columns=self._dset.get_classes() + ["Other"],
        )

    def predict(self, expr_df: pd.DataFrame) -> pd.DataFrame:
        torch = _import_torch()
        x_t = self._dset.standardize_external(expr_df)
        with torch.no_grad():
            gamma, _ = self._recog(x_t)
            probs_np = gamma.detach().cpu().numpy()
        return pd.DataFrame(
            probs_np,
            index=list(expr_df.index),
            columns=self._dset.get_classes() + ["Other"],
        )

    def get_assignment(self) -> pd.DataFrame:
        if self._assignment.empty:
            raise RuntimeError("The cell-type model has not been trained yet.")
        return self._assignment.copy()

    def get_losses(self) -> np.ndarray:
        return np.asarray(self._losses, dtype=np.float64)

    def is_converged(self) -> bool:
        return self._is_converged


class ASTIRCellStateModel:
    """Variational Bayes cell-state inference model matching ``astir.models.CellStateModel``."""

    def __init__(
        self,
        dset: ASTIRDataset,
        *,
        const: int = 2,
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        random_seed: int = 42,
        dtype: Any = None,
        device: str = "cpu",
    ) -> None:
        torch = _import_torch()
        self._dset = dset
        self._const = int(const)
        self._dropout_rate = float(dropout_rate)
        self._batch_norm = bool(batch_norm)
        self._random_seed = int(random_seed)
        self._device = torch.device(device)
        self._dtype = _resolve_dtype(dtype, device)

        torch.manual_seed(self._random_seed)
        np.random.seed(self._random_seed)

        self._optimizer: Optional[Any] = None
        self._is_converged = False
        self._losses: List[float] = []
        self._assignment: pd.DataFrame = pd.DataFrame()
        self._init_parameters()

    def _init_parameters(self) -> None:
        torch = _import_torch()
        c_dim = self._dset.get_n_classes()
        g_dim = self._dset.get_n_features()

        unif = torch.distributions.Uniform(
            torch.tensor(0.0, dtype=self._dtype),
            torch.tensor(1.5, dtype=self._dtype),
        )
        log_w_init = torch.log(torch.clamp(unif.sample((c_dim, g_dim)), min=1e-6)).to(self._device)

        self._variables: Dict[str, Any] = {
            "log_sigma": torch.log(self._dset.get_sigma().mean()).to(self._device).detach().requires_grad_(True),
            "mu": self._dset.get_mu().reshape(1, -1).to(self._device).detach().requires_grad_(True),
            "log_w": log_w_init.clone().detach().requires_grad_(True),
        }
        self._rho = self._dset.get_marker_mat().T.to(self._device)
        self._recog = _build_state_recognition_net(
            c_dim,
            g_dim,
            const=self._const,
            dropout_rate=self._dropout_rate,
            batch_norm=self._batch_norm,
            dtype=self._dtype,
            device=str(self._device),
        )

    def _forward(self, x_in: Any) -> Tuple[Any, Any, Any]:
        torch = _import_torch()
        mu_z, std_z = self._recog(x_in)
        std = torch.exp(torch.clamp(std_z, min=-20.0, max=10.0))
        eps = torch.randn_like(std)
        z_sample = eps * std + mu_z
        return mu_z, std_z, z_sample

    def _loss_fn(self, mu_z: Any, std_z: Any, z_sample: Any, x_in: Any) -> Any:
        torch = _import_torch()
        n_batch = max(1, int(x_in.shape[0]))

        q_z = torch.distributions.Normal(
            loc=mu_z, scale=torch.exp(torch.clamp(std_z, min=-20.0, max=10.0))
        )
        log_q_z = q_z.log_prob(z_sample)

        rho_w = self._rho * torch.exp(torch.clamp(self._variables["log_w"], min=-20.0, max=20.0))
        mean = self._variables["mu"] + torch.matmul(z_sample, rho_w)
        std = torch.exp(torch.clamp(self._variables["log_sigma"], min=-20.0, max=10.0)).reshape(1, -1)
        p_y_given_z = torch.distributions.Normal(loc=mean, scale=std)
        log_p_y_given_z = p_y_given_z.log_prob(x_in)

        p_z = torch.distributions.Normal(
            torch.zeros_like(z_sample),
            torch.ones_like(z_sample),
        )
        log_p_z = p_z.log_prob(z_sample)

        return (1.0 / n_batch) * (
            torch.sum(log_q_z) - torch.sum(log_p_y_given_z) - torch.sum(log_p_z)
        )

    def fit(
        self,
        *,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        delta_loss: float = 1e-3,
        delta_loss_batch: int = 10,
    ) -> None:
        torch = _import_torch()
        if self._is_converged:
            return

        if self._optimizer is None:
            opt_params = list(self._recog.parameters()) + list(self._variables.values())
            self._optimizer = torch.optim.Adam(opt_params, lr=float(learning_rate))

        x_all = self._dset.get_standardized()
        n_cells = len(self._dset)
        bs = max(1, min(int(batch_size), n_cells))

        self._recog.train()
        for _ in range(int(max_epochs)):
            last_loss_val = 0.0
            for start in range(0, n_cells, bs):
                x_batch = x_all[start : start + bs]
                if self._batch_norm and x_batch.shape[0] < 2:
                    continue
                self._optimizer.zero_grad()
                mu_z, std_z, z_sample = self._forward(x_batch)
                loss = self._loss_fn(mu_z, std_z, z_sample, x_batch)
                loss.backward()
                self._optimizer.step()
                last_loss_val = float(loss.detach().cpu().item())

            self._losses.append(last_loss_val)

            if len(self._losses) > delta_loss_batch:
                curr_mean = float(np.mean(self._losses[-delta_loss_batch:]))
                prev_mean = float(np.mean(self._losses[-delta_loss_batch - 1 : -1]))
                if abs(prev_mean) > 1e-12:
                    curr_delta = (prev_mean - curr_mean) / prev_mean
                    if 0.0 <= curr_delta < delta_loss:
                        self._is_converged = True
                        break

        self._recog.eval()
        with torch.no_grad():
            mu_z, _ = self._recog(x_all)
            g_np = mu_z.detach().cpu().numpy()

        self._assignment = pd.DataFrame(
            g_np,
            index=self._dset.get_cell_names(),
            columns=self._dset.get_classes(),
        )

    def predict_raw(self, expr_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        torch = _import_torch()
        self._recog.eval()
        x_in = (
            self._dset.get_standardized()
            if expr_df is None
            else self._dset.standardize_external(expr_df)
        )
        with torch.no_grad():
            mu_z, _ = self._recog(x_in)
            g_np = mu_z.detach().cpu().numpy()
        idx = self._dset.get_cell_names() if expr_df is None else list(expr_df.index)
        return pd.DataFrame(g_np, index=idx, columns=self._dset.get_classes())

    def get_assignment(self) -> pd.DataFrame:
        if self._assignment.empty:
            raise RuntimeError("The cell-state model has not been trained yet.")
        return self._assignment.copy()

    def get_losses(self) -> np.ndarray:
        return np.asarray(self._losses, dtype=np.float64)

    def is_converged(self) -> bool:
        return self._is_converged


def assign_celltypes_from_probabilities(
    prob_df: pd.DataFrame,
    *,
    threshold: float = 0.7,
    assignment_type: Literal["threshold", "max"] = "threshold",
) -> pd.DataFrame:
    """Convert cell-type posterior matrix into discrete labels with ``Unknown`` gating."""
    if assignment_type not in ("threshold", "max"):
        warnings.warn(
            f"Unknown assignment_type {assignment_type!r}; defaulting to 'threshold'.",
            UserWarning,
            stacklevel=2,
        )
        assignment_type = "threshold"

    probs = prob_df.to_numpy(dtype=np.float64)
    cell_types = [str(c) for c in prob_df.columns]
    if probs.shape[0] == 0:
        return pd.DataFrame({"cell_type": pd.Series(dtype=object)}, index=prob_df.index)

    max_probs = probs.max(axis=1)
    argmax_idx = probs.argmax(axis=1)

    labels: List[str] = []
    for row_idx in range(probs.shape[0]):
        mp = float(max_probs[row_idx])
        if assignment_type == "threshold":
            if mp < threshold:
                labels.append("Unknown")
                continue
        elif assignment_type == "max":
            if int(np.sum(probs[row_idx] == mp)) > 1:
                labels.append("Unknown")
                continue
        labels.append(cell_types[int(argmax_idx[row_idx])])

    return pd.DataFrame({"cell_type": labels}, index=prob_df.index)


def compute_hierarchy_probabilities(
    prob_df: pd.DataFrame,
    hierarchy_dict: Mapping[str, Any],
    *,
    depth: int = 1,
) -> pd.DataFrame:
    """Aggregate cell-type posterior probabilities along a hierarchical dictionary."""

    def _sum_subtree(sub: Any) -> pd.Series:
        if isinstance(sub, (list, tuple)):
            cols = [c for c in sub if c in prob_df.columns]
            if not cols:
                return pd.Series(0.0, index=prob_df.index)
            return prob_df[cols].sum(axis=1)
        acc = pd.Series(0.0, index=prob_df.index)
        for child in sub.values():
            acc = acc + _sum_subtree(child)
        return acc

    def _collect_level(out_df: pd.DataFrame, dic: Mapping[str, Any], cur_depth: int) -> None:
        if cur_depth <= 1:
            for key, val in dic.items():
                out_df[str(key)] = _sum_subtree(val)
        else:
            for _, cells in dic.items():
                if isinstance(cells, (list, tuple)):
                    for cell in cells:
                        if cell in prob_df.columns:
                            out_df[str(cell)] = prob_df[cell]
                elif isinstance(cells, Mapping):
                    _collect_level(out_df, cells, cur_depth - 1)

    hier_df = pd.DataFrame(index=prob_df.index)
    _collect_level(hier_df, hierarchy_dict, depth)
    return hier_df


def run_celltype_diagnostics(
    expr_df: pd.DataFrame,
    rho: np.ndarray,
    classes: Sequence[str],
    features: Sequence[str],
    assigned_labels: Sequence[str],
    *,
    alpha: float = 0.01,
) -> pd.DataFrame:
    """Check whether each assigned cell type over-expresses its canonical markers."""
    col_names = [
        "feature",
        "should be expressed higher in",
        "than",
        "mean cell type 1",
        "mean cell type 2",
        "p-value",
        "note",
    ]
    g_df = pd.DataFrame(rho, index=list(features), columns=list(classes) + ["Other"])
    expr_np = expr_df[list(features)].to_numpy(dtype=np.float64)
    labels_arr = np.asarray(list(assigned_labels), dtype=object)

    problems: List[Dict[str, Any]] = []
    for curr_type in classes:
        if curr_type not in labels_arr:
            continue
        curr_markers = g_df.index[g_df[curr_type] == 1]
        for marker in curr_markers:
            feat_idx = list(features).index(marker)
            other_types = g_df.columns[g_df.loc[marker] == 0]
            for comp_type in other_types:
                if comp_type not in labels_arr:
                    continue
                x = expr_np[labels_arr == curr_type, feat_idx]
                y = expr_np[labels_arr == comp_type, feat_idx]
                stat = np.nan
                pval = np.inf
                note: Optional[str] = "Only 1 cell in a type: comparison not possible"
                if len(x) > 1 and len(y) > 1:
                    tt = stats.ttest_ind(x, y)
                    stat = float(tt.statistic)
                    pval = float(tt.pvalue)
                    note = None
                if not (stat > 0.0 and pval < alpha):
                    problems.append(
                        {
                            "feature": marker,
                            "should be expressed higher in": curr_type,
                            "than": comp_type,
                            "mean cell type 1": float(np.mean(x)) if len(x) > 0 else np.nan,
                            "mean cell type 2": float(np.mean(y)) if len(y) > 0 else np.nan,
                            "p-value": pval,
                            "note": note,
                        }
                    )

    if not problems:
        return pd.DataFrame(columns=col_names)
    return pd.DataFrame(problems)[col_names]


def run_cellstate_diagnostics(
    expr_df: pd.DataFrame,
    state_scores: pd.DataFrame,
    rho: np.ndarray,
    state_names: Sequence[str],
    features: Sequence[str],
) -> pd.DataFrame:
    """Flag non-marker proteins more correlated with a state than its weakest marker."""
    col_names = [
        "pathway",
        "protein A",
        "correlation of protein A",
        "protein B",
        "correlation of protein B",
        "note",
    ]
    expr_np = expr_df[list(features)].to_numpy(dtype=np.float64)
    state_np = state_scores[list(state_names)].to_numpy(dtype=np.float64)

    n_states = len(state_names)
    n_feat = len(features)
    corr_mat = np.zeros((n_states, n_feat), dtype=np.float64)
    for c in range(n_states):
        for g in range(n_feat):
            s_vec = state_np[:, c]
            p_vec = expr_np[:, g]
            if np.std(s_vec) > 0 and np.std(p_vec) > 0:
                corr_mat[c, g] = float(np.corrcoef(p_vec, s_vec)[0, 1])

    rho_state = rho.T  # (n_states, n_feat)
    marker_corr = np.where(rho_state == 1, corr_mat, np.inf)
    min_marker_corr = np.min(marker_corr, axis=1).reshape(-1, 1)
    min_marker_idx = np.argmin(marker_corr, axis=1)

    non_marker_corr = np.where(rho_state == 0, corr_mat, -np.inf)
    bad = np.argwhere(non_marker_corr > min_marker_corr)

    problems: List[Dict[str, Any]] = []
    for c_idx, g_idx in bad:
        state_name = state_names[int(c_idx)]
        marker_prot = features[int(min_marker_idx[int(c_idx)])]
        non_marker_prot = features[int(g_idx)]
        problems.append(
            {
                "pathway": state_name,
                "protein A": marker_prot,
                "correlation of protein A": float(min_marker_corr[int(c_idx), 0]),
                "protein B": non_marker_prot,
                "correlation of protein B": float(non_marker_corr[int(c_idx), int(g_idx)]),
                "note": f"{marker_prot} is marker for {state_name} but {non_marker_prot} isn't",
            }
        )

    if not problems:
        return pd.DataFrame(columns=col_names)
    return pd.DataFrame(problems)[col_names]


class ASTIRPipeline:
    """End-to-end ASTIR runner supporting both IRIS PyTorch VI and official ``astir``."""

    def __init__(
        self,
        expr_input: Union[
            pd.DataFrame,
            Any,
            Tuple[Union[np.ndarray, Any], Sequence[str], Sequence[Any]],
        ],
        marker_config: Union[str, Path, Mapping[str, Any], CellTypePanel],
        *,
        marker_panel: Optional[MarkerPanel] = None,
        design: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        batch_col: Optional[str] = "batch",
        create_design_mat: bool = True,
        apply_arcsinh: Union[bool, Literal["auto"]] = False,
        arcsinh_cofactor: float = 5.0,
        winsorize_percentiles: Optional[Tuple[float, float]] = None,
        preferred_stat: str = "mean",
        random_seed: int = 1234,
        dtype: Optional[Union[str, Any]] = None,
        device: Optional[str] = None,
        engine: Literal["auto", "iris", "official"] = "iris",
    ) -> None:
        self.random_seed = int(random_seed)
        self.preferred_stat = preferred_stat
        self.device = resolve_device(device)

        type_dict, state_dict, hierarchy_dict = normalize_marker_config(
            marker_config, marker_panel=marker_panel
        )
        self._type_dict = type_dict
        self._state_dict = state_dict
        self._hierarchy_dict = hierarchy_dict

        all_markers: List[str] = []
        if type_dict:
            all_markers.extend(m for ms in type_dict.values() for m in ms)
        if state_dict:
            all_markers.extend(m for ms in state_dict.values() for m in ms)
        unique_markers = sorted(set(all_markers)) if all_markers else None

        raw_df, design_arr = extract_expression_and_design(
            expr_input,
            requested_markers=unique_markers,
            preferred_stat=preferred_stat,
            design=design,
            batch_col=batch_col,
            create_design_mat=create_design_mat,
        )
        self._expr_df, self.did_arcsinh = preprocess_expression(
            raw_df,
            apply_arcsinh=apply_arcsinh,
            arcsinh_cofactor=arcsinh_cofactor,
            winsorize_percentiles=winsorize_percentiles,
        )
        self._design_arr = design_arr

        if engine == "auto":
            self.engine_used = "official" if is_official_astir_available() else "iris_pytorch"
        elif engine == "official":
            if not is_official_astir_available():
                raise ImportError("Official 'astir' package is not installed in this environment.")
            self.engine_used = "official"
        elif engine == "iris":
            self.engine_used = "iris_pytorch"
        else:
            raise ValueError(f"Unsupported engine {engine!r}; use 'auto', 'iris', or 'official'.")

        self._dtype = _resolve_dtype(dtype, self.device)
        self._official_astir: Optional[Any] = None
        self._type_dset: Optional[ASTIRDataset] = None
        self._state_dset: Optional[ASTIRDataset] = None
        self._type_model: Optional[ASTIRCellTypeModel] = None
        self._state_model: Optional[ASTIRCellStateModel] = None
        self._type_losses: Optional[np.ndarray] = None
        self._state_losses: Optional[np.ndarray] = None

        if self.engine_used == "official":
            from astir.astir import Astir

            marker_payload: Dict[str, Any] = {}
            if type_dict is not None:
                marker_payload["cell_types"] = type_dict
            if state_dict is not None:
                marker_payload["cell_states"] = state_dict
            if hierarchy_dict is not None:
                marker_payload["hierarchy"] = hierarchy_dict
            self._official_astir = Astir(
                self._expr_df,
                marker_payload,
                design=self._design_arr,
                random_seed=self.random_seed,
                dtype=self._dtype,
            )
        else:
            if type_dict is not None:
                self._type_dset = ASTIRDataset(
                    self._expr_df,
                    type_dict,
                    include_other_column=True,
                    design=self._design_arr,
                    dtype=self._dtype,
                    device=self.device,
                )
            if state_dict is not None:
                self._state_dset = ASTIRDataset(
                    self._expr_df,
                    state_dict,
                    include_other_column=False,
                    design=self._design_arr,
                    dtype=self._dtype,
                    device=self.device,
                )

    def fit_type(
        self,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        delta_loss: float = 1e-3,
        n_init: int = 5,
        n_init_epochs: int = 5,
    ) -> "ASTIRPipeline":
        """Fit cell-type Bayesian variational inference model with multi-start restarts."""
        if self.engine_used == "official":
            assert self._official_astir is not None
            self._official_astir.fit_type(
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                delta_loss=delta_loss,
                n_init=n_init,
                n_init_epochs=n_init_epochs,
            )
            self._type_losses = self._official_astir.get_type_losses().detach().cpu().numpy()
            return self

        if self._type_dset is None:
            raise RuntimeError("No cell_types markers were provided.")

        np.random.seed(self.random_seed)
        seeds = np.random.randint(1, 100_000_000, size=max(1, int(n_init)))
        init_epochs = min(int(max_epochs), max(1, int(n_init_epochs)))

        candidates: List[ASTIRCellTypeModel] = []
        final_losses: List[float] = []
        for seed in seeds:
            model = ASTIRCellTypeModel(
                self._type_dset,
                random_seed=int(seed),
                dtype=self._dtype,
                device=self.device,
            )
            model.fit(
                max_epochs=init_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                delta_loss=delta_loss,
            )
            candidates.append(model)
            final_losses.append(float(model.get_losses()[-1]))

        best_idx = int(np.argmin(final_losses))
        best_model = candidates[best_idx]
        remaining_epochs = max(1, int(max_epochs))
        best_model.fit(
            max_epochs=remaining_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            delta_loss=delta_loss,
        )
        self._type_model = best_model
        self._type_losses = best_model.get_losses()
        iris_log(
            "IRIS-Phenotyper",
            "ASTIR",
            f"Fitted cell-type VI model ({len(self._type_dset):,} cells, "
            f"{len(self._type_dset.get_classes())} cell types, epochs={remaining_epochs}).",
        )
        return self

    def fit_state(
        self,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        delta_loss: float = 1e-3,
        n_init: int = 5,
        n_init_epochs: int = 5,
        delta_loss_batch: int = 10,
        const: int = 2,
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
    ) -> "ASTIRPipeline":
        """Fit cell-state Bayesian variational inference model."""
        if self.engine_used == "official":
            assert self._official_astir is not None
            self._official_astir.fit_state(
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                delta_loss=delta_loss,
                n_init=n_init,
                n_init_epochs=n_init_epochs,
                delta_loss_batch=delta_loss_batch,
                const=const,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
            )
            losses_t = self._official_astir.get_state_losses()
            self._state_losses = (
                losses_t.detach().cpu().numpy() if hasattr(losses_t, "detach") else np.asarray(losses_t)
            )
            return self

        if self._state_dset is None:
            raise RuntimeError("No cell_states markers were provided.")

        init_epochs = min(int(max_epochs), max(1, int(n_init_epochs)))
        candidates: List[ASTIRCellStateModel] = []
        mean_losses: List[float] = []

        for i in range(max(1, int(n_init))):
            model = ASTIRCellStateModel(
                self._state_dset,
                const=const,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                random_seed=self.random_seed + i,
                dtype=self._dtype,
                device=self.device,
            )
            model.fit(
                max_epochs=init_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                delta_loss=delta_loss,
                delta_loss_batch=delta_loss_batch,
            )
            candidates.append(model)
            losses = model.get_losses()[:init_epochs]
            mean_losses.append(float(np.mean(losses[-delta_loss_batch:])))

        best_model = candidates[int(np.argmin(mean_losses))]
        best_model.fit(
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            delta_loss=delta_loss,
            delta_loss_batch=delta_loss_batch,
        )
        self._state_model = best_model
        self._state_losses = best_model.get_losses()
        return self

    def get_celltype_probabilities(self) -> pd.DataFrame:
        """Return posterior probabilities ``(N, C + 1)`` over cell types and ``Other``."""
        if self.engine_used == "official":
            assert self._official_astir is not None
            return self._official_astir.get_celltype_probabilities()
        if self._type_model is None:
            raise RuntimeError("The cell-type model has not been trained yet.")
        return self._type_model.get_assignment()

    def get_celltypes(
        self,
        threshold: float = 0.7,
        assignment_type: Literal["threshold", "max"] = "threshold",
    ) -> pd.DataFrame:
        """Return discrete cell-type assignments with ``Unknown`` below ``threshold``."""
        if self.engine_used == "official":
            assert self._official_astir is not None
            return self._official_astir.get_celltypes(
                threshold=threshold, assignment_type=assignment_type
            )
        prob_df = self.get_celltype_probabilities()
        return assign_celltypes_from_probabilities(
            prob_df, threshold=threshold, assignment_type=assignment_type
        )

    def predict_celltypes(
        self,
        new_expr: Union[pd.DataFrame, Any],
        *,
        threshold: float = 0.7,
        assignment_type: Literal["threshold", "max"] = "threshold",
        apply_arcsinh: Union[bool, Literal["auto"]] = False,
        arcsinh_cofactor: float = 5.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Predict cell-type probabilities and labels on unseen data using recognition net."""
        if self.engine_used == "official":
            assert self._official_astir is not None
            features = self._official_astir.get_type_dataset().get_features()
            raw_df, _ = extract_expression_and_design(
                new_expr, requested_markers=features, preferred_stat=self.preferred_stat
            )
            proc_df, _ = preprocess_expression(
                raw_df, apply_arcsinh=apply_arcsinh, arcsinh_cofactor=arcsinh_cofactor
            )
            from astir.data import SCDataset

            new_dset = SCDataset(
                proc_df,
                self._official_astir.get_type_dataset()._marker_dict,
                include_other_column=True,
                dtype=self._dtype,
                device=torch_device(self.device),
            )
            probs = self._official_astir.predict_celltypes(new_dset)
            labels = assign_celltypes_from_probabilities(
                probs, threshold=threshold, assignment_type=assignment_type
            )
            return labels, probs

        if self._type_model is None or self._type_dset is None:
            raise RuntimeError("The cell-type model has not been trained yet.")

        features = self._type_dset.get_features()
        raw_df, _ = extract_expression_and_design(
            new_expr,
            requested_markers=features,
            preferred_stat=self.preferred_stat,
            create_design_mat=False,
        )
        proc_df, _ = preprocess_expression(
            raw_df, apply_arcsinh=apply_arcsinh, arcsinh_cofactor=arcsinh_cofactor
        )
        probs = self._type_model.predict(proc_df)
        labels = assign_celltypes_from_probabilities(
            probs, threshold=threshold, assignment_type=assignment_type
        )
        return labels, probs

    def get_cellstates(self) -> pd.DataFrame:
        """Return min-max rescaled cell-state activations in ``[0, 1]``."""
        if self.engine_used == "official":
            assert self._official_astir is not None
            return self._official_astir.get_cellstates()
        if self._state_model is None:
            raise RuntimeError("The cell-state model has not been trained yet.")
        raw = self._state_model.get_assignment()
        denom = (raw.max(axis=0) - raw.min(axis=0)).replace(0.0, 1.0)
        return (raw - raw.min(axis=0)) / denom

    def assign_celltype_hierarchy(self, depth: int = 1) -> pd.DataFrame:
        """Return summed cell-type probabilities at the requested hierarchy depth."""
        if self.engine_used == "official":
            assert self._official_astir is not None
            return self._official_astir.assign_celltype_hierarchy(depth=depth)
        if not self._hierarchy_dict:
            raise RuntimeError("No hierarchy dictionary was provided in marker_config.")
        prob_df = self.get_celltype_probabilities()
        return compute_hierarchy_probabilities(prob_df, self._hierarchy_dict, depth=depth)

    def diagnostics_celltype(self, threshold: float = 0.5, alpha: float = 0.01) -> pd.DataFrame:
        """Run marker over-expression t-test diagnostics across assigned cell types."""
        if self.engine_used == "official":
            assert self._official_astir is not None
            return self._official_astir.diagnostics_celltype(threshold=threshold, alpha=alpha)
        if self._type_dset is None or self._type_model is None:
            raise RuntimeError("The cell-type model has not been trained yet.")
        labels = list(self.get_celltypes(threshold=threshold)["cell_type"])
        rho = self._type_dset.get_marker_mat().detach().cpu().numpy()
        return run_celltype_diagnostics(
            self._type_dset.get_exprs_df(),
            rho,
            self._type_dset.get_classes(),
            self._type_dset.get_features(),
            labels,
            alpha=alpha,
        )

    def diagnostics_cellstate(self) -> pd.DataFrame:
        """Run correlation-based marker specificity diagnostics for cell states."""
        if self.engine_used == "official":
            assert self._official_astir is not None
            return self._official_astir.diagnostics_cellstate()
        if self._state_dset is None or self._state_model is None:
            raise RuntimeError("The cell-state model has not been trained yet.")
        raw_states = self._state_model.get_assignment()
        rho = self._state_dset.get_marker_mat().detach().cpu().numpy()
        return run_cellstate_diagnostics(
            self._state_dset.get_exprs_df(),
            raw_states,
            rho,
            self._state_dset.get_classes(),
            self._state_dset.get_features(),
        )

    def get_type_losses(self) -> np.ndarray:
        """Return training ELBO loss trajectory for cell-type fitting."""
        if self._type_losses is None:
            raise RuntimeError("The cell-type model has not been trained yet.")
        return np.asarray(self._type_losses, dtype=np.float64)

    def get_state_losses(self) -> np.ndarray:
        """Return training ELBO loss trajectory for cell-state fitting."""
        if self._state_losses is None:
            raise RuntimeError("The cell-state model has not been trained yet.")
        return np.asarray(self._state_losses, dtype=np.float64)


def torch_device(device: str) -> Any:
    """Return ``torch.device`` for the given device string."""
    torch = _import_torch()
    return torch.device(device)
