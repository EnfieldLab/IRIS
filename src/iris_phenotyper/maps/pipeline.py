"""
MAPS Single-Cell Phenotyping Pipeline
=====================================
Faithful PyTorch implementation of MAPS (Machine learning for Analysis of
Proteomics in Spatial biology; Shaban et al., Nature Communications 2024,
DOI: 10.1038/s41467-023-44188-w).

MAPS is a supervised feed-forward neural network (4-layer MLP with ReLU and
Dropout followed by a linear classifier) trained with class-balanced weighted
sampling and z-score standardization on single-cell marker expressions plus
cell size/area.
"""
from __future__ import annotations

import copy
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# Guard against macOS dual-libomp crashes when PyPI torch and conda numpy/scipy coexist.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, SequentialSampler, WeightedRandomSampler

    if sys.platform == "darwin":
        try:
            torch.set_num_threads(1)
        except RuntimeError:
            pass
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - guarded fallback when torch is absent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    Dataset = object  # type: ignore[assignment,misc]
    SequentialSampler = None  # type: ignore[assignment,misc]
    WeightedRandomSampler = None  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False


DEFAULT_SEED = 7325111
DEFAULT_SCALE_DIVISOR = 255.0


def is_maps_available() -> bool:
    """Return True if PyTorch and required scientific Python packages are importable."""
    return _TORCH_AVAILABLE


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for MAPS training and inference. "
            "Install torch>=2.0 to enable MAPSPhenotyper."
        )


def resolve_device(device: str = "auto") -> "torch.device":
    """Resolve execution device across ``cpu``, ``cuda``, and Apple Silicon ``mps``.

    Parameters
    ----------
    device:
        Target device specification (``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:0"``,
        or ``"mps"``).

    Returns
    -------
    torch.device
    """
    _require_torch()
    req = (device or "auto").strip().lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if req.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested device '{device}' is not available on this host.")
    if req == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError("Requested device 'mps' is not available on this host.")
    return torch.device(req)


def resolve_dtype(device: "torch.device", dtype: str = "auto") -> "torch.dtype":
    """Resolve tensor dtype matching upstream float64 on CPU/CUDA and float32 on MPS."""
    _require_torch()
    req = (dtype or "auto").strip().lower()
    if req == "auto":
        return torch.float32 if device.type == "mps" else torch.float64
    if req in ("float32", "fp32", "float"):
        return torch.float32
    if req in ("float64", "fp64", "double"):
        if device.type == "mps":
            raise ValueError("MPS backend does not support float64; use dtype='float32' or 'auto'.")
        return torch.float64
    raise ValueError(f"Unsupported dtype '{dtype}'. Expected 'auto', 'float32', or 'float64'.")


def set_global_seed(seed: int, device: Optional["torch.device"] = None) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible training."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if device is not None and device.type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def standardize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    scale_divisor: float = DEFAULT_SCALE_DIVISOR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardize cell features and apply upstream ``scale_divisor`` scaling.

    Matches ``CellExpressionCSV`` in ``mahmoodlab/MAPS``:
    ``((X - mean) / std) / scale_divisor`` where ``scale_divisor = 255.0`` by default.
    Zero-variance features safely fall back to unit denominator without altering
    non-constant features.
    """
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature array (N, F), got shape {arr.shape}.")
    if scale_divisor <= 0.0:
        raise ValueError(f"scale_divisor must be positive, got {scale_divisor}.")

    if mean is None:
        computed_mean = np.mean(arr, axis=0)
    else:
        computed_mean = np.asarray(mean, dtype=np.float64)
        if computed_mean.shape != (arr.shape[1],):
            raise ValueError(
                f"mean shape {computed_mean.shape} does not match {arr.shape[1]} features."
            )

    if std is None:
        computed_std = np.std(arr, axis=0)
    else:
        computed_std = np.asarray(std, dtype=np.float64)
        if computed_std.shape != (arr.shape[1],):
            raise ValueError(
                f"std shape {computed_std.shape} does not match {arr.shape[1]} features."
            )

    safe_std = np.where(computed_std > 1e-12, computed_std, 1.0)
    scaled = ((arr - computed_mean) / safe_std) / float(scale_divisor)
    return scaled, computed_mean, computed_std


if _TORCH_AVAILABLE:

    class MAPSMLP(nn.Module):
        """Multi-Layer Perceptron matching ``mahmoodlab/MAPS`` ``networks.MLP``.

        Architecture: 4x [Linear -> ReLU -> Dropout] hidden blocks followed by
        a linear classification head returning ``(logits, probs)``.
        Module attribute names (``fc``, ``classifier``) match upstream checkpoints
        so official pretrained ``.pt`` weights load with ``strict=True``.
        """

        def __init__(
            self,
            input_dim: int = 47,
            hidden_dim: int = 512,
            num_classes: int = 12,
            dropout: float = 0.10,
        ) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.hidden_dim = int(hidden_dim)
            self.num_classes = int(num_classes)
            self.dropout = float(dropout)

            self.fc = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
            )
            self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

        def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Run forward pass returning ``(logits, softmax_probabilities)``."""
            features = self.fc(batch)
            logits = self.classifier(features)
            probs = torch.softmax(logits, dim=-1)
            return logits, probs

    class MAPSTabularDataset(Dataset):
        """In-memory and CSV dataset mirroring ``CellExpressionCSV``."""

        def __init__(
            self,
            features: np.ndarray,
            labels: Optional[np.ndarray] = None,
            is_train: bool = True,
            mean: Optional[np.ndarray] = None,
            std: Optional[np.ndarray] = None,
            scale_divisor: float = DEFAULT_SCALE_DIVISOR,
            tensor_dtype: torch.dtype = torch.float64,
        ) -> None:
            self.is_train = bool(is_train)
            self.scale_divisor = float(scale_divisor)
            self.tensor_dtype = tensor_dtype

            if not is_train and (mean is None or std is None):
                raise ValueError("mean and std must be provided when is_train=False.")

            scaled, self.mean, self.std = standardize_features(
                features,
                mean=mean if not is_train else None,
                std=std if not is_train else None,
                scale_divisor=self.scale_divisor,
            )
            np_dtype = np.float32 if tensor_dtype == torch.float32 else np.float64
            self.x = np.ascontiguousarray(scaled, dtype=np_dtype)
            self.y = None if labels is None else np.asarray(labels, dtype=np.int64)

        @classmethod
        def from_csv(
            cls,
            csv_path: Union[str, Path],
            label_col: str = "cell_label",
            is_train: bool = True,
            mean: Optional[np.ndarray] = None,
            std: Optional[np.ndarray] = None,
            scale_divisor: float = DEFAULT_SCALE_DIVISOR,
            tensor_dtype: torch.dtype = torch.float64,
        ) -> "MAPSTabularDataset":
            """Load a MAPS CSV file containing markers + cellSize (+ optional label_col)."""
            df = pd.read_csv(csv_path)
            y: Optional[np.ndarray] = None
            if label_col in df.columns:
                y = df[label_col].to_numpy(dtype=np.int64)
                df = df.drop(columns=[label_col])
            return cls(
                features=df.to_numpy(dtype=np.float64),
                labels=y,
                is_train=is_train,
                mean=mean,
                std=std,
                scale_divisor=scale_divisor,
                tensor_dtype=tensor_dtype,
            )

        def __len__(self) -> int:
            return int(self.x.shape[0])

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
            feat = torch.from_numpy(self.x[idx])
            gt = int(self.y[idx]) if self.y is not None else -1
            return feat, gt

        @staticmethod
        def get_data_loader(
            dataset: "MAPSTabularDataset",
            batch_size: int = 128,
            is_train: bool = False,
            num_workers: int = 0,
            use_class_balanced_sampler: bool = True,
            seed: Optional[int] = None,
        ) -> DataLoader:
            """Create a DataLoader with class-balanced WeightedRandomSampler for training."""
            if is_train and use_class_balanced_sampler and dataset.y is not None:
                labels = dataset.y
                _, inverse, counts = np.unique(
                    labels, return_inverse=True, return_counts=True
                )
                n_samples = float(len(dataset))
                class_weights = n_samples / np.maximum(counts.astype(np.float64), 1.0)
                sample_weights = class_weights[inverse]
                generator = None
                if seed is not None:
                    generator = torch.Generator()
                    generator.manual_seed(int(seed))
                sampler = WeightedRandomSampler(
                    weights=sample_weights.tolist(),
                    num_samples=len(dataset),
                    replacement=True,
                    generator=generator,
                )
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    drop_last=len(dataset) >= batch_size,
                    num_workers=num_workers,
                )

            sampler = SequentialSampler(dataset)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=False,
                num_workers=num_workers,
            )

else:  # pragma: no cover - fallback placeholders when torch is absent

    class MAPSMLP:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

    class MAPSTabularDataset:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


@dataclass
class MAPSTrainingConfig:
    """Hyperparameters matching ``mahmoodlab/MAPS`` Trainer & Predictor."""

    hidden_dim: int = 512
    dropout: float = 0.10
    batch_size: int = 128
    learning_rate: float = 0.001
    max_epochs: int = 500
    min_epochs: int = 250
    patience: int = 100
    scale_divisor: float = DEFAULT_SCALE_DIVISOR
    seed: int = DEFAULT_SEED
    num_workers: int = 0
    device: str = "auto"
    dtype: str = "auto"
    use_class_balanced_sampler: bool = True
    val_fraction: float = 0.2
    verbose: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MAPSTrainer:
    """Faithful trainer and inference engine for MAPS MLP phenotyping."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        config: Optional[MAPSTrainingConfig] = None,
        results_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        _require_torch()
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.config = config or MAPSTrainingConfig()
        self.results_dir = Path(results_dir) if results_dir is not None else None

        self.device = resolve_device(self.config.device)
        self.dtype = resolve_dtype(self.device, self.config.dtype)

        self.model: Optional[MAPSMLP] = None
        self.train_mean: Optional[np.ndarray] = None
        self.train_std: Optional[np.ndarray] = None
        self.best_state_dict: Optional[Dict[str, Any]] = None
        self.history: Dict[str, List[float]] = {}
        self.best_epoch: int = -1
        self.best_valid_loss: float = float("inf")

    def init_model(self) -> MAPSMLP:
        """Instantiate the 4-layer MAPS MLP on the resolved device and dtype."""
        model = MAPSMLP(
            input_dim=self.num_features,
            hidden_dim=self.config.hidden_dim,
            num_classes=self.num_classes,
            dropout=self.config.dropout,
        )
        self.model = model.to(device=self.device, dtype=self.dtype)
        return self.model

    @staticmethod
    def _split_train_valid(
        X: np.ndarray,
        y: np.ndarray,
        val_fraction: float,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split a single dataset into train and validation arrays."""
        n = len(y)
        if n < 2:
            return X, y, X, y

        val_count = max(1, int(round(n * val_fraction)))
        if val_count >= n:
            val_count = max(1, n // 5)

        _, counts = np.unique(y, return_counts=True)
        can_stratify = np.all(counts >= 2) and val_count >= len(counts)
        if can_stratify:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=val_count, random_state=seed
            )
            tr_idx, va_idx = next(splitter.split(X, y))
        else:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n)
            va_idx = perm[:val_count]
            tr_idx = perm[val_count:]
            if len(tr_idx) == 0:
                tr_idx = va_idx

        return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]

    def _safe_auc(self, y_true: np.ndarray, probs: np.ndarray) -> float:
        """Compute multi-class OVO ROC-AUC when at least two classes are present."""
        present = np.unique(y_true)
        if len(present) < 2:
            return float("nan")
        try:
            if len(present) == 2 and self.num_classes == 2:
                return float(roc_auc_score(y_true, probs[:, 1]))
            return float(
                roc_auc_score(
                    y_true,
                    probs,
                    multi_class="ovo",
                    labels=list(range(self.num_classes)),
                )
            )
        except Exception:
            return float("nan")

    @staticmethod
    def _compute_class_sample_weights(y: np.ndarray) -> "torch.Tensor":
        """Compute per-sample inverse-frequency weights matching WeightedRandomSampler."""
        _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
        n_samples = float(len(y))
        class_weights = n_samples / np.maximum(counts.astype(np.float64), 1.0)
        return torch.from_numpy(class_weights[inverse])

    def _run_tensor_epoch(
        self,
        x_dev: "torch.Tensor",
        y_dev: "torch.Tensor",
        batch_size: int,
        loss_fn: "nn.Module",
        optimizer: Optional["optim.Optimizer"] = None,
        sample_weights: Optional["torch.Tensor"] = None,
        rng_generator: Optional["torch.Generator"] = None,
        compute_auc: bool = False,
    ) -> Tuple[float, float, float]:
        assert self.model is not None
        is_train = optimizer is not None
        self.model.train(is_train)

        n_samples = int(x_dev.shape[0])
        if n_samples == 0:
            return 0.0, 0.0, float("nan")

        if is_train:
            if sample_weights is not None:
                order = torch.multinomial(
                    sample_weights,
                    num_samples=n_samples,
                    replacement=True,
                    generator=rng_generator,
                ).to(self.device)
            else:
                order = torch.randperm(n_samples, generator=rng_generator).to(self.device)
            drop_last = n_samples >= batch_size
            limit = (n_samples // batch_size) * batch_size if drop_last else n_samples
        else:
            order = None
            limit = n_samples

        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        batch_count = 0
        gt_chunks: List[torch.Tensor] = []
        pred_chunks: List[torch.Tensor] = []
        prob_chunks: List[torch.Tensor] = []

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for start in range(0, limit, batch_size):
                end = min(start + batch_size, limit)
                if order is not None:
                    idx = order[start:end]
                    features_batch = x_dev.index_select(0, idx)
                    label_batch = y_dev.index_select(0, idx)
                else:
                    features_batch = x_dev[start:end]
                    label_batch = y_dev[start:end]

                if is_train:
                    optimizer.zero_grad()

                logits, probs = self.model(features_batch)
                loss = loss_fn(logits, label_batch)

                if is_train:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.detach().to(dtype=torch.float32)
                batch_count += 1
                gt_chunks.append(label_batch)
                pred_chunks.append(torch.argmax(probs.detach(), dim=1))
                if compute_auc:
                    prob_chunks.append(probs.detach())

        if batch_count == 0:
            return 0.0, 0.0, float("nan")

        y_true_t = torch.cat(gt_chunks, dim=0)
        y_pred_t = torch.cat(pred_chunks, dim=0)
        avg_loss = float(total_loss.cpu().item()) / batch_count
        acc = float((y_true_t == y_pred_t).to(dtype=torch.float32).mean().cpu().item())
        auc = float("nan")
        if compute_auc and prob_chunks:
            y_true_np = y_true_t.cpu().numpy()
            probs_np = torch.cat(prob_chunks, dim=0).cpu().numpy()
            auc = self._safe_auc(y_true_np, probs_np)
        return avg_loss, acc, auc

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        *,
        class_names: Optional[Sequence[str]] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Fit the MAPS MLP with early stopping on validation cross-entropy loss."""
        set_global_seed(self.config.seed, self.device)

        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=np.int64)
        if X_valid is None or y_valid is None:
            X_tr, y_tr, X_va, y_va = self._split_train_valid(
                X_tr, y_tr, self.config.val_fraction, self.config.seed
            )
        else:
            X_va = np.asarray(X_valid, dtype=np.float64)
            y_va = np.asarray(y_valid, dtype=np.int64)

        self.init_model()
        assert self.model is not None

        train_ds = MAPSTabularDataset(
            X_tr,
            y_tr,
            is_train=True,
            scale_divisor=self.config.scale_divisor,
            tensor_dtype=self.dtype,
        )
        self.train_mean = train_ds.mean.copy()
        self.train_std = train_ds.std.copy()

        valid_ds = MAPSTabularDataset(
            X_va,
            y_va,
            is_train=False,
            mean=self.train_mean,
            std=self.train_std,
            scale_divisor=self.config.scale_divisor,
            tensor_dtype=self.dtype,
        )

        x_tr_dev = torch.from_numpy(train_ds.x).to(device=self.device, dtype=self.dtype)
        y_tr_dev = torch.from_numpy(train_ds.y).to(device=self.device, dtype=torch.long)
        x_va_dev = torch.from_numpy(valid_ds.x).to(device=self.device, dtype=self.dtype)
        y_va_dev = torch.from_numpy(valid_ds.y).to(device=self.device, dtype=torch.long)

        sample_weights = (
            self._compute_class_sample_weights(train_ds.y)
            if self.config.use_class_balanced_sampler
            else None
        )
        rng_gen = torch.Generator()
        rng_gen.manual_seed(int(self.config.seed))

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
        )
        loss_fn = nn.CrossEntropyLoss()

        self.history = {
            "train_loss": [],
            "valid_loss": [],
            "train_acc": [],
            "valid_acc": [],
            "train_auc": [],
            "valid_auc": [],
        }
        self.best_valid_loss = float("inf")
        self.best_epoch = -1
        self.best_state_dict = None
        counter = 0

        if self.results_dir is not None:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.time()
        compute_auc = self.results_dir is not None or self.config.verbose > 0
        for epoch in range(self.config.max_epochs):
            tr_loss, tr_acc, tr_auc = self._run_tensor_epoch(
                x_tr_dev,
                y_tr_dev,
                self.config.batch_size,
                loss_fn,
                optimizer=optimizer,
                sample_weights=sample_weights,
                rng_generator=rng_gen,
                compute_auc=compute_auc and (epoch % 10 == 0),
            )
            va_loss, va_acc, va_auc = self._run_tensor_epoch(
                x_va_dev,
                y_va_dev,
                max(self.config.batch_size, 1024),
                loss_fn,
                optimizer=None,
                compute_auc=compute_auc,
            )

            self.history["train_loss"].append(tr_loss)
            self.history["valid_loss"].append(va_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["valid_acc"].append(va_acc)
            self.history["train_auc"].append(tr_auc)
            self.history["valid_auc"].append(va_auc)

            if va_loss < self.best_valid_loss:
                self.best_valid_loss = va_loss
                self.best_epoch = epoch
                self.best_state_dict = {
                    k: copy.deepcopy(v.detach().cpu())
                    for k, v in self.model.state_dict().items()
                }
                counter = 0
                if self.results_dir is not None:
                    self.save_checkpoint(
                        self.results_dir / "best_checkpoint.pt",
                        class_names=class_names,
                        feature_names=feature_names,
                    )
            else:
                counter += 1

            if self.config.verbose > 0 and (epoch % 10 == 0 or epoch == self.config.max_epochs - 1):
                print(
                    f"[MAPS] Epoch {epoch:03d}: "
                    f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                    f"valid_loss={va_loss:.4f} valid_acc={va_acc:.4f}"
                )

            if self.results_dir is not None:
                pd.DataFrame(self.history).to_csv(self.results_dir / "training_logs.csv")

            if counter > self.config.patience and epoch >= self.config.min_epochs:
                break

        if self.best_state_dict is not None:
            loaded = {
                k: v.to(device=self.device, dtype=self.dtype)
                for k, v in self.best_state_dict.items()
            }
            self.model.load_state_dict(loaded, strict=True)

        return {
            "best_epoch": self.best_epoch,
            "best_valid_loss": self.best_valid_loss,
            "epochs_run": len(self.history["train_loss"]),
            "elapsed_sec": time.time() - t_start,
            "history": self.history,
        }

    def predict(
        self,
        X: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class indices and softmax probabilities for feature matrix ``X``."""
        _require_torch()
        if self.model is None or self.train_mean is None or self.train_std is None:
            raise RuntimeError("MAPS model must be fitted or loaded before calling predict().")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array (N, F), got shape {X_arr.shape}.")
        if X_arr.shape[1] != self.num_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.num_features}, got {X_arr.shape[1]}."
            )

        if X_arr.shape[0] == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0, self.num_classes), dtype=np.float64)

        ds = MAPSTabularDataset(
            X_arr,
            labels=None,
            is_train=False,
            mean=self.train_mean,
            std=self.train_std,
            scale_divisor=self.config.scale_divisor,
            tensor_dtype=self.dtype,
        )
        bs = max(int(batch_size or self.config.batch_size), 1)
        x_dev = torch.from_numpy(ds.x).to(device=self.device, dtype=self.dtype)

        self.model.eval()
        prob_chunks: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x_dev.shape[0], bs):
                _, probs = self.model(x_dev[start : start + bs])
                prob_chunks.append(probs.detach().cpu().numpy().astype(np.float64))

        probs_all = np.concatenate(prob_chunks, axis=0)
        labels_all = np.argmax(probs_all, axis=1).astype(np.int64)
        return labels_all, probs_all

    def save_checkpoint(
        self,
        path: Union[str, Path],
        *,
        class_names: Optional[Sequence[str]] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> Path:
        """Save checkpoint compatible with upstream ``mahmoodlab/MAPS`` format."""
        _require_torch()
        if self.model is None or self.train_mean is None or self.train_std is None:
            raise RuntimeError("Cannot save checkpoint before model is fitted or loaded.")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        state = self.best_state_dict
        if state is None:
            state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        payload: Dict[str, Any] = {
            "model_parameters": state,
            "train_data_mean": np.asarray(self.train_mean, dtype=np.float64),
            "train_data_std": np.asarray(self.train_std, dtype=np.float64),
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "hidden_dim": self.config.hidden_dim,
            "dropout": self.config.dropout,
            "scale_divisor": self.config.scale_divisor,
            "class_names": list(class_names) if class_names is not None else None,
            "feature_names": list(feature_names) if feature_names is not None else None,
            "iris_format_version": "1.0",
        }
        torch.save(payload, out_path)
        return out_path

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        *,
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 128,
        scale_divisor: Optional[float] = None,
        dropout: float = 0.10,
    ) -> Tuple["MAPSTrainer", Dict[str, Any]]:
        """Load an IRIS or upstream ``mahmoodlab/MAPS`` checkpoint."""
        _require_torch()
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MAPS checkpoint not found: {ckpt_path}")

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_params = loaded["model_parameters"]

        first_weight = model_params["fc.0.weight"]
        cls_weight = model_params["classifier.weight"]
        hidden_dim = int(loaded.get("hidden_dim", first_weight.shape[0]))
        num_features = int(loaded.get("num_features", first_weight.shape[1]))
        num_classes = int(loaded.get("num_classes", cls_weight.shape[0]))
        ckpt_dropout = float(loaded.get("dropout", dropout))
        ckpt_scale_divisor = float(
            scale_divisor
            if scale_divisor is not None
            else loaded.get("scale_divisor", DEFAULT_SCALE_DIVISOR)
        )

        config = MAPSTrainingConfig(
            hidden_dim=hidden_dim,
            dropout=ckpt_dropout,
            batch_size=batch_size,
            scale_divisor=ckpt_scale_divisor,
            device=device,
            dtype=dtype,
        )
        trainer = cls(num_features=num_features, num_classes=num_classes, config=config)
        model = trainer.init_model()
        cast_params = {
            k: v.to(device=trainer.device, dtype=trainer.dtype)
            for k, v in model_params.items()
        }
        model.load_state_dict(cast_params, strict=True)
        model.eval()

        trainer.train_mean = np.asarray(loaded["train_data_mean"], dtype=np.float64)
        trainer.train_std = np.asarray(loaded["train_data_std"], dtype=np.float64)
        trainer.best_state_dict = {
            k: v.detach().cpu() for k, v in model_params.items()
        }

        meta = {
            "class_names": loaded.get("class_names"),
            "feature_names": loaded.get("feature_names"),
            "scale_divisor": ckpt_scale_divisor,
            "hidden_dim": hidden_dim,
            "num_features": num_features,
            "num_classes": num_classes,
            "checkpoint_path": str(ckpt_path),
        }
        return trainer, meta
