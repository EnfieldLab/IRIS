"""
IRIS Phenotyper Evaluation & Benchmark Reporting
================================================
Standardized classification metrics, confusion matrices, and Markdown report
formatting shared across all phenotyping benchmarks.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    """Container for standardized phenotyping classification evaluation metrics.

    Supports both attribute access (``res.macro_f1``) and mapping-style lookup
    (``res["macro_f1"]``) for interoperability across pipelines and scripts.
    """

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    cohens_kappa: float
    per_class_df: pd.DataFrame
    confusion_matrix_raw: pd.DataFrame
    confusion_matrix_norm: pd.DataFrame
    classes: List[str] = field(default_factory=list)
    n_samples: int = 0
    log_loss: Optional[float] = None
    roc_auc_macro: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        return default

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.__dataclass_fields__

    def keys(self) -> List[str]:
        return list(self.__dataclass_fields__.keys())

    def to_dict(self, include_tables: bool = True) -> Dict[str, Any]:
        """Return evaluation metrics as a plain dictionary."""
        out: Dict[str, Any] = {
            "accuracy": float(self.accuracy),
            "balanced_accuracy": float(self.balanced_accuracy),
            "macro_f1": float(self.macro_f1),
            "weighted_f1": float(self.weighted_f1),
            "cohens_kappa": float(self.cohens_kappa),
            "classes": list(self.classes),
            "n_samples": int(self.n_samples),
            "log_loss": self.log_loss,
            "roc_auc_macro": self.roc_auc_macro,
            "metadata": dict(self.metadata),
        }
        if include_tables:
            out["per_class_df"] = self.per_class_df.copy()
            out["confusion_matrix_raw"] = self.confusion_matrix_raw.copy()
            out["confusion_matrix_norm"] = self.confusion_matrix_norm.copy()
        return out

    def to_json_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        out = self.to_dict(include_tables=False)
        out["per_class"] = {
            str(idx): {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "support": int(row["support"]),
            }
            for idx, row in self.per_class_df.iterrows()
        }
        out["confusion_matrix_raw"] = {
            str(idx): {str(col): int(val) for col, val in row.items()}
            for idx, row in self.confusion_matrix_raw.iterrows()
        }
        out["confusion_matrix_norm"] = {
            str(idx): {str(col): float(val) for col, val in row.items()}
            for idx, row in self.confusion_matrix_norm.iterrows()
        }
        return out


def _align_inputs(
    y_true: Union[pd.Series, Sequence[Any], np.ndarray],
    y_pred: Union[pd.Series, Sequence[Any], np.ndarray],
    y_prob: Optional[Union[pd.DataFrame, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[Union[pd.DataFrame, np.ndarray]]]:
    """Align true labels, predicted labels, and optional probability matrix."""
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        if not y_true.index.equals(y_pred.index):
            if (
                len(y_true) == len(y_pred)
                and y_true.index.is_unique
                and y_pred.index.is_unique
                and set(y_true.index) == set(y_pred.index)
            ):
                y_pred = y_pred.reindex(y_true.index)
                if isinstance(y_prob, pd.DataFrame) and set(y_true.index).issubset(
                    set(y_prob.index)
                ):
                    y_prob = y_prob.reindex(y_true.index)

    yt_raw = np.asarray(y_true, dtype=object).ravel()
    yp_raw = np.asarray(y_pred, dtype=object).ravel()

    if len(yt_raw) != len(yp_raw):
        raise ValueError(
            f"y_true and y_pred length mismatch: {len(yt_raw)} != {len(yp_raw)}."
        )
    if len(yt_raw) == 0:
        raise ValueError("Cannot evaluate empty predictions.")

    valid_mask = ~pd.isna(yt_raw)
    if not np.any(valid_mask):
        raise ValueError("y_true contains only missing/NaN values.")

    yt_clean = yt_raw[valid_mask]
    yp_clean = yp_raw[valid_mask]
    yp_clean = np.where(pd.isna(yp_clean), "Unassigned", yp_clean)

    yt_str = np.asarray([str(v) for v in yt_clean], dtype=object)
    yp_str = np.asarray([str(v) for v in yp_clean], dtype=object)

    prob_aligned: Optional[Union[pd.DataFrame, np.ndarray]] = None
    if y_prob is not None:
        if isinstance(y_prob, pd.DataFrame):
            if len(y_prob) != len(yt_raw):
                raise ValueError(
                    f"y_prob length mismatch: {len(y_prob)} != {len(yt_raw)}."
                )
            prob_aligned = y_prob.iloc[valid_mask].copy()
        else:
            prob_arr = np.asarray(y_prob, dtype=float)
            if len(prob_arr) != len(yt_raw):
                raise ValueError(
                    f"y_prob length mismatch: {len(prob_arr)} != {len(yt_raw)}."
                )
            prob_aligned = prob_arr[valid_mask]

    return yt_str, yp_str, prob_aligned


def _compute_prob_metrics(
    yt_str: np.ndarray,
    y_prob: Union[pd.DataFrame, np.ndarray],
    true_classes: List[str],
) -> tuple[Optional[float], Optional[float]]:
    """Compute optional log_loss and macro ROC-AUC when class probabilities align."""
    if len(true_classes) < 2:
        return None, None

    try:
        if isinstance(y_prob, pd.DataFrame):
            missing = [c for c in true_classes if c not in y_prob.columns]
            if missing:
                return None, None
            probs = y_prob[true_classes].to_numpy(dtype=float)
        else:
            probs = np.asarray(y_prob, dtype=float)
            if probs.ndim != 2 or probs.shape[1] != len(true_classes):
                return None, None

        if not np.all(np.isfinite(probs)):
            return None, None

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
        probs_norm = np.clip(probs / row_sums, 1e-15, 1.0)
        probs_norm = probs_norm / probs_norm.sum(axis=1, keepdims=True)

        ll = float(log_loss(yt_str, probs_norm, labels=true_classes))
        if len(true_classes) == 2:
            auc = float(roc_auc_score((yt_str == true_classes[1]).astype(int), probs_norm[:, 1]))
        else:
            auc = float(
                roc_auc_score(
                    yt_str,
                    probs_norm,
                    labels=true_classes,
                    multi_class="ovr",
                    average="macro",
                )
            )
        return ll, auc
    except Exception:
        return None, None


def evaluate_predictions(
    y_true: Union[pd.Series, Sequence[Any], np.ndarray],
    y_pred: Union[pd.Series, Sequence[Any], np.ndarray],
    y_prob: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    *,
    labels: Optional[Sequence[str]] = None,
    zero_division: int = 0,
) -> EvaluationResult:
    """Compute standardized single-cell phenotyping classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth cell-type labels.
    y_pred:
        Predicted cell-type labels.
    y_prob:
        Optional class probability matrix (rows = cells, columns = class names).
    labels:
        Optional explicit ordered list of class labels for per-class reporting and
        confusion matrices. Defaults to the sorted union of observed true and
        predicted classes.
    zero_division:
        Value returned by precision/recall/F1 when a class has zero support or
        zero predicted positives.

    Returns
    -------
    EvaluationResult
        Dataclass (also dict-subscriptable) with overall metrics, ``per_class_df``,
        and raw/normalized confusion matrices.
    """
    yt_str, yp_str, prob_aligned = _align_inputs(y_true, y_pred, y_prob=y_prob)

    if labels is not None:
        class_labels = [str(c) for c in labels]
    else:
        class_labels = sorted(set(yt_str.tolist()) | set(yp_str.tolist()))

    acc = float(accuracy_score(yt_str, yp_str))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        bal_acc = float(balanced_accuracy_score(yt_str, yp_str))

    macro_f1 = float(
        f1_score(yt_str, yp_str, labels=class_labels, average="macro", zero_division=zero_division)
    )
    weighted_f1 = float(
        f1_score(
            yt_str, yp_str, labels=class_labels, average="weighted", zero_division=zero_division
        )
    )

    if len(set(yt_str.tolist()) | set(yp_str.tolist())) <= 1:
        kappa = 1.0 if acc == 1.0 else 0.0
    else:
        kappa_raw = float(cohen_kappa_score(yt_str, yp_str, labels=class_labels))
        kappa = 0.0 if np.isnan(kappa_raw) else kappa_raw

    prec, rec, f1_per_cls, supp = precision_recall_fscore_support(
        yt_str,
        yp_str,
        labels=class_labels,
        average=None,
        zero_division=zero_division,
    )
    per_class_df = pd.DataFrame(
        {
            "precision": np.asarray(prec, dtype=float),
            "recall": np.asarray(rec, dtype=float),
            "f1": np.asarray(f1_per_cls, dtype=float),
            "support": np.asarray(supp, dtype=int),
        },
        index=pd.Index(class_labels, name="cell_type"),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cm_raw_arr = confusion_matrix(yt_str, yp_str, labels=class_labels)
    cm_raw_df = pd.DataFrame(
        cm_raw_arr,
        index=pd.Index(class_labels, name="true"),
        columns=pd.Index(class_labels, name="predicted"),
    )

    row_sums = cm_raw_arr.sum(axis=1, keepdims=True).astype(float)
    cm_norm_arr = np.divide(
        cm_raw_arr.astype(float),
        row_sums,
        out=np.zeros_like(cm_raw_arr, dtype=float),
        where=row_sums > 0.0,
    )
    cm_norm_df = pd.DataFrame(
        cm_norm_arr,
        index=pd.Index(class_labels, name="true"),
        columns=pd.Index(class_labels, name="predicted"),
    )

    ll: Optional[float] = None
    auc_macro: Optional[float] = None
    if prob_aligned is not None:
        true_classes = sorted(set(yt_str.tolist()))
        ll, auc_macro = _compute_prob_metrics(yt_str, prob_aligned, true_classes)

    return EvaluationResult(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        cohens_kappa=kappa,
        per_class_df=per_class_df,
        confusion_matrix_raw=cm_raw_df,
        confusion_matrix_norm=cm_norm_df,
        classes=class_labels,
        n_samples=int(len(yt_str)),
        log_loss=ll,
        roc_auc_macro=auc_macro,
    )


def _format_df_markdown(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> List[str]:
    """Render a DataFrame into a GitHub-flavored Markdown table without extra deps."""
    index_name = df.index.name or "index"
    headers = [str(index_name)] + [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in df.columns]) + " |",
    ]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for val in row:
            if isinstance(val, (int, np.integer)):
                cells.append(str(int(val)))
            elif isinstance(val, (float, np.floating)):
                cells.append("nan" if np.isnan(val) else float_fmt.format(float(val)))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _extract_metric(obj: Union[EvaluationResult, Mapping[str, Any]], key: str) -> Any:
    if isinstance(obj, EvaluationResult):
        return getattr(obj, key, None)
    return obj.get(key)


def format_benchmark_report(
    evaluation: Union[EvaluationResult, Mapping[str, Any]],
    method_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    paper_metrics: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    include_per_class: bool = True,
    include_confusion_matrix: bool = True,
) -> str:
    """Format an evaluation result or method comparison mapping into Markdown.

    Parameters
    ----------
    evaluation:
        Either a single ``EvaluationResult`` (or metric dict) OR a mapping from
        method name to ``EvaluationResult`` (for multi-method comparisons).
    method_name:
        Method display name when formatting a single result.
    dataset_name:
        Optional benchmark dataset identifier.
    paper_metrics:
        Optional published reference metrics (e.g., ``{"accuracy": 0.962, "macro_f1": 0.941}``)
        shown alongside reproduced values for single-method reports.
    metadata:
        Optional run metadata displayed in a summary list.
    include_per_class:
        Whether to include the per-class precision/recall/F1/support table.
    include_confusion_matrix:
        Whether to include the raw confusion matrix table.
    """
    is_single = isinstance(evaluation, EvaluationResult) or (
        isinstance(evaluation, Mapping)
        and any(k in evaluation for k in ("accuracy", "macro_f1", "per_class_df"))
    )

    lines: List[str] = []
    if is_single:
        title_parts = [p for p in (method_name, dataset_name) if p]
        title = " - ".join(title_parts) if title_parts else "Single Method"
        lines.append(f"# Phenotyping Benchmark Report: {title}")
        lines.append("")

        if dataset_name or method_name or metadata:
            if method_name:
                lines.append(f"- **Method**: {method_name}")
            if dataset_name:
                lines.append(f"- **Dataset**: {dataset_name}")
            n_samples = _extract_metric(evaluation, "n_samples")
            if n_samples is not None:
                lines.append(f"- **Samples Evaluated**: {n_samples}")
            if metadata:
                for k, v in metadata.items():
                    lines.append(f"- **{k}**: {v}")
            lines.append("")

        lines.append("## Overall Metrics")
        lines.append("")
        metric_specs: Iterable[tuple[str, str]] = (
            ("accuracy", "Accuracy"),
            ("balanced_accuracy", "Balanced Accuracy"),
            ("macro_f1", "Macro F1"),
            ("weighted_f1", "Weighted F1"),
            ("cohens_kappa", "Cohen's Kappa"),
            ("log_loss", "Log Loss"),
            ("roc_auc_macro", "ROC-AUC (Macro OvR)"),
        )
        if paper_metrics:
            lines.append("| Metric | IRIS Reproduced | Published Paper | Delta |")
            lines.append("| --- | ---: | ---: | ---: |")
        else:
            lines.append("| Metric | Value |")
            lines.append("| --- | ---: |")

        for key, label in metric_specs:
            val = _extract_metric(evaluation, key)
            if val is None:
                continue
            val_str = f"{float(val):.4f}"
            if paper_metrics:
                ref = paper_metrics.get(key)
                if ref is not None and isinstance(ref, (int, float, np.floating)):
                    delta = float(val) - float(ref)
                    lines.append(f"| {label} | {val_str} | {float(ref):.4f} | {delta:+.4f} |")
                else:
                    lines.append(f"| {label} | {val_str} | - | - |")
            else:
                lines.append(f"| {label} | {val_str} |")
        lines.append("")

        per_class_df = _extract_metric(evaluation, "per_class_df")
        if include_per_class and isinstance(per_class_df, pd.DataFrame) and not per_class_df.empty:
            lines.append("## Per-Class Performance")
            lines.append("")
            lines.extend(_format_df_markdown(per_class_df))
            lines.append("")

        cm_raw = _extract_metric(evaluation, "confusion_matrix_raw")
        if (
            include_confusion_matrix
            and isinstance(cm_raw, pd.DataFrame)
            and not cm_raw.empty
        ):
            lines.append("## Confusion Matrix (Raw Counts)")
            lines.append("")
            lines.extend(_format_df_markdown(cm_raw))
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    # Multi-method comparison report
    title_suffix = f": {dataset_name}" if dataset_name else ""
    lines.append(f"# Phenotyping Benchmark Comparison{title_suffix}")
    lines.append("")
    if metadata:
        for k, v in metadata.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("## Summary Comparison")
    lines.append("")
    lines.append(
        "| Method | N | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Cohen's Kappa |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    for m_name, m_eval in evaluation.items():
        n_val = _extract_metric(m_eval, "n_samples")
        acc = _extract_metric(m_eval, "accuracy")
        b_acc = _extract_metric(m_eval, "balanced_accuracy")
        m_f1 = _extract_metric(m_eval, "macro_f1")
        w_f1 = _extract_metric(m_eval, "weighted_f1")
        kap = _extract_metric(m_eval, "cohens_kappa")

        def _fmt(v: Any) -> str:
            return "-" if v is None else f"{float(v):.4f}"

        lines.append(
            f"| {m_name} | {n_val if n_val is not None else '-'} | "
            f"{_fmt(acc)} | {_fmt(b_acc)} | {_fmt(m_f1)} | {_fmt(w_f1)} | {_fmt(kap)} |"
        )
    lines.append("")

    if include_per_class:
        for m_name, m_eval in evaluation.items():
            per_class_df = _extract_metric(m_eval, "per_class_df")
            if isinstance(per_class_df, pd.DataFrame) and not per_class_df.empty:
                lines.append(f"## Per-Class Performance: {m_name}")
                lines.append("")
                lines.extend(_format_df_markdown(per_class_df))
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"
