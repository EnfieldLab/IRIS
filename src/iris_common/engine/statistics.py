"""
IRIS Common Statistical Engine
==============================
Sample-aware hypothesis testing, multiple-testing correction, effect sizes,
and publication-grade significance formatting.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from iris_common.schema import COL_SAMPLE, resolve_marker_column

VALID_TESTS = (
    "mann_whitney",
    "welch_t",
    "student_t",
    "wilcoxon_paired",
    "kruskal",
    "anova_oneway",
)

VALID_CORRECTIONS = ("fdr_bh", "bonferroni", "holm", "none", "tukey")


@dataclass
class StatTestResult:
    """Result for a single pairwise group comparison."""

    group1: str
    group2: str
    test_name: str
    statistic: float
    p_value: float
    q_value: float
    n1: int
    n2: int
    aggregation_level: str = "sample"
    correction_method: str = "fdr_bh"
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    mean1: Optional[float] = None
    mean2: Optional[float] = None
    median1: Optional[float] = None
    median2: Optional[float] = None
    significant: bool = False
    significance_label: str = "ns"
    omnibus_test: Optional[str] = None
    omnibus_statistic: Optional[float] = None
    omnibus_p_value: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this comparison result to a plain dictionary."""
        return asdict(self)


class ComparisonResultList(list):
    """List of ``StatTestResult`` with attached omnibus metadata and DataFrame export."""

    def __init__(
        self,
        items: Sequence[StatTestResult] = (),
        *,
        aggregation_level: str = "sample",
        correction: str = "fdr_bh",
        omnibus_test: Optional[str] = None,
        omnibus_statistic: Optional[float] = None,
        omnibus_p_value: Optional[float] = None,
        sample_summary_df: Optional[pd.DataFrame] = None,
    ) -> None:
        super().__init__(items)
        self.aggregation_level = aggregation_level
        self.correction = correction
        self.omnibus_test = omnibus_test
        self.omnibus_statistic = omnibus_statistic
        self.omnibus_p_value = omnibus_p_value
        self.sample_summary_df = sample_summary_df

    def to_dataframe(self) -> pd.DataFrame:
        """Convert comparison results to a tidy DataFrame."""
        return results_to_dataframe(self)


def results_to_dataframe(results: Sequence[StatTestResult]) -> pd.DataFrame:
    """Convert a sequence of ``StatTestResult`` objects into a pandas DataFrame."""
    if not results:
        return pd.DataFrame(
            columns=[f for f in StatTestResult.__dataclass_fields__]
        )
    return pd.DataFrame([r.to_dict() for r in results])


def adjust_p_values(
    p_values: Sequence[float] | np.ndarray,
    method: str = "fdr_bh",
) -> np.ndarray:
    """Adjust p-values for multiple hypothesis testing using pure NumPy.

    Supported methods: ``'fdr_bh'``, ``'bonferroni'``, ``'holm'``, ``'none'``.
    NaN values are preserved at their original indices.
    """
    arr = np.asarray(p_values, dtype=float)
    if arr.size == 0:
        return arr.copy()

    norm_method = method.lower().strip()
    if norm_method == "none":
        return np.where(np.isfinite(arr), np.clip(arr, 0.0, 1.0), np.nan)

    out = np.full(arr.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(arr)
    valid_p = np.clip(arr[valid_mask], 0.0, 1.0)
    m = valid_p.size
    if m == 0:
        return out

    if norm_method == "bonferroni":
        adj = np.minimum(1.0, valid_p * m)
    elif norm_method == "holm":
        order = np.argsort(valid_p, kind="mergesort")
        sorted_p = valid_p[order]
        multipliers = np.arange(m, 0, -1, dtype=float)
        scaled = sorted_p * multipliers
        monotonic = np.minimum(1.0, np.maximum.accumulate(scaled))
        adj = np.empty(m, dtype=float)
        adj[order] = monotonic
    elif norm_method in ("fdr_bh", "bh", "fdr"):
        order = np.argsort(valid_p, kind="mergesort")
        sorted_p = valid_p[order]
        ranks = np.arange(1, m + 1, dtype=float)
        scaled = (sorted_p * m) / ranks
        monotonic = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
        adj = np.empty(m, dtype=float)
        adj[order] = monotonic
    else:
        raise ValueError(
            f"Unsupported multiple-testing correction {method!r}. "
            f"Expected one of {VALID_CORRECTIONS}."
        )

    out[valid_mask] = adj
    return out


def format_significance(p_or_q: Optional[float], mode: str = "stars") -> str:
    """Format a p-value or q-value into significance stars or text.

    Parameters
    ----------
    p_or_q:
        P-value or FDR-adjusted q-value.
    mode:
        One of ``'stars'``, ``'p_value'``, ``'q_value'``, or ``'stars_and_p'``.
    """
    m = mode.lower().strip()
    if p_or_q is None or not np.isfinite(p_or_q):
        return "ns" if m == "stars" else "N/A"

    val = float(p_or_q)
    if val < 1e-4:
        stars = "****"
    elif val < 1e-3:
        stars = "***"
    elif val < 1e-2:
        stars = "**"
    elif val < 0.05:
        stars = "*"
    else:
        stars = "ns"

    if m == "stars":
        return stars
    if m in ("p_value", "p", "value"):
        return f"p = {val:.2e}" if val < 1e-3 else f"p = {val:.3f}"
    if m in ("q_value", "q"):
        return f"q = {val:.2e}" if val < 1e-3 else f"q = {val:.3f}"
    if m in ("stars_and_p", "combined"):
        p_txt = f"{val:.2e}" if val < 1e-3 else f"{val:.3f}"
        return f"{stars} (p = {p_txt})"
    raise ValueError(f"Unsupported significance format mode: {mode!r}")


def _normalize_test_name(test: str) -> str:
    mapping = {
        "mann_whitney": "mann_whitney",
        "mann-whitney": "mann_whitney",
        "mannwhitney": "mann_whitney",
        "mwu": "mann_whitney",
        "welch_t": "welch_t",
        "welch": "welch_t",
        "student_t": "student_t",
        "t_test": "student_t",
        "ttest": "student_t",
        "wilcoxon_paired": "wilcoxon_paired",
        "wilcoxon": "wilcoxon_paired",
        "kruskal": "kruskal",
        "kruskal_wallis": "kruskal",
        "anova_oneway": "anova_oneway",
        "anova": "anova_oneway",
        "oneway_anova": "anova_oneway",
    }
    key = test.lower().strip()
    if key not in mapping:
        raise ValueError(f"Unsupported statistical test {test!r}. Expected one of {VALID_TESTS}.")
    return mapping[key]


def _cohens_d(v1: np.ndarray, v2: np.ndarray, *, welch: bool = False) -> Optional[float]:
    if v1.size < 2 or v2.size < 2:
        return None
    m1, m2 = float(np.mean(v1)), float(np.mean(v2))
    s1 = float(np.var(v1, ddof=1))
    s2 = float(np.var(v2, ddof=1))
    if welch:
        denom = float(np.sqrt(0.5 * (s1 + s2)))
    else:
        n1, n2 = v1.size, v2.size
        pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / max(n1 + n2 - 2, 1)
        denom = float(np.sqrt(pooled))
    if denom <= 0.0:
        return 0.0 if np.isclose(m1, m2) else float(np.sign(m1 - m2) * np.inf)
    return (m1 - m2) / denom


def _align_paired_vectors(
    g1_df: pd.DataFrame,
    g2_df: pd.DataFrame,
    value_col: str,
    group1: str,
    group2: str,
    pair_key: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Align observations for paired tests via pair_key or positional length."""
    if pair_key is not None and pair_key in g1_df.columns and pair_key in g2_df.columns:
        s1 = g1_df.groupby(pair_key, observed=True)[value_col].mean()
        s2 = g2_df.groupby(pair_key, observed=True)[value_col].mean()
        common = s1.index.intersection(s2.index)
        if len(common) == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        return s1.loc[common].to_numpy(dtype=float), s2.loc[common].to_numpy(dtype=float)

    v1 = g1_df[value_col].to_numpy(dtype=float)
    v2 = g2_df[value_col].to_numpy(dtype=float)
    if v1.size != v2.size:
        raise ValueError(
            f"Paired test between {group1!r} (n={v1.size}) and {group2!r} (n={v2.size}) "
            "requires equal sample sizes or a valid 'paired_by' column."
        )
    return v1, v2


def _run_pairwise_stat(
    g1_df: pd.DataFrame,
    g2_df: pd.DataFrame,
    value_col: str,
    group1: str,
    group2: str,
    pair_test: str,
    pair_key: Optional[str],
) -> Tuple[float, float, Optional[float], Optional[str], int, int, str]:
    """Execute a single pairwise test returning (stat, p, eff, eff_name, n1, n2, notes)."""
    if pair_test == "wilcoxon_paired":
        v1, v2 = _align_paired_vectors(g1_df, g2_df, value_col, group1, group2, pair_key)
        n1, n2 = int(v1.size), int(v2.size)
        if n1 == 0:
            return np.nan, 1.0, None, None, 0, 0, "No matched pairs available"
        diffs = v1 - v2
        if np.allclose(diffs, 0.0):
            return 0.0, 1.0, 0.0, "matched_rank_biserial", n1, n2, "All paired differences are zero"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.wilcoxon(v1, v2, alternative="two-sided", zero_method="wilcox")
        pos = float(np.sum(diffs > 0))
        neg = float(np.sum(diffs < 0))
        total = pos + neg
        eff = (pos - neg) / total if total > 0 else 0.0
        return float(res.statistic), float(res.pvalue), eff, "matched_rank_biserial", n1, n2, ""

    v1 = g1_df[value_col].to_numpy(dtype=float)
    v2 = g2_df[value_col].to_numpy(dtype=float)
    n1, n2 = int(v1.size), int(v2.size)

    if n1 == 0 or n2 == 0:
        return np.nan, 1.0, None, None, n1, n2, "Insufficient sample size"

    if pair_test == "mann_whitney":
        res = stats.mannwhitneyu(v1, v2, alternative="two-sided")
        eff = (2.0 * float(res.statistic)) / (n1 * n2) - 1.0
        return float(res.statistic), float(res.pvalue), eff, "rank_biserial", n1, n2, ""

    if pair_test in ("welch_t", "student_t"):
        if n1 < 2 or n2 < 2:
            return np.nan, 1.0, None, None, n1, n2, "t-test requires >= 2 observations per group"
        equal_var = pair_test == "student_t"
        if np.var(v1) == 0.0 and np.var(v2) == 0.0:
            if np.isclose(np.mean(v1), np.mean(v2)):
                return 0.0, 1.0, 0.0, "cohens_d", n1, n2, "Zero variance in both groups"
            return (
                float(np.sign(np.mean(v1) - np.mean(v2)) * np.inf),
                0.0,
                float(np.sign(np.mean(v1) - np.mean(v2)) * np.inf),
                "cohens_d",
                n1,
                n2,
                "Zero variance in both groups with distinct means",
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.ttest_ind(v1, v2, equal_var=equal_var)
        eff = _cohens_d(v1, v2, welch=not equal_var)
        return float(res.statistic), float(res.pvalue), eff, "cohens_d", n1, n2, ""

    raise ValueError(f"Unsupported pairwise test: {pair_test!r}")


def run_group_comparison(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    sample_col: str = COL_SAMPLE,
    aggregate_by_sample: bool = True,
    test: str = "mann_whitney",
    correction: str = "fdr_bh",
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    paired_by: Optional[str] = None,
    *,
    aggregation_level: Optional[Literal["sample", "cell"]] = None,
    sample_agg_func: Literal["mean", "median"] = "mean",
    alpha: float = 0.05,
) -> ComparisonResultList:
    """Compare a numeric readout across groups with optional sample-level aggregation.

    Defaults to aggregating cell-level rows to sample-level summaries per
    ``[group_col, sample_col]`` so biological sample N (not cell count) drives
    inference and avoids pseudoreplication.
    """
    resolved_val_col = value_col if value_col in df.columns else resolve_marker_column(df, value_col)
    if resolved_val_col is None:
        raise KeyError(f"Value column {value_col!r} not found in DataFrame.")
    if group_col not in df.columns:
        raise KeyError(f"Group column {group_col!r} not found in DataFrame.")

    norm_test = _normalize_test_name(test)
    norm_corr = correction.lower().strip()
    if norm_corr not in VALID_CORRECTIONS:
        raise ValueError(
            f"Unsupported correction {correction!r}. Expected one of {VALID_CORRECTIONS}."
        )
    if norm_corr == "tukey" and norm_test != "anova_oneway":
        raise ValueError("correction='tukey' is only supported with test='anova_oneway'.")

    if aggregation_level is not None:
        if aggregation_level not in ("sample", "cell"):
            raise ValueError(
                f"aggregation_level must be 'sample' or 'cell', got {aggregation_level!r}."
            )
        use_sample_agg = aggregation_level == "sample"
        if use_sample_agg and sample_col not in df.columns:
            raise KeyError(
                f"Sample column {sample_col!r} not found for aggregation_level='sample'."
            )
        effective_level = aggregation_level
    else:
        use_sample_agg = bool(aggregate_by_sample and sample_col in df.columns)
        effective_level = "sample" if use_sample_agg else "cell"

    keep_cols = [group_col, resolved_val_col]
    if use_sample_agg and sample_col in df.columns and sample_col not in keep_cols:
        keep_cols.append(sample_col)
    if paired_by and paired_by in df.columns and paired_by not in keep_cols:
        keep_cols.append(paired_by)
    elif sample_col in df.columns and sample_col not in keep_cols:
        keep_cols.append(sample_col)

    work = df[keep_cols].copy()
    work[resolved_val_col] = pd.to_numeric(work[resolved_val_col], errors="coerce")
    work = work.dropna(subset=[group_col, resolved_val_col])
    work = work[np.isfinite(work[resolved_val_col].to_numpy(dtype=float))]
    if work.empty:
        return ComparisonResultList([], aggregation_level=effective_level, correction=norm_corr)

    # Preserve categorical group order when present; otherwise sort alphabetically
    if isinstance(df[group_col].dtype, pd.CategoricalDtype):
        present_set = {str(v) for v in work[group_col].unique()}
        groups = [str(c) for c in df[group_col].cat.categories if str(c) in present_set]
    else:
        groups = sorted({str(v) for v in work[group_col].unique()})
    work["_group_str"] = work[group_col].astype(str)

    if use_sample_agg:
        agg_cols = ["_group_str", sample_col]
        if paired_by and paired_by in work.columns and paired_by not in (group_col, sample_col):
            agg_cols.append(paired_by)
        grouped = work.groupby(agg_cols, observed=True, as_index=False)[resolved_val_col]
        work_eval = grouped.median() if sample_agg_func == "median" else grouped.mean()
    else:
        work_eval = work

    group_frames: Dict[str, pd.DataFrame] = {
        g: work_eval[work_eval["_group_str"] == g].copy() for g in groups
    }

    if pairs is None:
        pair_list: List[Tuple[str, str]] = list(itertools.combinations(groups, 2))
    else:
        pair_list = [(str(g1), str(g2)) for g1, g2 in pairs]

    if not pair_list:
        return ComparisonResultList(
            [],
            aggregation_level=effective_level,
            correction=norm_corr,
            sample_summary_df=work_eval if use_sample_agg else None,
        )

    # Omnibus step for multi-group tests
    omnibus_test: Optional[str] = None
    omnibus_stat: Optional[float] = None
    omnibus_p: Optional[float] = None
    nonempty_arrays = [
        group_frames[g][resolved_val_col].to_numpy(dtype=float)
        for g in groups
        if g in group_frames and len(group_frames[g]) > 0
    ]

    if norm_test == "kruskal" and len(nonempty_arrays) >= 2:
        omnibus_test = "kruskal"
        try:
            k_res = stats.kruskal(*nonempty_arrays)
            omnibus_stat, omnibus_p = float(k_res.statistic), float(k_res.pvalue)
        except ValueError:
            omnibus_stat, omnibus_p = 0.0, 1.0
    elif norm_test == "anova_oneway" and len(nonempty_arrays) >= 2:
        omnibus_test = "anova_oneway"
        try:
            f_res = stats.f_oneway(*nonempty_arrays)
            omnibus_stat = float(f_res.statistic) if np.isfinite(f_res.statistic) else 0.0
            omnibus_p = float(f_res.pvalue) if np.isfinite(f_res.pvalue) else 1.0
        except ValueError:
            omnibus_stat, omnibus_p = 0.0, 1.0

    # Handle Tukey HSD post-hoc for one-way ANOVA when requested
    if norm_test == "anova_oneway" and norm_corr == "tukey":
        tukey_res = stats.tukey_hsd(*[group_frames[g][resolved_val_col].to_numpy(dtype=float) for g in groups])
        group_idx = {g: idx for idx, g in enumerate(groups)}
        results: List[StatTestResult] = []
        for g1, g2 in pair_list:
            df1 = group_frames.get(g1, work_eval.iloc[0:0])
            df2 = group_frames.get(g2, work_eval.iloc[0:0])
            v1 = df1[resolved_val_col].to_numpy(dtype=float)
            v2 = df2[resolved_val_col].to_numpy(dtype=float)
            i, j = group_idx[g1], group_idx[g2]
            stat_val = float(tukey_res.statistic[i, j])
            p_val = float(tukey_res.pvalue[i, j])
            q_val = p_val
            eff = _cohens_d(v1, v2, welch=False)
            results.append(
                StatTestResult(
                    group1=g1,
                    group2=g2,
                    test_name="anova_oneway_tukey",
                    statistic=stat_val,
                    p_value=p_val,
                    q_value=q_val,
                    n1=int(v1.size),
                    n2=int(v2.size),
                    aggregation_level=effective_level,
                    correction_method="tukey",
                    effect_size=eff,
                    effect_size_name="cohens_d" if eff is not None else None,
                    mean1=float(np.mean(v1)) if v1.size else None,
                    mean2=float(np.mean(v2)) if v2.size else None,
                    median1=float(np.median(v1)) if v1.size else None,
                    median2=float(np.median(v2)) if v2.size else None,
                    significant=bool(np.isfinite(q_val) and q_val < alpha),
                    significance_label=format_significance(q_val, mode="stars"),
                    omnibus_test=omnibus_test,
                    omnibus_statistic=omnibus_stat,
                    omnibus_p_value=omnibus_p,
                )
            )
        return ComparisonResultList(
            results,
            aggregation_level=effective_level,
            correction="tukey",
            omnibus_test=omnibus_test,
            omnibus_statistic=omnibus_stat,
            omnibus_p_value=omnibus_p,
            sample_summary_df=work_eval if use_sample_agg else None,
        )

    # Determine pairwise test used (direct or post-hoc)
    if norm_test == "kruskal":
        pair_test = "mann_whitney"
        reported_name = "kruskal_mann_whitney"
    elif norm_test == "anova_oneway":
        pair_test = "student_t"
        reported_name = "anova_oneway_pairwise_t"
    else:
        pair_test = norm_test
        reported_name = norm_test

    pair_key = paired_by or (sample_col if sample_col in work_eval.columns else None)

    raw_records: List[Dict[str, Any]] = []
    raw_p_values: List[float] = []
    for g1, g2 in pair_list:
        df1 = group_frames.get(g1, work_eval.iloc[0:0])
        df2 = group_frames.get(g2, work_eval.iloc[0:0])
        v1 = df1[resolved_val_col].to_numpy(dtype=float)
        v2 = df2[resolved_val_col].to_numpy(dtype=float)

        stat_val, p_val, eff, eff_name, n1, n2, notes = _run_pairwise_stat(
            df1, df2, resolved_val_col, g1, g2, pair_test, pair_key
        )
        raw_p_values.append(p_val)
        raw_records.append(
            {
                "group1": g1,
                "group2": g2,
                "statistic": stat_val,
                "p_value": p_val,
                "n1": n1,
                "n2": n2,
                "effect_size": eff,
                "effect_size_name": eff_name,
                "mean1": float(np.mean(v1)) if v1.size else None,
                "mean2": float(np.mean(v2)) if v2.size else None,
                "median1": float(np.median(v1)) if v1.size else None,
                "median2": float(np.median(v2)) if v2.size else None,
                "notes": notes,
            }
        )

    q_values = adjust_p_values(raw_p_values, method=norm_corr)
    results = []
    for rec, q_val in zip(raw_records, q_values):
        q_float = float(q_val) if np.isfinite(q_val) else 1.0
        results.append(
            StatTestResult(
                group1=rec["group1"],
                group2=rec["group2"],
                test_name=reported_name,
                statistic=rec["statistic"],
                p_value=rec["p_value"],
                q_value=q_float,
                n1=rec["n1"],
                n2=rec["n2"],
                aggregation_level=effective_level,
                correction_method=norm_corr,
                effect_size=rec["effect_size"],
                effect_size_name=rec["effect_size_name"],
                mean1=rec["mean1"],
                mean2=rec["mean2"],
                median1=rec["median1"],
                median2=rec["median2"],
                significant=bool(np.isfinite(q_float) and q_float < alpha),
                significance_label=format_significance(q_float, mode="stars"),
                omnibus_test=omnibus_test,
                omnibus_statistic=omnibus_stat,
                omnibus_p_value=omnibus_p,
                notes=rec["notes"],
            )
        )

    return ComparisonResultList(
        results,
        aggregation_level=effective_level,
        correction=norm_corr,
        omnibus_test=omnibus_test,
        omnibus_statistic=omnibus_stat,
        omnibus_p_value=omnibus_p,
        sample_summary_df=work_eval if use_sample_agg else None,
    )


__all__ = [
    "StatTestResult",
    "ComparisonResultList",
    "VALID_TESTS",
    "VALID_CORRECTIONS",
    "adjust_p_values",
    "format_significance",
    "results_to_dataframe",
    "run_group_comparison",
]
