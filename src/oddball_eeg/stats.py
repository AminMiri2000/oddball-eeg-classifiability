"""Statistical helpers shared by the subject-level analyses.

These functions were previously re-defined in several analysis notebooks
(17, 17b, 18, 19, 21, 21b). They are kept behaviourally identical to those
definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

__all__ = [
    "benjamini_hochberg",
    "safe_corr",
    "strength_word",
    "zscore_within_group",
]


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Returns q-values in the original order. NaN p-values remain NaN and are
    excluded from the number of tests.
    """
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q

    p_valid = p[valid]
    m = len(p_valid)

    order = np.argsort(p_valid)
    ranked_p = p_valid[order]

    raw_q = ranked_p * m / np.arange(1, m + 1)
    monotonic_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    monotonic_q = np.minimum(monotonic_q, 1.0)

    q_valid = np.empty_like(monotonic_q)
    q_valid[order] = monotonic_q
    q[valid] = q_valid
    return q


def safe_corr(x, y, method: str = "pearson") -> tuple[float, float, int]:
    """Correlation on finite pairs; NaN if fewer than 3 pairs or constant input.

    Returns ``(r, p, n)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan, int(len(x))

    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return np.nan, np.nan, int(len(x))

    if method == "pearson":
        r, p = pearsonr(x, y)
    elif method == "spearman":
        r, p = spearmanr(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(r), float(p), int(len(x))


def zscore_within_group(s: pd.Series) -> pd.Series:
    """Z-score a series (population SD, ``ddof=0``); all-NaN if the SD is ~0.

    Typically used as ``df.groupby("dataset")[col].transform(zscore_within_group)``.
    """
    x = s.astype(float)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)

    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=s.index)

    return (x - mu) / sd


def strength_word(r_abs: float) -> str:
    """Verbal label for an absolute correlation coefficient."""
    if not np.isfinite(r_abs):
        return "unclear"
    if r_abs >= 0.7:
        return "strong"
    if r_abs >= 0.5:
        return "moderate-to-strong"
    if r_abs >= 0.3:
        return "moderate"
    if r_abs >= 0.15:
        return "weak-to-moderate"
    return "weak"
