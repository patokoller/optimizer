"""
app/ml/scoring.py
─────────────────────────────────────────────────────────────────────────────
Composite score formula — Equation (2), Cohen et al. (2025):

  CombinedScore(i,t) = w × MLScore(i,t) + (1-w) × LLMScore(i,t)

LOCKED optimal ML weights — Table 1, Cohen et al. (2025). DO NOT MODIFY.
"""
import numpy as np
from typing import Optional

# ── Locked ML weights ─────────────────────────────────────────────────────
# Source: Table 1, Cohen et al., Entropy 2025, 27, 550
OPTIMAL_WEIGHTS: dict[tuple[str, str], dict[str, float]] = {
    ("technical",   "monthly"):   {"ml": 1.00, "llm": 0.00},
    ("fundamental", "monthly"):   {"ml": 0.15, "llm": 0.85},
    ("entropy",     "monthly"):   {"ml": 0.70, "llm": 0.30},
    ("technical",   "quarterly"): {"ml": 0.45, "llm": 0.55},
    ("fundamental", "quarterly"): {"ml": 0.00, "llm": 1.00},
    ("entropy",     "quarterly"): {"ml": 0.40, "llm": 0.60},
}


def combined_score(
    ml_score: float,
    llm_score: Optional[float],
    strategy: str,
    frequency: str,
    llm_failed: bool = False,
) -> float:
    """
    Compute combined score per Equation (2).

    If llm_failed=True or llm_score is None, falls back to pure ML (w=1.0).
    This matches the paper's methodology while handling API failures gracefully.
    """
    key = (strategy.lower(), frequency.lower())
    weights = OPTIMAL_WEIGHTS.get(key)
    if weights is None:
        raise ValueError(f"Unknown strategy/frequency: {strategy}/{frequency}")

    if llm_failed or llm_score is None:
        # Fallback to pure ML — log this upstream
        return float(np.clip(ml_score, 0.0, 1.0))

    w = weights["ml"]
    score = w * ml_score + (1 - w) * llm_score
    return float(np.clip(score, 0.0, 1.0))


def normalize_scores(raw_scores: list[float], clip_pct: float = 0.01) -> list[float]:
    """
    Percentile-based normalization to [0, 1].
    Clips at 1st/99th percentiles to reduce outlier distortion.
    Source: Section 3.4, Cohen et al. (2025).
    """
    arr = np.array(raw_scores, dtype=float)
    lo = np.nanpercentile(arr, clip_pct * 100)
    hi = np.nanpercentile(arr, (1 - clip_pct) * 100)
    if hi == lo:
        return [0.5] * len(raw_scores)
    normed = np.clip(arr, lo, hi)
    return list((normed - lo) / (hi - lo))


def rank_normalize(raw_by_ticker: dict[str, Optional[float]]) -> dict[str, float]:
    """
    Cross-sectional percentile-RANK normalization to [0, 1].

    Each ticker's score becomes its fractional rank across the universe:
        score = (#values strictly below + 0.5 × #ties) / n   ∈ (0, 1)

    This is the robust reading of the paper's "percentile-based normalization"
    (Section 3.4, Cohen et al. 2025). Unlike min-max scaling — `(x-min)/(max-min)`
    — it is immune to right-skew: a single large prediction can no longer set the
    max and crush every other name toward zero. The output is ~uniform on [0, 1]
    regardless of the raw distribution's shape, which is what a cross-sectional
    SELECTION task needs (relative ordering, evenly spread, no flooring).

    Note: a uniform spread is produced *by construction*; it certifies
    differentiation, NOT predictive power. The IC harness remains the test of
    whether the ranks are actually informative.

    Ties receive the average (mid) rank. None/NaN inputs are dropped.
    n == 0 → {}; n == 1 → {ticker: 0.5}.
    """
    items = [
        (t, float(v)) for t, v in raw_by_ticker.items()
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0][0]: 0.5}

    vals = np.array([v for _, v in items], dtype=float)
    order = np.argsort(vals, kind="mergesort")
    sorted_vals = vals[order]
    # Average-rank for ties: assign each value the mean of the fractional ranks
    # it would occupy. Fractional rank r_k = (k + 0.5) / n for position k (0-based).
    frac = (np.arange(n) + 0.5) / n
    out_sorted = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        out_sorted[i : j + 1] = frac[i : j + 1].mean()
        i = j + 1
    result = {}
    for pos, idx in enumerate(order):
        result[items[idx][0]] = float(out_sorted[pos])
    return result


def percentile_into(reference_values: list[float], x: float) -> float:
    """
    Rank a single value into a reference distribution, on the SAME scale that
    rank_normalize produces. This is what lets an on-demand single-ticker score
    be cross-sectionally comparable to scores computed during a full universe run.

    Returns (count_below + 0.5 * count_equal) / n  ∈ (0, 1), where the counts are
    taken over `reference_values` PLUS x itself (so a value at the extreme top of
    the reference doesn't collapse to exactly 1.0, matching rank_normalize where
    every member is ranked among the full set).

    Empty reference → 0.5 (no basis to rank). NaNs in the reference are dropped.
    """
    ref = [
        float(v) for v in reference_values
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.5
    x = float(x)
    n = len(ref) + 1  # include x itself in the ranking population
    below = sum(1 for v in ref if v < x)
    equal = sum(1 for v in ref if v == x) + 1  # x ties with itself
    return (below + 0.5 * equal) / n


def select_top_n(scores: dict[str, float], n: int = 10) -> list[str]:
    """
    Select top-N tickers by score — equal-weighted portfolio formation.
    Source: Section 3.4 (portfolio_size = 10 per paper methodology).
    """
    return sorted(scores, key=lambda t: scores[t], reverse=True)[:n]
