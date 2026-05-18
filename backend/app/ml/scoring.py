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


def select_top_n(scores: dict[str, float], n: int = 10) -> list[str]:
    """
    Select top-N tickers by score — equal-weighted portfolio formation.
    Source: Section 3.4 (portfolio_size = 10 per paper methodology).
    """
    return sorted(scores, key=lambda t: scores[t], reverse=True)[:n]
