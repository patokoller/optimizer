"""
Portfolio risk analytics — pure functions over a returns matrix and weights.

Used by the portfolio report (Feature B) to produce current-vs-proposed risk
comparisons. No I/O; all inputs are plain arrays/dicts so this is unit-testable.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def portfolio_returns(returns_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted daily return series from a (T x n) returns frame and weights dict.
    Tickers absent from the frame are dropped; weights are renormalised over what
    remains so a missing name doesn't silently zero the series."""
    cols = [t for t in weights if t in returns_df.columns]
    if not cols:
        return pd.Series(dtype=float)
    w = np.array([weights[t] for t in cols], dtype=float)
    s = w.sum()
    if s <= 0:
        return pd.Series(dtype=float)
    w = w / s
    return returns_df[cols].fillna(0.0).dot(w)


def annualized_return(port_ret: pd.Series) -> Optional[float]:
    if port_ret.empty:
        return None
    return float(port_ret.mean() * TRADING_DAYS)


def annualized_vol(port_ret: pd.Series) -> Optional[float]:
    if port_ret.empty or len(port_ret) < 2:
        return None
    return float(port_ret.std(ddof=1) * math.sqrt(TRADING_DAYS))


def sharpe_ratio(port_ret: pd.Series, risk_free: float = 0.0) -> Optional[float]:
    """Annualized Sharpe (zero risk-free by default, matching the paper)."""
    vol = annualized_vol(port_ret)
    ann = annualized_return(port_ret)
    if vol is None or ann is None or vol == 0:
        return None
    return float((ann - risk_free) / vol)


def max_drawdown(port_ret: pd.Series) -> Optional[float]:
    """Largest peak-to-trough decline of the cumulative return path (<= 0)."""
    if port_ret.empty:
        return None
    cum = (1.0 + port_ret).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def herfindahl(weights: dict[str, float]) -> Optional[float]:
    """HHI concentration in [0,1]; 1/n (diversified) → 1 (single name)."""
    w = np.array([v for v in weights.values() if v is not None], dtype=float)
    s = w.sum()
    if s <= 0:
        return None
    w = w / s
    return float(np.sum(w ** 2))


def sector_weights(weights: dict[str, float], sectors: dict[str, str]) -> dict[str, float]:
    """Aggregate position weights by sector (unknown → 'Unknown')."""
    out: dict[str, float] = {}
    total = sum(v for v in weights.values() if v) or 1.0
    for t, w in weights.items():
        if not w:
            continue
        sec = sectors.get(t) or "Unknown"
        out[sec] = out.get(sec, 0.0) + w / total
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def risk_summary(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    sectors: Optional[dict[str, str]] = None,
) -> dict:
    """Bundle the headline risk metrics for one weight vector."""
    pr = portfolio_returns(returns_df, weights)
    return {
        "annualized_return": annualized_return(pr),
        "annualized_vol": annualized_vol(pr),
        "sharpe": sharpe_ratio(pr),
        "max_drawdown": max_drawdown(pr),
        "hhi": herfindahl(weights),
        "n_positions": len([w for w in weights.values() if w]),
        "sector_weights": sector_weights(weights, sectors or {}),
    }
