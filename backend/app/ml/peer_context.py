"""
app/ml/peer_context.py
─────────────────────────────────────────────────────────────────────────────
Cross-sectional peer context for the LLM scorer (#19).

The LLM currently scores each stock in isolation from absolute numbers and has
no relative anchor, so its scores compress toward the middle — everything looks
"fine." This computes where each name sits *relative to the period's universe*
on a few interpretable features and injects that as a compact percentile block,
giving the model the relative positioning it needs to differentiate.

No-lookahead: fundamentals use only reports dated before the rebalance date;
prices use only bars before it.

This is the percentile-injection half of #19. The costlier second-pass top-20
head-to-head ranking is a separate, measured/gated step — not included here.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def _f(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _pct_ranks(values: dict) -> dict:
    """Cross-sectional percentile rank (0–100) per ticker for one feature.
    Single/empty peer set → neutral 50.0 (no relative information)."""
    items = [(t, v) for t, v in values.items() if v is not None]
    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0][0]: 50.0}
    vals = [v for _, v in items]
    out = {}
    for t, v in items:
        less = sum(1 for x in vals if x < v)
        eq = sum(1 for x in vals if x == v)
        out[t] = round(100.0 * (less + 0.5 * eq) / n, 1)
    return out


# feature key → (display label, higher_is_stronger)
_LABELS = [
    ("revenue_growth",   "revenue growth"),
    ("net_margin",       "net margin"),
    ("operating_margin", "operating margin"),
    ("momentum_6m",      "6-month momentum"),
    ("momentum_3m",      "3-month momentum"),
    ("volatility",       "volatility (lower = calmer)"),
]


def compute_peer_percentiles(fundamentals_df, prices_df, tickers, rebalance_date) -> dict:
    """Return {ticker: {feature: percentile_0_100}} computed cross-sectionally
    across `tickers`, using only data available before `rebalance_date`."""
    feats = {k: {} for k, _ in _LABELS}
    rb = pd.Timestamp(rebalance_date)

    if fundamentals_df is not None and len(fundamentals_df):
        fdf = fundamentals_df.copy()
        fdf["period_date"] = pd.to_datetime(fdf["period_date"])
        fdf = fdf[fdf["period_date"] < rb]
        for tk, g in fdf.groupby("ticker"):
            g = g.sort_values("period_date")
            last = g.iloc[-1]
            nm = _f(last.get("net_margin")) if "net_margin" in g else None
            om = _f(last.get("operating_margin")) if "operating_margin" in g else None
            if nm is not None:
                feats["net_margin"][tk] = nm
            if om is not None:
                feats["operating_margin"][tk] = om
            if "revenue" in g and len(g) >= 5:
                cur = _f(g.iloc[-1]["revenue"])
                prior = _f(g.iloc[-5]["revenue"])   # 4 quarters earlier
                if cur is not None and prior not in (None, 0):
                    feats["revenue_growth"][tk] = cur / prior - 1.0

    if prices_df is not None and len(prices_df):
        pdf = prices_df.copy()
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf[pdf["date"] < rb]
        for tk, g in pdf.groupby("ticker"):
            g = g.sort_values("date")
            close = g["close"].astype(float).values
            if len(close) >= 127 and close[-127] > 0:
                feats["momentum_6m"][tk] = close[-1] / close[-127] - 1.0
            if len(close) >= 64 and close[-64] > 0:
                feats["momentum_3m"][tk] = close[-1] / close[-64] - 1.0
                rets = np.diff(close[-64:]) / close[-64:-1]
                if len(rets) and np.all(np.isfinite(rets)):
                    feats["volatility"][tk] = float(np.std(rets) * np.sqrt(252))

    ranks = {k: _pct_ranks(v) for k, v in feats.items()}
    out = {}
    for tk in tickers:
        d = {k: ranks[k][tk] for k, _ in _LABELS if tk in ranks[k]}
        out[tk] = d
    return out


def format_peer_context(percentiles: dict, universe_n: int | None = None) -> str:
    """Render one ticker's percentile dict as a compact prompt block, or '' if none."""
    if not percentiles:
        return ""
    parts = [f"{label} {int(round(percentiles[k]))}th pct"
             for k, label in _LABELS if k in percentiles]
    if not parts:
        return ""
    head = "Peer position vs this period's scoreable universe"
    if universe_n:
        head += f" ({universe_n} names)"
    return head + " — percentile rank (higher = stronger, except where noted): " + "; ".join(parts) + "."
