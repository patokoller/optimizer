"""
app/ml/validation.py

Pure functions for the discovery validation harness — no DB, no network, so
they stay unit-testable. The orchestration (fetch prices, read/write rows) lives
in workers/tasks.py::backfill_forward_returns.

What this measures
------------------
Whether a score column assigned at time t actually predicted the cross-section
of forward returns realised by t+h. Two complementary metrics per run/horizon:

  rank_ic     Spearman correlation of score vs realised forward return across
              the universe. The standard quant "information coefficient". A
              monthly IC that is reliably positive (even ~0.03–0.05) is a real
              edge; one indistinguishable from zero is not.

  topk_spread Mean forward return of the top-10 names by score minus the
              universe equal-weight mean. This mirrors exactly what the paper's
              strategy does (select top-10, equal weight), so it is the most
              decision-relevant number: did picking the highest scores beat
              picking everything?

Horizons are 21 and 63 trading days (≈ 1 month / 1 quarter), matching the
paper's monthly and quarterly rebalancing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Score columns we validate. Keys must match DiscoveryForwardReturn attributes
# and DiscoveryScore attributes.
SCORE_COLUMNS = ["combined_score", "technical_score",
                 "fundamental_score", "entropy_score", "llm_score"]

HORIZONS = (21, 63)          # trading days
TOPK = 10                    # paper selects top-10
MIN_PAIRS = 5                # below this, IC is meaningless


def rank_ic(scores, returns) -> tuple[float | None, int]:
    """Spearman rank IC = Pearson correlation of ranks. Dependency-free.

    Returns (ic, n) where n is the number of pairwise-complete observations.
    ic is None if fewer than MIN_PAIRS valid pairs or either side is constant.
    """
    s = np.asarray(scores, dtype=float)
    r = np.asarray(returns, dtype=float)
    mask = ~(np.isnan(s) | np.isnan(r))
    s, r = s[mask], r[mask]
    n = int(len(s))
    if n < MIN_PAIRS:
        return None, n
    rs = pd.Series(s).rank().to_numpy()
    rr = pd.Series(r).rank().to_numpy()
    if np.std(rs) == 0 or np.std(rr) == 0:
        return None, n
    return float(np.corrcoef(rs, rr)[0, 1]), n


def topk_spread(scores, returns, k: int = TOPK) -> tuple[float | None, float | None]:
    """Return (top-k mean fwd return minus universe mean, universe mean).

    Both None if fewer than k complete observations.
    """
    df = pd.DataFrame({"s": np.asarray(scores, float),
                       "r": np.asarray(returns, float)}).dropna()
    if len(df) < k:
        return None, (float(df["r"].mean()) if len(df) else None)
    uni = float(df["r"].mean())
    top = float(df.nlargest(k, "s")["r"].mean())
    return top - uni, uni


def forward_returns_from_bars(
    bars: pd.DataFrame,
    run_date,
    horizons=HORIZONS,
) -> dict[str, dict]:
    """Compute per-ticker forward returns from a long OHLCV frame.

    bars: columns [date, ticker, close] (adjusted close). `date` may be
          datetime.date or datetime64. run_date is the scoring timestamp.

    For each ticker we take the anchor = first bar on/after run_date, then the
    close `h` trading days later (by position in that ticker's own sorted
    series). Forward return = close[anchor_idx + h] / close[anchor_idx] - 1.

    Returns {ticker: {"anchor_close": float,
                      "fwd_return_21d": float|None,
                      "fwd_return_63d": float|None}}.
    A horizon is None if that ticker doesn't yet have enough future bars.
    """
    out: dict[str, dict] = {}
    if bars is None or len(bars) == 0:
        return out

    df = bars.copy()
    df["date"] = pd.to_datetime(df["date"])
    anchor_ts = pd.to_datetime(getattr(run_date, "date", lambda: run_date)()) \
        if hasattr(run_date, "date") else pd.to_datetime(run_date)
    anchor_ts = pd.Timestamp(anchor_ts).normalize()

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        future = g[g["date"] >= anchor_ts]
        if len(future) == 0:
            continue
        anchor_pos = future.index[0]
        anchor_close = float(g.loc[anchor_pos, "close"])
        if anchor_close <= 0 or np.isnan(anchor_close):
            continue
        rec = {"anchor_close": anchor_close,
               "fwd_return_21d": None, "fwd_return_63d": None}
        for h in horizons:
            tgt = anchor_pos + h
            if tgt < len(g):
                c = float(g.loc[tgt, "close"])
                if c > 0 and not np.isnan(c):
                    rec[f"fwd_return_{h}d"] = c / anchor_close - 1.0
        out[ticker] = rec
    return out


def score_distribution(values) -> dict:
    """Per-run summary of a score column for compression/calibration monitoring.

    Returns count, mean, std, min, p10/p25/median/p75/p90, max, IQR, a 10-bin
    decile histogram over [0,1], and a `compressed` flag. Score compression
    (everything bunched near the middle) is the failure mode that hurts a
    top-10 selection most, so this makes it visible per run and lets later
    scoring changes be judged by their effect on the spread.
    """
    import statistics
    vals = sorted(float(v) for v in values if v is not None)
    n = len(vals)
    if n == 0:
        return {"n": 0}

    def pct(p):
        if n == 1:
            return vals[0]
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        return vals[f] + (vals[c] - vals[f]) * (k - f)

    mean = sum(vals) / n
    std = statistics.pstdev(vals) if n > 1 else 0.0
    p25, p50, p75 = pct(0.25), pct(0.5), pct(0.75)
    iqr = p75 - p25
    bins = [0] * 10
    for v in vals:
        bins[min(9, max(0, int(v * 10)))] += 1
    return {
        "n": n, "mean": round(mean, 4), "std": round(std, 4),
        "min": round(vals[0], 4), "p10": round(pct(0.1), 4), "p25": round(p25, 4),
        "median": round(p50, 4), "p75": round(p75, 4), "p90": round(pct(0.9), 4),
        "max": round(vals[-1], 4), "iqr": round(iqr, 4),
        "histogram_deciles": bins,
        "compressed": bool(std < 0.08 or iqr < 0.10),
    }
