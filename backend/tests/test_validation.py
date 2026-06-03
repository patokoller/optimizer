"""Unit tests for the discovery validation harness (pure logic, no DB/network)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from datetime import datetime
from app.ml.validation import (
    rank_ic, topk_spread, forward_returns_from_bars,
)


def test_rank_ic_perfect_and_inverse():
    s = np.arange(98, dtype=float)
    assert abs(rank_ic(s, s * 0.01)[0] - 1.0) < 1e-9
    assert abs(rank_ic(s, -s * 0.01)[0] + 1.0) < 1e-9


def test_rank_ic_noise_is_near_zero():
    rng = np.random.default_rng(42)
    ics = [rank_ic(rng.normal(size=98), rng.normal(size=98))[0] for _ in range(1000)]
    assert abs(np.mean(ics)) < 0.03


def test_rank_ic_nan_safe_and_degenerate():
    assert rank_ic([1, 2, np.nan, np.nan], [1, np.nan, 3, np.nan])[0] is None  # <5 pairs
    assert rank_ic([5, 5, 5, 5, 5, 5], [1, 2, 3, 4, 5, 6])[0] is None         # constant score


def test_topk_spread():
    s = np.arange(50, dtype=float)
    r = np.arange(50, dtype=float) * 0.01
    spread, uni = topk_spread(s, r, k=10)
    assert spread > 0 and uni is not None


def test_forward_returns_weekend_anchor_and_horizons():
    dates = pd.bdate_range("2026-01-02", periods=120)
    rows = [{"date": d.date(), "ticker": "AAA", "close": 100.0 + i}
            for i, d in enumerate(dates)]
    bars = pd.DataFrame(rows)
    # 2026-01-03 is Saturday → anchor snaps to Monday (index 1, close 101.0)
    res = forward_returns_from_bars(bars, datetime(2026, 1, 3))
    assert abs(res["AAA"]["anchor_close"] - 101.0) < 1e-9
    assert abs(res["AAA"]["fwd_return_21d"] - ((101.0 + 21) / 101.0 - 1.0)) < 1e-9
    assert abs(res["AAA"]["fwd_return_63d"] - ((101.0 + 63) / 101.0 - 1.0)) < 1e-9


def test_forward_returns_maturity_and_empty():
    dates = pd.bdate_range("2026-01-02", periods=30)
    bars = pd.DataFrame([{"date": d.date(), "ticker": "AAA", "close": 100.0 + i}
                         for i, d in enumerate(dates)])
    res = forward_returns_from_bars(bars, datetime(2026, 1, 2))
    assert res["AAA"]["fwd_return_21d"] is not None     # 21 < 30 bars
    assert res["AAA"]["fwd_return_63d"] is None          # 63 > 30 bars → not matured
    assert forward_returns_from_bars(None, datetime(2026, 1, 2)) == {}
