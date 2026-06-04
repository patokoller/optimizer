"""Test cross-sectional peer-percentile context (#19)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from app.ml.peer_context import compute_peer_percentiles, format_peer_context

RB = pd.Timestamp("2026-05-01")


def _fundamentals():
    rows = []
    for tk, nm, om, growth in [("AAA", 0.30, 0.35, 0.20), ("BBB", 0.15, 0.18, 0.05), ("CCC", 0.05, 0.07, -0.05)]:
        rev = 1000.0
        for q in ["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01", "2026-01-01", "2026-04-01"]:
            rows.append({"ticker": tk, "period_date": q, "revenue": rev, "net_margin": nm, "operating_margin": om})
            rev *= (1 + growth)
    return pd.DataFrame(rows)


def _prices():
    rng = np.random.default_rng(7)
    rows = []
    for tk, drift, vol in [("AAA", 0.0015, 0.010), ("BBB", 0.0, 0.018), ("CCC", -0.001, 0.035)]:
        px = 100.0
        for d in pd.bdate_range("2025-07-01", "2026-04-25"):
            px *= (1 + drift + rng.normal(0, vol))
            rows.append({"ticker": tk, "date": d, "close": max(px, 1.0)})
    return pd.DataFrame(rows)


def test_ranking_directions():
    p = compute_peer_percentiles(_fundamentals(), _prices(), ["AAA", "BBB", "CCC"], RB)
    assert p["AAA"]["net_margin"] > p["BBB"]["net_margin"] > p["CCC"]["net_margin"]
    assert p["AAA"]["revenue_growth"] > p["CCC"]["revenue_growth"]
    assert p["AAA"]["momentum_6m"] > p["CCC"]["momentum_6m"]
    assert p["CCC"]["volatility"] > p["AAA"]["volatility"]   # CCC is the noisiest


def test_no_lookahead():
    f = _fundamentals()
    base = compute_peer_percentiles(f, _prices(), ["AAA", "BBB", "CCC"], RB)
    leaked = pd.concat([f, pd.DataFrame([{"ticker": "AAA", "period_date": "2026-07-01",
                                          "revenue": 9999, "net_margin": 0.99, "operating_margin": 0.99}])])
    after = compute_peer_percentiles(leaked, _prices(), ["AAA", "BBB", "CCC"], RB)
    assert after["AAA"]["net_margin"] == base["AAA"]["net_margin"]   # future report ignored


def test_format_and_empty():
    p = compute_peer_percentiles(_fundamentals(), _prices(), ["AAA"], RB)
    s = format_peer_context(p["AAA"], universe_n=3)
    assert "percentile rank" in s and "net margin" in s
    assert format_peer_context({}) == ""


def test_missing_data_graceful():
    # no fundamentals → only price features; no prices → only fundamentals
    only_px = compute_peer_percentiles(None, _prices(), ["AAA", "CCC"], RB)
    assert "momentum_6m" in only_px["AAA"] and "net_margin" not in only_px["AAA"]
    only_f = compute_peer_percentiles(_fundamentals(), None, ["AAA", "CCC"], RB)
    assert "net_margin" in only_f["AAA"] and "momentum_6m" not in only_f["AAA"]
    both_none = compute_peer_percentiles(None, None, ["AAA"], RB)
    assert both_none["AAA"] == {}
