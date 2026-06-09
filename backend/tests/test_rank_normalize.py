"""Tests for scoring.rank_normalize — the percentile-rank cross-sectional
normalizer that replaced the min-max scaling which floored skewed ML predictions
near zero (tech_ml/fund_ml histograms in the 2026-06-09 discovery run)."""
import numpy as np
from app.ml.scoring import rank_normalize


def test_decompresses_right_skewed_distribution():
    # 95 tiny clustered predictions + 1 huge outlier == the logged pathology.
    raw = {f"T{i}": 0.001 * i for i in range(95)}
    raw["BIG"] = 50.0
    out = rank_normalize(raw)
    vals = np.array(list(out.values()))
    # min-max gave std ~0.10 and 64/96 in the bottom decile; rank gives ~uniform.
    assert vals.std() > 0.25
    assert abs(vals.mean() - 0.5) < 0.02
    hist = np.histogram(vals, bins=10, range=(0, 1))[0]
    assert hist.min() >= 7  # every decile populated, none floored/empty


def test_monotonic_order_preserved():
    raw = {"a": 0.1, "b": 0.5, "c": 0.9, "d": 100.0}
    out = rank_normalize(raw)
    assert out["a"] < out["b"] < out["c"] < out["d"]
    assert out["d"] == max(out.values())


def test_ties_get_average_rank():
    out = rank_normalize({"a": 1.0, "b": 1.0, "c": 1.0, "d": 5.0})
    assert out["a"] == out["b"] == out["c"]
    assert out["d"] > out["a"]


def test_edge_cases_and_nan_handling():
    assert rank_normalize({}) == {}
    assert rank_normalize({"x": 0.7}) == {"x": 0.5}
    out = rank_normalize({"a": 1.0, "b": None, "c": float("nan"), "d": 2.0})
    assert set(out) == {"a", "d"}  # None/NaN dropped
    assert out["d"] > out["a"]


def test_output_bounded_open_interval():
    out = rank_normalize({f"T{i}": float(i) for i in range(50)})
    assert all(0.0 < v < 1.0 for v in out.values())
