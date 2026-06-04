"""Mocked tests for the AV macro overlay (no network).

Verifies the '.'-sentinel hygiene the probe flagged, CPI YoY, the 10Y-2Y curve,
and that the hybrid snapshot overlays AV onto FRED only when AV returns a value
(VIX always stays FRED; AV failures leave FRED values intact)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app.data.av_macro as M
import app.data.fred_client as F


def test_latest_valid_skips_sentinel():
    data = {"data": [
        {"date": "2026-05-01", "value": "."},   # newest is sentinel → skip
        {"date": "2026-04-01", "value": "4.33"},
        {"date": "2026-03-01", "value": "4.58"}]}
    assert M._latest_valid(data) == 4.33
    assert M._latest_valid({"data": [{"date": "2026-01-01", "value": "."}]}) is None
    assert M._latest_valid({}) is None


def test_cpi_yoy_sentinel_safe():
    rows = [{"date": f"2026-{m:02d}-01", "value": "." if m == 10 else str(300 + m)} for m in range(1, 13)]
    rows += [{"date": f"2025-{m:02d}-01", "value": str(280 + m)} for m in range(1, 13)]
    yoy = M.cpi_yoy_from_payload({"data": rows})
    assert yoy is not None and yoy > 0
    assert M.cpi_yoy_from_payload({"data": [{"date": "2026-01-01", "value": "300"}]}) is None  # <13


def test_curve():
    cl = M.AVMacroClient(); cl.api_key = "K"
    cl._get = lambda p: ({"data": [{"date": "2026-04-01", "value": "4.50"}]}
                         if p.get("maturity") == "10year"
                         else {"data": [{"date": "2026-04-01", "value": "4.10"}]})
    assert abs(cl.yield_curve_10y_2y() - 0.40) < 1e-9


def test_hybrid_overlay_and_fallback(monkeypatch):
    monkeypatch.setattr(F.FREDClient, "get_macro_snapshot", lambda self: {
        "vix": 18.0, "vix_trend": "stable", "yield_curve": 0.50, "curve_trend": "stable",
        "fed_funds": 5.0, "cpi_yoy": 3.0, "errors": ["FRED DFF 429", "FRED CPI 429"]})

    monkeypatch.setattr(M.AVMacroClient, "fed_funds_rate", lambda self: 4.33)
    monkeypatch.setattr(M.AVMacroClient, "cpi_yoy", lambda self: 2.7)
    monkeypatch.setattr(M.AVMacroClient, "yield_curve_10y_2y", lambda self: 0.40)
    snap = M.get_hybrid_macro_snapshot()
    assert snap["fed_funds"] == 4.33 and snap["fed_funds_source"] == "alpha_vantage"
    assert snap["cpi_yoy"] == 2.7 and snap["yield_curve"] == 0.40
    assert snap["vix"] == 18.0  # VIX never overlaid

    monkeypatch.setattr(M.AVMacroClient, "fed_funds_rate", lambda self: None)
    monkeypatch.setattr(M.AVMacroClient, "cpi_yoy", lambda self: None)
    monkeypatch.setattr(M.AVMacroClient, "yield_curve_10y_2y", lambda self: None)
    snap2 = M.get_hybrid_macro_snapshot()
    assert snap2["fed_funds"] == 5.0 and "fed_funds_source" not in snap2
