"""Tests for Feature B: portfolio_risk metrics, report action/narrative logic,
and the PDF renderer (smoke)."""

import os
os.environ.setdefault("DATABASE_URL", "postgresql://u:p@localhost:5432/x")

import io
import numpy as np
import pandas as pd

from app.ml import portfolio_risk as pr
from app.services.portfolio_report import (
    derive_actions, compose_rationale, fallback_narrative,
)
from app.services.report_pdf import build_report_pdf


# ── portfolio_risk ───────────────────────────────────────────────────────────
def _rets():
    np.random.seed(0)
    return pd.DataFrame({
        "A": np.random.normal(0.0005, 0.008, 400),
        "B": np.random.normal(0.0007, 0.015, 400),
        "C": np.random.normal(0.0010, 0.030, 400),
    })


def test_risk_summary_fields():
    s = pr.risk_summary(_rets(), {"A": 0.5, "B": 0.3, "C": 0.2},
                        sectors={"A": "Tech", "B": "Tech", "C": "Energy"})
    assert s["annualized_vol"] > 0
    assert s["sharpe"] is not None
    assert 0 < s["hhi"] < 1
    assert s["max_drawdown"] <= 0
    assert abs(sum(s["sector_weights"].values()) - 1.0) < 1e-9


def test_hhi_bounds():
    assert abs(pr.herfindahl({"a": 1, "b": 1, "c": 1, "d": 1}) - 0.25) < 1e-9
    assert abs(pr.herfindahl({"a": 1.0, "b": 0.0}) - 1.0) < 1e-9
    assert pr.herfindahl({"a": 0.0}) is None


def test_missing_ticker_renormalizes():
    rets = _rets()
    s = pr.portfolio_returns(rets, {"A": 0.5, "Z": 0.5})  # Z absent → all on A
    assert abs(s.mean() - rets["A"].mean()) < 1e-9


def test_empty_returns_safe():
    assert pr.sharpe_ratio(pd.Series(dtype=float)) is None
    assert pr.max_drawdown(pd.Series(dtype=float)) is None


# ── action derivation ────────────────────────────────────────────────────────
def test_derive_actions_classification_and_order():
    wc = {"NVDA": 0.30, "PAYX": 0.22, "AAPL": 0.28, "ZM": 0.20}
    wp = {"NVDA": 0.24, "PAYX": 0.0, "AAPL": 0.38, "ZM": 0.20}
    sc = {"NVDA": 0.71, "PAYX": 0.38, "AAPL": 0.55, "ZM": 0.33}
    dr = {"NVDA": "STABLE", "PAYX": "DETERIORATING", "AAPL": "STABLE", "ZM": "DETERIORATING"}
    acts = derive_actions(wc, wp, sc, dr)
    by = {a["ticker"]: a for a in acts}
    assert by["PAYX"]["action"] == "EXIT"
    assert by["NVDA"]["action"] == "TRIM"
    assert by["AAPL"]["action"] == "ADD"
    assert by["ZM"]["action"] == "HOLD"
    assert acts[0]["action"] == "EXIT"  # most impactful first


def test_rationale_anchors_to_evidence():
    r = compose_rationale(0.38, "DETERIORATING", "EXIT")
    assert "bottom-tercile" in r.lower() and "deteriorating" in r.lower()
    r2 = compose_rationale(0.71, "STABLE", "TRIM")
    assert "top-tercile" in r2.lower()


def test_fallback_narrative_uses_numbers():
    data = {
        "holdings": [{"ticker": "X"}] * 4,
        "watch_items": ["PAYX", "ZM"],
        "risk_current": {"sharpe": 0.67, "hhi": 0.265, "annualized_vol": 0.27,
                         "annualized_return": 0.18, "sector_weights": {"Tech": 0.8}},
        "risk_proposed": {"sharpe": 0.83, "hhi": 0.22, "annualized_vol": 0.21,
                          "annualized_return": 0.175},
    }
    nar = fallback_narrative(data)
    assert set(nar) == {"exec_summary", "risk_commentary", "closing"}
    assert "0.67" in nar["exec_summary"] and "0.83" in nar["exec_summary"]
    assert "PAYX" in nar["exec_summary"]


# ── PDF render (smoke) ───────────────────────────────────────────────────────
def test_build_report_pdf_smoke():
    data = {
        "portfolio_name": "Test", "as_of": "2026-06-15",
        "regime": {"label": "Neutral / Mixed", "confidence": 0.7},
        "narrative": {"exec_summary": "E.", "risk_commentary": "R.", "closing": "C."},
        "holdings": [
            {"ticker": "NVDA", "weight": 0.5, "overall_score": 0.71,
             "strategies": {"technical": {"combined": 0.8}, "fundamental": {"combined": 0.6},
                            "entropy": {"combined": 0.65}}, "drift_trend": "STABLE"},
            {"ticker": "ZM", "weight": 0.5, "overall_score": 0.33,
             "strategies": {"technical": {"combined": 0.3}, "fundamental": {"combined": 0.35},
                            "entropy": {"combined": 0.34}}, "drift_trend": "DETERIORATING"},
        ],
        "risk_current": {"annualized_return": 0.18, "annualized_vol": 0.27, "sharpe": 0.67,
                         "max_drawdown": -0.34, "hhi": 0.5, "n_positions": 2,
                         "sector_weights": {"Tech": 1.0}},
        "risk_proposed": {"annualized_return": 0.17, "annualized_vol": 0.21, "sharpe": 0.8,
                          "max_drawdown": -0.26, "hhi": 0.5, "n_positions": 2,
                          "sector_weights": {"Tech": 1.0}},
        "optimizer": "MVO",
        "actions": [{"ticker": "ZM", "action": "EXIT", "delta": -0.5, "rationale": "Weak."}],
        "watch_items": ["ZM"],
        "stress_test": {"series": None, "note": "Note."},
    }
    pdf = build_report_pdf(data)
    assert pdf[:4] == b"%PDF"
    from pypdf import PdfReader
    r = PdfReader(io.BytesIO(pdf))
    assert len(r.pages) >= 2
    assert "Portfolio Analysis" in r.pages[0].extract_text()
