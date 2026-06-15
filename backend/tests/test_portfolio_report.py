"""Tests for Feature B: portfolio_risk metrics, report action/narrative logic,
and the PDF renderer (smoke)."""

import os
os.environ.setdefault("DATABASE_URL", "postgresql://u:p@localhost:5432/x")

import io
import numpy as np
import pandas as pd

from app.ml import portfolio_risk as pr
from app.services.portfolio_report import (
    derive_actions, compose_rationale, fallback_narrative, fallback_advisor_view,
    fallback_review_outlook,
)
from app.ml.portfolio_risk import monthly_movers
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


def test_fallback_advisor_view_is_opinionated_and_grounded():
    data = {
        "portfolio_name": "P", "risk_current": {"sharpe": 0.67, "hhi": 0.27, "annualized_vol": 0.27},
        "risk_proposed": {"sharpe": 0.83, "annualized_vol": 0.21},
        "watch_items": ["PAYX", "ZM"],
        "actions": [], "overall_posture_score": 0.5,
        "holdings": [
            {"ticker": "NVDA", "weight": 0.30, "overall_score": 0.71, "drift_trend": "STABLE", "llm": {}},
            {"ticker": "AAPL", "weight": 0.28, "overall_score": 0.55, "drift_trend": "STABLE", "llm": {}},
            {"ticker": "PAYX", "weight": 0.22, "overall_score": 0.38, "drift_trend": "DETERIORATING", "llm": {}},
            {"ticker": "ZM", "weight": 0.20, "overall_score": 0.33, "drift_trend": "DETERIORATING", "llm": {}},
        ],
    }
    av = fallback_advisor_view(data)
    assert set(av) == {"stance", "conviction", "key_points", "recommended_posture"}
    assert av["conviction"] == "cautious"  # concentrated + watch list
    assert "PAYX" in av["stance"] and "0.67" in av["stance"] and "0.83" in av["stance"]
    assert isinstance(av["key_points"], list) and av["key_points"]
    assert av["recommended_posture"]


def test_fallback_advisor_view_balanced_book():
    data = {
        "portfolio_name": "P", "risk_current": {"sharpe": 0.7, "hhi": 0.12, "annualized_vol": 0.15},
        "risk_proposed": {"sharpe": 0.72, "annualized_vol": 0.14}, "watch_items": [],
        "actions": [], "overall_posture_score": 0.6,
        "holdings": [{"ticker": t, "weight": 0.25, "overall_score": 0.6, "drift_trend": "STABLE", "llm": {}}
                     for t in ("A", "B", "C", "D")],
    }
    av = fallback_advisor_view(data)
    assert av["conviction"] in ("moderate", "high")  # not cautious when balanced


def test_monthly_movers_orders_and_attributes():
    rets = pd.DataFrame({
        "A": np.r_[np.zeros(40), np.full(21, 0.0)],
        "B": np.r_[np.zeros(40), np.full(21, -0.004)],
        "C": np.r_[np.zeros(40), np.full(21, 0.006)],
    })
    m = monthly_movers(rets, {"A": 0.4, "B": 0.3, "C": 0.3}, window=21)
    assert m[0]["ticker"] == "C" and m[-1]["ticker"] == "B"
    assert abs(m[0]["contribution"] - 0.3 * m[0]["period_return"]) < 1e-9
    assert monthly_movers(pd.DataFrame(), {"A": 1.0}) == []


def test_fallback_review_outlook_grounded_and_labeled():
    data = {
        "regime": {"label": "Neutral / Mixed"}, "watch_items": ["PAYX", "ZM"],
        "movers": [
            {"ticker": "NVDA", "weight": 0.30, "period_return": 0.12, "contribution": 0.036},
            {"ticker": "ZM", "weight": 0.20, "period_return": -0.09, "contribution": -0.018},
        ],
        "actions": [{"ticker": "PAYX", "action": "EXIT", "delta": -0.22},
                    {"ticker": "AAPL", "action": "ADD", "delta": 0.10}],
        "holdings": [
            {"ticker": "NVDA", "drift_trend": "STABLE",
             "llm": {"key_positives": ["Datacenter demand still outrunning supply"]}},
            {"ticker": "ZM", "drift_trend": "DETERIORATING",
             "llm": {"key_risks": ["Enterprise competition intensifying"]}},
        ],
    }
    r = fallback_review_outlook(data)
    assert set(r) == {"key_developments", "future_positioning"}
    assert "NVDA" in r["key_developments"] and "+12.0%" in r["key_developments"]
    assert "supply" in r["key_developments"].lower()          # supply-chain woven in
    assert "PAYX, ZM" in r["key_developments"]                 # drift note
    assert "model-derived" in r["future_positioning"].lower()  # honesty label
    assert "neutral" in r["future_positioning"].lower()        # regime grounded


def test_fallback_review_outlook_empty_safe():
    r = fallback_review_outlook({"movers": [], "holdings": [], "actions": [], "regime": None})
    assert "No single position" in r["key_developments"]
    assert r["future_positioning"]



    data = {
        "portfolio_name": "Test", "as_of": "2026-06-15",
        "regime": {"label": "Neutral / Mixed", "confidence": 0.7},
        "overall_posture_score": 0.52,
        "narrative": {"exec_summary": "E.", "risk_commentary": "R.", "closing": "C."},
        "advisor_view": {"stance": "I would reduce concentration first.", "conviction": "moderate",
                         "key_points": ["Top names dominate risk.", "Two watch-list names."],
                         "recommended_posture": "Trim the top, exit the weakest, hold the rest."},
        "review": {"key_developments": "NVDA led; ZM lagged.",
                   "future_positioning": "Model-derived defensive tilt."},
        "movers": [{"ticker": "NVDA", "weight": 0.5, "period_return": 0.1, "contribution": 0.05}],
        "holdings": [
            {"ticker": "NVDA", "company": "NVIDIA", "weight": 0.5, "overall_score": 0.71,
             "strategies": {"technical": {"combined": 0.8}, "fundamental": {"combined": 0.6},
                            "entropy": {"combined": 0.65}}, "drift_trend": "STABLE",
             "llm": {"key_positives": ["Demand strong"], "key_risks": ["Valuation rich"]}},
            {"ticker": "ZM", "company": "Zoom", "weight": 0.5, "overall_score": 0.33,
             "strategies": {"technical": {"combined": 0.3}, "fundamental": {"combined": 0.35},
                            "entropy": {"combined": 0.34}}, "drift_trend": "DETERIORATING",
             "llm": {"key_positives": ["Net cash"], "key_risks": ["Growth stalled", "Competition"]}},
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
    assert len(r.pages) >= 3  # cover + content
    # Advisor's view present in the document text
    alltext = " ".join(p.extract_text() for p in r.pages)
    assert "Advisor" in alltext and "Executive summary" in alltext
    assert "Review" in alltext and "Future positioning" in alltext
