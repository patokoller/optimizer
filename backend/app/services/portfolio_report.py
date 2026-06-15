"""
Portfolio report orchestrator (Feature B).

Assembles ReportData from existing pieces — scoring (score_one, cached), risk
analytics (portfolio_risk), the MVO/HRP optimizer, and per-holding drift/LLM
context — then asks Claude for the narrative sections. The PDF is rendered by
app/services/report_pdf.build_report_pdf.

The agent ORCHESTRATES; it introduces no new model. Its voice is advisory: every
proposed action ties to a holding's score and is left for the user to accept,
adjust, or reject (scores are an unvalidated research signal, not a verdict).

Pure helpers (derive_actions, compose_rationale, fallback_narrative) are unit-
tested; build_report_data is the live wrapper.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from app.ml.portfolio_risk import risk_summary
from app.services.score_one import score_one

logger = logging.getLogger(__name__)

_ADD_THRESHOLD = 0.02   # weight delta above which we call it ADD
_TRIM_THRESHOLD = -0.02
_EXIT_FLOOR = 0.005     # proposed weight at/below this (from a real position) = EXIT


def compose_rationale(score: Optional[float], drift: Optional[str], action: str) -> str:
    """One-sentence, evidence-anchored rationale tying the action to the score."""
    sc = "no score" if score is None else (
        "top-tercile score" if score >= 0.66 else
        "mid-pack score" if score >= 0.4 else "bottom-tercile score"
    )
    drift_clause = ""
    if drift == "DETERIORATING":
        drift_clause = "; management language deteriorating across recent calls"
    elif drift == "IMPROVING":
        drift_clause = "; management language improving across recent calls"

    if action == "EXIT":
        return f"{sc.capitalize()}{drift_clause}. Optimizer drops the position."
    if action == "TRIM":
        return f"{sc.capitalize()}{drift_clause}. Trim to reduce its weight/risk contribution."
    if action == "ADD":
        return f"{sc.capitalize()}{drift_clause}. Optimizer raises the weight."
    return f"{sc.capitalize()}{drift_clause}. Hold at roughly the current weight."


def derive_actions(
    weights_current: dict[str, float],
    weights_proposed: dict[str, float],
    scores: dict[str, Optional[float]],
    drift: dict[str, Optional[str]],
) -> list[dict]:
    """Pure: turn current/proposed weights + scores + drift into ranked actions."""
    actions = []
    for t in weights_current:
        cw = weights_current.get(t, 0.0) or 0.0
        pw = weights_proposed.get(t, 0.0) or 0.0
        delta = pw - cw
        if pw <= _EXIT_FLOOR < cw:
            action = "EXIT"
        elif delta > _ADD_THRESHOLD:
            action = "ADD"
        elif delta < _TRIM_THRESHOLD:
            action = "TRIM"
        else:
            action = "HOLD"
        actions.append({
            "ticker": t,
            "action": action,
            "delta": delta,
            "rationale": compose_rationale(scores.get(t), drift.get(t), action),
        })
    # Most impactful first: EXIT/large moves at the top.
    order = {"EXIT": 0, "TRIM": 1, "ADD": 2, "HOLD": 3}
    actions.sort(key=lambda a: (order[a["action"]], -abs(a["delta"])))
    return actions


def fallback_narrative(data: dict) -> dict:
    """Deterministic narrative from the numbers — used when the LLM is unavailable
    so the report is never blank."""
    cur, prop = data.get("risk_current", {}), data.get("risk_proposed", {})
    watch = data.get("watch_items") or []
    n = len(data.get("holdings", []))
    top_sec = next(iter(cur.get("sector_weights", {})), None)
    exec_s = (
        f"This {n}-position portfolio's largest exposure is "
        f"{top_sec or 'n/a'}. "
        f"{('Holdings flagged for review: ' + ', '.join(watch) + '. ') if watch else ''}"
        f"The proposed reallocation moves the Sharpe ratio from "
        f"{_n(cur.get('sharpe'))} to {_n(prop.get('sharpe'))} and concentration (HHI) "
        f"from {_n(cur.get('hhi'),3)} to {_n(prop.get('hhi'),3)}."
    )
    risk_s = (
        f"Proposed volatility {_p(prop.get('annualized_vol'))} vs current "
        f"{_p(cur.get('annualized_vol'))}; annualized return "
        f"{_p(prop.get('annualized_return'))} vs {_p(cur.get('annualized_return'))}."
    )
    return {
        "exec_summary": exec_s,
        "risk_commentary": risk_s,
        "closing": "Proposals are advisory and tie to each holding's score; the decision rests with you.",
    }


def _p(x, d=1):
    return "-" if x is None else f"{x*100:.{d}f}%"


def _n(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def generate_narrative(llm_scorer, data: dict) -> dict:
    """Ask Claude to write the three narrative sections from the structured data.
    Falls back to a deterministic narrative on any failure."""
    try:
        client = getattr(llm_scorer, "client", None)
        if client is None:
            return fallback_narrative(data)
        # Compact, number-grounded context (no raw holdings dump beyond essentials).
        ctx = {
            "portfolio": data.get("portfolio_name"),
            "risk_current": data.get("risk_current"),
            "risk_proposed": data.get("risk_proposed"),
            "watch_items": data.get("watch_items"),
            "actions": [{k: a[k] for k in ("ticker", "action", "delta")} for a in data.get("actions", [])],
            "holdings": [{"ticker": h["ticker"], "overall_score": h.get("overall_score"),
                          "drift": h.get("drift_trend")} for h in data.get("holdings", [])],
        }
        prompt = (
            "You are a portfolio analyst writing a client-grade memo from the JSON below. "
            "Write in a precise, neutral, advisory voice — propose, do not command; tie claims "
            "to the numbers. Do NOT invent figures not present. Return ONLY JSON with keys "
            "exec_summary (3-4 sentences), risk_commentary (2-3 sentences), closing (1-2 sentences).\n\n"
            f"{json.dumps(ctx, default=str)}"
        )
        resp = client.messages.create(
            model=llm_scorer.model, max_tokens=700, temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        # Require all three keys; else fall back.
        if all(k in parsed for k in ("exec_summary", "risk_commentary", "closing")):
            return parsed
        return fallback_narrative(data)
    except Exception as e:
        logger.warning(f"Narrative generation failed ({e}) — using deterministic fallback")
        return fallback_narrative(data)


def _returns_from_bars(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot OHLCV long frame (date,ticker,close) → wide daily returns."""
    if prices_df is None or prices_df.empty:
        return pd.DataFrame()
    df = prices_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="ticker", values="close").sort_index()
    return wide.pct_change().dropna(how="all")


def build_report_data(
    db,
    portfolio_id: str,
    *,
    optimizer: str = "MVO",
    alpaca=None,
    llm_scorer=None,
) -> dict:
    """Live: assemble ReportData for a portfolio. Returns {"error": ...} if the
    portfolio is empty or unscoreable."""
    from app import models
    from app.optimizer.deep_rl import mvo_optimize, hrp_optimize

    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if portfolio is None:
        return {"error": "portfolio_not_found", "portfolio_id": portfolio_id}
    holdings = list(portfolio.holdings)
    if not holdings:
        return {"error": "empty_portfolio", "portfolio_id": portfolio_id}

    if alpaca is None:
        from app.data import clients as _clients
        alpaca = _clients.AlpacaClient()
    if llm_scorer is None:
        from app.ml.llm_scoring import LLMScorer
        llm_scorer = LLMScorer()

    tickers = [h.ticker for h in holdings]
    from datetime import datetime
    end = datetime.utcnow()
    start = end - timedelta(days=730)

    # Prices → returns
    try:
        prices_df = alpaca.get_ohlcv(tickers, start, end)
    except Exception as e:
        logger.warning(f"report: price fetch failed: {e}")
        prices_df = None
    returns_df = _returns_from_bars(prices_df)

    # Current weights from latest market value (fallback: cost basis).
    last_price = {}
    if returns_df is not None and not returns_df.empty and prices_df is not None:
        for t in tickers:
            sub = prices_df[prices_df["ticker"] == t]
            if not sub.empty:
                last_price[t] = float(sub.sort_values("date")["close"].iloc[-1])
    mv = {}
    for h in holdings:
        px = last_price.get(h.ticker) or (h.cost_basis or 0.0)
        mv[h.ticker] = (h.shares or 0.0) * px
    total_mv = sum(mv.values()) or 1.0
    weights_current = {t: mv[t] / total_mv for t in tickers}

    # Per-holding scores (cached LLM) + drift.
    scores_overall: dict[str, Optional[float]] = {}
    drift: dict[str, Optional[str]] = {}
    holding_rows = []
    sectors = {}
    for h in holdings:
        payload = score_one(db, h.ticker, alpaca=alpaca, llm_scorer=llm_scorer)
        ov = payload.get("overall_score")
        scores_overall[h.ticker] = ov
        dtrend = _lookup_drift(db, h.ticker)
        drift[h.ticker] = dtrend
        sectors[h.ticker] = getattr(h, "sector", None)
        holding_rows.append({
            "ticker": h.ticker,
            "company": payload.get("company_name"),
            "sector": getattr(h, "sector", None),
            "weight": weights_current.get(h.ticker),
            "overall_score": ov,
            "strategies": payload.get("strategies", {}),
            "drift_trend": dtrend,
            "llm": payload.get("llm", {}),
        })

    # Risk current.
    risk_current = risk_summary(returns_df, weights_current, sectors)

    # Proposed weights via optimizer (fallback to equal-weight on failure).
    weights_proposed = {}
    try:
        if optimizer.upper() == "HRP":
            weights_proposed = hrp_optimize(returns_df)
        else:
            weights_proposed = mvo_optimize(returns_df, target="max_sharpe")
    except Exception as e:
        logger.warning(f"report: optimizer {optimizer} failed ({e}) — equal weight")
    if not weights_proposed:
        weights_proposed = {t: 1.0 / len(tickers) for t in tickers}
    risk_proposed = risk_summary(returns_df, weights_proposed, sectors)

    actions = derive_actions(weights_current, weights_proposed, scores_overall, drift)
    watch_items = [t for t in tickers if drift.get(t) == "DETERIORATING"]

    regime = _latest_regime(db)

    data = {
        "portfolio_name": portfolio.name,
        "as_of": end.strftime("%Y-%m-%d"),
        "regime": regime,
        "holdings": holding_rows,
        "risk_current": risk_current,
        "risk_proposed": risk_proposed,
        "proposed_weights": weights_proposed,
        "optimizer": optimizer.upper(),
        "actions": actions,
        "watch_items": watch_items,
        "stress_test": {
            "series": None,  # per-portfolio COVID series not reconstructed; note only
            "note": ("Per the source paper, no strategy collapsed during the Feb-May 2020 "
                     "COVID crash; the technical track rebounded fastest. Framework-level "
                     "robustness evidence, not a portfolio-specific simulation."),
        },
    }
    data["narrative"] = generate_narrative(llm_scorer, data)
    return data


def _lookup_drift(db, ticker: str) -> Optional[str]:
    """Best-effort drift trend from the enrichment cache; None if unavailable."""
    try:
        from app.data.enrichment_cache import get_drift_cached
        cached = get_drift_cached(db, ticker)
        if isinstance(cached, dict):
            return cached.get("trend")
        if isinstance(cached, str) and cached:
            obj = json.loads(cached)
            return obj.get("trend")
    except Exception:
        pass
    return None


def _latest_regime(db) -> dict:
    try:
        from app import models
        row = (db.query(models.MarketRegime)
               .order_by(models.MarketRegime.created_at.desc()).first())
        if row:
            return {"label": row.regime_label, "confidence": row.regime_confidence}
    except Exception:
        pass
    return {}
