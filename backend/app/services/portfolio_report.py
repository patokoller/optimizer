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


def compose_rationale(
    score: Optional[float],
    drift: Optional[str],
    action: str,
    excluded_reason: Optional[str] = None,
) -> str:
    """One-sentence, evidence-anchored rationale tying the action to the score.

    If ``excluded_reason`` is set, the holding could not be scored/optimized for a
    data reason (e.g. no price data for an unsupported symbol). In that case we say
    so plainly rather than attributing the outcome to an optimizer decision.
    """
    if excluded_reason:
        return excluded_reason
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
    excluded: Optional[dict[str, str]] = None,
) -> list[dict]:
    """Pure: turn current/proposed weights + scores + drift into ranked actions.

    ``excluded`` maps ticker -> human-readable reason for tickers that could not be
    priced/optimized (e.g. Alpaca-unsupported symbols). Such tickers are reported
    as EXCLUDED with the true reason, never as an optimizer EXIT.
    """
    excluded = excluded or {}
    actions = []
    for t in weights_current:
        if t in excluded:
            cw = weights_current.get(t, 0.0) or 0.0
            actions.append({
                "ticker": t,
                "action": "EXCLUDED",
                "delta": 0.0,
                "rationale": compose_rationale(
                    scores.get(t), drift.get(t), "EXCLUDED", excluded_reason=excluded[t]
                ),
            })
            continue
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
    # Most impactful first: EXIT/large moves at the top; EXCLUDED sorted last.
    order = {"EXIT": 0, "TRIM": 1, "ADD": 2, "HOLD": 3, "EXCLUDED": 4}
    actions.sort(key=lambda a: (order[a["action"]], -abs(a["delta"])))
    return actions


def fallback_narrative(data: dict) -> dict:
    """Deterministic narrative from the numbers — used when the LLM is unavailable
    so the report is never blank."""
    cur, prop = data.get("risk_current", {}), data.get("risk_proposed", {})
    watch = data.get("watch_items") or []
    n = data.get("n_holdings", len(data.get("holdings", [])))
    n_excl = data.get("n_excluded", 0)
    excl_clause = (f" {n_excl} holding(s) are excluded from optimization for missing "
                   f"price data and shown separately.") if n_excl else ""
    top_sec = next(iter(cur.get("sector_weights", {})), None)
    sector_clause = (f"This {n}-position portfolio's largest exposure is {top_sec}."
                     if data.get("sector_data_available") and top_sec
                     else f"This portfolio holds {n} positions.")
    exec_s = (
        f"{sector_clause}{excl_clause} "
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
        "closing": "Proposals are advisory and optimizer-derived; each holding's score is shown for context, and the decision rests with you.",
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
        # Drop sector_weights when there's no real sector data, so the model does
        # not write about a "100% Unknown sector" that is just a missing dimension.
        def _strip_sector(rd):
            if not isinstance(rd, dict):
                return rd
            if data.get("sector_data_available"):
                return rd
            return {k: v for k, v in rd.items() if k != "sector_weights"}
        ctx = {
            "portfolio": data.get("portfolio_name"),
            "position_counts": {
                "total": data.get("n_holdings"),
                "priced_and_optimized": data.get("n_priced"),
                "excluded_no_price_data": data.get("n_excluded"),
            },
            "risk_current": _strip_sector(data.get("risk_current")),
            "risk_proposed": _strip_sector(data.get("risk_proposed")),
            "watch_items": data.get("watch_items"),
            "actions": [{k: a[k] for k in ("ticker", "action", "delta")} for a in data.get("actions", [])],
            "holdings": [{"ticker": h["ticker"], "overall_score": h.get("overall_score"),
                          "drift": h.get("drift_trend")} for h in data.get("holdings", [])],
        }
        prompt = (
            "You are a portfolio analyst writing a client-grade memo from the JSON below. "
            "Write in a precise, neutral, advisory voice — propose, do not command; tie claims "
            "to the numbers. Do NOT invent figures not present. When stating how many positions "
            "the portfolio holds, use position_counts.total exactly; never count rows yourself. "
            + ("Language-drift data is unavailable this period — do NOT mention drift, "
               "management-tone trends, or 'drift scores'. "
               if not data.get("drift_data_available") else "")
            + "Return ONLY JSON with keys "
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


# ── Advisor's View (the differentiator) ──────────────────────────────────────
def fallback_advisor_view(data: dict) -> dict:
    """Deterministic Advisor's View when the LLM is unavailable. Opinionated but
    grounded purely in the computed numbers, so the panel is never empty."""
    cur, prop = data.get("risk_current", {}), data.get("risk_proposed", {})
    holdings = data.get("holdings", [])
    watch = data.get("watch_items") or []
    hhi = cur.get("hhi")
    sh_c, sh_p = cur.get("sharpe"), prop.get("sharpe")
    # rank holdings by weight to name the concentration
    top = sorted([h for h in holdings if h.get("weight")], key=lambda h: h["weight"], reverse=True)
    top3 = top[:3]
    top3_w = sum(h["weight"] for h in top3)
    concentrated = hhi is not None and hhi > 0.2

    conviction = "moderate"
    if concentrated and watch:
        conviction = "cautious"
    elif sh_p is not None and sh_c is not None and sh_p - sh_c > 0.15:
        conviction = "high"

    pieces = []
    if concentrated and top3:
        names = ", ".join(h["ticker"] for h in top3)
        pieces.append(f"the book leans heavily on a few names ({names} are roughly "
                      f"{top3_w*100:.0f}% of capital)")
    if watch:
        pieces.append(f"{', '.join(watch)} pair weaker scores with deteriorating call language")
    if sh_p is not None and sh_c is not None and sh_p > sh_c:
        pieces.append(f"the proposed weights lift the Sharpe ratio from {sh_c:.2f} to {sh_p:.2f}, "
                      f"mostly by lowering volatility rather than chasing return")
    stance = ("Reading the numbers as an advisor would: "
              + "; ".join(pieces) + ". "
              + ("The clearest action is to reduce concentration and address the watch-list names "
                 "before adding anywhere else." if concentrated or watch
                 else "The allocation looks reasonably balanced; changes here are refinements, not fixes."))

    key_points = []
    if concentrated and top3:
        key_points.append(f"Concentration is the dominant risk: the top {len(top3)} positions are "
                          f"~{top3_w*100:.0f}% of capital.")
    if watch:
        key_points.append(f"{', '.join(watch)} are the clearest candidates to trim or exit on "
                          f"weak scores and softening management tone.")
    if sh_p is not None and sh_c is not None:
        key_points.append("The proposed rebalance improves risk-adjusted return primarily by "
                          "cutting volatility — a higher-quality kind of gain.")
    if not key_points:
        key_points.append("No single risk dominates; treat the proposals as incremental tuning.")

    posture = ("Reduce the largest positions toward the proposed weights, act on the watch-list "
               "names, and hold the remainder pending fresh data."
               if concentrated or watch else
               "Maintain current positioning and revisit at the next scoring cycle.")

    # Bull vs bear — the strongest case each way, grounded in the data.
    movers = data.get("movers") or []
    best = movers[0] if movers else None
    worst = movers[-1] if movers and len(movers) > 1 else None
    scored = [h for h in holdings if h.get("overall_score") is not None]
    top_scored = max(scored, key=lambda h: h["overall_score"], default=None)

    bull = []
    if sh_p is not None and sh_c is not None and sh_p > sh_c:
        bull.append(f"The proposed reallocation lifts the Sharpe ratio to {sh_p:.2f} from {sh_c:.2f}.")
    if best and best.get("contribution", 0) > 0:
        bull.append(f"{best['ticker']} is carrying the book ({best['period_return']*100:+.1f}% last month).")
    if top_scored and top_scored["overall_score"] >= 0.6:
        bull.append(f"{top_scored['ticker']} anchors quality at the top of the score range "
                    f"({top_scored['overall_score']:.2f}).")
    for h in holdings:
        if (h.get("drift_trend") == "IMPROVING"):
            bull.append(f"{h['ticker']}'s management tone is improving across recent calls.")
            break
    if not bull:
        bull.append("Risk metrics are within normal ranges; no acute red flags in the current book.")

    bear = []
    if concentrated and top3:
        bear.append(f"Concentration is high — the top {len(top3)} names are ~{top3_w*100:.0f}% of capital.")
    if watch:
        bear.append(f"{', '.join(watch)} pair weak scores with deteriorating management tone.")
    if worst and worst.get("contribution", 0) < 0:
        bear.append(f"{worst['ticker']} detracted ({worst['period_return']*100:+.1f}%) and weighs on the book.")
    cur_vol = cur.get("annualized_vol")
    if cur_vol is not None and cur_vol > 0.25:
        bear.append(f"Volatility is elevated at {cur_vol*100:.0f}% annualized.")
    if not bear:
        bear.append("Scores are an unvalidated research signal; treat any single read with caution.")

    return {"stance": stance, "conviction": conviction,
            "bull_case": bull[:4], "bear_case": bear[:4],
            "key_points": key_points, "recommended_posture": posture}


def generate_advisor_view(llm_scorer, data: dict) -> dict:
    """Ask Claude to act as the advisor and give its own reasoned opinion — the
    differentiating feature. Opinionated but advisory; grounded in the data.
    Falls back to a deterministic view on any failure."""
    try:
        client = getattr(llm_scorer, "client", None)
        if client is None:
            return fallback_advisor_view(data)
        ctx = {
            "portfolio": data.get("portfolio_name"),
            "regime": data.get("regime"),
            "position_counts": {
                "total": data.get("n_holdings"),
                "priced_and_optimized": data.get("n_priced"),
                "excluded_no_price_data": data.get("n_excluded"),
            },
            "overall_posture_score": data.get("overall_posture_score"),
            "risk_current": data.get("risk_current"),
            "risk_proposed": data.get("risk_proposed"),
            "movers": [{"ticker": m["ticker"], "period_return": round(m["period_return"], 4)}
                       for m in (data.get("movers") or [])],
            "watch_items": data.get("watch_items"),
            "actions": [{k: a[k] for k in ("ticker", "action", "delta")} for a in data.get("actions", [])],
            "holdings": [{"ticker": h["ticker"], "weight": h.get("weight"),
                          "overall_score": h.get("overall_score"), "drift": h.get("drift_trend"),
                          "key_risks": (h.get("llm") or {}).get("key_risks", [])[:2]}
                         for h in data.get("holdings", [])],
        }
        prompt = (
            "You are a seasoned portfolio advisor. From the JSON below, give YOUR OWN opinion on "
            "this portfolio — not a neutral summary. Take a clear position on what matters and what "
            "you would do, in a confident but advisory voice (you propose; the client decides). "
            "Ground every claim in the data; invent no figures. When stating how many positions the "
            "portfolio holds, use position_counts.total exactly; never count rows yourself. "
            + ("Language-drift data is unavailable this period — do NOT mention drift or "
               "management-tone trends as inputs. "
               if not data.get("drift_data_available") else "")
            + "Scores are an unvalidated research "
            "signal, so lean on concentration, risk, drift and the proposed changes as much as on "
            "scores. Also argue both sides like a research desk: a bull_case (what would have to go "
            "right / the strongest reasons to stay constructive) and a bear_case (the strongest "
            "reasons for caution) — each a list of 2-4 short, data-grounded sentences. Return ONLY "
            "JSON with keys: stance (one opinionated paragraph, 4-6 sentences), conviction (one of: "
            "high, moderate, low, cautious), bull_case (list), bear_case (list), key_points (list of "
            "2-4 short sentences), recommended_posture (one sentence stating what you would actually "
            "do).\n\n"
            f"{json.dumps(ctx, default=str)}"
        )
        resp = client.messages.create(
            model=llm_scorer.model, max_tokens=1100, temperature=0.4,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        required = ("stance", "conviction", "key_points", "recommended_posture", "bull_case", "bear_case")
        if all(k in parsed for k in required):
            if isinstance(parsed["key_points"], list) and parsed["stance"] \
               and isinstance(parsed["bull_case"], list) and isinstance(parsed["bear_case"], list):
                return parsed
        return fallback_advisor_view(data)
    except Exception as e:
        logger.warning(f"Advisor view generation failed ({e}) — using deterministic fallback")
        return fallback_advisor_view(data)


# ── Review & outlook (Julius-Baer-style) ─────────────────────────────────────
def _holding_lookup(data: dict) -> dict:
    return {h["ticker"]: h for h in data.get("holdings", [])}


def fallback_review_outlook(data: dict) -> dict:
    """Deterministic 'Key developments' + 'Future positioning' prose, grounded in
    computed movers, drift, and each holding's already-extracted semantic signals
    (key_positives / key_risks). Never invents macro facts."""
    movers = data.get("movers") or []
    hl = _holding_lookup(data)
    regime = (data.get("regime") or {}).get("label")
    watch = data.get("watch_items") or []
    actions = data.get("actions", [])

    # Key developments: best/worst contributor + a drift note + one company outlook signal.
    dev = []
    if movers:
        best = movers[0]
        worst = movers[-1]
        if best["contribution"] > 0:
            dev.append(f"{best['ticker']} was the largest positive contributor over the trailing "
                       f"month ({best['period_return']*100:+.1f}%)")
        if worst["contribution"] < 0 and worst["ticker"] != best["ticker"]:
            dev.append(f"{worst['ticker']} was the main detractor ({worst['period_return']*100:+.1f}%)")
    # company outlook / supply-chain signal from the LLM's reading, for a named mover
    outlook_bits = []
    for m in (movers[:1] + movers[-1:]):
        h = hl.get(m["ticker"], {})
        llm = h.get("llm") or {}
        sig = (llm.get("key_positives") or llm.get("key_risks") or [None])[0]
        if sig:
            outlook_bits.append(f"{m['ticker']}: {sig.lower()}")
    drift_note = ""
    if watch:
        drift_note = (f" Management language at {', '.join(watch)} continued to soften across "
                      f"recent calls.")
    key_developments = (
        ((". ".join(dev) + ".") if dev else "No single position dominated the month's move.")
        + (f" On the company outlook, {('; '.join(outlook_bits))}." if outlook_bits else "")
        + drift_note
    )

    # Future positioning: regime + proposed posture synthesis (model-derived).
    n_exit = sum(1 for a in actions if a["action"] == "EXIT")
    n_trim = sum(1 for a in actions if a["action"] == "TRIM")
    n_add = sum(1 for a in actions if a["action"] == "ADD")
    moves = []
    if n_trim or n_exit:
        moves.append(f"reduce {n_trim + n_exit} position(s)")
    if n_add:
        moves.append(f"add to {n_add}")
    posture = (" and ".join(moves)) if moves else "hold current positioning"
    regime_clause = (f"With the regime read as {regime.lower()}, " if regime else "")
    macro = data.get("macro") or {}
    macro_bits = []
    if macro.get("fed_funds") is not None:
        macro_bits.append(f"fed funds at {macro['fed_funds']:.2f}%")
    if macro.get("cpi_yoy") is not None:
        macro_bits.append(f"CPI running {macro['cpi_yoy']:.1f}% YoY")
    if macro.get("yield_curve") is not None:
        c = macro["yield_curve"]
        macro_bits.append(f"the 10Y-2Y curve at {c:+.2f}{' (inverted)' if c < 0 else ''}")
    macro_clause = (f" Against a backdrop of {', '.join(macro_bits)}, " if macro_bits else " ")
    future_positioning = (
        f"{regime_clause}the model leans toward {'a more defensive tilt' if (n_trim + n_exit) > n_add else 'maintaining risk'}: "
        f"the proposal would {posture}, concentrating weight in the higher-scored, lower-volatility "
        f"names." + macro_clause +
        f"this is a model-derived stance from the scores, drift, and risk inputs above — not a "
        f"market forecast — and is offered for the reader to weigh against their own view."
    )
    return {"key_developments": key_developments, "future_positioning": future_positioning}


def generate_review_outlook(llm_scorer, data: dict) -> dict:
    """Ask Claude for concise 'Key developments last month' and 'Future positioning'
    prose (Julius-Baer style). Company outlook / supply-chain context must come from
    each holding's provided key_positives / key_risks — not invented. Falls back to a
    deterministic version on any failure."""
    try:
        client = getattr(llm_scorer, "client", None)
        if client is None:
            return fallback_review_outlook(data)
        hl = _holding_lookup(data)
        movers = data.get("movers") or []
        ctx = {
            "regime": data.get("regime"),
            "macro": data.get("macro"),
            "movers": [{"ticker": m["ticker"], "period_return": round(m["period_return"], 4),
                        "contribution": round(m["contribution"], 4)} for m in movers],
            "holdings_semantics": [
                {"ticker": t, "drift": (h.get("drift_trend")),
                 "key_positives": (h.get("llm") or {}).get("key_positives", [])[:3],
                 "key_risks": (h.get("llm") or {}).get("key_risks", [])[:3]}
                for t, h in hl.items()
            ],
            "actions": [{k: a[k] for k in ("ticker", "action", "delta")} for a in data.get("actions", [])],
            "watch_items": data.get("watch_items"),
        }
        _drift_ok = data.get("drift_data_available")
        _tone_clause = ("plus notable changes in management tone (use 'drift'). "
                        if _drift_ok else "")
        _derived_clause = ("model-derived from scores/drift/risk" if _drift_ok
                           else "model-derived from scores and risk")
        prompt = (
            "You are writing two concise sections of a portfolio report, in the style of a private "
            "bank factsheet. Be specific and name holdings, but keep each section to 4-6 sentences.\n\n"
            "1) key_developments: what actually moved over the trailing month (use the 'movers' data — "
            "name the biggest contributor and detractor with their returns), "
            + _tone_clause +
            "For the named movers, add a short forward-looking note on "
            "the company's outlook and supply-chain/demand context — but ONLY using each holding's "
            "provided key_positives / key_risks. Do NOT invent macro or company facts not in that data.\n"
            "2) future_positioning: synthesize the regime and the proposed actions into a forward stance. "
            "If 'macro' figures are present (fed funds, CPI, 10Y-2Y curve, VIX — all real, sourced data), "
            "reference the relevant ones factually to frame the backdrop; do not invent any macro number "
            "not given. Explicitly frame the positioning as " + _derived_clause + ", not a "
            "market forecast.\n\n"
            + ("" if _drift_ok else
               "Language-drift data is unavailable this period — do NOT mention drift or "
               "management-tone trends anywhere.\n")
            + "Return ONLY JSON with keys key_developments and future_positioning (strings).\n\n"
            f"{json.dumps(ctx, default=str)}"
        )
        resp = client.messages.create(
            model=llm_scorer.model, max_tokens=900, temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        if parsed.get("key_developments") and parsed.get("future_positioning"):
            return {"key_developments": parsed["key_developments"],
                    "future_positioning": parsed["future_positioning"]}
        return fallback_review_outlook(data)
    except Exception as e:
        logger.warning(f"Review/outlook generation failed ({e}) — using deterministic fallback")
        return fallback_review_outlook(data)


def _returns_from_bars(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot OHLCV long frame (date,ticker,close) → wide daily returns."""
    if prices_df is None or prices_df.empty:
        return pd.DataFrame()
    df = prices_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="ticker", values="close").sort_index()
    return wide.pct_change(fill_method=None).dropna(how="all")


def _bundle_refresh_running(db) -> bool:
    """True if a discovery/training run is currently pending or running (so a
    score-less report can tell the user models are being prepared)."""
    try:
        from app.services.bundle_maintenance import bundle_status
        return bool(bundle_status(db).get("refresh_in_progress"))
    except Exception:
        return False


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

    # Identify holdings that could not be priced, so they are reported honestly
    # (as data exclusions) rather than mislabeled as optimizer EXIT decisions.
    # Two causes: symbols Alpaca's feed cannot handle (e.g. BRK.B), and any other
    # ticker that simply came back with no bars.
    unsupported = getattr(type(alpaca), "_ALPACA_UNSUPPORTED", set())
    priced_tickers = (
        set(prices_df["ticker"].unique()) if prices_df is not None and not prices_df.empty else set()
    )
    excluded: dict[str, str] = {}
    for t in tickers:
        if t in unsupported:
            excluded[t] = (
                "No price data: symbol unsupported by the price feed (e.g. share-class "
                "tickers like BRK.B). Excluded from scoring and optimization."
            )
        elif t not in priced_tickers:
            excluded[t] = (
                "No price data returned for this symbol over the lookback window. "
                "Excluded from scoring and optimization."
            )
    if excluded:
        logger.info(
            f"report: {len(excluded)} holding(s) excluded for missing price data: "
            f"{sorted(excluded)}"
        )

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

    # Per-holding scores (cached LLM) + drift. Load the model bundle once and
    # reuse it across holdings (avoids a DB round-trip per ticker).
    from app.ml.model_bundle import load_latest_bundle
    bundle = load_latest_bundle(db)
    scores_overall: dict[str, Optional[float]] = {}
    drift: dict[str, Optional[str]] = {}
    holding_rows = []
    sectors = {}
    for h in holdings:
        # When no bundle exists yet, every score_one call would just return
        # "no_model_bundle" — skip them (avoids N redundant DB loads + log spam).
        payload = score_one(db, h.ticker, alpaca=alpaca, llm_scorer=llm_scorer,
                            bundle=bundle) if bundle is not None else {}
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

    actions = derive_actions(weights_current, weights_proposed, scores_overall, drift, excluded=excluded)
    watch_items = [t for t in tickers if drift.get(t) == "DETERIORATING"]

    regime = _latest_regime(db)
    macro = _latest_macro(db)

    overall_scores = [s for s in scores_overall.values() if s is not None]
    overall_posture = sum(overall_scores) / len(overall_scores) if overall_scores else None

    from app.ml.portfolio_risk import monthly_movers
    movers = monthly_movers(returns_df, weights_current, window=21)

    # Explicit, consistent position counts so the narrative never disagrees with
    # the holdings table. total = every holding the user uploaded; priced = those
    # the optimizer could act on; excluded = those with no price data.
    n_total = len(holding_rows)
    n_excluded = len(excluded)
    n_priced = n_total - n_excluded

    # Whether any holding has a real sector. Uploads carry only ticker/shares/cost,
    # so sector is typically absent — flag it so the renderer can suppress a
    # misleading "100% Unknown" sector chart rather than displaying a dead axis.
    sector_data_available = any(
        (s and str(s).strip().lower() not in ("", "unknown", "none"))
        for s in sectors.values()
    )

    # Whether any holding has a computed language-drift trend. When the whole
    # column is empty (the common case — drift needs multi-quarter enrichment
    # history), the narrative must not claim it "derived from drift scores".
    drift_data_available = any(
        (d and str(d).strip().upper() in ("IMPROVING", "DETERIORATING", "STABLE"))
        for d in drift.values()
    )

    data = {
        "portfolio_name": portfolio.name,
        "as_of": end.strftime("%Y-%m-%d"),
        "regime": regime,
        "macro": macro,
        "overall_posture_score": overall_posture,
        "movers": movers,
        "scores_available": bundle is not None,
        "scores_status": ("ready" if bundle is not None else
                          ("training" if _bundle_refresh_running(db) else "absent")),
        "holdings": holding_rows,
        "n_holdings": n_total,
        "n_priced": n_priced,
        "n_excluded": n_excluded,
        "sector_data_available": sector_data_available,
        "drift_data_available": drift_data_available,
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
    data["advisor_view"] = generate_advisor_view(llm_scorer, data)
    data["review"] = generate_review_outlook(llm_scorer, data)
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
               .order_by(models.MarketRegime.computed_at.desc()).first())
        if row:
            return {"label": row.regime_label, "confidence": row.regime_confidence}
    except Exception:
        pass
    return {}


def _label_description(label: Optional[str]) -> Optional[str]:
    """Map a regime label back to its description from the regime catalogue."""
    if not label:
        return None
    try:
        from app.ml.regime import REGIMES
        for r in REGIMES.values():
            if r.get("label") == label:
                return r.get("description")
    except Exception:
        pass
    return None


def _latest_macro(db) -> dict:
    """Real, sourced macro snapshot from the most recent regime computation
    (FRED / Alpha Vantage). Empty dict if none has been computed yet."""
    try:
        from app import models
        row = (db.query(models.MarketRegime)
               .order_by(models.MarketRegime.computed_at.desc()).first())
        if not row:
            return {}
        return {
            "regime_label": row.regime_label,
            "regime_description": _label_description(row.regime_label),
            "dominant_factor": row.dominant_factor,
            "transition_risk": row.transition_risk,
            "vix": row.vix,
            "yield_curve": row.yield_curve_10y2y,
            "fed_funds": row.fed_funds_rate,
            "cpi_yoy": row.cpi_yoy,
            "as_of": row.computed_at.strftime("%Y-%m-%d") if row.computed_at else None,
            "source": "FRED / Alpha Vantage",
        }
    except Exception as e:
        logger.warning(f"macro snapshot lookup failed: {e}")
        return {}
