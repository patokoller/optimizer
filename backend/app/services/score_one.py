"""
On-demand single-ticker scoring.

Scores one arbitrary US-listed ticker WITHOUT a full universe run by reusing the
most recently persisted ModelBundle:

  1. Load the latest discovery bundle (fitted models + universe raw-ensemble
     reference distributions).
  2. Fetch the ticker's features + enrichment.
  3. For each strategy: predict the ticker's raw ensemble, then rank it INTO the
     bundle's universe distribution (scoring.percentile_into) → a
     cross-sectionally-correct ML percentile on the same scale as a full run.
  4. Score the LLM layer synchronously via two-stage (extract → score).
  5. Blend per strategy (combined_score) and assemble a detail payload.

The cross-sectional subtlety: an ad-hoc ticker has no universe of its own, so its
ML score is only meaningful relative to a reference. That reference is the saved
discovery universe — surfaced to the caller so the number is never context-free.

The pure `assemble_score_one` holds all the math and is unit-tested with fakes;
`score_one` is the thin live wrapper that does the I/O.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Optional

from app.ml.scoring import combined_score, percentile_into
from app.ml.model_bundle import load_latest_bundle, LoadedBundle

logger = logging.getLogger(__name__)

_STRATEGIES = ("fundamental", "technical", "entropy")


def _format_fundamentals_context(fundamentals_df, ticker: str, max_quarters: int = 4) -> str:
    """Turn the fetched quarterly fundamentals into a compact text block for the
    LLM prompt.

    The fundamentals were already fetched from Alpha Vantage (revenue, operating
    income, net income, margins) but were not being passed into the scoring
    prompt — so the LLM reported a 'complete information void' even for names
    whose financials we had. This converts them into the FINANCIAL STATEMENTS
    section the prompt expects. Returns '' when no usable data is present.
    """
    if fundamentals_df is None or getattr(fundamentals_df, "empty", True):
        return ""
    try:
        cols = {"period_date", "revenue", "operating_income", "net_income",
                "operating_margin", "net_margin"}
        if not cols.issubset(set(fundamentals_df.columns)):
            return ""
        df = fundamentals_df.sort_values("period_date").tail(max_quarters)
        if df.empty:
            return ""

        def _money(v) -> str:
            try:
                v = float(v)
            except (TypeError, ValueError):
                return "n/a"
            if abs(v) >= 1e9:
                return f"${v / 1e9:.2f}B"
            if abs(v) >= 1e6:
                return f"${v / 1e6:.1f}M"
            return f"${v:,.0f}"

        def _pct(v) -> str:
            try:
                return f"{float(v) * 100:.1f}%"
            except (TypeError, ValueError):
                return "n/a"

        lines = [f"Quarterly income-statement history for {ticker} "
                 f"(most recent {len(df)} quarters, source: Alpha Vantage):"]
        for _, r in df.iterrows():
            period = r["period_date"]
            period_s = period.strftime("%Y-%m") if hasattr(period, "strftime") else str(period)
            lines.append(
                f"  {period_s}: revenue {_money(r['revenue'])}, "
                f"operating income {_money(r['operating_income'])}, "
                f"net income {_money(r['net_income'])}, "
                f"operating margin {_pct(r['operating_margin'])}, "
                f"net margin {_pct(r['net_margin'])}"
            )
        # Trend note: compare oldest vs newest revenue when we have ≥2 quarters.
        if len(df) >= 2:
            first_rev = df.iloc[0]["revenue"]
            last_rev = df.iloc[-1]["revenue"]
            try:
                if first_rev and float(first_rev) != 0:
                    chg = (float(last_rev) - float(first_rev)) / abs(float(first_rev)) * 100
                    direction = "growth" if chg >= 0 else "decline"
                    lines.append(f"  Revenue {direction} over the window: {chg:+.1f}%.")
            except (TypeError, ValueError):
                pass
        return "\n".join(lines)
    except Exception as e:  # never let formatting break scoring
        logger.warning(f"_format_fundamentals_context failed for {ticker}: {e}")
        return ""


def assemble_score_one(
    bundle: LoadedBundle,
    *,
    ticker: str,
    raw_ensemble_by_strategy: dict[str, Optional[float]],
    llm_result: Optional[dict],
    data_availability: Optional[dict[str, bool]] = None,
) -> dict:
    """
    Pure assembly: given a ticker's raw ensemble per strategy and the LLM result,
    produce the full single-ticker score payload. No I/O.

    raw_ensemble_by_strategy: {strategy: raw_float_or_None}. None ⇒ that strategy
    couldn't be computed (missing data) and is reported as unavailable.
    """
    frequency = bundle.frequency or "monthly"
    llm_score = None
    llm_failed = True
    if isinstance(llm_result, dict) and llm_result.get("score") is not None:
        llm_score = float(llm_result["score"])
        llm_failed = False

    strategies_out: dict[str, Any] = {}
    combined_values: list[float] = []

    for strat in _STRATEGIES:
        raw = raw_ensemble_by_strategy.get(strat)
        if strat not in (bundle.strategies or []) or raw is None:
            strategies_out[strat] = {
                "available": False,
                "ml_percentile": None,
                "combined": None,
            }
            continue

        ml_pct = percentile_into(bundle.reference_raw(strat), raw)
        try:
            comb = combined_score(ml_pct, llm_score, strat, frequency, llm_failed)
        except ValueError as e:
            # Unknown strategy/frequency — should not happen for paper strategies.
            logger.error(f"combined_score failed for {strat}/{frequency}: {e}")
            strategies_out[strat] = {
                "available": False, "ml_percentile": ml_pct, "combined": None,
            }
            continue

        combined_values.append(comb)
        strategies_out[strat] = {
            "available": True,
            "ml_percentile": round(ml_pct, 4),
            "combined": round(comb, 4),
            "ml_raw": round(float(raw), 6),
        }

    overall = round(sum(combined_values) / len(combined_values), 4) if combined_values else None

    # LLM detail (derivation + rationale + fact sheet) for the detail view.
    llm_out: dict[str, Any] = {"available": not llm_failed, "score": llm_score}
    if isinstance(llm_result, dict):
        for k in ("band_base", "adjustments", "key_positives", "key_risks",
                  "confidence", "two_stage", "fact_sheet"):
            if k in llm_result:
                llm_out[k] = llm_result[k]

    return {
        "ticker": ticker.upper(),
        "as_of": bundle.rebalance_date.isoformat() if bundle.rebalance_date else None,
        "frequency": frequency,
        "comparison_universe": {
            "source_run": bundle.run_id,
            "size": len(bundle.universe or []),
            "label": "discovery universe (NASDAQ-100)",
        },
        "overall_score": overall,
        "strategies": strategies_out,
        "llm": llm_out,
        "data_availability": data_availability or {},
        "bundle_age": bundle.created_at.isoformat() if bundle.created_at else None,
    }


def _predict_raw(model, strat: str, ticker: str, prices_df, fundamentals_df, rebalance_date) -> Optional[float]:
    """Run a strategy model's predict for one ticker and pull its raw_ensemble."""
    try:
        if strat == "fundamental":
            out = model.predict([ticker], fundamentals_df)
        else:  # technical / entropy
            out = model.predict([ticker], prices_df, rebalance_date)
        rec = out.get(ticker) if isinstance(out, dict) else None
        if not isinstance(rec, dict):
            return None
        raw = rec.get("raw_ensemble")
        return float(raw) if raw is not None else None
    except Exception as e:
        logger.warning(f"score_one: {strat} predict failed for {ticker}: {e}")
        return None


def score_one(
    db,
    ticker: str,
    *,
    alpaca=None,
    av=None,
    edgar=None,
    llm_scorer=None,
    enrichment=None,
    bundle=None,
) -> dict:
    """
    Live single-ticker scoring. Dependencies are injectable for testing; in
    production they default to the real clients. A preloaded `bundle` may be
    passed to avoid reloading it on every call (e.g. when scoring a whole
    portfolio); when omitted it is loaded once here.

    Returns the assembled payload, or {"error": ...} when no bundle exists or the
    ticker yields no usable data.
    """
    ticker = ticker.strip().upper()

    if bundle is None:
        bundle = load_latest_bundle(db)
    if bundle is None:
        return {
            "error": "no_model_bundle",
            "message": "No trained models are available yet. Run a discovery job first.",
            "ticker": ticker,
        }

    # Lazy real-client defaults (kept out of import path for testability).
    if alpaca is None or av is None or edgar is None:
        from app.data import clients as _clients
        alpaca = alpaca or _clients.AlpacaClient()
        av = av or _clients.AlphaVantageClient()
        edgar = edgar or _clients.EDGARClient()
    if llm_scorer is None:
        from app.ml.llm_scoring import LLMScorer
        llm_scorer = LLMScorer()

    rebalance_date = bundle.rebalance_date
    frequency = bundle.frequency or "monthly"
    training_start = (rebalance_date - timedelta(days=730)) if rebalance_date else None

    # ── Fetch features (soft-failure: each gap degrades one strategy, not all) ──
    prices_df = fundamentals_df = None
    avail = {"prices": False, "fundamentals": False, "filings": False}
    try:
        prices_df = alpaca.get_ohlcv([ticker], training_start, rebalance_date)
        avail["prices"] = prices_df is not None and getattr(prices_df, "empty", True) is False
    except Exception as e:
        logger.warning(f"score_one: price fetch failed for {ticker}: {e}")
    try:
        fundamentals_df = av.get_fundamentals_batch([ticker])
        avail["fundamentals"] = fundamentals_df is not None and getattr(fundamentals_df, "empty", True) is False
    except Exception as e:
        logger.warning(f"score_one: fundamentals fetch failed for {ticker}: {e}")

    # ── ML: predict raw ensemble per strategy, rank into the saved universe ──
    raw_by_strategy: dict[str, Optional[float]] = {}
    for strat in _STRATEGIES:
        model = bundle.models.get(strat)
        if model is None:
            raw_by_strategy[strat] = None
            continue
        raw_by_strategy[strat] = _predict_raw(
            model, strat, ticker, prices_df, fundamentals_df, rebalance_date
        )

    # ── LLM: synchronous two-stage score ──
    llm_result = None
    try:
        company_name = ticker
        filing_ctx = ""
        try:
            filing_ctx = edgar.get_filing_context(ticker, rebalance_date) or ""
            avail["filings"] = bool(filing_ctx)
        except Exception as e:
            logger.warning(f"score_one: filing context failed for {ticker}: {e}")

        # Fundamentals context for the LLM, via a quarterly cache. If this run's
        # live fetch (used by the ML models above) already returned data, format
        # and cache it. If the live fetch came back empty (commonly AV's daily
        # rate-limit on a multi-ticker report run), fall back to the cached
        # context from an earlier run this quarter — so a transient throttle no
        # longer turns a real company like GOOGL into a "complete information void".
        from app.data import enrichment_cache as _ec
        if fundamentals_df is not None and getattr(fundamentals_df, "empty", True) is False:
            _fund_ctx = _format_fundamentals_context(fundamentals_df, ticker)
            if _fund_ctx:
                _ec.set_fundamentals_cached(db, ticker, _fund_ctx)
        else:
            # Live fetch yielded nothing this run; reuse a cached quarter value if present.
            _fund_ctx = _ec.get_fundamentals_cached(db, ticker) or ""
            if _fund_ctx == _ec.FUNDAMENTALS_EMPTY_SENTINEL:
                _fund_ctx = ""
        # Diagnostic: surface whether fundamentals actually reached the prompt.
        logger.info(
            f"score_one[{ticker}]: fundamentals_df rows="
            f"{0 if fundamentals_df is None else len(fundamentals_df)}, "
            f"income_statement_context chars={len(_fund_ctx)}, "
            f"filing_context chars={len(filing_ctx)}"
        )
        prompt = llm_scorer.build_prompt(
            ticker=ticker,
            company_name=company_name,
            frequency=frequency,
            period=rebalance_date.strftime("%Y-%m") if rebalance_date else "",
            filing_context=filing_ctx,
            income_statement_context=_fund_ctx,
        )
        from app.ml.llm_cache import score_sync_cached
        period = rebalance_date.strftime("%Y-%m") if rebalance_date else ""
        import hashlib as _hl
        logger.info(f"score_one[{ticker}]: prompt len={len(prompt)} "
                    f"hash={_hl.sha256(prompt.encode()).hexdigest()[:12]}")
        llm_result = score_sync_cached(db, llm_scorer, ticker, prompt, period)
    except Exception as e:
        logger.warning(f"score_one: LLM scoring failed for {ticker}: {e}")

    payload = assemble_score_one(
        bundle,
        ticker=ticker,
        raw_ensemble_by_strategy=raw_by_strategy,
        llm_result=llm_result,
        data_availability=avail,
    )

    if payload["overall_score"] is None and llm_result is None:
        payload["error"] = "no_usable_data"
        payload["message"] = (
            f"{ticker} could not be scored — no price/fundamental data matched the "
            f"trained models and the LLM layer was unavailable."
        )
    return payload
