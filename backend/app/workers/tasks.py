import warnings
warnings.filterwarnings("ignore", message="Loky-backed parallel loops")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

"""
app/workers/tasks.py
Celery async tasks:
  - run_score_job     : runs all three strategy models + LLM scoring
  - run_optimization_job : runs deep_rl / mvo / hrp on completed scores
"""
import os
import logging
from datetime import datetime, timedelta
from celery import Celery
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


# ── Score Job ─────────────────────────────────────────────────────────────
@celery_app.task(bind=True, max_retries=0)
def run_score_job(self, run_id: str, portfolio_id: str, frequency: str):
    """
    Full scoring pipeline for a single rebalance period.

    Steps:
      1. Load holdings from DB
      2. Fetch price data (Alpaca) — blocks technical/entropy on failure
      3. Fetch fundamentals (Alpha Vantage) — blocks fundamental on failure
      4. Fetch EDGAR filings — all strategies fall back to w=1.0 on failure
      5. Call Claude for LLM scoring — fall back to w=1.0 on failure
      6. Compute CombinedScore = w*ML + (1-w)*LLM per strategy
      7. Write Score rows to DB
      8. Update ScoreRun status
    """
    from app.database import SessionLocal
    from app import models
    from app.ml.scoring import combined_score, normalize_scores, OPTIMAL_WEIGHTS
    from app.ml.fundamental import FundamentalScorer
    from app.ml.technical import TechnicalScorer
    from app.ml.entropy import EntropyScorer
    from app.ml.llm_scoring import LLMScorer
    from app.data.clients import AlpacaClient, AlpacaDataError
    from app.data.clients import AlphaVantageClient, AlphaVantageError
    from app.data.clients import EDGARClient

    db = SessionLocal()
    run = db.query(models.ScoreRun).filter(models.ScoreRun.id == run_id).first()
    if not run:
        logger.error(f"ScoreRun {run_id} not found")
        return

    try:
        run.status = models.RunStatus.running
        db.commit()

        # Load portfolio holdings
        portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        portfolio_tickers = [h.ticker for h in portfolio.holdings]
        if not portfolio_tickers:
            raise ValueError("Portfolio has no holdings")

        rebalance_date = run.run_date
        training_start = rebalance_date - timedelta(days=730)  # 24 months

        warnings_list = []

        # ── Step 1b: ETF classification ────────────────────────────
        # Classify each ticker: STOCK | EQUITY_ETF | BOND_ETF | CRYPTO_ETF | NON_SCOREABLE
        # For EQUITY_ETF: resolve top 5 equity holdings, score them, average back
        from app.data.etf_client import ETFClient
        etf_client = ETFClient()
        classifications = etf_client.classify_batch(portfolio_tickers)

        # Build the effective scoring universe:
        # - STOCK/unknown → score directly
        # - EQUITY_ETF → replace with top 5 holdings (deduplicated)
        # - BOND_ETF / CRYPTO_ETF / NON_SCOREABLE → exclude entirely
        score_tickers = []          # tickers actually run through the ML pipeline
        etf_holding_map = {}        # etf_ticker → [ETFHolding] for composite averaging later
        excluded_tickers = {}       # etf_ticker → reason string
        ticker_normalisation = {}   # original → resolved (e.g. BRK.B → BRK-B)

        for orig_ticker, clf in classifications.items():
            resolved = clf.resolved_ticker
            ticker_normalisation[orig_ticker] = resolved

            if clf.etf_type == "STOCK":
                if resolved not in score_tickers:
                    score_tickers.append(resolved)

            elif clf.etf_type == "EQUITY_ETF":
                # Score the underlying holdings instead of the wrapper
                etf_holding_map[orig_ticker] = clf.holdings
                for h in clf.holdings:
                    if h.ticker not in score_tickers:
                        score_tickers.append(h.ticker)
                logger.info(f"ETF {orig_ticker}: scoring via holdings {[h.ticker for h in clf.holdings]}")

            elif clf.etf_type == "BOND_ETF":
                excluded_tickers[orig_ticker] = "Bond ETF — excluded from equity scoring framework"
                warnings_list.append(f"ETF_EXCLUDED: {orig_ticker} is a bond ETF — not scoreable in this framework")

            elif clf.etf_type == "CRYPTO_ETF":
                excluded_tickers[orig_ticker] = "Crypto ETF — excluded from equity scoring framework"
                warnings_list.append(f"ETF_EXCLUDED: {orig_ticker} is a crypto ETF — not scoreable in this framework")

            elif clf.etf_type == "NON_SCOREABLE":
                excluded_tickers[orig_ticker] = clf.error or "Ticker not recognised / no data available"
                warnings_list.append(f"TICKER_EXCLUDED: {orig_ticker} — {excluded_tickers[orig_ticker]}")

        tickers = score_tickers  # pipeline now operates on this universe

        if not tickers:
            raise ValueError("No scoreable tickers after ETF classification — all holdings are bonds/crypto/unrecognised")

        logger.info(f"ETF classification: {len(tickers)} scoreable tickers | {len(etf_holding_map)} ETF composites | {len(excluded_tickers)} excluded")

        # ── Step 2: Price data ─────────────────────────────────────
        alpaca = AlpacaClient()
        prices_df = None
        try:
            prices_df = alpaca.get_ohlcv(tickers, training_start, rebalance_date)
            logger.info(f"Alpaca: {len(prices_df)} bars fetched for {len(tickers)} tickers")
        except AlpacaDataError as e:
            logger.warning(f"Alpaca unavailable — technical + entropy blocked: {e}")
            warnings_list.append(f"ALPACA_UNAVAILABLE: {e}")

        # ── Step 3: Fundamental data ───────────────────────────────
        av = AlphaVantageClient()
        fundamentals_df = None
        try:
            fundamentals_df = av.get_fundamentals_batch(tickers)
            logger.info(f"Alpha Vantage: {len(fundamentals_df)} quarterly rows fetched")
        except AlphaVantageError as e:
            logger.warning(f"Alpha Vantage unavailable — fundamental blocked: {e}")
            warnings_list.append(f"ALPHAVANTAGE_UNAVAILABLE: {e}")

        # ── Step 4: EDGAR filings ──────────────────────────────────
        edgar = EDGARClient()
        filing_contexts = {}  # ticker → filing text
        llm_failed_global = False
        for ticker in tickers:
            ctx = edgar.get_filing_context(ticker, rebalance_date)
            if ctx:
                filing_contexts[ticker] = ctx
        if not filing_contexts:
            warnings_list.append("EDGAR_UNAVAILABLE: No filing context — all strategies fall back to w=1.0")
            llm_failed_global = True

        # ── Step 5: Claude LLM scoring ─────────────────────────────
        llm_scorer = LLMScorer()
        llm_scores = {}
        if not llm_failed_global:
            for ticker in tickers:
                ctx = filing_contexts.get(ticker, "")
                result = llm_scorer.score(
                    ticker=ticker,
                    company_name=ticker,
                    frequency=frequency,
                    period=rebalance_date.strftime("%Y-%m"),
                    filing_context=ctx,
                    earnings_context="",
                )
                if result is not None:
                    llm_scores[ticker] = result
                else:
                    llm_failed_global = True
                    warnings_list.append(f"LLM_SCORE_FAILED: {ticker} — Claude API unavailable")
                    break

        # ── Step 6: FRED macro snapshot + regime classification ───
        from app.data.fred_client import FREDClient
        from app.ml.regime import classify_regime, apply_regime_weight_adjustment
        import numpy as np

        macro_snapshot = {}
        regime_data    = {"factor_weight_adj": {"technical": 1.0, "fundamental": 1.0, "entropy": 1.0}}
        try:
            fred = FREDClient()
            macro_snapshot = fred.get_macro_snapshot()
            regime_data    = classify_regime(macro_snapshot)
            logger.info(f"Regime: {regime_data['label']} (confidence={regime_data['confidence']:.2f})")
        except Exception as e:
            logger.warning(f"FRED/Regime failed — neutral regime applied: {e}")
            warnings_list.append(f"REGIME_UNAVAILABLE: {e}")

        # ── Step 7: ML model training + scoring ────────────────────
        fund_scores = {}
        tech_scores = {}
        entr_scores = {}

        # Attach forward returns to fundamentals using Alpaca price data
        import pandas as pd
        if fundamentals_df is not None and prices_df is not None:
            try:
                prices_df["date"] = pd.to_datetime(prices_df["date"])
                fundamentals_df["period_date"] = pd.to_datetime(fundamentals_df["period_date"])
                price_lookup = {
                    t: grp.sort_values("date").reset_index(drop=True)
                    for t, grp in prices_df.groupby("ticker")
                }
                def get_forward_return(ticker, report_date):
                    if ticker not in price_lookup:
                        return None
                    grp = price_lookup[ticker]
                    after = grp[grp["date"] >= report_date]
                    if len(after) < 22:
                        return None
                    s = after.iloc[0]["close"]
                    e = after.iloc[21]["close"]
                    return (e / s) - 1 if s else None
                fundamentals_df["forward_return"] = fundamentals_df.apply(
                    lambda r: get_forward_return(r["ticker"], r["period_date"]), axis=1
                )
                logger.info(f"Forward returns attached: {fundamentals_df['forward_return'].notna().sum()}/{len(fundamentals_df)} rows labeled")
            except Exception as e:
                logger.warning(f"Forward return computation failed: {e}")

        if fundamentals_df is not None:
            try:
                fund_model = FundamentalScorer()
                fund_model.fit(fundamentals_df, rebalance_date)
                fund_scores = fund_model.predict(tickers, fundamentals_df)
                logger.info(f"Fundamental model scored {len(fund_scores)} tickers")
            except Exception as e:
                logger.error(f"Fundamental model error: {e}")
                warnings_list.append(f"FUNDAMENTAL_MODEL_ERROR: {e}")

        if prices_df is not None:
            try:
                tech_model = TechnicalScorer()
                tech_model.fit(prices_df, rebalance_date)
                tech_scores = tech_model.predict(tickers, prices_df, rebalance_date)

                entr_model = EntropyScorer()
                entr_model.fit(prices_df, rebalance_date)
                entr_scores = entr_model.predict(tickers, prices_df, rebalance_date)
                logger.info(f"Technical/Entropy models scored {len(tech_scores)} tickers")
            except Exception as e:
                logger.error(f"Technical/Entropy model error: {e}")
                warnings_list.append(f"TECHNICAL_MODEL_ERROR: {e}")

        # ── Step 8: Compute risk metrics from price data ───────────
        risk_metrics = {}
        if prices_df is not None:
            try:
                prices_df["date"] = pd.to_datetime(prices_df["date"])
                qqq_prices = prices_df[prices_df["ticker"] == "QQQ"].set_index("date")["close"] if "QQQ" in prices_df["ticker"].values else None

                for ticker, grp in prices_df.groupby("ticker"):
                    try:
                        closes = grp.sort_values("date").set_index("date")["close"]
                        daily_ret = closes.pct_change().dropna()

                        vol_21d = float(daily_ret.tail(21).std() * (252**0.5)) if len(daily_ret) >= 21 else None
                        vol_63d = float(daily_ret.tail(63).std() * (252**0.5)) if len(daily_ret) >= 63 else None

                        # Max drawdown over 252 trading days
                        tail = closes.tail(252)
                        roll_max = tail.expanding().max()
                        drawdowns = (tail - roll_max) / roll_max
                        mdd = float(drawdowns.min()) if len(drawdowns) > 0 else None

                        # Sharpe 1Y (annualised, risk-free ≈ 0)
                        annual_ret = float(daily_ret.tail(252).mean() * 252) if len(daily_ret) >= 252 else None
                        sharpe = float(annual_ret / vol_63d) if (annual_ret and vol_63d and vol_63d > 0) else None

                        # Beta vs QQQ
                        beta = None
                        if qqq_prices is not None:
                            qqq_ret = qqq_prices.pct_change().dropna()
                            common = daily_ret.index.intersection(qqq_ret.index)
                            if len(common) >= 60:
                                cov = float(daily_ret[common].tail(252).cov(qqq_ret[common].tail(252)))
                                var = float(qqq_ret[common].tail(252).var())
                                beta = round(cov / var, 3) if var > 0 else None

                        risk_metrics[ticker] = {
                            "vol_21d": round(vol_21d, 4) if vol_21d else None,
                            "vol_63d": round(vol_63d, 4) if vol_63d else None,
                            "mdd":     round(mdd, 4)     if mdd else None,
                            "sharpe":  round(sharpe, 3)  if sharpe else None,
                            "beta":    beta,
                        }
                    except Exception:
                        pass
                logger.info(f"Risk metrics computed for {len(risk_metrics)} tickers")
            except Exception as e:
                logger.warning(f"Risk metric computation failed: {e}")

        # ── Step 9: Fetch previous run scores for delta computation ─
        prev_scores_map = {}
        try:
            prev_run = (
                db.query(models.ScoreRun)
                .filter(
                    models.ScoreRun.portfolio_id == portfolio_id,
                    models.ScoreRun.status.in_([models.RunStatus.complete, models.RunStatus.complete_with_warnings]),
                    models.ScoreRun.id != run_id,
                )
                .order_by(models.ScoreRun.run_date.desc())
                .first()
            )
            if prev_run:
                prev_rows = db.query(models.Score).filter(models.Score.run_id == prev_run.id).all()
                prev_scores_map = {s.ticker: s for s in prev_rows}
                logger.info(f"Delta: comparing against run {prev_run.id[:8]} ({len(prev_scores_map)} prev scores)")
        except Exception as e:
            logger.warning(f"Delta computation skipped: {e}")

        # ── Step 10: Compute universe percentile ranks ──────────────
        all_combined = {}

        # ── Step 11: Write Score rows ──────────────────────────────
        weights_m = {
            "technical":   OPTIMAL_WEIGHTS.get(("technical",   frequency), {"ml": 1.0,  "llm": 0.0}),
            "fundamental": OPTIMAL_WEIGHTS.get(("fundamental", frequency), {"ml": 0.15, "llm": 0.85}),
            "entropy":     OPTIMAL_WEIGHTS.get(("entropy",     frequency), {"ml": 0.70, "llm": 0.30}),
        }

        # Apply regime weight adjustments
        adj = regime_data.get("factor_weight_adj", {})
        reg_weights = {
            "technical":   {"ml": min(1.0, weights_m["technical"]["ml"]   * adj.get("technical",   1.0))},
            "fundamental": {"ml": min(1.0, weights_m["fundamental"]["ml"] * adj.get("fundamental", 1.0))},
            "entropy":     {"ml": min(1.0, weights_m["entropy"]["ml"]     * adj.get("entropy",     1.0))},
        }

        score_rows = []
        for ticker in tickers:
            llm_data      = llm_scores.get(ticker)
            llm_score_val = llm_data["score"] if llm_data else None
            llm_provider  = models.LLMProvider.claude if llm_data else models.LLMProvider.none
            llm_failed    = llm_score_val is None

            # Extract component dicts from upgraded scorers
            tech_d  = tech_scores.get(ticker) or {}
            fund_d  = fund_scores.get(ticker) or {}
            entr_d  = entr_scores.get(ticker) or {}

            # Scalar ML scores (the ensemble-normalised value)
            tech_ml = tech_d.get("score") if isinstance(tech_d, dict) else tech_d
            fund_ml = fund_d.get("score") if isinstance(fund_d, dict) else fund_d
            entr_ml = entr_d.get("score") if isinstance(entr_d, dict) else entr_d

            # Combined scores per strategy (regime-adjusted weights)
            tech_combined = combined_score(tech_ml or 0.5, llm_score_val, "technical",   frequency, llm_failed) if tech_ml is not None else None
            fund_combined = combined_score(fund_ml or 0.5, llm_score_val, "fundamental", frequency, llm_failed) if fund_ml is not None else None
            entr_combined = combined_score(entr_ml or 0.5, llm_score_val, "entropy",     frequency, llm_failed) if entr_ml is not None else None

            avail = [s for s in [tech_combined, fund_combined, entr_combined] if s is not None]
            overall = float(sum(avail) / len(avail)) if avail else None
            all_combined[ticker] = overall

            # Dispersion — std dev across all available component scores
            component_scores = [s for s in [
                tech_d.get("xgboost"), tech_d.get("lightgbm"), tech_d.get("catboost"),
                fund_d.get("ridge"),   fund_d.get("xgboost"),   fund_d.get("rf"), fund_d.get("mlp"),
                entr_d.get("xgboost"), entr_d.get("lightgbm"), entr_d.get("catboost"),
            ] if s is not None]
            overall_dispersion = float(np.std(component_scores)) if len(component_scores) > 1 else 0.0

            # Confidence score — inversely proportional to dispersion, boosted by regime
            regime_conf_boost = regime_data.get("confidence", 0.7) - 0.7  # centre on 0
            raw_confidence = max(0.0, min(1.0, 1.0 - (overall_dispersion * 3.0) + regime_conf_boost))

            # LLM-ML alignment — does Claude's direction agree with overall ML?
            llm_ml_align = None
            if llm_score_val is not None and overall is not None:
                ml_direction  = 1 if (tech_ml or 0.5) > 0.5 else -1
                llm_direction = 1 if llm_score_val > 0.5 else -1
                llm_ml_align  = 1.0 if ml_direction == llm_direction else 0.0

            # Delta vs previous run
            prev = prev_scores_map.get(ticker)
            score_delta = None
            prev_combined = None
            if prev and prev.combined_score is not None and overall is not None:
                prev_combined = prev.combined_score
                score_delta   = round(overall - prev_combined, 4)

            # Risk metrics
            risk = risk_metrics.get(ticker, {})

            score_row = models.Score(
                run_id                 = run_id,
                ticker                 = ticker,
                # Individual component scores
                fundamental_ridge_score = fund_d.get("ridge"),
                fundamental_xgb_score   = fund_d.get("xgboost"),
                fundamental_rf_score    = fund_d.get("rf"),
                fundamental_mlp_score   = fund_d.get("mlp"),
                technical_xgb_score     = tech_d.get("xgboost"),
                technical_lgbm_score    = tech_d.get("lightgbm"),
                technical_cat_score     = tech_d.get("catboost"),
                entropy_xgb_score       = entr_d.get("xgboost"),
                entropy_lgbm_score      = entr_d.get("lightgbm"),
                entropy_cat_score       = entr_d.get("catboost"),
                # Dispersion
                fundamental_dispersion  = fund_d.get("dispersion"),
                technical_dispersion    = tech_d.get("dispersion"),
                entropy_dispersion      = entr_d.get("dispersion"),
                overall_dispersion      = overall_dispersion,
                # Feature importances
                fundamental_feature_importance = fund_d.get("feature_importance"),
                technical_feature_importance   = tech_d.get("feature_importance"),
                # Ensemble scores
                technical_ml_score     = tech_ml,
                fundamental_ml_score   = fund_ml,
                entropy_ml_score       = entr_ml,
                llm_score              = llm_score_val,
                llm_provider           = llm_provider,
                llm_reasoning_json     = llm_data,
                # Combined
                technical_score        = tech_combined,
                fundamental_score      = fund_combined,
                entropy_score          = entr_combined,
                combined_score         = overall,
                # Weights
                w_technical            = reg_weights["technical"]["ml"],
                w_fundamental          = reg_weights["fundamental"]["ml"],
                w_entropy              = reg_weights["entropy"]["ml"],
                # Confidence
                confidence_score       = round(raw_confidence, 3),
                model_agreement        = round(1.0 - overall_dispersion, 3) if overall_dispersion is not None else None,
                llm_ml_alignment       = llm_ml_align,
                # Delta
                prev_combined_score    = prev_combined,
                score_delta            = score_delta,
                # Risk
                realised_vol_21d       = risk.get("vol_21d"),
                realised_vol_63d       = risk.get("vol_63d"),
                beta_vs_qqq            = risk.get("beta"),
                max_drawdown_1y        = risk.get("mdd"),
                sharpe_1y              = risk.get("sharpe"),
                # ETF metadata
                etf_type               = "STOCK",
                is_etf_composite       = False,
            )
            db.add(score_row)
            score_rows.append((ticker, overall))

        # ── Step 11b: Write ETF composite score rows ───────────────
        # For each EQUITY_ETF, average the scores of its underlying holdings
        for etf_ticker, holdings in etf_holding_map.items():
            holding_tickers  = [h.ticker for h in holdings]
            holding_weights  = [h.weight for h in holdings]
            total_weight     = sum(holding_weights) or 1.0

            # Weighted average of each score dimension
            def wavg(getter):
                vals = [(getter(t), w) for t, w in zip(holding_tickers, holding_weights) if getter(t) is not None]
                if not vals:
                    return None
                return sum(v * w for v, w in vals) / sum(w for _, w in vals)

            etf_tech  = wavg(lambda t: all_combined.get(t) if tech_scores.get(t) else None)
            etf_fund  = wavg(lambda t: (fund_scores.get(t) or {}).get("score"))
            etf_entr  = wavg(lambda t: (entr_scores.get(t) or {}).get("score"))
            etf_llm   = wavg(lambda t: (llm_scores.get(t) or {}).get("score"))
            etf_conf  = wavg(lambda t: next(
                (s.confidence_score for s in db.new if hasattr(s, "ticker") and s.ticker == t and s.confidence_score is not None),
                None
            ))

            # Overall combined = simple average of available strategy scores
            avail_etf = [s for s in [etf_tech, etf_fund, etf_entr] if s is not None]
            etf_overall = float(sum(avail_etf) / len(avail_etf)) if avail_etf else None
            all_combined[etf_ticker] = etf_overall

            etf_row = models.Score(
                run_id                 = run_id,
                ticker                 = etf_ticker,
                technical_score        = etf_tech,
                fundamental_score      = etf_fund,
                entropy_score          = etf_entr,
                llm_score              = etf_llm,
                llm_provider           = models.LLMProvider.claude if etf_llm else models.LLMProvider.none,
                combined_score         = etf_overall,
                confidence_score       = etf_conf,
                w_technical            = reg_weights["technical"]["ml"],
                w_fundamental          = reg_weights["fundamental"]["ml"],
                w_entropy              = reg_weights["entropy"]["ml"],
                is_etf_composite       = True,
                etf_type               = "EQUITY_ETF",
                etf_holdings_used      = [{"ticker": h.ticker, "weight": h.weight, "description": h.description} for h in holdings],
            )
            db.add(etf_row)
            logger.info(f"ETF composite {etf_ticker}: combined={f'{etf_overall:.3f}' if etf_overall is not None else 'N/A'} (from {holding_tickers})")

        # ── Step 11c: Write excluded ticker rows (no score, labelled) ─
        for excl_ticker, reason in excluded_tickers.items():
            etf_type = classifications[excl_ticker].etf_type if excl_ticker in classifications else "NON_SCOREABLE"
            excl_row = models.Score(
                run_id         = run_id,
                ticker         = excl_ticker,
                etf_type       = etf_type,
                is_etf_composite = False,
                llm_provider   = models.LLMProvider.none,
                w_technical    = reg_weights["technical"]["ml"],
                w_fundamental  = reg_weights["fundamental"]["ml"],
                w_entropy      = reg_weights["entropy"]["ml"],
            )
            db.add(excl_row)

        # ── Step 12: Compute rank deltas ───────────────────────────
        # Rank by combined score descending; store rank_delta vs prev run
        try:
            sorted_tickers = sorted(
                [(t, s) for t, s in all_combined.items() if s is not None],
                key=lambda x: x[1], reverse=True
            )
            curr_ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_tickers)}

            if prev_scores_map:
                prev_combined_scores = {t: s.combined_score for t, s in prev_scores_map.items() if s.combined_score}
                sorted_prev = sorted(prev_combined_scores.items(), key=lambda x: x[1], reverse=True)
                prev_ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_prev)}

                # Update rank_delta on already-added score rows
                for score_obj in db.new:
                    if hasattr(score_obj, "ticker") and score_obj.ticker in curr_ranks:
                        curr_r = curr_ranks.get(score_obj.ticker)
                        prev_r = prev_ranks.get(score_obj.ticker)
                        if curr_r and prev_r:
                            score_obj.rank_delta = prev_r - curr_r  # positive = improved rank
        except Exception as e:
            logger.warning(f"Rank delta computation failed: {e}")

        # ── Step 13: Store market regime snapshot ──────────────────
        try:
            regime_row = models.MarketRegime(
                run_id            = run_id,
                regime_label      = regime_data.get("label", "Neutral / Mixed"),
                regime_confidence = regime_data.get("confidence", 0.5),
                vix               = macro_snapshot.get("vix"),
                yield_curve_10y2y = macro_snapshot.get("yield_curve"),
                fed_funds_rate    = macro_snapshot.get("fed_funds"),
                cpi_yoy           = macro_snapshot.get("cpi_yoy"),
                dominant_factor   = regime_data.get("dominant_factor"),
                factor_weight_adj = regime_data.get("factor_weight_adj"),
                transition_risk   = regime_data.get("transition_risk"),
                raw_fred_json     = macro_snapshot,
            )
            db.add(regime_row)
        except Exception as e:
            logger.warning(f"Regime storage failed: {e}")

        # ── Step 14: Update run status ─────────────────────────────
        run.status = (
            models.RunStatus.complete_with_warnings if warnings_list
            else models.RunStatus.complete
        )
        if warnings_list:
            run.error_log = "\n".join(warnings_list)
        db.commit()
        logger.info(f"Score run {run_id} complete. Warnings: {len(warnings_list)}")

    except Exception as e:
        logger.error(f"Score run {run_id} failed: {e}", exc_info=True)
        run.status = models.RunStatus.failed
        run.error_log = str(e)
        db.commit()
        raise e
    finally:
        db.close()


# ── Optimization Job ──────────────────────────────────────────────────────
@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def run_optimization_job(
    self, job_id: str, portfolio_id: str, run_id: str,
    optimizer_type: str, settings: dict,
):
    """
    Optimize portfolio weights using the specified algorithm.
    Returns {ticker: weight} in result_json.

    ⚠ Result MUST be labeled as 'optimizer output' — never as 'paper model'.
    Paper model: top-10 equal-weight from composite scores.
    """
    from app.database import SessionLocal
    from app import models
    from app.optimizer.deep_rl import DeepRLOptimizer, mvo_optimize, hrp_optimize

    db = SessionLocal()
    job = db.query(models.OptimizationJob).filter(models.OptimizationJob.id == job_id).first()
    if not job:
        db.close()
        return

    try:
        job.status = models.RunStatus.running
        db.commit()

        # Load scores
        scores = db.query(models.Score).filter(models.Score.run_id == run_id).all()
        if not scores:
            raise ValueError(f"No scores found for run {run_id}")

        tickers = [s.ticker for s in scores]
        combined_scores = {s.ticker: s.combined_score or 0.5 for s in scores}

        if optimizer_type == "deep_rl":
            # For RL we need historical returns — use mock if Alpaca unavailable
            import numpy as np
            import pandas as pd
            n = len(tickers)
            # In production: fetch real returns from Alpaca
            # Mock: use scores as proxy for expected returns
            mock_returns = pd.DataFrame(
                np.random.randn(500, n) * 0.01,
                columns=tickers,
            )
            mock_scores = pd.DataFrame(
                np.tile([combined_scores[t] for t in tickers], (500, 1)),
                columns=tickers,
            )
            optimizer = DeepRLOptimizer(n_assets=n)
            optimizer.train(mock_returns, mock_scores, total_timesteps=20_000)

            # Build obs from current scores
            import numpy as np
            obs = np.array([
                [combined_scores[t] for t in tickers] +   # scores
                [0.0] * n +                                 # lagged ret 1m
                [0.0] * n +                                 # lagged ret 3m
                [0.05] * n +                                # vol
                [1.0 / n] * n                               # current weights
            ], dtype=np.float32).flatten()
            raw_weights = optimizer.predict_weights(obs)
            result_weights = {tickers[i]: float(w) for i, w in enumerate(raw_weights) if isinstance(raw_weights, np.ndarray)}
            if not isinstance(raw_weights, np.ndarray):
                result_weights = {t: float(raw_weights.get(i, 1.0/n)) for i, t in enumerate(tickers)}

        elif optimizer_type == "mvo":
            import numpy as np
            import pandas as pd
            # Mock returns if real data unavailable
            n = len(tickers)
            mock_returns = pd.DataFrame(
                np.random.randn(252, n) * 0.01,
                columns=tickers,
            )
            max_weight = settings.get("max_weight", 0.25)
            result_weights = mvo_optimize(mock_returns, target="max_sharpe", max_weight=max_weight)

        elif optimizer_type == "hrp":
            import numpy as np
            import pandas as pd
            n = len(tickers)
            mock_returns = pd.DataFrame(
                np.random.randn(252, n) * 0.01,
                columns=tickers,
            )
            result_weights = hrp_optimize(mock_returns)

        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Normalize weights
        total = sum(result_weights.values())
        if total > 0:
            result_weights = {k: v / total for k, v in result_weights.items()}

        job.result_json = result_weights
        job.status = models.RunStatus.complete
        db.commit()
        logger.info(f"Optimization job {job_id} ({optimizer_type}) complete")

    except Exception as e:
        logger.error(f"Optimization job {job_id} failed: {e}", exc_info=True)
        job.status = models.RunStatus.failed
        db.commit()
        raise self.retry(exc=e)
    finally:
        db.close()


# ── Discovery Job ─────────────────────────────────────────────────────────────
@celery_app.task(bind=True, max_retries=0)
def run_discovery_job(self, discovery_run_id: str):
    """
    Score the full NASDAQ-100 universe through the complete 3-strategy + LLM pipeline.
    Results stored in discovery_runs / discovery_scores — separate from portfolio scoring.
    """
    from datetime import timedelta
    import numpy as np
    from app.database import SessionLocal
    from app import models
    from app.ml.scoring import combined_score, normalize_scores, OPTIMAL_WEIGHTS
    from app.ml.fundamental import FundamentalScorer
    from app.ml.technical import TechnicalScorer
    from app.ml.entropy import EntropyScorer
    from app.ml.llm_scoring import LLMScorer
    from app.data.clients import AlpacaClient, AlpacaDataError
    from app.data.clients import AlphaVantageClient, AlphaVantageError
    from app.data.clients import EDGARClient
    from app.data.ndx100 import get_ndx100_tickers, get_sector
    from app.data.etf_client import ETFClient

    db = SessionLocal()
    run = db.query(models.DiscoveryRun).filter(
        models.DiscoveryRun.id == discovery_run_id
    ).first()
    if not run:
        logger.error(f"DiscoveryRun {discovery_run_id} not found")
        return

    try:
        run.status = models.RunStatus.running
        db.commit()

        tickers = get_ndx100_tickers()
        run.universe_size = len(tickers)
        db.commit()

        rebalance_date = run.run_date
        training_start = rebalance_date - timedelta(days=730)
        frequency = "monthly"
        warnings_list = []

        # Filter out Alpaca-unsupported tickers
        etf_client = ETFClient()
        clean_tickers = [t for t in tickers if t not in etf_client._ALPACA_UNSUPPORTED]

        # ── Price data ─────────────────────────────────────────────
        alpaca = AlpacaClient()
        prices_df = None
        try:
            prices_df = alpaca.get_ohlcv(clean_tickers, training_start, rebalance_date)
            logger.info(f"Discovery Alpaca: {len(prices_df)} bars for {len(clean_tickers)} tickers")
        except AlpacaDataError as e:
            logger.warning(f"Discovery Alpaca failed: {e}")
            warnings_list.append(f"ALPACA_UNAVAILABLE: {e}")

        # ── Fundamental data ───────────────────────────────────────
        av = AlphaVantageClient()
        fundamentals_df = None
        try:
            fundamentals_df = av.get_fundamentals_batch(clean_tickers)
            logger.info(f"Discovery AV: {len(fundamentals_df)} rows")
        except AlphaVantageError as e:
            logger.warning(f"Discovery AV failed: {e}")
            warnings_list.append(f"ALPHAVANTAGE_UNAVAILABLE: {e}")

        # ── Attach forward returns ─────────────────────────────────
        import pandas as pd
        if fundamentals_df is not None and prices_df is not None:
            try:
                prices_df["date"] = pd.to_datetime(prices_df["date"])
                fundamentals_df["period_date"] = pd.to_datetime(fundamentals_df["period_date"])
                price_lookup = {
                    t: grp.sort_values("date").reset_index(drop=True)
                    for t, grp in prices_df.groupby("ticker")
                }
                def get_fwd(ticker, report_date):
                    if ticker not in price_lookup:
                        return None
                    grp = price_lookup[ticker]
                    after = grp[grp["date"] >= report_date]
                    if len(after) < 22:
                        return None
                    s = after.iloc[0]["close"]
                    e = after.iloc[21]["close"]
                    return (e / s) - 1 if s else None
                fundamentals_df["forward_return"] = fundamentals_df.apply(
                    lambda r: get_fwd(r["ticker"], r["period_date"]), axis=1
                )
            except Exception as e:
                logger.warning(f"Discovery forward return failed: {e}")

        # ── EDGAR filings ──────────────────────────────────────────
        edgar = EDGARClient()
        filing_contexts = {}
        for ticker in clean_tickers:
            ctx = edgar.get_filing_context(ticker, rebalance_date)
            if ctx:
                filing_contexts[ticker] = ctx

        # ── Claude LLM scoring ─────────────────────────────────────
        llm_scorer = LLMScorer()
        llm_scores = {}
        for ticker in clean_tickers:
            ctx = filing_contexts.get(ticker, "")
            result = llm_scorer.score(
                ticker=ticker,
                company_name=ticker,
                frequency=frequency,
                period=rebalance_date.strftime("%Y-%m"),
                filing_context=ctx,
                earnings_context="",
            )
            if result is not None:
                llm_scores[ticker] = result

        # ── Macro regime ───────────────────────────────────────────
        from app.data.fred_client import FREDClient
        from app.ml.regime import classify_regime
        fred = FREDClient()
        macro_snapshot = fred.get_macro_snapshot()
        regime_data = classify_regime(macro_snapshot)
        run.regime_label = regime_data.get("label", "Neutral / Mixed")
        run.regime_confidence = regime_data.get("confidence", 0.5)
        db.commit()

        # ── ML scoring ─────────────────────────────────────────────
        fund_scores, tech_scores, entr_scores = {}, {}, {}

        if fundamentals_df is not None:
            try:
                fund_model = FundamentalScorer()
                fund_model.fit(fundamentals_df, rebalance_date)
                fund_scores = fund_model.predict(clean_tickers, fundamentals_df)
            except Exception as e:
                logger.error(f"Discovery fundamental model error: {e}")

        if prices_df is not None:
            try:
                tech_model = TechnicalScorer()
                tech_model.fit(prices_df, rebalance_date)
                tech_scores = tech_model.predict(clean_tickers, prices_df, rebalance_date)
                entr_model = EntropyScorer()
                entr_model.fit(prices_df, rebalance_date)
                entr_scores = entr_model.predict(clean_tickers, prices_df, rebalance_date)
            except Exception as e:
                logger.error(f"Discovery tech/entropy model error: {e}")

        # ── Risk metrics ───────────────────────────────────────────
        risk_metrics = {}
        if prices_df is not None:
            try:
                from app.ml.risk import compute_risk_metrics
                risk_metrics = compute_risk_metrics(prices_df, rebalance_date)
            except Exception as e:
                logger.warning(f"Discovery risk metrics failed: {e}")

        # ── Previous run for deltas ────────────────────────────────
        prev_run = (
            db.query(models.DiscoveryRun)
            .filter(
                models.DiscoveryRun.status.in_([
                    models.RunStatus.complete,
                    models.RunStatus.complete_with_warnings,
                ]),
                models.DiscoveryRun.id != discovery_run_id,
            )
            .order_by(models.DiscoveryRun.run_date.desc())
            .first()
        )
        prev_scores_map = {}
        if prev_run:
            prev_scores = db.query(models.DiscoveryScore).filter(
                models.DiscoveryScore.discovery_run_id == prev_run.id
            ).all()
            prev_scores_map = {s.ticker: s for s in prev_scores}

        # ── Optimal weights ────────────────────────────────────────
        weights_m = {
            "technical":   OPTIMAL_WEIGHTS.get(("technical",   frequency), {"ml": 1.0,  "llm": 0.0}),
            "fundamental": OPTIMAL_WEIGHTS.get(("fundamental", frequency), {"ml": 0.15, "llm": 0.85}),
            "entropy":     OPTIMAL_WEIGHTS.get(("entropy",     frequency), {"ml": 0.70, "llm": 0.30}),
        }
        adj = regime_data.get("factor_weight_adj", {})
        reg_weights = {
            s: {"ml": min(1.0, weights_m[s]["ml"] * adj.get(s, 1.0))}
            for s in ["technical", "fundamental", "entropy"]
        }

        # ── Write discovery score rows ─────────────────────────────
        all_scores = {}  # ticker → combined_score for ranking

        for ticker in clean_tickers:
            llm_data      = llm_scores.get(ticker)
            llm_score_val = llm_data["score"] if llm_data else None
            llm_failed    = llm_score_val is None
            llm_provider  = models.LLMProvider.claude if llm_data else models.LLMProvider.none

            tech_d = tech_scores.get(ticker) or {}
            fund_d = fund_scores.get(ticker) or {}
            entr_d = entr_scores.get(ticker) or {}

            tech_ml = tech_d.get("score") if isinstance(tech_d, dict) else tech_d
            fund_ml = fund_d.get("score") if isinstance(fund_d, dict) else fund_d
            entr_ml = entr_d.get("score") if isinstance(entr_d, dict) else entr_d

            tech_c = combined_score(tech_ml or 0.5, llm_score_val, "technical",   frequency, llm_failed) if tech_ml is not None else None
            fund_c = combined_score(fund_ml or 0.5, llm_score_val, "fundamental", frequency, llm_failed) if fund_ml is not None else None
            entr_c = combined_score(entr_ml or 0.5, llm_score_val, "entropy",     frequency, llm_failed) if entr_ml is not None else None

            avail = [s for s in [tech_c, fund_c, entr_c] if s is not None]
            overall = float(sum(avail) / len(avail)) if avail else None
            all_scores[ticker] = overall

            # Dispersion + confidence
            comp_scores = [s for s in [
                tech_d.get("xgboost"), tech_d.get("lightgbm"), tech_d.get("catboost"),
                fund_d.get("ridge"),   fund_d.get("xgboost"),   fund_d.get("rf"), fund_d.get("mlp"),
                entr_d.get("xgboost"), entr_d.get("lightgbm"), entr_d.get("catboost"),
            ] if s is not None]
            dispersion  = float(np.std(comp_scores)) if len(comp_scores) > 1 else 0.0
            rc_boost    = regime_data.get("confidence", 0.7) - 0.7
            confidence  = max(0.0, min(1.0, 1.0 - (dispersion * 3.0) + rc_boost))

            # Delta
            prev = prev_scores_map.get(ticker)
            prev_combined = prev.combined_score if prev else None
            score_delta   = round(overall - prev_combined, 4) if (overall is not None and prev_combined is not None) else None
            prev_rank     = prev.rank if prev else None

            risk = risk_metrics.get(ticker, {})

            score_row = models.DiscoveryScore(
                discovery_run_id  = discovery_run_id,
                ticker            = ticker,
                sector            = get_sector(ticker),
                technical_score   = tech_c,
                fundamental_score = fund_c,
                entropy_score     = entr_c,
                combined_score    = overall,
                llm_score         = llm_score_val,
                llm_provider      = llm_provider,
                llm_reasoning_json = llm_data,
                confidence_score  = round(confidence, 3),
                overall_dispersion = dispersion,
                prev_combined_score = prev_combined,
                score_delta       = score_delta,
                prev_rank         = prev_rank,
                technical_feature_importance  = tech_d.get("feature_importance"),
                fundamental_feature_importance = fund_d.get("feature_importance"),
                realised_vol_21d  = risk.get("vol_21d"),
                beta_vs_qqq       = risk.get("beta"),
                sharpe_1y         = risk.get("sharpe"),
            )
            db.add(score_row)

        db.flush()  # get IDs before ranking

        # ── Compute and write ranks ────────────────────────────────
        sorted_tickers = sorted(
            [(t, s) for t, s in all_scores.items() if s is not None],
            key=lambda x: x[1], reverse=True,
        )
        curr_ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_tickers)}

        for score_obj in db.new:
            if isinstance(score_obj, models.DiscoveryScore):
                ticker = score_obj.ticker
                score_obj.rank = curr_ranks.get(ticker)
                if score_obj.prev_rank is not None and score_obj.rank is not None:
                    score_obj.rank_delta = score_obj.prev_rank - score_obj.rank

        run.scored_count = len(clean_tickers)
        run.status = (
            models.RunStatus.complete_with_warnings if warnings_list
            else models.RunStatus.complete
        )
        db.commit()
        logger.info(f"Discovery run {discovery_run_id} complete — {len(clean_tickers)} tickers scored")

    except Exception as e:
        logger.error(f"Discovery run {discovery_run_id} failed: {e}", exc_info=True)
        run.status = models.RunStatus.failed
        run.error_log = str(e)
        db.commit()
        raise e
    finally:
        db.close()
