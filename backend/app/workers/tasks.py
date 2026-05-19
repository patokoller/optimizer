"""
app/workers/tasks.py
─────────────────────────────────────────────────────────────────────────────
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
@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
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

        tickers = [h.ticker for h in portfolio.holdings]
        if not tickers:
            raise ValueError("Portfolio has no holdings")

        rebalance_date = run.run_date
        training_start = rebalance_date - timedelta(days=730)  # 24 months

        warnings_list = []

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
                    company_name=ticker,  # In production: resolve company name
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
                    break  # If Claude fails for one, assume it fails for all

        # ── Step 6: ML model training + scoring ────────────────────
        fund_scores = {}
        tech_scores = {}
        entr_scores = {}

        # Attach forward returns to fundamentals using Alpaca price data
        import pandas as pd
        if fundamentals_df is not None and prices_df is not None:
            try:
                prices_df["date"] = pd.to_datetime(prices_df["date"])
                fundamentals_df["period_date"] = pd.to_datetime(fundamentals_df["period_date"])

                # Build lookup: ticker → sorted price series
                price_lookup = {
                    ticker: grp.sort_values("date").reset_index(drop=True)
                    for ticker, grp in prices_df.groupby("ticker")
                }

                def get_forward_return(ticker, report_date):
                    if ticker not in price_lookup:
                        return None
                    grp = price_lookup[ticker]
                    after = grp[grp["date"] >= report_date]
                    if len(after) < 22:
                        return None
                    start_price = after.iloc[0]["close"]
                    end_price   = after.iloc[21]["close"]
                    return (end_price / start_price) - 1 if start_price else None

                fundamentals_df["forward_return"] = fundamentals_df.apply(
                    lambda row: get_forward_return(row["ticker"], row["period_date"]), axis=1
                )
                labeled = fundamentals_df["forward_return"].notna().sum()
                logger.info(f"Forward returns attached: {labeled}/{len(fundamentals_df)} rows labeled")
            except Exception as e:
                logger.warning(f"Forward return computation failed: {e}")

        if fundamentals_df is not None:
            try:
                fund_model = FundamentalScorer()
                fund_model.fit(fundamentals_df, rebalance_date)
                fund_scores = fund_model.predict(tickers, fundamentals_df)
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
            except Exception as e:
                logger.error(f"Technical/Entropy model error: {e}")
                warnings_list.append(f"TECHNICAL_MODEL_ERROR: {e}")

        # ── Step 7: Write Score rows ───────────────────────────────
        weights_m = {
            "technical":   OPTIMAL_WEIGHTS.get(("technical",   frequency), {"ml": 1.0, "llm": 0.0}),
            "fundamental": OPTIMAL_WEIGHTS.get(("fundamental", frequency), {"ml": 0.15, "llm": 0.85}),
            "entropy":     OPTIMAL_WEIGHTS.get(("entropy",     frequency), {"ml": 0.70, "llm": 0.30}),
        }

        for ticker in tickers:
            llm_data = llm_scores.get(ticker)
            llm_score_val = llm_data["score"] if llm_data else None
            llm_provider = models.LLMProvider.claude if llm_data else models.LLMProvider.none
            llm_failed = llm_score_val is None

            tech_ml  = tech_scores.get(ticker)
            fund_ml  = fund_scores.get(ticker)
            entr_ml  = entr_scores.get(ticker)

            # Combined scores per strategy
            tech_combined  = combined_score(tech_ml or 0.5,  llm_score_val, "technical",   frequency, llm_failed) if tech_ml  is not None else None
            fund_combined  = combined_score(fund_ml or 0.5,  llm_score_val, "fundamental", frequency, llm_failed) if fund_ml  is not None else None
            entr_combined  = combined_score(entr_ml or 0.5,  llm_score_val, "entropy",     frequency, llm_failed) if entr_ml  is not None else None

            # Overall combined = mean of available strategy scores
            avail = [s for s in [tech_combined, fund_combined, entr_combined] if s is not None]
            overall = float(sum(avail) / len(avail)) if avail else None

            score_row = models.Score(
                run_id                 = run_id,
                ticker                 = ticker,
                technical_ml_score     = tech_ml,
                fundamental_ml_score   = fund_ml,
                entropy_ml_score       = entr_ml,
                llm_score              = llm_score_val,
                llm_provider           = llm_provider,
                llm_reasoning_json     = llm_data,
                technical_score        = tech_combined,
                fundamental_score      = fund_combined,
                entropy_score          = entr_combined,
                combined_score         = overall,
                w_technical            = weights_m["technical"]["ml"],
                w_fundamental          = weights_m["fundamental"]["ml"],
                w_entropy              = weights_m["entropy"]["ml"],
            )
            db.add(score_row)

        # ── Step 8: Update run status ──────────────────────────────
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
        raise self.retry(exc=e)
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
