"""
app/routers/backtest.py

Backtest router — two concerns kept strictly separate:

1. PAPER BENCHMARKS (source-backed, locked):
   Table 1, Cohen et al., Entropy 2025, 27, 550.
   Six strategy-frequency configurations with locked metrics.
   Time-series NOT fabricated — returns required_fields schema instead.

2. LIVE CURRENT TOP-10 PERFORMANCE (forward, real data):
   How the current discovery run's top-10 has performed since the
   run_date, using actual Alpaca close prices.
   Clearly labelled as forward performance, not backtested.
"""
import csv
import io
import logging
from datetime import datetime, timedelta, date

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.data.clients import AlpacaClient, AlpacaDataError

logger = logging.getLogger("backtest")
router = APIRouter()

# ── Locked benchmark facts — DO NOT MODIFY ────────────────────────────────
# Source: Table 1, Cohen et al., Entropy 2025, 27, 550
# cumulative_return is the decimal multiplier (19.7771 = 1977.71%)
LOCKED_BENCHMARKS = [
    {
        "id":                "tech-m",
        "strategy":          "technical",
        "frequency":         "monthly",
        "ml_weight":         1.00,
        "llm_weight":        0.00,
        "sharpe_ratio":      0.6934,
        "average_return":    0.0750,
        "volatility":        0.1082,
        "cumulative_return": 19.7771,
        "badge":             "Best Cumulative Return",
        "badge_color":       "#4f8ef7",
        "note":              "Pure ML; momentum and short-term price signals fully captured by quantitative models. Semantic enrichment adds no measurable value at monthly horizon.",
    },
    {
        "id":                "ent-m",
        "strategy":          "entropy",
        "frequency":         "monthly",
        "ml_weight":         0.70,
        "llm_weight":        0.30,
        "sharpe_ratio":      0.4207,
        "average_return":    0.0523,
        "volatility":        0.1244,
        "cumulative_return": 7.0052,
        "badge":             None,
        "badge_color":       None,
        "note":              "Balanced blend; entropy benefits from semantic context to disambiguate complex structural signals. Neither extreme of the weight range performs well.",
    },
    {
        "id":                "fund-m",
        "strategy":          "fundamental",
        "frequency":         "monthly",
        "ml_weight":         0.15,
        "llm_weight":        0.85,
        "sharpe_ratio":      0.5001,
        "average_return":    0.0432,
        "volatility":        0.0863,
        "cumulative_return": 5.7840,
        "badge":             "Lowest Volatility (Monthly)",
        "badge_color":       "#3ecf8e",
        "note":              "Heavily semantic; LLM contextual reading of earnings calls, filings, and macro commentary drove most predictive value. Lowest volatility in monthly set.",
    },
    {
        "id":                "tech-q",
        "strategy":          "technical",
        "frequency":         "quarterly",
        "ml_weight":         0.45,
        "llm_weight":        0.55,
        "sharpe_ratio":      1.2967,
        "average_return":    0.2499,
        "volatility":        0.1927,
        "cumulative_return": 5.7337,
        "badge":             "Best Sharpe Ratio",
        "badge_color":       "#f5a623",
        "note":              "Highest Sharpe ratio across all configurations. Semantic enrichment at quarterly horizon reduced risk without proportionally reducing return — the core risk-adjusted case for LLM blending.",
    },
    {
        "id":                "ent-q",
        "strategy":          "entropy",
        "frequency":         "quarterly",
        "ml_weight":         0.40,
        "llm_weight":        0.60,
        "sharpe_ratio":      0.6048,
        "average_return":    0.2025,
        "volatility":        0.3348,
        "cumulative_return": 5.3436,
        "badge":             None,
        "badge_color":       None,
        "note":              "Slight semantic lean; highest volatility in the set. Semantic inputs help at quarterly horizon but do not resolve entropy's inherent structural complexity.",
    },
    {
        "id":                "fund-q",
        "strategy":          "fundamental",
        "frequency":         "quarterly",
        "ml_weight":         0.00,
        "llm_weight":        1.00,
        "sharpe_ratio":      0.4899,
        "average_return":    0.1471,
        "volatility":        0.3002,
        "cumulative_return": 3.2612,
        "badge":             "Pure Semantic",
        "badge_color":       "#8b90a7",
        "note":              "Pure LLM; optimal empirical result — zero ML weight. LLM semantic trackers of fundamental metrics outperformed structured algorithmic signals at quarterly horizon.",
    },
]

# Derived metrics computed from locked values (zero-risk-free rate, per paper)
def _derived(b: dict) -> dict:
    """Compute derived metrics from locked Table 1 values."""
    # Period: 2020-01-01 to 2025-01-01 = 60 months = 5 years
    n_years   = 5.0
    cum       = b["cumulative_return"]    # decimal multiplier
    avg_ret   = b["average_return"]       # per period (monthly or quarterly)
    vol       = b["volatility"]           # annualised
    sharpe    = b["sharpe_ratio"]

    freq      = b["frequency"]
    periods   = 60 if freq == "monthly" else 20

    # CAGR from cumulative return
    cagr = (1 + cum) ** (1 / n_years) - 1

    # Sortino — approximate using downside vol heuristic (0.7 × total vol)
    # Note: exact downside vol not published; flagged as estimated
    sortino_est = round(avg_ret * (12 if freq == "monthly" else 4) / (vol * 0.70), 3)

    # Calmar — CAGR / |max drawdown|
    # Max drawdown not in Table 1; mark as not available
    calmar = None

    # Win rate — not published; mark as not available
    win_rate = None

    return {
        **b,
        "cagr":          round(cagr, 4),
        "sortino_est":   sortino_est,
        "sortino_note":  "Estimated — exact downside vol not published",
        "calmar":        calmar,
        "win_rate":      win_rate,
        "n_periods":     periods,
        "backtest_period": "2020-01-01 to 2025-01-01",
        "universe":      "NASDAQ-100",
        "portfolio_size": 10,
    }

BENCHMARKS_FULL = [_derived(b) for b in LOCKED_BENCHMARKS]


@router.get("/benchmarks")
def get_benchmarks(frequency: str = Query(default="all")):
    """
    Return locked benchmark configurations with derived metrics.
    Source: Table 1, Cohen et al., Entropy 2025, 27, 550.
    """
    if frequency == "all":
        return {"benchmarks": BENCHMARKS_FULL, "source": "Table 1, Cohen et al., Entropy 2025, 27, 550"}
    filtered = [b for b in BENCHMARKS_FULL if b["frequency"] == frequency]
    return {"benchmarks": filtered, "source": "Table 1, Cohen et al., Entropy 2025, 27, 550"}


@router.get("/live-performance")
def get_live_performance(
    portfolio_id: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    Forward performance of the current discovery top-10 since the last run date.

    This is NOT backtested — it is the out-of-sample forward return of the
    paper model's current selection, using live Alpaca price data.
    Clearly distinguished from the locked paper benchmarks.
    """
    # Latest completed discovery run
    run = (
        db.query(models.DiscoveryRun)
        .filter(models.DiscoveryRun.status.in_(["complete", "complete_with_warnings"]))
        .order_by(models.DiscoveryRun.run_date.desc())
        .first()
    )
    if not run:
        return {"available": False, "reason": "No completed discovery run found"}

    # Top-10 tickers
    top10 = (
        db.query(models.DiscoveryScore)
        .filter(models.DiscoveryScore.discovery_run_id == run.id)
        .order_by(models.DiscoveryScore.combined_score.desc())
        .limit(10)
        .all()
    )
    if not top10:
        return {"available": False, "reason": "No scores in latest run"}

    tickers     = [s.ticker for s in top10]
    scores_map  = {s.ticker: round(s.combined_score, 3) for s in top10 if s.combined_score}
    run_date    = run.run_date.date()
    today       = date.today()

    # Need at least 1 trading day of data
    if (today - run_date).days < 1:
        return {
            "available": False,
            "reason":    "Run completed today — no forward return data yet",
            "run_date":  run_date.isoformat(),
            "tickers":   tickers,
        }

    # Fetch prices: from run_date to today
    alpaca = AlpacaClient()
    try:
        all_tickers = list(set(tickers + ["QQQ"]))
        df = alpaca.get_ohlcv(
            all_tickers,
            start=datetime.combine(run_date, datetime.min.time()),
            end=datetime.combine(today, datetime.min.time()),
        )
    except AlpacaDataError as e:
        return {"available": False, "reason": f"Alpaca unavailable: {e}"}

    if df.empty:
        return {"available": False, "reason": "No price data returned from Alpaca"}

    df["date"] = pd.to_datetime(df["date"])
    first_date = df["date"].min()
    last_date  = df["date"].max()
    n_days     = (last_date - first_date).days

    if n_days < 1:
        return {"available": False, "reason": "Insufficient price history"}

    # Per-ticker return since run date
    ticker_returns = []
    for ticker in tickers:
        t_df = df[df["ticker"] == ticker].sort_values("date")
        if len(t_df) < 2:
            continue
        entry = float(t_df.iloc[0]["close"])
        exit_ = float(t_df.iloc[-1]["close"])
        ret   = (exit_ - entry) / entry if entry > 0 else None
        ticker_returns.append({
            "ticker":        ticker,
            "entry_price":   round(entry, 2),
            "current_price": round(exit_, 2),
            "return":        round(ret, 4) if ret is not None else None,
            "combined_score": scores_map.get(ticker),
        })

    if not ticker_returns:
        return {"available": False, "reason": "No valid price data for top-10 tickers"}

    # Equal-weight portfolio return
    valid_rets = [t["return"] for t in ticker_returns if t["return"] is not None]
    port_return = sum(valid_rets) / len(valid_rets) if valid_rets else None

    # QQQ return over same period
    qqq_df = df[df["ticker"] == "QQQ"].sort_values("date")
    qqq_return = None
    if len(qqq_df) >= 2:
        qqq_start = float(qqq_df.iloc[0]["close"])
        qqq_end   = float(qqq_df.iloc[-1]["close"])
        qqq_return = round((qqq_end - qqq_start) / qqq_start, 4) if qqq_start > 0 else None

    alpha = round(port_return - qqq_return, 4) if (port_return is not None and qqq_return is not None) else None

    # Daily portfolio series for sparkline
    daily_series = []
    try:
        port_df = df[df["ticker"].isin(tickers)].copy()
        daily_value = (
            port_df
            .groupby("date")["close"]
            .mean()  # equal-weight proxy
            .reset_index()
            .rename(columns={"close": "price"})
        )
        base_price = float(daily_value.iloc[0]["price"])
        daily_series = [
            {
                "date":           d["date"].strftime("%Y-%m-%d"),
                "cumulative_ret": round((float(d["price"]) - base_price) / base_price, 4),
            }
            for _, d in daily_value.iterrows()
        ]
    except Exception as e:
        logger.debug(f"Daily series failed: {e}")

    return {
        "available":      True,
        "label":          "Current Top-10 (Forward, Equal-Weight)",
        "type":           "forward",
        "run_date":       run_date.isoformat(),
        "data_through":   last_date.strftime("%Y-%m-%d"),
        "n_trading_days": len(df["date"].unique()),
        "tickers":        tickers,
        "ticker_returns": sorted(ticker_returns, key=lambda x: x["return"] or 0, reverse=True),
        "portfolio_return": round(port_return, 4) if port_return is not None else None,
        "qqq_return":     qqq_return,
        "alpha":          alpha,
        "daily_series":   daily_series,
        "disclaimer":     (
            "Forward performance only — not a backtest. "
            "Based on equal-weight top-10 selection from the most recent discovery run. "
            "Past performance does not predict future results."
        ),
    }


@router.post("/run")
def run_backtest(data: schemas.BacktestRequest, db: Session = Depends(get_db)):
    filtered = [b for b in BENCHMARKS_FULL if b["strategy"] in data.strategies]
    return {
        "source":           "Table 1, Cohen et al., Entropy 2025, 27, 550",
        "period":           f"{data.start_date} to {data.end_date}",
        "universe":         "NASDAQ-100",
        "results":          filtered,
        "series_available": False,
        "series_note": (
            "Full cumulative return and monthly return series not included in paper attachment. "
            "Supply underlying data to render time-series charts. "
            "Required fields: period_index, strategy_type, rebalance_frequency, cumulative_return"
        ),
    }


@router.get("/{job_id}/results")
def get_backtest_results(job_id: str, db: Session = Depends(get_db)):
    return {"job_id": job_id, "benchmarks": BENCHMARKS_FULL, "series_available": False}


# ── Export router ─────────────────────────────────────────────────────────
export_router = APIRouter()

@export_router.get("/trades/{proposal_id}")
def export_trades(proposal_id: str, fmt: str = "csv", db: Session = Depends(get_db)):
    trades = db.query(models.Trade).filter(models.Trade.proposal_id == proposal_id).all()
    if not trades:
        raise HTTPException(status_code=404, detail="No trades found")
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ticker", "action", "shares", "estimated_price", "estimated_value"])
    for t in trades:
        writer.writerow([t.ticker, t.action, t.shares, t.estimated_price, t.estimated_value])
    from fastapi.responses import StreamingResponse
    output.seek(0)
    return StreamingResponse(output, media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename=trades_{proposal_id}.csv"})

@export_router.get("/report/{proposal_id}")
def export_report(proposal_id: str, db: Session = Depends(get_db)):
    return {"proposal_id": proposal_id, "status": "PDF export coming in Phase 4"}
