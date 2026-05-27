"""
Dashboard KPI endpoint.

GET /api/dashboard/kpis?portfolio_id=...

Returns:
  - portfolio_value:   current market value (shares × latest close)
  - mtd_return:        month-to-date return of the portfolio
  - mtd_alpha:         portfolio MTD return minus QQQ MTD return
  - active_risk:       annualised portfolio volatility (21-day rolling)
  - rebalance_due:     next rebalance date (1st of next month)
  - days_to_rebalance: calendar days until next rebalance
  - last_score_run:    date/time of most recent completed discovery run
  - top_ticker:        highest-scored ticker from latest discovery run
  - top_score:         its combined score
  - holdings_count:    number of positions in portfolio
  - data_date:         date of latest Alpaca price data used
  - alpaca_ok:         whether Alpaca data was available (bool)
"""
from datetime import datetime, date, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app import models
from app.data.clients import AlpacaClient, AlpacaDataError

import pandas as pd
import logging

logger = logging.getLogger("dashboard")
router = APIRouter()


def _next_first(today: date) -> date:
    """Return the 1st of the next month."""
    if today.month == 12:
        return date(today.year + 1, 1, 1)
    return date(today.year, today.month + 1, 1)


@router.get("/kpis")
def get_dashboard_kpis(
    portfolio_id: str = Query(...),
    db: Session = Depends(get_db),
):
    today = date.today()
    rebalance_date = _next_first(today)
    days_to_rebalance = (rebalance_date - today).days

    # ── Portfolio holdings ─────────────────────────────────────────────
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id
    ).first()

    if not portfolio or not portfolio.holdings:
        return _empty_response(days_to_rebalance, rebalance_date, db)

    holdings = [h for h in portfolio.holdings if h.shares and h.shares > 0]
    if not holdings:
        return _empty_response(days_to_rebalance, rebalance_date, db)

    tickers       = list({h.ticker for h in holdings})
    shares_map    = {h.ticker: h.shares for h in holdings}
    cost_map      = {h.ticker: (h.cost_basis or 0) for h in holdings}

    # ── Fetch prices from Alpaca ───────────────────────────────────────
    alpaca = AlpacaClient()
    alpaca_ok = False
    prices_df = None
    data_date = None

    try:
        # Fetch last 30 days to compute MTD return + volatility
        start = date(today.year, today.month, 1) - timedelta(days=5)  # buffer for weekends
        all_tickers = list(set(tickers + ["QQQ"]))
        prices_df = alpaca.get_ohlcv(all_tickers, start=datetime.combine(start, datetime.min.time()), end=datetime.combine(today, datetime.min.time()))
        alpaca_ok = True
        if not prices_df.empty:
            data_date = str(prices_df["date"].max())
    except AlpacaDataError as e:
        logger.warning(f"Alpaca unavailable for dashboard KPIs: {e}")

    # ── Compute KPIs from price data ───────────────────────────────────
    portfolio_value  = None
    mtd_return       = None
    mtd_alpha        = None
    active_risk      = None

    if alpaca_ok and prices_df is not None and not prices_df.empty:
        prices_df["date"] = pd.to_datetime(prices_df["date"])
        latest_date = prices_df["date"].max()

        # Latest close per ticker
        latest = (
            prices_df[prices_df["date"] == latest_date]
            .set_index("ticker")["close"]
            .to_dict()
        )

        # Portfolio market value = sum(shares × latest_close)
        port_value = sum(
            shares_map.get(t, 0) * latest.get(t, cost_map.get(t, 0))
            for t in tickers
        )
        portfolio_value = round(port_value, 2) if port_value > 0 else None

        # MTD return — compare first available close this month vs latest
        month_start = datetime(today.year, today.month, 1)
        month_prices = prices_df[prices_df["date"] >= month_start]

        if not month_prices.empty:
            # Portfolio MTD
            first_date_port = month_prices["date"].min()
            first_prices = (
                month_prices[month_prices["date"] == first_date_port]
                .set_index("ticker")["close"]
                .to_dict()
            )
            port_cost_mtd = sum(
                shares_map.get(t, 0) * first_prices.get(t, cost_map.get(t, 0))
                for t in tickers
            )
            port_value_mtd = sum(
                shares_map.get(t, 0) * latest.get(t, cost_map.get(t, 0))
                for t in tickers
            )
            if port_cost_mtd > 0:
                mtd_return = round((port_value_mtd - port_cost_mtd) / port_cost_mtd, 4)

            # QQQ MTD
            qqq_month = month_prices[month_prices["ticker"] == "QQQ"].sort_values("date")
            if len(qqq_month) >= 2:
                qqq_start = float(qqq_month.iloc[0]["close"])
                qqq_end   = float(qqq_month.iloc[-1]["close"])
                qqq_mtd   = (qqq_end - qqq_start) / qqq_start if qqq_start > 0 else None
                if mtd_return is not None and qqq_mtd is not None:
                    mtd_alpha = round(mtd_return - qqq_mtd, 4)

        # Active risk — portfolio daily returns, 21-day annualised vol
        try:
            # Build daily portfolio value series
            port_prices = prices_df[prices_df["ticker"].isin(tickers)].copy()
            port_daily = (
                port_prices
                .assign(value=lambda df: df["ticker"].map(shares_map).fillna(0) * df["close"])
                .groupby("date")["value"]
                .sum()
                .sort_index()
            )
            if len(port_daily) >= 5:
                daily_ret  = port_daily.pct_change().dropna()
                active_risk = round(float(daily_ret.tail(21).std() * (252 ** 0.5)), 4)
        except Exception as e:
            logger.debug(f"Active risk calc failed: {e}")

    # ── Latest discovery run metadata ─────────────────────────────────
    last_run = (
        db.query(models.DiscoveryRun)
        .filter(models.DiscoveryRun.status.in_(["complete", "complete_with_warnings"]))
        .order_by(models.DiscoveryRun.run_date.desc())
        .first()
    )
    last_score_run = last_run.run_date.isoformat() if last_run else None

    # Top scorer from latest discovery run
    top_ticker = None
    top_score  = None
    if last_run:
        top = (
            db.query(models.DiscoveryScore)
            .filter(models.DiscoveryScore.discovery_run_id == last_run.id)
            .order_by(models.DiscoveryScore.combined_score.desc())
            .first()
        )
        if top:
            top_ticker = top.ticker
            top_score  = round(top.combined_score, 3) if top.combined_score else None

    return {
        "portfolio_value":    portfolio_value,
        "mtd_return":         mtd_return,
        "mtd_alpha":          mtd_alpha,
        "active_risk":        active_risk,
        "rebalance_due":      rebalance_date.strftime("%b %d"),
        "days_to_rebalance":  days_to_rebalance,
        "last_score_run":     last_score_run,
        "top_ticker":         top_ticker,
        "top_score":          top_score,
        "holdings_count":     len(holdings),
        "data_date":          data_date,
        "alpaca_ok":          alpaca_ok,
    }


def _empty_response(days_to_rebalance: int, rebalance_date: date, db: Session) -> dict:
    last_run = (
        db.query(models.DiscoveryRun)
        .filter(models.DiscoveryRun.status.in_(["complete", "complete_with_warnings"]))
        .order_by(models.DiscoveryRun.run_date.desc())
        .first()
    )
    return {
        "portfolio_value":   None,
        "mtd_return":        None,
        "mtd_alpha":         None,
        "active_risk":       None,
        "rebalance_due":     rebalance_date.strftime("%b %d"),
        "days_to_rebalance": days_to_rebalance,
        "last_score_run":    last_run.run_date.isoformat() if last_run else None,
        "top_ticker":        None,
        "top_score":         None,
        "holdings_count":    0,
        "data_date":         None,
        "alpaca_ok":         False,
    }
