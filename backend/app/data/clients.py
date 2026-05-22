"""
app/data/alpaca_client.py
─────────────────────────────────────────────────────────────────────────────
Alpaca Markets price/OHLCV client.
Used by: Technical + Entropy strategies.
Failure impact: blocks both strategies; Fundamental can still run.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed — price data unavailable")


class AlpacaDataError(Exception):
    """Raised when Alpaca API is unavailable — blocks Technical + Entropy strategies."""
    pass


class AlpacaClient:
    def __init__(self):
        api_key = os.environ.get("ALPACA_API_KEY")
        secret  = os.environ.get("ALPACA_SECRET_KEY")
        if not (api_key and secret) or not ALPACA_AVAILABLE:
            self._client = None
            logger.warning("Alpaca credentials not set or alpaca-py not installed.")
        else:
            self._client = StockHistoricalDataClient(api_key=api_key, secret_key=secret)

    # Tickers Alpaca's IEX feed cannot handle — filtered before the batch call
    _ALPACA_UNSUPPORTED = {"BRK-B", "BRK-A", "BRK.B", "BRK.A", "BF-B", "BF.B"}

    def get_ohlcv(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for tickers in [start, end].

        Returns DataFrame: date, ticker, open, high, low, close, volume
        Raises AlpacaDataError if client is unavailable.
        """
        if self._client is None:
            raise AlpacaDataError("Alpaca client not initialized — check ALPACA_API_KEY + ALPACA_SECRET_KEY")

        # Filter out tickers known to cause Alpaca 400 errors before the batch call
        clean_tickers = [t for t in tickers if t not in self._ALPACA_UNSUPPORTED]
        skipped = [t for t in tickers if t in self._ALPACA_UNSUPPORTED]
        if skipped:
            logger.warning(f"Alpaca: skipping unsupported tickers {skipped}")

        if not clean_tickers:
            raise AlpacaDataError("No valid tickers after filtering unsupported symbols")

        try:
            req = StockBarsRequest(
                symbol_or_symbols=clean_tickers,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=start,
                end=end,
                adjustment="all",
                feed="iex",   # IEX feed — available on free/paper accounts; SIP requires paid subscription
            )
            bars = self._client.get_stock_bars(req).df
            bars = bars.reset_index()
            # Rename columns to match internal schema
            bars = bars.rename(columns={
                "symbol": "ticker",
                "timestamp": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            })
            bars["date"] = pd.to_datetime(bars["date"]).dt.date
            return bars[["date", "ticker", "open", "high", "low", "close", "volume"]]

        except Exception as e:
            logger.error(f"Alpaca fetch error: {e}", exc_info=True)
            raise AlpacaDataError(f"Alpaca data fetch failed: {e}") from e


# ────────────────────────────────────────────────────────────────────────────
# app/data/alphavantage.py
# ────────────────────────────────────────────────────────────────────────────
"""
Alpha Vantage quarterly fundamentals client.
Used by: Fundamental strategy only.
Failure impact: blocks only Fundamental; Technical + Entropy unaffected.
"""
import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime

logger_av = logging.getLogger("alphavantage")


class AlphaVantageError(Exception):
    """Raised when Alpha Vantage API fails — blocks Fundamental strategy only."""
    pass


def _safe_float(val) -> float:
    """Convert AV field to float — handles None, 'None', '', and missing."""
    if val is None or val == "None" or val == "":
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


class AlphaVantageClient:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self):
        self.api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            logger_av.warning("ALPHA_VANTAGE_API_KEY not set — fundamental data unavailable.")

    def get_income_statement(self, ticker: str) -> pd.DataFrame:
        """
        Fetch quarterly income statements for a single ticker.
        Timeout: 30s with one retry on transient failures.
        """
        if not self.api_key:
            raise AlphaVantageError("Alpha Vantage API key not configured")

        for attempt in range(2):
            try:
                resp = requests.get(
                    self.BASE_URL,
                    params={
                        "function": "INCOME_STATEMENT",
                        "symbol": ticker,
                        "apikey": self.api_key,
                    },
                    timeout=30,  # increased from 15s
                )
                resp.raise_for_status()
                data = resp.json()

                if "quarterlyReports" not in data:
                    error_msg = data.get("Note") or data.get("Information") or "Unknown error"
                    raise AlphaVantageError(f"Alpha Vantage error for {ticker}: {error_msg}")

                rows = []
                for q in data["quarterlyReports"]:
                    revenue    = _safe_float(q.get("totalRevenue"))
                    op_income  = _safe_float(q.get("operatingIncome"))
                    net_income = _safe_float(q.get("netIncome"))
                    op_margin  = float(np.clip(op_income / revenue, -10, 10)) if revenue != 0 else 0.0
                    net_margin = float(np.clip(net_income / revenue, -10, 10)) if revenue != 0 else 0.0
                    rows.append({
                        "ticker":           ticker,
                        "period_date":      pd.to_datetime(q["fiscalDateEnding"]),
                        "revenue":          revenue,
                        "operating_income": op_income,
                        "net_income":       net_income,
                        "operating_margin": op_margin,
                        "net_margin":       net_margin,
                    })

                df = pd.DataFrame(rows)
                return df.sort_values("period_date")

            except AlphaVantageError:
                raise
            except Exception as e:
                if attempt == 0:
                    logger_av.warning(f"AV timeout for {ticker}, retrying in 5s: {e}")
                    time.sleep(5)
                    continue
                logger_av.error(f"Alpha Vantage request failed for {ticker}: {e}", exc_info=True)
                raise AlphaVantageError(f"Alpha Vantage fetch failed: {e}") from e

    def get_fundamentals_batch(
        self,
        tickers: list[str],
        delay_sec: float = 12.0,
    ) -> pd.DataFrame:
        """
        Fetch quarterly fundamentals for all tickers.
        Applies a delay between requests to respect free-tier rate limits.
        """
        frames = []
        for ticker in tickers:
            try:
                df = self.get_income_statement(ticker)
                frames.append(df)
                logger_av.info(f"Alpha Vantage: fetched {ticker} ({len(df)} quarters)")
            except AlphaVantageError as e:
                logger_av.warning(f"Skipping {ticker}: {e}")
            time.sleep(delay_sec)  # AV free tier: 5 calls/min

        if not frames:
            raise AlphaVantageError("No fundamental data fetched")
        return pd.concat(frames, ignore_index=True)

    def get_earnings_transcript(self, ticker: str, quarters: int = 2) -> str:
        """
        Fetch the most recent earnings call transcript(s) via AV EARNINGS_CALL_TRANSCRIPT.

        Returns a formatted string ready for Claude's earnings_context slot.
        Falls back to empty string on any error — never blocks the scoring pipeline.

        AV returns transcripts quarterly. We fetch the last `quarters` transcripts
        and concatenate them so Claude sees management commentary across 2 periods.
        """
        if not self.api_key:
            return ""

        texts = []
        # AV transcript endpoint requires year + quarter parameters
        # We try the most recent 2 quarters using current date
        from datetime import datetime
        now = datetime.utcnow()
        # Build (year, quarter) pairs for the last `quarters` periods
        periods = []
        y, q = now.year, (now.month - 1) // 3 + 1
        for _ in range(quarters):
            periods.append((y, q))
            q -= 1
            if q == 0:
                q = 4
                y -= 1

        for year, quarter in periods:
            try:
                resp = requests.get(
                    self.BASE_URL,
                    params={
                        "function": "EARNINGS_CALL_TRANSCRIPT",
                        "symbol":   ticker,
                        "year":     year,
                        "quarter":  quarter,
                        "apikey":   self.api_key,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

                # Handle rate limit / permission messages
                if "Information" in data or "Note" in data:
                    break

                transcript = data.get("transcript", [])
                if not transcript:
                    continue

                # Transcript is a list of {speaker, title, content} dicts
                header = f"\n=== EARNINGS CALL Q{quarter} {year} ===\n"
                body = "\n".join(
                    f"[{t.get('title', t.get('speaker', 'Speaker'))}] {t.get('content', '')}"
                    for t in transcript
                    if t.get("content")
                )
                if body:
                    texts.append(header + body)

                time.sleep(1.0)  # Respect rate limits

            except Exception as e:
                logger_av.debug(f"Transcript fetch failed {ticker} Q{quarter}/{year}: {e}")
                continue

        if not texts:
            logger_av.debug(f"No transcript available for {ticker}")
            return ""

        result = "\n\n".join(texts)
        logger_av.info(f"AV transcript: {ticker} — {len(result):,} chars from {len(texts)} quarter(s)")
        return result

    def get_news_sentiment(self, ticker: str, limit: int = 20) -> str:
        """
        Fetch recent news articles with sentiment scores via AV NEWS_SENTIMENT.

        Returns a formatted string for Claude's news_context slot.
        Includes: headline, source, relevance score, sentiment label, and summary.
        Filters to articles with relevance score > 0.3 for the specific ticker.
        """
        if not self.api_key:
            return ""

        try:
            resp = requests.get(
                self.BASE_URL,
                params={
                    "function":   "NEWS_SENTIMENT",
                    "tickers":    ticker,
                    "limit":      limit,
                    "sort":       "LATEST",
                    "apikey":     self.api_key,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            if "Information" in data or "Note" in data:
                return ""

            feed = data.get("feed", [])
            if not feed:
                return ""

            lines = [f"=== RECENT NEWS & SENTIMENT ({ticker}) ===\n"]
            for article in feed[:limit]:
                # Find this ticker's relevance and sentiment in the ticker_sentiments list
                ticker_sentiment = next(
                    (ts for ts in article.get("ticker_sentiment", [])
                     if ts.get("ticker") == ticker),
                    None,
                )
                relevance = float(ticker_sentiment.get("relevance_score", 0)) if ticker_sentiment else 0
                if relevance < 0.3:
                    continue  # Skip low-relevance articles

                sentiment_label = ticker_sentiment.get("ticker_sentiment_label", "Neutral") if ticker_sentiment else "Neutral"
                sentiment_score = float(ticker_sentiment.get("ticker_sentiment_score", 0)) if ticker_sentiment else 0

                title   = article.get("title", "")
                source  = article.get("source", "")
                summary = article.get("summary", "")[:400]  # Trim long summaries
                time_pub = article.get("time_published", "")[:8]  # YYYYMMDD

                lines.append(
                    f"[{time_pub} | {source} | Relevance: {relevance:.2f} | Sentiment: {sentiment_label} ({sentiment_score:+.2f})]\n"
                    f"  {title}\n"
                    f"  {summary}\n"
                )

            if len(lines) == 1:  # Only the header — no relevant articles
                return ""

            result = "\n".join(lines)
            logger_av.info(f"AV news: {ticker} — {len(lines)-1} relevant articles")
            return result

        except Exception as e:
            logger_av.debug(f"News sentiment fetch failed {ticker}: {e}")
            return ""

    def get_earnings_history(self, ticker: str) -> str:
        """
        Fetch EPS and revenue surprise history via AV EARNINGS.

        Returns a formatted string showing the last 8 quarters of:
        - Reported vs estimated EPS
        - Surprise % (beat/miss)
        - Revenue reported vs estimated

        This gives Claude the "earnings quality" context the paper highlights:
        consistent beats signal management credibility; misses signal execution risk.
        """
        if not self.api_key:
            return ""

        try:
            resp = requests.get(
                self.BASE_URL,
                params={
                    "function": "EARNINGS",
                    "symbol":   ticker,
                    "apikey":   self.api_key,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            if "Information" in data or "Note" in data:
                return ""

            quarterly = data.get("quarterlyEarnings", [])
            if not quarterly:
                return ""

            lines = [f"=== EARNINGS HISTORY ({ticker}) — Last 8 Quarters ===\n"]
            beats = 0
            for q in quarterly[:8]:
                date       = q.get("fiscalDateEnding", "")
                reported   = q.get("reportedEPS", "None")
                estimated  = q.get("estimatedEPS", "None")
                surprise   = q.get("surprisePercentage", "None")
                try:
                    surp_f = float(surprise)
                    direction = "BEAT" if surp_f > 0 else "MISS"
                    surp_str  = f"{surp_f:+.1f}% ({direction})"
                    if surp_f > 0:
                        beats += 1
                except (ValueError, TypeError):
                    surp_str = "N/A"

                lines.append(
                    f"  {date}: EPS reported={reported} vs estimated={estimated} | Surprise: {surp_str}"
                )

            # Summary line — management credibility signal
            shown = len([q for q in quarterly[:8]])
            if shown > 0:
                lines.append(f"\n  Beat rate last {shown} quarters: {beats}/{shown} ({100*beats//shown}%)")

            result = "\n".join(lines)
            logger_av.info(f"AV earnings: {ticker} — {shown} quarters of EPS history")
            return result

        except Exception as e:
            logger_av.debug(f"Earnings history fetch failed {ticker}: {e}")
            return ""

    def get_earnings_transcript(self, ticker: str, quarters: int = 2) -> str:
        """
        Fetch the most recent earnings call transcript(s) via AV EARNINGS_CALL_TRANSCRIPT.

        Correct parameter format: quarter=2025Q1 (not year + quarter separately).
        Returns a formatted string for Claude's earnings_context slot.
        """
        if not self.api_key:
            return ""

        texts = []
        now = datetime.utcnow()
        # Build quarter strings in format "2025Q1", "2024Q4", etc.
        y, q = now.year, (now.month - 1) // 3 + 1
        periods = []
        for _ in range(quarters):
            periods.append(f"{y}Q{q}")
            q -= 1
            if q == 0:
                q = 4
                y -= 1

        for quarter_str in periods:
            try:
                resp = requests.get(
                    self.BASE_URL,
                    params={
                        "function": "EARNINGS_CALL_TRANSCRIPT",
                        "symbol":   ticker,
                        "quarter":  quarter_str,
                        "apikey":   self.api_key,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

                if "Information" in data or "Note" in data:
                    logger_av.debug(f"Transcript rate limit for {ticker} {quarter_str}")
                    break

                transcript = data.get("transcript", [])
                if not transcript:
                    continue

                header = f"\n=== EARNINGS CALL {quarter_str} ===\n"
                body = "\n".join(
                    f"[{t.get('title', t.get('speaker', 'Speaker'))}] {t.get('content', '')}"
                    for t in transcript
                    if t.get("content")
                )
                if body:
                    texts.append(header + body)
                    logger_av.info(f"AV transcript: {ticker} {quarter_str} — {len(body):,} chars")

                time.sleep(1.0)

            except Exception as e:
                logger_av.debug(f"Transcript fetch failed {ticker} {quarter_str}: {e}")
                continue

        if not texts:
            return ""

        return "\n\n".join(texts)

    def get_enriched_llm_context(
        self,
        ticker: str,
        delay_sec: float = 1.0,
    ) -> dict[str, str]:
        """
        Fetch Phase A + B + C enrichment signals for Claude LLM scoring.

        Phase A: earnings transcript, news sentiment, EPS surprise history
        Phase B: company overview (valuation + analyst target), balance sheet, cash flow
        Phase C: insider transactions (Form 4), institutional holdings

        Never raises — all failures return empty strings.
        """
        transcript       = self.get_earnings_transcript(ticker)
        time.sleep(delay_sec)
        news             = self.get_news_sentiment(ticker)
        time.sleep(delay_sec)
        earnings_history = self.get_earnings_history(ticker)
        time.sleep(delay_sec)
        overview         = self.get_company_overview(ticker)
        time.sleep(delay_sec)
        balance_sheet    = self.get_balance_sheet(ticker)
        time.sleep(delay_sec)
        cash_flow        = self.get_cash_flow(ticker)
        time.sleep(delay_sec)
        insider          = self.get_insider_transactions(ticker)
        time.sleep(delay_sec)
        institutional    = self.get_institutional_holdings(ticker)
        time.sleep(delay_sec)

        return {
            "transcript":       transcript,
            "news":             news,
            "earnings_history": earnings_history,
            "overview":         overview,
            "balance_sheet":    balance_sheet,
            "cash_flow":        cash_flow,
            "insider":          insider,
            "institutional":    institutional,
        }

    # ── Phase B methods ────────────────────────────────────────────────────

    def _av_get(self, params: dict, timeout: int = 20) -> dict | None:
        """
        Thin helper: GET to AV base URL, return parsed JSON or None on any error.
        Handles rate-limit messages gracefully.
        """
        if not self.api_key:
            return None
        try:
            resp = requests.get(
                self.BASE_URL,
                params={**params, "apikey": self.api_key},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if "Information" in data or "Note" in data:
                logger_av.debug(f"AV rate limit for {params.get('symbol', '?')}: {params.get('function')}")
                return None
            return data
        except Exception as e:
            logger_av.debug(f"AV {params.get('function')} failed for {params.get('symbol', '?')}: {e}")
            return None

    def get_company_overview(self, ticker: str) -> str:
        """
        Fetch company overview: valuation multiples, analyst target, market cap.

        Returns a formatted string for Claude's overview_context slot.
        Key signals for Claude:
          - Analyst target price vs current: upside/downside framing
          - P/E, EV/EBITDA: valuation relative to sector
          - Dividend yield: income profile
          - 52-week range: momentum context
        """
        data = self._av_get({"function": "OVERVIEW", "symbol": ticker})
        if not data or "Symbol" not in data:
            return ""

        def fmt(key: str, prefix: str = "", suffix: str = "") -> str:
            v = data.get(key, "N/A")
            if v in ("None", "0", "", None):
                return "N/A"
            return f"{prefix}{v}{suffix}"

        lines = [
            f"=== COMPANY OVERVIEW ({ticker}) ===",
            f"Name:               {data.get('Name', 'N/A')}",
            f"Sector / Industry:  {data.get('Sector', 'N/A')} / {data.get('Industry', 'N/A')}",
            f"Market Cap:         {fmt('MarketCapitalization', '$')}",
            f"",
            f"--- Valuation ---",
            f"P/E (TTM):          {fmt('TrailingPE')}",
            f"Forward P/E:        {fmt('ForwardPE')}",
            f"Price/Sales:        {fmt('PriceToSalesRatioTTM')}",
            f"Price/Book:         {fmt('PriceToBookRatio')}",
            f"EV/EBITDA:          {fmt('EVToEBITDA')}",
            f"EV/Revenue:         {fmt('EVToRevenue')}",
            f"",
            f"--- Analyst Consensus ---",
            f"Analyst Target:     {fmt('AnalystTargetPrice', '$')}",
            f"Analyst Count:      {fmt('AnalystRatingStrongBuy')} strong buy | {fmt('AnalystRatingBuy')} buy | {fmt('AnalystRatingHold')} hold | {fmt('AnalystRatingSell')} sell",
            f"",
            f"--- Price Context ---",
            f"52-Week High:       {fmt('52WeekHigh', '$')}",
            f"52-Week Low:        {fmt('52WeekLow', '$')}",
            f"50-Day MA:          {fmt('50DayMovingAverage', '$')}",
            f"200-Day MA:         {fmt('200DayMovingAverage', '$')}",
            f"",
            f"--- Income & Growth ---",
            f"Dividend Yield:     {fmt('DividendYield', suffix='%')}",
            f"EPS (TTM):          {fmt('EPS', '$')}",
            f"Revenue/Share:      {fmt('RevenuePerShareTTM', '$')}",
            f"Revenue Growth YoY: {fmt('RevenueGrowthYOY', suffix='%')}",
            f"Earnings Growth YoY:{fmt('EarningsGrowthYOY', suffix='%')}",
            f"Profit Margin:      {fmt('ProfitMargin', suffix='%')}",
            f"Operating Margin:   {fmt('OperatingMarginTTM', suffix='%')}",
            f"ROE:                {fmt('ReturnOnEquityTTM', suffix='%')}",
            f"ROA:                {fmt('ReturnOnAssetsTTM', suffix='%')}",
            f"Beta:               {fmt('Beta')}",
        ]

        result = "\n".join(lines)
        logger_av.info(f"AV overview: {ticker} — {len(result):,} chars")
        return result

    def get_balance_sheet(self, ticker: str) -> str:
        """
        Fetch the last 4 quarters of balance sheet data.

        Key signals for Claude:
          - Debt/equity: leverage risk
          - Current ratio: liquidity
          - Cash & equivalents: financial flexibility
          - Goodwill/intangibles: acquisition premium risk
        """
        data = self._av_get({"function": "BALANCE_SHEET", "symbol": ticker}, timeout=30)
        if not data or "quarterlyReports" not in data:
            return ""

        reports = data["quarterlyReports"][:4]  # Last 4 quarters
        if not reports:
            return ""

        lines = [f"=== BALANCE SHEET ({ticker}) — Last 4 Quarters ===\n"]
        for q in reports:
            date         = q.get("fiscalDateEnding", "")
            total_assets = _safe_float(q.get("totalAssets"))
            total_liab   = _safe_float(q.get("totalLiabilities"))
            total_equity = _safe_float(q.get("totalShareholderEquity"))
            cash         = _safe_float(q.get("cashAndCashEquivalentsAtCarryingValue"))
            short_debt   = _safe_float(q.get("shortTermDebt") or q.get("shortLongTermDebtTotal"))
            long_debt    = _safe_float(q.get("longTermDebt"))
            total_debt   = (short_debt or 0) + (long_debt or 0)
            current_assets = _safe_float(q.get("totalCurrentAssets"))
            current_liab   = _safe_float(q.get("totalCurrentLiabilities"))
            goodwill     = _safe_float(q.get("goodwill"))

            # Derived ratios
            debt_equity  = round(total_debt / total_equity, 2) if total_equity and total_equity != 0 else None
            current_ratio = round(current_assets / current_liab, 2) if current_liab and current_liab != 0 else None

            def m(v):  # Format as $M
                return f"${v/1e6:.0f}M" if v else "N/A"

            lines.append(
                f"  {date}: Assets={m(total_assets)} | Liabilities={m(total_liab)} | Equity={m(total_equity)}\n"
                f"    Cash={m(cash)} | Total Debt={m(total_debt)} | D/E={debt_equity} | "
                f"Current Ratio={current_ratio} | Goodwill={m(goodwill)}"
            )

        result = "\n".join(lines)
        logger_av.info(f"AV balance sheet: {ticker} — {len(reports)} quarters")
        return result

    def get_cash_flow(self, ticker: str) -> str:
        """
        Fetch the last 4 quarters of cash flow data.

        Key signals for Claude:
          - Free cash flow = operating CF - capex: quality of earnings signal
          - Operating CF vs net income: cash conversion quality
          - Share buybacks: capital allocation signal
          - Capex trend: investment cycle phase
        """
        data = self._av_get({"function": "CASH_FLOW", "symbol": ticker}, timeout=30)
        if not data or "quarterlyReports" not in data:
            return ""

        reports = data["quarterlyReports"][:4]
        if not reports:
            return ""

        lines = [f"=== CASH FLOW ({ticker}) — Last 4 Quarters ===\n"]
        for q in reports:
            date       = q.get("fiscalDateEnding", "")
            op_cf      = _safe_float(q.get("operatingCashflow"))
            capex      = _safe_float(q.get("capitalExpenditures"))
            net_income = _safe_float(q.get("netIncome"))
            buybacks   = _safe_float(q.get("repurchaseOfCommonStock"))
            dividends  = _safe_float(q.get("dividendPayout"))

            # Free cash flow
            fcf = (op_cf or 0) - abs(capex or 0) if op_cf is not None else None

            # Cash conversion ratio
            ccr = round(op_cf / net_income, 2) if (net_income and net_income != 0 and op_cf) else None

            def m(v):
                return f"${v/1e6:.0f}M" if v else "N/A"
            def sign_m(v):  # Buybacks reported negative in AV
                if v is None: return "N/A"
                return f"${abs(v)/1e6:.0f}M"

            lines.append(
                f"  {date}: Op CF={m(op_cf)} | Capex={m(capex)} | FCF={m(fcf)}\n"
                f"    Net Income={m(net_income)} | Cash Conversion={ccr} | "
                f"Buybacks={sign_m(buybacks)} | Dividends={sign_m(dividends)}"
            )

        result = "\n".join(lines)
        logger_av.info(f"AV cash flow: {ticker} — {len(reports)} quarters")
        return result

    def get_insider_transactions(self, ticker: str, lookback_days: int = 180) -> str:
        """
        Fetch recent insider transactions via AV INSIDER_TRANSACTIONS.

        Returns a formatted string for Claude's insider_context slot.
        Covers the last `lookback_days` of SEC Form 4 filings.

        Key signals for Claude:
          - CEO/CFO buying: strongest conviction signal (they have most to lose)
          - Cluster buying: multiple insiders buying simultaneously is very bullish
          - Mass selling: bearish, especially if concentrated near guidance periods
          - Option exercises followed by immediate sale: routine, low signal
          - Planned 10b5-1 sales: scheduled in advance, lower signal than discretionary
        """
        data = self._av_get({"function": "INSIDER_TRANSACTIONS", "symbol": ticker})
        if not data:
            return ""

        # AV returns list under 'data' key
        transactions = data.get("data", [])
        if not transactions:
            return ""

        from datetime import datetime, timedelta
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)

        # Filter to recent transactions and parse
        recent = []
        for t in transactions:
            try:
                date_str = t.get("transactionDate", "")
                if not date_str:
                    continue
                tx_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                if tx_date < cutoff:
                    continue

                tx_type    = t.get("transactionType", "")
                shares     = _safe_float(t.get("shares"))
                price      = _safe_float(t.get("sharePrice"))
                value      = (shares or 0) * (price or 0)
                name       = t.get("executiveName", "Unknown")
                title      = t.get("executiveTitle", "")
                plan_10b51 = t.get("plan10b51", "false").lower() == "true"

                recent.append({
                    "date":      date_str[:10],
                    "name":      name,
                    "title":     title,
                    "type":      tx_type,
                    "shares":    shares,
                    "price":     price,
                    "value":     value,
                    "plan10b51": plan_10b51,
                })
            except Exception:
                continue

        if not recent:
            return ""

        # Sort by date descending, cap at 20 most recent
        recent.sort(key=lambda x: x["date"], reverse=True)
        recent = recent[:20]

        # Summarise buy/sell sentiment
        buys  = [t for t in recent if "P" in t["type"] or "A" in t["type"]]  # Purchase / Award
        sells = [t for t in recent if "S" in t["type"]]                      # Sale

        buy_value  = sum(t["value"] for t in buys  if t["value"])
        sell_value = sum(t["value"] for t in sells if t["value"])

        lines = [
            f"=== INSIDER TRANSACTIONS ({ticker}) — Last {lookback_days} days ===",
            f"Summary: {len(buys)} buys (${buy_value/1e3:.0f}K total) | "
            f"{len(sells)} sells (${sell_value/1e3:.0f}K total)\n",
        ]

        for t in recent:
            plan_flag = " [10b5-1 plan]" if t["plan10b51"] else ""
            val_str   = f"${t['value']/1e3:.0f}K" if t["value"] else "N/A"
            lines.append(
                f"  {t['date']} | {t['name']} ({t['title']}) | "
                f"{t['type']}{plan_flag} | "
                f"{int(t['shares'] or 0):,} shares @ ${t['price'] or 'N/A'} = {val_str}"
            )

        result = "\n".join(lines)
        logger_av.info(f"AV insider: {ticker} — {len(recent)} transactions")
        return result

    def get_institutional_holdings(self, ticker: str) -> str:
        """
        Fetch institutional holdings summary via AV INSTITUTIONAL_HOLDINGS.

        Returns a formatted string for Claude's institutional_context slot.

        Key signals for Claude:
          - % institutional ownership: high = sophisticated money consensus
          - QoQ change in ownership: increasing = accumulation; decreasing = distribution
          - Number of holders: breadth of institutional conviction
          - Top holders: if Berkshire/Sequoia/index funds dominate = different signal
            than if hedge funds dominate (more tactical)
          - Recent buyers vs sellers: net flow direction
        """
        data = self._av_get({"function": "INSTITUTIONAL_HOLDINGS", "symbol": ticker})
        if not data:
            return ""

        # AV returns ownership_summary + institutional_ownership list
        summary = data.get("ownership_summary", {})
        holders = data.get("institutional_ownership", [])

        if not summary and not holders:
            return ""

        lines = [f"=== INSTITUTIONAL HOLDINGS ({ticker}) ===\n"]

        # Summary block
        if summary:
            inst_pct     = summary.get("institutionalSharesHeldPercent", "N/A")
            total_shares = _safe_float(summary.get("totalSharesOutstanding"))
            inst_shares  = _safe_float(summary.get("institutionalSharesHeld"))
            num_holders  = summary.get("numberOfInstitutionalShareHolders", "N/A")
            qoq_change   = summary.get("quarterlyChange", "N/A")

            lines.append(
                f"Institutional Ownership: {inst_pct}%\n"
                f"Institutional Shares:    {inst_shares/1e6:.1f}M of {total_shares/1e6:.1f}M outstanding\n"
                f"Number of Holders:       {num_holders}\n"
                f"QoQ Change:              {qoq_change}%\n"
            )

        # Top 10 holders
        if holders:
            lines.append("--- Top Institutional Holders ---")
            for h in holders[:10]:
                name         = h.get("name", "Unknown")
                shares       = _safe_float(h.get("sharesHeld"))
                shares_chg   = _safe_float(h.get("changeInShares"))
                pct_portfolio = h.get("portfolioPercent", "")
                date_reported = h.get("reportDate", "")

                chg_str = ""
                if shares_chg is not None:
                    direction = "▲" if shares_chg > 0 else "▼"
                    chg_str   = f" ({direction}{abs(shares_chg)/1e6:.1f}M shares QoQ)"

                lines.append(
                    f"  {name}: {shares/1e6:.1f}M shares{chg_str} "
                    f"({pct_portfolio}% of portfolio, reported {date_reported})"
                )

        result = "\n".join(lines)
        logger_av.info(f"AV institutional: {ticker} — {len(holders)} holders")
        return result
# ────────────────────────────────────────────────────────────────────────────
"""
SEC EDGAR API client — 10-K, 10-Q, 8-K filings.
Used by: Claude LLM scoring (all three strategies need this for LLM context).
Failure impact: falls back to w=1.0 (pure ML) for ALL strategies.
"""
import re
import os
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger_edgar = logging.getLogger("edgar")

EDGAR_BASE  = "https://data.sec.gov"
EDGAR_FULL  = "https://efts.sec.gov/LATEST/search-index"
USER_AGENT  = os.environ.get("EDGAR_USER_AGENT", "ai-portfolio-platform research@alphalens.io")


class EDGARError(Exception):
    """EDGAR unavailable — all strategies fall back to w=1.0."""
    pass


# Cache company_tickers.json to avoid refetching on every ticker
_COMPANY_TICKERS_CACHE: dict = {}

# Form types for US domestic companies
US_FORM_TYPES = ["10-K", "10-Q", "8-K"]
# Form types for foreign private issuers (BABA, NVO, JD, GGAL, NU, YPF etc.)
FOREIGN_FORM_TYPES = ["20-F", "6-K", "40-F"]
# All form types to try
ALL_FORM_TYPES = US_FORM_TYPES + FOREIGN_FORM_TYPES

# Tickers known to be ETFs/funds with no individual company filings
ETF_TICKERS = {
    "TLT", "SPY", "QQQ", "IWM", "VTI", "GLD", "SLV", "USO",
    "QCLN", "IHI", "IBIT", "ETHA", "ARKK", "XLK", "XLF",
}


class EDGARClient:
    """
    SEC EDGAR filing fetcher.
    Retrieves 10-K / 10-Q / 20-F / 6-K text for Claude LLM context.
    Handles both US domestic companies and foreign private issuers.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})

    def _load_company_tickers(self) -> dict:
        """Load and cache the SEC company tickers JSON (fetched once per process)."""
        global _COMPANY_TICKERS_CACHE
        if _COMPANY_TICKERS_CACHE:
            return _COMPANY_TICKERS_CACHE
        try:
            resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": USER_AGENT},
                timeout=15,
            )
            resp.raise_for_status()
            _COMPANY_TICKERS_CACHE = resp.json()
            logger_edgar.info(f"Loaded {len(_COMPANY_TICKERS_CACHE)} company tickers from SEC")
        except Exception as e:
            logger_edgar.warning(f"Could not load company_tickers.json: {e}")
        return _COMPANY_TICKERS_CACHE

    def get_cik(self, ticker: str) -> str:
        """
        Look up company CIK by ticker symbol using SEC company_tickers.json.
        Returns zero-padded 10-digit CIK string.
        """
        if ticker.upper() in ETF_TICKERS:
            raise EDGARError(f"{ticker} is an ETF — no individual company filing on EDGAR")

        company_data = self._load_company_tickers()
        for val in company_data.values():
            if val.get("ticker", "").upper() == ticker.upper():
                return str(val["cik_str"]).zfill(10)

        # Fallback: try BRK.B → BRK-B or strip suffix
        clean = ticker.replace(".", "-").split("-")[0]
        if clean != ticker:
            for val in company_data.values():
                if val.get("ticker", "").upper() == clean.upper():
                    return str(val["cik_str"]).zfill(10)

        raise EDGARError(f"CIK not found for {ticker}")

    def get_recent_filings(
        self,
        ticker: str,
        n: int = 3,
    ) -> list[dict]:
        """
        Get n most recent filings for a ticker.
        Tries US form types first (10-K, 10-Q, 8-K), then foreign types (20-F, 6-K).
        Returns list of {form_type, filing_date, accession, text_url}.
        """
        try:
            cik = self.get_cik(ticker)
            resp = self.session.get(
                f"{EDGAR_BASE}/submissions/CIK{cik}.json",
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            filings = data.get("filings", {}).get("recent", {})

            forms      = filings.get("form", [])
            dates      = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])

            results = []
            # Try all form types (US + foreign) to maximize coverage
            for form, date, acc in zip(forms, dates, accessions):
                if form in ALL_FORM_TYPES and len(results) < n:
                    acc_clean = acc.replace("-", "")
                    cik_int   = int(cik)
                    results.append({
                        "form_type":   form,
                        "filing_date": date,
                        "accession":   acc,
                        "text_url":    f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{acc}-index.htm",
                    })

            if not results:
                logger_edgar.warning(f"No filings found for {ticker} (CIK {cik}) in {ALL_FORM_TYPES}")
            return results

        except EDGARError:
            raise
        except Exception as e:
            raise EDGARError(f"EDGAR filings lookup failed for {ticker}: {e}") from e

    def get_filing_text(
        self,
        text_url: str,
        max_chars: int = 150_000,
    ) -> str:
        """
        Download and extract plain text from a filing.
        Returns truncated text suitable for Claude's 200K context window.
        """
        try:
            resp = self.session.get(text_url, timeout=30)
            resp.raise_for_status()
            # Minimal HTML strip
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_chars]
        except Exception as e:
            logger_edgar.warning(f"Failed to fetch filing text from {text_url}: {e}")
            return ""

    def get_filing_context(
        self,
        ticker: str,
        rebalance_date: datetime,
        max_chars: int = 150_000,
    ) -> str:
        """
        Build LLM context string from most recent available filings.
        Accepts filings up to 2 years old to ensure coverage for all company types.
        Returns empty string on failure (caller scores with w=1.0 fallback).
        """
        try:
            filings = self.get_recent_filings(ticker, n=3)
            if not filings:
                return ""

            # Accept filings within last 2 years — not strict exact timestamp
            cutoff = pd.Timestamp(rebalance_date) - pd.DateOffset(years=2)
            relevant = [f for f in filings if pd.to_datetime(f["filing_date"]) >= cutoff]

            # Fall back to whatever exists if nothing in 2-year window
            if not relevant:
                relevant = filings[:2]
                logger_edgar.info(f"Using older filings for {ticker}: {filings[0]['filing_date']}")

            sections = []
            per_filing_max = max_chars // max(1, len(relevant))
            for filing in relevant:
                text = self.get_filing_text(filing["text_url"], max_chars=per_filing_max)
                if text:
                    sections.append(f"[{filing['form_type']} | {filing['filing_date']}]\n{text}")
                time.sleep(0.1)

            result = "\n\n".join(sections)
            if result:
                logger_edgar.info(f"EDGAR: {ticker} — {len(relevant)} filing(s), {len(result)} chars")
            return result

        except EDGARError as e:
            logger_edgar.warning(f"EDGAR unavailable for {ticker}: {e}")
            return ""
        except Exception as e:
            logger_edgar.error(f"EDGAR context error for {ticker}: {e}", exc_info=True)
            return ""
