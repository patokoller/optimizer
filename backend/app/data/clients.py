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

        Returns DataFrame with columns:
            ticker, period_date, revenue, operating_income,
            net_income, operating_margin, net_margin
        """
        if not self.api_key:
            raise AlphaVantageError("Alpha Vantage API key not configured")

        try:
            resp = requests.get(
                self.BASE_URL,
                params={
                    "function": "INCOME_STATEMENT",
                    "symbol": ticker,
                    "apikey": self.api_key,
                },
                timeout=15,
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
                # Guard against division by zero and clip extreme margin values
                op_margin  = float(np.clip(op_income / revenue, -10, 10)) if revenue != 0 else 0.0
                net_margin = float(np.clip(net_income / revenue, -10, 10)) if revenue != 0 else 0.0

                rows.append({
                    "ticker": ticker,
                    "period_date": pd.to_datetime(q["fiscalDateEnding"]),
                    "revenue": revenue,
                    "operating_income": op_income,
                    "net_income": net_income,
                    "operating_margin": op_margin,
                    "net_margin": net_margin,
                })

            df = pd.DataFrame(rows)
            return df.sort_values("period_date")

        except AlphaVantageError:
            raise
        except Exception as e:
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

    def get_enriched_llm_context(
        self,
        ticker: str,
        delay_sec: float = 1.0,
    ) -> dict[str, str]:
        """
        Fetch Phase A enrichment signals: news sentiment + earnings history.

        NOTE: EARNINGS_CALL_TRANSCRIPT requires AV Premium plan above Plan 30.
        Transcript fetch is disabled — returns empty string until plan is upgraded.
        News and earnings history are available on Plan 30.

        Returns dict with keys: transcript, news, earnings_history.
        Never raises — all failures return empty strings.
        """
        # Transcript disabled — requires higher AV premium tier
        transcript = ""

        news = self.get_news_sentiment(ticker)
        time.sleep(delay_sec)
        earnings_history = self.get_earnings_history(ticker)
        time.sleep(delay_sec)

        return {
            "transcript":       transcript,
            "news":             news,
            "earnings_history": earnings_history,
        }


# ────────────────────────────────────────────────────────────────────────────
# app/data/edgar_client.py
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
