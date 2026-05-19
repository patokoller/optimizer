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

        try:
            req = StockBarsRequest(
                symbol_or_symbols=tickers,
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


class EDGARClient:
    """
    SEC EDGAR filing fetcher.
    Retrieves 10-K / 10-Q / 8-K text for Claude LLM context.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})

    def get_cik(self, ticker: str) -> str:
        """Look up company CIK by ticker symbol."""
        try:
            resp = self.session.get(
                f"{EDGAR_BASE}/submissions/CIK.json",
                params={"action": "getcompany", "company": ticker, "type": "", "dateb": "", "owner": "include", "count": "1", "search_text": ""},
                timeout=10,
            )
            # Try the ticker lookup endpoint directly
            resp2 = self.session.get(
                "https://efts.sec.gov/LATEST/search-index?q=%22" + ticker + "%22&dateRange=custom&startdt=2023-01-01&enddt=2025-01-01&forms=10-K",
                timeout=10,
            )
            # Simpler: use company tickers JSON
            company_resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": USER_AGENT},
                timeout=10,
            )
            company_data = company_resp.json()
            for key, val in company_data.items():
                if val.get("ticker", "").upper() == ticker.upper():
                    return str(val["cik_str"]).zfill(10)
            raise EDGARError(f"CIK not found for {ticker}")
        except EDGARError:
            raise
        except Exception as e:
            raise EDGARError(f"EDGAR CIK lookup failed for {ticker}: {e}") from e

    def get_recent_filings(
        self,
        ticker: str,
        form_types: list[str] = ["10-K", "10-Q", "8-K"],
        n: int = 4,
    ) -> list[dict]:
        """
        Get n most recent filings of the given types.
        Returns list of {form_type, filing_date, description, text_url}.
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

            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])

            results = []
            for form, date, acc in zip(forms, dates, accessions):
                if form in form_types and len(results) < n:
                    acc_clean = acc.replace("-", "")
                    results.append({
                        "form_type": form,
                        "filing_date": date,
                        "accession": acc,
                        "text_url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{acc}-index.htm",
                    })
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
        Build LLM context string from most recent filings before rebalance_date.
        Returns empty string on failure (caller uses w=1.0 fallback).
        """
        try:
            filings = self.get_recent_filings(ticker)
            # Filter to filings before rebalance_date
            relevant = [
                f for f in filings
                if pd.to_datetime(f["filing_date"]) < pd.Timestamp(rebalance_date)
            ][:3]

            if not relevant:
                logger_edgar.warning(f"No recent filings before {rebalance_date} for {ticker}")
                return ""

            sections = []
            per_filing_max = max_chars // max(1, len(relevant))
            for filing in relevant:
                text = self.get_filing_text(filing["text_url"], max_chars=per_filing_max)
                if text:
                    sections.append(
                        f"[{filing['form_type']} | {filing['filing_date']}]\n{text}"
                    )
                time.sleep(0.1)  # Rate limiting per EDGAR robots.txt

            return "\n\n".join(sections)

        except EDGARError as e:
            logger_edgar.warning(f"EDGAR unavailable for {ticker}: {e}")
            return ""
        except Exception as e:
            logger_edgar.error(f"EDGAR context error for {ticker}: {e}", exc_info=True)
            return ""
