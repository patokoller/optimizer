"""
app/data/etf_client.py
────────────────────────────────────────────────────────────────────────────
ETF classification and holdings resolution.

Classifies portfolio tickers into:
  - EQUITY_ETF:  Equity basket (QCLN, IHI, XLK, etc.)
                 → fetch top 5 equity holdings, score them, average back
  - BOND_ETF:    Bond/fixed-income wrapper (TLT, AGG, BND)
                 → exclude from scoring, not scoreable via paper framework
  - CRYPTO_ETF:  Crypto spot ETF (IBIT, ETHA, ARKB)
                 → exclude from scoring
  - STOCK:       Individual operating company → score normally
  - UNKNOWN:     Cannot classify → treat as STOCK (attempt scoring)

Holdings data: Alpha Vantage ETF_PROFILE endpoint.
Same API key, same rate limits. One call per ETF.
"""
import os
import time
import logging
import requests
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("etf_client")

# ── Known-ETF overrides (classification without needing an API call) ─────────
# Keys: uppercase ticker. Values: (etf_type, description)
KNOWN_BOND_ETFS = {
    "TLT", "AGG", "BND", "IEF", "SHY", "GOVT", "LQD", "HYG", "JNK",
    "MUB", "TIP", "VTIP", "SCHZ", "BNDX", "EMB",
}

KNOWN_CRYPTO_ETFS = {
    "IBIT", "ETHA", "ARKB", "FBTC", "BITB", "HODL", "EZBC", "BTCW",
    "ETHW", "CETH", "ETHV", "QBTC",
}

# Tickers that simply don't exist as scoreable instruments
NON_SCOREABLE = {
    "JBCG",   # unrecognised / delisted
    "STRV",   # unrecognised / delisted
    "BRK.B",  # Alpaca rejects BRK-B ticker format; crashes entire batch request
    "BRK.A",  # same issue
}

# Known equity ETFs — these get top-5 holdings resolution via ETF_PROFILE
# Only add ETFs that are actually in user portfolios and need composite scoring
KNOWN_EQUITY_ETFS = {
    "QCLN", "IHI", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP",
    "XLB", "XLU", "XLRE", "SMH", "SOXX", "QQQ", "SPY", "IWM", "VTI",
    "ARKK", "ARKG", "ARKF", "ARKQ", "ARKW",
}

# BRK.B fix: Berkshire Hathaway uses a non-standard ticker — map to scoreable form
TICKER_NORMALISATION = {
    "BRK.B": "BRK-B",
    "BRK.A": "BRK-A",
    "BF.B":  "BF-B",
}


@dataclass
class ETFHolding:
    ticker: str
    weight: float        # as decimal (0.08 = 8%)
    description: str = ""


@dataclass
class TickerClassification:
    original_ticker: str
    resolved_ticker: str          # after normalisation
    etf_type: str                 # "STOCK" | "EQUITY_ETF" | "BOND_ETF" | "CRYPTO_ETF" | "NON_SCOREABLE"
    holdings: list[ETFHolding] = field(default_factory=list)
    error: Optional[str] = None


class ETFClient:
    """
    Classifies tickers and resolves ETF holdings via Alpha Vantage ETF_PROFILE.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self):
        self.api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")

    def classify(self, ticker: str) -> TickerClassification:
        """
        Classify a single ticker. Returns TickerClassification with etf_type set.
        For EQUITY_ETF, also fetches and returns top 5 equity holdings.
        """
        resolved = TICKER_NORMALISATION.get(ticker.upper(), ticker)

        # Fast-path: known non-scoreable
        if ticker.upper() in NON_SCOREABLE:
            return TickerClassification(
                original_ticker=ticker,
                resolved_ticker=resolved,
                etf_type="NON_SCOREABLE",
            )

        # Fast-path: known bond ETF
        if ticker.upper() in KNOWN_BOND_ETFS:
            return TickerClassification(
                original_ticker=ticker,
                resolved_ticker=resolved,
                etf_type="BOND_ETF",
            )

        # Fast-path: known crypto ETF
        if ticker.upper() in KNOWN_CRYPTO_ETFS:
            return TickerClassification(
                original_ticker=ticker,
                resolved_ticker=resolved,
                etf_type="CRYPTO_ETF",
            )

        # Fast-path: known equity ETF — fetch holdings via ETF_PROFILE
        if ticker.upper() in KNOWN_EQUITY_ETFS:
            if self.api_key:
                try:
                    result = self._fetch_etf_profile(ticker, resolved)
                    if result is not None:
                        return result
                except Exception as e:
                    logger.warning(f"ETF_PROFILE failed for {ticker}: {e} — treating as STOCK")
            return TickerClassification(
                original_ticker=ticker,
                resolved_ticker=resolved,
                etf_type="STOCK",
            )

        # Default: treat as individual stock — skip ETF_PROFILE API call entirely
        # This avoids wasting AV rate-limit quota on obvious stocks (GOOGL, MSFT, etc.)
        return TickerClassification(
            original_ticker=ticker,
            resolved_ticker=resolved,
            etf_type="STOCK",
        )

    def _fetch_etf_profile(self, original: str, resolved: str) -> Optional[TickerClassification]:
        """
        Call Alpha Vantage ETF_PROFILE. Returns classification if it's an ETF,
        None if it appears to be a regular stock (profile not found).
        """
        resp = requests.get(
            self.BASE_URL,
            params={
                "function": "ETF_PROFILE",
                "symbol":   resolved,
                "apikey":   self.api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        # If AV returns rate-limit note or no ETF data, treat as stock
        if "Information" in data or "Note" in data:
            return None
        if "holdings" not in data and "net_assets" not in data:
            return None

        # It's an ETF — classify by asset type
        asset_type = data.get("asset_allocation", {})
        holdings_raw = data.get("holdings", [])

        # Determine ETF category from allocation
        bond_pct  = _safe_pct(asset_type.get("bond_allocation"))
        cash_pct  = _safe_pct(asset_type.get("cash_allocation"))
        equity_pct = _safe_pct(asset_type.get("equity_allocation"))

        # Bond-heavy ETF
        if bond_pct > 50 or (bond_pct + cash_pct > 70):
            logger.info(f"{original}: BOND_ETF (bond={bond_pct:.0f}%)")
            return TickerClassification(
                original_ticker=original,
                resolved_ticker=resolved,
                etf_type="BOND_ETF",
            )

        # Equity-heavy ETF → extract top 5 equity holdings
        equity_holdings = []
        for h in holdings_raw:
            hticker = h.get("symbol", "").strip().upper()
            if not hticker or hticker in ("", "N/A", "-"):
                continue
            # Skip bond/cash positions within the ETF
            desc = h.get("description", "").lower()
            if any(w in desc for w in ["treasury", "bond", "note", "bill", "cash", "etf"]):
                continue
            weight = _safe_pct(h.get("weight")) / 100.0  # convert % to decimal
            equity_holdings.append(ETFHolding(
                ticker=hticker,
                weight=weight,
                description=h.get("description", ""),
            ))

        # Take top 5 by weight
        top5 = sorted(equity_holdings, key=lambda x: x.weight, reverse=True)[:5]

        if not top5:
            # Can't resolve equity holdings — exclude
            logger.warning(f"{original}: EQUITY_ETF but no equity holdings found — excluding")
            return TickerClassification(
                original_ticker=original,
                resolved_ticker=resolved,
                etf_type="NON_SCOREABLE",
                error="No equity holdings found in ETF_PROFILE",
            )

        logger.info(f"{original}: EQUITY_ETF — top holdings: {[h.ticker for h in top5]}")
        return TickerClassification(
            original_ticker=original,
            resolved_ticker=resolved,
            etf_type="EQUITY_ETF",
            holdings=top5,
        )

    def classify_batch(self, tickers: list[str]) -> dict[str, TickerClassification]:
        """
        Classify all tickers. Rate-limited: 1s between ETF_PROFILE calls.
        Returns {original_ticker: TickerClassification}.
        """
        results = {}
        for ticker in tickers:
            results[ticker] = self.classify(ticker)
            # Only AV calls need rate limiting
            if results[ticker].etf_type == "EQUITY_ETF":
                time.sleep(1.0)
        return results


def _safe_pct(val) -> float:
    if val is None or val == "None" or val == "":
        return 0.0
    try:
        s = str(val).replace("%", "").strip()
        return float(s)
    except (ValueError, TypeError):
        return 0.0
