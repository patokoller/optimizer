"""
app/data/av_options.py
─────────────────────────────────────────────────────────────────────────────
Alpha Vantage HISTORICAL_OPTIONS → at-the-money implied-volatility feature.

Status: a ready, ISOLATED capability. It is NOT wired into the scoring models.
ATM IV is a signal the paper never used (Recommendation, outside the locked
benchmarks), so it must be measured by the validation harness before it is
allowed to influence any score.

Why ATM filtering is mandatory (confirmed against the live API):
raw per-contract `implied_volatility` is degenerate at the wings — a ~0.0149
floor on deep-ITM contracts and values up to ~10.0 (≈999%) on deep-OTM /
near-worthless strikes (delta pinned at 1.0, gamma/vega 0). Feeding the raw
field would be garbage. Restricting to liquid near-the-money contracts
(volume>0, 0.3<|delta|<0.7) and taking the median yields sane levels
(e.g. NVDA ~0.45, MSFT ~0.34, AAPL ~0.27).

Coverage note: options coverage is universe-wide, independent of the
EARNINGS_ESTIMATES mega-cap gap — it works on AAPL/MSFT/NVDA etc.

A single chain mixes expirations (term structure). v1 takes the median across
the |delta| band over all expirations, which is what was validated; an optional
`nearest_expirations` cap is exposed for later term-structure refinement.
IV *rank* requires a time series of these levels (store per run, rank downstream)
— a single snapshot only gives the level.
"""
from __future__ import annotations
import os
import logging
import statistics
import requests

from app.data.clients import _classify_av_response

logger = logging.getLogger("av_options")
AV_URL = "https://www.alphavantage.co/query"


def _f(val):
    """Parse an AV string field to float, or None if missing/non-numeric."""
    if val is None or val == "" or val == ".":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def extract_atm_iv(
    contracts: list[dict],
    vol_min: float = 1.0,
    delta_lo: float = 0.30,
    delta_hi: float = 0.70,
    nearest_expirations: int | None = None,
) -> dict:
    """Median ATM implied volatility from an options chain.

    Filters to liquid near-the-money contracts (volume >= vol_min and
    delta_lo < |delta| < delta_hi), which excludes the degenerate-IV wings.

    Returns a dict with:
      atm_iv        median IV of the filtered set, or None if none qualify
      n_atm         number of contracts used
      n_total       total contracts in the chain
      raw_iv_min/max  range of raw IV across the whole chain (diagnostic — shows
                      the wing degeneracy the filter is removing)
    Never raises; returns atm_iv=None on empty/garbage input.
    """
    out = {"atm_iv": None, "n_atm": 0, "n_total": 0,
           "raw_iv_min": None, "raw_iv_max": None}
    if not contracts:
        return out
    out["n_total"] = len(contracts)

    # Optional term-structure restriction to the soonest N expirations.
    rows = contracts
    if nearest_expirations:
        exps = sorted({c.get("expiration") for c in contracts if c.get("expiration")})
        keep = set(exps[:nearest_expirations])
        rows = [c for c in contracts if c.get("expiration") in keep]

    raw_ivs, atm_ivs = [], []
    for c in rows:
        iv = _f(c.get("implied_volatility"))
        if iv is not None:
            raw_ivs.append(iv)
        vol = _f(c.get("volume")) or 0.0
        delta = _f(c.get("delta"))
        if iv is None or delta is None:
            continue
        if vol >= vol_min and delta_lo < abs(delta) < delta_hi:
            atm_ivs.append(iv)

    if raw_ivs:
        out["raw_iv_min"] = min(raw_ivs)
        out["raw_iv_max"] = max(raw_ivs)
    if atm_ivs:
        out["atm_iv"] = float(statistics.median(atm_ivs))
        out["n_atm"] = len(atm_ivs)
    return out


class AVOptionsClient:
    def __init__(self):
        self.api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        self.session = requests.Session()

    def get_chain(self, ticker: str, date: str | None = None) -> list[dict] | None:
        """Fetch an options chain. Returns the list of contract dicts, or None on
        throttle/failure (so a throttle is never mistaken for an empty chain)."""
        if not self.api_key:
            return None
        params = {"function": "HISTORICAL_OPTIONS", "symbol": ticker, "apikey": self.api_key}
        if date:
            params["date"] = date
        try:
            resp = self.session.get(AV_URL, params=params, timeout=40)
            resp.raise_for_status()
            data = resp.json()
            kind, msg = _classify_av_response(data)
            if kind:
                logger.warning(f"AV options {kind} for {ticker}: {msg[:100]}")
                return None
            return data.get("data") or []
        except Exception as e:
            logger.warning(f"AV options fetch failed for {ticker}: {e}")
            return None

    def atm_iv(self, ticker: str, date: str | None = None, **kw) -> dict:
        """Convenience: fetch chain + extract ATM IV for one ticker.
        Returns extract_atm_iv(...) with atm_iv=None if the chain is unavailable."""
        chain = self.get_chain(ticker, date=date)
        if chain is None:
            return {"atm_iv": None, "n_atm": 0, "n_total": 0,
                    "raw_iv_min": None, "raw_iv_max": None, "unavailable": True}
        return extract_atm_iv(chain, **kw)
