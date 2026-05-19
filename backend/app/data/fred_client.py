"""
app/data/fred_client.py
────────────────────────────────────────────────────────────────────────────
FRED (Federal Reserve Economic Data) client.
Used for: market regime classification.

Series fetched:
  VIXCLS      — CBOE VIX daily close
  T10Y2Y      — 10Y minus 2Y Treasury yield spread
  FEDFUNDS    — Effective federal funds rate
  CPIAUCSL    — CPI All Urban Consumers (for YoY calc)

No rate limits for most series. FRED allows 120 API calls per 60 seconds.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger("fred")

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = os.environ.get("FRED_API_KEY", "50dc7177994133c91cc1c0a5cf1bf529")


class FREDError(Exception):
    """FRED unavailable — regime classification falls back to neutral."""
    pass


class FREDClient:
    """
    Lightweight FRED client for macro regime inputs.
    All series returned as the most recent available value.
    """

    def __init__(self):
        self.api_key = FRED_API_KEY
        self.session = requests.Session()

    def _fetch_series(
        self,
        series_id: str,
        n_obs: int = 30,
        observation_start: Optional[str] = None,
    ) -> pd.Series:
        """
        Fetch the last n_obs observations for a FRED series.
        Returns a pandas Series with DatetimeIndex.
        """
        if observation_start is None:
            start = (datetime.utcnow() - timedelta(days=n_obs * 2)).strftime("%Y-%m-%d")
        else:
            start = observation_start

        try:
            resp = self.session.get(
                FRED_BASE,
                params={
                    "series_id":         series_id,
                    "api_key":           self.api_key,
                    "file_type":         "json",
                    "observation_start": start,
                    "sort_order":        "desc",
                    "limit":             n_obs,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            if "observations" not in data or not data["observations"]:
                raise FREDError(f"No observations for {series_id}")

            obs = data["observations"]
            dates  = [o["date"] for o in obs if o["value"] != "."]
            values = [float(o["value"]) for o in obs if o["value"] != "."]

            if not values:
                raise FREDError(f"All values missing for {series_id}")

            s = pd.Series(values, index=pd.to_datetime(dates))
            return s.sort_index()

        except FREDError:
            raise
        except Exception as e:
            raise FREDError(f"FRED fetch failed for {series_id}: {e}") from e

    def get_latest_value(self, series_id: str) -> float:
        """Get the single most recent non-null value for a series."""
        s = self._fetch_series(series_id, n_obs=10)
        return float(s.iloc[-1])

    def get_macro_snapshot(self) -> dict:
        """
        Fetch all macro inputs needed for regime classification.

        Returns:
            {
                "vix":           float,   # CBOE VIX
                "yield_curve":   float,   # 10Y - 2Y spread (negative = inverted)
                "fed_funds":     float,   # current Fed funds rate
                "cpi_yoy":       float,   # CPI year-over-year %
                "vix_trend":     str,     # "rising" | "falling" | "stable"
                "curve_trend":   str,     # "steepening" | "flattening" | "stable"
                "errors":        list,    # any series that failed
            }
        """
        result = {}
        errors = []

        # VIX
        try:
            vix_series = self._fetch_series("VIXCLS", n_obs=30)
            result["vix"] = float(vix_series.iloc[-1])
            result["vix_1m_ago"] = float(vix_series.iloc[0]) if len(vix_series) >= 20 else result["vix"]
            result["vix_trend"] = (
                "rising"  if result["vix"] > result["vix_1m_ago"] * 1.05 else
                "falling" if result["vix"] < result["vix_1m_ago"] * 0.95 else
                "stable"
            )
        except FREDError as e:
            logger.warning(f"FRED VIX: {e}")
            result["vix"] = 20.0  # neutral default
            result["vix_1m_ago"] = 20.0
            result["vix_trend"] = "stable"
            errors.append(str(e))

        # 10Y-2Y yield curve
        try:
            curve_series = self._fetch_series("T10Y2Y", n_obs=30)
            result["yield_curve"] = float(curve_series.iloc[-1])
            prior = float(curve_series.iloc[0]) if len(curve_series) >= 20 else result["yield_curve"]
            result["curve_trend"] = (
                "steepening" if result["yield_curve"] > prior + 0.05 else
                "flattening" if result["yield_curve"] < prior - 0.05 else
                "stable"
            )
        except FREDError as e:
            logger.warning(f"FRED T10Y2Y: {e}")
            result["yield_curve"] = 0.5  # neutral
            result["curve_trend"] = "stable"
            errors.append(str(e))

        # Federal funds rate
        try:
            result["fed_funds"] = self._fetch_latest_monthly("DFF")
        except FREDError as e:
            logger.warning(f"FRED DFF: {e}")
            result["fed_funds"] = 5.0  # current-ish neutral
            errors.append(str(e))

        # CPI year-over-year
        try:
            cpi = self._fetch_series("CPIAUCSL", n_obs=14)
            if len(cpi) >= 13:
                current = float(cpi.iloc[-1])
                year_ago = float(cpi.iloc[-13])
                result["cpi_yoy"] = ((current - year_ago) / year_ago) * 100
            else:
                result["cpi_yoy"] = 3.0  # neutral default
        except FREDError as e:
            logger.warning(f"FRED CPI: {e}")
            result["cpi_yoy"] = 3.0
            errors.append(str(e))

        result["errors"] = errors
        result["fetched_at"] = datetime.utcnow().isoformat()
        return result

    def _fetch_latest_monthly(self, series_id: str) -> float:
        """Fetch most recent monthly observation."""
        s = self._fetch_series(series_id, n_obs=5)
        return float(s.iloc[-1])
