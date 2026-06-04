"""
app/data/av_macro.py
─────────────────────────────────────────────────────────────────────────────
Alpha Vantage macro source for the regime classifier, with FRED fallback.

Why this exists: the discovery logs show FRED returning HTTP 429 on DFF
(fed funds) and CPIAUCSL (CPI) on essentially every run, so the regime
classifier has been silently running on neutral fallback values for those two
inputs. Alpha Vantage — which we already pay for and which has far higher limits
— serves the same series (FEDERAL_FUNDS_RATE, CPI, TREASURY_YIELD) with no
symbol required. This module fetches them from AV and overlays them onto the
FRED snapshot, so AV fixes the recurring degradation while FRED remains the
fallback (and the sole source for VIX, which these AV endpoints don't provide).

Data hygiene: AV economic series use "." as a missing-data sentinel (observed on
CPI 2025-10). Every parser here drops "." rather than coercing it to 0.0, which
would corrupt a level series and any YoY computed off it.

None is returned per-field on throttle/failure so the caller keeps the FRED
value; this never raises into the regime path.
"""
from __future__ import annotations
import os
import logging
import requests

from app.data.clients import _classify_av_response  # consistent throttle detection

logger = logging.getLogger("av_macro")
AV_URL = "https://www.alphavantage.co/query"


def _rows(data: dict) -> list[dict]:
    """AV economic endpoints return {'name':..., 'data':[{'date','value'},...]}."""
    if not isinstance(data, dict):
        return []
    return data.get("data") or []


def _latest_valid(data: dict) -> float | None:
    """Most recent non-sentinel value from an AV economic series."""
    rows = [r for r in _rows(data) if r.get("value") not in (None, ".", "")]
    if not rows:
        return None
    rows.sort(key=lambda r: r["date"], reverse=True)
    try:
        return float(rows[0]["value"])
    except (ValueError, TypeError):
        return None


def cpi_yoy_from_payload(data: dict) -> float | None:
    """Year-over-year % from a monthly CPI level series, '.'-safe."""
    rows = [r for r in _rows(data) if r.get("value") not in (None, ".", "")]
    if len(rows) < 13:
        return None
    rows.sort(key=lambda r: r["date"], reverse=True)
    try:
        latest = float(rows[0]["value"])
        year_ago = float(rows[12]["value"])
    except (ValueError, TypeError, IndexError):
        return None
    if year_ago == 0:
        return None
    return (latest - year_ago) / year_ago * 100.0


class AVMacroClient:
    def __init__(self):
        self.api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        self.session = requests.Session()

    def _get(self, params: dict) -> dict | None:
        if not self.api_key:
            return None
        try:
            resp = self.session.get(AV_URL, params={**params, "apikey": self.api_key}, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            kind, msg = _classify_av_response(data)
            if kind:
                logger.warning(f"AV macro {kind} on {params.get('function')}: {msg[:100]}")
                return None
            return data
        except Exception as e:
            logger.warning(f"AV macro fetch failed for {params.get('function')}: {e}")
            return None

    def fed_funds_rate(self) -> float | None:
        return _latest_valid(self._get({"function": "FEDERAL_FUNDS_RATE", "interval": "monthly"}) or {})

    def cpi_yoy(self) -> float | None:
        return cpi_yoy_from_payload(self._get({"function": "CPI", "interval": "monthly"}) or {})

    def treasury_yield(self, maturity: str) -> float | None:
        return _latest_valid(self._get(
            {"function": "TREASURY_YIELD", "interval": "monthly", "maturity": maturity}) or {})

    def yield_curve_10y_2y(self) -> float | None:
        ten = self.treasury_yield("10year")
        two = self.treasury_yield("2year")
        if ten is None or two is None:
            return None
        return ten - two


def get_hybrid_macro_snapshot() -> dict:
    """FRED snapshot (VIX + fallbacks + trends) with AV overlaid on the fields
    that chronically 429 on FRED. AV failures leave the FRED value untouched."""
    from app.data.fred_client import FREDClient

    snap = FREDClient().get_macro_snapshot()
    av = AVMacroClient()

    ff = av.fed_funds_rate()
    if ff is not None:
        snap["fed_funds"] = ff
        snap["fed_funds_source"] = "alpha_vantage"

    cpi = av.cpi_yoy()
    if cpi is not None:
        snap["cpi_yoy"] = cpi
        snap["cpi_source"] = "alpha_vantage"

    curve = av.yield_curve_10y_2y()
    if curve is not None:
        snap["yield_curve"] = curve
        snap["curve_source"] = "alpha_vantage"

    sources = [k for k in ("fed_funds_source", "cpi_source", "curve_source") if k in snap]
    logger.info(f"Macro snapshot: AV overlaid {len(sources)}/3 fields ({', '.join(sources) or 'none'}); "
                f"VIX from FRED; FRED errors={len(snap.get('errors', []))}")
    return snap
