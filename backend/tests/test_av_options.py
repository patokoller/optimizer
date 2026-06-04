"""Tests for ATM-IV extraction from AV HISTORICAL_OPTIONS (no network).

The point of the extractor is to strip the degenerate-IV wings the live API
returns (a ~0.0149 floor deep-ITM, ~10.0 deep-OTM) and keep only liquid
near-the-money contracts, so these tests reproduce that degeneracy explicitly.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.data.av_options import extract_atm_iv, AVOptionsClient


def _chain_with_wings():
    chain = []
    for _ in range(20):
        chain.append({"implied_volatility": "0.0149", "delta": "0.99", "volume": "0", "expiration": "2026-06-19"})
        chain.append({"implied_volatility": "9.99", "delta": "0.02", "volume": "0", "expiration": "2026-06-19"})
    for iv in [0.28, 0.29, 0.30, 0.31, 0.32, 0.30, 0.29, 0.33]:
        chain.append({"implied_volatility": str(iv), "delta": "0.45", "volume": "120", "expiration": "2026-06-19"})
        chain.append({"implied_volatility": str(iv + 0.01), "delta": "-0.48", "volume": "80", "expiration": "2026-06-19"})
    return chain


def test_wings_excluded_atm_median_sane():
    r = extract_atm_iv(_chain_with_wings())
    assert r["atm_iv"] is not None and 0.27 < r["atm_iv"] < 0.34
    assert r["n_atm"] == 16            # calls + puts in the |delta| band
    assert r["raw_iv_min"] < 0.02 and r["raw_iv_max"] > 9.0  # wings present raw, excluded from ATM


def test_volume_gate():
    assert extract_atm_iv([{"implied_volatility": "0.5", "delta": "0.45", "volume": "0"}])["atm_iv"] is None


def test_empty_and_none_safe():
    assert extract_atm_iv([])["atm_iv"] is None
    assert extract_atm_iv(None)["atm_iv"] is None


def test_garbage_fields_skipped():
    chain = [{"implied_volatility": "", "delta": "0.45", "volume": "100"},
             {"implied_volatility": "0.31", "delta": "", "volume": "100"},
             {"implied_volatility": "0.30", "delta": "0.50", "volume": "100"}]
    r = extract_atm_iv(chain)
    assert r["n_atm"] == 1 and abs(r["atm_iv"] - 0.30) < 1e-9


def test_nearest_expirations():
    chain = [{"implied_volatility": "0.30", "delta": "0.45", "volume": "100", "expiration": "2026-06-19"},
             {"implied_volatility": "0.60", "delta": "0.45", "volume": "100", "expiration": "2027-01-15"}]
    r = extract_atm_iv(chain, nearest_expirations=1)
    assert r["n_atm"] == 1 and abs(r["atm_iv"] - 0.30) < 1e-9


def test_unavailable_chain_graceful():
    cl = AVOptionsClient(); cl.api_key = "K"
    cl.get_chain = lambda t, date=None: None
    assert cl.atm_iv("AAPL")["unavailable"] is True
