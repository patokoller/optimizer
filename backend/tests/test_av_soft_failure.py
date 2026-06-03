"""Mocked tests for Alpha Vantage soft-failure hardening (no network).

Verifies that an AV throttle is distinguished from a legitimately-empty ticker
and from a real error, so throttling can no longer silently degrade fundamental
coverage (the root cause of the drift 0/98 symptom and silent coverage loss).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app.data.clients as C


class _FakeResp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p


def _seq(payloads):
    it = iter(payloads)
    def fake_get(url, params=None, timeout=None):
        return _FakeResp(next(it))
    return fake_get


RATE = {"Information": "Our standard API rate limit is 25 requests per day."}
PREMIUM = {"Information": "This is a premium endpoint. Subscribe to a premium plan."}
ERR = {"Error Message": "Invalid API call."}
DATA = {"quarterlyReports": [
    {"fiscalDateEnding": "2025-12-31", "totalRevenue": "1000", "operatingIncome": "200", "netIncome": "150"},
    {"fiscalDateEnding": "2025-09-30", "totalRevenue": "900",  "operatingIncome": "180", "netIncome": "120"},
]}
EMPTY = {"symbol": "ZZZ", "annualReports": []}


def test_classifier():
    assert C._classify_av_response(RATE)[0] == "rate_limit"
    assert C._classify_av_response(PREMIUM)[0] == "premium_gate"
    assert C._classify_av_response(ERR)[0] == "error"
    assert C._classify_av_response(DATA)[0] is None


def test_income_statement_throttle_raises_soft_failure():
    cl = C.AlphaVantageClient(); cl.api_key = "K"
    C.requests.get = _seq([RATE])
    try:
        cl.get_income_statement("AAPL"); assert False
    except C.AlphaVantageSoftFailure as e:
        assert e.reason == "rate_limit"


def test_income_statement_legit_empty_returns_empty_df():
    cl = C.AlphaVantageClient(); cl.api_key = "K"
    C.requests.get = _seq([EMPTY])
    df = cl.get_income_statement("ZZZ")
    assert df.empty


def test_income_statement_valid_data():
    cl = C.AlphaVantageClient(); cl.api_key = "K"
    C.requests.get = _seq([DATA])
    df = cl.get_income_statement("AAPL")
    assert len(df) == 2 and abs(df.iloc[-1]["operating_margin"] - 0.2) < 1e-9


def test_batch_recovers_after_throttle():
    cl = C.AlphaVantageClient(); cl.api_key = "K"
    C.requests.get = _seq([RATE, DATA])
    out = cl.get_fundamentals_batch(["AAPL"], delay_sec=0, throttle_backoff_sec=0, max_throttle_retries=1)
    assert len(out) == 2


def test_batch_all_throttled_raises_flagging_throttle():
    cl = C.AlphaVantageClient(); cl.api_key = "K"
    C.requests.get = _seq([RATE, RATE])
    try:
        cl.get_fundamentals_batch(["AAPL"], delay_sec=0, throttle_backoff_sec=0, max_throttle_retries=1)
        assert False
    except C.AlphaVantageError as e:
        assert "throttl" in str(e).lower()


def test_batch_isolates_error_and_empty_keeps_good():
    cl = C.AlphaVantageClient(); cl.api_key = "K"
    C.requests.get = _seq([DATA, ERR, EMPTY])
    out = cl.get_fundamentals_batch(["AAA", "BBB", "CCC"], delay_sec=0,
                                    throttle_backoff_sec=0, max_throttle_retries=1)
    assert set(out["ticker"]) == {"AAA"}
