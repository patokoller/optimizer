"""Regression test for the BRK.B batch custom_id bug.

A ticker with a dot (BRK.B) produced an Anthropic batch custom_id that violated
^[a-zA-Z0-9_-]{1,64}$, which 400'd the ENTIRE batch and zeroed LLM scores for
every name in the run. _safe_custom_id_map must sanitize all ids, keep them
unique, and round-trip back to the original tickers.
"""
import re, os
_src = open(os.path.join(os.path.dirname(__file__), "..", "app", "ml", "llm_scoring.py")).read()
_block = re.search(r"def _safe_custom_id_map.*?return id_to_ticker", _src, re.S).group(0)
_ns = {"_re": re}
exec(_block, _ns)
safe_map = _ns["_safe_custom_id_map"]

PAT = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def test_all_ids_valid_and_unique_and_roundtrip():
    tickers = ["AAPL", "BRK.B", "BF.B", "BRK_B", "AIRO", "GGAL"]
    m = safe_map(tickers)                       # {custom_id: ticker}
    ids = list(m.keys())
    assert all(PAT.match(i) for i in ids)       # every id valid
    assert len(set(ids)) == len(tickers)        # unique, 1:1
    assert set(m.values()) == set(tickers)      # round-trips every ticker


def test_dot_ticker_specifically():
    m = safe_map(["BRK.B"])
    cid = list(m.keys())[0]
    assert PAT.match(cid) and m[cid] == "BRK.B"


def test_collision_resolved():
    m = safe_map(["BRK.B", "BRK_B"])            # both sanitize toward BRK_B
    assert len(m) == 2 and set(m.values()) == {"BRK.B", "BRK_B"}


def test_empty_and_weird():
    m = safe_map(["", "...", "A" * 80])
    ids = list(m.keys())
    assert all(PAT.match(i) for i in ids) and len(ids) == 3
