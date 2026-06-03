#!/usr/bin/env python3
"""
probe_alphavantage.py — one-shot inspector for Alpha Vantage endpoints we may
consolidate onto (Premium plan already paid). Run locally where the key and
network exist; this decides whether AV can replace the planned FMP estimates
feed and the flaky FRED macro dependency BEFORE any client code is written.

Usage:
    export ALPHA_VANTAGE_API_KEY=...        # your Premium key
    python backend/scripts/probe_alphavantage.py [TICKER]   # default AAPL

For each endpoint it prints: HTTP ok, whether the payload is non-empty, the
top-level keys, and a trimmed sample of the most decision-relevant fields —
specifically whether EARNINGS_ESTIMATES carries a revision trail + analyst
count (the two things that make estimates predictive).
"""
import os
import sys
import json
import time
import requests

KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
TICKER = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
BASE = "https://www.alphavantage.co/query"

if not KEY:
    sys.exit("Set ALPHA_VANTAGE_API_KEY in the environment first.")

# (label, params, what we're checking for)
PROBES = [
    ("EARNINGS_ESTIMATES (gap-filler: analyst consensus + revisions)",
     {"function": "EARNINGS_ESTIMATES", "symbol": TICKER},
     "Look for: forward EPS/revenue estimates, a revision trail (estimate "
     "changes over time), and number of analysts. If present → no FMP needed."),
    ("EARNINGS (already used — confirm it also carries estimate + surprise)",
     {"function": "EARNINGS", "symbol": TICKER}, ""),
    ("EARNINGS_CALENDAR (forward report dates for event-aware scoring)",
     {"function": "EARNINGS_CALENDAR", "symbol": TICKER, "horizon": "3month"}, ""),
    ("SHARES_OUTSTANDING (float / buyback-dilution signal)",
     {"function": "SHARES_OUTSTANDING", "symbol": TICKER}, ""),
    ("FEDERAL_FUNDS_RATE (macro → could retire FRED dependency)",
     {"function": "FEDERAL_FUNDS_RATE", "interval": "daily"}, ""),
    ("CPI (macro → could retire FRED dependency)",
     {"function": "CPI", "interval": "monthly"}, ""),
    ("TREASURY_YIELD 10y (yield-curve input for regime classifier)",
     {"function": "TREASURY_YIELD", "interval": "daily", "maturity": "10year"}, ""),
    ("HISTORICAL_PUT_CALL_RATIO? (options sentiment — confirm exact fn name)",
     {"function": "HISTORICAL_OPTIONS", "symbol": TICKER}, ""),
]


def trim(obj, depth=0):
    """Compact structural preview of a JSON payload."""
    pad = "  " * depth
    if isinstance(obj, dict):
        keys = list(obj.keys())
        out = f"dict[{len(keys)}] keys: {keys[:12]}"
        if keys and depth < 2:
            first = keys[0]
            out += f"\n{pad}  e.g. {first!r} -> " + trim(obj[first], depth + 1)
        return out
    if isinstance(obj, list):
        out = f"list[{len(obj)}]"
        if obj and depth < 2:
            out += " first elem -> " + trim(obj[0], depth + 1)
        return out
    s = str(obj)
    return s[:120]


def probe(label, params, note):
    print("\n" + "=" * 78)
    print(label)
    if note:
        print("  → " + note)
    try:
        r = requests.get(BASE, params={**params, "apikey": KEY}, timeout=30)
        ok = r.status_code == 200
        try:
            data = r.json()
        except Exception:
            print(f"  HTTP {r.status_code}; non-JSON body: {r.text[:200]}")
            return
        # AV soft-errors: {"Information": ...} (rate limit / premium gate), {"Note": ...}
        for soft in ("Information", "Note", "Error Message"):
            if isinstance(data, dict) and soft in data:
                print(f"  HTTP {r.status_code}; AV {soft}: {str(data[soft])[:180]}")
        empty = (data == {} or data is None)
        print(f"  HTTP {r.status_code} | non_empty={not empty}")
        if not empty:
            print("  structure: " + trim(data))
    except Exception as e:
        print(f"  request failed: {e}")


if __name__ == "__main__":
    print(f"Probing Alpha Vantage for {TICKER} (key ...{KEY[-4:]})")
    for label, params, note in PROBES:
        probe(label, params, note)
        time.sleep(1.0)  # be gentle with the rate limit
    print("\n" + "=" * 78)
    print("Decision guide:")
    print("  • EARNINGS_ESTIMATES has revision history + analyst count → build the")
    print("    estimate-revision-momentum + SUE signals on AV; skip FMP entirely.")
    print("  • EARNINGS_ESTIMATES is only a current snapshot → snapshot it monthly")
    print("    ourselves to build the revision series (still no new vendor).")
    print("  • FEDERAL_FUNDS_RATE / CPI / TREASURY_YIELD non-empty → migrate the")
    print("    regime classifier off FRED to kill the recurring 429s.")
    print("  • Options endpoint non-empty → put-call sentiment is backtestable here,")
    print("    retiring the Phase E paid-options line item.")
