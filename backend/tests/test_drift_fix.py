"""Regression tests for #6 — language drift was 0/98 across all runs.

Root cause: AlphaVantageClient had TWO get_earnings_transcript definitions; the
second (supporting specific_quarters) shadowed the first, but compute_language_drift
called `quarters=[quarter_str]` — a list into an int param — raising an instant
TypeError that an `except Exception: pass` swallowed. Drift returned "" in ~3ms
for every ticker. The fix passes specific_quarters, deletes the shadowed dead
method, and counts/logs fetch errors instead of silencing them."""
from app.data.phase_d import compute_language_drift


class FakeAV:
    """Mirrors the real (post-fix) client signature."""

    def __init__(self):
        self.calls = []

    def get_earnings_transcript(self, ticker, quarters=2, specific_quarters=None):
        self.calls.append({"quarters": quarters, "specific_quarters": specific_quarters})
        assert isinstance(quarters, int)  # the old bug passed a list here
        q = specific_quarters[0]
        hedge = "we believe approximately may potentially uncertain " * (3 if q >= "2025Q3" else 1)
        return (
            f"=== EARNINGS CALL {q} ===\n[CEO] Revenue was $5.2 billion, up 12%. {hedge}"
            "Operator: We will now begin the question-and-answer session.\n"
            f"[Analyst] Question on margins? [CFO] Margins were 42%. {hedge}"
        ) * 6


class BrokenAV:
    """A client whose signature rejects the call — must degrade loudly, not crash."""

    def get_earnings_transcript(self, ticker, quarters=2):
        raise TypeError("unexpected keyword argument")


def test_drift_uses_specific_quarters_kwarg():
    fake = FakeAV()
    compute_language_drift("TEST", fake, n_quarters=8)
    assert len(fake.calls) >= 2
    assert all(c["specific_quarters"] is not None for c in fake.calls)
    assert all(isinstance(c["specific_quarters"], list) for c in fake.calls)


def test_drift_produces_analysis_from_valid_transcripts():
    drift = compute_language_drift("TEST", FakeAV(), n_quarters=8)
    assert drift != ""  # was "" for 98/98 tickers before the fix
    assert "LANGUAGE DRIFT" in drift
    assert "hedge" in drift.lower() or "Hedg" in drift


def test_drift_degrades_to_empty_on_client_errors_without_raising():
    assert compute_language_drift("TEST", BrokenAV(), n_quarters=8) == ""


def test_client_has_single_transcript_method_with_specific_quarters():
    """Guard against the shadowed-definition bug returning."""
    import inspect
    from app.data import clients

    src = inspect.getsource(clients.AlphaVantageClient)
    assert src.count("def get_earnings_transcript") == 1
    sig = inspect.signature(clients.AlphaVantageClient.get_earnings_transcript)
    assert "specific_quarters" in sig.parameters
