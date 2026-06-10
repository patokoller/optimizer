"""Regression tests for #20 part 2 — two-stage extract-then-score.

The anchored rubric (#20 part 1) fixed range usage but not within-band
anchoring: 23/44 names still piled at 0.7-0.8 with the 0.6-0.7 decile empty.
Stage 1 extracts a structured fact sheet (no score); stage 2 scores from the
fact sheet via mandatory adjustment arithmetic (band_base + itemized deltas),
which makes round-value anchoring arithmetically unnatural.

Safety invariants tested here:
- per-ticker fallback: extraction failure NEVER costs a ticker its LLM score
- LLM_TWO_STAGE=0 kill switch restores single-stage behavior exactly
- score JSON contract (score/key_positives/key_risks/confidence) unchanged;
  extra fields (band_base/adjustments) tolerated and stored
- arithmetic drift is warned about, not fatal
"""
import json
import os

import app.ml.llm_scoring as L


def _scorer_with_fake_transport(fake_run):
    s = L.LLMScorer.__new__(L.LLMScorer)
    s.client = object()
    s.model = "test-model"
    s._run_batch_raw = fake_run
    return s


def _single_stage_prompt(ticker="NVDA"):
    s = L.LLMScorer.__new__(L.LLMScorer)
    return L.LLMScorer.build_prompt(
        s, ticker=ticker, company_name=f"{ticker} Corp", frequency="monthly",
        period="2026-06", peer_context="revenue_growth: 92nd pctile",
        filing_context="10-K...", earnings_context="Q&A...",
    )


def test_flag_default_on_and_kill_values():
    for v in ("0", "false", "no", "off"):
        os.environ["LLM_TWO_STAGE"] = v
        assert not L.two_stage_enabled()
    os.environ.pop("LLM_TWO_STAGE", None)
    assert L.two_stage_enabled()


def test_extraction_prompt_strips_scoring_keeps_materials():
    sp = _single_stage_prompt()
    ep = L.build_extraction_prompt(sp)
    assert ep is not None
    assert "FACT EXTRACTION ONLY" in ep
    assert "SCORE CALIBRATION" not in ep
    assert "92nd pctile" in ep  # peer context retained in materials
    assert L.build_extraction_prompt("no calibration marker") is None


def test_stage2_prompt_mandates_arithmetic():
    p2 = L.build_stage2_prompt("NVDA", {"peer_standing": "92nd pctile"})
    for needle in ("band_base", "DERIVATION REQUIREMENT", "sum of deltas", "92nd pctile"):
        assert needle in p2


def test_two_stage_end_to_end_with_per_ticker_fallback():
    os.environ.pop("LLM_TWO_STAGE", None)
    calls = []

    def fake_run(prompts, label, max_tokens=800, poll_interval=60, max_wait=7200):
        calls.append({"label": label, "prompts": dict(prompts)})
        if label == "extraction":
            return {
                "NVDA": json.dumps({"peer_standing": "92nd"}),
                "AAPL": "NOT JSON",  # parse-fail -> fallback
                # BRK.B absent from results -> fallback
            }
        return {
            t: json.dumps({
                "band_base": 0.55,
                "adjustments": [{"reason": "guidance", "delta": 0.06}],
                "score": 0.61,
                "key_positives": ["a"], "key_risks": ["b"], "confidence": "high",
            })
            for t in prompts
        }

    s = _scorer_with_fake_transport(fake_run)
    sp = _single_stage_prompt()
    prompts = {t: sp for t in ("NVDA", "AAPL", "BRK.B")}
    res = s.score_batch(prompts)

    assert [c["label"] for c in calls] == ["extraction", "scoring"]
    assert "FACT SHEET" in calls[1]["prompts"]["NVDA"]
    assert "FACT SHEET" not in calls[1]["prompts"]["AAPL"]
    assert "FACT SHEET" not in calls[1]["prompts"]["BRK.B"]
    assert res["NVDA"]["two_stage"] is True
    assert res["AAPL"]["two_stage"] is False
    assert len(res) == 3  # nobody lost a score


def test_kill_switch_single_stage_one_batch():
    os.environ["LLM_TWO_STAGE"] = "0"
    calls = []

    def fake_run(prompts, label, max_tokens=800, poll_interval=60, max_wait=7200):
        calls.append(label)
        return {t: json.dumps({"score": 0.5, "key_positives": [], "key_risks": [],
                               "confidence": "low"}) for t in prompts}

    s = _scorer_with_fake_transport(fake_run)
    res = s.score_batch({"NVDA": _single_stage_prompt()})
    os.environ.pop("LLM_TWO_STAGE", None)
    assert calls == ["scoring"]
    assert res["NVDA"]["two_stage"] is False
    assert res["NVDA"]["score"] == 0.5


def test_arithmetic_drift_warns_but_keeps_score():
    s = L.LLMScorer.__new__(L.LLMScorer)
    bad = json.dumps({"band_base": 0.50, "adjustments": [{"delta": 0.05}],
                      "score": 0.80, "key_positives": [], "key_risks": [],
                      "confidence": "low"})
    parsed = s._parse_score_json("TEST", bad)
    assert parsed is not None and parsed["score"] == 0.80
