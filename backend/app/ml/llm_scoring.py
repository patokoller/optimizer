"""
app/ml/llm_scoring.py
─────────────────────────────────────────────────────────────────────────────
Claude LLM semantic scoring — Section 3.3, Cohen et al. (2025).

Features:
  • Per-ticker, per-month LLM score cache (DB-backed)
  • Structured prompt: SEC 10-K/10-Q/8-K + earnings call context (≤200K tokens)
  • Parsed JSON output: score, key_positives, key_risks, confidence
  • Graceful fallback: if Claude API fails → returns None → caller uses w=1.0
"""
import os
import json
import logging
from datetime import datetime
from typing import Optional

import anthropic
from anthropic import APIConnectionError, APITimeoutError, RateLimitError

logger = logging.getLogger(__name__)

# ── Prompt template (Section 3.3, Cohen et al. 2025) ─────────────────────
import re as _re

_QA_MARKERS = [
    r"Questions?\s*(?:and\s*)?Answers?\s*Session",
    r"Q&A\s*Session",
    r"We'll now begin.*question",
    r"Operator.*questions",
    r"^\s*Operator\s*$",
    r"QUESTION.*ANSWER",
]


def _safe_custom_id_map(tickers: list[str]) -> dict:
    """Map a list of tickers to Anthropic-batch-safe custom_ids.

    The batch custom_id must match ^[a-zA-Z0-9_-]{1,64}$. Tickers like BRK.B
    contain characters (dots) that violate it; a single bad id 400s the whole
    batch. This replaces invalid chars with '_', truncates to 64, and resolves
    collisions (e.g. BRK.B vs a real BRK_B) with a numeric suffix.

    Returns {custom_id: original_ticker} (1:1).
    """
    id_to_ticker: dict[str, str] = {}
    for tkr in tickers:
        base = _re.sub(r"[^a-zA-Z0-9_-]", "_", tkr)[:64] or "T"
        cid = base
        n = 0
        while cid in id_to_ticker:
            n += 1
            suffix = f"_{n}"
            cid = base[:64 - len(suffix)] + suffix
        id_to_ticker[cid] = tkr
    return id_to_ticker


def _qa_weighted_truncate(text: str, limit: int = 15_000, qa_fraction: float = 0.6) -> str:
    """Truncate an earnings transcript toward the analyst Q&A rather than the
    scripted prepared remarks.

    A naive text[:limit] keeps the operator intro + management's prepared script
    and routinely cuts off the Q&A entirely — but the unscripted Q&A (analysts
    probing, management answering live) is where the differentiating signal is.
    This keeps a head of the prepared section for guidance context and reserves
    the majority of the budget for the Q&A. Falls back to a plain head cut when
    no Q&A boundary is detectable, so behaviour is never worse than before.
    """
    if not text or len(text) <= limit:
        return text or ""

    split = len(text)
    for marker in _QA_MARKERS:
        m = _re.search(marker, text, _re.IGNORECASE | _re.MULTILINE)
        if m and m.start() < split:
            split = m.start()
    if split >= len(text) * 0.9:          # no plausible Q&A boundary
        return text[:limit]

    prepared, qa = text[:split], text[split:]
    sep = "\n[... prepared remarks truncated ...]\n"
    budget = limit - len(sep)
    qa_budget = int(budget * qa_fraction)
    prep_budget = budget - qa_budget
    # reallocate slack from whichever side is shorter to the other
    prep_take = min(len(prepared), prep_budget)
    qa_take = min(len(qa), qa_budget + (prep_budget - prep_take))
    prep_take = min(len(prepared), prep_take + (qa_budget - min(len(qa), qa_budget)))
    return prepared[:prep_take] + sep + qa[:qa_take]


# ── Two-stage extract-then-score (#20 part 2) ───────────────────────────
# Stage 1 extracts a structured fact sheet from the briefing materials (no
# score). Stage 2 scores from the fact sheet via explicit adjustment
# arithmetic (band_base + itemized deltas), which breaks the holistic
# "feels like a B+ -> 0.72" anchoring the rubric alone did not fix.
# Kill switch: set env LLM_TWO_STAGE=0 to restore single-stage scoring.

_CALIBRATION_MARKER = "SCORE CALIBRATION (anchor the number"
_MATERIALS_MARKER = "\nTICKER:"


def two_stage_enabled() -> bool:
    return os.environ.get("LLM_TWO_STAGE", "1").strip().lower() not in ("0", "false", "no", "off")


def split_prompt_materials(prompt: str) -> Optional[str]:
    """Return the company-materials portion of a single-stage prompt — the
    'TICKER: ... / --- COMPANY OVERVIEW --- / --- SEC FILINGS --- /
    --- INCOME STATEMENT --- ...' block that carries the actual evidence.

    In the current template the scoring rubric and SCORE CALIBRATION block come
    FIRST, and the materials come AFTER. The previous implementation returned
    everything *before* the calibration marker — i.e. the rubric with NO company
    data — so the stage-1 extractor saw no filings, overview, or income statement
    and emitted "none stated" for every field, which stage 2 then scored as a
    "complete information void". Returning the materials block fixes that.

    Falls back to the old behavior only if the materials marker is absent.
    Returns None (→ single-stage path) if neither marker is found.
    """
    m_idx = prompt.find(_MATERIALS_MARKER)
    if m_idx > 0:
        return prompt[m_idx:].strip()
    # Fallback: if the template ever changes, keep two-stage working off the
    # pre-calibration portion rather than silently dropping to single-stage.
    idx = prompt.find(_CALIBRATION_MARKER)
    if idx <= 0:
        return None
    return prompt[:idx].rstrip()


EXTRACTION_TAIL = """
TASK — FACT EXTRACTION ONLY (do NOT score, do NOT give an outlook rating):
From the materials above, extract a structured fact sheet. Be factual and
specific; copy concrete numbers where present; write "none stated" where a
field has no support in the materials. Faithfully carry over the PEER
POSITION percentiles into peer_standing.

Return ONLY valid JSON with this exact structure:
{
  "revenue_trajectory": "<1-2 factual sentences with figures>",
  "margins_earnings_quality": "<1-2 factual sentences>",
  "guidance": {"direction": "<raised|maintained|lowered|none>", "specifics": "<concrete guidance figures or 'none stated'>"},
  "management_tone": {"hedging": "<low|moderate|high>", "notable": "<1 sentence>"},
  "competitive_position": "<1-2 factual sentences>",
  "peer_standing": "<the peer percentile figures from the PEER POSITION section, verbatim>",
  "catalysts": ["<catalyst 1>", "<catalyst 2>"],
  "key_risks": ["<risk 1>", "<risk 2>"],
  "red_flags": ["<red flag or empty list>"]
}"""


STAGE2_SCORE_TEMPLATE = """You are a quantitative analyst scoring {ticker} for a monthly portfolio rebalancing.
You are given ONLY a structured fact sheet extracted from this company's filings,
earnings calls, and news — score from these facts alone.

=== FACT SHEET ({ticker}) ===
{fact_sheet}

SCORE CALIBRATION (anchor the number — do NOT default to 0.70/0.75):
  - 0.85-1.00  top-decile conviction; evidence decisively strong (expect few)
  - 0.65-0.85  clearly above the peer median; strong but not exceptional
  - 0.45-0.65  around the peer median; balanced or genuinely mixed evidence
  - 0.25-0.45  below the peer median; identifiable weakness or elevated risk
  - 0.00-0.25  conviction sell; serious deterioration or red flags

DERIVATION REQUIREMENT (mandatory arithmetic):
1. Set band_base from peer_standing: a value reflecting where this name's
   overall peer standing falls on [0,1] (roughly the average peer percentile
   divided by 100, weighted toward the metrics most relevant to this company).
2. List adjustments for qualitative evidence the peer metrics do not capture
   (guidance, tone, catalysts, red flags). Each adjustment: a reason and a
   signed delta with |delta| <= 0.05; at most 4 adjustments; |sum of deltas| <= 0.15.
3. score = band_base + sum(deltas), reported to two decimals. The reported
   score MUST equal that sum (within 0.01). Scores like 0.58 or 0.41 are
   expected; repeated round values like 0.70/0.75 indicate calibration failure.

Return ONLY valid JSON with this exact structure:
{{
  "band_base": <float>,
  "adjustments": [{{"reason": "<short>", "delta": <signed float>}}],
  "score": <float 0.0-1.0 = band_base + sum of deltas>,
  "key_positives": ["<signal 1>", "<signal 2>"],
  "key_risks": ["<risk 1>", "<risk 2>"],
  "confidence": "<low|medium|high>"
}}"""


def build_extraction_prompt(single_stage_prompt: str) -> Optional[str]:
    materials = split_prompt_materials(single_stage_prompt)
    if materials is None:
        return None
    return materials + "\n" + EXTRACTION_TAIL


def build_stage2_prompt(ticker: str, fact_sheet: dict) -> str:
    return STAGE2_SCORE_TEMPLATE.format(
        ticker=ticker, fact_sheet=json.dumps(fact_sheet, indent=2)
    )




SCORE_PROMPT_TEMPLATE = """You are a quantitative analyst scoring {ticker} ({company_name}) for a {frequency} portfolio rebalancing.

You have been provided with six layers of information. Use all available layers — weight recent evidence more heavily.

PEER POSITION (relative rank across this period's scoreable universe — use this to DIFFERENTIATE this name from its peers; do not anchor every score near the middle):
{peer_context}

SCORING CRITERIA:
1. Revenue growth trajectory and quality (not just magnitude — consistency matters)
2. Margin expansion or compression (operating leverage signals)
3. Management credibility: do they beat their own guidance? What does their tone signal?
4. Competitive positioning and moat durability
5. Balance sheet health: leverage, liquidity, cash generation quality
6. Valuation relative to analyst consensus and historical norms
7. Macro and regulatory tailwinds or headwinds specific to this company

EARNINGS QUALITY HEURISTIC (from earnings history):
- Beat rate > 75%: strong management credibility, score positively
- Beat rate 50-75%: neutral
- Beat rate < 50%: execution risk, score negatively
- Trend matters: improving beat rate is bullish; deteriorating is bearish

VALUATION HEURISTIC (from company overview):
- Analyst target > 20% above current price: positive signal (consensus upside)
- Analyst target < current price: negative signal (consensus downside)
- EV/EBITDA > 40x: premium valuation — requires strong growth to justify
- P/E compression trend: bearish; expansion: bullish

BALANCE SHEET HEURISTIC:
- D/E > 2.0: elevated leverage risk, especially in rising rate environment
- Current ratio < 1.0: near-term liquidity concern
- High goodwill relative to equity: acquisition integration risk

CASH FLOW HEURISTIC:
- FCF > net income: high-quality earnings (cash conversion > 1.0)
- FCF < 0 while profitable: earnings quality concern
- Rising buybacks + FCF positive: strong capital allocation signal

INSIDER TRANSACTIONS HEURISTIC:
- CEO/CFO buying with own money: strongest bullish signal
- Cluster buying (3+ insiders): very bullish — coordinated conviction
- Discretionary selling: mildly bearish (could be liquidity, diversification)
- 10b5-1 plan sales: pre-scheduled, lower signal — note but don't overweight
- Mass selling ahead of guidance period: bearish warning signal

INSTITUTIONAL HOLDINGS HEURISTIC:
- QoQ increase in institutional ownership: accumulation signal
- QoQ decrease: distribution signal — especially meaningful if >5% change
- High # of holders with rising concentration: institutional conviction building
- Hedge fund dominance: tactical/shorter-term view; mutual fund dominance: longer-term thesis

SEC COMMENT LETTERS HEURISTIC:
- UPLOAD filing (SEC questioning): elevated accounting risk — treat as significant negative
- Multiple rounds of correspondence: SEC not satisfied — strong negative signal
- Topics: revenue recognition, goodwill, going concern = highest severity
- No correspondence in 2 years: positive signal (clean regulatory relationship)
{optional_signal_heuristics}
SCORE CALIBRATION (anchor the number — do NOT default to 0.70/0.75):
Place this name into a band using its standing across the PEER POSITION metrics
above, then adjust within/across the band for evidence those metrics do not
capture. Use the full 0.0–1.0 range — in a ~100-name universe most names should
NOT receive the same score, and only a handful belong in the top band.
  - 0.85-1.00  top-decile conviction; evidence decisively strong (expect few)
  - 0.65-0.85  clearly above the peer median; strong but not exceptional
  - 0.45-0.65  around the peer median; balanced or genuinely mixed evidence
  - 0.25-0.45  below the peer median; identifiable weakness or elevated risk
  - 0.00-0.25  conviction sell; serious deterioration or red flags
Anchoring rule: let this name's peer standing set the band, then move at most
~0.15 for qualitative evidence (guidance, filings, insider/institutional signals)
not reflected in the peer metrics. Report the precise value your evidence implies
(e.g. 0.58, 0.41) rather than rounding to 0.70/0.75 — clustering at round values
is a calibration failure.

Return ONLY valid JSON with this exact structure:
{{
  "score": <float 0.0-1.0 per SCORE CALIBRATION above; 1.0 = strongest conviction buy, 0.0 = strongest conviction sell>,
  "key_positives": [<2-4 specific, evidence-backed positive factors citing which source they come from>],
  "key_risks": [<2-4 specific, evidence-backed risk factors citing which source they come from>],
  "confidence": "<low|medium|high>"
}}

Confidence guide: high = multiple sources align; medium = mixed signals; low = insufficient data.
Do not include any text outside the JSON object.

TICKER: {ticker}
COMPANY: {company_name}
PERIOD: {period} ({frequency})

--- COMPANY OVERVIEW (valuation, analyst consensus, market context) ---
{overview_context}

--- SEC FILINGS (10-K / 10-Q / 8-K) ---
{filing_context}

--- EARNINGS CALL TRANSCRIPT ---
{earnings_context}

--- EARNINGS CALL — PREPARED REMARKS vs Q&A SPLIT (most recent quarter) ---
{transcript_qa_split_context}

--- EARNINGS SURPRISE HISTORY ---
{earnings_history_context}

--- INCOME STATEMENT (quarterly: revenue, operating/net income, margins) ---
{income_statement_context}

--- BALANCE SHEET (last 4 quarters) ---
{balance_sheet_context}

--- CASH FLOW (last 4 quarters) ---
{cash_flow_context}

--- INSIDER TRANSACTIONS (Form 4 — last 180 days) ---
{insider_context}

--- INSTITUTIONAL HOLDINGS (latest 13F filings) ---
{institutional_context}

--- SEC COMMENT LETTERS / CORRESPONDENCE ---
{comment_letters_context}
{optional_signals_block}
--- RECENT NEWS & SENTIMENT ---
{news_context}

{concentration_instruction}
"""


def _build_optional_signals(language_drift_context: str = "",
                            short_interest_context: str = "") -> tuple[str, str]:
    """Assemble the language-drift / short-interest fragments only when data is
    present. Returns (heuristics, data_block); both empty when neither signal
    has data, so dead labeled sections and their instructions never ship.
    Drift and short interest are frequently unavailable, and an empty labeled
    section plus its heuristic is wasted tokens that can dilute attention."""
    drift = (language_drift_context or "").strip()
    short = (short_interest_context or "").strip()
    heur_parts, data_parts = [], []
    if drift:
        heur_parts.append(
            "LANGUAGE DRIFT HEURISTIC:\n"
            "- DETERIORATING trend: rising hedging + falling specificity across 6+ quarters → bearish\n"
            "- IMPROVING trend: falling hedging + rising specificity → bullish forward guidance quality\n"
            "- Q&A divergence > 1.5: management is MORE cautious in Q&A than in prepared remarks\n"
            "  This means analysts are extracting information management didn't volunteer — red flag"
        )
        data_parts.append("--- MANAGEMENT LANGUAGE DRIFT (8-quarter trend) ---\n" + drift[:4_000])
    if short:
        heur_parts.append(
            "SHORT INTEREST HEURISTIC:\n"
            "- Days-to-cover > 10 + high Claude score: market disagrees — investigate thesis carefully\n"
            "- Days-to-cover > 10 + rising stock: potential short squeeze setup\n"
            "- Short interest rising while price rising: building contrarian pressure\n"
            "- Short interest falling: shorts losing conviction, covering — mild bullish signal"
        )
        data_parts.append("--- SHORT INTEREST (FINRA) ---\n" + short[:2_000])
    heuristics = ("\n" + "\n\n".join(heur_parts) + "\n") if heur_parts else ""
    block = ("\n" + "\n\n".join(data_parts) + "\n") if data_parts else ""
    return heuristics, block


class LLMScoringError(Exception):
    """Raised when Claude API is unavailable — triggers w=1.0 fallback."""
    pass


class LLMScorer:
    """
    Wraps the Anthropic API for per-ticker LLM scoring.

    Usage:
        scorer = LLMScorer()
        result = scorer.score(ticker="NVDA", company_name="NVIDIA Corp.", ...)
        if result is None:
            # API failed — fall back to w=1.0
    """

    def __init__(self, model: str | None = None):
        # Model is env-overridable so it can be changed without a redeploy when
        # Anthropic rotates/retires model strings. The previous hard default
        # (claude-sonnet-4-20250514) began returning 404 once it was retired.
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — LLM scoring unavailable.")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"LLMScorer using model: {self.model}")

    def score(
        self,
        ticker: str,
        company_name: str,
        frequency: str,
        period: str,
        filing_context: str,
        earnings_context: str,
        peer_context: str = "",
        news_context: str = "",
        earnings_history_context: str = "",
        overview_context: str = "",
        income_statement_context: str = "",
        balance_sheet_context: str = "",
        cash_flow_context: str = "",
        insider_context: str = "",
        institutional_context: str = "",
        transcript_qa_split_context: str = "",
        # Phase D
        comment_letters_context: str = "",
        language_drift_context: str = "",
        short_interest_context: str = "",
        concentration_instruction: str = "",
    ) -> Optional[dict]:
        """
        Call Claude API with full enriched context (Phases A + B + C + D).

        Token budget (200K context window):
          - Company overview:       up to  5K chars
          - SEC filings:            up to 45K chars
          - Earnings transcript:    up to 30K chars
          - Earnings history:       up to  3K chars
          - Balance sheet:          up to  5K chars
          - Cash flow:              up to  5K chars
          - Insider:                up to  4K chars
          - Institutional:          up to  4K chars
          - Comment letters:        up to  6K chars  ← Phase D
          - Language drift:         up to  4K chars  ← Phase D
          - Short interest:         up to  2K chars  ← Phase D
          - News:                   up to  6K chars
        """
        if self.client is None:
            logger.warning(f"LLM scorer not initialized — skipping {ticker}")
            return None

        _osh, _osb = _build_optional_signals(language_drift_context, short_interest_context)
        prompt = SCORE_PROMPT_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            frequency=frequency,
            period=period,
            peer_context=(peer_context or "(peer context unavailable this period)")[:1_000],
            overview_context=overview_context[:5_000],
            filing_context=filing_context[:45_000],
            earnings_context=_qa_weighted_truncate(earnings_context, 15_000),  # Q&A-weighted (#21)
            transcript_qa_split_context=transcript_qa_split_context[:8_000],
            earnings_history_context=earnings_history_context[:3_000],
            income_statement_context=income_statement_context[:5_000],
            balance_sheet_context=balance_sheet_context[:5_000],
            cash_flow_context=cash_flow_context[:5_000],
            insider_context=insider_context[:4_000],
            institutional_context=institutional_context[:4_000],
            comment_letters_context=comment_letters_context[:6_000],
            optional_signal_heuristics=_osh,
            optional_signals_block=_osb,
            news_context=news_context[:6_000],
            concentration_instruction=concentration_instruction[:1_000],
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,   # key_positives/key_risks strings can hit 600+ tokens; 400 was too low
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent financial scoring
            )
            raw = response.content[0].text.strip()
            # Strip any accidental markdown code fences
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)

            # Validate required fields
            assert "score" in parsed, "Missing 'score' field"
            assert 0.0 <= float(parsed["score"]) <= 1.0, "Score out of range"
            parsed["score"] = float(parsed["score"])
            return parsed

        except (APIConnectionError, APITimeoutError) as e:
            logger.error(f"Claude API connection error for {ticker}: {e}")
            return None
        except RateLimitError as e:
            logger.error(f"Claude API rate limit for {ticker}: {e}")
            return None
        except (json.JSONDecodeError, AssertionError, KeyError) as e:
            logger.error(f"LLM score parse error for {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected LLM error for {ticker}: {e}", exc_info=True)
            return None

    def _call_sync(self, prompt: str, max_tokens: int = 800) -> Optional[str]:
        """One synchronous Claude text call. Returns raw text, or None on error."""
        if self.client is None:
            return None
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Sync Claude call failed: {e}")
            return None

    def score_two_stage_sync(self, ticker: str, single_stage_prompt: str) -> Optional[dict]:
        """
        Synchronous two-stage score for ONE ticker (on-demand path), mirroring the
        batch two-stage logic without the batch poll: stage 1 extracts a fact
        sheet, stage 2 scores from it with the calibration/arithmetic contract.

        Falls back to a single synchronous call on the original prompt if the
        ticker has no calibration marker (no extraction prompt) or if extraction
        fails — so on-demand scoring never loses its number to the machinery.
        Result is tagged "two_stage": bool and carries the extracted "fact_sheet"
        (when available) for the single-stock detail view.
        """
        if self.client is None:
            return None

        # No two-stage requested, or no extractable materials → single call.
        extraction_prompt = build_extraction_prompt(single_stage_prompt) if two_stage_enabled() else None
        if extraction_prompt is None:
            raw = self._call_sync(single_stage_prompt)
            if raw is None:
                return None
            parsed = self._parse_score_json(ticker, raw)
            if parsed is not None:
                parsed["two_stage"] = False
            return parsed

        # Stage 1 — fact extraction.
        fact_raw = self._call_sync(extraction_prompt)
        fact_sheet = None
        if fact_raw is not None:
            try:
                fact_sheet = json.loads(self._strip_fences(fact_raw))
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.warning(f"Sync extraction parse failed for {ticker}: {e} — falling back to single-stage")

        if not isinstance(fact_sheet, dict):
            raw = self._call_sync(single_stage_prompt)
            if raw is None:
                return None
            parsed = self._parse_score_json(ticker, raw)
            if parsed is not None:
                parsed["two_stage"] = False
            return parsed

        # Stage 2 — score from the fact sheet.
        stage2_raw = self._call_sync(build_stage2_prompt(ticker, fact_sheet))
        if stage2_raw is None:
            return None
        parsed = self._parse_score_json(ticker, stage2_raw)
        if parsed is not None:
            parsed["two_stage"] = True
            parsed["fact_sheet"] = fact_sheet
        return parsed

    def build_prompt(
        self,
        ticker: str,
        company_name: str,
        frequency: str,
        period: str,
        filing_context: str = "",
        earnings_context: str = "",
        peer_context: str = "",
        news_context: str = "",
        earnings_history_context: str = "",
        overview_context: str = "",
        income_statement_context: str = "",
        balance_sheet_context: str = "",
        cash_flow_context: str = "",
        insider_context: str = "",
        institutional_context: str = "",
        transcript_qa_split_context: str = "",
        comment_letters_context: str = "",
        language_drift_context: str = "",
        short_interest_context: str = "",
        concentration_instruction: str = "",
    ) -> str:
        """Build the formatted prompt string for a single ticker (used by score_batch)."""
        _osh, _osb = _build_optional_signals(language_drift_context, short_interest_context)
        return SCORE_PROMPT_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            frequency=frequency,
            period=period,
            peer_context=(peer_context or "(peer context unavailable this period)")[:1_000],
            overview_context=overview_context[:5_000],
            filing_context=filing_context[:45_000],
            earnings_context=_qa_weighted_truncate(earnings_context, 15_000),
            transcript_qa_split_context=transcript_qa_split_context[:8_000],
            earnings_history_context=earnings_history_context[:3_000],
            income_statement_context=income_statement_context[:5_000],
            balance_sheet_context=balance_sheet_context[:5_000],
            cash_flow_context=cash_flow_context[:5_000],
            insider_context=insider_context[:4_000],
            institutional_context=institutional_context[:4_000],
            comment_letters_context=comment_letters_context[:6_000],
            optional_signal_heuristics=_osh,
            optional_signals_block=_osb,
            news_context=news_context[:6_000],
            concentration_instruction=concentration_instruction[:1_000],
        )

    def _run_batch_raw(
        self,
        prompts: dict,           # {ticker: prompt_string}
        label: str,
        max_tokens: int = 800,
        poll_interval: int = 60,
        max_wait: int = 7200,
    ) -> dict:
        """
        Submit prompts as one Anthropic Message Batch; poll; return raw text.

        Returns {ticker: raw_response_text} for succeeded results only.
        Shared machinery for single-stage scoring, extraction, and stage-2
        scoring — identical custom_id sanitization, polling, and logging.
        """
        if self.client is None:
            logger.warning("LLM scorer not initialized — skipping batch scoring")
            return {}
        if not prompts:
            return {}

        import time as _time

        # Anthropic batch custom_id must match ^[a-zA-Z0-9_-]{1,64}$ — tickers
        # like BRK.B (dot) violate it and 400 the ENTIRE batch, zeroing LLM
        # scores for every name. Sanitize to a safe id and keep a reverse map.
        id_to_ticker = _safe_custom_id_map(list(prompts.keys()))
        ticker_to_id = {t: i for i, t in id_to_ticker.items()}

        requests = [
            {
                "custom_id": ticker_to_id[ticker],
                "params": {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
            for ticker, prompt in prompts.items()
        ]

        try:
            batch = self.client.beta.messages.batches.create(requests=requests)
            logger.info(f"Batch submitted ({label}): {batch.id} — {len(requests)} requests")
        except Exception as e:
            logger.error(f"Batch submission failed ({label}): {e}", exc_info=True)
            return {}

        elapsed = 0
        while elapsed < max_wait:
            _time.sleep(poll_interval)
            elapsed += poll_interval
            try:
                status = self.client.beta.messages.batches.retrieve(batch.id)
            except Exception as e:
                logger.warning(f"Batch poll error (will retry): {e}")
                continue

            counts = status.request_counts
            logger.info(
                f"Batch {batch.id} ({label}): {status.processing_status} | "
                f"processing={counts.processing} succeeded={counts.succeeded} "
                f"errored={counts.errored} elapsed={elapsed}s"
            )
            if status.processing_status == "ended":
                break
        else:
            logger.error(f"Batch {batch.id} ({label}) timed out after {max_wait}s")
            return {}

        raw_by_ticker = {}
        try:
            for result in self.client.beta.messages.batches.results(batch.id):
                ticker = id_to_ticker.get(result.custom_id, result.custom_id)
                if result.result.type != "succeeded":
                    logger.warning(f"Batch result {ticker} ({label}): {result.result.type}")
                    continue
                try:
                    raw = result.result.message.content[0].text.strip()
                    raw_by_ticker[ticker] = raw
                except (IndexError, AttributeError) as e:
                    logger.error(f"Batch raw-text error for {ticker} ({label}): {e}")
        except Exception as e:
            logger.error(f"Batch results retrieval failed ({label}): {e}", exc_info=True)

        return raw_by_ticker

    @staticmethod
    def _strip_fences(raw: str) -> str:
        return raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    def _parse_score_json(self, ticker: str, raw: str) -> Optional[dict]:
        """Parse a scoring response. Requires score in [0,1]; tolerates extra
        fields (band_base, adjustments) which flow into llm_reasoning_json.
        Lightly validates the two-stage arithmetic without rejecting scores."""
        try:
            parsed = json.loads(self._strip_fences(raw))
            assert "score" in parsed
            assert 0.0 <= float(parsed["score"]) <= 1.0
            parsed["score"] = float(parsed["score"])
        except (json.JSONDecodeError, AssertionError, IndexError, TypeError, ValueError) as e:
            logger.error(f"Batch parse error for {ticker}: {e}")
            return None

        if "band_base" in parsed and isinstance(parsed.get("adjustments"), list):
            try:
                implied = float(parsed["band_base"]) + sum(
                    float(a.get("delta", 0)) for a in parsed["adjustments"]
                )
                if abs(implied - parsed["score"]) > 0.02:
                    logger.warning(
                        f"Two-stage arithmetic drift {ticker}: score={parsed['score']} "
                        f"vs band_base+deltas={implied:.3f} (keeping reported score)"
                    )
            except (TypeError, ValueError):
                pass
        return parsed

    def score_batch(
        self,
        prompts: dict,          # {ticker: prompt_string}
        poll_interval: int = 60,
        max_wait: int = 7200,   # 2 hours hard ceiling
    ) -> dict:
        """
        Score all tickers via Anthropic Message Batches (50% cost reduction).

        Single-stage (LLM_TWO_STAGE=0): one batch with the provided prompts.
        Two-stage (default): batch 1 extracts a structured fact sheet per
        ticker from the prompt's materials; batch 2 scores from the fact
        sheet with mandatory adjustment arithmetic. Any ticker whose
        extraction fails (parse error, batch error, missing marker) falls
        back to its original single-stage prompt within batch 2 — no ticker
        loses its LLM score to the two-stage machinery.

        Returns {ticker: parsed_score_dict}; each result is tagged with
        "two_stage": bool for auditing. Tickers that error or time out are
        omitted → caller uses w=1.0 fallback.
        """
        if not prompts:
            return {}

        if not two_stage_enabled():
            raw = self._run_batch_raw(
                prompts, "scoring", max_tokens=800,
                poll_interval=poll_interval, max_wait=max_wait,
            )
            scores = {}
            for ticker, txt in raw.items():
                parsed = self._parse_score_json(ticker, txt)
                if parsed is not None:
                    parsed["two_stage"] = False
                    scores[ticker] = parsed
            logger.info(f"Batch complete: {len(scores)}/{len(prompts)} tickers scored successfully")
            return scores

        # ── Stage 1: fact extraction ──────────────────────────────
        extract_prompts = {}
        for ticker, prompt in prompts.items():
            ep = build_extraction_prompt(prompt)
            if ep is not None:
                extract_prompts[ticker] = ep
        logger.info(f"Two-stage LLM enabled: extracting facts for {len(extract_prompts)}/{len(prompts)} tickers")

        raw1 = self._run_batch_raw(
            extract_prompts, "extraction", max_tokens=1024,
            poll_interval=poll_interval, max_wait=max_wait,
        )
        facts = {}
        parse_failed = 0
        for ticker, txt in raw1.items():
            try:
                fs = json.loads(self._strip_fences(txt))
                assert isinstance(fs, dict) and fs
                facts[ticker] = fs
            except (json.JSONDecodeError, AssertionError) as e:
                parse_failed += 1
                logger.warning(f"Extraction parse failed {ticker} — falling back to single-stage: {e}")

        fallback = [t for t in prompts if t not in facts]
        logger.info(
            f"Two-stage extraction: ok={len(facts)} parse_failed={parse_failed} "
            f"batch_missing={len(fallback) - parse_failed if len(fallback) >= parse_failed else 0} | "
            f"single-stage fallback for {len(fallback)} ticker(s)"
        )

        # ── Stage 2: score (fact sheet where available, original prompt otherwise)
        stage2_prompts = {}
        for ticker, prompt in prompts.items():
            if ticker in facts:
                stage2_prompts[ticker] = build_stage2_prompt(ticker, facts[ticker])
            else:
                stage2_prompts[ticker] = prompt

        raw2 = self._run_batch_raw(
            stage2_prompts, "scoring", max_tokens=1024,
            poll_interval=poll_interval, max_wait=max_wait,
        )
        scores = {}
        for ticker, txt in raw2.items():
            parsed = self._parse_score_json(ticker, txt)
            if parsed is not None:
                parsed["two_stage"] = ticker in facts
                scores[ticker] = parsed
        logger.info(f"Batch complete: {len(scores)}/{len(prompts)} tickers scored successfully")
        return scores
