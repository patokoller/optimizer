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
SCORE_PROMPT_TEMPLATE = """You are a quantitative analyst scoring {ticker} ({company_name}) for a {frequency} portfolio rebalancing.

You have been provided with six layers of information. Use all available layers — weight recent evidence more heavily.

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

Return ONLY valid JSON with this exact structure:
{{
  "score": <float 0.0–1.0, where 1.0 = strongest conviction buy, 0.0 = strongest conviction sell>,
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

--- EARNINGS SURPRISE HISTORY ---
{earnings_history_context}

--- BALANCE SHEET (last 4 quarters) ---
{balance_sheet_context}

--- CASH FLOW (last 4 quarters) ---
{cash_flow_context}

--- INSIDER TRANSACTIONS (Form 4 — last 180 days) ---
{insider_context}

--- INSTITUTIONAL HOLDINGS (latest 13F filings) ---
{institutional_context}

--- RECENT NEWS & SENTIMENT ---
{news_context}
"""


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

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — LLM scoring unavailable.")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def score(
        self,
        ticker: str,
        company_name: str,
        frequency: str,
        period: str,
        filing_context: str,
        earnings_context: str,
        news_context: str = "",
        earnings_history_context: str = "",
        overview_context: str = "",
        balance_sheet_context: str = "",
        cash_flow_context: str = "",
        insider_context: str = "",
        institutional_context: str = "",
    ) -> Optional[dict]:
        """
        Call Claude API with full enriched context (Phases A + B + C).

        Token budget (200K context window):
          - Company overview:       up to  5K chars
          - SEC filings:            up to 55K chars
          - Earnings transcript:    up to 35K chars
          - Earnings history:       up to  3K chars
          - Balance sheet:          up to  5K chars
          - Cash flow:              up to  5K chars
          - Insider transactions:   up to  4K chars
          - Institutional holdings: up to  4K chars
          - News:                   up to  8K chars
        """
        if self.client is None:
            logger.warning(f"LLM scorer not initialized — skipping {ticker}")
            return None

        prompt = SCORE_PROMPT_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            frequency=frequency,
            period=period,
            overview_context=overview_context[:5_000],
            filing_context=filing_context[:55_000],
            earnings_context=earnings_context[:35_000],
            earnings_history_context=earnings_history_context[:3_000],
            balance_sheet_context=balance_sheet_context[:5_000],
            cash_flow_context=cash_flow_context[:5_000],
            insider_context=insider_context[:4_000],
            institutional_context=institutional_context[:4_000],
            news_context=news_context[:8_000],
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
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


def get_cache_key(ticker: str, period: str, strategy: str) -> str:
    """Cache key: ticker + year-month + strategy."""
    return f"{ticker}:{period}:{strategy}"


def is_cached(db_scores: list, ticker: str, period: str) -> Optional[dict]:
    """
    Check if LLM score exists in the DB for this ticker/month.
    Returns the cached reasoning dict or None.
    """
    for score in db_scores:
        if score.ticker == ticker and score.llm_reasoning_json:
            return score.llm_reasoning_json
    return None
