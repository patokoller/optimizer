"""
app/data/phase_d.py
─────────────────────────────────────────────────────────────────────────────
Phase D enrichment signals — five legally-available, structurally underexploited
data sources that complement the Phase A/B/C pipeline.

P3-1  SEC Comment Letters     — EDGAR correspondence filings
P3-2  Language Drift          — 8-quarter transcript trend analysis
P3-3  Q&A Separation          — prepared remarks vs analyst Q&A split
P3-4  Customer Concentration  — extracted from 10-K footnotes via Claude
P3-5  FINRA Short Interest    — twice-monthly short interest + days-to-cover

All methods:
  - Never raise — return empty string on failure
  - Cache-aware where possible (slow signals)
  - Designed to plug into enrichment_cache as new context slots
"""
import re
import time
import logging
import requests
from datetime import datetime, date, timedelta
from typing import Optional

logger = logging.getLogger("phase_d")

EDGAR_BASE  = "https://data.sec.gov"
SEC_BASE    = "https://www.sec.gov"
USER_AGENT  = "AlphaLens Research contact@alphalens.io"


# ─────────────────────────────────────────────────────────────────────────────
# P3-1  SEC Comment Letters
# ─────────────────────────────────────────────────────────────────────────────

def get_sec_comment_letters(
    ticker: str,
    edgar_client,
    lookback_days: int = 730,
) -> str:
    """
    Fetch SEC comment letters (CORRESP / UPLOAD form types) from EDGAR.

    Comment letters are formal SEC staff queries to companies about their
    accounting treatment, disclosure adequacy, or financial statement presentation.
    They typically precede restatements or guidance cuts by 2–6 months and are
    publicly available but rarely systematically tracked.

    EDGAR form types:
      CORRESP  — company's response to SEC comment letter
      UPLOAD   — SEC staff's original comment letter

    Key signals for Claude:
      - UPLOAD present: SEC is actively questioning this company's accounting
      - Topics: revenue recognition, goodwill impairment, segment reporting,
        going concern, related-party transactions
      - Recency: letter in last 6 months = elevated risk
      - Multiple rounds: back-and-forth indicates SEC not satisfied
    """
    try:
        cik = edgar_client.get_cik(ticker)
    except Exception as e:
        logger.debug(f"Comment letters: CIK lookup failed for {ticker}: {e}")
        return ""

    try:
        resp = requests.get(
            f"{EDGAR_BASE}/submissions/CIK{cik}.json",
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        data     = resp.json()
        filings  = data.get("filings", {}).get("recent", {})
        forms    = filings.get("form", [])
        dates    = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
    except Exception as e:
        logger.debug(f"Comment letters: EDGAR fetch failed for {ticker}: {e}")
        return ""

    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    comment_filings = []

    for form, filing_date, acc in zip(forms, dates, accessions):
        if form not in ("CORRESP", "UPLOAD", "DFAN14A"):
            continue
        try:
            fd = datetime.strptime(filing_date[:10], "%Y-%m-%d")
        except ValueError:
            continue
        if fd < cutoff:
            continue
        comment_filings.append({
            "form":         form,
            "filing_date":  filing_date[:10],
            "accession":    acc,
        })

    if not comment_filings:
        return ""

    lines = [
        f"=== SEC COMMENT LETTERS / CORRESPONDENCE ({ticker}) — Last {lookback_days // 365} years ===",
        f"⚠ {len(comment_filings)} correspondence filing(s) found. Active SEC comment letters",
        f"often precede restatements or guidance cuts by 2–6 months.\n",
    ]

    # Fetch full text of the most recent 2 letters (they're short)
    for filing in sorted(comment_filings, key=lambda x: x["filing_date"], reverse=True)[:2]:
        lines.append(f"  [{filing['form']} | {filing['filing_date']}]")
        try:
            acc_clean = filing["accession"].replace("-", "")
            cik_int   = int(cik)
            index_url = (
                f"{SEC_BASE}/Archives/edgar/data/{cik_int}"
                f"/{acc_clean}/{filing['accession']}-index.htm"
            )
            r = requests.get(index_url, headers={"User-Agent": USER_AGENT}, timeout=20)
            r.raise_for_status()
            # Extract the first document link from the index
            doc_links = re.findall(
                r'href="(/Archives/edgar/data/\S+?\.(?:htm|txt))"',
                r.text, re.IGNORECASE
            )
            if doc_links:
                doc_url = f"{SEC_BASE}{doc_links[0]}"
                dr = requests.get(doc_url, headers={"User-Agent": USER_AGENT}, timeout=20)
                dr.raise_for_status()
                text = re.sub(r"<[^>]+>", " ", dr.text)
                text = re.sub(r"\s+", " ", text).strip()
                lines.append(text[:4_000])  # cap per letter
        except Exception as e:
            logger.debug(f"Comment letter text fetch failed for {ticker}: {e}")
        lines.append("")
        time.sleep(0.3)

    result = "\n".join(lines)
    logger.info(f"Phase D comment letters: {ticker} — {len(comment_filings)} filing(s)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# P3-2  Management Language Drift  (8-quarter trend)
# ─────────────────────────────────────────────────────────────────────────────

# Hedging / uncertainty language markers (from academic NLP literature on earnings call analysis)
_HEDGE_PATTERNS = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bpossibly\b",
    r"\bsubject to\b", r"\bwe continue to monitor\b", r"\bremains uncertain\b",
    r"\bchallenging\b", r"\bheadwinds\b", r"\bpressure\b", r"\bvolatility\b",
    r"\bif conditions\b", r"\bdepending on\b", r"\bwe believe\b",
]

_SPECIFICITY_PATTERNS = [
    r"\$[\d,]+\s*(?:billion|million|B|M)\b",   # concrete dollar amounts
    r"\b\d+\.?\d*\s*%\b",                       # specific percentages
    r"\bQ[1-4]\s*\d{4}\b",                      # specific quarters
    r"\bfiscal\s*\d{4}\b",                       # fiscal years
]


def _hedge_score(text: str) -> float:
    """Return hedging density: hedge pattern hits per 1000 words."""
    if not text:
        return 0.0
    words = len(text.split())
    if words == 0:
        return 0.0
    hits = sum(len(re.findall(p, text, re.IGNORECASE)) for p in _HEDGE_PATTERNS)
    return round(hits / words * 1000, 2)


def _specificity_score(text: str) -> float:
    """Return guidance specificity: concrete numbers per 1000 words."""
    if not text:
        return 0.0
    words = len(text.split())
    if words == 0:
        return 0.0
    hits = sum(len(re.findall(p, text, re.IGNORECASE)) for p in _SPECIFICITY_PATTERNS)
    return round(hits / words * 1000, 2)


def _split_qa(text: str) -> tuple[str, str]:
    """
    Split transcript into prepared remarks and Q&A session.

    Heuristic: Q&A typically starts with "Operator:" or "Questions and Answers"
    or "Question-and-Answer Session" marker.
    """
    qa_markers = [
        r"Questions?\s*(?:and\s*)?Answers?\s*Session",
        r"Q&A\s*Session",
        r"We'll now begin.*question",
        r"Operator.*questions",
        r"^\s*Operator\s*$",
        r"QUESTION.*ANSWER",
    ]
    earliest_pos = len(text)
    for marker in qa_markers:
        m = re.search(marker, text, re.IGNORECASE | re.MULTILINE)
        if m and m.start() < earliest_pos:
            earliest_pos = m.start()

    if earliest_pos < len(text) * 0.9:  # found a plausible split point
        return text[:earliest_pos], text[earliest_pos:]
    return text, ""  # no clear split — treat all as prepared


def compute_language_drift(
    ticker: str,
    av_client,
    n_quarters: int = 8,
) -> str:
    """
    P3-2 + P3-3: Fetch 8 quarters of transcripts, compute language drift metrics,
    and split each into prepared remarks vs Q&A.

    Returns a structured analysis string for Claude with:
      - Hedging trend per quarter (density score)
      - Specificity trend per quarter
      - Divergence between prepared and Q&A tone
      - Overall trend label: improving / stable / deteriorating

    This is a significant signal: a company whose hedging language increases
    monotonically over 4+ quarters is statistically more likely to miss guidance.
    """
    from app.data.clients import AlphaVantageClient  # avoid circular

    # Try to fetch up to 8 quarters
    quarterly_data = []
    current_year = datetime.utcnow().year

    for year_offset in range(2):          # last 2 years
        for quarter in [1, 2, 3, 4]:
            year = current_year - year_offset
            # Skip future quarters
            if year == current_year and quarter > (datetime.utcnow().month - 1) // 3 + 1:
                continue
            quarter_str = f"{year}Q{quarter}"
            try:
                transcript = av_client.get_earnings_transcript(ticker, quarters=[quarter_str])
                if transcript and len(transcript) > 500:
                    quarterly_data.append({
                        "quarter": quarter_str,
                        "text":    transcript,
                    })
            except Exception:
                pass
            if len(quarterly_data) >= n_quarters:
                break
        if len(quarterly_data) >= n_quarters:
            break

    if len(quarterly_data) < 2:
        return ""  # not enough data for drift analysis

    # Compute metrics per quarter (oldest first)
    quarterly_data = sorted(quarterly_data, key=lambda x: x["quarter"])

    rows = []
    qa_divergences = []
    for q in quarterly_data:
        prepared, qa = _split_qa(q["text"])
        hedge_prep  = _hedge_score(prepared)
        hedge_qa    = _hedge_score(qa)
        spec_prep   = _specificity_score(prepared)
        spec_qa     = _specificity_score(qa)

        qa_divergence = None
        if qa:
            # Positive divergence = management is MORE hedged in Q&A than prepared remarks
            # This is a red flag — they knew more than they said
            qa_divergence = round(hedge_qa - hedge_prep, 2)
            qa_divergences.append(qa_divergence)

        rows.append({
            "quarter":    q["quarter"],
            "hedge_prep": hedge_prep,
            "spec_prep":  spec_prep,
            "qa_div":     qa_divergence,
            "has_qa":     bool(qa),
        })

    # Trend detection: compare first half vs second half of available data
    mid = len(rows) // 2
    early_hedge = sum(r["hedge_prep"] for r in rows[:mid]) / max(mid, 1)
    late_hedge  = sum(r["hedge_prep"] for r in rows[mid:]) / max(len(rows) - mid, 1)
    early_spec  = sum(r["spec_prep"]  for r in rows[:mid]) / max(mid, 1)
    late_spec   = sum(r["spec_prep"]  for r in rows[mid:]) / max(len(rows) - mid, 1)

    hedge_change = late_hedge - early_hedge
    spec_change  = late_spec  - early_spec

    if hedge_change > 1.5 and spec_change < -0.5:
        trend = "DETERIORATING — rising hedging language, declining specificity"
    elif hedge_change < -1.0 and spec_change > 0.3:
        trend = "IMPROVING — declining hedging language, increasing specificity"
    else:
        trend = "STABLE"

    # Average Q&A divergence
    avg_qa_div = round(sum(qa_divergences) / len(qa_divergences), 2) if qa_divergences else None

    lines = [
        f"=== MANAGEMENT LANGUAGE DRIFT ({ticker}) — Last {len(rows)} Quarters ===",
        f"Overall trend: {trend}",
        f"Early period hedge density: {early_hedge:.2f} / Late period: {late_hedge:.2f} (per 1000 words)",
        f"Early guidance specificity: {early_spec:.2f} / Late: {late_spec:.2f} (concrete numbers per 1000 words)",
    ]
    if avg_qa_div is not None:
        lines.append(
            f"Avg Q&A vs Prepared divergence: {avg_qa_div:+.2f} "
            f"({'⚠ higher hedging in Q&A than prepared — management more cautious under questioning' if avg_qa_div > 1.5 else 'normal'})"
        )

    lines.append("\nQuarter-by-quarter:")
    for r in rows:
        qa_str = f" | Q&A div: {r['qa_div']:+.2f}" if r["qa_div"] is not None else ""
        lines.append(
            f"  {r['quarter']}: hedge={r['hedge_prep']:.2f}, specificity={r['spec_prep']:.2f}{qa_str}"
        )

    lines.append(
        "\nNote: hedge density = frequency of uncertainty language per 1000 words. "
        "Specificity = frequency of concrete numbers/dates per 1000 words. "
        "Trend comparison uses first half vs second half of available quarters."
    )

    result = "\n".join(lines)
    logger.info(f"Phase D language drift: {ticker} — {len(rows)} quarters, trend={trend.split('—')[0].strip()}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# P3-3  Q&A Separation  (split already done in compute_language_drift above)
#       Also expose as a standalone function for use in the prompt template
# ─────────────────────────────────────────────────────────────────────────────

def split_transcript_qa(transcript: str) -> tuple[str, str]:
    """
    Public wrapper around _split_qa for use in the enrichment pipeline.
    Returns (prepared_remarks, qa_session).
    If no clear split found, returns (full_text, "").
    """
    return _split_qa(transcript)


# ─────────────────────────────────────────────────────────────────────────────
# P3-5  FINRA Short Interest
# ─────────────────────────────────────────────────────────────────────────────

FINRA_API = "https://api.finra.org/data/group/OTCMarket/name/otcShortInterest"

def get_short_interest(ticker: str) -> str:
    """
    Fetch short interest data from FINRA public API.

    FINRA publishes short interest twice monthly (settlement dates ~15th and ~end).
    Data: short interest shares, days-to-cover (short interest / avg daily volume).

    Key signals for Claude:
      - Days-to-cover > 10: heavily shorted, potential squeeze
      - Rising short interest + rising stock price: short squeeze building
      - High short + high Claude score: market disagrees — investigate why
      - Declining short interest + positive momentum: shorts covering = bullish
      - Days-to-cover < 1: negligible short position

    FINRA API is free, no auth required for public short interest data.
    """
    try:
        # FINRA short interest API: https://api.finra.org
        resp = requests.get(
            FINRA_API,
            params={
                "limit": 10,
                "compareFilters": f"[{{\"fieldName\":\"issueSymbolIdentifier\","
                                  f"\"compareType\":\"equal\",\"fieldValue\":\"{ticker}\"}}]",
                "fields": "issueSymbolIdentifier,settlementDate,shortInterestQty,"
                          "daysToCoverQty,revisionFlag",
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug(f"FINRA short interest failed for {ticker}: {e}")
        return ""

    if not data or not isinstance(data, list) or len(data) == 0:
        return ""

    # Sort by settlement date desc
    records = sorted(data, key=lambda x: x.get("settlementDate", ""), reverse=True)

    lines = [f"=== FINRA SHORT INTEREST ({ticker}) ===\n"]

    for r in records[:4]:  # last 4 settlement dates (~2 months)
        settle     = r.get("settlementDate", "?")
        short_qty  = r.get("shortInterestQty")
        dtc        = r.get("daysToCoverQty")

        short_str = f"{int(short_qty):,}" if short_qty else "N/A"
        dtc_str   = f"{float(dtc):.1f} days" if dtc else "N/A"

        # Signal interpretation
        if dtc and float(dtc) > 10:
            signal = "⚠ HIGH — potential squeeze risk"
        elif dtc and float(dtc) > 5:
            signal = "Elevated"
        elif dtc and float(dtc) < 1:
            signal = "Negligible"
        else:
            signal = "Moderate"

        lines.append(
            f"  {settle}: short_interest={short_str} shares | "
            f"days_to_cover={dtc_str} | signal={signal}"
        )

    # Trend: compare most recent to 2 periods ago
    if len(records) >= 3:
        recent_si = records[0].get("shortInterestQty", 0) or 0
        older_si  = records[2].get("shortInterestQty", 0) or 0
        if older_si > 0:
            change_pct = (recent_si - older_si) / older_si * 100
            direction  = "▲ rising" if change_pct > 5 else ("▼ falling" if change_pct < -5 else "→ stable")
            lines.append(f"\nShort interest trend (2 periods): {direction} ({change_pct:+.1f}%)")

    result = "\n".join(lines)
    logger.info(f"Phase D short interest: {ticker} — {len(records)} periods")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# P3-4  Customer Concentration  (Claude-extracted from 10-K text)
#       This runs as a prompt instruction, not a data fetch.
#       We inject a concentration extraction task into the filing context.
# ─────────────────────────────────────────────────────────────────────────────

CONCENTRATION_INSTRUCTION = """
ADDITIONAL EXTRACTION TASK — CUSTOMER CONCENTRATION:
While reading the SEC filings above, identify and list:
1. Any customer, partner, or government agency representing >10% of revenue
   (required disclosure under ASC 280 / FASB Topic 280)
2. Any language about customer contract renewal risk, exclusivity clauses,
   or customers building equivalent capabilities in-house
3. Any geographic concentration risk (single country/region >30% of revenue)

Include this in your analysis as a "concentration_risks" factor in your JSON output.
If no material concentration is found, note "no material customer concentration disclosed."
"""

def get_concentration_instruction() -> str:
    """
    Return the concentration extraction instruction to prepend to filing context.
    This is injected into the prompt — no API call needed.
    """
    return CONCENTRATION_INSTRUCTION
