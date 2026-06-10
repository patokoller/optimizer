"""
Enrichment cache — monthly AV API response cache.

Caches the 8 slower enrichment signals per ticker per calendar month:
  transcript, earnings_history, overview, balance_sheet, cash_flow,
  insider, institutional

NOT cached (re-fetched every run):
  news — time-sensitive, cheap (1 call), and stale news is worse than no news

Cache key: (ticker, YYYY-MM)
Invalidation: automatic on new month; manual via cache_bust=True param

Why cache: 8 AV calls × 100 tickers × ~1.2s = ~16 min of AV calls per run.
With cache: ~2 min (only news re-fetched, ~1 call × 100 tickers).
"""
import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger("enrichment_cache")

# Signals that are slow-changing and safe to cache for a full month
CACHEABLE_KEYS = {
    # Phase A
    "transcript",
    "transcript_qa_split",
    "earnings_history",
    # Phase B
    "overview",
    "balance_sheet",
    "cash_flow",
    # Phase C
    "insider",
    "institutional",
    # Phase D (slow — cached monthly)
    "comment_letters",
    "language_drift",
    # short_interest and concentration_instruction not cached:
    # short_interest refreshes every 2 weeks (FINRA schedule)
    # concentration_instruction is a static string
}

# Signal that must be re-fetched every run (time-sensitive)
REALTIME_KEYS = {"news", "short_interest", "concentration_instruction"}


def _cache_month() -> str:
    """Current calendar month as YYYY-MM."""
    from datetime import timezone
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _cache_quarter() -> str:
    """Current calendar quarter as YYYY-QN (e.g. '2026-Q2').

    Used as cache key for language drift — drift is computed from 8 quarters
    of transcript history and only becomes stale when a new quarter begins.
    Storing with a quarterly key means we only recompute 4×/year instead of
    12×/year, saving ~588 AV transcript calls annually.
    """
    from datetime import timezone
    now = datetime.now(timezone.utc)
    q = (now.month - 1) // 3 + 1
    return f"{now.year}-Q{q}"


def get_cached(db: Session, ticker: str, month: Optional[str] = None) -> Optional[dict]:
    """
    Return cached enrichment context for ticker in the given month.
    Returns None if no cache entry exists.
    """
    month = month or _cache_month()
    try:
        row = db.execute(
            text("SELECT context FROM enrichment_cache WHERE ticker = :t AND cache_month = :m"),
            {"t": ticker, "m": month},
        ).fetchone()
        if row:
            ctx = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            logger.debug(f"Cache HIT: {ticker} {month}")
            return ctx
    except Exception as e:
        logger.warning(f"Cache read failed for {ticker}: {e}")
    return None


def set_cached(db: Session, ticker: str, context: dict, month: Optional[str] = None) -> None:
    """
    Upsert enrichment context for ticker in the given month.
    Only stores CACHEABLE_KEYS — strips realtime signals before persisting.
    """
    month = month or _cache_month()
    cacheable = {k: v for k, v in context.items() if k in CACHEABLE_KEYS}
    try:
        db.execute(
            text("""
                INSERT INTO enrichment_cache (ticker, cache_month, context, fetched_at)
                VALUES (:t, :m, CAST(:ctx AS jsonb), now())
                ON CONFLICT (ticker, cache_month)
                DO UPDATE SET context = EXCLUDED.context, fetched_at = now()
            """),
            {"t": ticker, "m": month, "ctx": json.dumps(cacheable)},
        )
        db.commit()
        logger.debug(f"Cache SET: {ticker} {month}")
    except Exception as e:
        logger.warning(f"Cache write failed for {ticker}: {e}")
        db.rollback()


# ── Quarterly drift cache helpers ─────────────────────────────────────────────
# Language drift uses 8 quarters of transcript history and changes once per quarter.
# We store it under a YYYY-QN key in the same enrichment_cache table so it is
# only recomputed 4× per year (saving ~588 AV calls/year vs monthly recompute).

def get_drift_cached(db: Session, ticker: str, quarter: Optional[str] = None) -> Optional[str]:
    """Return cached language_drift string for the given quarter, or None on miss."""
    quarter = quarter or _cache_quarter()
    try:
        row = db.execute(
            text("SELECT context FROM enrichment_cache WHERE ticker = :t AND cache_month = :q"),
            {"t": ticker, "q": quarter},
        ).fetchone()
        if row:
            ctx = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            return ctx.get("language_drift", "") or None
    except Exception as e:
        logger.warning(f"Drift cache read failed for {ticker}: {e}")
    return None


def set_drift_cached(db: Session, ticker: str, drift_text: str, quarter: Optional[str] = None) -> None:
    """Persist language_drift under a quarterly cache key."""
    quarter = quarter or _cache_quarter()
    try:
        db.execute(
            text("""
                INSERT INTO enrichment_cache (ticker, cache_month, context, fetched_at)
                VALUES (:t, :q, CAST(:ctx AS jsonb), now())
                ON CONFLICT (ticker, cache_month)
                DO UPDATE SET context = EXCLUDED.context, fetched_at = now()
            """),
            {"t": ticker, "q": quarter, "ctx": json.dumps({"language_drift": drift_text})},
        )
        db.commit()
    except Exception as e:
        logger.warning(f"Drift cache write failed for {ticker}: {e}")
        db.rollback()


DRIFT_EMPTY_SENTINEL = "__NO_DRIFT_DATA__"


def get_or_fetch_drift(db: Session, ticker: str, av_client) -> str:
    """
    Return language drift analysis for ticker, using quarterly cache.

    Cache miss → calls compute_language_drift() (8 AV transcript calls, ~15s).
    Cache hit  → returns instantly from DB.

    Empty results (no transcripts: ETFs, dotted/foreign tickers, micro-caps)
    are cached as a tombstone for the quarter — otherwise these names re-attempt
    their AV transcript fetches on EVERY run, forever. Trade-off: a name whose
    first-ever transcript appears mid-quarter won't get drift until next
    quarter; acceptable since the affected set is dominated by names that will
    never have transcripts.

    Called once per ticker per quarter; all monthly runs within a quarter
    after the first will hit the cache at no cost.
    """
    from app.data.phase_d import compute_language_drift
    quarter = _cache_quarter()

    cached = get_drift_cached(db, ticker, quarter)
    if cached:
        if cached == DRIFT_EMPTY_SENTINEL:
            logger.debug(f"Drift cache HIT {ticker} ({quarter}) — empty tombstone")
            return ""
        logger.debug(f"Drift cache HIT {ticker} ({quarter})")
        return cached

    logger.info(f"Drift cache MISS {ticker} ({quarter}) — computing language drift")
    try:
        drift = compute_language_drift(ticker, av_client, n_quarters=8)
    except Exception as e:
        logger.warning(f"Language drift failed for {ticker}: {e}")
        drift = ""

    if drift:
        set_drift_cached(db, ticker, drift, quarter)
        logger.info(f"Drift cached {ticker} ({quarter}) — {len(drift)} chars")
    else:
        set_drift_cached(db, ticker, DRIFT_EMPTY_SENTINEL, quarter)
        logger.info(
            f"Drift empty-cached {ticker} ({quarter}) — no transcripts; "
            "will not retry until next quarter"
        )

    return drift


def _make_qa_split(transcript: str) -> str:
    """
    Split the most recent transcript into prepared remarks + Q&A sections.
    Returns a formatted string for the prompt, or '' if no Q&A found.
    Zero AV calls — derived from the already-cached transcript.
    """
    if not transcript:
        return ""
    from app.data.phase_d import split_transcript_qa
    try:
        prepared, qa = split_transcript_qa(transcript)
        if not qa:
            return ""
        return (
            f"=== PREPARED REMARKS (most recent quarter) ===\n{prepared[:12_000]}\n\n"
            f"=== Q&A SESSION (most recent quarter) ===\n{qa[:8_000]}"
        )
    except Exception:
        return ""


def get_or_fetch(
    db: Session,
    ticker: str,
    av_client,
    edgar_client=None,
    cache_bust: bool = False,
) -> dict:
    """
    Main entry point. Returns full enrichment context for a ticker.

    Monthly cache (YYYY-MM):  transcript, overview, balance sheet, cash flow,
                              insider, institutional, comment letters, QA split
    Quarterly cache (YYYY-QN): language drift (8-quarter history, expensive)
    Realtime (every run):     news, short interest, concentration instruction

    Language drift is fetched via get_or_fetch_drift() which maintains its own
    quarterly cache — it is always populated regardless of monthly cache state.
    QA split is derived from the cached transcript at zero extra API cost.
    """
    import time

    month   = _cache_month()
    cached  = None if cache_bust else get_cached(db, ticker, month)

    if cached:
        # Monthly cache hit — fetch realtime signals + quarterly drift
        from app.data.phase_d import get_short_interest, get_concentration_instruction
        news           = av_client.get_news_sentiment(ticker)
        time.sleep(1.0)
        short_interest  = get_short_interest(ticker)
        concentration   = get_concentration_instruction()
        language_drift  = get_or_fetch_drift(db, ticker, av_client)
        # QA split from cached transcript — zero extra calls
        transcript_qa_split = _make_qa_split(cached.get("transcript", ""))
        logger.info(
            f"Cache HIT {ticker} ({month}) — fetched realtime signals only"
            f" | drift={'yes' if language_drift else 'no'}"
            f" | qa_split={'yes' if transcript_qa_split else 'no'}"
        )
        return {
            **cached,
            "language_drift":            language_drift,
            "transcript_qa_split":       transcript_qa_split,
            "news":                      news,
            "short_interest":            short_interest,
            "concentration_instruction": concentration,
        }

    # Monthly cache miss — fetch all slow signals
    logger.info(f"Cache MISS {ticker} ({month}) — fetching all signals")
    full_ctx = av_client.get_enriched_llm_context(ticker, edgar_client=edgar_client)

    # QA split from freshly-fetched transcript
    full_ctx["transcript_qa_split"] = _make_qa_split(full_ctx.get("transcript", ""))

    # Language drift via quarterly cache (may be a cache hit even on monthly miss)
    full_ctx["language_drift"] = get_or_fetch_drift(db, ticker, av_client)

    # Persist the slow monthly signals (drift stored separately under quarterly key)
    set_cached(db, ticker, full_ctx, month)

    return full_ctx


def bust_ticker(db: Session, ticker: str, month: Optional[str] = None) -> None:
    """
    Invalidate cache for a specific ticker (e.g. after surprise earnings release).
    """
    month = month or _cache_month()
    try:
        db.execute(
            text("DELETE FROM enrichment_cache WHERE ticker = :t AND cache_month = :m"),
            {"t": ticker, "m": month},
        )
        db.commit()
        logger.info(f"Cache busted: {ticker} {month}")
    except Exception as e:
        logger.warning(f"Cache bust failed for {ticker}: {e}")
        db.rollback()


def cache_stats(db: Session) -> dict:
    """Return cache statistics for monitoring."""
    try:
        row = db.execute(text("""
            SELECT
                COUNT(DISTINCT ticker)    AS tickers_cached,
                COUNT(*)                  AS total_entries,
                MAX(fetched_at)           AS last_fetch,
                cache_month
            FROM enrichment_cache
            GROUP BY cache_month
            ORDER BY cache_month DESC
            LIMIT 3
        """)).fetchall()
        return [
            {
                "month":           r[3],
                "tickers_cached":  r[0],
                "total_entries":   r[1],
                "last_fetch":      str(r[2]),
            }
            for r in row
        ]
    except Exception:
        return []


def extract_company_name(overview_text: str, ticker: str) -> str:
    """
    Extract company name from the cached AV overview text.
    Falls back to ticker if not found.

    The overview text contains: "Name:               Apple Inc."
    """
    if not overview_text:
        return ticker
    for line in overview_text.splitlines():
        if line.strip().startswith("Name:"):
            name = line.split("Name:", 1)[-1].strip()
            if name and name != "N/A":
                return name
    return ticker
