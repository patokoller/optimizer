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
    return datetime.utcnow().strftime("%Y-%m")


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
                VALUES (:t, :m, :ctx::jsonb, now())
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


def get_or_fetch(
    db: Session,
    ticker: str,
    av_client,
    edgar_client=None,
    cache_bust: bool = False,
) -> dict:
    """
    Main entry point. Returns full enrichment context for a ticker.

    1. Check cache for slow signals (transcript, overview, balance sheet, etc.)
    2. Always fetch news fresh (time-sensitive)
    3. If cache miss (or cache_bust=True), fetch all slow signals from AV and cache them
    4. Merge cached slow signals + fresh news and return

    Args:
        db:          SQLAlchemy session
        ticker:      Stock ticker
        av_client:   AlphaVantageClient instance
        cache_bust:  Force re-fetch even if cache is warm (use after earnings release)
    """
    import time

    month = _cache_month()
    cached = None if cache_bust else get_cached(db, ticker, month)

    if cached:
        # Cache hit — fetch realtime signals fresh (news + short interest)
        from app.data.phase_d import get_short_interest, get_concentration_instruction
        news           = av_client.get_news_sentiment(ticker)
        time.sleep(1.0)
        short_interest = get_short_interest(ticker)
        concentration  = get_concentration_instruction()
        logger.info(f"Cache HIT {ticker} ({month}) — fetched realtime signals only")
        return {
            **cached,
            "news":                    news,
            "short_interest":          short_interest,
            "concentration_instruction": concentration,
        }

    # Cache miss — fetch all signals
    logger.info(f"Cache MISS {ticker} ({month}) — fetching all signals")
    full_ctx = av_client.get_enriched_llm_context(ticker, edgar_client=edgar_client)

    # Persist the slow signals
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
