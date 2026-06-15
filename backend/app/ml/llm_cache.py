"""
LLM score caching.

Caches Claude scores per ticker per month, keyed by a fingerprint of the prompt.
The prompt encodes every semantic input (filings, transcripts, peer standing,
macro, language drift), so:

  - unchanged inputs  → identical fingerprint → cache HIT (no API call)
  - any new filing / transcript / macro shift → different prompt → fingerprint
    changes → cache MISS → fresh score

This gives correct invalidation without explicit event hooks: the cache can never
return a score computed from stale inputs, because changed inputs change the key.

Two entry points wrap the existing scorers:
  - score_batch_cached: for full runs (splits hits from misses, batches only the
    misses, stores the fresh results)
  - score_sync_cached:  for on-demand single-ticker scoring
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from app import models

logger = logging.getLogger(__name__)

_PRUNE_KEEP_MONTHS = 4  # retain ~4 months of cache rows


def prompt_fingerprint(prompt: str) -> str:
    """Stable sha256 of the prompt text — the cache key's variable part."""
    return hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()


def get_cached(db, ticker: str, period: str, prompt_hash: str) -> Optional[dict]:
    row = (
        db.query(models.LLMScoreCache)
        .filter(
            models.LLMScoreCache.ticker == ticker,
            models.LLMScoreCache.period == period,
            models.LLMScoreCache.prompt_hash == prompt_hash,
        )
        .first()
    )
    return row.result_json if row else None


def put_cached(db, ticker: str, period: str, prompt_hash: str, result: dict,
               two_stage: bool = False) -> None:
    """
    Store a fresh result. Any prior rows for (ticker, period) with a DIFFERENT
    fingerprint are stale (inputs changed) and are removed, keeping the table to
    roughly one row per ticker per month. Caller is responsible for commit.
    """
    if not isinstance(result, dict):
        return
    db.query(models.LLMScoreCache).filter(
        models.LLMScoreCache.ticker == ticker,
        models.LLMScoreCache.period == period,
        models.LLMScoreCache.prompt_hash != prompt_hash,
    ).delete(synchronize_session=False)

    existing = (
        db.query(models.LLMScoreCache)
        .filter(
            models.LLMScoreCache.ticker == ticker,
            models.LLMScoreCache.period == period,
            models.LLMScoreCache.prompt_hash == prompt_hash,
        )
        .first()
    )
    if existing:
        existing.result_json = result
        existing.two_stage = bool(two_stage)
    else:
        db.add(models.LLMScoreCache(
            ticker=ticker, period=period, prompt_hash=prompt_hash,
            result_json=result, two_stage=bool(two_stage),
        ))


def score_batch_cached(db, scorer, prompts: dict, period: str) -> dict:
    """
    Cache-aware wrapper over LLMScorer.score_batch. Returns {ticker: result}.

    Splits prompts into cache hits and misses; only the misses go to the batch
    API; fresh results are written back to the cache. On any cache error, falls
    back to scoring everything (cache is an optimisation, never a dependency).
    """
    if not prompts:
        return {}

    try:
        fingerprints = {t: prompt_fingerprint(p) for t, p in prompts.items()}
        cached: dict = {}
        to_score: dict = {}
        for t, p in prompts.items():
            hit = get_cached(db, t, period, fingerprints[t])
            if hit is not None:
                cached[t] = hit
            else:
                to_score[t] = p
        logger.info(
            f"LLM cache ({period}): {len(cached)} hit / {len(to_score)} miss "
            f"of {len(prompts)}"
        )
    except Exception as e:
        logger.warning(f"LLM cache read failed ({e}) — scoring all uncached")
        return scorer.score_batch(prompts)

    fresh = scorer.score_batch(to_score) if to_score else {}

    try:
        for t, res in fresh.items():
            put_cached(db, t, period, fingerprints[t], res,
                       two_stage=bool(res.get("two_stage")) if isinstance(res, dict) else False)
        if fresh:
            db.commit()
            _prune(db)
    except Exception as e:
        logger.warning(f"LLM cache write failed ({e}) — results still returned")
        try:
            db.rollback()
        except Exception:
            pass

    return {**cached, **fresh}


def score_sync_cached(db, scorer, ticker: str, prompt: str, period: str) -> Optional[dict]:
    """Cache-aware wrapper over LLMScorer.score_two_stage_sync (on-demand path)."""
    fp = prompt_fingerprint(prompt)
    try:
        hit = get_cached(db, ticker, period, fp)
        if hit is not None:
            logger.info(f"LLM cache HIT (sync) {ticker} {period}")
            return hit
    except Exception as e:
        logger.warning(f"LLM cache read failed for {ticker} ({e})")

    res = scorer.score_two_stage_sync(ticker, prompt)
    if isinstance(res, dict):
        try:
            put_cached(db, ticker, period, fp, res, two_stage=bool(res.get("two_stage")))
            db.commit()
        except Exception as e:
            logger.warning(f"LLM cache write failed for {ticker} ({e})")
            try:
                db.rollback()
            except Exception:
                pass
    return res


def _prune(db) -> None:
    """Delete cache rows older than the most recent _PRUNE_KEEP_MONTHS periods."""
    try:
        periods = [
            r[0] for r in db.query(models.LLMScoreCache.period)
            .distinct().order_by(models.LLMScoreCache.period.desc()).all()
        ]
        if len(periods) <= _PRUNE_KEEP_MONTHS:
            return
        stale_periods = periods[_PRUNE_KEEP_MONTHS:]
        deleted = db.query(models.LLMScoreCache).filter(
            models.LLMScoreCache.period.in_(stale_periods)
        ).delete(synchronize_session=False)
        db.commit()
        if deleted:
            logger.info(f"LLM cache: pruned {deleted} row(s) from {stale_periods}")
    except Exception as e:
        logger.warning(f"LLM cache prune failed ({e})")
        try:
            db.rollback()
        except Exception:
            pass
