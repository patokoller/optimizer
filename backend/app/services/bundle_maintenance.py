"""
Bundle maintenance — keeps a trained ModelBundle available so the user never has
to think about "discovery" to score a portfolio or a single stock.

The scoring models (the "bundle") are universe-level: trained once over the
NASDAQ-100, then reused to score ANY US-listed ticker by percentiling it into
that reference distribution. So a bundle is a one-time (periodically refreshed)
system prerequisite, not something tied to a particular portfolio. This module
makes that prerequisite self-healing:

  * `bundle_status(db)`  — is a bundle present, how old, is a refresh running.
  * `ensure_bundle_fresh(db)` — if missing or stale and nothing is already
        training, kick off a training run (the existing discovery job) in the
        background. Safe to call on every report/score request; it never blocks
        and never double-enqueues.

Training is the ~20-minute cost of a discovery run, so this is always async: the
caller proceeds with whatever it can do now (risk analytics, macro, allocation)
and the bundle becomes available for the next request.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# A bundle older than this is considered stale and eligible for a background
# refresh. Training is expensive, so this is deliberately generous.
BUNDLE_MAX_AGE_DAYS = 7


def bundle_status(db, max_age_days: int = BUNDLE_MAX_AGE_DAYS) -> dict:
    """Lightweight freshness read — does NOT deserialize the models."""
    from app import models
    row = (
        db.query(models.ModelBundle.created_at)
        .order_by(models.ModelBundle.created_at.desc())
        .first()
    )
    refresh = _refresh_in_progress(db)
    if row is None:
        return {
            "exists": False, "age_days": None, "fresh": False,
            "refresh_in_progress": refresh is not None,
            "refresh_run_id": refresh.id if refresh else None,
        }
    age = (datetime.utcnow() - row[0]).total_seconds() / 86400.0
    return {
        "exists": True,
        "age_days": round(age, 1),
        "fresh": age <= max_age_days,
        "refresh_in_progress": refresh is not None,
        "refresh_run_id": refresh.id if refresh else None,
    }


def _refresh_in_progress(db):
    """A discovery run that is pending or running counts as a bundle refresh
    already underway (discovery trains and persists the bundle)."""
    from app import models
    return (
        db.query(models.DiscoveryRun)
        .filter(models.DiscoveryRun.status.in_([
            models.RunStatus.pending, models.RunStatus.running,
        ]))
        .order_by(models.DiscoveryRun.created_at.desc())
        .first()
    )


def _start_training(db) -> str:
    """Create a discovery run and enqueue the training job. Returns the run id."""
    from app import models
    run = models.DiscoveryRun(
        id=str(uuid.uuid4()),
        status=models.RunStatus.pending,
        run_date=datetime.utcnow(),
    )
    db.add(run)
    db.commit()
    from app.workers.tasks import run_discovery_job
    run_discovery_job.delay(run.id)
    return run.id


def ensure_bundle_fresh(db, max_age_days: int = BUNDLE_MAX_AGE_DAYS) -> dict:
    """Ensure a reasonably fresh bundle exists, training one in the background if
    not. Returns the status (with `refresh_started` set when a new run was
    enqueued). Never blocks; never enqueues a second run if one is already
    pending/running.
    """
    status = bundle_status(db, max_age_days)
    status["refresh_started"] = False

    if status["fresh"] or status["refresh_in_progress"]:
        return status  # nothing to do — fresh, or already training

    # Missing or stale, and nothing in flight → start a background training run.
    try:
        run_id = _start_training(db)
        reason = "missing" if not status["exists"] else f"stale ({status['age_days']}d)"
        logger.info(f"Bundle {reason} — kicked off background training run {run_id}")
        status.update(refresh_started=True, refresh_in_progress=True, refresh_run_id=run_id)
    except Exception as e:
        logger.warning(f"Could not start background bundle refresh: {e}")
        db.rollback()
    return status
