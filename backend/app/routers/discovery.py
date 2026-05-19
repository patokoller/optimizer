"""
app/routers/discovery.py
Discovery (NASDAQ-100 universe) scoring endpoints.
"""
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app import models
from app.workers.tasks import run_discovery_job

router = APIRouter(prefix="/api/discovery", tags=["discovery"])


@router.post("/run")
def start_discovery_run(db: Session = Depends(get_db)):
    """Trigger a full NASDAQ-100 discovery scoring run."""
    # Check for already-running discovery
    running = db.query(models.DiscoveryRun).filter(
        models.DiscoveryRun.status.in_([
            models.RunStatus.pending,
            models.RunStatus.running,
        ])
    ).first()
    if running:
        return {"run_id": running.id, "status": running.status, "message": "Discovery run already in progress"}

    run = models.DiscoveryRun(
        id       = str(uuid.uuid4()),
        status   = models.RunStatus.pending,
        run_date = datetime.utcnow(),
    )
    db.add(run)
    db.commit()

    run_discovery_job.delay(run.id)
    return {"run_id": run.id, "status": "pending"}


@router.get("/latest")
def get_latest_discovery(db: Session = Depends(get_db)):
    """Return the latest completed discovery run with all scores."""
    run = (
        db.query(models.DiscoveryRun)
        .filter(models.DiscoveryRun.status.in_([
            models.RunStatus.complete,
            models.RunStatus.complete_with_warnings,
        ]))
        .order_by(models.DiscoveryRun.run_date.desc())
        .first()
    )
    if not run:
        raise HTTPException(status_code=404, detail="No completed discovery run found")

    scores = (
        db.query(models.DiscoveryScore)
        .filter(models.DiscoveryScore.discovery_run_id == run.id)
        .order_by(models.DiscoveryScore.combined_score.desc().nullslast())
        .all()
    )

    return {
        "run": _run_out(run),
        "scores": [_score_out(s) for s in scores],
    }


@router.get("/status/{run_id}")
def get_discovery_status(run_id: str, db: Session = Depends(get_db)):
    """Poll status of a discovery run."""
    run = db.query(models.DiscoveryRun).filter(models.DiscoveryRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status in [models.RunStatus.complete, models.RunStatus.complete_with_warnings]:
        scores = (
            db.query(models.DiscoveryScore)
            .filter(models.DiscoveryScore.discovery_run_id == run_id)
            .order_by(models.DiscoveryScore.combined_score.desc().nullslast())
            .all()
        )
        return {"run": _run_out(run), "scores": [_score_out(s) for s in scores]}

    return {"run": _run_out(run), "scores": []}


@router.get("/runs")
def list_discovery_runs(limit: int = 10, db: Session = Depends(get_db)):
    """List recent discovery runs."""
    runs = (
        db.query(models.DiscoveryRun)
        .order_by(models.DiscoveryRun.run_date.desc())
        .limit(limit)
        .all()
    )
    return [_run_out(r) for r in runs]


# ── Serialisers ────────────────────────────────────────────────────────────

def _run_out(run: models.DiscoveryRun) -> dict:
    return {
        "id":               run.id,
        "status":           run.status,
        "run_date":         run.run_date.isoformat() if run.run_date else None,
        "universe":         run.universe,
        "universe_size":    run.universe_size,
        "scored_count":     run.scored_count,
        "regime_label":     run.regime_label,
        "regime_confidence": run.regime_confidence,
        "error_log":        run.error_log,
    }


def _score_out(s: models.DiscoveryScore) -> dict:
    return {
        "id":               s.id,
        "discovery_run_id": s.discovery_run_id,
        "ticker":           s.ticker,
        "sector":           s.sector,
        "technical_score":  s.technical_score,
        "fundamental_score": s.fundamental_score,
        "entropy_score":    s.entropy_score,
        "combined_score":   s.combined_score,
        "llm_score":        s.llm_score,
        "llm_provider":     s.llm_provider,
        "llm_reasoning_json": s.llm_reasoning_json,
        "confidence_score": s.confidence_score,
        "overall_dispersion": s.overall_dispersion,
        "prev_combined_score": s.prev_combined_score,
        "score_delta":      s.score_delta,
        "rank":             s.rank,
        "prev_rank":        s.prev_rank,
        "rank_delta":       s.rank_delta,
        "technical_feature_importance":  s.technical_feature_importance,
        "fundamental_feature_importance": s.fundamental_feature_importance,
        "realised_vol_21d": s.realised_vol_21d,
        "beta_vs_qqq":      s.beta_vs_qqq,
        "sharpe_1y":        s.sharpe_1y,
    }
