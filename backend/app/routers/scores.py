"""
Scores router.
POST /api/scores/run           → dispatches Celery task, returns {jobId, runId}
GET  /api/scores/latest        → latest complete run + scores
GET  /api/scores/{run_id}      → specific run
GET  /api/scores/history       → all runs for portfolio
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.workers.tasks import run_score_job

router = APIRouter()


@router.post("/run", response_model=dict)
def run_scores(data: schemas.ScoreRunRequest, db: Session = Depends(get_db)):
    """
    Kick off async scoring job.
    Returns immediately with jobId + runId for polling.
    """
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == data.portfolio_id
    ).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    run = models.ScoreRun(
        portfolio_id=data.portfolio_id,
        run_date=datetime.utcnow(),
        frequency=data.frequency,
        status=models.RunStatus.pending,
        model_version="1.0.0",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Dispatch Celery task
    task = run_score_job.delay(str(run.id), data.portfolio_id, data.frequency)
    return {"jobId": task.id, "runId": str(run.id)}


@router.get("/latest", response_model=schemas.ScoreRunWithScores)
def get_latest_scores(
    portfolio_id: str = Query(...),
    db: Session = Depends(get_db),
):
    run = (
        db.query(models.ScoreRun)
        .filter(
            models.ScoreRun.portfolio_id == portfolio_id,
            models.ScoreRun.status.in_(["complete", "complete_with_warnings"]),
        )
        .order_by(models.ScoreRun.run_date.desc())
        .first()
    )
    if not run:
        raise HTTPException(status_code=404, detail="No completed score runs found")

    scores = db.query(models.Score).filter(models.Score.run_id == run.id).all()
    return {"run": run, "scores": scores}


@router.get("/history", response_model=list[schemas.ScoreRunOut])
def get_score_history(
    portfolio_id: str = Query(...),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    runs = (
        db.query(models.ScoreRun)
        .filter(models.ScoreRun.portfolio_id == portfolio_id)
        .order_by(models.ScoreRun.run_date.desc())
        .limit(limit)
        .all()
    )
    return runs


@router.get("/{run_id}", response_model=schemas.ScoreRunWithScores)
def get_score_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(models.ScoreRun).filter(models.ScoreRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Score run not found")
    scores = db.query(models.Score).filter(models.Score.run_id == run_id).all()
    return {"run": run, "scores": scores}
