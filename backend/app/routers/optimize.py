"""
optimize.py — POST /api/optimize/deep-rl | mvo | hrp  +  GET /api/optimize/{job_id}
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app import models, schemas
from app.workers.tasks import run_optimization_job

router = APIRouter()


def _dispatch_optimize(db: Session, data: schemas.OptimizeRequest, opt_type: str) -> dict:
    job = models.OptimizationJob(
        portfolio_id=data.portfolio_id,
        run_id=data.run_id,
        optimizer_type=opt_type,
        status=models.RunStatus.pending,
        settings_json=data.settings,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    task = run_optimization_job.delay(str(job.id), data.portfolio_id, data.run_id, opt_type, data.settings)
    return {"jobId": str(job.id), "taskId": task.id}


@router.post("/deep-rl",  response_model=schemas.OptimizeOut)
def optimize_deep_rl(data: schemas.OptimizeRequest, db: Session = Depends(get_db)):
    return _dispatch_optimize(db, data, "deep_rl")


@router.post("/mvo", response_model=schemas.OptimizeOut)
def optimize_mvo(data: schemas.OptimizeRequest, db: Session = Depends(get_db)):
    return _dispatch_optimize(db, data, "mvo")


@router.post("/hrp", response_model=schemas.OptimizeOut)
def optimize_hrp(data: schemas.OptimizeRequest, db: Session = Depends(get_db)):
    return _dispatch_optimize(db, data, "hrp")


@router.get("/{job_id}", response_model=schemas.OptimizationJobOut)
def get_optimization_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(models.OptimizationJob).filter(models.OptimizationJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Optimization job not found")
    return job
