"""
app/routers/report.py
Portfolio-analysis report (Feature B).

  POST /api/report/run            — start a report job for a portfolio
  GET  /api/report/{id}           — status + JSON summary (for on-screen preview)
  GET  /api/report/{id}/download  — the rendered PDF
  GET  /api/report/portfolio/{pid}/latest — most recent report for a portfolio
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app import models
from app.workers.tasks import run_portfolio_report_job

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/report", tags=["report"])


class ReportRequest(BaseModel):
    portfolio_id: str
    optimizer: str = "MVO"  # MVO | HRP


@router.post("/run")
def start_report(req: ReportRequest, db: Session = Depends(get_db)):
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == req.portfolio_id
    ).first()
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if not portfolio.holdings:
        raise HTTPException(status_code=400, detail="Portfolio has no holdings to analyze")

    opt = (req.optimizer or "MVO").upper()
    if opt not in ("MVO", "HRP"):
        raise HTTPException(status_code=400, detail="optimizer must be MVO or HRP")

    report = models.PortfolioReport(
        id=str(uuid.uuid4()),
        portfolio_id=req.portfolio_id,
        optimizer=opt,
        status=models.RunStatus.pending,
    )
    db.add(report)
    db.commit()
    run_portfolio_report_job.delay(report.id)
    return {"report_id": report.id, "status": report.status}


@router.get("/{report_id}")
def get_report(report_id: str, db: Session = Depends(get_db)):
    report = db.query(models.PortfolioReport).filter(
        models.PortfolioReport.id == report_id
    ).first()
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return {
        "report_id": report.id,
        "portfolio_id": report.portfolio_id,
        "status": report.status,
        "optimizer": report.optimizer,
        "summary": report.summary_json,
        "pdf_size": report.pdf_size,
        "has_pdf": report.pdf_bytes is not None,
        "error": report.error_log,
        "created_at": report.created_at.isoformat() if report.created_at else None,
        "completed_at": report.completed_at.isoformat() if report.completed_at else None,
    }


@router.get("/{report_id}/download")
def download_report(report_id: str, db: Session = Depends(get_db)):
    report = db.query(models.PortfolioReport).filter(
        models.PortfolioReport.id == report_id
    ).first()
    if report is None or report.pdf_bytes is None:
        raise HTTPException(status_code=404, detail="Report PDF not available")
    filename = f"portfolio_report_{report_id[:8]}.pdf"
    return Response(
        content=bytes(report.pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/portfolio/{portfolio_id}/latest")
def latest_for_portfolio(portfolio_id: str, db: Session = Depends(get_db)):
    report = (
        db.query(models.PortfolioReport)
        .filter(models.PortfolioReport.portfolio_id == portfolio_id)
        .order_by(models.PortfolioReport.created_at.desc())
        .first()
    )
    if report is None:
        raise HTTPException(status_code=404, detail="No report yet for this portfolio")
    return {"report_id": report.id, "status": report.status,
            "created_at": report.created_at.isoformat() if report.created_at else None}
