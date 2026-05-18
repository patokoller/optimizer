"""
Portfolio router — upload, retrieve, update constraints.
"""
import csv
import io
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Body
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas

router = APIRouter()


@router.post("/upload", response_model=schemas.PortfolioOut)
async def upload_portfolio(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a portfolio CSV with columns: ticker, shares, cost_basis, currency.
    Creates or replaces holdings for the default demo user.
    """
    content = await file.read()
    text = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))

    # For MVP: use/create a demo user
    user = db.query(models.User).filter(models.User.email == "demo@alphalens.io").first()
    if not user:
        user = models.User(email="demo@alphalens.io", name="Demo User")
        db.add(user)
        db.flush()

    # Create portfolio
    portfolio = models.Portfolio(
        user_id=user.id,
        name=f"Uploaded {datetime.utcnow().strftime('%Y-%m-%d')}",
        universe="NASDAQ-100",
        benchmark="QQQ",
    )
    db.add(portfolio)
    db.flush()

    # Create default constraints
    constraint = models.Constraint(portfolio_id=portfolio.id)
    db.add(constraint)

    # Parse holdings
    required_cols = {"ticker", "shares"}
    holdings = []
    for row in reader:
        cols = {k.strip().lower() for k in row}
        if not required_cols.issubset(cols):
            raise HTTPException(
                status_code=422,
                detail=f"CSV missing required columns. Found: {cols}. Required: {required_cols}",
            )
        ticker = row.get("ticker", "").strip().upper()
        if not ticker:
            continue
        try:
            shares = float(row.get("shares", 0))
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid shares value for {ticker}")

        holding = models.Holding(
            portfolio_id=portfolio.id,
            ticker=ticker,
            shares=shares,
            cost_basis=float(row.get("cost_basis", 0) or 0),
            currency=row.get("currency", "USD").strip().upper(),
        )
        db.add(holding)
        holdings.append(holding)

    db.commit()
    db.refresh(portfolio)
    return portfolio


@router.get("/{portfolio_id}", response_model=schemas.PortfolioOut)
def get_portfolio(portfolio_id: str, db: Session = Depends(get_db)):
    p = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return p


@router.put("/{portfolio_id}/constraints")
def update_constraints(
    portfolio_id: str,
    data: schemas.PortfolioConstraintsUpdate,
    db: Session = Depends(get_db),
):
    p = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    c = db.query(models.Constraint).filter(models.Constraint.portfolio_id == portfolio_id).first()
    if not c:
        c = models.Constraint(portfolio_id=portfolio_id)
        db.add(c)

    c.max_position_pct  = data.max_position_pct
    c.sector_cap_pct    = data.sector_cap_pct
    c.min_cash_pct      = data.min_cash_pct
    c.max_cash_pct      = data.max_cash_pct
    c.excluded_tickers  = data.excluded_tickers
    c.esg_filter        = data.esg_filter
    db.commit()
    return {"status": "updated"}
