"""
Rebalance router.
POST /api/rebalance/propose          → create proposal from opt job
PUT  /api/rebalance/{id}/approve
PUT  /api/rebalance/{id}/modify
PUT  /api/rebalance/{id}/reject
GET  /api/rebalance/{id}/trades
GET  /api/rebalance/history
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas

router = APIRouter()


@router.post("/propose")
def propose_rebalance(data: schemas.RebalanceProposeRequest, db: Session = Depends(get_db)):
    """
    Build a rebalance proposal from a completed optimization job.
    Computes current vs proposed weights and generates trade actions.
    """
    job = db.query(models.OptimizationJob).filter(
        models.OptimizationJob.id == data.optimization_job_id,
        models.OptimizationJob.status.in_(["complete", "complete_with_warnings"]),
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Optimization job not complete")

    proposed_weights: dict = job.result_json or {}

    # Get current portfolio weights
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == data.portfolio_id
    ).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # Compute turnover
    holdings_map = {h.ticker: h for h in portfolio.holdings}
    total_value = sum(h.shares * 100 for h in portfolio.holdings)  # placeholder price
    current_weights = {}
    if total_value > 0:
        for h in portfolio.holdings:
            current_weights[h.ticker] = (h.shares * 100) / total_value

    all_tickers = set(list(current_weights.keys()) + list(proposed_weights.keys()))
    turnover = sum(
        abs(proposed_weights.get(t, 0) - current_weights.get(t, 0))
        for t in all_tickers
    ) / 2

    proposal = models.RebalanceProposal(
        portfolio_id=data.portfolio_id,
        optimization_job_id=data.optimization_job_id,
        proposed_weights_json=proposed_weights,
        estimated_turnover=turnover,
        estimated_cost=turnover * total_value * 0.001,  # ~10bps cost estimate
        status="pending",
    )
    db.add(proposal)

    # Generate trades
    for ticker in all_tickers:
        curr = current_weights.get(ticker, 0)
        prop = proposed_weights.get(ticker, 0)
        delta = prop - curr
        action = "BUY" if delta > 0.005 else "SELL" if delta < -0.005 else "HOLD"
        trade = models.Trade(
            proposal_id=proposal.id,
            ticker=ticker,
            action=action,
            shares=abs(round(delta * total_value / 100, 0)),
            estimated_price=100.0,
            estimated_value=abs(delta * total_value),
        )
        db.add(trade)

    db.commit()
    db.refresh(proposal)
    return proposal


@router.put("/{proposal_id}/approve")
def approve_rebalance(proposal_id: str, db: Session = Depends(get_db)):
    proposal = _get_proposal(proposal_id, db)
    proposal.status = "approved"
    decision = models.RebalanceDecision(proposal_id=proposal_id, decision="approved")
    db.add(decision)
    db.commit()
    return {"status": "approved"}


@router.put("/{proposal_id}/modify")
def modify_rebalance(
    proposal_id: str,
    data: schemas.RebalanceModifyRequest,
    db: Session = Depends(get_db),
):
    proposal = _get_proposal(proposal_id, db)
    proposal.status = "approved"
    decision = models.RebalanceDecision(
        proposal_id=proposal_id,
        decision="modified",
        modified_weights_json=data.weights,
    )
    db.add(decision)
    db.commit()
    return {"status": "modified"}


@router.put("/{proposal_id}/reject")
def reject_rebalance(
    proposal_id: str,
    data: schemas.RejectRequest,
    db: Session = Depends(get_db),
):
    proposal = _get_proposal(proposal_id, db)
    proposal.status = "rejected"
    decision = models.RebalanceDecision(
        proposal_id=proposal_id,
        decision="rejected",
        reason=data.reason,
    )
    db.add(decision)
    db.commit()
    return {"status": "rejected"}


@router.get("/{proposal_id}/trades", response_model=list[schemas.TradeOut])
def get_trades(proposal_id: str, db: Session = Depends(get_db)):
    trades = db.query(models.Trade).filter(models.Trade.proposal_id == proposal_id).all()
    return trades


@router.get("/history")
def get_rebalance_history(portfolio_id: str, db: Session = Depends(get_db)):
    proposals = (
        db.query(models.RebalanceProposal)
        .filter(models.RebalanceProposal.portfolio_id == portfolio_id)
        .order_by(models.RebalanceProposal.created_at.desc())
        .limit(20)
        .all()
    )
    return proposals


def _get_proposal(proposal_id: str, db: Session) -> models.RebalanceProposal:
    p = db.query(models.RebalanceProposal).filter(
        models.RebalanceProposal.id == proposal_id
    ).first()
    if not p:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return p
