"""
Rebalance router.
GET  /api/rebalance/live-proposal      → build proposal from latest scores (no optimizer needed)
POST /api/rebalance/propose            → create proposal from opt job
PUT  /api/rebalance/{id}/approve
PUT  /api/rebalance/{id}/modify
PUT  /api/rebalance/{id}/reject
GET  /api/rebalance/{id}/trades
GET  /api/rebalance/history
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas

router = APIRouter()


@router.get("/live-proposal")
def get_live_proposal(
    portfolio_id: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    Build a rebalance proposal directly from the latest score run.

    Uses the paper model approach: select the top-10 highest combined_score tickers,
    assign equal weights (10% each), compare against current holdings.

    No optimizer required — this is the paper model selection, not Deep RL.
    The Deep RL optimizer is a Phase 3 addition.
    """
    # ── Get portfolio + latest score run ──────────────────────────────
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id
    ).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # ── Use Discovery run (full NASDAQ-100 universe) ───────────────────
    # Discovery runs score the full universe; ScoreRuns are portfolio-specific.
    # The rebalance proposal should select from the full universe, not just
    # tickers already in the portfolio.
    latest_run = (
        db.query(models.DiscoveryRun)
        .filter(models.DiscoveryRun.status.in_(["complete", "complete_with_warnings"]))
        .order_by(models.DiscoveryRun.run_date.desc())
        .first()
    )
    if not latest_run:
        return {"proposal": None, "reason": "no_scores", "trades": []}

    # ── Scores sorted by combined_score ───────────────────────────────
    scores = (
        db.query(models.DiscoveryScore)
        .filter(models.DiscoveryScore.discovery_run_id == latest_run.id)
        .order_by(models.DiscoveryScore.combined_score.desc())
        .all()
    )
    if not scores:
        return {"proposal": None, "reason": "no_scores", "trades": []}

    # ── Current weights from holdings ─────────────────────────────────
    # Use cost_basis × shares as a rough market value proxy
    # (Alpaca live prices not fetched here to keep this endpoint fast)
    holdings_map = {h.ticker: h for h in portfolio.holdings}
    total_cost = sum(
        (h.shares * (h.cost_basis or 0)) for h in portfolio.holdings
    ) or 1.0  # avoid div-by-zero

    current_weights = {
        h.ticker: (h.shares * (h.cost_basis or 0)) / total_cost
        for h in portfolio.holdings
    }

    # ── Paper model: top-10 equal-weight ──────────────────────────────
    TOP_N = 10
    top10 = [s for s in scores if s.combined_score is not None][:TOP_N]
    proposed_weight = 1.0 / TOP_N  # equal weight

    # ── Build trade list ───────────────────────────────────────────────
    proposed_tickers = {s.ticker for s in top10}
    all_tickers = set(list(current_weights.keys()) + list(proposed_tickers))

    trades = []
    for ticker in sorted(all_tickers):
        curr  = current_weights.get(ticker, 0.0)
        prop  = proposed_weight if ticker in proposed_tickers else 0.0
        delta = prop - curr

        if abs(delta) < 0.001:
            action = "HOLD"
        elif delta > 0:
            action = "BUY"
        else:
            action = "SELL"

        score_row = next((s for s in scores if s.ticker == ticker), None)

        trades.append({
            "ticker":           ticker,
            "action":           action,
            "current_weight":   round(curr, 4),
            "proposed_weight":  round(prop, 4),
            "delta_weight":     round(delta, 4),
            "combined_score":   round(score_row.combined_score, 3) if score_row and score_row.combined_score else None,
            "technical_score":  round(score_row.technical_score, 3) if score_row and score_row.technical_score else None,
            "fundamental_score":round(score_row.fundamental_score, 3) if score_row and score_row.fundamental_score else None,
            "entropy_score":    round(score_row.entropy_score, 3) if score_row and score_row.entropy_score else None,
            "confidence_score": round(score_row.confidence_score, 3) if score_row and score_row.confidence_score else None,
            "llm_reasoning":    score_row.llm_reasoning_json if score_row else None,
            "score_delta":      round(score_row.score_delta, 3) if score_row and score_row.score_delta else None,
            "beta_vs_qqq":      round(score_row.beta_vs_qqq, 3) if score_row and score_row.beta_vs_qqq else None,
            "vol_21d":          round(score_row.realised_vol_21d, 3) if score_row and score_row.realised_vol_21d else None,
            "in_current":       ticker in holdings_map,
            "in_proposed":      ticker in proposed_tickers,
        })

    # Sort: BUY first, then SELL, then HOLD; within each group by |delta| desc
    order = {"BUY": 0, "SELL": 1, "HOLD": 2}
    trades.sort(key=lambda t: (order[t["action"]], -abs(t["delta_weight"])))

    # ── Summary metrics ────────────────────────────────────────────────
    turnover = sum(abs(t["delta_weight"]) for t in trades if t["action"] != "HOLD") / 2
    buys  = [t for t in trades if t["action"] == "BUY"]
    sells = [t for t in trades if t["action"] == "SELL"]

    return {
        "proposal": {
            "run_id":        latest_run.id,
            "run_date":      latest_run.run_date.isoformat(),
            "portfolio_id":  portfolio_id,
            "method":        "paper_model_top10_equal_weight",
            "top_n":         TOP_N,
            "turnover":      round(turnover, 4),
            "n_buys":        len(buys),
            "n_sells":       len(sells),
            "n_holds":       len([t for t in trades if t["action"] == "HOLD"]),
        },
        "trades": trades,
    }


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
