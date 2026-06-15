"""
app/routers/search.py
Single-stock search: score any US-listed ticker on demand by reusing the latest
trained ModelBundle (no full universe run).

  GET  /api/search/resolve/{ticker}  — fast validity + company name (no scoring)
  POST /api/search/score             — full on-demand score (ML percentiles +
                                       synchronous two-stage LLM); ~15-30s
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.data import clients as _clients
from app.services.score_one import score_one

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])

# One shared EdgarClient so company_tickers.json is cached process-wide.
_edgar = _clients.EDGARClient()


class ScoreRequest(BaseModel):
    ticker: str


@router.get("/resolve/{ticker}")
def resolve(ticker: str):
    """Fast identity check for the search box — is this a real US-listed ticker,
    and what's the company name? No scoring, no LLM, no model bundle needed."""
    info = _edgar.resolve_ticker(ticker)
    if not info["valid"]:
        raise HTTPException(
            status_code=404,
            detail=f"'{ticker.upper()}' is not a recognised US-listed ticker.",
        )
    return info


@router.post("/score")
def score(req: ScoreRequest, db: Session = Depends(get_db)):
    """
    Score a single ticker against the latest discovery universe.

    Validates the ticker first (cheap), then runs the full on-demand path. The
    response carries per-strategy ML percentiles, the blended scores, the LLM
    derivation + fact sheet, and explicit data-availability flags so partial
    data is visible rather than silently filled.
    """
    info = _edgar.resolve_ticker(req.ticker)
    if not info["valid"]:
        raise HTTPException(
            status_code=404,
            detail=f"'{req.ticker.upper()}' is not a recognised US-listed ticker.",
        )

    try:
        payload = score_one(db, req.ticker)
    except Exception as e:
        logger.error(f"search.score failed for {req.ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Scoring failed — see server logs.")

    # No bundle yet → a clear, actionable 409 rather than a generic error.
    if payload.get("error") == "no_model_bundle":
        raise HTTPException(
            status_code=409,
            detail="No trained models are available yet. Run a discovery job first.",
        )

    payload.setdefault("company_name", info.get("company_name"))
    payload["is_etf"] = info.get("is_etf", False)
    return payload
