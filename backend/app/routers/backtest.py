"""app/routers/backtest.py"""
import csv
import io
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app import models, schemas

router = APIRouter()

# Locked benchmark facts — source: Table 1, Cohen et al. (2025)
LOCKED_BENCHMARKS = [
    {"strategy": "technical",   "frequency": "monthly",   "ml_weight": 1.00, "llm_weight": 0.00, "sharpe_ratio": 0.6934, "average_return": 0.0750, "volatility": 0.1082, "cumulative_return": 19.7771, "notes": "Best cumulative return overall"},
    {"strategy": "entropy",     "frequency": "monthly",   "ml_weight": 0.70, "llm_weight": 0.30, "sharpe_ratio": 0.4207, "average_return": 0.0523, "volatility": 0.1244, "cumulative_return":  7.0052, "notes": "Balanced blend"},
    {"strategy": "fundamental", "frequency": "monthly",   "ml_weight": 0.15, "llm_weight": 0.85, "sharpe_ratio": 0.5001, "average_return": 0.0432, "volatility": 0.0863, "cumulative_return":  5.7840, "notes": "Lowest volatility monthly"},
    {"strategy": "technical",   "frequency": "quarterly", "ml_weight": 0.45, "llm_weight": 0.55, "sharpe_ratio": 1.2967, "average_return": 0.2499, "volatility": 0.1927, "cumulative_return":  5.7337, "notes": "Highest Sharpe ratio"},
    {"strategy": "entropy",     "frequency": "quarterly", "ml_weight": 0.40, "llm_weight": 0.60, "sharpe_ratio": 0.6048, "average_return": 0.2025, "volatility": 0.3348, "cumulative_return":  5.3436, "notes": "Slight semantic lean"},
    {"strategy": "fundamental", "frequency": "quarterly", "ml_weight": 0.00, "llm_weight": 1.00, "sharpe_ratio": 0.4899, "average_return": 0.1471, "volatility": 0.3002, "cumulative_return":  3.2612, "notes": "Pure semantic"},
]


@router.post("/run")
def run_backtest(data: schemas.BacktestRequest, db: Session = Depends(get_db)):
    """
    Returns locked benchmark facts for the requested strategies.
    NOTE: Paper benchmarks are returned as-is (source-backed, locked values).
    Full time-series data would require supplying the paper's underlying dataset.
    """
    filtered = [
        b for b in LOCKED_BENCHMARKS
        if b["strategy"] in data.strategies
    ]
    return {
        "source": "Table 1, Cohen et al., Entropy 2025, 27, 550",
        "period": f"{data.start_date} to {data.end_date}",
        "universe": "NASDAQ-100",
        "results": filtered,
        "series_available": False,
        "series_note": (
            "Full cumulative return and monthly return series not included in paper attachment. "
            "Supply underlying data to render time-series charts. "
            "Required fields: period_index, strategy_type, rebalance_frequency, cumulative_return"
        ),
    }


@router.get("/{job_id}/results")
def get_backtest_results(job_id: str, db: Session = Depends(get_db)):
    return {
        "job_id": job_id,
        "benchmarks": LOCKED_BENCHMARKS,
        "series_available": False,
    }


# ──────────────────────────────────────────────────────────────
# app/routers/export.py
# ──────────────────────────────────────────────────────────────
"""Export router — CSV, IBKR, Schwab, PDF memo"""
from fastapi import APIRouter as ExportRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.orm import Session
from app.database import get_db
from app import models

export_router = ExportRouter()


def _get_trades(proposal_id: str, db: Session) -> list[models.Trade]:
    trades = db.query(models.Trade).filter(models.Trade.proposal_id == proposal_id).all()
    if not trades:
        raise HTTPException(status_code=404, detail="No trades found for proposal")
    return trades


@export_router.get("/trades/{proposal_id}")
def export_trades(
    proposal_id: str,
    format: str = Query("csv", pattern="^(csv|ibkr|schwab|pdf)$"),
    db: Session = Depends(get_db),
):
    trades = _get_trades(proposal_id, db)

    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["ticker", "action", "shares", "estimated_price", "estimated_value"])
        writer.writeheader()
        for t in trades:
            writer.writerow({
                "ticker": t.ticker,
                "action": t.action,
                "shares": t.shares,
                "estimated_price": t.estimated_price,
                "estimated_value": t.estimated_value,
            })
        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=trades_{proposal_id}.csv"},
        )

    elif format == "ibkr":
        # Interactive Brokers order file format
        output = io.StringIO()
        output.write("Action,Quantity,Symbol,SecType,Exchange,Currency\n")
        for t in trades:
            if t.action in ("BUY", "SELL"):
                output.write(f"{t.action},{int(t.shares or 0)},{t.ticker},STK,SMART,USD\n")
        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=ibkr_orders_{proposal_id}.csv"},
        )

    elif format == "schwab":
        output = io.StringIO()
        output.write("Symbol,Instruction,Quantity,Order Type\n")
        for t in trades:
            if t.action in ("BUY", "SELL"):
                output.write(f"{t.ticker},{t.action.capitalize()},{int(t.shares or 0)},Market\n")
        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=schwab_orders_{proposal_id}.csv"},
        )

    elif format == "pdf":
        # Simplified PDF memo — in production use reportlab or weasyprint
        lines = [
            "AI Portfolio Decision-Support Platform",
            "Rebalance Memo — Advisory Only",
            f"Proposal ID: {proposal_id}",
            "",
            "DISCLAIMER: Backtested results only. Not a representation of live performance.",
            "This tool never executes trades automatically.",
            "",
            "TRADE LIST:",
            f"{'TICKER':<10} {'ACTION':<6} {'SHARES':>8} {'EST. PRICE':>12} {'EST. VALUE':>12}",
            "-" * 55,
        ]
        for t in trades:
            lines.append(
                f"{t.ticker:<10} {t.action:<6} {int(t.shares or 0):>8} "
                f"{(t.estimated_price or 0):>12.2f} {(t.estimated_value or 0):>12.2f}"
            )
        lines.append("")
        lines.append("Source: Cohen, Aiche & Eichel (2025), Entropy 27, 550")
        content = "\n".join(lines).encode()
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename=memo_{proposal_id}.txt"},
        )


@export_router.get("/report/{proposal_id}")
def export_report(proposal_id: str, db: Session = Depends(get_db)):
    trades = _get_trades(proposal_id, db)
    lines = [
        "=== Rebalance Report ===",
        f"Proposal: {proposal_id}",
        "Source: Cohen, Aiche & Eichel (2025), Entropy 27, 550",
        "DISCLAIMER: Advisory only. Backtested results 2020-2025. Not live performance.",
    ]
    return Response(
        content="\n".join(lines).encode(),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=report_{proposal_id}.txt"},
    )
