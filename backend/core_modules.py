"""
AI Portfolio Decision-Support Platform — FastAPI Backend
Cohen, Aiche & Eichel (2025), Entropy 27, 550
"""

# ────────────────────────────────────────────────────────────
# app/main.py
# ────────────────────────────────────────────────────────────
MAIN = '''
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.database import engine, Base
from app.routers import portfolio, scores, optimize, rebalance, backtest, export

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AI Portfolio Decision-Support API",
    description="Based on Cohen, Aiche & Eichel (2025), Entropy 27, 550",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-app.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(req: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(scores.router,    prefix="/api/scores",    tags=["scores"])
app.include_router(optimize.router,  prefix="/api/optimize",  tags=["optimize"])
app.include_router(rebalance.router, prefix="/api/rebalance", tags=["rebalance"])
app.include_router(backtest.router,  prefix="/api/backtest",  tags=["backtest"])
app.include_router(export.router,    prefix="/api/export",    tags=["export"])


@app.get("/health")
async def health():
    return {"status": "ok"}
'''

# ────────────────────────────────────────────────────────────
# app/database.py
# ────────────────────────────────────────────────────────────
DATABASE = '''
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ["DATABASE_URL"]
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
'''

# ────────────────────────────────────────────────────────────
# app/models.py — SQLAlchemy ORM
# ────────────────────────────────────────────────────────────
MODELS = '''
import uuid
import enum
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime,
    ForeignKey, Enum, Text, ARRAY, UniqueConstraint, CheckConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from app.database import Base


def _uuid():
    return str(uuid.uuid4())


class StrategyType(str, enum.Enum):
    fundamental = "fundamental"
    technical   = "technical"
    entropy     = "entropy"


class RebalanceFreq(str, enum.Enum):
    monthly   = "monthly"
    quarterly = "quarterly"


class OptimizerType(str, enum.Enum):
    deep_rl = "deep_rl"
    mvo     = "mvo"
    hrp     = "hrp"


class DecisionType(str, enum.Enum):
    approved = "approved"
    modified = "modified"
    rejected = "rejected"


class RunStatus(str, enum.Enum):
    pending   = "pending"
    running   = "running"
    complete  = "complete"
    failed    = "failed"
    complete_with_warnings = "complete_with_warnings"


class LLMProvider(str, enum.Enum):
    claude = "claude"
    none   = "none"


class User(Base):
    __tablename__ = "users"
    id         = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    email      = Column(String, unique=True, nullable=False, index=True)
    name       = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")


class Portfolio(Base):
    __tablename__ = "portfolios"
    id         = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    user_id    = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    name       = Column(String, nullable=False)
    universe   = Column(String, nullable=False, default="NASDAQ-100")
    benchmark  = Column(String, nullable=False, default="QQQ")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    user        = relationship("User", back_populates="portfolios")
    holdings    = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")
    constraints = relationship("Constraint", back_populates="portfolio", uselist=False, cascade="all, delete-orphan")
    score_runs  = relationship("ScoreRun", back_populates="portfolio")


class Holding(Base):
    __tablename__ = "holdings"
    id           = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    portfolio_id = Column(UUID(as_uuid=False), ForeignKey("portfolios.id"), nullable=False)
    ticker       = Column(String(10), nullable=False, index=True)
    shares       = Column(Float, nullable=False)
    cost_basis   = Column(Float)
    currency     = Column(String(3), nullable=False, default="USD")
    uploaded_at  = Column(DateTime, default=datetime.utcnow, nullable=False)
    portfolio    = relationship("Portfolio", back_populates="holdings")


class Constraint(Base):
    __tablename__ = "constraints"
    id               = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    portfolio_id     = Column(UUID(as_uuid=False), ForeignKey("portfolios.id"), nullable=False, unique=True)
    max_position_pct = Column(Float, nullable=False, default=0.25)
    sector_cap_pct   = Column(Float, nullable=False, default=0.40)
    min_cash_pct     = Column(Float, nullable=False, default=0.02)
    max_cash_pct     = Column(Float, nullable=False, default=0.10)
    excluded_tickers = Column(ARRAY(String), nullable=False, default=list)
    esg_filter       = Column(Boolean, nullable=False, default=False)
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    portfolio        = relationship("Portfolio", back_populates="constraints")


class ScoreRun(Base):
    __tablename__ = "score_runs"
    id            = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    portfolio_id  = Column(UUID(as_uuid=False), ForeignKey("portfolios.id"), nullable=False)
    run_date      = Column(DateTime, nullable=False)
    frequency     = Column(Enum(RebalanceFreq), nullable=False, default=RebalanceFreq.monthly)
    status        = Column(Enum(RunStatus), nullable=False, default=RunStatus.pending)
    model_version = Column(String)
    error_log     = Column(Text)
    created_at    = Column(DateTime, default=datetime.utcnow, nullable=False)
    portfolio     = relationship("Portfolio", back_populates="score_runs")
    scores        = relationship("Score", back_populates="run", cascade="all, delete-orphan")


class Score(Base):
    __tablename__ = "scores"
    id                      = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    run_id                  = Column(UUID(as_uuid=False), ForeignKey("score_runs.id"), nullable=False)
    ticker                  = Column(String(10), nullable=False, index=True)

    # Raw per-strategy ML scores
    technical_ml_score      = Column(Float)
    fundamental_ml_score    = Column(Float)
    entropy_ml_score        = Column(Float)

    # LLM semantic score (shared)
    llm_score               = Column(Float)
    llm_provider            = Column(Enum(LLMProvider), nullable=False, default=LLMProvider.claude)
    llm_reasoning_json      = Column(JSONB)  # {score, key_positives, key_risks, confidence}

    # Combined scores: w*ML + (1-w)*LLM  — weights locked from Table 1
    technical_score         = Column(Float)
    fundamental_score       = Column(Float)
    entropy_score           = Column(Float)
    combined_score          = Column(Float)

    # Locked ML weights (Table 1, Cohen et al. 2025)
    w_technical             = Column(Float, nullable=False, default=1.00)
    w_fundamental           = Column(Float, nullable=False, default=0.15)
    w_entropy               = Column(Float, nullable=False, default=0.70)

    forward_return_forecast = Column(Float)
    created_at              = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (UniqueConstraint("run_id", "ticker"),)
    run = relationship("ScoreRun", back_populates="scores")


class OptimizationJob(Base):
    __tablename__ = "optimization_jobs"
    id             = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    portfolio_id   = Column(UUID(as_uuid=False), ForeignKey("portfolios.id"), nullable=False)
    run_id         = Column(UUID(as_uuid=False), ForeignKey("score_runs.id"))
    optimizer_type = Column(Enum(OptimizerType), nullable=False)
    status         = Column(Enum(RunStatus), nullable=False, default=RunStatus.pending)
    settings_json  = Column(JSONB)
    result_json    = Column(JSONB)  # {ticker: weight}
    created_at     = Column(DateTime, default=datetime.utcnow, nullable=False)


class RebalanceProposal(Base):
    __tablename__ = "rebalance_proposals"
    id                    = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    portfolio_id          = Column(UUID(as_uuid=False), ForeignKey("portfolios.id"), nullable=False)
    optimization_job_id   = Column(UUID(as_uuid=False), ForeignKey("optimization_jobs.id"))
    status                = Column(String, nullable=False, default="pending")
    proposed_weights_json = Column(JSONB, nullable=False)
    rationale_json        = Column(JSONB)
    estimated_turnover    = Column(Float)
    estimated_cost        = Column(Float)
    created_at            = Column(DateTime, default=datetime.utcnow, nullable=False)
    decisions             = relationship("RebalanceDecision", back_populates="proposal", cascade="all, delete-orphan")
    trades                = relationship("Trade", back_populates="proposal", cascade="all, delete-orphan")


class RebalanceDecision(Base):
    __tablename__ = "rebalance_decisions"
    id                    = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    proposal_id           = Column(UUID(as_uuid=False), ForeignKey("rebalance_proposals.id"), nullable=False)
    decision              = Column(Enum(DecisionType), nullable=False)
    modified_weights_json = Column(JSONB)
    reason                = Column(Text)
    decided_at            = Column(DateTime, default=datetime.utcnow, nullable=False)
    proposal              = relationship("RebalanceProposal", back_populates="decisions")


class Trade(Base):
    __tablename__ = "trades"
    id              = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    proposal_id     = Column(UUID(as_uuid=False), ForeignKey("rebalance_proposals.id"), nullable=False)
    ticker          = Column(String(10), nullable=False)
    action          = Column(String(4), nullable=False)
    shares          = Column(Float)
    estimated_price = Column(Float)
    estimated_value = Column(Float)
    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False)
    proposal        = relationship("RebalanceProposal", back_populates="trades")


class BenchmarkFact(Base):
    """Seed-only table — locked from Table 1, Cohen et al. (2025). Read-only at runtime."""
    __tablename__ = "benchmark_facts"
    id                = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    strategy          = Column(Enum(StrategyType), nullable=False)
    frequency         = Column(Enum(RebalanceFreq), nullable=False)
    ml_weight         = Column(Float, nullable=False)
    llm_weight        = Column(Float, nullable=False)
    sharpe_ratio      = Column(Float, nullable=False)
    average_return    = Column(Float, nullable=False)
    volatility        = Column(Float, nullable=False)
    cumulative_return = Column(Float, nullable=False)
    notes             = Column(Text)
    source            = Column(String, nullable=False, default="Table 1, Cohen et al., Entropy 2025, 27, 550")
    __table_args__    = (UniqueConstraint("strategy", "frequency"),)
'''

# ────────────────────────────────────────────────────────────
# app/schemas.py — Pydantic request/response schemas
# ────────────────────────────────────────────────────────────
SCHEMAS = '''
from __future__ import annotations
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class PortfolioCreate(BaseModel):
    name: str = "My Portfolio"
    universe: str = "NASDAQ-100"
    benchmark: str = "QQQ"


class PortfolioConstraintsUpdate(BaseModel):
    max_position_pct: float = Field(0.25, ge=0, le=1)
    sector_cap_pct:   float = Field(0.40, ge=0, le=1)
    min_cash_pct:     float = Field(0.02, ge=0, le=1)
    max_cash_pct:     float = Field(0.10, ge=0, le=1)
    excluded_tickers: list[str] = Field(default_factory=list)
    esg_filter:       bool = False


class HoldingOut(BaseModel):
    id: str; portfolio_id: str; ticker: str; shares: float
    cost_basis: Optional[float]; currency: str; uploaded_at: datetime
    class Config: from_attributes = True


class PortfolioOut(BaseModel):
    id: str; user_id: str; name: str; universe: str
    benchmark: str; created_at: datetime; updated_at: datetime
    holdings: list[HoldingOut]
    class Config: from_attributes = True


class ScoreRunRequest(BaseModel):
    portfolio_id: str
    frequency: str = "monthly"


class ScoreRunOut(BaseModel):
    id: str; portfolio_id: str; run_date: datetime
    frequency: str; status: str; model_version: Optional[str]
    error_log: Optional[str]; created_at: datetime
    class Config: from_attributes = True


class LLMReasoningOut(BaseModel):
    score: float; key_positives: list[str]
    key_risks: list[str]; confidence: str


class ScoreOut(BaseModel):
    id: str; run_id: str; ticker: str
    technical_ml_score: Optional[float]; fundamental_ml_score: Optional[float]
    entropy_ml_score: Optional[float]; llm_score: Optional[float]
    llm_provider: str; llm_reasoning_json: Optional[dict[str, Any]]
    technical_score: Optional[float]; fundamental_score: Optional[float]
    entropy_score: Optional[float]; combined_score: Optional[float]
    w_technical: float; w_fundamental: float; w_entropy: float
    forward_return_forecast: Optional[float]; created_at: datetime
    class Config: from_attributes = True


class ScoreRunWithScores(BaseModel):
    run: ScoreRunOut; scores: list[ScoreOut]


class OptimizeRequest(BaseModel):
    portfolio_id: str; run_id: str
    settings: dict[str, Any] = Field(default_factory=dict)


class OptimizeOut(BaseModel):
    job_id: str


class OptimizationJobOut(BaseModel):
    id: str; portfolio_id: str; run_id: Optional[str]
    optimizer_type: str; status: str
    settings_json: Optional[dict]; result_json: Optional[dict]
    created_at: datetime
    class Config: from_attributes = True


class RebalanceProposeRequest(BaseModel):
    portfolio_id: str; optimization_job_id: str


class RebalanceModifyRequest(BaseModel):
    weights: dict[str, float]

    @field_validator("weights")
    @classmethod
    def weights_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"Weights must sum to ~1.0, got {total:.4f}")
        return v


class RejectRequest(BaseModel):
    reason: Optional[str] = None


class TradeOut(BaseModel):
    ticker: str; action: str; shares: Optional[float]
    estimated_price: Optional[float]; estimated_value: Optional[float]
    class Config: from_attributes = True


class BacktestRequest(BaseModel):
    portfolio_id: str
    strategies: list[str] = ["technical", "fundamental", "entropy"]
    start_date: str = "2020-01-01"
    end_date:   str = "2025-01-01"


class BenchmarkFactOut(BaseModel):
    strategy: str; frequency: str; ml_weight: float; llm_weight: float
    sharpe_ratio: float; average_return: float; volatility: float
    cumulative_return: float; notes: Optional[str]; source: str
    class Config: from_attributes = True
'''

# Write all files
import os

FILES = {
    "app/main.py":    MAIN,
    "app/database.py": DATABASE,
    "app/models.py":  MODELS,
    "app/schemas.py": SCHEMAS,
}

for path, content in FILES.items():
    print(f"# ── {path}")
    print(content.strip())
    print()
