"""
SQLAlchemy ORM models — AI Portfolio Decision-Support Platform
Schema: Cohen, Aiche & Eichel (2025), Entropy 27, 550
"""
import uuid
import enum
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime,
    ForeignKey, Enum, Text, ARRAY, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
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
    id         = Column(String, primary_key=True, default=_uuid)
    email      = Column(String, unique=True, nullable=False, index=True)
    name       = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")


class Portfolio(Base):
    __tablename__ = "portfolios"
    id         = Column(String, primary_key=True, default=_uuid)
    user_id    = Column(String, ForeignKey("users.id"), nullable=False)
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
    id           = Column(String, primary_key=True, default=_uuid)
    portfolio_id = Column(String, ForeignKey("portfolios.id"), nullable=False)
    ticker       = Column(String(10), nullable=False, index=True)
    shares       = Column(Float, nullable=False)
    cost_basis   = Column(Float)
    currency     = Column(String(3), nullable=False, default="USD")
    uploaded_at  = Column(DateTime, default=datetime.utcnow, nullable=False)
    portfolio    = relationship("Portfolio", back_populates="holdings")


class Constraint(Base):
    __tablename__ = "constraints"
    id               = Column(String, primary_key=True, default=_uuid)
    portfolio_id     = Column(String, ForeignKey("portfolios.id"), nullable=False, unique=True)
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
    id            = Column(String, primary_key=True, default=_uuid)
    portfolio_id  = Column(String, ForeignKey("portfolios.id"), nullable=False)
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
    id                      = Column(String, primary_key=True, default=_uuid)
    run_id                  = Column(String, ForeignKey("score_runs.id"), nullable=False)
    ticker                  = Column(String(10), nullable=False, index=True)
    technical_ml_score      = Column(Float)
    fundamental_ml_score    = Column(Float)
    entropy_ml_score        = Column(Float)
    llm_score               = Column(Float)
    llm_provider            = Column(Enum(LLMProvider), nullable=False, default=LLMProvider.claude)
    llm_reasoning_json      = Column(JSONB)
    technical_score         = Column(Float)
    fundamental_score       = Column(Float)
    entropy_score           = Column(Float)
    combined_score          = Column(Float)
    w_technical             = Column(Float, nullable=False, default=1.00)
    w_fundamental           = Column(Float, nullable=False, default=0.15)
    w_entropy               = Column(Float, nullable=False, default=0.70)
    forward_return_forecast = Column(Float)
    created_at              = Column(DateTime, default=datetime.utcnow, nullable=False)
    __table_args__          = (UniqueConstraint("run_id", "ticker"),)
    run = relationship("ScoreRun", back_populates="scores")


class OptimizationJob(Base):
    __tablename__ = "optimization_jobs"
    id             = Column(String, primary_key=True, default=_uuid)
    portfolio_id   = Column(String, ForeignKey("portfolios.id"), nullable=False)
    run_id         = Column(String, ForeignKey("score_runs.id"))
    optimizer_type = Column(Enum(OptimizerType), nullable=False)
    status         = Column(Enum(RunStatus), nullable=False, default=RunStatus.pending)
    settings_json  = Column(JSONB)
    result_json    = Column(JSONB)
    created_at     = Column(DateTime, default=datetime.utcnow, nullable=False)


class RebalanceProposal(Base):
    __tablename__ = "rebalance_proposals"
    id                    = Column(String, primary_key=True, default=_uuid)
    portfolio_id          = Column(String, ForeignKey("portfolios.id"), nullable=False)
    optimization_job_id   = Column(String, ForeignKey("optimization_jobs.id"))
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
    id                    = Column(String, primary_key=True, default=_uuid)
    proposal_id           = Column(String, ForeignKey("rebalance_proposals.id"), nullable=False)
    decision              = Column(Enum(DecisionType), nullable=False)
    modified_weights_json = Column(JSONB)
    reason                = Column(Text)
    decided_at            = Column(DateTime, default=datetime.utcnow, nullable=False)
    proposal              = relationship("RebalanceProposal", back_populates="decisions")


class Trade(Base):
    __tablename__ = "trades"
    id              = Column(String, primary_key=True, default=_uuid)
    proposal_id     = Column(String, ForeignKey("rebalance_proposals.id"), nullable=False)
    ticker          = Column(String(10), nullable=False)
    action          = Column(String(4), nullable=False)
    shares          = Column(Float)
    estimated_price = Column(Float)
    estimated_value = Column(Float)
    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False)
    proposal        = relationship("RebalanceProposal", back_populates="trades")


class BenchmarkFact(Base):
    """Seed-only. Locked from Table 1, Cohen et al. (2025). Read-only at runtime."""
    __tablename__ = "benchmark_facts"
    id                = Column(String, primary_key=True, default=_uuid)
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
