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
    is_etf       = Column(Boolean, nullable=False, default=False)  # user-set flag
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

    # Ensemble component scores (individual models — enables real dispersion)
    fundamental_ridge_score = Column(Float)
    fundamental_xgb_score   = Column(Float)
    fundamental_rf_score    = Column(Float)
    fundamental_mlp_score   = Column(Float)
    technical_xgb_score     = Column(Float)
    technical_lgbm_score    = Column(Float)
    technical_cat_score     = Column(Float)
    entropy_xgb_score       = Column(Float)
    entropy_lgbm_score      = Column(Float)
    entropy_cat_score       = Column(Float)

    # Ensemble dispersion (std dev of component scores — real confidence proxy)
    fundamental_dispersion  = Column(Float)
    technical_dispersion    = Column(Float)
    entropy_dispersion      = Column(Float)
    overall_dispersion      = Column(Float)

    # XGBoost feature importances per strategy (enables real factor attribution)
    fundamental_feature_importance = Column(JSONB)  # {feature: importance}
    technical_feature_importance   = Column(JSONB)
    entropy_feature_importance     = Column(JSONB)

    # LLM semantic score
    technical_ml_score      = Column(Float)
    fundamental_ml_score    = Column(Float)
    entropy_ml_score        = Column(Float)
    llm_score               = Column(Float)
    llm_provider            = Column(Enum(LLMProvider), nullable=False, default=LLMProvider.claude)
    llm_reasoning_json      = Column(JSONB)

    # Combined scores
    technical_score         = Column(Float)
    fundamental_score       = Column(Float)
    entropy_score           = Column(Float)
    combined_score          = Column(Float)

    # Locked ML weights (Table 1, Cohen et al. 2025)
    w_technical             = Column(Float, nullable=False, default=1.00)
    w_fundamental           = Column(Float, nullable=False, default=0.15)
    w_entropy               = Column(Float, nullable=False, default=0.70)

    # Derived confidence metrics
    confidence_score        = Column(Float)   # 0-1, higher = more confident
    model_agreement         = Column(Float)   # 0-1, agreement across strategies
    llm_ml_alignment        = Column(Float)   # 0-1, does LLM agree with ML direction

    # Delta vs previous run (populated after second run)
    prev_combined_score     = Column(Float)
    score_delta             = Column(Float)
    rank_delta              = Column(Integer)
    confidence_delta        = Column(Float)

    # Risk metrics computed from Alpaca price data
    realised_vol_21d        = Column(Float)
    realised_vol_63d        = Column(Float)
    beta_vs_qqq             = Column(Float)
    max_drawdown_1y         = Column(Float)
    sharpe_1y               = Column(Float)

    forward_return_forecast = Column(Float)
    # ETF composite metadata
    is_etf_composite     = Column(Boolean, default=False)
    etf_type             = Column(String)   # "STOCK"|"EQUITY_ETF"|"BOND_ETF"|"CRYPTO_ETF"|"NON_SCOREABLE"
    etf_holdings_used    = Column(JSONB)    # [{ticker, weight}] for EQUITY_ETF
    created_at              = Column(DateTime, default=datetime.utcnow, nullable=False)
    __table_args__          = (UniqueConstraint("run_id", "ticker"),)
    run = relationship("ScoreRun", back_populates="scores")


class MarketRegime(Base):
    """
    Stores the market regime snapshot for each score run.
    Computed from FRED macro data: VIX, yield curve, Fed funds, CPI.
    """
    __tablename__ = "market_regimes"
    id                   = Column(String, primary_key=True, default=_uuid)
    run_id               = Column(String, ForeignKey("score_runs.id"), nullable=False, unique=True)
    regime_label         = Column(String, nullable=False)   # e.g. "Risk-On Momentum"
    regime_confidence    = Column(Float, nullable=False)    # 0-1
    vix                  = Column(Float)
    yield_curve_10y2y    = Column(Float)
    fed_funds_rate       = Column(Float)
    cpi_yoy              = Column(Float)
    dominant_factor      = Column(String)   # e.g. "Momentum", "Quality", "Defensive"
    factor_weight_adj    = Column(JSONB)    # {"technical": 1.1, "fundamental": 0.9, ...}
    transition_risk      = Column(String)   # "low" | "medium" | "high"
    raw_fred_json        = Column(JSONB)
    computed_at          = Column(DateTime, default=datetime.utcnow, nullable=False)


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


# ── Discovery (NASDAQ-100 universe scoring) ─────────────────────────────────

class DiscoveryRun(Base):
    """A full NASDAQ-100 universe scoring run, separate from portfolio scoring."""
    __tablename__ = "discovery_runs"
    id               = Column(String, primary_key=True, default=_uuid)
    status           = Column(Enum(RunStatus), default=RunStatus.pending, nullable=False)
    run_date         = Column(DateTime, default=datetime.utcnow, nullable=False)
    universe         = Column(String, default="NASDAQ-100", nullable=False)
    universe_size    = Column(Integer)
    scored_count     = Column(Integer)
    regime_label     = Column(String)
    regime_confidence = Column(Float)
    error_log        = Column(Text)
    created_at       = Column(DateTime, default=datetime.utcnow, nullable=False)
    scores           = relationship("DiscoveryScore", back_populates="run",
                                    cascade="all, delete-orphan")


class DiscoveryScore(Base):
    """Score for a single NASDAQ-100 ticker within a discovery run."""
    __tablename__ = "discovery_scores"
    id                   = Column(String, primary_key=True, default=_uuid)
    discovery_run_id     = Column(String, ForeignKey("discovery_runs.id"), nullable=False)
    ticker               = Column(String(10), nullable=False)
    sector               = Column(String)
    # Strategy scores
    technical_score      = Column(Float)
    fundamental_score    = Column(Float)
    entropy_score        = Column(Float)
    combined_score       = Column(Float)
    # LLM
    llm_score            = Column(Float)
    llm_provider         = Column(Enum(LLMProvider), default=LLMProvider.none)
    llm_reasoning_json   = Column(JSONB)
    # Confidence & dispersion
    confidence_score     = Column(Float)
    overall_dispersion   = Column(Float)
    # Delta vs previous discovery run
    prev_combined_score  = Column(Float)
    score_delta          = Column(Float)
    rank                 = Column(Integer)
    prev_rank            = Column(Integer)
    rank_delta           = Column(Integer)
    # Feature importances
    technical_feature_importance  = Column(JSONB)
    fundamental_feature_importance = Column(JSONB)
    # Risk metrics
    realised_vol_21d     = Column(Float)
    beta_vs_qqq          = Column(Float)
    sharpe_1y            = Column(Float)
    created_at           = Column(DateTime, default=datetime.utcnow, nullable=False)
    run                  = relationship("DiscoveryRun", back_populates="scores")
