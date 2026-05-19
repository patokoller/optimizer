"""Pydantic v2 request/response schemas."""
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
    cost_basis: Optional[float] = None; currency: str; uploaded_at: datetime
    model_config = {"from_attributes": True}


class PortfolioOut(BaseModel):
    id: str; user_id: str; name: str; universe: str
    benchmark: str; created_at: datetime; updated_at: datetime
    holdings: list[HoldingOut]
    model_config = {"from_attributes": True}


class ScoreRunRequest(BaseModel):
    portfolio_id: str
    frequency: str = "monthly"


class ScoreRunOut(BaseModel):
    id: str; portfolio_id: str; run_date: datetime
    frequency: str; status: str; model_version: Optional[str] = None
    error_log: Optional[str] = None; created_at: datetime
    model_config = {"from_attributes": True, "protected_namespaces": ()}


class ScoreOut(BaseModel):
    id: str; run_id: str; ticker: str

    # Individual model component scores
    fundamental_ridge_score: Optional[float] = None
    fundamental_xgb_score:   Optional[float] = None
    fundamental_rf_score:    Optional[float] = None
    fundamental_mlp_score:   Optional[float] = None
    technical_xgb_score:     Optional[float] = None
    technical_lgbm_score:    Optional[float] = None
    technical_cat_score:     Optional[float] = None
    entropy_xgb_score:       Optional[float] = None
    entropy_lgbm_score:      Optional[float] = None
    entropy_cat_score:       Optional[float] = None

    # Ensemble dispersion (std dev of components — real confidence proxy)
    fundamental_dispersion: Optional[float] = None
    technical_dispersion:   Optional[float] = None
    entropy_dispersion:     Optional[float] = None
    overall_dispersion:     Optional[float] = None

    # Feature importances
    fundamental_feature_importance: Optional[dict[str, Any]] = None
    technical_feature_importance:   Optional[dict[str, Any]] = None
    entropy_feature_importance:     Optional[dict[str, Any]] = None

    # Ensemble + LLM scores
    technical_ml_score:   Optional[float] = None
    fundamental_ml_score: Optional[float] = None
    entropy_ml_score:     Optional[float] = None
    llm_score:            Optional[float] = None
    llm_provider:         str
    llm_reasoning_json:   Optional[dict[str, Any]] = None

    # Combined strategy scores
    technical_score:   Optional[float] = None
    fundamental_score: Optional[float] = None
    entropy_score:     Optional[float] = None
    combined_score:    Optional[float] = None

    # Locked ML weights
    w_technical: float; w_fundamental: float; w_entropy: float

    # Confidence metrics
    confidence_score:  Optional[float] = None
    model_agreement:   Optional[float] = None
    llm_ml_alignment:  Optional[float] = None

    # Delta vs previous run
    prev_combined_score: Optional[float] = None
    score_delta:         Optional[float] = None
    rank_delta:          Optional[int]   = None
    confidence_delta:    Optional[float] = None

    # Risk metrics
    realised_vol_21d: Optional[float] = None
    realised_vol_63d: Optional[float] = None
    beta_vs_qqq:      Optional[float] = None
    max_drawdown_1y:  Optional[float] = None
    sharpe_1y:        Optional[float] = None

    forward_return_forecast: Optional[float] = None
    created_at: datetime
    model_config = {"from_attributes": True}


class RegimeOut(BaseModel):
    """Market regime snapshot for the latest score run."""
    id: str
    run_id: str
    regime_label: str
    regime_confidence: float
    vix: Optional[float] = None
    yield_curve_10y2y: Optional[float] = None
    fed_funds_rate: Optional[float] = None
    cpi_yoy: Optional[float] = None
    dominant_factor: Optional[str] = None
    factor_weight_adj: Optional[dict[str, Any]] = None
    transition_risk: Optional[str] = None
    computed_at: datetime
    model_config = {"from_attributes": True}


class ScoreRunWithScores(BaseModel):
    run: ScoreRunOut
    scores: list[ScoreOut]


class OptimizeRequest(BaseModel):
    portfolio_id: str
    run_id: str
    settings: dict[str, Any] = Field(default_factory=dict)


class OptimizeOut(BaseModel):
    job_id: str


class OptimizationJobOut(BaseModel):
    id: str; portfolio_id: str; run_id: Optional[str] = None
    optimizer_type: str; status: str
    settings_json: Optional[dict] = None
    result_json: Optional[dict] = None
    created_at: datetime
    model_config = {"from_attributes": True}


class RebalanceProposeRequest(BaseModel):
    portfolio_id: str
    optimization_job_id: str


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
    ticker: str; action: str; shares: Optional[float] = None
    estimated_price: Optional[float] = None
    estimated_value: Optional[float] = None
    model_config = {"from_attributes": True}


class BacktestRequest(BaseModel):
    portfolio_id: str
    strategies: list[str] = ["technical", "fundamental", "entropy"]
    start_date: str = "2020-01-01"
    end_date: str   = "2025-01-01"


class BenchmarkFactOut(BaseModel):
    strategy: str; frequency: str; ml_weight: float; llm_weight: float
    sharpe_ratio: float; average_return: float; volatility: float
    cumulative_return: float; notes: Optional[str] = None; source: str
    model_config = {"from_attributes": True}
