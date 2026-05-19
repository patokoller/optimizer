"""Phase 1 upgrades: component scores, dispersion, feature importances, confidence, delta, risk metrics, market_regimes table

Revision ID: 002_phase1_scoring
Revises: 001_initial_schema
Create Date: 2026-05-19
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "002_phase1_scoring"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()

    # ── Add columns to scores with IF NOT EXISTS (idempotent) ────────────
    float_cols = [
        "fundamental_ridge_score", "fundamental_xgb_score",
        "fundamental_rf_score",    "fundamental_mlp_score",
        "technical_xgb_score",     "technical_lgbm_score",  "technical_cat_score",
        "entropy_xgb_score",       "entropy_lgbm_score",    "entropy_cat_score",
        "fundamental_dispersion",  "technical_dispersion",
        "entropy_dispersion",      "overall_dispersion",
        "confidence_score",        "model_agreement",        "llm_ml_alignment",
        "prev_combined_score",     "score_delta",            "confidence_delta",
        "realised_vol_21d",        "realised_vol_63d",       "beta_vs_qqq",
        "max_drawdown_1y",         "sharpe_1y",
    ]
    for col in float_cols:
        conn.execute(sa.text(
            f"ALTER TABLE scores ADD COLUMN IF NOT EXISTS {col} FLOAT"
        ))

    conn.execute(sa.text(
        "ALTER TABLE scores ADD COLUMN IF NOT EXISTS rank_delta INTEGER"
    ))

    jsonb_cols = [
        "fundamental_feature_importance",
        "technical_feature_importance",
        "entropy_feature_importance",
    ]
    for col in jsonb_cols:
        conn.execute(sa.text(
            f"ALTER TABLE scores ADD COLUMN IF NOT EXISTS {col} JSONB"
        ))

    # ── Create market_regimes table with IF NOT EXISTS ────────────────────
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS market_regimes (
            id                VARCHAR      PRIMARY KEY,
            run_id            VARCHAR      NOT NULL UNIQUE REFERENCES score_runs(id),
            regime_label      VARCHAR      NOT NULL,
            regime_confidence FLOAT        NOT NULL,
            vix               FLOAT,
            yield_curve_10y2y FLOAT,
            fed_funds_rate    FLOAT,
            cpi_yoy           FLOAT,
            dominant_factor   VARCHAR,
            factor_weight_adj JSONB,
            transition_risk   VARCHAR,
            raw_fred_json     JSONB,
            computed_at       TIMESTAMP    NOT NULL DEFAULT NOW()
        )
    """))


def downgrade():
    conn = op.get_bind()
    conn.execute(sa.text("DROP TABLE IF EXISTS market_regimes"))

    cols = [
        "fundamental_ridge_score", "fundamental_xgb_score", "fundamental_rf_score",
        "fundamental_mlp_score", "technical_xgb_score", "technical_lgbm_score",
        "technical_cat_score", "entropy_xgb_score", "entropy_lgbm_score",
        "entropy_cat_score", "fundamental_dispersion", "technical_dispersion",
        "entropy_dispersion", "overall_dispersion", "fundamental_feature_importance",
        "technical_feature_importance", "entropy_feature_importance",
        "confidence_score", "model_agreement", "llm_ml_alignment",
        "prev_combined_score", "score_delta", "rank_delta", "confidence_delta",
        "realised_vol_21d", "realised_vol_63d", "beta_vs_qqq",
        "max_drawdown_1y", "sharpe_1y",
    ]
    for col in cols:
        conn.execute(sa.text(f"ALTER TABLE scores DROP COLUMN IF EXISTS {col}"))

