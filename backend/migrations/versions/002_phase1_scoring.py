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
    # ── Individual model component scores ─────────────────────────────────
    for col in [
        "fundamental_ridge_score", "fundamental_xgb_score",
        "fundamental_rf_score",    "fundamental_mlp_score",
        "technical_xgb_score",     "technical_lgbm_score",  "technical_cat_score",
        "entropy_xgb_score",       "entropy_lgbm_score",    "entropy_cat_score",
    ]:
        op.add_column("scores", sa.Column(col, sa.Float(), nullable=True))

    # ── Ensemble dispersion ───────────────────────────────────────────────
    for col in [
        "fundamental_dispersion", "technical_dispersion",
        "entropy_dispersion",     "overall_dispersion",
    ]:
        op.add_column("scores", sa.Column(col, sa.Float(), nullable=True))

    # ── Feature importances (JSONB) ────────────────────────────────────────
    for col in [
        "fundamental_feature_importance",
        "technical_feature_importance",
        "entropy_feature_importance",
    ]:
        op.add_column("scores", sa.Column(col, JSONB(), nullable=True))

    # ── Confidence metrics ────────────────────────────────────────────────
    op.add_column("scores", sa.Column("confidence_score",  sa.Float(), nullable=True))
    op.add_column("scores", sa.Column("model_agreement",   sa.Float(), nullable=True))
    op.add_column("scores", sa.Column("llm_ml_alignment",  sa.Float(), nullable=True))

    # ── Delta vs previous run ─────────────────────────────────────────────
    op.add_column("scores", sa.Column("prev_combined_score", sa.Float(),   nullable=True))
    op.add_column("scores", sa.Column("score_delta",         sa.Float(),   nullable=True))
    op.add_column("scores", sa.Column("rank_delta",          sa.Integer(), nullable=True))
    op.add_column("scores", sa.Column("confidence_delta",    sa.Float(),   nullable=True))

    # ── Risk metrics from Alpaca prices ───────────────────────────────────
    op.add_column("scores", sa.Column("realised_vol_21d", sa.Float(), nullable=True))
    op.add_column("scores", sa.Column("realised_vol_63d", sa.Float(), nullable=True))
    op.add_column("scores", sa.Column("beta_vs_qqq",      sa.Float(), nullable=True))
    op.add_column("scores", sa.Column("max_drawdown_1y",  sa.Float(), nullable=True))
    op.add_column("scores", sa.Column("sharpe_1y",        sa.Float(), nullable=True))

    # ── Market regimes table ──────────────────────────────────────────────
    op.create_table(
        "market_regimes",
        sa.Column("id",                sa.String(),  primary_key=True),
        sa.Column("run_id",            sa.String(),  sa.ForeignKey("score_runs.id"), nullable=False, unique=True),
        sa.Column("regime_label",      sa.String(),  nullable=False),
        sa.Column("regime_confidence", sa.Float(),   nullable=False),
        sa.Column("vix",               sa.Float(),   nullable=True),
        sa.Column("yield_curve_10y2y", sa.Float(),   nullable=True),
        sa.Column("fed_funds_rate",    sa.Float(),   nullable=True),
        sa.Column("cpi_yoy",           sa.Float(),   nullable=True),
        sa.Column("dominant_factor",   sa.String(),  nullable=True),
        sa.Column("factor_weight_adj", JSONB(),      nullable=True),
        sa.Column("transition_risk",   sa.String(),  nullable=True),
        sa.Column("raw_fred_json",     JSONB(),      nullable=True),
        sa.Column("computed_at",       sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("market_regimes")

    for col in [
        "realised_vol_21d", "realised_vol_63d", "beta_vs_qqq",
        "max_drawdown_1y", "sharpe_1y",
        "prev_combined_score", "score_delta", "rank_delta", "confidence_delta",
        "confidence_score", "model_agreement", "llm_ml_alignment",
        "fundamental_feature_importance", "technical_feature_importance", "entropy_feature_importance",
        "fundamental_dispersion", "technical_dispersion", "entropy_dispersion", "overall_dispersion",
        "fundamental_ridge_score", "fundamental_xgb_score", "fundamental_rf_score", "fundamental_mlp_score",
        "technical_xgb_score", "technical_lgbm_score", "technical_cat_score",
        "entropy_xgb_score", "entropy_lgbm_score", "entropy_cat_score",
    ]:
        op.drop_column("scores", col)
