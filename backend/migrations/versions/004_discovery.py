"""Add discovery_runs and discovery_scores tables

Revision ID: 004_discovery
Revises: 003_etf_composite
Create Date: 2026-05-19
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "004_discovery"
down_revision = "003_etf_composite"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()

    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS discovery_runs (
            id                VARCHAR      PRIMARY KEY,
            status            VARCHAR      NOT NULL DEFAULT 'pending',
            run_date          TIMESTAMP    NOT NULL DEFAULT NOW(),
            universe          VARCHAR      NOT NULL DEFAULT 'NASDAQ-100',
            universe_size     INTEGER,
            scored_count      INTEGER,
            regime_label      VARCHAR,
            regime_confidence FLOAT,
            error_log         TEXT,
            created_at        TIMESTAMP    NOT NULL DEFAULT NOW()
        )
    """))

    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS discovery_scores (
            id                              VARCHAR   PRIMARY KEY,
            discovery_run_id                VARCHAR   NOT NULL REFERENCES discovery_runs(id),
            ticker                          VARCHAR(10) NOT NULL,
            sector                          VARCHAR,
            technical_score                 FLOAT,
            fundamental_score               FLOAT,
            entropy_score                   FLOAT,
            combined_score                  FLOAT,
            llm_score                       FLOAT,
            llm_provider                    VARCHAR   DEFAULT 'none',
            llm_reasoning_json              JSONB,
            confidence_score                FLOAT,
            overall_dispersion              FLOAT,
            prev_combined_score             FLOAT,
            score_delta                     FLOAT,
            rank                            INTEGER,
            prev_rank                       INTEGER,
            rank_delta                      INTEGER,
            technical_feature_importance    JSONB,
            fundamental_feature_importance  JSONB,
            realised_vol_21d                FLOAT,
            beta_vs_qqq                     FLOAT,
            sharpe_1y                       FLOAT,
            created_at                      TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """))

    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_discovery_scores_run_id ON discovery_scores(discovery_run_id)"
    ))
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_discovery_scores_combined ON discovery_scores(discovery_run_id, combined_score DESC)"
    ))


def downgrade():
    conn = op.get_bind()
    conn.execute(sa.text("DROP TABLE IF EXISTS discovery_scores"))
    conn.execute(sa.text("DROP TABLE IF EXISTS discovery_runs"))
