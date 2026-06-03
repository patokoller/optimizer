"""Add score-validation tables: discovery_forward_returns + discovery_ic_metrics

Revision ID: 008_score_validation
Revises: 007_discovery_score_regime
Create Date: 2026-06-04

Stores realized forward returns per (discovery_run, ticker) and the
cross-sectional rank-IC per (run, horizon, score_column), so the discovery
engine can be measured against subsequent returns rather than assumed correct.

Note: tables are also auto-created by Base.metadata.create_all() on app
startup; this migration keeps the Alembic chain consistent and is idempotent.
"""
from alembic import op
import sqlalchemy as sa

revision = "008_score_validation"
down_revision = "007_discovery_score_regime"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS discovery_forward_returns (
            id                VARCHAR PRIMARY KEY,
            discovery_run_id  VARCHAR NOT NULL REFERENCES discovery_runs(id),
            ticker            VARCHAR(10) NOT NULL,
            run_date          TIMESTAMP NOT NULL,
            combined_score    DOUBLE PRECISION,
            technical_score   DOUBLE PRECISION,
            fundamental_score DOUBLE PRECISION,
            entropy_score     DOUBLE PRECISION,
            llm_score         DOUBLE PRECISION,
            rank              INTEGER,
            anchor_close      DOUBLE PRECISION,
            fwd_return_21d    DOUBLE PRECISION,
            fwd_return_63d    DOUBLE PRECISION,
            filled_21d_at     TIMESTAMP,
            filled_63d_at     TIMESTAMP,
            created_at        TIMESTAMP NOT NULL DEFAULT now(),
            CONSTRAINT uq_fwd_run_ticker UNIQUE (discovery_run_id, ticker)
        )
    """))
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_fwd_run ON discovery_forward_returns (discovery_run_id)"
    ))
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_fwd_ticker ON discovery_forward_returns (ticker)"
    ))
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS discovery_ic_metrics (
            id               VARCHAR PRIMARY KEY,
            discovery_run_id VARCHAR NOT NULL REFERENCES discovery_runs(id),
            run_date         TIMESTAMP NOT NULL,
            horizon_days     INTEGER NOT NULL,
            score_column     VARCHAR NOT NULL,
            rank_ic          DOUBLE PRECISION,
            topk_spread      DOUBLE PRECISION,
            universe_mean    DOUBLE PRECISION,
            n                INTEGER NOT NULL,
            computed_at      TIMESTAMP NOT NULL DEFAULT now(),
            CONSTRAINT uq_ic_run_horizon_col UNIQUE (discovery_run_id, horizon_days, score_column)
        )
    """))
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS ix_ic_run ON discovery_ic_metrics (discovery_run_id)"
    ))


def downgrade():
    conn = op.get_bind()
    conn.execute(sa.text("DROP TABLE IF EXISTS discovery_ic_metrics"))
    conn.execute(sa.text("DROP TABLE IF EXISTS discovery_forward_returns"))
