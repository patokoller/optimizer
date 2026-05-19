"""Add ETF composite fields to scores table

Revision ID: 003_etf_composite
Revises: 002_phase1_scoring
Create Date: 2026-05-19
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "003_etf_composite"
down_revision = "002_phase1_scoring"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    conn.execute(sa.text(
        "ALTER TABLE scores ADD COLUMN IF NOT EXISTS is_etf_composite BOOLEAN DEFAULT FALSE"
    ))
    conn.execute(sa.text(
        "ALTER TABLE scores ADD COLUMN IF NOT EXISTS etf_type VARCHAR"
    ))
    conn.execute(sa.text(
        "ALTER TABLE scores ADD COLUMN IF NOT EXISTS etf_holdings_used JSONB"
    ))


def downgrade():
    conn = op.get_bind()
    for col in ["is_etf_composite", "etf_type", "etf_holdings_used"]:
        conn.execute(sa.text(f"ALTER TABLE scores DROP COLUMN IF EXISTS {col}"))
