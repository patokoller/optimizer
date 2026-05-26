"""Add enrichment_cache table for AV API response caching

Revision ID: 006_enrichment_cache
Revises: 005_holding_is_etf
Create Date: 2026-05-22
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "006_enrichment_cache"
down_revision = "005_holding_is_etf"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS enrichment_cache (
            id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
            ticker      TEXT NOT NULL,
            cache_month TEXT NOT NULL,  -- YYYY-MM format
            context     JSONB NOT NULL, -- all enrichment signals
            fetched_at  TIMESTAMP NOT NULL DEFAULT now(),
            UNIQUE (ticker, cache_month)
        )
    """))
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS idx_enrichment_cache_lookup "
        "ON enrichment_cache (ticker, cache_month)"
    ))


def downgrade():
    conn = op.get_bind()
    conn.execute(sa.text("DROP TABLE IF EXISTS enrichment_cache"))
