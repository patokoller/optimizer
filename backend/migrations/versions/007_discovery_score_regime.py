"""Add regime_adjusted and regime_label to discovery_scores

Revision ID: 007_discovery_score_regime
Revises: 006_enrichment_cache
Create Date: 2026-05-27
"""
from alembic import op
import sqlalchemy as sa

revision = "007_discovery_score_regime"
down_revision = "006_enrichment_cache"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    conn.execute(sa.text("""
        ALTER TABLE discovery_scores
            ADD COLUMN IF NOT EXISTS regime_adjusted BOOLEAN NOT NULL DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS regime_label     TEXT
    """))


def downgrade():
    conn = op.get_bind()
    conn.execute(sa.text("""
        ALTER TABLE discovery_scores
            DROP COLUMN IF EXISTS regime_adjusted,
            DROP COLUMN IF EXISTS regime_label
    """))
