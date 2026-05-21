"""Add is_etf flag to holdings table

Revision ID: 005_holding_is_etf
Revises: 004_discovery
Create Date: 2026-05-21
"""
from alembic import op
import sqlalchemy as sa

revision = "005_holding_is_etf"
down_revision = "004_discovery"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    conn.execute(sa.text(
        "ALTER TABLE holdings ADD COLUMN IF NOT EXISTS is_etf BOOLEAN NOT NULL DEFAULT FALSE"
    ))


def downgrade():
    conn = op.get_bind()
    conn.execute(sa.text("ALTER TABLE holdings DROP COLUMN IF EXISTS is_etf"))
