"""Initial schema — AI Portfolio Decision-Support Platform

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-05-18 00:00:00.000000

Source: Cohen, Aiche & Eichel (2025), Entropy 27, 550
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── users ──────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id",         sa.String, primary_key=True),
        sa.Column("email",      sa.String, nullable=False, unique=True),
        sa.Column("name",       sa.String),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_users_email", "users", ["email"])

    # ── portfolios ─────────────────────────────────────────────────────────
    op.create_table(
        "portfolios",
        sa.Column("id",         sa.String, primary_key=True),
        sa.Column("user_id",    sa.String, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name",       sa.String, nullable=False),
        sa.Column("universe",   sa.String, nullable=False, server_default="NASDAQ-100"),
        sa.Column("benchmark",  sa.String, nullable=False, server_default="QQQ"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
    )

    # ── holdings ───────────────────────────────────────────────────────────
    op.create_table(
        "holdings",
        sa.Column("id",           sa.String, primary_key=True),
        sa.Column("portfolio_id", sa.String, sa.ForeignKey("portfolios.id"), nullable=False),
        sa.Column("ticker",       sa.String(10), nullable=False),
        sa.Column("shares",       sa.Float, nullable=False),
        sa.Column("cost_basis",   sa.Float),
        sa.Column("currency",     sa.String(3), nullable=False, server_default="USD"),
        sa.Column("uploaded_at",  sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_holdings_ticker", "holdings", ["ticker"])

    # ── constraints ────────────────────────────────────────────────────────
    op.create_table(
        "constraints",
        sa.Column("id",               sa.String, primary_key=True),
        sa.Column("portfolio_id",     sa.String, sa.ForeignKey("portfolios.id"), nullable=False, unique=True),
        sa.Column("max_position_pct", sa.Float, nullable=False, server_default="0.25"),
        sa.Column("sector_cap_pct",   sa.Float, nullable=False, server_default="0.40"),
        sa.Column("min_cash_pct",     sa.Float, nullable=False, server_default="0.02"),
        sa.Column("max_cash_pct",     sa.Float, nullable=False, server_default="0.10"),
        sa.Column("excluded_tickers", postgresql.ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("esg_filter",       sa.Boolean, nullable=False, server_default="false"),
        sa.Column("updated_at",       sa.DateTime, server_default=sa.text("NOW()")),
    )

    # ── score_runs ─────────────────────────────────────────────────────────
    op.create_table(
        "score_runs",
        sa.Column("id",            sa.String, primary_key=True),
        sa.Column("portfolio_id",  sa.String, sa.ForeignKey("portfolios.id"), nullable=False),
        sa.Column("run_date",      sa.DateTime, nullable=False),
        sa.Column("frequency",     sa.String, nullable=False, server_default="monthly"),
        sa.Column("status",        sa.String, nullable=False, server_default="pending"),
        sa.Column("model_version", sa.String),
        sa.Column("error_log",     sa.Text),
        sa.Column("created_at",    sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_score_runs_portfolio_id", "score_runs", ["portfolio_id"])
    op.create_index("ix_score_runs_run_date",     "score_runs", ["run_date"])

    # ── scores ─────────────────────────────────────────────────────────────
    op.create_table(
        "scores",
        sa.Column("id",                       sa.String, primary_key=True),
        sa.Column("run_id",                   sa.String, sa.ForeignKey("score_runs.id"), nullable=False),
        sa.Column("ticker",                   sa.String(10), nullable=False),
        sa.Column("technical_ml_score",       sa.Float),
        sa.Column("fundamental_ml_score",     sa.Float),
        sa.Column("entropy_ml_score",         sa.Float),
        sa.Column("llm_score",                sa.Float),
        sa.Column("llm_provider",             sa.String, nullable=False, server_default="claude"),
        sa.Column("llm_reasoning_json",       postgresql.JSONB),
        sa.Column("technical_score",          sa.Float),
        sa.Column("fundamental_score",        sa.Float),
        sa.Column("entropy_score",            sa.Float),
        sa.Column("combined_score",           sa.Float),
        sa.Column("w_technical",              sa.Float, nullable=False, server_default="1.0"),
        sa.Column("w_fundamental",            sa.Float, nullable=False, server_default="0.15"),
        sa.Column("w_entropy",                sa.Float, nullable=False, server_default="0.70"),
        sa.Column("forward_return_forecast",  sa.Float),
        sa.Column("created_at",               sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
        sa.UniqueConstraint("run_id", "ticker"),
    )
    op.create_index("ix_scores_ticker", "scores", ["ticker"])
    op.create_index("ix_scores_run_id", "scores", ["run_id"])

    # ── optimization_jobs ──────────────────────────────────────────────────
    op.create_table(
        "optimization_jobs",
        sa.Column("id",             sa.String, primary_key=True),
        sa.Column("portfolio_id",   sa.String, sa.ForeignKey("portfolios.id"), nullable=False),
        sa.Column("run_id",         sa.String, sa.ForeignKey("score_runs.id")),
        sa.Column("optimizer_type", sa.String, nullable=False),
        sa.Column("status",         sa.String, nullable=False, server_default="pending"),
        sa.Column("settings_json",  postgresql.JSONB),
        sa.Column("result_json",    postgresql.JSONB),
        sa.Column("created_at",     sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
    )

    # ── rebalance_proposals ────────────────────────────────────────────────
    op.create_table(
        "rebalance_proposals",
        sa.Column("id",                    sa.String, primary_key=True),
        sa.Column("portfolio_id",          sa.String, sa.ForeignKey("portfolios.id"), nullable=False),
        sa.Column("optimization_job_id",   sa.String, sa.ForeignKey("optimization_jobs.id")),
        sa.Column("status",                sa.String, nullable=False, server_default="pending"),
        sa.Column("proposed_weights_json", postgresql.JSONB, nullable=False),
        sa.Column("rationale_json",        postgresql.JSONB),
        sa.Column("estimated_turnover",    sa.Float),
        sa.Column("estimated_cost",        sa.Float),
        sa.Column("created_at",            sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
    )

    # ── rebalance_decisions ────────────────────────────────────────────────
    op.create_table(
        "rebalance_decisions",
        sa.Column("id",                    sa.String, primary_key=True),
        sa.Column("proposal_id",           sa.String, sa.ForeignKey("rebalance_proposals.id"), nullable=False),
        sa.Column("decision",              sa.String, nullable=False),
        sa.Column("modified_weights_json", postgresql.JSONB),
        sa.Column("reason",                sa.Text),
        sa.Column("decided_at",            sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
    )

    # ── trades ─────────────────────────────────────────────────────────────
    op.create_table(
        "trades",
        sa.Column("id",              sa.String, primary_key=True),
        sa.Column("proposal_id",     sa.String, sa.ForeignKey("rebalance_proposals.id"), nullable=False),
        sa.Column("ticker",          sa.String(10), nullable=False),
        sa.Column("action",          sa.String(4), nullable=False),
        sa.Column("shares",          sa.Float),
        sa.Column("estimated_price", sa.Float),
        sa.Column("estimated_value", sa.Float),
        sa.Column("created_at",      sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
    )

    # ── benchmark_facts (seed-only, locked) ────────────────────────────────
    op.create_table(
        "benchmark_facts",
        sa.Column("id",                sa.String, primary_key=True),
        sa.Column("strategy",          sa.String, nullable=False),
        sa.Column("frequency",         sa.String, nullable=False),
        sa.Column("ml_weight",         sa.Float, nullable=False),
        sa.Column("llm_weight",        sa.Float, nullable=False),
        sa.Column("sharpe_ratio",      sa.Float, nullable=False),
        sa.Column("average_return",    sa.Float, nullable=False),
        sa.Column("volatility",        sa.Float, nullable=False),
        sa.Column("cumulative_return", sa.Float, nullable=False),
        sa.Column("notes",             sa.Text),
        sa.Column("source",            sa.String, nullable=False, server_default="Table 1, Cohen et al., Entropy 2025, 27, 550"),
        sa.UniqueConstraint("strategy", "frequency"),
    )

    # ── Seed benchmark_facts ───────────────────────────────────────────────
    # LOCKED VALUES — Table 1, Cohen et al. (2025). DO NOT MODIFY.
    import uuid
    op.bulk_insert(
        sa.table(
            "benchmark_facts",
            sa.column("id"),          sa.column("strategy"),    sa.column("frequency"),
            sa.column("ml_weight"),   sa.column("llm_weight"),  sa.column("sharpe_ratio"),
            sa.column("average_return"), sa.column("volatility"), sa.column("cumulative_return"),
            sa.column("notes"),       sa.column("source"),
        ),
        [
            {"id": str(uuid.uuid4()), "strategy": "technical",   "frequency": "monthly",   "ml_weight": 1.00, "llm_weight": 0.00, "sharpe_ratio": 0.6934, "average_return": 0.0750, "volatility": 0.1082, "cumulative_return": 19.7771, "notes": "Best cumulative return (1977.71%); pure ML weighting",                     "source": "Table 1, Cohen et al., Entropy 2025, 27, 550"},
            {"id": str(uuid.uuid4()), "strategy": "entropy",     "frequency": "monthly",   "ml_weight": 0.70, "llm_weight": 0.30, "sharpe_ratio": 0.4207, "average_return": 0.0523, "volatility": 0.1244, "cumulative_return":  7.0052, "notes": "Balanced blend; semantic context disambiguates entropy signals",          "source": "Table 1, Cohen et al., Entropy 2025, 27, 550"},
            {"id": str(uuid.uuid4()), "strategy": "fundamental", "frequency": "monthly",   "ml_weight": 0.15, "llm_weight": 0.85, "sharpe_ratio": 0.5001, "average_return": 0.0432, "volatility": 0.0863, "cumulative_return":  5.7840, "notes": "Lowest volatility (8.63% monthly); LLM reads 10-K/10-Q/earnings calls", "source": "Table 1, Cohen et al., Entropy 2025, 27, 550"},
            {"id": str(uuid.uuid4()), "strategy": "technical",   "frequency": "quarterly", "ml_weight": 0.45, "llm_weight": 0.55, "sharpe_ratio": 1.2967, "average_return": 0.2499, "volatility": 0.1927, "cumulative_return":  5.7337, "notes": "Best Sharpe ratio (1.2967); semantic blend at quarterly horizon",         "source": "Table 1, Cohen et al., Entropy 2025, 27, 550"},
            {"id": str(uuid.uuid4()), "strategy": "entropy",     "frequency": "quarterly", "ml_weight": 0.40, "llm_weight": 0.60, "sharpe_ratio": 0.6048, "average_return": 0.2025, "volatility": 0.3348, "cumulative_return":  5.3436, "notes": "Slight semantic lean; high volatility at quarterly horizon",              "source": "Table 1, Cohen et al., Entropy 2025, 27, 550"},
            {"id": str(uuid.uuid4()), "strategy": "fundamental", "frequency": "quarterly", "ml_weight": 0.00, "llm_weight": 1.00, "sharpe_ratio": 0.4899, "average_return": 0.1471, "volatility": 0.3002, "cumulative_return":  3.2612, "notes": "Pure semantic (w=0); lowest cumulative return in study",                  "source": "Table 1, Cohen et al., Entropy 2025, 27, 550"},
        ],
    )


def downgrade() -> None:
    op.drop_table("trades")
    op.drop_table("rebalance_decisions")
    op.drop_table("rebalance_proposals")
    op.drop_table("optimization_jobs")
    op.drop_table("scores")
    op.drop_table("score_runs")
    op.drop_table("constraints")
    op.drop_table("holdings")
    op.drop_table("portfolios")
    op.drop_table("users")
    op.drop_table("benchmark_facts")
