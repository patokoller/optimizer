"""
tests/test_api.py — API integration tests using the CI PostgreSQL service.
"""
import io
import os
import pytest
from fastapi.testclient import TestClient

# Use the real DATABASE_URL from environment (set by CI)
# This avoids SQLite incompatibility with JSONB/ARRAY columns
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_portfolio")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

from app.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_csv() -> bytes:
    return b"ticker,shares,cost_basis,currency\nNVDA,50,480.20,USD\nMSFT,30,340.10,USD\nAAPL,80,168.50,USD\n"


@pytest.fixture(scope="module")
def uploaded_portfolio(sample_csv):
    resp = client.post(
        "/api/portfolio/upload",
        files={"file": ("holdings.csv", io.BytesIO(sample_csv), "text/csv")},
    )
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    return resp.json()


# ── Health ─────────────────────────────────────────────────────────────
def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── Portfolio ──────────────────────────────────────────────────────────
def test_portfolio_upload(sample_csv):
    resp = client.post(
        "/api/portfolio/upload",
        files={"file": ("h.csv", io.BytesIO(sample_csv), "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "id" in body
    assert len(body["holdings"]) == 3
    assert {h["ticker"] for h in body["holdings"]} == {"NVDA", "MSFT", "AAPL"}


def test_portfolio_upload_missing_column():
    bad = b"symbol,qty\nNVDA,50\n"
    resp = client.post(
        "/api/portfolio/upload",
        files={"file": ("bad.csv", io.BytesIO(bad), "text/csv")},
    )
    assert resp.status_code == 422


def test_get_portfolio(uploaded_portfolio):
    pid = uploaded_portfolio["id"]
    resp = client.get(f"/api/portfolio/{pid}")
    assert resp.status_code == 200
    assert resp.json()["id"] == pid


def test_get_portfolio_not_found():
    resp = client.get("/api/portfolio/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_update_constraints(uploaded_portfolio):
    pid = uploaded_portfolio["id"]
    resp = client.put(
        f"/api/portfolio/{pid}/constraints",
        json={
            "max_position_pct": 0.20,
            "sector_cap_pct": 0.40,
            "min_cash_pct": 0.02,
            "max_cash_pct": 0.10,
            "excluded_tickers": ["TSLA"],
            "esg_filter": False,
        },
    )
    assert resp.status_code == 200


# ── Backtest — locked benchmark values ────────────────────────────────
def test_backtest_returns_locked_benchmarks(uploaded_portfolio):
    resp = client.post(
        "/api/backtest/run",
        json={
            "portfolio_id": uploaded_portfolio["id"],
            "strategies": ["technical", "fundamental", "entropy"],
            "start_date": "2020-01-01",
            "end_date": "2025-01-01",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "Table 1, Cohen et al., Entropy 2025, 27, 550"

    results = {(r["strategy"], r["frequency"]): r for r in body["results"]}
    assert len(results) == 6

    # F-2: Technical monthly — locked values
    tm = results[("technical", "monthly")]
    assert abs(tm["cumulative_return"] - 19.7771) < 0.001
    assert abs(tm["ml_weight"] - 1.00) < 0.001
    assert abs(tm["sharpe_ratio"] - 0.6934) < 0.001

    # F-3: Technical quarterly — highest Sharpe
    tq = results[("technical", "quarterly")]
    assert abs(tq["sharpe_ratio"] - 1.2967) < 0.001

    # F-4: Fundamental quarterly — pure semantic
    fq = results[("fundamental", "quarterly")]
    assert abs(fq["ml_weight"] - 0.00) < 0.001
    assert abs(fq["llm_weight"] - 1.00) < 0.001


def test_backtest_series_unavailable(uploaded_portfolio):
    resp = client.post(
        "/api/backtest/run",
        json={"portfolio_id": uploaded_portfolio["id"], "strategies": ["technical"]},
    )
    body = resp.json()
    assert body["series_available"] is False


# ── Export ─────────────────────────────────────────────────────────────
def test_export_trades_not_found():
    resp = client.get("/api/export/trades/nonexistent-id?format=csv")
    assert resp.status_code == 404


# ── Rebalance weight validation ────────────────────────────────────────
def test_rebalance_modify_invalid_weights():
    resp = client.put(
        "/api/rebalance/fake-id/modify",
        json={"weights": {"NVDA": 0.5, "MSFT": 0.8}},
    )
    assert resp.status_code == 422
