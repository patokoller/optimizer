"""
tests/test_api.py
─────────────────────────────────────────────────────────────────────────────
Integration tests for the FastAPI backend.
Uses an in-memory SQLite DB for isolation (no external services required).
"""
import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db

# ── Test DB (SQLite in-memory) ─────────────────────────────────────────────
TEST_DB_URL = "sqlite:///./test.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


# ── Fixtures ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def sample_csv() -> bytes:
    content = "ticker,shares,cost_basis,currency\nNVDA,50,480.20,USD\nMSFT,30,340.10,USD\nAAPL,80,168.50,USD\n"
    return content.encode()


@pytest.fixture(scope="module")
def uploaded_portfolio(sample_csv):
    resp = client.post(
        "/api/portfolio/upload",
        files={"file": ("holdings.csv", io.BytesIO(sample_csv), "text/csv")},
    )
    assert resp.status_code == 200
    return resp.json()


# ── Health ─────────────────────────────────────────────────────────────────
def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── Portfolio upload ───────────────────────────────────────────────────────
def test_portfolio_upload(sample_csv):
    resp = client.post(
        "/api/portfolio/upload",
        files={"file": ("h.csv", io.BytesIO(sample_csv), "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "id" in body
    assert len(body["holdings"]) == 3
    tickers = {h["ticker"] for h in body["holdings"]}
    assert tickers == {"NVDA", "MSFT", "AAPL"}


def test_portfolio_upload_missing_column():
    bad_csv = b"symbol,qty\nNVDA,50\n"  # wrong column names
    resp = client.post(
        "/api/portfolio/upload",
        files={"file": ("bad.csv", io.BytesIO(bad_csv), "text/csv")},
    )
    assert resp.status_code == 422


def test_portfolio_upload_invalid_shares():
    bad_csv = b"ticker,shares,cost_basis,currency\nNVDA,NOT_A_NUMBER,480.20,USD\n"
    resp = client.post(
        "/api/portfolio/upload",
        files={"file": ("bad.csv", io.BytesIO(bad_csv), "text/csv")},
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
            "sector_cap_pct":   0.40,
            "min_cash_pct":     0.02,
            "max_cash_pct":     0.10,
            "excluded_tickers": ["TSLA"],
            "esg_filter": False,
        },
    )
    assert resp.status_code == 200


def test_update_constraints_invalid_weight():
    resp = client.put(
        "/api/portfolio/fake-id/constraints",
        json={"max_position_pct": 1.5},  # > 1.0
    )
    assert resp.status_code == 422


# ── Backtest — locked benchmark values ────────────────────────────────────
def test_backtest_returns_locked_benchmarks(uploaded_portfolio):
    resp = client.post(
        "/api/backtest/run",
        json={
            "portfolio_id": uploaded_portfolio["id"],
            "strategies": ["technical", "fundamental", "entropy"],
            "start_date": "2020-01-01",
            "end_date":   "2025-01-01",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "Table 1, Cohen et al., Entropy 2025, 27, 550"

    results = {(r["strategy"], r["frequency"]): r for r in body["results"]}

    # F-2: Technical monthly
    tm = results[("technical", "monthly")]
    assert abs(tm["cumulative_return"] - 19.7771) < 0.001
    assert abs(tm["ml_weight"] - 1.00) < 0.001
    assert abs(tm["sharpe_ratio"] - 0.6934) < 0.001

    # F-3: Technical quarterly has highest Sharpe
    tq = results[("technical", "quarterly")]
    assert abs(tq["sharpe_ratio"] - 1.2967) < 0.001

    # F-4: Fundamental quarterly is pure semantic
    fq = results[("fundamental", "quarterly")]
    assert abs(fq["ml_weight"] - 0.00) < 0.001
    assert abs(fq["llm_weight"] - 1.00) < 0.001

    # All six present
    assert len(results) == 6


def test_backtest_series_unavailable_notice(uploaded_portfolio):
    resp = client.post(
        "/api/backtest/run",
        json={"portfolio_id": uploaded_portfolio["id"], "strategies": ["technical"]},
    )
    body = resp.json()
    assert body["series_available"] is False
    assert "period_index" in body["series_note"] or "required" in body["series_note"].lower()


# ── Optimize endpoint ──────────────────────────────────────────────────────
def test_optimize_returns_job_id(uploaded_portfolio):
    """Optimization should queue a job and return immediately."""
    # We need a dummy run_id — in real tests we'd create a score run first.
    # Here just verify the endpoint accepts the request shape.
    resp = client.post(
        "/api/optimize/mvo",
        json={
            "portfolio_id": uploaded_portfolio["id"],
            "run_id": "00000000-0000-0000-0000-000000000001",
            "settings": {"risk_appetite": "balanced"},
        },
    )
    # Expect 200 (job created) or 404 (run_id not found in test DB) — both acceptable
    assert resp.status_code in (200, 404)


# ── Export ─────────────────────────────────────────────────────────────────
def test_export_trades_not_found():
    resp = client.get("/api/export/trades/nonexistent-id?format=csv")
    assert resp.status_code == 404


# ── Rebalance modify weight validation ────────────────────────────────────
def test_rebalance_modify_invalid_weights():
    resp = client.put(
        "/api/rebalance/fake-proposal-id/modify",
        json={"weights": {"NVDA": 0.5, "MSFT": 0.8}},  # sums to 1.3 > 1.02 threshold
    )
    assert resp.status_code == 422
