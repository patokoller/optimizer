"""Decision-logic tests for the self-healing bundle maintenance, with the
freshness read and the training enqueue stubbed (no DB / no Celery)."""

import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.services.bundle_maintenance as bm
from app import models


def _patch(monkeypatch, status, started=[]):
    monkeypatch.setattr(bm, "bundle_status", lambda db, max_age_days=bm.BUNDLE_MAX_AGE_DAYS: dict(status))
    monkeypatch.setattr(bm, "_start_training", lambda db: started.append("run") or "run-1")
    monkeypatch.setattr(bm, "_expire_stale_runs", lambda db: 0)


def test_missing_bundle_kicks_training(monkeypatch):
    started = []
    _patch(monkeypatch, {"exists": False, "age_days": None, "fresh": False,
                         "refresh_in_progress": False, "refresh_run_id": None}, started)
    s = bm.ensure_bundle_fresh(object())
    assert s["refresh_started"] is True and started == ["run"]
    assert s["refresh_run_id"] == "run-1"


def test_fresh_bundle_no_training(monkeypatch):
    started = []
    _patch(monkeypatch, {"exists": True, "age_days": 2.0, "fresh": True,
                         "refresh_in_progress": False, "refresh_run_id": None}, started)
    s = bm.ensure_bundle_fresh(object())
    assert s["refresh_started"] is False and started == []


def test_stale_bundle_kicks_training(monkeypatch):
    started = []
    _patch(monkeypatch, {"exists": True, "age_days": 10.0, "fresh": False,
                         "refresh_in_progress": False, "refresh_run_id": None}, started)
    s = bm.ensure_bundle_fresh(object())
    assert s["refresh_started"] is True and started == ["run"]


def test_stale_but_already_running_no_double_enqueue(monkeypatch):
    started = []
    _patch(monkeypatch, {"exists": True, "age_days": 10.0, "fresh": False,
                         "refresh_in_progress": True, "refresh_run_id": "existing"}, started)
    s = bm.ensure_bundle_fresh(object())
    assert s["refresh_started"] is False and started == []


# ── Staleness recovery (real in-memory SQLite, DiscoveryRun table only) ───────
@pytest.fixture
def sqlite_db():
    eng = create_engine("sqlite:///:memory:")
    models.DiscoveryRun.__table__.create(eng)
    db = sessionmaker(bind=eng)()
    yield db
    db.close()


def _add_run(db, status, age_min):
    db.add(models.DiscoveryRun(
        id=str(uuid.uuid4()), status=status, run_date=datetime.utcnow(),
        created_at=datetime.utcnow() - timedelta(minutes=age_min)))
    db.commit()


def test_fresh_running_run_blocks(sqlite_db):
    _add_run(sqlite_db, models.RunStatus.running, age_min=5)
    assert bm._refresh_in_progress(sqlite_db) is not None  # genuinely training


def test_stale_running_run_ignored(sqlite_db):
    _add_run(sqlite_db, models.RunStatus.running, age_min=bm.STALE_RUN_MINUTES + 15)
    assert bm._refresh_in_progress(sqlite_db) is None  # dead run no longer blocks


def test_expire_stale_marks_failed(sqlite_db):
    _add_run(sqlite_db, models.RunStatus.running, age_min=bm.STALE_RUN_MINUTES + 15)
    assert bm._expire_stale_runs(sqlite_db) == 1
    assert sqlite_db.query(models.DiscoveryRun).first().status == models.RunStatus.failed


def test_expire_leaves_fresh_running_alone(sqlite_db):
    _add_run(sqlite_db, models.RunStatus.running, age_min=5)
    assert bm._expire_stale_runs(sqlite_db) == 0
    assert sqlite_db.query(models.DiscoveryRun).first().status == models.RunStatus.running
