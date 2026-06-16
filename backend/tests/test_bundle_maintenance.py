"""Decision-logic tests for the self-healing bundle maintenance, with the
freshness read and the training enqueue stubbed (no DB / no Celery)."""

import app.services.bundle_maintenance as bm


def _patch(monkeypatch, status, started=[]):
    monkeypatch.setattr(bm, "bundle_status", lambda db, max_age_days=bm.BUNDLE_MAX_AGE_DAYS: dict(status))
    monkeypatch.setattr(bm, "_start_training", lambda db: started.append("run") or "run-1")


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
