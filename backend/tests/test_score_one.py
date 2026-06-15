"""
Tests for the on-demand single-ticker scoring chain:
  - scoring.percentile_into            (rank a value into a saved distribution)
  - ModelBundle save/load/prune        (model persistence)
  - LLMScorer.score_two_stage_sync     (synchronous two-stage)
  - assemble_score_one / score_one     (blend + orchestration)

These use fakes throughout — no DB, network, or trained models required.
"""

import json
import os
import uuid
from datetime import datetime, timedelta

os.environ.setdefault("DATABASE_URL", "postgresql://u:p@localhost:5432/x")

import pytest

from app.ml.scoring import rank_normalize, percentile_into


# ── percentile_into ────────────────────────────────────────────────────────
def test_percentile_into_matches_rank_normalize_scale():
    universe = {f"T{i}": float(i) for i in range(20)}
    rn = rank_normalize(universe)
    for t, v in universe.items():
        others = [u for k, u in universe.items() if k != t]
        assert abs(percentile_into(others, v) - rn[t]) < 1e-9


def test_percentile_into_monotonic_and_bounded():
    ref = [0.1, 0.2, 0.5, 0.9, 5.0]
    vals = [percentile_into(ref, x) for x in [-1, 0.15, 0.5, 4.0, 99.0]]
    assert vals == sorted(vals)
    assert all(0.0 < v < 1.0 for v in vals)


def test_percentile_into_edges():
    assert percentile_into([], 0.7) == 0.5
    assert percentile_into([1, 2, 3], None) == 0.5
    assert abs(percentile_into([1.0, 1.0, 1.0], 1.0) - 0.5) < 1e-9
    assert 0.8 < percentile_into([0.1, 0.2, 0.3], 9.0) < 1.0  # high but not exactly 1.0


# ── ModelBundle ────────────────────────────────────────────────────────────
class _FakeModel:
    def __init__(self, trained=True):
        self._trained = trained

    def predict(self, tickers, *a):
        return {tickers[0]: {"raw_ensemble": 7.0}}


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def order_by(self, *a):
        return self

    def limit(self, n):
        return self

    def offset(self, n):
        return _FakeQuery(self._rows[n:])

    def all(self):
        return list(self._rows)


class _FakeDB:
    def __init__(self):
        self.rows = []

    def add(self, r):
        self.rows.append(r)

    def commit(self):
        pass

    def rollback(self):
        pass

    def delete(self, r):
        self.rows.remove(r)

    def query(self, _):
        return _FakeQuery(sorted(self.rows, key=lambda r: r.created_at, reverse=True))


def _install_fake_bundle_row(monkeypatch):
    import app.models as M

    ctr = {"n": 0}

    class _Col:
        def desc(self):
            return None

    class FakeBundleRow:
        created_at = _Col()

        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.id = str(uuid.uuid4())
            ctr["n"] += 1
            self.created_at = datetime.now() + timedelta(seconds=ctr["n"])

    monkeypatch.setattr(M, "ModelBundle", FakeBundleRow)


def test_bundle_save_load_roundtrip(monkeypatch):
    from app.ml import model_bundle as mb

    _install_fake_bundle_row(monkeypatch)
    db = _FakeDB()
    score_dicts = {
        "technical": {f"T{i}": {"raw_ensemble": float(i)} for i in range(5)},
        "fundamental": {f"T{i}": {"raw_ensemble": float(i * 2)} for i in range(5)},
    }
    models_by = {
        "technical": _FakeModel(),
        "fundamental": _FakeModel(),
        "entropy": _FakeModel(trained=False),  # untrained → skipped
    }
    bid = mb.save_bundle(
        db, run_id="r", run_type="discovery", rebalance_date=datetime.now(),
        frequency="monthly", universe=[f"T{i}" for i in range(5)],
        models_by_strategy=models_by, score_dicts_by_strategy=score_dicts,
    )
    assert bid
    loaded = mb.load_latest_bundle(db)
    assert set(loaded.strategies) == {"fundamental", "technical"}
    assert loaded.reference_raw("technical") == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_bundle_require_strategies_and_prune(monkeypatch):
    from app.ml import model_bundle as mb

    _install_fake_bundle_row(monkeypatch)
    db = _FakeDB()
    mb.save_bundle(
        db, run_id="r", run_type="discovery", rebalance_date=None, frequency="monthly",
        universe=["T0"], models_by_strategy={"technical": _FakeModel()},
        score_dicts_by_strategy={"technical": {"T0": {"raw_ensemble": 1.0}}},
    )
    assert mb.load_latest_bundle(db, require_strategies=["entropy"]) is None
    assert mb.load_latest_bundle(db, require_strategies=["technical"]) is not None
    for k in range(mb._MAX_BUNDLES + 3):
        mb.save_bundle(
            db, run_id=f"r{k}", run_type="discovery", rebalance_date=None, frequency="monthly",
            universe=["T0"], models_by_strategy={"technical": _FakeModel()},
            score_dicts_by_strategy={"technical": {"T0": {"raw_ensemble": 1.0}}},
        )
    assert len(db.rows) == mb._MAX_BUNDLES


def test_bundle_no_trained_models_returns_none(monkeypatch):
    from app.ml import model_bundle as mb

    _install_fake_bundle_row(monkeypatch)
    db = _FakeDB()
    r = mb.save_bundle(
        db, run_id="x", run_type="score", rebalance_date=None, frequency="monthly",
        universe=["T0"], models_by_strategy={"technical": _FakeModel(trained=False)},
        score_dicts_by_strategy={},
    )
    assert r is None and len(db.rows) == 0


# ── synchronous two-stage ──────────────────────────────────────────────────
class _FakeContent:
    def __init__(self, text):
        self.text = text


class _FakeResp:
    def __init__(self, text):
        self.content = [_FakeContent(text)]


class _FakeMessages:
    def __init__(self, outs):
        self.outs = outs
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return _FakeResp(self.outs[len(self.calls) - 1])


class _FakeClient:
    def __init__(self, outs):
        self.messages = _FakeMessages(outs)


def _scorer(outs):
    from app.ml.llm_scoring import LLMScorer

    s = LLMScorer.__new__(LLMScorer)
    s.model = "fake"
    s.client = _FakeClient(outs)
    return s


def _marker_prompt():
    from app.ml.llm_scoring import _CALIBRATION_MARKER

    return f"MATERIALS for AAPL...\n{_CALIBRATION_MARKER}\nrest"


def test_two_stage_sync_happy_path():
    fact = {"revenue_trajectory": "up", "peer_standing": "60th"}
    score = {"band_base": 0.55, "adjustments": [{"reason": "g", "delta": 0.04}],
             "score": 0.59, "key_positives": ["x"], "key_risks": ["y"], "confidence": "medium"}
    s = _scorer([json.dumps(fact), json.dumps(score)])
    r = s.score_two_stage_sync("AAPL", _marker_prompt())
    assert r["two_stage"] is True
    assert abs(r["score"] - 0.59) < 1e-9
    assert r["fact_sheet"]["peer_standing"] == "60th"
    assert len(s.client.messages.calls) == 2


def test_two_stage_sync_extraction_failure_falls_back():
    score = {"score": 0.5, "key_positives": [], "key_risks": [], "confidence": "low"}
    s = _scorer(["NOT JSON", json.dumps(score)])
    r = s.score_two_stage_sync("AAPL", _marker_prompt())
    assert r["two_stage"] is False
    assert len(s.client.messages.calls) == 2


def test_two_stage_sync_no_marker_single_call():
    score = {"score": 0.7, "key_positives": [], "key_risks": [], "confidence": "low"}
    s = _scorer([json.dumps(score)])
    r = s.score_two_stage_sync("AAPL", "plain prompt no marker")
    assert r["two_stage"] is False
    assert len(s.client.messages.calls) == 1


def test_two_stage_sync_no_client_returns_none():
    from app.ml.llm_scoring import LLMScorer

    s = LLMScorer.__new__(LLMScorer)
    s.model = "fake"
    s.client = None
    assert s.score_two_stage_sync("AAPL", _marker_prompt()) is None


# ── assemble_score_one ─────────────────────────────────────────────────────
def _bundle(strategies=("fundamental", "technical", "entropy")):
    from app.ml.model_bundle import LoadedBundle

    ref = {f"T{i}": float(i) for i in range(20)}
    return LoadedBundle(
        run_id="d1", run_type="discovery", rebalance_date=datetime(2026, 6, 15),
        frequency="monthly", universe=list(ref), strategies=list(strategies),
        models={}, raw_vectors={s: dict(ref) for s in strategies},
        lib_versions={}, created_at=datetime(2026, 6, 15, 16, 0),
    )


def test_assemble_blend_and_weights():
    from app.services.score_one import assemble_score_one

    raw = {"technical": 9.5, "fundamental": 18.0, "entropy": 1.0}
    llm = {"score": 0.62, "band_base": 0.55, "adjustments": [],
           "key_positives": ["a"], "key_risks": ["b"], "confidence": "high",
           "two_stage": True, "fact_sheet": {"peer_standing": "high"}}
    p = assemble_score_one(_bundle(), ticker="nvda", raw_ensemble_by_strategy=raw, llm_result=llm)
    # technical monthly w_ml=1.0 → combined == ml_percentile
    t = p["strategies"]["technical"]
    assert abs(t["combined"] - t["ml_percentile"]) < 1e-6
    # fundamental monthly w_ml=0.15 → 0.15*pct + 0.85*0.62
    f = p["strategies"]["fundamental"]
    assert abs(f["combined"] - (0.15 * f["ml_percentile"] + 0.85 * 0.62)) < 1e-3
    assert p["llm"]["two_stage"] is True
    assert p["overall_score"] is not None


def test_assemble_missing_strategy_degrades():
    from app.services.score_one import assemble_score_one

    raw = {"technical": 9.5, "fundamental": 18.0, "entropy": None}
    p = assemble_score_one(_bundle(), ticker="x", raw_ensemble_by_strategy=raw, llm_result=None)
    assert p["strategies"]["entropy"]["available"] is False
    assert p["strategies"]["technical"]["available"] is True
    assert p["overall_score"] is not None


def test_assemble_llm_failure_pure_ml():
    from app.services.score_one import assemble_score_one

    raw = {"technical": 9.5, "fundamental": 18.0, "entropy": 1.0}
    p = assemble_score_one(_bundle(), ticker="x", raw_ensemble_by_strategy=raw, llm_result=None)
    f = p["strategies"]["fundamental"]
    assert abs(f["combined"] - f["ml_percentile"]) < 1e-6  # w→1.0 on llm failure
    assert p["llm"]["available"] is False


def test_assemble_respects_bundle_strategy_subset():
    from app.services.score_one import assemble_score_one

    raw = {"technical": 9.5, "fundamental": 18.0, "entropy": 1.0}
    p = assemble_score_one(_bundle(strategies=("technical",)), ticker="x",
                           raw_ensemble_by_strategy=raw, llm_result=None)
    assert p["strategies"]["technical"]["available"] is True
    assert p["strategies"]["fundamental"]["available"] is False


# ── score_one orchestrator ─────────────────────────────────────────────────
class _NullCacheDB:
    """Minimal db whose cache queries always miss and writes no-op."""
    def query(self, *a):
        class _Q:
            def filter(self, *a): return self
            def first(self): return None
            def delete(self, *a, **k): return 0
        return _Q()
    def add(self, *a): pass
    def commit(self): pass
    def rollback(self): pass


def test_score_one_no_bundle(monkeypatch):
    import app.services.score_one as so

    monkeypatch.setattr(so, "load_latest_bundle", lambda db: None)
    out = so.score_one(_NullCacheDB(), "nvda", alpaca=object(), av=object(),
                       edgar=object(), llm_scorer=object())
    assert out["error"] == "no_model_bundle"


def test_score_one_happy_path(monkeypatch):
    import app.services.score_one as so

    monkeypatch.setattr(so, "load_latest_bundle", lambda db: _bundle())
    # bundle needs models for the orchestrator to predict
    b = _bundle()

    class FM:
        def predict(self, tickers, *a):
            return {tickers[0]: {"raw_ensemble": 7.0}}

    b.models = {"fundamental": FM(), "technical": FM(), "entropy": FM()}
    monkeypatch.setattr(so, "load_latest_bundle", lambda db: b)

    class DF:
        empty = False

    class A:
        def get_ohlcv(self, *a):
            return DF()

    class V:
        def get_fundamentals_batch(self, *a):
            return DF()

    class E:
        def get_filing_context(self, *a):
            return "FILING"

    class L:
        def build_prompt(self, **kw):
            return "p"

        def score_two_stage_sync(self, t, p):
            return {"score": 0.6, "band_base": 0.55, "adjustments": [],
                    "key_positives": ["p"], "key_risks": ["r"], "confidence": "medium",
                    "two_stage": True, "fact_sheet": {"x": 1}}

    out = so.score_one(_NullCacheDB(), "nvda", alpaca=A(), av=V(), edgar=E(), llm_scorer=L())
    assert out["ticker"] == "NVDA"
    assert out["overall_score"] is not None
    assert out["data_availability"] == {"prices": True, "fundamentals": True, "filings": True}
    assert out["llm"]["two_stage"] is True


def test_score_one_full_degradation(monkeypatch):
    import app.services.score_one as so

    b = _bundle()

    class Dead:
        def predict(self, *a):
            raise RuntimeError("no features")

    b.models = {"fundamental": Dead(), "technical": Dead(), "entropy": Dead()}
    monkeypatch.setattr(so, "load_latest_bundle", lambda db: b)

    class A:
        def get_ohlcv(self, *a):
            raise RuntimeError("down")

    class V:
        def get_fundamentals_batch(self, *a):
            raise RuntimeError("down")

    class E:
        def get_filing_context(self, *a):
            return ""

    class L:
        def build_prompt(self, **kw):
            return "p"

        def score_two_stage_sync(self, *a):
            return None

    out = so.score_one(_NullCacheDB(), "zzzz", alpaca=A(), av=V(), edgar=E(), llm_scorer=L())
    assert out["error"] == "no_usable_data"
    assert out["data_availability"]["prices"] is False
