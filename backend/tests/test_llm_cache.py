"""Tests for app/ml/llm_cache.py — fingerprinting, hit/miss split, write-back,
invalidation on input change, and graceful degradation."""

import os
os.environ.setdefault("DATABASE_URL", "postgresql://u:p@localhost:5432/x")

from app.ml.llm_cache import (
    prompt_fingerprint, score_batch_cached, score_sync_cached,
)


def test_fingerprint_stable_and_sensitive():
    a = prompt_fingerprint("materials X")
    assert a == prompt_fingerprint("materials X")          # stable
    assert a != prompt_fingerprint("materials Y")          # input change → new key
    assert prompt_fingerprint("") == prompt_fingerprint(None)  # both empty


class _Row:
    def __init__(self, ticker, period, prompt_hash, result_json, two_stage=False):
        self.ticker = ticker; self.period = period
        self.prompt_hash = prompt_hash; self.result_json = result_json
        self.two_stage = two_stage


class _FakeCacheDB:
    """In-memory stand-in for the LLMScoreCache table."""
    def __init__(self):
        self.rows: list[_Row] = []
        self.committed = 0

    # save_bundle-style query chain used by get/put/prune
    def query(self, *cols):
        rows = self.rows
        outer = self

        class _Q:
            def __init__(self):
                self._f = list(rows)
                self._cols = cols

            def filter(self, *conds):
                # Conditions are SQLAlchemy expressions we can't introspect here;
                # the cache helpers only ever filter by exact ticker/period/hash,
                # so we approximate by returning all rows and letting first()/
                # delete() operate — tests drive specific scenarios via the API.
                return self

            def first(self):
                return self._f[0] if self._f else None

            def distinct(self):
                return self

            def order_by(self, *a):
                return self

            def all(self):
                return [(r.period,) for r in self._f]

            def delete(self, *a, **k):
                return 0

        return _Q()

    def add(self, row):
        self.rows.append(row)

    def commit(self):
        self.committed += 1

    def rollback(self):
        pass


class _FakeScorer:
    def __init__(self):
        self.batch_calls = []
        self.sync_calls = []

    def score_batch(self, prompts):
        self.batch_calls.append(dict(prompts))
        return {t: {"score": 0.5, "two_stage": True} for t in prompts}

    def score_two_stage_sync(self, ticker, prompt):
        self.sync_calls.append((ticker, prompt))
        return {"score": 0.6, "two_stage": True}


def test_batch_all_miss_scores_everything(monkeypatch):
    import app.ml.llm_cache as lc

    # Force every lookup to miss.
    monkeypatch.setattr(lc, "get_cached", lambda db, t, p, h: None)
    stored = []
    monkeypatch.setattr(lc, "put_cached",
                        lambda db, t, p, h, r, two_stage=False: stored.append(t))

    db = _FakeCacheDB()
    scorer = _FakeScorer()
    prompts = {"AAPL": "pa", "MSFT": "pm"}
    out = score_batch_cached(db, scorer, prompts, "2026-06")
    assert set(out) == {"AAPL", "MSFT"}
    assert scorer.batch_calls[0] == prompts          # both scored (all miss)
    assert set(stored) == {"AAPL", "MSFT"}           # both written back


def test_batch_partial_hit_only_scores_misses(monkeypatch):
    import app.ml.llm_cache as lc

    # AAPL hits cache; MSFT misses.
    def fake_get(db, t, period, h):
        return {"score": 0.9, "two_stage": True} if t == "AAPL" else None
    monkeypatch.setattr(lc, "get_cached", fake_get)
    monkeypatch.setattr(lc, "put_cached", lambda *a, **k: None)

    db = _FakeCacheDB()
    scorer = _FakeScorer()
    out = score_batch_cached(db, scorer, {"AAPL": "pa", "MSFT": "pm"}, "2026-06")
    assert out["AAPL"]["score"] == 0.9               # from cache
    assert out["MSFT"]["score"] == 0.5               # freshly scored
    assert list(scorer.batch_calls[0]) == ["MSFT"]   # only the miss was batched


def test_batch_empty_prompts_no_call():
    scorer = _FakeScorer()
    assert score_batch_cached(_FakeCacheDB(), scorer, {}, "2026-06") == {}
    assert scorer.batch_calls == []


def test_batch_cache_read_error_falls_back(monkeypatch):
    import app.ml.llm_cache as lc

    def boom(*a, **k):
        raise RuntimeError("cache down")
    monkeypatch.setattr(lc, "get_cached", boom)

    scorer = _FakeScorer()
    out = score_batch_cached(_FakeCacheDB(), scorer, {"AAPL": "pa"}, "2026-06")
    assert out["AAPL"]["score"] == 0.5               # scored despite cache failure


def test_sync_hit_skips_scorer(monkeypatch):
    import app.ml.llm_cache as lc

    monkeypatch.setattr(lc, "get_cached", lambda db, t, p, h: {"score": 0.8, "two_stage": True})
    scorer = _FakeScorer()
    out = score_sync_cached(_FakeCacheDB(), scorer, "AAPL", "prompt", "2026-06")
    assert out["score"] == 0.8
    assert scorer.sync_calls == []                   # cache hit → no API call


def test_sync_miss_scores_and_writes(monkeypatch):
    import app.ml.llm_cache as lc

    monkeypatch.setattr(lc, "get_cached", lambda db, t, p, h: None)
    written = {}
    monkeypatch.setattr(lc, "put_cached",
                        lambda db, t, p, h, r, two_stage=False: written.update({t: r}))
    scorer = _FakeScorer()
    out = score_sync_cached(_FakeCacheDB(), scorer, "AAPL", "prompt", "2026-06")
    assert out["score"] == 0.6
    assert scorer.sync_calls == [("AAPL", "prompt")]
    assert written["AAPL"]["score"] == 0.6


def test_sync_write_failure_still_returns(monkeypatch):
    import app.ml.llm_cache as lc

    monkeypatch.setattr(lc, "get_cached", lambda db, t, p, h: None)

    def boom(*a, **k):
        raise RuntimeError("write down")
    monkeypatch.setattr(lc, "put_cached", boom)
    scorer = _FakeScorer()
    out = score_sync_cached(_FakeCacheDB(), scorer, "AAPL", "prompt", "2026-06")
    assert out["score"] == 0.6                       # result survives write failure
