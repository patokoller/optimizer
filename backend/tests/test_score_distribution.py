"""Test per-run score-distribution monitoring (#22)."""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.ml.validation import score_distribution


def test_healthy_spread_not_compressed():
    random.seed(1)
    d = score_distribution([random.random() for _ in range(98)])
    assert d["n"] == 98 and not d["compressed"] and d["std"] > 0.2


def test_bunched_scores_flagged_compressed():
    random.seed(2)
    d = score_distribution([0.5 + random.uniform(-0.03, 0.03) for _ in range(98)])
    assert d["compressed"] is True


def test_histogram_sums_to_n_and_ignores_none():
    d = score_distribution([None, 0.05, 0.15, 0.95, None, 0.5])
    assert d["n"] == 4 and sum(d["histogram_deciles"]) == 4


def test_empty_safe():
    assert score_distribution([]) == {"n": 0}
    assert score_distribution([None, None]) == {"n": 0}
