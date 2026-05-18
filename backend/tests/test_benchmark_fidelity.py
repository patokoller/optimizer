"""
tests/test_benchmark_fidelity.py
─────────────────────────────────────────────────────────────────────────────
Fidelity guard for locked benchmark facts.
Source: Table 1, Cohen et al., Entropy 2025, 27, 550.

Run in CI to ensure no code change silently alters the paper's locked values.
Also tests the scoring formula and normalization.

Usage:
    pytest tests/test_benchmark_fidelity.py -v
    python tests/test_benchmark_fidelity.py        # standalone
"""
import sys
import os
import math

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Locked benchmark facts ────────────────────────────────────────────────
# Source: Table 1, Cohen et al., Entropy 2025, 27, 550
# DO NOT MODIFY — any change here must be accompanied by a paper citation update.
LOCKED_BENCHMARKS = [
    # strategy,      frequency,    ml_w,  llm_w,  sharpe,  avg_ret,  vol,     cum_ret
    ("technical",   "monthly",    1.00,  0.00,   0.6934,  0.0750,   0.1082,  19.7771),
    ("entropy",     "monthly",    0.70,  0.30,   0.4207,  0.0523,   0.1244,   7.0052),
    ("fundamental", "monthly",    0.15,  0.85,   0.5001,  0.0432,   0.0863,   5.7840),
    ("technical",   "quarterly",  0.45,  0.55,   1.2967,  0.2499,   0.1927,   5.7337),
    ("entropy",     "quarterly",  0.40,  0.60,   0.6048,  0.2025,   0.3348,   5.3436),
    ("fundamental", "quarterly",  0.00,  1.00,   0.4899,  0.1471,   0.3002,   3.2612),
]

# Convenience lookup
BENCHMARKS_BY_KEY = {
    (s, f): {"ml_weight": ml, "llm_weight": llm, "sharpe": sh,
             "avg_ret": ar, "vol": v, "cum_ret": cr}
    for s, f, ml, llm, sh, ar, v, cr in LOCKED_BENCHMARKS
}


def approx(a: float, b: float, tol: float = 1e-4) -> bool:
    """True if a and b agree within tol."""
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


# ────────────────────────────────────────────────────────────────────────────
# F-1: All six benchmark rows present and complete
# ────────────────────────────────────────────────────────────────────────────
def test_f1_all_six_rows_present():
    """F-1: Exactly six strategy-frequency pairs must be present."""
    assert len(LOCKED_BENCHMARKS) == 6, (
        f"Expected 6 benchmark rows, got {len(LOCKED_BENCHMARKS)}"
    )
    expected_keys = {
        ("technical",   "monthly"),
        ("entropy",     "monthly"),
        ("fundamental", "monthly"),
        ("technical",   "quarterly"),
        ("entropy",     "quarterly"),
        ("fundamental", "quarterly"),
    }
    actual_keys = {(s, f) for s, f, *_ in LOCKED_BENCHMARKS}
    assert actual_keys == expected_keys, f"Missing keys: {expected_keys - actual_keys}"


# ────────────────────────────────────────────────────────────────────────────
# F-2: Technical monthly — best cumulative return, pure ML
# ────────────────────────────────────────────────────────────────────────────
def test_f2_technical_monthly():
    """F-2: Technical monthly must have ml_weight=1.00 and cumulative_return≈1977.71%."""
    b = BENCHMARKS_BY_KEY[("technical", "monthly")]
    assert approx(b["ml_weight"], 1.00), f"ml_weight={b['ml_weight']}, expected 1.00"
    assert approx(b["llm_weight"], 0.00), f"llm_weight={b['llm_weight']}, expected 0.00"
    assert approx(b["cum_ret"], 19.7771), (
        f"cumulative_return={b['cum_ret']}, expected 19.7771 (1977.71%)"
    )
    assert approx(b["sharpe"], 0.6934), f"sharpe={b['sharpe']}, expected 0.6934"


# ────────────────────────────────────────────────────────────────────────────
# F-3: Best Sharpe = technical quarterly (NOT technical monthly)
# ────────────────────────────────────────────────────────────────────────────
def test_f3_best_sharpe_is_technical_quarterly():
    """
    F-3: Technical quarterly has the highest Sharpe ratio (1.2967).
    This must NOT be confused with the best cumulative return (technical monthly).
    """
    all_sharpes = {(s, f): vals["sharpe"] for (s, f), vals in BENCHMARKS_BY_KEY.items()}
    best = max(all_sharpes, key=all_sharpes.__getitem__)
    assert best == ("technical", "quarterly"), (
        f"Best Sharpe should be (technical, quarterly), got {best}"
    )
    assert approx(all_sharpes[("technical", "quarterly")], 1.2967), (
        f"Expected Sharpe 1.2967, got {all_sharpes[('technical', 'quarterly')]}"
    )


# ────────────────────────────────────────────────────────────────────────────
# F-4: Fundamental quarterly — pure semantic (ml_weight=0.00)
# ────────────────────────────────────────────────────────────────────────────
def test_f4_fundamental_quarterly_pure_semantic():
    """F-4: Fundamental quarterly must have ml_weight=0.00 (pure LLM)."""
    b = BENCHMARKS_BY_KEY[("fundamental", "quarterly")]
    assert approx(b["ml_weight"],  0.00), f"ml_weight={b['ml_weight']}, expected 0.00"
    assert approx(b["llm_weight"], 1.00), f"llm_weight={b['llm_weight']}, expected 1.00"


# ────────────────────────────────────────────────────────────────────────────
# F-5: ml_weight + llm_weight = 1.00 for all rows
# ────────────────────────────────────────────────────────────────────────────
def test_f5_weights_sum_to_one():
    """F-5: ml_weight + llm_weight must equal 1.00 for every benchmark."""
    for s, f, ml, llm, *_ in LOCKED_BENCHMARKS:
        total = ml + llm
        assert approx(total, 1.00), (
            f"({s},{f}): ml_weight={ml} + llm_weight={llm} = {total} ≠ 1.00"
        )


# ────────────────────────────────────────────────────────────────────────────
# F-6: Entropy uses balanced blending at both horizons
# ────────────────────────────────────────────────────────────────────────────
def test_f6_entropy_balanced_blending():
    """F-6: Entropy must have 0 < ml_weight < 1 at both horizons (balanced blend)."""
    for freq in ["monthly", "quarterly"]:
        b = BENCHMARKS_BY_KEY[("entropy", freq)]
        ml = b["ml_weight"]
        assert 0 < ml < 1, (
            f"Entropy {freq}: ml_weight={ml} must be between 0 and 1 exclusive"
        )


# ────────────────────────────────────────────────────────────────────────────
# I-1: Technical monthly < technical quarterly on Sharpe
# ────────────────────────────────────────────────────────────────────────────
def test_i1_monthly_technical_lower_sharpe_than_quarterly():
    """
    I-1: Technical monthly cumulative return > quarterly, but monthly Sharpe < quarterly.
    The app must communicate this distinction correctly.
    """
    tech_m = BENCHMARKS_BY_KEY[("technical", "monthly")]
    tech_q = BENCHMARKS_BY_KEY[("technical", "quarterly")]
    assert tech_m["cum_ret"] > tech_q["cum_ret"], (
        "Technical monthly cumulative return should exceed quarterly"
    )
    assert tech_m["sharpe"] < tech_q["sharpe"], (
        "Technical monthly Sharpe should be LOWER than quarterly"
    )


# ────────────────────────────────────────────────────────────────────────────
# Scoring formula — Equation (2), Cohen et al.
# ────────────────────────────────────────────────────────────────────────────
def test_combined_score_formula():
    """CombinedScore = w*ML + (1-w)*LLM, clipped to [0,1]."""
    from app.ml.scoring import combined_score

    # Technical monthly: w=1.00 → pure ML score
    result = combined_score(0.8, 0.6, "technical", "monthly", llm_failed=False)
    expected = 1.00 * 0.8 + 0.00 * 0.6
    assert approx(result, expected), f"technical/monthly: {result} != {expected}"

    # Fundamental monthly: w=0.15
    result = combined_score(0.8, 0.6, "fundamental", "monthly", llm_failed=False)
    expected = 0.15 * 0.8 + 0.85 * 0.6
    assert approx(result, expected), f"fundamental/monthly: {result} != {expected}"

    # Entropy monthly: w=0.70
    result = combined_score(0.8, 0.6, "entropy", "monthly", llm_failed=False)
    expected = 0.70 * 0.8 + 0.30 * 0.6
    assert approx(result, expected), f"entropy/monthly: {result} != {expected}"


def test_llm_fallback_uses_pure_ml():
    """When llm_failed=True, combined_score must return the ML score directly (w=1.0)."""
    from app.ml.scoring import combined_score

    ml_score = 0.75
    llm_score = 0.30  # Would normally reduce the score

    for strategy in ["technical", "fundamental", "entropy"]:
        for freq in ["monthly", "quarterly"]:
            result = combined_score(ml_score, llm_score, strategy, freq, llm_failed=True)
            assert approx(result, ml_score), (
                f"{strategy}/{freq}: LLM fallback gave {result}, expected {ml_score}"
            )


def test_scores_clipped_to_unit_interval():
    """Scores must always be clipped to [0, 1], regardless of raw ML predictions."""
    from app.ml.scoring import combined_score

    # Very high ML score
    result = combined_score(1.5, 0.5, "technical", "monthly", llm_failed=False)
    assert 0.0 <= result <= 1.0, f"Score {result} out of [0,1]"

    # Very low ML score
    result = combined_score(-0.5, 0.5, "entropy", "quarterly", llm_failed=False)
    assert 0.0 <= result <= 1.0, f"Score {result} out of [0,1]"


def test_select_top_n_returns_exactly_n():
    """Top-N selection must return exactly n tickers by score descending."""
    from app.ml.scoring import select_top_n

    scores = {f"TICK{i:02d}": i * 0.05 for i in range(20)}
    top10 = select_top_n(scores, n=10)
    assert len(top10) == 10, f"Expected 10, got {len(top10)}"
    # Verify ordering
    for i in range(len(top10) - 1):
        assert scores[top10[i]] >= scores[top10[i + 1]], "Top-N not sorted descending"


# ────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_f1_all_six_rows_present,
        test_f2_technical_monthly,
        test_f3_best_sharpe_is_technical_quarterly,
        test_f4_fundamental_quarterly_pure_semantic,
        test_f5_weights_sum_to_one,
        test_f6_entropy_balanced_blending,
        test_i1_monthly_technical_lower_sharpe_than_quarterly,
    ]

    print("=== Benchmark Fidelity Tests ===")
    print("Source: Table 1, Cohen et al., Entropy 2025, 27, 550\n")

    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ⚠ {t.__name__} (import error — run from repo root): {e}")

    print(f"\n{passed}/{passed+failed} passed")

    # Formula tests require app module — skip in standalone mode
    if failed == 0:
        print("\n✅ All benchmark fidelity tests passed.")
    else:
        print(f"\n❌ {failed} test(s) failed — locked benchmark values may be corrupted.")
        sys.exit(1)
