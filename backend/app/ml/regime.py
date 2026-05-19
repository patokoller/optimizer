"""
app/ml/regime.py
────────────────────────────────────────────────────────────────────────────
Market regime classifier using FRED macro inputs.

Regime taxonomy (8 regimes covering the full macro cycle):
  1. Risk-On Momentum        — Low VIX, normal/steep curve, moderate rates
  2. AI / Growth Expansion   — Low VIX, falling curve, high tech breadth
  3. Inflation Shock         — Rising VIX, high CPI, rising rates
  4. Defensive Rotation      — Rising VIX, flattening/inverted curve
  5. Macro Uncertainty       — Elevated VIX, mixed signals, low conviction
  6. Liquidity Expansion     — Falling rates, steep curve, falling VIX
  7. High Volatility Regime  — VIX > 30, any curve
  8. Neutral / Mixed         — No dominant signal

For each regime, we compute:
  - Factor weight adjustments (relative to paper optimal weights)
  - Dominant factor (what is driving returns in this regime)
  - Transition risk (likelihood of regime change)
"""
import logging
from typing import Any

logger = logging.getLogger("regime")

# ── Regime definitions ────────────────────────────────────────────────────
# Each entry: (label, description, dominant_factor, transition_risk,
#              factor_weight_adj, regime_confidence_modifier)
#
# factor_weight_adj multiplies the paper's optimal ML weights per strategy.
# Values > 1.0 = amplify that strategy's contribution
# Values < 1.0 = dampen that strategy's contribution

REGIMES = {
    "risk_on_momentum": {
        "label":         "Risk-On Momentum",
        "description":   "Low volatility, positive breadth, normal yield curve. Momentum and technical signals dominate.",
        "dominant_factor": "Technical Momentum",
        "factor_weight_adj": {
            "technical":   1.20,   # amplify technical (momentum works in risk-on)
            "fundamental": 0.85,   # slight dampening (valuation less important)
            "entropy":     1.00,
        },
        "transition_risk": "low",
        "confidence_boost": 0.10,  # add to regime_confidence
    },
    "ai_growth_expansion": {
        "label":         "AI / Growth Expansion",
        "description":   "Low VIX, steep curve normalising, technology sector leadership. Growth fundamentals rewarded.",
        "dominant_factor": "Quality Growth",
        "factor_weight_adj": {
            "technical":   1.10,
            "fundamental": 1.15,   # fundamentals matter more in growth expansion
            "entropy":     0.90,
        },
        "transition_risk": "medium",
        "confidence_boost": 0.05,
    },
    "inflation_shock": {
        "label":         "Inflation Shock",
        "description":   "High CPI, rising rates, Fed tightening. Value and defensive factors rewarded; growth penalised.",
        "dominant_factor": "Inflation Sensitivity",
        "factor_weight_adj": {
            "technical":   0.80,   # momentum breaks down in inflation shocks
            "fundamental": 1.20,   # earnings quality matters more
            "entropy":     1.10,   # entropy/stability signals valuable
        },
        "transition_risk": "high",
        "confidence_boost": -0.10,
    },
    "defensive_rotation": {
        "label":         "Defensive Rotation",
        "description":   "Rising volatility, flattening/inverted curve. Defensive quality and low-beta preferred.",
        "dominant_factor": "Defensive Quality",
        "factor_weight_adj": {
            "technical":   0.75,
            "fundamental": 1.25,
            "entropy":     1.15,
        },
        "transition_risk": "medium",
        "confidence_boost": -0.05,
    },
    "macro_uncertainty": {
        "label":         "Macro Uncertainty",
        "description":   "Elevated and rising VIX, mixed macro signals. Low conviction environment; widen uncertainty bands.",
        "dominant_factor": "Mixed / Low Conviction",
        "factor_weight_adj": {
            "technical":   0.90,
            "fundamental": 1.00,
            "entropy":     1.10,
        },
        "transition_risk": "high",
        "confidence_boost": -0.15,
    },
    "liquidity_expansion": {
        "label":         "Liquidity Expansion",
        "description":   "Falling rates, steep yield curve, easing financial conditions. Risk assets broadly rewarded.",
        "dominant_factor": "Liquidity / Duration",
        "factor_weight_adj": {
            "technical":   1.15,
            "fundamental": 1.05,
            "entropy":     0.95,
        },
        "transition_risk": "low",
        "confidence_boost": 0.08,
    },
    "high_volatility": {
        "label":         "High Volatility Compression",
        "description":   "VIX above 30. Correlations spike, factor premia compress. Reduce all signal confidence.",
        "dominant_factor": "Vol / Macro Risk",
        "factor_weight_adj": {
            "technical":   0.70,
            "fundamental": 0.90,
            "entropy":     1.20,   # entropy/stability signals most reliable in high vol
        },
        "transition_risk": "high",
        "confidence_boost": -0.20,
    },
    "neutral": {
        "label":         "Neutral / Mixed",
        "description":   "No dominant macro signal. Paper optimal weights apply without adjustment.",
        "dominant_factor": "Balanced",
        "factor_weight_adj": {
            "technical":   1.00,
            "fundamental": 1.00,
            "entropy":     1.00,
        },
        "transition_risk": "medium",
        "confidence_boost": 0.0,
    },
}


def classify_regime(macro: dict) -> dict:
    """
    Classify current market regime from FRED macro snapshot.

    Args:
        macro: dict from FREDClient.get_macro_snapshot()

    Returns:
        {
            "regime_key":        str,
            "label":             str,
            "confidence":        float,   # 0-1
            "dominant_factor":   str,
            "factor_weight_adj": dict,
            "transition_risk":   str,
            "description":       str,
            "inputs_used":       dict,
        }
    """
    vix         = macro.get("vix", 20.0)
    vix_trend   = macro.get("vix_trend", "stable")
    yield_curve = macro.get("yield_curve", 0.5)
    curve_trend = macro.get("curve_trend", "stable")
    fed_funds   = macro.get("fed_funds", 5.0)
    cpi_yoy     = macro.get("cpi_yoy", 3.0)
    errors      = macro.get("errors", [])

    # ── Rule-based classification ─────────────────────────────────────────
    # Priority order: high-vol > inflation > defensive > liquidity > risk-on > growth > neutral

    regime_key = "neutral"
    base_confidence = 0.70

    if vix > 30:
        regime_key = "high_volatility"
        base_confidence = 0.80  # high vol is unambiguous

    elif cpi_yoy > 4.5 and fed_funds > 4.0 and vix_trend == "rising":
        regime_key = "inflation_shock"
        base_confidence = 0.75

    elif yield_curve < -0.20 and vix_trend in ("rising", "stable") and vix > 18:
        regime_key = "defensive_rotation"
        base_confidence = 0.70

    elif vix > 22 and vix_trend == "rising":
        regime_key = "macro_uncertainty"
        base_confidence = 0.65

    elif fed_funds < 3.5 and yield_curve > 0.5 and curve_trend == "steepening":
        regime_key = "liquidity_expansion"
        base_confidence = 0.72

    elif vix < 16 and yield_curve > 0.0 and cpi_yoy < 3.5:
        # Low-vol, positive curve, benign inflation
        if cpi_yoy < 2.5 and fed_funds < 4.0:
            regime_key = "ai_growth_expansion"
        else:
            regime_key = "risk_on_momentum"
        base_confidence = 0.75

    elif vix < 20 and vix_trend == "falling":
        regime_key = "risk_on_momentum"
        base_confidence = 0.68

    # Reduce confidence if FRED errors occurred
    if errors:
        base_confidence -= 0.05 * len(errors)
        base_confidence = max(0.40, base_confidence)

    regime = REGIMES[regime_key]

    # Final confidence with regime-specific boost
    final_confidence = min(0.95, max(0.30, base_confidence + regime["confidence_boost"]))

    result = {
        "regime_key":        regime_key,
        "label":             regime["label"],
        "description":       regime["description"],
        "confidence":        round(final_confidence, 3),
        "dominant_factor":   regime["dominant_factor"],
        "factor_weight_adj": regime["factor_weight_adj"],
        "transition_risk":   regime["transition_risk"],
        "inputs_used": {
            "vix":         vix,
            "vix_trend":   vix_trend,
            "yield_curve": yield_curve,
            "curve_trend": curve_trend,
            "fed_funds":   fed_funds,
            "cpi_yoy":     cpi_yoy,
        },
    }

    logger.info(
        f"Regime: {result['label']} | "
        f"Confidence: {result['confidence']:.2f} | "
        f"VIX={vix:.1f} | Curve={yield_curve:.2f} | "
        f"Fed={fed_funds:.2f} | CPI={cpi_yoy:.1f}%"
    )

    return result


def apply_regime_weight_adjustment(
    base_weights: dict[str, float],
    factor_weight_adj: dict[str, float],
) -> dict[str, float]:
    """
    Apply regime factor weight adjustments to the paper's optimal ML weights.

    The adjustment multiplies the paper's optimal weight, then re-normalises
    the total so it remains a valid blending coefficient.

    Args:
        base_weights:      {"technical": 1.00, "fundamental": 0.15, "entropy": 0.70}
        factor_weight_adj: {"technical": 1.20, "fundamental": 0.85, "entropy": 1.00}

    Returns:
        Adjusted weights dict, each value still in [0, 1]
    """
    adjusted = {
        k: min(1.0, max(0.0, base_weights.get(k, 0.5) * factor_weight_adj.get(k, 1.0)))
        for k in base_weights
    }
    return adjusted
