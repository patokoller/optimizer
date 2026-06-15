"""
app/ml/fundamental.py
─────────────────────────────────────────────────────────────────────────────
Fundamental ML scoring — Section 3.2.1, Cohen et al. (2025).

Ensemble composition (equal weight per strategy):
    Ridge Regression    30%
    XGBoost             30%
    Random Forest       20%
    MLP                 20%

Features (quarterly fundamentals from Alpha Vantage):
    revenue, operating_income, net_income,
    operating_margin, net_margin,
    revenue_growth_yoy, net_income_growth_yoy

No-lookahead: train on data available BEFORE rebalance_date only.
"""
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("xgboost not installed — XGB disabled in fundamental model")

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "revenue",
    "operating_income",
    "net_income",
    "operating_margin",
    "net_margin",
    "revenue_growth_yoy",
    "net_income_growth_yoy",
]

# Model weights per paper
MODEL_WEIGHTS = {"ridge": 0.30, "xgboost": 0.30, "rf": 0.20, "mlp": 0.20}


def _make_models() -> dict:
    models = {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=[0.1, 1.0, 10.0])),
        ]),
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", max_iter=500, random_state=42)),
        ]),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            objective="reg:squarederror",
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0, n_jobs=-1,
        )
    else:
        # Increase RF weight if XGB missing
        models["rf_xgb_fallback"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=300, max_depth=8, random_state=43, n_jobs=-1)),
        ])
    return models


def _add_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add YoY growth rates — no lookahead, uses only past 4 quarters."""
    df = df.sort_values("period_date")
    df["revenue_growth_yoy"] = df.groupby("ticker")["revenue"].pct_change(4)
    df["net_income_growth_yoy"] = df.groupby("ticker")["net_income"].pct_change(4)
    return df.dropna(subset=["revenue_growth_yoy", "net_income_growth_yoy"])


class FundamentalScorer:
    """
    Trains the fundamental ensemble on historical quarterly financials,
    then scores a list of tickers at a given rebalance date.

    Strict no-lookahead: only data with period_date < rebalance_date is used.
    """

    def __init__(self):
        self.models = _make_models()
        self._trained = False
        self._scaler = StandardScaler()

    def fit(
        self,
        fundamentals_df: pd.DataFrame,
        rebalance_date: datetime,
    ) -> "FundamentalScorer":
        """
        Train all ensemble members on data available before rebalance_date.

        fundamentals_df columns:
            ticker, period_date, revenue, operating_income, net_income,
            operating_margin, net_margin, forward_return (target, 1-quarter ahead)
        """
        df = fundamentals_df.copy()
        df["period_date"] = pd.to_datetime(df["period_date"])
        df = df[df["period_date"] < pd.Timestamp(rebalance_date)]
        df = _add_growth_features(df)
        df = df.dropna(subset=FEATURE_COLS + ["forward_return"])

        if len(df) < 30:
            logger.warning(f"Insufficient training samples for fundamental model ({len(df)} rows)")
            return self

        X = df[FEATURE_COLS].values.astype(float)
        y = df["forward_return"].values.astype(float)

        # Sanitize: replace inf, -inf, and NaN with 0 before scaling
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip extreme revenue values (in billions) to prevent scale issues
        X = np.clip(X, -1e12, 1e12)

        for name, model in self.models.items():
            try:
                model.fit(X, y)
                logger.info(f"Fundamental [{name}] trained on {len(X)} samples")
            except Exception as e:
                logger.error(f"Fundamental [{name}] training failed: {e}")

        self._trained = True
        return self

    def predict(
        self,
        tickers: list[str],
        current_fundamentals: pd.DataFrame,
    ) -> dict[str, dict]:
        """
        Predict fundamental ML score for each ticker.

        Returns dict per ticker:
        {
            "score":        float,         # normalized ensemble [0,1]
            "ridge":        float,         # raw Ridge prediction
            "xgboost":      float,         # raw XGBoost prediction
            "rf":           float,         # raw RF prediction
            "mlp":          float,         # raw MLP prediction
            "dispersion":   float,         # std dev of component predictions
            "feature_importance": dict,    # {feature: importance} from XGBoost
        }
        """
        if not self._trained:
            logger.warning("Fundamental model not trained — returning neutral 0.5")
            return {t: {"score": 0.5, "dispersion": 0.0, "feature_importance": {}} for t in tickers}

        df = current_fundamentals.copy()
        df = _add_growth_features(df)
        df = df[df["ticker"].isin(tickers)]

        if df.empty:
            return {t: {"score": 0.5, "dispersion": 0.0, "feature_importance": {}} for t in tickers}

        # Extract XGBoost feature importances (once, not per ticker)
        xgb_importances = {}
        if "xgboost" in self.models:
            try:
                xgb_model = self.models["xgboost"]
                # Handle Pipeline vs raw model
                raw_xgb = xgb_model.named_steps.get("model", xgb_model) if hasattr(xgb_model, "named_steps") else xgb_model
                if hasattr(raw_xgb, "feature_importances_"):
                    imp = raw_xgb.feature_importances_
                    xgb_importances = {f: round(float(v), 4) for f, v in zip(FEATURE_COLS, imp)}
            except Exception as e:
                logger.warning(f"Could not extract XGBoost importances: {e}")

        raw_results = {}
        for _, row in df.drop_duplicates(subset=["ticker"]).iterrows():
            ticker = row["ticker"]
            try:
                x = np.array([[row.get(f, 0.0) for f in FEATURE_COLS]], dtype=float)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                x = np.clip(x, -1e12, 1e12)

                component_preds = {}
                for name, model in self.models.items():
                    try:
                        pred = float(model.predict(x)[0])
                        component_preds[name] = pred
                    except Exception:
                        pass

                if component_preds:
                    # Weighted ensemble
                    weighted = sum(
                        component_preds[n] * MODEL_WEIGHTS.get(n, 0.25)
                        for n in component_preds
                    )
                    total_weight = sum(MODEL_WEIGHTS.get(n, 0.25) for n in component_preds)
                    ensemble = weighted / total_weight if total_weight else 0.5

                    # Dispersion = std dev of raw component predictions
                    vals = list(component_preds.values())
                    dispersion = float(np.std(vals)) if len(vals) > 1 else 0.0

                    raw_results[ticker] = {
                        "raw_ensemble": ensemble,
                        "ridge":        component_preds.get("ridge"),
                        "xgboost":      component_preds.get("xgboost"),
                        "rf":           component_preds.get("rf"),
                        "mlp":          component_preds.get("mlp"),
                        "dispersion":   dispersion,
                        "feature_importance": xgb_importances,
                    }
                else:
                    raw_results[ticker] = {"raw_ensemble": 0.5, "dispersion": 0.0, "feature_importance": {}}

            except Exception as e:
                logger.error(f"Fundamental prediction error for {ticker}: {e}")
                raw_results[ticker] = {"raw_ensemble": 0.5, "dispersion": 0.0, "feature_importance": {}}

        # Normalize ensemble scores to [0, 1] across the universe (percentile-rank;
        # min-max here floored skewed predictions near 0 — see scoring.rank_normalize)
        from app.ml.scoring import rank_normalize
        raw_vals = {t: v["raw_ensemble"] for t, v in raw_results.items()}
        norm_by_ticker = rank_normalize(raw_vals)

        results = {}
        for ticker in tickers:
            if ticker not in raw_results:
                results[ticker] = {"score": 0.5, "dispersion": 0.0, "feature_importance": {}}
                continue
            r = raw_results[ticker]
            norm = norm_by_ticker.get(ticker, 0.5)
            results[ticker] = {
                "score":              norm,
                "raw_ensemble":       r.get("raw_ensemble"),
                "ridge":              r.get("ridge"),
                "xgboost":            r.get("xgboost"),
                "rf":                 r.get("rf"),
                "mlp":                r.get("mlp"),
                "dispersion":         r.get("dispersion", 0.0),
                "feature_importance": r.get("feature_importance", {}),
            }

        return results
