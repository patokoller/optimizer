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
from sklearn.linear_model import Ridge
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
            ("model", Ridge(alpha=1.0)),
        ]),
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)),
        ]),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
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

        X = df[FEATURE_COLS].values
        y = df["forward_return"].values

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
    ) -> dict[str, float]:
        """
        Predict fundamental ML score for each ticker.
        Returns normalized scores ∈ [0, 1].
        """
        if not self._trained:
            logger.warning("Fundamental model not trained — returning neutral 0.5")
            return {t: 0.5 for t in tickers}

        df = current_fundamentals.copy()
        df = _add_growth_features(df)
        df = df[df["ticker"].isin(tickers)]

        if df.empty:
            return {t: 0.5 for t in tickers}

        results = {}
        for _, row in df.iterrows():
            ticker = row["ticker"]
            try:
                x = np.array([[row.get(f, 0.0) for f in FEATURE_COLS]])
                preds = []
                weights = []
                for name, model in self.models.items():
                    weight = MODEL_WEIGHTS.get(name, MODEL_WEIGHTS.get("rf", 0.25))
                    try:
                        pred = float(model.predict(x)[0])
                        preds.append(pred * weight)
                        weights.append(weight)
                    except Exception:
                        pass
                if preds:
                    results[ticker] = sum(preds) / sum(weights) if weights else 0.5
                else:
                    results[ticker] = 0.5
            except Exception as e:
                logger.error(f"Fundamental prediction error for {ticker}: {e}")
                results[ticker] = 0.5

        # Normalize predictions to [0, 1]
        if results:
            vals = np.array(list(results.values()))
            lo, hi = vals.min(), vals.max()
            if hi > lo:
                for k in results:
                    results[k] = float((results[k] - lo) / (hi - lo))

        # Fill missing tickers with 0.5
        for t in tickers:
            if t not in results:
                results[t] = 0.5

        return results
