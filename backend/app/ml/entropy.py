"""
app/ml/entropy.py
─────────────────────────────────────────────────────────────────────────────
Entropy ML scoring — Section 3.2.3, Cohen et al. (2025).

Features: Fuzzy entropy over 30-day rolling windows applied to:
    close price returns, volume, intraday range (high-low/close)

Ensemble: same as Technical — XGBoost + LightGBM + CatBoost + LSTM (equal weight).
Rolling 24-month training window; strictly no lookahead.
"""
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


# ── Fuzzy Entropy ─────────────────────────────────────────────────────────
def fuzzy_entropy(series: np.ndarray, m: int = 2, r_coef: float = 0.2) -> float:
    """
    Compute Fuzzy Entropy (FuzzyEn) for a 1-D time series.

    Based on Chen et al. (2007) / Liu et al. (2012).
    Measures irregularity / complexity of the time series.

    Args:
        series : 1-D array of values (e.g., 30-day rolling returns)
        m      : embedding dimension (default 2)
        r_coef : tolerance as fraction of std (default 0.2)

    Returns:
        Fuzzy entropy value (non-negative float; higher = more complex/uncertain)
    """
    N = len(series)
    if N < m + 1:
        return np.nan
    r = r_coef * np.std(series)
    if r == 0:
        return 0.0

    def _phi(m_dim: int) -> float:
        """Compute the template-matching probability for dimension m_dim."""
        count = 0.0
        total = 0.0
        for i in range(N - m_dim):
            template = series[i : i + m_dim]
            for j in range(N - m_dim):
                if i == j:
                    continue
                candidate = series[j : j + m_dim]
                d = np.max(np.abs(template - candidate))
                # Fuzzy membership function: exponential
                count += np.exp(-(d ** 2) / r)
                total += 1.0
        return count / total if total > 0 else 0.0

    phi_m  = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m == 0:
        return 0.0
    return -np.log(phi_m1 / phi_m) if phi_m1 > 0 else 0.0


def build_entropy_features(
    prices_df: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """
    Compute fuzzy entropy features for each ticker over rolling 30-day windows.

    prices_df: columns = [date, ticker, open, high, low, close, volume]
    Returns wide-format DataFrame with one row per (date, ticker).
    """
    result_rows = []

    for ticker, grp in prices_df.groupby("ticker"):
        grp = grp.sort_values("date").copy()

        # Base series
        ret      = grp["close"].pct_change().fillna(0).values
        vol      = (np.log(grp["volume"] + 1)).values
        rng      = ((grp["high"] - grp["low"]) / grp["close"].replace(0, np.nan)).fillna(0).values

        n = len(ret)
        fe_ret, fe_vol, fe_rng = [], [], []
        fe_ret_std, fe_vol_std = [], []

        for i in range(n):
            if i < window - 1:
                fe_ret.append(np.nan)
                fe_vol.append(np.nan)
                fe_rng.append(np.nan)
                fe_ret_std.append(np.nan)
                fe_vol_std.append(np.nan)
            else:
                w_ret  = ret[i - window + 1 : i + 1]
                w_vol  = vol[i - window + 1 : i + 1]
                w_rng  = rng[i - window + 1 : i + 1]
                fe_ret.append(fuzzy_entropy(w_ret))
                fe_vol.append(fuzzy_entropy(w_vol))
                fe_rng.append(fuzzy_entropy(w_rng))
                fe_ret_std.append(np.std(w_ret))
                fe_vol_std.append(np.std(w_vol))

        grp["fe_return"]        = fe_ret
        grp["fe_volume"]        = fe_vol
        grp["fe_intraday_range"]= fe_rng
        grp["fe_return_std"]    = fe_ret_std
        grp["fe_volume_std"]    = fe_vol_std
        # Also include lagged returns for context
        grp["ret_1d"]  = grp["close"].pct_change(1)
        grp["ret_5d"]  = grp["close"].pct_change(5)
        grp["ret_21d"] = grp["close"].pct_change(21)
        grp["forward_return"] = grp["close"].pct_change(21).shift(-21)

        result_rows.append(grp)

    return pd.concat(result_rows, ignore_index=True)


FEATURE_COLS = [
    "fe_return", "fe_volume", "fe_intraday_range",
    "fe_return_std", "fe_volume_std",
    "ret_1d", "ret_5d", "ret_21d",
]


class EntropyScorer:
    """
    Entropy ML ensemble — XGBoost + LightGBM + CatBoost + LSTM.
    Uses fuzzy entropy features over 30-day rolling windows.
    Rolling 24-month training window; strictly no lookahead.
    """

    def __init__(self):
        self.models: dict = {}
        self.scaler = MinMaxScaler()
        self._trained = False

    def fit(
        self,
        prices_df: pd.DataFrame,
        rebalance_date: datetime,
        training_months: int = 24,
    ) -> "EntropyScorer":
        """Train on data available before rebalance_date."""
        logger.info(f"Building entropy features for {len(prices_df)} price rows…")
        df = build_entropy_features(prices_df)
        df["date"] = pd.to_datetime(df["date"])
        cutoff_start = pd.Timestamp(rebalance_date) - pd.DateOffset(months=training_months)
        df = df[
            (df["date"] < pd.Timestamp(rebalance_date)) &
            (df["date"] >= cutoff_start)
        ]
        df = df.dropna(subset=FEATURE_COLS + ["forward_return"])

        if len(df) < 50:
            logger.warning(f"Insufficient entropy training data ({len(df)} rows)")
            return self

        X = self.scaler.fit_transform(df[FEATURE_COLS].values)
        y = df["forward_return"].values

        if XGBOOST_AVAILABLE:
            try:
                xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
                xgb.fit(X, y)
                self.models["xgboost"] = xgb
            except Exception as e:
                logger.error(f"Entropy XGBoost failed: {e}")

        if LIGHTGBM_AVAILABLE:
            try:
                lgbm = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
                lgbm.fit(X, y)
                self.models["lightgbm"] = lgbm
            except Exception as e:
                logger.error(f"Entropy LightGBM failed: {e}")

        if CATBOOST_AVAILABLE:
            try:
                cat = CatBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=0)
                cat.fit(X, y)
                self.models["catboost"] = cat
            except Exception as e:
                logger.error(f"Entropy CatBoost failed: {e}")

        self._trained = bool(self.models)
        if len(self.models) < 4:
            logger.warning(
                f"Entropy ensemble trained with {len(self.models)}/4 members — "
                "LSTM absent (no LSTM implementation in entropy.py and TensorFlow "
                "not installed; paper spec is XGBoost+LightGBM+CatBoost+LSTM)"
            )
        logger.info(f"Entropy model trained ({len(self.models)} members) on {len(X)} samples")
        return self

    def predict(
        self,
        tickers: list[str],
        prices_df: pd.DataFrame,
        rebalance_date: datetime,
    ) -> dict[str, dict]:
        """
        Predict entropy score for each ticker.
        Returns dict per ticker with component scores and dispersion.
        """
        if not self._trained:
            return {t: {"score": 0.5, "dispersion": 0.0, "feature_importance": {}} for t in tickers}

        df = build_entropy_features(prices_df)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] < pd.Timestamp(rebalance_date)]
        latest = df.sort_values("date").groupby("ticker").last().reset_index()
        latest = latest[latest["ticker"].isin(tickers)]
        latest = latest.dropna(subset=FEATURE_COLS)

        # XGBoost feature importances
        xgb_importances = {}
        if "xgboost" in self.models:
            try:
                xgb_model = self.models["xgboost"]
                if hasattr(xgb_model, "feature_importances_"):
                    xgb_importances = {f: round(float(v), 4) for f, v in zip(FEATURE_COLS, xgb_model.feature_importances_)}
            except Exception as e:
                logger.warning(f"Entropy XGBoost importances unavailable: {e}")

        raw_results = {}
        for _, row in latest.iterrows():
            ticker = row["ticker"]
            try:
                x = self.scaler.transform(np.array([[row[f] for f in FEATURE_COLS]]))
                component_preds = {}
                for name, model in self.models.items():
                    try:
                        component_preds[name] = float(model.predict(x)[0])
                    except Exception:
                        pass

                if component_preds:
                    ensemble = float(np.mean(list(component_preds.values())))
                    dispersion = float(np.std(list(component_preds.values()))) if len(component_preds) > 1 else 0.0
                    raw_results[ticker] = {
                        "raw_ensemble": ensemble,
                        "xgboost":      component_preds.get("xgboost"),
                        "lightgbm":     component_preds.get("lightgbm"),
                        "catboost":     component_preds.get("catboost"),
                        "dispersion":   dispersion,
                        "feature_importance": xgb_importances,
                    }
                else:
                    raw_results[ticker] = {"raw_ensemble": 0.5, "dispersion": 0.0, "feature_importance": {}}
            except Exception as e:
                logger.error(f"Entropy predict error {ticker}: {e}")
                raw_results[ticker] = {"raw_ensemble": 0.5, "dispersion": 0.0, "feature_importance": {}}

        # Normalize ensemble to [0, 1] (percentile-rank; min-max floored skewed preds)
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
                "xgboost":            r.get("xgboost"),
                "lightgbm":           r.get("lightgbm"),
                "catboost":           r.get("catboost"),
                "dispersion":         r.get("dispersion", 0.0),
                "feature_importance": r.get("feature_importance", {}),
            }

        return results
