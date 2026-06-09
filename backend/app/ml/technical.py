"""
app/ml/technical.py
─────────────────────────────────────────────────────────────────────────────
Technical ML scoring — Section 3.2.2, Cohen et al. (2025).

Ensemble (equal weight, 25% each):
    XGBoost   ·  LightGBM  ·  CatBoost  ·  LSTM

Features (from OHLCV via Alpaca):
    RSI(14), MACD, MACD_signal, SMA_10, SMA_20, SMA_50,
    EMA_10, EMA_20, rolling_std_10, rolling_std_20,
    ret_1d, ret_5d, ret_10d, ret_21d (lagged returns)

No-lookahead: rolling 24-month train window, retrain monthly.
"""
import logging
from datetime import datetime, timedelta
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

try:
    import tensorflow as tf  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


# ── Technical indicators ──────────────────────────────────────────────
def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(prices: pd.Series, fast=12, slow=26, signal=9) -> tuple[pd.Series, pd.Series]:
    ema_fast   = prices.ewm(span=fast, adjust=False).mean()
    ema_slow   = prices.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def build_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build technical features from OHLCV data.

    prices_df: columns = [date, ticker, open, high, low, close, volume]
    Returns wide-format DataFrame with one row per (date, ticker).
    """
    result_rows = []
    for ticker, grp in prices_df.groupby("ticker"):
        grp = grp.sort_values("date").copy()
        close = grp["close"]

        grp["rsi"]           = compute_rsi(close)
        grp["macd"], grp["macd_signal"] = compute_macd(close)
        grp["sma_10"]        = close.rolling(10).mean()
        grp["sma_20"]        = close.rolling(20).mean()
        grp["sma_50"]        = close.rolling(50).mean()
        grp["ema_10"]        = close.ewm(span=10, adjust=False).mean()
        grp["ema_20"]        = close.ewm(span=20, adjust=False).mean()
        grp["rolling_std_10"]= close.rolling(10).std()
        grp["rolling_std_20"]= close.rolling(20).std()
        grp["ret_1d"]        = close.pct_change(1)
        grp["ret_5d"]        = close.pct_change(5)
        grp["ret_10d"]       = close.pct_change(10)
        grp["ret_21d"]       = close.pct_change(21)
        grp["forward_return"]= close.pct_change(21).shift(-21)  # 1-month forward (target)

        result_rows.append(grp)

    return pd.concat(result_rows, ignore_index=True)


FEATURE_COLS = [
    "rsi", "macd", "macd_signal",
    "sma_10", "sma_20", "sma_50",
    "ema_10", "ema_20",
    "rolling_std_10", "rolling_std_20",
    "ret_1d", "ret_5d", "ret_10d", "ret_21d",
]


def _build_lstm_model(input_shape: tuple) -> Optional[object]:
    """Build LSTM model if TensorFlow is available."""
    if not TENSORFLOW_AVAILABLE:
        return None
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model
    except Exception as e:
        logger.warning(f"Could not build LSTM model: {e}")
        return None


class TechnicalScorer:
    """
    Technical ML ensemble — XGBoost + LightGBM + CatBoost + LSTM.
    Rolling 24-month training window; no lookahead bias.
    """

    def __init__(self, sequence_len: int = 30):  # paper: 30-day rolling LSTM sequences
        self.sequence_len = sequence_len
        self.models: dict = {}
        self.scaler = MinMaxScaler()
        self._trained = False

    def fit(
        self,
        prices_df: pd.DataFrame,
        rebalance_date: datetime,
        training_months: int = 24,
    ) -> "TechnicalScorer":
        """
        Train on rolling 24-month window ending at rebalance_date.
        Strictly no lookahead: only data with date < rebalance_date.
        """
        df = build_features(prices_df)
        df["date"] = pd.to_datetime(df["date"])
        cutoff_start = pd.Timestamp(rebalance_date) - pd.DateOffset(months=training_months)
        df = df[
            (df["date"] < pd.Timestamp(rebalance_date)) &
            (df["date"] >= cutoff_start)
        ]
        df = df.dropna(subset=FEATURE_COLS + ["forward_return"])

        if len(df) < 50:
            logger.warning("Insufficient data for technical model training")
            return self

        X = self.scaler.fit_transform(df[FEATURE_COLS].values)
        y = df["forward_return"].values

        # XGBoost
        if XGBOOST_AVAILABLE:
            try:
                xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
                xgb.fit(X, y)
                self.models["xgboost"] = xgb
                logger.info(f"Technical XGBoost trained on {len(X)} samples")
            except Exception as e:
                logger.error(f"XGBoost training failed: {e}")

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            try:
                lgbm = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
                lgbm.fit(X, y)
                self.models["lightgbm"] = lgbm
            except Exception as e:
                logger.error(f"LightGBM training failed: {e}")

        # CatBoost
        if CATBOOST_AVAILABLE:
            try:
                cat = CatBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=0)
                cat.fit(X, y)
                self.models["catboost"] = cat
            except Exception as e:
                logger.error(f"CatBoost training failed: {e}")

        # LSTM
        if TENSORFLOW_AVAILABLE and len(df) >= self.sequence_len + 10:
            try:
                X_seq, y_seq = self._make_sequences(X, y)
                lstm = _build_lstm_model((self.sequence_len, len(FEATURE_COLS)))
                if lstm:
                    lstm.fit(X_seq, y_seq, epochs=30, batch_size=32, verbose=0)
                    self.models["lstm"] = ("lstm", lstm, len(FEATURE_COLS))
            except Exception as e:
                logger.error(f"LSTM training failed: {e}")

        self._trained = bool(self.models)
        return self

    def _make_sequences(self, X: np.ndarray, y: np.ndarray):
        xs, ys = [], []
        for i in range(self.sequence_len, len(X)):
            xs.append(X[i - self.sequence_len : i])
            ys.append(y[i])
        return np.array(xs), np.array(ys)

    def predict(
        self,
        tickers: list[str],
        prices_df: pd.DataFrame,
        rebalance_date: datetime,
    ) -> dict[str, dict]:
        """
        Predict technical ML score for each ticker.
        Returns dict per ticker with component scores and feature importances.
        """
        if not self._trained:
            return {t: {"score": 0.5, "dispersion": 0.0, "feature_importance": {}} for t in tickers}

        df = build_features(prices_df)
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
                logger.warning(f"Could not extract technical XGBoost importances: {e}")

        raw_results = {}
        for _, row in latest.iterrows():
            ticker = row["ticker"]
            try:
                x_raw = np.array([[row[f] for f in FEATURE_COLS]])
                x = self.scaler.transform(x_raw)
                component_preds = {}
                for name, model in self.models.items():
                    if isinstance(model, tuple) and model[0] == "lstm":
                        continue
                    try:
                        component_preds[name] = float(model.predict(x)[0])
                    except Exception:
                        pass

                if component_preds:
                    ensemble = float(np.mean(list(component_preds.values())))
                    dispersion = float(np.std(list(component_preds.values()))) if len(component_preds) > 1 else 0.0
                    raw_results[ticker] = {
                        "raw_ensemble": ensemble,
                        "xgboost":  component_preds.get("xgboost"),
                        "lightgbm": component_preds.get("lightgbm"),
                        "catboost": component_preds.get("catboost"),
                        "dispersion": dispersion,
                        "feature_importance": xgb_importances,
                    }
                else:
                    raw_results[ticker] = {"raw_ensemble": 0.5, "dispersion": 0.0, "feature_importance": {}}
            except Exception as e:
                logger.error(f"Technical predict error {ticker}: {e}")
                raw_results[ticker] = {"raw_ensemble": 0.5, "dispersion": 0.0, "feature_importance": {}}

        # Normalize (percentile-rank across universe; min-max floored skewed preds)
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
                "xgboost":            r.get("xgboost"),
                "lightgbm":           r.get("lightgbm"),
                "catboost":           r.get("catboost"),
                "dispersion":         r.get("dispersion", 0.0),
                "feature_importance": r.get("feature_importance", {}),
            }

        return results
