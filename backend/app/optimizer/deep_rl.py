"""
app/optimizer/deep_rl.py
─────────────────────────────────────────────────────────────────────────────
Deep RL portfolio optimizer — Section 3.5, Cohen et al. (2025).

Algorithm : PPO via stable-baselines3
State     : composite scores + lagged returns + volatility + current weights
Reward    : R(t) = Sharpe(t) − 0.01 × Turnover(t)
Training  : Rolling 24-month window, monthly retraining (no lookahead)

⚠ Output is ALWAYS labeled "Deep RL Optimizer" — NEVER presented as the
   paper model (top-10 equal-weight selection from composite scores).
"""
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("stable-baselines3 not installed — Deep RL optimizer unavailable, falling back to MVO/HRP")


class PortfolioEnv(gym.Env):
    """
    Custom Gymnasium environment for portfolio allocation.

    State space:
        composite_scores      (n_assets,)
        lagged_returns_1m     (n_assets,)
        lagged_returns_3m     (n_assets,)
        volatility_21d        (n_assets,)
        current_weights       (n_assets,)

    Action space:
        portfolio weights ∈ [0, 1]^n (softmaxed to sum=1)

    Reward:
        R(t) = Sharpe(t) - 0.01 * Turnover(t)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns_df: pd.DataFrame,         # (T, n_assets) — rolling training window
        scores_df: pd.DataFrame,           # (T, n_assets) — composite scores
        rebalance_freq: int = 21,          # trading days per rebalance
    ):
        super().__init__()
        self.returns_df = returns_df
        self.scores_df  = scores_df
        self.n_assets   = returns_df.shape[1]
        self.rebalance_freq = rebalance_freq
        self.tickers = list(returns_df.columns)

        # 5 feature types × n_assets
        obs_dim = 5 * self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 30  # start after 30-day burn-in for vol estimation
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.prev_portfolio_value = 1.0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        t = min(self.t, len(self.returns_df) - 1)
        ret_hist = self.returns_df.iloc[max(0, t-21) : t + 1].values

        scores    = self.scores_df.iloc[t].fillna(0.5).values if t < len(self.scores_df) else np.full(self.n_assets, 0.5)
        ret_1m    = self.returns_df.iloc[t].fillna(0).values if t < len(self.returns_df) else np.zeros(self.n_assets)
        ret_3m_df = self.returns_df.iloc[max(0, t-63) : t + 1]
        ret_3m    = ret_3m_df.mean().fillna(0).values if len(ret_3m_df) > 0 else np.zeros(self.n_assets)
        vol       = np.nanstd(ret_hist, axis=0) if len(ret_hist) > 1 else np.zeros(self.n_assets)
        weights   = self.current_weights

        obs = np.concatenate([scores, ret_1m, ret_3m, vol, weights]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, action: np.ndarray):
        # Softmax to enforce sum-to-one and non-negativity
        action = np.exp(action - action.max())
        new_weights = action / action.sum()

        t = self.t
        if t >= len(self.returns_df):
            return self._obs(), 0.0, True, False, {}

        period_returns = self.returns_df.iloc[t].fillna(0).values
        portfolio_return = float(np.dot(new_weights, period_returns))

        # Compute rolling Sharpe (21-day)
        recent = self.returns_df.iloc[max(0, t-20) : t + 1].values
        portfolio_hist = (recent * self.current_weights).sum(axis=1)
        sharpe = (portfolio_hist.mean() / (portfolio_hist.std() + 1e-8)) * np.sqrt(252)

        # Turnover penalty
        turnover = float(np.sum(np.abs(new_weights - self.current_weights)))

        reward = float(sharpe - 0.01 * turnover)

        self.current_weights = new_weights
        self.t += 1

        done = self.t >= len(self.returns_df)
        return self._obs(), reward, done, False, {}


class DeepRLOptimizer:
    """
    PPO-based portfolio weight optimizer.

    ⚠ Output labeled 'Deep RL Optimizer' — separate from paper model.
    """

    def __init__(self, n_assets: int, seed: int = 42):
        self.n_assets = n_assets
        self.seed = seed
        self.model: Optional[PPO] = None

    def train(
        self,
        returns_df: pd.DataFrame,
        scores_df: pd.DataFrame,
        total_timesteps: int = 100_000,
    ) -> "DeepRLOptimizer":
        if not SB3_AVAILABLE:
            logger.warning("stable-baselines3 unavailable — cannot train DeepRL optimizer")
            return self

        try:
            env = DummyVecEnv([lambda: PortfolioEnv(returns_df, scores_df)])
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                verbose=0,
                seed=self.seed,
            )
            self.model.learn(total_timesteps=total_timesteps)
            logger.info(f"DeepRL optimizer trained ({total_timesteps} steps)")
        except Exception as e:
            logger.error(f"PPO training failed: {e}", exc_info=True)
        return self

    def predict_weights(
        self,
        current_obs: np.ndarray,
        deterministic: bool = True,
    ) -> dict[str, float]:
        """
        Predict optimal weights given current observation.
        Falls back to equal weight if model not trained.
        """
        if self.model is None or not SB3_AVAILABLE:
            logger.warning("DeepRL model not trained — returning equal weights")
            return {i: 1.0 / self.n_assets for i in range(self.n_assets)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            action, _ = self.model.predict(current_obs, deterministic=deterministic)

        action = np.exp(action - action.max())
        weights = action / action.sum()
        return weights


# ────────────────────────────────────────────────────────────────────────────
# app/optimizer/mvo.py — Mean-Variance Optimization fallback
# ────────────────────────────────────────────────────────────────────────────
"""
MVO optimizer fallback using scipy.
Implements Markowitz minimum-variance and max-Sharpe portfolios.
"""
import numpy as np
import pandas as pd
import logging

logger_mvo = logging.getLogger("optimizer.mvo")

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def mvo_optimize(
    returns_df: pd.DataFrame,
    target: str = "max_sharpe",
    risk_free_rate: float = 0.045,
    max_weight: float = 0.25,
) -> dict[str, float]:
    """
    Mean-Variance Optimization.

    Args:
        returns_df  : (T × n_assets) daily returns, no lookahead
        target      : 'max_sharpe' | 'min_variance' | 'equal_weight'
        risk_free_rate : annual risk-free rate
        max_weight  : max position size constraint

    Returns:
        {ticker: weight} dict summing to ~1.0
    """
    tickers = list(returns_df.columns)
    n = len(tickers)

    if n == 0 or not SCIPY_AVAILABLE or target == "equal_weight":
        return {t: 1.0 / n for t in tickers}

    mu  = returns_df.mean().values * 252  # annualized
    cov = returns_df.cov().values * 252

    rf_daily = risk_free_rate / 252

    def portfolio_sharpe(w: np.ndarray) -> float:
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        return -(ret - risk_free_rate) / (vol + 1e-8)

    def portfolio_variance(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    obj = portfolio_sharpe if target == "max_sharpe" else portfolio_variance

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight)] * n
    w0 = np.ones(n) / n

    try:
        result = minimize(
            obj, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if result.success:
            weights = np.maximum(result.x, 0)
            weights /= weights.sum()
            return {t: float(w) for t, w in zip(tickers, weights)}
    except Exception as e:
        logger_mvo.error(f"MVO optimization failed: {e}")

    return {t: 1.0 / n for t in tickers}


# ────────────────────────────────────────────────────────────────────────────
# app/optimizer/hrp.py — Hierarchical Risk Parity fallback
# ────────────────────────────────────────────────────────────────────────────
"""
Hierarchical Risk Parity (Lopez de Prado, 2016).
A robust diversification method that doesn't require inverting the covariance matrix.
"""
import numpy as np
import pandas as pd
import logging

logger_hrp = logging.getLogger("optimizer.hrp")


def _corr_dist(corr: pd.DataFrame) -> pd.DataFrame:
    """Euclidean distance from correlation matrix."""
    return np.sqrt((1 - corr) / 2.0)


def _cluster_variance(cov: pd.DataFrame, items: list) -> float:
    sub_cov = cov.loc[items, items]
    ivp = 1.0 / np.diag(sub_cov.values)
    ivp /= ivp.sum()
    return float(ivp @ sub_cov.values @ ivp)


def _get_quasi_diag(link: np.ndarray) -> list[int]:
    """Sort clustered items by dendrogram leaves."""
    link = link.astype(int)
    sorted_items = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]

    while sorted_items.max() >= num_items:
        sorted_items.index = range(0, sorted_items.shape[0] * 2, 2)
        df = sorted_items[sorted_items >= num_items]
        i = df.index
        j = df.values - num_items
        sorted_items[i] = link[j, 0]
        sorted_items = sorted_items.append(pd.Series(link[j, 1], index=i + 1))
        sorted_items = sorted_items.sort_index()
        sorted_items.index = range(sorted_items.shape[0])

    return sorted_items.tolist()


def hrp_optimize(returns_df: pd.DataFrame) -> dict[str, float]:
    """
    Hierarchical Risk Parity portfolio weights.

    Args:
        returns_df : (T × n_assets) daily returns, no lookahead

    Returns:
        {ticker: weight} dict summing to ~1.0
    """
    tickers = list(returns_df.columns)
    n = len(tickers)

    if n <= 1:
        return {t: 1.0 / n for t in tickers}

    try:
        from scipy.cluster.hierarchy import linkage

        cov  = returns_df.cov()
        corr = returns_df.corr()
        dist = _corr_dist(corr)

        link = linkage(dist.values[np.triu_indices(n, k=1)], method="single")
        sorted_idx = _get_quasi_diag(link)
        sorted_items = corr.index[sorted_idx].tolist()

        # Recursive bisection
        weights = pd.Series(1.0, index=sorted_items)
        clusters = [sorted_items]

        while clusters:
            clusters = [
                i[j:k]
                for i in clusters
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            for i in range(0, len(clusters), 2):
                if i + 1 >= len(clusters):
                    break
                c0, c1 = clusters[i], clusters[i + 1]
                v0 = _cluster_variance(cov, c0)
                v1 = _cluster_variance(cov, c1)
                alpha = 1 - v0 / (v0 + v1)
                weights[c0] *= alpha
                weights[c1] *= 1 - alpha

        weights = weights.reindex(tickers).fillna(1.0 / n)
        weights = weights / weights.sum()
        return weights.to_dict()

    except Exception as e:
        logger_hrp.error(f"HRP optimization failed: {e}")
        return {t: 1.0 / n for t in tickers}
