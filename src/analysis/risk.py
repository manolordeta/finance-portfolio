"""
Risk metrics: VaR and CVaR (Expected Shortfall) for a portfolio.

Three VaR methods:
  - historical    : empirical quantile of past P&L distribution
  - parametric    : Gaussian assumption (mean + z * sigma)
  - monte_carlo   : simulate future returns via multivariate normal

All results are expressed as positive numbers representing losses
(i.e., VaR 5% at 0.95 confidence means: we lose at least 5% with prob 5%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(42)


# -----------------------------------------------------------------------
# Historical VaR / CVaR
# -----------------------------------------------------------------------

def var_historical(
    portfolio_returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """
    Historical simulation VaR.
    horizon > 1 scales by sqrt(horizon) (iid assumption).
    """
    q = np.quantile(portfolio_returns, 1 - confidence)
    return -q * np.sqrt(horizon)


def cvar_historical(
    portfolio_returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """Expected Shortfall (CVaR) — mean of losses beyond VaR."""
    q = np.quantile(portfolio_returns, 1 - confidence)
    tail = portfolio_returns[portfolio_returns <= q]
    return -tail.mean() * np.sqrt(horizon)


# -----------------------------------------------------------------------
# Parametric VaR / CVaR (Gaussian)
# -----------------------------------------------------------------------

def var_parametric(
    portfolio_returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    z = stats.norm.ppf(1 - confidence)
    return -(mu * horizon + z * sigma * np.sqrt(horizon))


def cvar_parametric(
    portfolio_returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """
    Analytical CVaR under Gaussian assumption:
    CVaR = -mu*h + sigma*sqrt(h) * phi(z) / (1 - confidence)
    where phi is the standard normal PDF.
    """
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    z = stats.norm.ppf(1 - confidence)
    cvar_1d = -mu + sigma * stats.norm.pdf(z) / (1 - confidence)
    return cvar_1d * np.sqrt(horizon)


# -----------------------------------------------------------------------
# Monte Carlo VaR / CVaR
# -----------------------------------------------------------------------

def var_monte_carlo(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95,
    horizon: int = 1,
    n_simulations: int = 10_000,
) -> float:
    """
    Multivariate normal Monte Carlo.
    Draws from the empirical mean vector and covariance matrix.
    """
    mu = returns.mean().values
    cov = returns.cov().values
    w = np.asarray(weights)

    # Simulate horizon-day cumulative returns
    sim = RNG.multivariate_normal(mu, cov, size=(n_simulations, horizon))
    # sim shape: (n_simulations, horizon, n_assets)
    portfolio_sim = sim.dot(w)            # (n_simulations, horizon)
    cumulative = portfolio_sim.sum(axis=1)  # (n_simulations,)

    q = np.quantile(cumulative, 1 - confidence)
    return -q


def cvar_monte_carlo(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95,
    horizon: int = 1,
    n_simulations: int = 10_000,
) -> float:
    mu = returns.mean().values
    cov = returns.cov().values
    w = np.asarray(weights)

    sim = RNG.multivariate_normal(mu, cov, size=(n_simulations, horizon))
    portfolio_sim = sim.dot(w)
    cumulative = portfolio_sim.sum(axis=1)

    q = np.quantile(cumulative, 1 - confidence)
    tail = cumulative[cumulative <= q]
    return -tail.mean()


# -----------------------------------------------------------------------
# Unified interface
# -----------------------------------------------------------------------

def compute_risk_table(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: list[float] = [0.95, 0.99],
    horizons: list[int] = [1, 5, 21],
    methods: list[str] = ["historical", "parametric", "monte_carlo"],
    n_simulations: int = 10_000,
) -> pd.DataFrame:
    """
    Returns a MultiIndex DataFrame with VaR and CVaR for all combinations
    of method × confidence × horizon.

    Index: (method, metric)   e.g. ('historical', 'VaR')
    Columns: MultiIndex (confidence, horizon)   e.g. (0.95, 1)
    """
    from src.analysis.portfolio import portfolio_returns as port_ret

    pr = port_ret(returns, weights)
    w = np.asarray(weights)

    records = []
    for method in methods:
        for conf in confidence_levels:
            for h in horizons:
                if method == "historical":
                    var = var_historical(pr, conf, h)
                    cvar = cvar_historical(pr, conf, h)
                elif method == "parametric":
                    var = var_parametric(pr, conf, h)
                    cvar = cvar_parametric(pr, conf, h)
                elif method == "monte_carlo":
                    var = var_monte_carlo(returns, w, conf, h, n_simulations)
                    cvar = cvar_monte_carlo(returns, w, conf, h, n_simulations)
                else:
                    raise ValueError(f"Unknown method: {method}")

                records.append({
                    "Method": method,
                    "Metric": "VaR",
                    "Confidence": conf,
                    "Horizon (days)": h,
                    "Value": var,
                })
                records.append({
                    "Method": method,
                    "Metric": "CVaR",
                    "Confidence": conf,
                    "Horizon (days)": h,
                    "Value": cvar,
                })

    df = pd.DataFrame(records)
    pivot = df.pivot_table(
        index=["Method", "Metric"],
        columns=["Confidence", "Horizon (days)"],
        values="Value",
    )
    return pivot
