"""
Portfolio statistics: returns, volatility, Sharpe, drawdown, correlation.
All functions operate on a returns DataFrame (dates × tickers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Weighted daily portfolio returns."""
    w = np.asarray(weights)
    return returns.dot(w)


def annualized_return(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Geometric annualized return."""
    n = len(returns)
    total = (1 + returns).prod()
    return total ** (TRADING_DAYS / n) - 1


def annualized_volatility(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Annualized standard deviation of daily returns."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.045) -> float:
    """Annualized Sharpe ratio."""
    excess = annualized_return(returns) - risk_free_rate
    vol = annualized_volatility(returns)
    return excess / vol if vol != 0 else np.nan


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown time series."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    return (cumulative - running_max) / running_max


def calmar_ratio(returns: pd.Series) -> float:
    """Annualized return / |Max Drawdown|."""
    mdd = abs(max_drawdown(returns))
    return annualized_return(returns) / mdd if mdd != 0 else np.nan


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def covariance_matrix(returns: pd.DataFrame, annualized: bool = True) -> pd.DataFrame:
    cov = returns.cov()
    return cov * TRADING_DAYS if annualized else cov


def summary_stats(
    returns: pd.DataFrame,
    weights: np.ndarray | None = None,
    risk_free_rate: float = 0.045,
) -> pd.DataFrame:
    """
    Per-asset summary statistics. If weights are provided, also includes
    a 'Portfolio' column with weighted aggregate stats.
    """
    stats: dict[str, dict] = {}

    for col in returns.columns:
        r = returns[col]
        stats[col] = {
            "Ann. Return": annualized_return(r),
            "Ann. Volatility": annualized_volatility(r),
            "Sharpe Ratio": sharpe_ratio(r, risk_free_rate),
            "Max Drawdown": max_drawdown(r),
            "Calmar Ratio": calmar_ratio(r),
            "Skewness": r.skew(),
            "Kurtosis": r.kurt(),
        }

    if weights is not None:
        pr = portfolio_returns(returns, weights)
        stats["Portfolio"] = {
            "Ann. Return": annualized_return(pr),
            "Ann. Volatility": annualized_volatility(pr),
            "Sharpe Ratio": sharpe_ratio(pr, risk_free_rate),
            "Max Drawdown": max_drawdown(pr),
            "Calmar Ratio": calmar_ratio(pr),
            "Skewness": pr.skew(),
            "Kurtosis": pr.kurt(),
        }

    return pd.DataFrame(stats).T
