"""
Portfolio optimization via cvxpy.

Three objectives:
  - max_sharpe    : maximize Sharpe ratio (solved as min-variance on excess returns)
  - min_variance  : global minimum variance portfolio
  - risk_parity   : equal risk contribution (ERC) — solved iteratively

All return a dict with keys: weights, expected_return, volatility, sharpe_ratio
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize

from src.analysis.portfolio import annualized_return, annualized_volatility, TRADING_DAYS

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _portfolio_stats(
    weights: np.ndarray,
    mu: np.ndarray,          # annualized mean returns
    cov: np.ndarray,         # annualized covariance
    risk_free_rate: float,
) -> dict:
    w = np.asarray(weights)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else np.nan
    return {"weights": w, "expected_return": ret, "volatility": vol, "sharpe_ratio": sharpe}


def _annualized_params(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return annualized mu vector and covariance matrix."""
    mu = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values * TRADING_DAYS
    return mu, cov


# -----------------------------------------------------------------------
# Max Sharpe (Markowitz tangency portfolio)
# -----------------------------------------------------------------------

def max_sharpe(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.045,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
) -> dict:
    """
    Maximize Sharpe ratio via the Sharpe-Lintner transformation:
      minimize  w^T Σ w
      subject to  (mu - rf)^T w = 1,  lb <= w/sum(w) <= ub
    Then normalize w to get portfolio weights.
    """
    mu, cov = _annualized_params(returns)
    n = len(mu)
    excess = mu - risk_free_rate

    if np.all(excess <= 0):
        warnings.warn("All assets have negative excess returns; max Sharpe undefined. Falling back to min variance.")
        return min_variance(returns, weight_bounds=weight_bounds)

    y = cp.Variable(n)  # y = w / (excess^T w), will normalize after
    objective = cp.Minimize(cp.quad_form(y, cov))
    constraints = [
        excess @ y == 1,
        y >= weight_bounds[0],
    ]
    if weight_bounds[1] < 1.0:
        constraints.append(y <= weight_bounds[1])

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Max Sharpe optimization failed: {prob.status}")

    raw = y.value
    w = raw / raw.sum()
    return _portfolio_stats(w, mu, cov, risk_free_rate)


# -----------------------------------------------------------------------
# Minimum Variance
# -----------------------------------------------------------------------

def min_variance(
    returns: pd.DataFrame,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    risk_free_rate: float = 0.045,
) -> dict:
    mu, cov = _annualized_params(returns)
    n = len(mu)

    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cov))
    constraints = [
        cp.sum(w) == 1,
        w >= weight_bounds[0],
        w <= weight_bounds[1],
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Min variance optimization failed: {prob.status}")

    return _portfolio_stats(w.value, mu, cov, risk_free_rate)


# -----------------------------------------------------------------------
# Risk Parity (Equal Risk Contribution)
# -----------------------------------------------------------------------

def risk_parity(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.045,
) -> dict:
    """
    Equal risk contribution portfolio.
    Solves: minimize sum_i (w_i * (Σw)_i - vol/n)^2
    using scipy since this is a non-convex formulation.
    """
    mu, cov = _annualized_params(returns)
    n = len(mu)
    target = np.ones(n) / n  # equal risk budget

    def risk_contributions(w):
        vol = np.sqrt(w @ cov @ w)
        marginal = cov @ w
        rc = w * marginal / vol
        return rc / rc.sum()  # normalized

    def objective(w):
        rc = risk_contributions(w)
        return np.sum((rc - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 1.0)] * n  # ERC needs positive weights
    w0 = np.ones(n) / n

    result = minimize(
        objective, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        logger.warning("Risk parity solver: %s", result.message)

    w = result.x / result.x.sum()
    return _portfolio_stats(w, mu, cov, risk_free_rate)


# -----------------------------------------------------------------------
# Efficient Frontier
# -----------------------------------------------------------------------

def efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 100,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    risk_free_rate: float = 0.045,
) -> pd.DataFrame:
    """
    Trace the efficient frontier by solving min-variance for a grid of
    target returns between min-variance and max-return portfolios.

    Returns DataFrame with columns: volatility, expected_return, sharpe_ratio
    """
    mu, cov = _annualized_params(returns)
    n = len(mu)

    # Bounds for target return grid
    ret_min = min_variance(returns, weight_bounds, risk_free_rate)["expected_return"]
    ret_max = float(mu.max())
    targets = np.linspace(ret_min, ret_max * 0.99, n_points)

    records = []
    for target_ret in targets:
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cov))
        constraints = [
            cp.sum(w) == 1,
            w @ mu >= target_ret,
            w >= weight_bounds[0],
            w <= weight_bounds[1],
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL, verbose=False)

        if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
            wv = w.value
            vol = float(np.sqrt(wv @ cov @ wv))
            ret = float(wv @ mu)
            sharpe = (ret - risk_free_rate) / vol if vol > 0 else np.nan
            records.append({
                "volatility": vol,
                "expected_return": ret,
                "sharpe_ratio": sharpe,
                "weights": wv.tolist(),
            })

    return pd.DataFrame(records)


# -----------------------------------------------------------------------
# Unified interface: run all objectives
# -----------------------------------------------------------------------

def run_optimization(
    returns: pd.DataFrame,
    objectives: list[str] = ["max_sharpe", "min_variance", "risk_parity"],
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    risk_free_rate: float = 0.045,
) -> dict[str, dict]:
    """
    Run all requested optimization objectives.
    Returns a dict: { objective_name -> stats_dict }
    """
    tickers = list(returns.columns)
    results = {}

    for obj in objectives:
        logger.info("Optimizing: %s", obj)
        if obj == "max_sharpe":
            res = max_sharpe(returns, risk_free_rate, weight_bounds)
        elif obj == "min_variance":
            res = min_variance(returns, weight_bounds, risk_free_rate)
        elif obj == "risk_parity":
            res = risk_parity(returns, risk_free_rate)
        else:
            raise ValueError(f"Unknown objective: {obj}")

        res["tickers"] = tickers
        results[obj] = res

    return results


def optimization_summary(opt_results: dict[str, dict]) -> pd.DataFrame:
    """
    Format optimization results into a readable DataFrame.
    Rows: objectives. Columns: return, vol, sharpe + per-ticker weights.
    """
    records = {}
    for obj, res in opt_results.items():
        row = {
            "Ann. Return": res["expected_return"],
            "Ann. Volatility": res["volatility"],
            "Sharpe Ratio": res["sharpe_ratio"],
        }
        for ticker, w in zip(res["tickers"], res["weights"]):
            row[ticker] = w
        records[obj] = row

    return pd.DataFrame(records).T
