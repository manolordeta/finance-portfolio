"""
Black-Litterman Portfolio Optimization.

Combines:
  1. Market equilibrium returns (CAPM prior)
  2. Quantitative views from our ranking system
  3. Optional human views (from BBVA partner)

Produces optimal portfolio weights that balance
market consensus with our signal-based views.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BLResult:
    """Result of Black-Litterman optimization."""
    tickers: list[str]
    weights: dict[str, float]  # ticker → weight (sums to 1)
    expected_returns: dict[str, float]  # posterior expected returns (annualized)
    risk_contribution: dict[str, float]  # % of portfolio risk from each position
    portfolio_vol: float  # annualized portfolio volatility
    portfolio_return: float  # expected annualized return
    portfolio_sharpe: float
    method: str  # 'black_litterman' | 'max_sharpe' | 'min_variance'


def black_litterman(
    covariance: pd.DataFrame,
    market_caps: dict[str, float] | None = None,
    views: dict[str, float] | None = None,
    view_confidence: dict[str, float] | None = None,
    risk_free_rate: float = 0.045,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> dict[str, float]:
    """
    Compute Black-Litterman posterior expected returns.

    Args:
        covariance: annualized covariance matrix (N × N)
        market_caps: market cap per ticker (for equilibrium weights)
                     if None, uses equal-weight as prior
        views: dict of ticker → expected excess return (our signal)
        view_confidence: dict of ticker → confidence [0, 1]
        risk_free_rate: annual risk-free rate
        tau: scalar controlling prior uncertainty (typically 0.02-0.10)
        risk_aversion: risk aversion coefficient (delta)

    Returns:
        dict of ticker → posterior expected return (annualized)
    """
    tickers = list(covariance.columns)
    N = len(tickers)
    Sigma = covariance.values  # N × N

    # 1. Equilibrium returns (CAPM prior)
    if market_caps:
        total_cap = sum(market_caps.get(t, 0) for t in tickers)
        if total_cap > 0:
            w_mkt = np.array([market_caps.get(t, 0) / total_cap for t in tickers])
        else:
            w_mkt = np.ones(N) / N
    else:
        w_mkt = np.ones(N) / N  # equal-weight prior

    Pi = risk_aversion * Sigma @ w_mkt  # equilibrium excess returns

    # 2. If no views, return equilibrium + risk-free
    if not views:
        return {t: Pi[i] + risk_free_rate for i, t in enumerate(tickers)}

    # 3. Build view matrices
    # P: K × N pick matrix (which tickers have views)
    # Q: K × 1 view returns
    # Omega: K × K view uncertainty (diagonal)
    view_tickers = [t for t in tickers if t in views]
    K = len(view_tickers)

    if K == 0:
        return {t: Pi[i] + risk_free_rate for i, t in enumerate(tickers)}

    P = np.zeros((K, N))
    Q = np.zeros(K)
    omega_diag = np.zeros(K)

    for k, vt in enumerate(view_tickers):
        idx = tickers.index(vt)
        P[k, idx] = 1.0
        Q[k] = views[vt]  # expected excess return from our signal

        # View uncertainty: inversely proportional to confidence
        conf = (view_confidence or {}).get(vt, 0.5)
        conf = max(conf, 0.05)  # floor at 5% confidence
        # Omega_k = (1/conf - 1) * tau * Sigma[idx, idx]
        omega_diag[k] = (1.0 / conf - 1.0) * tau * Sigma[idx, idx]

    Omega = np.diag(omega_diag)

    # 4. Black-Litterman formula
    # μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]
    tau_Sigma_inv = np.linalg.inv(tau * Sigma)
    Omega_inv = np.linalg.inv(Omega)

    # Posterior precision
    M = tau_Sigma_inv + P.T @ Omega_inv @ P
    M_inv = np.linalg.inv(M)

    # Posterior mean
    mu_BL = M_inv @ (tau_Sigma_inv @ Pi + P.T @ Omega_inv @ Q)

    return {t: mu_BL[i] + risk_free_rate for i, t in enumerate(tickers)}


def optimize_weights(
    expected_returns: dict[str, float],
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.045,
    max_weight: float = 0.15,
    min_weight: float = 0.00,
    max_sector_weight: float = 0.35,
    sectors: dict[str, str] | None = None,
    method: str = "max_sharpe",
) -> BLResult:
    """
    Optimize portfolio weights given expected returns and covariance.

    Args:
        expected_returns: ticker → expected annual return (from B-L or direct)
        covariance: annualized covariance matrix
        risk_free_rate: annual risk-free rate
        max_weight: maximum weight per position
        min_weight: minimum weight per position (0 = allow exclusion)
        max_sector_weight: maximum weight per sector
        sectors: ticker → sector name (for sector constraints)
        method: 'max_sharpe' | 'min_variance' | 'risk_parity'

    Returns:
        BLResult with optimal weights and portfolio metrics.
    """
    tickers = [t for t in covariance.columns if t in expected_returns]
    N = len(tickers)

    if N == 0:
        return BLResult(tickers=[], weights={}, expected_returns={},
                        risk_contribution={}, portfolio_vol=0, portfolio_return=0,
                        portfolio_sharpe=0, method=method)

    mu = np.array([expected_returns[t] - risk_free_rate for t in tickers])  # excess returns
    Sigma = covariance.loc[tickers, tickers].values

    try:
        import cvxpy as cp

        w = cp.Variable(N)
        ret = mu @ w
        risk = cp.quad_form(w, Sigma)

        constraints = [
            cp.sum(w) == 1,
            w >= min_weight,
            w <= max_weight,
        ]

        # Sector constraints
        if sectors and max_sector_weight < 1.0:
            sector_groups = {}
            for i, t in enumerate(tickers):
                s = sectors.get(t, "Unknown")
                sector_groups.setdefault(s, []).append(i)
            for s, indices in sector_groups.items():
                constraints.append(cp.sum(w[indices]) <= max_sector_weight)

        if method in ("max_sharpe", "mean_variance"):
            # Mean-variance utility maximization: E[r] - 0.5 * δ * σ²
            # Note: this approximates max Sharpe but is not identical.
            # True max Sharpe requires fractional programming (Dinkelbach).
            # With constraints, the difference is minimal in practice.
            objective = cp.Maximize(ret - 0.5 * 2.5 * risk)
        elif method == "min_variance":
            objective = cp.Minimize(risk)
        else:  # risk_parity approximation
            objective = cp.Maximize(ret - 0.5 * 2.5 * risk)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        if prob.status in ("optimal", "optimal_inaccurate"):
            weights_arr = w.value
        else:
            logger.warning("Optimization failed (%s), using equal weight", prob.status)
            weights_arr = np.ones(N) / N

    except ImportError:
        logger.warning("cvxpy not available, using analytical max-Sharpe")
        # Analytical solution (no constraints except sum=1)
        Sigma_inv = np.linalg.inv(Sigma)
        ones = np.ones(N)
        w_raw = Sigma_inv @ mu
        weights_arr = w_raw / w_raw.sum()
        weights_arr = np.clip(weights_arr, min_weight, max_weight)
        weights_arr /= weights_arr.sum()

    # Clean up small weights
    weights_arr = np.where(np.abs(weights_arr) < 0.005, 0, weights_arr)
    if weights_arr.sum() > 0:
        weights_arr /= weights_arr.sum()

    # Portfolio metrics
    port_ret = float(mu @ weights_arr) + risk_free_rate
    port_var = float(weights_arr @ Sigma @ weights_arr)
    port_vol = np.sqrt(port_var)
    port_sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0

    # Risk contribution
    marginal_risk = Sigma @ weights_arr
    risk_contrib = weights_arr * marginal_risk
    total_risk = risk_contrib.sum()
    risk_pct = risk_contrib / total_risk if total_risk > 0 else np.zeros(N)

    weights_dict = {t: float(weights_arr[i]) for i, t in enumerate(tickers) if weights_arr[i] > 0.005}
    returns_dict = {t: float(mu[i]) + risk_free_rate for i, t in enumerate(tickers)}
    risk_dict = {t: float(risk_pct[i]) for i, t in enumerate(tickers) if weights_arr[i] > 0.005}

    return BLResult(
        tickers=[t for t in tickers if t in weights_dict],
        weights=weights_dict,
        expected_returns=returns_dict,
        risk_contribution=risk_dict,
        portfolio_vol=port_vol,
        portfolio_return=port_ret,
        portfolio_sharpe=port_sharpe,
        method=method,
    )


def optimize_target_vol(
    expected_returns: dict[str, float],
    covariance: pd.DataFrame,
    target_vol: float,
    risk_free_rate: float = 0.045,
    max_weight: float = 0.15,
    min_weight: float = 0.00,
    max_sector_weight: float = 0.35,
    sectors: dict[str, str] | None = None,
    ranking_positions: dict[str, int] | None = None,
) -> BLResult:
    """
    Optimize portfolio to maximize return at a target volatility level.

    High-conviction tickers (top 10 in ranking) get a minimum weight floor
    so the optimizer can't exclude them entirely.
    """
    tickers = [t for t in covariance.columns if t in expected_returns]
    N = len(tickers)

    if N == 0:
        return BLResult(tickers=[], weights={}, expected_returns={},
                        risk_contribution={}, portfolio_vol=0, portfolio_return=0,
                        portfolio_sharpe=0, method=f"target_vol_{target_vol:.0%}")

    mu = np.array([expected_returns[t] - risk_free_rate for t in tickers])
    Sigma = covariance.loc[tickers, tickers].values

    try:
        import cvxpy as cp

        w = cp.Variable(N)
        ret = mu @ w
        risk = cp.quad_form(w, Sigma)

        # Min weight: top-ranked tickers get a floor
        min_weights = np.full(N, min_weight)
        if ranking_positions:
            for i, t in enumerate(tickers):
                rank = ranking_positions.get(t, 999)
                if rank <= 5:
                    min_weights[i] = 0.03  # top 5: at least 3%
                elif rank <= 10:
                    min_weights[i] = 0.02  # top 10: at least 2%
                elif rank <= 20:
                    min_weights[i] = 0.01  # top 20: at least 1%

        constraints = [
            cp.sum(w) == 1,
            w >= min_weights,
            w <= max_weight,
            risk <= target_vol ** 2,  # volatility constraint
        ]

        if sectors and max_sector_weight < 1.0:
            sector_groups = {}
            for i, t in enumerate(tickers):
                s = sectors.get(t, "Unknown")
                sector_groups.setdefault(s, []).append(i)
            for s, indices in sector_groups.items():
                constraints.append(cp.sum(w[indices]) <= max_sector_weight)

        # Maximize return subject to vol constraint
        objective = cp.Maximize(ret)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        if prob.status in ("optimal", "optimal_inaccurate"):
            weights_arr = w.value
        else:
            logger.warning("Target vol optimization failed (%s), trying relaxed", prob.status)
            # Relax: try without vol constraint, use max_sharpe
            constraints_relaxed = [cp.sum(w) == 1, w >= 0, w <= max_weight]
            prob2 = cp.Problem(cp.Maximize(ret - 0.5 * 2.5 * risk), constraints_relaxed)
            prob2.solve(solver=cp.SCS, verbose=False)
            weights_arr = w.value if prob2.status in ("optimal", "optimal_inaccurate") else np.ones(N) / N

    except ImportError:
        logger.warning("cvxpy not available, using equal weight")
        weights_arr = np.ones(N) / N

    # Clean up
    weights_arr = np.where(np.abs(weights_arr) < 0.005, 0, weights_arr)
    if weights_arr.sum() > 0:
        weights_arr /= weights_arr.sum()

    port_ret = float(mu @ weights_arr) + risk_free_rate
    port_var = float(weights_arr @ Sigma @ weights_arr)
    port_vol = np.sqrt(port_var)
    port_sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0

    marginal_risk = Sigma @ weights_arr
    risk_contrib = weights_arr * marginal_risk
    total_risk = risk_contrib.sum()
    risk_pct = risk_contrib / total_risk if total_risk > 0 else np.zeros(N)

    weights_dict = {t: float(weights_arr[i]) for i, t in enumerate(tickers) if weights_arr[i] > 0.005}
    returns_dict = {t: float(mu[i]) + risk_free_rate for i, t in enumerate(tickers)}
    risk_dict = {t: float(risk_pct[i]) for i, t in enumerate(tickers) if weights_arr[i] > 0.005}

    return BLResult(
        tickers=[t for t in tickers if t in weights_dict],
        weights=weights_dict,
        expected_returns=returns_dict,
        risk_contribution=risk_dict,
        portfolio_vol=port_vol,
        portfolio_return=port_ret,
        portfolio_sharpe=port_sharpe,
        method=f"target_vol_{target_vol:.0%}",
    )


def compute_efficient_frontier(
    expected_returns: dict[str, float],
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.045,
    max_weight: float = 0.15,
    max_sector_weight: float = 0.35,
    sectors: dict[str, str] | None = None,
    ranking_positions: dict[str, int] | None = None,
    n_points: int = 20,
) -> list[dict]:
    """
    Compute the efficient frontier by optimizing at multiple target volatilities.

    Returns list of {vol, ret, sharpe, weights} dicts for plotting.
    """
    tickers = [t for t in covariance.columns if t in expected_returns]
    if not tickers:
        return []

    # Find vol range
    # Min vol portfolio
    min_var_result = optimize_weights(
        expected_returns, covariance, risk_free_rate,
        max_weight, 0.0, max_sector_weight, sectors, "min_variance"
    )
    min_vol = min_var_result.portfolio_vol

    # Max vol: equal-weight of highest vol stocks
    N = len(tickers)
    mu = np.array([expected_returns[t] - risk_free_rate for t in tickers])
    Sigma = covariance.loc[tickers, tickers].values
    ew = np.ones(N) / N
    max_vol = np.sqrt(ew @ Sigma @ ew) * 1.3  # 30% above equal-weight

    # Generate frontier points
    vol_range = np.linspace(min_vol * 0.95, max_vol, n_points)
    frontier = []

    for target in vol_range:
        result = optimize_target_vol(
            expected_returns, covariance, float(target),
            risk_free_rate, max_weight, 0.0, max_sector_weight,
            sectors, ranking_positions,
        )
        if result.portfolio_vol > 0:
            frontier.append({
                "vol": result.portfolio_vol,
                "ret": result.portfolio_return,
                "sharpe": result.portfolio_sharpe,
                "n_positions": len(result.weights),
                "weights": result.weights,
            })

    return frontier


def scores_to_views(
    ranking_scores: dict[str, float],
    scale: float = 0.15,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Convert composite ranking scores to Black-Litterman views.

    Args:
        ranking_scores: ticker → composite score [-1, +1]
        scale: maximum expected excess return (±scale maps to ±1 score)

    Returns:
        (views, confidence) dicts ready for black_litterman()
    """
    views = {}
    confidence = {}

    for ticker, score in ranking_scores.items():
        # Map score to expected excess return
        # score=+1.0 → +scale annual excess return
        # score=-0.5 → -scale/2 excess return
        views[ticker] = score * scale

        # Confidence proportional to absolute score
        # Strong signal (|score| > 0.3) → high confidence
        # Weak signal (|score| < 0.1) → low confidence
        conf = min(abs(score) * 2, 0.95)
        conf = max(conf, 0.10)
        confidence[ticker] = conf

    return views, confidence
