"""
GJR-GARCH(1,1) volatility modeling.

Estimates dynamic volatility σ(t) per stock for:
  - Position sizing (inversely proportional to σ)
  - Black-Litterman covariance matrix
  - VaR/CVaR computation
  - Risk-adjusted scoring
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GARCHResult:
    """Result of fitting GARCH to a single stock."""
    ticker: str
    omega: float
    alpha: float
    beta: float
    gamma: float  # leverage effect (GJR)
    current_vol: float  # annualized σ today
    forecast_vol_5d: float  # annualized σ forecast 5 days
    forecast_vol_21d: float  # annualized σ forecast 21 days
    persistence: float  # α + β + γ/2 (should be < 1)
    log_likelihood: float
    success: bool


def fit_garch(returns: pd.Series, ticker: str = "") -> GARCHResult:
    """
    Fit GJR-GARCH(1,1) to a return series.

    Uses the `arch` library. Falls back to simple historical vol
    if fitting fails.
    """
    try:
        from arch import arch_model

        # Clean returns
        r = returns.dropna()
        if len(r) < 100:
            raise ValueError(f"Insufficient data: {len(r)} observations")

        # Scale returns to percentage for numerical stability
        r_pct = r * 100

        # Fit GJR-GARCH(1,1)
        model = arch_model(r_pct, vol="GARCH", p=1, o=1, q=1, mean="Zero", dist="normal")
        result = model.fit(disp="off", show_warning=False)

        params = result.params
        omega = params.get("omega", 0)
        alpha = params.get("alpha[1]", 0)
        gamma = params.get("gamma[1]", 0)  # GJR leverage
        beta = params.get("beta[1]", 0)

        # Current conditional variance (last fitted value)
        cond_vol = result.conditional_volatility
        current_vol_daily = cond_vol.iloc[-1] / 100  # back to decimal
        current_vol_annual = current_vol_daily * np.sqrt(252)

        # Forecast
        forecasts = result.forecast(horizon=21)
        var_forecasts = forecasts.variance.iloc[-1].values / 10000  # back to decimal variance

        vol_5d = np.sqrt(var_forecasts[4]) * np.sqrt(252) if len(var_forecasts) > 4 else current_vol_annual
        vol_21d = np.sqrt(var_forecasts[20]) * np.sqrt(252) if len(var_forecasts) > 20 else current_vol_annual

        persistence = alpha + beta + gamma / 2

        return GARCHResult(
            ticker=ticker,
            omega=omega,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            current_vol=current_vol_annual,
            forecast_vol_5d=vol_5d,
            forecast_vol_21d=vol_21d,
            persistence=persistence,
            log_likelihood=result.loglikelihood,
            success=True,
        )

    except Exception as e:
        logger.warning("GARCH fit failed for %s: %s — using historical vol", ticker, e)
        r = returns.dropna()
        hist_vol = r.std() * np.sqrt(252) if len(r) > 20 else 0.30

        return GARCHResult(
            ticker=ticker,
            omega=0, alpha=0, beta=0, gamma=0,
            current_vol=hist_vol,
            forecast_vol_5d=hist_vol,
            forecast_vol_21d=hist_vol,
            persistence=0,
            log_likelihood=0,
            success=False,
        )


def fit_universe(returns: pd.DataFrame, tickers: list[str] | None = None) -> dict[str, GARCHResult]:
    """
    Fit GARCH for multiple stocks.

    Returns dict mapping ticker → GARCHResult.
    """
    tickers = tickers or list(returns.columns)
    results = {}

    for i, ticker in enumerate(tickers):
        if ticker not in returns.columns:
            continue
        results[ticker] = fit_garch(returns[ticker], ticker)
        if (i + 1) % 50 == 0:
            logger.info("  GARCH fitted %d/%d tickers", i + 1, len(tickers))

    succeeded = sum(1 for r in results.values() if r.success)
    logger.info("GARCH: %d/%d succeeded, %d fell back to historical vol",
                succeeded, len(results), len(results) - succeeded)

    return results


def build_covariance_matrix(
    returns: pd.DataFrame,
    garch_results: dict[str, GARCHResult],
    tickers: list[str],
    method: str = "shrinkage",
) -> pd.DataFrame:
    """
    Build covariance matrix using GARCH volatilities.

    Method:
      - 'sample': standard sample covariance (noisy with many tickers)
      - 'shrinkage': Ledoit-Wolf shrinkage (better conditioned)
      - 'garch_adjusted': sample correlations × GARCH vols (hybrid)
    """
    valid_tickers = [t for t in tickers if t in returns.columns and t in garch_results]
    r = returns[valid_tickers].dropna()

    if method == "shrinkage":
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(r)
            cov = pd.DataFrame(lw.covariance_ * 252, index=valid_tickers, columns=valid_tickers)
        except Exception:
            cov = r.cov() * 252

    elif method == "garch_adjusted":
        # Sample correlations × GARCH daily vols → covariance
        corr = r.corr()
        daily_vols = pd.Series({t: garch_results[t].current_vol / np.sqrt(252) for t in valid_tickers})
        # Cov = diag(σ) @ Corr @ diag(σ) × 252
        D = np.diag(daily_vols.values)
        cov_matrix = D @ corr.values @ D * 252
        cov = pd.DataFrame(cov_matrix, index=valid_tickers, columns=valid_tickers)

    else:  # sample
        cov = r.cov() * 252

    return cov
