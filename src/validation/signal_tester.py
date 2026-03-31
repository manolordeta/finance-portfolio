"""
Signal validation framework implementing the Research Protocol.

Evaluates each signal against forward returns with:
  - Information Coefficient (IC)
  - Quintile/decile analysis
  - Factor attribution
  - Stability by regime
  - Turnover estimation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field


@dataclass
class SignalReport:
    """Standardized evaluation report for a single signal."""

    name: str
    horizon_days: int
    n_dates: int
    n_tickers_avg: float

    # IC metrics
    ic_mean: float
    ic_std: float
    ic_tstat: float
    ic_by_year: dict[int, float] = field(default_factory=dict)
    ic_bull: float | None = None
    ic_bear: float | None = None
    ic_high_vol: float | None = None
    ic_low_vol: float | None = None

    # Quintile analysis
    quintile_returns: list[float] = field(default_factory=list)
    spread_q5_q1: float = 0.0
    spread_annualized: float = 0.0
    monotonicity: int = 0       # out of n_groups-1
    n_groups: int = 5

    # Factor attribution
    ic_residual: float | None = None
    factor_exposures: dict[str, float] = field(default_factory=dict)

    # Turnover
    turnover_monthly: float | None = None

    @property
    def verdict(self) -> str:
        """PASS / MARGINAL / FAIL based on Research Protocol thresholds."""
        if self.ic_mean >= 0.03 and self.ic_tstat >= 2.0:
            if self.spread_q5_q1 > 0 and self.monotonicity >= (self.n_groups - 2):
                return "PASS"
            return "MARGINAL"
        return "FAIL"

    def summary(self) -> str:
        """Human-readable evaluation report."""
        v = self.verdict
        icon = {"PASS": "✅", "MARGINAL": "⚠️", "FAIL": "❌"}[v]

        lines = [
            "═" * 60,
            "SIGNAL EVALUATION REPORT",
            "═" * 60,
            f"",
            f"Signal:    {self.name}",
            f"Horizon:   {self.horizon_days} days",
            f"Dates:     {self.n_dates} cross-sections, ~{self.n_tickers_avg:.0f} tickers/date",
            f"",
            f"IC METRICS:",
            f"  IC mean:              {self.ic_mean:+.4f}  {'✓' if self.ic_mean >= 0.03 else '✗'} (threshold: 0.03)",
            f"  IC std:               {self.ic_std:.4f}",
            f"  IC t-stat:            {self.ic_tstat:.2f}  {'✓' if self.ic_tstat >= 2.0 else '✗'} (threshold: 2.0)",
        ]

        if self.ic_by_year:
            lines.append(f"  IC by year:")
            for yr, ic in sorted(self.ic_by_year.items()):
                lines.append(f"    {yr}: {ic:+.4f}")

        if self.ic_bull is not None:
            lines.append(f"  IC (bull market):     {self.ic_bull:+.4f}")
        if self.ic_bear is not None:
            lines.append(f"  IC (bear market):     {self.ic_bear:+.4f}")

        if self.ic_high_vol is not None:
            lines.append(f"  IC (high vol):        {self.ic_high_vol:+.4f}")
        if self.ic_low_vol is not None:
            lines.append(f"  IC (low vol):         {self.ic_low_vol:+.4f}")

        lines.append(f"")
        lines.append(f"QUINTILE ANALYSIS:")
        for i, ret in enumerate(self.quintile_returns, 1):
            lines.append(f"  Q{i}: {ret:+.4f} ({ret * 252 / self.horizon_days:+.1%} ann.)")
        lines.append(f"  Spread Q{self.n_groups}-Q1:       {self.spread_q5_q1:+.4f} "
                      f"({self.spread_annualized:+.1%} ann.)")
        lines.append(f"  Monotonicity:         {self.monotonicity}/{self.n_groups - 1}")

        if self.ic_residual is not None:
            lines.append(f"")
            lines.append(f"FACTOR ATTRIBUTION:")
            lines.append(f"  IC residual:          {self.ic_residual:+.4f}  "
                         f"{'✓' if self.ic_residual >= 0.02 else '✗'} (threshold: 0.02)")
            for fname, fval in self.factor_exposures.items():
                lines.append(f"  β_{fname:12s}:     {fval:+.4f}")

        if self.turnover_monthly is not None:
            lines.append(f"")
            lines.append(f"TURNOVER:")
            lines.append(f"  Monthly turnover:     {self.turnover_monthly:.1%}  "
                         f"{'✓' if self.turnover_monthly <= 0.60 else '✗'} (threshold: 60%)")

        lines.append(f"")
        lines.append(f"VERDICT:  {icon} {v}")
        lines.append("═" * 60)
        return "\n".join(lines)


class SignalTester:
    """
    Evaluate signals according to the Research Protocol.

    Usage:
        tester = SignalTester()
        report = tester.full_evaluation(
            signal_name="momentum_12_1",
            signal=signal_df,        # DataFrame(dates × tickers), values in [-1, +1]
            prices=prices_df,        # DataFrame(dates × tickers), adjusted close
            horizon=21,
        )
        print(report.summary())
    """

    def compute_forward_returns(self, prices: pd.DataFrame,
                                horizon: int = 21) -> pd.DataFrame:
        """Compute forward returns at horizon h for each date × ticker."""
        return prices.pct_change(horizon).shift(-horizon)

    # ── IC ────────────────────────────────────────────────────────

    def compute_ic(self, signal: pd.DataFrame,
                   forward_returns: pd.DataFrame) -> pd.Series:
        """
        Cross-sectional rank IC for each date.

        IC_t = spearman_corr(signal_t, fwd_return_t) across tickers.
        Returns a Series indexed by date.
        """
        common_dates = signal.index.intersection(forward_returns.index)
        common_tickers = signal.columns.intersection(forward_returns.columns)

        ics = {}
        for date in common_dates:
            s = signal.loc[date, common_tickers].dropna()
            r = forward_returns.loc[date, common_tickers].dropna()
            common = s.index.intersection(r.index)
            if len(common) < 5:
                continue
            corr, _ = stats.spearmanr(s[common], r[common])
            if np.isfinite(corr):
                ics[date] = corr

        return pd.Series(ics, dtype=float)

    def ic_stats(self, ic_series: pd.Series) -> dict:
        """Compute IC statistics: mean, std, t-stat."""
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        n = len(ic_series)
        ic_tstat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 and n > 1 else 0.0
        return {"ic_mean": ic_mean, "ic_std": ic_std, "ic_tstat": ic_tstat, "n": n}

    def ic_by_year(self, ic_series: pd.Series) -> dict[int, float]:
        """IC broken down by calendar year."""
        result = {}
        for year, group in ic_series.groupby(ic_series.index.year):
            if len(group) >= 5:
                result[year] = group.mean()
        return result

    # ── Quintile Analysis ────────────────────────────────────────

    def quintile_analysis(self, signal: pd.DataFrame,
                          forward_returns: pd.DataFrame,
                          n_groups: int = 5) -> dict:
        """
        Divide universe into n_groups by signal score each date.
        Compute average forward return per group.
        """
        common_dates = signal.index.intersection(forward_returns.index)
        common_tickers = signal.columns.intersection(forward_returns.columns)

        group_returns = {g: [] for g in range(1, n_groups + 1)}

        for date in common_dates:
            s = signal.loc[date, common_tickers].dropna()
            r = forward_returns.loc[date, common_tickers].dropna()
            common = s.index.intersection(r.index)
            if len(common) < n_groups * 2:
                continue

            s_sorted = s[common].rank(pct=True)
            for ticker in common:
                pct = s_sorted[ticker]
                group = min(int(pct * n_groups) + 1, n_groups)
                group_returns[group].append(r[ticker])

        avg_returns = []
        for g in range(1, n_groups + 1):
            vals = group_returns[g]
            avg_returns.append(np.mean(vals) if vals else 0.0)

        # Monotonicity: count how many consecutive groups increase
        mono = sum(1 for i in range(len(avg_returns) - 1)
                   if avg_returns[i + 1] > avg_returns[i])

        spread = avg_returns[-1] - avg_returns[0]

        return {
            "quintile_returns": avg_returns,
            "spread": spread,
            "monotonicity": mono,
            "n_groups": n_groups,
        }

    # ── Regime Analysis ──────────────────────────────────────────

    def ic_by_regime(self, ic_series: pd.Series,
                     benchmark_returns: pd.Series,
                     lookback: int = 252) -> dict:
        """
        Split IC into bull/bear and high/low vol regimes.

        Bull/Bear: trailing 12m benchmark return > 0 or < 0
        High/Low vol: trailing 63d realized vol above/below median
        """
        result = {}

        # Align dates
        common = ic_series.index.intersection(benchmark_returns.index)
        if len(common) < 20:
            return result

        bm = benchmark_returns.reindex(common)

        # Bull vs Bear: trailing 12m return
        trailing_ret = bm.rolling(lookback, min_periods=lookback // 2).sum()
        bull_dates = trailing_ret[trailing_ret > 0].index.intersection(ic_series.index)
        bear_dates = trailing_ret[trailing_ret <= 0].index.intersection(ic_series.index)

        if len(bull_dates) >= 5:
            result["ic_bull"] = ic_series.reindex(bull_dates).mean()
        if len(bear_dates) >= 5:
            result["ic_bear"] = ic_series.reindex(bear_dates).mean()

        # High vs Low vol: trailing 63d realized vol
        trailing_vol = bm.rolling(63, min_periods=30).std() * np.sqrt(252)
        median_vol = trailing_vol.median()
        high_vol_dates = trailing_vol[trailing_vol > median_vol].index.intersection(ic_series.index)
        low_vol_dates = trailing_vol[trailing_vol <= median_vol].index.intersection(ic_series.index)

        if len(high_vol_dates) >= 5:
            result["ic_high_vol"] = ic_series.reindex(high_vol_dates).mean()
        if len(low_vol_dates) >= 5:
            result["ic_low_vol"] = ic_series.reindex(low_vol_dates).mean()

        return result

    # ── Turnover ─────────────────────────────────────────────────

    def compute_turnover(self, signal: pd.DataFrame,
                         top_pct: float = 0.2,
                         rebalance_freq: int = 21) -> float:
        """
        Monthly turnover of the top quintile.

        Turnover = fraction of top quintile that changes each rebalance.
        """
        dates = signal.index[::rebalance_freq]
        turnovers = []

        prev_top = None
        for date in dates:
            row = signal.loc[date].dropna()
            if len(row) < 5:
                continue
            threshold = row.quantile(1 - top_pct)
            current_top = set(row[row >= threshold].index)

            if prev_top is not None and len(current_top) > 0:
                overlap = len(current_top & prev_top)
                turnover = 1 - overlap / max(len(current_top), 1)
                turnovers.append(turnover)

            prev_top = current_top

        return np.mean(turnovers) if turnovers else 0.0

    # ── Factor Attribution ───────────────────────────────────────

    def factor_attribution(self, signal: pd.DataFrame,
                           forward_returns: pd.DataFrame,
                           factor_signals: dict[str, pd.DataFrame]) -> dict:
        """
        Compute IC residual after controlling for known factors.

        Approach: for each date, regress signal on factor signals,
        take residual, compute IC of residual vs forward returns.
        """
        common_dates = signal.index.intersection(forward_returns.index)
        common_tickers = signal.columns.intersection(forward_returns.columns)

        residual_ics = []
        exposures = {name: [] for name in factor_signals}

        for date in common_dates:
            s = signal.loc[date, common_tickers].dropna()
            r = forward_returns.loc[date, common_tickers].dropna()
            common = s.index.intersection(r.index)

            # Build factor matrix for this date
            factor_vals = {}
            for fname, fsig in factor_signals.items():
                if date in fsig.index:
                    frow = fsig.loc[date, common_tickers].reindex(common)
                    factor_vals[fname] = frow

            if len(common) < 10 or not factor_vals:
                continue

            # Regress signal on factors
            F = pd.DataFrame(factor_vals).reindex(common).dropna()
            valid = F.index.intersection(s.index).intersection(r.index)
            if len(valid) < 10:
                continue

            F_valid = F.loc[valid]
            s_valid = s[valid]
            r_valid = r[valid]

            try:
                # OLS: signal = α + β'F + ε
                X = np.column_stack([np.ones(len(valid)), F_valid.values])
                betas = np.linalg.lstsq(X, s_valid.values, rcond=None)[0]
                residual = s_valid.values - X @ betas

                # IC of residual vs forward return
                corr, _ = stats.spearmanr(residual, r_valid.values)
                if np.isfinite(corr):
                    residual_ics.append(corr)

                # Store factor exposures (betas, skip intercept)
                for i, fname in enumerate(factor_vals.keys()):
                    if i + 1 < len(betas):
                        exposures[fname].append(betas[i + 1])
            except (np.linalg.LinAlgError, ValueError):
                continue

        result = {
            "ic_residual": np.mean(residual_ics) if residual_ics else None,
            "factor_exposures": {
                name: np.mean(vals) if vals else 0.0
                for name, vals in exposures.items()
            },
        }
        return result

    # ── Full Evaluation ──────────────────────────────────────────

    def full_evaluation(
        self,
        signal_name: str,
        signal: pd.DataFrame,
        prices: pd.DataFrame,
        horizon: int = 21,
        benchmark_ticker: str = "SPY",
        factor_signals: dict[str, pd.DataFrame] | None = None,
        n_groups: int = 5,
    ) -> SignalReport:
        """
        Run the complete Research Protocol evaluation.

        Returns a SignalReport with all metrics.
        """
        fwd_returns = self.compute_forward_returns(prices, horizon)

        # IC
        ic_series = self.compute_ic(signal, fwd_returns)
        ic = self.ic_stats(ic_series)
        ic_years = self.ic_by_year(ic_series)

        # Count average tickers per date
        n_tickers_avg = signal.notna().sum(axis=1).mean()

        # Quintile analysis
        qa = self.quintile_analysis(signal, fwd_returns, n_groups)

        # Regime analysis
        regime = {}
        if benchmark_ticker in prices.columns:
            bm_returns = prices[benchmark_ticker].pct_change().dropna()
            regime = self.ic_by_regime(ic_series, bm_returns)

        # Turnover
        turnover = self.compute_turnover(signal)

        # Factor attribution
        fa = {}
        if factor_signals:
            fa = self.factor_attribution(signal, fwd_returns, factor_signals)

        # Annualize spread
        periods_per_year = 252 / horizon
        spread_ann = qa["spread"] * periods_per_year

        return SignalReport(
            name=signal_name,
            horizon_days=horizon,
            n_dates=ic["n"],
            n_tickers_avg=n_tickers_avg,
            ic_mean=ic["ic_mean"],
            ic_std=ic["ic_std"],
            ic_tstat=ic["ic_tstat"],
            ic_by_year=ic_years,
            ic_bull=regime.get("ic_bull"),
            ic_bear=regime.get("ic_bear"),
            ic_high_vol=regime.get("ic_high_vol"),
            ic_low_vol=regime.get("ic_low_vol"),
            quintile_returns=qa["quintile_returns"],
            spread_q5_q1=qa["spread"],
            spread_annualized=spread_ann,
            monotonicity=qa["monotonicity"],
            n_groups=n_groups,
            ic_residual=fa.get("ic_residual"),
            factor_exposures=fa.get("factor_exposures", {}),
            turnover_monthly=turnover,
        )
