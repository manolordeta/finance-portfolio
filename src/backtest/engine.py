"""
Walk-Forward Backtesting Engine

Tests three hypotheses:
  A) Global: one set of weights for all tickers
  B) GICS: sector-specific weights (regularized with global)
  C) Clusters: correlation-cluster-specific weights (regularized with global)

Walk-forward design:
  - Rolling train window (12 months)
  - Non-overlapping test window (3 months)
  - Weights calibrated in train, applied frozen in test
  - Transaction costs applied
  - Rebalance at start of each test period (or monthly within)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for the walk-forward backtest."""
    train_months: int = 12
    test_months: int = 3
    rebalance_days: int = 21          # rebalance within test period
    cost_bps: float = 12.0            # round-trip cost in basis points
    top_pct: float = 0.20             # top quintile
    min_ic: float = 0.0               # min IC to include signal (0 = include all)
    n_clusters: int = 8               # for Model C
    regularization_alpha: float = 0.5 # blend with global weights
    min_signals: int = 5              # min signals per ticker to include


@dataclass
class PeriodResult:
    """Results for one train/test period."""
    period_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    # Per-model results
    model_a: dict = field(default_factory=dict)
    model_b: dict = field(default_factory=dict)
    model_c: dict = field(default_factory=dict)
    # Signal ICs
    signal_ics_global: dict = field(default_factory=dict)
    signal_ics_by_sector: dict = field(default_factory=dict)
    signal_ics_by_cluster: dict = field(default_factory=dict)
    # Cluster assignments
    cluster_labels: dict = field(default_factory=dict)


@dataclass
class BacktestResults:
    """Aggregate results across all periods."""
    periods: list[PeriodResult] = field(default_factory=list)
    config: BacktestConfig = field(default_factory=BacktestConfig)
    benchmark_returns: pd.Series = field(default_factory=pd.Series)


class WalkForwardEngine:
    """
    Walk-forward backtesting with three model hypotheses.

    Usage:
        engine = WalkForwardEngine(prices, signals_tech, signals_fund, signals_val, sectors)
        results = engine.run()
        results.summary()
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        signals_tech: dict[str, pd.DataFrame],
        signals_fund: dict[str, dict[str, float]],
        signals_val: dict[str, pd.DataFrame],
        sectors: dict[str, str],
        config: BacktestConfig | None = None,
    ):
        self.prices = prices
        self.signals_tech = signals_tech
        self.signals_fund = signals_fund
        self.signals_val = signals_val
        self.sectors = sectors
        self.config = config or BacktestConfig()

        # Compute returns (keep same index as prices, first row = NaN)
        self.returns = prices.pct_change()
        self.tickers = list(prices.columns)

        # Build signal matrix: (dates, tickers, signals)
        self._build_signal_matrix()

    def _build_signal_matrix(self):
        """Combine all signals into a unified structure."""
        self.signal_names = []
        self.signal_dfs = {}

        # Technical signals (already DataFrames with dates × tickers)
        for name, df in self.signals_tech.items():
            self.signal_names.append(f"T_{name}")
            self.signal_dfs[f"T_{name}"] = df.reindex(
                index=self.prices.index, columns=self.tickers
            )

        # Valuation signals (same format)
        for name, df in self.signals_val.items():
            self.signal_names.append(f"V_{name}")
            self.signal_dfs[f"V_{name}"] = df.reindex(
                index=self.prices.index, columns=self.tickers
            )

        # Fundamental signals (static dict → broadcast to all dates)
        for name, scores in self.signals_fund.items():
            self.signal_names.append(f"F_{name}")
            s = pd.Series(scores)
            df = pd.DataFrame(
                np.tile(s.values, (len(self.prices), 1)),
                index=self.prices.index,
                columns=s.index,
            ).reindex(columns=self.tickers)
            self.signal_dfs[f"F_{name}"] = df

        logger.info(
            "Signal matrix: %d signals × %d dates × %d tickers",
            len(self.signal_names), len(self.prices), len(self.tickers),
        )

    def _generate_periods(self) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate train/test period boundaries."""
        dates = self.prices.index
        train_days = self.config.train_months * 21  # approx trading days
        test_days = self.config.test_months * 21
        periods = []

        start = 0
        while start + train_days + test_days <= len(dates):
            train_start = dates[start]
            train_end = dates[start + train_days - 1]
            test_start = dates[start + train_days]
            test_end_idx = min(start + train_days + test_days - 1, len(dates) - 1)
            test_end = dates[test_end_idx]
            periods.append((train_start, train_end, test_start, test_end))
            start += test_days  # step by test_months (non-overlapping tests)

        return periods

    def _compute_forward_returns(self, horizon: int = 21) -> pd.DataFrame:
        """Forward returns at horizon h for IC computation."""
        return self.prices.pct_change(horizon).shift(-horizon)

    def _compute_ic(
        self, signal: pd.DataFrame, fwd_returns: pd.DataFrame,
        date_mask: pd.Series, ticker_mask: list[str] | None = None,
    ) -> float:
        """
        Compute average IC (Spearman rank correlation) over masked dates and tickers.
        """
        tickers = ticker_mask if ticker_mask else self.tickers
        if isinstance(date_mask, pd.Index):
            valid_dates = date_mask.intersection(signal.index).intersection(fwd_returns.index)
        else:
            valid_dates = date_mask[date_mask].index.intersection(signal.index).intersection(fwd_returns.index)
        if len(valid_dates) == 0:
            return 0.0
        sig = signal.loc[valid_dates, tickers].copy()
        ret = fwd_returns.loc[valid_dates, tickers].copy()

        ics = []
        for date in sig.index:
            s_row = sig.loc[date].dropna()
            r_row = ret.loc[date].dropna()
            common = s_row.index.intersection(r_row.index)
            if len(common) < 10:
                continue
            ic = s_row[common].corr(r_row[common], method="spearman")
            if np.isfinite(ic):
                ics.append(ic)

        return np.mean(ics) if ics else 0.0

    def _compute_weights_global(
        self, train_dates: pd.Index, fwd_returns: pd.DataFrame,
    ) -> dict[str, float]:
        """Model A: compute IC-proportional weights over all tickers."""
        weights = {}
        for name in self.signal_names:
            ic = self._compute_ic(self.signal_dfs[name], fwd_returns, train_dates)
            weights[name] = max(ic, self.config.min_ic)
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Equal weight fallback
            weights = {k: 1.0 / len(self.signal_names) for k in self.signal_names}
        return weights

    def _compute_weights_by_group(
        self, train_dates: pd.Index, fwd_returns: pd.DataFrame,
        group_map: dict[str, str],  # ticker → group_name
        global_weights: dict[str, float],
    ) -> dict[str, dict[str, float]]:
        """Model B/C: compute weights per group, regularized with global."""
        alpha = self.config.regularization_alpha
        groups = set(group_map.values())
        group_weights = {}

        for group in groups:
            group_tickers = [t for t, g in group_map.items() if g == group and t in self.tickers]
            if len(group_tickers) < 20:
                # Too few tickers → use global weights
                group_weights[group] = global_weights.copy()
                continue

            raw_weights = {}
            for name in self.signal_names:
                ic = self._compute_ic(
                    self.signal_dfs[name], fwd_returns, train_dates, group_tickers
                )
                raw_weights[name] = max(ic, 0.0)

            total = sum(raw_weights.values())
            if total > 0:
                raw_weights = {k: v / total for k, v in raw_weights.items()}
            else:
                raw_weights = global_weights.copy()

            # Regularize: blend with global
            blended = {}
            for name in self.signal_names:
                blended[name] = alpha * global_weights.get(name, 0) + (1 - alpha) * raw_weights.get(name, 0)

            # Re-normalize
            total = sum(blended.values())
            if total > 0:
                blended = {k: v / total for k, v in blended.items()}
            group_weights[group] = blended

        return group_weights

    def _compute_clusters(self, train_dates: pd.Index) -> dict[str, str]:
        """Compute spectral clusters from correlation matrix in train period."""
        common = train_dates.intersection(self.returns.index)
        train_returns = self.returns.loc[common].dropna(how="all")
        # Only use tickers with enough data
        valid = train_returns.columns[train_returns.notna().sum() > len(train_returns) * 0.5]
        if len(valid) < self.config.n_clusters * 3:
            # Not enough data for clustering — return all same cluster
            return {t: "cluster_0" for t in self.tickers}
        train_returns = train_returns[valid].dropna(axis=0, how="all")

        corr = train_returns.corr().fillna(0).values.copy()
        affinity = (corr + 1) / 2
        np.fill_diagonal(affinity, 1)

        # Handle NaN in affinity
        affinity = np.nan_to_num(affinity, nan=0.5)

        sc = SpectralClustering(
            n_clusters=self.config.n_clusters,
            affinity="precomputed",
            n_init=10,
            random_state=42,
        )
        labels = sc.fit_predict(affinity)

        cluster_map = {}
        for ticker, label in zip(valid, labels):
            cluster_map[ticker] = f"cluster_{label}"

        # Assign unclustered tickers to nearest cluster
        for t in self.tickers:
            if t not in cluster_map:
                cluster_map[t] = "cluster_0"  # default

        return cluster_map

    def _build_composite(
        self, date: pd.Timestamp, weights: dict[str, float],
    ) -> pd.Series:
        """Build composite score for all tickers on a given date."""
        scores = pd.DataFrame(index=self.tickers, dtype=float)
        for name, w in weights.items():
            if name in self.signal_dfs:
                if date in self.signal_dfs[name].index:
                    scores[name] = self.signal_dfs[name].loc[date] * w

        # Count available signals per ticker
        n_available = scores.notna().sum(axis=1)
        composite = scores.sum(axis=1)

        # Mask tickers with too few signals
        composite[n_available < self.config.min_signals] = np.nan
        return composite

    def _build_composite_by_group(
        self, date: pd.Timestamp,
        group_weights: dict[str, dict[str, float]],
        group_map: dict[str, str],
    ) -> pd.Series:
        """Build composite using group-specific weights."""
        composite = pd.Series(index=self.tickers, dtype=float)

        for ticker in self.tickers:
            group = group_map.get(ticker, list(group_weights.keys())[0])
            weights = group_weights.get(group, {})
            score = 0.0
            n = 0
            for name, w in weights.items():
                if name in self.signal_dfs and date in self.signal_dfs[name].index:
                    val = self.signal_dfs[name].loc[date].get(ticker)
                    if val is not None and np.isfinite(val):
                        score += val * w
                        n += 1
            composite[ticker] = score if n >= self.config.min_signals else np.nan

        return composite

    def _simulate_portfolio(
        self, composite_scores: pd.Series, test_dates: pd.Index,
        rebalance_days: int,
    ) -> dict:
        """
        Simulate equal-weight top-quintile portfolio over test period.
        Returns performance metrics.
        """
        valid = composite_scores.dropna().sort_values(ascending=False)
        n_top = max(1, int(len(valid) * self.config.top_pct))
        top_tickers = valid.index[:n_top].tolist()

        if not top_tickers:
            return {"return": 0.0, "sharpe": 0.0, "max_dd": 0.0, "n_stocks": 0}

        # Equal-weight portfolio returns
        common_dates = test_dates.intersection(self.returns.index)
        if len(common_dates) == 0:
            return {"return": 0.0, "ann_return": 0.0, "ann_vol": 0.0,
                    "sharpe": 0.0, "max_dd": 0.0, "n_stocks": 0,
                    "spread_q5_q1": 0.0, "equity_curve": pd.Series(dtype=float)}
        port_returns = self.returns.loc[common_dates].reindex(columns=top_tickers).mean(axis=1).dropna()

        # Apply transaction costs at start
        cost = self.config.cost_bps / 10000
        port_returns.iloc[0] -= cost * 2  # buy cost

        # Rebalance costs
        for i in range(rebalance_days, len(port_returns), rebalance_days):
            if i < len(port_returns):
                port_returns.iloc[i] -= cost  # turnover cost approx

        # Metrics
        total_return = (1 + port_returns).prod() - 1
        ann_factor = 252 / len(port_returns) if len(port_returns) > 0 else 1
        ann_return = (1 + total_return) ** ann_factor - 1
        ann_vol = port_returns.std() * np.sqrt(252) if len(port_returns) > 1 else 0
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Max drawdown
        cum = (1 + port_returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = dd.min()

        # Bottom quintile for spread
        bottom_tickers = valid.index[-n_top:].tolist()
        bottom_returns = self.returns.loc[common_dates].reindex(columns=bottom_tickers).mean(axis=1).dropna()
        bottom_total = (1 + bottom_returns).prod() - 1

        return {
            "return": total_return,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "n_stocks": len(top_tickers),
            "spread_q5_q1": total_return - bottom_total,
            "equity_curve": (1 + port_returns).cumprod(),
        }

    def run(self) -> BacktestResults:
        """Execute the full walk-forward backtest."""
        periods = self._generate_periods()
        fwd_returns = self._compute_forward_returns(21)

        logger.info("Walk-forward: %d periods, train=%dm test=%dm",
                     len(periods), self.config.train_months, self.config.test_months)

        # Benchmark (SPY or equal-weight market)
        if "SPY" in self.returns.columns:
            benchmark = self.returns["SPY"]
        else:
            benchmark = self.returns.mean(axis=1)

        results = BacktestResults(config=self.config, benchmark_returns=benchmark)

        for i, (tr_start, tr_end, te_start, te_end) in enumerate(periods):
            logger.info("  Period %d: train %s→%s | test %s→%s",
                        i + 1, tr_start.date(), tr_end.date(),
                        te_start.date(), te_end.date())

            train_dates = self.prices.index[(self.prices.index >= tr_start) & (self.prices.index <= tr_end)]
            test_dates = self.prices.index[(self.prices.index >= te_start) & (self.prices.index <= te_end)]

            if len(train_dates) < 100 or len(test_dates) < 20:
                logger.warning("  Skipping period %d: insufficient data", i + 1)
                continue

            period = PeriodResult(
                period_idx=i + 1,
                train_start=str(tr_start.date()),
                train_end=str(tr_end.date()),
                test_start=str(te_start.date()),
                test_end=str(te_end.date()),
            )

            # ── Model A: Global weights ──────────────────────
            global_weights = self._compute_weights_global(train_dates, fwd_returns)
            period.signal_ics_global = {
                name: self._compute_ic(self.signal_dfs[name], fwd_returns, train_dates)
                for name in self.signal_names
            }

            # Score at start of test using train weights
            composite_a = self._build_composite(te_start, global_weights)
            period.model_a = self._simulate_portfolio(
                composite_a, test_dates, self.config.rebalance_days
            )
            period.model_a["weights"] = global_weights

            # ── Model B: GICS sector weights ─────────────────
            gics_map = {t: self.sectors.get(t, "Unknown") for t in self.tickers}
            gics_weights = self._compute_weights_by_group(
                train_dates, fwd_returns, gics_map, global_weights
            )
            composite_b = self._build_composite_by_group(
                te_start, gics_weights, gics_map
            )
            period.model_b = self._simulate_portfolio(
                composite_b, test_dates, self.config.rebalance_days
            )
            period.model_b["weights"] = gics_weights

            # ── Model C: Cluster weights ─────────────────────
            cluster_map = self._compute_clusters(train_dates)
            period.cluster_labels = cluster_map
            cluster_weights = self._compute_weights_by_group(
                train_dates, fwd_returns, cluster_map, global_weights
            )
            composite_c = self._build_composite_by_group(
                te_start, cluster_weights, cluster_map
            )
            period.model_c = self._simulate_portfolio(
                composite_c, test_dates, self.config.rebalance_days
            )
            period.model_c["weights"] = cluster_weights

            # ── Benchmark ────────────────────────────────────
            spy_dates = test_dates.intersection(benchmark.index)
            spy_ret = benchmark.loc[spy_dates]
            spy_total = (1 + spy_ret).prod() - 1
            period.model_a["spy_return"] = spy_total
            period.model_b["spy_return"] = spy_total
            period.model_c["spy_return"] = spy_total

            logger.info("    A(global):  ret=%+.2f%% sharpe=%.2f spread=%.2f%%",
                        period.model_a["return"] * 100,
                        period.model_a["sharpe"],
                        period.model_a.get("spread_q5_q1", 0) * 100)
            logger.info("    B(GICS):    ret=%+.2f%% sharpe=%.2f spread=%.2f%%",
                        period.model_b["return"] * 100,
                        period.model_b["sharpe"],
                        period.model_b.get("spread_q5_q1", 0) * 100)
            logger.info("    C(cluster): ret=%+.2f%% sharpe=%.2f spread=%.2f%%",
                        period.model_c["return"] * 100,
                        period.model_c["sharpe"],
                        period.model_c.get("spread_q5_q1", 0) * 100)
            logger.info("    SPY:        ret=%+.2f%%", spy_total * 100)

            results.periods.append(period)

        return results


def print_summary(results: BacktestResults) -> None:
    """Print formatted summary of backtest results."""
    if not results.periods:
        print("No results to summarize.")
        return

    print(f"\n{'='*75}")
    print(f"  WALK-FORWARD BACKTEST RESULTS")
    print(f"  {len(results.periods)} periods | "
          f"train={results.config.train_months}m test={results.config.test_months}m | "
          f"costs={results.config.cost_bps}bps")
    print(f"{'='*75}")

    # Aggregate by model
    for model_name, model_key in [("A (Global)", "model_a"),
                                   ("B (GICS)", "model_b"),
                                   ("C (Clusters)", "model_c")]:
        returns = [getattr(p, model_key)["return"] for p in results.periods]
        sharpes = [getattr(p, model_key)["sharpe"] for p in results.periods]
        spreads = [getattr(p, model_key).get("spread_q5_q1", 0) for p in results.periods]
        spy_rets = [getattr(p, model_key).get("spy_return", 0) for p in results.periods]
        max_dds = [getattr(p, model_key)["max_dd"] for p in results.periods]

        # Compound total return
        total = 1.0
        spy_total = 1.0
        for r, s in zip(returns, spy_rets):
            total *= (1 + r)
            spy_total *= (1 + s)

        avg_sharpe = np.mean(sharpes)
        avg_spread = np.mean(spreads)
        worst_dd = min(max_dds)
        win_rate = sum(1 for r, s in zip(returns, spy_rets) if r > s) / len(returns)

        print(f"\n  Model {model_name}:")
        print(f"    Cumulative return:  {(total-1)*100:+.1f}%")
        print(f"    SPY cumulative:     {(spy_total-1)*100:+.1f}%")
        print(f"    Alpha:              {(total-spy_total)*100:+.1f}%")
        print(f"    Avg Sharpe (period):{avg_sharpe:+.2f}")
        print(f"    Avg spread Q5-Q1:   {avg_spread*100:+.2f}%")
        print(f"    Worst drawdown:     {worst_dd*100:.1f}%")
        print(f"    Win rate vs SPY:    {win_rate*100:.0f}%")

    # Period-by-period
    print(f"\n  {'─'*75}")
    print(f"  Period-by-period comparison:")
    print(f"  {'Period':>8s}  {'Test':>22s}  {'A(Global)':>10s}  {'B(GICS)':>10s}  {'C(Cluster)':>10s}  {'SPY':>8s}")
    print(f"  {'─'*75}")
    for p in results.periods:
        print(f"  {p.period_idx:>8d}  {p.test_start}→{p.test_end}  "
              f"{p.model_a['return']*100:>+9.1f}%  "
              f"{p.model_b['return']*100:>+9.1f}%  "
              f"{p.model_c['return']*100:>+9.1f}%  "
              f"{p.model_a.get('spy_return',0)*100:>+7.1f}%")

    # Signal IC summary (from last period)
    if results.periods:
        last = results.periods[-1]
        print(f"\n  {'─'*75}")
        print(f"  Signal ICs (last period, global):")
        sorted_ics = sorted(last.signal_ics_global.items(), key=lambda x: -x[1])
        for name, ic in sorted_ics:
            bar = "█" * int(abs(ic) * 100)
            sign = "+" if ic > 0 else "-"
            print(f"    {name:30s} {ic:+.4f} {sign}{bar}")
