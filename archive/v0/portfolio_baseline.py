"""
Portfolio baseline: convert ranking into investable portfolio and backtest.

Tests the critical question: does our ranking generate alpha after costs?

Approach:
  - At each rebalance date, buy equal-weight top quintile of the ranking
  - Apply caps (max per name, max per sector)
  - Simulate with transaction costs (spread + slippage)
  - Compare vs SPY buy & hold and simple factor baselines
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class BacktestConfig:
    """Configuration for portfolio backtest."""
    top_pct: float = 0.20            # top quintile
    rebalance_freq: int = 21         # days (monthly)
    max_weight_per_name: float = 0.05  # 5% max per stock
    max_weight_per_sector: float = 0.30  # 30% max per sector
    cost_per_trade_bps: float = 12   # 12 bps per side (spread + slippage)
    initial_capital: float = 100_000


@dataclass
class BacktestResult:
    """Complete backtest output."""
    # Portfolio series
    portfolio_value: pd.Series       # daily portfolio value
    benchmark_value: pd.Series       # daily benchmark value

    # Summary metrics
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    hit_rate_monthly: float          # % of months with positive return

    # Benchmark comparison
    benchmark_return: float
    benchmark_sharpe: float
    alpha_annualized: float

    # Costs
    total_turnover: float            # annualized
    total_costs: float               # annualized drag
    gross_return: float              # before costs
    net_return: float                # after costs

    # Holdings info
    avg_holdings: float
    rebalance_dates: list

    def summary(self) -> str:
        lines = [
            "═" * 65,
            "  PORTFOLIO BACKTEST RESULTS",
            "═" * 65,
            "",
            f"  Period:           {self.portfolio_value.index[0].date()} → {self.portfolio_value.index[-1].date()}",
            f"  Rebalance dates:  {len(self.rebalance_dates)}",
            f"  Avg holdings:     {self.avg_holdings:.0f} stocks",
            "",
            "  RETURNS:",
            f"    Gross return (ann.):  {self.gross_return:+.1%}",
            f"    Net return (ann.):    {self.net_return:+.1%}",
            f"    Benchmark (SPY):      {self.benchmark_return:+.1%}",
            f"    Alpha (ann.):         {self.alpha_annualized:+.1%}",
            "",
            "  RISK:",
            f"    Volatility (ann.):    {self.annualized_vol:.1%}",
            f"    Sharpe ratio:         {self.sharpe_ratio:.2f}",
            f"    Benchmark Sharpe:     {self.benchmark_sharpe:.2f}",
            f"    Max drawdown:         {self.max_drawdown:.1%}",
            f"    Calmar ratio:         {self.calmar_ratio:.2f}",
            f"    Hit rate (monthly):   {self.hit_rate_monthly:.0%}",
            "",
            "  COSTS:",
            f"    Turnover (ann.):      {self.total_turnover:.0%}",
            f"    Cost drag (ann.):     {self.total_costs:.2%}",
            "",
        ]

        # Verdict
        if self.sharpe_ratio > self.benchmark_sharpe and self.alpha_annualized > 0:
            lines.append("  VERDICT: ✅ Portfolio beats benchmark after costs")
        elif self.gross_return > self.benchmark_return:
            lines.append("  VERDICT: ⚠️ Beats gross but costs may erode alpha")
        else:
            lines.append("  VERDICT: ❌ Does not beat benchmark")

        lines.append("═" * 65)
        return "\n".join(lines)


def _compute_drawdown(values: pd.Series) -> pd.Series:
    """Running drawdown from peak."""
    peak = values.expanding().max()
    return (values - peak) / peak


def backtest_top_quintile(
    composite_history: pd.DataFrame,
    prices: pd.DataFrame,
    sectors: dict[str, str] | None = None,
    benchmark: str = "SPY",
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Backtest equal-weight top quintile portfolio.

    Args:
        composite_history: DataFrame(dates × tickers) with composite scores.
                          Each row is the ranking for that date.
        prices: DataFrame(dates × tickers) with adjusted close prices.
        sectors: {ticker: sector} for sector caps.
        benchmark: benchmark ticker.
        config: backtest configuration.

    Returns:
        BacktestResult with all metrics.
    """
    cfg = config or BacktestConfig()

    # Align dates
    common_dates = composite_history.index.intersection(prices.index)
    common_tickers = composite_history.columns.intersection(prices.columns)
    composite_history = composite_history.loc[common_dates, common_tickers]

    # Remove benchmark from investable universe
    invest_tickers = [t for t in common_tickers if t != benchmark]

    # Rebalance dates
    rebal_dates = common_dates[::cfg.rebalance_freq]

    # Initialize
    capital = cfg.initial_capital
    portfolio_values = []
    holdings = {}  # ticker -> shares
    weights_history = []
    turnover_list = []
    costs_total = 0.0

    # Benchmark series
    bm_prices = prices[benchmark].reindex(common_dates)
    bm_start = bm_prices.iloc[0]

    prev_weights = {}

    for i, date in enumerate(common_dates):
        # Current portfolio value
        if holdings:
            current_prices = prices.loc[date, list(holdings.keys())]
            port_value = sum(
                holdings[t] * current_prices.get(t, 0)
                for t in holdings
            )
        else:
            port_value = capital

        portfolio_values.append({"date": date, "value": port_value})

        # Rebalance?
        if date not in rebal_dates:
            continue

        # Get ranking for this date
        scores = composite_history.loc[date, invest_tickers].dropna()
        if len(scores) < 10:
            continue

        # Top quintile
        threshold = scores.quantile(1 - cfg.top_pct)
        top_tickers = scores[scores >= threshold].index.tolist()
        n_top = len(top_tickers)

        if n_top == 0:
            continue

        # Equal weight with caps
        raw_weight = 1.0 / n_top
        target_weights = {}

        for t in top_tickers:
            w = min(raw_weight, cfg.max_weight_per_name)
            target_weights[t] = w

        # Sector cap
        if sectors:
            sector_weights = {}
            for t, w in target_weights.items():
                s = sectors.get(t, "Unknown")
                sector_weights[s] = sector_weights.get(s, 0) + w

            for t in list(target_weights.keys()):
                s = sectors.get(t, "Unknown")
                if sector_weights[s] > cfg.max_weight_per_sector:
                    # Scale down proportionally
                    scale = cfg.max_weight_per_sector / sector_weights[s]
                    target_weights[t] *= scale

        # Normalize weights to sum to 1
        total_w = sum(target_weights.values())
        if total_w > 0:
            target_weights = {t: w / total_w for t, w in target_weights.items()}

        # Compute turnover
        all_tickers_involved = set(target_weights.keys()) | set(prev_weights.keys())
        turnover = sum(
            abs(target_weights.get(t, 0) - prev_weights.get(t, 0))
            for t in all_tickers_involved
        ) / 2  # divide by 2 (buys + sells counted once)
        turnover_list.append(turnover)

        # Transaction costs
        cost = turnover * 2 * (cfg.cost_per_trade_bps / 10_000) * port_value
        costs_total += cost
        port_value -= cost

        # Update holdings
        holdings = {}
        for t, w in target_weights.items():
            p = prices.loc[date].get(t, 0)
            if p > 0:
                holdings[t] = (w * port_value) / p

        prev_weights = target_weights.copy()
        weights_history.append({"date": date, "n_holdings": len(holdings),
                                "turnover": turnover})

    # Build result series
    port_df = pd.DataFrame(portfolio_values).set_index("date")["value"]

    # Compute metrics
    port_returns = port_df.pct_change().dropna()
    bm_returns = bm_prices.pct_change().dropna()

    n_years = len(port_returns) / 252

    total_ret = (port_df.iloc[-1] / port_df.iloc[0]) - 1
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    dd = _compute_drawdown(port_df)
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Monthly hit rate
    monthly_ret = port_returns.resample("ME").sum()
    hit_rate = (monthly_ret > 0).mean()

    # Benchmark metrics
    bm_total = (bm_prices.iloc[-1] / bm_prices.iloc[0]) - 1
    bm_ann = (1 + bm_total) ** (1 / n_years) - 1 if n_years > 0 else 0
    bm_vol = bm_returns.std() * np.sqrt(252)
    bm_sharpe = bm_ann / bm_vol if bm_vol > 0 else 0

    # Costs
    avg_turnover = np.mean(turnover_list) if turnover_list else 0
    ann_turnover = avg_turnover * (252 / cfg.rebalance_freq)
    ann_cost_drag = ann_turnover * 2 * (cfg.cost_per_trade_bps / 10_000)

    # Gross return (add back costs)
    gross_ret = ann_ret + ann_cost_drag

    # Average holdings
    avg_hold = np.mean([w["n_holdings"] for w in weights_history]) if weights_history else 0

    bm_value = bm_prices / bm_start * cfg.initial_capital

    return BacktestResult(
        portfolio_value=port_df,
        benchmark_value=bm_value,
        total_return=total_ret,
        annualized_return=ann_ret,
        annualized_vol=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        hit_rate_monthly=hit_rate,
        benchmark_return=bm_ann,
        benchmark_sharpe=bm_sharpe,
        alpha_annualized=ann_ret - bm_ann,
        total_turnover=ann_turnover,
        total_costs=ann_cost_drag,
        gross_return=gross_ret,
        net_return=ann_ret,
        avg_holdings=avg_hold,
        rebalance_dates=[w["date"] for w in weights_history],
    )
