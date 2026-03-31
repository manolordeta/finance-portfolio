"""
Rotation Strategy Engine — Short-term GARCH-scaled barrier trading.

Core idea:
  1. Buy top-ranked stocks from the composite ranking
  2. Set take-profit and stop-loss barriers scaled to each stock's GARCH volatility
  3. Exit when any barrier is hit (tp, sl, or time limit)
  4. Rotate capital into the next opportunity

Barriers are scaled by volatility:
  tp = k_tp * sigma_daily * sqrt(expected_holding_days)
  sl = k_sl * sigma_daily * sqrt(expected_holding_days)

Where k_tp and k_sl are universal parameters optimized via backtesting.

From optimal stopping theory (GBM with two absorbing barriers):
  P(tp first) = (1 - e^(-2μ·sl/σ²)) / (1 - e^(-2μ·(tp+sl)/σ²))
  If μ > 0 (positive drift from ranking), P(tp) > 0.5 → edge exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """A single trade in the rotation strategy."""
    ticker: str
    entry_date: str
    entry_price: float
    tp_pct: float           # take profit barrier (e.g., 0.05 = +5%)
    sl_pct: float           # stop loss barrier (e.g., 0.03 = -3%)
    max_days: int           # time limit
    sigma_daily: float      # GARCH daily vol at entry
    rank_at_entry: int      # composite ranking position at entry
    # Filled on exit
    exit_date: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""   # "tp" | "sl" | "time" | "open"
    return_pct: float = 0.0
    holding_days: int = 0


@dataclass
class RotationResult:
    """Results of a rotation strategy backtest."""
    trades: list[Trade]
    k_tp: float
    k_sl: float
    max_days: int
    n_positions: int
    # Computed metrics
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    hit_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    avg_holding_days: float = 0.0
    exits_by_tp: int = 0
    exits_by_sl: int = 0
    exits_by_time: int = 0
    cost_total: float = 0.0


def compute_garch_vol(returns: pd.Series, window: int = 252) -> float:
    """
    Simple GARCH(1,1)-like daily volatility estimate.
    Uses exponentially weighted std for speed in backtesting.
    """
    if len(returns) < 21:
        return returns.std() if len(returns) > 1 else 0.02
    ewm_vol = returns.ewm(span=21, min_periods=10).std().iloc[-1]
    return float(ewm_vol) if np.isfinite(ewm_vol) and ewm_vol > 0 else 0.02


def run_backtest(
    prices: pd.DataFrame,
    ranking_scores: pd.DataFrame,
    k_tp: float = 1.5,
    k_sl: float = 1.0,
    max_days: int = 15,
    n_positions: int = 10,
    recheck_interval: int = 5,
    cost_per_trade_bps: float = 12,
    min_vol: float = 0.005,
    max_vol: float = 0.10,
) -> RotationResult:
    """
    Backtest the rotation strategy.

    Parameters:
      prices:           DataFrame of daily close prices (dates × tickers)
      ranking_scores:   DataFrame of daily composite scores (dates × tickers)
      k_tp:             take profit multiplier (tp = k_tp * σ * √max_days)
      k_sl:             stop loss multiplier (sl = k_sl * σ * √max_days)
      max_days:         maximum holding period in trading days
      n_positions:      max simultaneous positions
      recheck_interval: every N days, check for new entries if slots available
      cost_per_trade_bps: round-trip cost in basis points
      min_vol:          minimum daily vol (floor for very stable stocks)
      max_vol:          maximum daily vol (cap for extreme cases)

    Returns:
      RotationResult with all trades and metrics.
    """
    dates = prices.index
    returns = prices.pct_change()
    all_tickers = prices.columns.tolist()

    trades: list[Trade] = []
    open_positions: dict[str, Trade] = {}  # ticker → Trade
    equity_curve = [1.0]
    capital_per_slot = 1.0 / n_positions

    # We need some warmup for vol estimation
    start_idx = 63  # 3 months warmup

    for i in range(start_idx, len(dates)):
        date = dates[i]
        date_str = str(date.date()) if hasattr(date, 'date') else str(date)[:10]

        # ── Check exits for open positions ──
        closed_tickers = []
        for ticker, trade in list(open_positions.items()):
            if ticker not in prices.columns:
                continue

            current_price = prices.loc[date, ticker]
            if pd.isna(current_price):
                continue

            trade.holding_days += 1
            ret = (current_price - trade.entry_price) / trade.entry_price

            # Check barriers
            if ret >= trade.tp_pct:
                trade.exit_reason = "tp"
                trade.exit_price = current_price
                trade.exit_date = date_str
                trade.return_pct = ret
                closed_tickers.append(ticker)
            elif ret <= -trade.sl_pct:
                trade.exit_reason = "sl"
                trade.exit_price = current_price
                trade.exit_date = date_str
                trade.return_pct = ret
                closed_tickers.append(ticker)
            elif trade.holding_days >= trade.max_days:
                trade.exit_reason = "time"
                trade.exit_price = current_price
                trade.exit_date = date_str
                trade.return_pct = ret
                closed_tickers.append(ticker)

        for t in closed_tickers:
            trades.append(open_positions.pop(t))

        # ── Check for new entries ──
        slots_available = n_positions - len(open_positions)
        should_check = (i - start_idx) % recheck_interval == 0

        if slots_available > 0 and should_check:
            # Get today's ranking
            if date not in ranking_scores.index:
                continue

            day_scores = ranking_scores.loc[date].dropna()
            if len(day_scores) == 0:
                continue

            # Sort by score, exclude already held
            candidates = day_scores.drop(
                labels=[t for t in open_positions.keys() if t in day_scores.index],
                errors='ignore',
            ).sort_values(ascending=False)

            # Take top N candidates
            for ticker in candidates.head(slots_available * 2).index:
                if len(open_positions) >= n_positions:
                    break
                if ticker in open_positions:
                    continue
                if ticker not in prices.columns:
                    continue

                entry_price = prices.loc[date, ticker]
                if pd.isna(entry_price) or entry_price <= 0:
                    continue

                # Compute GARCH vol for this ticker
                hist_returns = returns[ticker].iloc[max(0, i - 252):i].dropna()
                sigma_daily = compute_garch_vol(hist_returns)
                sigma_daily = np.clip(sigma_daily, min_vol, max_vol)

                # Scale barriers by volatility
                tp_pct = k_tp * sigma_daily * np.sqrt(max_days)
                sl_pct = k_sl * sigma_daily * np.sqrt(max_days)

                # Sanity: min 1% tp, min 0.5% sl
                tp_pct = max(tp_pct, 0.01)
                sl_pct = max(sl_pct, 0.005)

                # Get rank
                rank = int((day_scores > day_scores[ticker]).sum()) + 1

                trade = Trade(
                    ticker=ticker,
                    entry_date=date_str,
                    entry_price=float(entry_price),
                    tp_pct=float(tp_pct),
                    sl_pct=float(sl_pct),
                    max_days=max_days,
                    sigma_daily=float(sigma_daily),
                    rank_at_entry=rank,
                )
                open_positions[ticker] = trade

        # ── Update equity curve ──
        daily_pnl = 0.0
        for ticker, trade in open_positions.items():
            if ticker in returns.columns and date in returns.index:
                r = returns.loc[date, ticker]
                if np.isfinite(r):
                    daily_pnl += r * capital_per_slot

        equity_curve.append(equity_curve[-1] * (1 + daily_pnl))

    # Close any remaining open positions at last price
    last_date = str(dates[-1].date()) if hasattr(dates[-1], 'date') else str(dates[-1])[:10]
    for ticker, trade in open_positions.items():
        if ticker in prices.columns:
            trade.exit_price = float(prices.iloc[-1][ticker])
            trade.exit_date = last_date
            trade.exit_reason = "open"
            trade.return_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
            trades.append(trade)

    # ── Compute metrics ──
    result = RotationResult(
        trades=trades,
        k_tp=k_tp,
        k_sl=k_sl,
        max_days=max_days,
        n_positions=n_positions,
    )

    if not trades:
        return result

    cost_per_trade = cost_per_trade_bps / 10000
    completed = [t for t in trades if t.exit_reason != "open"]

    result.total_trades = len(completed)
    result.wins = sum(1 for t in completed if t.return_pct > 0)
    result.losses = sum(1 for t in completed if t.return_pct <= 0)
    result.hit_rate = result.wins / max(result.total_trades, 1)

    win_returns = [t.return_pct - cost_per_trade for t in completed if t.return_pct > 0]
    loss_returns = [t.return_pct - cost_per_trade for t in completed if t.return_pct <= 0]

    result.avg_win = np.mean(win_returns) if win_returns else 0
    result.avg_loss = np.mean(loss_returns) if loss_returns else 0
    result.avg_holding_days = np.mean([t.holding_days for t in completed]) if completed else 0

    result.exits_by_tp = sum(1 for t in completed if t.exit_reason == "tp")
    result.exits_by_sl = sum(1 for t in completed if t.exit_reason == "sl")
    result.exits_by_time = sum(1 for t in completed if t.exit_reason == "time")

    # Cost
    result.cost_total = len(completed) * cost_per_trade

    # Equity curve metrics
    eq = np.array(equity_curve)
    total_days = len(eq) - 1
    result.total_return = eq[-1] / eq[0] - 1
    years = total_days / 252
    result.annualized_return = (eq[-1] / eq[0]) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0

    # Sharpe
    daily_returns = pd.Series(eq).pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        result.sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        result.sharpe = 0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    result.max_drawdown = float(dd.min())

    return result


def grid_search(
    prices: pd.DataFrame,
    ranking_scores: pd.DataFrame,
    k_tp_range: list[float] = [1.0, 1.5, 2.0, 2.5],
    k_sl_range: list[float] = [0.5, 0.75, 1.0, 1.5],
    max_days_range: list[int] = [10, 15, 21],
    n_positions: int = 10,
    cost_bps: float = 12,
) -> pd.DataFrame:
    """
    Grid search over (k_tp, k_sl, max_days) to find optimal parameters.

    Returns DataFrame sorted by Sharpe ratio.
    """
    results = []
    total = len(k_tp_range) * len(k_sl_range) * len(max_days_range)
    count = 0

    for k_tp in k_tp_range:
        for k_sl in k_sl_range:
            for max_days in max_days_range:
                count += 1
                if count % 10 == 0:
                    logger.info("  Grid search: %d/%d", count, total)

                r = run_backtest(
                    prices=prices,
                    ranking_scores=ranking_scores,
                    k_tp=k_tp,
                    k_sl=k_sl,
                    max_days=max_days,
                    n_positions=n_positions,
                    cost_per_trade_bps=cost_bps,
                )

                results.append({
                    "k_tp": k_tp,
                    "k_sl": k_sl,
                    "max_days": max_days,
                    "total_trades": r.total_trades,
                    "hit_rate": r.hit_rate,
                    "avg_win": r.avg_win,
                    "avg_loss": r.avg_loss,
                    "total_return": r.total_return,
                    "ann_return": r.annualized_return,
                    "sharpe": r.sharpe,
                    "max_drawdown": r.max_drawdown,
                    "avg_hold_days": r.avg_holding_days,
                    "tp_exits": r.exits_by_tp,
                    "sl_exits": r.exits_by_sl,
                    "time_exits": r.exits_by_time,
                    "expectancy": r.hit_rate * r.avg_win + (1 - r.hit_rate) * r.avg_loss,
                })

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    return df
