"""
Timing Strategy Engine — Context-aware entry/exit conditions.

Unlike the barrier-based rotation, this strategy uses SIGNALS to decide
when to enter and exit, not fixed price thresholds.

Entry: top quintile + at least one timing condition met
Exit:  when N of M holding conditions turn negative

This can be backtested against the baseline (monthly rebalance, no timing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Timing Conditions ────────────────────────────────────────────────

def cond_ranking_improving(
    ranking_scores: pd.DataFrame, ticker: str, date_idx: int,
    lookback: int = 5,
) -> bool:
    """Ranking improved vs N days ago (momentum of ranking position)."""
    if date_idx < lookback:
        return False
    current = ranking_scores.iloc[date_idx].get(ticker)
    past = ranking_scores.iloc[date_idx - lookback].get(ticker)
    if current is None or past is None or not np.isfinite(current) or not np.isfinite(past):
        return False
    return current > past  # higher score = better ranking


def cond_vol_decreasing(
    returns: pd.DataFrame, ticker: str, date_idx: int,
    short_window: int = 10, long_window: int = 30,
) -> bool:
    """GARCH-like vol is decreasing (short-term vol < long-term vol)."""
    if date_idx < long_window:
        return False
    col = returns.iloc[max(0, date_idx - long_window):date_idx + 1][ticker].dropna()
    if len(col) < long_window:
        return False
    short_vol = col.iloc[-short_window:].std()
    long_vol = col.std()
    if long_vol == 0:
        return False
    return short_vol < long_vol * 0.9  # short vol at least 10% lower


def cond_earnings_surprise_recent(
    earnings_dates: dict[str, list[tuple[str, float]]],
    ticker: str, current_date: str,
    max_days_after: int = 21, min_surprise: float = 0.05,
) -> bool:
    """Recent earnings surprise > threshold (PEAD opportunity)."""
    if ticker not in earnings_dates:
        return False
    current = pd.Timestamp(current_date)
    for earn_date_str, surprise in earnings_dates[ticker]:
        earn_date = pd.Timestamp(earn_date_str)
        days_since = (current - earn_date).days
        if 0 <= days_since <= max_days_after and surprise >= min_surprise:
            return True
    return False


def cond_price_above_sma50(
    prices: pd.DataFrame, ticker: str, date_idx: int,
) -> bool:
    """Price is above SMA50 (uptrend confirmation)."""
    if date_idx < 50:
        return False
    sma50 = prices[ticker].iloc[date_idx - 49:date_idx + 1].mean()
    current = prices[ticker].iloc[date_idx]
    if pd.isna(current) or pd.isna(sma50) or sma50 == 0:
        return False
    return current > sma50


def cond_fresh_golden_cross(
    prices: pd.DataFrame, ticker: str, date_idx: int,
    max_days_since: int = 10,
) -> bool:
    """SMA50 crossed above SMA200 within the last N days."""
    if date_idx < 200 + max_days_since:
        return False
    for d in range(max(0, date_idx - max_days_since), date_idx + 1):
        if d < 200:
            continue
        sma50_now = prices[ticker].iloc[d - 49:d + 1].mean()
        sma200_now = prices[ticker].iloc[d - 199:d + 1].mean()
        sma50_prev = prices[ticker].iloc[d - 50:d].mean()
        sma200_prev = prices[ticker].iloc[d - 200:d].mean()
        if (pd.notna(sma50_now) and pd.notna(sma200_now) and
            pd.notna(sma50_prev) and pd.notna(sma200_prev)):
            if sma50_now > sma200_now and sma50_prev <= sma200_prev:
                return True
    return False


# ── Exit Conditions ──────────────────────────────────────────────────

def exit_ranking_dropped(
    ranking_scores: pd.DataFrame, ticker: str, date_idx: int,
    threshold_pct: float = 0.40,
) -> bool:
    """Ticker dropped below top X% of ranking."""
    if date_idx >= len(ranking_scores):
        return False
    day_scores = ranking_scores.iloc[date_idx].dropna()
    if len(day_scores) == 0 or ticker not in day_scores.index:
        return True  # if we can't find it, exit
    score = day_scores[ticker]
    percentile = (day_scores > score).sum() / len(day_scores)
    return percentile > (1 - threshold_pct)  # below top X%


def exit_vol_spiking(
    returns: pd.DataFrame, ticker: str, date_idx: int,
    entry_vol: float, spike_threshold: float = 1.5,
) -> bool:
    """Current vol is X times higher than at entry."""
    if date_idx < 10:
        return False
    current_vol = returns[ticker].iloc[date_idx - 9:date_idx + 1].std()
    if not np.isfinite(current_vol) or entry_vol == 0:
        return False
    return current_vol > entry_vol * spike_threshold


def exit_price_below_sma50(
    prices: pd.DataFrame, ticker: str, date_idx: int,
) -> bool:
    """Price dropped below SMA50 (trend broken)."""
    if date_idx < 50:
        return False
    sma50 = prices[ticker].iloc[date_idx - 49:date_idx + 1].mean()
    current = prices[ticker].iloc[date_idx]
    if pd.isna(current) or pd.isna(sma50):
        return False
    return current < sma50 * 0.97  # 3% buffer to avoid whipsaw


# ── Backtest Engine ──────────────────────────────────────────────────

@dataclass
class TimingTrade:
    ticker: str
    entry_date: str
    entry_price: float
    entry_conditions: list[str]
    entry_vol: float
    rank_at_entry: int
    exit_date: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    return_pct: float = 0.0
    holding_days: int = 0


@dataclass
class TimingResult:
    trades: list[TimingTrade]
    config: dict
    # Metrics
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    hit_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_days: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    # Comparison
    baseline_return: float = 0.0
    baseline_sharpe: float = 0.0


def run_timing_backtest(
    prices: pd.DataFrame,
    ranking_scores: pd.DataFrame,
    earnings_data: dict[str, list[tuple[str, float]]] | None = None,
    n_positions: int = 10,
    min_entry_conditions: int = 2,
    exit_negative_threshold: int = 2,
    ranking_threshold: float = 0.20,
    exit_ranking_threshold: float = 0.40,
    vol_spike_threshold: float = 1.5,
    recheck_interval: int = 5,
    cost_per_trade_bps: float = 12,
    max_hold_days: int = 63,
) -> TimingResult:
    """
    Backtest the timing-based strategy.

    Entry: top quintile + min_entry_conditions timing conditions met
    Exit: exit_negative_threshold exit conditions met, OR max_hold_days
    """
    dates = prices.index
    returns = prices.pct_change()
    cost = cost_per_trade_bps / 10000

    trades: list[TimingTrade] = []
    open_positions: dict[str, TimingTrade] = {}
    equity_curve = [1.0]
    capital_per_slot = 1.0 / n_positions

    earnings = earnings_data or {}
    start_idx = 252  # 1 year warmup for SMA200

    for i in range(start_idx, len(dates)):
        date = dates[i]
        date_str = str(date.date()) if hasattr(date, 'date') else str(date)[:10]

        # ── Check exit conditions for open positions ──
        closed = []
        for ticker, trade in list(open_positions.items()):
            trade.holding_days += 1

            # Count negative exit conditions
            neg_conditions = 0
            exit_reasons = []

            if exit_ranking_dropped(ranking_scores, ticker, i, exit_ranking_threshold):
                neg_conditions += 1
                exit_reasons.append("ranking_drop")

            if exit_vol_spiking(returns, ticker, i, trade.entry_vol, vol_spike_threshold):
                neg_conditions += 1
                exit_reasons.append("vol_spike")

            if exit_price_below_sma50(prices, ticker, i):
                neg_conditions += 1
                exit_reasons.append("below_sma50")

            # Max hold
            if trade.holding_days >= max_hold_days:
                neg_conditions = exit_negative_threshold  # force exit
                exit_reasons.append("max_hold")

            if neg_conditions >= exit_negative_threshold:
                current_price = prices.loc[date, ticker]
                if pd.notna(current_price):
                    trade.exit_date = date_str
                    trade.exit_price = float(current_price)
                    trade.return_pct = (current_price - trade.entry_price) / trade.entry_price - cost
                    trade.exit_reason = "+".join(exit_reasons[:2])
                    closed.append(ticker)

        for t in closed:
            trades.append(open_positions.pop(t))

        # ── Check entry conditions ──
        slots = n_positions - len(open_positions)
        should_check = (i - start_idx) % recheck_interval == 0

        if slots > 0 and should_check and date in ranking_scores.index:
            day_scores = ranking_scores.loc[date].dropna()
            if len(day_scores) == 0:
                continue

            # Top quintile
            threshold_score = day_scores.quantile(1 - ranking_threshold)
            candidates = day_scores[day_scores >= threshold_score].sort_values(ascending=False)

            for ticker in candidates.index:
                if len(open_positions) >= n_positions:
                    break
                if ticker in open_positions:
                    continue
                if ticker not in prices.columns:
                    continue

                entry_price = prices.loc[date, ticker]
                if pd.isna(entry_price) or entry_price <= 0:
                    continue

                # Check timing conditions
                conditions_met = []

                if cond_ranking_improving(ranking_scores, ticker, i, lookback=5):
                    conditions_met.append("rank_improving")

                if cond_vol_decreasing(returns, ticker, i):
                    conditions_met.append("vol_decreasing")

                if cond_earnings_surprise_recent(earnings, ticker, date_str):
                    conditions_met.append("earnings_surprise")

                if cond_price_above_sma50(prices, ticker, i):
                    conditions_met.append("above_sma50")

                if cond_fresh_golden_cross(prices, ticker, i, max_days_since=15):
                    conditions_met.append("golden_cross")

                # Need minimum conditions to enter
                if len(conditions_met) < min_entry_conditions:
                    continue

                # Compute entry vol
                hist_ret = returns[ticker].iloc[max(0, i - 21):i].dropna()
                entry_vol = float(hist_ret.std()) if len(hist_ret) > 5 else 0.02

                rank = int((day_scores > day_scores[ticker]).sum()) + 1

                trade = TimingTrade(
                    ticker=ticker,
                    entry_date=date_str,
                    entry_price=float(entry_price),
                    entry_conditions=conditions_met,
                    entry_vol=entry_vol,
                    rank_at_entry=rank,
                )
                open_positions[ticker] = trade

        # ── Equity curve ──
        daily_pnl = 0.0
        for ticker, trade in open_positions.items():
            if ticker in returns.columns and date in returns.index:
                r = returns.loc[date, ticker]
                if np.isfinite(r):
                    daily_pnl += r * capital_per_slot
        equity_curve.append(equity_curve[-1] * (1 + daily_pnl))

    # Close remaining
    last_date = str(dates[-1].date()) if hasattr(dates[-1], 'date') else str(dates[-1])[:10]
    for ticker, trade in open_positions.items():
        if ticker in prices.columns:
            trade.exit_price = float(prices.iloc[-1][ticker])
            trade.exit_date = last_date
            trade.exit_reason = "open"
            trade.return_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
            trades.append(trade)

    # ── Compute baseline (monthly rebalance, no timing) ──
    baseline_eq = [1.0]
    for i in range(start_idx, len(dates)):
        date = dates[i]
        if date in ranking_scores.index:
            day_scores = ranking_scores.loc[date].dropna()
            if len(day_scores) > 0:
                threshold = day_scores.quantile(0.80)
                top_quintile = day_scores[day_scores >= threshold].index
                top_ret = returns.loc[date].reindex(top_quintile).mean()
                if np.isfinite(top_ret):
                    baseline_eq.append(baseline_eq[-1] * (1 + top_ret))
                else:
                    baseline_eq.append(baseline_eq[-1])
            else:
                baseline_eq.append(baseline_eq[-1])
        else:
            baseline_eq.append(baseline_eq[-1])

    # ── Metrics ──
    result = TimingResult(
        trades=trades,
        config={
            "n_positions": n_positions,
            "min_entry_conditions": min_entry_conditions,
            "exit_negative_threshold": exit_negative_threshold,
            "ranking_threshold": ranking_threshold,
            "exit_ranking_threshold": exit_ranking_threshold,
            "vol_spike_threshold": vol_spike_threshold,
            "max_hold_days": max_hold_days,
        },
    )

    completed = [t for t in trades if t.exit_reason != "open"]
    result.total_trades = len(completed)
    result.wins = sum(1 for t in completed if t.return_pct > 0)
    result.losses = sum(1 for t in completed if t.return_pct <= 0)
    result.hit_rate = result.wins / max(result.total_trades, 1)

    win_rets = [t.return_pct for t in completed if t.return_pct > 0]
    loss_rets = [t.return_pct for t in completed if t.return_pct <= 0]
    result.avg_win = np.mean(win_rets) if win_rets else 0
    result.avg_loss = np.mean(loss_rets) if loss_rets else 0
    result.avg_holding_days = np.mean([t.holding_days for t in completed]) if completed else 0

    eq = np.array(equity_curve)
    total_days = len(eq) - 1
    years = total_days / 252
    result.total_return = eq[-1] / eq[0] - 1
    result.annualized_return = (eq[-1] / eq[0]) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0

    daily_rets = pd.Series(eq).pct_change().dropna()
    if len(daily_rets) > 1 and daily_rets.std() > 0:
        result.sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    result.max_drawdown = float(dd.min())

    # Baseline metrics
    beq = np.array(baseline_eq)
    result.baseline_return = beq[-1] / beq[0] - 1
    base_daily = pd.Series(beq).pct_change().dropna()
    if len(base_daily) > 1 and base_daily.std() > 0:
        result.baseline_sharpe = (base_daily.mean() / base_daily.std()) * np.sqrt(252)

    return result
