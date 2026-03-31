"""
Hard baselines for signal comparison.

These represent the "dumb" strategies our signals must beat.
If our composite can't beat momentum_simple or value_simple,
we don't have an edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(prices: pd.DataFrame,
                 benchmark: str = "SPY") -> pd.Series:
    """
    Buy & hold benchmark returns.
    The simplest and hardest to beat baseline.
    """
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark {benchmark} not in prices")
    return prices[benchmark].pct_change()


def momentum_simple(prices: pd.DataFrame,
                    lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """
    Simple momentum signal: 12-month return, skip most recent month.

    Literature: Jegadeesh & Titman (1993).
    This is the single hardest baseline for most signal systems.
    """
    ret_full = prices.pct_change(lookback)
    ret_skip = prices.pct_change(skip)
    return ret_full - ret_skip


def value_simple(prices: pd.DataFrame,
                 pe_ratios: dict[str, float] | pd.Series | None = None,
                 fcf_yields: dict[str, float] | pd.Series | None = None) -> pd.DataFrame:
    """
    Simple value signal: inverse P/E or FCF yield.

    Uses FCF yield if available, falls back to inverse P/E.
    Static for now (latest available ratio), applied to all dates.
    """
    if fcf_yields is not None:
        vals = pd.Series(fcf_yields) if isinstance(fcf_yields, dict) else fcf_yields
    elif pe_ratios is not None:
        vals = pd.Series(pe_ratios) if isinstance(pe_ratios, dict) else pe_ratios
        vals = 1.0 / vals.replace(0, np.nan)  # inverse P/E
    else:
        raise ValueError("Need either pe_ratios or fcf_yields")

    # Broadcast static values across all dates
    signal = pd.DataFrame(
        np.tile(vals.values, (len(prices), 1)),
        index=prices.index,
        columns=vals.index,
    )
    return signal.reindex(columns=prices.columns)


def quality_simple(roe_scores: dict[str, float] | pd.Series,
                   leverage_scores: dict[str, float] | pd.Series,
                   prices: pd.DataFrame) -> pd.DataFrame:
    """
    Simple quality signal: ROE rank + low leverage rank, averaged.

    Literature: Novy-Marx (2013), Asness et al. (2019).
    """
    roe = pd.Series(roe_scores) if isinstance(roe_scores, dict) else roe_scores
    lev = pd.Series(leverage_scores) if isinstance(leverage_scores, dict) else leverage_scores

    # Rank each (higher = better for both: high ROE, low leverage already inverted)
    roe_rank = roe.rank(pct=True)
    lev_rank = lev.rank(pct=True)

    combined = (roe_rank + lev_rank) / 2.0

    # Broadcast across dates
    signal = pd.DataFrame(
        np.tile(combined.values, (len(prices), 1)),
        index=prices.index,
        columns=combined.index,
    )
    return signal.reindex(columns=prices.columns)


def equal_weight_factor_combo(
    momentum_sig: pd.DataFrame,
    value_sig: pd.DataFrame | None = None,
    quality_sig: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Equal-weight combination of simple factor signals.

    This is the composite baseline: if our smart composite can't beat
    this dumb combination, we haven't added value.
    """
    signals = [momentum_sig]
    if value_sig is not None:
        signals.append(value_sig)
    if quality_sig is not None:
        signals.append(quality_sig)

    # Rank-normalize each, then average
    ranked = []
    for sig in signals:
        ranked.append(sig.rank(axis=1, pct=True))

    return sum(ranked) / len(ranked)
