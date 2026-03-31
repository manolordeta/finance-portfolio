"""
Price valuation signals — is the current price a good entry/exit point?

Unlike technical signals (cross-sectional ranking), these compare each stock's
price against its OWN history. They answer: "is THIS stock cheap/expensive
relative to where it's been?"

Can be used both cross-sectionally (normalized) and per-stock (absolute).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.technical import normalize


def price_vs_sma200(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Price / 200-day SMA - 1.

    Below SMA200 → negative (potential buy, stock is "cheap" vs trend).
    Above SMA200 → positive (stock is above long-term trend).

    Inverted for value signal: below average = bullish.
    Hypothesis: mean reversion to long-term trend.
    Natural horizon: 21-63 days.
    """
    sma200 = prices.rolling(200, min_periods=150).mean()
    raw = -(prices / sma200 - 1)  # inverted: below SMA = positive score
    return normalize(raw)


def price_vs_sma50(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Price / 50-day SMA - 1.

    Same logic as SMA200 but shorter term.
    Natural horizon: 5-21 days.
    """
    sma50 = prices.rolling(50, min_periods=40).mean()
    raw = -(prices / sma50 - 1)
    return normalize(raw)


def drawdown_from_high(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Drawdown from 52-week high: (price - max) / max.

    Large drawdown (e.g., -30%) could be:
    - Opportunity (oversold, will recover)
    - Danger (falling knife, fundamentals broken)

    We score it as bullish (large drawdown = buying opportunity)
    but this signal MUST be combined with fundamental quality to avoid traps.

    Hypothesis: stocks with large drawdowns but intact fundamentals revert.
    Natural horizon: 21-63 days.
    """
    rolling_max = prices.rolling(window, min_periods=int(window * 0.7)).max()
    drawdown = (prices - rolling_max) / rolling_max  # always <= 0
    # Invert: large drawdown → high positive score (potential opportunity)
    raw = -drawdown  # maps e.g. -0.30 → +0.30
    return normalize(raw)


def distance_from_low(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Distance from 52-week low: (price - min) / min.

    Near the low → potentially oversold (positive for buying).
    Far from the low → already recovered, less upside.

    Inverted: close to low = bullish.
    Natural horizon: 21-63 days.
    """
    rolling_min = prices.rolling(window, min_periods=int(window * 0.7)).min()
    distance = (prices - rolling_min) / rolling_min  # always >= 0
    # Invert: close to low → high score
    raw = -distance
    return normalize(raw)


def price_vs_21d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Price / 21-day average - 1.

    Short-term mean reversion: if price dropped below its 21-day average,
    it may bounce back.

    Inverted: below average = bullish for mean reversion.
    Natural horizon: 5-21 days.
    """
    sma21 = prices.rolling(21, min_periods=15).mean()
    raw = -(prices / sma21 - 1)
    return normalize(raw)


def mean_reversion_63d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score of price vs its own 63-day (3-month) rolling mean and std.

    High z-score → overextended above mean (bearish for reversion).
    Low z-score → oversold below mean (bullish for reversion).

    Inverted: low z-score = bullish.
    Natural horizon: 21-63 days.
    """
    sma63 = prices.rolling(63, min_periods=50).mean()
    std63 = prices.rolling(63, min_periods=50).std()
    z = (prices - sma63) / std63.replace(0, np.nan)
    raw = -z  # inverted: oversold = positive
    return normalize(raw)


def golden_cross(prices: pd.DataFrame) -> pd.DataFrame:
    """
    SMA50 / SMA200 - 1.

    Positive → SMA50 above SMA200 ("golden cross" territory, bullish trend).
    Negative → SMA50 below SMA200 ("death cross" territory, bearish trend).

    NOT inverted: golden cross = bullish.
    Natural horizon: 21-63 days.
    """
    sma50 = prices.rolling(50, min_periods=40).mean()
    sma200 = prices.rolling(200, min_periods=150).mean()
    raw = sma50 / sma200 - 1
    return normalize(raw)


# ── Convenience ──────────────────────────────────────────────────────

ALL_SIGNALS = {
    "price_vs_sma200": price_vs_sma200,
    "price_vs_sma50": price_vs_sma50,
    "drawdown_from_high": drawdown_from_high,
    "distance_from_low": distance_from_low,
    "price_vs_21d": price_vs_21d,
    "mean_reversion_63d": mean_reversion_63d,
    "golden_cross": golden_cross,
}


def compute_all(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute all valuation signals. Returns {name: DataFrame(dates × tickers)}."""
    results = {}
    for name, fn in ALL_SIGNALS.items():
        results[name] = fn(prices)
    return results
