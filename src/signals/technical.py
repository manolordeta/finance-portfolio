"""
Technical signals for cross-sectional stock ranking.

Each function takes price/volume data and returns a DataFrame of scores
normalized to [-1, +1] via cross-sectional z-score + winsorization.

Convention:
  +1 = strongly bullish signal
   0 = neutral
  -1 = strongly bearish signal
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Normalization helpers ────────────────────────────────────────────

def _zscore_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row (date) across all tickers."""
    mu = df.mean(axis=1)
    sigma = df.std(axis=1)
    sigma = sigma.replace(0, np.nan)
    return df.sub(mu, axis=0).div(sigma, axis=0)


def _winsorize_and_clip(df: pd.DataFrame, n_std: float = 3.0) -> pd.DataFrame:
    """Clip z-scores to [-n_std, n_std] then map to [-1, +1]."""
    clipped = df.clip(lower=-n_std, upper=n_std)
    return clipped / n_std


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score → winsorize → [-1, +1]."""
    return _winsorize_and_clip(_zscore_cross_sectional(raw))


# ── Technical Signals ────────────────────────────────────────────────

def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum 12-minus-1: retorno 12 meses menos retorno 1 mes.

    Hypothesis: medium-term momentum with short-term reversal filtered out.
    Literature: Jegadeesh & Titman (1993).
    Natural horizon: 21-63 days.
    """
    ret_12m = prices.pct_change(252)    # ~12 months
    ret_1m = prices.pct_change(21)      # ~1 month
    raw = ret_12m - ret_1m
    return normalize(raw)


def momentum_1m(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Short-term momentum: retorno ultimo mes.

    Hypothesis: short-term continuation / trend following.
    Natural horizon: 5-21 days.
    """
    raw = prices.pct_change(21)
    return normalize(raw)


def rsi_14(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    RSI (Relative Strength Index) 14 days, normalized.

    Hypothesis: mean reversion — overbought (RSI>70) and oversold (RSI<30) revert.
    Natural horizon: 5-21 days.
    Score: RSI=50 → 0, RSI>70 → negative (overbought = bearish for reversal),
           RSI<30 → positive (oversold = bullish for reversal).
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing (EWM with com=period-1)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Normalize: RSI 50 → 0, RSI 30 → +0.67, RSI 70 → -0.67
    # (inverted because low RSI = oversold = bullish for mean reversion)
    raw = -(rsi - 50) / 50  # maps [0, 100] to [+1, -1]
    return raw.clip(-1, 1)


def macd_signal(prices: pd.DataFrame,
                fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD minus Signal line, normalized by price.

    Hypothesis: trend change detection — MACD crossing signal = momentum shift.
    Natural horizon: 5-21 days.
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()

    # Normalize by price to make comparable across stocks
    raw = (macd - signal_line) / prices
    return normalize(raw)


def bollinger_position(prices: pd.DataFrame,
                       window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """
    Position within Bollinger Bands.

    Hypothesis: reversion from bands — price near upper band = overbought.
    Natural horizon: 5-21 days.
    Score: at lower band → +1 (bullish), at upper band → -1 (bearish).
    """
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()

    upper = sma + n_std * std
    lower = sma - n_std * std
    band_width = upper - lower

    # Position: 0 at lower, 1 at upper → invert for mean reversion signal
    position = (prices - lower) / band_width.replace(0, np.nan)

    # Map [0, 1] → [+1, -1] (inverted: low position = bullish)
    raw = -(position * 2 - 1)
    return raw.clip(-1, 1)


def volume_ratio(prices: pd.DataFrame, volumes: pd.DataFrame,
                 window: int = 20) -> pd.DataFrame:
    """
    Current volume relative to 20-day average.

    Hypothesis: unusual volume signals institutional interest / upcoming move.
    Natural horizon: 5-21 days.
    Note: directionally ambiguous — high volume isn't inherently bullish.
    Combined with momentum: high volume + positive momentum = bullish confirmation.
    """
    avg_vol = volumes.rolling(window=window).mean()
    raw = volumes / avg_vol.replace(0, np.nan)
    return normalize(raw)


# ── Convenience: compute all signals at once ─────────────────────────

ALL_SIGNALS = {
    "momentum_12_1": {"fn": "momentum_12_1", "needs_volume": False},
    "momentum_1m": {"fn": "momentum_1m", "needs_volume": False},
    "rsi_14": {"fn": "rsi_14", "needs_volume": False},
    "macd_signal": {"fn": "macd_signal", "needs_volume": False},
    "bollinger_position": {"fn": "bollinger_position", "needs_volume": False},
    "volume_ratio": {"fn": "volume_ratio", "needs_volume": True},
}


def compute_all(prices: pd.DataFrame,
                volumes: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """
    Compute all technical signals.

    Returns dict mapping signal_name → DataFrame(dates × tickers) with values in [-1, +1].
    """
    results = {}

    results["momentum_12_1"] = momentum_12_1(prices)
    results["momentum_1m"] = momentum_1m(prices)
    results["rsi_14"] = rsi_14(prices)
    results["macd_signal"] = macd_signal(prices)
    results["bollinger_position"] = bollinger_position(prices)

    if volumes is not None:
        results["volume_ratio"] = volume_ratio(prices, volumes)

    return results
