"""
Data fetcher with parquet-based local cache.

Cache strategy:
  - Each ticker gets its own parquet file: data/cache/{TICKER}.parquet
  - On fetch, check if file exists and if last modified < expiry_days
  - If stale or missing, download from yfinance and overwrite
  - Always returns a DataFrame with DatetimeIndex, columns = tickers
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, cache_dir: str = "data/cache", expiry_days: int = 1, frequency: str = "1d"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expiry_days = expiry_days
        self.frequency = frequency

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prices(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Return adjusted close prices for all tickers.
        Shape: (dates, tickers), DatetimeIndex.
        """
        end = end_date or datetime.today().strftime("%Y-%m-%d")
        frames: dict[str, pd.Series] = {}

        for ticker in tickers:
            frames[ticker] = self._get_ticker(ticker, start_date, end)

        prices = pd.DataFrame(frames).sort_index()
        prices.index = pd.to_datetime(prices.index)

        # Drop rows where ALL tickers are NaN (weekends already excluded by yfinance)
        prices.dropna(how="all", inplace=True)

        missing = prices.isnull().sum()
        if missing.any():
            logger.warning("NaN counts per ticker:\n%s", missing[missing > 0])

        return prices

    def get_returns(self, prices: pd.DataFrame, log_returns: bool = False) -> pd.DataFrame:
        """
        Compute daily returns from price DataFrame.
        log_returns=True  → log returns ln(P_t / P_{t-1})
        log_returns=False → simple returns (P_t - P_{t-1}) / P_{t-1}
        """
        if log_returns:
            import numpy as np
            return np.log(prices / prices.shift(1)).dropna()
        return prices.pct_change().dropna()

    # ------------------------------------------------------------------
    # Cache logic
    # ------------------------------------------------------------------

    def _get_ticker(self, ticker: str, start_date: str, end_date: str) -> pd.Series:
        cache_path = self.cache_dir / f"{ticker}.parquet"

        if self._cache_is_fresh(cache_path):
            logger.info("[cache] %s — loading from cache", ticker)
            df = pd.read_parquet(cache_path)
            # Trim to requested range in case cache spans a wider period
            series = df["Close"].squeeze()
            return series.loc[start_date:end_date]

        logger.info("[fetch] %s — downloading from yfinance", ticker)
        t = yf.Ticker(ticker)
        raw = t.history(
            start=start_date,
            end=end_date,
            interval=self.frequency,
            auto_adjust=True,
        )

        if raw.empty:
            logger.error("No data returned for %s", ticker)
            return pd.Series(name=ticker, dtype=float)

        raw.index = pd.to_datetime(raw.index.tz_localize(None))
        raw.to_parquet(cache_path)
        return raw["Close"].rename(ticker)

    def _cache_is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < timedelta(days=self.expiry_days)

    def invalidate_cache(self, tickers: list[str] | None = None) -> None:
        """Delete cache files. Pass None to clear all."""
        targets = (
            [self.cache_dir / f"{t}.parquet" for t in tickers]
            if tickers
            else list(self.cache_dir.glob("*.parquet"))
        )
        for p in targets:
            if p.exists():
                p.unlink()
                logger.info("Cache cleared: %s", p.name)
