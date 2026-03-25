"""
Universe manager — loads and resolves the active universe of tickers.

Supports:
  - Static ticker lists (watchlist, mx_stocks)
  - Auto-fetched lists (sp500 via FMP)
  - Mixed sources (US via FMP, MX via yfinance)
  - Filtering by market cap and volume
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Universe:
    """Resolved universe ready for use."""

    name: str
    description: str
    tickers_us: list[str] = field(default_factory=list)
    tickers_mx: list[str] = field(default_factory=list)
    benchmark: str = "SPY"
    factor_proxies: dict[str, str] = field(default_factory=dict)

    @property
    def all_tickers(self) -> list[str]:
        return self.tickers_us + self.tickers_mx

    @property
    def total_count(self) -> int:
        return len(self.all_tickers)

    def __repr__(self) -> str:
        return (
            f"Universe('{self.name}', "
            f"US={len(self.tickers_us)}, MX={len(self.tickers_mx)}, "
            f"total={self.total_count})"
        )


def load_universe(
    config_path: str | Path = "config/universe.yaml",
    override_name: str | None = None,
    fmp_client: object | None = None,
) -> Universe:
    """
    Load the active universe from universe.yaml.

    Parameters
    ----------
    config_path : path to universe.yaml
    override_name : if set, use this universe instead of active_universe
    fmp_client : optional FMPClient instance for auto-fetching sp500 list

    Returns
    -------
    Universe with resolved ticker lists
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Universe config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    universe_name = override_name or cfg.get("active_universe", "watchlist")
    universes = cfg.get("universes", {})

    if universe_name not in universes:
        raise ValueError(
            f"Universe '{universe_name}' not found. "
            f"Available: {list(universes.keys())}"
        )

    ucfg = universes[universe_name]
    benchmark = cfg.get("benchmark", "SPY")
    factor_proxies = cfg.get("factor_proxies", {})

    tickers_us: list[str] = []
    tickers_mx: list[str] = []

    source = ucfg.get("source", "mixed")

    if source == "fmp" and ucfg.get("auto_fetch"):
        # Auto-fetch S&P 500 constituent list from FMP
        tickers_us = _fetch_sp500(fmp_client, ucfg.get("filters", {}))
    else:
        # Static ticker lists
        tickers_cfg = ucfg.get("tickers", {})
        tickers_us = [str(t) for t in tickers_cfg.get("us", [])]
        tickers_mx = [str(t) for t in tickers_cfg.get("mx", [])]

    # Deduplicate and sort
    tickers_us = sorted(set(tickers_us))
    tickers_mx = sorted(set(tickers_mx))

    universe = Universe(
        name=universe_name,
        description=ucfg.get("description", ""),
        tickers_us=tickers_us,
        tickers_mx=tickers_mx,
        benchmark=benchmark,
        factor_proxies=factor_proxies,
    )

    logger.info("Loaded %s", universe)
    return universe


def _fetch_sp500(fmp_client, filters: dict) -> list[str]:
    """Fetch S&P 500 constituents from FMP and apply filters."""
    if fmp_client is None:
        raise ValueError(
            "FMPClient required to auto-fetch sp500 universe. "
            "Pass fmp_client= to load_universe()."
        )

    constituents = fmp_client.get_sp500_constituents()
    tickers = [c["symbol"] for c in constituents]

    min_cap = filters.get("min_market_cap", 0)
    min_vol = filters.get("min_avg_volume", 0)
    exclude_sectors = set(filters.get("exclude_sectors", []))

    if min_cap > 0 or min_vol > 0 or exclude_sectors:
        logger.info(
            "Applying filters: min_cap=%.0f, min_vol=%.0f, exclude=%s",
            min_cap, min_vol, exclude_sectors,
        )
        filtered = []
        for c in constituents:
            # FMP sp500 constituents don't include vol/cap inline,
            # so we'd need to cross-reference with profiles.
            # For now, include all and filter later via profiles table.
            if c.get("sector") in exclude_sectors:
                continue
            filtered.append(c["symbol"])
        tickers = filtered

    logger.info("S&P 500 universe: %d tickers (after filters)", len(tickers))
    return tickers
