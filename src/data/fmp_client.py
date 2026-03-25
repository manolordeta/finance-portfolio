"""
Financial Modeling Prep (FMP) API client — /stable/ endpoints.

Plan: Growth ($49/mo) — 750 calls/min, quarterly fundamentals, US coverage.
Docs: https://site.financialmodelingprep.com/developer/docs

NOTE: FMP migrated from /api/v3/ to /stable/ in Aug 2025.
      All new subscriptions MUST use /stable/ endpoints.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

FMP_BASE = "https://financialmodelingprep.com/stable"


class FMPClient:
    """REST client for Financial Modeling Prep /stable/ API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FMP API key required. Set FMP_API_KEY in .env or pass api_key=."
            )
        self._session = requests.Session()
        self._call_count = 0
        self._last_call_time = 0.0

    # ── Core request ──────────────────────────────────────────

    def _get(self, endpoint: str, params: dict | None = None) -> Any:
        """Authenticated GET to /stable/{endpoint}."""
        url = f"{FMP_BASE}/{endpoint}"
        p = {"apikey": self.api_key}
        if params:
            p.update(params)

        self._rate_limit()

        try:
            resp = self._session.get(url, params=p, timeout=30)
            resp.raise_for_status()
            self._call_count += 1
            data = resp.json()

            if isinstance(data, dict) and "Error Message" in data:
                logger.error("FMP error [%s]: %s", endpoint, data["Error Message"])
                return []

            return data

        except requests.exceptions.HTTPError as e:
            logger.error("FMP HTTP %s for %s: %s", resp.status_code, endpoint, e)
            raise
        except requests.exceptions.RequestException as e:
            logger.error("FMP request failed for %s: %s", endpoint, e)
            raise

    def _rate_limit(self) -> None:
        """Stay under 750 calls/min (~12.5/sec). We cap at ~10/sec."""
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self._last_call_time = time.time()

    # ── Company Profile ───────────────────────────────────────

    def get_profile(self, ticker: str) -> dict:
        """Company profile: name, sector, market cap, etc."""
        data = self._get("profile", {"symbol": ticker})
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return {}

    def get_profiles_batch(self, tickers: list[str]) -> list[dict]:
        """Batch profiles (comma-separated, max ~50)."""
        results = []
        for i in range(0, len(tickers), 50):
            batch = tickers[i : i + 50]
            data = self._get("profile", {"symbol": ",".join(batch)})
            if isinstance(data, list):
                results.extend(data)
        return results

    # ── Financial Statements (quarterly) ──────────────────────

    def get_income_statement(
        self, ticker: str, period: str = "quarterly", limit: int = 8,
    ) -> list[dict]:
        """Income statement. Key: revenue, netIncome, eps, fillingDate."""
        return self._get(
            "income-statement",
            {"symbol": ticker, "period": period, "limit": limit},
        )

    def get_balance_sheet(
        self, ticker: str, period: str = "quarterly", limit: int = 8,
    ) -> list[dict]:
        """Balance sheet. Key: totalAssets, totalDebt, totalEquity."""
        return self._get(
            "balance-sheet-statement",
            {"symbol": ticker, "period": period, "limit": limit},
        )

    def get_cash_flow(
        self, ticker: str, period: str = "quarterly", limit: int = 8,
    ) -> list[dict]:
        """Cash flow statement. Key: freeCashFlow, operatingCashFlow."""
        return self._get(
            "cash-flow-statement",
            {"symbol": ticker, "period": period, "limit": limit},
        )

    # ── Ratios & Metrics ──────────────────────────────────────

    def get_ratios(
        self, ticker: str, period: str = "quarterly", limit: int = 8,
    ) -> list[dict]:
        """Financial ratios: P/E, P/B, ROE, ROA, etc."""
        return self._get(
            "ratios",
            {"symbol": ticker, "period": period, "limit": limit},
        )

    def get_key_metrics(
        self, ticker: str, period: str = "quarterly", limit: int = 8,
    ) -> list[dict]:
        """Key metrics: EV/EBITDA, FCF yield, etc."""
        return self._get(
            "key-metrics",
            {"symbol": ticker, "period": period, "limit": limit},
        )

    def get_financial_growth(
        self, ticker: str, period: str = "quarterly", limit: int = 8,
    ) -> list[dict]:
        """Growth metrics: revenue growth, EPS growth, etc."""
        return self._get(
            "financial-growth",
            {"symbol": ticker, "period": period, "limit": limit},
        )

    # ── Earnings ──────────────────────────────────────────────

    def get_earnings_calendar(
        self, symbol: str | None = None, limit: int = 50,
        from_date: str | None = None, to_date: str | None = None,
    ) -> list[dict]:
        """
        Earnings calendar with actual vs estimated EPS/revenue.
        Use as earnings surprise source: epsActual vs epsEstimated.
        """
        params: dict[str, Any] = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._get("earnings-calendar", params)

    # ── Analyst Estimates ─────────────────────────────────────

    def get_analyst_estimates(
        self, ticker: str, period: str = "quarterly", limit: int = 8,
    ) -> list[dict]:
        """Analyst consensus: epsAvg, epsHigh, epsLow, revenueAvg, etc."""
        return self._get(
            "analyst-estimates",
            {"symbol": ticker, "period": period, "limit": limit},
        )

    # ── Stock Price ───────────────────────────────────────────

    def get_historical_price(
        self, ticker: str, from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict]:
        """Historical daily OHLCV."""
        params: dict[str, Any] = {"symbol": ticker}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = self._get("historical-price-eod/full", params)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("historical", [])
        return []

    def get_quote(self, ticker: str) -> dict:
        """Real-time quote: price, volume, market cap, P/E, etc."""
        data = self._get("quote", {"symbol": ticker})
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return {}

    # ── News (FMP articles) ───────────────────────────────────

    def get_fmp_articles(self, limit: int = 50) -> list[dict]:
        """
        FMP-authored market articles with analysis.
        NOTE: /stable/ plan doesn't have per-ticker stock_news.
              Use fmp-articles for general market intel.
        """
        return self._get("fmp-articles", {"limit": limit})

    # ── S&P 500 Constituents ──────────────────────────────────

    def get_sp500_constituents(self) -> list[dict]:
        """Current S&P 500 list: symbol, name, sector, etc."""
        return self._get("sp500-constituent")

    # ── Sector Performance ────────────────────────────────────

    def get_sector_performance(self) -> list[dict]:
        """Daily sector performance."""
        return self._get("sector-performance")

    # ── Utility ───────────────────────────────────────────────

    @property
    def calls_made(self) -> int:
        return self._call_count

    def test_connection(self) -> bool:
        """Quick test that the API key works."""
        try:
            data = self.get_profile("AAPL")
            ok = bool(data and data.get("symbol") == "AAPL")
            if ok:
                logger.info("FMP connection OK")
            return ok
        except Exception as e:
            logger.error("FMP connection failed: %s", e)
            return False
