"""
Point-in-Time Fundamental Signals.

Unlike fundamental.py which uses the LATEST data for all dates (look-ahead bias),
this module reconstructs what data was ACTUALLY AVAILABLE at each historical date
using filing_date from the database.

For each date t in the backtest:
  - Only uses fundamentals with filing_date <= t
  - Computes YoY growth by comparing the latest available quarter vs same quarter 1 year ago
  - Returns a DataFrame (dates × tickers) like technical signals, not a static dict

Requires: fundamentals with filing_date in the database (we have this).
Limitation: only covers dates where we have 5+ quarters of history (~18 months).
"""

from __future__ import annotations

import json
import logging
import sqlite3

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_fundamentals_history(db_path: str, tickers: list[str]) -> dict:
    """
    Load all fundamentals with filing dates from DB.
    Returns: {ticker: [(filing_date, period_date, data_dict), ...]} sorted by filing_date
    """
    conn = sqlite3.connect(db_path)
    result = {}

    for ticker in tickers:
        rows = conn.execute(
            """SELECT filing_date, period_date, data_json
               FROM fundamentals
               WHERE ticker = ? AND statement_type = 'income'
               ORDER BY filing_date ASC""",
            (ticker,)
        ).fetchall()

        parsed = []
        for filing_date, period_date, data_json in rows:
            try:
                data = json.loads(data_json) if isinstance(data_json, str) else data_json
                parsed.append((filing_date, period_date, data))
            except (json.JSONDecodeError, TypeError):
                continue

        if parsed:
            result[ticker] = parsed

    conn.close()
    return result


def _load_ratios_history(db_path: str, tickers: list[str]) -> dict:
    """Load all ratios with filing dates."""
    conn = sqlite3.connect(db_path)
    result = {}

    for ticker in tickers:
        rows = conn.execute(
            """SELECT filing_date, period_date, data_json
               FROM ratios
               WHERE ticker = ? ORDER BY filing_date ASC""",
            (ticker,)
        ).fetchall()

        parsed = []
        for filing_date, period_date, data_json in rows:
            try:
                data = json.loads(data_json) if isinstance(data_json, str) else data_json
                parsed.append((filing_date, period_date, data))
            except (json.JSONDecodeError, TypeError):
                continue

        if parsed:
            result[ticker] = parsed

    conn.close()
    return result


def _load_earnings_history(db_path: str, tickers: list[str]) -> dict:
    """Load all earnings with dates."""
    conn = sqlite3.connect(db_path)
    result = {}

    for ticker in tickers:
        rows = conn.execute(
            """SELECT earnings_date, eps_actual, eps_estimated
               FROM earnings
               WHERE ticker = ? AND eps_actual IS NOT NULL AND eps_estimated IS NOT NULL
               ORDER BY earnings_date ASC""",
            (ticker,)
        ).fetchall()

        if rows:
            result[ticker] = [(r[0], float(r[1]), float(r[2])) for r in rows]

    conn.close()
    return result


def _get_latest_as_of(history: list[tuple], as_of_date: str) -> dict | None:
    """Get the most recent filing with filing_date <= as_of_date."""
    latest = None
    for filing_date, period_date, data in history:
        if filing_date <= as_of_date:
            latest = data
        else:
            break  # sorted by filing_date, so we can stop
    return latest


def _get_yoy_pair_as_of(history: list[tuple], as_of_date: str) -> tuple | None:
    """
    Get the latest quarter AND the same quarter 1 year ago,
    both with filing_date <= as_of_date.
    Returns (latest_data, year_ago_data) or None.
    """
    available = [(fd, pd_, d) for fd, pd_, d in history if fd <= as_of_date]
    if len(available) < 5:  # need at least 5 quarters for YoY
        return None

    latest = available[-1]
    latest_period = latest[1]  # e.g., "2024-09-30"

    # Find same quarter 1 year ago
    # Look for period_date that's ~1 year before latest period
    try:
        latest_pd = pd.Timestamp(latest_period)
        target_pd = latest_pd - pd.DateOffset(years=1)

        best_match = None
        best_diff = None
        for fd, pd_, d in available[:-1]:
            diff = abs((pd.Timestamp(pd_) - target_pd).days)
            if diff < 120 and (best_diff is None or diff < best_diff):
                best_match = d
                best_diff = diff

        if best_match is not None:
            return (latest[2], best_match)
    except Exception:
        pass

    return None


def _normalize_cross_sectional(values: dict[str, float | None]) -> dict[str, float]:
    """Z-score across tickers, clip to [-1, +1]."""
    vals = [v for v in values.values() if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return {k: 0.0 for k in values}
    mu = np.mean(vals)
    sigma = np.std(vals)
    if sigma < 1e-10:
        return {k: 0.0 for k in values}
    result = {}
    for k, v in values.items():
        if v is None or not np.isfinite(v):
            result[k] = 0.0
        else:
            z = (v - mu) / sigma
            result[k] = float(np.clip(z / 3.0, -1.0, 1.0))
    return result


def compute_all_pit(
    db_path: str,
    tickers: list[str],
    dates: pd.DatetimeIndex,
    recompute_every: int = 21,
) -> dict[str, pd.DataFrame]:
    """
    Compute all fundamental signals point-in-time.

    For each date, only uses data with filing_date <= that date.
    Recomputes every `recompute_every` days (fundamentals don't change daily).

    Returns: {signal_name: DataFrame(dates × tickers)} with values in [-1, +1]
    """
    logger.info("Loading fundamental history for PIT computation...")

    income_hist = _load_fundamentals_history(db_path, tickers)
    ratios_hist = _load_ratios_history(db_path, tickers)
    earnings_hist = _load_earnings_history(db_path, tickers)

    logger.info("  Income history: %d tickers", len(income_hist))
    logger.info("  Ratios history: %d tickers", len(ratios_hist))
    logger.info("  Earnings history: %d tickers", len(earnings_hist))

    # Initialize output DataFrames
    signal_names = [
        "pe_relative", "ev_ebitda_relative", "fcf_yield", "roe",
        "gross_margin_delta", "earnings_surprise", "revenue_growth", "debt_equity_inv",
    ]
    results = {name: pd.DataFrame(np.nan, index=dates, columns=tickers) for name in signal_names}

    # Compute at each recompute point
    last_scores = {name: {} for name in signal_names}

    for i, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")

        if i % recompute_every == 0:
            # Recompute all signals for this date
            raw_scores = {name: {} for name in signal_names}

            for ticker in tickers:
                # ── PE relative ──
                if ticker in ratios_hist:
                    ratio_data = _get_latest_as_of(ratios_hist[ticker], date_str)
                    if ratio_data:
                        pe = ratio_data.get("priceToEarningsRatio") or ratio_data.get("priceEarningsRatio")
                        if pe is not None:
                            try:
                                pe_val = float(pe)
                                if pe_val > 0:
                                    raw_scores["pe_relative"][ticker] = -pe_val
                            except (ValueError, TypeError):
                                pass

                        # EV/EBITDA
                        ev = ratio_data.get("enterpriseValueMultiple")
                        if ev is not None:
                            try:
                                ev_val = float(ev)
                                if ev_val > 0:
                                    raw_scores["ev_ebitda_relative"][ticker] = -ev_val
                            except (ValueError, TypeError):
                                pass

                        # FCF yield
                        fcf = ratio_data.get("freeCashFlowPerShare")
                        if fcf is not None:
                            try:
                                raw_scores["fcf_yield"][ticker] = float(fcf)
                            except (ValueError, TypeError):
                                pass

                        # ROE
                        ni = ratio_data.get("netIncomePerShare")
                        eq = ratio_data.get("shareholdersEquityPerShare")
                        if ni is not None and eq is not None:
                            try:
                                ni_v = float(ni)
                                eq_v = float(eq)
                                if abs(eq_v) > 1e-6:
                                    raw_scores["roe"][ticker] = ni_v / eq_v
                            except (ValueError, TypeError):
                                pass

                        # Debt/Equity
                        de = ratio_data.get("debtToEquityRatio") or ratio_data.get("debtEquityRatio")
                        if de is not None:
                            try:
                                de_val = float(de)
                                if de_val >= 0:
                                    raw_scores["debt_equity_inv"][ticker] = -de_val
                            except (ValueError, TypeError):
                                pass

                # ── Revenue growth (YoY) ──
                if ticker in income_hist:
                    yoy = _get_yoy_pair_as_of(income_hist[ticker], date_str)
                    if yoy:
                        latest_data, year_ago_data = yoy
                        try:
                            rev_now = float(latest_data.get("revenue", 0) or 0)
                            rev_then = float(year_ago_data.get("revenue", 0) or 0)
                            if abs(rev_then) > 1e-6:
                                raw_scores["revenue_growth"][ticker] = (rev_now - rev_then) / abs(rev_then)
                        except (ValueError, TypeError):
                            pass

                        # Gross margin delta
                        try:
                            gp_now = float(latest_data.get("grossProfit", 0) or 0)
                            r_now = float(latest_data.get("revenue", 1) or 1)
                            gp_then = float(year_ago_data.get("grossProfit", 0) or 0)
                            r_then = float(year_ago_data.get("revenue", 1) or 1)
                            m_now = gp_now / r_now if r_now else 0
                            m_then = gp_then / r_then if r_then else 0
                            if abs(m_then) > 1e-6:
                                raw_scores["gross_margin_delta"][ticker] = m_now - m_then
                        except (ValueError, TypeError):
                            pass

                # ── Earnings surprise ──
                if ticker in earnings_hist:
                    # Find most recent earnings with date <= as_of_date
                    latest_earn = None
                    for earn_date, eps_a, eps_e in earnings_hist[ticker]:
                        if earn_date <= date_str:
                            latest_earn = (eps_a, eps_e)
                        else:
                            break
                    if latest_earn:
                        eps_a, eps_e = latest_earn
                        if abs(eps_e) > 1e-6:
                            raw_scores["earnings_surprise"][ticker] = (eps_a - eps_e) / abs(eps_e)

            # Normalize cross-sectionally
            for name in signal_names:
                last_scores[name] = _normalize_cross_sectional(raw_scores[name])

        # Fill in the DataFrame row
        for name in signal_names:
            for ticker, score in last_scores[name].items():
                if ticker in results[name].columns:
                    results[name].at[date, ticker] = score

    # Count coverage
    for name in signal_names:
        coverage = results[name].notna().any(axis=0).sum()
        logger.info("  %s: %d tickers with data", name, coverage)

    return results
