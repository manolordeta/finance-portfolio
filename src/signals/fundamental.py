"""
Fundamental signals for cross-sectional stock ranking.

Each function reads from the MarketDB (SQLite) and returns a dict
mapping ticker -> score in [-1, +1].

Convention:
  +1 = strongly bullish signal
   0 = neutral
  -1 = strongly bearish signal
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from src.data.database import MarketDB


# ── Normalization ────────────────────────────────────────────────────

def _normalize_dict(scores: dict[str, float | None]) -> dict[str, float]:
    """Cross-sectional z-score -> clip to [-1, +1]."""
    vals = [v for v in scores.values() if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return {k: 0.0 for k in scores}

    mu = np.mean(vals)
    sigma = np.std(vals)
    if sigma < 1e-10:
        return {k: 0.0 for k in scores}

    result = {}
    for k, v in scores.items():
        if v is None or not np.isfinite(v):
            result[k] = 0.0
        else:
            z = (v - mu) / sigma
            result[k] = float(np.clip(z / 3.0, -1.0, 1.0))
    return result


def _ratio_field(db: MarketDB, ticker: str, field: str) -> float | None:
    """Get a field from the latest ratios row."""
    row = db.get_latest_ratios(ticker)
    if row is None:
        return None
    data = row.get("data", {})
    val = data.get(field)
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    return None


def _income_field(db: MarketDB, ticker: str, field: str) -> float | None:
    """Get a field from the latest income statement."""
    row = db.get_latest_fundamentals(ticker, "income")
    if row is None:
        return None
    data = row.get("data", {})
    val = data.get(field)
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    return None


def _yoy_change(db: MarketDB, ticker: str, field: str) -> float | None:
    """Compute YoY change from income statement history."""
    rows = db.get_fundamentals_history(ticker, "income", limit=5)
    if len(rows) < 5:
        return None
    try:
        v_now = float(rows[0]["data"].get(field, 0) or 0)
        v_then = float(rows[4]["data"].get(field, 0) or 0)
        if abs(v_then) < 1e-10:
            return None
        return (v_now - v_then) / abs(v_then)
    except (ValueError, TypeError, KeyError):
        return None


# ── Fundamental Signals ──────────────────────────────────────────────

def pe_relative(db: MarketDB, tickers: list[str]) -> dict[str, float]:
    """
    P/E relative, inverted (low P/E = bullish for value).
    Hypothesis: cheap stocks tend to outperform. Fama & French (1992).
    """
    raw = {}
    for t in tickers:
        val = _ratio_field(db, t, "priceToEarningsRatio")
        raw[t] = -val if val is not None and val > 0 else None
    return _normalize_dict(raw)


def ev_ebitda_relative(db: MarketDB, tickers: list[str]) -> dict[str, float]:
    """
    EV/EBITDA relative, inverted (low = cheap).
    Capital-structure-neutral valuation metric.
    """
    raw = {}
    for t in tickers:
        val = _ratio_field(db, t, "enterpriseValueMultiple")
        raw[t] = -val if val is not None and val > 0 else None
    return _normalize_dict(raw)


def fcf_yield(db: MarketDB, tickers: list[str]) -> dict[str, float]:
    """
    Free Cash Flow per Share (higher = better).
    Companies generating real cash relative to price outperform.
    """
    raw = {}
    for t in tickers:
        val = _ratio_field(db, t, "freeCashFlowPerShare")
        raw[t] = val
    return _normalize_dict(raw)


def roe(db: MarketDB, tickers: list[str]) -> dict[str, float]:
    """
    Return on Equity.
    High ROE = quality. Novy-Marx (2013).
    """
    raw = {}
    for t in tickers:
        # ROE = netIncomePerShare / shareholdersEquityPerShare
        ni = _ratio_field(db, t, "netIncomePerShare")
        eq = _ratio_field(db, t, "shareholdersEquityPerShare")
        if ni is not None and eq is not None and abs(eq) > 1e-6:
            raw[t] = ni / eq
        else:
            raw[t] = None
    return _normalize_dict(raw)


def gross_margin_delta(db: MarketDB, tickers: list[str]) -> dict[str, float]:
    """
    Change in gross margin YoY.
    Improving margins signal operational improvement.
    """
    raw = {}
    for t in tickers:
        # Try from ratios history: grossProfitMargin
        rows = db.get_fundamentals_history(ticker=t, statement_type="income", limit=5)
        if len(rows) >= 5:
            try:
                gp_now = float(rows[0]["data"].get("grossProfit", 0) or 0)
                rev_now = float(rows[0]["data"].get("revenue", 1) or 1)
                gp_then = float(rows[4]["data"].get("grossProfit", 0) or 0)
                rev_then = float(rows[4]["data"].get("revenue", 1) or 1)
                margin_now = gp_now / rev_now if rev_now else 0
                margin_then = gp_then / rev_then if rev_then else 0
                if abs(margin_then) > 1e-6:
                    raw[t] = margin_now - margin_then  # absolute change in margin
                else:
                    raw[t] = None
            except (ValueError, TypeError):
                raw[t] = None
        else:
            raw[t] = None
    return _normalize_dict(raw)


def earnings_surprise(db: MarketDB, tickers: list[str]) -> dict[str, float]:
    """
    Most recent earnings surprise: (actual - estimated) / |estimated|.
    PEAD: market under-reacts to earnings. Bernard & Thomas (1989).
    """
    raw = {}
    for t in tickers:
        df = db.get_earnings_history(t, limit=4)
        actual = df[df["eps_actual"].notna() & df["eps_estimated"].notna()]
        if len(actual) > 0:
            r = actual.iloc[0]
            eps_a = float(r["eps_actual"])
            eps_e = float(r["eps_estimated"])
            if abs(eps_e) > 1e-6:
                raw[t] = (eps_a - eps_e) / abs(eps_e)
            else:
                raw[t] = None
        else:
            raw[t] = None
    return _normalize_dict(raw)


def revenue_growth(db: MarketDB, tickers: list[str]) -> dict[str, float]:
    """Revenue growth YoY. Growth factor."""
    raw = {}
    for t in tickers:
        val = _yoy_change(db, t, "revenue")
        raw[t] = val
    return _normalize_dict(raw)


def debt_equity_inv(db: MarketDB, tickers: list[str]) -> dict[str, float]:
    """Inverse of Debt/Equity (lower leverage = better). Quality/solvency."""
    raw = {}
    for t in tickers:
        val = _ratio_field(db, t, "debtToEquityRatio")
        raw[t] = -val if val is not None and val >= 0 else None
    return _normalize_dict(raw)


# ── Convenience ──────────────────────────────────────────────────────

ALL_SIGNALS = {
    "pe_relative": pe_relative,
    "ev_ebitda_relative": ev_ebitda_relative,
    "fcf_yield": fcf_yield,
    "roe": roe,
    "gross_margin_delta": gross_margin_delta,
    "earnings_surprise": earnings_surprise,
    "revenue_growth": revenue_growth,
    "debt_equity_inv": debt_equity_inv,
}


def compute_all(db: MarketDB, tickers: list[str]) -> dict[str, dict[str, float]]:
    """Compute all fundamental signals. Returns {signal_name: {ticker: score}}."""
    results = {}
    for name, fn in ALL_SIGNALS.items():
        results[name] = fn(db, tickers)
    return results
