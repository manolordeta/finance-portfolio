"""
Data quality module — explicit handling of missing data.

Philosophy: a TRUE NEUTRAL (score=0 because signals are mixed) is fundamentally
different from a MISSING DATA neutral (score=0 because we have no information).
The system must distinguish these cases and report them transparently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ScoreStatus(Enum):
    """Why a signal has a particular value."""
    VALID = "valid"                    # real signal computed from available data
    MISSING_PRICE = "missing_price"    # no price data for this ticker/date
    MISSING_FUNDAMENTAL = "missing_fundamental"  # no fundamental data (FMP gap)
    MISSING_NEWS = "missing_news"      # no news articles available
    INSUFFICIENT_HISTORY = "insufficient_history"  # not enough data points (e.g., IPO)
    API_ERROR = "api_error"            # data source returned an error
    EXCLUDED = "excluded"              # ticker excluded from this signal by design


@dataclass
class ScoredValue:
    """A signal value with explicit data quality metadata."""
    value: float              # the score in [-1, +1]
    status: ScoreStatus       # why this value exists
    detail: str = ""          # human-readable explanation

    @property
    def is_valid(self) -> bool:
        return self.status == ScoreStatus.VALID

    @property
    def is_missing(self) -> bool:
        return self.status != ScoreStatus.VALID

    @property
    def display_value(self) -> str:
        """Human-readable: shows value + flag if missing."""
        if self.is_valid:
            return f"{self.value:+.3f}"
        return f"{self.value:+.3f}*"  # asterisk = missing data

    @property
    def display_status(self) -> str:
        """Short status for reports."""
        if self.is_valid:
            return ""
        return f"[{self.status.value}]"


@dataclass
class TickerDataQuality:
    """Complete data quality report for a single ticker."""
    ticker: str
    signals: dict[str, ScoredValue] = field(default_factory=dict)

    @property
    def valid_count(self) -> int:
        return sum(1 for s in self.signals.values() if s.is_valid)

    @property
    def missing_count(self) -> int:
        return sum(1 for s in self.signals.values() if s.is_missing)

    @property
    def total_count(self) -> int:
        return len(self.signals)

    @property
    def coverage_pct(self) -> float:
        """Percentage of signals with valid data."""
        if self.total_count == 0:
            return 0.0
        return self.valid_count / self.total_count * 100

    @property
    def quality_grade(self) -> str:
        """A/B/C/D grade based on data coverage."""
        pct = self.coverage_pct
        if pct >= 90:
            return "A"
        elif pct >= 70:
            return "B"
        elif pct >= 50:
            return "C"
        else:
            return "D"

    def missing_signals(self) -> list[str]:
        """List of signal names with missing data."""
        return [name for name, sv in self.signals.items() if sv.is_missing]

    def summary(self) -> str:
        """One-line summary."""
        missing = self.missing_signals()
        if not missing:
            return f"{self.ticker}: {self.valid_count}/{self.total_count} signals (Grade {self.quality_grade})"
        return (
            f"{self.ticker}: {self.valid_count}/{self.total_count} signals "
            f"(Grade {self.quality_grade}) — missing: {', '.join(missing)}"
        )


def check_price_quality(prices_series, ticker: str, min_days: int = 200) -> ScoreStatus:
    """Check if a ticker has sufficient price history."""
    if prices_series is None:
        return ScoreStatus.MISSING_PRICE
    valid = prices_series.dropna()
    if len(valid) < min_days:
        return ScoreStatus.INSUFFICIENT_HISTORY
    return ScoreStatus.VALID


def check_fundamental_quality(db, ticker: str, statement_type: str = "income") -> ScoreStatus:
    """Check if a ticker has fundamental data available."""
    row = db.get_latest_fundamentals(ticker, statement_type)
    if row is None:
        return ScoreStatus.MISSING_FUNDAMENTAL
    return ScoreStatus.VALID


def build_quality_report(
    ticker: str,
    signal_scores: dict[str, float],
    signal_statuses: dict[str, ScoreStatus],
) -> TickerDataQuality:
    """Build a complete quality report for a ticker."""
    report = TickerDataQuality(ticker=ticker)
    for name, score in signal_scores.items():
        status = signal_statuses.get(name, ScoreStatus.VALID)
        detail = ""
        if status == ScoreStatus.MISSING_FUNDAMENTAL:
            detail = "No FMP data available"
        elif status == ScoreStatus.MISSING_NEWS:
            detail = "No recent news articles"
        elif status == ScoreStatus.INSUFFICIENT_HISTORY:
            detail = "Insufficient price history (IPO or recently listed)"
        elif status == ScoreStatus.MISSING_PRICE:
            detail = "Price data unavailable"

        report.signals[name] = ScoredValue(
            value=score,
            status=status,
            detail=detail,
        )
    return report
