"""
Universe ranker: takes composite scores and produces actionable ranking.

Outputs:
  - Ranked list of tickers with score breakdown
  - Comparison vs baselines
  - Alerts for significant score changes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RankedTicker:
    """A single ticker's ranking entry."""
    rank: int
    ticker: str
    composite_score: float
    signal_scores: dict[str, float]
    sector: str = ""
    company_name: str = ""


class UniverseRanker:
    """
    Rank the universe by composite score with signal breakdown.

    Usage:
        ranker = UniverseRanker(db=db)
        ranking = ranker.rank(composite_scores, signal_scores)
        print(ranker.format_ranking(ranking))
    """

    def __init__(self, db=None):
        self.db = db

    def rank(
        self,
        composite_scores: dict[str, float],
        signal_scores: dict[str, dict[str, float]] | None = None,
    ) -> list[RankedTicker]:
        """
        Rank tickers by composite score (descending).

        Args:
            composite_scores: {ticker: composite_score}
            signal_scores: {signal_name: {ticker: score}} for breakdown
        """
        # Sort by composite descending
        sorted_tickers = sorted(
            composite_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        ranking = []
        for i, (ticker, score) in enumerate(sorted_tickers, 1):
            # Gather per-signal breakdown
            breakdown = {}
            if signal_scores:
                for sig_name, scores in signal_scores.items():
                    breakdown[sig_name] = scores.get(ticker, 0.0)

            # Get profile from DB if available
            sector = ""
            company = ""
            if self.db is not None:
                profile = self.db.get_profile(ticker)
                if profile:
                    sector = profile.get("sector", "")
                    company = profile.get("company_name", "")

            ranking.append(RankedTicker(
                rank=i,
                ticker=ticker,
                composite_score=score,
                signal_scores=breakdown,
                sector=sector,
                company_name=company,
            ))

        return ranking

    def format_ranking(
        self,
        ranking: list[RankedTicker],
        title: str = "Universe Ranking",
        date: str | None = None,
    ) -> str:
        """Format ranking as human-readable table."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Determine signal columns from first entry
        sig_names = []
        if ranking and ranking[0].signal_scores:
            sig_names = list(ranking[0].signal_scores.keys())

        # Header
        lines = [
            f"\n{'═' * 90}",
            f"  {title} — {date}",
            f"{'═' * 90}",
            "",
        ]

        # Column header
        header = f"{'Rank':>4s}  {'Ticker':6s}  {'Score':>6s}"
        for s in sig_names:
            # Abbreviate signal names
            short = s[:8]
            header += f"  {short:>8s}"
        header += f"  {'Sector':15s}  {'Company':20s}"
        lines.append(header)
        lines.append("─" * len(header))

        # Rows
        for entry in ranking:
            row = f"{entry.rank:>4d}  {entry.ticker:6s}  {entry.composite_score:+.3f}"
            for s in sig_names:
                val = entry.signal_scores.get(s, 0.0)
                row += f"  {val:+8.3f}"
            row += f"  {entry.sector:15s}  {entry.company_name:20s}"
            lines.append(row)

        lines.append("─" * len(header) if ranking else "")

        # Summary
        if ranking:
            scores = [e.composite_score for e in ranking]
            lines.append(f"\n  Top 3:    {', '.join(e.ticker for e in ranking[:3])}")
            lines.append(f"  Bottom 3: {', '.join(e.ticker for e in ranking[-3:])}")
            lines.append(f"  Score range: [{min(scores):+.3f}, {max(scores):+.3f}]")

            # Sector distribution of top quartile
            n_top = max(1, len(ranking) // 4)
            top_sectors = {}
            for e in ranking[:n_top]:
                s = e.sector or "Unknown"
                top_sectors[s] = top_sectors.get(s, 0) + 1
            lines.append(f"  Top quartile sectors: {top_sectors}")

        lines.append(f"{'═' * 90}")
        return "\n".join(lines)

    def to_dataframe(self, ranking: list[RankedTicker]) -> pd.DataFrame:
        """Convert ranking to DataFrame for analysis."""
        rows = []
        for entry in ranking:
            row = {
                "rank": entry.rank,
                "ticker": entry.ticker,
                "composite": entry.composite_score,
                "sector": entry.sector,
                "company": entry.company_name,
            }
            row.update(entry.signal_scores)
            rows.append(row)
        return pd.DataFrame(rows).set_index("ticker")

    def detect_changes(
        self,
        current: list[RankedTicker],
        previous: list[RankedTicker],
        alert_threshold: int = 3,
    ) -> list[str]:
        """
        Detect significant ranking changes between two periods.

        Returns list of alert strings for tickers that moved >= alert_threshold positions.
        """
        prev_ranks = {e.ticker: e.rank for e in previous}
        alerts = []

        for entry in current:
            if entry.ticker in prev_ranks:
                prev_rank = prev_ranks[entry.ticker]
                change = prev_rank - entry.rank  # positive = improved
                if abs(change) >= alert_threshold:
                    direction = "↑" if change > 0 else "↓"
                    alerts.append(
                        f"  {direction} {entry.ticker:6s}  "
                        f"rank {prev_rank} → {entry.rank} "
                        f"({change:+d} positions)  "
                        f"score={entry.composite_score:+.3f}"
                    )

        return alerts
