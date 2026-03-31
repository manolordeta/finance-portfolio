"""
Composite scorer: combines validated signals into a single score per ticker.

Only signals that PASS or MARGINAL in the Research Protocol enter the composite.
Weights are proportional to IC (information coefficient) by default.

Supports separate composites by horizon (21d, 63d).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ScorerConfig:
    """Configuration for which signals enter the composite and with what weight."""

    # signal_name → weight (if None, use equal weights)
    weights: dict[str, float] | None = None

    # signal_name → IC (used to auto-weight if weights is None)
    signal_ics: dict[str, float] = field(default_factory=dict)

    # Minimum IC to include a signal (filter out noise)
    min_ic: float = 0.03

    def get_weights(self) -> dict[str, float]:
        """Compute final weights. IC-proportional if no explicit weights."""
        if self.weights is not None:
            total = sum(self.weights.values())
            if total > 0:
                return {k: v / total for k, v in self.weights.items()}
            return self.weights

        # Filter by min_ic, then weight proportional to IC
        valid = {k: v for k, v in self.signal_ics.items() if v >= self.min_ic}
        if not valid:
            # Fallback: equal weight all signals
            return {k: 1.0 / len(self.signal_ics) for k in self.signal_ics}

        total_ic = sum(valid.values())
        return {k: v / total_ic for k, v in valid.items()}


class CompositeScorer:
    """
    Combine multiple signal DataFrames into a composite score.

    Usage:
        scorer = CompositeScorer(config)
        composite = scorer.score(signals_dict)
        # composite is DataFrame(dates × tickers) with values in [-1, +1]
    """

    def __init__(self, config: ScorerConfig | None = None):
        self.config = config or ScorerConfig()
        self._weights: dict[str, float] | None = None

    @property
    def weights(self) -> dict[str, float]:
        if self._weights is None:
            self._weights = self.config.get_weights()
        return self._weights

    def score(self, signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute weighted composite score across all signals.

        Args:
            signals: dict mapping signal_name → DataFrame(dates × tickers)
                     Values should be in [-1, +1].

        Returns:
            DataFrame(dates × tickers) with composite scores in [-1, +1].
        """
        weights = self.weights
        active_signals = {k: v for k, v in signals.items() if k in weights}

        if not active_signals:
            raise ValueError("No signals match the configured weights")

        # Align all signals to common index and columns
        all_dates = sorted(
            set.intersection(*[set(s.index) for s in active_signals.values()])
        )
        all_tickers = sorted(
            set.intersection(*[set(s.columns) for s in active_signals.values()])
        )

        if not all_dates or not all_tickers:
            raise ValueError("No common dates/tickers across signals")

        # Weighted sum
        composite = pd.DataFrame(0.0, index=all_dates, columns=all_tickers)
        total_weight = 0.0

        for name, sig in active_signals.items():
            w = weights.get(name, 0.0)
            if w <= 0:
                continue
            aligned = sig.reindex(index=all_dates, columns=all_tickers)
            # Where signal is NaN, don't count its weight
            mask = aligned.notna()
            composite += aligned.fillna(0) * w
            total_weight += w

        # Normalize by total weight
        if total_weight > 0:
            composite = composite / total_weight

        # Clip to [-1, +1]
        return composite.clip(-1, 1)

    def score_snapshot(self, signal_scores: dict[str, dict[str, float]]) -> dict[str, float]:
        """
        Compute composite for a single point in time (e.g., today).

        Args:
            signal_scores: {signal_name: {ticker: score}}

        Returns:
            {ticker: composite_score}
        """
        weights = self.weights

        all_tickers = set()
        for scores in signal_scores.values():
            all_tickers.update(scores.keys())

        result = {}
        for ticker in all_tickers:
            weighted_sum = 0.0
            weight_sum = 0.0
            for sig_name, scores in signal_scores.items():
                w = weights.get(sig_name, 0.0)
                if w <= 0 or ticker not in scores:
                    continue
                val = scores[ticker]
                if val is not None and np.isfinite(val):
                    weighted_sum += val * w
                    weight_sum += w

            if weight_sum > 0:
                result[ticker] = weighted_sum / weight_sum
            else:
                result[ticker] = 0.0

        return result

    def weight_summary(self) -> str:
        """Human-readable weight breakdown."""
        lines = ["Composite Weights:"]
        for name, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            ic = self.config.signal_ics.get(name, 0)
            lines.append(f"  {name:25s}  w={w:.3f}  (IC={ic:+.4f})")
        return "\n".join(lines)
