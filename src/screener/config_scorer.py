"""
Config-driven composite scorer.

Reads signals.yaml to determine which signals are enabled, their weights,
and horizon-specific overrides. Produces a ranked list of the universe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_signals_config(path: str = "config/signals.yaml") -> dict:
    """Load the master signals configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_enabled_signals(config: dict, horizon: str = "21d") -> dict[str, dict]:
    """
    Get all enabled signals for a given horizon, respecting overrides.

    Returns dict of signal_name → {ic, weight, category, ...}
    """
    composite_cfg = config.get("composite", {})
    overrides = composite_cfg.get("horizon_overrides", {}).get(horizon, {})
    exclude = set(overrides.get("exclude", []))

    enabled = {}
    for category in ["technical", "fundamental", "valuation", "sentiment"]:
        for sig in config.get("signals", {}).get(category, []):
            name = sig["name"]
            if not sig.get("enabled", False):
                continue
            if name in exclude:
                continue

            # Get IC for this horizon
            ic_key = f"ic_{horizon}"
            ic_val = sig.get(ic_key)

            enabled[name] = {
                "category": category,
                "ic": ic_val,
                "t_stat": sig.get(f"t_stat_{horizon}"),
                "verdict": sig.get("verdict", ""),
                "weight": sig.get("weight", "auto"),
            }

    return enabled


def compute_weights(enabled_signals: dict[str, dict],
                    method: str = "auto") -> dict[str, float]:
    """
    Compute weights for enabled signals.

    method='auto': weight proportional to abs(IC). Signals with null IC get min weight.
    method='equal': uniform 1/N.
    """
    if method == "equal" or not enabled_signals:
        n = len(enabled_signals)
        return {name: 1.0 / n for name in enabled_signals} if n > 0 else {}

    # Auto: proportional to abs(IC)
    ics = {}
    for name, info in enabled_signals.items():
        ic = info.get("ic")
        if ic is not None and np.isfinite(ic) and ic > 0:
            ics[name] = abs(ic)
        else:
            ics[name] = 0.005  # minimum weight for signals without IC

    total = sum(ics.values())
    if total < 1e-10:
        n = len(ics)
        return {name: 1.0 / n for name in ics}

    return {name: ic / total for name, ic in ics.items()}


class ConfigScorer:
    """
    Composite scorer driven by signals.yaml configuration.

    Usage:
        scorer = ConfigScorer("config/signals.yaml")
        ranking = scorer.rank(tech_signals, fund_signals, val_signals, sentiment)
    """

    def __init__(self, config_path: str = "config/signals.yaml"):
        self.config = load_signals_config(config_path)
        self.horizon = self.config.get("composite", {}).get("horizon", "21d")
        self.weighting = self.config.get("composite", {}).get("weighting", "auto")
        self.min_signals = self.config.get("composite", {}).get("min_signals", 5)

        self.enabled = get_enabled_signals(self.config, self.horizon)
        self.weights = compute_weights(self.enabled, self.weighting)

        logger.info(
            "ConfigScorer initialized: %d signals, horizon=%s, weighting=%s",
            len(self.weights), self.horizon, self.weighting,
        )

    def rank(
        self,
        tech_signals: dict[str, pd.DataFrame],
        fund_signals: dict[str, dict[str, float]],
        val_signals: dict[str, pd.DataFrame],
        sentiment_scores: dict[str, float],
    ) -> pd.DataFrame:
        """
        Produce a ranked DataFrame of the universe.

        Returns DataFrame with columns:
            ticker, composite_score, rank, and one column per signal.
        """
        # Collect latest scores per ticker
        all_tickers = set()
        ticker_scores: dict[str, dict[str, float]] = {}

        # Technical signals (time-series → take latest row)
        for name, df in tech_signals.items():
            if name not in self.weights:
                continue
            latest = df.iloc[-1].dropna()
            for ticker, val in latest.items():
                if ticker not in ticker_scores:
                    ticker_scores[ticker] = {}
                ticker_scores[ticker][name] = float(val)
                all_tickers.add(ticker)

        # Valuation signals (same structure as technical)
        for name, df in val_signals.items():
            if name not in self.weights:
                continue
            latest = df.iloc[-1].dropna()
            for ticker, val in latest.items():
                if ticker not in ticker_scores:
                    ticker_scores[ticker] = {}
                ticker_scores[ticker][name] = float(val)
                all_tickers.add(ticker)

        # Fundamental signals (dict per signal)
        for name, scores_dict in fund_signals.items():
            if name not in self.weights:
                continue
            for ticker, val in scores_dict.items():
                if ticker not in ticker_scores:
                    ticker_scores[ticker] = {}
                ticker_scores[ticker][name] = float(val)
                all_tickers.add(ticker)

        # Sentiment
        if "llm_sentiment" in self.weights:
            for ticker, val in sentiment_scores.items():
                if ticker not in ticker_scores:
                    ticker_scores[ticker] = {}
                ticker_scores[ticker]["llm_sentiment"] = float(val)

        # Compute weighted composite
        rows = []
        for ticker in sorted(all_tickers):
            scores = ticker_scores.get(ticker, {})
            available = {k: v for k, v in scores.items()
                        if k in self.weights and np.isfinite(v)}

            if len(available) < self.min_signals:
                continue

            # Weighted average
            weighted_sum = sum(self.weights.get(k, 0) * v
                             for k, v in available.items())
            weight_sum = sum(self.weights.get(k, 0) for k in available)
            composite = weighted_sum / weight_sum if weight_sum > 0 else 0

            row = {"ticker": ticker, "composite_score": composite}
            row.update(scores)
            rows.append(row)

        df = pd.DataFrame(rows)
        if len(df) == 0:
            return df

        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        return df

    def get_watchlist(self) -> dict:
        """Get watchlist configuration."""
        return self.config.get("watchlist", {})

    def get_alert_config(self) -> dict:
        """Get alert system configuration."""
        return self.config.get("alerts", {})

    def get_policy(self) -> dict:
        """Get investment policy."""
        return self.config.get("policy", {})

    def summary(self) -> str:
        """Print human-readable summary of active configuration."""
        lines = [
            f"Horizon: {self.horizon}",
            f"Weighting: {self.weighting}",
            f"Enabled signals: {len(self.weights)}",
            "",
        ]
        for name, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            info = self.enabled.get(name, {})
            cat = info.get("category", "?")[:4]
            ic = info.get("ic")
            ic_str = f"IC={ic:+.4f}" if ic is not None else "IC=measuring"
            lines.append(f"  [{cat:4s}] {name:25s} w={w:.1%}  {ic_str}")

        return "\n".join(lines)
