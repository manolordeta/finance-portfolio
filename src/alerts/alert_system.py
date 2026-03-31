"""
LLM-powered alert system for active portfolio monitoring.

Runs ONLY on watchlist tickers (not full universe) to optimize costs.
Each run produces:
  1. Sentiment score per ticker (reuses sentiment scoring)
  2. Event detection (earnings, upgrades, insider, regulatory, etc.)
  3. Severity classification (HIGH / MEDIUM / LOW)
  4. Actionable recommendation (HOLD / REVIEW / CONSIDER_EXIT / CONSIDER_ENTRY)

Cost: ~$0.004/run for 12 tickers with DeepSeek.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import yaml
from openai import OpenAI

from src.data.database import MarketDB

logger = logging.getLogger(__name__)


ALERT_SYSTEM_PROMPT = """You are a portfolio monitoring assistant for a quantitative investment system.
Your job is to analyze recent news and market context for stocks in an active portfolio
and generate actionable alerts.

You must be calibrated and avoid false alarms:
- Most days, most stocks should have NO material alerts
- Only flag events that could materially impact the investment thesis
- Be specific about what happened, why it matters, and what action to consider

Severity levels:
- HIGH: material event requiring immediate attention (earnings miss >10%, fraud, CEO departure, FDA rejection, major lawsuit)
- MEDIUM: notable event worth reviewing (guidance change, analyst upgrade/downgrade, sector rotation, unusual volume)
- LOW: minor but relevant context (industry news, competitor developments, macro shifts)
- NONE: nothing material happened

Action recommendations:
- HOLD: no action needed, thesis intact
- REVIEW: worth looking at more carefully with your partner
- CONSIDER_EXIT: negative signals suggest considering reducing/exiting position
- CONSIDER_ENTRY: positive signals suggest considering adding/entering position
- URGENT_REVIEW: potential high-impact negative event, review ASAP"""


ALERT_USER_PROMPT = """Analyze the following stock in my active portfolio:

TICKER: {ticker}
COMPANY: {company_name}
SECTOR: {sector}
CURRENT POSITION: {position_status}

CONTEXT:
- Composite ranking: #{rank} out of {total} (percentile: {percentile})
- Ranking trend: {rank_trend}
- 1-month return: {ret_1m}
- Earnings surprise (last): {earnings_context}

RECENT NEWS (last 48-72 hours):
{news_text}

SIGNAL BREAKDOWN:
{signal_breakdown}

Based on all available information, generate an alert assessment.
Respond ONLY with valid JSON:
{{
  "severity": "<NONE|LOW|MEDIUM|HIGH>",
  "action": "<HOLD|REVIEW|CONSIDER_EXIT|CONSIDER_ENTRY|URGENT_REVIEW>",
  "events_detected": ["<list of material events detected, or empty>"],
  "headline": "<1 sentence summary of the most important thing>",
  "analysis": "<2-3 sentences explaining the situation and reasoning>",
  "risk_factors": ["<current risks to monitor>"],
  "positive_factors": ["<current positive drivers>"],
  "sentiment_score": <float -1.0 to +1.0>
}}"""


@dataclass
class Alert:
    """A single alert for a portfolio ticker."""
    ticker: str
    severity: str           # NONE | LOW | MEDIUM | HIGH
    action: str             # HOLD | REVIEW | CONSIDER_EXIT | CONSIDER_ENTRY | URGENT_REVIEW
    events_detected: list[str]
    headline: str
    analysis: str
    risk_factors: list[str]
    positive_factors: list[str]
    sentiment_score: float
    # Context
    rank: int = 0
    total: int = 0
    ret_1m: str = ""
    timestamp: str = ""


@dataclass
class AlertReport:
    """Complete alert report for a portfolio run."""
    alerts: list[Alert]
    timestamp: str
    total_cost_usd: float = 0.0
    model: str = ""

    @property
    def high_alerts(self) -> list[Alert]:
        return [a for a in self.alerts if a.severity == "HIGH"]

    @property
    def medium_alerts(self) -> list[Alert]:
        return [a for a in self.alerts if a.severity == "MEDIUM"]

    @property
    def actionable_alerts(self) -> list[Alert]:
        return [a for a in self.alerts if a.severity in ("HIGH", "MEDIUM")]

    def format_summary(self) -> str:
        """Human-readable summary of the alert report."""
        lines = [
            f"{'═' * 75}",
            f"  PORTFOLIO ALERT REPORT — {self.timestamp}",
            f"  {len(self.alerts)} tickers monitored | Model: {self.model}",
            f"{'═' * 75}",
        ]

        # High severity first
        if self.high_alerts:
            lines.append(f"\n  🔴 HIGH SEVERITY ({len(self.high_alerts)}):")
            for a in self.high_alerts:
                lines.append(f"    {a.ticker:6s} | {a.action:15s} | {a.headline}")
                lines.append(f"           {a.analysis}")
                if a.events_detected:
                    lines.append(f"           Events: {', '.join(a.events_detected)}")

        # Medium severity
        if self.medium_alerts:
            lines.append(f"\n  🟡 MEDIUM SEVERITY ({len(self.medium_alerts)}):")
            for a in self.medium_alerts:
                lines.append(f"    {a.ticker:6s} | {a.action:15s} | {a.headline}")
                lines.append(f"           {a.analysis}")

        # Low / None — compact
        low = [a for a in self.alerts if a.severity == "LOW"]
        none_ = [a for a in self.alerts if a.severity == "NONE"]

        if low:
            lines.append(f"\n  ⚪ LOW SEVERITY ({len(low)}):")
            for a in low:
                lines.append(f"    {a.ticker:6s} | {a.headline}")

        if none_:
            lines.append(f"\n  ✅ NO ALERTS ({len(none_)}):")
            tickers_none = ", ".join(a.ticker for a in none_)
            lines.append(f"    {tickers_none}")

        # Portfolio summary
        lines.append(f"\n  {'─' * 75}")
        lines.append(f"  Portfolio Sentiment: {self._avg_sentiment():+.2f}")
        lines.append(f"  Estimated cost: ${self.total_cost_usd:.4f}")

        return "\n".join(lines)

    def _avg_sentiment(self) -> float:
        scores = [a.sentiment_score for a in self.alerts]
        return sum(scores) / len(scores) if scores else 0.0


class AlertSystem:
    """
    LLM-powered portfolio alert system.

    Monitors only watchlist tickers. Produces severity-classified alerts
    with actionable recommendations.
    """

    def __init__(self, config_path: str = "config/signals.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        alert_cfg = self.config.get("alerts", {})
        llm_cfg = self.config.get("llm", {}).get("batch", {})

        self.model = llm_cfg.get("model", "deepseek-chat")
        self.temperature = llm_cfg.get("temperature", 0.1)
        self.max_tokens = 600
        self.cost_per_ticker = self.config.get("llm", {}).get("cost_per_ticker", 0.00025)

        # Initialize LLM client
        provider = llm_cfg.get("provider", "deepseek")
        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            base_url = "https://api.deepseek.com"
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            base_url = "https://api.anthropic.com/v1"

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _call_llm(self, system: str, user: str) -> dict:
        """Make LLM call and parse JSON response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        raw = response.choices[0].message.content.strip()

        # Parse JSON
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]

        return json.loads(raw.strip())

    def _get_news_text(self, db: MarketDB, ticker: str, limit: int = 5) -> str:
        """Get recent news for a ticker."""
        articles = db.get_news(ticker, limit=limit)
        if not articles:
            return "(No recent news available for this ticker)"

        parts = []
        for i, art in enumerate(articles, 1):
            title = art.get("title", "No title")
            date = art.get("published_at", "?")
            source = art.get("source_name", "?")
            text = art.get("text_snippet", "")
            if len(text) > 300:
                text = text[:300] + "..."
            parts.append(f"[{i}] ({date} | {source}) {title}\n    {text}")

        return "\n\n".join(parts)

    def _get_signal_breakdown(self, ranking_row: dict) -> str:
        """Format signal breakdown for the prompt."""
        skip = {"ticker", "composite_score", "rank"}
        parts = []
        for key, val in ranking_row.items():
            if key in skip or not isinstance(val, (int, float)):
                continue
            try:
                v = float(val)
                if abs(v) > 0.001:
                    direction = "bullish" if v > 0 else "bearish"
                    parts.append(f"  {key}: {v:+.3f} ({direction})")
            except (ValueError, TypeError):
                pass
        return "\n".join(parts) if parts else "(no signal breakdown available)"

    def run(
        self,
        db: MarketDB,
        ranking: "pd.DataFrame",
        ret_1m: "pd.Series | None" = None,
        force_refresh: bool = False,
    ) -> AlertReport:
        """
        Run alert system on all watchlist tickers.

        Uses daily cache: if alerts were already generated today, returns cached
        version. Pass force_refresh=True to regenerate.

        Args:
            db: MarketDB instance
            ranking: DataFrame from ConfigScorer.rank()
            force_refresh: if True, ignore cache and regenerate
            ret_1m: 1-month returns Series (optional)
        """
        import pandas as pd
        import numpy as np
        import time

        today = datetime.now().strftime("%Y-%m-%d")

        # ── Cache check: if alerts exist for today, return them ──
        if not force_refresh and db.has_alerts_for_date(today):
            logger.info("[alert] Using cached alerts for %s", today)
            cached = db.get_cached_alerts(today)
            cached_alerts = []
            for c in cached:
                cached_alerts.append(Alert(
                    ticker=c["ticker"],
                    severity=c["severity"],
                    action=c["action"],
                    events_detected=json.loads(c.get("events_json", "[]") or "[]"),
                    headline=c.get("headline", ""),
                    analysis=c.get("analysis", ""),
                    risk_factors=json.loads(c.get("risk_factors_json", "[]") or "[]"),
                    positive_factors=json.loads(c.get("positive_factors_json", "[]") or "[]"),
                    sentiment_score=c.get("sentiment_score", 0.0),
                    rank=c.get("rank", 0),
                    total=c.get("total_ranked", 0),
                    ret_1m=c.get("ret_1m", ""),
                    timestamp=c.get("created_at", ""),
                ))
            return AlertReport(
                alerts=cached_alerts,
                timestamp=f"{today} (cached)",
                total_cost_usd=0.0,
                model=self.model + " (cached)",
            )

        # ── Fresh run ────────────────────────────────────────────
        watchlist = self.config.get("watchlist", {}).get("active", [])
        total_ranked = len(ranking)
        alerts = []

        for i, ticker in enumerate(watchlist):
            logger.info("[alert] processing %s (%d/%d)", ticker, i + 1, len(watchlist))

            # Get context
            profile = db.get_profile(ticker) or {}
            company_name = profile.get("company_name", ticker)
            sector = profile.get("sector", "Unknown")

            # Ranking position
            match = ranking[ranking["ticker"] == ticker]
            if len(match) > 0:
                row = match.iloc[0]
                rank = int(row["rank"])
                percentile = f"top {rank/total_ranked*100:.0f}%"
                rank_trend = "stable"  # TODO: compare with previous run
                signal_breakdown = self._get_signal_breakdown(row.to_dict())
            else:
                rank = 0
                percentile = "not ranked"
                rank_trend = "N/A"
                signal_breakdown = "(not in ranking universe)"

            # Returns
            r1m = "N/A"
            if ret_1m is not None and ticker in ret_1m.index:
                v = ret_1m[ticker]
                if np.isfinite(v):
                    r1m = f"{v:+.1%}"

            # Earnings context
            earnings_df = db.get_earnings_history(ticker, limit=2)
            earnings_ctx = "No recent earnings data"
            if len(earnings_df) > 0:
                actual = earnings_df[earnings_df["eps_actual"].notna()]
                if len(actual) > 0:
                    e = actual.iloc[0]
                    eps_a = e["eps_actual"]
                    eps_e = e["eps_estimated"]
                    if eps_e and abs(eps_e) > 1e-6:
                        surp = (eps_a - eps_e) / abs(eps_e)
                        earnings_ctx = f"EPS {eps_a} vs est {eps_e} (surprise {surp:+.1%})"
                    else:
                        earnings_ctx = f"EPS {eps_a}"

            # News
            news_text = self._get_news_text(db, ticker)

            # Build prompt
            user_prompt = ALERT_USER_PROMPT.format(
                ticker=ticker,
                company_name=company_name,
                sector=sector,
                position_status="in watchlist / considering",
                rank=rank,
                total=total_ranked,
                percentile=percentile,
                rank_trend=rank_trend,
                ret_1m=r1m,
                earnings_context=earnings_ctx,
                news_text=news_text,
                signal_breakdown=signal_breakdown,
            )

            try:
                parsed = self._call_llm(ALERT_SYSTEM_PROMPT, user_prompt)

                alert = Alert(
                    ticker=ticker,
                    severity=parsed.get("severity", "NONE"),
                    action=parsed.get("action", "HOLD"),
                    events_detected=parsed.get("events_detected", []),
                    headline=parsed.get("headline", "No headline"),
                    analysis=parsed.get("analysis", ""),
                    risk_factors=parsed.get("risk_factors", []),
                    positive_factors=parsed.get("positive_factors", []),
                    sentiment_score=max(-1, min(1, float(parsed.get("sentiment_score", 0)))),
                    rank=rank,
                    total=total_ranked,
                    ret_1m=r1m,
                    timestamp=datetime.now().isoformat(),
                )
            except Exception as e:
                logger.error("Alert failed for %s: %s", ticker, e)
                alert = Alert(
                    ticker=ticker,
                    severity="NONE",
                    action="HOLD",
                    events_detected=[],
                    headline=f"Scoring failed: {e}",
                    analysis="",
                    risk_factors=[],
                    positive_factors=[],
                    sentiment_score=0.0,
                    rank=rank,
                    total=total_ranked,
                    ret_1m=r1m,
                    timestamp=datetime.now().isoformat(),
                )

            alerts.append(alert)

            if i < len(watchlist) - 1:
                time.sleep(0.5)

        # Sort: HIGH first, then MEDIUM, then LOW, then NONE
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
        alerts.sort(key=lambda a: severity_order.get(a.severity, 4))

        # ── Save to cache ────────────────────────────────────────
        for alert in alerts:
            db.save_alert_cache(today, alert)
        logger.info("[alert] Cached %d alerts for %s", len(alerts), today)

        return AlertReport(
            alerts=alerts,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            total_cost_usd=len(watchlist) * self.cost_per_ticker,
            model=self.model,
        )
