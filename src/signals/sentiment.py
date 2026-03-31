"""
LLM-based sentiment scoring for stock news.

Provider-agnostic: supports DeepSeek (cheap testing), Anthropic (production),
or any OpenAI-compatible API.

Each ticker gets a sentiment score in [-1, +1] based on recent news articles.
Scores are stored with full versioning (model, prompt version, timestamp)
for reproducibility and forward IC measurement.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openai import OpenAI

from src.data.database import MarketDB

logger = logging.getLogger(__name__)

# ── Prompt versioning ────────────────────────────────────────────────

PROMPT_VERSION = "v1.0"

SYSTEM_PROMPT = """You are a quantitative financial analyst. Your job is to evaluate
the sentiment of recent news articles about a stock and produce a numerical score.

You must be objective and calibrated:
- Score +1.0 means extremely bullish news (major positive catalyst)
- Score +0.5 means moderately positive
- Score 0.0 means neutral or mixed
- Score -0.5 means moderately negative
- Score -1.0 means extremely bearish news (major negative catalyst)

Most news is routine and should score close to 0. Reserve extreme scores for
genuinely significant events (earnings beats/misses, M&A, regulatory actions,
major product launches/failures, management changes).

Respond ONLY with valid JSON. No markdown, no explanation outside the JSON."""

USER_PROMPT_TEMPLATE = """Evaluate the sentiment of these recent news articles about {ticker} ({company_name}).

Context:
- Sector: {sector}
- Recent earnings surprise: {earnings_context}

NEWS ARTICLES (last 48-72 hours):
{articles_text}

Respond with this exact JSON structure:
{{
  "score": <float between -1.0 and +1.0>,
  "confidence": <float between 0.0 and 1.0, how confident you are in the score>,
  "horizon": "<short|medium|long — when will the news impact materialize>",
  "key_drivers": ["<top 3 positive factors>"],
  "key_risks": ["<top 3 negative factors>"],
  "events_detected": ["<any material events: earnings, upgrade, downgrade, insider, regulatory, lawsuit, product_launch, management_change, partnership, or none>"],
  "summary": "<2-3 sentence summary of overall sentiment>"
}}"""


# ── Provider configuration ───────────────────────────────────────────

@dataclass
class LLMProvider:
    """Configuration for an LLM API provider."""
    name: str
    base_url: str
    api_key: str
    model: str
    max_tokens: int = 500
    temperature: float = 0.1  # low temp for consistent scoring


def get_deepseek_provider() -> LLMProvider:
    key = os.getenv("DEEPSEEK_API_KEY", "")
    return LLMProvider(
        name="deepseek",
        base_url="https://api.deepseek.com",
        api_key=key,
        model="deepseek-chat",
    )


def get_anthropic_provider() -> LLMProvider:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    return LLMProvider(
        name="anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key=key,
        model="claude-haiku-4-5-20251001",
    )


# ── Sentiment Scorer ─────────────────────────────────────────────────

@dataclass
class SentimentResult:
    """Result of scoring a ticker's news sentiment."""
    ticker: str
    score: float
    confidence: float
    horizon: str
    key_drivers: list[str]
    key_risks: list[str]
    events_detected: list[str]
    summary: str
    # Metadata for reproducibility
    model: str = ""
    prompt_version: str = PROMPT_VERSION
    scored_at: str = ""
    n_articles: int = 0
    raw_response: str = ""


class SentimentScorer:
    """
    Scores stock sentiment from news articles using LLM.

    Usage:
        scorer = SentimentScorer()  # uses DeepSeek by default
        result = scorer.score_ticker("MSFT", db)
        results = scorer.score_batch(["MSFT", "NVDA", "META"], db)
    """

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider or get_deepseek_provider()
        self.client = OpenAI(
            api_key=self.provider.api_key,
            base_url=self.provider.base_url,
        )

    def _build_articles_text(self, articles: list[dict], max_articles: int = 8) -> str:
        """Format articles for the prompt. Limit to most recent max_articles."""
        if not articles:
            return "(No recent news articles available)"

        articles = articles[:max_articles]
        parts = []
        for i, art in enumerate(articles, 1):
            date = art.get("published_date", art.get("publishedDate", "?"))
            title = art.get("title", "No title")
            text = art.get("text", "")
            # Truncate long articles to ~300 chars
            if len(text) > 400:
                text = text[:400] + "..."
            source = art.get("site", art.get("source", "?"))
            parts.append(f"[{i}] ({date} | {source})\n    {title}\n    {text}")

        return "\n\n".join(parts)

    def _get_earnings_context(self, db: MarketDB, ticker: str) -> str:
        """Get recent earnings context for the prompt."""
        df = db.get_earnings_history(ticker, limit=2)
        if len(df) == 0:
            return "No recent earnings data"

        actual = df[df["eps_actual"].notna()]
        if len(actual) == 0:
            return "No recent earnings data"

        r = actual.iloc[0]
        eps_a = r["eps_actual"]
        eps_e = r["eps_estimated"]
        if eps_e and abs(eps_e) > 1e-6:
            surp = (eps_a - eps_e) / abs(eps_e)
            return f"EPS actual={eps_a}, estimated={eps_e}, surprise={surp:+.1%}"
        return f"EPS actual={eps_a}"

    def _call_llm(self, system: str, user: str) -> str:
        """Make the LLM API call. Returns raw response text."""
        response = self.client.chat.completions.create(
            model=self.provider.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.provider.max_tokens,
            temperature=self.provider.temperature,
        )
        return response.choices[0].message.content.strip()

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown fences."""
        text = raw
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())

    def score_ticker(self, ticker: str, db: MarketDB,
                     max_articles: int = 8) -> SentimentResult:
        """
        Score a single ticker's sentiment from its news in the DB.

        Returns SentimentResult with score in [-1, +1].
        """
        # Get profile
        profile = db.get_profile(ticker) or {}
        company_name = profile.get("company_name", ticker)
        sector = profile.get("sector", "Unknown")

        # Get news from DB
        articles = db.get_news(ticker, limit=max_articles)

        # Build prompt
        articles_text = self._build_articles_text(articles, max_articles)
        earnings_ctx = self._get_earnings_context(db, ticker)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            earnings_context=earnings_ctx,
            articles_text=articles_text,
        )

        # Call LLM
        try:
            raw = self._call_llm(SYSTEM_PROMPT, user_prompt)
            parsed = self._parse_response(raw)
        except Exception as e:
            logger.error("LLM scoring failed for %s: %s", ticker, e)
            return SentimentResult(
                ticker=ticker, score=0.0, confidence=0.0,
                horizon="unknown", key_drivers=[], key_risks=[],
                events_detected=[], summary=f"Scoring failed: {e}",
                model=self.provider.model, scored_at=datetime.now().isoformat(),
                n_articles=len(articles), raw_response=str(e),
            )

        # Clamp score to [-1, +1]
        score = max(-1.0, min(1.0, float(parsed.get("score", 0))))

        return SentimentResult(
            ticker=ticker,
            score=score,
            confidence=float(parsed.get("confidence", 0.5)),
            horizon=parsed.get("horizon", "medium"),
            key_drivers=parsed.get("key_drivers", []),
            key_risks=parsed.get("key_risks", []),
            events_detected=parsed.get("events_detected", []),
            summary=parsed.get("summary", ""),
            model=self.provider.model,
            prompt_version=PROMPT_VERSION,
            scored_at=datetime.now().isoformat(),
            n_articles=len(articles),
            raw_response=raw,
        )

    def score_batch(self, tickers: list[str], db: MarketDB,
                    delay: float = 0.5, max_articles: int = 8,
                    ) -> list[SentimentResult]:
        """
        Score multiple tickers. Adds delay between calls to respect rate limits.

        Returns list of SentimentResult sorted by score (highest first).
        """
        results = []
        for i, ticker in enumerate(tickers):
            logger.info("[sentiment] scoring %s (%d/%d)", ticker, i + 1, len(tickers))
            result = self.score_ticker(ticker, db, max_articles)
            results.append(result)
            if i < len(tickers) - 1:
                time.sleep(delay)

        results.sort(key=lambda r: -r.score)
        return results

    def save_results(self, results: list[SentimentResult], db: MarketDB) -> None:
        """Save sentiment results to the database for forward IC tracking."""
        for r in results:
            db.upsert_sentiment(r.ticker, {
                "score": r.score,
                "confidence": r.confidence,
                "horizon": r.horizon,
                "key_drivers": r.key_drivers,
                "key_risks": r.key_risks,
                "events_detected": r.events_detected,
                "summary": r.summary,
                "model": r.model,
                "prompt_version": r.prompt_version,
                "scored_at": r.scored_at,
                "n_articles": r.n_articles,
            })


# ── Convenience function ─────────────────────────────────────────────

def score_universe(db: MarketDB, tickers: list[str],
                   provider: LLMProvider | None = None) -> list[SentimentResult]:
    """Score all tickers in the universe and save to DB."""
    scorer = SentimentScorer(provider)
    results = scorer.score_batch(tickers, db)
    scorer.save_results(results, db)
    return results
