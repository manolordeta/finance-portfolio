"""
SQLite database module for the quantitative investment system.

Stores fundamentals, ratios, news, signal scores, rankings, and run metadata.
All timestamps use filing_date (when data became publicly available),
NOT period_date, to prevent look-ahead bias.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Fundamentals: income statement, balance sheet, cash flow (quarterly)
CREATE TABLE IF NOT EXISTS fundamentals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    period_date     TEXT    NOT NULL,   -- end of fiscal quarter (e.g. 2025-12-31)
    filing_date     TEXT    NOT NULL,   -- when it became public (e.g. 2026-01-28)
    statement_type  TEXT    NOT NULL,   -- 'income' | 'balance' | 'cashflow'
    data_json       TEXT    NOT NULL,   -- full statement as JSON
    source          TEXT    NOT NULL DEFAULT 'fmp',
    fetched_at      TEXT    NOT NULL,
    UNIQUE(ticker, period_date, statement_type, source)
);
CREATE INDEX IF NOT EXISTS idx_fund_ticker ON fundamentals(ticker);
CREATE INDEX IF NOT EXISTS idx_fund_filing ON fundamentals(filing_date);

-- Ratios & key metrics (quarterly or TTM)
CREATE TABLE IF NOT EXISTS ratios (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    period_date     TEXT    NOT NULL,
    filing_date     TEXT    NOT NULL,
    data_json       TEXT    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'fmp',
    fetched_at      TEXT    NOT NULL,
    UNIQUE(ticker, period_date, source)
);
CREATE INDEX IF NOT EXISTS idx_ratios_ticker ON ratios(ticker);

-- Earnings calendar & surprises
CREATE TABLE IF NOT EXISTS earnings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    earnings_date   TEXT    NOT NULL,   -- actual report date
    fiscal_period   TEXT,               -- e.g. 'Q4 2025'
    eps_estimated   REAL,
    eps_actual      REAL,
    revenue_estimated REAL,
    revenue_actual    REAL,
    surprise_pct    REAL,               -- (actual - est) / |est|
    source          TEXT    NOT NULL DEFAULT 'fmp',
    fetched_at      TEXT    NOT NULL,
    UNIQUE(ticker, earnings_date, source)
);
CREATE INDEX IF NOT EXISTS idx_earn_ticker ON earnings(ticker);
CREATE INDEX IF NOT EXISTS idx_earn_date ON earnings(earnings_date);

-- Company profile / reference data
CREATE TABLE IF NOT EXISTS profiles (
    ticker          TEXT    PRIMARY KEY,
    company_name    TEXT,
    sector          TEXT,
    industry        TEXT,
    market_cap      REAL,
    country         TEXT,
    exchange        TEXT,
    data_json       TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);

-- News articles (raw + scored)
CREATE TABLE IF NOT EXISTS news (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    published_at    TEXT    NOT NULL,   -- when article was published
    title           TEXT    NOT NULL,
    url             TEXT,
    source_name     TEXT,
    text_snippet    TEXT,               -- first ~500 chars
    -- LLM scoring fields (NULL until scored)
    sentiment_score REAL,               -- [-1, +1]
    confidence      REAL,               -- [0, 1]
    horizon         TEXT,               -- 'short' | 'medium' | 'long'
    drivers_json    TEXT,               -- JSON array of drivers
    risks_json      TEXT,               -- JSON array of risks
    llm_summary     TEXT,
    llm_model       TEXT,               -- e.g. 'claude-sonnet-4-20250514'
    prompt_version  TEXT,               -- version of the scoring prompt
    scored_at       TEXT,               -- when LLM scored it
    fetched_at      TEXT    NOT NULL,
    UNIQUE(ticker, url)
);
CREATE INDEX IF NOT EXISTS idx_news_ticker ON news(ticker);
CREATE INDEX IF NOT EXISTS idx_news_pub ON news(published_at);

-- Signal scores (raw, per signal per ticker per date)
CREATE TABLE IF NOT EXISTS signal_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    signal_date     TEXT    NOT NULL,   -- date the signal was computed for
    signal_name     TEXT    NOT NULL,   -- e.g. 'momentum_12_1', 'pe_relative'
    signal_value    REAL    NOT NULL,   -- raw value before normalization
    score           REAL    NOT NULL,   -- normalized [-1, +1]
    horizon         TEXT,               -- '21d' | '63d'
    computed_at     TEXT    NOT NULL,
    UNIQUE(ticker, signal_date, signal_name)
);
CREATE INDEX IF NOT EXISTS idx_sig_ticker ON signal_scores(ticker);
CREATE INDEX IF NOT EXISTS idx_sig_date ON signal_scores(signal_date);
CREATE INDEX IF NOT EXISTS idx_sig_name ON signal_scores(signal_name);

-- Composite rankings (daily snapshots)
CREATE TABLE IF NOT EXISTS rankings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ranking_date    TEXT    NOT NULL,
    horizon         TEXT    NOT NULL,   -- '21d' | '63d'
    ticker          TEXT    NOT NULL,
    rank_position   INTEGER NOT NULL,
    composite_score REAL    NOT NULL,
    score_breakdown TEXT    NOT NULL,   -- JSON: {"momentum_12_1": 0.5, "pe_relative": -0.2, ...}
    computed_at     TEXT    NOT NULL,
    UNIQUE(ranking_date, horizon, ticker)
);
CREATE INDEX IF NOT EXISTS idx_rank_date ON rankings(ranking_date);

-- Run metadata (track every system execution)
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date        TEXT    NOT NULL,
    run_type        TEXT    NOT NULL,   -- 'daily_full' | 'data_refresh' | 'signal_only' | 'manual'
    universe_name   TEXT    NOT NULL,
    tickers_count   INTEGER NOT NULL,
    signals_computed TEXT,              -- JSON list of signal names
    duration_secs   REAL,
    status          TEXT    NOT NULL,   -- 'success' | 'partial' | 'failed'
    error_msg       TEXT,
    metadata_json   TEXT,              -- any extra info
    created_at      TEXT    NOT NULL
);

-- Human views (from BBVA partner or manual input)
CREATE TABLE IF NOT EXISTS human_views (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    view_date       TEXT    NOT NULL,
    view            TEXT    NOT NULL,   -- 'bullish' | 'neutral' | 'bearish'
    expected_excess_return REAL,        -- vs market, annualized
    confidence      REAL,              -- [0, 1]
    horizon_months  INTEGER,
    rationale       TEXT,
    analyst         TEXT,              -- e.g. 'BBVA_WM'
    created_at      TEXT    NOT NULL,
    UNIQUE(ticker, view_date, analyst)
);
CREATE INDEX IF NOT EXISTS idx_hv_ticker ON human_views(ticker);

-- Signal evaluation results (research protocol)
CREATE TABLE IF NOT EXISTS signal_evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_name     TEXT    NOT NULL,
    eval_date       TEXT    NOT NULL,
    horizon         TEXT    NOT NULL,
    period_start    TEXT    NOT NULL,
    period_end      TEXT    NOT NULL,
    is_oos          INTEGER NOT NULL,   -- 1 = out-of-sample, 0 = in-sample
    ic_mean         REAL,
    ic_tstat        REAL,
    ic_bull         REAL,
    ic_bear         REAL,
    ic_highvol      REAL,
    ic_lowvol       REAL,
    spread_q5_q1    REAL,              -- annualized
    monotonicity    REAL,              -- fraction of quintiles in order
    ic_residual     REAL,              -- after factor attribution
    turnover_monthly REAL,
    alpha_net       REAL,              -- after costs
    verdict         TEXT,              -- 'PASS' | 'FAIL' | 'MARGINAL'
    notes           TEXT,
    created_at      TEXT    NOT NULL
);
"""


class MarketDB:
    """Interface to the SQLite market database."""

    def __init__(self, db_path: str | Path = "data/db/market.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.info("Database schema initialized at %s", self.db_path)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Fundamentals ──────────────────────────────────────────

    def upsert_fundamentals(
        self,
        ticker: str,
        period_date: str,
        filing_date: str,
        statement_type: str,
        data: dict,
        source: str = "fmp",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO fundamentals (ticker, period_date, filing_date,
                    statement_type, data_json, source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, period_date, statement_type, source)
                DO UPDATE SET data_json=excluded.data_json,
                              filing_date=excluded.filing_date,
                              fetched_at=excluded.fetched_at
                """,
                (ticker, period_date, filing_date, statement_type,
                 json.dumps(data), source, _now()),
            )

    def get_latest_fundamentals(
        self, ticker: str, statement_type: str = "income"
    ) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM fundamentals
                WHERE ticker = ? AND statement_type = ?
                ORDER BY period_date DESC LIMIT 1
                """,
                (ticker, statement_type),
            ).fetchone()
        if row is None:
            return None
        return {**dict(row), "data": json.loads(row["data_json"])}

    def get_fundamentals_history(
        self,
        ticker: str,
        statement_type: str = "income",
        limit: int = 20,
    ) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM fundamentals
                WHERE ticker = ? AND statement_type = ?
                ORDER BY period_date DESC LIMIT ?
                """,
                (ticker, statement_type, limit),
            ).fetchall()
        return [
            {**dict(r), "data": json.loads(r["data_json"])} for r in rows
        ]

    # ── Ratios ────────────────────────────────────────────────

    def upsert_ratios(
        self, ticker: str, period_date: str, filing_date: str,
        data: dict, source: str = "fmp",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO ratios (ticker, period_date, filing_date,
                    data_json, source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, period_date, source)
                DO UPDATE SET data_json=excluded.data_json,
                              fetched_at=excluded.fetched_at
                """,
                (ticker, period_date, filing_date,
                 json.dumps(data), source, _now()),
            )

    def get_latest_ratios(self, ticker: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM ratios WHERE ticker = ? ORDER BY period_date DESC LIMIT 1",
                (ticker,),
            ).fetchone()
        if row is None:
            return None
        return {**dict(row), "data": json.loads(row["data_json"])}

    # ── Earnings ──────────────────────────────────────────────

    def upsert_earnings(self, ticker: str, earnings: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO earnings (ticker, earnings_date, fiscal_period,
                    eps_estimated, eps_actual, revenue_estimated, revenue_actual,
                    surprise_pct, source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, earnings_date, source) DO UPDATE SET
                    eps_actual=excluded.eps_actual,
                    revenue_actual=excluded.revenue_actual,
                    surprise_pct=excluded.surprise_pct,
                    fetched_at=excluded.fetched_at
                """,
                (
                    ticker,
                    earnings.get("date"),
                    earnings.get("fiscalDateEnding"),
                    earnings.get("epsEstimated"),
                    earnings.get("eps"),
                    earnings.get("revenueEstimated"),
                    earnings.get("revenue"),
                    _calc_surprise(earnings.get("eps"), earnings.get("epsEstimated")),
                    "fmp",
                    _now(),
                ),
            )

    def get_earnings_history(self, ticker: str, limit: int = 12) -> pd.DataFrame:
        with self._conn() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM earnings WHERE ticker = ? ORDER BY earnings_date DESC LIMIT ?",
                conn,
                params=(ticker, limit),
            )
        return df

    # ── Profiles ──────────────────────────────────────────────

    def upsert_profile(self, ticker: str, data: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO profiles (ticker, company_name, sector, industry,
                    market_cap, country, exchange, data_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    company_name=excluded.company_name,
                    sector=excluded.sector,
                    market_cap=excluded.market_cap,
                    data_json=excluded.data_json,
                    updated_at=excluded.updated_at
                """,
                (
                    ticker,
                    data.get("companyName"),
                    data.get("sector"),
                    data.get("industry"),
                    data.get("mktCap"),
                    data.get("country"),
                    data.get("exchangeShortName"),
                    json.dumps(data),
                    _now(),
                ),
            )

    def get_profile(self, ticker: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE ticker = ?", (ticker,)
            ).fetchone()
        if row is None:
            return None
        return {**dict(row), "data": json.loads(row["data_json"])}

    def get_all_sectors(self) -> dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT ticker, sector FROM profiles").fetchall()
        return {r["ticker"]: r["sector"] for r in rows}

    # ── News ──────────────────────────────────────────────────

    def upsert_news(self, ticker: str, article: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO news (ticker, published_at, title, url,
                    source_name, text_snippet, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, url) DO NOTHING
                """,
                (
                    ticker,
                    article.get("publishedDate"),
                    article.get("title"),
                    article.get("url"),
                    article.get("site"),
                    (article.get("text") or "")[:500],
                    _now(),
                ),
            )

    def update_news_sentiment(
        self, news_id: int, score: float, confidence: float,
        horizon: str, drivers: list, risks: list, summary: str,
        model: str, prompt_version: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE news SET
                    sentiment_score = ?, confidence = ?, horizon = ?,
                    drivers_json = ?, risks_json = ?, llm_summary = ?,
                    llm_model = ?, prompt_version = ?, scored_at = ?
                WHERE id = ?
                """,
                (
                    score, confidence, horizon,
                    json.dumps(drivers), json.dumps(risks), summary,
                    model, prompt_version, _now(), news_id,
                ),
            )

    def get_unscored_news(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM news
                WHERE sentiment_score IS NULL
                ORDER BY published_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Signal Scores ─────────────────────────────────────────

    def upsert_signal(
        self, ticker: str, signal_date: str, signal_name: str,
        signal_value: float, score: float, horizon: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO signal_scores (ticker, signal_date, signal_name,
                    signal_value, score, horizon, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, signal_date, signal_name) DO UPDATE SET
                    signal_value=excluded.signal_value,
                    score=excluded.score,
                    computed_at=excluded.computed_at
                """,
                (ticker, signal_date, signal_name, signal_value, score,
                 horizon, _now()),
            )

    def get_signals_for_date(
        self, signal_date: str, signal_name: str | None = None,
    ) -> pd.DataFrame:
        with self._conn() as conn:
            if signal_name:
                df = pd.read_sql_query(
                    "SELECT * FROM signal_scores WHERE signal_date = ? AND signal_name = ?",
                    conn, params=(signal_date, signal_name),
                )
            else:
                df = pd.read_sql_query(
                    "SELECT * FROM signal_scores WHERE signal_date = ?",
                    conn, params=(signal_date,),
                )
        return df

    def get_signal_history(
        self, ticker: str, signal_name: str, limit: int = 252,
    ) -> pd.DataFrame:
        with self._conn() as conn:
            df = pd.read_sql_query(
                """
                SELECT signal_date, signal_value, score FROM signal_scores
                WHERE ticker = ? AND signal_name = ?
                ORDER BY signal_date DESC LIMIT ?
                """,
                conn, params=(ticker, signal_name, limit),
            )
        return df

    # ── Rankings ──────────────────────────────────────────────

    def save_ranking(
        self, ranking_date: str, horizon: str,
        rankings: list[dict],
    ) -> None:
        with self._conn() as conn:
            for r in rankings:
                conn.execute(
                    """
                    INSERT INTO rankings (ranking_date, horizon, ticker,
                        rank_position, composite_score, score_breakdown, computed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ranking_date, horizon, ticker) DO UPDATE SET
                        rank_position=excluded.rank_position,
                        composite_score=excluded.composite_score,
                        score_breakdown=excluded.score_breakdown,
                        computed_at=excluded.computed_at
                    """,
                    (
                        ranking_date, horizon, r["ticker"],
                        r["rank"], r["composite_score"],
                        json.dumps(r.get("breakdown", {})), _now(),
                    ),
                )

    def get_ranking(self, ranking_date: str, horizon: str = "21d") -> pd.DataFrame:
        with self._conn() as conn:
            df = pd.read_sql_query(
                """
                SELECT * FROM rankings
                WHERE ranking_date = ? AND horizon = ?
                ORDER BY rank_position ASC
                """,
                conn, params=(ranking_date, horizon),
            )
        return df

    # ── Runs ──────────────────────────────────────────────────

    def log_run(
        self, run_type: str, universe_name: str, tickers_count: int,
        signals: list[str] | None = None, duration: float | None = None,
        status: str = "success", error: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs (run_date, run_type, universe_name,
                    tickers_count, signals_computed, duration_secs,
                    status, error_msg, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now()[:10], run_type, universe_name, tickers_count,
                    json.dumps(signals) if signals else None,
                    duration, status, error,
                    json.dumps(metadata) if metadata else None,
                    _now(),
                ),
            )
            return cursor.lastrowid

    # ── Human Views ───────────────────────────────────────────

    def upsert_human_view(
        self, ticker: str, view_date: str, view: str,
        expected_excess_return: float | None = None,
        confidence: float | None = None,
        horizon_months: int | None = None,
        rationale: str | None = None,
        analyst: str = "manual",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO human_views (ticker, view_date, view,
                    expected_excess_return, confidence, horizon_months,
                    rationale, analyst, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, view_date, analyst) DO UPDATE SET
                    view=excluded.view,
                    expected_excess_return=excluded.expected_excess_return,
                    confidence=excluded.confidence,
                    rationale=excluded.rationale,
                    created_at=excluded.created_at
                """,
                (ticker, view_date, view, expected_excess_return,
                 confidence, horizon_months, rationale, analyst, _now()),
            )

    # ── Signal Evaluations ────────────────────────────────────

    def save_signal_evaluation(self, eval_data: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO signal_evaluations (
                    signal_name, eval_date, horizon, period_start, period_end,
                    is_oos, ic_mean, ic_tstat, ic_bull, ic_bear,
                    ic_highvol, ic_lowvol, spread_q5_q1, monotonicity,
                    ic_residual, turnover_monthly, alpha_net,
                    verdict, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eval_data["signal_name"], _now()[:10],
                    eval_data["horizon"],
                    eval_data["period_start"], eval_data["period_end"],
                    1 if eval_data.get("is_oos") else 0,
                    eval_data.get("ic_mean"),
                    eval_data.get("ic_tstat"),
                    eval_data.get("ic_bull"),
                    eval_data.get("ic_bear"),
                    eval_data.get("ic_highvol"),
                    eval_data.get("ic_lowvol"),
                    eval_data.get("spread_q5_q1"),
                    eval_data.get("monotonicity"),
                    eval_data.get("ic_residual"),
                    eval_data.get("turnover_monthly"),
                    eval_data.get("alpha_net"),
                    eval_data.get("verdict", "PENDING"),
                    eval_data.get("notes"),
                    _now(),
                ),
            )

    # ── Bulk queries ──────────────────────────────────────────

    def get_all_fundamentals_for_universe(
        self, tickers: list[str], statement_type: str = "income",
    ) -> pd.DataFrame:
        placeholders = ",".join("?" * len(tickers))
        with self._conn() as conn:
            df = pd.read_sql_query(
                f"""
                SELECT ticker, period_date, filing_date, data_json
                FROM fundamentals
                WHERE ticker IN ({placeholders}) AND statement_type = ?
                ORDER BY ticker, period_date DESC
                """,
                conn,
                params=[*tickers, statement_type],
            )
        return df


# ── Helpers ───────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _calc_surprise(actual: float | None, estimated: float | None) -> float | None:
    if actual is None or estimated is None or estimated == 0:
        return None
    return (actual - estimated) / abs(estimated)
