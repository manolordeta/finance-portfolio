#!/usr/bin/env python3
"""
MM Quant Capital — Portfolio Tracker

Two modes:
  1. IBKR API: reads positions automatically from TWS/Gateway
  2. Manual: you input positions via config file

Usage:
  python run_tracker.py                    # auto-detect: IBKR if available, else manual
  python run_tracker.py --ibkr             # force IBKR connection (live, port 7496)
  python run_tracker.py --ibkr --paper     # IBKR paper trading (port 7497)
  python run_tracker.py --manual           # read from config/positions.yaml
  python run_tracker.py --history          # show P&L history
  python run_tracker.py --compare          # compare portfolio vs ranking vs SPY
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tracker")

DB_PATH = "data/db/market.db"
POSITIONS_PATH = "config/positions.yaml"


# ── Database setup ────────────────────────────────────────────────────

def init_tracker_tables():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date   TEXT NOT NULL,
            source          TEXT NOT NULL,   -- 'ibkr_live' | 'ibkr_paper' | 'manual'
            ticker          TEXT NOT NULL,
            shares          REAL,
            avg_cost        REAL,
            current_price   REAL,
            market_value    REAL,
            weight_pct      REAL,           -- % of total portfolio
            pnl_pct         REAL,           -- gain/loss from avg_cost
            rank_position   INTEGER,
            composite_score REAL,
            rank_zone       TEXT,           -- 'TOP20' | 'TOP40' | 'MID' | 'BTM'
            created_at      TEXT NOT NULL,
            UNIQUE(snapshot_date, source, ticker)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_daily (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date   TEXT NOT NULL,
            source          TEXT NOT NULL,
            total_value     REAL,
            total_cost      REAL,
            total_pnl       REAL,
            total_pnl_pct   REAL,
            n_positions     INTEGER,
            spy_price       REAL,
            spy_return_since_start REAL,
            portfolio_return_since_start REAL,
            alpha_since_start REAL,
            created_at      TEXT NOT NULL,
            UNIQUE(snapshot_date, source)
        )
    """)
    conn.commit()
    conn.close()


# ── Position loading ──────────────────────────────────────────────────

def load_positions_manual(path: str = POSITIONS_PATH) -> list[dict]:
    """Load positions from YAML config file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    positions = []
    for pos in cfg.get("positions", []):
        positions.append({
            "ticker": pos["ticker"],
            "shares": float(pos["shares"]),
            "avg_cost": float(pos["avg_cost"]),
        })
    return positions


def load_positions_ibkr(port: int = 7496) -> list[dict]:
    """Load positions from IBKR TWS/Gateway."""
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect("127.0.0.1", port, clientId=10, timeout=10)
        log.info("Connected to IBKR on port %d", port)

        positions = []
        for pos in ib.positions():
            if pos.position == 0:
                continue
            positions.append({
                "ticker": pos.contract.symbol,
                "shares": float(pos.position),
                "avg_cost": float(pos.avgCost),
            })

        ib.disconnect()
        return positions

    except Exception as e:
        log.error("IBKR connection failed: %s", e)
        return []


def enrich_with_prices(positions: list[dict]) -> list[dict]:
    """Add current prices from yfinance."""
    import yfinance as yf

    tickers = [p["ticker"] for p in positions]
    if not tickers:
        return positions

    data = yf.download(tickers, period="5d", progress=False)
    if len(data) == 0:
        return positions

    latest_prices = data["Close"].iloc[-1]

    for pos in positions:
        t = pos["ticker"]
        price = latest_prices.get(t)
        if price is not None and np.isfinite(price):
            pos["current_price"] = float(price)
            pos["market_value"] = pos["shares"] * float(price)
            pos["pnl_pct"] = (float(price) - pos["avg_cost"]) / pos["avg_cost"]
        else:
            pos["current_price"] = pos["avg_cost"]
            pos["market_value"] = pos["shares"] * pos["avg_cost"]
            pos["pnl_pct"] = 0.0

    # Calculate weights
    total = sum(p["market_value"] for p in positions)
    for pos in positions:
        pos["weight_pct"] = pos["market_value"] / total if total > 0 else 0

    return positions


def enrich_with_ranking(positions: list[dict]) -> list[dict]:
    """Add ranking data from DB."""
    conn = sqlite3.connect(DB_PATH)
    ranking = {}
    try:
        rows = conn.execute(
            """SELECT ticker, rank_position, composite_score
               FROM dual_rankings
               WHERE ranking_date = (SELECT MAX(ranking_date) FROM dual_rankings)
               AND model = 'gics'"""
        ).fetchall()
        total = len(rows)
        for r in rows:
            ranking[r[0]] = {"rank": r[1], "score": r[2], "total": total}
    except Exception:
        total = 0

    profiles = {}
    try:
        rows = conn.execute("SELECT ticker, company_name, sector FROM profiles").fetchall()
        for r in rows:
            profiles[r[0]] = {"name": r[1], "sector": r[2]}
    except Exception:
        pass
    conn.close()

    for pos in positions:
        t = pos["ticker"]
        r = ranking.get(t, {})
        p = profiles.get(t, {})
        pos["rank"] = r.get("rank")
        pos["score"] = r.get("score")
        pos["total_ranked"] = r.get("total", total)
        pos["company"] = p.get("name", "")
        pos["sector"] = p.get("sector", "")

        if pos["rank"] and pos["total_ranked"]:
            pct = pos["rank"] / pos["total_ranked"]
            pos["zone"] = "TOP20" if pct <= 0.20 else "TOP40" if pct <= 0.40 else "MID" if pct <= 0.60 else "BTM"
        else:
            pos["zone"] = "N/A"

    return positions


# ── Display ───────────────────────────────────────────────────────────

def display_portfolio(positions: list[dict], source: str):
    total_value = sum(p["market_value"] for p in positions)
    total_cost = sum(p["shares"] * p["avg_cost"] for p in positions)
    total_pnl = total_value - total_cost
    total_pnl_pct = total_pnl / total_cost if total_cost > 0 else 0

    print(f"\n{'=' * 85}")
    print(f"  MM QUANT CAPITAL — Portfolio Tracker")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Source: {source}")
    print(f"{'=' * 85}")

    print(f"\n  SUMMARY:")
    print(f"    Total Value:    ${total_value:>10,.2f}")
    print(f"    Total Cost:     ${total_cost:>10,.2f}")
    print(f"    Total P&L:      ${total_pnl:>+10,.2f} ({total_pnl_pct:+.2%})")
    print(f"    Positions:      {len(positions)}")

    print(f"\n  {'Zone':>6s} {'Ticker':>7s} {'Shares':>7s} {'AvgCost':>9s} {'Price':>9s} "
          f"{'Value':>10s} {'Wt%':>5s} {'P&L%':>7s} {'Rank':>6s} {'Score':>7s} {'Sector':>16s}")
    print(f"  {'-' * 92}")

    for pos in sorted(positions, key=lambda p: -p["market_value"]):
        zone = pos.get("zone", "N/A")
        if zone == "TOP20":
            z = "🟢"
        elif zone == "TOP40":
            z = "🟡"
        elif zone == "BTM":
            z = "🔴"
        else:
            z = "  "

        rank_str = f"#{pos['rank']}" if pos.get("rank") else "N/A"
        score_str = f"{pos['score']:+.3f}" if pos.get("score") else "N/A"

        print(f"  {z}{zone:>4s} {pos['ticker']:>7s} {pos['shares']:>7.1f} "
              f"${pos['avg_cost']:>8.2f} ${pos.get('current_price', 0):>8.2f} "
              f"${pos['market_value']:>9.2f} {pos['weight_pct']*100:>4.1f}% "
              f"{pos.get('pnl_pct', 0):>+6.2%} {rank_str:>6s} {score_str:>7s} "
              f"{pos.get('sector', '?')[:16]:>16s}")

    # Warnings
    warnings_list = []
    for pos in positions:
        if pos.get("zone") == "BTM":
            warnings_list.append(
                f"  ⚠️  {pos['ticker']} ranked #{pos['rank']} — BOTTOM zone, review position")
        elif pos.get("zone") == "MID":
            warnings_list.append(
                f"  📌 {pos['ticker']} ranked #{pos['rank']} — MID zone, monitor")

    if warnings_list:
        print(f"\n  ALERTS:")
        for w in warnings_list:
            print(w)

    print(f"\n{'=' * 85}")
    return total_value, total_cost, total_pnl, total_pnl_pct


def save_snapshot(positions: list[dict], source: str,
                  total_value: float, total_cost: float):
    """Save to DB for historical tracking."""
    conn = sqlite3.connect(DB_PATH)
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().isoformat()

    for pos in positions:
        conn.execute(
            """INSERT OR REPLACE INTO portfolio_snapshots
               (snapshot_date, source, ticker, shares, avg_cost, current_price,
                market_value, weight_pct, pnl_pct, rank_position, composite_score,
                rank_zone, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (today, source, pos["ticker"], pos["shares"], pos["avg_cost"],
             pos.get("current_price"), pos["market_value"], pos["weight_pct"],
             pos.get("pnl_pct"), pos.get("rank"), pos.get("score"),
             pos.get("zone"), now),
        )

    # Get SPY price for comparison
    try:
        import yfinance as yf
        spy = yf.download("SPY", period="5d", progress=False)
        spy_price = float(spy["Close"].iloc[-1])
    except Exception:
        spy_price = None

    total_pnl = total_value - total_cost
    total_pnl_pct = total_pnl / total_cost if total_cost > 0 else 0

    conn.execute(
        """INSERT OR REPLACE INTO portfolio_daily
           (snapshot_date, source, total_value, total_cost, total_pnl,
            total_pnl_pct, n_positions, spy_price,
            spy_return_since_start, portfolio_return_since_start,
            alpha_since_start, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (today, source, total_value, total_cost, total_pnl,
         total_pnl_pct, len(positions), spy_price,
         None, None, None, now),
    )

    conn.commit()
    conn.close()
    log.info("Snapshot saved for %s (%d positions)", today, len(positions))


def show_history():
    """Show P&L history."""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            """SELECT snapshot_date, source, total_value, total_pnl, total_pnl_pct, n_positions
               FROM portfolio_daily
               ORDER BY snapshot_date DESC LIMIT 30"""
        ).fetchall()

        if not rows:
            print("\nNo history yet. Run the tracker first.")
            return

        print(f"\n{'=' * 70}")
        print(f"  Portfolio P&L History")
        print(f"{'=' * 70}")
        print(f"  {'Date':>12s} {'Source':>8s} {'Value':>12s} {'P&L':>10s} {'P&L%':>8s} {'#Pos':>5s}")
        print(f"  {'-' * 58}")

        for r in rows:
            val = f"${r[2]:,.2f}" if r[2] else "N/A"
            pnl = f"${r[3]:+,.2f}" if r[3] else "N/A"
            pnl_pct = f"{r[4]:+.2%}" if r[4] else "N/A"
            print(f"  {r[0]:>12s} {r[1]:>8s} {val:>12s} {pnl:>10s} {pnl_pct:>8s} {r[5]:>5d}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Portfolio Tracker")
    parser.add_argument("--ibkr", action="store_true", help="Connect to IBKR")
    parser.add_argument("--paper", action="store_true", help="IBKR paper trading port")
    parser.add_argument("--manual", action="store_true", help="Read from config/positions.yaml")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--history", action="store_true", help="Show P&L history")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    init_tracker_tables()

    if args.history:
        show_history()
        return

    # Load positions
    if args.ibkr:
        port = args.port or (7497 if args.paper else 7496)
        source = "ibkr_paper" if args.paper else "ibkr_live"
        positions = load_positions_ibkr(port)
        if not positions:
            log.error("No positions from IBKR. Falling back to manual.")
            args.manual = True

    if args.manual or not args.ibkr:
        source = "manual"
        if not Path(POSITIONS_PATH).exists():
            log.info("Creating template positions file: %s", POSITIONS_PATH)
            template = {
                "# MM Quant Capital — Current Positions": None,
                "# Update this file with your actual positions": None,
                "# Then run: python run_tracker.py --manual": None,
                "positions": [
                    {"ticker": "SPY", "shares": 1, "avg_cost": 560.00},
                ],
            }
            with open(POSITIONS_PATH, "w") as f:
                yaml.dump({"positions": template["positions"]}, f, default_flow_style=False)
            log.info("Edit %s with your real positions, then run again", POSITIONS_PATH)
            return

        positions = load_positions_manual()

    if not positions:
        log.error("No positions found")
        return

    # Enrich with current prices and ranking
    log.info("Fetching current prices...")
    positions = enrich_with_prices(positions)
    positions = enrich_with_ranking(positions)

    # Display
    total_value, total_cost, total_pnl, total_pnl_pct = display_portfolio(positions, source)

    # Save
    if not args.no_save:
        save_snapshot(positions, source, total_value, total_cost)


if __name__ == "__main__":
    main()
