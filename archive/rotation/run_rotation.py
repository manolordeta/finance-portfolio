#!/usr/bin/env python3
"""
MM Quant Capital — Rotation Strategy

Short-term GARCH-scaled barrier trading:
  Buy top-ranked stocks, set volatility-adjusted take-profit and stop-loss,
  exit on barrier hit, rotate into next opportunity.

Usage:
  python run_rotation.py backtest              # find optimal parameters
  python run_rotation.py scan                  # what to buy today
  python run_rotation.py status               # check open positions
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

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
log = logging.getLogger("rotation")


def cmd_backtest(args):
    """Grid search for optimal rotation parameters."""
    import numpy as np
    import pandas as pd
    import sqlite3
    import yfinance as yf

    from src.signals import technical, fundamental, valuation
    from src.data.database import MarketDB
    from src.rotation.engine import grid_search, run_backtest

    log.info("=" * 65)
    log.info("  ROTATION STRATEGY — Parameter Optimization")
    log.info("=" * 65)

    # Load data
    log.info("[1/4] Loading data...")
    db = MarketDB("data/db/market.db")
    conn = sqlite3.connect("data/db/market.db")
    all_tickers = [r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM profiles").fetchall()]
    conn.close()

    data = yf.download(all_tickers + ["SPY"], start="2023-01-01", progress=False)
    prices = data["Close"].dropna(axis=1, thresh=int(len(data) * 0.5))
    volumes = data["Volume"].reindex(columns=prices.columns)
    log.info("  %d days x %d tickers", len(prices), len(prices.columns))

    # Compute signals and ranking scores
    log.info("[2/4] Computing signals...")
    tech = technical.compute_all(prices, volumes)
    tickers_with_data = [t for t in all_tickers if t in prices.columns]
    fund = fundamental.compute_all(db, tickers_with_data)
    val = valuation.compute_all(prices)

    # Build composite score time series (equal-weight for backtest simplicity)
    log.info("  Building composite ranking time series...")
    signal_dfs = {}
    for name, df in tech.items():
        signal_dfs[name] = df.reindex(columns=prices.columns)
    for name, df in val.items():
        signal_dfs[name] = df.reindex(columns=prices.columns)
    for name, scores in fund.items():
        s = pd.Series(scores)
        signal_dfs[name] = pd.DataFrame(
            np.tile(s.values, (len(prices), 1)),
            index=prices.index, columns=s.index,
        ).reindex(columns=prices.columns)

    # Simple equal-weight composite (for backtesting, not IC-weighted)
    all_signals = list(signal_dfs.values())
    ranking_scores = sum(all_signals) / len(all_signals)
    ranking_scores = ranking_scores.dropna(axis=1, thresh=int(len(ranking_scores) * 0.5))

    # Grid search
    log.info("[3/4] Running grid search...")
    log.info("  k_tp:     %s", args.k_tp_range)
    log.info("  k_sl:     %s", args.k_sl_range)
    log.info("  max_days: %s", args.max_days_range)
    log.info("  positions: %d", args.n_positions)
    t0 = time.time()

    results = grid_search(
        prices=prices,
        ranking_scores=ranking_scores,
        k_tp_range=args.k_tp_range,
        k_sl_range=args.k_sl_range,
        max_days_range=args.max_days_range,
        n_positions=args.n_positions,
        cost_bps=args.cost_bps,
    )

    log.info("  Grid search complete (%.1fs, %d combinations)", time.time() - t0, len(results))

    # Display results
    log.info("[4/4] Results:\n")

    print("=" * 100)
    print("  ROTATION STRATEGY — PARAMETER OPTIMIZATION RESULTS")
    print("  Period: %s to %s (%d days)" % (
        prices.index[63].strftime("%Y-%m-%d"),
        prices.index[-1].strftime("%Y-%m-%d"),
        len(prices) - 63,
    ))
    print("  Universe: %d tickers | Cost: %d bps/trade" % (len(ranking_scores.columns), args.cost_bps))
    print("=" * 100)

    print("\n  TOP 10 CONFIGURATIONS (by Sharpe):\n")
    print("  %5s  %5s  %5s  %6s  %6s  %8s  %8s  %8s  %8s  %6s  %5s  %5s  %5s" % (
        "k_tp", "k_sl", "days", "trades", "hit%", "avg_win", "avg_loss", "ann_ret", "sharpe", "maxDD", "tp%", "sl%", "tm%"))
    print("  " + "─" * 95)

    for _, row in results.head(10).iterrows():
        tp_pct = row["tp_exits"] / max(row["total_trades"], 1) * 100
        sl_pct = row["sl_exits"] / max(row["total_trades"], 1) * 100
        tm_pct = row["time_exits"] / max(row["total_trades"], 1) * 100
        print("  %5.1f  %5.2f  %5d  %6d  %5.1f%%  %+7.2f%%  %+7.2f%%  %+7.1f%%  %+7.2f  %+5.1f%%  %4.0f%%  %4.0f%%  %4.0f%%" % (
            row["k_tp"], row["k_sl"], row["max_days"],
            row["total_trades"], row["hit_rate"] * 100,
            row["avg_win"] * 100, row["avg_loss"] * 100,
            row["ann_return"] * 100, row["sharpe"],
            row["max_drawdown"] * 100,
            tp_pct, sl_pct, tm_pct,
        ))

    # Show best config detail
    best = results.iloc[0]
    print("\n" + "─" * 100)
    print("  BEST CONFIGURATION:")
    print("    k_tp = %.1f  (take profit = %.1f × σ × √days)" % (best["k_tp"], best["k_tp"]))
    print("    k_sl = %.2f  (stop loss = %.2f × σ × √days)" % (best["k_sl"], best["k_sl"]))
    print("    max_days = %d" % best["max_days"])
    print()
    print("    Example barriers for different volatilities:")
    for name, sigma in [("Low vol (σ=2%%)", 0.02), ("Med vol (σ=4%%)", 0.04), ("High vol (σ=6%%)", 0.06)]:
        tp = best["k_tp"] * sigma * (best["max_days"] ** 0.5) * 100
        sl = best["k_sl"] * sigma * (best["max_days"] ** 0.5) * 100
        print("      %s: tp=+%.1f%%, sl=-%.1f%%" % (name, tp, sl))
    print()
    print("    Expectancy per trade: %+.3f%%" % (best["expectancy"] * 100))
    print("    Estimated annual trades: %d" % (best["total_trades"] / max((len(prices) - 63) / 252, 0.01)))
    print("=" * 100)

    # Run detailed backtest with best params
    log.info("\n  Running detailed backtest with best params...")
    detail = run_backtest(
        prices=prices,
        ranking_scores=ranking_scores,
        k_tp=best["k_tp"],
        k_sl=best["k_sl"],
        max_days=int(best["max_days"]),
        n_positions=args.n_positions,
        cost_per_trade_bps=args.cost_bps,
    )

    # Show sample trades
    completed = [t for t in detail.trades if t.exit_reason != "open"]
    if completed:
        print("\n  SAMPLE TRADES (last 15):\n")
        print("  %6s  %10s  %10s  %5s  %8s  %6s  %6s  σ_day  exit" % (
            "ticker", "entry", "exit", "days", "return", "tp", "sl"))
        print("  " + "─" * 75)
        for t in completed[-15:]:
            print("  %6s  %10s  %10s  %5d  %+7.2f%%  %+5.1f%%  %+5.1f%%  %4.1f%%  %s" % (
                t.ticker, t.entry_date, t.exit_date, t.holding_days,
                t.return_pct * 100, t.tp_pct * 100, -t.sl_pct * 100,
                t.sigma_daily * 100, t.exit_reason,
            ))

    # Save results
    results.to_csv("reports/rotation_grid_search.csv", index=False)
    log.info("\n  Grid search saved: reports/rotation_grid_search.csv")

    # Save best params to config
    log.info("  Best params: k_tp=%.1f, k_sl=%.2f, max_days=%d, sharpe=%.2f",
             best["k_tp"], best["k_sl"], best["max_days"], best["sharpe"])


def cmd_scan(args):
    """Scan for today's rotation opportunities."""
    import numpy as np
    import pandas as pd
    import sqlite3
    import yfinance as yf
    import yaml

    from src.data.database import MarketDB
    from src.rotation.engine import compute_garch_vol

    log.info("=" * 65)
    log.info("  ROTATION STRATEGY — Daily Scan")
    log.info("=" * 65)

    # Load config
    with open("config/signals.yaml") as f:
        cfg = yaml.safe_load(f)
    rot = cfg.get("rotation", {})
    k_tp = args.k_tp or rot.get("k_tp", 1.5)
    k_sl = args.k_sl or rot.get("k_sl", 0.75)
    max_days = args.max_days or rot.get("max_days", 15)
    n_positions = args.n_positions or rot.get("n_positions", 10)

    # Load latest ranking from DB
    db = MarketDB("data/db/market.db")
    conn = sqlite3.connect("data/db/market.db")

    ranking_rows = conn.execute(
        """SELECT ticker, rank_position, composite_score
           FROM dual_rankings
           WHERE ranking_date = (SELECT MAX(ranking_date) FROM dual_rankings)
           AND model = 'gics'
           ORDER BY rank_position ASC"""
    ).fetchall()

    profiles = {r[0]: {"name": r[1], "sector": r[2]} for r in conn.execute(
        "SELECT ticker, company_name, sector FROM profiles").fetchall()}
    conn.close()

    if not ranking_rows:
        log.error("No ranking data found. Run run_daily.py first.")
        return

    # Get top N candidates
    top_tickers = [r[0] for r in ranking_rows[:n_positions * 2]]

    # Load open positions if any
    positions_file = Path("data/rotation_positions.json")
    open_positions = {}
    if positions_file.exists():
        open_positions = json.loads(positions_file.read_text())

    # Exclude already held
    candidates = [t for t in top_tickers if t not in open_positions][:n_positions]

    # Download recent prices for vol calculation
    log.info("[1/2] Downloading prices for vol estimation...")
    data = yf.download(candidates + ["SPY"], start="2024-06-01", progress=False)
    prices = data["Close"]
    returns = prices.pct_change()

    log.info("[2/2] Computing barriers...\n")

    today = datetime.now().strftime("%Y-%m-%d")

    print("=" * 85)
    print("  ROTATION SCAN — %s" % today)
    print("  Parameters: k_tp=%.1f  k_sl=%.2f  max_days=%d" % (k_tp, k_sl, max_days))
    print("=" * 85)

    print("\n  %5s  %6s  %5s  %7s  %7s  %6s  %5s  %20s  %s" % (
        "Rank", "Ticker", "Score", "TP", "SL", "σ/day", "Days", "Company", "Sector"))
    print("  " + "─" * 82)

    scan_results = []
    for ticker in candidates:
        rank_info = next((r for r in ranking_rows if r[0] == ticker), None)
        if rank_info is None:
            continue
        rank, score = rank_info[1], rank_info[2]

        if ticker not in returns.columns:
            continue

        sigma = compute_garch_vol(returns[ticker].dropna())
        tp_pct = k_tp * sigma * np.sqrt(max_days)
        sl_pct = k_sl * sigma * np.sqrt(max_days)

        tp_pct = max(tp_pct, 0.01)
        sl_pct = max(sl_pct, 0.005)

        p = profiles.get(ticker, {"name": "?", "sector": "?"})
        current_price = prices[ticker].dropna().iloc[-1]

        print("  %5d  %6s  %+.3f  %+6.1f%%  %+6.1f%%  %5.1f%%  %5d  %20s  %s" % (
            rank, ticker, score,
            tp_pct * 100, -sl_pct * 100,
            sigma * 100, max_days,
            p["name"][:20], p["sector"][:18],
        ))

        scan_results.append({
            "ticker": ticker,
            "rank": rank,
            "score": score,
            "price": float(current_price),
            "sigma_daily": float(sigma),
            "tp_pct": float(tp_pct),
            "sl_pct": float(sl_pct),
            "tp_price": float(current_price * (1 + tp_pct)),
            "sl_price": float(current_price * (1 - sl_pct)),
            "max_days": max_days,
        })

    # Show price targets
    print("\n  ENTRY TARGETS:")
    print("  %6s  %10s  %10s  %10s" % ("Ticker", "Entry", "TP Price", "SL Price"))
    print("  " + "─" * 42)
    for r in scan_results:
        print("  %6s  $%8.2f  $%8.2f  $%8.2f" % (
            r["ticker"], r["price"], r["tp_price"], r["sl_price"]))

    if open_positions:
        print("\n  OPEN POSITIONS: %d (slots used)" % len(open_positions))
        print("  Slots available: %d" % (n_positions - len(open_positions)))

    print("\n" + "=" * 85)


def cmd_status(args):
    """Check status of open rotation positions."""
    import numpy as np
    import yfinance as yf

    positions_file = Path("data/rotation_positions.json")

    if not positions_file.exists():
        print("No open positions. Use 'scan' to find opportunities.")
        return

    positions = json.loads(positions_file.read_text())
    if not positions:
        print("No open positions.")
        return

    tickers = list(positions.keys())
    data = yf.download(tickers, period="5d", progress=False)
    prices = data["Close"]

    today = datetime.now().strftime("%Y-%m-%d")
    print("=" * 80)
    print("  ROTATION POSITIONS — %s" % today)
    print("=" * 80)

    print("\n  %6s  %10s  %8s  %8s  %8s  %6s  %5s  %6s" % (
        "Ticker", "Entry", "Entry$", "Now$", "Return", "TP", "SL", "Days"))
    print("  " + "─" * 68)

    for ticker, pos in positions.items():
        current = prices[ticker].dropna().iloc[-1] if ticker in prices.columns else pos["entry_price"]
        ret = (current - pos["entry_price"]) / pos["entry_price"]
        entry_date = pos.get("entry_date", "?")
        days_held = (datetime.now() - datetime.strptime(entry_date, "%Y-%m-%d")).days if entry_date != "?" else 0

        # Status indicator
        if ret >= pos["tp_pct"]:
            status = " ✅ HIT TP"
        elif ret <= -pos["sl_pct"]:
            status = " 🔴 HIT SL"
        elif days_held >= pos["max_days"]:
            status = " ⏰ TIME UP"
        else:
            status = ""

        print("  %6s  %10s  $%7.2f  $%7.2f  %+6.1f%%  %+5.1f%%  %5.1f%%  %5d%s" % (
            ticker, entry_date, pos["entry_price"], current,
            ret * 100, pos["tp_pct"] * 100, -pos["sl_pct"] * 100,
            days_held, status,
        ))

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="MM Quant Capital — Rotation Strategy")
    sub = parser.add_subparsers(dest="command")

    # Backtest
    bt = sub.add_parser("backtest", help="Find optimal parameters via grid search")
    bt.add_argument("--k-tp", type=str, default="1.0,1.5,2.0,2.5",
                    help="k_tp values to test (comma-separated)")
    bt.add_argument("--k-sl", type=str, default="0.5,0.75,1.0,1.5",
                    help="k_sl values to test (comma-separated)")
    bt.add_argument("--max-days", type=str, default="10,15,21",
                    help="max_days values to test (comma-separated)")
    bt.add_argument("--n-positions", type=int, default=10, help="Max simultaneous positions")
    bt.add_argument("--cost-bps", type=float, default=12, help="Cost per trade in bps")

    # Scan
    sc = sub.add_parser("scan", help="Scan for today's opportunities")
    sc.add_argument("--k-tp", type=float, default=None, help="k_tp override")
    sc.add_argument("--k-sl", type=float, default=None, help="k_sl override")
    sc.add_argument("--max-days", type=int, default=None, help="max_days override")
    sc.add_argument("--n-positions", type=int, default=None, help="Max positions")

    # Status
    sub.add_parser("status", help="Check open positions")

    args = parser.parse_args()

    if args.command == "backtest":
        args.k_tp_range = [float(x) for x in args.k_tp.split(",")]
        args.k_sl_range = [float(x) for x in args.k_sl.split(",")]
        args.max_days_range = [int(x) for x in args.max_days.split(",")]
        cmd_backtest(args)
    elif args.command == "scan":
        cmd_scan(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
