#!/usr/bin/env python3
"""
MM Quant Capital — Daily Run Script (v2 — Dual Model)

Executes the full daily pipeline with GICS + Cluster models in parallel:
  1. Refresh price data (yfinance)
  2. Compute all signals (technical + fundamental + valuation)
  3. Build dual rankings: GICS-weighted + Cluster-weighted
  4. Run LLM alerts on watchlist (cached — won't regenerate same day)
  5. Track forward test (compare models)
  6. Generate PDF report

Usage:
  python run_daily.py                  # normal daily run
  python run_daily.py --refresh-alerts # force regenerate alerts
  python run_daily.py --weekly         # weekly tasks (refresh fundamentals)
  python run_daily.py --calibrate      # monthly recalibration (recalculate ICs + clusters)
  python run_daily.py --no-report      # skip PDF generation

Cost per run:  ~$0.003 (daily) | ~$0.15 (weekly)
Time per run:  ~45s (daily) | ~5 min (weekly) | ~8 min (calibrate)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
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
log = logging.getLogger("daily_run")


def main():
    parser = argparse.ArgumentParser(description="MM Quant Capital — Daily Pipeline v2")
    parser.add_argument("--refresh-alerts", action="store_true",
                        help="Force regenerate LLM alerts (ignore cache)")
    parser.add_argument("--weekly", action="store_true",
                        help="Run weekly tasks (refresh fundamentals from FMP)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Monthly recalibration (recompute ICs, clusters, weights)")
    parser.add_argument("--no-report", action="store_true",
                        help="Run pipeline but skip PDF generation")
    args = parser.parse_args()

    t_start = time.time()
    today = datetime.now().strftime("%Y-%m-%d")
    log.info("=" * 65)
    log.info("  MM QUANT CAPITAL — Daily Pipeline v2 (Dual Model)")
    log.info("  Date: %s", today)
    log.info("=" * 65)

    import numpy as np
    import pandas as pd
    import yfinance as yf

    from src.data.database import MarketDB
    from src.data.fmp_client import FMPClient
    from src.signals import technical, fundamental, valuation
    from src.backtest.calibration import (
        calibrate_gics, calibrate_clusters,
        save_calibration, print_calibration_report,
    )
    from src.alerts.alert_system import AlertSystem
    from src.reports.pdf_report import generate_report

    db = MarketDB("data/db/market.db")

    # ── Step 1: Load universe ────────────────────────────────────
    log.info("[1/7] Loading universe...")
    conn = sqlite3.connect("data/db/market.db")
    all_tickers = [r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM profiles").fetchall()]
    sectors = {r[0]: r[1] for r in conn.execute(
        "SELECT ticker, sector FROM profiles").fetchall()}
    profiles = {r[0]: {"name": r[1], "sector": r[2]} for r in conn.execute(
        "SELECT ticker, company_name, sector FROM profiles").fetchall()}

    # Load latest calibration weights
    calib_rows = conn.execute(
        """SELECT model, group_name, signal_name, weight
           FROM calibrations
           WHERE calibration_date = (SELECT MAX(calibration_date) FROM calibrations)"""
    ).fetchall()
    conn.close()

    # Build weight lookups: {model: {group: {signal: weight}}}
    calib_weights = {"gics": {}, "cluster": {}}
    for row in calib_rows:
        model, group, signal, weight = row
        calib_weights.setdefault(model, {}).setdefault(group, {})[signal] = weight

    has_calibration = len(calib_rows) > 0
    log.info("  Universe: %d tickers | Calibration: %s",
             len(all_tickers), "loaded" if has_calibration else "NOT FOUND — run with --calibrate")

    # ── Step 2: Refresh prices ───────────────────────────────────
    log.info("[2/7] Downloading prices...")
    t0 = time.time()
    data = yf.download(all_tickers + ["SPY"], start="2023-01-01", progress=False)
    prices = data["Close"].dropna(axis=1, thresh=int(len(data) * 0.5))
    volumes = data["Volume"].reindex(columns=prices.columns)
    returns = prices.pct_change()
    ret_1m = prices.pct_change(21).iloc[-1]
    ret_3m = prices.pct_change(63).iloc[-1]
    tickers_with_data = [t for t in all_tickers if t in prices.columns]
    log.info("  Got %d days x %d tickers (%.1fs)",
             len(prices), len(prices.columns), time.time() - t0)

    # ── Step 2b: Weekly refresh fundamentals ─────────────────────
    if args.weekly:
        log.info("[2b] WEEKLY: Refreshing fundamentals from FMP...")
        client = FMPClient()
        refreshed = 0
        for ticker in all_tickers:
            try:
                for stmt_type, fn in [("income", client.get_income_statement),
                                      ("balance", client.get_balance_sheet),
                                      ("cashflow", client.get_cash_flow)]:
                    stmts = fn(ticker, period="quarterly", limit=4)
                    if stmts:
                        for s in stmts:
                            db.upsert_fundamentals(ticker, s.get("date", ""),
                                s.get("fillingDate", s.get("date", "")), stmt_type, s)
                ratios = client.get_ratios(ticker, period="quarterly", limit=4)
                if ratios:
                    for r in ratios:
                        db.upsert_ratios(ticker, r.get("date", ""), r.get("date", ""), r)
                refreshed += 1
                if refreshed % 50 == 0:
                    log.info("    Refreshed %d/%d", refreshed, len(all_tickers))
            except Exception as e:
                log.warning("    Failed %s: %s", ticker, e)
        log.info("  Refreshed %d tickers", refreshed)

    # ── Step 3: Compute signals ──────────────────────────────────
    log.info("[3/7] Computing signals...")
    t0 = time.time()
    tech = technical.compute_all(prices, volumes)
    fund = fundamental.compute_all(db, tickers_with_data)
    val = valuation.compute_all(prices)
    log.info("  Tech: %d | Fund: %d | Val: %d (%.1fs)",
             len(tech), len(fund), len(val), time.time() - t0)

    # Build unified signal DFs
    signal_dfs = {}
    for name, df in tech.items():
        signal_dfs[f"T_{name}"] = df.reindex(index=prices.index, columns=prices.columns)
    for name, df in val.items():
        signal_dfs[f"V_{name}"] = df.reindex(index=prices.index, columns=prices.columns)
    for name, scores in fund.items():
        s = pd.Series(scores)
        signal_dfs[f"F_{name}"] = pd.DataFrame(
            np.tile(s.values, (len(prices), 1)),
            index=prices.index, columns=s.index,
        ).reindex(columns=prices.columns)

    # ── Step 3b: Calibrate (monthly) ─────────────────────────────
    if args.calibrate:
        log.info("[3b] CALIBRATION: Recomputing ICs, clusters, weights...")
        t0 = time.time()
        fwd_21 = prices.pct_change(21).shift(-21)
        train_dates = prices.index[-252:]

        gics_result = calibrate_gics(
            signal_dfs, fwd_21, train_dates, sectors, tickers_with_data)
        cluster_result = calibrate_clusters(
            signal_dfs, fwd_21, returns, train_dates, sectors, tickers_with_data,
            n_clusters=8)

        save_calibration(gics_result, db)
        save_calibration(cluster_result, db)

        # Reload weights
        conn = sqlite3.connect("data/db/market.db")
        calib_rows = conn.execute(
            """SELECT model, group_name, signal_name, weight
               FROM calibrations
               WHERE calibration_date = (SELECT MAX(calibration_date) FROM calibrations)"""
        ).fetchall()
        conn.close()
        calib_weights = {"gics": {}, "cluster": {}}
        for row in calib_rows:
            model, group, signal, weight = row
            calib_weights.setdefault(model, {}).setdefault(group, {})[signal] = weight
        has_calibration = True

        print_calibration_report(gics_result, cluster_result)
        log.info("  Calibration complete (%.1fs)", time.time() - t0)

    if not has_calibration:
        log.error("  No calibration found! Run: python run_daily.py --calibrate")
        return

    # ── Step 4: Build dual rankings ──────────────────────────────
    log.info("[4/7] Building dual rankings (GICS + Clusters)...")
    today_date = prices.index[-1]

    # Get cluster assignments
    conn = sqlite3.connect("data/db/market.db")
    cluster_assign = {r[0]: r[1] for r in conn.execute(
        """SELECT ticker, cluster_id FROM cluster_assignments
           WHERE assignment_date = (SELECT MAX(assignment_date) FROM cluster_assignments)"""
    ).fetchall()}
    conn.close()

    def score_ticker(ticker, weights_by_group, group_map):
        """Score a single ticker using group-specific weights."""
        group = group_map.get(ticker)
        if group is None or group not in weights_by_group:
            return np.nan
        weights = weights_by_group[group]
        score = 0.0
        n = 0
        for sig_name, w in weights.items():
            if sig_name in signal_dfs and today_date in signal_dfs[sig_name].index:
                val_s = signal_dfs[sig_name].loc[today_date].get(ticker)
                if val_s is not None and np.isfinite(val_s):
                    score += val_s * w
                    n += 1
        return score if n >= 5 else np.nan

    # GICS ranking
    gics_scores = {}
    for t in tickers_with_data:
        sector = sectors.get(t, "Unknown")
        gics_scores[t] = score_ticker(t, calib_weights["gics"], {t: sector for t in tickers_with_data})

    # Cluster ranking
    cluster_scores = {}
    for t in tickers_with_data:
        cluster_scores[t] = score_ticker(t, calib_weights["cluster"], cluster_assign)

    # Build DataFrames
    def build_ranking_df(scores, model_name):
        df = pd.DataFrame([
            {"ticker": t, "composite_score": s}
            for t, s in scores.items() if np.isfinite(s)
        ]).sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        df["model"] = model_name
        df["ret_1m"] = df["ticker"].map(
            lambda t: f"{ret_1m.get(t, np.nan):+.1%}" if np.isfinite(ret_1m.get(t, np.nan)) else "N/A")
        df["ret_3m"] = df["ticker"].map(
            lambda t: f"{ret_3m.get(t, np.nan):+.1%}" if np.isfinite(ret_3m.get(t, np.nan)) else "N/A")
        df["company"] = df["ticker"].map(lambda t: profiles.get(t, {}).get("name", ""))
        df["sector"] = df["ticker"].map(lambda t: profiles.get(t, {}).get("sector", ""))
        return df

    ranking_gics = build_ranking_df(gics_scores, "gics")
    ranking_cluster = build_ranking_df(cluster_scores, "cluster")

    # Primary ranking = GICS (stable, production), shadow = Clusters
    ranking = ranking_gics  # primary for alerts and report

    log.info("  GICS ranking:    %d tickers | Top: %s (%+.4f)",
             len(ranking_gics), ranking_gics.iloc[0]["ticker"], ranking_gics.iloc[0]["composite_score"])
    log.info("  Cluster ranking: %d tickers | Top: %s (%+.4f)",
             len(ranking_cluster), ranking_cluster.iloc[0]["ticker"], ranking_cluster.iloc[0]["composite_score"])

    # Save both rankings to DB
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn = sqlite3.connect("data/db/market.db")
    for df in [ranking_gics, ranking_cluster]:
        model = df.iloc[0]["model"]
        for _, row in df.iterrows():
            group = sectors.get(row["ticker"], "Unknown") if model == "gics" else cluster_assign.get(row["ticker"], "C0")
            conn.execute(
                """INSERT OR REPLACE INTO dual_rankings
                   (ranking_date, model, ticker, rank_position, composite_score, group_name, group_weights, computed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (today, model, row["ticker"], int(row["rank"]), row["composite_score"],
                 group, json.dumps(calib_weights.get(model, {}).get(group, {})), now),
            )
    conn.commit()

    # ── Step 4b: Forward test tracking ───────────────────────────
    # Track daily returns of top quintile for both models
    if len(returns) > 1:
        today_ret = returns.iloc[-1]
        for model_name, rank_df in [("gics", ranking_gics), ("cluster", ranking_cluster)]:
            n_top = max(1, int(len(rank_df) * 0.20))
            top_tickers_model = rank_df.head(n_top)["ticker"].tolist()
            bottom_tickers_model = rank_df.tail(n_top)["ticker"].tolist()

            top_ret = today_ret.reindex(top_tickers_model).mean()
            bottom_ret = today_ret.reindex(bottom_tickers_model).mean()
            spy_ret = today_ret.get("SPY", 0)

            # Get cumulative alpha
            prev = conn.execute(
                "SELECT cumulative_alpha FROM forward_test WHERE model=? ORDER BY test_date DESC LIMIT 1",
                (model_name,)).fetchone()
            prev_alpha = prev[0] if prev else 0.0
            daily_alpha = (top_ret - spy_ret) if np.isfinite(top_ret) and np.isfinite(spy_ret) else 0.0
            cum_alpha = prev_alpha + daily_alpha

            conn.execute(
                """INSERT OR REPLACE INTO forward_test
                   (test_date, model, top_quintile_return, bottom_quintile_return,
                    spread, spy_return, cumulative_alpha, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (today, model_name, float(top_ret) if np.isfinite(top_ret) else None,
                 float(bottom_ret) if np.isfinite(bottom_ret) else None,
                 float(top_ret - bottom_ret) if np.isfinite(top_ret) and np.isfinite(bottom_ret) else None,
                 float(spy_ret) if np.isfinite(spy_ret) else None,
                 cum_alpha, now),
            )
        conn.commit()
        log.info("  Forward test tracked for both models")

    conn.close()

    # ── Step 5: Alerts ───────────────────────────────────────────
    log.info("[5/7] Running alerts on watchlist...")
    t0 = time.time()

    # Load watchlist from config
    import yaml
    with open("config/signals.yaml") as f:
        cfg = yaml.safe_load(f)
    watchlist_tickers = cfg.get("watchlist", {}).get("active", [])

    alert_sys = AlertSystem("config/signals.yaml")
    alert_report = alert_sys.run(db, ranking, ret_1m, force_refresh=args.refresh_alerts)
    log.info("  Alerts: %d HIGH, %d MEDIUM, %d LOW (%.1fs)",
             len(alert_report.high_alerts), len(alert_report.medium_alerts),
             len([a for a in alert_report.alerts if a.severity == "LOW"]),
             time.time() - t0)

    if alert_report.high_alerts or alert_report.medium_alerts:
        log.info("")
        for a in alert_report.high_alerts + alert_report.medium_alerts:
            icon = "🔴" if a.severity == "HIGH" else "🟡"
            log.info("  %s %s %s — %s", icon, a.ticker, a.action, a.headline[:65])
        log.info("")

    # ── Step 6: Compare models ───────────────────────────────────
    log.info("[6/7] Model comparison...")

    # Show where GICS and Cluster rankings agree/disagree
    top20_gics = set(ranking_gics.head(int(len(ranking_gics) * 0.20))["ticker"])
    top20_cluster = set(ranking_cluster.head(int(len(ranking_cluster) * 0.20))["ticker"])
    overlap = top20_gics & top20_cluster
    only_gics = top20_gics - top20_cluster
    only_cluster = top20_cluster - top20_gics

    log.info("  Top quintile overlap: %d/%d (%.0f%%)",
             len(overlap), len(top20_gics), len(overlap) / max(len(top20_gics), 1) * 100)
    if only_gics:
        sample = sorted(only_gics)[:5]
        log.info("  Only in GICS top:    %s...", sample)
    if only_cluster:
        sample = sorted(only_cluster)[:5]
        log.info("  Only in Cluster top: %s...", sample)

    # Forward test summary
    conn = sqlite3.connect("data/db/market.db")
    for model_name in ["gics", "cluster"]:
        fwd = conn.execute(
            "SELECT COUNT(*), cumulative_alpha FROM forward_test WHERE model=? ORDER BY test_date DESC LIMIT 1",
            (model_name,)).fetchone()
        if fwd and fwd[0] > 0:
            log.info("  Forward test %s: %d days tracked, cumulative alpha: %+.2f%%",
                     model_name, fwd[0], (fwd[1] or 0) * 100)
    conn.close()

    # ── Step 7: PDF Report ───────────────────────────────────────
    if not args.no_report:
        log.info("[7/7] Generating PDF report...")

        total = len(ranking)
        watchlist_status = []
        for t in watchlist_tickers:
            # Use GICS ranking as primary
            m = ranking_gics[ranking_gics["ticker"] == t]
            mc = ranking_cluster[ranking_cluster["ticker"] == t]
            if len(m) > 0:
                r = m.iloc[0]
                rk = int(r["rank"])
                pct = rk / total
                zone = "TOP20" if pct <= 0.20 else "TOP40" if pct <= 0.40 else "MID" if pct <= 0.60 else "BTM"
                r1 = ret_1m.get(t, np.nan)
                am = [a for a in alert_report.alerts if a.ticker == t]

                # Cluster rank for comparison
                crk = int(mc.iloc[0]["rank"]) if len(mc) > 0 else "?"

                watchlist_status.append({
                    "zone": zone, "ticker": t,
                    "rank_gics": f"{rk}/{total}",
                    "rank_cluster": f"{crk}/{total}" if isinstance(crk, int) else "?",
                    "score": r["composite_score"],
                    "ret_1m": f"{r1:+.1%}" if np.isfinite(r1) else "N/A",
                    "notes": am[0].headline[:48] if am else "",
                })

        # Discoveries: in top 20 of BOTH models but not in watchlist
        active_set = set(watchlist_tickers)
        both_top = top20_gics & top20_cluster
        discoveries = []
        for _, row in ranking_gics.iterrows():
            if row["ticker"] in both_top and row["ticker"] not in active_set and row["ticker"] != "SPY":
                mc = ranking_cluster[ranking_cluster["ticker"] == row["ticker"]]
                crk = int(mc.iloc[0]["rank"]) if len(mc) > 0 else "?"
                discoveries.append({
                    "rank_gics": int(row["rank"]),
                    "rank_cluster": crk,
                    "ticker": row["ticker"],
                    "score": row["composite_score"],
                    "ret_1m": row["ret_1m"],
                    "company": row["company"],
                    "sector": row["sector"],
                })
            if len(discoveries) >= 15:
                break

        # Get calibration weights for report
        conn = sqlite3.connect("data/db/market.db")
        calib_data = conn.execute(
            """SELECT model, group_name, signal_name, ic_value, weight
               FROM calibrations
               WHERE calibration_date = (SELECT MAX(calibration_date) FROM calibrations)
               ORDER BY model, group_name, weight DESC"""
        ).fetchall()
        conn.close()

        # Build calibration summary for report
        calib_summary = {"gics": {}, "cluster": {}}
        for model, group, signal, ic, weight in calib_data:
            calib_summary.setdefault(model, {}).setdefault(group, []).append({
                "signal": signal, "ic": ic, "weight": weight,
            })

        # Cluster sizes and sample tickers
        conn2 = sqlite3.connect("data/db/market.db")
        cluster_info = {}
        for cid, ticker in conn2.execute(
            """SELECT cluster_id, ticker FROM cluster_assignments
               WHERE assignment_date = (SELECT MAX(assignment_date) FROM cluster_assignments)
               ORDER BY cluster_id, ticker"""
        ).fetchall():
            cluster_info.setdefault(cid, []).append(ticker)
        conn2.close()

        report_filename = f"reports/MM_Quant_Capital_{today}.pdf"
        pdf_path = generate_report(
            ranking_df=ranking_gics,
            ranking_cluster_df=ranking_cluster,
            alerts=alert_report.alerts,
            watchlist_status=watchlist_status,
            discoveries=discoveries,
            signal_weights=calib_weights.get("gics", {}).get(
                list(calib_weights.get("gics", {}).keys())[0] if calib_weights.get("gics") else "", {}),
            data_quality={},
            output_path=report_filename,
            calibration_summary=calib_summary,
            cluster_info=cluster_info,
            cluster_assignments=cluster_assign,
            profiles=profiles,
        )
        log.info("  Report: %s (%.1f KB)", pdf_path, os.path.getsize(pdf_path) / 1024)
    else:
        log.info("[7/7] Skipped (--no-report)")

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - t_start
    log.info("")
    log.info("=" * 65)
    log.info("  PIPELINE COMPLETE")
    log.info("  Time: %.1fs | Tickers: %d | Models: GICS + Cluster", elapsed, len(ranking))
    log.info("  GICS top: %s | Cluster top: %s | Overlap: %.0f%%",
             ranking_gics.iloc[0]["ticker"], ranking_cluster.iloc[0]["ticker"],
             len(overlap) / max(len(top20_gics), 1) * 100)
    log.info("=" * 65)

    db.log_run(
        run_type="calibrate" if args.calibrate else "weekly_full" if args.weekly else "daily",
        universe_name="sp500+nasdaq",
        tickers_count=len(ranking),
        signals=list(signal_dfs.keys()),
        duration=elapsed,
        status="success",
    )


if __name__ == "__main__":
    main()
