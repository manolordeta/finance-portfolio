#!/usr/bin/env python3
"""
MM Quant Capital — Backtest: Signal Decay Exit

Tests whether selling positions when their composite score drops
significantly BETWEEN monthly rebalances improves performance.

Hypothesis: if the reason you bought (high composite score) degrades
before the next rebalance, exiting early avoids losses.

Compares:
  A) Monthly rebalance only (baseline — what works now)
  B) Monthly rebalance + weekly signal check
     → sell if composite score dropped >X% from entry score
  C) Monthly rebalance + weekly signal check
     → sell if ticker dropped out of top quintile

Usage:
  python backtests/03_signal_decay_exit.py
  python backtests/03_signal_decay_exit.py --decay-thresholds 30,50,70
  python backtests/03_signal_decay_exit.py --check-freq 5  # check every 5 days
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Signal Decay Exit Backtest")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--decay-thresholds", default="30,50,70",
                        help="Comma-separated decay %% thresholds to test")
    parser.add_argument("--check-freq", type=int, default=5,
                        help="Check signal decay every N trading days")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--rebal-days", type=int, default=21)
    parser.add_argument("--top-pct", type=float, default=0.20)
    args = parser.parse_args()

    decay_thresholds = [int(x) for x in args.decay_thresholds.split(",")]
    train_days = args.train_months * 21

    t_start = time.time()
    print("=" * 85)
    print("  MM QUANT CAPITAL — Signal Decay Exit Backtest")
    print(f"  Decay thresholds: {decay_thresholds}%")
    print(f"  Check frequency: every {args.check_freq} days")
    print(f"  Rebalance: every {args.rebal_days} days")
    print("=" * 85)

    # ── Load data ─────────────────────────────────────────────
    import sqlite3
    import yfinance as yf
    from src.data.database import MarketDB
    from src.signals import technical, fundamental, valuation

    db = MarketDB("data/db/market.db")
    conn = sqlite3.connect("data/db/market.db")
    all_tickers = [r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM profiles").fetchall()]
    sectors_map = {r[0]: r[1] for r in conn.execute(
        "SELECT ticker, sector FROM profiles").fetchall()}
    conn.close()

    print("\n[1/3] Loading data...")
    data = yf.download(all_tickers + ["SPY"], start=args.start, progress=False)
    prices = data["Close"].dropna(axis=1, thresh=int(len(data) * 0.5))
    volumes = data["Volume"].reindex(columns=prices.columns)
    returns = prices.pct_change()
    tickers = [t for t in all_tickers if t in prices.columns and t != "SPY"]
    N = len(tickers)
    print(f"  {len(prices)} days, {N} tickers")

    # ── Signals ───────────────────────────────────────────────
    print("[2/3] Computing signals...")
    tech = technical.compute_all(prices, volumes)
    fund = fundamental.compute_all(db, tickers)
    val = valuation.compute_all(prices)

    sig_names = []
    sig_list = []
    for name, df in tech.items():
        sig_names.append(f"T_{name}")
        sig_list.append(df.reindex(columns=tickers).values)
    for name, df in val.items():
        sig_names.append(f"V_{name}")
        sig_list.append(df.reindex(columns=tickers).values)
    for name, scores in fund.items():
        sig_names.append(f"F_{name}")
        s = pd.Series(scores).reindex(tickers).values
        sig_list.append(np.tile(s, (len(prices), 1)))

    S = np.stack(sig_list, axis=2)  # (T, N, K)
    K = len(sig_names)
    fwd_21 = prices[tickers].pct_change(21).shift(-21).values
    spy_daily = returns["SPY"].values
    ret_daily = returns[tickers].values

    # GICS weights (same as main backtest)
    ticker_sectors = [sectors_map.get(t, "Unknown") for t in tickers]
    sector_idx = {}
    for i, s in enumerate(ticker_sectors):
        sector_idx.setdefault(s, []).append(i)

    print(f"  {K} signals")

    def compute_gics_weights(train_sl):
        S_tr = S[train_sl]
        F_tr = fwd_21[train_sl]
        global_ic = np.zeros(K)
        for k in range(K):
            ics = []
            for t in range(0, len(S_tr), 5):
                sv = S_tr[t, :, k]; fv = F_tr[t, :]
                mask = np.isfinite(sv) & np.isfinite(fv)
                if mask.sum() >= 30:
                    ic, _ = spearmanr(sv[mask], fv[mask])
                    if np.isfinite(ic): ics.append(ic)
            global_ic[k] = np.mean(ics) if ics else 0
        global_ic = np.maximum(global_ic, 0)
        g_total = global_ic.sum()
        global_w = global_ic / g_total if g_total > 0 else np.ones(K) / K

        group_weights = {}
        for gname, gidx in sector_idx.items():
            if len(gidx) < 10:
                group_weights[gname] = global_w.copy(); continue
            group_ic = np.zeros(K)
            for k in range(K):
                ics = []
                for t in range(0, len(S_tr), 5):
                    sv = S_tr[t, gidx, k]; fv = F_tr[t, gidx]
                    mask = np.isfinite(sv) & np.isfinite(fv)
                    if mask.sum() >= 8:
                        ic, _ = spearmanr(sv[mask], fv[mask])
                        if np.isfinite(ic): ics.append(ic)
                group_ic[k] = np.mean(ics) if ics else 0
            group_ic = np.maximum(group_ic, 0)
            g_t = group_ic.sum()
            group_w = group_ic / g_t if g_t > 0 else np.ones(K) / K
            blended = 0.5 * global_w + 0.5 * group_w
            b_t = blended.sum()
            group_weights[gname] = blended / b_t if b_t > 0 else np.ones(K) / K
        return group_weights

    def score_all(day_idx, gics_weights):
        """Score all tickers for a given day."""
        day_signals = S[day_idx]
        scores = np.full(N, np.nan)
        for j in range(N):
            sector = ticker_sectors[j]
            if sector not in gics_weights: continue
            w = gics_weights[sector]
            vals = day_signals[j]
            mask = np.isfinite(vals)
            if mask.sum() >= 5:
                scores[j] = np.dot(vals[mask], w[mask])
        return scores

    # ── Walk-forward ──────────────────────────────────────────
    print("[3/3] Walk-forward backtest...")

    # Models: baseline + each decay threshold + quintile exit
    model_names = ["baseline"] + [f"decay_{d}pct" for d in decay_thresholds] + ["quintile_exit"]
    results = {m: [] for m in model_names + ["spy"]}
    trade_stats = {m: {"total": 0, "early_exits": 0, "early_exit_returns": []}
                   for m in model_names}

    test_days = 63  # 3 months
    i = train_days
    period = 0

    while i + args.rebal_days <= len(prices):
        period += 1
        train_sl = slice(max(0, i - train_days), i)
        test_end = min(i + test_days, len(prices))

        gics_weights = compute_gics_weights(train_sl)

        # Monthly rebalance points
        rebal_points = list(range(i, test_end, args.rebal_days))

        for rb_idx, rb in enumerate(rebal_points):
            rb_end = min(rb + args.rebal_days, test_end)

            # Score and select top quintile at rebalance
            scores_at_entry = score_all(rb, gics_weights)
            valid = np.where(np.isfinite(scores_at_entry))[0]
            if len(valid) < 20:
                for day in range(rb, rb_end):
                    if day >= len(spy_daily): break
                    for m in model_names: results[m].append(0)
                    results["spy"].append(spy_daily[day] if np.isfinite(spy_daily[day]) else 0)
                continue

            sorted_valid = valid[np.argsort(-scores_at_entry[valid])]
            n_top = max(1, int(len(sorted_valid) * args.top_pct))
            top_idx = sorted_valid[:n_top].tolist()
            quintile_threshold = scores_at_entry[sorted_valid[n_top - 1]]

            # Track active positions per model
            # Baseline: hold all, no changes
            # Decay models: check periodically and remove decayed ones
            active = {m: set(top_idx) for m in model_names}
            entry_scores = {j: scores_at_entry[j] for j in top_idx}

            for day in range(rb, rb_end):
                if day >= len(ret_daily) or day >= len(spy_daily):
                    break

                # Check for signal decay every check_freq days
                days_since_rebal = day - rb
                if days_since_rebal > 0 and days_since_rebal % args.check_freq == 0:
                    current_scores = score_all(day, gics_weights)

                    for m in model_names:
                        if m == "baseline":
                            continue  # baseline never exits early

                        to_remove = []
                        for j in list(active[m]):
                            if not np.isfinite(current_scores[j]):
                                continue

                            if m == "quintile_exit":
                                # Exit if dropped out of top quintile entirely
                                # Recompute quintile with current scores
                                valid_now = np.where(np.isfinite(current_scores))[0]
                                if len(valid_now) > 0:
                                    threshold_now = np.percentile(
                                        current_scores[valid_now],
                                        100 * (1 - args.top_pct))
                                    if current_scores[j] < threshold_now:
                                        to_remove.append(j)
                            else:
                                # Decay threshold check
                                decay_pct = int(m.split("_")[1].replace("pct", ""))
                                entry_score = entry_scores.get(j, 0)
                                if entry_score > 0:
                                    decay = (entry_score - current_scores[j]) / entry_score
                                    if decay > decay_pct / 100:
                                        to_remove.append(j)

                        for j in to_remove:
                            active[m].discard(j)
                            trade_stats[m]["early_exits"] += 1
                            # Track return at exit
                            if j < len(ret_daily[0]):
                                cum_ret = 1.0
                                for d in range(rb, day):
                                    if d < len(ret_daily):
                                        r = ret_daily[d, j]
                                        if np.isfinite(r):
                                            cum_ret *= (1 + r)
                                trade_stats[m]["early_exit_returns"].append(cum_ret - 1)

                # Compute daily returns for each model
                spy_r = spy_daily[day] if np.isfinite(spy_daily[day]) else 0

                for m in model_names:
                    port = list(active[m])
                    trade_stats[m]["total"] += 1
                    if not port:
                        results[m].append(0)
                        continue
                    day_rets = ret_daily[day, port]
                    valid_rets = day_rets[np.isfinite(day_rets)]
                    results[m].append(float(np.mean(valid_rets)) if len(valid_rets) > 0 else 0)

                results["spy"].append(spy_r)

        if period % 4 == 0:
            print(f"  Period {period}...")

        i += test_days

    # ═══════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    n_days = len(results["spy"])
    years = n_days / 252

    spy_cumul = (1 + pd.Series(results["spy"])).cumprod()
    spy_total = spy_cumul.iloc[-1] - 1

    print(f"\n{'=' * 90}")
    print(f"  SIGNAL DECAY EXIT — RESULTS")
    print(f"  {n_days} days, {years:.1f} years | Check every {args.check_freq} days")
    print(f"{'=' * 90}\n")

    print(f"  {'Model':25s} {'Total':>8s} {'Ann':>8s} {'Sharpe':>8s} {'MaxDD':>8s} "
          f"{'Alpha':>8s} {'EarlyX':>7s} {'AvgExRet':>9s}")
    print(f"  {'-' * 82}")

    for model in model_names + ["spy"]:
        rets = pd.Series(results[model])
        eq = (1 + rets).cumprod()
        total = eq.iloc[-1] - 1
        ann = eq.iloc[-1] ** (1 / max(years, 0.1)) - 1
        rf_daily = 0.045 / 252
        sharpe = ((rets.mean() - rf_daily) / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        dd = ((eq - eq.cummax()) / eq.cummax()).min()
        alpha = total - spy_total if model != "spy" else 0

        stats = trade_stats.get(model, {})
        early = stats.get("early_exits", 0)
        early_rets = stats.get("early_exit_returns", [])
        avg_exit_ret = f"{np.mean(early_rets):+.2%}" if early_rets else "N/A"

        labels = {
            "baseline": "Monthly Rebal (baseline)",
            "quintile_exit": "Quintile Exit (weekly)",
            "spy": "SPY Buy & Hold",
        }
        for d in decay_thresholds:
            labels[f"decay_{d}pct"] = f"Decay >{d}% Exit"

        print(f"  {labels.get(model, model):25s} {total*100:+7.1f}% {ann*100:+7.1f}% "
              f"{sharpe:+7.2f} {dd*100:+7.1f}% {alpha*100:+7.1f}% "
              f"{early:>7d} {avg_exit_ret:>9s}")

    # Year by year
    all_dates = prices.index[train_days:train_days + n_days]
    print(f"\n  Year-by-Year Alpha vs SPY:")
    header = f"  {'':>6s}"
    for m in model_names:
        short = m.replace("baseline", "Base").replace("decay_", "D").replace("pct", "%").replace("quintile_exit", "QExit")
        header += f" {short:>10s}"
    print(header)
    print(f"  {'-' * (6 + 11 * len(model_names))}")

    for year in sorted(set(all_dates.year)):
        mask = (all_dates.year == year)[:n_days]
        if mask.sum() < 20:
            continue
        spy_yr = (1 + pd.Series(results["spy"])[mask]).prod() - 1
        row = f"  {year:6d}"
        for m in model_names:
            yr_ret = (1 + pd.Series(results[m])[mask]).prod() - 1
            row += f" {(yr_ret - spy_yr)*100:+9.1f}%"
        print(row)

    # Analysis of early exits
    print(f"\n{'=' * 90}")
    print(f"  EARLY EXIT ANALYSIS")
    print(f"{'=' * 90}\n")

    for m in model_names:
        if m == "baseline":
            continue
        stats = trade_stats[m]
        er = stats["early_exit_returns"]
        if not er:
            continue
        er = np.array(er)
        wins = (er > 0).sum()
        losses = (er <= 0).sum()
        print(f"  {labels.get(m, m):25s}")
        print(f"    Early exits:     {stats['early_exits']}")
        print(f"    Avg return at exit: {np.mean(er):+.2%}")
        print(f"    Win/Loss:        {wins}/{losses} ({wins/(wins+losses)*100:.0f}% profitable at exit)")
        print(f"    Avoided losses:  {(er < -0.02).sum()} positions exited before >2% loss")
        print(f"    Missed gains:    {(er > 0.02).sum()} positions exited while still gaining >2%")
        print()

    print(f"  Completed in {elapsed:.0f}s")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
