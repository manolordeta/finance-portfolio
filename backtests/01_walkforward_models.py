#!/usr/bin/env python3
"""
MM Quant Capital — Walk-Forward Backtest

Compares 4 models:
  A) Equal Weight:    all signals weighted equally
  B) GICS Calibrated: IC-weighted per GICS sector (regularized 50/50 with global)
  C) Cluster Calibr.: IC-weighted per correlation cluster (regularized)
  D) Momentum Only:   momentum_12_1 signal alone (baseline)

vs SPY buy & hold.

Walk-forward: 12-month train → 3-month test, rolling.
Monthly rebalance within each test period.
Top quintile long-only, equal-weight positions.

Usage:
  python run_backtest.py
  python run_backtest.py --start 2022-01-01 --clusters 8 --train-months 12 --test-months 3
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
from sklearn.cluster import SpectralClustering

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Backtest")
    parser.add_argument("--start", default="2022-01-01", help="Start date for data")
    parser.add_argument("--clusters", type=int, default=8, help="Number of clusters")
    parser.add_argument("--train-months", type=int, default=12, help="Training window months")
    parser.add_argument("--test-months", type=int, default=3, help="Test window months")
    parser.add_argument("--rebal-days", type=int, default=21, help="Rebalance frequency in trading days")
    parser.add_argument("--top-pct", type=float, default=0.20, help="Top percentile to buy (0.20 = quintile)")
    parser.add_argument("--reg-alpha", type=float, default=0.5, help="Regularization: 0=pure group, 1=pure global")
    args = parser.parse_args()

    t_start = time.time()
    train_days = args.train_months * 21
    test_days = args.test_months * 21

    print("=" * 80)
    print("  MM QUANT CAPITAL — Walk-Forward Backtest")
    print(f"  Train: {args.train_months}m | Test: {args.test_months}m | Rebal: {args.rebal_days}d")
    print(f"  Top: {args.top_pct:.0%} | Clusters: {args.clusters} | Reg α: {args.reg_alpha}")
    print("=" * 80)

    # ── Load data ─────────────────────────────────────────────────
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

    print("\n[1/4] Downloading prices...")
    t0 = time.time()
    data = yf.download(all_tickers + ["SPY"], start=args.start, progress=False)
    prices = data["Close"].dropna(axis=1, thresh=int(len(data) * 0.5))
    volumes = data["Volume"].reindex(columns=prices.columns)
    returns = prices.pct_change()
    tickers = [t for t in all_tickers if t in prices.columns and t != "SPY"]
    N = len(tickers)
    print(f"  {len(prices)} days, {N} tickers ({time.time()-t0:.0f}s)")

    # ── Compute signals ───────────────────────────────────────────
    print("[2/4] Computing signals...")
    t0 = time.time()
    tech = technical.compute_all(prices, volumes)
    fund = fundamental.compute_all(db, tickers)
    val = valuation.compute_all(prices)

    # Build 3D array: (dates, tickers, signals)
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
    print(f"  {K} signals, shape {S.shape} ({time.time()-t0:.0f}s)")

    # ── Setup groups ──────────────────────────────────────────────
    print("[3/4] Setting up groups...")

    # GICS sectors
    ticker_sectors = [sectors_map.get(t, "Unknown") for t in tickers]
    unique_sectors = sorted(set(ticker_sectors))
    sector_idx = {s: [i for i, ts in enumerate(ticker_sectors) if ts == s]
                  for s in unique_sectors}

    # Clusters on first train window correlation
    corr_data = returns[tickers].iloc[:train_days].dropna()
    corr_mat = np.corrcoef(corr_data.T)
    corr_mat = np.nan_to_num(corr_mat, nan=0)
    affinity = (corr_mat + 1) / 2
    np.fill_diagonal(affinity, 1)
    sc = SpectralClustering(n_clusters=args.clusters, affinity="precomputed",
                            random_state=42, n_init=5)
    cluster_labels = sc.fit_predict(affinity)
    cluster_idx = {}
    for i, c in enumerate(cluster_labels):
        cluster_idx.setdefault(c, []).append(i)

    cluster_sizes = [len(v) for v in cluster_idx.values()]
    print(f"  GICS: {len(unique_sectors)} sectors")
    print(f"  Clusters: {len(cluster_idx)} groups, sizes: {cluster_sizes}")

    # ── Walk-forward ──────────────────────────────────────────────
    print("[4/4] Walk-forward backtest...")

    spy_daily = returns["SPY"].values
    ret_daily = returns[tickers].values  # (T, N)

    # Forward returns for IC calculation
    fwd_21 = prices[tickers].pct_change(21).shift(-21).values  # (T, N)

    def compute_ic_per_group(group_indices, train_slice, sample_step=5):
        """Compute IC per signal per group on train data. Returns {group: {sig_idx: weight}}."""
        S_tr = S[train_slice]
        F_tr = fwd_21[train_slice]

        # Global IC first
        global_ic = np.zeros(K)
        for k in range(K):
            ics = []
            for t in range(0, len(S_tr), sample_step):
                sv = S_tr[t, :, k]
                fv = F_tr[t, :]
                mask = np.isfinite(sv) & np.isfinite(fv)
                if mask.sum() >= 30:
                    ic, _ = spearmanr(sv[mask], fv[mask])
                    if np.isfinite(ic):
                        ics.append(ic)
            global_ic[k] = np.mean(ics) if ics else 0

        # Clamp negative ICs to 0 (don't short signals)
        global_ic = np.maximum(global_ic, 0)
        g_total = global_ic.sum()
        global_w = global_ic / g_total if g_total > 0 else np.ones(K) / K

        # Per-group IC
        group_weights = {}
        for gname, gidx in group_indices.items():
            if len(gidx) < 10:
                # Too few tickers, use global
                group_weights[gname] = global_w.copy()
                continue

            group_ic = np.zeros(K)
            for k in range(K):
                ics = []
                for t in range(0, len(S_tr), sample_step):
                    sv = S_tr[t, gidx, k]
                    fv = F_tr[t, gidx]
                    mask = np.isfinite(sv) & np.isfinite(fv)
                    if mask.sum() >= 8:
                        ic, _ = spearmanr(sv[mask], fv[mask])
                        if np.isfinite(ic):
                            ics.append(ic)
                group_ic[k] = np.mean(ics) if ics else 0

            # Clamp negative to 0
            group_ic = np.maximum(group_ic, 0)
            g_total = group_ic.sum()
            group_w = group_ic / g_total if g_total > 0 else np.ones(K) / K

            # Regularize: alpha * global + (1-alpha) * group
            blended = args.reg_alpha * global_w + (1 - args.reg_alpha) * group_w
            b_total = blended.sum()
            group_weights[gname] = blended / b_total if b_total > 0 else np.ones(K) / K

        return group_weights, global_w

    def score_day(day_idx, weights_by_group, ticker_to_group):
        """Score all tickers for one day using group-specific weights."""
        day_signals = S[day_idx]  # (N, K)
        scores = np.full(N, np.nan)

        for j in range(N):
            group = ticker_to_group[j]
            if group not in weights_by_group:
                continue
            w = weights_by_group[group]
            vals = day_signals[j]
            mask = np.isfinite(vals)
            if mask.sum() >= 5:
                scores[j] = np.dot(vals[mask], w[mask])

        return scores

    # Group mappings for each model
    ticker_to_sector = {j: ticker_sectors[j] for j in range(N)}
    ticker_to_cluster = {j: cluster_labels[j] for j in range(N)}

    # Results storage
    models = ["equal", "gics", "cluster", "momentum_only", "spy"]
    period_results = {m: [] for m in models}
    period_info = []

    # Momentum signal index
    mom_idx = sig_names.index("T_momentum_12_1")

    i = train_days
    period_num = 0

    while i + args.rebal_days <= len(prices):
        period_num += 1
        train_sl = slice(max(0, i - train_days), i)
        test_end = min(i + test_days, len(prices))

        period_label = f"{prices.index[i].strftime('%Y-%m')} to {prices.index[min(test_end-1, len(prices)-1)].strftime('%Y-%m')}"

        # ── Train: compute weights ──
        gics_weights, global_w = compute_ic_per_group(sector_idx, train_sl)
        cluster_weights, _ = compute_ic_per_group(cluster_idx, train_sl)

        # ── Test: monthly rebalance within test period ──
        rebal_idx = list(range(i, test_end, args.rebal_days))

        for rb_start in rebal_idx:
            rb_end = min(rb_start + args.rebal_days, test_end)

            # Score at rebalance date
            eq_scores = np.nanmean(S[rb_start], axis=1)  # equal weight
            gics_scores = score_day(rb_start, gics_weights, ticker_to_sector)
            clust_scores = score_day(rb_start, cluster_weights, ticker_to_cluster)
            mom_scores = S[rb_start, :, mom_idx]

            # Select top quintile for each model
            portfolios = {}
            for model_name, sc in [("equal", eq_scores), ("gics", gics_scores),
                                    ("cluster", clust_scores), ("momentum_only", mom_scores)]:
                valid = np.where(np.isfinite(sc))[0]
                if len(valid) < 20:
                    portfolios[model_name] = []
                    continue
                sorted_valid = valid[np.argsort(-sc[valid])]
                n_top = max(1, int(len(sorted_valid) * args.top_pct))
                portfolios[model_name] = sorted_valid[:n_top].tolist()

            # Compute daily returns for the rebalance period
            for day in range(rb_start, rb_end):
                if day >= len(ret_daily) or day >= len(spy_daily):
                    break

                spy_r = spy_daily[day] if np.isfinite(spy_daily[day]) else 0.0

                for model_name in ["equal", "gics", "cluster", "momentum_only"]:
                    port = portfolios[model_name]
                    if not port:
                        period_results[model_name].append(0.0)
                        continue
                    day_rets = ret_daily[day, port]
                    valid_rets = day_rets[np.isfinite(day_rets)]
                    if len(valid_rets) > 0:
                        period_results[model_name].append(float(np.mean(valid_rets)))
                    else:
                        period_results[model_name].append(0.0)

                period_results["spy"].append(spy_r)

        period_info.append(period_label)
        print(f"  Period {period_num}: {period_label}")

        i += test_days

    # ═══════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start

    print(f"\n{'=' * 80}")
    print(f"  WALK-FORWARD BACKTEST RESULTS")
    print(f"  {prices.index[train_days].strftime('%Y-%m')} to {prices.index[-1].strftime('%Y-%m')}")
    print(f"  {period_num} periods | {len(period_results['spy'])} trading days | {elapsed:.0f}s")
    print(f"{'=' * 80}\n")

    spy_cumul = (1 + pd.Series(period_results["spy"])).cumprod()
    spy_total = spy_cumul.iloc[-1] - 1
    n_days = len(period_results["spy"])
    years = n_days / 252

    print(f"  {'Model':20s} {'Total':>8s} {'Ann.Ret':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Alpha':>8s}")
    print(f"  {'-' * 58}")

    for model in ["equal", "gics", "cluster", "momentum_only", "spy"]:
        rets = pd.Series(period_results[model])
        eq = (1 + rets).cumprod()
        total = eq.iloc[-1] - 1
        ann = (eq.iloc[-1]) ** (1 / max(years, 0.1)) - 1
        rf_daily = 0.045 / 252
        sharpe = ((rets.mean() - rf_daily) / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        dd = ((eq - eq.cummax()) / eq.cummax()).min()
        alpha = total - spy_total if model != "spy" else 0

        label = {
            "equal": "Equal Weight",
            "gics": "GICS Calibrated",
            "cluster": "Cluster Calibr.",
            "momentum_only": "Momentum Only",
            "spy": "SPY Buy&Hold",
        }[model]

        marker = " ←" if model == "gics" else ""
        print(f"  {label:20s} {total*100:+7.1f}% {ann*100:+7.1f}% {sharpe:+7.2f} {dd*100:+7.1f}% {alpha*100:+7.1f}%{marker}")

    # Year-by-year
    all_dates = prices.index[train_days:train_days + n_days]
    print(f"\n  Year-by-Year Alpha vs SPY:")
    print(f"  {'':>6s} {'Equal':>10s} {'GICS':>10s} {'Cluster':>10s} {'Mom':>10s}")
    print(f"  {'-' * 50}")

    for year in sorted(set(all_dates.year)):
        mask = (all_dates.year == year)[:n_days]
        if mask.sum() < 20:
            continue
        spy_yr = (1 + pd.Series(period_results["spy"])[mask]).prod() - 1
        row = {}
        for model in ["equal", "gics", "cluster", "momentum_only"]:
            yr_ret = (1 + pd.Series(period_results[model])[mask]).prod() - 1
            row[model] = yr_ret - spy_yr
        print(f"  {year:6d} {row['equal']*100:+9.1f}% {row['gics']*100:+9.1f}% "
              f"{row['cluster']*100:+9.1f}% {row['momentum_only']*100:+9.1f}%")

    # Top signal weights for each GICS sector (last calibration)
    print(f"\n  Signal Weights by GICS Sector (last period):")
    print(f"  {'Sector':25s} {'Top Signal':>25s} {'Weight':>8s} {'IC':>8s}")
    print(f"  {'-' * 70}")
    for sname in sorted(sector_idx.keys()):
        if sname not in gics_weights:
            continue
        w = gics_weights[sname]
        top_k = np.argmax(w)
        print(f"  {sname:25s} {sig_names[top_k]:>25s} {w[top_k]*100:7.1f}% ")

    print(f"\n{'=' * 80}")
    print(f"  Completed in {elapsed:.0f}s")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
