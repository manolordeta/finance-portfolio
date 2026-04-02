#!/usr/bin/env python3
"""
MM Quant Capital — Look-Ahead Bias Test

Tests whether the +136%/yr is real or inflated by look-ahead in fundamentals.

Compares simplified B-L (max 10%) using:
  A) ALL signals (21): technical + fundamental + valuation  ← the +136%
  B) TECHNICAL ONLY (6): momentum, MACD, volume             ← NO look-ahead possible
  C) VALUATION ONLY (7): SMA crosses, mean reversion         ← NO look-ahead
  D) TECHNICAL + VALUATION (13): all price-based signals     ← NO look-ahead
  E) FUNDAMENTAL ONLY (8): P/E, ROE, earnings surprise       ← HAS look-ahead

If A >> D, then the extra return comes from fundamentals (likely look-ahead).
If A ≈ D, then the return is real (price-based signals drive it).

Usage:
  python backtests/05_lookahead_test.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()


def optimize_bl(scores, ret_history, top_indices, rf=0.045/252, max_weight=0.10):
    n = len(top_indices)
    if n < 5: return np.ones(n) / n
    ts = scores[top_indices] - scores[top_indices].min()
    total = ts.sum()
    if total <= 0: return np.ones(n) / n
    mu = (ts / total) * 0.15 / 252
    rr = np.nan_to_num(ret_history[:, top_indices], nan=0)
    cov = np.cov(rr.T)
    if cov.ndim < 2: return np.ones(n) / n
    cov = 0.8 * cov + 0.2 * np.diag(np.diag(cov))
    def ns(w):
        pr = w @ mu; pv = w @ cov @ w
        return -(pr - rf) / np.sqrt(pv) if pv > 0 else 1e10
    try:
        r = minimize(ns, np.ones(n)/n, method="SLSQP",
                    bounds=[(0, max_weight)]*n,
                    constraints=[{"type":"eq","fun":lambda w: np.sum(w)-1}],
                    options={"maxiter":200,"ftol":1e-8})
        if r.success and np.all(np.isfinite(r.x)):
            w = np.maximum(r.x, 0); return w / w.sum()
    except: pass
    return np.ones(n) / n


def main():
    t_start = time.time()

    print("=" * 85)
    print("  LOOK-AHEAD BIAS TEST")
    print("  Does +136%/yr come from real signals or look-ahead in fundamentals?")
    print("=" * 85)

    import sqlite3, yfinance as yf
    from src.data.database import MarketDB
    from src.signals import technical, fundamental, valuation

    db = MarketDB("data/db/market.db")
    conn = sqlite3.connect("data/db/market.db")
    all_tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM profiles").fetchall()]
    sectors_map = dict(conn.execute("SELECT ticker, sector FROM profiles").fetchall())
    conn.close()

    print("\n[1/3] Loading data...")
    data = yf.download(all_tickers + ["SPY"], start="2022-01-01", progress=False)
    prices = data["Close"].dropna(axis=1, thresh=int(len(data)*0.5))
    volumes = data["Volume"].reindex(columns=prices.columns)
    returns = prices.pct_change()
    tickers = [t for t in all_tickers if t in prices.columns and t != "SPY"]
    N = len(tickers)
    print(f"  {len(prices)} days, {N} tickers")

    print("[2/3] Computing signals by category...")
    tech = technical.compute_all(prices, volumes)
    fund = fundamental.compute_all(db, tickers)
    val = valuation.compute_all(prices)

    # Build separate signal groups
    tech_sigs = {}  # NO look-ahead
    for name, df in tech.items():
        tech_sigs[f"T_{name}"] = df.reindex(columns=tickers).values

    val_sigs = {}  # NO look-ahead
    for name, df in val.items():
        val_sigs[f"V_{name}"] = df.reindex(columns=tickers).values

    fund_sigs = {}  # HAS look-ahead
    for name, scores in fund.items():
        s = pd.Series(scores).reindex(tickers).values
        fund_sigs[f"F_{name}"] = np.tile(s, (len(prices), 1))

    # Combine into different groups
    signal_groups = {
        "A) ALL (21 signals)": {**tech_sigs, **val_sigs, **fund_sigs},
        "B) TECHNICAL only (6)": tech_sigs,
        "C) VALUATION only (7)": val_sigs,
        "D) TECH+VAL (13, no lookahead)": {**tech_sigs, **val_sigs},
        "E) FUNDAMENTAL only (8, lookahead)": fund_sigs,
    }

    spy_daily = returns["SPY"].values
    ret_daily = returns[tickers].values
    ticker_sectors = [sectors_map.get(t, "Unknown") for t in tickers]
    sector_idx = {}
    for i, s in enumerate(ticker_sectors):
        sector_idx.setdefault(s, []).append(i)

    # ── Walk-forward for each signal group ────────────────────
    print("[3/3] Walk-forward backtest per signal group...")

    train_days = 252
    test_days = 63
    results = {name: [] for name in signal_groups}
    results["SPY"] = []

    for group_name, sigs in signal_groups.items():
        sig_names_g = list(sigs.keys())
        sig_arrays = list(sigs.values())
        S_g = np.stack(sig_arrays, axis=2)
        K_g = len(sig_names_g)
        fwd_21 = prices[tickers].pct_change(21).shift(-21).values

        i = train_days
        while i + 21 <= len(prices):
            train_sl = slice(max(0, i - train_days), i)
            test_end = min(i + test_days, len(prices))

            # Calibrate GICS weights for this signal group
            S_tr = S_g[train_sl]; F_tr = fwd_21[train_sl]
            global_ic = np.zeros(K_g)
            for k in range(K_g):
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
            global_w = global_ic / g_total if g_total > 0 else np.ones(K_g) / K_g

            gics_weights = {}
            for gname, gidx in sector_idx.items():
                if len(gidx) < 10:
                    gics_weights[gname] = global_w.copy(); continue
                gic = np.zeros(K_g)
                for k in range(K_g):
                    ics = []
                    for t in range(0, len(S_tr), 5):
                        sv = S_tr[t, gidx, k]; fv = F_tr[t, gidx]
                        mask = np.isfinite(sv) & np.isfinite(fv)
                        if mask.sum() >= 8:
                            ic, _ = spearmanr(sv[mask], fv[mask])
                            if np.isfinite(ic): ics.append(ic)
                    gic[k] = np.mean(ics) if ics else 0
                gic = np.maximum(gic, 0); gt = gic.sum()
                gw = gic / gt if gt > 0 else np.ones(K_g) / K_g
                b = 0.5 * global_w + 0.5 * gw; bt = b.sum()
                gics_weights[gname] = b / bt if bt > 0 else np.ones(K_g) / K_g

            # Rebalance monthly
            for rb in range(i, test_end, 21):
                rb_end = min(rb + 21, test_end)

                scores = np.full(N, np.nan)
                for j in range(N):
                    sec = ticker_sectors[j]
                    if sec not in gics_weights: continue
                    w = gics_weights[sec]
                    vals = S_g[rb][j]
                    mask = np.isfinite(vals)
                    if mask.sum() >= max(3, K_g // 3):
                        scores[j] = np.dot(vals[mask], w[mask])

                valid = np.where(np.isfinite(scores))[0]
                if len(valid) < 20:
                    for day in range(rb, rb_end):
                        if day < len(spy_daily):
                            results[group_name].append(0)
                            if group_name == list(signal_groups.keys())[0]:
                                results["SPY"].append(spy_daily[day] if np.isfinite(spy_daily[day]) else 0)
                    continue

                sorted_valid = valid[np.argsort(-scores[valid])]
                n_top = max(1, len(sorted_valid) // 5)
                top_idx = sorted_valid[:n_top]
                train_rets = ret_daily[max(0, rb-252):rb]

                bl_w = optimize_bl(scores, train_rets, top_idx, max_weight=0.10)

                for day in range(rb, rb_end):
                    if day >= len(ret_daily) or day >= len(spy_daily): break
                    dr = np.nan_to_num(ret_daily[day, top_idx], nan=0)
                    results[group_name].append(float(np.sum(bl_w * dr)))
                    if group_name == list(signal_groups.keys())[0]:
                        results["SPY"].append(spy_daily[day] if np.isfinite(spy_daily[day]) else 0)

            i += test_days

        print(f"  {group_name}: {len(results[group_name])} days computed")

    # ═══════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════
    n_days = len(results["SPY"])
    years = n_days / 252

    spy_total = (1 + pd.Series(results["SPY"])).cumprod().iloc[-1] - 1

    print(f"\n{'=' * 85}")
    print(f"  RESULTS — {n_days} days, {years:.1f} years")
    print(f"  All models: GICS calibrated + Simplified B-L (max 10%)")
    print(f"{'=' * 85}\n")

    print(f"  {'Signal Group':40s} {'Total':>8s} {'Ann':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Alpha':>8s}")
    print(f"  {'-' * 76}")

    for name in list(signal_groups.keys()) + ["SPY"]:
        rets = pd.Series(results[name][:n_days])
        eq = (1 + rets).cumprod()
        total = eq.iloc[-1] - 1
        ann = eq.iloc[-1] ** (1/max(years,0.1)) - 1
        rf_d = 0.045 / 252
        sharpe = ((rets.mean() - rf_d) / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        dd = ((eq - eq.cummax()) / eq.cummax()).min()
        alpha = total - spy_total if name != "SPY" else 0

        marker = ""
        if "no lookahead" in name.lower(): marker = " ← CLEAN"
        if "lookahead" in name.lower() and "no" not in name.lower(): marker = " ← BIASED"

        print(f"  {name:40s} {total*100:+7.1f}% {ann*100:+7.1f}% "
              f"{sharpe:+7.2f} {dd*100:+7.1f}% {alpha*100:+7.1f}%{marker}")

    # Year by year
    all_dates = prices.index[train_days:train_days + n_days]
    print(f"\n  Year-by-Year Ann. Return:")
    header = f"  {'':>6s}"
    short_names = {"A) ALL (21 signals)": "ALL", "B) TECHNICAL only (6)": "TECH",
                   "C) VALUATION only (7)": "VAL", "D) TECH+VAL (13, no lookahead)": "T+V",
                   "E) FUNDAMENTAL only (8, lookahead)": "FUND", "SPY": "SPY"}
    for name in list(signal_groups.keys()) + ["SPY"]:
        header += f" {short_names.get(name, name[:6]):>8s}"
    print(header)
    print(f"  {'-' * (6 + 9 * (len(signal_groups) + 1))}")

    for year in sorted(set(all_dates.year)):
        mask = (all_dates.year == year)[:n_days]
        if mask.sum() < 20: continue
        row = f"  {year:6d}"
        for name in list(signal_groups.keys()) + ["SPY"]:
            yr_ret = (1 + pd.Series(results[name][:n_days])[mask]).prod() - 1
            row += f" {yr_ret*100:+7.1f}%"
        print(row)

    # Interpretation
    all_ret = (1 + pd.Series(results["A) ALL (21 signals)"][:n_days])).cumprod().iloc[-1] - 1
    clean_ret = (1 + pd.Series(results["D) TECH+VAL (13, no lookahead)"][:n_days])).cumprod().iloc[-1] - 1
    fund_ret = (1 + pd.Series(results["E) FUNDAMENTAL only (8, lookahead)"][:n_days])).cumprod().iloc[-1] - 1

    print(f"\n{'=' * 85}")
    print(f"  INTERPRETATION:")
    print(f"")
    print(f"  ALL signals (with fundamentals):    {all_ret*100:+.0f}%")
    print(f"  CLEAN signals (no look-ahead):      {clean_ret*100:+.0f}%")
    print(f"  FUNDAMENTAL only (has look-ahead):  {fund_ret*100:+.0f}%")
    print(f"  SPY:                                {spy_total*100:+.0f}%")
    print(f"")

    if clean_ret > all_ret * 0.7:
        print(f"  VERDICT: Most of the return comes from CLEAN signals.")
        print(f"  The look-ahead bias adds ~{(all_ret - clean_ret) / all_ret * 100:.0f}% to the total.")
        print(f"  The strategy has REAL alpha from price-based signals.")
    elif clean_ret > spy_total:
        print(f"  VERDICT: Clean signals beat SPY but much less than ALL signals.")
        print(f"  ~{(all_ret - clean_ret) / all_ret * 100:.0f}% of total return is from look-ahead bias.")
        print(f"  There IS real alpha, but it's inflated significantly.")
    else:
        print(f"  VERDICT: Clean signals DON'T beat SPY consistently.")
        print(f"  Most of the +136% was look-ahead bias in fundamentals.")
        print(f"  The strategy needs fundamental signals that have look-ahead.")

    print(f"\n{'=' * 85}")
    print(f"  Completed in {time.time() - t_start:.0f}s")
    print(f"{'=' * 85}")


if __name__ == "__main__":
    main()
