#!/usr/bin/env python3
"""
MM Quant Capital — Frontier from 100 Candidates (Top Quintile)

The KEY question: with 100 candidates instead of 20, can the optimizer
find better combinations by exploiting low correlations?

Compares:
  - Equal Weight Top 20 by score (the current winner, +46%/yr)
  - Target Vol from 100 candidates (optimizer chooses WHICH 10-20 to buy)
  - Max Sharpe from 100 candidates
  - Min Variance from 100 candidates

The difference vs backtest 09: optimizer has 5x more candidates to choose from.

Usage:
  python backtests/10_frontier_100_candidates.py
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


def optimize_target_vol_100(mu, cov, target_vol, max_weight=0.10):
    """Maximize return subject to vol <= target, from 100 candidates."""
    n = len(mu)
    if n < 5: return np.ones(n) / n
    target_var = target_vol ** 2

    def neg_return(w):
        return -(w @ mu)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: target_var - w @ cov @ w},
    ]
    bounds = [(0, max_weight)] * n

    # Try multiple starting points
    best_w = np.ones(n) / n
    best_ret = -np.inf

    for attempt in range(3):
        if attempt == 0:
            w0 = np.ones(n) / n
        elif attempt == 1:
            # Start concentrated in top 10 by mu
            w0 = np.zeros(n)
            top10 = np.argsort(-mu)[:10]
            w0[top10] = 0.10
        else:
            # Random start
            w0 = np.random.dirichlet(np.ones(n))

        try:
            r = minimize(neg_return, w0, method="SLSQP",
                        bounds=bounds, constraints=constraints,
                        options={"maxiter": 500, "ftol": 1e-10})
            if r.success and np.all(np.isfinite(r.x)):
                w = np.maximum(r.x, 0)
                if w.sum() > 0:
                    w = w / w.sum()
                    ret = w @ mu
                    vol = np.sqrt(w @ cov @ w)
                    if ret > best_ret and vol <= target_vol * 1.05:
                        best_w = w
                        best_ret = ret
        except Exception:
            pass

    return best_w


def optimize_max_sharpe_100(mu, cov, rf, max_weight=0.10):
    """Max Sharpe from 100 candidates."""
    n = len(mu)
    if n < 5: return np.ones(n) / n

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf) / vol if vol > 1e-10 else 1e10

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n

    best_w = np.ones(n) / n
    best_sharpe = -np.inf

    for attempt in range(3):
        if attempt == 0:
            w0 = np.ones(n) / n
        elif attempt == 1:
            w0 = np.zeros(n)
            top10 = np.argsort(-mu)[:10]
            w0[top10] = 0.10
        else:
            w0 = np.random.dirichlet(np.ones(n))

        try:
            r = minimize(neg_sharpe, w0, method="SLSQP",
                        bounds=bounds, constraints=constraints,
                        options={"maxiter": 500, "ftol": 1e-10})
            if r.success and np.all(np.isfinite(r.x)):
                w = np.maximum(r.x, 0)
                if w.sum() > 0:
                    w = w / w.sum()
                    s = -(neg_sharpe(w))
                    if s > best_sharpe:
                        best_w = w
                        best_sharpe = s
        except Exception:
            pass

    return best_w


def optimize_min_var_100(cov, max_weight=0.10):
    """Min variance from 100 candidates."""
    n = cov.shape[0]
    def var(w): return w @ cov @ w
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n
    try:
        r = minimize(var, np.ones(n)/n, method="SLSQP",
                    bounds=bounds, constraints=constraints)
        if r.success:
            w = np.maximum(r.x, 0); return w / w.sum()
    except: pass
    return np.ones(n) / n


def main():
    t_start = time.time()
    np.random.seed(42)

    print("=" * 90)
    print("  EFFICIENT FRONTIER — 100 Candidates (Top Quintile)")
    print("  Can the optimizer exploit low correlations from a larger pool?")
    print("=" * 90)

    import sqlite3, yfinance as yf
    from src.signals import technical, valuation

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

    spy_daily = returns["SPY"].values
    ret_daily = returns[tickers].values

    print("[2/3] Computing tech+val signals...")
    tech = technical.compute_all(prices, volumes)
    val = valuation.compute_all(prices)
    tv_arrays = []
    for _, df in tech.items(): tv_arrays.append(df.reindex(columns=tickers).values)
    for _, df in val.items(): tv_arrays.append(df.reindex(columns=tickers).values)
    S = np.stack(tv_arrays, axis=2); K = S.shape[2]
    fwd_21 = prices[tickers].pct_change(21).shift(-21).values

    ticker_sectors = [sectors_map.get(t, "Unknown") for t in tickers]
    sector_idx = {}
    for i, s in enumerate(ticker_sectors): sector_idx.setdefault(s, []).append(i)

    print("[3/3] Walk-forward backtest...")

    target_vols = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    models = (["eq_top20", "eq_top100"] +
              [f"100_tvol_{int(v*100)}" for v in target_vols] +
              ["100_maxsharpe", "100_minvar", "SPY"])
    results = {m: [] for m in models}

    train_days = 252; test_days = 63
    i = train_days; period = 0

    while i + 21 <= len(prices):
        period += 1
        train_sl = slice(max(0, i - train_days), i)
        test_end = min(i + test_days, len(prices))

        # GICS calibration
        S_tr = S[train_sl]; F_tr = fwd_21[train_sl]
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

        gics_w = {}
        for gname, gidx in sector_idx.items():
            if len(gidx) < 10: gics_w[gname] = global_w.copy(); continue
            gic = np.zeros(K)
            for k in range(K):
                ics = []
                for t in range(0, len(S_tr), 5):
                    sv = S_tr[t, gidx, k]; fv = F_tr[t, gidx]
                    mask = np.isfinite(sv) & np.isfinite(fv)
                    if mask.sum() >= 8:
                        ic, _ = spearmanr(sv[mask], fv[mask])
                        if np.isfinite(ic): ics.append(ic)
                gic[k] = np.mean(ics) if ics else 0
            gic = np.maximum(gic, 0); gt = gic.sum()
            gw = gic / gt if gt > 0 else np.ones(K) / K
            b = 0.5 * global_w + 0.5 * gw; bt = b.sum()
            gics_w[gname] = b / bt if bt > 0 else np.ones(K) / K

        for rb in range(i, test_end, 21):
            rb_end = min(rb + 21, test_end)

            scores = np.full(N, np.nan)
            for j in range(N):
                sec = ticker_sectors[j]
                if sec not in gics_w: continue
                w = gics_w[sec]; vals = S[rb][j]
                mask = np.isfinite(vals)
                if mask.sum() >= 3:
                    scores[j] = np.dot(vals[mask], w[mask])

            valid = np.where(np.isfinite(scores))[0]
            if len(valid) < 20:
                for day in range(rb, rb_end):
                    if day < len(spy_daily):
                        for m in models: results[m].append(0)
                continue

            sorted_valid = valid[np.argsort(-scores[valid])]
            top20 = sorted_valid[:20]
            n_q = max(1, len(sorted_valid) // 5)
            top100 = sorted_valid[:n_q]

            # Mu and cov for top 100
            top100_scores = scores[top100]
            top100_shifted = top100_scores - top100_scores.min() + 0.001
            mu_100 = (top100_shifted / top100_shifted.sum()) * 0.15 / 252

            hist_100 = np.nan_to_num(ret_daily[max(0, rb-252):rb, :][:, top100], nan=0)
            cov_100 = np.cov(hist_100.T)
            if cov_100.ndim < 2: cov_100 = np.eye(len(top100)) * 0.001
            cov_100 = 0.8 * cov_100 + 0.2 * np.diag(np.diag(cov_100))

            rf_d = 0.045 / 252

            portfolios = {}

            # Baselines
            w_eq20 = np.zeros(N); w_eq20[top20] = 1.0/20
            portfolios["eq_top20"] = w_eq20

            w_eq100 = np.zeros(N); w_eq100[top100] = 1.0/len(top100)
            portfolios["eq_top100"] = w_eq100

            # Target vol from 100
            for tv in target_vols:
                tv_d = tv / np.sqrt(252)
                w_opt = optimize_target_vol_100(mu_100, cov_100, tv_d, max_weight=0.10)
                w_full = np.zeros(N)
                for idx, j in enumerate(top100):
                    w_full[j] = w_opt[idx]
                portfolios[f"100_tvol_{int(tv*100)}"] = w_full

            # Max Sharpe from 100
            w_ms = optimize_max_sharpe_100(mu_100, cov_100, rf_d, max_weight=0.10)
            w_full = np.zeros(N)
            for idx, j in enumerate(top100): w_full[j] = w_ms[idx]
            portfolios["100_maxsharpe"] = w_full

            # Min var from 100
            w_mv = optimize_min_var_100(cov_100, max_weight=0.10)
            w_full = np.zeros(N)
            for idx, j in enumerate(top100): w_full[j] = w_mv[idx]
            portfolios["100_minvar"] = w_full

            # Compute returns
            for day in range(rb, rb_end):
                if day >= len(ret_daily) or day >= len(spy_daily): break
                dr = ret_daily[day]
                spy_r = spy_daily[day] if np.isfinite(spy_daily[day]) else 0
                for m, w in portfolios.items():
                    active = np.where(w > 0.0001)[0]
                    r = sum(w[j] * dr[j] for j in active if np.isfinite(dr[j]))
                    results[m].append(r)
                results["SPY"].append(spy_r)

        if period % 4 == 0:
            print(f"  Period {period}...")
        i += test_days

    # ═══════════════════════════════════════════════════════════
    n_days = min(len(r) for r in results.values())
    years = n_days / 252
    spy_total = (1 + pd.Series(results["SPY"][:n_days])).cumprod().iloc[-1] - 1

    labels = {
        "eq_top20": "EqWt Top 20 (baseline)",
        "eq_top100": "EqWt Top 100 (quintile)",
        "100_maxsharpe": "Max Sharpe from 100",
        "100_minvar": "Min Variance from 100",
        "SPY": "SPY Buy & Hold",
    }
    for tv in target_vols:
        labels[f"100_tvol_{int(tv*100)}"] = f"Target Vol {tv:.0%} from 100"

    print(f"\n{'=' * 95}")
    print(f"  EFFICIENT FRONTIER FROM 100 CANDIDATES — {n_days} days, {years:.1f} years")
    print(f"  Tech+val signals, GICS calibrated, max 10% per position")
    print(f"{'=' * 95}\n")

    print(f"  {'Model':35s} {'Total':>8s} {'Ann':>8s} {'RealVol':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'#Pos':>5s} {'Alpha':>8s}")
    print(f"  {'-' * 88}")

    for model in models:
        rets = pd.Series(results[model][:n_days])
        eq = (1 + rets).cumprod()
        total = eq.iloc[-1] - 1
        ann = eq.iloc[-1] ** (1/max(years,0.1)) - 1
        real_vol = rets.std() * np.sqrt(252)
        rf_d = 0.045 / 252
        sharpe = ((rets.mean() - rf_d) / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        dd = ((eq - eq.cummax()) / eq.cummax()).min()
        alpha = total - spy_total if model != "SPY" else 0

        # Count avg positions
        # (approximate from weights)
        n_pos = "~20" if "top20" in model else "~100" if "top100" in model else "~10"

        print(f"  {labels[model]:35s} {total*100:+7.1f}% {ann*100:+7.1f}% "
              f"{real_vol*100:7.1f}% {sharpe:+7.2f} {dd*100:+7.1f}% {n_pos:>5s} {alpha*100:+7.1f}%")

    # Year by year
    all_dates = prices.index[train_days:train_days + n_days]
    print(f"\n  Year-by-Year Alpha vs SPY:")
    print(f"  {'':>6s} {'EqT20':>8s} {'EqT100':>8s} {'TV15':>8s} {'TV25':>8s} {'TV35':>8s} {'MaxSh':>8s} {'MinV':>8s}")
    print(f"  {'-' * 62}")

    show_models = ["eq_top20", "eq_top100", "100_tvol_15", "100_tvol_25", "100_tvol_35", "100_maxsharpe", "100_minvar"]
    for year in sorted(set(all_dates.year)):
        mask = (all_dates.year == year)[:n_days]
        if mask.sum() < 20: continue
        spy_yr = (1 + pd.Series(results["SPY"][:n_days])[mask]).prod() - 1
        row = f"  {year:6d}"
        for m in show_models:
            yr = (1 + pd.Series(results[m][:n_days])[mask]).prod() - 1
            row += f" {(yr-spy_yr)*100:+7.1f}%"
        print(row)

    print(f"\n{'=' * 95}")
    print(f"  Completed in {time.time() - t_start:.0f}s")
    print(f"{'=' * 95}")


if __name__ == "__main__":
    main()
