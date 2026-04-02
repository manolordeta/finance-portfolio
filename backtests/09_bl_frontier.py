#!/usr/bin/env python3
"""
MM Quant Capital — B-L on the Efficient Frontier

The RIGHT way to use B-L: optimize top 20 by score, but move along
the efficient frontier by targeting different volatility levels.

Tests:
  A) EqWt Top 20 (baseline, no optimization)
  B) B-L Top 20, target vol = 20% (conservative)
  C) B-L Top 20, target vol = 30% (moderate)
  D) B-L Top 20, target vol = 40% (aggressive)
  E) B-L Top 20, target vol = 50% (very aggressive)
  F) B-L Top 20, max Sharpe (let optimizer choose)
  G) B-L Top 20, min variance (minimum risk)

B-L here uses scores as expected returns (not CAPM prior),
sample covariance with shrinkage, and proper target vol optimization.
Top 20 by score, NOT top 100 — so optimizer can't just pick volatile stocks.

Usage:
  python backtests/09_bl_frontier.py
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


def optimize_target_vol(mu, cov, target_vol, max_weight=0.15):
    """Maximize return for a given target volatility level."""
    n = len(mu)
    if n < 3:
        return np.ones(n) / n

    target_var = target_vol ** 2

    def neg_return(w):
        return -(w @ mu)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: target_var - w @ cov @ w},  # vol <= target
    ]
    bounds = [(0, max_weight)] * n

    try:
        r = minimize(neg_return, np.ones(n) / n, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"maxiter": 300, "ftol": 1e-10})
        if r.success and np.all(np.isfinite(r.x)):
            w = np.maximum(r.x, 0)
            if w.sum() > 0:
                return w / w.sum()
    except Exception:
        pass
    return np.ones(n) / n


def optimize_max_sharpe(mu, cov, rf_daily, max_weight=0.15):
    """True max Sharpe ratio optimization."""
    n = len(mu)
    if n < 3:
        return np.ones(n) / n

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf_daily) / vol if vol > 0 else 1e10

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n

    try:
        r = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"maxiter": 300, "ftol": 1e-10})
        if r.success and np.all(np.isfinite(r.x)):
            w = np.maximum(r.x, 0)
            if w.sum() > 0:
                return w / w.sum()
    except Exception:
        pass
    return np.ones(n) / n


def optimize_min_variance(cov, max_weight=0.15):
    """Minimum variance portfolio."""
    n = cov.shape[0]
    if n < 3:
        return np.ones(n) / n

    def variance(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n

    try:
        r = minimize(variance, np.ones(n) / n, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"maxiter": 300, "ftol": 1e-10})
        if r.success and np.all(np.isfinite(r.x)):
            w = np.maximum(r.x, 0)
            if w.sum() > 0:
                return w / w.sum()
    except Exception:
        pass
    return np.ones(n) / n


def main():
    t_start = time.time()
    print("=" * 85)
    print("  B-L ON THE EFFICIENT FRONTIER")
    print("  Top 20 by score, different target volatilities")
    print("=" * 85)

    import sqlite3, yfinance as yf
    from src.data.database import MarketDB
    from src.signals import technical, valuation

    db_path = "data/db/market.db"
    conn = sqlite3.connect(db_path)
    all_tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM profiles").fetchall()]
    sectors_map = dict(conn.execute("SELECT ticker, sector FROM profiles").fetchall())
    conn.close()

    print("\n[1/3] Loading data...")
    data = yf.download(all_tickers + ["SPY"], start="2022-01-01", progress=False)
    prices = data["Close"].dropna(axis=1, thresh=int(len(data) * 0.5))
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
    for _, df in tech.items():
        tv_arrays.append(df.reindex(columns=tickers).values)
    for _, df in val.items():
        tv_arrays.append(df.reindex(columns=tickers).values)
    S = np.stack(tv_arrays, axis=2)
    K = S.shape[2]
    fwd_21 = prices[tickers].pct_change(21).shift(-21).values

    ticker_sectors = [sectors_map.get(t, "Unknown") for t in tickers]
    sector_idx = {}
    for i, s in enumerate(ticker_sectors):
        sector_idx.setdefault(s, []).append(i)

    print("[3/3] Walk-forward backtest...")

    target_vols = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    models = ["eq_top20"] + [f"tvol_{int(v*100)}" for v in target_vols] + ["max_sharpe", "min_var", "SPY"]
    results = {m: [] for m in models}

    train_days = 252
    test_days = 63
    max_weight = 0.15  # allow up to 15% to see differentiation
    i = train_days
    period = 0

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

            # Expected returns from scores (normalized to daily)
            top_scores = scores[top20]
            top_scores_shifted = top_scores - top_scores.min() + 0.001
            mu_daily = (top_scores_shifted / top_scores_shifted.sum()) * 0.15 / 252

            # Covariance (daily, from last 252 days)
            hist_rets = np.nan_to_num(ret_daily[max(0, rb-252):rb, :][:, top20], nan=0)
            cov_daily = np.cov(hist_rets.T)
            if cov_daily.ndim < 2:
                cov_daily = np.eye(20) * 0.001
            cov_daily = 0.8 * cov_daily + 0.2 * np.diag(np.diag(cov_daily))

            rf_daily = 0.045 / 252

            # Build portfolios
            portfolios = {}

            # Equal weight
            portfolios["eq_top20"] = np.ones(20) / 20

            # Target vol portfolios
            for tv in target_vols:
                tv_daily = tv / np.sqrt(252)
                w = optimize_target_vol(mu_daily, cov_daily, tv_daily, max_weight)
                portfolios[f"tvol_{int(tv*100)}"] = w

            # Max Sharpe
            portfolios["max_sharpe"] = optimize_max_sharpe(mu_daily, cov_daily, rf_daily, max_weight)

            # Min variance
            portfolios["min_var"] = optimize_min_variance(cov_daily, max_weight)

            # Compute daily returns
            for day in range(rb, rb_end):
                if day >= len(ret_daily) or day >= len(spy_daily): break
                dr = np.nan_to_num(ret_daily[day, top20], nan=0)
                spy_r = spy_daily[day] if np.isfinite(spy_daily[day]) else 0

                for model_name, w in portfolios.items():
                    results[model_name].append(float(w @ dr))
                results["SPY"].append(spy_r)

        if period % 4 == 0:
            print(f"  Period {period}...")
        i += test_days

    # ═══════════════════════════════════════════════════════════
    n_days = min(len(r) for r in results.values())
    years = n_days / 252
    spy_total = (1 + pd.Series(results["SPY"][:n_days])).cumprod().iloc[-1] - 1

    labels = {
        "eq_top20": "Equal Weight Top 20 (baseline)",
        "max_sharpe": "Max Sharpe Top 20",
        "min_var": "Min Variance Top 20",
        "SPY": "SPY Buy & Hold",
    }
    for tv in target_vols:
        labels[f"tvol_{int(tv*100)}"] = f"Target Vol {tv:.0%} Top 20"

    print(f"\n{'=' * 90}")
    print(f"  B-L EFFICIENT FRONTIER — {n_days} days, {years:.1f} years")
    print(f"  Top 20 by score, max 15% per position, tech+val signals only")
    print(f"{'=' * 90}\n")

    print(f"  {'Model':35s} {'Total':>8s} {'Ann':>8s} {'RealVol':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Alpha':>8s} {'#Pos':>5s}")
    print(f"  {'-' * 85}")

    for model in models:
        rets = pd.Series(results[model][:n_days])
        eq = (1 + rets).cumprod()
        total = eq.iloc[-1] - 1
        ann = eq.iloc[-1] ** (1 / max(years, 0.1)) - 1
        real_vol = rets.std() * np.sqrt(252)
        rf_d = 0.045 / 252
        sharpe = ((rets.mean() - rf_d) / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        dd = ((eq - eq.cummax()) / eq.cummax()).min()
        alpha = total - spy_total if model != "SPY" else 0

        print(f"  {labels[model]:35s} {total*100:+7.1f}% {ann*100:+7.1f}% "
              f"{real_vol*100:7.1f}% {sharpe:+7.2f} {dd*100:+7.1f}% {alpha*100:+7.1f}%")

    # Year by year
    all_dates = prices.index[train_days:train_days + n_days]
    print(f"\n  Year-by-Year Return:")
    header = f"  {'':>6s}"
    for m in models:
        if m == "SPY": continue
        short = labels[m].split("(")[0].strip()[:8]
        header += f" {short:>8s}"
    header += f" {'SPY':>8s}"
    print(header)
    print(f"  {'-' * (6 + 9 * len(models))}")

    for year in sorted(set(all_dates.year)):
        mask = (all_dates.year == year)[:n_days]
        if mask.sum() < 20: continue
        row = f"  {year:6d}"
        for m in models:
            yr = (1 + pd.Series(results[m][:n_days])[mask]).prod() - 1
            row += f" {yr*100:+7.1f}%"
        print(row)

    print(f"\n  KEY INSIGHT:")
    best = max((m for m in models if m != "SPY"),
               key=lambda m: ((pd.Series(results[m][:n_days]).mean() - 0.045/252) /
                              pd.Series(results[m][:n_days]).std()) * np.sqrt(252)
               if pd.Series(results[m][:n_days]).std() > 0 else 0)
    print(f"  Best risk-adjusted: {labels[best]}")
    print(f"\n{'=' * 90}")
    print(f"  Completed in {time.time() - t_start:.0f}s")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
