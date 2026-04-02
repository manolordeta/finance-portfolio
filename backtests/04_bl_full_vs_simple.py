#!/usr/bin/env python3
"""
MM Quant Capital — Backtest: Full B-L (GARCH+CAPM) vs Simplified B-L

Compares the TWO optimization approaches side by side:
  A) Simplified B-L (backtest replica): scores → mu direct, sample cov
  B) Full B-L: CAPM prior + views + GARCH covariance + sector constraints
  C) Equal weight top quintile (baseline)

Both use GICS-calibrated ranking for scoring.
Walk-forward: 12m train, 3m test, monthly rebalance.

Usage:
  python backtests/04_bl_full_vs_simple.py
  python backtests/04_bl_full_vs_simple.py --max-weight 0.10
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
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()


# ── Simplified B-L (from backtest 02) ────────────────────────────────

def optimize_simple(scores, ret_history, top_indices, rf=0.045/252, max_weight=0.10):
    """Scores → mu direct, sample cov with shrinkage, max Sharpe."""
    n = len(top_indices)
    if n < 5:
        return np.ones(n) / n

    top_scores = scores[top_indices] - scores[top_indices].min()
    total = top_scores.sum()
    if total <= 0:
        return np.ones(n) / n

    mu = (top_scores / total) * 0.15 / 252

    recent_rets = np.nan_to_num(ret_history[:, top_indices], nan=0)
    cov = np.cov(recent_rets.T)
    if cov.ndim < 2:
        return np.ones(n) / n
    cov = 0.8 * cov + 0.2 * np.diag(np.diag(cov))

    def neg_sharpe(w):
        pr = w @ mu
        pv = w @ cov @ w
        return -(pr - rf) / np.sqrt(pv) if pv > 0 else 1e10

    try:
        r = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                    bounds=[(0, max_weight)] * n,
                    constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                    options={"maxiter": 200, "ftol": 1e-8})
        if r.success and np.all(np.isfinite(r.x)):
            w = np.maximum(r.x, 0)
            return w / w.sum()
    except Exception:
        pass
    return np.ones(n) / n


# ── Full B-L (from run_portfolio.py) ─────────────────────────────────

def optimize_full_bl(scores, ret_history, top_indices, sectors_list,
                     rf_annual=0.045, max_weight=0.10, max_sector=0.35,
                     tau=0.05, risk_aversion=2.5, view_scale=0.15):
    """Full B-L: CAPM prior + views + sector constraints."""
    n = len(top_indices)
    if n < 5:
        return np.ones(n) / n

    recent_rets = np.nan_to_num(ret_history[:, top_indices], nan=0)
    Sigma = np.cov(recent_rets.T) * 252  # annualized
    if Sigma.ndim < 2:
        return np.ones(n) / n
    Sigma = 0.8 * Sigma + 0.2 * np.diag(np.diag(Sigma))

    # 1. Market equilibrium returns (CAPM prior)
    w_eq = np.ones(n) / n  # assume equal market cap
    Pi = risk_aversion * Sigma @ w_eq

    # 2. Views from scores
    top_scores = scores[top_indices]
    score_min = top_scores.min()
    score_max = top_scores.max()
    score_range = score_max - score_min if score_max > score_min else 1

    P = np.eye(n)  # absolute views
    Q = np.array([(s - score_min) / score_range * view_scale for s in top_scores])

    # View uncertainty
    omega_diag = np.array([(1.0 / 0.5 - 1.0) * tau * Sigma[i, i] for i in range(n)])
    Omega = np.diag(omega_diag)

    # 3. B-L posterior
    try:
        tau_Sigma_inv = np.linalg.inv(tau * Sigma)
        Omega_inv = np.linalg.inv(Omega)
        M = tau_Sigma_inv + P.T @ Omega_inv @ P
        M_inv = np.linalg.inv(M)
        mu_BL = M_inv @ (tau_Sigma_inv @ Pi + P.T @ Omega_inv @ Q)
    except np.linalg.LinAlgError:
        return np.ones(n) / n

    # 4. Optimize with sector constraints
    # Mean-variance utility: max (w'μ - 0.5 * δ * w'Σw)
    def neg_utility(w):
        ret = w @ mu_BL
        risk = w @ (Sigma / 252) @ w  # daily
        return -(ret / 252 - 0.5 * risk_aversion * risk)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # Sector constraints
    sector_groups = {}
    for i, s in enumerate(sectors_list):
        sector_groups.setdefault(s, []).append(i)
    for sec, indices in sector_groups.items():
        idx_copy = list(indices)  # capture
        constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=idx_copy: max_sector - sum(w[i] for i in idx)
        })

    bounds = [(0, max_weight)] * n

    try:
        r = minimize(neg_utility, np.ones(n) / n, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"maxiter": 300, "ftol": 1e-8})
        if r.success and np.all(np.isfinite(r.x)):
            w = np.maximum(r.x, 0)
            if w.sum() > 0:
                return w / w.sum()
    except Exception:
        pass
    return np.ones(n) / n


def main():
    parser = argparse.ArgumentParser(description="Full B-L vs Simplified B-L")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--max-weight", type=float, default=0.10)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--top-pct", type=float, default=0.20)
    args = parser.parse_args()

    train_days = args.train_months * 21
    test_days = args.test_months * 21

    t_start = time.time()
    print("=" * 85)
    print("  Full B-L (GARCH+CAPM+Sectors) vs Simplified B-L (Scores Direct)")
    print(f"  Max weight: {args.max_weight:.0%} | Train: {args.train_months}m | Test: {args.test_months}m")
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
    sectors_map = dict(conn.execute("SELECT ticker, sector FROM profiles").fetchall())
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

    S = np.stack(sig_list, axis=2)
    K = len(sig_names)
    fwd_21 = prices[tickers].pct_change(21).shift(-21).values
    spy_daily = returns["SPY"].values
    ret_daily = returns[tickers].values

    ticker_sectors = [sectors_map.get(t, "Unknown") for t in tickers]
    sector_idx = {}
    for i, s in enumerate(ticker_sectors):
        sector_idx.setdefault(s, []).append(i)

    # ── GICS calibration function ────────────────────────────
    def compute_gics_weights(train_sl):
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
        gw = {}
        for gname, gidx in sector_idx.items():
            if len(gidx) < 10: gw[gname] = global_w.copy(); continue
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
            gw2 = gic / gt if gt > 0 else np.ones(K) / K
            b = 0.5 * global_w + 0.5 * gw2; bt = b.sum()
            gw[gname] = b / bt if bt > 0 else np.ones(K) / K
        return gw

    # ── Walk-forward ──────────────────────────────────────────
    print("[3/3] Walk-forward backtest...")

    models = ["equal", "simple_bl", "full_bl", "spy"]
    results = {m: [] for m in models}

    i = train_days
    period = 0

    while i + 21 <= len(prices):
        period += 1
        train_sl = slice(max(0, i - train_days), i)
        test_end = min(i + test_days, len(prices))

        gics_weights = compute_gics_weights(train_sl)

        for rb in range(i, test_end, 21):
            rb_end = min(rb + 21, test_end)

            # Score all tickers
            scores = np.full(N, np.nan)
            for j in range(N):
                sec = ticker_sectors[j]
                if sec not in gics_weights: continue
                w = gics_weights[sec]
                vals = S[rb][j]
                mask = np.isfinite(vals)
                if mask.sum() >= 5:
                    scores[j] = np.dot(vals[mask], w[mask])

            valid = np.where(np.isfinite(scores))[0]
            if len(valid) < 20:
                for day in range(rb, rb_end):
                    if day >= len(spy_daily): break
                    for m in models: results[m].append(0)
                continue

            sorted_valid = valid[np.argsort(-scores[valid])]
            n_top = max(1, int(len(sorted_valid) * args.top_pct))
            top_idx = sorted_valid[:n_top]

            train_rets = ret_daily[max(0, rb - 252):rb]

            # Equal weight
            eq_w = np.ones(len(top_idx)) / len(top_idx)

            # Simplified B-L
            simple_w = optimize_simple(scores, train_rets, top_idx,
                                       max_weight=args.max_weight)

            # Full B-L (CAPM + sectors)
            top_sectors = [ticker_sectors[idx] for idx in top_idx]
            full_w = optimize_full_bl(scores, train_rets, top_idx,
                                      sectors_list=top_sectors,
                                      max_weight=args.max_weight)

            # Daily returns
            for day in range(rb, rb_end):
                if day >= len(ret_daily) or day >= len(spy_daily): break
                spy_r = spy_daily[day] if np.isfinite(spy_daily[day]) else 0
                dr = np.nan_to_num(ret_daily[day, top_idx], nan=0)

                results["equal"].append(float(np.sum(eq_w * dr)))
                results["simple_bl"].append(float(np.sum(simple_w * dr)))
                results["full_bl"].append(float(np.sum(full_w * dr)))
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

    print(f"\n{'=' * 85}")
    print(f"  RESULTS — {period} periods, {n_days} days, {years:.1f} years")
    print(f"  Max weight: {args.max_weight:.0%}")
    print(f"{'=' * 85}\n")

    labels = {
        "equal": "Equal Weight (top quintile)",
        "simple_bl": "Simplified B-L (scores→mu)",
        "full_bl": "Full B-L (CAPM+sectors)",
        "spy": "SPY Buy & Hold",
    }

    print(f"  {'Model':35s} {'Total':>8s} {'Ann':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Alpha':>8s}")
    print(f"  {'-' * 72}")

    for model in models:
        rets = pd.Series(results[model])
        eq = (1 + rets).cumprod()
        total = eq.iloc[-1] - 1
        ann = eq.iloc[-1] ** (1 / max(years, 0.1)) - 1
        rf_daily = 0.045 / 252
        sharpe = ((rets.mean() - rf_daily) / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        dd = ((eq - eq.cummax()) / eq.cummax()).min()
        alpha = total - spy_total if model != "spy" else 0

        print(f"  {labels[model]:35s} {total*100:+7.1f}% {ann*100:+7.1f}% "
              f"{sharpe:+7.2f} {dd*100:+7.1f}% {alpha*100:+7.1f}%")

    # Year by year
    all_dates = prices.index[train_days:train_days + n_days]
    print(f"\n  Year-by-Year Alpha vs SPY:")
    print(f"  {'':>6s} {'Equal':>12s} {'Simple BL':>12s} {'Full BL':>12s}")
    print(f"  {'-' * 44}")

    for year in sorted(set(all_dates.year)):
        mask = (all_dates.year == year)[:n_days]
        if mask.sum() < 20: continue
        spy_yr = (1 + pd.Series(results["spy"])[mask]).prod() - 1
        row = f"  {year:6d}"
        for m in ["equal", "simple_bl", "full_bl"]:
            yr_ret = (1 + pd.Series(results[m])[mask]).prod() - 1
            row += f" {(yr_ret - spy_yr)*100:+11.1f}%"
        print(row)

    # Concentration analysis
    print(f"\n  Portfolio Concentration (last rebalance):")
    for name, w in [("Equal", eq_w), ("Simple B-L", simple_w), ("Full B-L", full_w)]:
        nonzero = (w > 0.001).sum()
        top1 = np.max(w) * 100
        top5 = np.sort(w)[-min(5, len(w)):].sum() * 100
        print(f"    {name:15s}  positions={nonzero:3d}  max={top1:5.1f}%  top5={top5:5.1f}%")

    print(f"\n{'=' * 85}")
    print(f"  Completed in {elapsed:.0f}s")
    print(f"{'=' * 85}")


if __name__ == "__main__":
    main()
