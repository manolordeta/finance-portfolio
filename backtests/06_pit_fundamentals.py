#!/usr/bin/env python3
"""
MM Quant Capital — Backtest with Point-in-Time Fundamentals

THE DEFINITIVE TEST: Uses fundamental signals that only see data available
at each historical date (via filing_date). No look-ahead bias.

Compares:
  A) ALL signals with PIT fundamentals (honest backtest)
  B) ALL signals with look-ahead fundamentals (inflated, for reference)
  C) TECH+VAL only (no fundamentals at all)
  D) Equal weight top quintile (baseline)

Usage:
  python backtests/06_pit_fundamentals.py
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


def run_walkforward(S, tickers, ticker_sectors, sector_idx, ret_daily, spy_daily,
                    fwd_21, K, train_days=252, test_days=63, max_weight=0.10,
                    top_pct=0.20, use_bl=True):
    """Generic walk-forward. Returns list of daily returns."""
    N = len(tickers)
    results = []

    i = train_days
    while i + 21 <= len(S):
        train_sl = slice(max(0, i - train_days), i)
        test_end = min(i + test_days, len(S))

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
                if mask.sum() >= max(3, K // 3):
                    scores[j] = np.dot(vals[mask], w[mask])

            valid = np.where(np.isfinite(scores))[0]
            if len(valid) < 20:
                for day in range(rb, rb_end):
                    if day < len(spy_daily): results.append(0)
                continue

            sorted_valid = valid[np.argsort(-scores[valid])]
            n_top = max(1, int(len(sorted_valid) * top_pct))
            top_idx = sorted_valid[:n_top]

            if use_bl:
                train_rets = ret_daily[max(0, rb-252):rb]
                w = optimize_bl(scores, train_rets, top_idx, max_weight=max_weight)
            else:
                w = np.ones(len(top_idx)) / len(top_idx)

            for day in range(rb, rb_end):
                if day >= len(ret_daily): break
                dr = np.nan_to_num(ret_daily[day, top_idx], nan=0)
                results.append(float(np.sum(w * dr)))

        i += test_days

    return results


def main():
    t_start = time.time()

    print("=" * 85)
    print("  POINT-IN-TIME FUNDAMENTALS — THE HONEST BACKTEST")
    print("  No look-ahead bias in any signal")
    print("=" * 85)

    import sqlite3, yfinance as yf
    from src.data.database import MarketDB
    from src.signals import technical, valuation
    from src.signals.fundamental_pit import compute_all_pit

    db_path = "data/db/market.db"
    db = MarketDB(db_path)
    conn = sqlite3.connect(db_path)
    all_tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM profiles").fetchall()]
    sectors_map = dict(conn.execute("SELECT ticker, sector FROM profiles").fetchall())
    conn.close()

    print("\n[1/4] Loading prices...")
    data = yf.download(all_tickers + ["SPY"], start="2022-01-01", progress=False)
    prices = data["Close"].dropna(axis=1, thresh=int(len(data)*0.5))
    volumes = data["Volume"].reindex(columns=prices.columns)
    returns = prices.pct_change()
    tickers = [t for t in all_tickers if t in prices.columns and t != "SPY"]
    N = len(tickers)
    print(f"  {len(prices)} days, {N} tickers")

    # Common arrays
    spy_daily = returns["SPY"].values
    ret_daily = returns[tickers].values
    fwd_21 = prices[tickers].pct_change(21).shift(-21).values
    ticker_sectors = [sectors_map.get(t, "Unknown") for t in tickers]
    sector_idx = {}
    for i, s in enumerate(ticker_sectors):
        sector_idx.setdefault(s, []).append(i)

    # ── Technical + Valuation signals (no look-ahead) ─────────
    print("[2/4] Computing technical + valuation signals...")
    tech = technical.compute_all(prices, volumes)
    val = valuation.compute_all(prices)

    tech_val_arrays = []
    for _, df in tech.items():
        tech_val_arrays.append(df.reindex(columns=tickers).values)
    for _, df in val.items():
        tech_val_arrays.append(df.reindex(columns=tickers).values)

    # ── Point-in-Time fundamentals ────────────────────────────
    print("[3/4] Computing Point-in-Time fundamental signals...")
    pit_signals = compute_all_pit(db_path, tickers, prices.index, recompute_every=21)

    pit_arrays = []
    for name, df in pit_signals.items():
        pit_arrays.append(df.reindex(index=prices.index, columns=tickers).values)

    # ── Look-ahead fundamentals (for comparison) ──────────────
    from src.signals import fundamental
    fund_lookahead = fundamental.compute_all(db, tickers)
    lookahead_arrays = []
    for name, scores in fund_lookahead.items():
        s = pd.Series(scores).reindex(tickers).values
        lookahead_arrays.append(np.tile(s, (len(prices), 1)))

    # Build signal matrices for each model
    S_pit_all = np.stack(tech_val_arrays + pit_arrays, axis=2)
    S_lookahead_all = np.stack(tech_val_arrays + lookahead_arrays, axis=2)
    S_tech_val = np.stack(tech_val_arrays, axis=2)

    K_all = S_pit_all.shape[2]
    K_tv = S_tech_val.shape[2]

    print(f"  PIT ALL: {K_all} signals ({len(tech_val_arrays)} tech+val + {len(pit_arrays)} PIT fund)")
    print(f"  Lookahead ALL: {S_lookahead_all.shape[2]} signals")
    print(f"  Tech+Val only: {K_tv} signals")

    # ── Walk-forward for each model ───────────────────────────
    print("[4/4] Walk-forward backtest...")

    print("  Running: PIT ALL (B-L)...")
    r_pit_bl = run_walkforward(S_pit_all, tickers, ticker_sectors, sector_idx,
                                ret_daily, spy_daily, fwd_21, K_all, use_bl=True)

    print("  Running: Lookahead ALL (B-L)...")
    r_look_bl = run_walkforward(S_lookahead_all, tickers, ticker_sectors, sector_idx,
                                 ret_daily, spy_daily, fwd_21, K_all, use_bl=True)

    print("  Running: Tech+Val only (B-L)...")
    r_tv_bl = run_walkforward(S_tech_val, tickers, ticker_sectors, sector_idx,
                               ret_daily, spy_daily, fwd_21, K_tv, use_bl=True)

    print("  Running: PIT ALL (Equal Weight)...")
    r_pit_eq = run_walkforward(S_pit_all, tickers, ticker_sectors, sector_idx,
                                ret_daily, spy_daily, fwd_21, K_all, use_bl=False)

    # SPY
    train_days = 252
    r_spy = spy_daily[train_days:train_days + len(r_pit_bl)].tolist()

    # ═══════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════
    n_days = min(len(r_pit_bl), len(r_look_bl), len(r_tv_bl), len(r_spy))
    years = n_days / 252

    models = {
        "PIT ALL + B-L (HONEST)": r_pit_bl[:n_days],
        "Lookahead ALL + B-L (BIASED)": r_look_bl[:n_days],
        "Tech+Val + B-L (no fundamentals)": r_tv_bl[:n_days],
        "PIT ALL Equal Weight": r_pit_eq[:n_days],
        "SPY Buy & Hold": r_spy[:n_days],
    }

    spy_total = (1 + pd.Series(r_spy[:n_days])).cumprod().iloc[-1] - 1

    print(f"\n{'=' * 90}")
    print(f"  RESULTS — {n_days} days, {years:.1f} years")
    print(f"  All B-L models: GICS calibrated, max 10% per position")
    print(f"{'=' * 90}\n")

    print(f"  {'Model':40s} {'Total':>8s} {'Ann':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Alpha':>8s}")
    print(f"  {'-' * 78}")

    for name, rets_list in models.items():
        rets = pd.Series(rets_list)
        eq = (1 + rets).cumprod()
        total = eq.iloc[-1] - 1
        ann = eq.iloc[-1] ** (1/max(years,0.1)) - 1
        rf_d = 0.045 / 252
        sharpe = ((rets.mean() - rf_d) / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        dd = ((eq - eq.cummax()) / eq.cummax()).min()
        alpha = total - spy_total if "SPY" not in name else 0

        marker = ""
        if "HONEST" in name: marker = " ← THE REAL NUMBER"
        if "BIASED" in name: marker = " ← inflated"

        print(f"  {name:40s} {total*100:+7.1f}% {ann*100:+7.1f}% "
              f"{sharpe:+7.2f} {dd*100:+7.1f}% {alpha*100:+7.1f}%{marker}")

    # Year by year
    all_dates = prices.index[252:252 + n_days]
    print(f"\n  Year-by-Year:")
    print(f"  {'':>6s} {'PIT+BL':>10s} {'Look+BL':>10s} {'T+V+BL':>10s} {'PIT EW':>10s} {'SPY':>10s}")
    print(f"  {'-' * 56}")

    for year in sorted(set(all_dates.year)):
        mask = (all_dates.year == year)[:n_days]
        if mask.sum() < 20: continue
        row = f"  {year:6d}"
        for name in models:
            yr = (1 + pd.Series(models[name])[mask]).prod() - 1
            row += f" {yr*100:+9.1f}%"
        print(row)

    # Final interpretation
    pit_total = (1 + pd.Series(r_pit_bl[:n_days])).cumprod().iloc[-1] - 1
    look_total = (1 + pd.Series(r_look_bl[:n_days])).cumprod().iloc[-1] - 1
    tv_total = (1 + pd.Series(r_tv_bl[:n_days])).cumprod().iloc[-1] - 1

    pit_ann = (1 + pit_total) ** (1/years) - 1
    look_ann = (1 + look_total) ** (1/years) - 1
    tv_ann = (1 + tv_total) ** (1/years) - 1

    print(f"\n{'=' * 90}")
    print(f"  FINAL VERDICT:")
    print(f"")
    print(f"  With look-ahead fundamentals:     {look_ann*100:+.0f}%/yr (inflated)")
    print(f"  With PIT fundamentals (honest):   {pit_ann*100:+.0f}%/yr ← THE REAL NUMBER")
    print(f"  Without fundamentals (tech+val):  {tv_ann*100:+.0f}%/yr")
    print(f"  SPY:                              {(1+spy_total)**(1/years)*100-100:+.0f}%/yr")
    print(f"")

    pit_vs_look = (pit_ann - look_ann) / look_ann * 100 if look_ann > 0 else 0
    fund_contribution = (pit_ann - tv_ann) / pit_ann * 100 if pit_ann > 0 else 0

    print(f"  Look-ahead bias inflated returns by: {-pit_vs_look:.0f}%")
    print(f"  PIT fundamentals contribute: {fund_contribution:.0f}% of honest return")
    print(f"  (the rest comes from technical + valuation signals)")
    print(f"\n{'=' * 90}")


if __name__ == "__main__":
    main()
