#!/usr/bin/env python3
"""
MM Quant Capital — Backtest: Fundamentals as Filter vs Additive Signal

Tests different ways to combine fundamentals with technical signals:

  A) Tech+Val ONLY (no fundamentals) — baseline
  B) ADDITIVE: score = tech_score + fund_score (what we do now)
  C) MULTIPLICATIVE: score = tech_score × (1 + 0.5 × fund_quality)
     where fund_quality = average of PIT fundamental signals
  D) FILTER: only buy if fund_quality > 0 (positive fundamentals required)
     score = tech_score (but filtered universe)
  E) BOOST: score = tech_score × (1 + fund_quality) if fund > 0
            score = tech_score × 0.5 if fund < 0 (penalize bad fundamentals)

All use PIT fundamentals (no look-ahead).
GICS calibrated + Simplified B-L (max 10%).

Usage:
  python backtests/07_fundamentals_as_filter.py
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
    print("  FUNDAMENTALS: Additive vs Multiplicative vs Filter")
    print("  All using PIT fundamentals (no look-ahead)")
    print("=" * 85)

    import sqlite3, yfinance as yf
    from src.data.database import MarketDB
    from src.signals import technical, valuation
    from src.signals.fundamental_pit import compute_all_pit

    db_path = "data/db/market.db"
    conn = sqlite3.connect(db_path)
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

    print("[2/3] Computing signals...")
    tech = technical.compute_all(prices, volumes)
    val = valuation.compute_all(prices)

    # Tech+Val signal matrix
    tv_arrays = []
    for _, df in tech.items():
        tv_arrays.append(df.reindex(columns=tickers).values)
    for _, df in val.items():
        tv_arrays.append(df.reindex(columns=tickers).values)
    S_tv = np.stack(tv_arrays, axis=2)  # (T, N, K_tv)
    K_tv = S_tv.shape[2]

    # PIT fundamentals
    print("  Computing PIT fundamentals...")
    pit_signals = compute_all_pit(db_path, tickers, prices.index, recompute_every=21)
    pit_arrays = []
    for name, df in pit_signals.items():
        pit_arrays.append(df.reindex(index=prices.index, columns=tickers).values)
    S_fund = np.stack(pit_arrays, axis=2)  # (T, N, K_fund)
    K_fund = S_fund.shape[2]

    # Fund quality = average of all PIT fundamental signals per ticker per day
    fund_quality = np.nanmean(S_fund, axis=2)  # (T, N) — average of 8 fund signals

    # Build composite scores for each model
    # We compute tech_score first (GICS calibrated), then combine differently
    ticker_sectors = [sectors_map.get(t, "Unknown") for t in tickers]
    sector_idx = {}
    for i, s in enumerate(ticker_sectors):
        sector_idx.setdefault(s, []).append(i)

    fwd_21 = prices[tickers].pct_change(21).shift(-21).values

    print("[3/3] Walk-forward backtest...")

    models = ["A_tech_val", "B_additive", "C_multiplicative", "D_filter", "E_boost", "SPY"]
    results = {m: [] for m in models}

    train_days = 252
    test_days = 63
    i = train_days
    period = 0

    while i + 21 <= len(prices):
        period += 1
        train_sl = slice(max(0, i - train_days), i)
        test_end = min(i + test_days, len(prices))

        # Calibrate GICS weights for tech+val only
        S_tr = S_tv[train_sl]; F_tr = fwd_21[train_sl]
        global_ic = np.zeros(K_tv)
        for k in range(K_tv):
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
        global_w = global_ic / g_total if g_total > 0 else np.ones(K_tv) / K_tv

        gics_w = {}
        for gname, gidx in sector_idx.items():
            if len(gidx) < 10: gics_w[gname] = global_w.copy(); continue
            gic = np.zeros(K_tv)
            for k in range(K_tv):
                ics = []
                for t in range(0, len(S_tr), 5):
                    sv = S_tr[t, gidx, k]; fv = F_tr[t, gidx]
                    mask = np.isfinite(sv) & np.isfinite(fv)
                    if mask.sum() >= 8:
                        ic, _ = spearmanr(sv[mask], fv[mask])
                        if np.isfinite(ic): ics.append(ic)
                gic[k] = np.mean(ics) if ics else 0
            gic = np.maximum(gic, 0); gt = gic.sum()
            gw = gic / gt if gt > 0 else np.ones(K_tv) / K_tv
            b = 0.5 * global_w + 0.5 * gw; bt = b.sum()
            gics_w[gname] = b / bt if bt > 0 else np.ones(K_tv) / K_tv

        for rb in range(i, test_end, 21):
            rb_end = min(rb + 21, test_end)

            # Tech+Val score per ticker
            tech_scores = np.full(N, np.nan)
            for j in range(N):
                sec = ticker_sectors[j]
                if sec not in gics_w: continue
                w = gics_w[sec]; vals = S_tv[rb][j]
                mask = np.isfinite(vals)
                if mask.sum() >= 3:
                    tech_scores[j] = np.dot(vals[mask], w[mask])

            # Fund quality per ticker (average of 8 PIT signals)
            fq = fund_quality[rb]  # (N,)

            # Build scores for each model
            model_scores = {}

            # A) Tech+Val only
            model_scores["A_tech_val"] = tech_scores.copy()

            # B) Additive: tech + fund (equally weighted)
            additive = tech_scores.copy()
            for j in range(N):
                if np.isfinite(tech_scores[j]) and np.isfinite(fq[j]):
                    additive[j] = tech_scores[j] + fq[j]
            model_scores["B_additive"] = additive

            # C) Multiplicative: tech × (1 + 0.5 × fund)
            mult = tech_scores.copy()
            for j in range(N):
                if np.isfinite(tech_scores[j]) and np.isfinite(fq[j]):
                    mult[j] = tech_scores[j] * (1 + 0.5 * fq[j])
            model_scores["C_multiplicative"] = mult

            # D) Filter: only include if fund > 0
            filtered = tech_scores.copy()
            for j in range(N):
                if np.isfinite(fq[j]) and fq[j] < 0:
                    filtered[j] = np.nan  # exclude bad fundamentals
            model_scores["D_filter"] = filtered

            # E) Boost: amplify good fund, penalize bad fund
            boosted = tech_scores.copy()
            for j in range(N):
                if np.isfinite(tech_scores[j]) and np.isfinite(fq[j]):
                    if fq[j] > 0:
                        boosted[j] = tech_scores[j] * (1 + fq[j])
                    else:
                        boosted[j] = tech_scores[j] * (1 + 0.3 * fq[j])  # mild penalty
            model_scores["E_boost"] = boosted

            # Optimize and compute returns for each model
            train_rets = ret_daily[max(0, rb-252):rb]

            for model_name in ["A_tech_val", "B_additive", "C_multiplicative", "D_filter", "E_boost"]:
                scores = model_scores[model_name]
                valid = np.where(np.isfinite(scores))[0]
                if len(valid) < 20:
                    for day in range(rb, rb_end):
                        if day < len(spy_daily): results[model_name].append(0)
                    continue

                sorted_valid = valid[np.argsort(-scores[valid])]
                n_top = max(1, len(sorted_valid) // 5)
                top_idx = sorted_valid[:n_top]
                bl_w = optimize_bl(scores, train_rets, top_idx, max_weight=0.10)

                for day in range(rb, rb_end):
                    if day >= len(ret_daily): break
                    dr = np.nan_to_num(ret_daily[day, top_idx], nan=0)
                    results[model_name].append(float(np.sum(bl_w * dr)))

            # SPY
            for day in range(rb, rb_end):
                if day >= len(spy_daily): break
                results["SPY"].append(spy_daily[day] if np.isfinite(spy_daily[day]) else 0)

        if period % 4 == 0:
            print(f"  Period {period}...")
        i += test_days

    # ═══════════════════════════════════════════════════════════
    n_days = min(len(r) for r in results.values())
    years = n_days / 252
    spy_total = (1 + pd.Series(results["SPY"][:n_days])).cumprod().iloc[-1] - 1

    labels = {
        "A_tech_val": "A) Tech+Val ONLY (baseline)",
        "B_additive": "B) + Fund ADDITIVE (current method)",
        "C_multiplicative": "C) + Fund MULTIPLICATIVE (×1.5 boost)",
        "D_filter": "D) + Fund as FILTER (exclude bad fund)",
        "E_boost": "E) + Fund BOOST (amplify good, mild penalty bad)",
        "SPY": "SPY Buy & Hold",
    }

    print(f"\n{'=' * 90}")
    print(f"  RESULTS — {n_days} days, {years:.1f} years")
    print(f"  All models: GICS calibrated tech+val, B-L max 10%, PIT fundamentals")
    print(f"{'=' * 90}\n")

    print(f"  {'Model':45s} {'Total':>8s} {'Ann':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Alpha':>8s}")
    print(f"  {'-' * 82}")

    for model in models:
        rets = pd.Series(results[model][:n_days])
        eq = (1 + rets).cumprod()
        total = eq.iloc[-1] - 1
        ann = eq.iloc[-1] ** (1/max(years,0.1)) - 1
        rf_d = 0.045 / 252
        sharpe = ((rets.mean() - rf_d) / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        dd = ((eq - eq.cummax()) / eq.cummax()).min()
        alpha = total - spy_total if model != "SPY" else 0

        best = " ←" if sharpe == max(
            ((pd.Series(results[m][:n_days]).mean() - rf_d) / pd.Series(results[m][:n_days]).std()) * np.sqrt(252)
            if pd.Series(results[m][:n_days]).std() > 0 else 0
            for m in models if m != "SPY"
        ) else ""

        print(f"  {labels[model]:45s} {total*100:+7.1f}% {ann*100:+7.1f}% "
              f"{sharpe:+7.2f} {dd*100:+7.1f}% {alpha*100:+7.1f}%{best}")

    # Year by year
    all_dates = prices.index[train_days:train_days + n_days]
    print(f"\n  Year-by-Year Alpha vs SPY:")
    print(f"  {'':>6s} {'A)T+V':>9s} {'B)Add':>9s} {'C)Mult':>9s} {'D)Filt':>9s} {'E)Boost':>9s}")
    print(f"  {'-' * 52}")

    for year in sorted(set(all_dates.year)):
        mask = (all_dates.year == year)[:n_days]
        if mask.sum() < 20: continue
        spy_yr = (1 + pd.Series(results["SPY"][:n_days])[mask]).prod() - 1
        row = f"  {year:6d}"
        for m in ["A_tech_val", "B_additive", "C_multiplicative", "D_filter", "E_boost"]:
            yr = (1 + pd.Series(results[m][:n_days])[mask]).prod() - 1
            row += f" {(yr-spy_yr)*100:+8.1f}%"
        print(row)

    print(f"\n{'=' * 90}")
    print(f"  Completed in {time.time() - t_start:.0f}s")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
