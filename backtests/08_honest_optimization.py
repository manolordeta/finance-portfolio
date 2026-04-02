#!/usr/bin/env python3
"""
MM Quant Capital — Honest Optimization Comparison

Now that we know simplified B-L was just selecting high volatility,
let's test REAL portfolio construction methods:

  A) Equal weight top quintile (100 positions, +32%/yr baseline)
  B) Equal weight top 10 by SCORE (concentrated, simple)
  C) Equal weight top 20 by SCORE (more diversified)
  D) Risk Parity top 20: weight ∝ 1/vol (less in volatile, more in stable)
  E) Score-weighted top 20: weight ∝ score (more in high conviction)
  F) Score/Vol weighted top 20: weight ∝ score/vol (risk-adjusted conviction)
  G) Full B-L on top quintile (CAPM prior, the conservative one)

All using GICS-calibrated tech+val signals (no look-ahead).

Usage:
  python backtests/08_honest_optimization.py
"""

from __future__ import annotations

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
    t_start = time.time()
    print("=" * 85)
    print("  HONEST OPTIMIZATION — Real Portfolio Construction Methods")
    print("  Tech+Val signals only (no look-ahead)")
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

    models = [
        "A_eq_quintile",
        "B_eq_top10",
        "C_eq_top20",
        "D_riskparity_top20",
        "E_scorewt_top20",
        "F_scorevol_top20",
        "G_full_bl_quintile",
        "SPY",
    ]
    results = {m: [] for m in models}

    train_days = 252
    test_days = 63
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
            if len(gidx) < 10:
                gics_w[gname] = global_w.copy()
                continue
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

        # Rebalance monthly
        for rb in range(i, test_end, 21):
            rb_end = min(rb + 21, test_end)

            # Score all tickers
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

            # Compute volatilities for each ticker (last 63 days)
            vols = np.full(N, np.nan)
            for j in valid:
                hist = ret_daily[max(0, rb-63):rb, j]
                hist = hist[np.isfinite(hist)]
                if len(hist) > 20:
                    vols[j] = np.std(hist) * np.sqrt(252)

            # ── Build portfolios ──────────────────────────────

            portfolios = {}

            # A) Equal weight top quintile (~100)
            n_q = max(1, len(sorted_valid) // 5)
            top_q = sorted_valid[:n_q]
            w_a = np.zeros(N); w_a[top_q] = 1.0 / len(top_q)
            portfolios["A_eq_quintile"] = w_a

            # B) Equal weight top 10 by score
            top10 = sorted_valid[:10]
            w_b = np.zeros(N); w_b[top10] = 1.0 / 10
            portfolios["B_eq_top10"] = w_b

            # C) Equal weight top 20 by score
            top20 = sorted_valid[:20]
            w_c = np.zeros(N); w_c[top20] = 1.0 / 20
            portfolios["C_eq_top20"] = w_c

            # D) Risk parity top 20: weight ∝ 1/vol
            w_d = np.zeros(N)
            inv_vols = []
            for j in top20:
                v = vols[j]
                if np.isfinite(v) and v > 0:
                    inv_vols.append((j, 1.0 / v))
                else:
                    inv_vols.append((j, 1.0))
            total_inv = sum(iv for _, iv in inv_vols)
            for j, iv in inv_vols:
                w_d[j] = iv / total_inv
            portfolios["D_riskparity_top20"] = w_d

            # E) Score-weighted top 20: weight ∝ score
            w_e = np.zeros(N)
            top20_scores = scores[top20]
            top20_scores_pos = top20_scores - top20_scores.min() + 0.001
            total_s = top20_scores_pos.sum()
            for idx, j in enumerate(top20):
                w_e[j] = top20_scores_pos[idx] / total_s
            portfolios["E_scorewt_top20"] = w_e

            # F) Score/Vol weighted top 20: weight ∝ score/vol
            w_f = np.zeros(N)
            score_vol_ratios = []
            for idx, j in enumerate(top20):
                s = top20_scores_pos[idx]
                v = vols[j] if np.isfinite(vols[j]) and vols[j] > 0 else 0.30
                score_vol_ratios.append((j, s / v))
            total_sv = sum(sv for _, sv in score_vol_ratios)
            for j, sv in score_vol_ratios:
                w_f[j] = sv / total_sv if total_sv > 0 else 1.0 / 20
            portfolios["F_scorevol_top20"] = w_f

            # G) Full B-L on top quintile (CAPM prior)
            w_g = np.zeros(N)
            risk_aversion = 2.5
            tau = 0.05
            hist_rets = np.nan_to_num(ret_daily[max(0, rb-252):rb, :][:, top_q], nan=0)
            Sigma = np.cov(hist_rets.T) * 252
            if Sigma.ndim == 2 and Sigma.shape[0] > 1:
                Sigma = 0.8 * Sigma + 0.2 * np.diag(np.diag(Sigma))
                w_eq = np.ones(len(top_q)) / len(top_q)
                Pi = risk_aversion * Sigma @ w_eq
                # Views from scores
                q_scores = scores[top_q]
                q_min, q_max = q_scores.min(), q_scores.max()
                q_range = q_max - q_min if q_max > q_min else 1
                Q = np.array([(s - q_min) / q_range * 0.15 for s in q_scores])
                P = np.eye(len(top_q))
                omega_diag = np.array([(1.0/0.5 - 1.0) * tau * Sigma[ii, ii]
                                       for ii in range(len(top_q))])
                Omega = np.diag(omega_diag)
                try:
                    tS_inv = np.linalg.inv(tau * Sigma)
                    O_inv = np.linalg.inv(Omega)
                    M = tS_inv + P.T @ O_inv @ P
                    M_inv = np.linalg.inv(M)
                    mu_BL = M_inv @ (tS_inv @ Pi + P.T @ O_inv @ Q)
                    # Mean-variance optimize
                    opt_w = np.linalg.inv(risk_aversion * Sigma) @ mu_BL
                    opt_w = np.maximum(opt_w, 0)
                    opt_w = np.minimum(opt_w, 0.10)  # cap at 10%
                    if opt_w.sum() > 0:
                        opt_w = opt_w / opt_w.sum()
                    else:
                        opt_w = w_eq
                    for idx, j in enumerate(top_q):
                        w_g[j] = opt_w[idx]
                except np.linalg.LinAlgError:
                    for j in top_q: w_g[j] = 1.0 / len(top_q)
            else:
                for j in top_q: w_g[j] = 1.0 / len(top_q)
            portfolios["G_full_bl_quintile"] = w_g

            # ── Compute daily returns ─────────────────────────
            for day in range(rb, rb_end):
                if day >= len(ret_daily) or day >= len(spy_daily): break
                dr = ret_daily[day]
                spy_r = spy_daily[day] if np.isfinite(spy_daily[day]) else 0

                for model_name, w in portfolios.items():
                    port_ret = 0.0
                    for j in np.where(w > 0.0001)[0]:
                        r = dr[j]
                        if np.isfinite(r):
                            port_ret += w[j] * r
                    results[model_name].append(port_ret)

                results["SPY"].append(spy_r)

        if period % 4 == 0:
            print(f"  Period {period}...")
        i += test_days

    # ═══════════════════════════════════════════════════════════
    n_days = min(len(r) for r in results.values())
    years = n_days / 252
    spy_total = (1 + pd.Series(results["SPY"][:n_days])).cumprod().iloc[-1] - 1

    labels = {
        "A_eq_quintile": "A) EqWt Top Quintile (~100 pos)",
        "B_eq_top10": "B) EqWt Top 10 by Score",
        "C_eq_top20": "C) EqWt Top 20 by Score",
        "D_riskparity_top20": "D) Risk Parity Top 20 (wt∝1/vol)",
        "E_scorewt_top20": "E) Score-Weighted Top 20 (wt∝score)",
        "F_scorevol_top20": "F) Score/Vol Top 20 (wt∝score/vol)",
        "G_full_bl_quintile": "G) Full B-L Top Quintile (CAPM)",
        "SPY": "SPY Buy & Hold",
    }

    print(f"\n{'=' * 90}")
    print(f"  RESULTS — {n_days} days, {years:.1f} years")
    print(f"  All using GICS-calibrated tech+val signals (no look-ahead)")
    print(f"{'=' * 90}\n")

    print(f"  {'Model':42s} {'Total':>8s} {'Ann':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Alpha':>8s}")
    print(f"  {'-' * 78}")

    sharpes = {}
    for model in models:
        rets = pd.Series(results[model][:n_days])
        eq = (1 + rets).cumprod()
        total = eq.iloc[-1] - 1
        ann = eq.iloc[-1] ** (1 / max(years, 0.1)) - 1
        rf_d = 0.045 / 252
        sharpe = ((rets.mean() - rf_d) / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        dd = ((eq - eq.cummax()) / eq.cummax()).min()
        alpha = total - spy_total if model != "SPY" else 0
        sharpes[model] = sharpe

        best = " ←" if model != "SPY" and sharpe == max(
            v for k, v in sharpes.items() if k != "SPY") else ""
        print(f"  {labels[model]:42s} {total*100:+7.1f}% {ann*100:+7.1f}% "
              f"{sharpe:+7.2f} {dd*100:+7.1f}% {alpha*100:+7.1f}%{best}")

    # Year by year
    all_dates = prices.index[train_days:train_days + n_days]
    print(f"\n  Year-by-Year Alpha vs SPY:")
    short = {"A_eq_quintile":"EqQ100","B_eq_top10":"EqT10","C_eq_top20":"EqT20",
             "D_riskparity_top20":"RskPar","E_scorewt_top20":"ScWt","F_scorevol_top20":"Sc/Vol",
             "G_full_bl_quintile":"FullBL"}
    header = f"  {'':>6s}"
    for m in models:
        if m == "SPY": continue
        header += f" {short.get(m,'?'):>8s}"
    print(header)
    print(f"  {'-' * (6 + 9 * (len(models)-1))}")

    for year in sorted(set(all_dates.year)):
        mask = (all_dates.year == year)[:n_days]
        if mask.sum() < 20: continue
        spy_yr = (1 + pd.Series(results["SPY"][:n_days])[mask]).prod() - 1
        row = f"  {year:6d}"
        for m in models:
            if m == "SPY": continue
            yr = (1 + pd.Series(results[m][:n_days])[mask]).prod() - 1
            row += f" {(yr-spy_yr)*100:+7.1f}%"
        print(row)

    print(f"\n  KEY INSIGHT:")
    best_model = max((m for m in models if m != "SPY"), key=lambda m: sharpes[m])
    print(f"  Best Sharpe: {labels[best_model]} (Sharpe={sharpes[best_model]:.2f})")
    print(f"\n{'=' * 90}")
    print(f"  Completed in {time.time() - t_start:.0f}s")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
