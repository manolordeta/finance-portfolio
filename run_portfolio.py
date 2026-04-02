#!/usr/bin/env python3
"""
MM Quant Capital — Portfolio Optimizer

Uses the EXACT same optimization from backtests that was validated.
Scores universe with GICS-calibrated tech+val signals, takes top quintile
(~100 candidates), and optimizes portfolio weights.

Modes:
  equal20   — Equal weight top 20 by score (Sharpe 1.22, +46%/yr in backtest)
  equal100  — Equal weight top quintile (~100 positions, +29%/yr)
  tvol      — Target volatility from 100 candidates (you choose risk level)
  maxsharpe — Max Sharpe from 100 candidates (+110%/yr in backtest, aggressive)
  minvar    — Min variance from 100 candidates (safest, +14%/yr)

Usage:
  python run_portfolio.py                          # default: equal20
  python run_portfolio.py --mode equal20           # top 20 equal weight (best Sharpe)
  python run_portfolio.py --mode tvol --vol 35     # target 35% vol from 100
  python run_portfolio.py --mode maxsharpe         # max Sharpe from 100 (aggressive)
  python run_portfolio.py --mode minvar            # minimum variance (conservative)
  python run_portfolio.py --tickers MU,CIEN,LITE   # specific tickers, equal weight
  python run_portfolio.py --max-weight 0.15        # allow up to 15% per position
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from scipy.optimize import minimize

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
log = logging.getLogger("portfolio")


# ── Optimizers (same as validated backtests) ─────────────────────────

def optimize_target_vol(mu, cov, target_vol, max_weight=0.10):
    """Maximize return subject to portfolio vol <= target."""
    n = len(mu)
    if n < 3: return np.ones(n) / n
    target_var = target_vol ** 2

    def neg_return(w): return -(w @ mu)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: target_var - w @ cov @ w},
    ]
    bounds = [(0, max_weight)] * n

    best_w = np.ones(n) / n
    best_ret = -np.inf
    for attempt in range(3):
        w0 = (np.ones(n) / n if attempt == 0
              else np.zeros(n) if attempt == 1
              else np.random.dirichlet(np.ones(n)))
        if attempt == 1:
            top10 = np.argsort(-mu)[:min(10, n)]
            w0[top10] = 1.0 / len(top10)
        try:
            r = minimize(neg_return, w0, method="SLSQP", bounds=bounds,
                        constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})
            if r.success and np.all(np.isfinite(r.x)):
                w = np.maximum(r.x, 0)
                if w.sum() > 0:
                    w = w / w.sum()
                    if w @ mu > best_ret and np.sqrt(w @ cov @ w) <= target_vol * 1.05:
                        best_w = w; best_ret = w @ mu
        except Exception: pass
    return best_w


def optimize_max_sharpe(mu, cov, rf, max_weight=0.10):
    """Maximize Sharpe ratio."""
    n = len(mu)
    if n < 3: return np.ones(n) / n

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu - rf) / vol if vol > 1e-10 else 1e10

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n

    best_w = np.ones(n) / n; best_s = -np.inf
    for attempt in range(3):
        w0 = (np.ones(n) / n if attempt == 0
              else np.zeros(n) if attempt == 1
              else np.random.dirichlet(np.ones(n)))
        if attempt == 1:
            top10 = np.argsort(-mu)[:min(10, n)]
            w0[top10] = 1.0 / len(top10)
        try:
            r = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                        constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})
            if r.success and np.all(np.isfinite(r.x)):
                w = np.maximum(r.x, 0)
                if w.sum() > 0:
                    w = w / w.sum()
                    s = -neg_sharpe(w)
                    if s > best_s: best_w = w; best_s = s
        except Exception: pass
    return best_w


def optimize_min_var(cov, max_weight=0.10):
    """Minimum variance portfolio."""
    n = cov.shape[0]
    if n < 3: return np.ones(n) / n
    def var(w): return w @ cov @ w
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n
    try:
        r = minimize(var, np.ones(n)/n, method="SLSQP", bounds=bounds,
                    constraints=constraints)
        if r.success:
            w = np.maximum(r.x, 0); return w / w.sum()
    except: pass
    return np.ones(n) / n


# ── Main ─────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    parser = argparse.ArgumentParser(description="MM Quant Capital — Portfolio Optimizer")
    parser.add_argument("--mode", choices=["equal20", "equal100", "tvol", "maxsharpe", "minvar"],
                        default="equal20",
                        help="equal20=top 20 EW (best Sharpe), tvol=target vol, maxsharpe=aggressive")
    parser.add_argument("--vol", type=float, default=35.0,
                        help="Target annualized vol %% for tvol mode (default: 35)")
    parser.add_argument("--max-weight", type=float, default=0.10,
                        help="Max weight per position (default: 0.10)")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers (overrides mode)")
    parser.add_argument("--no-chart", action="store_true")
    args = parser.parse_args()

    t_start = time.time()
    today = datetime.now().strftime("%Y-%m-%d")

    with open("config/signals.yaml") as f:
        cfg = yaml.safe_load(f)

    log.info("=" * 70)
    log.info("  MM QUANT CAPITAL — Portfolio Optimizer")
    log.info("  Date: %s | Mode: %s", today, args.mode)
    if args.mode == "tvol":
        log.info("  Target Vol: %.0f%% | Max Weight: %.0f%%", args.vol, args.max_weight * 100)
    log.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────
    import yfinance as yf
    from src.data.database import MarketDB
    from src.signals import technical, valuation

    db = MarketDB("data/db/market.db")
    conn = sqlite3.connect("data/db/market.db")
    all_tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM profiles").fetchall()]
    sectors_map = dict(conn.execute("SELECT ticker, sector FROM profiles").fetchall())
    profiles_map = {r[0]: {"name": r[1], "sector": r[2]} for r in conn.execute(
        "SELECT ticker, company_name, sector FROM profiles").fetchall()}
    conn.close()

    log.info("[1/4] Downloading prices...")
    data = yf.download(all_tickers + ["SPY"], start="2023-01-01", progress=False)
    prices = data["Close"].dropna(axis=1, thresh=int(len(data) * 0.5))
    volumes = data["Volume"].reindex(columns=prices.columns)
    returns = prices.pct_change()
    tickers = [t for t in all_tickers if t in prices.columns and t != "SPY"]
    N = len(tickers)
    log.info("  %d days x %d tickers", len(prices), N)

    # ── Signals (tech + val only, no look-ahead) ──────────────
    log.info("[2/4] Computing tech+val signals...")
    tech = technical.compute_all(prices, volumes)
    val = valuation.compute_all(prices)

    tv_arrays = []
    sig_names = []
    for name, df in tech.items():
        sig_names.append(f"T_{name}")
        tv_arrays.append(df.reindex(columns=tickers).values)
    for name, df in val.items():
        sig_names.append(f"V_{name}")
        tv_arrays.append(df.reindex(columns=tickers).values)
    S = np.stack(tv_arrays, axis=2)
    K = S.shape[2]
    fwd_21 = prices[tickers].pct_change(21).shift(-21).values

    ticker_sectors = [sectors_map.get(t, "Unknown") for t in tickers]
    sector_idx = {}
    for i, s in enumerate(ticker_sectors):
        sector_idx.setdefault(s, []).append(i)

    log.info("  %d signals (tech+val, no look-ahead)", K)

    # ── GICS calibration (last 12 months) ─────────────────────
    log.info("[3/4] Calibrating GICS weights...")
    train_sl = slice(max(0, len(prices) - 252), len(prices))
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

    # ── Score and optimize ────────────────────────────────────
    log.info("[4/4] Scoring and optimizing...")
    day_idx = len(prices) - 1

    scores = np.full(N, np.nan)
    for j in range(N):
        sec = ticker_sectors[j]
        if sec not in gics_w: continue
        w = gics_w[sec]; vals = S[day_idx][j]
        mask = np.isfinite(vals)
        if mask.sum() >= 3:
            scores[j] = np.dot(vals[mask], w[mask])

    valid = np.where(np.isfinite(scores))[0]
    sorted_valid = valid[np.argsort(-scores[valid])]

    # Select candidates based on mode
    if args.tickers:
        ticker_list = [t.strip().upper() for t in args.tickers.split(",")]
        candidates = np.array([np.where(np.array(tickers) == t)[0][0]
                               for t in ticker_list if t in tickers])
    elif args.mode in ("equal20",):
        candidates = sorted_valid[:20]
    else:  # all quintile modes (tvol, maxsharpe, minvar, equal100)
        n_q = max(1, len(sorted_valid) // 5)
        candidates = sorted_valid[:n_q]

    n_cand = len(candidates)

    # Expected returns from scores
    cand_scores = scores[candidates]
    cand_shifted = cand_scores - cand_scores.min() + 0.001
    mu_daily = (cand_shifted / cand_shifted.sum()) * 0.15 / 252

    # Covariance (sample + shrinkage)
    hist = np.nan_to_num(returns[tickers].iloc[-252:].values[:, candidates], nan=0)
    cov_daily = np.cov(hist.T)
    if cov_daily.ndim < 2: cov_daily = np.eye(n_cand) * 0.001
    cov_daily = 0.8 * cov_daily + 0.2 * np.diag(np.diag(cov_daily))

    rf_daily = 0.045 / 252

    # Optimize
    if args.mode == "equal20" or (args.tickers and not args.mode):
        weights = np.ones(n_cand) / n_cand
        method_label = f"Equal Weight ({n_cand} positions)"
    elif args.mode == "equal100":
        weights = np.ones(n_cand) / n_cand
        method_label = f"Equal Weight Top Quintile ({n_cand} positions)"
    elif args.mode == "tvol":
        tv_daily = args.vol / 100 / np.sqrt(252)
        weights = optimize_target_vol(mu_daily, cov_daily, tv_daily, args.max_weight)
        method_label = f"Target Vol {args.vol:.0f}% ({n_cand} candidates)"
    elif args.mode == "maxsharpe":
        weights = optimize_max_sharpe(mu_daily, cov_daily, rf_daily, args.max_weight)
        method_label = f"Max Sharpe ({n_cand} candidates)"
    elif args.mode == "minvar":
        weights = optimize_min_var(cov_daily, args.max_weight)
        method_label = f"Min Variance ({n_cand} candidates)"
    else:
        weights = np.ones(n_cand) / n_cand
        method_label = "Equal Weight"

    # ── Portfolio metrics ─────────────────────────────────────
    sel_mask = weights > 0.001
    sel_w = weights[sel_mask]; sel_w = sel_w / sel_w.sum()
    sel_rets = np.nan_to_num(hist[:, sel_mask], nan=0)
    port_daily = sel_rets @ sel_w
    port_vol = float(np.std(port_daily) * np.sqrt(252))
    port_ret_ann = float(np.mean(port_daily) * 252)
    port_sharpe = (port_ret_ann - 0.045) / port_vol if port_vol > 0 else 0
    n_positions = int(sel_mask.sum())

    # Avg correlation
    if n_positions > 1:
        corr_m = np.corrcoef(sel_rets.T)
        avg_corr = float((corr_m.sum() - n_positions) / (n_positions * (n_positions - 1)))
    else:
        avg_corr = 1.0

    # Sector allocation
    sector_alloc = {}
    for idx in np.where(sel_mask)[0]:
        t = tickers[candidates[idx]]
        sec = sectors_map.get(t, "?")
        sector_alloc[sec] = sector_alloc.get(sec, 0) + weights[idx]

    # ── Build positions list ──────────────────────────────────
    ret_1m = prices.pct_change(21).iloc[-1]
    ret_3m = prices.pct_change(63).iloc[-1]
    positions = []

    for idx in np.argsort(-weights):
        if weights[idx] < 0.001: continue
        t = tickers[candidates[idx]]
        t_vol = np.std(hist[:, idx]) * np.sqrt(252)
        rank_pos = int(np.where(sorted_valid == candidates[idx])[0][0]) + 1
        r1 = ret_1m.get(t, np.nan)
        r3 = ret_3m.get(t, np.nan)
        positions.append({
            "ticker": t, "weight": float(weights[idx]),
            "score": float(scores[candidates[idx]]),
            "vol": float(t_vol), "rank": rank_pos,
            "sector": sectors_map.get(t, "?"),
            "company": profiles_map.get(t, {}).get("name", ""),
            "ret_1m": float(r1) if np.isfinite(r1) else None,
            "ret_3m": float(r3) if np.isfinite(r3) else None,
        })

    # ── Compute efficient frontier ──────────────────────────────
    frontier_vols = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    frontier_points = []
    for fv in frontier_vols:
        fv_d = fv / np.sqrt(252)
        fw = optimize_target_vol(mu_daily, cov_daily, fv_d, args.max_weight)
        f_daily = np.nan_to_num(hist, nan=0) @ fw
        rv = float(np.std(f_daily) * np.sqrt(252))
        rr = float(np.mean(f_daily) * 252)
        nf = int((fw > 0.001).sum())
        sf = (rr - 0.045) / rv if rv > 0 else 0
        frontier_points.append({"target_vol": fv, "realized_vol": rv, "realized_ret": rr, "sharpe": sf, "n_pos": nf})

    ms_w = optimize_max_sharpe(mu_daily, cov_daily, rf_daily, args.max_weight)
    ms_daily = np.nan_to_num(hist, nan=0) @ ms_w
    ms_vol = float(np.std(ms_daily) * np.sqrt(252))
    ms_ret = float(np.mean(ms_daily) * 252)
    ms_sharpe = (ms_ret - 0.045) / ms_vol if ms_vol > 0 else 0

    mv_w = optimize_min_var(cov_daily, args.max_weight)
    mv_daily = np.nan_to_num(hist, nan=0) @ mv_w
    mv_vol = float(np.std(mv_daily) * np.sqrt(252))
    mv_ret = float(np.mean(mv_daily) * 252)

    individual = []
    for idx in range(n_cand):
        t = tickers[candidates[idx]]
        t_ret = float(np.mean(hist[:, idx]) * 252)
        t_vol = float(np.std(hist[:, idx]) * np.sqrt(252))
        individual.append({"ticker": t, "ret": t_ret, "vol": t_vol,
                          "selected": bool(sel_mask[idx]), "score": float(scores[candidates[idx]])})

    # ── Display ───────────────────────────────────────────────
    print(f"\n{'=' * 85}")
    print(f"  PORTFOLIO — {today}")
    print(f"  Method: {method_label}")
    print(f"  Max weight: {args.max_weight:.0%}")
    print(f"{'=' * 85}")

    print(f"\n  PORTFOLIO CHARACTERISTICS:")
    print(f"    Positions:          {n_positions}")
    print(f"    Expected return:    {port_ret_ann*100:+.1f}% (based on last 252 days)")
    print(f"    Portfolio vol:      {port_vol*100:.1f}%")
    print(f"    Sharpe ratio:       {port_sharpe:.2f}")
    print(f"    Avg pairwise corr:  {avg_corr:.2f}")
    print(f"    Sectors:            {len(sector_alloc)}")

    # Backtest reference
    refs = {
        "equal20": "+46%/yr, Sharpe 1.22, MaxDD -33%",
        "equal100": "+29%/yr, Sharpe 1.14, MaxDD -24%",
        "tvol": f"+{10 + args.vol * 0.85:.0f}%/yr est. (varies with target vol)",
        "maxsharpe": "+110%/yr, Sharpe 1.78, MaxDD -37% (aggressive)",
        "minvar": "+14%/yr, Sharpe 0.68, MaxDD -17% (conservative)",
    }
    print(f"\n  BACKTEST REFERENCE (2023-2026, tech+val signals):")
    print(f"    {refs.get(args.mode, 'N/A')}")

    print(f"\n  POSITIONS:")
    print(f"  {'Ticker':>8s} {'Weight':>7s} {'Score':>7s} {'Rank':>5s} {'Vol':>5s} "
          f"{'1m':>7s} {'3m':>7s} {'Sector':>20s}  Company")
    print(f"  {'-' * 85}")

    for p in positions:
        r1 = f"{p['ret_1m']:+.1%}" if p["ret_1m"] is not None else "N/A"
        r3 = f"{p['ret_3m']:+.1%}" if p["ret_3m"] is not None else "N/A"
        print(f"  {p['ticker']:>8s} {p['weight']*100:6.1f}% {p['score']:+.3f} "
              f"#{p['rank']:>3d} {p['vol']*100:4.0f}% {r1:>7s} {r3:>7s} "
              f"{p['sector'][:20]:>20s}  {p['company'][:25]}")

    # Not selected (for quintile modes)
    if n_cand > n_positions and n_cand > 20:
        not_sel = [(tickers[candidates[i]], scores[candidates[i]],
                    np.std(hist[:, i]) * np.sqrt(252))
                   for i in range(n_cand) if not sel_mask[i]]
        not_sel.sort(key=lambda x: -x[1])
        if not_sel:
            print(f"\n  NOT SELECTED (top 10 by score):")
            print(f"  {'Ticker':>8s} {'Score':>7s} {'Vol':>5s} {'Sector':>20s}")
            for t, sc, vol in not_sel[:10]:
                print(f"  {t:>8s} {sc:+.3f} {vol*100:4.0f}% {sectors_map.get(t, '?')[:20]:>20s}")

    print(f"\n  SECTOR ALLOCATION:")
    for sec, sw in sorted(sector_alloc.items(), key=lambda x: -x[1]):
        bar = "█" * int(sw * 40)
        print(f"    {sec:25s} {sw*100:5.1f}% {bar}")

    # Frontier table
    print(f"\n  EFFICIENT FRONTIER:")
    print(f"    {'Target':>8s} {'Real Vol':>8s} {'E[r]':>8s} {'Sharpe':>8s} {'#Pos':>5s}")
    print(f"    {'-' * 42}")
    print(f"    {'MinVar':>8s} {mv_vol*100:7.1f}% {mv_ret*100:+7.1f}% {(mv_ret-0.045)/mv_vol if mv_vol>0 else 0:7.2f} {'~10':>5s}")
    for fp in frontier_points:
        marker = " ← YOU" if abs(fp["realized_vol"] - port_vol) < 0.03 else ""
        print(f"    {fp['target_vol']*100:7.0f}% {fp['realized_vol']*100:7.1f}% {fp['realized_ret']*100:+7.1f}% "
              f"{fp['sharpe']:7.2f} {fp['n_pos']:>5d}{marker}")
    print(f"    {'MaxSh':>8s} {ms_vol*100:7.1f}% {ms_ret*100:+7.1f}% {ms_sharpe:7.2f} {'~10':>5s}")

    # ── Chart: Efficient Frontier ─────────────────────────────
    if not args.no_chart:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(13, 7.5))

        # Individual stocks
        for d in individual:
            if d["selected"]:
                ax.scatter(d["vol"] * 100, d["ret"] * 100, s=60, color="#2C5F9E",
                          alpha=0.9, zorder=5, edgecolors="white", linewidth=0.5)
                ax.annotate(d["ticker"], (d["vol"] * 100, d["ret"] * 100),
                           fontsize=7, ha="center", va="bottom",
                           xytext=(0, 6), textcoords="offset points",
                           color="#1B3A6B", fontweight="bold")
            else:
                ax.scatter(d["vol"] * 100, d["ret"] * 100, s=12, color="#C0C0C0",
                          alpha=0.25, zorder=2)

        # Frontier curve
        f_vols = [p["realized_vol"] * 100 for p in frontier_points]
        f_rets = [p["realized_ret"] * 100 for p in frontier_points]
        sorted_f = sorted(zip(f_vols, f_rets))
        ax.plot([x[0] for x in sorted_f], [x[1] for x in sorted_f],
               "-", color="#2C5F9E", linewidth=2.5, alpha=0.7, label="Efficient Frontier")
        ax.fill_between([x[0] for x in sorted_f], [x[1] for x in sorted_f],
                       alpha=0.05, color="#2C5F9E")

        # Selected portfolio point
        ax.scatter(port_vol * 100, port_ret_ann * 100, s=200, color="#E74C3C", marker="D",
                  zorder=10, edgecolors="white", linewidth=2,
                  label=f"Selected (σ={port_vol*100:.1f}%, E[r]={port_ret_ann*100:.1f}%)")

        # Max Sharpe point
        ax.scatter(ms_vol * 100, ms_ret * 100, s=120, color="#F39C12", marker="*",
                  zorder=9, edgecolors="white", linewidth=1,
                  label=f"Max Sharpe (σ={ms_vol*100:.1f}%, E[r]={ms_ret*100:.1f}%)")

        # Min Var point
        ax.scatter(mv_vol * 100, mv_ret * 100, s=80, color="#27AE60", marker="s",
                  zorder=9, edgecolors="white", linewidth=1,
                  label=f"Min Var (σ={mv_vol*100:.1f}%, E[r]={mv_ret*100:.1f}%)")

        # Risk-free rate
        ax.axhline(y=4.5, color="#9A9A9A", linestyle="--", alpha=0.4, label="Rf=4.5%")

        ax.set_xlabel("Annualized Volatility (%)", fontsize=11)
        ax.set_ylabel("Expected Annual Return (%)", fontsize=11)
        ax.set_title(f"MM Quant Capital — Efficient Frontier ({today})\n{method_label}",
                    fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.15)

        stats_text = (f"Positions: {n_positions}\nPortfolio σ: {port_vol*100:.1f}%\n"
                     f"E[r]: {port_ret_ann*100:.1f}%\nSharpe: {port_sharpe:.2f}\n"
                     f"Avg corr: {avg_corr:.2f}\nSectors: {len(sector_alloc)}")
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=8.5,
               va="bottom", ha="right",
               bbox=dict(boxstyle="round,pad=0.5", facecolor="#F7F8FA", edgecolor="#E2E5EA"))

        import tempfile
        chart_path = os.path.join(tempfile.gettempdir(), f"portfolio_chart_{today}.png")
        fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

    # ── PDF Report ────────────────────────────────────────────
    if not args.no_chart:
        from src.reports.pdf_report import QuantPDF, _s, ML, MR, CW, PAGE_W, C_NAVY, \
            C_COBALT, C_SLATE, C_MID, C_MIST, C_CLOUD, C_LINE, C_GREEN, C_RED, CONTENT_TOP

        pdf = QuantPDF()
        pdf._is_cover = True
        pdf.add_page()

        # Cover
        pdf.set_font("InterSB", "", 7.5)
        pdf.set_text_color(*C_MIST)
        pdf.set_xy(ML, 18)
        pdf.cell(CW / 2, 5, "PREPARED BY MM QUANT CAPITAL", align="L")
        pdf.cell(CW / 2, 5, today.upper(), align="R")

        pdf.set_fill_color(*C_CLOUD)
        pdf.rect(0, 55, PAGE_W, 135, "F")

        pdf.set_xy(ML, 80)
        pdf.set_font("Inter", "", 44)
        pdf.set_text_color(*C_NAVY)
        pdf.cell(0, 17, "Portfolio", ln=True)
        pdf.set_x(ML)
        pdf.set_font("InterSB", "", 32)
        pdf.cell(0, 15, "Recommendation", ln=True)

        pdf.set_x(ML); pdf.set_font("InterMed", "", 10); pdf.set_text_color(*C_COBALT)
        pdf.cell(0, 9, f"Method: {method_label}", ln=True)

        pdf.set_x(ML); pdf.set_font("Inter", "", 8.5); pdf.set_text_color(*C_MID)
        pdf.cell(0, 6, f"{n_positions} positions  |  Portfolio vol: {port_vol*100:.1f}%  |  {today}", ln=True)

        pdf.set_xy(ML, 155)
        pdf.set_font("Inter", "", 7.5)
        pdf.set_text_color(*C_MIST); pdf.cell(28, 4.5, "Mode:"); pdf.set_text_color(*C_SLATE)
        pdf.cell(0, 4.5, _s(method_label), ln=True)
        pdf.set_x(ML); pdf.set_text_color(*C_MIST); pdf.cell(28, 4.5, "Positions:"); pdf.set_text_color(*C_SLATE)
        pdf.cell(0, 4.5, str(n_positions), ln=True)
        pdf.set_x(ML); pdf.set_text_color(*C_MIST); pdf.cell(28, 4.5, "Expected return:"); pdf.set_text_color(*C_SLATE)
        pdf.cell(0, 4.5, f"{port_ret_ann*100:+.1f}%", ln=True)
        pdf.set_x(ML); pdf.set_text_color(*C_MIST); pdf.cell(28, 4.5, "Portfolio vol:"); pdf.set_text_color(*C_SLATE)
        pdf.cell(0, 4.5, f"{port_vol*100:.1f}%", ln=True)
        pdf.set_x(ML); pdf.set_text_color(*C_MIST); pdf.cell(28, 4.5, "Sharpe ratio:"); pdf.set_text_color(*C_SLATE)
        pdf.cell(0, 4.5, f"{port_sharpe:.2f}", ln=True)
        pdf.set_x(ML); pdf.set_text_color(*C_MIST); pdf.cell(28, 4.5, "Avg corr:"); pdf.set_text_color(*C_SLATE)
        pdf.cell(0, 4.5, f"{avg_corr:.2f}", ln=True)

        # Content pages
        pdf._is_cover = False

        # Page 2: Chart
        if os.path.exists(chart_path):
            pdf.content_page()
            pdf.set_font("InterSB", "", 13); pdf.set_text_color(*C_NAVY); pdf.set_x(ML)
            pdf.cell(CW, 9, "SCORE VS VOLATILITY", ln=True); pdf.ln(3)
            pdf.image(chart_path, x=ML, w=CW)

        # Page 3: Positions table
        pdf.content_page()
        pdf.set_font("InterSB", "", 13); pdf.set_text_color(*C_NAVY); pdf.set_x(ML)
        pdf.cell(CW, 9, "PORTFOLIO POSITIONS", ln=True); pdf.ln(3)

        # Backtest reference
        pdf.set_font("Inter", "", 8.5); pdf.set_text_color(*C_MID); pdf.set_x(ML)
        ref = refs.get(args.mode, "N/A")
        pdf.multi_cell(CW, 4.5, _s(f"Backtest reference (2023-2026, tech+val): {ref}"))
        pdf.ln(4)

        # Table header
        cols = [("#", 8, "C"), ("Ticker", 16, "C"), ("Weight", 18, "C"),
                ("Score", 18, "C"), ("Vol", 14, "C"), ("1m", 16, "C"), ("3m", 16, "C"),
                ("Company", CW - 136, "L"), ("Sector", 30, "L")]
        pdf.set_font("InterSB", "", 7); pdf.set_text_color(*C_NAVY)
        x = ML
        for text, w, a in cols:
            pdf.set_xy(x, pdf.get_y()); pdf.cell(w, 5.5, text.upper(), align=a); x += w
        pdf.ln(5.5)
        y = pdf.get_y(); pdf.set_draw_color(*C_NAVY); pdf.set_line_width(0.4)
        pdf.line(ML, y, PAGE_W - MR, y); pdf.ln(1.5)

        for i, p in enumerate(positions):
            r1 = f"{p['ret_1m']:+.1%}" if p["ret_1m"] is not None else "N/A"
            r3 = f"{p['ret_3m']:+.1%}" if p["ret_3m"] is not None else "N/A"
            color = C_GREEN if p["score"] > 0.1 else C_RED if p["score"] < -0.05 else C_SLATE
            pdf.set_font("Inter", "", 7.5); pdf.set_text_color(*color)
            x = ML
            vals = [(str(i+1), 8, "C"), (p["ticker"], 16, "C"),
                    (f"{p['weight']*100:.1f}%", 18, "C"), (f"{p['score']:+.3f}", 18, "C"),
                    (f"{p['vol']*100:.0f}%", 14, "C"), (r1, 16, "C"), (r3, 16, "C"),
                    (p["company"][:25], CW - 136, "L"), (p["sector"][:18], 30, "L")]
            for text, w, a in vals:
                pdf.set_xy(x, pdf.get_y()); pdf.cell(w, 5, _s(str(text)), align=a); x += w
            pdf.ln(5)
            y = pdf.get_y()
            pdf.set_draw_color(*C_LINE); pdf.set_line_width(0.15 if i < len(positions)-1 else 0.3)
            pdf.line(ML, y, PAGE_W - MR, y); pdf.ln(0.8)

        # Sector allocation
        pdf.ln(6)
        pdf.set_font("InterSB", "", 13); pdf.set_text_color(*C_NAVY); pdf.set_x(ML)
        pdf.cell(CW, 9, "SECTOR ALLOCATION", ln=True); pdf.ln(3)

        for sec, sw in sorted(sector_alloc.items(), key=lambda x: -x[1]):
            pdf.set_font("Inter", "", 8.5); pdf.set_text_color(*C_SLATE); pdf.set_x(ML)
            pdf.cell(50, 5, _s(sec[:25]))
            pdf.cell(15, 5, f"{sw*100:.1f}%")
            # Bar
            bar_w = sw * 80
            y = pdf.get_y() + 1
            pdf.set_fill_color(*C_COBALT)
            pdf.rect(ML + 68, y, bar_w, 3, "F")
            pdf.ln(6)

        # Page: Efficient Frontier table
        pdf.content_page()
        pdf.set_font("InterSB", "", 13); pdf.set_text_color(*C_NAVY); pdf.set_x(ML)
        pdf.cell(CW, 9, "EFFICIENT FRONTIER", ln=True); pdf.ln(3)

        pdf.set_font("Inter", "", 8.5); pdf.set_text_color(*C_MID); pdf.set_x(ML)
        pdf.multi_cell(CW, 4.5, _s(
            "Each row shows a portfolio optimized for a different risk level. "
            "Move along the frontier to choose your risk/return trade-off. "
            "Backtest ref: EqWt Top 20 = +46%/yr Sharpe 1.22, MaxSharpe = +110%/yr Sharpe 1.78"))
        pdf.ln(4)

        # Frontier table
        f_cols = [("Target Vol", 25, "C"), ("Real Vol", 22, "C"), ("E[r]", 22, "C"),
                  ("Sharpe", 22, "C"), ("#Pos", 15, "C"), ("Note", CW - 106, "L")]
        pdf.set_font("InterSB", "", 7); pdf.set_text_color(*C_NAVY)
        x = ML
        for text, w, a in f_cols:
            pdf.set_xy(x, pdf.get_y()); pdf.cell(w, 5.5, text.upper(), align=a); x += w
        pdf.ln(5.5)
        y = pdf.get_y(); pdf.set_draw_color(*C_NAVY); pdf.set_line_width(0.4)
        pdf.line(ML, y, PAGE_W - MR, y); pdf.ln(1.5)

        # Min var row
        pdf.set_font("Inter", "", 7.5); pdf.set_text_color(*C_SLATE)
        vals_mv = [("Min Var", 25, "C"), (f"{mv_vol*100:.1f}%", 22, "C"),
                   (f"{mv_ret*100:+.1f}%", 22, "C"),
                   (f"{(mv_ret-0.045)/mv_vol:.2f}" if mv_vol > 0 else "0", 22, "C"),
                   ("~10", 15, "C"), ("Most conservative", CW - 106, "L")]
        x = ML
        for text, w, a in vals_mv:
            pdf.set_xy(x, pdf.get_y()); pdf.cell(w, 5, _s(text), align=a); x += w
        pdf.ln(5); pdf.set_draw_color(*C_LINE); pdf.set_line_width(0.15)
        pdf.line(ML, pdf.get_y(), PAGE_W - MR, pdf.get_y()); pdf.ln(0.8)

        # Frontier rows
        for fp in frontier_points:
            is_you = abs(fp["realized_vol"] - port_vol) < 0.03
            color = C_COBALT if is_you else C_SLATE
            pdf.set_text_color(*color)
            note = "YOUR PORTFOLIO" if is_you else ""
            vals_fp = [(f"{fp['target_vol']*100:.0f}%", 25, "C"),
                       (f"{fp['realized_vol']*100:.1f}%", 22, "C"),
                       (f"{fp['realized_ret']*100:+.1f}%", 22, "C"),
                       (f"{fp['sharpe']:.2f}", 22, "C"),
                       (str(fp["n_pos"]), 15, "C"), (note, CW - 106, "L")]
            x = ML
            for text, w, a in vals_fp:
                pdf.set_xy(x, pdf.get_y()); pdf.cell(w, 5, _s(text), align=a); x += w
            pdf.ln(5); pdf.set_draw_color(*C_LINE); pdf.set_line_width(0.15)
            pdf.line(ML, pdf.get_y(), PAGE_W - MR, pdf.get_y()); pdf.ln(0.8)

        # Max Sharpe row
        pdf.set_text_color(*C_SLATE)
        vals_ms = [("Max Sharpe", 25, "C"), (f"{ms_vol*100:.1f}%", 22, "C"),
                   (f"{ms_ret*100:+.1f}%", 22, "C"), (f"{ms_sharpe:.2f}", 22, "C"),
                   ("~10", 15, "C"), ("Most aggressive", CW - 106, "L")]
        x = ML
        for text, w, a in vals_ms:
            pdf.set_xy(x, pdf.get_y()); pdf.cell(w, 5, _s(text), align=a); x += w
        pdf.ln(5); pdf.set_draw_color(*(200, 204, 210)); pdf.set_line_width(0.3)
        pdf.line(ML, pdf.get_y(), PAGE_W - MR, pdf.get_y())

        # Save PDF
        pdf_path = f"reports/Portfolio_{today}.pdf"
        pdf.output(pdf_path)
        log.info("  PDF: %s (%.1f KB)", pdf_path, os.path.getsize(pdf_path) / 1024)
        print(f"  PDF: {pdf_path}")

    # ── Save to DB ────────────────────────────────────────────
    conn = sqlite3.connect("data/db/market.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rec_date TEXT NOT NULL,
            method TEXT NOT NULL,
            ticker TEXT NOT NULL,
            weight REAL NOT NULL,
            score REAL,
            rank_position INTEGER,
            vol REAL,
            sector TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(rec_date, method, ticker)
        )
    """)
    now = datetime.now().isoformat()
    for p in positions:
        conn.execute(
            """INSERT OR REPLACE INTO portfolio_recommendations
               (rec_date, method, ticker, weight, score, rank_position, vol, sector, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (today, args.mode, p["ticker"], p["weight"], p["score"],
             p["rank"], p["vol"], p["sector"], now))
    conn.commit(); conn.close()
    log.info("Portfolio saved to DB")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 85}")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'=' * 85}")


if __name__ == "__main__":
    main()
