#!/usr/bin/env python3
"""
MM Quant Capital — Backtest: GICS Calibrated + Black-Litterman Optimization

Compares portfolio construction methods on top quintile of GICS-calibrated ranking:
  A) Equal Weight (baseline)
  B) B-L max 5% per position
  C) B-L max 10% per position
  D) B-L max 15% per position
  E) B-L unconstrained

Walk-forward: 12-month train → 3-month test, monthly rebalance.
Shows detailed trades for each rebalance period.

Usage:
  python backtests/02_bl_optimization.py
  python backtests/02_bl_optimization.py --max-weights 0.05,0.10,0.15,0.20
  python backtests/02_bl_optimization.py --show-trades 12  # show last 12 months of trades
  python backtests/02_bl_optimization.py --start 2021-01-01  # longer history
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


def compute_group_weights(S, fwd_21, train_sl, sector_idx, K, reg_alpha=0.5):
    """Compute IC-weighted signal weights per GICS sector."""
    S_tr = S[train_sl]
    F_tr = fwd_21[train_sl]

    # Global IC
    global_ic = np.zeros(K)
    for k in range(K):
        ics = []
        for t in range(0, len(S_tr), 5):
            sv = S_tr[t, :, k]
            fv = F_tr[t, :]
            mask = np.isfinite(sv) & np.isfinite(fv)
            if mask.sum() >= 30:
                ic, _ = spearmanr(sv[mask], fv[mask])
                if np.isfinite(ic):
                    ics.append(ic)
        global_ic[k] = np.mean(ics) if ics else 0

    global_ic = np.maximum(global_ic, 0)
    g_total = global_ic.sum()
    global_w = global_ic / g_total if g_total > 0 else np.ones(K) / K

    group_weights = {}
    for gname, gidx in sector_idx.items():
        if len(gidx) < 10:
            group_weights[gname] = global_w.copy()
            continue
        group_ic = np.zeros(K)
        for k in range(K):
            ics = []
            for t in range(0, len(S_tr), 5):
                sv = S_tr[t, gidx, k]
                fv = F_tr[t, gidx]
                mask = np.isfinite(sv) & np.isfinite(fv)
                if mask.sum() >= 8:
                    ic, _ = spearmanr(sv[mask], fv[mask])
                    if np.isfinite(ic):
                        ics.append(ic)
            group_ic[k] = np.mean(ics) if ics else 0
        group_ic = np.maximum(group_ic, 0)
        g_t = group_ic.sum()
        group_w = group_ic / g_t if g_t > 0 else np.ones(K) / K
        blended = reg_alpha * global_w + (1 - reg_alpha) * group_w
        b_t = blended.sum()
        group_weights[gname] = blended / b_t if b_t > 0 else np.ones(K) / K

    return group_weights


def optimize_bl(scores, ret_history, top_indices, rf=0.045/252, max_weight=0.10):
    """Black-Litterman-inspired max Sharpe optimization on selected tickers."""
    n = len(top_indices)
    if n < 5:
        return np.ones(n) / n

    top_scores = scores[top_indices]
    top_scores = top_scores - top_scores.min()
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

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n

    try:
        result = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                         bounds=bounds, constraints=constraints,
                         options={"maxiter": 200, "ftol": 1e-8})
        if result.success and np.all(np.isfinite(result.x)):
            w = np.maximum(result.x, 0)
            return w / w.sum()
    except Exception:
        pass
    return np.ones(n) / n


def main():
    parser = argparse.ArgumentParser(description="B-L Optimization Backtest")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--max-weights", default="0.05,0.10,0.15",
                        help="Comma-separated max weight per position to test")
    parser.add_argument("--show-trades", type=int, default=6,
                        help="Show last N months of detailed trades")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--top-pct", type=float, default=0.20)
    parser.add_argument("--top-n", default=None,
                        help="Also test B-L on top N tickers directly (e.g., '10,15')")
    parser.add_argument("--reg-alpha", type=float, default=0.5)
    args = parser.parse_args()

    max_weight_list = [float(x) for x in args.max_weights.split(",")]
    top_n_list = [int(x) for x in args.top_n.split(",")] if args.top_n else []
    train_days = args.train_months * 21
    test_days = args.test_months * 21

    t_start = time.time()
    print("=" * 85)
    print("  MM QUANT CAPITAL — B-L Optimization Backtest")
    print(f"  Max weights tested: {[f'{w:.0%}' for w in max_weight_list]}")
    print(f"  Train: {args.train_months}m | Test: {args.test_months}m | Top: {args.top_pct:.0%}")
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

    S = np.stack(sig_list, axis=2)
    K = len(sig_names)
    fwd_21 = prices[tickers].pct_change(21).shift(-21).values
    spy_daily = returns["SPY"].values
    ret_daily = returns[tickers].values

    ticker_sectors = [sectors_map.get(t, "Unknown") for t in tickers]
    sector_idx = {}
    for i, s in enumerate(ticker_sectors):
        sector_idx.setdefault(s, []).append(i)

    print(f"  {K} signals")

    # ── Walk-forward ──────────────────────────────────────────
    print("[3/3] Walk-forward backtest...")

    model_names = ["equal"] + [f"bl_{int(w*100)}pct" for w in max_weight_list]
    # Add top-N models: B-L optimized on ONLY the top N tickers (not top quintile)
    for tn in top_n_list:
        model_names.append(f"top{tn}_eq")
        model_names.append(f"top{tn}_bl")
    model_names.append("spy")
    results = {m: [] for m in model_names}
    all_trades = {m: [] for m in model_names if m != "spy"}

    i = train_days
    period = 0

    while i + 21 <= len(prices):
        period += 1
        train_sl = slice(max(0, i - train_days), i)
        test_end = min(i + test_days, len(prices))

        gics_weights = compute_group_weights(S, fwd_21, train_sl, sector_idx, K, args.reg_alpha)

        rebal_points = list(range(i, test_end, 21))

        for rb in rebal_points:
            rb_end = min(rb + 21, test_end)
            date_str = prices.index[rb].strftime("%Y-%m-%d")

            # Score all tickers
            day_signals = S[rb]
            scores = np.full(N, np.nan)
            for j in range(N):
                sector = ticker_sectors[j]
                if sector not in gics_weights:
                    continue
                w = gics_weights[sector]
                vals = day_signals[j]
                mask = np.isfinite(vals)
                if mask.sum() >= 5:
                    scores[j] = np.dot(vals[mask], w[mask])

            valid = np.where(np.isfinite(scores))[0]
            if len(valid) < 20:
                for day in range(rb, rb_end):
                    if day >= len(spy_daily):
                        break
                    for m in model_names:
                        results[m].append(0)
                continue

            sorted_valid = valid[np.argsort(-scores[valid])]
            n_top = max(1, int(len(sorted_valid) * args.top_pct))
            top_idx = sorted_valid[:n_top]

            # Compute weights for each model
            eq_w = np.ones(len(top_idx)) / len(top_idx)
            train_rets = ret_daily[max(0, rb - 252):rb]

            bl_weights_dict = {}
            for mw in max_weight_list:
                bl_w = optimize_bl(scores, train_rets, top_idx, max_weight=mw)
                bl_weights_dict[f"bl_{int(mw*100)}pct"] = bl_w

            # Top-N models: optimize on ONLY the top N tickers
            topn_weights = {}
            for tn in top_n_list:
                topn_idx = sorted_valid[:tn]  # just the top N, not top quintile
                # Equal weight on top N
                topn_eq = np.ones(len(topn_idx)) / len(topn_idx)
                topn_weights[f"top{tn}_eq"] = (topn_idx, topn_eq)
                # B-L on top N (unconstrained since N is small)
                topn_bl = optimize_bl(scores, train_rets, topn_idx, max_weight=1.0/max(tn//3, 1))
                topn_weights[f"top{tn}_bl"] = (topn_idx, topn_bl)

            # Log trades
            for model_name, w in [("equal", eq_w)] + list(bl_weights_dict.items()):
                top10 = np.argsort(-w)[:10]
                trades = []
                for idx in top10:
                    if w[idx] > 0.001:
                        t_name = tickers[top_idx[idx]]
                        end_rb = min(rb + 21, len(prices))
                        if end_rb < len(prices):
                            act_ret = prices[t_name].iloc[end_rb] / prices[t_name].iloc[rb] - 1
                        else:
                            act_ret = np.nan
                        trades.append({
                            "ticker": t_name, "weight": w[idx],
                            "score": scores[top_idx[idx]], "ret_21d": act_ret,
                            "sector": sectors_map.get(t_name, "?"),
                        })
                all_trades[model_name].append({"date": date_str, "positions": trades})

            # Compute daily returns
            for day in range(rb, rb_end):
                if day >= len(ret_daily) or day >= len(spy_daily):
                    break
                spy_r = spy_daily[day] if np.isfinite(spy_daily[day]) else 0

                # Equal weight on top quintile
                dr = np.nan_to_num(ret_daily[day, top_idx], nan=0)
                results["equal"].append(float(np.sum(eq_w * dr)))

                # B-L on top quintile
                for mw_name, bl_w in bl_weights_dict.items():
                    results[mw_name].append(float(np.sum(bl_w * dr)))

                # Top-N models
                for tn_name, (tn_idx, tn_w) in topn_weights.items():
                    dr_tn = np.nan_to_num(ret_daily[day, tn_idx], nan=0)
                    results[tn_name].append(float(np.sum(tn_w * dr_tn)))

                results["spy"].append(spy_r)

        if period % 3 == 0:
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
    print(f"  RESULTS — {period} periods, {n_days} trading days, {years:.1f} years")
    print(f"{'=' * 85}\n")

    print(f"  {'Model':30s} {'Total':>8s} {'Ann':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Alpha':>8s}")
    print(f"  {'-' * 68}")

    for model in model_names:
        rets = pd.Series(results[model])
        eq = (1 + rets).cumprod()
        total = eq.iloc[-1] - 1
        ann = eq.iloc[-1] ** (1 / max(years, 0.1)) - 1
        rf_daily = 0.045 / 252
        sharpe = ((rets.mean() - rf_daily) / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        dd = ((eq - eq.cummax()) / eq.cummax()).min()
        alpha = total - spy_total if model != "spy" else 0

        labels = {"equal": "GICS EqWt (top quintile)", "spy": "SPY Buy & Hold"}
        for mw in max_weight_list:
            labels[f"bl_{int(mw*100)}pct"] = f"GICS+BL quintile (max {mw:.0%})"
        for tn in top_n_list:
            labels[f"top{tn}_eq"] = f"Top {tn} Equal Weight"
            labels[f"top{tn}_bl"] = f"Top {tn} + B-L Optimized"

        print(f"  {labels.get(model, model):30s} {total*100:+7.1f}% {ann*100:+7.1f}% "
              f"{sharpe:+7.2f} {dd*100:+7.1f}% {alpha*100:+7.1f}%")

    # Year by year
    all_dates = prices.index[train_days:train_days + n_days]
    print(f"\n  Year-by-Year Alpha vs SPY:")
    header = f"  {'':>6s}"
    for m in model_names:
        if m == "spy":
            continue
        short = m.replace("bl_", "BL").replace("pct", "%").replace("equal", "EqWt")
        header += f" {short:>10s}"
    print(header)
    print(f"  {'-' * (10 + 11 * (len(model_names) - 1))}")

    for year in sorted(set(all_dates.year)):
        mask = (all_dates.year == year)[:n_days]
        if mask.sum() < 20:
            continue
        spy_yr = (1 + pd.Series(results["spy"])[mask]).prod() - 1
        row = f"  {year:6d}"
        for m in model_names:
            if m == "spy":
                continue
            yr_ret = (1 + pd.Series(results[m])[mask]).prod() - 1
            row += f" {(yr_ret - spy_yr)*100:+9.1f}%"
        print(row)

    # Detailed trades
    n_show = args.show_trades
    print(f"\n{'=' * 85}")
    print(f"  DETAILED TRADES — Last {n_show} rebalances (B-L {int(max_weight_list[1]*100 if len(max_weight_list)>1 else max_weight_list[0]*100)}%)")
    print(f"{'=' * 85}")

    trade_model = f"bl_{int(max_weight_list[1]*100 if len(max_weight_list) > 1 else max_weight_list[0]*100)}pct"
    for trade in all_trades.get(trade_model, [])[-n_show:]:
        print(f"\n  Rebalance: {trade['date']}")
        print(f"  {'Ticker':8s} {'Weight':>8s} {'Score':>8s} {'21d Ret':>8s} {'Sector':>22s}")
        print(f"  {'-' * 58}")
        for p in trade["positions"]:
            ret_str = f"{p['ret_21d']:+.1%}" if np.isfinite(p["ret_21d"]) else "N/A"
            print(f"  {p['ticker']:8s} {p['weight']*100:7.1f}% {p['score']:+.4f} "
                  f"{ret_str:>8s} {p['sector']:>22s}")

    print(f"\n{'=' * 85}")
    print(f"  Completed in {elapsed:.0f}s")
    print(f"{'=' * 85}")


if __name__ == "__main__":
    main()
