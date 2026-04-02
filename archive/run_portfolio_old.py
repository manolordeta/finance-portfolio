#!/usr/bin/env python3
"""
MM Quant Capital — Portfolio Optimizer

Produces optimal portfolio weights using:
  - Current ranking (from daily run)
  - GARCH volatility estimates
  - Black-Litterman optimization
  - Efficient frontier visualization

Usage:
  python run_portfolio.py --mode watchlist                    # max Sharpe on watchlist
  python run_portfolio.py --mode discover                     # top N from ranking
  python run_portfolio.py --mode discover --target-vol 40     # target 40% vol
  python run_portfolio.py --tickers NVDA,MU,CIEN,ASTS        # specific tickers
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


def generate_portfolio_pdf(
    result, frontier, max_sharpe_result, garch_results, ranking_positions,
    sectors, profiles, target_tickers, today, output_path,
    frontier_unconstrained=None, config=None,
):
    """Generate PDF report with efficient frontier chart and portfolio details."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from fpdf import FPDF
    import tempfile

    FONT_DIR = "assets/fonts"
    NAVY = (27, 58, 107)
    COBALT = (44, 95, 158)
    SLATE = (61, 61, 61)
    MID = (107, 114, 128)
    GREEN = (34, 120, 60)
    RED = (180, 40, 40)

    # ── Generate efficient frontier plot ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor("white")

    # Unconstrained frontier (dashed, shows theoretical max)
    if frontier_unconstrained:
        vols_u = [p["vol"] * 100 for p in frontier_unconstrained]
        rets_u = [p["ret"] * 100 for p in frontier_unconstrained]
        ax.plot(vols_u, rets_u, "--", color="#9A9A9A", linewidth=1.5,
                label="Unconstrained", alpha=0.7)

    # Constrained frontier (solid, what you can actually achieve)
    if frontier:
        vols = [p["vol"] * 100 for p in frontier]
        rets = [p["ret"] * 100 for p in frontier]
        ax.plot(vols, rets, "-", color="#2C5F9E", linewidth=2.5, label="Constrained Frontier")
        ax.fill_between(vols, rets, alpha=0.08, color="#2C5F9E")

    # Max Sharpe point
    if max_sharpe_result and max_sharpe_result.portfolio_vol > 0:
        ax.scatter(
            [max_sharpe_result.portfolio_vol * 100],
            [max_sharpe_result.portfolio_return * 100],
            s=120, color="#1B3A6B", zorder=5, marker="*",
            label=f"Max Sharpe (σ={max_sharpe_result.portfolio_vol*100:.1f}%)",
        )

    # Selected portfolio point
    if result.portfolio_vol > 0:
        ax.scatter(
            [result.portfolio_vol * 100], [result.portfolio_return * 100],
            s=150, color="#E74C3C", zorder=6, marker="D",
            label=f"Selected (σ={result.portfolio_vol*100:.1f}%)",
        )

    # Individual stocks
    for t in target_tickers:
        gr = garch_results.get(t)
        er = result.expected_returns.get(t, 0)
        if gr and er:
            ax.scatter([gr.forecast_vol_21d * 100], [er * 100],
                       s=30, color="#9A9A9A", alpha=0.6, zorder=3)
            ax.annotate(t, (gr.forecast_vol_21d * 100, er * 100),
                        fontsize=7, color="#6B7280", ha="center", va="bottom")

    # Risk-free rate line
    rf = 4.5
    if frontier:
        ax.axhline(y=rf, color="#9A9A9A", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(min(vols) - 1, rf + 0.3, f"Rf={rf:.1f}%", fontsize=8, color="#9A9A9A")

    ax.set_xlabel("Annualized Volatility (%)", fontsize=11, color="#3D3D3D")
    ax.set_ylabel("Expected Annual Return (%)", fontsize=11, color="#3D3D3D")
    ax.set_title("Efficient Frontier — Black-Litterman + GARCH", fontsize=13,
                 fontweight="bold", color="#1B3A6B", pad=15)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    chart_path = tempfile.mktemp(suffix=".png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── Build PDF ──
    pdf = FPDF()

    # Register Inter font
    if os.path.exists(f"{FONT_DIR}/Inter-Regular.ttf"):
        pdf.add_font("Inter", "", f"{FONT_DIR}/Inter-Regular.ttf", uni=True)
        pdf.add_font("Inter", "B", f"{FONT_DIR}/Inter-Bold.ttf", uni=True)
        font = "Inter"
    else:
        font = "Helvetica"

    # ── Cover page ──
    pdf.add_page()
    pdf.set_fill_color(240, 243, 248)
    pdf.rect(0, 0, 210, 297, "F")

    pdf.set_y(50)
    pdf.set_font(font, "B", 32)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 15, "Portfolio", ln=True, align="L")
    pdf.cell(0, 15, "Optimization", ln=True, align="L")

    pdf.set_y(90)
    pdf.set_font(font, "", 14)
    pdf.set_text_color(*COBALT)
    pdf.cell(0, 8, f"MM QUANT CAPITAL  |  {today}", ln=True)

    pdf.set_y(110)
    pdf.set_font(font, "", 11)
    pdf.set_text_color(*SLATE)
    method = result.method.replace("_", " ").title()
    pdf.cell(0, 7, f"Method: Black-Litterman + GARCH | {method}", ln=True)
    pdf.cell(0, 7, f"Positions: {len(result.weights)} | Universe: {len(target_tickers)} candidates", ln=True)
    pdf.cell(0, 7, f"E[r]: {result.portfolio_return:+.1%} | Vol: {result.portfolio_vol:.1%} | Sharpe: {result.portfolio_sharpe:.2f}", ln=True)

    # ── Efficient Frontier page ──
    pdf.add_page()

    # Header
    pdf.set_font(font, "B", 9)
    pdf.set_text_color(*NAVY)
    pdf.cell(140, 6, "MM QUANT CAPITAL", ln=False)
    pdf.set_font(font, "", 9)
    pdf.set_text_color(*MID)
    pdf.cell(0, 6, today, ln=True, align="R")
    pdf.set_draw_color(*COBALT)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    pdf.set_font(font, "B", 16)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 10, "EFFICIENT FRONTIER", ln=True)
    pdf.ln(4)

    # Chart
    pdf.image(chart_path, x=10, w=190)
    pdf.ln(6)

    # Key metrics box
    pdf.set_font(font, "", 9)
    pdf.set_text_color(*SLATE)
    pdf.cell(63, 6, f"Expected Return: {result.portfolio_return:+.1%}", ln=False)
    pdf.cell(63, 6, f"Volatility: {result.portfolio_vol:.1%}", ln=False)
    pdf.cell(0, 6, f"Sharpe: {result.portfolio_sharpe:.2f}", ln=True)

    # ── Portfolio Weights page ──
    pdf.add_page()
    pdf.set_font(font, "B", 9)
    pdf.set_text_color(*NAVY)
    pdf.cell(140, 6, "MM QUANT CAPITAL", ln=False)
    pdf.set_font(font, "", 9)
    pdf.set_text_color(*MID)
    pdf.cell(0, 6, today, ln=True, align="R")
    pdf.set_draw_color(*COBALT)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    pdf.set_font(font, "B", 16)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 10, "PORTFOLIO WEIGHTS", ln=True)
    pdf.ln(4)

    # Weights table
    col_widths = [22, 18, 18, 22, 18, 14, 48, 30]
    headers = ["TICKER", "WEIGHT", "E[r]", "VOL", "RISK%", "RANK", "COMPANY", "SECTOR"]

    # Header row
    pdf.set_font(font, "B", 8)
    pdf.set_text_color(*NAVY)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, ln=False, align="C")
    pdf.ln()
    pdf.set_draw_color(*NAVY)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    # Data rows
    sorted_weights = sorted(result.weights.items(), key=lambda x: -x[1])
    pdf.set_font(font, "", 8)

    for ticker, weight in sorted_weights:
        er = result.expected_returns.get(ticker, 0)
        gr = garch_results.get(ticker)
        vol = gr.forecast_vol_21d if gr else 0
        risk_pct = result.risk_contribution.get(ticker, 0)
        rank = ranking_positions.get(ticker, "?")
        company = profiles.get(ticker, {}).get("name", "")[:25]
        sector = sectors.get(ticker, "")[:15]

        pdf.set_text_color(*SLATE)
        pdf.cell(col_widths[0], 7, ticker, ln=False, align="C")

        # Weight in green
        pdf.set_text_color(*GREEN)
        pdf.cell(col_widths[1], 7, f"{weight:.1%}", ln=False, align="C")

        pdf.set_text_color(*SLATE)
        pdf.cell(col_widths[2], 7, f"{er:+.1%}", ln=False, align="C")
        pdf.cell(col_widths[3], 7, f"{vol:.1%}", ln=False, align="C")
        pdf.cell(col_widths[4], 7, f"{risk_pct:.1%}", ln=False, align="C")
        pdf.cell(col_widths[5], 7, str(rank), ln=False, align="C")
        pdf.cell(col_widths[6], 7, company, ln=False, align="L")
        pdf.cell(col_widths[7], 7, sector, ln=False, align="L")
        pdf.ln()

        # Hairline
        pdf.set_draw_color(220, 220, 220)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())

    # Excluded tickers
    excluded = [t for t in target_tickers if t not in result.weights]
    if excluded:
        pdf.ln(6)
        pdf.set_font(font, "B", 10)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 8, "EXCLUDED TICKERS", ln=True)
        pdf.set_font(font, "", 8)
        for t in excluded:
            gr = garch_results.get(t)
            vol = gr.forecast_vol_21d if gr else 0
            rank = ranking_positions.get(t, "?")
            pdf.set_text_color(*RED)
            pdf.cell(0, 6, f"  {t}  rank={rank}  vol={vol:.0%}  — excluded by optimizer", ln=True)

    # Sector allocation
    pdf.ln(4)
    pdf.set_font(font, "B", 10)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 8, "SECTOR ALLOCATION", ln=True)
    pdf.set_font(font, "", 9)

    sector_weights = {}
    for ticker, weight in result.weights.items():
        s = sectors.get(ticker, "Unknown")
        sector_weights[s] = sector_weights.get(s, 0) + weight

    for s, w in sorted(sector_weights.items(), key=lambda x: -x[1]):
        bar_width = w * 120
        pdf.set_text_color(*SLATE)
        pdf.cell(50, 6, s, ln=False)
        pdf.cell(15, 6, f"{w:.1%}", ln=False)
        # Draw bar
        y = pdf.get_y() + 1
        pdf.set_fill_color(*COBALT)
        pdf.rect(75, y, bar_width, 4, "F")
        pdf.ln(6)

    # ── GARCH Details page ──
    pdf.add_page()
    pdf.set_font(font, "B", 9)
    pdf.set_text_color(*NAVY)
    pdf.cell(140, 6, "MM QUANT CAPITAL", ln=False)
    pdf.set_font(font, "", 9)
    pdf.set_text_color(*MID)
    pdf.cell(0, 6, today, ln=True, align="R")
    pdf.set_draw_color(*COBALT)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    pdf.set_font(font, "B", 16)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 10, "VOLATILITY ANALYSIS (GARCH)", ln=True)
    pdf.ln(4)

    garch_headers = ["TICKER", "VOL NOW", "VOL 21D", "PERSIST.", "LEVERAGE", "STATUS"]
    garch_widths = [25, 25, 25, 25, 25, 40]

    pdf.set_font(font, "B", 8)
    pdf.set_text_color(*NAVY)
    for i, h in enumerate(garch_headers):
        pdf.cell(garch_widths[i], 8, h, ln=False, align="C")
    pdf.ln()
    pdf.set_draw_color(*NAVY)
    pdf.line(10, pdf.get_y(), 175, pdf.get_y())
    pdf.ln(2)

    pdf.set_font(font, "", 8)
    for t in target_tickers:
        gr = garch_results.get(t)
        if not gr:
            continue
        pdf.set_text_color(*SLATE)
        pdf.cell(garch_widths[0], 7, t, ln=False, align="C")
        pdf.cell(garch_widths[1], 7, f"{gr.current_vol:.1%}", ln=False, align="C")
        pdf.cell(garch_widths[2], 7, f"{gr.forecast_vol_21d:.1%}", ln=False, align="C")
        pdf.cell(garch_widths[3], 7, f"{gr.persistence:.3f}", ln=False, align="C")

        lev_color = GREEN if gr.gamma > 0.01 else MID
        pdf.set_text_color(*lev_color)
        pdf.cell(garch_widths[4], 7, "Yes" if gr.gamma > 0.01 else "No", ln=False, align="C")

        status_color = GREEN if gr.success else RED
        pdf.set_text_color(*status_color)
        pdf.cell(garch_widths[5], 7, "GARCH" if gr.success else "Historical fallback", ln=False, align="C")
        pdf.ln()

        pdf.set_draw_color(220, 220, 220)
        pdf.line(10, pdf.get_y(), 175, pdf.get_y())

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pdf.output(output_path)

    # Cleanup
    if os.path.exists(chart_path):
        os.remove(chart_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="MM Quant Capital — Portfolio Optimizer")
    parser.add_argument("--mode", choices=["watchlist", "discover", "quintile"], default="watchlist",
                        help="watchlist=your tickers, discover=top N, quintile=top 20%% (best backtest)")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers (overrides --mode)")
    parser.add_argument("--target-vol", type=float, default=None,
                        help="Target annualized volatility %% (overrides config)")
    parser.add_argument("--max-positions", type=int, default=None,
                        help="Max positions (overrides config)")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF generation")
    args = parser.parse_args()

    # Load config
    import yaml
    with open("config/signals.yaml") as f:
        cfg = yaml.safe_load(f)

    po = cfg.get("portfolio", {})
    global_cfg = cfg.get("global", {})

    # Aliases for backward compat
    bl_cfg = po  # B-L params are now flat in portfolio section
    ef_cfg = po  # frontier params too
    garch_cfg = po

    # Apply config with CLI overrides
    max_weight = po.get("max_weight", 0.10)
    max_sector = po.get("max_sector_weight", 0.35)
    risk_free_rate = global_cfg.get("risk_free_rate", 0.045)
    max_positions = args.max_positions or po.get("max_positions", 15)
    target_vol_pct = args.target_vol or (
        po.get("target_volatility_pct") if po.get("default_method") == "target_vol" else None
    )

    t_start = time.time()
    today = datetime.now().strftime("%Y-%m-%d")
    mode_label = args.tickers or args.mode
    if args.target_vol:
        mode_label += f" @{args.target_vol:.0f}% vol"

    log.info("=" * 65)
    log.info("  MM QUANT CAPITAL — Portfolio Optimizer")
    log.info("  Date: %s | Mode: %s", today, mode_label)
    log.info("=" * 65)

    import numpy as np
    import pandas as pd
    import yaml
    import yfinance as yf

    from src.data.database import MarketDB
    from src.portfolio.garch import fit_universe, build_covariance_matrix
    from src.portfolio.black_litterman import (
        black_litterman, optimize_weights, optimize_target_vol,
        compute_efficient_frontier, scores_to_views,
    )

    db = MarketDB("data/db/market.db")

    # ── Step 1: Determine tickers ────────────────────────────────
    log.info("[1/5] Selecting tickers...")

    conn = sqlite3.connect("data/db/market.db")
    sectors = {r[0]: r[1] for r in conn.execute(
        "SELECT ticker, sector FROM profiles").fetchall()}
    profiles = {r[0]: {"name": r[1], "sector": r[2]} for r in conn.execute(
        "SELECT ticker, company_name, sector FROM profiles").fetchall()}

    ranking_rows = conn.execute(
        """SELECT ticker, composite_score, rank_position
           FROM dual_rankings
           WHERE ranking_date = (SELECT MAX(ranking_date) FROM dual_rankings)
           AND model = 'gics'
           ORDER BY rank_position"""
    ).fetchall()
    ranking_scores = {r[0]: r[1] for r in ranking_rows}
    ranking_positions = {r[0]: r[2] for r in ranking_rows}
    conn.close()

    if not ranking_rows:
        log.error("No ranking found. Run: python run_daily.py first")
        return

    if args.tickers:
        target_tickers = [t.strip().upper() for t in args.tickers.split(",")]
    elif args.mode == "watchlist":
        target_tickers = cfg.get("watchlist", {}).get("active", [])
    elif args.mode == "quintile":
        # Top 20% of universe — the strategy that backtested +136%/yr
        n_quintile = max(1, len(ranking_rows) // 5)
        target_tickers = [r[0] for r in ranking_rows[:n_quintile]]
        log.info("  Quintile mode: top %d of %d tickers", n_quintile, len(ranking_rows))
    else:
        target_tickers = [r[0] for r in ranking_rows[:max_positions]]

    target_tickers = [t for t in target_tickers if t in ranking_scores]
    log.info("  Tickers: %d — %s", len(target_tickers), target_tickers)

    # ── Step 2: Prices ───────────────────────────────────────────
    log.info("[2/5] Downloading prices...")
    t0 = time.time()
    data = yf.download(target_tickers + ["SPY"], start="2023-01-01", progress=False)
    prices = data["Close"].dropna(axis=1, thresh=int(len(data) * 0.5))
    returns = prices.pct_change().dropna()
    valid_tickers = [t for t in target_tickers if t in returns.columns]
    log.info("  %d days x %d tickers (%.1fs)", len(prices), len(valid_tickers), time.time() - t0)

    # ── Step 3: GARCH ────────────────────────────────────────────
    log.info("[3/5] Fitting GARCH...")
    t0 = time.time()
    garch_results = fit_universe(returns, valid_tickers)
    log.info("  GARCH fitted in %.1fs", time.time() - t0)

    for t in valid_tickers:
        g = garch_results.get(t)
        if g:
            log.info("    %s  σ=%.1f%%  21d=%.1f%%  persist=%.3f  lev=%s%s",
                     t, g.current_vol*100, g.forecast_vol_21d*100,
                     g.persistence, "Y" if g.gamma > 0.01 else "N",
                     "" if g.success else " (fallback)")

    # ── Step 4: Optimize ─────────────────────────────────────────
    log.info("[4/5] Optimizing portfolio...")

    cov = build_covariance_matrix(returns, garch_results, valid_tickers, method="garch_adjusted")
    target_scores = {t: ranking_scores[t] for t in valid_tickers}
    views, confidence = scores_to_views(target_scores, scale=0.15)

    posterior_returns = black_litterman(
        covariance=cov, views=views, view_confidence=confidence,
        risk_free_rate=risk_free_rate,
        tau=po.get("tau", 0.05),
        risk_aversion=po.get("risk_aversion", 2.5),
    )

    # Max Sharpe (always compute as reference)
    max_sharpe_result = optimize_weights(
        expected_returns=posterior_returns, covariance=cov,
        risk_free_rate=risk_free_rate, max_weight=max_weight,
        max_sector_weight=max_sector, sectors=sectors, method="max_sharpe",
    )

    # Target vol or max Sharpe
    if target_vol_pct:
        target = target_vol_pct / 100.0
        result = optimize_target_vol(
            expected_returns=posterior_returns, covariance=cov,
            target_vol=target, risk_free_rate=risk_free_rate,
            max_weight=max_weight, max_sector_weight=max_sector,
            sectors=sectors, ranking_positions=ranking_positions,
        )
    else:
        result = max_sharpe_result

    # Efficient frontier — constrained
    n_frontier = po.get("frontier_points", 25)
    log.info("  Computing efficient frontier (constrained)...")
    frontier = compute_efficient_frontier(
        expected_returns=posterior_returns, covariance=cov,
        risk_free_rate=risk_free_rate, max_weight=max_weight,
        max_sector_weight=max_sector, sectors=sectors,
        ranking_positions=ranking_positions, n_points=n_frontier,
    )

    # Efficient frontier — unconstrained (no caps, for comparison)
    frontier_unconstrained = []
    if po.get("show_unconstrained", True):
        log.info("  Computing efficient frontier (unconstrained)...")
        frontier_unconstrained = compute_efficient_frontier(
            expected_returns=posterior_returns, covariance=cov,
            risk_free_rate=risk_free_rate, max_weight=1.0,
            max_sector_weight=1.0, sectors=None,
            ranking_positions=ranking_positions, n_points=n_frontier,
        )

    # ── Step 5: Display + PDF ────────────────────────────────────
    log.info("[5/5] Results:")
    log.info("")

    print(f"\n{'='*80}")
    print(f"  PORTFOLIO RECOMMENDATION — {today}")
    print(f"  Method: Black-Litterman + GARCH | {result.method}")
    print(f"  Constraints: max {max_weight:.0%}/pos, max {max_sector:.0%}/sector")
    print(f"{'='*80}")

    print(f"\n  METRICS:")
    print(f"    E[r] (ann.):    {result.portfolio_return:+.1%}")
    print(f"    Vol (ann.):     {result.portfolio_vol:.1%}")
    print(f"    Sharpe:         {result.portfolio_sharpe:.2f}")
    print(f"    Positions:      {len(result.weights)}")
    print(f"    Config:         max_wt={max_weight:.0%} max_sect={max_sector:.0%} tau={po.get('tau', 0.05)}")

    if target_vol_pct:
        print(f"    Target vol:     {target_vol_pct:.0f}%")
        print(f"    Max Sharpe ref: E[r]={max_sharpe_result.portfolio_return:+.1%} σ={max_sharpe_result.portfolio_vol:.1%} Sh={max_sharpe_result.portfolio_sharpe:.2f}")

    print(f"\n  {'─'*80}")
    print(f"  {'Ticker':>8s}  {'Weight':>8s}  {'E[r]':>7s}  {'σ(GARCH)':>10s}  {'Risk%':>7s}  {'Rank':>6s}  {'Sector':>20s}")
    print(f"  {'─'*80}")

    for ticker, weight in sorted(result.weights.items(), key=lambda x: -x[1]):
        er = posterior_returns.get(ticker, 0)
        gr = garch_results.get(ticker)
        vol = gr.forecast_vol_21d if gr else 0
        risk_pct = result.risk_contribution.get(ticker, 0)
        rank = ranking_positions.get(ticker, "?")
        sector = sectors.get(ticker, "?")[:20]
        bar = "█" * max(1, int(weight * 30))
        print(f"  {ticker:>8s}  {weight:7.1%}  {er:+6.1%}  {vol:9.1%}  {risk_pct:6.1%}  {rank:>6}  {sector:>20s}  {bar}")

    excluded = [t for t in valid_tickers if t not in result.weights]
    if excluded:
        print(f"\n  EXCLUDED:")
        for t in excluded:
            rank = ranking_positions.get(t, "?")
            gr = garch_results.get(t)
            vol = gr.forecast_vol_21d if gr else 0
            print(f"    {t:8s}  rank={rank:>5}  σ={vol:.0%}")

    # Frontier summary
    if frontier:
        print(f"\n  {'─'*80}")
        print(f"  EFFICIENT FRONTIER ({len(frontier)} points):")
        print(f"  {'Vol':>8s}  {'E[r]':>8s}  {'Sharpe':>8s}  {'#Pos':>6s}")
        for i, p in enumerate(frontier):
            if i % 5 == 0 or i == len(frontier) - 1:
                marker = " ← YOU" if abs(p["vol"] - result.portfolio_vol) < 0.01 else ""
                print(f"  {p['vol']*100:7.1f}%  {p['ret']*100:+7.1f}%  {p['sharpe']:7.2f}  {p['n_positions']:>6d}{marker}")

    # PDF
    if not args.no_pdf:
        pdf_path = f"reports/Portfolio_{today}.pdf"
        generate_portfolio_pdf(
            result=result,
            frontier=frontier,
            frontier_unconstrained=frontier_unconstrained,
            max_sharpe_result=max_sharpe_result,
            garch_results=garch_results,
            ranking_positions=ranking_positions,
            sectors=sectors,
            profiles=profiles,
            target_tickers=valid_tickers,
            today=today,
            output_path=pdf_path,
            config=po,
        )
        log.info("  PDF report: %s (%.1f KB)", pdf_path, os.path.getsize(pdf_path) / 1024)

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
