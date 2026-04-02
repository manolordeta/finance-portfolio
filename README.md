# MM Quant Capital

Quantitative investment system for S&P 500 screening, portfolio optimization, and forward testing.

**What it does:**
- Ranks ~520 US stocks daily using 13 GICS-calibrated signals (technical + valuation)
- Optimizes portfolios via efficient frontier (equal weight, target vol, max Sharpe, min variance)
- Generates PDF reports with positions, frontier chart, and sector allocation
- Monitors portfolio via IBKR connection, compares positions against ranking
- Runs LLM alerts on active watchlist for event detection

**Backtest results (2023-2026, tech+val signals only, no look-ahead):**
- Equal Weight Top 20: +46%/yr, Sharpe 1.22, MaxDD -33%
- Target Vol 35-40%: +41-45%/yr, Sharpe 1.03-1.08
- Max Sharpe from 100: +110%/yr, Sharpe 1.78 (aggressive)
- SPY benchmark: +20%/yr

---

## Daily Operations

### Every evening after market close (~6:00 PM MX)

```bash
# 1. Daily ranking + alerts + PDF report (~50s, ~$0.003)
.venv/bin/python run_daily.py

# 2. Track portfolio from IBKR (requires TWS running)
.venv/bin/python run_tracker.py --ibkr
```

### When you want to invest or rebalance

```bash
# Top 20 equal weight — best risk-adjusted (Sharpe 1.22 in backtest)
.venv/bin/python run_portfolio.py --mode equal20

# Choose your risk level on the efficient frontier
.venv/bin/python run_portfolio.py --mode tvol --vol 25    # conservative
.venv/bin/python run_portfolio.py --mode tvol --vol 35    # balanced
.venv/bin/python run_portfolio.py --mode tvol --vol 40    # aggressive

# Other modes
.venv/bin/python run_portfolio.py --mode maxsharpe        # max return, high vol
.venv/bin/python run_portfolio.py --mode minvar            # safest, still beats SPY
.venv/bin/python run_portfolio.py --mode equal100          # 100 positions
.venv/bin/python run_portfolio.py --tickers MU,CIEN,LITE   # specific tickers
```

### Weekly (Sundays)

```bash
# Refresh fundamentals from FMP + batch LLM sentiment
.venv/bin/python run_daily.py --weekly
```

### Monthly (first Sunday)

```bash
# Recalibrate signal weights by GICS sector and clusters
.venv/bin/python run_daily.py --calibrate
```

---

## Backtesting

```bash
# Walk-forward: Equal vs GICS vs Clusters vs SPY
.venv/bin/python backtests/01_walkforward_models.py

# B-L optimization with different max weights
.venv/bin/python backtests/02_bl_optimization.py

# Full B-L (CAPM) vs Simplified B-L comparison
.venv/bin/python backtests/04_bl_full_vs_simple.py

# Look-ahead bias test (tech-only vs all signals)
.venv/bin/python backtests/05_lookahead_test.py

# Point-in-time fundamentals (honest backtest)
.venv/bin/python backtests/06_pit_fundamentals.py

# Fundamentals as filter vs additive vs multiplicative
.venv/bin/python backtests/07_fundamentals_as_filter.py

# Honest optimization comparison (7 methods)
.venv/bin/python backtests/08_honest_optimization.py

# Efficient frontier from 100 candidates
.venv/bin/python backtests/10_frontier_100_candidates.py
```

---

## Project Structure

```
run_daily.py          — daily pipeline: ranking + alerts + PDF
run_portfolio.py      — portfolio optimizer with efficient frontier
run_tracker.py        — IBKR portfolio tracker
run_quintile.py       — backtest replica (standalone)

config/
  signals.yaml        — master config (signals, watchlist, alerts, portfolio)
  universe.yaml       — universe definitions (S&P 500, watchlist)

src/
  data/               — FMP client, yfinance fetcher, SQLite database
  signals/            — technical (6), fundamental (8), valuation (7), sentiment
  portfolio/          — GARCH volatility, Black-Litterman optimization
  backtest/           — calibration engine, walk-forward
  screener/           — GICS-calibrated composite scoring
  alerts/             — LLM event detection for watchlist
  reports/            — PDF generation (Inter font, MM Quant Capital brand)
  validation/         — signal IC testing, baselines

backtests/            — standalone backtest scripts (01-10)
data/db/market.db     — SQLite: 520 tickers, fundamentals, rankings, tracker
```
