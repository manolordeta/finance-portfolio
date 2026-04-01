# Known Limitations & Biases

> Documented honestly so we don't fool ourselves.
> "In quant, half the competitive advantage is simply not self-deceiving."

---

## 1. Survivorship Bias (HIGH impact)

**What:** The backtest universe uses current S&P 500 constituents. Companies that
were delisted, went bankrupt, or were removed from the index between 2022-2026
are not included. We only test on "survivors."

**Impact:** Academic studies estimate survivorship bias inflates backtested returns
by 2-4% annually. Our +30.8%/year GICS result should be interpreted as ~+27-28%.

**Why we haven't fixed it:** FMP Growth plan does not provide historical index
composition. Obtaining point-in-time S&P 500 membership requires specialized
data providers (e.g., CRSP, Compustat).

**Mitigation:** The forward test (running since 2026-03-27) has zero survivorship
bias. In 3-6 months we'll have unbiased performance data.

**Status:** Accepted limitation. Forward test is the real validation.

---

## 2. Look-Ahead Bias in Fundamental Signals (HIGH impact on fundamental ICs)

**What:** Fundamental signals (P/E, ROE, revenue_growth, earnings_surprise, etc.)
use the MOST RECENT quarterly data for all historical dates. In January 2024,
the system "sees" Q4 2025 data that wasn't yet published.

**Impact:** ICs for fundamental signals are inflated. The true predictive power
of earnings_surprise or revenue_growth is likely lower than measured.

**Why we haven't fixed it:** Implementing point-in-time fundamentals requires
using `filing_date` to reconstruct which quarterly data was available at each
historical date. We have the data (`filing_date` is stored in the DB) but
haven't built the reconstruction logic yet.

**Important nuance:** Technical signals (momentum, MACD, volume) and valuation
signals (golden_cross, SMA ratios) use only price data and have ZERO look-ahead
bias. These are the majority of the composite weight.

**Walk-forward mitigation:** The walk-forward calibration partially mitigates this
because inflated ICs in the train period will produce poor predictions in the test
period, which the walk-forward detects. However, since fundamentals are static
throughout the backtest, this mitigation is incomplete.

**Status:** Will implement point-in-time fundamentals when time permits.
Not urgent because the forward test validates in real-time.

---

## 3. Cluster Stability in Backtest (MEDIUM impact)

**What:** In `01_walkforward_models.py`, clusters are computed once on the first
train window and kept fixed for the entire backtest. In production (`run_daily.py
--calibrate`), clusters are recomputed each calibration.

**Impact:** The backtest slightly overfits to the initial correlation structure.
Clusters that formed during 2022 bear market may not be optimal for 2024 bull
market.

**Why:** Recomputing spectral clustering for 500 tickers at each walk-forward
period is computationally expensive (~45+ minutes). Previous attempts timed out.

**Mitigation:** GICS sectors (which don't change) perform similarly to clusters
in backtest results. The cluster approach adds marginal value but the system
doesn't depend on it.

**Status:** Will optimize clustering performance when we revisit.

---

## 4. Mean-Variance Utility ≠ True Max Sharpe (LOW impact)

**What:** The portfolio optimizer maximizes `E[r] - 0.5 * δ * σ²` (mean-variance
utility with risk aversion δ=2.5), not the Sharpe ratio `(E[r] - rf) / σ`.
These are related but not identical.

**Impact:** With our constraints (max weight per position, sector caps), the
practical difference is minimal. Both produce similar portfolios.

**Why:** True max-Sharpe with constraints requires fractional programming
(Dinkelbach's algorithm), which is more complex for negligible benefit.

**Status:** Documented. Method correctly named as `mean_variance` in code.

---

## 5. Backtest Validates Equal-Weight, Production Uses B-L (MEDIUM impact)

**What:** The primary walk-forward backtest (`01_walkforward_models.py`) validates
an equal-weight top-quintile strategy (~100 positions). Production uses
Black-Litterman optimization (~10-20 concentrated positions). These are
different strategies.

**Impact:** The +30.8%/year Sharpe 1.49 result applies to equal-weight.
B-L concentrated showed +61%/year in a separate backtest but without full
walk-forward calibration rigor.

**What's needed:** A walk-forward backtest that replicates EXACTLY the production
pipeline: calibrate GICS weights → rank → B-L optimize → rebalance monthly.

**Status:** `02_bl_optimization.py` partially addresses this but doesn't
recalibrate GICS weights per period. Full integration pending.

---

## 6. LLM Sentiment Not Validated (MEDIUM impact)

**What:** The LLM sentiment signal was scored for 335 tickers but cannot be
backtested because the LLM's training data includes future market events.

**Impact:** Currently disabled in config (`enabled: false`). Zero impact on
production decisions.

**Mitigation:** Forward measurement started 2026-03-26. Will evaluate IC
after 3 months of live predictions vs realized returns.

**Status:** Disabled. Measuring forward. Will re-enable if IC > 0.03.

---

## 7. Transaction Costs Approximated (LOW impact)

**What:** Backtests use a flat 12 bps cost per trade. Real costs depend on
bid-ask spread, market impact, timing, and order size.

**Impact:** For liquid S&P 500 stocks, 12 bps is reasonable. For smaller
or more volatile names, actual costs may be higher (20-30 bps).

**Status:** Acceptable approximation for current universe.

---

## How We Validate Despite These Limitations

The most reliable validation is the **forward test** running in production:

```
Started:     2026-03-27
Tracks:      Daily returns of GICS top quintile vs SPY
No biases:   Uses only live data, no look-ahead, no survivorship
Timeline:    3 months → preliminary IC validation
             6 months → reliable performance assessment
             12 months → full track record
```

After 6+ months of forward data, the backtest limitations become irrelevant
because we'll have unbiased live performance to evaluate.
