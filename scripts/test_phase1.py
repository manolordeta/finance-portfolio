#!/usr/bin/env python3
"""
Phase 1 integration test.
Run: python scripts/test_phase1.py
(Reads FMP_API_KEY from .env automatically via src/__init__)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src  # noqa: F401 — triggers .env loading + dir creation
from src.data.fmp_client import FMPClient
from src.data.universe import load_universe
from src.data.database import MarketDB


def main():
    print("=" * 60)
    print("PHASE 1 — Integration Test")
    print("=" * 60)

    # ── 1. Universe ───────────────────────────────────────────
    print("\n[1] Loading universe...")
    universe = load_universe("config/universe.yaml")
    print(f"    {universe}")
    print(f"    US tickers:  {universe.tickers_us}")
    print(f"    MX tickers:  {universe.tickers_mx}")
    print(f"    Benchmark:   {universe.benchmark}")

    # ── 2. Database ───────────────────────────────────────────
    print("\n[2] Initializing database...")
    db = MarketDB("data/db/market.db")
    print("    SQLite initialized OK")

    # ── 3. FMP Client ─────────────────────────────────────────
    print("\n[3] Testing FMP connection...")
    try:
        client = FMPClient()
    except ValueError as e:
        print(f"    SKIP — {e}")
        return

    ok = client.test_connection()
    if not ok:
        print("    FAILED — check your API key")
        return
    print("    Connection OK (/stable/ endpoints)")

    # ── 4. Fetch data for each US ticker ──────────────────────
    print(f"\n[4] Fetching data for {len(universe.tickers_us)} US tickers...")

    for ticker in universe.tickers_us:
        print(f"\n  ── {ticker} ──")

        # Profile
        profile = client.get_profile(ticker)
        if profile:
            print(f"    Profile: {profile.get('companyName', '?')} | "
                  f"{profile.get('sector', '?')} | "
                  f"MCap ${profile.get('mktCap', 0) / 1e9:.1f}B")
            db.upsert_profile(ticker, profile)

        # Income statement (quarterly, last 8)
        income = client.get_income_statement(ticker, period="quarterly", limit=8)
        if income:
            for stmt in income:
                db.upsert_fundamentals(
                    ticker=ticker,
                    period_date=stmt.get("date", ""),
                    filing_date=stmt.get("fillingDate", stmt.get("date", "")),
                    statement_type="income",
                    data=stmt,
                )
            print(f"    Income:  {len(income)} quarters (latest: {income[0].get('date')})")

        # Balance sheet
        balance = client.get_balance_sheet(ticker, period="quarterly", limit=8)
        if balance:
            for stmt in balance:
                db.upsert_fundamentals(
                    ticker=ticker,
                    period_date=stmt.get("date", ""),
                    filing_date=stmt.get("fillingDate", stmt.get("date", "")),
                    statement_type="balance",
                    data=stmt,
                )
            print(f"    Balance: {len(balance)} quarters")

        # Cash flow
        cashflow = client.get_cash_flow(ticker, period="quarterly", limit=8)
        if cashflow:
            for stmt in cashflow:
                db.upsert_fundamentals(
                    ticker=ticker,
                    period_date=stmt.get("date", ""),
                    filing_date=stmt.get("fillingDate", stmt.get("date", "")),
                    statement_type="cashflow",
                    data=stmt,
                )
            print(f"    CashFlow:{len(cashflow)} quarters")

        # Ratios
        ratios = client.get_ratios(ticker, period="quarterly", limit=8)
        if ratios:
            for r in ratios:
                db.upsert_ratios(
                    ticker=ticker,
                    period_date=r.get("date", ""),
                    filing_date=r.get("date", ""),
                    data=r,
                )
            print(f"    Ratios:  {len(ratios)} quarters "
                  f"(P/E={ratios[0].get('priceEarningsRatio', 'N/A')})")

        # Key metrics
        metrics = client.get_key_metrics(ticker, period="quarterly", limit=8)
        if metrics:
            m = metrics[0]
            ev_ebitda = m.get("enterpriseValueOverEBITDA")
            fcf_y = m.get("freeCashFlowYield")
            print(f"    Metrics: EV/EBITDA={ev_ebitda}, FCFYield={fcf_y}")

        # Earnings calendar (used as surprise source)
        earnings = client.get_earnings_calendar(symbol=ticker, limit=12)
        if earnings:
            for e in earnings:
                db.upsert_earnings(ticker, {
                    "date": e.get("date"),
                    "fiscalDateEnding": e.get("date"),
                    "eps": e.get("epsActual"),
                    "epsEstimated": e.get("epsEstimated"),
                    "revenue": e.get("revenueActual"),
                    "revenueEstimated": e.get("revenueEstimated"),
                })
            # Show latest with actual data
            with_actual = [e for e in earnings if e.get("epsActual") is not None]
            if with_actual:
                lat = with_actual[0]
                surp = ""
                if lat.get("epsEstimated") and lat["epsEstimated"] != 0:
                    s = (lat["epsActual"] - lat["epsEstimated"]) / abs(lat["epsEstimated"])
                    surp = f" surprise={s:+.1%}"
                print(f"    Earnings:{len(earnings)} dates, "
                      f"latest actual: EPS={lat.get('epsActual')}{surp}")

        # Analyst estimates
        estimates = client.get_analyst_estimates(ticker, period="quarterly", limit=4)
        if estimates:
            est = estimates[0]
            print(f"    Analyst: EPS est={est.get('epsAvg')} "
                  f"({est.get('numAnalystsEps', '?')} analysts)")

    # ── 5. Verify DB reads ────────────────────────────────────
    print(f"\n\n[5] Verifying DB reads...")
    test_t = universe.tickers_us[0]

    p = db.get_profile(test_t)
    print(f"    {test_t} profile: {p['company_name'] if p else 'NOT FOUND'}")

    f = db.get_latest_fundamentals(test_t, "income")
    print(f"    {test_t} latest income: {f['period_date'] if f else 'NOT FOUND'}")

    r = db.get_latest_ratios(test_t)
    print(f"    {test_t} latest ratios: {r['period_date'] if r else 'NOT FOUND'}")

    e = db.get_earnings_history(test_t)
    print(f"    {test_t} earnings history: {len(e)} records")

    sectors = db.get_all_sectors()
    print(f"    Sectors in DB: {dict(list(sectors.items())[:5])}...")

    # ── 6. Log run ────────────────────────────────────────────
    run_id = db.log_run(
        run_type="phase1_full_load",
        universe_name=universe.name,
        tickers_count=len(universe.tickers_us),
        status="success",
    )

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print(f"  Tickers loaded:  {len(universe.tickers_us)}")
    print(f"  API calls made:  {client.calls_made}")
    print(f"  Run ID:          {run_id}")
    print(f"  DB location:     data/db/market.db")
    print("=" * 60)


if __name__ == "__main__":
    main()
