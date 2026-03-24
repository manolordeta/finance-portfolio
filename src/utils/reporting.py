"""
Save run results to disk (CSV + JSON) with timestamps for historical tracking.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def save_run(
    portfolio_name: str,
    stats: pd.DataFrame,
    risk_table: pd.DataFrame,
    weights: dict[str, float],
    cfg: dict,
    reports_dir: str = "reports",
    historical_dir: str = "data/historical",
    opt_summary: pd.DataFrame | None = None,
) -> Path:
    """
    Save the full output of a single portfolio run.
    Returns the path of the run directory.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{portfolio_name}_{ts}"

    # Latest report (overwritten each run)
    latest_dir = Path(reports_dir) / portfolio_name
    latest_dir.mkdir(parents=True, exist_ok=True)

    # Historical snapshot (never overwritten)
    hist_dir = Path(historical_dir) / portfolio_name / ts
    hist_dir.mkdir(parents=True, exist_ok=True)

    for out_dir in [latest_dir, hist_dir]:
        stats.to_csv(out_dir / "stats.csv")
        risk_table.to_csv(out_dir / "risk.csv")
        if opt_summary is not None:
            opt_summary.to_csv(out_dir / "optimization.csv")

        meta = {
            "run_timestamp": ts,
            "portfolio_name": portfolio_name,
            "tickers": cfg["portfolios"][portfolio_name]["tickers"],
            "start_date": cfg["portfolios"][portfolio_name]["start_date"],
            "end_date": cfg["portfolios"][portfolio_name]["end_date"],
            "weights": weights,
            "risk_free_rate": cfg["optimization"]["risk_free_rate"],
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    return hist_dir
