"""
Entry point for portfolio analysis.

Usage:
    python run_analysis.py                          # uses config/portfolio.yaml
    python run_analysis.py --config path/to/cfg.yaml
    python run_analysis.py --portfolio tech_concentrated
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from src.utils.config import load_config, get_active_portfolios
from src.data.fetcher import DataFetcher
from src.analysis.portfolio import summary_stats, correlation_matrix
from src.analysis.risk import compute_risk_table
from src.analysis.optimization import run_optimization, optimization_summary
from src.utils.reporting import save_run

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Portfolio quant analysis")
    p.add_argument("--config", default="config/portfolio.yaml")
    p.add_argument("--portfolio", default=None, help="Override active_portfolios with a single name")
    p.add_argument("--weights", default="equal", help="'equal' or comma-separated floats summing to 1")
    return p.parse_args()


def resolve_weights(tickers: list[str], weights_arg: str) -> np.ndarray:
    if weights_arg == "equal":
        n = len(tickers)
        return np.ones(n) / n

    w = np.array([float(x) for x in weights_arg.split(",")])
    if len(w) != len(tickers):
        raise ValueError(f"Expected {len(tickers)} weights, got {len(w)}")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1, got {w.sum():.4f}")
    return w


def run_portfolio(name: str, pcfg: dict, cfg: dict, weights_arg: str) -> None:
    console.rule(f"[bold blue]Portfolio: {name}")

    # --- Fetch data -------------------------------------------------------
    fetcher = DataFetcher(
        cache_dir=cfg["data"]["cache_dir"],
        expiry_days=cfg["data"]["cache_expiry_days"],
        frequency=cfg["data"]["frequency"],
    )
    prices = fetcher.get_prices(pcfg["tickers"], pcfg["start_date"], pcfg["end_date"])
    returns = fetcher.get_returns(prices)

    tickers = list(returns.columns)
    weights = resolve_weights(tickers, weights_arg)
    rfr = cfg["optimization"]["risk_free_rate"]

    # --- Stats ------------------------------------------------------------
    stats = summary_stats(returns, weights, risk_free_rate=rfr)
    console.print("\n[bold]Portfolio & Asset Statistics[/bold]")
    _print_df(stats.round(4))

    # --- Risk -------------------------------------------------------------
    risk_cfg = cfg["risk"]
    risk_table = compute_risk_table(
        returns,
        weights,
        confidence_levels=risk_cfg["var_confidence"],
        horizons=risk_cfg["horizon_days"],
        methods=risk_cfg["var_methods"],
        n_simulations=risk_cfg["mc_simulations"],
    )
    console.print("\n[bold]VaR / CVaR Table[/bold]")
    _print_df(risk_table.round(4))

    # --- Correlation ------------------------------------------------------
    corr = correlation_matrix(returns)
    console.print("\n[bold]Correlation Matrix[/bold]")
    _print_df(corr.round(3))

    # --- Optimization -----------------------------------------------------
    opt_cfg = cfg["optimization"]
    opt_results = run_optimization(
        returns,
        objectives=opt_cfg["objectives"],
        weight_bounds=tuple(opt_cfg["weight_bounds"]),
        risk_free_rate=rfr,
    )
    opt_summary = optimization_summary(opt_results)
    console.print("\n[bold]Optimal Portfolios[/bold]")
    _print_df(opt_summary.round(4))

    # --- Save -------------------------------------------------------------
    weights_dict = dict(zip(tickers, weights.tolist()))
    out_path = save_run(
        portfolio_name=name,
        stats=stats,
        risk_table=risk_table,
        opt_summary=opt_summary,
        weights=weights_dict,
        cfg=cfg,
        reports_dir=cfg["output"]["reports_dir"],
        historical_dir=cfg["output"]["historical_dir"],
    )
    console.print(f"\n[green]Results saved → {out_path}[/green]")


def _print_df(df) -> None:
    """Render a DataFrame as a Rich table (handles MultiIndex columns)."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("", style="bold")
    for col in df.columns:
        table.add_column(str(col), justify="right")
    for idx, row in df.iterrows():
        idx_str = " | ".join(str(i) for i in idx) if isinstance(idx, tuple) else str(idx)
        table.add_row(idx_str, *[str(v) for v in row])
    console.print(table)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.portfolio:
        cfg["active_portfolios"] = [args.portfolio]

    portfolios = get_active_portfolios(cfg)
    if not portfolios:
        logger.error("No portfolios to process. Check active_portfolios in config.")
        sys.exit(1)

    for name, pcfg in portfolios.items():
        run_portfolio(name, pcfg, cfg, args.weights)


if __name__ == "__main__":
    main()
