"""Load and validate the YAML config."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import yaml


def load_config(path: str = "config/portfolio.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Resolve null end_dates to today
    for name, portfolio in cfg.get("portfolios", {}).items():
        if portfolio.get("end_date") is None:
            portfolio["end_date"] = datetime.today().strftime("%Y-%m-%d")

    return cfg


def get_active_portfolios(cfg: dict) -> dict[str, dict]:
    """Return only the portfolios to be processed in this run."""
    active = cfg.get("active_portfolios", [])
    all_portfolios = cfg.get("portfolios", {})

    if not active:
        return all_portfolios

    return {name: all_portfolios[name] for name in active if name in all_portfolios}
