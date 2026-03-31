"""
Monthly Calibration Module

Computes signal weights by group (GICS sectors and correlation clusters),
saves calibration snapshots, and generates comparison data for the
calibration report.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Full calibration output for one model type."""
    model: str  # 'gics' or 'cluster'
    date: str
    # group_name → {signal_name: {"ic": float, "weight": float}}
    group_signals: dict[str, dict[str, dict]] = field(default_factory=dict)
    # group_name → list of tickers
    group_members: dict[str, list[str]] = field(default_factory=dict)
    # group_name → description
    group_descriptions: dict[str, str] = field(default_factory=dict)
    # Global weights for reference
    global_weights: dict[str, float] = field(default_factory=dict)
    global_ics: dict[str, float] = field(default_factory=dict)


def compute_ic(signal_df: pd.DataFrame, fwd_returns: pd.DataFrame,
               dates: pd.Index, tickers: list[str]) -> float:
    """Compute average Spearman IC over dates for given tickers."""
    valid_dates = dates.intersection(signal_df.index).intersection(fwd_returns.index)
    valid_tickers = [t for t in tickers if t in signal_df.columns and t in fwd_returns.columns]

    if len(valid_dates) < 20 or len(valid_tickers) < 10:
        return 0.0

    ics = []
    for date in valid_dates:
        s = signal_df.loc[date, valid_tickers].dropna()
        r = fwd_returns.loc[date, valid_tickers].dropna()
        common = s.index.intersection(r.index)
        if len(common) < 8:
            continue
        ic = s[common].corr(r[common], method="spearman")
        if np.isfinite(ic):
            ics.append(ic)

    return float(np.mean(ics)) if ics else 0.0


def calibrate_gics(
    signal_dfs: dict[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    train_dates: pd.Index,
    sectors: dict[str, str],
    all_tickers: list[str],
    regularization_alpha: float = 0.5,
) -> CalibrationResult:
    """Calibrate signal weights by GICS sector."""
    result = CalibrationResult(
        model="gics",
        date=str(train_dates[-1].date()),
    )

    signal_names = sorted(signal_dfs.keys())

    # 1. Global ICs
    for name in signal_names:
        ic = compute_ic(signal_dfs[name], fwd_returns, train_dates, all_tickers)
        result.global_ics[name] = ic

    # Global weights (IC-proportional, only positive)
    total_pos = sum(max(ic, 0) for ic in result.global_ics.values())
    if total_pos > 0:
        result.global_weights = {k: max(v, 0) / total_pos for k, v in result.global_ics.items()}
    else:
        result.global_weights = {k: 1.0 / len(signal_names) for k in signal_names}

    # 2. Per-sector ICs and weights
    sector_groups = {}
    for t in all_tickers:
        s = sectors.get(t, "Unknown")
        sector_groups.setdefault(s, []).append(t)

    for sector, members in sector_groups.items():
        result.group_members[sector] = members
        result.group_descriptions[sector] = f"{sector} ({len(members)} stocks)"

        if len(members) < 20:
            # Too few — use global weights
            result.group_signals[sector] = {
                name: {"ic": result.global_ics[name], "weight": result.global_weights[name],
                       "source": "global_fallback"}
                for name in signal_names
            }
            continue

        sector_ics = {}
        for name in signal_names:
            ic = compute_ic(signal_dfs[name], fwd_returns, train_dates, members)
            sector_ics[name] = ic

        # Regularize with global
        raw_total = sum(max(v, 0) for v in sector_ics.values())
        if raw_total > 0:
            raw_weights = {k: max(v, 0) / raw_total for k, v in sector_ics.items()}
        else:
            raw_weights = result.global_weights.copy()

        blended = {}
        for name in signal_names:
            blended[name] = (
                regularization_alpha * result.global_weights[name]
                + (1 - regularization_alpha) * raw_weights[name]
            )

        # Re-normalize
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        result.group_signals[sector] = {
            name: {"ic": sector_ics[name], "weight": blended[name], "source": "sector"}
            for name in signal_names
        }

    return result


def calibrate_clusters(
    signal_dfs: dict[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    returns: pd.DataFrame,
    train_dates: pd.Index,
    sectors: dict[str, str],
    all_tickers: list[str],
    n_clusters: int = 8,
    regularization_alpha: float = 0.5,
) -> CalibrationResult:
    """Calibrate signal weights by correlation clusters."""
    result = CalibrationResult(
        model="cluster",
        date=str(train_dates[-1].date()),
    )

    signal_names = sorted(signal_dfs.keys())

    # 1. Global ICs (same as GICS)
    for name in signal_names:
        ic = compute_ic(signal_dfs[name], fwd_returns, train_dates, all_tickers)
        result.global_ics[name] = ic

    total_pos = sum(max(ic, 0) for ic in result.global_ics.values())
    if total_pos > 0:
        result.global_weights = {k: max(v, 0) / total_pos for k, v in result.global_ics.items()}
    else:
        result.global_weights = {k: 1.0 / len(signal_names) for k in signal_names}

    # 2. Compute clusters
    common_dates = train_dates.intersection(returns.index)
    train_returns = returns.loc[common_dates].dropna(how="all")
    valid = train_returns.columns[train_returns.notna().sum() > len(train_returns) * 0.5]
    valid_tickers = [t for t in valid if t in all_tickers]

    if len(valid_tickers) < n_clusters * 5:
        logger.warning("Not enough tickers for clustering (%d)", len(valid_tickers))
        # Fallback: one cluster
        result.group_members["cluster_all"] = all_tickers
        result.group_signals["cluster_all"] = {
            name: {"ic": result.global_ics[name], "weight": result.global_weights[name],
                   "source": "global_fallback"}
            for name in signal_names
        }
        return result

    corr = train_returns[valid_tickers].corr().fillna(0).values.copy()
    affinity = (corr + 1) / 2
    np.fill_diagonal(affinity, 1)
    affinity = np.nan_to_num(affinity, nan=0.5)

    sc = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed",
        n_init=10, random_state=42,
    )
    labels = sc.fit_predict(affinity)
    cluster_map = dict(zip(valid_tickers, labels))

    # Assign unclustered tickers
    for t in all_tickers:
        if t not in cluster_map:
            cluster_map[t] = 0

    # 3. Build groups with descriptions
    for c in range(n_clusters):
        cid = f"C{c}"
        members = [t for t, cl in cluster_map.items() if cl == c]
        result.group_members[cid] = members

        # Name by dominant sector
        sector_dist = Counter(sectors.get(t, "?") for t in members)
        top2 = sector_dist.most_common(2)
        if top2[0][1] / max(len(members), 1) > 0.5:
            desc = f"{top2[0][0]} ({len(members)})"
        else:
            desc = f"{top2[0][0][:12]}+{top2[1][0][:12]} ({len(members)})" if len(top2) > 1 else f"Mixed ({len(members)})"
        result.group_descriptions[cid] = desc

        # Compute ICs and weights
        if len(members) < 15:
            result.group_signals[cid] = {
                name: {"ic": result.global_ics[name], "weight": result.global_weights[name],
                       "source": "global_fallback"}
                for name in signal_names
            }
            continue

        cluster_ics = {}
        for name in signal_names:
            ic = compute_ic(signal_dfs[name], fwd_returns, train_dates, members)
            cluster_ics[name] = ic

        raw_total = sum(max(v, 0) for v in cluster_ics.values())
        if raw_total > 0:
            raw_weights = {k: max(v, 0) / raw_total for k, v in cluster_ics.items()}
        else:
            raw_weights = result.global_weights.copy()

        blended = {}
        for name in signal_names:
            blended[name] = (
                regularization_alpha * result.global_weights[name]
                + (1 - regularization_alpha) * raw_weights[name]
            )
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        result.group_signals[cid] = {
            name: {"ic": cluster_ics[name], "weight": blended[name], "source": "cluster"}
            for name in signal_names
        }

    return result


def save_calibration(result: CalibrationResult, db) -> None:
    """Save calibration to database."""
    import sqlite3
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    conn = sqlite3.connect(str(db.db_path))
    for group, signals in result.group_signals.items():
        n_members = len(result.group_members.get(group, []))
        for sig_name, data in signals.items():
            conn.execute(
                """INSERT OR REPLACE INTO calibrations
                   (calibration_date, model, group_name, signal_name,
                    ic_value, weight, n_tickers, n_observations, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (result.date, result.model, group, sig_name,
                 data["ic"], data["weight"], n_members, 0, now),
            )

    # Save cluster assignments if cluster model
    if result.model == "cluster":
        for group, members in result.group_members.items():
            desc = result.group_descriptions.get(group, "")
            for ticker in members:
                conn.execute(
                    """INSERT OR REPLACE INTO cluster_assignments
                       (assignment_date, ticker, cluster_id, cluster_desc, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (result.date, ticker, group, desc, now),
                )

    conn.commit()
    conn.close()
    logger.info("Saved calibration: %s model, %d groups, %d signals",
                result.model, len(result.group_signals), len(next(iter(result.group_signals.values()))))


def print_calibration_report(gics_result: CalibrationResult,
                              cluster_result: CalibrationResult) -> None:
    """Print formatted calibration comparison."""
    signal_names = sorted(gics_result.global_ics.keys())

    print(f"\n{'='*95}")
    print(f"  CALIBRATION REPORT — {gics_result.date}")
    print(f"{'='*95}")

    # Global ICs
    print(f"\n  GLOBAL SIGNAL ICs (all tickers):")
    print(f"  {'Signal':>30s}  {'IC':>8s}  {'Weight':>8s}")
    print(f"  {'-'*50}")
    for name in sorted(signal_names, key=lambda n: -gics_result.global_ics[n]):
        ic = gics_result.global_ics[name]
        w = gics_result.global_weights[name]
        bar = "█" * int(abs(ic) * 80) if ic > 0 else ""
        print(f"  {name:>30s}  {ic:+.4f}  {w:6.1%}  {bar}")

    # GICS breakdown
    print(f"\n{'─'*95}")
    print(f"  GICS SECTOR SIGNAL WEIGHTS")
    print(f"{'─'*95}")

    # Header
    gics_groups = sorted(gics_result.group_signals.keys())
    top_signals = sorted(signal_names, key=lambda n: -gics_result.global_ics[n])[:10]

    for group in gics_groups:
        desc = gics_result.group_descriptions.get(group, group)
        signals = gics_result.group_signals[group]
        n = len(gics_result.group_members.get(group, []))

        # Top 5 signals for this sector
        sorted_sigs = sorted(signals.items(), key=lambda x: -x[1]["weight"])[:5]
        top_str = ", ".join(f"{s[7:] if len(s)>7 else s}({d['weight']:.0%})" for s, d in sorted_sigs)

        # Top IC signals
        sorted_ics = sorted(signals.items(), key=lambda x: -x[1]["ic"])[:3]
        ic_str = ", ".join(f"{s[2:]}({d['ic']:+.3f})" for s, d in sorted_ics)

        print(f"\n  {group} ({n} stocks)")
        print(f"    Top weights: {top_str}")
        print(f"    Top ICs:     {ic_str}")

    # Cluster breakdown
    print(f"\n{'─'*95}")
    print(f"  CLUSTER SIGNAL WEIGHTS")
    print(f"{'─'*95}")

    cluster_groups = sorted(cluster_result.group_signals.keys())
    for group in cluster_groups:
        desc = cluster_result.group_descriptions.get(group, group)
        signals = cluster_result.group_signals[group]
        n = len(cluster_result.group_members.get(group, []))

        sorted_sigs = sorted(signals.items(), key=lambda x: -x[1]["weight"])[:5]
        top_str = ", ".join(f"{s[2:]}({d['weight']:.0%})" for s, d in sorted_sigs)

        sorted_ics = sorted(signals.items(), key=lambda x: -x[1]["ic"])[:3]
        ic_str = ", ".join(f"{s[2:]}({d['ic']:+.3f})" for s, d in sorted_ics)

        print(f"\n  {desc}")
        print(f"    Top weights: {top_str}")
        print(f"    Top ICs:     {ic_str}")

    # Key differences
    print(f"\n{'─'*95}")
    print(f"  KEY DIFFERENCES: SIGNALS THAT VARY MOST ACROSS GROUPS")
    print(f"{'─'*95}")

    for sig in signal_names:
        gics_ics = [gics_result.group_signals[g][sig]["ic"] for g in gics_groups
                     if sig in gics_result.group_signals[g]]
        if gics_ics:
            spread = max(gics_ics) - min(gics_ics)
            if spread > 0.10:
                best_g = gics_groups[np.argmax(gics_ics)]
                worst_g = gics_groups[np.argmin(gics_ics)]
                print(f"  {sig:>30s}  spread={spread:.3f}  best={best_g}({max(gics_ics):+.3f})  worst={worst_g}({min(gics_ics):+.3f})")
