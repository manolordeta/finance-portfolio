"""
PDF Report Generator for MM Quant Capital.

Design: Inter font, clean typography, tables with navy headers and hairlines.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fpdf import FPDF

# ── Brand Design Tokens ──────────────────────────────────────────────

BRAND = "MM QUANT CAPITAL"
FONT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"

# Colors (R, G, B)
C_NAVY = (27, 58, 107)
C_COBALT = (44, 95, 158)
C_SLATE = (61, 61, 61)
C_MID = (107, 114, 128)
C_MIST = (154, 154, 154)
C_CLOUD = (247, 248, 250)
C_LINE = (226, 229, 234)
C_WHITE = (255, 255, 255)
C_GREEN = (22, 120, 60)
C_RED = (180, 40, 40)
C_AMBER = (160, 120, 10)

# Layout
ML = 22          # margin left
MR = 22          # margin right
PAGE_W = 210
CW = PAGE_W - ML - MR  # 166mm content width
CONTENT_TOP = 24  # Y where content starts (below header)


class QuantPDF(FPDF):

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=22)
        self.set_top_margin(CONTENT_TOP)  # ensures auto page break respects top margin
        self._is_cover = True
        # Register Inter font family
        self.add_font("Inter", "", str(FONT_DIR / "Inter-Regular.ttf"), uni=True)
        self.add_font("Inter", "B", str(FONT_DIR / "Inter-Bold.ttf"), uni=True)
        self.add_font("InterMed", "", str(FONT_DIR / "Inter-Medium.ttf"), uni=True)
        self.add_font("InterSB", "", str(FONT_DIR / "Inter-SemiBold.ttf"), uni=True)

    def header(self):
        if self._is_cover or self.page_no() <= 1:
            return
        self.set_font("InterSB", "", 7.5)
        self.set_text_color(*C_NAVY)
        self.set_xy(ML, 10)
        self.cell(CW / 2, 4, BRAND, align="L")
        self.set_font("Inter", "", 7.5)
        self.set_text_color(*C_MIST)
        self.cell(CW / 2, 4, datetime.now().strftime("%B %d, %Y"), align="R")
        # Thin line
        self.set_draw_color(*C_LINE)
        self.set_line_width(0.2)
        self.line(ML, 16, PAGE_W - MR, 16)
        # Push cursor below header so content never overlaps
        self.set_y(CONTENT_TOP)

    def footer(self):
        if self._is_cover:
            return
        self.set_y(-14)
        self.set_draw_color(*C_LINE)
        self.set_line_width(0.15)
        self.line(ML, self.get_y(), PAGE_W - MR, self.get_y())
        self.set_font("Inter", "", 6.5)
        self.set_text_color(*C_MIST)
        self.set_x(ML)
        self.cell(CW / 2, 8, f"{BRAND}  |  Quantitative Investment System", align="L")
        pg = self.page_no() - 1
        if pg > 0:
            self.cell(CW / 2, 8, f"Page {pg}", align="R")

    def content_page(self):
        """Add a new content page and set Y to proper start position."""
        self.add_page()
        self.set_y(CONTENT_TOP)


def _s(text) -> str:
    """Safe string for PDF."""
    if not isinstance(text, str):
        text = str(text)
    return text.replace("\u2014", "-").replace("\u2013", "-").replace("\u2018", "'").replace(
        "\u2019", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u2026", "...")


def generate_report(
    ranking_df,
    alerts: list,
    watchlist_status: list[dict],
    discoveries: list[dict],
    signal_weights: dict[str, float],
    data_quality: dict[str, Any] | None = None,
    output_path: str = "reports/MM_Quant_Capital_Daily_Report.pdf",
    calibration_summary: dict | None = None,
    cluster_info: dict | None = None,
    cluster_assignments: dict | None = None,
    profiles: dict | None = None,
    ranking_cluster_df=None,
) -> str:
    import numpy as np

    pdf = QuantPDF()
    today = datetime.now().strftime("%B %d, %Y")
    today_short = datetime.now().strftime("%Y-%m-%d")
    n_tickers = len(ranking_df) if ranking_df is not None else 0
    n_signals = len(signal_weights)

    # ── Helpers ──────────────────────────────────────────────────

    def h1(title: str):
        pdf.set_x(ML)
        pdf.set_font("InterSB", "", 13)
        pdf.set_text_color(*C_NAVY)
        pdf.cell(CW, 9, title.upper(), ln=True)
        pdf.ln(3)

    def h2(title: str):
        pdf.set_x(ML)
        pdf.set_font("InterMed", "", 10)
        pdf.set_text_color(*C_COBALT)
        pdf.cell(CW, 7, title, ln=True)
        pdf.ln(2)

    def body(text: str):
        pdf.set_x(ML)
        pdf.set_font("Inter", "", 9)
        pdf.set_text_color(*C_SLATE)
        pdf.multi_cell(CW, 5, _s(text))
        pdf.ln(3)

    def meta(label: str, value: str):
        pdf.set_x(ML)
        pdf.set_font("Inter", "", 7.5)
        pdf.set_text_color(*C_MIST)
        pdf.cell(28, 4.5, label)
        pdf.set_text_color(*C_SLATE)
        pdf.cell(0, 4.5, _s(value), ln=True)

    def tbl_header(cols: list[tuple[str, int, str]]):
        pdf.set_font("InterSB", "", 7)
        pdf.set_text_color(*C_NAVY)
        x = ML
        for text, w, a in cols:
            pdf.set_xy(x, pdf.get_y())
            pdf.cell(w, 5.5, text.upper(), align=a)
            x += w
        pdf.ln(5.5)
        y = pdf.get_y()
        pdf.set_draw_color(*C_NAVY)
        pdf.set_line_width(0.4)
        pdf.line(ML, y, PAGE_W - MR, y)
        pdf.ln(1.5)

    def tbl_row(vals: list[tuple[str, int, str]], last: bool = False,
                color: tuple = C_SLATE):
        pdf.set_font("Inter", "", 7.5)
        pdf.set_text_color(*color)
        x = ML
        for text, w, a in vals:
            pdf.set_xy(x, pdf.get_y())
            pdf.cell(w, 5, _s(str(text)), align=a)
            x += w
        pdf.ln(5)
        y = pdf.get_y()
        pdf.set_draw_color(*(C_LINE if not last else (200, 204, 210)))
        pdf.set_line_width(0.15 if not last else 0.3)
        pdf.line(ML, y, PAGE_W - MR, y)
        pdf.ln(0.8)

    # ════════════════════════════════════════════════════════════════
    # COVER
    # ════════════════════════════════════════════════════════════════
    pdf._is_cover = True
    pdf.add_page()

    # Top metadata
    pdf.set_font("InterSB", "", 7.5)
    pdf.set_text_color(*C_MIST)
    pdf.set_xy(ML, 18)
    pdf.cell(CW / 2, 5, f"PREPARED BY {BRAND}", align="L")
    pdf.cell(CW / 2, 5, today.upper(), align="R")

    # Blue band
    pdf.set_fill_color(*C_CLOUD)
    pdf.rect(0, 55, PAGE_W, 135, "F")

    # Title
    pdf.set_xy(ML, 80)
    pdf.set_font("Inter", "", 44)
    pdf.set_text_color(*C_NAVY)
    pdf.cell(0, 17, "Daily", ln=True)
    pdf.set_x(ML)
    pdf.set_font("InterSB", "", 32)
    pdf.cell(0, 15, "Investment", ln=True)
    pdf.set_x(ML)
    pdf.cell(0, 15, "Report", ln=True)

    # Subtitle
    pdf.set_x(ML)
    pdf.set_font("InterMed", "", 10)
    pdf.set_text_color(*C_COBALT)
    pdf.cell(0, 9, BRAND, ln=True)

    pdf.set_x(ML)
    pdf.set_font("Inter", "", 8.5)
    pdf.set_text_color(*C_MID)
    pdf.cell(0, 6, f"{n_tickers} tickers  |  {n_signals} signals  |  S&P 500  |  {today_short}", ln=True)

    # Bottom metadata (inside the blue band area, stays on page 1)
    pdf.set_xy(ML, 155)
    meta("Universe:", f"S&P 500 ({n_tickers} tickers)")
    meta("Signals:", f"{n_signals} active (IC-weighted)")
    meta("Model:", "v2.0")
    meta("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M"))

    # ════════════════════════════════════════════════════════════════
    # PAGE: EXECUTIVE SUMMARY + ALERTS
    # ════════════════════════════════════════════════════════════════
    pdf._is_cover = False
    pdf.content_page()

    h1("Executive Summary")

    high_alerts = [a for a in alerts if a.severity == "HIGH"]
    med_alerts = [a for a in alerts if a.severity == "MEDIUM"]
    low_alerts = [a for a in alerts if a.severity == "LOW"]
    no_alerts = [a for a in alerts if a.severity == "NONE"]
    sents = [a.sentiment_score for a in alerts if hasattr(a, "sentiment_score")]
    avg_s = sum(sents) / len(sents) if sents else 0

    txt = (
        f"The MM Quant Capital system monitors {n_tickers} tickers across the S&P 500 "
        f"using {n_signals} validated signals. The active watchlist contains "
        f"{len(watchlist_status)} positions under monitoring. "
    )
    if high_alerts:
        txt += f"{len(high_alerts)} HIGH severity alerts require immediate review. "
    if med_alerts:
        txt += f"{len(med_alerts)} MEDIUM severity alerts warrant attention. "
    if not high_alerts and not med_alerts:
        txt += "No high-severity alerts detected. Portfolio thesis intact for monitored positions. "
    txt += f"Overall portfolio sentiment: {avg_s:+.2f}."
    body(txt)

    pdf.ln(2)
    h1("Portfolio Alerts")

    def render_alert(a, sev_color):
        pdf.set_x(ML)
        pdf.set_font("InterSB", "", 9.5)
        pdf.set_text_color(*sev_color)
        pdf.cell(15, 5.5, a.ticker)
        pdf.set_font("InterMed", "", 8)
        pdf.set_text_color(*C_MID)
        pdf.cell(0, 5.5, a.action, ln=True)
        pdf.set_x(ML)
        pdf.set_font("Inter", "", 8.5)
        pdf.set_text_color(*C_SLATE)
        pdf.multi_cell(CW, 4.5, _s(a.headline))
        pdf.set_x(ML)
        pdf.set_font("Inter", "", 8)
        pdf.set_text_color(*C_MID)
        pdf.multi_cell(CW, 4.5, _s(a.analysis))
        pdf.ln(3)

    if high_alerts:
        h2("High Severity")
        for a in high_alerts:
            render_alert(a, C_RED)

    if med_alerts:
        h2("Medium Severity")
        for a in med_alerts:
            render_alert(a, C_AMBER)

    if low_alerts:
        h2("Low Severity")
        for a in low_alerts:
            pdf.set_x(ML)
            pdf.set_font("InterSB", "", 8)
            pdf.set_text_color(*C_SLATE)
            pdf.cell(15, 4.5, a.ticker)
            pdf.set_font("Inter", "", 7.5)
            pdf.set_text_color(*C_MID)
            pdf.cell(CW - 15, 4.5, _s(a.headline[:75]), ln=True)

    if no_alerts:
        pdf.ln(2)
        pdf.set_x(ML)
        pdf.set_font("Inter", "", 7.5)
        pdf.set_text_color(*C_MIST)
        pdf.cell(0, 4.5, f"No alerts: {', '.join(a.ticker for a in no_alerts)}", ln=True)

    # ── WATCHLIST STATUS ────────────────────────────────────────
    pdf.content_page()
    h1("Watchlist Status")

    # Same table style as ranking tables
    wcols = [
        ("Ticker", 18, "C"), ("Zone", 18, "C"), ("Rank", 28, "C"),
        ("Score", 25, "C"), ("1m Return", 25, "C"), ("Alert Summary", CW - 114, "L"),
    ]
    tbl_header(wcols)

    for i, ws_item in enumerate(watchlist_status):
        zone = ws_item.get("zone", "?")
        color = C_GREEN if "TOP20" in zone else C_RED if "BTM" in zone else C_SLATE
        score_val = ws_item.get("score", 0)
        score_str = f"{score_val:+.4f}" if isinstance(score_val, float) else str(score_val)
        tbl_row([
            (ws_item.get("ticker", ""), 18, "C"),
            (zone, 18, "C"),
            (ws_item.get("rank", ""), 28, "C"),
            (score_str, 25, "C"),
            (ws_item.get("ret_1m", ""), 25, "C"),
            (ws_item.get("notes", "")[:48], CW - 114, "L"),
        ], last=(i == len(watchlist_status) - 1), color=color)

    pdf.ln(8)

    # ── Top 25 Ranking ───────────────────────────────────────────
    h1("Universe Ranking - Top 25")

    rcols = [
        ("#", 8, "C"), ("Ticker", 16, "C"), ("Score", 22, "C"),
        ("1m", 18, "C"), ("3m", 18, "C"),
        ("Company", CW - 112, "L"), ("Sector", 30, "L"),
    ]
    tbl_header(rcols)

    if ranking_df is not None:
        top = ranking_df.head(25)
        for i, (_, row) in enumerate(top.iterrows()):
            tbl_row([
                (str(int(row.get("rank", 0))), 8, "C"),
                (row.get("ticker", ""), 16, "C"),
                (f"{row.get('composite_score', 0):+.4f}", 22, "C"),
                (str(row.get("ret_1m", "")), 18, "C"),
                (str(row.get("ret_3m", "")), 18, "C"),
                (str(row.get("company", ""))[:30], CW - 112, "L"),
                (str(row.get("sector", ""))[:18], 30, "L"),
            ], last=(i == len(top) - 1))

    # ── Cluster Ranking - Top 25 ────────────────────────────────────
    if ranking_cluster_df is not None:
        pdf.content_page()
        h1("Universe Ranking - Top 25 (Clusters)")
        body("Ranking using cluster-specific signal weights. Cluster ID shown for reference.")

        crcols = [
            ("#", 8, "C"), ("Ticker", 16, "C"), ("Cluster", 14, "C"),
            ("Score", 22, "C"), ("1m", 18, "C"), ("3m", 18, "C"),
            ("Company", CW - 126, "L"), ("Sector", 30, "L"),
        ]
        tbl_header(crcols)

        top_c = ranking_cluster_df.head(25)
        for i, (_, row) in enumerate(top_c.iterrows()):
            cid = cluster_assignments.get(row.get("ticker", ""), "?") if cluster_assignments else "?"
            tbl_row([
                (str(int(row.get("rank", 0))), 8, "C"),
                (row.get("ticker", ""), 16, "C"),
                (str(cid), 14, "C"),
                (f"{row.get('composite_score', 0):+.4f}", 22, "C"),
                (str(row.get("ret_1m", "")), 18, "C"),
                (str(row.get("ret_3m", "")), 18, "C"),
                (str(row.get("company", ""))[:25], CW - 126, "L"),
                (str(row.get("sector", ""))[:18], 30, "L"),
            ], last=(i == len(top_c) - 1))

    # ════════════════════════════════════════════════════════════════
    # PAGE: DISCOVERIES + SIGNALS + DATA QUALITY
    # ════════════════════════════════════════════════════════════════
    pdf.content_page()

    h1("New Discoveries")
    body(
        "Tickers ranking highly that are not currently in the active watchlist. "
        "These may warrant fundamental due diligence by the investment team."
    )

    if discoveries:
        dcols = [
            ("#", 8, "C"), ("Ticker", 16, "C"), ("Score", 22, "C"),
            ("1m", 18, "C"), ("Company", CW - 94, "L"), ("Sector", 30, "L"),
        ]
        tbl_header(dcols)
        for i, d in enumerate(discoveries[:15]):
            tbl_row([
                (str(d.get("rank", "")), 8, "C"),
                (d.get("ticker", ""), 16, "C"),
                (f"{d.get('score', 0):+.4f}", 22, "C"),
                (str(d.get("ret_1m", "")), 18, "C"),
                (str(d.get("company", ""))[:30], CW - 94, "L"),
                (str(d.get("sector", ""))[:18], 30, "L"),
            ], last=(i == min(14, len(discoveries) - 1)))

    pdf.ln(8)

    # ── Signal Configuration ─────────────────────────────────────
    h1("Active Signal Configuration")
    body(
        "Signals weighted proportionally to their validated Information Coefficient (IC). "
        "Signals marked (*) are in forward measurement period."
    )

    scols = [
        ("Signal", 52, "L"), ("Category", 22, "C"),
        ("Weight", 20, "C"), ("Status", 25, "C"), ("IC (21d)", CW - 119, "C"),
    ]
    tbl_header(scols)

    for i, (name, weight) in enumerate(sorted(signal_weights.items(), key=lambda x: -x[1])):
        cat = ("Tech" if name in ("momentum_12_1", "momentum_1m", "macd_signal", "volume_ratio")
               else "Fund" if name in ("pe_relative", "ev_ebitda_relative", "fcf_yield", "roe",
                                        "gross_margin_delta", "earnings_surprise", "revenue_growth",
                                        "debt_equity_inv")
               else "Val" if name in ("golden_cross", "mean_reversion_63d", "price_vs_sma50", "price_vs_21d")
               else "Sent")
        status = "Measuring*" if "sentiment" in name else "Validated"
        tbl_row([
            (name, 52, "L"), (cat, 22, "C"),
            (f"{weight:.1%}", 20, "C"), (status, 25, "C"),
            ("-" if "sentiment" in name else "ok", CW - 119, "C"),
        ], last=(i == len(signal_weights) - 1))

    # ── GICS Signal Weights by Sector ──────────────────────────────
    if calibration_summary and "gics" in calibration_summary:
        pdf.content_page()
        h1("Signal Weights by Sector (GICS)")
        body(
            "Top 3 dominant signals per sector. Weights are IC-proportional "
            "and regularized (50% global + 50% sector-specific)."
        )

        gcols = [
            ("Sector", 40, "L"), ("Signal 1", 30, "L"), ("Wt", 12, "C"),
            ("Signal 2", 30, "L"), ("Wt", 12, "C"),
            ("Signal 3", 30, "L"), ("Wt", 12, "C"),
        ]
        tbl_header(gcols)

        gics_data = calibration_summary["gics"]
        sorted_sectors = sorted(gics_data.keys())
        for i, sector in enumerate(sorted_sectors):
            sigs = sorted(gics_data[sector], key=lambda x: -x["weight"])[:3]
            row_data = [(sector[:22], 40, "L")]
            for s in sigs:
                short_name = s["signal"].replace("F_", "").replace("T_", "").replace("V_", "")
                row_data.append((short_name[:18], 30, "L"))
                row_data.append((f"{s['weight']:.0%}", 12, "C"))
            # Pad if fewer than 3 signals
            while len(row_data) < 7:
                row_data.append(("-", 30 if len(row_data) % 2 == 1 else 12, "C"))
            tbl_row(row_data, last=(i == len(sorted_sectors) - 1))

    # ── Cluster Overview ──────────────────────────────────────────
    if calibration_summary and "cluster" in calibration_summary and cluster_info:
        pdf.ln(8)
        h1("Cluster Overview")
        body(
            "Clusters derived from correlation-based spectral clustering. "
            "Recalculated monthly during calibration."
        )

        ccols = [
            ("Cluster", 14, "C"), ("Size", 12, "C"),
            ("Top Signal", 30, "L"), ("Wt", 12, "C"),
            ("Signal 2", 26, "L"), ("Wt", 10, "C"),
            ("Sample Tickers", CW - 104, "L"),
        ]
        tbl_header(ccols)

        cluster_data = calibration_summary["cluster"]
        sorted_clusters = sorted(cluster_data.keys())
        for i, cid in enumerate(sorted_clusters):
            sigs = sorted(cluster_data[cid], key=lambda x: -x["weight"])[:2]
            tickers_in = cluster_info.get(cid, [])
            n_tickers_cluster = len(tickers_in)

            # Get sample tickers (first 5, prefer recognizable names)
            sample = tickers_in[:6]
            sample_str = ", ".join(sample[:5])
            if len(tickers_in) > 5:
                sample_str += "..."

            row_data = [
                (str(cid), 14, "C"),
                (str(n_tickers_cluster), 12, "C"),
            ]
            for s in sigs[:2]:
                short_name = s["signal"].replace("F_", "").replace("T_", "").replace("V_", "")
                row_data.append((short_name[:16], 30 if len(row_data) == 2 else 26, "L"))
                row_data.append((f"{s['weight']:.0%}", 12 if len(row_data) == 3 else 10, "C"))
            # Pad
            while len(row_data) < 6:
                row_data.append(("-", 26, "L"))
                row_data.append(("-", 10, "C"))

            row_data.append((sample_str[:35], CW - 104, "L"))
            tbl_row(row_data, last=(i == len(sorted_clusters) - 1))

    # ── Data Quality ─────────────────────────────────────────────
    if data_quality:
        pdf.ln(6)
        h1("Data Quality Notes")
        body(
            "Tickers below have signals scored as neutral (0.0) due to missing data. "
            "This is NOT a true neutral - it indicates absence of information."
        )
        for ticker, info in data_quality.items():
            if info.get("missing"):
                pdf.set_x(ML)
                pdf.set_font("InterSB", "", 8)
                pdf.set_text_color(*C_SLATE)
                pdf.cell(15, 4.5, ticker)
                pdf.set_font("Inter", "", 7.5)
                pdf.set_text_color(*C_MID)
                pdf.cell(0, 4.5, f"Missing: {', '.join(info['missing'])}", ln=True)

    # ── Disclaimer ───────────────────────────────────────────────
    pdf.content_page()
    h1("Disclaimer")
    body(
        "This report is generated by the MM Quant Capital quantitative investment system "
        "for internal research purposes only. It does not constitute investment advice, "
        "a recommendation, or an offer to buy or sell any security."
    )
    body(
        "Past performance and quantitative signals do not guarantee future results. "
        "Always perform independent due diligence before making investment decisions."
    )
    pdf.ln(4)
    meta("Report generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    meta("Model version:", "v2.0 - IC-proportional composite")
    meta("Universe:", f"S&P 500 ({n_tickers} tickers)")
    meta("Signals:", f"{n_signals} active")

    # ── Save ─────────────────────────────────────────────────────
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output))
    return str(output)
