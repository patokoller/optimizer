"""
Portfolio report PDF renderer (Feature B).

Consumes the structured ReportData produced by app/services/portfolio_report.py
and renders a consulting-grade PDF: executive summary, holdings & allocation,
per-holding scorecard, risk analytics (current vs proposed), proposed actions,
COVID stress note, and caveats.

Charts are matplotlib PNGs embedded as reportlab Images; layout is Platypus.
No Unicode sub/superscript glyphs are used (they render as black boxes in the
built-in fonts — per the pdf skill).
"""

from __future__ import annotations

import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

# ── Palette (professional, print-friendly on white) ──────────────────────────
NAVY = colors.HexColor("#1a2236")
BLUE = colors.HexColor("#4f8ef7")
GREEN = colors.HexColor("#2f9e6f")
AMBER = colors.HexColor("#d6932b")
RED = colors.HexColor("#d24a4a")
GREY = colors.HexColor("#6b7280")
LIGHT = colors.HexColor("#eef1f6")

_MPL_BLUE = "#4f8ef7"
_MPL_GREEN = "#2f9e6f"
_MPL_AMBER = "#d6932b"
_MPL_RED = "#d24a4a"
_MPL_GREY = "#9aa0b0"


def _score_hex(s: Optional[float]) -> colors.Color:
    if s is None:
        return GREY
    if s >= 0.7:
        return GREEN
    if s >= 0.4:
        return AMBER
    return RED


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("H1x", parent=ss["Heading1"], textColor=NAVY, fontSize=16, spaceAfter=6))
    ss.add(ParagraphStyle("H2x", parent=ss["Heading2"], textColor=NAVY, fontSize=12, spaceBefore=10, spaceAfter=4))
    ss.add(ParagraphStyle("Bodyx", parent=ss["Normal"], fontSize=9.5, leading=14, textColor=colors.HexColor("#222630")))
    ss.add(ParagraphStyle("Smallx", parent=ss["Normal"], fontSize=8, leading=11, textColor=GREY))
    ss.add(ParagraphStyle("Cellx", parent=ss["Normal"], fontSize=8.5, leading=11))
    ss.add(ParagraphStyle("TitleBig", parent=ss["Title"], textColor=NAVY, fontSize=22))
    return ss


def _fig_to_image(fig, width_in: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_in * inch
    img.drawHeight = width_in * inch * ratio
    return img


def _allocation_donut(holdings: list[dict]) -> Image:
    labels = [h["ticker"] for h in holdings if h.get("weight")]
    sizes = [h["weight"] for h in holdings if h.get("weight")]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    if sizes:
        ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90,
               wedgeprops=dict(width=0.42, edgecolor="white"),
               textprops={"fontsize": 8})
    ax.set_title("Current allocation", fontsize=11, color="#1a2236")
    return _fig_to_image(fig, 3.1)


def _sector_bar(sector_w: dict[str, float]) -> Image:
    items = list(sector_w.items())[:8]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    if items:
        names = [k for k, _ in items][::-1]
        vals = [v * 100 for _, v in items][::-1]
        ax.barh(names, vals, color=_MPL_BLUE)
        ax.set_xlabel("Weight (%)", fontsize=8)
        ax.tick_params(labelsize=8)
    ax.set_title("Sector exposure", fontsize=11, color="#1a2236")
    return _fig_to_image(fig, 3.3)


def _risk_compare_bar(cur: dict, prop: dict) -> Image:
    metrics = [("Ann. return", "annualized_return", 100),
               ("Volatility", "annualized_vol", 100),
               ("Sharpe", "sharpe", 1),
               ("Max drawdown", "max_drawdown", 100)]
    labels = [m[0] for m in metrics]
    cur_v = [(cur.get(m[1]) or 0) * m[2] for m in metrics]
    prop_v = [(prop.get(m[1]) or 0) * m[2] for m in metrics]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.bar([i - 0.2 for i in x], cur_v, width=0.4, label="Current", color=_MPL_GREY)
    ax.bar([i + 0.2 for i in x], prop_v, width=0.4, label="Proposed", color=_MPL_BLUE)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8)
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.set_title("Risk: current vs proposed (returns/vol/drawdown in %)", fontsize=10, color="#1a2236")
    return _fig_to_image(fig, 6.2)


def _stress_chart(stress: dict) -> Optional[Image]:
    series = stress.get("series") if isinstance(stress, dict) else None
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    for name, pts in series.items():
        xs = [p[0] for p in pts]
        ys = [p[1] * 100 for p in pts]
        ax.plot(xs, ys, label=name, linewidth=1.6)
    ax.set_title("COVID crash stress test (Feb-May 2020)", fontsize=10, color="#1a2236")
    ax.set_ylabel("Cumulative return (%)", fontsize=8)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)
    return _fig_to_image(fig, 6.0)


def _fmt_pct(x: Optional[float], digits=1) -> str:
    return "-" if x is None else f"{x * 100:.{digits}f}%"


def _fmt_num(x: Optional[float], digits=2) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def build_report_pdf(data: dict) -> bytes:
    """Render ReportData → PDF bytes."""
    ss = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title=f"Portfolio Analysis — {data.get('portfolio_name', '')}",
    )
    story: list = []
    nar = data.get("narrative", {}) or {}

    # ── Title + executive summary ────────────────────────────────────────────
    story.append(Paragraph("Portfolio Analysis Report", ss["TitleBig"]))
    story.append(Paragraph(
        f"{data.get('portfolio_name', 'Portfolio')} &nbsp;·&nbsp; as of {data.get('as_of', '')}",
        ss["Smallx"]))
    reg = data.get("regime") or {}
    if reg.get("label"):
        story.append(Paragraph(
            f"Market regime: <b>{reg['label']}</b> (confidence {_fmt_num(reg.get('confidence'))})",
            ss["Smallx"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Executive summary", ss["H2x"]))
    story.append(Paragraph(nar.get("exec_summary", "Summary unavailable."), ss["Bodyx"]))
    story.append(Spacer(1, 6))

    # ── Allocation charts ────────────────────────────────────────────────────
    story.append(Paragraph("Holdings &amp; allocation", ss["H2x"]))
    holdings = data.get("holdings", [])
    charts = Table(
        [[_allocation_donut(holdings), _sector_bar(data.get("risk_current", {}).get("sector_weights", {}))]],
        colWidths=[3.3 * inch, 3.5 * inch])
    charts.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(charts)

    # ── Per-holding scorecard ────────────────────────────────────────────────
    story.append(Paragraph("Per-holding scorecard", ss["H2x"]))
    head = ["Ticker", "Weight", "Overall", "Tech", "Fund", "Entr", "Drift", "Flag"]
    rows = [head]
    for h in holdings:
        st = h.get("strategies", {})
        drift = h.get("drift_trend") or "-"
        flag = "WATCH" if drift == "DETERIORATING" else ""
        rows.append([
            h["ticker"],
            _fmt_pct(h.get("weight")),
            _fmt_num(h.get("overall_score")),
            _fmt_num((st.get("technical") or {}).get("combined")),
            _fmt_num((st.get("fundamental") or {}).get("combined")),
            _fmt_num((st.get("entropy") or {}).get("combined")),
            drift,
            flag,
        ])
    tbl = Table(rows, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7dbe6")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    # colour the Overall column + WATCH flags
    for i, h in enumerate(holdings, start=1):
        style.append(("TEXTCOLOR", (2, i), (2, i), _score_hex(h.get("overall_score"))))
        if (h.get("drift_trend") or "") == "DETERIORATING":
            style.append(("TEXTCOLOR", (7, i), (7, i), RED))
            style.append(("FONTNAME", (7, i), (7, i), "Helvetica-Bold"))
    tbl.setStyle(TableStyle(style))
    story.append(tbl)

    # Watch items
    watch = data.get("watch_items") or []
    if watch:
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "<b>Watch items:</b> " + ", ".join(watch) +
            " — language deteriorating across recent calls; review before adding.",
            ss["Smallx"]))

    story.append(PageBreak())

    # ── Risk analytics ───────────────────────────────────────────────────────
    story.append(Paragraph("Risk analytics", ss["H2x"]))
    cur, prop = data.get("risk_current", {}), data.get("risk_proposed", {})
    story.append(_risk_compare_bar(cur, prop))
    rrows = [["Metric", "Current", "Proposed"],
             ["Annualized return", _fmt_pct(cur.get("annualized_return")), _fmt_pct(prop.get("annualized_return"))],
             ["Annualized volatility", _fmt_pct(cur.get("annualized_vol")), _fmt_pct(prop.get("annualized_vol"))],
             ["Sharpe ratio", _fmt_num(cur.get("sharpe")), _fmt_num(prop.get("sharpe"))],
             ["Max drawdown", _fmt_pct(cur.get("max_drawdown")), _fmt_pct(prop.get("max_drawdown"))],
             ["Concentration (HHI)", _fmt_num(cur.get("hhi"), 3), _fmt_num(prop.get("hhi"), 3)],
             ["Positions", str(cur.get("n_positions", "-")), str(prop.get("n_positions", "-"))]]
    rtbl = Table(rrows, colWidths=[2.4 * inch, 1.6 * inch, 1.6 * inch], hAlign="LEFT")
    rtbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7dbe6")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(rtbl)
    if nar.get("risk_commentary"):
        story.append(Spacer(1, 6))
        story.append(Paragraph(nar["risk_commentary"], ss["Bodyx"]))

    # ── Proposed actions ─────────────────────────────────────────────────────
    story.append(Paragraph("Proposed actions", ss["H2x"]))
    story.append(Paragraph(
        f"Optimizer: <b>{data.get('optimizer', 'MVO')}</b>. Proposals are advisory — "
        f"each ties to the holding's score and is yours to accept, adjust, or reject.",
        ss["Smallx"]))
    arows = [["Ticker", "Action", "Weight Δ", "Rationale"]]
    for a in data.get("actions", []):
        arows.append([
            a["ticker"], a["action"],
            (("+" if (a.get("delta") or 0) >= 0 else "") + _fmt_pct(a.get("delta"))),
            Paragraph(a.get("rationale", ""), ss["Cellx"]),
        ])
    atbl = Table(arows, colWidths=[0.8 * inch, 0.8 * inch, 0.9 * inch, 4.3 * inch], repeatRows=1, hAlign="LEFT")
    astyle = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7dbe6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    action_color = {"ADD": GREEN, "TRIM": AMBER, "EXIT": RED, "HOLD": GREY}
    for i, a in enumerate(data.get("actions", []), start=1):
        astyle.append(("TEXTCOLOR", (1, i), (1, i), action_color.get(a["action"], GREY)))
        astyle.append(("FONTNAME", (1, i), (1, i), "Helvetica-Bold"))
    atbl.setStyle(TableStyle(astyle))
    story.append(atbl)

    # ── Stress test ──────────────────────────────────────────────────────────
    stress_img = _stress_chart(data.get("stress_test") or {})
    if stress_img is not None:
        story.append(Paragraph("Stress test", ss["H2x"]))
        story.append(stress_img)
        note = (data.get("stress_test") or {}).get("note")
        if note:
            story.append(Paragraph(note, ss["Smallx"]))

    # ── Closing + caveats ────────────────────────────────────────────────────
    if nar.get("closing"):
        story.append(Spacer(1, 6))
        story.append(Paragraph(nar["closing"], ss["Bodyx"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Caveats", ss["H2x"]))
    story.append(Paragraph(data.get("disclaimer", _DEFAULT_DISCLAIMER), ss["Smallx"]))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


_DEFAULT_DISCLAIMER = (
    "This report is decision-support, not investment advice. Scores derive from a "
    "published research framework (Cohen, Aiche &amp; Eichel, 2025) whose live "
    "predictive power on current data is still being validated; treat them as one "
    "input, not a verdict. Backtested figures (2020-2025) are not a representation "
    "of live performance. Proposed weights are model output, not the paper's "
    "equal-weight selection. All decisions rest with the reader."
)
