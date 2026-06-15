"""
Portfolio report PDF renderer (Feature B) — professional redesign.

Renders the structured ReportData from app/services/portfolio_report.py into a
consulting-grade PDF modelled on institutional client reports (Addepar / Julius
Baer / Morningstar): a cover page, running header/footer, a "Key figures" stat
panel, an editorial Advisor's View, allocation visuals, a per-holding scorecard,
per-holding detail cards, risk analytics, proposed actions, a stress note, and
caveats.

Design system lives in the constants below. Charts are matplotlib PNGs on a
clean theme; layout is reportlab Platypus over a BaseDocTemplate with separate
cover/content page templates. No Unicode sub/superscripts (per the pdf skill).
"""

from __future__ import annotations

import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, NextPageTemplate, HRFlowable, KeepTogether, Flowable,
)

# ── Design tokens ────────────────────────────────────────────────────────────
INK     = colors.HexColor("#1b2a4a")   # deep navy — titles, headers
INK2    = colors.HexColor("#2a3656")
ACCENT  = colors.HexColor("#2c7a7b")   # teal — kickers, rules
GOLD    = colors.HexColor("#b8893b")
TEXT    = colors.HexColor("#2a2f3a")
MUTED   = colors.HexColor("#6a7180")
FAINT   = colors.HexColor("#9aa1b0")
HAIR    = colors.HexColor("#d9dee8")
PANEL   = colors.HexColor("#f5f7fb")
PANEL2  = colors.HexColor("#eaeef6")
WHITE   = colors.white
GREEN   = colors.HexColor("#2f8f5b")
AMBER   = colors.HexColor("#c98a1e")
RED     = colors.HexColor("#c0492f")

# matplotlib palette (hex strings)
M_INK = "#1b2a4a"; M_TEAL = "#2c7a7b"; M_GREEN = "#2f8f5b"
M_AMBER = "#c98a1e"; M_RED = "#c0492f"; M_GREY = "#9aa1b0"; M_HAIR = "#d9dee8"
M_SERIES = ["#2c7a7b", "#1b2a4a", "#b8893b", "#6a7180", "#2f8f5b"]

PAGE_W, PAGE_H = letter
MARGIN = 0.78 * inch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#c4ccd9", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e6eaf1", "grid.linewidth": 0.7,
    "xtick.color": "#6a7180", "ytick.color": "#6a7180",
    "axes.labelcolor": "#6a7180", "text.color": "#2a2f3a",
})


def _score_color(s: Optional[float]) -> colors.Color:
    if s is None:
        return FAINT
    return GREEN if s >= 0.66 else AMBER if s >= 0.4 else RED


def _hex(c: colors.Color) -> str:
    """'#rrggbb' string for use inside Paragraph <font color=...> markup."""
    return "#" + c.hexval()[2:]


def _styles():
    ss = getSampleStyleSheet()
    def add(name, **kw):
        ss.add(ParagraphStyle(name, parent=ss["Normal"], **kw))
    add("CoverTitle", fontName="Helvetica-Bold", fontSize=30, leading=34, textColor=INK)
    add("CoverSub", fontName="Helvetica", fontSize=13, leading=18, textColor=ACCENT)
    add("CoverMeta", fontName="Helvetica", fontSize=9.5, leading=14, textColor=MUTED)
    add("Kicker", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=ACCENT,
        spaceAfter=1)
    add("H2", fontName="Helvetica-Bold", fontSize=14.5, leading=18, textColor=INK, spaceAfter=2)
    add("H3", fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=INK)
    add("Body", fontName="Helvetica", fontSize=9.5, leading=15, textColor=TEXT)
    add("BodyTight", fontName="Helvetica", fontSize=9, leading=13, textColor=TEXT)
    add("Small", fontName="Helvetica", fontSize=8, leading=11.5, textColor=MUTED)
    add("Tiny", fontName="Helvetica", fontSize=7, leading=9.5, textColor=FAINT)
    add("Label", fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=MUTED)
    add("StatValue", fontName="Helvetica-Bold", fontSize=17, leading=19, textColor=INK)
    add("StatValueSm", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=INK)
    add("Cell", fontName="Helvetica", fontSize=8, leading=11, textColor=TEXT)
    add("CellMuted", fontName="Helvetica", fontSize=8, leading=11, textColor=MUTED)
    add("AdvisorLead", fontName="Helvetica", fontSize=10, leading=16, textColor=TEXT)
    add("AdvisorBullet", fontName="Helvetica", fontSize=9, leading=13.5, textColor=TEXT,
        leftIndent=10, bulletIndent=0)
    return ss


# ── formatting helpers ───────────────────────────────────────────────────────
def _p(x, d=1):
    return "-" if x is None else f"{x*100:.{d}f}%"


def _n(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def _signed_p(x, d=1):
    if x is None:
        return "-"
    return f"{'+' if x >= 0 else ''}{x*100:.{d}f}%"


# ── chart builders ───────────────────────────────────────────────────────────
def _fig_to_image(fig, width_in: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_in * inch
    img.drawHeight = width_in * inch * ratio
    return img


def _donut(holdings, width_in=2.9):
    items = [(h["ticker"], h["weight"]) for h in holdings if h.get("weight")]
    items.sort(key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    sizes = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    if sizes:
        cols = (M_SERIES * ((len(sizes) // len(M_SERIES)) + 1))[:len(sizes)]
        wedges, *_ = ax.pie(
            sizes, startangle=90, counterclock=False,
            wedgeprops=dict(width=0.40, edgecolor="white", linewidth=1.5),
            colors=cols,
        )
        ax.legend(wedges, [f"{l}  {s*100:.0f}%" for l, s in items],
                  loc="center", fontsize=7.5, frameon=False, ncol=1,
                  bbox_to_anchor=(0.5, 0.5), handlelength=1.0, labelspacing=0.35)
    ax.set(aspect="equal")
    return _fig_to_image(fig, width_in)


def _sector_bar(sector_w, width_in=3.2):
    items = list(sector_w.items())[:8][::-1]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    if items:
        names = [k if len(k) <= 22 else k[:21] + "…" for k, _ in items]
        vals = [v * 100 for _, v in items]
        ax.barh(names, vals, color=M_TEAL, height=0.62)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.xaxis.grid(True); ax.yaxis.grid(False)
        ax.tick_params(labelsize=8, length=0)
        ax.set_xlabel("Weight (%)", fontsize=7.5)
    return _fig_to_image(fig, width_in)


def _risk_bars(cur, prop, width_in=6.6):
    metrics = [("Ann. return", "annualized_return", 100),
               ("Volatility", "annualized_vol", 100),
               ("Sharpe", "sharpe", 1),
               ("Max drawdown", "max_drawdown", 100)]
    labels = [m[0] for m in metrics]
    cv = [(cur.get(m[1]) or 0) * m[2] for m in metrics]
    pv = [(prop.get(m[1]) or 0) * m[2] for m in metrics]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    ax.bar([i - 0.205 for i in x], cv, width=0.4, label="Current", color=M_GREY)
    ax.bar([i + 0.205 for i in x], pv, width=0.4, label="Proposed", color=M_TEAL)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.axhline(0, color="#c4ccd9", linewidth=0.9)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    ax.tick_params(length=0)
    ax.yaxis.grid(True); ax.xaxis.grid(False)
    return _fig_to_image(fig, width_in)


def _score_strip(strategies, width_in=1.7):
    """Mini horizontal bar of the three strategy combined scores for a holding."""
    order = ["technical", "fundamental", "entropy"]
    vals = [((strategies.get(k) or {}).get("combined")) for k in order]
    fig, ax = plt.subplots(figsize=(2.6, 1.05))
    ys = range(len(order))
    cols = [(_hex(_score_color(v)) if v is not None else "#9aa1b0") for v in vals]
    ax.barh(list(ys), [(v or 0) for v in vals], color=cols, height=0.6)
    ax.set_xlim(0, 1)
    ax.set_yticks(list(ys)); ax.set_yticklabels(["Tech", "Fund", "Entr"], fontsize=7)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", ".5", "1"], fontsize=6)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    ax.invert_yaxis()
    return _fig_to_image(fig, width_in)


def _stress_chart(stress, width_in=6.6):
    series = stress.get("series") if isinstance(stress, dict) else None
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    for i, (name, pts) in enumerate(series.items()):
        ax.plot([p[0] for p in pts], [p[1] * 100 for p in pts],
                label=name, linewidth=1.8, color=M_SERIES[i % len(M_SERIES)])
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_ylabel("Cum. return (%)", fontsize=8)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    ax.tick_params(labelsize=7.5, length=0)
    return _fig_to_image(fig, width_in)


# ── reusable flowables ───────────────────────────────────────────────────────
class Rule(Flowable):
    def __init__(self, width, thickness=0.6, color=HAIR, space_before=0, space_after=0):
        super().__init__()
        self.width = width; self.thickness = thickness; self.color = color
        self.sb = space_before; self.sa = space_after
        self.height = thickness + space_before + space_after

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        y = self.sa
        self.canv.line(0, y, self.width, y)


def _section(ss, kicker, title, content_w):
    """Kicker + title + accent rule as a grouped header (returns a flowable list)."""
    return [
        Paragraph(kicker.upper(), ss["Kicker"]),
        Paragraph(title, ss["H2"]),
        Spacer(1, 3),
        HRFlowable(width=content_w, thickness=1.4, color=ACCENT,
                   spaceBefore=0, spaceAfter=8),
    ]


def _stat_panel(ss, stats, content_w):
    """Julius-Baer-style 'key figures' band: a row of label/value cells on a tint."""
    n = len(stats)
    cells = []
    for label, value, color in stats:
        v = Paragraph(value, ParagraphStyle("sv", parent=ss["StatValueSm"],
                                            textColor=color or INK))
        l = Paragraph(label.upper(), ss["Label"])
        cells.append([v, l])
    # transpose into a single-row table of stacked cells
    col_w = content_w / n
    data = [[_stack(c) for c in cells]]
    t = Table(data, colWidths=[col_w] * n)
    style = [
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(1, n):
        style.append(("LINEBEFORE", (i, 0), (i, 0), 0.6, HAIR))
    t.setStyle(TableStyle(style))
    return t


def _stack(rows):
    """Inner micro-table stacking value over label with no padding."""
    t = Table([[rows[0]], [rows[1]]], colWidths=[None])
    t.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (0, 0), 3),
        ("BOTTOMPADDING", (0, 1), (0, 1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def _advisor_panel(ss, advisor, content_w):
    """Editorial Advisor's View: accent left bar + tinted panel with stance,
    conviction badge, key points, and recommended posture."""
    conviction = (advisor.get("conviction") or "moderate").lower()
    conv_color = {"high": GREEN, "moderate": ACCENT, "low": AMBER, "cautious": AMBER}.get(conviction, ACCENT)

    inner = []
    head = Table([[
        Paragraph("ADVISOR'S VIEW", ParagraphStyle("av", parent=ss["Kicker"], textColor=WHITE, fontSize=9)),
        Paragraph(f"{conviction.capitalize()} conviction",
                  ParagraphStyle("cv", parent=ss["Label"], textColor=WHITE, alignment=TA_RIGHT)),
    ]], colWidths=[content_w * 0.6 - 24, content_w * 0.4 - 12])
    head.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (0, 0), 14), ("RIGHTPADDING", (-1, 0), (-1, 0), 14),
    ]))
    inner.append(head)

    body = [Paragraph(advisor.get("stance", ""), ss["AdvisorLead"]), Spacer(1, 7)]
    pts = advisor.get("key_points") or []
    if pts:
        body.append(Paragraph("What matters most", ss["H3"]))
        body.append(Spacer(1, 3))
        for pt in pts:
            body.append(Paragraph(f"•&nbsp;&nbsp;{pt}", ss["AdvisorBullet"]))
            body.append(Spacer(1, 2))
    posture = advisor.get("recommended_posture")
    if posture:
        body.append(Spacer(1, 5))
        pb = Table([[Paragraph("<b>Recommended posture:</b>&nbsp; " + posture, ss["BodyTight"])]],
                   colWidths=[content_w - 28])
        pb.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PANEL2),
            ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LINEBEFORE", (0, 0), (0, 0), 2.5, conv_color),
        ]))
        body.append(pb)

    body_tbl = Table([[body]], colWidths=[content_w])
    body_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 11), ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    return KeepTogether([inner[0], body_tbl])


# ── page furniture ───────────────────────────────────────────────────────────
def _make_header_footer(meta):
    def draw(canvas, doc):
        canvas.saveState()
        # header
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(INK)
        canvas.drawString(MARGIN, PAGE_H - 0.5 * inch, meta["portfolio_name"][:60])
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.5 * inch,
                               f"Portfolio Analysis  ·  {meta['as_of']}")
        canvas.setStrokeColor(HAIR); canvas.setLineWidth(0.6)
        canvas.line(MARGIN, PAGE_H - 0.58 * inch, PAGE_W - MARGIN, PAGE_H - 0.58 * inch)
        # footer
        canvas.setStrokeColor(HAIR)
        canvas.line(MARGIN, 0.62 * inch, PAGE_W - MARGIN, 0.62 * inch)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(FAINT)
        canvas.drawString(MARGIN, 0.46 * inch,
                          "Confidential — decision-support, not investment advice.")
        canvas.drawCentredString(PAGE_W / 2, 0.46 * inch, "Generated by the AI Portfolio Analyst")
        canvas.drawRightString(PAGE_W - MARGIN, 0.46 * inch, f"Page {doc.page - 1}")
        canvas.restoreState()
    return draw


def _make_cover(meta):
    def draw(canvas, doc):
        canvas.saveState()
        # top navy band
        band_h = 3.6 * inch
        canvas.setFillColor(INK)
        canvas.rect(0, PAGE_H - band_h, PAGE_W, band_h, fill=1, stroke=0)
        # accent rule
        canvas.setFillColor(ACCENT)
        canvas.rect(0, PAGE_H - band_h - 6, PAGE_W, 6, fill=1, stroke=0)
        # geometric accent (thin gold ticks)
        canvas.setStrokeColor(GOLD); canvas.setLineWidth(2)
        for i in range(6):
            x = PAGE_W - MARGIN - i * 9
            canvas.line(x, PAGE_H - band_h + 26, x, PAGE_H - band_h + 26 + (i + 1) * 7)
        # title block on the band
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(MARGIN, PAGE_H - 1.05 * inch, "PORTFOLIO ANALYSIS REPORT")
        canvas.setFont("Helvetica-Bold", 30)
        canvas.drawString(MARGIN, PAGE_H - 1.7 * inch, meta["portfolio_name"][:34])
        canvas.setFillColor(colors.HexColor("#aeb9cf"))
        canvas.setFont("Helvetica", 12)
        canvas.drawString(MARGIN, PAGE_H - 2.05 * inch, f"As of {meta['as_of']}")
        if meta.get("regime"):
            canvas.setFont("Helvetica", 9.5)
            canvas.drawString(MARGIN, PAGE_H - 2.42 * inch,
                              f"Market regime:  {meta['regime']}")
        # "In this report" contents list in the lower area
        cy = PAGE_H - 4.7 * inch
        canvas.setFillColor(ACCENT)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(MARGIN, cy, "IN THIS REPORT")
        canvas.setStrokeColor(HAIR); canvas.setLineWidth(0.6)
        canvas.line(MARGIN, cy - 6, MARGIN + 2.1 * inch, cy - 6)
        contents = meta.get("contents") or []
        canvas.setFont("Helvetica", 10.5)
        for i, item in enumerate(contents):
            yy = cy - 26 - i * 19
            canvas.setFillColor(GOLD)
            canvas.setFont("Helvetica-Bold", 9)
            canvas.drawString(MARGIN, yy, f"{i+1:02d}")
            canvas.setFillColor(TEXT)
            canvas.setFont("Helvetica", 10.5)
            canvas.drawString(MARGIN + 0.32 * inch, yy, item)
        # footer of cover
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(MARGIN, 0.7 * inch, "Generated by the AI Portfolio Analyst")
        canvas.drawRightString(PAGE_W - MARGIN, 0.7 * inch,
                               "Decision-support — not investment advice")
        canvas.setStrokeColor(HAIR); canvas.setLineWidth(0.6)
        canvas.line(MARGIN, 0.85 * inch, PAGE_W - MARGIN, 0.85 * inch)
        canvas.restoreState()
    return draw


# ── main entry ───────────────────────────────────────────────────────────────
def build_report_pdf(data: dict) -> bytes:
    ss = _styles()
    content_w = PAGE_W - 2 * MARGIN
    nar = data.get("narrative", {}) or {}
    advisor = data.get("advisor_view", {}) or {}
    holdings = data.get("holdings", [])
    cur = data.get("risk_current", {}) or {}
    prop = data.get("risk_proposed", {}) or {}

    meta = {
        "portfolio_name": data.get("portfolio_name", "Portfolio"),
        "as_of": data.get("as_of", ""),
        "regime": (data.get("regime") or {}).get("label"),
    }
    contents = ["Executive summary"]
    if advisor.get("stance"):
        contents.append("Advisor's view")
    contents += ["Holdings & allocation", "Risk analytics", "Proposed actions"]
    if (data.get("stress_test") or {}).get("note") or (data.get("stress_test") or {}).get("series"):
        contents.append("Stress test")
    meta["contents"] = contents

    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=0.75 * inch, bottomMargin=0.8 * inch,
        title=f"Portfolio Analysis — {meta['portfolio_name']}",
        author="AI Portfolio Analyst",
    )
    cover_frame = Frame(0, 0, PAGE_W, PAGE_H, id="cover",
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    content_frame = Frame(MARGIN, 0.72 * inch, content_w,
                          PAGE_H - 0.72 * inch - 0.95 * inch, id="content")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[cover_frame], onPage=_make_cover(meta)),
        PageTemplate(id="content", frames=[content_frame], onPage=_make_header_footer(meta)),
    ])

    story = [NextPageTemplate("content"), PageBreak()]

    # ── Executive summary + key figures ──────────────────────────────────────
    story += _section(ss, "Overview", "Executive summary", content_w)
    story.append(Paragraph(nar.get("exec_summary", "Summary unavailable."), ss["Body"]))
    story.append(Spacer(1, 10))
    watch_n = len(data.get("watch_items") or [])
    story.append(_stat_panel(ss, [
        ("Overall score", _n(data.get("overall_posture_score") or _avg_overall(holdings)), _score_color(_avg_overall(holdings))),
        ("Holdings", str(len(holdings)), INK),
        ("Sharpe (cur → prop)", f"{_n(cur.get('sharpe'))} → {_n(prop.get('sharpe'))}", INK),
        ("Concentration", _n(cur.get("hhi"), 2), INK),
        ("Watch items", str(watch_n), RED if watch_n else INK),
    ], content_w))
    story.append(Spacer(1, 16))

    # ── Advisor's View ───────────────────────────────────────────────────────
    if advisor.get("stance"):
        story.append(KeepTogether(_section(ss, "The analyst's take", "Advisor's view", content_w)
                                  + [_advisor_panel(ss, advisor, content_w)]))
        story.append(Spacer(1, 16))

    # ── Allocation ───────────────────────────────────────────────────────────
    charts = Table([[_donut(holdings), _sector_bar(cur.get("sector_weights", {}))]],
                   colWidths=[content_w * 0.46, content_w * 0.54])
    charts.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(KeepTogether(_section(ss, "Composition", "Holdings & allocation", content_w) + [charts]))
    story.append(Spacer(1, 6))
    story.append(_scorecard_table(ss, holdings, content_w))
    watch = data.get("watch_items") or []
    if watch:
        story.append(Spacer(1, 5))
        story.append(_callout(ss, "Watch", ", ".join(watch) +
                              " — management language deteriorating across recent calls; review before adding.",
                              AMBER, content_w))

    # ── Per-holding detail cards ─────────────────────────────────────────────
    detailed = [h for h in holdings if (h.get("llm") or {}).get("key_positives") or (h.get("llm") or {}).get("key_risks")]
    if detailed:
        story.append(Spacer(1, 14))
        story += _section(ss, "Position detail", "Holding scorecards", content_w)
        for h in detailed:
            story.append(_holding_card(ss, h, content_w))
            story.append(Spacer(1, 8))

    # ── Risk analytics ───────────────────────────────────────────────────────
    story.append(PageBreak())
    story += _section(ss, "Risk", "Risk analytics", content_w)
    story.append(_risk_bars(cur, prop))
    story.append(Spacer(1, 6))
    story.append(_risk_table(ss, cur, prop, content_w))
    if nar.get("risk_commentary"):
        story.append(Spacer(1, 8))
        story.append(Paragraph(nar["risk_commentary"], ss["Body"]))

    # ── Proposed actions ─────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(KeepTogether(
        _section(ss, "Recommendations", "Proposed actions", content_w) + [
            Paragraph(
                f"Optimizer: <b>{data.get('optimizer', 'MVO')}</b>. Proposals are advisory — each "
                f"ties to the holding's score and is yours to accept, adjust, or reject.", ss["Small"]),
            Spacer(1, 6),
        ]))
    story.append(_actions_table(ss, data.get("actions", []), content_w))

    # ── Stress test ──────────────────────────────────────────────────────────
    stress_img = _stress_chart(data.get("stress_test") or {})
    note = (data.get("stress_test") or {}).get("note")
    if stress_img is not None or note:
        story.append(Spacer(1, 16))
        block = _section(ss, "Resilience", "Stress test", content_w)
        if stress_img is not None:
            block.append(stress_img)
        if note:
            block.append(Spacer(1, 4))
            block.append(Paragraph(note, ss["Small"]))
        story.append(KeepTogether(block))

    # ── Closing + caveats ────────────────────────────────────────────────────
    if nar.get("closing"):
        story.append(Spacer(1, 14))
        story.append(Paragraph(nar["closing"], ss["Body"]))
    story.append(Spacer(1, 12))
    story.append(Rule(content_w, 0.6, HAIR, space_after=6))
    story.append(Paragraph("IMPORTANT INFORMATION", ss["Label"]))
    story.append(Spacer(1, 3))
    story.append(Paragraph(data.get("disclaimer", _DEFAULT_DISCLAIMER), ss["Tiny"]))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ── table/card builders ──────────────────────────────────────────────────────
def _avg_overall(holdings):
    vals = [h.get("overall_score") for h in holdings if h.get("overall_score") is not None]
    return sum(vals) / len(vals) if vals else None


def _callout(ss, tag, text, color, content_w):
    t = Table([[Paragraph(f"<b>{tag}:</b>&nbsp; {text}", ss["Small"])]], colWidths=[content_w])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("LINEBEFORE", (0, 0), (0, 0), 2.5, color),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _scorecard_table(ss, holdings, content_w):
    head = ["Ticker", "Name", "Weight", "Overall", "Tech", "Fund", "Entr", "Drift"]
    rows = [head]
    for h in holdings:
        st = h.get("strategies", {})
        rows.append([
            h["ticker"],
            Paragraph((h.get("company") or "")[:26], ss["CellMuted"]),
            _p(h.get("weight")),
            _n(h.get("overall_score")),
            _n((st.get("technical") or {}).get("combined")),
            _n((st.get("fundamental") or {}).get("combined")),
            _n((st.get("entropy") or {}).get("combined")),
            (h.get("drift_trend") or "-").title(),
        ])
    cw = [0.7, 1.9, 0.75, 0.75, 0.62, 0.62, 0.62, 1.05]
    scale = content_w / (sum(cw) * inch)
    t = Table(rows, colWidths=[c * inch * scale for c in cw], repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 1), (-1, -1), TEXT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PANEL]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, HAIR),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (0, -1), 8),
    ]
    for i, h in enumerate(holdings, start=1):
        style.append(("TEXTCOLOR", (3, i), (3, i), _score_color(h.get("overall_score"))))
        style.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))
        if (h.get("drift_trend") or "") == "DETERIORATING":
            style.append(("TEXTCOLOR", (7, i), (7, i), RED))
        elif (h.get("drift_trend") or "") == "IMPROVING":
            style.append(("TEXTCOLOR", (7, i), (7, i), GREEN))
    t.setStyle(TableStyle(style))
    return t


def _holding_card(ss, h, content_w):
    llm = h.get("llm") or {}
    pos = llm.get("key_positives") or []
    risks = llm.get("key_risks") or []
    drift = h.get("drift_trend") or "-"
    drift_color = RED if drift == "DETERIORATING" else GREEN if drift == "IMPROVING" else MUTED

    # left: identity + score strip ; right: positives/risks
    left = [
        Paragraph(f"<b>{h['ticker']}</b>  <font size=8 color='#6a7180'>{(h.get('company') or '')[:24]}</font>", ss["H3"]),
        Spacer(1, 2),
        Paragraph(f"Overall <b><font color='{_hex(_score_color(h.get('overall_score')))}'>{_n(h.get('overall_score'))}</font></b>"
                  f"&nbsp;&nbsp;·&nbsp;&nbsp;Weight {_p(h.get('weight'))}", ss["Small"]),
        Spacer(1, 4),
        _score_strip(h.get("strategies", {})),
        Spacer(1, 2),
        Paragraph(f"Drift: <font color='{_hex(drift_color)}'>{drift.title()}</font>", ss["Tiny"]),
    ]
    rcell = []
    if pos:
        rcell.append(Paragraph("Key positives", ParagraphStyle("kp", parent=ss["Label"], textColor=GREEN)))
        for p in pos[:3]:
            rcell.append(Paragraph(f"+ {p}", ss["Cell"]))
        rcell.append(Spacer(1, 3))
    if risks:
        rcell.append(Paragraph("Key risks", ParagraphStyle("kr", parent=ss["Label"], textColor=RED)))
        for r in risks[:3]:
            rcell.append(Paragraph(f"– {r}", ss["Cell"]))
    if not rcell:
        rcell.append(Paragraph("No semantic detail available.", ss["CellMuted"]))

    card = Table([[left, rcell]], colWidths=[content_w * 0.36, content_w * 0.64])
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LINEBEFORE", (0, 0), (0, 0), 2.5, _score_color(h.get("overall_score"))),
        ("LINEAFTER", (0, 0), (0, 0), 0.6, HAIR),
    ]))
    return KeepTogether([card])


def _risk_table(ss, cur, prop, content_w):
    rows = [["Metric", "Current", "Proposed", ""]]
    spec = [("Annualized return", "annualized_return", _p),
            ("Annualized volatility", "annualized_vol", _p),
            ("Sharpe ratio", "sharpe", lambda x: _n(x)),
            ("Max drawdown", "max_drawdown", _p),
            ("Concentration (HHI)", "hhi", lambda x: _n(x, 2)),
            ("Positions", "n_positions", lambda x: str(x) if x is not None else "-")]
    for label, key, fmt in spec:
        c, p = cur.get(key), prop.get(key)
        better = ""
        if isinstance(c, (int, float)) and isinstance(p, (int, float)) and key in ("sharpe", "annualized_vol", "max_drawdown", "hhi"):
            improved = (p > c) if key == "sharpe" else (p > c if key == "max_drawdown" else p < c)
            better = "improved" if improved else ""
        rows.append([label, fmt(c), fmt(p), better])
    cw = [2.5, 1.5, 1.5, 1.0]
    scale = content_w / (sum(cw) * inch)
    t = Table(rows, colWidths=[c * inch * scale for c in cw], hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 1), (-1, -1), TEXT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PANEL]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (0, -1), 8),
        ("FONTSIZE", (3, 1), (3, -1), 7), ("TEXTCOLOR", (3, 1), (3, -1), GREEN),
    ]
    t.setStyle(TableStyle(style))
    return t


def _actions_table(ss, actions, content_w):
    rows = [["Ticker", "Action", "Weight Δ", "Rationale"]]
    for a in actions:
        rows.append([a["ticker"], a["action"], _signed_p(a.get("delta")),
                     Paragraph(a.get("rationale", ""), ss["Cell"])])
    cw = [0.8, 0.85, 0.9, 4.0]
    scale = content_w / (sum(cw) * inch)
    t = Table(rows, colWidths=[c * inch * scale for c in cw], repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"), ("TEXTCOLOR", (0, 1), (-1, -1), TEXT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PANEL]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("ALIGN", (1, 0), (2, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (0, -1), 8),
    ]
    ac = {"ADD": GREEN, "TRIM": AMBER, "EXIT": RED, "HOLD": MUTED}
    for i, a in enumerate(actions, start=1):
        style.append(("TEXTCOLOR", (1, i), (1, i), ac.get(a["action"], MUTED)))
        style.append(("FONTNAME", (1, i), (1, i), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


_DEFAULT_DISCLAIMER = (
    "This report is decision-support, not investment advice, and does not constitute a personal "
    "recommendation. Scores derive from a published research framework (Cohen, Aiche &amp; Eichel, "
    "2025) whose live predictive power on current data is still being validated; treat them as one "
    "input among many, not a verdict. Backtested figures (2020-2025) represent past performance, "
    "which does not guarantee future results. Proposed weights are model output, distinct from the "
    "source paper's equal-weight selection. The Advisor's View is an automated interpretation of the "
    "data shown and may be incomplete or wrong. All investment decisions rest with the reader."
)
