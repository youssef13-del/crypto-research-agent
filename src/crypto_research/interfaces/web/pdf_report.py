"""Deterministic, in-memory PDF exports for stored ChainScope research."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from io import BytesIO
from math import isfinite
from threading import Lock
from typing import Any, Literal

from pymupdf_fonts import myfont  # type: ignore[import-untyped]
from reportlab.graphics.shapes import (  # type: ignore[import-untyped]
    Drawing,
    Line,
    PolyLine,
    Rect,
    String,
)
from reportlab.lib import colors  # type: ignore[import-untyped]
from reportlab.lib.enums import TA_CENTER, TA_LEFT  # type: ignore[import-untyped]
from reportlab.lib.pagesizes import A4  # type: ignore[import-untyped]
from reportlab.lib.styles import (  # type: ignore[import-untyped]
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import mm  # type: ignore[import-untyped]
from reportlab.pdfbase import pdfmetrics  # type: ignore[import-untyped]
from reportlab.pdfbase.ttfonts import TTFont  # type: ignore[import-untyped]
from reportlab.platypus import (  # type: ignore[import-untyped]
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from crypto_research.domain.evidence import DerivativesEvidence
from crypto_research.domain.history import StoredResearchRun
from crypto_research.domain.market import MarketEvidence
from crypto_research.domain.research import ResearchReport
from crypto_research.interfaces.web.presentation import build_research_presentation
from crypto_research.interfaces.web.runtime import AgentPanelPresentation, ResearchPresentation
from crypto_research.shared.security import clean_text, normalize_http_url, redact_secrets

_MAX_REPORTS = 20
_FONT_REGISTRATION_LOCK = Lock()
_FONTS_REGISTERED = False
_FONT_REGULAR = "ChainScopeNotoSans"
_FONT_BOLD = "ChainScopeNotoSansBold"
_INK = colors.HexColor("#172033")
_MUTED = colors.HexColor("#5F6B82")
_SUBTLE = colors.HexColor("#8791A5")
_BORDER = colors.HexColor("#DDE2EA")
_SURFACE = colors.HexColor("#F6F7FB")
_ACCENT = colors.HexColor("#6D4AFF")
_BLUE = colors.HexColor("#1976D2")
_POSITIVE = colors.HexColor("#11875D")
_AMBER = colors.HexColor("#B86B00")
_SERIES_COLORS = ("#6D4AFF", "#1976D2", "#0C9B77", "#D15C1F", "#9A3CB7")
_SAFE_FILENAME = re.compile(r"[^a-z0-9]+")


def render_research_pdf(
    runs: Sequence[StoredResearchRun],
    *,
    prepared_for: str | None = None,
    generated_at: datetime | None = None,
) -> bytes:
    """Render one to twenty stored reports without fetching new data."""

    ordered = _validated_runs(runs)
    generated = _utc(generated_at or datetime.now(UTC))
    prepared_name = _safe_text(prepared_for or "", limit=120) or None
    _register_fonts()
    output = BytesIO()
    title = (
        _report_title(ordered[0])
        if len(ordered) == 1
        else f"ChainScope Research Bundle ({len(ordered)} reports)"
    )
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title=title,
        author="ChainScope",
        subject="Evidence-backed cryptocurrency research",
        creator="ChainScope research workspace",
        pageCompression=1,
    )
    styles = _styles()
    story: list[Any] = []
    if len(ordered) > 1:
        story.extend(_bundle_cover(ordered, prepared_name, generated, styles))
        story.append(PageBreak())
    for index, stored in enumerate(ordered):
        if index:
            story.append(PageBreak())
        story.extend(_report_story(stored, prepared_name, generated, styles))

    def draw_page(canvas: Any, doc: Any) -> None:
        _page_chrome(canvas, doc, generated=generated)

    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return output.getvalue()


def research_pdf_filename(
    runs: Sequence[StoredResearchRun],
    *,
    generated_at: datetime | None = None,
) -> str:
    """Return a stable, filesystem-safe PDF download name."""

    ordered = _validated_runs(runs)
    generated = _utc(generated_at or datetime.now(UTC))
    date = generated.strftime("%Y%m%d")
    if len(ordered) > 1:
        return f"chainscope-research-bundle-{date}.pdf"
    assets = "-".join(ordered[0].summary.assets) or "market-research"
    slug = _SAFE_FILENAME.sub("-", assets.casefold()).strip("-")[:60] or "research"
    return f"chainscope-{slug}-{date}.pdf"


def _validated_runs(runs: Sequence[StoredResearchRun]) -> tuple[StoredResearchRun, ...]:
    if not runs:
        raise ValueError("PDF export requires at least one stored research report.")
    if len(runs) > _MAX_REPORTS:
        raise ValueError("PDF export is limited to 20 stored research reports.")
    unique: dict[str, StoredResearchRun] = {}
    for stored in runs:
        unique.setdefault(stored.summary.id, stored)
    if len(unique) != len(runs):
        raise ValueError("PDF export cannot contain duplicate research runs.")
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: _utc(item.summary.completed_at or item.summary.created_at),
            reverse=True,
        )
    )


def _register_fonts() -> None:
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    with _FONT_REGISTRATION_LOCK:
        if _FONTS_REGISTERED:
            return
        pdfmetrics.registerFont(TTFont(_FONT_REGULAR, BytesIO(myfont("notos"))))
        pdfmetrics.registerFont(TTFont(_FONT_BOLD, BytesIO(myfont("notosbo"))))
        pdfmetrics.registerFontFamily(
            "ChainScopeNoto",
            normal=_FONT_REGULAR,
            bold=_FONT_BOLD,
            italic=_FONT_REGULAR,
            boldItalic=_FONT_BOLD,
        )
        _FONTS_REGISTERED = True


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover": ParagraphStyle(
            "ChainScopeCover",
            parent=base["Title"],
            fontName=_FONT_BOLD,
            fontSize=25,
            leading=31,
            textColor=_INK,
            spaceAfter=12,
        ),
        "eyebrow": ParagraphStyle(
            "ChainScopeEyebrow",
            parent=base["Normal"],
            fontName=_FONT_BOLD,
            fontSize=8,
            leading=10,
            textColor=_ACCENT,
            spaceAfter=7,
        ),
        "h1": ParagraphStyle(
            "ChainScopeH1",
            parent=base["Heading1"],
            fontName=_FONT_BOLD,
            fontSize=18,
            leading=23,
            textColor=_INK,
            spaceBefore=5,
            spaceAfter=9,
        ),
        "h2": ParagraphStyle(
            "ChainScopeH2",
            parent=base["Heading2"],
            fontName=_FONT_BOLD,
            fontSize=12,
            leading=16,
            textColor=_INK,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "ChainScopeH3",
            parent=base["Heading3"],
            fontName=_FONT_BOLD,
            fontSize=9.5,
            leading=13,
            textColor=_BLUE,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "ChainScopeBody",
            parent=base["BodyText"],
            fontName=_FONT_REGULAR,
            fontSize=8.7,
            leading=13,
            textColor=_INK,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "ChainScopeSmall",
            parent=base["BodyText"],
            fontName=_FONT_REGULAR,
            fontSize=7.2,
            leading=10,
            textColor=_MUTED,
            spaceAfter=3,
        ),
        "metric": ParagraphStyle(
            "ChainScopeMetric",
            parent=base["BodyText"],
            fontName=_FONT_BOLD,
            fontSize=8,
            leading=11,
            textColor=_INK,
            alignment=TA_LEFT,
        ),
        "center": ParagraphStyle(
            "ChainScopeCenter",
            parent=base["BodyText"],
            fontName=_FONT_REGULAR,
            fontSize=8,
            leading=11,
            textColor=_MUTED,
            alignment=TA_CENTER,
        ),
    }


def _bundle_cover(
    runs: tuple[StoredResearchRun, ...],
    prepared_for: str | None,
    generated_at: datetime,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    story: list[Any] = [
        Spacer(1, 17 * mm),
        _p("CHAIN SCOPE / RESEARCH BUNDLE", styles["eyebrow"]),
        _p("Evidence-backed research, organized for review", styles["cover"]),
        _p(
            f"{len(runs)} saved reports generated from validated stored evidence. "
            "No market data or analysis was refreshed during export.",
            styles["body"],
        ),
        Spacer(1, 7 * mm),
        _metadata_table(
            (
                ("Prepared for", prepared_for or "Private ChainScope workspace"),
                ("Generated", _date_time(generated_at)),
                ("Reports", str(len(runs))),
                ("Newest research", _date_time(_run_time(runs[0]))),
            ),
            styles,
        ),
        Spacer(1, 9 * mm),
        _p("Bundle index", styles["h2"]),
    ]
    rows: list[list[Any]] = [
        [
            _p("#", styles["small"]),
            _p("Assets", styles["small"]),
            _p("Completed", styles["small"]),
            _p("Status", styles["small"]),
        ]
    ]
    for index, stored in enumerate(runs, start=1):
        rows.append(
            [
                _p(str(index), styles["body"]),
                _p(", ".join(stored.summary.assets) or "Market discovery", styles["body"]),
                _p(_date_time(_run_time(stored)), styles["body"]),
                _p(stored.summary.state.title(), styles["body"]),
            ]
        )
    story.append(_table(rows, (12 * mm, 66 * mm, 52 * mm, 28 * mm), header=True))
    story.append(Spacer(1, 7 * mm))
    story.append(_p("Educational research only. Not financial advice.", styles["small"]))
    return story


def _report_story(
    stored: StoredResearchRun,
    prepared_for: str | None,
    generated_at: datetime,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    report = stored.report
    route = tuple(item.agent for item in report.agent_statuses)
    presentation = build_research_presentation(report, route=route)
    assets = ", ".join(stored.summary.assets) or "Market discovery"
    story: list[Any] = [
        _p("CHAIN SCOPE / SAVED RESEARCH", styles["eyebrow"]),
        _p(_report_title(stored), styles["cover"]),
        _p(stored.summary.question, styles["body"]),
        Spacer(1, 3 * mm),
        _metadata_table(
            (
                ("Assets", assets),
                ("Status", stored.summary.state.title()),
                ("Evidence confidence", f"{report.evidence_confidence:.0%}"),
                ("Research completed", _date_time(_run_time(stored))),
                ("Exchange / timeframe", _exchange_timeframe(stored)),
                ("Prepared for", prepared_for or "Private ChainScope workspace"),
                ("Export generated", _date_time(generated_at)),
                ("Report ID", stored.summary.id),
            ),
            styles,
        ),
    ]
    story.extend(_executive_summary(presentation, styles))
    story.extend(_coverage_section(report, styles))
    story.extend(_data_sections(presentation, styles))
    story.extend(_chart_sections(report, styles))
    story.extend(_agent_sections(presentation, styles))
    story.extend(_risk_and_limitations(presentation, report, styles))
    story.extend(_sources_section(presentation, styles))
    story.append(Spacer(1, 4 * mm))
    story.append(_p(report.disclaimer, styles["small"]))
    return story


def _executive_summary(
    presentation: ResearchPresentation,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    story: list[Any] = [_p("Executive analysis", styles["h1"])]
    available = [panel for panel in presentation.agent_panels if panel.answer is not None]
    if not available:
        story.append(_p("No specialist narrative was stored for this report.", styles["body"]))
        return story
    rows: list[list[Any]] = []
    for panel in available:
        assert panel.answer is not None
        analysis = panel.answer.structured_analysis
        verdict = analysis.verdict if analysis is not None else panel.answer.answer
        rows.append(
            [
                _p(panel.title, styles["metric"]),
                _p(verdict, styles["body"]),
                _p(f"{panel.answer.confidence:.0%}", styles["center"]),
            ]
        )
    story.append(_table(rows, (42 * mm, 103 * mm, 16 * mm), header=False))
    return story


def _coverage_section(report: ResearchReport, styles: dict[str, ParagraphStyle]) -> list[Any]:
    coverage = report.evidence_coverage_summary
    capability_count = len(
        {item.capability for item in coverage.entries}
        | {item.capability for item in report.capability_coverage}
    )
    story: list[Any] = [_p("Evidence coverage", styles["h2"])]
    totals = (
        ("Collected", str(coverage.total_collected_records)),
        ("Accepted", str(coverage.total_accepted_records)),
        ("Excluded", str(coverage.total_excluded_records)),
        ("Capabilities", str(capability_count)),
    )
    story.append(_metadata_table(totals, styles))
    if coverage.entries:
        rows: list[list[Any]] = [
            [
                _p("Asset", styles["small"]),
                _p("Topic", styles["small"]),
                _p("Accepted", styles["small"]),
                _p("Providers", styles["small"]),
            ]
        ]
        rows.extend(
            [
                _p(entry.asset, styles["body"]),
                _p(entry.capability.value.title(), styles["body"]),
                _p(str(entry.accepted_records), styles["body"]),
                _p(", ".join(entry.providers) or "Unavailable", styles["small"]),
            ]
            for entry in coverage.entries
        )
        story.append(_table(rows, (37 * mm, 38 * mm, 22 * mm, 64 * mm), header=True))
    return story


def _data_sections(
    presentation: ResearchPresentation,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    cards = [card for panel in presentation.agent_panels for card in panel.data]
    if not cards:
        return []
    story: list[Any] = [_p("Validated numerical snapshot", styles["h1"])]
    for card in cards:
        facts = [
            [_p(label, styles["small"]), _p(value, styles["metric"])] for label, value in card.facts
        ]
        content: list[Any] = [_p(card.title, styles["h3"])]
        if facts:
            content.append(_table(facts, (70 * mm, 91 * mm), header=False))
        else:
            content.append(_p("No validated numerical facts were stored.", styles["small"]))
        if card.limitation:
            content.append(_p("Limitation: " + card.limitation, styles["small"]))
        story.append(KeepTogether(content))
    return story


def _chart_sections(report: ResearchReport, styles: dict[str, ParagraphStyle]) -> list[Any]:
    series = list(_report_series(report))
    if not series:
        return []
    story: list[Any] = [_p("Stored quantitative trends", styles["h1"])]
    for index, (title, timestamps, values, kind) in enumerate(series):
        drawing = _vector_chart(
            timestamps,
            values,
            color=_SERIES_COLORS[index % len(_SERIES_COLORS)],
            kind=kind,
        )
        if drawing is None:
            continue
        story.append(KeepTogether([_p(title, styles["h3"]), drawing, Spacer(1, 2 * mm)]))
    return story


def _agent_sections(
    presentation: ResearchPresentation,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    if not presentation.agent_panels:
        return []
    story: list[Any] = [_p("Specialist analysis", styles["h1"])]
    for panel in presentation.agent_panels:
        story.extend(_agent_panel(panel, styles))
    return story


def _agent_panel(
    panel: AgentPanelPresentation,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    status = panel.state_label or panel.status.title()
    story: list[Any] = [
        _p(f"{panel.title} · {status}", styles["h2"]),
        _p("Topics: " + (", ".join(panel.capabilities) or "Not recorded"), styles["small"]),
    ]
    if panel.answer is None:
        story.append(_p(panel.limitation or "Analysis was unavailable.", styles["body"]))
        return story
    answer = panel.answer
    structured = answer.structured_analysis
    if structured is not None:
        story.append(_p(structured.verdict, styles["metric"]))
        for section in structured.sections:
            story.append(_p(f"{section.asset} / {section.scope.title()}", styles["h3"]))
            story.append(_p(section.text, styles["body"]))
        if structured.comparison:
            story.append(_p("Comparison: " + structured.comparison, styles["body"]))
    else:
        story.append(_p(answer.answer, styles["body"]))
        if answer.analysis:
            story.append(_p(answer.analysis, styles["body"]))
    story.append(_p(f"Confidence: {answer.confidence:.0%}", styles["small"]))
    if answer.claims:
        story.append(_p("Supported claims", styles["h3"]))
        for claim in answer.claims:
            identifiers = ", ".join(claim.evidence_ids)
            story.append(
                _p(
                    f"• {claim.statement} "
                    f"[{claim.claim_kind}; {claim.confidence:.0%}; {identifiers}]",
                    styles["small"],
                )
            )
    for label, values in (
        ("Uncertainty", answer.uncertainty),
        ("Limitations", answer.limitations),
    ):
        if values:
            story.append(_p(label + ": " + " • ".join(values), styles["small"]))
    return story


def _risk_and_limitations(
    presentation: ResearchPresentation,
    report: ResearchReport,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    risks = _unique((*presentation.risks, *presentation.limitations, *report.warnings))
    if not risks:
        return []
    story: list[Any] = [_p("Risks, warnings, and limitations", styles["h1"])]
    story.extend(_p("• " + value, styles["body"]) for value in risks)
    return story


def _sources_section(
    presentation: ResearchPresentation,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    story: list[Any] = [_p("Evidence and source appendix", styles["h1"])]
    if not presentation.sources:
        story.append(_p("No separate source records were attached.", styles["body"]))
        return story
    rows: list[list[Any]] = [
        [
            _p("Provider", styles["small"]),
            _p("Source", styles["small"]),
            _p("Time", styles["small"]),
        ]
    ]
    for source in presentation.sources:
        link = _source_link(source.title, source.url, styles)
        rows.append(
            [
                _p(f"{source.publisher}\n{source.kind}", styles["small"]),
                link,
                _p(f"{source.time_context}\n{_date_time(source.published_at)}", styles["small"]),
            ]
        )
    story.append(
        LongTable(
            rows, colWidths=(38 * mm, 83 * mm, 40 * mm), repeatRows=1, style=_table_style(True)
        )
    )
    return story


def _report_series(
    report: ResearchReport,
) -> Iterable[
    tuple[str, tuple[datetime, ...], tuple[float, ...], Literal["number", "price", "percent"]]
]:
    for market, derivatives in _market_evidence(report):
        candles = market.candles[-180:]
        if len(candles) >= 2:
            yield (
                f"{market.symbol} · completed-candle close",
                tuple(item.timestamp for item in candles),
                tuple(item.close for item in candles),
                "price",
            )
        if derivatives is not None:
            funding = derivatives.funding_history
            if len(funding) >= 2:
                yield (
                    f"{derivatives.asset} · perpetual funding rate",
                    tuple(item.observed_at for item in funding),
                    tuple(item.rate * 100 for item in funding),
                    "percent",
                )
            interest = derivatives.open_interest_history
            if len(interest) >= 2:
                yield (
                    f"{derivatives.asset} · perpetual open interest",
                    tuple(item.observed_at for item in interest),
                    tuple(item.value_usd for item in interest),
                    "number",
                )
    if report.onchain_result is not None:
        for bundle in report.onchain_result.asset_results:
            if bundle.onchain is None:
                continue
            for metric in bundle.onchain.metrics:
                if len(metric.observations) < 2:
                    continue
                yield (
                    f"{bundle.asset.symbol} · {metric.label}",
                    tuple(item.observed_at for item in metric.observations),
                    tuple(item.value for item in metric.observations),
                    "number",
                )


def _market_evidence(
    report: ResearchReport,
) -> tuple[tuple[MarketEvidence, DerivativesEvidence | None], ...]:
    if report.market_comparison_result is not None:
        return tuple(
            (item.market, item.derivatives) for item in report.market_comparison_result.assets
        )
    if report.market_result is not None:
        return ((report.market_result.market, report.market_result.derivatives),)
    return ()


def _vector_chart(
    timestamps: Sequence[datetime],
    values: Sequence[float],
    *,
    color: str,
    kind: Literal["number", "price", "percent"],
) -> Drawing | None:
    points = [
        (timestamp, float(value))
        for timestamp, value in zip(timestamps, values, strict=True)
        if isfinite(value)
    ]
    if len(points) < 2:
        return None
    width, height = 470.0, 150.0
    left, right, bottom, top = 54.0, 12.0, 24.0, 12.0
    chart_width = width - left - right
    chart_height = height - bottom - top
    minimum = min(value for _, value in points)
    maximum = max(value for _, value in points)
    spread = maximum - minimum
    if spread == 0:
        padding = max(abs(maximum) * 0.05, 1.0)
        minimum -= padding
        maximum += padding
        spread = maximum - minimum
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, rx=6, ry=6, fillColor=_SURFACE, strokeColor=_BORDER))
    for step in range(5):
        y = bottom + chart_height * step / 4
        drawing.add(Line(left, y, width - right, y, strokeColor=_BORDER, strokeWidth=0.5))
        label_value = minimum + spread * step / 4
        drawing.add(
            String(
                left - 5,
                y - 2,
                _format_chart_value(label_value, kind),
                textAnchor="end",
                fontName=_FONT_REGULAR,
                fontSize=6.5,
                fillColor=_MUTED,
            )
        )
    coordinates: list[tuple[float, float]] = []
    for index, (_, value) in enumerate(points):
        x = left + chart_width * index / (len(points) - 1)
        y = bottom + chart_height * (value - minimum) / spread
        coordinates.append((x, y))
    drawing.add(PolyLine(coordinates, strokeColor=colors.HexColor(color), strokeWidth=2.2))
    first_time, last_time = points[0][0], points[-1][0]
    drawing.add(
        String(
            left,
            7,
            first_time.strftime("%d %b %Y"),
            fontName=_FONT_REGULAR,
            fontSize=6.5,
            fillColor=_MUTED,
        )
    )
    drawing.add(
        String(
            width - right,
            7,
            last_time.strftime("%d %b %Y"),
            textAnchor="end",
            fontName=_FONT_REGULAR,
            fontSize=6.5,
            fillColor=_MUTED,
        )
    )
    latest_x, latest_y = coordinates[-1]
    drawing.add(
        Rect(
            latest_x - 2.2,
            latest_y - 2.2,
            4.4,
            4.4,
            rx=2.2,
            ry=2.2,
            fillColor=colors.HexColor(color),
            strokeColor=colors.white,
        )
    )
    return drawing


def _metadata_table(
    values: Sequence[tuple[str, str]],
    styles: dict[str, ParagraphStyle],
) -> Table:
    rows: list[list[Any]] = []
    for start in range(0, len(values), 2):
        pair = values[start : start + 2]
        row: list[Any] = []
        for label, value in pair:
            row.extend((_p(label, styles["small"]), _p(value, styles["metric"])))
        if len(pair) == 1:
            row.extend(("", ""))
        rows.append(row)
    return _table(rows, (28 * mm, 50.5 * mm, 28 * mm, 54.5 * mm), header=False)


def _table(rows: Sequence[Sequence[Any]], widths: Sequence[float], *, header: bool) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    table.setStyle(_table_style(header))
    return table


def _table_style(header: bool) -> TableStyle:
    commands: list[tuple[Any, ...]] = [
        ("FONTNAME", (0, 0), (-1, -1), _FONT_REGULAR),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), _INK),
        ("GRID", (0, 0), (-1, -1), 0.45, _BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), (colors.white, _SURFACE)),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _INK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), _FONT_BOLD),
            ]
        )
    return TableStyle(commands)


def _p(value: object, style: ParagraphStyle) -> Paragraph:
    text = _safe_text(str(value), limit=6000).replace("\n", "<br/>")
    return Paragraph(html.escape(text, quote=True), style)


def _source_link(title: str, url: str | None, styles: dict[str, ParagraphStyle]) -> Paragraph:
    label = html.escape(_safe_text(title, limit=500), quote=True)
    safe_url = normalize_http_url(url)
    if safe_url is None:
        return Paragraph(label, styles["body"])
    href = html.escape(safe_url, quote=True)
    return Paragraph(f'<link href="{href}" color="#1976D2">{label}</link>', styles["body"])


def _safe_text(value: str, *, limit: int) -> str:
    return clean_text(redact_secrets(value), max_length=limit)


def _format_chart_value(value: float, kind: str) -> str:
    if kind == "price":
        if abs(value) >= 1:
            return "$" + _compact_number(value)
        return f"${value:.4g}"
    if kind == "percent":
        return f"{value:.3g}%"
    return _compact_number(value)


def _compact_number(value: float) -> str:
    magnitude = abs(value)
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= divisor:
            return f"{value / divisor:.3g}{suffix}"
    return f"{value:.4g}"


def _report_title(stored: StoredResearchRun) -> str:
    assets = ", ".join(stored.summary.assets)
    return f"{assets} Research Report" if assets else "Market Discovery Research Report"


def _exchange_timeframe(stored: StoredResearchRun) -> str:
    values = [value for value in (stored.summary.exchange, stored.summary.timeframe) if value]
    return " / ".join(values) if values else "Topic-only research"


def _run_time(stored: StoredResearchRun) -> datetime:
    return _utc(stored.summary.completed_at or stored.summary.created_at)


def _date_time(value: datetime) -> str:
    return _utc(value).strftime("%d %b %Y %H:%M UTC")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("PDF timestamps must be timezone-aware.")
    return value.astimezone(UTC)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(value for value in (_safe_text(item, limit=1000) for item in values) if value)
    )


def _page_chrome(canvas: Any, document: Any, *, generated: datetime) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(17 * mm, height - 12 * mm, width - 17 * mm, height - 12 * mm)
    canvas.setFont(_FONT_BOLD, 7)
    canvas.setFillColor(_ACCENT)
    canvas.drawString(17 * mm, height - 9 * mm, "CHAIN SCOPE")
    canvas.setFont(_FONT_REGULAR, 6.7)
    canvas.setFillColor(_SUBTLE)
    canvas.drawRightString(width - 17 * mm, height - 9 * mm, "VERIFIED RESEARCH EXPORT")
    canvas.line(17 * mm, 12 * mm, width - 17 * mm, 12 * mm)
    canvas.drawString(17 * mm, 8 * mm, f"Generated {_date_time(generated)}")
    canvas.drawRightString(width - 17 * mm, 8 * mm, f"Page {document.page}")
    canvas.restoreState()


__all__ = ["render_research_pdf", "research_pdf_filename"]
