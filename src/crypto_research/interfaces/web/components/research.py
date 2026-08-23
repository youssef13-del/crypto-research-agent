"""Streamlit rendering for research presentations."""

from __future__ import annotations

import html
from collections import defaultdict
from datetime import UTC

import pandas as pd
import streamlit as st

from crypto_research.domain.forecast import ForecastRun
from crypto_research.domain.research import ResearchCapability
from crypto_research.interfaces.web.components.charts import format_price, render_line_chart
from crypto_research.interfaces.web.components.layout import render_data_table, render_disclaimer
from crypto_research.interfaces.web.presentation import (
    SOURCE_ICONS,
    AgentAnalysisSectionPresentation,
    AgentClaimPresentation,
    AgentPanelPresentation,
    CapabilityDataPresentation,
    DiscoveryPresentation,
    ResearchPresentation,
    ResearchTurn,
    SourcePresentation,
    StructuredAgentAnalysisPresentation,
    _agent_state_label,
    _capability_label,
)
from crypto_research.interfaces.web.runtime import AGENT_LABELS, safe_markdown
from crypto_research.interfaces.web.theme import active_palette


def render_panel_section_heading(title: str, description: str) -> None:
    st.markdown(
        '<div class="cs-panel-section-heading">'
        f"<strong>{html.escape(title)}</strong><span>{html.escape(description)}</span></div>",
        unsafe_allow_html=True,
    )


def render_claim_support(claims: tuple[AgentClaimPresentation, ...]) -> None:
    for claim in claims:
        evidence = "".join(
            f'<span class="cs-evidence-id">{html.escape(item)}</span>'
            for item in claim.evidence_ids
        )
        evidence_html = (
            f'<div class="cs-evidence-ids"><span>Evidence</span>{evidence}</div>'
            if evidence
            else '<div class="cs-evidence-ids"><span>Evidence</span>'
            "<em>No record attached</em></div>"
        )
        st.markdown(
            '<article class="cs-claim-card">'
            f"<p>{html.escape(claim.statement)}</p>{evidence_html}</article>",
            unsafe_allow_html=True,
        )


def render_capability_cards(cards: tuple[CapabilityDataPresentation, ...]) -> None:
    columns = st.columns(2) if len(cards) > 1 else (st.container(),)
    for index, card in enumerate(cards):
        labels = {"complete": "Validated", "partial": "Partial", "unavailable": "Unavailable"}
        status_label = labels.get(card.status, card.status.replace("_", " ").title())
        status_class = "is-complete" if card.status == "complete" else "is-limited"
        facts = "".join(
            '<div class="cs-resource-fact">'
            f"<dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>"
            for label, value in card.facts
        )
        if not facts:
            facts = (
                '<div class="cs-resource-fact"><dt>Data status</dt>'
                "<dd>No validated values available</dd></div>"
            )
        limitation = (
            '<div class="cs-resource-card-note"><strong>Note</strong>'
            f"<span>{html.escape(card.limitation)}</span></div>"
            if card.limitation
            else ""
        )
        with columns[index % len(columns)]:
            st.markdown(
                '<article class="cs-resource-card"><header><strong>'
                f"{html.escape(card.title)}</strong>"
                f'<span class="cs-resource-state {status_class}">{html.escape(status_label)}</span>'
                f"</header><dl>{facts}</dl>{limitation}</article>",
                unsafe_allow_html=True,
            )


def render_structured_agent_analysis(
    panel: AgentPanelPresentation,
    analysis: StructuredAgentAnalysisPresentation,
) -> None:
    live = panel.analysis_state == "live"
    kicker = (
        "Live answer | Overall view"
        if live
        else "Live evidence answer | Overall view"
        if panel.source_state == "live"
        else "Evidence answer | Overall view"
    )
    confidence = (
        f'<span class="cs-analysis-confidence">Confidence {panel.answer.confidence:.0%}</span>'
        if panel.answer is not None
        else ""
    )
    st.markdown(
        '<section class="cs-analysis-verdict" aria-label="Overall view">'
        '<div class="cs-analysis-verdict-header">'
        f'<span class="cs-analysis-kicker">{html.escape(kicker)}</span>'
        f"{confidence}</div>"
        f"<p>{html.escape(analysis.verdict)}</p></section>",
        unsafe_allow_html=True,
    )
    ordered_assets = tuple(dict.fromkeys(section.asset for section in analysis.sections))
    for asset in ordered_assets:
        sections = tuple(section for section in analysis.sections if section.asset == asset)
        _render_asset_card(panel, asset=asset, sections=sections)
    if analysis.comparison and len(ordered_assets) > 1:
        st.markdown(
            '<section class="cs-analysis-comparison" aria-label="Cross-asset comparison">'
            '<span class="cs-analysis-kicker">Cross-asset comparison</span>'
            f"<p>{html.escape(analysis.comparison)}</p></section>",
            unsafe_allow_html=True,
        )


def _render_asset_card(
    panel: AgentPanelPresentation,
    *,
    asset: str,
    sections: tuple[AgentAnalysisSectionPresentation, ...],
) -> None:
    data = {card.capability: card for card in panel.data if _same_asset(card.asset, asset)}
    chips = "".join(
        '<span class="cs-analysis-chip">'
        f'<span class="cs-analysis-chip-label">{html.escape(label)}</span>'
        f"<strong>{html.escape(value)}</strong></span>"
        for label, value in _visible_metrics(panel.agent, data)
    )
    body: list[str] = []
    if panel.agent == "news_agent":
        body.append(_news_development_html(data.get(ResearchCapability.NEWS.value)))
    scope_labels = {
        "market": "Opportunity analysis"
        if ResearchCapability.DISCOVERY.value in panel.capabilities
        else "Market view",
        "risk": "Risk interpretation",
        "derivatives": "Derivatives positioning",
        "fundamentals": "Fundamentals interpretation",
        "defi": "DeFi view",
        "news": "Why it matters",
        "forecast": "Model interpretation",
        "onchain": "Network activity",
    }
    for section in sections:
        scope_class = " cs-analysis-block-risk" if section.scope == "risk" else ""
        body.append(
            f'<div class="cs-analysis-block{scope_class}">'
            f"<span>{html.escape(scope_labels[section.scope])}</span>"
            f"<p>{html.escape(section.text)}</p></div>"
        )
    limitations = tuple(dict.fromkeys(card.limitation for card in data.values() if card.limitation))
    limitation_html = (
        '<div class="cs-analysis-gap"><strong>Coverage:</strong> '
        + html.escape(limitations[0])
        + "</div>"
        if limitations
        else ""
    )
    band = _fact_value(data.get(ResearchCapability.RISK.value), "Band").casefold()
    risk_class = f" cs-risk-{band.replace(' ', '-')}" if band else ""
    st.markdown(
        f'<article class="cs-analysis-asset{risk_class}">'
        f'<div class="cs-analysis-asset-header"><h5>{html.escape(asset)}</h5></div>'
        f'<div class="cs-analysis-chips">{chips}</div>'
        + "".join(body)
        + limitation_html
        + "</article>",
        unsafe_allow_html=True,
    )


def _same_asset(left: str | None, right: str) -> bool:
    if left is None:
        return False
    left_key = left.split("/", maxsplit=1)[0].casefold()
    right_key = right.split("/", maxsplit=1)[0].casefold()
    return left_key == right_key


def _fact_value(card: CapabilityDataPresentation | None, label: str) -> str:
    if card is None:
        return ""
    return next((value for key, value in card.facts if key == label), "")


def _visible_metrics(
    agent: str,
    cards: dict[str, CapabilityDataPresentation],
) -> tuple[tuple[str, str], ...]:
    label_sets: dict[str, dict[str, tuple[str, ...]]] = {
        "market_agent": {
            "market": ("Current price", "24h change", "Trend", "RSI"),
            "derivatives": ("Latest funding", "Open interest", "24h OI change"),
            "risk": ("Risk score", "Band", "Evidence confidence"),
        },
        "fundamentals_agent": {
            "fundamentals": (
                "Market cap",
                "Rank",
                "Commits (4 weeks)",
                "Repository stars",
                "Circulating supply",
                "Categories",
            ),
            "defi": ("TVL", "1d change", "7d change", "Chains"),
        },
        "news_agent": {"news": ("Validated items", "Latest publisher", "Published", "Freshness")},
        "forecast_agent": {
            "forecast": (
                "Current price",
                "Model output",
                "Predicted return",
                "Target time",
                "Interval",
                "Model",
                "Quality",
            )
        },
    }
    labels = label_sets.get(agent, {})
    metrics: list[tuple[str, str]] = []
    for capability, selected_labels in labels.items():
        card = cards.get(capability)
        if card is None:
            continue
        facts = dict(card.facts)
        metrics.extend((label, facts[label]) for label in selected_labels if facts.get(label))
    return tuple(metrics[:7])


def _news_development_html(card: CapabilityDataPresentation | None) -> str:
    title = _fact_value(card, "Latest title")
    if not title:
        return (
            '<div class="cs-analysis-empty"><span>Coverage</span>'
            "<p>No recent validated news item was available for this asset.</p></div>"
        )
    return (
        '<div class="cs-analysis-development"><span>Key development</span>'
        f"<p>{html.escape(title)}</p></div>"
    )


def render_forecast_output(panel: AgentPanelPresentation) -> None:
    """Render deterministic forecast cards with one batch-level trust warning."""

    st.markdown("**Deterministic predictions**")
    untrusted_assets = [
        card.asset or card.title
        for card in panel.data
        if card.status == "partial" and dict(card.facts).get("Quality") == "Not trusted"
    ]
    if untrusted_assets:
        st.warning(
            "ChainScope does not trust the model output for "
            + ", ".join(untrusted_assets)
            + " because its validation checks did not pass."
        )
    for card in panel.data:
        facts = dict(card.facts)
        with st.container(border=True):
            st.markdown(f"#### {safe_markdown(card.asset or card.title)}")
            st.markdown(f"### {safe_markdown(facts.get('Model output', 'Unavailable'))}")
            if facts.get("Quality") == "Validation passed":
                st.caption("Validation passed")
            columns = st.columns(3)
            for column, label in zip(
                columns,
                ("Current price", "Predicted return", "Target time"),
                strict=True,
            ):
                column.metric(label, facts.get(label, "Unavailable"))
            if interval := facts.get("Interval"):
                st.caption("Prediction interval: " + safe_markdown(interval))
            if card.limitation:
                st.caption(safe_markdown(card.limitation))
    if len(panel.data) > 1:
        rows = [
            {
                "Asset": card.asset or card.title,
                "Predicted price": dict(card.facts).get("Model output", "Unavailable"),
                "Predicted return": dict(card.facts).get("Predicted return", "Unavailable"),
            }
            for card in panel.data
        ]
        st.markdown("**Deterministic comparison**")
        render_data_table(rows, label="Deterministic forecast comparison")
    if panel.answer is not None:
        st.markdown("**Model summary**")
        st.caption(
            "Live answer"
            if panel.answer.analysis_state == "live"
            else "Summary unavailable | deterministic results remain available"
        )
        _render_forecast_summary_rows(panel.data)
        structured = panel.answer.structured_analysis
        model_view = (
            structured.verdict
            if structured is not None and panel.answer.analysis_state == "live"
            else panel.answer.answer
            if panel.answer.analysis_state == "live"
            else "Live model context is currently unavailable."
        )
        st.markdown("**Model view**")
        st.write(model_view)
        if panel.answer.analysis_state != "live":
            st.caption("The live summary was unavailable; deterministic outputs remain unchanged.")


def _render_forecast_summary_rows(cards: tuple[CapabilityDataPresentation, ...]) -> None:
    rows: list[str] = []
    for card in cards:
        facts = dict(card.facts)
        quality_class = (
            "is-trusted" if facts.get("Quality") == "Validation passed" else "is-untrusted"
        )
        rows.append(
            '<div class="cs-forecast-summary-row">'
            f"<strong>{html.escape(card.asset or card.title)}</strong>"
            '<span class="cs-forecast-price-path">'
            f"{html.escape(facts.get('Current price', 'Unavailable'))} "
            f"<b>-&gt; {html.escape(facts.get('Model output', 'Unavailable'))}</b></span>"
            f"<span>{html.escape(facts.get('Predicted return', 'Unavailable'))}</span>"
            '<span class="cs-forecast-summary-meta">'
            f"<small>{html.escape(facts.get('Target time', 'Unavailable'))}</small>"
            f'<small class="cs-forecast-quality {quality_class}">'
            f"{html.escape(facts.get('Quality', 'Unavailable'))}</small>"
            "</span>"
            "</div>"
        )
    st.markdown(
        '<div class="cs-forecast-summary" aria-label="Predicted price summary">'
        '<span class="cs-analysis-kicker">Predicted prices</span>' + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


def render_forecast_resources(research: ResearchPresentation) -> None:
    batch = research.forecast_result
    if batch is None:
        return
    palette = active_palette()
    for item in batch.asset_results:
        if not isinstance(item, ForecastRun):
            continue
        st.markdown(f"**{safe_markdown(item.request.symbol)} model diagnostics**")
        point = item.model_output
        history = item.market.candles[-120:]
        frame = pd.DataFrame(
            {"Historical close": [candle.close for candle in history]},
            index=pd.DatetimeIndex([candle.timestamp for candle in history], name="timestamp"),
        )
        frame.loc[history[-1].timestamp, "Model output"] = history[-1].close
        frame.loc[point.timestamp, "Model output"] = point.predicted_price
        render_line_chart(
            frame,
            height=220,
            colors=(palette.blue, palette.positive),
            value_kind="price",
            allow_future=True,
        )
        st.caption(
            f"MAE {item.metrics.mae:.4%} | baseline {item.metrics.baseline_mae:.4%} | "
            f"directional accuracy {item.metrics.directional_accuracy:.1%}"
        )
        failed = [reason for reason in item.quality.reasons if reason.startswith("failed")]
        if failed:
            st.caption("Failed quality gates: " + "; ".join(failed))


def render_research_turn(turn: ResearchTurn) -> None:
    render_research_response(turn.content, turn.research)


def render_research_response(content: str, research: ResearchPresentation) -> None:
    st.markdown(f"#### {safe_markdown(research.title)}")
    if research.agent_panels:
        _render_run_summary(research)
        _render_called_agents(research)
    else:
        # Third-party integrations can omit agent panels; keep their response readable.
        render_analysis_text(content)
    if research.disclaimer:
        render_disclaimer(research.disclaimer)
    with st.expander("Sources and data notes", expanded=False):
        _render_details(research)


def _render_run_summary(research: ResearchPresentation) -> None:
    complete = sum(panel.status == "complete" for panel in research.agent_panels)
    live = sum(panel.analysis_state == "live" for panel in research.agent_panels)
    cached = sum(panel.source_state == "cached" for panel in research.agent_panels)
    evidence_only = sum(panel.analysis_state == "evidence_only" for panel in research.agent_panels)
    unavailable = sum(panel.analysis_state == "unavailable" for panel in research.agent_panels)
    metrics = st.columns(5)
    for column, label, value in zip(
        metrics,
        ("Complete", "Live", "Cached", "Evidence only", "Unavailable"),
        (complete, live, cached, evidence_only, unavailable),
        strict=True,
    ):
        column.metric(label, value)
    if research.retry_of_run_id:
        st.caption(
            "Combined retry report | Retried: "
            + ", ".join(
                AGENT_LABELS.get(agent, agent.replace("_", " ").title())
                for agent in research.retried_agents
            )
        )


def _render_called_agents(research: ResearchPresentation) -> None:
    st.markdown("**Agent outputs**")
    st.caption("Open an agent result to review its analysis and supporting evidence.")
    for panel in research.agent_panels:
        with st.expander(_agent_dropdown_label(panel), expanded=False):
            _render_agent_panel_header(panel, show_identity=False)
            _render_agent_analysis(panel)
            _render_agent_resources(research, panel)


def _agent_dropdown_label(panel: AgentPanelPresentation) -> str:
    label = panel.state_label or _agent_state_label(None, panel.status)
    return f"{panel.title} | {_display_state_label(label)}"


def _display_state_label(label: str) -> str:
    return "Live answer" if label == "Live" else label


def _render_agent_panel_header(
    panel: AgentPanelPresentation,
    *,
    show_identity: bool = True,
) -> None:
    label = panel.state_label or _agent_state_label(None, panel.status)
    if show_identity:
        display_label = _display_state_label(label)
        state_class = (
            "cs-agent-state-live"
            if label in {"Live", "Live answer", "Live evidence answer", "Cached"}
            else "cs-agent-state-partial"
        )
        st.markdown(
            f'<div class="cs-agent-header"><strong>{html.escape(panel.title)}</strong>'
            f'<span class="cs-agent-state {state_class}">'
            f"{html.escape(display_label)}</span></div>",
            unsafe_allow_html=True,
        )
    if panel.capabilities:
        scopes = " | ".join(_capability_label(capability) for capability in panel.capabilities)
        st.markdown(
            '<div class="cs-agent-meta"><span>Research scope</span>'
            f"<strong>{html.escape(scopes)}</strong></div>",
            unsafe_allow_html=True,
        )
    if panel.coverage_state == "partial":
        coverage_message = (
            "Live analysis uses the validated data that was available."
            if panel.analysis_state == "live"
            else "Validated evidence remains available, but live interpretation is incomplete."
        )
        st.markdown(
            '<div class="cs-coverage-note">Some supporting data was unavailable. '
            f"{coverage_message} Evidence and limitations are included below.</div>",
            unsafe_allow_html=True,
        )
    if panel.source_state == "cached":
        st.caption("Provider data: validated cached records.")


def _render_agent_analysis(panel: AgentPanelPresentation) -> None:
    if panel.agent == "forecast_agent":
        render_forecast_output(panel)
        return
    if panel.answer is None:
        if ResearchCapability.RISK.value in panel.capabilities:
            st.caption(
                "This specialist contributes inputs to the deterministic Risk "
                "assessment shown in the Market card."
            )
        elif panel.data:
            st.caption("Validated data is included below.")
        return
    if panel.answer.structured_analysis is not None:
        render_structured_agent_analysis(panel, panel.answer.structured_analysis)
        if panel.answer.uncertainty:
            uncertainty = " ".join(safe_markdown(item) for item in panel.answer.uncertainty)
            st.caption("Uncertainty: " + uncertainty)
        return
    if panel.agent == "fundamentals_agent":
        st.markdown("**Fundamentals analysis**")
    else:
        st.markdown("**Analysis**")
    render_analysis_text(panel.answer.answer)
    if panel.answer.analysis:
        has_defi = any(card.capability == ResearchCapability.DEFI.value for card in panel.data)
        if panel.agent == "fundamentals_agent" and has_defi:
            st.markdown("**DeFi analysis**")
        elif panel.agent == "market_agent":
            st.markdown("**Risk interpretation**")
        elif panel.agent == "news_agent":
            st.markdown("**News context**")
        else:
            st.markdown("**Analysis details**")
        st.markdown(safe_markdown(panel.answer.analysis, preserve_paragraphs=True))
    if panel.answer.uncertainty:
        uncertainty = " ".join(safe_markdown(item) for item in panel.answer.uncertainty)
        st.caption("Uncertainty: " + uncertainty)


def _render_agent_resources(research: ResearchPresentation, panel: AgentPanelPresentation) -> None:
    claims = panel.answer.claims if panel.answer is not None else ()
    item_count = len(panel.data) + len(claims)
    evidence_label = (
        f"Evidence & resources | {item_count} item{'s' if item_count != 1 else ''}"
        if item_count
        else "Evidence & resources | No records"
    )
    with st.expander(evidence_label, expanded=False):
        if panel.agent == "market_agent":
            _render_discovery_results(research.discovery)
        if panel.agent == "forecast_agent":
            render_forecast_resources(research)
        if panel.data and panel.agent != "forecast_agent":
            render_panel_section_heading("Validated data", "Current inputs used for this answer.")
            derivatives = tuple(
                card
                for card in panel.data
                if card.capability == ResearchCapability.DERIVATIVES.value
            )
            other_cards = tuple(card for card in panel.data if card not in derivatives)
            if other_cards:
                render_capability_cards(other_cards)
            if derivatives:
                _render_derivatives_table(derivatives)
        if claims:
            render_panel_section_heading(
                "Supported claims", "Trace conclusions to evidence records."
            )
            render_claim_support(claims)
        if panel.limitation:
            st.markdown(
                '<div class="cs-resource-note"><strong>Coverage note</strong><span>'
                f"{html.escape(panel.limitation)}</span></div>",
                unsafe_allow_html=True,
            )
        if not panel.data and not claims and panel.limitation is None:
            st.caption("No separate evidence records were attached to this agent answer.")


def _render_derivatives_table(cards: tuple[CapabilityDataPresentation, ...]) -> None:
    st.markdown("**Derivatives positioning**")
    rows = []
    for card in cards:
        facts = dict(card.facts)
        rows.append(
            {
                "Asset": card.asset or card.title,
                "Contract": facts.get("Contract", "Not applicable"),
                "Funding": facts.get("Latest funding", "Unavailable"),
                "Open interest": facts.get("Open interest", "Unavailable"),
                "24h change": facts.get("24h OI change", "Unavailable"),
                "Venue": facts.get("Venue", "Binance USD-M Futures"),
                "Freshness": facts.get("Freshness", "Unavailable"),
                "Cache": facts.get("Cache state", "Unavailable"),
                "Limitations": card.limitation or "None reported",
            }
        )
    render_data_table(rows)


def _render_discovery_results(discovery: DiscoveryPresentation | None) -> None:
    """Show the full deterministic scan rather than hiding prices behind the LLM brief."""

    if discovery is None:
        return
    st.markdown("**Ranked market screen**")
    st.caption(
        f"{discovery.exchange.title()} | {discovery.timeframe} | "
        f"collected {discovery.collected_at:%d %b %Y %H:%M UTC}"
    )
    render_data_table(
        [
            {
                "#": candidate.rank,
                "Asset": candidate.symbol,
                "Price": format_price(candidate.current_price),
                "24h momentum": f"{candidate.momentum_24h:+.2f}%",
                "24h volatility": f"{candidate.volatility_24h:.2f}%",
                "Score": f"{candidate.score:.1f}/100",
                "Trend": candidate.trend.title(),
            }
            for candidate in discovery.candidates
        ],
        columns=("#", "Asset", "Price", "24h momentum", "24h volatility", "Score", "Trend"),
        label="Ranked market screen",
    )
    st.markdown("**Why they ranked**")
    for candidate in discovery.candidates[:4]:
        st.markdown(
            f"**#{candidate.rank} {safe_markdown(candidate.symbol)}** - "
            f"{safe_markdown(candidate.reason)}"
        )
    st.caption(safe_markdown(discovery.summary))


def render_analysis_text(content: str) -> None:
    cleaned = _preserve_paragraphs(content)
    if not cleaned:
        return
    st.markdown(safe_markdown(cleaned, preserve_paragraphs=True))


def _preserve_paragraphs(value: str) -> str:
    normalized: list[str] = []
    for line in (line.strip() for line in value.splitlines()):
        if not line:
            if normalized and normalized[-1]:
                normalized.append("")
            continue
        normalized.append(" ".join(line.split()))
    return "\n".join(normalized)


def _render_details(research: ResearchPresentation) -> None:
    """Render traceable sources and validation notes without answer-side charts."""

    if research.sources:
        st.markdown("**Sources**")
        _render_source_list(research.sources)
    if research.warnings:
        st.caption("Data notes")
        for warning in research.warnings:
            st.caption("- " + safe_markdown(warning))


def _render_source_list(sources: tuple[SourcePresentation, ...]) -> None:
    grouped: dict[str, list[SourcePresentation]] = defaultdict(list)
    for source in sources:
        grouped[source.kind].append(source)
    for kind, items in grouped.items():
        st.markdown(f"##### {SOURCE_ICONS.get(kind, '*')} {kind}")
        for source in items:
            title = safe_markdown(source.title)
            publisher = safe_markdown(source.publisher)
            published = source.published_at.astimezone(UTC).strftime("%d %b %Y %H:%M UTC")
            time_context = source.time_context.casefold()
            if source.url:
                st.markdown(f"[{title}]({source.url}) | {publisher} | {time_context} {published}")
            else:
                st.markdown(f"{title} | {publisher} | {time_context} {published}")
