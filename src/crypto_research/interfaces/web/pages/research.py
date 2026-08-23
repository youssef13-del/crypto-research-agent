"""Guided specialist research workspace."""

from __future__ import annotations

import logging

import streamlit as st
from pydantic import ValidationError

from crypto_research.bootstrap import load_research_repository
from crypto_research.domain.core import (
    MAX_COMPARISON_ASSETS,
    SUPPORTED_EXCHANGES,
    ResearchCapability,
)
from crypto_research.domain.forecast import ForecastSettings
from crypto_research.interfaces.web import runtime
from crypto_research.interfaces.web.components.layout import render_page_header
from crypto_research.interfaces.web.components.research import render_research_turn
from crypto_research.interfaces.web.presentation import ResearchTurn
from crypto_research.interfaces.web.research_jobs import (
    ResearchJobAlreadyRunning,
    ResearchJobSnapshot,
    latest_research_job,
    start_guided_research_job,
    start_retry_research_job,
)
from crypto_research.interfaces.web.runtime import (
    LATEST_RESEARCH_TURN_STATE_KEY,
    initialize_state,
)
from crypto_research.orchestration.planning import (
    GuidedResearchPlan,
    agent_labels,
    asset_display_name,
    available_assets,
    capability_groups,
    capability_options,
    compile_guided_research_plan,
    defi_eligible_assets,
)

LOGGER = logging.getLogger(__name__)
_CAPABILITY_LABELS = {cap.value: label for cap, label, _ in capability_options()}
_AGENT_LABELS = agent_labels()
_GUIDED_MARKET_TIMEFRAMES = ("1h", "4h", "1d")
_LATEST_RESEARCH_TURN_KEY = LATEST_RESEARCH_TURN_STATE_KEY
_GUIDED_MODE_KEY = "guided-mode"
_RESEARCH_MODE = "Custom research"
_DISCOVERY_MODE = "Find opportunities"


def research_page() -> None:
    initialize_state()
    _restore_latest_saved_research()
    render_page_header(
        "Research",
        "Guided Research",
        "Configure and run an evidence-backed specialist report.",
    )
    _render_research_workspace()


def _render_research_workspace() -> None:
    _render_guided_panel()
    active_job = _current_research_job()
    if active_job is not None and active_job.active:
        st.info(
            f"Research continues in the background: {active_job.label} "
            "You can change pages without stopping it."
        )
    turn = st.session_state.get(_LATEST_RESEARCH_TURN_KEY)
    if isinstance(turn, ResearchTurn):
        retryable = _retryable_agents(turn)
        toolbar = st.columns([4, 1.8])
        toolbar[0].markdown("### Latest research")
        toolbar[0].caption("Saved to your Research Library and restored when you return.")
        if retryable:
            run_id = turn.research.run_id
            if toolbar[1].button(
                f"Retry failed agents ({len(retryable)})",
                key="guided-retry-failed",
                icon=":material/replay:",
                width="stretch",
                disabled=run_id is None or (active_job is not None and active_job.active),
                help=(
                    "Wait for the active research task to finish."
                    if active_job is not None and active_job.active
                    else "Creates a new immutable combined report and retains successful cards."
                    if run_id is not None
                    else "Retry requires an original report saved in Research Library."
                ),
            ):
                assert run_id is not None
                _submit_failed_agent_retry(run_id)
                st.rerun()
        render_research_turn(turn)
    else:
        st.info("Choose a research scope and run it to create a report.")


def _restore_latest_saved_research() -> None:
    existing = st.session_state.get(_LATEST_RESEARCH_TURN_KEY)
    if isinstance(existing, ResearchTurn):
        return
    workspace = runtime.current_workspace()
    if workspace is None:
        return
    try:
        settings = runtime.load_runtime_settings()
        repository = load_research_repository(
            settings.database_url,
            settings.research_retention_days,
        )
        stored = (
            repository.for_owner(runtime.current_owner_id()).latest_run()
            if repository is not None
            else None
        )
        if stored is not None:
            st.session_state[_LATEST_RESEARCH_TURN_KEY] = runtime.turn_from_stored_research(stored)
    except (RuntimeError, ValueError, ValidationError) as exc:
        LOGGER.warning("Latest saved research could not be restored (%s).", type(exc).__name__)


def _retryable_agents(turn: ResearchTurn) -> tuple[str, ...]:
    return tuple(
        panel.agent
        for panel in turn.research.agent_panels
        if panel.analysis_state in {"unavailable", "evidence_only"}
    )


def _submit_failed_agent_retry(run_id: str) -> None:
    try:
        settings = runtime.load_runtime_settings()
    except ValidationError:
        st.error("Valid provider settings are required before failed agents can be retried.")
        return
    displayed = st.session_state.get(_LATEST_RESEARCH_TURN_KEY)
    retryable = _retryable_agents(displayed) if isinstance(displayed, ResearchTurn) else ()
    labels = tuple(_AGENT_LABELS.get(agent, agent) for agent in retryable)
    try:
        start_retry_research_job(
            owner_id=runtime.current_owner_id(),
            settings=settings,
            run_id=run_id,
            question="Retry failed agents: " + ", ".join(labels or ("specialists",)),
            agents=retryable,
        )
    except ResearchJobAlreadyRunning:
        st.info("Research is already running for this workspace.")
        return
    except (RuntimeError, ValueError) as exc:
        LOGGER.error("Failed-agent retry could not start (%s).", type(exc).__name__)
        st.error("Failed agents could not be retried. The original report is unchanged.")
        return
    st.toast("Retry started. It will continue while you use other pages.")


def _render_guided_panel() -> None:
    with st.container(border=True):
        st.markdown("#### Guided Research")
        mode = st.segmented_control(
            "Research mode",
            (_RESEARCH_MODE, _DISCOVERY_MODE),
            default=_RESEARCH_MODE,
            key=_GUIDED_MODE_KEY,
            width="stretch",
        )
        if mode == _DISCOVERY_MODE:
            _render_discovery_brief()
        else:
            _render_asset_research_brief()


def _render_asset_research_brief() -> None:
    workspace = runtime.current_workspace()
    if workspace is not None:
        st.session_state.setdefault("guided-exchange", workspace.preferences.default_exchange)
        if workspace.preferences.default_timeframe in _GUIDED_MARKET_TIMEFRAMES:
            st.session_state.setdefault(
                "guided-timeframe",
                workspace.preferences.default_timeframe,
            )
    assets = st.multiselect(
        "Coins",
        options=available_assets(),
        format_func=asset_display_name,
        max_selections=MAX_COMPARISON_ASSETS,
        help="Select up to four supported coins.",
        key="guided-assets",
    )
    count = len(assets)
    comparison_note = (
        "Automatic comparison enabled."
        if count > 1
        else "Select 2-4 coins to compare automatically."
    )
    st.caption(f"{count}/{MAX_COMPARISON_ASSETS} selected · {comparison_note}")
    selected_caps = _render_research_topics(assets)
    exchange, timeframe, forecast_settings = _render_data_settings(selected_caps)
    plan, message = _visible_guided_plan(
        mode="asset",
        assets=assets,
        capabilities=selected_caps,
        exchange=exchange,
        timeframe=timeframe,
        forecast_settings=forecast_settings,
    )
    _render_research_brief(plan, message)
    job_running = _research_job_is_active()
    if job_running:
        st.caption("A research task is already running. Its progress is available in the sidebar.")
    if st.button(
        _research_action_label(assets),
        type="primary",
        width="stretch",
        disabled=plan is None or job_running,
        key="guided-run",
    ):
        assert plan is not None
        _submit_guided_research(plan)
        st.rerun()


def _render_discovery_brief() -> None:
    exchange, timeframe = _render_market_settings(prefix="discovery")
    plan, message = _visible_guided_plan(
        mode="discovery",
        assets=[],
        capabilities=[ResearchCapability.DISCOVERY],
        exchange=exchange,
        timeframe=timeframe,
    )
    _render_research_brief(plan, message)
    job_running = _research_job_is_active()
    if job_running:
        st.caption("A research task is already running. Its progress is available in the sidebar.")
    if st.button(
        "Find opportunities",
        type="primary",
        width="stretch",
        disabled=plan is None or job_running,
        key="guided-discovery-run",
    ):
        assert plan is not None
        _submit_guided_research(plan)
        st.rerun()


def _render_research_topics(assets: list[str]) -> list[ResearchCapability]:
    st.markdown("**What do you want to research?**")
    st.caption(
        "Choose topics directly. Each selected topic calls its specialist; Recent news calls "
        "the News Agent."
    )
    options = {
        capability: (label, description) for capability, label, description in capability_options()
    }
    eligible_defi_assets = defi_eligible_assets(assets)
    defi_enabled = bool(eligible_defi_assets)
    if not defi_enabled:
        st.session_state.pop("cap-defi", None)
    selected: list[ResearchCapability] = []
    columns = st.columns(len(capability_groups()))
    for column, (group, capabilities) in zip(columns, capability_groups(), strict=True):
        with column:
            st.markdown(f"**{group}**")
            for capability in capabilities:
                label, description = options[capability]
                disabled = capability is ResearchCapability.DEFI and not defi_enabled
                help_text = (
                    description
                    if not disabled
                    else "Select a supported DeFi protocol coin to enable this topic."
                )
                if st.checkbox(
                    label,
                    key=f"cap-{capability.value}",
                    help=help_text,
                    disabled=disabled,
                ):
                    selected.append(capability)
    if eligible_defi_assets and len(eligible_defi_assets) < len(assets):
        st.caption("DeFi will run for: " + ", ".join(eligible_defi_assets))
    return selected


def _render_data_settings(
    capabilities: list[ResearchCapability],
) -> tuple[str, str, ForecastSettings | None]:
    selected = set(capabilities)
    needs_exchange = bool(
        selected
        & {
            ResearchCapability.MARKET,
            ResearchCapability.RISK,
            ResearchCapability.FORECAST,
            ResearchCapability.DERIVATIVES,
        }
    )
    needs_timeframe = bool(
        selected
        & {
            ResearchCapability.MARKET,
            ResearchCapability.RISK,
            ResearchCapability.DERIVATIVES,
        }
    )
    needs_forecast = ResearchCapability.FORECAST in selected
    if not needs_exchange:
        return "kraken", "1h", None
    with st.expander("Data settings", expanded=needs_forecast):
        if ResearchCapability.DERIVATIVES in selected:
            st.caption(
                "Spot market settings remain below. Derivatives always use Binance USD-M Futures."
            )
        if needs_timeframe:
            exchange, timeframe = _render_market_settings(timeframe_label="Analysis timeframe")
        else:
            exchange = st.selectbox(
                "Exchange",
                options=SUPPORTED_EXCHANGES,
                key="guided-exchange",
            )
            timeframe = "1h"
        forecast_settings = _render_forecast_settings() if needs_forecast else None
    return exchange, timeframe, forecast_settings


def _render_market_settings(
    prefix: str = "guided",
    *,
    timeframe_label: str = "Timeframe",
) -> tuple[str, str]:
    timeframe_key = f"{prefix}-timeframe"
    exchange_key = f"{prefix}-exchange"
    if st.session_state.get(timeframe_key) not in _GUIDED_MARKET_TIMEFRAMES:
        st.session_state[timeframe_key] = "1h"
    columns = st.columns(2)
    with columns[0]:
        exchange = st.selectbox("Exchange", options=SUPPORTED_EXCHANGES, key=exchange_key)
    with columns[1]:
        timeframe = st.selectbox(
            timeframe_label,
            options=_GUIDED_MARKET_TIMEFRAMES,
            key=timeframe_key,
        )
    return exchange, timeframe


def _render_forecast_settings() -> ForecastSettings:
    st.markdown("**Forecast settings**")
    first = st.columns(2)
    with first[0]:
        model_id = st.selectbox(
            "Model",
            ("gradient_boosting_huber", "ridge"),
            format_func=lambda value: (
                "Gradient Boosting (Huber)" if value == "gradient_boosting_huber" else "Ridge"
            ),
            key="guided-forecast-model",
        )
    with first[1]:
        forecast_timeframe = st.selectbox(
            "Forecast timeframe", ("1h", "4h"), key="guided-forecast-timeframe"
        )
    second = st.columns(3)
    with second[0]:
        horizon = st.selectbox(
            "Horizon",
            (4, 8, 12, 24, 48),
            index=3,
            format_func=lambda value: f"{value}h",
            key="guided-forecast-horizon",
        )
    with second[1]:
        confidence = st.selectbox(
            "Interval",
            (0.8, 0.9),
            format_func=lambda value: f"{value:.0%}",
            key="guided-forecast-confidence",
        )
    with second[2]:
        lookback = st.selectbox(
            "History",
            (500, 750, 1000, 1500),
            index=1,
            format_func=lambda value: f"{value} candles",
            key="guided-forecast-lookback",
        )
    return ForecastSettings(
        timeframe=forecast_timeframe,
        horizon_hours=horizon,
        model_id=model_id,
        confidence_level=confidence,
        lookback_candles=lookback,
    )


def _visible_guided_plan(
    *,
    mode: str,
    assets: list[str],
    capabilities: list[ResearchCapability],
    exchange: str,
    timeframe: str,
    forecast_settings: ForecastSettings | None = None,
) -> tuple[GuidedResearchPlan | None, str | None]:
    try:
        return (
            compile_guided_research_plan(
                assets,
                capabilities,
                mode=mode,
                exchange=exchange,
                timeframe=timeframe,
                forecast_settings=forecast_settings,
            ),
            None,
        )
    except ValueError as exc:
        return None, str(exc)


def _render_research_brief(plan: GuidedResearchPlan | None, message: str | None) -> None:
    st.divider()
    st.markdown("**Research brief**")
    if plan is None:
        st.caption(message or "Choose a valid research scope.")
        return
    request = plan.action.request
    assert request is not None
    if plan.mode == "discovery":
        st.caption("Scope: Market-wide opportunity scan")
        st.caption(f"Data: {request.exchange.title()} · {request.timeframe}")
        st.caption("Agent: " + _AGENT_LABELS["market_agent"])
        return
    labels = [_CAPABILITY_LABELS.get(cap.value, cap.value) for cap in plan.display_capabilities]
    assets = [asset.requested_name for asset in request.assets]
    st.caption("Coins: " + ", ".join(assets))
    st.caption("Topics: " + ", ".join(labels))
    st.caption(
        "Agents: " + ", ".join(_AGENT_LABELS.get(agent, agent) for agent in plan.expected_agents)
    )
    scope: list[str] = []
    if len(request.assets) > 1:
        scope.append("Automatic comparison")
    selected = set(plan.display_capabilities)
    if selected & {
        ResearchCapability.MARKET,
        ResearchCapability.RISK,
        ResearchCapability.DERIVATIVES,
    }:
        scope.append(f"{request.exchange.title()} · {request.timeframe}")
    elif ResearchCapability.FORECAST in selected:
        scope.append(request.exchange.title())
    if request.forecast_settings is not None:
        settings = request.forecast_settings
        scope.append(f"Forecast {settings.timeframe} · {settings.horizon_hours}h horizon")
    if ResearchCapability.DERIVATIVES in selected:
        scope.append("Binance USD-M Futures derivatives")
    st.caption("Scope: " + " · ".join(scope or ["Selected provider coverage"]))


def _research_action_label(assets: list[str]) -> str:
    if len(assets) > 1:
        return f"Compare {len(assets)} coins"
    if assets:
        return f"Research {assets[0]}"
    return "Run custom research"


def _submit_guided_research(plan: GuidedResearchPlan) -> None:
    request = plan.action.request
    assert request is not None
    question = _format_selection(plan)

    try:
        settings = runtime.load_runtime_settings()
    except ValidationError:
        st.error("Valid provider settings are required before live research can run.")
        return

    try:
        start_guided_research_job(
            owner_id=runtime.current_owner_id(),
            settings=settings,
            action=plan.action,
            question=question,
        )
    except ResearchJobAlreadyRunning:
        st.info("Research is already running for this workspace.")
        return
    except (RuntimeError, ValueError) as exc:
        LOGGER.error("Guided research could not start (%s).", type(exc).__name__)
        st.error("The research services are temporarily unavailable. Please try again.")
        return
    st.toast("Research started. It will continue while you use other pages.")


def _current_research_job() -> ResearchJobSnapshot | None:
    workspace = runtime.current_workspace()
    if workspace is None:
        return None
    try:
        return latest_research_job(runtime.current_owner_id())
    except RuntimeError, ValueError:
        return None


def _research_job_is_active() -> bool:
    snapshot = _current_research_job()
    return snapshot is not None and snapshot.active


def _format_selection(plan: GuidedResearchPlan) -> str:
    request = plan.action.request
    assert request is not None
    if plan.mode == "discovery":
        return f"Discover {request.exchange.title()} market opportunities"
    cap_names = [_CAPABILITY_LABELS.get(cap.value, cap.value) for cap in plan.display_capabilities]
    assets = [asset.requested_name for asset in request.assets]
    verb = "Compare" if len(assets) > 1 else "Research"
    return f"{verb} {', '.join(assets)} - {' + '.join(cap_names)}"
