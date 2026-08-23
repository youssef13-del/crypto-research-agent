"""Interactive market dashboard page."""

# Page copy is kept as complete product language.
# ruff: noqa: E501

from __future__ import annotations

import logging
from datetime import UTC, datetime

import streamlit as st
from pydantic import ValidationError

from crypto_research.bootstrap import resolve_asset_request
from crypto_research.config import Settings
from crypto_research.domain.core import MAX_COMPARISON_ASSETS
from crypto_research.domain.market import MarketEvidence
from crypto_research.domain.research import (
    COIN_ID_BY_ASSET,
    SUPPORTED_EXCHANGES,
    AnalysisRequest,
    SupportedExchange,
    SupportedTimeframe,
    TechnicalSnapshot,
    build_market_symbol,
    extract_supported_assets,
)
from crypto_research.interfaces.web import runtime
from crypto_research.interfaces.web.components.charts import (
    candle_frame,
    format_price,
    render_candlestick_chart,
    render_market_chart,
)
from crypto_research.interfaces.web.components.layout import (
    render_data_table,
    render_disclaimer,
    render_empty_panel,
    render_page_header,
    render_section_header,
)
from crypto_research.interfaces.web.presentation import DashboardView
from crypto_research.shared.security import redact_secrets
from crypto_research.shared.time import timeframe_delta

DASHBOARD_TIMEFRAMES: tuple[SupportedTimeframe, ...] = ("1h", "4h", "1d")
DASHBOARD_LIMITS = (48, 120, 240, 480)
MAX_DASHBOARD_ASSETS = MAX_COMPARISON_ASSETS
DASHBOARD_RESULT_KEY = "dashboard_last_result"
DASHBOARD_ASSETS_KEY = "dashboard-assets-input"
LOGGER = logging.getLogger(__name__)


def dashboard_page() -> None:
    workspace = runtime.current_workspace()
    default_assets = (
        ", ".join(workspace.watchlist[:MAX_DASHBOARD_ASSETS])
        if workspace is not None and workspace.watchlist
        else "BTC, ETH"
    )
    default_exchange = workspace.preferences.default_exchange if workspace is not None else "kraken"
    preferred_timeframe = workspace.preferences.default_timeframe if workspace is not None else "1h"
    default_timeframe = preferred_timeframe if preferred_timeframe in DASHBOARD_TIMEFRAMES else "1h"
    st.session_state.setdefault(DASHBOARD_ASSETS_KEY, default_assets)
    st.session_state.setdefault("dashboard-exchange", default_exchange)
    st.session_state.setdefault("dashboard-timeframe", default_timeframe)
    render_page_header(
        "Markets",
        "Market Dashboard",
        "Compare up to four assets on the same exchange, timeframe, and observation window.",
    )
    render_section_header(
        "Configure view",
        "Use one asset or compare up to four on a shared time window.",
    )

    with st.form("market-dashboard-form", clear_on_submit=False):
        asset_query = st.text_input(
            "Assets to compare",
            key=DASHBOARD_ASSETS_KEY,
            placeholder="BTC, ETH, SOL",
            help="Enter names or tickers separated by commas. Asset identity is resolved against the curated catalog and provider data.",
        )
        controls = st.columns(3)
        with controls[0]:
            exchange = st.selectbox(
                "Exchange",
                SUPPORTED_EXCHANGES,
                format_func=str.title,
                key="dashboard-exchange",
            )
        with controls[1]:
            timeframe = st.selectbox(
                "Candle timeframe",
                DASHBOARD_TIMEFRAMES,
                key="dashboard-timeframe",
            )
        with controls[2]:
            candle_limit = st.selectbox(
                "Observation window",
                DASHBOARD_LIMITS,
                index=1,
                format_func=lambda value: _window_label(value, timeframe),
            )
        submitted = st.form_submit_button(
            "Build market view",
            type="primary",
            width="stretch",
            icon=":material/query_stats:",
        )

    if not submitted:
        stored = st.session_state.get(DASHBOARD_RESULT_KEY)
        if isinstance(stored, DashboardView):
            _render_dashboard_result(stored)
        else:
            _render_dashboard_intro()
        _render_dashboard_disclaimer()
        return

    st.session_state.pop(DASHBOARD_RESULT_KEY, None)
    try:
        settings = runtime.load_runtime_settings()
    except ValidationError:
        st.error("The application configuration is invalid. Check the provider settings.")
        return
    symbols, coin_ids, resolution_warnings = _symbols_from_input(
        asset_query,
        exchange,
        settings=settings,
    )
    if not symbols:
        for warning in resolution_warnings:
            st.error(warning)
        if not resolution_warnings:
            st.warning("Enter at least one recognizable cryptocurrency.")
        return

    with st.status("Preparing comparison…", expanded=True) as status:
        status.write(f"Fetching {timeframe} candles from {exchange.title()}")
        try:
            request = AnalysisRequest(
                user_intent=f"Compare {', '.join(symbols)}",
                exchange=exchange,
                symbol=symbols[0],
                timeframe=timeframe,
                candle_limit=int(candle_limit),
                coin_id=coin_ids[0],
                comparison_symbols=symbols,
            )
            snapshots, provider_warnings = runtime.load_market_dashboard(request)
        except Exception as exc:
            LOGGER.error("Market dashboard request failed (%s).", type(exc).__name__)
            status.update(label="Market data unavailable", state="error")
            st.error("The market provider could not complete this request. Please try again.")
            return
        status.update(label="Comparison ready", state="complete", expanded=False)

    warnings = tuple(
        redact_secrets(warning)
        for warning in dict.fromkeys([*resolution_warnings, *provider_warnings])
    )
    if not snapshots:
        for warning in warnings:
            st.warning(warning)
        render_empty_panel(
            "No market snapshots",
            "The provider returned no complete asset history for this request.",
            icon="—",
        )
        return
    view = DashboardView(snapshots=tuple(snapshots), warnings=warnings)
    st.session_state[DASHBOARD_RESULT_KEY] = view
    _render_dashboard_result(view)
    _render_dashboard_disclaimer()


def _render_dashboard_disclaimer() -> None:
    render_disclaimer(
        "Market data and calculations are informational. This dashboard does not recommend buying, "
        "selling, holding, shorting, or trading any asset."
    )


def _render_dashboard_result(view: DashboardView) -> None:
    snapshots = list(view.snapshots)
    for warning in view.warnings:
        st.warning(warning)

    render_section_header(
        "Summary",
        "Latest prices and shared-window returns.",
    )
    _render_summary(snapshots)
    if len(snapshots) == 1:
        render_section_header(
            "Market history and metrics",
            "The selected asset uses one exact observation window and source timestamp.",
        )
        _render_comparison_table(snapshots)
        tabs = st.tabs(["Price history", "Candles"])
        with tabs[0]:
            render_market_chart(snapshots)
        with tabs[1]:
            _render_candles(snapshots)
        _render_data_quality(snapshots)
        return
    if not _snapshots_share_window(snapshots):
        render_section_header(
            "Individual market histories",
            "A standardized comparison was not possible, so unaligned metrics are not shown.",
        )
        st.warning(
            "The available assets do not share one complete timestamp window. "
            "Their individual candles remain visible below."
        )
        tabs = st.tabs(["Price history", "Candles"])
        with tabs[0]:
            render_market_chart(snapshots)
        with tabs[1]:
            _render_candles(snapshots)
        _render_data_quality(snapshots)
        return
    render_section_header(
        "Shared comparison",
        "Returns, volatility, range, and volume use the same window for every asset.",
    )
    _render_comparison_table(snapshots)
    tabs = st.tabs(["Indexed performance", "Price history", "Candles"])
    with tabs[0]:
        render_market_chart(snapshots, normalized=True)
    with tabs[1]:
        render_market_chart(snapshots)
    with tabs[2]:
        _render_candles(snapshots)
    _render_data_quality(snapshots)


def _symbols_from_input(
    value: str,
    exchange: SupportedExchange,
    *,
    settings: Settings | None = None,
) -> tuple[list[str], list[str | None], list[str]]:
    symbols: list[str] = []
    coin_ids: list[str | None] = []
    warnings: list[str] = []
    queries = [part.strip() for part in value.split(",") if part.strip()]
    known_queries = [extract_supported_assets(item) for item in queries]
    if sum(len(matches) == 1 for matches in known_queries) > MAX_DASHBOARD_ASSETS:
        return (
            [],
            [],
            [f"Choose no more than {MAX_DASHBOARD_ASSETS} assets for one market comparison."],
        )
    for item in queries:
        known = extract_supported_assets(item)
        if settings is None:
            if len(known) != 1:
                warnings.append(f"Could not verify '{item}' against the curated asset catalog.")
                continue
            base = known[0]
            symbol = build_market_symbol(base, exchange)
            coin_id = COIN_ID_BY_ASSET.get(base)
        else:
            base = known[0] if len(known) == 1 else "TOKEN"
            unresolved = AnalysisRequest(
                user_intent=f"Resolve {item} for a market comparison",
                asset_query=item,
                exchange=exchange,
                symbol=build_market_symbol(base, exchange),
                candle_limit=20,
                coin_id=COIN_ID_BY_ASSET.get(base),
            )
            resolved = resolve_asset_request(unresolved, settings)
            identity = resolved.asset_resolution
            if identity is None or identity.status != "confirmed" or identity.selected is None:
                if identity is not None:
                    warnings.extend(identity.warnings)
                    if identity.status == "ambiguous":
                        choices = ", ".join(
                            f"{candidate.name} ({candidate.symbol})"
                            for candidate in identity.candidates[:4]
                        )
                        suffix = f": {choices}." if choices else "."
                        warnings.append(f"'{item}' is ambiguous. Use a more specific name{suffix}")
                        continue
                warnings.append(f"Could not verify '{item}' as a cryptocurrency.")
                continue
            symbol = resolved.symbol
            coin_id = resolved.coin_id
        if symbol in symbols:
            continue
        if len(symbols) >= MAX_DASHBOARD_ASSETS:
            return (
                [],
                [],
                [f"Choose no more than {MAX_DASHBOARD_ASSETS} assets for one market comparison."],
            )
        symbols.append(symbol)
        coin_ids.append(coin_id)
    return symbols, coin_ids, warnings


def _render_dashboard_intro() -> None:
    columns = st.columns(3)
    panels = (
        (
            "Consistent window",
            "Every asset uses the same exchange, timeframe, and candle count.",
            "◈",
        ),
        (
            "Evidence timestamps",
            "Collection time and candle boundaries remain attached to the result.",
            "◷",
        ),
        (
            "Partial by design",
            "A provider failure for one asset does not erase successful snapshots.",
            "◇",
        ),
    )
    for column, (title, body, icon) in zip(columns, panels, strict=True):
        with column:
            render_empty_panel(title, body, icon=icon)


def _render_summary(snapshots: list[tuple[MarketEvidence, TechnicalSnapshot]]) -> None:
    for start in range(0, len(snapshots), 4):
        row = snapshots[start : start + 4]
        columns = st.columns(len(row))
        for column, (market, _) in zip(columns, row, strict=True):
            with column:
                metrics = runtime.calculate_dashboard_metrics(market)
                st.markdown(f"**{market.symbol}**")
                st.markdown(f"{format_price(market.current_price)}")
                st.caption(f"Return {metrics.price_return:+.2%}")
                st.caption(
                    f"{market.exchange.title()} · {market.timeframe} · "
                    f"{len(market.candles)} candles"
                )
                st.caption(f"Collected {market.collected_at:%d %b %Y %H:%M UTC}")


def _render_comparison_table(snapshots: list[tuple[MarketEvidence, TechnicalSnapshot]]) -> None:
    rows: list[dict[str, str]] = []
    for market, _ in snapshots:
        metrics = runtime.calculate_dashboard_metrics(market)
        rows.append(
            {
                "Asset": market.symbol,
                "Last observed": format_price(market.current_price),
                "Return": f"{metrics.price_return:+.2%}",
                "Volatility": f"{metrics.volatility:.2%}",
                "High": format_price(metrics.high),
                "Low": format_price(metrics.low),
                "Volume": f"{metrics.total_volume:,.0f}",
                "Observations": str(metrics.observation_count),
                "Window": f"{metrics.period_start:%d %b %H:%M} – {metrics.period_end:%d %b %H:%M} UTC",
            }
        )
    render_data_table(rows, label="Market comparison")


def _render_candles(snapshots: list[tuple[MarketEvidence, TechnicalSnapshot]]) -> None:
    st.caption(
        "OHLC candles preserve the exact open, high, low, close, and timestamp for each asset."
    )
    for market, _ in snapshots:
        with st.expander(f"{market.symbol} candlestick chart", expanded=len(snapshots) == 1):
            render_candlestick_chart(market)
    for market, _ in snapshots:
        with st.expander(f"{market.symbol} raw OHLCV rows", expanded=False):
            st.dataframe(
                candle_frame(market.candles, limit=len(market.candles)),
                width="stretch",
                hide_index=True,
            )


def _render_data_quality(snapshots: list[tuple[MarketEvidence, TechnicalSnapshot]]) -> None:
    latest = max(market.collected_at for market, _ in snapshots)
    closed_candle_age = max(
        0,
        int(
            max(
                datetime.now(UTC) - (market.last_time + timeframe_delta(market.timeframe))
                for market, _ in snapshots
            ).total_seconds()
        ),
    )
    sources = ", ".join(dict.fromkeys(market.data_source for market, _ in snapshots))
    st.caption(
        f"Data quality: {len(snapshots)} assets returned · latest collection "
        f"{latest:%d %b %H:%M UTC} · closed-candle age {_format_age(closed_candle_age)} · "
        f"sources: {sources}"
    )


def _snapshots_share_window(
    snapshots: list[tuple[MarketEvidence, TechnicalSnapshot]],
) -> bool:
    if len(snapshots) < 2:
        return False
    expected = tuple(candle.timestamp for candle in snapshots[0][0].candles)
    return bool(expected) and all(
        tuple(candle.timestamp for candle in market.candles) == expected
        for market, _ in snapshots[1:]
    )


def _window_label(value: int, timeframe: SupportedTimeframe) -> str:
    total_hours = int(timeframe_delta(timeframe).total_seconds() * value // 3600)
    duration = (
        f"{total_hours // 24} days"
        if total_hours >= 24 and total_hours % 24 == 0
        else f"{total_hours} hours"
    )
    return f"{value} candles (~{duration})"


def _format_age(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    return f"{seconds // 3600}h ago"
