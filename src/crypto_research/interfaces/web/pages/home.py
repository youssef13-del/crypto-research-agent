"""ChainScope landing page with a resilient Kraken market pulse."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from statistics import mean, median

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from crypto_research.domain.core import ASSET_ALIASES
from crypto_research.domain.research import OpportunityCandidate, OpportunityScanResult
from crypto_research.interfaces.web import runtime
from crypto_research.interfaces.web.components.charts import format_price
from crypto_research.interfaces.web.components.layout import (
    render_disclaimer,
    render_empty_panel,
    render_section_header,
)
from crypto_research.interfaces.web.pages.dashboard import DASHBOARD_ASSETS_KEY, dashboard_page
from crypto_research.interfaces.web.pages.research import research_page

LOGGER = logging.getLogger(__name__)
_OVERVIEW_RESULT_KEY = "home_market_overview"
_OVERVIEW_STORED_AT_KEY = "home_market_overview_stored_at"
_OVERVIEW_CACHE_TTL = timedelta(minutes=5)
_OVERVIEW_STALE_LIMIT = timedelta(minutes=30)
_MAJOR_ASSETS = ("BTC", "ETH", "SOL", "XRP")


def home_page() -> None:
    workspace = runtime.current_workspace()
    _render_hero()
    _render_actions()

    header = st.columns([5, 1], vertical_alignment="bottom")
    with header[0]:
        render_section_header(
            "Kraken market pulse",
            (
                "One-hour market data for your saved assets."
                if workspace is not None
                else "One-hour market data for supported assets."
            ),
        )
    refresh = header[1].button(
        "Refresh",
        key="home-market-refresh",
        icon=":material/refresh:",
        width="stretch",
        help="Bypass the five-minute overview cache without clearing workspace state.",
    )

    with st.spinner("Loading Kraken market pulse…", show_time=True):
        overview, state, warning = _home_overview(force_refresh=refresh)
    if overview is None or not overview.candidates:
        render_empty_panel(
            "Market pulse unavailable",
            "No current or recently cached Kraken watchlist data is available.",
            icon="—",
        )
    else:
        _render_market_overview(overview, state=state, warning=warning)

    render_disclaimer(
        "Market observations and experimental forecasts are educational research, not financial "
        "advice, trading instructions, or guaranteed outcomes."
    )


def _render_hero() -> None:
    st.markdown(
        '<section class="cs-hero cs-home-hero">'
        '<div class="cs-eyebrow">Workspace</div>'
        "<h1>Crypto research, organized.</h1>"
        "<p>Run specialist research, compare market history, and review saved reports from "
        "one account.</p></section>",
        unsafe_allow_html=True,
    )


def _render_actions() -> None:
    st.markdown('<div class="cs-home-section-label">Start here</div>', unsafe_allow_html=True)
    with st.container(key="home-actions"):
        actions = st.columns([1.35, 1])
        _render_action(
            actions[0],
            "Guided Research",
            ("Build a sourced report for one asset or compare up to four."),
            research_page,
            "research",
            "Open Guided Research",
            ":material/science:",
            primary=True,
        )
        _render_action(
            actions[1],
            "Market Dashboard",
            "Compare synchronized prices, candles, and technical metrics.",
            dashboard_page,
            "dashboard",
            "Open Market Dashboard",
            ":material/monitoring:",
        )


def _home_overview(*, force_refresh: bool) -> tuple[OpportunityScanResult | None, str, str | None]:
    now = datetime.now(UTC)
    cached = st.session_state.get(_OVERVIEW_RESULT_KEY)
    stored_at = st.session_state.get(_OVERVIEW_STORED_AT_KEY)
    age = now - stored_at if isinstance(stored_at, datetime) else None
    if (
        not force_refresh
        and isinstance(cached, OpportunityScanResult)
        and age is not None
        and age <= _OVERVIEW_CACHE_TTL
    ):
        return cached, "Cached", None
    try:
        result = runtime.load_home_market_overview()
        if not result.candidates:
            raise ValueError("Kraken returned no usable watchlist candidates.")
    except Exception as exc:
        LOGGER.warning("Home market overview failed (%s).", type(exc).__name__)
        if (
            isinstance(cached, OpportunityScanResult)
            and cached.candidates
            and age is not None
            and age <= _OVERVIEW_STALE_LIMIT
        ):
            return cached, "Stale", "Live refresh failed; showing the latest recent market scan."
        return None, "Unavailable", None
    st.session_state[_OVERVIEW_RESULT_KEY] = result
    st.session_state[_OVERVIEW_STORED_AT_KEY] = now
    return result, "Live", None


def _render_market_overview(
    result: OpportunityScanResult, *, state: str, warning: str | None
) -> None:
    state_class = "cs-home-state-stale" if state == "Stale" else "cs-home-state-live"
    st.markdown(
        '<div class="cs-home-pulse-meta">'
        f'<span class="cs-home-state {state_class}">{state}</span>'
        f"<span>{result.exchange.title()}</span><span>{result.timeframe}</span>"
        f"<span>Observed {result.collected_at.strftime('%d %b %Y · %H:%M UTC')}</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    if warning:
        st.warning(warning)

    workspace = runtime.current_workspace()
    watchlist = workspace.watchlist if workspace is not None else _MAJOR_ASSETS
    if workspace is not None and not watchlist:
        render_empty_panel(
            "Your watchlist is empty",
            "Open Account to choose the assets shown in your personal market pulse.",
            icon="+",
        )
        return
    by_asset = {candidate.asset: candidate for candidate in result.candidates}
    for start in range(0, len(watchlist), 4):
        symbols = watchlist[start : start + 4]
        columns = st.columns(len(symbols))
        for column, symbol in zip(columns, symbols, strict=True):
            _render_major_asset(
                column,
                symbol=symbol,
                candidate=by_asset.get(symbol),
                quick_actions=workspace is not None,
            )

    candidates = (
        [candidate for candidate in result.candidates if candidate.asset in watchlist]
        if workspace is not None
        else result.candidates
    )
    if not candidates:
        st.warning("The provider did not return usable market data for your saved watchlist.")
        return
    bullish = sum(item.trend == "bullish" for item in candidates)
    summary = st.columns(4)
    requested_count = len(watchlist) if workspace is not None else len(ASSET_ALIASES)
    summary[0].metric("Watchlist coverage", f"{len(candidates)}/{requested_count}")
    summary[1].metric("Bullish breadth", f"{bullish / len(candidates):.0%}")
    summary[2].metric(
        "Median 24h return",
        f"{median(item.momentum_24h for item in candidates):+.2f}%",
    )
    summary[3].metric(
        "Average volatility",
        f"{mean(item.volatility_24h for item in candidates):.2f}%",
    )

    movers = sorted(candidates, key=lambda item: item.momentum_24h)
    mover_columns = st.columns(2)
    _render_mover(
        mover_columns[0],
        title="Strongest 24h mover",
        candidate=movers[-1],
        positive=True,
    )
    _render_mover(
        mover_columns[1],
        title="Weakest 24h mover",
        candidate=movers[0],
        positive=False,
    )

    requested_assets = watchlist if workspace is not None else tuple(ASSET_ALIASES)
    missing = [symbol for symbol in requested_assets if symbol not in by_asset]
    notes: list[str] = []
    if missing:
        notes.append("Unavailable assets: " + ", ".join(missing))
    if result.warnings:
        notes.append(f"Kraken returned {len(result.warnings)} provider warning(s).")
    if notes:
        st.markdown(
            '<div class="cs-home-coverage"><strong>Partial coverage</strong><br>'
            + " ".join(notes)
            + "</div>",
            unsafe_allow_html=True,
        )


def _render_major_asset(
    column: DeltaGenerator,
    *,
    symbol: str,
    candidate: OpportunityCandidate | None,
    quick_actions: bool = False,
) -> None:
    with column, st.container(border=True):
        st.markdown('<div class="cs-home-asset-card"></div>', unsafe_allow_html=True)
        if candidate is None:
            st.markdown(f"### {symbol}")
            st.caption("Temporarily unavailable")
        else:
            st.metric(
                symbol,
                format_price(candidate.current_price),
                f"{candidate.momentum_24h:+.2f}%",
            )
            st.caption(
                f"{candidate.trend.title()} trend · "
                f"{candidate.volatility_24h:.2f}% observed volatility"
            )
        if quick_actions:
            _render_asset_quick_actions(symbol)


def _render_asset_quick_actions(symbol: str) -> None:
    workspace = runtime.current_workspace()
    assert workspace is not None
    actions = st.columns(2)
    if actions[0].button(
        "Research",
        key=f"home-research-{symbol}",
        width="stretch",
    ):
        st.session_state["guided-assets"] = [symbol]
        st.session_state["guided-exchange"] = workspace.preferences.default_exchange
        st.session_state["guided-timeframe"] = workspace.preferences.default_timeframe
        st.switch_page(
            st.Page(
                research_page,
                title="Research",
                icon=":material/science:",
                url_path="research",
            )
        )
    if actions[1].button(
        "Chart",
        key=f"home-chart-{symbol}",
        width="stretch",
    ):
        st.session_state[DASHBOARD_ASSETS_KEY] = symbol
        st.switch_page(
            st.Page(
                dashboard_page,
                title="Market Dashboard",
                icon=":material/dashboard:",
                url_path="dashboard",
            )
        )


def _render_mover(
    column: DeltaGenerator,
    *,
    title: str,
    candidate: OpportunityCandidate,
    positive: bool,
) -> None:
    tone = "positive" if positive else "negative"
    with column:
        st.markdown(
            f'<article class="cs-home-mover cs-home-mover-{tone}">'
            f"<span>{title}</span><strong>{candidate.asset}</strong>"
            f"<p>{candidate.momentum_24h:+.2f}% · {candidate.trend.title()} trend</p>"
            "</article>",
            unsafe_allow_html=True,
        )


def _render_action(
    column: DeltaGenerator,
    title: str,
    body: str,
    page: Callable[[], None],
    url_path: str,
    label: str,
    icon: str,
    *,
    primary: bool = False,
) -> None:
    card_class = "cs-home-action cs-home-action-primary" if primary else "cs-home-action"
    kicker = "Primary workspace" if primary else "Explore"
    with column:
        st.markdown(
            f'<article class="{card_class}"><span>{kicker}</span>'
            f"<h3>{title}</h3><p>{body}</p></article>",
            unsafe_allow_html=True,
        )
        st.page_link(
            st.Page(page, title=title, icon=icon, url_path=url_path),
            label=label,
            icon=icon,
            width="stretch",
        )
