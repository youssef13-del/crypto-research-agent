"""Chart and market-card components used by the redesigned interface."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal, cast

import altair as alt
import pandas as pd
import streamlit as st

from crypto_research.domain.core import COIN_ID_BY_ASSET
from crypto_research.domain.market import Candle, MarketEvidence
from crypto_research.domain.research import TechnicalSnapshot
from crypto_research.interfaces.web.presentation import (
    CHART_HISTORY_LIMIT as CHART_HISTORY_LIMIT,
)
from crypto_research.interfaces.web.presentation import (
    build_asset_presentations as build_asset_presentations,
)
from crypto_research.interfaces.web.theme import ThemePalette, active_palette
from crypto_research.shared.time import timeframe_delta

ChartValueKind = Literal["price", "percent", "number"]
_SERIES_DASHES: tuple[tuple[int, ...], ...] = (
    (1, 0),
    (8, 4),
    (3, 3),
    (10, 3, 2, 3),
    (6, 2, 2, 2),
)


def format_price(value: float) -> str:
    absolute = abs(value)
    if absolute == 0 or absolute >= 1:
        return f"${value:,.2f}"
    if absolute < 1e-11:
        return f"${value:.4e}"
    decimals = min(12, max(2, 3 - math.floor(math.log10(absolute))))
    return f"${value:,.{decimals}f}"


def _chart_base_config(palette: ThemePalette) -> dict[str, object]:
    return {
        "axis": {
            "labelColor": palette.muted,
            "titleColor": palette.subtle,
            "gridColor": palette.chart_grid,
            "gridDash": [2, 4],
            "gridOpacity": 0.72,
            "domainColor": palette.border,
            "tickColor": palette.border,
            "labelPadding": 8,
            "titlePadding": 12,
        },
        "legend": {
            "labelColor": palette.muted,
            "titleColor": palette.subtle,
            "orient": "top",
            "direction": "horizontal",
            "symbolStrokeWidth": 3,
            "padding": 8,
        },
        "text": {"color": palette.ink},
        "title": {"color": palette.ink},
        "view": {"stroke": "transparent"},
    }


def asset_chart_color(symbol: str, palette: ThemePalette | None = None) -> str:
    """Return a stable accessible chart color for a supported asset symbol."""

    active = palette or active_palette()
    colors = (
        active.accent,
        active.blue,
        active.positive,
        active.teal,
        active.warning,
        active.danger,
        active.orange,
        active.lime,
    )
    base = symbol.split("/", maxsplit=1)[0].upper()
    supported = tuple(COIN_ID_BY_ASSET)
    if base in supported:
        return colors[supported.index(base) % len(colors)]
    return colors[sum(ord(character) for character in base) % len(colors)]


def render_line_chart(
    frame: pd.DataFrame,
    *,
    height: int,
    colors: Sequence[str],
    value_kind: ChartValueKind = "number",
    allow_future: bool = False,
) -> None:
    """Render a themed multi-series line chart that matches the active palette."""

    palette = active_palette()
    frame, warnings = prepare_line_chart_frame(frame, allow_future=allow_future)
    if frame.empty or frame.shape[1] == 0:
        st.info("No chart data is available.")
        return
    for warning in warnings:
        st.caption(warning)
    frame = frame.reset_index().rename(columns={frame.index.name or "index": "timestamp"})
    series_domain = [str(column) for column in frame.columns if column != "timestamp"]
    melted = frame.melt("timestamp", var_name="series", value_name="value").dropna()
    if melted.empty:
        st.info("No chart data is available.")
        return
    y_title, y_format = _line_axis(value_kind)
    color_scale = alt.Scale(
        domain=series_domain,
        range=[colors[index % len(colors)] for index in range(len(series_domain))]
        if colors
        else [palette.accent],
    )
    dash_scale = alt.Scale(
        domain=series_domain,
        range=[
            list(_SERIES_DASHES[index % len(_SERIES_DASHES)]) for index in range(len(series_domain))
        ],
    )
    hover = alt.selection_point(
        name="chart_hover",
        fields=["timestamp"],
        nearest=True,
        on="pointerover",
        empty=False,
        clear="pointerout",
    )
    base = alt.Chart(melted).encode(
        x=alt.X("timestamp:T", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(
            "value:Q",
            scale=alt.Scale(zero=False),
            axis=alt.Axis(title=y_title, format=y_format),
        ),
        color=alt.Color(
            "series:N",
            scale=color_scale,
            legend=alt.Legend(title=None),
        ),
        strokeDash=alt.StrokeDash(
            "series:N",
            scale=dash_scale,
            legend=None,
        ),
    )
    lines = base.mark_line(strokeWidth=2.5, interpolate="monotone")
    layers: list[Any] = []
    if len(series_domain) == 1:
        layers.append(
            base.mark_area(
                color=colors[0] if colors else palette.accent,
                opacity=0.08,
                interpolate="monotone",
            )
        )
    layers.append(lines)
    selectors = (
        alt.Chart(melted).mark_point(opacity=0).encode(x=alt.X("timestamp:T")).add_params(hover)
    )
    hover_points = base.mark_circle(size=70, stroke=palette.surface, strokeWidth=2).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Time (UTC)", format="%d %b %Y %H:%M UTC"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("value:Q", title="Value", format=_value_format(value_kind)),
        ],
    )
    hover_rule = (
        alt.Chart(melted)
        .mark_rule(color=palette.subtle, strokeDash=[3, 4], opacity=0.75)
        .encode(
            x=alt.X("timestamp:T"),
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        )
        .transform_filter(hover)
    )
    latest = melted.sort_values("timestamp").groupby("series", as_index=False).tail(1)
    latest_points = (
        alt.Chart(latest)
        .mark_circle(size=82, filled=True, stroke=palette.surface, strokeWidth=2)
        .encode(
            x=alt.X("timestamp:T"),
            y=alt.Y("value:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("series:N", title="Latest series"),
                alt.Tooltip("value:Q", title="Latest value", format=_value_format(value_kind)),
                alt.Tooltip("timestamp:T", title="Observed (UTC)", format="%d %b %Y %H:%M UTC"),
            ],
        )
    )
    chart = alt.layer(*layers, selectors, hover_points, hover_rule, latest_points).properties(
        height=height
    )
    chart = cast(Any, chart).interactive(bind_y=False)
    chart = cast(Any, chart).configure(**_chart_base_config(palette))
    st.altair_chart(chart, width="stretch", theme=None)


def render_market_chart(
    snapshots: list[tuple[MarketEvidence, TechnicalSnapshot]],
    *,
    normalized: bool = False,
) -> None:
    """Render a shared timestamp-aligned market chart."""

    prepared: list[tuple[MarketEvidence, pd.DataFrame]] = []
    for market, _ in snapshots:
        frame, warnings = prepare_market_chart_frame(market)
        if frame.empty:
            st.info(f"{market.symbol} has no valid candles for a chart.")
            continue
        for warning in warnings:
            st.caption(f"{market.symbol}: {warning}")
        prepared.append((market, frame))
    if not prepared:
        st.info("No chart data is available.")
        return
    rows: dict[str, list[float | None] | list[datetime]] = {}
    timestamps = sorted(
        {timestamp for _, frame in prepared for timestamp in frame["timestamp"].tolist()}
    )
    rows["timestamp"] = timestamps
    for market, frame in prepared:
        values = dict(zip(frame["timestamp"], frame["close"], strict=True))
        series = [values.get(timestamp) for timestamp in timestamps]
        if normalized:
            first = next((value for value in series if value is not None and value > 0), None)
            series = [
                ((value / first) - 1) * 100 if value is not None and first else None
                for value in series
            ]
        rows[market.symbol] = series
    frame = pd.DataFrame(rows).set_index("timestamp")
    palette = active_palette()
    series_colors = tuple(asset_chart_color(column, palette) for column in frame.columns)
    render_line_chart(
        frame,
        height=360,
        colors=series_colors,
        value_kind="percent" if normalized else "price",
    )
    if normalized:
        st.caption(
            "Indexed performance from the first shared observation; missing candles remain blank."
        )


def candle_frame(candles: Sequence[Candle], *, limit: int = 180) -> pd.DataFrame:
    """Build a timestamped OHLCV frame for candlestick and table consumers."""

    selected = list(candles[-limit:])
    return pd.DataFrame(
        {
            "timestamp": [candle.timestamp for candle in selected],
            "open": [candle.open for candle in selected],
            "high": [candle.high for candle in selected],
            "low": [candle.low for candle in selected],
            "close": [candle.close for candle in selected],
            "volume": [candle.volume for candle in selected],
        }
    )


def prepare_market_chart_frame(
    market: MarketEvidence,
    *,
    limit: int = 180,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Validate chart-only OHLCV rows without repairing provider data or changing exchanges."""

    frame = candle_frame(market.candles, limit=limit)
    if frame.empty:
        return frame, ()
    reference = market.collected_at.astimezone(UTC)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    numeric = frame[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    valid = (
        timestamps.notna()
        & numeric.notna().all(axis=1)
        & numeric.apply(lambda series: series.map(math.isfinite)).all(axis=1)
        & (numeric["volume"] >= 0)
        & (numeric["high"] >= numeric[["open", "close", "low"]].max(axis=1))
        & (numeric["low"] <= numeric[["open", "close", "high"]].min(axis=1))
        & (timestamps <= reference)
    )
    excluded = int((~valid).sum())
    normalized = numeric.loc[valid].copy()
    normalized.insert(0, "timestamp", timestamps.loc[valid])
    duplicate_count = int(normalized.duplicated(subset=["timestamp"], keep="last").sum())
    normalized = normalized.drop_duplicates(subset=["timestamp"], keep="last").sort_values(
        "timestamp"
    )
    gap_count = int(
        (normalized["timestamp"].diff() > pd.Timedelta(timeframe_delta(market.timeframe))).sum()
    )
    warnings: list[str] = []
    if excluded:
        warnings.append(f"{excluded} invalid or future candle(s) were omitted from this chart.")
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate timestamp(s) were omitted from this chart.")
    if gap_count:
        warnings.append(f"{gap_count} candle gap(s) remain; the chart is partial.")
    return normalized.reset_index(drop=True), tuple(warnings)


def prepare_line_chart_frame(
    frame: pd.DataFrame,
    *,
    allow_future: bool = False,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Normalize generic time-series frames while preserving intentional missing values."""

    if frame.empty or frame.shape[1] == 0:
        return frame, ()
    timestamps = pd.to_datetime(frame.index, utc=True, errors="coerce")
    normalized = frame.copy()
    normalized.index = timestamps
    normalized.index.name = str(frame.index.name or "timestamp")
    normalized = normalized.loc[normalized.index.notna()]
    future_rows = 0
    if not allow_future:
        future_rows = int((normalized.index > pd.Timestamp.now(tz="UTC")).sum())
        normalized = normalized.loc[normalized.index <= pd.Timestamp.now(tz="UTC")]
    for column in normalized.columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized.loc[
            ~normalized[column].map(
                lambda value: math.isfinite(value) if pd.notna(value) else True
            ),
            column,
        ] = None
    empty_rows = normalized.isna().all(axis=1)
    excluded = int(empty_rows.sum())
    normalized = normalized.loc[~empty_rows]
    duplicates = int(normalized.index.duplicated(keep="last").sum())
    normalized = normalized.loc[~normalized.index.duplicated(keep="last")].sort_index()
    warnings: list[str] = []
    if excluded:
        warnings.append(f"{excluded} invalid timestamp row(s) were omitted from this chart.")
    if duplicates:
        warnings.append(f"{duplicates} duplicate timestamp row(s) were omitted from this chart.")
    if future_rows:
        warnings.append(f"{future_rows} future timestamp row(s) were omitted from this chart.")
    return normalized, tuple(warnings)


def render_candlestick_chart(
    market: MarketEvidence,
    *,
    limit: int = 180,
    height: int = 420,
) -> None:
    """Render OHLC candles with bullish/bearish bodies and high/low wicks."""

    frame, warnings = prepare_market_chart_frame(market, limit=limit)
    if frame.empty:
        st.info("No OHLC candles are available for this chart.")
        return
    for warning in warnings:
        st.caption(warning)
    palette = active_palette()
    tooltips = [
        alt.Tooltip("timestamp:T", title="Timestamp (UTC)", format="%d %b %Y %H:%M UTC"),
        alt.Tooltip("open:Q", title="Open", format="$,.4~f"),
        alt.Tooltip("high:Q", title="High", format="$,.4~f"),
        alt.Tooltip("low:Q", title="Low", format="$,.4~f"),
        alt.Tooltip("close:Q", title="Close", format="$,.4~f"),
        alt.Tooltip("volume:Q", title="Volume", format=",.2~f"),
    ]
    hover = alt.selection_point(
        name="candle_hover",
        fields=["timestamp"],
        nearest=True,
        on="pointerover",
        empty=False,
        clear="pointerout",
    )
    zoom = alt.selection_interval(
        name="candle_zoom",
        bind="scales",
        encodings=["x"],
    )
    wicks = (
        alt.Chart(frame)
        .mark_rule(color=palette.subtle)
        .encode(
            x=alt.X("timestamp:T", title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y(
                "low:Q",
                scale=alt.Scale(zero=False),
                title="Price (USD)",
                axis=alt.Axis(format="$,.4~f"),
            ),
            y2="high:Q",
            tooltip=tooltips,
        )
    )
    bodies = (
        alt.Chart(frame)
        .mark_bar(size=6)
        .encode(
            x=alt.X("timestamp:T", title=None),
            y="open:Q",
            y2="close:Q",
            color=alt.condition(
                "datum.close >= datum.open",
                alt.value(palette.positive),
                alt.value(palette.danger),
            ),
            tooltip=tooltips,
        )
    )
    selectors = (
        alt.Chart(frame).mark_point(opacity=0).encode(x=alt.X("timestamp:T")).add_params(hover)
    )
    hover_rule = (
        alt.Chart(frame)
        .mark_rule(color=palette.subtle, strokeDash=[3, 4], opacity=0.8)
        .encode(x=alt.X("timestamp:T"))
        .transform_filter(hover)
    )
    price_chart = (
        alt.layer(wicks, bodies, selectors, hover_rule)
        .properties(height=max(260, height - 105))
        .add_params(zoom)
    )
    volume_chart = (
        alt.Chart(frame)
        .mark_bar(opacity=0.72)
        .encode(
            x=alt.X("timestamp:T", title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("volume:Q", title="Volume", axis=alt.Axis(format="~s")),
            color=alt.condition(
                "datum.close >= datum.open",
                alt.value(palette.positive),
                alt.value(palette.danger),
            ),
            tooltip=tooltips,
        )
        .properties(height=90)
    )
    chart = alt.vconcat(price_chart, volume_chart, spacing=8).resolve_scale(x="shared")
    chart = cast(Any, chart).configure(**_chart_base_config(palette))
    st.altair_chart(chart, width="stretch", theme=None)
    st.caption(
        f"{market.symbol} / {market.exchange.title()} / {market.timeframe} / "
        f"{len(frame)} candles / green up, red down."
    )


def _line_axis(value_kind: ChartValueKind) -> tuple[str | None, str]:
    labels = {
        "price": ("Price (USD)", "$,.4~f"),
        "percent": ("Return (%)", ".2f"),
        "number": (None, ".4~f"),
    }
    title, number_format = labels[value_kind]
    return title, number_format


def _value_format(value_kind: ChartValueKind) -> str:
    return _line_axis(value_kind)[1]
