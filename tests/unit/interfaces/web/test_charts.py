from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
import streamlit as st
from tests.support.fakes import fake_market_service

from crypto_research.interfaces.web.components import charts
from crypto_research.interfaces.web.components.charts import (
    asset_chart_color,
    candle_frame,
    prepare_line_chart_frame,
    prepare_market_chart_frame,
)
from crypto_research.interfaces.web.theme import DARK_PALETTE


def test_candle_frame_preserves_ohlcv_timestamps_and_values() -> None:
    market = fake_market_service()
    frame = candle_frame(market.candles, limit=3)

    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(frame) == 3
    assert frame.iloc[-1]["close"] == market.current_price
    assert frame.iloc[0]["timestamp"] == market.candles[-3].timestamp


def test_line_charts_use_the_active_palette_for_widget_and_axis_colours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(charts, "active_palette", lambda: DARK_PALETTE)
    monkeypatch.setattr(
        st,
        "altair_chart",
        lambda chart, **_kwargs: captured.update(spec=chart.to_dict()),
    )
    frame = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.DatetimeIndex([datetime(2026, 1, 1), datetime(2026, 1, 2)]),
    )

    charts.render_line_chart(frame, height=180, colors=(DARK_PALETTE.accent,))

    spec = captured["spec"]
    assert isinstance(spec, dict)
    assert spec["config"]["axis"]["labelColor"] == DARK_PALETTE.muted
    assert spec["config"]["axis"]["gridColor"] == DARK_PALETTE.chart_grid
    assert spec["config"]["legend"]["orient"] == "top"
    assert any(layer.get("mark", {}).get("type") == "area" for layer in spec["layer"])
    assert any("strokeDash" in layer.get("encoding", {}) for layer in spec["layer"])
    assert any(parameter.get("name") == "chart_hover" for parameter in spec["params"])


def test_asset_chart_colors_are_stable_and_distinct() -> None:
    assert asset_chart_color("BTC/USD", DARK_PALETTE) == asset_chart_color("BTC/USDT", DARK_PALETTE)
    assert asset_chart_color("BTC/USD", DARK_PALETTE) != asset_chart_color("ETH/USD", DARK_PALETTE)


def test_candlestick_chart_links_price_and_volume_with_hover_and_zoom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(charts, "active_palette", lambda: DARK_PALETTE)
    monkeypatch.setattr(
        st,
        "altair_chart",
        lambda chart, **_kwargs: captured.update(spec=chart.to_dict()),
    )

    charts.render_candlestick_chart(fake_market_service(), limit=24)

    spec = captured["spec"]
    assert isinstance(spec, dict)
    assert len(spec["vconcat"]) == 2
    assert spec["resolve"]["scale"]["x"] == "shared"
    assert any(parameter.get("name") == "candle_zoom" for parameter in spec["params"])
    assert spec["vconcat"][1]["encoding"]["y"]["title"] == "Volume"


def test_chart_preparation_excludes_future_and_duplicate_market_candles() -> None:
    market = fake_market_service()
    duplicate = market.candles[-2]
    future = market.candles[-1].model_copy(
        update={"timestamp": market.collected_at + timedelta(hours=1)}
    )
    corrupted = market.model_copy(update={"candles": [*market.candles[:-1], duplicate, future]})

    frame, warnings = prepare_market_chart_frame(corrupted, limit=len(corrupted.candles))

    assert not frame.empty
    assert all(frame["timestamp"] <= market.collected_at)
    assert any("future" in warning for warning in warnings)
    assert any("duplicate" in warning for warning in warnings)


def test_chart_preparation_marks_gapped_market_candles_as_partial() -> None:
    market = fake_market_service()
    gapped_candles = [market.candles[0], market.candles[2], market.candles[-1]]
    gapped = market.model_copy(
        update={
            "candles": gapped_candles,
            "first_time": gapped_candles[0].timestamp,
            "last_time": gapped_candles[-1].timestamp,
            "current_price": gapped_candles[-1].close,
        }
    )

    _, warnings = prepare_market_chart_frame(gapped, limit=len(gapped.candles))

    assert any("partial" in warning for warning in warnings)


def test_line_chart_preparation_drops_invalid_and_future_rows_without_filling_gaps() -> None:
    now = datetime.now(UTC)
    frame = pd.DataFrame(
        {"close": [1.0, float("nan"), 2.0, 3.0]},
        index=pd.DatetimeIndex(
            [
                now - timedelta(hours=2),
                now - timedelta(hours=1),
                now - timedelta(hours=2),
                now + timedelta(hours=1),
            ],
            name="timestamp",
        ),
    )

    prepared, warnings = prepare_line_chart_frame(frame)

    assert prepared.index.is_monotonic_increasing
    assert len(prepared) == 1
    assert prepared.iloc[0, 0] == 2.0
    assert any("future" in warning for warning in warnings)
