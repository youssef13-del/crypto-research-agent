from datetime import timedelta

import pytest
from tests.support.fakes import fake_market_service

from crypto_research.config import Settings
from crypto_research.domain.research import AnalysisRequest
from crypto_research.interfaces.web.app import main
from crypto_research.interfaces.web.pages.dashboard import (
    DASHBOARD_LIMITS,
    _snapshots_share_window,
    _symbols_from_input,
    _window_label,
)
from crypto_research.interfaces.web.pages.home import home_page
from crypto_research.interfaces.web.pages.research import research_page
from crypto_research.tools.market import calculate_indicators


def test_native_application_pages_are_importable() -> None:
    assert callable(main)
    assert callable(home_page)
    assert callable(research_page)


def test_dashboard_resolves_known_assets_and_preserves_unknown_warning() -> None:
    symbols, coin_ids, warnings = _symbols_from_input("Bitcoin, ETH, FAKE, ???", "kraken")

    assert symbols == ["BTC/USD", "ETH/USD"]
    assert coin_ids[:2] == ["bitcoin", "ethereum"]
    assert len(warnings) == 2


def test_dashboard_rejects_five_assets_without_resolving_any_provider_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "crypto_research.interfaces.web.pages.dashboard.resolve_asset_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not resolve")),
    )
    symbols, coin_ids, warnings = _symbols_from_input(
        "BTC, ETH, SOL, ADA, XRP",
        "kraken",
        settings=Settings(_env_file=None),
    )

    assert symbols == []
    assert coin_ids == []
    assert warnings == ["Choose no more than 4 assets for one market comparison."]


def test_dashboard_allows_four_assets() -> None:
    symbols, coin_ids, warnings = _symbols_from_input(
        "BTC, ETH, SOL, ADA",
        "kraken",
        settings=Settings(_env_file=None),
    )

    assert symbols == ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD"]
    assert coin_ids[:4] == ["bitcoin", "ethereum", "solana", "cardano"]
    assert warnings == []


def test_dashboard_windows_satisfy_market_request_contract() -> None:
    for candle_limit in DASHBOARD_LIMITS:
        request = AnalysisRequest(
            user_intent="Compare Bitcoin and Ethereum",
            candle_limit=candle_limit,
        )
        assert request.candle_limit == candle_limit


def test_dashboard_window_label_uses_selected_timeframe() -> None:
    assert _window_label(48, "1h") == "48 candles (~2 days)"
    assert _window_label(120, "4h") == "120 candles (~20 days)"
    assert _window_label(48, "1d") == "48 candles (~48 days)"


def test_dashboard_only_compares_exact_shared_timestamp_windows() -> None:
    market = fake_market_service()
    technical = calculate_indicators(market.candles)
    shifted_candles = [
        candle.model_copy(update={"timestamp": candle.timestamp + timedelta(hours=1)})
        for candle in market.candles
    ]
    shifted = market.model_copy(
        update={
            "candles": shifted_candles,
            "first_time": shifted_candles[0].timestamp,
            "last_time": shifted_candles[-1].timestamp,
        }
    )

    assert _snapshots_share_window([(market, technical), (market, technical)]) is True
    assert _snapshots_share_window([(market, technical), (shifted, technical)]) is False
