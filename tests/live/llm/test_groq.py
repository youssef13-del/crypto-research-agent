"""Opt-in smoke checks for the current strict Guided Research schemas."""

from __future__ import annotations

import os

import pytest

from crypto_research.agents.fundamentals.fundamentals_analyzer import FundamentalsLiveOutput
from crypto_research.agents.market.market_analyzer import MarketLiveOutput
from crypto_research.agents.news.news_analyzer import NewsLiveOutput
from crypto_research.config import LLMProvider, Settings
from crypto_research.llm.client import LLMRole
from crypto_research.llm.groq import GroqAdapter

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_GROQ") != "1",
    reason="Set RUN_LIVE_GROQ=1 to run live Groq specialist smoke checks.",
)


def _adapter() -> GroqAdapter:
    settings = Settings()
    if settings.llm_provider is not LLMProvider.GROQ or not settings.groq_api_key:
        pytest.skip("Groq is not configured.")
    return GroqAdapter(settings, model=settings.groq_model)


def test_live_one_asset_market_schema() -> None:
    result = _adapter().generate_structured(
        role=LLMRole.MARKET,
        system_prompt=(
            "Return the strict object. Use qualitative language only and include BTC/USD once."
        ),
        user_prompt=(
            '{"assets":[{"symbol":"BTC/USD","brief":"momentum is mixed; observed risk '
            'evidence has partial coverage"}]}'
        ),
        output_schema=MarketLiveOutput,
    )

    assert [item.symbol for item in result.assets] == ["BTC/USD"]


def test_live_four_asset_specialist_schemas() -> None:
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD"]
    compact_assets = ",".join(
        f'{{"symbol":"{symbol}","brief":"verified coverage"}}' for symbol in symbols
    )
    adapter = _adapter()
    fundamentals = adapter.generate_structured(
        role=LLMRole.FUNDAMENTALS,
        system_prompt=(
            "Return the strict object with exactly one fundamentals entry per asset in order. "
            "DeFi was not requested, so defi_assets must be empty. Use no numbers."
        ),
        user_prompt=f'{{"assets":[{compact_assets}]}}',
        output_schema=FundamentalsLiveOutput,
    )
    news = adapter.generate_structured(
        role=LLMRole.RESEARCH,
        system_prompt=(
            "Return the strict object with exactly one news entry per asset in order. Use no "
            "numbers and state qualitative coverage gaps."
        ),
        user_prompt=f'{{"assets":[{compact_assets}]}}',
        output_schema=NewsLiveOutput,
    )

    assert [item.symbol for item in fundamentals.assets] == symbols
    assert fundamentals.defi_assets == []
    assert [item.symbol for item in news.assets] == symbols


def test_live_mixed_asset_conditional_defi_schema() -> None:
    result = _adapter().generate_structured(
        role=LLMRole.FUNDAMENTALS,
        system_prompt=(
            "Return fundamentals for AAVE/USD then BTC/USD. Return a DeFi entry only for "
            "AAVE/USD because it is the sole eligible asset. Use qualitative language only."
        ),
        user_prompt=(
            '{"assets":[{"symbol":"AAVE/USD","defi_eligible":true},'
            '{"symbol":"BTC/USD","defi_eligible":false}],"defi_requested":true}'
        ),
        output_schema=FundamentalsLiveOutput,
    )

    assert [item.symbol for item in result.assets] == ["AAVE/USD", "BTC/USD"]
    assert [item.symbol for item in result.defi_assets] == ["AAVE/USD"]
