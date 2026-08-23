"""Tests for the simplified public agent manifest."""

from crypto_research.agents.registry import (
    REGISTRY,
    agents_for,
    capability_groups,
    capability_options,
    compile_capability_route,
    expand_capabilities,
    guided_capability,
    labels,
    manifest_for,
)
from crypto_research.domain.core import ResearchCapability


def test_manifest_registry_includes_only_public_agents() -> None:
    assert {manifest.id for manifest in REGISTRY} == {
        "market_agent",
        "fundamentals_agent",
        "news_agent",
        "forecast_agent",
        "onchain_agent",
    }


def test_agents_for_public_capabilities() -> None:
    assert [m.id for m in agents_for({ResearchCapability.MARKET})] == ["market_agent"]
    assert [m.id for m in agents_for({ResearchCapability.RISK})] == ["market_agent"]
    assert [m.id for m in agents_for({ResearchCapability.DERIVATIVES})] == ["market_agent"]
    assert [m.id for m in agents_for({ResearchCapability.NEWS})] == ["news_agent"]
    assert [m.id for m in agents_for({ResearchCapability.FUNDAMENTALS})] == ["fundamentals_agent"]
    assert [m.id for m in agents_for({ResearchCapability.DEFI})] == ["fundamentals_agent"]
    assert [m.id for m in agents_for({ResearchCapability.ONCHAIN})] == ["onchain_agent"]


def test_compile_capability_route_uses_news_agent() -> None:
    assert compile_capability_route([ResearchCapability.NEWS]) == ["news_agent"]
    assert compile_capability_route([ResearchCapability.MARKET]) == ["market_agent"]
    assert compile_capability_route([ResearchCapability.FUNDAMENTALS]) == ["fundamentals_agent"]
    assert compile_capability_route(
        [
            ResearchCapability.FORECAST,
            ResearchCapability.NEWS,
            ResearchCapability.MARKET,
            ResearchCapability.FUNDAMENTALS,
            ResearchCapability.ONCHAIN,
        ]
    ) == [
        "market_agent",
        "fundamentals_agent",
        "onchain_agent",
        "news_agent",
        "forecast_agent",
    ]


def test_labels_returns_all_public_agents() -> None:
    assert labels() == {
        "market_agent": "Market & Risk Agent",
        "fundamentals_agent": "Fundamentals Agent",
        "news_agent": "News Agent",
        "forecast_agent": "Forecasting Agent",
        "onchain_agent": "On-Chain Activity Agent",
    }


def test_capability_options_returns_independent_research_topics() -> None:
    options = capability_options()

    assert [capability for capability, _, _ in options] == [
        ResearchCapability.MARKET,
        ResearchCapability.RISK,
        ResearchCapability.DERIVATIVES,
        ResearchCapability.FUNDAMENTALS,
        ResearchCapability.DEFI,
        ResearchCapability.ONCHAIN,
        ResearchCapability.NEWS,
        ResearchCapability.FORECAST,
    ]
    assert [label for _, label, _ in options] == [
        "Market behavior",
        "Risk assessment",
        "Derivatives positioning",
        "Fundamentals",
        "DeFi activity",
        "On-Chain Activity",
        "Recent news",
        "Price forecast",
    ]


def test_capability_groups_use_only_exposed_toggles() -> None:
    exposed = {cap for cap, _, _ in capability_options()}
    for _title, caps in capability_groups():
        assert set(caps) <= exposed
    assert capability_groups() == (
        (
            "Market lens",
            (
                ResearchCapability.MARKET,
                ResearchCapability.RISK,
                ResearchCapability.DERIVATIVES,
                ResearchCapability.FORECAST,
            ),
        ),
        (
            "Project & network",
            (
                ResearchCapability.FUNDAMENTALS,
                ResearchCapability.DEFI,
                ResearchCapability.ONCHAIN,
            ),
        ),
        ("Information", (ResearchCapability.NEWS,)),
    )


def test_guided_capabilities_expand_to_internal_collection_scopes() -> None:
    market = guided_capability(ResearchCapability.MARKET)
    risk = guided_capability(ResearchCapability.RISK)
    derivatives = guided_capability(ResearchCapability.DERIVATIVES)
    fundamentals = guided_capability(ResearchCapability.FUNDAMENTALS)
    defi = guided_capability(ResearchCapability.DEFI)
    onchain = guided_capability(ResearchCapability.ONCHAIN)
    news = guided_capability(ResearchCapability.NEWS)
    forecast = guided_capability(ResearchCapability.FORECAST)

    assert market is not None
    assert market.scopes == {ResearchCapability.MARKET}
    assert market.agent_id == "market_agent"
    assert risk is not None
    assert risk.scopes == {ResearchCapability.RISK}
    assert risk.agent_id == "market_agent"
    assert derivatives is not None
    assert derivatives.scopes == {ResearchCapability.DERIVATIVES}
    assert derivatives.agent_id == "market_agent"
    assert "derivatives" in derivatives.collectors
    assert fundamentals is not None
    assert fundamentals.scopes == {ResearchCapability.FUNDAMENTALS}
    assert fundamentals.agent_id == "fundamentals_agent"
    assert defi is not None
    assert defi.scopes == {ResearchCapability.DEFI}
    assert defi.agent_id == "fundamentals_agent"
    assert onchain is not None
    assert onchain.scopes == {ResearchCapability.ONCHAIN}
    assert onchain.agent_id == "onchain_agent"
    assert news is not None
    assert news.scopes == {ResearchCapability.NEWS}
    assert news.agent_id == "news_agent"
    assert forecast is not None
    assert forecast.scopes == {ResearchCapability.FORECAST}
    assert forecast.agent_id == "forecast_agent"


def test_expand_capabilities_maps_aggregate_public_choices() -> None:
    assert expand_capabilities({ResearchCapability.NEWS}) == {ResearchCapability.NEWS}
    assert expand_capabilities({ResearchCapability.MARKET}) == {ResearchCapability.MARKET}
    assert expand_capabilities({ResearchCapability.RISK}) == {ResearchCapability.RISK}
    assert expand_capabilities({ResearchCapability.MARKET, ResearchCapability.RISK}) == {
        ResearchCapability.MARKET,
        ResearchCapability.RISK,
    }
    assert expand_capabilities({ResearchCapability.FUNDAMENTALS}) == {
        ResearchCapability.FUNDAMENTALS,
    }
    assert expand_capabilities({ResearchCapability.DEFI}) == {ResearchCapability.DEFI}
    assert expand_capabilities({ResearchCapability.ONCHAIN}) == {ResearchCapability.ONCHAIN}


def test_manifest_for_known_id() -> None:
    assert manifest_for("market_agent").label == "Market & Risk Agent"
    assert manifest_for("fundamentals_agent").label == "Fundamentals Agent"
    assert manifest_for("news_agent").label == "News Agent"
    assert manifest_for("onchain_agent").label == "On-Chain Activity Agent"
