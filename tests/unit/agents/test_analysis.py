"""Regression coverage for the no-synthesis specialist analysis pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, TypeVar, cast

import pytest
from pydantic import BaseModel
from tests.support.fakes import fake_fundamental_service, fake_market_service, fake_news_service

from crypto_research.agents.guardrails import evidence_first_answer, no_evidence_message
from crypto_research.agents.shared_analysis import SpecialistAnalysisRunner, _raw_result
from crypto_research.domain.analytics import evaluate_research_risk
from crypto_research.domain.evidence import DeveloperActivity
from crypto_research.domain.research import (
    AgentAnswer,
    AnalysisAsset,
    AnalysisInputs,
    AssetResearchBundle,
    CollectionContext,
    DefiEvidence,
    FundamentalsAgentResult,
    MarketAgentResult,
    MarketComparisonAsset,
    MarketComparisonResult,
    NewsEvidence,
    NewsItem,
    OpportunityCandidate,
    OpportunityScanResult,
    ResearchAgentResult,
    ResearchCapability,
    TechnicalSnapshot,
)
from crypto_research.llm.client import LLMRole
from crypto_research.orchestration.evidence import EvidenceSpec, build_specialist_evidence
from crypto_research.shared.text import estimate_tokens

OutputT = TypeVar("OutputT", bound=BaseModel)


class _RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.roles: list[LLMRole] = []
        self.last_call_used_fallback = False

    def generate_structured(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[OutputT],
    ) -> OutputT:
        del system_prompt
        payload = json.loads(user_prompt)
        assert isinstance(payload, dict)
        self.calls.append(cast(dict[str, object], payload))
        self.roles.append(role)
        raw_result = cast("dict[str, Any]", payload.get("analysis_briefs", {}))
        collection = cast("dict[str, Any]", raw_result.get("collection_status", {}))
        assets = [item["symbol"] for item in collection.get("requested_assets", [])]
        name = output_schema.__name__
        if name == "MarketLiveOutput":
            value = {
                "verdict": "The market posture is mixed across the selected assets.",
                "assets": [
                    {
                        "symbol": symbol,
                        "market_analysis": "Momentum is constructive but uneven.",
                        "risk_analysis": "Observed risk depends on evidence coverage.",
                    }
                    for symbol in assets
                ],
                "comparison": "Relative strength differs across the selected assets.",
                "limitations": [],
                "confidence": "high",
            }
        elif name == "FundamentalsLiveOutput":
            value = {
                "verdict": "Fundamental coverage provides a differentiated project read.",
                "assets": [
                    {"symbol": symbol, "analysis": "Project scale and supply context differ."}
                    for symbol in assets
                ],
                "defi_assets": [],
                "comparison": "Project coverage differs across the selected assets.",
                "limitations": [],
                "confidence": "high",
            }
        elif name in {"NewsLiveOutput", "OnchainLiveOutput"}:
            value = {
                "verdict": "Recent coverage is uneven across the selected assets.",
                "assets": [
                    {"symbol": symbol, "analysis": "The supplied headline shapes current context."}
                    for symbol in assets
                ],
                "comparison": "News themes differ across the selected assets.",
                "limitations": [],
                "confidence": "high",
            }
        else:
            return cast(
                OutputT,
                AgentAnswer(
                    agent=str(payload["agent"]),
                    answer="Verified evidence is available for each selected asset.",
                    confidence=0.8,
                ),
            )
        return output_schema.model_validate(value)


class _StrictSpecialistLLM:
    last_call_used_fallback = False

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []
        self.structured_calls = 0

    def generate_structured(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[OutputT],
    ) -> OutputT:
        self.structured_calls += 1
        self.calls.append(
            {"role": role, "system_prompt": system_prompt, "user_prompt": user_prompt}
        )
        return output_schema.model_validate(self.payload)


class _ProgrammingFailureLLM:
    last_call_used_fallback = False

    def generate_structured(self, **_kwargs: object) -> Any:
        raise RuntimeError("programming failure")


def test_market_raw_result_keeps_per_asset_summaries_for_four_coin_analysis() -> None:
    market = fake_market_service()
    inputs = AnalysisInputs(
        assets=[
            AnalysisAsset(requested_name=symbol, symbol=f"{symbol}/USD", coin_id=symbol.casefold())
            for symbol in ("BTC", "ETH", "SOL", "ADA")
        ],
        market_comparison_result=MarketComparisonResult(
            assets=[
                MarketComparisonAsset(
                    market=market.model_copy(
                        update={"symbol": f"{symbol}/USD", "coin_id": symbol.casefold()}
                    ),
                    technical=TechnicalSnapshot(trend="bullish", rsi=55.0 + index),
                )
                for index, symbol in enumerate(("BTC", "ETH", "SOL", "ADA"))
            ]
        ),
    )

    raw = _raw_result(
        inputs,
        agent="market_agent",
        capabilities=[ResearchCapability.MARKET, ResearchCapability.RISK],
    )
    per_asset_market = cast("list[dict[str, Any]]", raw["per_asset_market"])

    assert [item["symbol"] for item in per_asset_market] == [
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "ADA/USD",
    ]
    assert all("analysis_brief" in item for item in per_asset_market)


def test_news_raw_result_groups_relevant_news_by_selected_coin() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    inputs = AnalysisInputs(
        assets=[
            AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin"),
            AnalysisAsset(requested_name="ETH", symbol="ETH/USD", coin_id="ethereum"),
        ],
        research_result=_news_result(timestamp),
    )

    raw = _raw_result(inputs, agent="news_agent", capabilities=[ResearchCapability.NEWS])

    grouped = cast("list[dict[str, Any]]", raw["per_asset_news"])
    assert [item["symbol"] for item in grouped] == ["BTC/USD", "ETH/USD"]
    btc_items = cast("list[dict[str, Any]]", grouped[0]["items"])
    eth_items = cast("list[dict[str, Any]]", grouped[1]["items"])
    assert btc_items[0]["title"] == "BTC headline 1"
    assert eth_items[0]["title"] == "ETH headline 1"
    assert btc_items[0]["excerpt"] == "BTC relevant news."
    assert grouped[0]["coverage"] == {
        "validated_items": 1,
        "publisher_count": 1,
    }


def test_live_news_analysis_is_source_aware_without_repeating_the_headline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin")
    timestamp = datetime(2026, 8, 17, 12, 30, tzinfo=UTC)
    evidence = {
        "news.btc-usd.low": {
            "claim_type": "recent_news",
            "asset": asset.symbol,
            "source": "Unknown Blog",
            "observed_at": timestamp.isoformat(),
            "payload": {
                "title": "Unverified Bitcoin market commentary",
                "published_at": timestamp,
                "source_quality": "low",
            },
        },
        "news.btc-usd.0": {
            "claim_type": "recent_news",
            "asset": asset.symbol,
            "source": "CoinDesk",
            "payload": {
                "title": "Bitcoin developers publish a network upgrade proposal",
                "excerpt": "The proposal outlines a staged activation process.",
                "published_at": timestamp,
                "source_quality": "high",
            },
        },
        "news.btc-usd.1": {
            "claim_type": "recent_news",
            "asset": asset.symbol,
            "source": "Cointelegraph",
            "payload": {
                "title": "Developers discuss the same upgrade timeline",
                "excerpt": "The report describes open implementation questions.",
                "published_at": timestamp,
                "source_quality": "medium",
            },
        },
    }
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(available_evidence=evidence),
    )
    llm = _StrictSpecialistLLM(
        {
            "verdict": "**Verdict:** A proposed network change dominates current coverage.",
            "assets": [
                {
                    "symbol": asset.symbol,
                    "analysis": (
                        "### Why it matters: The proposal creates an implementation milestone, "
                        "while open activation details limit certainty. The proposal creates an "
                        "implementation milestone, while open activation details limit certainty."
                    ),
                }
            ],
            "comparison": "",
            "limitations": [],
            "confidence": "high",
        }
    )
    inputs = AnalysisInputs(
        assets=[asset],
        research_result=ResearchAgentResult(
            news=NewsEvidence(items=[], query="BTC", collected_at=timestamp),
            asset_results=[
                AssetResearchBundle(
                    asset=asset,
                    news=NewsEvidence(
                        items=[
                            NewsItem(
                                publisher="CoinDesk",
                                title="Bitcoin developers publish a network upgrade proposal",
                                excerpt="The proposal outlines a staged activation process.",
                                published_at=timestamp,
                                source_quality="high",
                            )
                        ],
                        query="BTC",
                        collected_at=timestamp,
                    ),
                )
            ],
            requested_capabilities=[ResearchCapability.NEWS],
        ),
    )

    answer = SpecialistAnalysisRunner(llm=llm, live_mode=True).run(
        "Explain the latest Bitcoin news.",
        inputs,
        agent="news_agent",
        capabilities=[ResearchCapability.NEWS],
    )

    assert answer.analysis_state == "live"
    assert answer.structured_analysis is not None
    assert answer.structured_analysis.verdict == (
        "A proposed network change dominates current coverage."
    )
    section = answer.structured_analysis.sections[0].text
    assert section == (
        "The proposal creates an implementation milestone, while open activation details "
        "limit certainty."
    )
    assert "Bitcoin developers publish" not in section
    assert 'CoinDesk reported "Bitcoin developers publish' in answer.answer
    assert "17 Aug 2026, 12:30 UTC" in answer.answer
    assert answer.evidence[0].evidence_ids == ["news.btc-usd.0", "news.btc-usd.1"]
    assert "Unknown Blog" not in answer.evidence[0].statement
    prompt = json.loads(cast(str, llm.calls[0]["user_prompt"]))
    brief = prompt["analysis_briefs"]["per_asset_news"][0]
    assert brief["coverage"]["validated_items"] == 1
    assert brief["stories"][0]["publisher"] == "CoinDesk"
    assert brief["stories"][0]["quality"] == "high"
    assert "staged activation" in brief["stories"][0]["excerpt"]


def test_live_news_rejects_unsupported_price_causality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin")
    evidence = {
        "news.btc-usd.0": {
            "claim_type": "recent_news",
            "asset": asset.symbol,
            "source": "CoinDesk",
            "payload": {"title": "Bitcoin network update"},
        }
    }
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(available_evidence=evidence),
    )
    llm = _StrictSpecialistLLM(
        {
            "verdict": "The report caused a Bitcoin price rally.",
            "assets": [
                {
                    "symbol": asset.symbol,
                    "analysis": "The report drove the price surge and will lift prices further.",
                }
            ],
            "comparison": "",
            "limitations": [],
            "confidence": "high",
        }
    )

    answer = SpecialistAnalysisRunner(llm=llm, live_mode=True).run(
        "Analyze Bitcoin news.",
        AnalysisInputs(assets=[asset]),
        agent="news_agent",
        capabilities=[ResearchCapability.NEWS],
    )

    assert answer.structured_analysis is not None
    combined = " ".join(
        [
            answer.structured_analysis.verdict,
            *(section.text for section in answer.structured_analysis.sections),
        ]
    ).casefold()
    assert "caused" not in combined
    assert "drove" not in combined
    assert "will lift" not in combined
    assert "market impact remains unproven" in combined


def test_four_asset_news_prompt_keeps_one_source_aware_story_per_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2026, 8, 17, 12, 30, tzinfo=UTC)
    assets = [
        AnalysisAsset(requested_name=symbol, symbol=f"{symbol}/USD", coin_id=symbol.casefold())
        for symbol in ("BTC", "ETH", "SOL", "ADA")
    ]
    evidence = {
        f"news.{asset.evidence_key}.0": {
            "claim_type": "recent_news",
            "asset": asset.symbol,
            "source": f"{asset.requested_name} Publisher",
            "payload": {
                "title": f"{asset.requested_name} developers publish a project update",
                "excerpt": "The report describes the next implementation milestone.",
                "published_at": timestamp,
                "source_quality": "high",
            },
        }
        for asset in assets
    }
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(available_evidence=evidence),
    )
    llm = _StrictSpecialistLLM(
        {
            "verdict": "Current project reporting differs across the selected assets.",
            "assets": [
                {
                    "symbol": asset.symbol,
                    "analysis": "The report defines a milestone while execution remains open.",
                }
                for asset in assets
            ],
            "comparison": "The reports differ in their implementation focus and source depth.",
            "limitations": [],
            "confidence": "high",
        }
    )
    bundles = [
        AssetResearchBundle(
            asset=asset,
            news=NewsEvidence(
                items=[
                    NewsItem(
                        publisher=f"{asset.requested_name} Publisher",
                        title=f"{asset.requested_name} developers publish a project update",
                        excerpt="The report describes the next implementation milestone.",
                        published_at=timestamp,
                        source_quality="high",
                    )
                ],
                query=asset.requested_name,
                collected_at=timestamp,
            ),
        )
        for asset in assets
    ]

    answer = SpecialistAnalysisRunner(llm=llm, live_mode=True).run(
        "Compare the latest project news.",
        AnalysisInputs(
            assets=assets,
            research_result=ResearchAgentResult(
                news=NewsEvidence(items=[], query="comparison", collected_at=timestamp),
                asset_results=bundles,
                requested_capabilities=[ResearchCapability.NEWS],
            ),
        ),
        agent="news_agent",
        capabilities=[ResearchCapability.NEWS],
    )

    assert answer.analysis_state == "live"
    prompt = json.loads(cast(str, llm.calls[0]["user_prompt"]))
    briefs = prompt["analysis_briefs"]["per_asset_news"]
    assert [brief["symbol"] for brief in briefs] == [asset.symbol for asset in assets]
    assert all(len(brief["stories"]) == 1 for brief in briefs)
    assert all(brief["stories"][0]["publisher"] for brief in briefs)
    assert (
        estimate_tokens(
            str(llm.calls[0]["system_prompt"]),
            str(llm.calls[0]["user_prompt"]),
        )
        <= 1_700
    )


def test_fundamentals_raw_result_separates_fundamentals_from_available_defi() -> None:
    inputs = AnalysisInputs(
        assets=[
            AnalysisAsset(requested_name="AAVE", symbol="AAVE/USD", coin_id="aave"),
            AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin"),
        ],
        fundamentals_result=FundamentalsAgentResult(
            asset_results=[
                AssetResearchBundle(
                    asset=AnalysisAsset(requested_name="AAVE", symbol="AAVE/USD", coin_id="aave"),
                    fundamentals=fake_fundamental_service().model_copy(
                        update={"name": "Aave", "symbol": "aave"}
                    ),
                    defi=DefiEvidence(
                        protocol="Aave",
                        tvl_usd=1_000_000.0,
                        chains=["Ethereum"],
                        status="available",
                        collected_at=datetime(2026, 1, 1, tzinfo=UTC),
                    ),
                ),
                AssetResearchBundle(
                    asset=AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin"),
                    fundamentals=fake_fundamental_service(),
                    defi=None,
                ),
            ],
            requested_capabilities=[ResearchCapability.FUNDAMENTALS, ResearchCapability.DEFI],
        ),
    )

    raw = _raw_result(
        inputs,
        agent="fundamentals_agent",
        capabilities=[ResearchCapability.FUNDAMENTALS, ResearchCapability.DEFI],
    )

    summaries = cast("list[dict[str, Any]]", raw["per_asset_fundamentals"])
    first_fundamentals = cast("dict[str, Any]", summaries[0]["fundamentals"])
    first_defi = cast("dict[str, Any]", summaries[0]["defi"])
    second_fundamentals = cast("dict[str, Any]", summaries[1]["fundamentals"])
    assert first_fundamentals["name"] == "Aave"
    assert "analysis_brief" in first_fundamentals
    assert first_defi["protocol"] == "Aave"
    assert "analysis_brief" in first_defi
    assert second_fundamentals["name"] == "Bitcoin"
    assert summaries[1]["defi"] is None


def test_live_fundamentals_replaces_quantified_model_copy_with_grounded_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin")
    fundamentals = fake_fundamental_service().model_copy(
        update={
            "developer_activity": DeveloperActivity(commits_4_weeks=108),
        }
    )
    evidence = {
        "fundamentals": {
            "claim_type": "project_fundamentals",
            "asset": asset.symbol,
            "source": "CoinGecko",
            "payload": fundamentals.model_dump(mode="json"),
        }
    }
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(available_evidence=evidence),
    )
    llm = _StrictSpecialistLLM(
        {
            "verdict": "The evidence differs across the selected assets.",
            "assets": [
                {
                    "symbol": asset.symbol,
                    "analysis": (
                        "Market cap exceeds a trillion dollars, supply is around twenty million, "
                        "and developer activity shows hundreds of commits."
                    ),
                }
            ],
            "defi_assets": [],
            "comparison": "",
            "limitations": [],
            "confidence": "high",
        }
    )
    inputs = AnalysisInputs(
        assets=[asset],
        requested_capabilities=[ResearchCapability.FUNDAMENTALS],
        fundamentals_result=FundamentalsAgentResult(
            fundamentals=fundamentals,
            asset_results=[AssetResearchBundle(asset=asset, fundamentals=fundamentals)],
            requested_capabilities=[ResearchCapability.FUNDAMENTALS],
        ),
    )

    answer = SpecialistAnalysisRunner(llm=llm, live_mode=True).run(
        "Give me a live fundamentals analysis of BTC.",
        inputs,
        agent="fundamentals_agent",
        capabilities=[ResearchCapability.FUNDAMENTALS],
    )

    assert answer.analysis_state == "live"
    assert answer.structured_analysis is not None
    assert answer.structured_analysis.verdict == (
        "The current provider snapshot supports a focused fundamentals read for this asset."
    )
    section = answer.structured_analysis.sections[0].text
    assert "maturity signal" in section
    assert "recent code activity" in section
    assert "$1.00 million" in section
    assert "provider rank was #1" in section
    assert "circulating supply was 19.00 million" in section
    assert "maximum supply was 21.00 million" in section
    assert "108 commits" in section
    assert not {"trillion", "twenty", "hundreds"} & set(section.casefold().split())
    prompt = json.loads(cast(str, llm.calls[0]["user_prompt"]))
    prompt_fundamentals = prompt["analysis_briefs"]["per_asset_fundamentals"][0]["fundamentals"]
    assert prompt_fundamentals["analysis_signals"] == {
        "market_position": "leading",
        "supply_profile": "limited_remaining_issuance",
        "development_activity": "active",
    }
    assert "market_cap" not in prompt_fundamentals


def test_fundamentals_specialist_evidence_reads_dedicated_fundamentals_result() -> None:
    asset = AnalysisAsset(requested_name="AAVE", symbol="AAVE/USD", coin_id="aave")
    fundamentals = fake_fundamental_service().model_copy(update={"name": "Aave", "symbol": "aave"})
    inputs = AnalysisInputs(
        assets=[asset],
        requested_capabilities=[ResearchCapability.FUNDAMENTALS],
        fundamentals_result=FundamentalsAgentResult(
            fundamentals=fundamentals,
            asset_results=[
                AssetResearchBundle(asset=asset, fundamentals=fundamentals),
            ],
            requested_capabilities=[ResearchCapability.FUNDAMENTALS],
        ),
    )

    spec = build_specialist_evidence(
        inputs,
        agent="fundamentals_agent",
        capabilities=[ResearchCapability.FUNDAMENTALS],
    )

    assert "fundamentals" in spec.available_evidence
    fundamentals_record = cast("dict[str, object]", spec.available_evidence["fundamentals"])
    assert fundamentals_record["claim_type"] == "project_fundamentals"


def test_fundamentals_missing_evidence_message_never_uses_news_copy() -> None:
    message = no_evidence_message(
        AnalysisInputs(requested_capabilities=[ResearchCapability.FUNDAMENTALS]),
        agent="fundamentals_agent",
    )

    assert "fundamentals" in message.casefold()
    assert "articles" not in message.casefold()


def test_market_fallback_groups_per_coin_and_names_risk_assets() -> None:
    answer = evidence_first_answer(
        agent="market_agent",
        question="Compare SOL and ADA market risk",
        message="Fallback analysis.",
        limitations=(),
        evidence={
            "market.sol-usd": {
                "claim_type": "market_snapshot",
                "asset": "SOL/USD",
                "payload": {
                    "snapshot": {"current_price": 76.44, "exchange": "kraken"},
                    "ohlcv_features": {"returns": {"24h": 1.16}},
                },
            },
            "risk.sol-usd": {
                "claim_type": "deterministic_risk_assessment",
                "asset": "SOL/USD",
                "payload": {"score": 12.0, "band": "low"},
            },
            "market.ada-usd": {
                "claim_type": "market_snapshot",
                "asset": "ADA/USD",
                "payload": {
                    "snapshot": {"current_price": 0.1964, "exchange": "kraken"},
                    "ohlcv_features": {"returns": {"24h": -1.47}},
                },
            },
            "risk.ada-usd": {
                "claim_type": "deterministic_risk_assessment",
                "asset": "ADA/USD",
                "payload": {"score": 0.0, "band": "low"},
            },
        },
    )

    assert "SOL/USD:" in answer.answer
    assert "ADA/USD:" in answer.answer
    assert "SOL/USD observed risk was low with score 12" in answer.answer
    assert "ADA/USD observed risk was low with score 0" in answer.answer
    assert answer.answer.startswith("The verified market read")
    assert "On comparison" in answer.answer
    assert answer.analysis_state == "evidence_only"
    assert answer.structured_analysis is not None
    assert answer.structured_analysis.verdict.startswith("Validated market evidence")
    assert [(section.asset, section.scope) for section in answer.structured_analysis.sections] == [
        ("SOL/USD", "market"),
        ("SOL/USD", "risk"),
        ("ADA/USD", "market"),
        ("ADA/USD", "risk"),
    ]
    assert "$76.44" in answer.structured_analysis.sections[0].text
    assert "score 12" in answer.structured_analysis.sections[1].text


@pytest.mark.parametrize(
    ("agent", "record", "expected_scope", "expected_text"),
    [
        (
            "fundamentals_agent",
            {
                "claim_type": "project_fundamentals",
                "asset": "BTC/USD",
                "payload": {"name": "Bitcoin", "market_cap": 1_000_000.0},
            },
            "fundamentals",
            "$1.00 million",
        ),
        (
            "news_agent",
            {
                "claim_type": "recent_news",
                "asset": "BTC/USD",
                "source": "Example",
                "payload": {"title": "Bitcoin network update"},
            },
            "news",
            "live interpretation is unavailable",
        ),
        (
            "onchain_agent",
            {
                "claim_type": "onchain_activity",
                "asset": "BTC/USD",
                "payload": {
                    "label": "Transactions",
                    "unit": "count",
                    "latest_value": 500_000.0,
                },
            },
            "onchain",
            "500,000",
        ),
    ],
)
def test_evidence_only_specialists_keep_structured_cards(
    agent: str,
    record: dict[str, object],
    expected_scope: str,
    expected_text: str,
) -> None:
    answer = evidence_first_answer(
        agent=cast("Any", agent),
        question="Review Bitcoin.",
        message="Live interpretation is unavailable.",
        limitations=(),
        evidence={"record.btc-usd": record},
    )

    assert answer.analysis_state == "evidence_only"
    assert answer.structured_analysis is not None
    assert [section.scope for section in answer.structured_analysis.sections] == [expected_scope]
    assert expected_text in answer.structured_analysis.sections[0].text


def test_risk_uses_fundamentals_result_for_per_coin_confidence() -> None:
    asset = AnalysisAsset(requested_name="AAVE", symbol="AAVE/USD", coin_id="aave")
    fundamentals = fake_fundamental_service().model_copy(update={"name": "Aave", "symbol": "aave"})
    risk = evaluate_research_risk(
        research_result=ResearchAgentResult(
            news=fake_news_service(),
            asset_results=[
                AssetResearchBundle(asset=asset, news=fake_news_service()),
            ],
            requested_capabilities=[ResearchCapability.NEWS],
        ),
        fundamentals_result=FundamentalsAgentResult(
            fundamentals=fundamentals,
            asset_results=[
                AssetResearchBundle(asset=asset, fundamentals=fundamentals),
            ],
            requested_capabilities=[ResearchCapability.FUNDAMENTALS],
        ),
        assets=[asset],
    )

    assert risk.asset_results[0].assessment.evidence_confidence == 30.0
    assert "Fundamental data is unavailable." not in risk.asset_results[0].assessment.coverage_gaps


@pytest.mark.parametrize(
    ("agent", "capabilities", "role", "expected_field"),
    [
        (
            "market_agent",
            [ResearchCapability.MARKET, ResearchCapability.RISK],
            LLMRole.MARKET,
            "per_asset_market",
        ),
        ("news_agent", [ResearchCapability.NEWS], LLMRole.RESEARCH, "per_asset_news"),
        (
            "fundamentals_agent",
            [ResearchCapability.FUNDAMENTALS, ResearchCapability.DEFI],
            LLMRole.FUNDAMENTALS,
            "per_asset_fundamentals",
        ),
    ],
)
def test_specialist_prompt_keeps_per_asset_payload_for_selected_agent(
    monkeypatch: pytest.MonkeyPatch,
    agent: str,
    capabilities: list[ResearchCapability],
    role: LLMRole,
    expected_field: str,
) -> None:
    llm = _RecordingLLM()
    inputs = _analysis_inputs_for(agent)
    claim_type, evidence_payload = {
        "market_agent": (
            "market_snapshot",
            {
                "snapshot": {"current_price": 100.0, "exchange": "kraken"},
                "ohlcv_features": {},
            },
        ),
        "news_agent": ("recent_news", {"title": "Bitcoin network update"}),
        "fundamentals_agent": (
            "project_fundamentals",
            {"name": "Bitcoin", "market_cap": 1_000_000.0},
        ),
    }[agent]
    evidence = {
        f"{capabilities[0].value}.BTC.0": {
            "evidence_id": f"{capabilities[0].value}.BTC.0",
            "claim_type": claim_type,
            "source": "verified provider",
            "asset": "BTC/USD",
            "payload": evidence_payload,
        }
    }
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(
            available_evidence=evidence,
            limitations=(),
            complete_data_digest={},
            analysis_data_digest={},
            coverage_summary=None,
        ),
    )

    answer = SpecialistAnalysisRunner(llm=llm).run(
        "Analyze the selected assets.",
        inputs,
        agent=cast("Any", agent),
        capabilities=capabilities,
    )

    assert answer.status == "complete"
    assert llm.roles == [role]
    prompt_raw_result = cast("dict[str, Any]", llm.calls[0]["analysis_briefs"])
    assert expected_field in prompt_raw_result


def test_guided_market_specialist_uses_strict_per_asset_live_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _StrictSpecialistLLM(
        {
            "verdict": "The market posture is constructive but evidence-bound.",
            "assets": [
                {
                    "symbol": "BTC/USD",
                    "market_analysis": "Momentum is constructive.",
                    "risk_analysis": "Observed risk remains bounded by coverage.",
                }
            ],
            "comparison": "",
            "limitations": [],
            "confidence": "high",
        }
    )
    evidence = {
        "market.btc-usd": {
            "claim_type": "market_snapshot",
            "asset": "BTC/USD",
            "payload": {"current_price": 100.0},
        },
        "risk.btc-usd": {
            "claim_type": "deterministic_risk_assessment",
            "asset": "BTC/USD",
            "payload": {"score": 12.0, "band": "low"},
        },
    }
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(
            available_evidence=evidence,
            limitations=(),
            complete_data_digest={},
            analysis_data_digest={},
            coverage_summary=None,
        ),
    )

    answer = SpecialistAnalysisRunner(llm=llm, live_mode=True).run(
        "Analyze BTC market risk.",
        _analysis_inputs_for("market_agent"),
        agent="market_agent",
        capabilities=[ResearchCapability.MARKET, ResearchCapability.RISK],
    )

    assert llm.calls
    assert llm.structured_calls == 1
    assert answer.agent == "market_agent"
    assert "BTC/USD:" in answer.answer
    assert answer.evidence
    assert answer.analysis_state == "live"


def test_guided_discovery_uses_live_candidate_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collected_at = datetime(2026, 1, 1, tzinfo=UTC)
    llm = _StrictSpecialistLLM(
        {
            "verdict": "The scan is led by assets with constructive relative momentum.",
            "assets": [
                {
                    "symbol": "SOL/USD",
                    "market_analysis": (
                        "Momentum and trend alignment make this candidate stand out."
                    ),
                    "risk_analysis": "The screen remains sensitive to changing conditions.",
                },
                {
                    "symbol": "ADA/USD",
                    "market_analysis": (
                        "The screen is constructive, though volatility tempers the read."
                    ),
                    "risk_analysis": "The screen remains sensitive to changing conditions.",
                },
            ],
            "comparison": "The leading candidates have different momentum profiles.",
            "limitations": [],
            "confidence": "high",
        }
    )
    candidates = [
        OpportunityCandidate(
            rank=1,
            asset="SOL",
            symbol="SOL/USD",
            current_price=150.0,
            score=82.0,
            momentum_24h=4.5,
            volatility_24h=2.0,
            trend="bullish",
            reason="Strong deterministic screen result.",
        ),
        OpportunityCandidate(
            rank=2,
            asset="ADA",
            symbol="ADA/USD",
            current_price=0.75,
            score=74.0,
            momentum_24h=2.5,
            volatility_24h=3.0,
            trend="bullish",
            reason="Constructive deterministic screen result.",
        ),
    ]
    evidence = {
        f"opportunity.{candidate.symbol}": {
            "claim_type": "market_screen",
            "asset": candidate.symbol,
            "payload": candidate.model_dump(mode="json"),
        }
        for candidate in candidates
    }
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(available_evidence=evidence),
    )

    answer = SpecialistAnalysisRunner(llm=llm, live_mode=True).run(
        "Discover Kraken market opportunities.",
        AnalysisInputs(
            opportunity_result=OpportunityScanResult(
                exchange="kraken",
                timeframe="1h",
                candidates=candidates,
                collected_at=collected_at,
                summary="SOL and ADA led the deterministic screen.",
            ),
            collection_context=CollectionContext(collected_at=collected_at),
            requested_capabilities=[ResearchCapability.DISCOVERY],
        ),
        agent="market_agent",
        capabilities=[ResearchCapability.DISCOVERY],
    )

    assert answer.analysis_state == "live"
    assert answer.structured_analysis is not None
    assert [section.asset for section in answer.structured_analysis.sections] == [
        "SOL/USD",
        "ADA/USD",
    ]
    assert all(section.scope == "market" for section in answer.structured_analysis.sections)
    assert "Momentum and trend alignment" in answer.structured_analysis.sections[0].text
    assert answer.evidence
    prompt = json.loads(cast(str, llm.calls[0]["user_prompt"]))
    assert prompt["requirements"]["assets_in_required_order"] == [
        "SOL/USD",
        "ADA/USD",
    ]
    assert len(prompt["analysis_briefs"]["discovery_candidates"]) == 2


@pytest.mark.parametrize(
    ("capabilities", "claim_type", "expected_scopes", "prompt_rule", "expected_number"),
    [
        (
            [ResearchCapability.MARKET],
            "market_snapshot",
            ["market"],
            "Interpret market posture only",
            "$100",
        ),
        (
            [ResearchCapability.RISK],
            "deterministic_risk_assessment",
            ["risk"],
            "Interpret observed risk only",
            "score 20",
        ),
        (
            [ResearchCapability.MARKET, ResearchCapability.RISK],
            "deterministic_risk_assessment",
            ["market", "risk"],
            "Interpret market posture and observed risk separately",
            "score 20",
        ),
    ],
)
def test_market_specialist_renders_only_requested_topic_scopes(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: list[ResearchCapability],
    claim_type: str,
    expected_scopes: list[str],
    prompt_rule: str,
    expected_number: str,
) -> None:
    llm = _StrictSpecialistLLM(
        {
            "verdict": "The selected evidence supports a focused current assessment.",
            "assets": [
                {
                    "symbol": "BTC/USD",
                    "market_analysis": "Momentum has a distinct qualitative posture.",
                    "risk_analysis": "Observed risk depends on the available evidence.",
                }
            ],
            "comparison": "",
            "limitations": [],
            "confidence": "high",
        }
    )
    evidence = {
        "selected.btc-usd": {
            "claim_type": claim_type,
            "asset": "BTC/USD",
            "payload": (
                {"score": 20.0}
                if "risk" in claim_type
                else {
                    "snapshot": {"current_price": 100.0, "exchange": "kraken"},
                    "ohlcv_features": {},
                }
            ),
        }
    }
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(available_evidence=evidence),
    )

    answer = SpecialistAnalysisRunner(llm=llm, live_mode=True).run(
        "Analyze the selected Bitcoin topic.",
        _analysis_inputs_for("market_agent"),
        agent="market_agent",
        capabilities=capabilities,
    )

    assert answer.analysis_state == "live"
    assert answer.structured_analysis is not None
    assert [section.scope for section in answer.structured_analysis.sections] == expected_scopes
    assert expected_number in " ".join(
        section.text for section in answer.structured_analysis.sections
    )
    assert prompt_rule in str(llm.calls[0]["system_prompt"])


def test_derivatives_analysis_includes_validated_positioning_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin")
    llm = _StrictSpecialistLLM(
        {
            "verdict": "The positioning evidence provides a bounded current view.",
            "assets": [
                {
                    "symbol": asset.symbol,
                    "market_analysis": "Positioning should be read without directional certainty.",
                    "risk_analysis": "Observed risk depends on evidence coverage.",
                }
            ],
            "comparison": "",
            "limitations": [],
            "confidence": "high",
        }
    )
    evidence = {
        "derivatives.btc-usd": {
            "claim_type": "derivatives_positioning",
            "asset": asset.symbol,
            "source": "Binance USD-M Futures",
            "payload": {
                "venue": "Binance USD-M Futures",
                "latest_funding_rate": 0.0001,
                "latest_open_interest_usd": 1_000_000_000.0,
                "open_interest_change_24h_pct": 2.5,
            },
        }
    }
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(available_evidence=evidence),
    )

    answer = SpecialistAnalysisRunner(llm=llm, live_mode=True).run(
        "Analyze BTC derivatives positioning.",
        AnalysisInputs(assets=[asset]),
        agent="market_agent",
        capabilities=[ResearchCapability.DERIVATIVES],
    )

    assert answer.structured_analysis is not None
    assert [section.scope for section in answer.structured_analysis.sections] == ["derivatives"]
    text = answer.structured_analysis.sections[0].text
    assert "0.0100%" in text
    assert "$1.00 billion" in text
    assert "+2.50% over 24 hours" in text


def test_malformed_strict_specialist_output_returns_evidence_only_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _StrictSpecialistLLM({"analysis": "Missing required output fields."})
    evidence = {
        "market.btc-usd": {
            "claim_type": "market_snapshot",
            "asset": "BTC/USD",
            "payload": {"current_price": 100.0},
        }
    }
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(
            available_evidence=evidence,
            limitations=(),
            complete_data_digest={},
            analysis_data_digest={},
            coverage_summary=None,
        ),
    )

    answer = SpecialistAnalysisRunner(llm=llm, live_mode=True).run(
        "Analyze BTC market risk.",
        _analysis_inputs_for("market_agent"),
        agent="market_agent",
        capabilities=[ResearchCapability.MARKET],
    )

    assert answer.status == "partial"
    assert "verified market read" in answer.answer.casefold()
    assert llm.structured_calls == 1
    assert answer.analysis_state == "evidence_only"


def test_unexpected_specialist_errors_reach_workflow_error_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(
            available_evidence={
                "market.btc-usd": {
                    "claim_type": "market_snapshot",
                    "asset": "BTC/USD",
                    "payload": {"current_price": 100.0},
                }
            }
        ),
    )

    with pytest.raises(RuntimeError, match="programming failure"):
        SpecialistAnalysisRunner(llm=cast("Any", _ProgrammingFailureLLM())).run(
            "Analyze BTC market risk.",
            _analysis_inputs_for("market_agent"),
            agent="market_agent",
            capabilities=[ResearchCapability.MARKET],
        )


@pytest.mark.parametrize("asset_count", [1, 2, 3, 4])
def test_strict_market_output_renders_every_asset_in_request_order(
    monkeypatch: pytest.MonkeyPatch,
    asset_count: int,
) -> None:
    symbols = ["BTC", "ETH", "SOL", "ADA"][:asset_count]
    assets = [
        AnalysisAsset(requested_name=symbol, symbol=f"{symbol}/USD", coin_id=symbol.casefold())
        for symbol in symbols
    ]
    llm = _StrictSpecialistLLM(
        {
            "verdict": "The selected assets show different market postures.",
            "assets": [
                {
                    "symbol": asset.symbol,
                    "market_analysis": "Momentum has a distinct qualitative posture.",
                    "risk_analysis": "Observed risk depends on the available evidence.",
                }
                for asset in reversed(assets)
            ],
            "comparison": "Relative strength and observed risk differ across the group.",
            "limitations": [],
            "confidence": "high",
        }
    )
    evidence = {
        f"risk.{asset.evidence_key}": {
            "claim_type": "deterministic_risk_assessment",
            "asset": asset.symbol,
            "source": "ChainScope deterministic risk model",
            "payload": {
                "score": float(index * 10),
                "band": "low",
                "evidence_confidence": 70.0,
                "factors": [],
                "coverage_gaps": [],
            },
        }
        for index, asset in enumerate(assets)
    }
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(available_evidence=evidence),
    )

    answer = SpecialistAnalysisRunner(llm=llm, live_mode=True).run(
        "Analyze the selected market and risk evidence.",
        AnalysisInputs(assets=assets),
        agent="market_agent",
        capabilities=[ResearchCapability.MARKET, ResearchCapability.RISK],
    )

    positions = [answer.answer.index(f"{asset.symbol}:") for asset in assets]
    assert positions == sorted(positions)
    assert len(answer.answer) <= 2_400
    assert all(paragraph.endswith((".", "!", "?")) for paragraph in answer.answer.split("\n\n"))
    assert answer.analysis_state == "live"
    assert answer.status == "complete"
    assert answer.structured_analysis is not None
    assert [section.asset for section in answer.structured_analysis.sections] == [
        asset.symbol for asset in assets for _scope in range(2)
    ]
    assert [section.scope for section in answer.structured_analysis.sections] == [
        scope for _asset in assets for scope in ("market", "risk")
    ]
    assert bool(answer.structured_analysis.comparison) is (asset_count > 1)
    call = llm.calls[0]
    assert estimate_tokens(str(call["system_prompt"]), str(call["user_prompt"])) <= 1_700


@pytest.mark.parametrize("asset_count", [1, 2, 3, 4])
@pytest.mark.parametrize("agent", ["fundamentals_agent", "news_agent"])
def test_compact_specialists_cover_assets_once_in_request_order(
    monkeypatch: pytest.MonkeyPatch,
    asset_count: int,
    agent: str,
) -> None:
    symbols = ["BTC", "ETH", "SOL", "ADA"][:asset_count]
    assets = [
        AnalysisAsset(requested_name=symbol, symbol=f"{symbol}/USD", coin_id=symbol.casefold())
        for symbol in symbols
    ]
    common = {
        "assets": [
            {
                "symbol": asset.symbol,
                "analysis": "The evidence gives this asset a distinct but qualified posture.",
            }
            for asset in reversed(assets)
        ],
        "comparison": "The selected assets differ in evidence strength and context.",
        "limitations": [],
        "confidence": "high",
    }
    payload: dict[str, object] = (
        {
            "verdict": "Fundamental evidence differentiates the selected assets.",
            "defi_assets": [],
            **common,
        }
        if agent == "fundamentals_agent"
        else {
            "verdict": "Recent coverage gives the selected assets different contexts.",
            **common,
        }
    )
    claim_type = "project_fundamentals" if agent == "fundamentals_agent" else "recent_news"
    evidence = {
        f"{claim_type}.{asset.evidence_key}": {
            "claim_type": claim_type,
            "asset": asset.symbol,
            "source": "Verified provider",
            "payload": (
                {"name": asset.requested_name, "market_cap": 1_000_000.0}
                if agent == "fundamentals_agent"
                else {"title": f"{asset.requested_name} network update"}
            ),
        }
        for asset in assets
    }
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(available_evidence=evidence),
    )

    llm = _StrictSpecialistLLM(payload)
    answer = SpecialistAnalysisRunner(
        llm=llm,
        live_mode=True,
    ).run(
        "Analyze the selected assets.",
        AnalysisInputs(assets=assets),
        agent=cast("Any", agent),
        capabilities=(
            [ResearchCapability.FUNDAMENTALS]
            if agent == "fundamentals_agent"
            else [ResearchCapability.NEWS]
        ),
    )

    positions = [answer.answer.index(f"{asset.symbol}:") for asset in assets]
    assert positions == sorted(positions)
    assert len(answer.answer) <= 2_400
    assert all(answer.answer.count(f"{asset.symbol}:") == 1 for asset in assets)
    assert answer.analysis_state == "live"
    assert answer.structured_analysis is not None
    assert [section.asset for section in answer.structured_analysis.sections] == [
        asset.symbol for asset in assets
    ]
    expected_scope = "fundamentals" if agent == "fundamentals_agent" else "news"
    assert {section.scope for section in answer.structured_analysis.sections} == {expected_scope}
    call = llm.calls[0]
    assert estimate_tokens(str(call["system_prompt"]), str(call["user_prompt"])) <= 1_700


def test_weak_live_verdict_is_replaced_with_a_complete_agent_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin")
    llm = _StrictSpecialistLLM(
        {
            "verdict": "informative",
            "assets": [
                {
                    "symbol": asset.symbol,
                    "analysis": "Coverage highlights a distinct current narrative for this asset.",
                }
            ],
            "comparison": "",
            "limitations": [],
            "confidence": "high",
        }
    )
    evidence = {
        "recent_news.btc-usd": {
            "claim_type": "recent_news",
            "asset": asset.symbol,
            "source": "Verified provider",
            "payload": {"title": "Bitcoin network update"},
        }
    }
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(available_evidence=evidence),
    )

    answer = SpecialistAnalysisRunner(llm=llm, live_mode=True).run(
        "Analyze recent Bitcoin news.",
        AnalysisInputs(assets=[asset]),
        agent="news_agent",
        capabilities=[ResearchCapability.NEWS],
    )

    assert answer.structured_analysis is not None
    assert answer.structured_analysis.verdict != "informative"
    assert len(answer.structured_analysis.verdict.split()) >= 5
    assert answer.structured_analysis.verdict.endswith(".")


def test_weak_onchain_verdict_stays_on_live_composition_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin")
    llm = _StrictSpecialistLLM(
        {
            "verdict": "informative",
            "assets": [
                {
                    "symbol": asset.symbol,
                    "analysis": "Network usage remains active within the available coverage.",
                }
            ],
            "comparison": "",
            "limitations": [],
            "confidence": "high",
        }
    )
    evidence = {
        "onchain.btc-usd.txcnt": {
            "claim_type": "onchain_activity",
            "asset": asset.symbol,
            "source": "Coin Metrics Community",
            "payload": {
                "metric": "TxCnt",
                "label": "Transactions",
                "unit": "count",
                "latest_value": 500_000.0,
            },
        }
    }
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(available_evidence=evidence),
    )

    answer = SpecialistAnalysisRunner(llm=llm, live_mode=True).run(
        "Analyze Bitcoin on-chain activity.",
        AnalysisInputs(assets=[asset]),
        agent="onchain_agent",
        capabilities=[ResearchCapability.ONCHAIN],
    )

    assert answer.analysis_state == "live"
    assert answer.status == "complete"
    assert answer.evidence
    assert answer.structured_analysis is not None
    assert answer.structured_analysis.verdict.startswith("The verified network metrics")
    assert [section.scope for section in answer.structured_analysis.sections] == ["onchain"]
    assert "500,000" in answer.structured_analysis.sections[0].text


@pytest.mark.parametrize(
    ("capabilities", "expected_scopes"),
    [
        (
            [ResearchCapability.FUNDAMENTALS, ResearchCapability.DEFI],
            ["fundamentals", "defi"],
        ),
        ([ResearchCapability.DEFI], ["defi"]),
    ],
)
def test_live_defi_section_stays_separate_for_an_eligible_asset(
    monkeypatch: pytest.MonkeyPatch,
    capabilities: list[ResearchCapability],
    expected_scopes: list[str],
) -> None:
    asset = AnalysisAsset(requested_name="AAVE", symbol="AAVE/USD", coin_id="aave")
    llm = _StrictSpecialistLLM(
        {
            "verdict": "Fundamental and protocol evidence provide complementary context.",
            "assets": [
                {
                    "symbol": asset.symbol,
                    "analysis": "Project scale should be assessed alongside its market position.",
                }
            ],
            "defi_assets": [
                {
                    "symbol": asset.symbol,
                    "analysis": "Protocol activity adds a distinct usage signal for this asset.",
                }
            ],
            "comparison": "",
            "limitations": [],
            "confidence": "high",
        }
    )
    evidence = {
        "fundamentals.aave-usd": {
            "claim_type": "project_fundamentals",
            "asset": asset.symbol,
            "source": "Verified provider",
            "payload": {"name": "Aave", "market_cap": 1_000_000.0},
        },
        "defi.aave-usd": {
            "claim_type": "defi_protocol_metrics",
            "asset": asset.symbol,
            "source": "DefiLlama",
            "payload": {"protocol": "Aave", "tvl_usd": 500_000.0},
        },
    }
    monkeypatch.setattr(
        "crypto_research.orchestration.evidence.build_specialist_evidence",
        lambda *_args, **_kwargs: EvidenceSpec(available_evidence=evidence),
    )

    answer = SpecialistAnalysisRunner(llm=llm, live_mode=True).run(
        "Analyze Aave fundamentals and DeFi usage.",
        AnalysisInputs(assets=[asset]),
        agent="fundamentals_agent",
        capabilities=capabilities,
    )

    assert answer.structured_analysis is not None
    assert [section.scope for section in answer.structured_analysis.sections] == expected_scopes
    sections = {section.scope: section.text for section in answer.structured_analysis.sections}
    if "fundamentals" in sections:
        assert "$1.00 million" in sections["fundamentals"]
    assert "$500,000" in sections["defi"]


def _analysis_inputs_for(agent: str) -> AnalysisInputs:
    market = fake_market_service()
    assets = [AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin")]
    if agent == "market_agent":
        return AnalysisInputs(
            assets=assets,
            market_result=MarketAgentResult(
                market=market,
                technical=TechnicalSnapshot(trend="bullish", rsi=58.0),
            ),
        )
    if agent == "news_agent":
        return AnalysisInputs(assets=assets, research_result=_news_result(market.collected_at))
    return AnalysisInputs(
        assets=assets,
        fundamentals_result=FundamentalsAgentResult(
            asset_results=[
                AssetResearchBundle(
                    asset=assets[0],
                    fundamentals=fake_fundamental_service(),
                    defi=DefiEvidence(
                        protocol="Aave",
                        tvl_usd=1_000_000.0,
                        status="available",
                        collected_at=market.collected_at,
                    ),
                )
            ],
            requested_capabilities=[ResearchCapability.FUNDAMENTALS, ResearchCapability.DEFI],
        ),
    )


def _news_result(timestamp: datetime) -> ResearchAgentResult:
    btc = AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin")
    eth = AnalysisAsset(requested_name="ETH", symbol="ETH/USD", coin_id="ethereum")
    return ResearchAgentResult(
        news=fake_news_service(),
        asset_results=[
            AssetResearchBundle(
                asset=btc,
                news=NewsEvidence(
                    items=[
                        NewsItem(
                            publisher="Example",
                            title="BTC headline 1",
                            excerpt="BTC relevant news.",
                            published_at=timestamp,
                        )
                    ],
                    query="BTC",
                    collected_at=timestamp,
                ),
            ),
            AssetResearchBundle(
                asset=eth,
                news=NewsEvidence(
                    items=[
                        NewsItem(
                            publisher="Example",
                            title="ETH headline 1",
                            excerpt="ETH relevant news.",
                            published_at=timestamp,
                        )
                    ],
                    query="ETH",
                    collected_at=timestamp,
                ),
            ),
        ],
        requested_capabilities=[ResearchCapability.NEWS],
    )
