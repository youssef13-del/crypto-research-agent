from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import cast

from tests.support.fakes import fake_market_service

from crypto_research.agents.guardrails import compile_answer_requirements
from crypto_research.domain.evidence import (
    DerivativesEvidence,
    FundingRateObservation,
    OpenInterestObservation,
)
from crypto_research.domain.research import (
    AnalysisAsset,
    AnalysisInputs,
    CollectionContext,
    MarketAgentResult,
    ResearchCapability,
    TechnicalSnapshot,
)
from crypto_research.llm.prompt_packing import build_specialist_analysis_payload
from crypto_research.orchestration.evidence import build_specialist_evidence


def test_derivatives_records_are_packed_as_an_independent_capability() -> None:
    cutoff = datetime(2026, 8, 14, 12, tzinfo=UTC)
    market = fake_market_service().model_copy(update={"collected_at": cutoff})
    derivatives = DerivativesEvidence(
        asset="BTC",
        contract_symbol="BTCUSDT",
        status="complete",
        funding_history=[
            FundingRateObservation(observed_at=cutoff - timedelta(hours=8), rate=0.0001)
        ],
        open_interest_history=[
            OpenInterestObservation(observed_at=cutoff - timedelta(hours=1), value_usd=1_250_000)
        ],
        latest_funding_rate=0.0001,
        average_funding_rate_24h=0.0001,
        latest_open_interest_usd=1_250_000,
        collected_at=cutoff,
    )
    inputs = AnalysisInputs(
        assets=[AnalysisAsset(requested_name="BTC", symbol="BTC/USD", coin_id="bitcoin")],
        requested_capabilities=[ResearchCapability.DERIVATIVES],
        collection_context=CollectionContext(collected_at=cutoff),
        market_result=MarketAgentResult(
            market=market,
            technical=TechnicalSnapshot(trend="neutral"),
            derivatives=derivatives,
        ),
    )

    evidence = build_specialist_evidence(
        inputs,
        agent="market_agent",
        capabilities=[ResearchCapability.DERIVATIVES],
    )
    record = evidence.available_evidence["derivatives.BTC"]
    assert isinstance(record, Mapping)
    assert record["claim_type"] == "derivatives_positioning"
    payload = record["payload"]
    assert isinstance(payload, Mapping)
    assert payload["research_capability"] == "derivatives"
    assert evidence.coverage_summary is not None
    assert evidence.coverage_summary.entries[0].capability is ResearchCapability.DERIVATIVES

    bundle = build_specialist_analysis_payload(
        {
            "specialist": "market_agent",
            "collection_status": {
                "requested_assets": [{"symbol": "BTC/USD"}],
                "requested_capabilities": ["derivatives"],
            },
        },
        cast("Mapping[object, object]", evidence.available_evidence),
        evidence.complete_data_digest,
    )
    assert "derivatives.BTC" in bundle.available_evidence


def test_derivatives_requirements_do_not_treat_liquidations_as_supplied_evidence() -> None:
    cutoff = datetime(2026, 8, 14, 12, tzinfo=UTC)
    record = {
        "claim_type": "derivatives_positioning",
        "asset": "BTC",
        "payload": {
            "research_capability": "derivatives",
            "latest_funding_rate": 0.0001,
            "latest_open_interest_usd": 1_250_000,
        },
    }
    requirements = compile_answer_requirements(
        question="What do BTC funding rate, open interest, and liquidations show now?",
        raw_result={
            "collection_status": {
                "requested_assets": [{"symbol": "BTC/USD"}],
                "requested_capabilities": ["derivatives"],
                "collected_at": cutoff.isoformat(),
            }
        },
        coverage_requirements=("derivatives",),
        evidence={"derivatives.BTC": record},
    )

    assert requirements.available_metrics == ("funding_rate", "open_interest")
    assert requirements.unavailable_metrics == ("liquidations",)
