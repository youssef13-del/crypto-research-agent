from datetime import UTC, datetime

from crypto_research.agents.onchain.onchain_agent import OnChainAgent
from crypto_research.agents.shared_analysis import SpecialistAnalysisRunner
from crypto_research.domain.evidence import (
    OnChainEvidence,
    OnChainMetricSeries,
    OnChainObservation,
)
from crypto_research.domain.research import (
    AnalysisAsset,
    AnalysisInputs,
    AnalysisRequest,
    AssetResearchBundle,
    CollectionContext,
    OnChainAgentResult,
    ResearchCapability,
)
from crypto_research.tools.types import OnChainServices


def test_onchain_agent_preserves_asset_order_and_summary() -> None:
    collected = datetime(2026, 8, 11, 12, tzinfo=UTC)

    def fetcher(*, asset: AnalysisAsset, **_: object) -> OnChainEvidence:
        symbol = asset.symbol
        point = OnChainObservation(observed_at=collected, value=100)
        return OnChainEvidence(
            asset=symbol,
            provider_asset=symbol.split("/")[0].casefold(),
            status="partial",
            metrics=[
                OnChainMetricSeries(
                    metric="TxCnt",
                    label="Transactions",
                    unit="count",
                    observations=[point],
                    latest_value=100,
                    latest_at=collected,
                    seven_day_change_pct=12.5,
                )
            ],
            collected_at=collected,
        )

    request = AnalysisRequest(
        user_intent="Compare BTC and ETH on-chain activity",
        comparison_symbols=["BTC/USD", "ETH/USD"],
    )
    result = OnChainAgent(
        services=OnChainServices(collect=fetcher),
        base_url="https://coinmetrics.test",
    ).run(request, collection_context=CollectionContext(collected_at=collected))

    assert [bundle.asset.symbol for bundle in result.asset_results] == ["BTC/USD", "ETH/USD"]
    assert result.capabilities == ["onchain"]
    assert "Transactions +12.5%" in (result.summary or "")


def test_onchain_analysis_falls_back_to_isolated_evidence() -> None:
    collected = datetime(2026, 8, 11, 12, tzinfo=UTC)
    asset = AnalysisAsset(requested_name="Bitcoin", symbol="BTC/USD", coin_id="bitcoin")
    point = OnChainObservation(observed_at=collected, value=100)
    evidence = OnChainEvidence(
        asset=asset.symbol,
        provider_asset="btc",
        status="partial",
        metrics=[
            OnChainMetricSeries(
                metric="TxCnt",
                label="Transactions",
                unit="count",
                observations=[point],
                latest_value=100,
                latest_at=collected,
            )
        ],
        collected_at=collected,
    )
    inputs = AnalysisInputs(
        assets=[asset],
        requested_capabilities=[ResearchCapability.ONCHAIN],
        onchain_result=OnChainAgentResult(
            asset_results=[AssetResearchBundle(asset=asset, onchain=evidence)],
            requested_capabilities=[ResearchCapability.ONCHAIN],
            capabilities=[ResearchCapability.ONCHAIN],
        ),
        collection_context=CollectionContext(collected_at=collected),
    )

    answer = SpecialistAnalysisRunner().run(
        "How active is Bitcoin on-chain?",
        inputs,
        agent="onchain_agent",
        capabilities=[ResearchCapability.ONCHAIN],
    )

    assert answer.agent == "onchain_agent"
    assert answer.analysis_state == "evidence_only"
    assert answer.evidence
    assert all(
        evidence_id.startswith("onchain.")
        for claim in answer.evidence
        for evidence_id in claim.evidence_ids
    )
    assert "holder concentration" not in answer.answer.casefold()
