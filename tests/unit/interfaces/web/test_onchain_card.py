from datetime import UTC, datetime

import pytest

from crypto_research.domain.evidence import (
    OnChainEvidence,
    OnChainMetricSeries,
    OnChainObservation,
)
from crypto_research.domain.research import (
    AnalysisAsset,
    AnalysisRequest,
    AssetResearchBundle,
    CollectionContext,
    OnChainAgentResult,
    ResearchReport,
)
from crypto_research.interfaces.web.components.research import render_structured_agent_analysis
from crypto_research.interfaces.web.presentation import onchain_data_cards
from crypto_research.interfaces.web.runtime import (
    AgentAnalysisSectionPresentation,
    AgentPanelPresentation,
    StructuredAgentAnalysisPresentation,
)


def test_onchain_card_formats_metrics_and_trends() -> None:
    collected = datetime(2026, 8, 11, 12, tzinfo=UTC)
    asset = AnalysisAsset(requested_name="Bitcoin", symbol="BTC/USD", coin_id="bitcoin")
    point = OnChainObservation(observed_at=collected, value=1_250_000)
    metrics = [
        OnChainMetricSeries(
            metric=metric,
            label=label,
            unit=unit,
            observations=[point],
            latest_value=point.value,
            latest_at=collected,
            seven_day_change_pct=8.5,
        )
        for metric, label, unit in (
            ("AdrActCnt", "Active addresses", "count"),
            ("TxCnt", "Transactions", "count"),
            ("FeeTotUSD", "Network fees", "usd"),
        )
    ]
    evidence = OnChainEvidence(
        asset=asset.symbol,
        provider_asset="btc",
        status="complete",
        metrics=metrics,
        collected_at=collected,
    )
    report = ResearchReport(
        request=AnalysisRequest(user_intent="Bitcoin on-chain activity"),
        onchain_result=OnChainAgentResult(
            asset_results=[AssetResearchBundle(asset=asset, onchain=evidence)]
        ),
        collection_context=CollectionContext(collected_at=collected),
    )

    cards = onchain_data_cards(report, cutoff=collected)

    assert len(cards) == 1
    assert cards[0].title == "BTC/USD on-chain activity"
    assert cards[0].status == "complete"
    assert cards[0].facts[0] == ("Active addresses", "1.25 million (+8.5% vs prior 7d)")
    assert cards[0].facts[2] == ("Network fees", "$1.25 million (+8.5% vs prior 7d)")


def test_unavailable_onchain_card_does_not_repeat_provider_warning() -> None:
    collected = datetime(2026, 8, 11, 12, tzinfo=UTC)
    asset = AnalysisAsset(requested_name="Solana", symbol="SOL/USD", coin_id="solana")
    evidence = OnChainEvidence(
        asset=asset.symbol,
        provider_asset="sol",
        status="unavailable",
        collected_at=collected,
        warnings=["Coin Metrics Community does not provide daily network metrics."],
    )
    report = ResearchReport(
        request=AnalysisRequest(user_intent="Solana on-chain activity"),
        onchain_result=OnChainAgentResult(
            asset_results=[AssetResearchBundle(asset=asset, onchain=evidence)]
        ),
        collection_context=CollectionContext(collected_at=collected),
    )

    card = onchain_data_cards(report, cutoff=collected)[0]

    assert card.facts == (("Data status", "Unavailable"),)
    assert card.limitation is None


def test_structured_onchain_analysis_renders_network_activity_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(
        "crypto_research.interfaces.web.components.research.st.markdown",
        lambda value, **_kwargs: rendered.append(value),
    )
    panel = AgentPanelPresentation(
        agent="onchain_agent",
        title="On-Chain Activity Agent",
        status="complete",
        analysis_state="evidence_only",
    )
    analysis = StructuredAgentAnalysisPresentation(
        verdict="Verified network activity is available.",
        sections=(
            AgentAnalysisSectionPresentation(
                asset="BTC/USD",
                scope="onchain",
                text="Network usage remains active within the available coverage.",
            ),
        ),
    )

    render_structured_agent_analysis(panel, analysis)

    assert any("Network activity" in value for value in rendered)
    assert any("Live evidence answer" in value for value in rendered)
