from datetime import UTC, datetime, timedelta
from io import BytesIO

import pytest
from pypdf import PdfReader
from tests.support.fakes import fake_market_service

from crypto_research.domain.history import ResearchRunSummary, StoredResearchRun
from crypto_research.domain.research import (
    AgentAnalysisSection,
    AgentAnswer,
    AgentExecutionStatus,
    AnalysisAsset,
    AnalysisRequest,
    CollectionContext,
    EvidenceClaim,
    MarketAgentResult,
    NewsEvidence,
    NewsItem,
    ResearchAgentResult,
    ResearchCapability,
    ResearchReport,
    StructuredAgentAnalysis,
    TechnicalSnapshot,
)
from crypto_research.interfaces.web.pdf_report import render_research_pdf, research_pdf_filename


def _stored_run(
    symbol: str,
    *,
    completed_at: datetime,
    run_id: str,
) -> StoredResearchRun:
    market = fake_market_service().model_copy(update={"symbol": symbol})
    asset = symbol.split("/", maxsplit=1)[0]
    statement = f"{symbol} closed at ${market.current_price:,.2f}."
    request = AnalysisRequest(
        user_intent=f"Research {asset}",
        assets=[AnalysisAsset(requested_name=asset, symbol=symbol)],
    )
    news_time = market.collected_at - timedelta(hours=1)
    report = ResearchReport(
        request=request,
        collection_context=CollectionContext(collected_at=market.collected_at),
        market_result=MarketAgentResult(
            market=market,
            technical=TechnicalSnapshot(
                trend="bullish",
                rsi=58.2,
                support=market.current_price * 0.95,
                resistance=market.current_price * 1.05,
            ),
        ),
        research_result=ResearchAgentResult(
            news=NewsEvidence(
                items=[
                    NewsItem(
                        publisher="Example Research",
                        title=f"{asset} network and market update",
                        excerpt="A current evidence-backed market update.",
                        url="https://example.com/research/update",
                        published_at=news_time,
                    )
                ],
                query=asset,
                collected_at=market.collected_at,
            )
        ),
        agent_statuses=[
            AgentExecutionStatus(
                agent="market_agent",
                status="complete",
                capabilities=[ResearchCapability.MARKET],
            )
        ],
        agent_answers=[
            AgentAnswer(
                agent="market_agent",
                answer=statement,
                structured_analysis=StructuredAgentAnalysis(
                    verdict=f"{asset} has a bullish completed-candle trend.",
                    sections=[
                        AgentAnalysisSection(
                            asset=symbol,
                            scope="market",
                            text=f"Price is ${market.current_price:,.2f} with RSI at 58.2.",
                        )
                    ],
                ),
                evidence=[
                    EvidenceClaim(
                        statement=statement,
                        evidence_ids=[f"market.{asset}.close"],
                        confidence=0.92,
                    )
                ],
                confidence=0.92,
            )
        ],
    )
    return StoredResearchRun(
        summary=ResearchRunSummary(
            id=run_id,
            created_at=completed_at - timedelta(minutes=2),
            completed_at=completed_at,
            state="complete",
            question=request.user_intent,
            assets=(symbol,),
            capabilities=("market", "news"),
            exchange="kraken",
            timeframe="1h",
            pinned=False,
            evidence_count=4,
        ),
        report=report,
    )


def _pdf_text(data: bytes) -> tuple[PdfReader, str]:
    reader = PdfReader(BytesIO(data))
    return reader, "\n".join(page.extract_text() or "" for page in reader.pages)


def test_single_pdf_contains_professional_sections_chart_sources_and_safe_profile() -> None:
    completed = datetime(2026, 8, 15, 12, tzinfo=UTC)
    stored = _stored_run(
        "BTC/USD",
        completed_at=completed,
        run_id="00000000-0000-0000-0000-000000000001",
    )

    data = render_research_pdf(
        (stored,),
        prepared_for="Youssef Ω <script>alert(1)</script> password=private-value",
        generated_at=datetime(2026, 8, 16, 9, tzinfo=UTC),
    )
    reader, text = _pdf_text(data)

    assert data.startswith(b"%PDF")
    assert reader.metadata is not None
    assert reader.metadata.title == "BTC/USD Research Report"
    assert len(reader.pages) >= 2
    assert "Executive analysis" in text
    assert "Validated numerical snapshot" in text
    assert "Stored quantitative trends" in text
    assert "Specialist analysis" in text
    assert "Evidence and source appendix" in text
    assert "BTC network and market update" in text
    assert "Youssef Ω" in text
    assert "private-value" not in text
    assert "script" not in text


def test_combined_pdf_sorts_newest_first_and_adds_bundle_index() -> None:
    older = _stored_run(
        "BTC/USD",
        completed_at=datetime(2026, 8, 14, 12, tzinfo=UTC),
        run_id="00000000-0000-0000-0000-000000000001",
    )
    newer = _stored_run(
        "ETH/USD",
        completed_at=datetime(2026, 8, 15, 12, tzinfo=UTC),
        run_id="00000000-0000-0000-0000-000000000002",
    )

    data = render_research_pdf((older, newer), generated_at=datetime(2026, 8, 16, tzinfo=UTC))
    reader, text = _pdf_text(data)

    assert len(reader.pages) >= 5
    assert reader.metadata is not None
    assert reader.metadata.title == "ChainScope Research Bundle (2 reports)"
    assert "Bundle index" in text
    assert text.index("ETH/USD Research Report") < text.index("BTC/USD Research Report")
    assert research_pdf_filename(
        (older, newer), generated_at=datetime(2026, 8, 16, tzinfo=UTC)
    ) == ("chainscope-research-bundle-20260816.pdf")


def test_pdf_export_rejects_empty_duplicate_oversized_and_naive_timestamp_inputs() -> None:
    stored = _stored_run(
        "BTC/USD",
        completed_at=datetime(2026, 8, 15, 12, tzinfo=UTC),
        run_id="00000000-0000-0000-0000-000000000001",
    )

    with pytest.raises(ValueError, match="at least one"):
        render_research_pdf(())
    with pytest.raises(ValueError, match="duplicate"):
        render_research_pdf((stored, stored))
    with pytest.raises(ValueError, match="limited to 20"):
        render_research_pdf(tuple(stored for _ in range(21)))
    with pytest.raises(ValueError, match="timezone-aware"):
        render_research_pdf((stored,), generated_at=datetime(2026, 8, 16))


def test_single_pdf_filename_is_sanitized() -> None:
    stored = _stored_run(
        "BTC/USD",
        completed_at=datetime(2026, 8, 15, 12, tzinfo=UTC),
        run_id="00000000-0000-0000-0000-000000000001",
    )

    assert (
        research_pdf_filename((stored,), generated_at=datetime(2026, 8, 16, tzinfo=UTC))
        == "chainscope-btc-usd-20260816.pdf"
    )
