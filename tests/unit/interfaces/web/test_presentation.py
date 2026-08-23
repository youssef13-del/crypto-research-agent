from datetime import UTC, datetime, timedelta

import pytest
from tests.support.fakes import fake_market_service

from crypto_research.domain.evidence import (
    DerivativesEvidence,
    FundingRateObservation,
    OpenInterestObservation,
)
from crypto_research.domain.research import (
    AgentAnalysisSection,
    AgentAnswer,
    AgentExecutionStatus,
    AnalysisAsset,
    AnalysisRequest,
    AnswerState,
    AssetResearchBundle,
    CollectionContext,
    DefiEvidence,
    EvidenceClaim,
    FundamentalEvidence,
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
    ResearchReport,
    RiskAssessment,
    RiskResult,
    StructuredAgentAnalysis,
    TechnicalSnapshot,
)
from crypto_research.interfaces.web.components.charts import build_asset_presentations
from crypto_research.interfaces.web.components.research import (
    _render_derivatives_table,
    _render_discovery_results,
    render_analysis_text,
)
from crypto_research.interfaces.web.presentation import (
    build_agent_answer_presentation,
    build_research_presentation,
    collect_warnings,
)


def test_collect_warnings_normalizes_current_warning_layers_once() -> None:
    market = fake_market_service()
    result = ResearchReport(
        status="partial",
        request=AnalysisRequest(user_intent="Review Bitcoin"),
        warnings=[" shared warning ", "result warning"],
        market_comparison_result=MarketComparisonResult(
            assets=[
                MarketComparisonAsset(market=market, technical=TechnicalSnapshot(trend="bullish")),
                MarketComparisonAsset(
                    market=market.model_copy(update={"symbol": "ETH/USD"}),
                    technical=TechnicalSnapshot(trend="neutral"),
                ),
            ],
            warnings=["comparison warning"],
        ),
        research_result=ResearchAgentResult(
            news=NewsEvidence(
                items=[],
                query="Bitcoin",
                collected_at=datetime(2026, 1, 1, tzinfo=UTC),
                warnings=["news warning"],
            ),
            fundamentals=FundamentalEvidence(warnings=["fundamentals warning"]),
        ),
    )

    assert collect_warnings(result) == [
        "shared warning",
        "result warning",
        "comparison warning",
        "news warning",
        "fundamentals warning",
    ]


def test_derivatives_presentation_uses_binance_venue_and_separate_capability_card() -> None:
    market = fake_market_service()
    cutoff = market.collected_at
    derivatives = DerivativesEvidence(
        asset="BTC",
        contract_symbol="BTCUSDT",
        status="complete",
        funding_history=[
            FundingRateObservation(observed_at=cutoff - timedelta(hours=8), rate=0.0001)
        ],
        open_interest_history=[
            OpenInterestObservation(observed_at=cutoff - timedelta(hours=25), value_usd=1_000_000),
            OpenInterestObservation(observed_at=cutoff - timedelta(hours=1), value_usd=1_250_000),
        ],
        latest_funding_rate=0.0001,
        average_funding_rate_24h=0.0001,
        latest_open_interest_usd=1_250_000,
        open_interest_change_24h_pct=25,
        collected_at=cutoff,
        warnings=["Funding is a positioning signal, not directional certainty."],
    )
    result = ResearchReport(
        request=AnalysisRequest(
            user_intent="Review BTC derivatives",
            assets=[AnalysisAsset(requested_name="BTC", symbol="BTC/USD")],
            exchange="coinbase",
        ),
        collection_context=CollectionContext(collected_at=cutoff),
        market_result=MarketAgentResult(
            market=market,
            technical=TechnicalSnapshot(trend="neutral"),
            derivatives=derivatives,
        ),
        agent_statuses=[
            AgentExecutionStatus(
                agent="market_agent",
                status="complete",
                capabilities=[ResearchCapability.DERIVATIVES],
            )
        ],
    )

    presentation = build_research_presentation(result)
    panel = presentation.agent_panels[0]
    card = next(
        item for item in panel.data if item.capability == ResearchCapability.DERIVATIVES.value
    )

    assert dict(card.facts)["Venue"] == "Binance USD-M Futures"
    assert dict(card.facts)["Latest funding"] == "0.0100%"
    assert dict(card.facts)["24h OI change"] == "+25.00%"
    assert any(source.kind == "Derivatives" for source in presentation.sources)
    assert any(source.publisher == "Binance USD-M Futures" for source in presentation.sources)
    assert "directional certainty" in " ".join(presentation.warnings)


def test_derivatives_resources_render_as_a_comparison_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[dict[str, object]]] = []
    monkeypatch.setattr(
        "crypto_research.interfaces.web.components.research.render_data_table",
        lambda rows: captured.append(rows),
    )
    from crypto_research.interfaces.web.runtime import CapabilityDataPresentation

    _render_derivatives_table(
        (
            CapabilityDataPresentation(
                agent="market_agent",
                capability="derivatives",
                title="BTC derivatives positioning",
                asset="BTC",
                facts=(
                    ("Contract", "BTCUSDT"),
                    ("Latest funding", "0.0100%"),
                    ("Open interest", "$1.25M"),
                    ("24h OI change", "+25.00%"),
                    ("Venue", "Binance USD-M Futures"),
                    ("Freshness", "1h ago"),
                ),
                limitation="Funding alone does not establish direction.",
            ),
        )
    )

    assert captured[0][0]["Venue"] == "Binance USD-M Futures"
    assert captured[0][0]["Limitations"] == "Funding alone does not establish direction."


def test_analysis_text_renders_as_plain_safe_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def capture_markdown(value: str, **kwargs: object) -> None:
        calls.append((value, kwargs))

    monkeypatch.setattr(
        "crypto_research.interfaces.web.components.research.st.markdown", capture_markdown
    )

    render_analysis_text("RSI and <script>alert(1)</script>")

    assert calls == [(r"RSI and \<script\>alert\(1\)\</script\>", {})]
    assert "cs-technical-term" not in calls[0][0]


def test_render_analysis_text_does_not_render_brittle_suggestion_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def capture_markdown(value: str, **kwargs: object) -> None:
        calls.append((value, kwargs))

    monkeypatch.setattr(
        "crypto_research.interfaces.web.components.research.st.markdown", capture_markdown
    )

    render_analysis_text("Compare Bitcoin and Ethereum and explain the main risks.")

    values = [value for value, _ in calls]
    assert "Suggested next steps" not in values
    assert all(not value.startswith("**Quick take**") for value in values)


def test_render_analysis_text_keeps_long_answers_plain_without_quick_take(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def capture_markdown(value: str, **kwargs: object) -> None:
        calls.append((value, kwargs))

    monkeypatch.setattr(
        "crypto_research.interfaces.web.components.research.st.markdown", capture_markdown
    )

    render_analysis_text(
        "Bitcoin remains relatively resilient as long as key support levels hold and "
        "volatility stays contained. The broader market remains sensitive to macro "
        "catalysts and funding conditions."
    )

    assert all(not value.startswith("**Quick take**") for value, _ in calls)
    assert any("relatively resilient" in value for value, _ in calls)


def test_build_asset_presentations_combines_market_comparison_assets() -> None:
    market = fake_market_service()
    result = ResearchReport(
        request=AnalysisRequest(user_intent="Compare BTC and ETH"),
        market_comparison_result=MarketComparisonResult(
            assets=[
                MarketComparisonAsset(market=market, technical=TechnicalSnapshot(trend="bullish")),
                MarketComparisonAsset(
                    market=market.model_copy(update={"symbol": "ETH/USD"}),
                    technical=TechnicalSnapshot(trend="neutral"),
                ),
            ]
        ),
    )

    assets = build_asset_presentations(result)

    assert [asset.symbol for asset in assets] == ["BTC/USD", "ETH/USD"]
    assert assets[0].current_price == market.current_price


def test_multi_asset_research_presentation_labels_assets_without_market_data() -> None:
    assets = [
        AnalysisAsset(requested_name="Bitcoin", symbol="BTC/USD", coin_id="bitcoin"),
        AnalysisAsset(requested_name="Ethereum", symbol="ETH/USD", coin_id="ethereum"),
    ]
    result = ResearchReport(
        request=AnalysisRequest(user_intent="Compare fundamentals", assets=assets),
        research_result=ResearchAgentResult(
            news=NewsEvidence(items=[], query="", collected_at=datetime.now(UTC)),
            fundamentals=FundamentalEvidence(name="Bitcoin", symbol="btc"),
            asset_results=[
                AssetResearchBundle(
                    asset=asset,
                    fundamentals=FundamentalEvidence(
                        name=asset.requested_name,
                        symbol=asset.symbol.split("/", maxsplit=1)[0],
                    ),
                )
                for asset in assets
            ],
        ),
    )

    presentation = build_research_presentation(result)

    assert presentation.title == "2-asset research comparison"
    assert [source.title for source in presentation.sources] == [
        "BTC/USD: Bitcoin fundamentals",
        "ETH/USD: Ethereum fundamentals",
    ]


def test_research_presentation_does_not_render_future_dated_sources() -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    result = ResearchReport(
        request=AnalysisRequest(user_intent="Review Bitcoin news"),
        collection_context=CollectionContext(collected_at=cutoff),
        research_result=ResearchAgentResult(
            news=NewsEvidence(
                items=[
                    NewsItem(
                        publisher="Example",
                        title="Future update",
                        excerpt="Future evidence must remain hidden.",
                        published_at=cutoff + timedelta(days=1),
                    )
                ],
                query="Bitcoin",
                collected_at=cutoff,
            )
        ),
    )

    assert build_research_presentation(result).sources == ()


def test_specialist_panels_quarantine_future_provider_observations() -> None:
    market = fake_market_service()
    cutoff = market.collected_at
    future_candles = [
        candle.model_copy(update={"timestamp": candle.timestamp + timedelta(days=1)})
        for candle in market.candles
    ]
    future_market = market.model_copy(
        update={
            "candles": future_candles,
            "first_time": future_candles[0].timestamp,
            "last_time": future_candles[-1].timestamp,
            "collected_at": cutoff + timedelta(days=1),
        }
    )
    future = cutoff + timedelta(days=1)
    result = ResearchReport(
        request=AnalysisRequest(user_intent="Review current evidence"),
        collection_context=CollectionContext(collected_at=cutoff),
        market_result=MarketAgentResult(
            market=future_market,
            technical=TechnicalSnapshot(trend="bullish", rsi=66.0),
        ),
        opportunity_result=OpportunityScanResult(
            exchange="kraken",
            timeframe="1h",
            collected_at=future,
            summary="Future scan summary.",
            candidates=[
                OpportunityCandidate(
                    rank=1,
                    asset="Future Asset",
                    symbol="FUT/USD",
                    current_price=999.0,
                    momentum_24h=12.0,
                    volatility_24h=3.0,
                    score=99.0,
                    trend="bullish",
                    reason="Future provider result.",
                )
            ],
        ),
        research_result=ResearchAgentResult(
            news=NewsEvidence(
                items=[
                    NewsItem(
                        publisher="Example",
                        title="Future headline",
                        excerpt="This must not reach the structured data card.",
                        published_at=future,
                    )
                ],
                query="Bitcoin",
                collected_at=future,
            ),
            requested_capabilities=[ResearchCapability.NEWS],
        ),
        fundamentals_result=FundamentalsAgentResult(
            fundamentals=FundamentalEvidence(
                name="Future Bitcoin",
                symbol="BTC",
                market_cap=1_234_567.0,
                collected_at=future,
            ),
            defi=DefiEvidence(
                protocol="Future protocol",
                tvl_usd=987_654.0,
                collected_at=future,
            ),
            requested_capabilities=[ResearchCapability.FUNDAMENTALS, ResearchCapability.DEFI],
        ),
    )

    presentation = build_research_presentation(result)
    panels = {panel.agent: panel for panel in presentation.agent_panels}

    assert presentation.discovery is None
    assert presentation.sources == ()
    assert panels["market_agent"].status == "partial"
    assert panels["market_agent"].state_label == "Partial"
    assert all(card.facts == (("Data status", "Excluded"),) for card in panels["market_agent"].data)
    research_facts = [fact for card in panels["news_agent"].data for fact in card.facts]
    assert ("Latest title", "Future headline") not in research_facts
    assert all("Future" not in value for _, value in research_facts)
    assert all(
        card.facts == (("Data status", "Excluded"),) for card in panels["fundamentals_agent"].data
    )


def test_specialist_panels_renormalize_stale_news_sources() -> None:
    cutoff = datetime(2026, 2, 1, tzinfo=UTC)
    stale = cutoff - timedelta(days=31)
    result = ResearchReport(
        request=AnalysisRequest(user_intent="Review research evidence"),
        collection_context=CollectionContext(collected_at=cutoff),
        research_result=ResearchAgentResult(
            news=NewsEvidence(
                items=[
                    NewsItem(
                        publisher="Example",
                        title="Stale headline",
                        excerpt="This expired item must not reach the structured data card.",
                        published_at=stale,
                    )
                ],
                query="Bitcoin",
                collected_at=cutoff,
            ),
            requested_capabilities=[ResearchCapability.NEWS],
        ),
    )

    presentation = build_research_presentation(result)
    panel = next(item for item in presentation.agent_panels if item.agent == "news_agent")
    facts = [fact for card in panel.data for fact in card.facts]

    assert presentation.sources == ()
    assert ("Latest title", "Stale headline") not in facts
    assert any(card.limitation and "stale" in card.limitation.casefold() for card in panel.data)


def test_discovery_presentation_renders_ranked_numeric_market_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collected_at = datetime(2026, 1, 1, tzinfo=UTC)
    result = ResearchReport(
        request=AnalysisRequest(user_intent="Discover market opportunities"),
        collection_context=CollectionContext(collected_at=collected_at),
        opportunity_result=OpportunityScanResult(
            exchange="kraken",
            timeframe="1h",
            collected_at=collected_at,
            summary="SOL led the deterministic market screen.",
            candidates=[
                OpportunityCandidate(
                    rank=1,
                    asset="SOL",
                    symbol="SOL/USD",
                    current_price=155.25,
                    score=82.4,
                    momentum_24h=4.75,
                    volatility_24h=2.2,
                    trend="bullish",
                    reason="Validated market-screen result.",
                )
            ],
        ),
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "crypto_research.interfaces.web.components.research.render_data_table",
        lambda rows, **kwargs: captured.update(rows=rows, table_kwargs=kwargs),
    )
    monkeypatch.setattr(
        "crypto_research.interfaces.web.components.research.st.dataframe",
        lambda *_args, **_kwargs: pytest.fail("Discovery must use the visible static table."),
    )
    monkeypatch.setattr(
        "crypto_research.interfaces.web.components.research.st.button",
        lambda *_args, **_kwargs: pytest.fail("Discovery must not render research buttons."),
    )

    presentation = build_research_presentation(result)
    _render_discovery_results(presentation.discovery)

    rows = captured["rows"]
    assert isinstance(rows, list)
    assert rows == [
        {
            "#": 1,
            "Asset": "SOL/USD",
            "Price": "$155.25",
            "24h momentum": "+4.75%",
            "24h volatility": "2.20%",
            "Score": "82.4/100",
            "Trend": "Bullish",
        }
    ]
    table_kwargs = captured["table_kwargs"]
    assert isinstance(table_kwargs, dict)
    assert table_kwargs["label"] == "Ranked market screen"
    assert table_kwargs["columns"] == (
        "#",
        "Asset",
        "Price",
        "24h momentum",
        "24h volatility",
        "Score",
        "Trend",
    )


def test_build_asset_presentations_keeps_the_freshest_duplicate_market() -> None:
    market = fake_market_service()
    latest_close = market.current_price + 0.25
    latest_candles = list(market.candles)
    latest_candles[-1] = latest_candles[-1].model_copy(update={"close": latest_close})
    latest = market.model_copy(
        update={
            "candles": latest_candles,
            "current_price": latest_close,
            "collected_at": market.collected_at + timedelta(minutes=1),
        }
    )
    result = ResearchReport(
        request=AnalysisRequest(user_intent="Review Bitcoin"),
        market_result=MarketAgentResult(
            market=market,
            technical=TechnicalSnapshot(trend="bullish"),
        ),
        market_comparison_result=MarketComparisonResult(
            assets=[
                MarketComparisonAsset(market=latest, technical=TechnicalSnapshot(trend="bullish")),
                MarketComparisonAsset(
                    market=market.model_copy(update={"symbol": "ETH/USD"}),
                    technical=TechnicalSnapshot(trend="neutral"),
                ),
            ]
        ),
    )

    assets = build_asset_presentations(result)

    assert [asset.symbol for asset in assets] == ["BTC/USD", "ETH/USD"]
    assert assets[0].current_price == latest_close


def test_research_presentation_contains_sanitized_metadata_and_labeled_sections() -> None:
    secret = "raw-provider-secret"
    market = fake_market_service()
    source = NewsItem(
        publisher=f"password={secret}",
        title="Bitcoin update",
        excerpt="Not retained by the UI.",
        url=f"https://example.test/article?token={secret}&page=2#section",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    result = ResearchReport(
        status="partial",
        request=AnalysisRequest(user_intent="Review Bitcoin"),
        collection_context=CollectionContext(collected_at=datetime(2026, 1, 1, tzinfo=UTC)),
        answer_state=AnswerState.PARTIAL,
        evidence_confidence=0.62,
        warnings=[f"api_key={secret}"],
        market_result=MarketAgentResult(
            market=market,
            technical=TechnicalSnapshot(trend="bullish"),
        ),
        research_result=ResearchAgentResult(
            news=NewsEvidence(
                items=[source], query="Bitcoin", collected_at=datetime(2026, 1, 1, tzinfo=UTC)
            ),
            fundamentals=FundamentalEvidence(),
            defi=DefiEvidence(protocol="Bitcoin DeFi", homepage="https://defi.example.test"),
            requested_capabilities=[ResearchCapability.NEWS, ResearchCapability.DEFI],
            capabilities=[ResearchCapability.NEWS, ResearchCapability.DEFI],
        ),
        agent_answers=[
            AgentAnswer(
                agent="news_agent",
                answer="The research evidence was reviewed.",
                confidence=0.8,
                evidence=[
                    EvidenceClaim(
                        statement=f"password={secret}",
                        evidence_ids=["market.BTC/USD"],
                        claim_kind="observed_fact",
                        confidence=0.8,
                    ),
                    EvidenceClaim(
                        statement="Observed market evidence was reviewed.",
                        evidence_ids=["market.BTC/USD"],
                        claim_kind="interpretation",
                        confidence=0.8,
                    ),
                    EvidenceClaim(
                        statement="Speculation: this scenario depends on stated assumptions.",
                        evidence_ids=["market.BTC/USD"],
                        claim_kind="speculation",
                        confidence=0.5,
                    ),
                    EvidenceClaim(
                        statement="Documented risks remain material.",
                        evidence_ids=["market.BTC/USD"],
                        claim_kind="risk",
                        confidence=0.7,
                    ),
                ],
            )
        ],
        disclaimer=f"Bearer {secret}",
        errors=[{"provider": secret}],
    )

    presentation = build_research_presentation(result, route=("news_agent",))

    assert secret not in repr(presentation)
    assert presentation.analysis_points
    assert presentation.speculation
    assert presentation.title == "BTC/USD market review"
    news_source = next(item for item in presentation.sources if item.kind == "News")
    assert news_source.url == "https://example.test/article?page=2"


def test_zero_confidence_risk_is_presented_as_unavailable() -> None:
    gap = "Current market and technical evidence is unavailable."
    result = ResearchReport(
        request=AnalysisRequest(user_intent="Review risk"),
        risk_result=RiskResult(
            assessment=RiskAssessment(
                score=0,
                band="low",
                evidence_confidence=0,
                coverage_gaps=[gap],
            )
        ),
    )

    presentation = build_research_presentation(result)

    assert gap in presentation.warnings


def test_agent_answer_presentation_preserves_evidence_labels_and_uncertainty() -> None:
    presentation = build_agent_answer_presentation(
        AgentAnswer(
            agent="market_agent",
            answer="The RSI evidence shows a bounded recent movement.",
            technical_terms=["RSI"],
            evidence=[
                EvidenceClaim(
                    statement="The observed move is a deterministic calculation.",
                    evidence_ids=["market.BTC/USD"],
                    claim_kind="calculation",
                    confidence=0.8,
                )
            ],
            uncertainty=["The window does not establish a future outcome."],
            limitations=["Provider freshness is limited to the collection timestamp."],
            confidence=0.75,
        )
    )

    assert presentation.title == "Market & Risk Agent"
    assert presentation.technical_terms == ("RSI",)
    assert presentation.claims[0].claim_kind == "calculation"
    assert presentation.claims[0].evidence_ids == ("market.BTC/USD",)
    assert presentation.uncertainty
    assert presentation.limitations


def test_agent_answer_presentation_preserves_structured_live_sections() -> None:
    presentation = build_agent_answer_presentation(
        AgentAnswer(
            agent="market_agent",
            answer="A compatibility summary remains available.",
            structured_analysis=StructuredAgentAnalysis(
                verdict="Market posture differs across the selected assets.",
                sections=[
                    AgentAnalysisSection(
                        asset="BTC/USD",
                        scope="market",
                        text="Momentum remains constructive within the observed window.",
                    ),
                    AgentAnalysisSection(
                        asset="BTC/USD",
                        scope="risk",
                        text="Observed risk remains bounded but evidence dependent.",
                    ),
                ],
                comparison="Bitcoin has the strongest relative posture in this sample.",
            ),
            confidence=0.8,
        )
    )

    assert presentation.structured_analysis is not None
    assert presentation.structured_analysis.verdict.startswith("Market posture")
    assert [section.scope for section in presentation.structured_analysis.sections] == [
        "market",
        "risk",
    ]
    assert presentation.structured_analysis.comparison.startswith("Bitcoin")


def test_research_presentation_orders_specialist_panels_and_keeps_owned_data() -> None:
    market = fake_market_service()
    collected_at = market.collected_at
    result = ResearchReport(
        request=AnalysisRequest(user_intent="Review Bitcoin market and news"),
        collection_context=CollectionContext(collected_at=collected_at),
        market_result=MarketAgentResult(
            market=market,
            technical=TechnicalSnapshot(trend="bullish", rsi=58.0),
        ),
        research_result=ResearchAgentResult(
            news=NewsEvidence(
                items=[
                    NewsItem(
                        publisher="Example",
                        title="Bitcoin update",
                        excerpt="A validated headline.",
                        published_at=collected_at,
                    )
                ],
                query="Bitcoin",
                collected_at=collected_at,
            ),
            requested_capabilities=[ResearchCapability.NEWS],
        ),
        fundamentals_result=FundamentalsAgentResult(
            fundamentals=FundamentalEvidence(
                name="Bitcoin",
                symbol="BTC",
                market_cap=1_000_000.0,
                collected_at=collected_at,
            ),
            defi=DefiEvidence(
                protocol="Bitcoin DeFi",
                tvl_usd=10_000.0,
                collected_at=collected_at,
            ),
            requested_capabilities=[ResearchCapability.FUNDAMENTALS, ResearchCapability.DEFI],
        ),
        agent_answers=[
            AgentAnswer(
                agent="market_agent",
                answer="The completed-candle trend is bullish.",
                confidence=0.8,
            ),
            AgentAnswer(
                agent="news_agent",
                answer="The normalized news item is current at collection time.",
                confidence=0.8,
            ),
            AgentAnswer(
                agent="fundamentals_agent",
                answer="The available fundamentals and DeFi metrics were reviewed.",
                confidence=0.8,
            ),
        ],
        agent_statuses=[
            AgentExecutionStatus(
                agent="market_agent",
                status="complete",
                capabilities=[ResearchCapability.MARKET],
            ),
            AgentExecutionStatus(
                agent="news_agent",
                status="complete",
                capabilities=[ResearchCapability.NEWS],
            ),
            AgentExecutionStatus(
                agent="fundamentals_agent",
                status="complete",
                capabilities=[ResearchCapability.FUNDAMENTALS, ResearchCapability.DEFI],
            ),
        ],
    )

    presentation = build_research_presentation(
        result,
        route=("news_agent", "market_agent", "fundamentals_agent"),
    )

    assert [panel.agent for panel in presentation.agent_panels] == [
        "news_agent",
        "market_agent",
        "fundamentals_agent",
    ]
    assert presentation.agent_panels[0].data[0].capability == "news"
    assert presentation.agent_panels[1].data[0].capability == "market"
    assert presentation.agent_panels[1].state_label == "Live answer"
    assert [card.capability for card in presentation.agent_panels[2].data] == [
        "fundamentals",
        "defi",
    ]
    fundamentals_facts = dict(presentation.agent_panels[2].data[0].facts)
    assert fundamentals_facts["Provider snapshot"] == "Live"
    assert fundamentals_facts["Collected"].endswith("UTC")


@pytest.mark.parametrize(
    ("capabilities", "expected_cards"),
    [
        ([ResearchCapability.MARKET], ["market"]),
        ([ResearchCapability.RISK], ["risk"]),
        ([ResearchCapability.MARKET, ResearchCapability.RISK], ["market", "risk"]),
    ],
)
def test_market_panel_resources_match_selected_topics(
    capabilities: list[ResearchCapability],
    expected_cards: list[str],
) -> None:
    market = fake_market_service()
    result = ResearchReport(
        request=AnalysisRequest(user_intent="Review Bitcoin"),
        collection_context=CollectionContext(collected_at=market.collected_at),
        market_result=MarketAgentResult(
            market=market,
            technical=TechnicalSnapshot(trend="bullish", rsi=58.0),
        ),
        risk_result=RiskResult(
            assessment=RiskAssessment(
                score=20,
                band="low",
                evidence_confidence=80,
            )
        ),
        agent_statuses=[
            AgentExecutionStatus(
                agent="market_agent",
                status="complete",
                capabilities=capabilities,
            )
        ],
    )

    panel = build_research_presentation(result).agent_panels[0]

    assert [card.capability for card in panel.data] == expected_cards


def test_research_presentation_omits_defi_resources_without_protocol_evidence() -> None:
    collected_at = datetime(2026, 1, 1, tzinfo=UTC)
    result = ResearchReport(
        request=AnalysisRequest(user_intent="Review Bitcoin fundamentals"),
        collection_context=CollectionContext(collected_at=collected_at),
        fundamentals_result=FundamentalsAgentResult(
            asset_results=[
                AssetResearchBundle(
                    asset=AnalysisAsset(
                        requested_name="BTC",
                        symbol="BTC/USD",
                        coin_id="bitcoin",
                    ),
                    fundamentals=FundamentalEvidence(
                        name="Bitcoin",
                        symbol="BTC",
                        market_cap=1_000_000.0,
                        collected_at=collected_at,
                    ),
                    defi=DefiEvidence(status="not_applicable", collected_at=collected_at),
                )
            ],
            requested_capabilities=[ResearchCapability.FUNDAMENTALS, ResearchCapability.DEFI],
        ),
        agent_answers=[
            AgentAnswer(
                agent="fundamentals_agent",
                answer="Bitcoin fundamentals were reviewed.",
                analysis="No DeFi protocol metrics were available.",
                confidence=0.8,
            )
        ],
        agent_statuses=[
            AgentExecutionStatus(
                agent="fundamentals_agent",
                status="complete",
                capabilities=[ResearchCapability.FUNDAMENTALS, ResearchCapability.DEFI],
            )
        ],
    )

    presentation = build_research_presentation(result)
    panel = presentation.agent_panels[0]

    assert panel.agent == "fundamentals_agent"
    assert panel.answer is not None
    assert panel.answer.answer == "Bitcoin fundamentals were reviewed."
    assert [card.capability for card in panel.data] == ["fundamentals"]


def test_live_analysis_badge_is_independent_from_partial_coverage() -> None:
    result = ResearchReport(
        request=AnalysisRequest(user_intent="Review Bitcoin"),
        status="partial",
        warnings=["Fundamentals coverage was unavailable."],
        agent_answers=[
            AgentAnswer(
                agent="market_agent",
                answer="Bitcoin market analysis completed from validated evidence.",
                limitations=["Fundamentals coverage was unavailable."],
                confidence=0.8,
                analysis_state="live",
                coverage_state="partial",
            )
        ],
        agent_statuses=[
            AgentExecutionStatus(
                agent="market_agent",
                status="complete",
                analysis_state="live",
                coverage_state="partial",
                capabilities=[ResearchCapability.MARKET],
            )
        ],
    )

    panel = build_research_presentation(result).agent_panels[0]

    assert panel.state_label == "Live answer"
    assert panel.status == "complete"
    assert panel.coverage_state == "partial"
