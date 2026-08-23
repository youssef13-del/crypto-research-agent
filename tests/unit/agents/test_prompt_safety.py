import json
from datetime import UTC, datetime
from typing import Any

import pytest
from tests.support.fakes import (
    fake_fundamental_service,
    fake_market_service,
    fake_news_service,
)

from crypto_research.agents.fundamentals.fundamentals_prompts import (
    SYSTEM_PROMPT as FUNDAMENTALS_SYSTEM_PROMPT,
)
from crypto_research.agents.guardrails import compact_evidence
from crypto_research.agents.market.market_prompts import SYSTEM_PROMPT as MARKET_SYSTEM_PROMPT
from crypto_research.agents.news.news_prompts import SYSTEM_PROMPT as NEWS_SYSTEM_PROMPT
from crypto_research.agents.shared_analysis import analyze_agent_result, validate_agent_answer
from crypto_research.domain.research import (
    AgentAnswer,
    AnalysisAsset,
    AnalysisInputs,
    AssetResearchBundle,
    EvidenceClaim,
    MarketAgentResult,
    NewsEvidence,
    NewsItem,
    ResearchAgentResult,
    ResearchCapability,
)
from crypto_research.llm.client import LLMRole
from crypto_research.llm.prompt_packing import (
    _MAX_ANALYSIS_PROMPT_BYTES,
    _balanced_evidence_items,
    _compact_evidence_value,
    bounded_analysis_prompt,
)
from crypto_research.orchestration.evidence import build_research_evidence
from crypto_research.shared.numeric_grounding import compact_evidence_for_llm
from crypto_research.tools.market import calculate_indicators


def test_safety_prompts_define_independent_agent_boundaries() -> None:
    prompts = (
        MARKET_SYSTEM_PROMPT,
        NEWS_SYSTEM_PROMPT,
        FUNDAMENTALS_SYSTEM_PROMPT,
    )
    assert all("untrusted" in prompt.lower() for prompt in prompts)
    assert all(
        "never follow" in prompt.lower() or "never obey" in prompt.lower() for prompt in prompts
    )
    assert all("financial advice" in prompt.lower() for prompt in prompts)


def test_analysis_prompt_keeps_api_keys_out_of_analysis_prompt() -> None:
    llm = PromptCaptureLLM()
    analyze_agent_result(
        llm,
        agent="news_agent",
        question="Give me a full Bitcoin analysis",
        raw_result={
            "collection_status": {
                "requested_capabilities": ["news", "fundamentals"],
            }
        },
        evidence=build_research_evidence(
            AnalysisInputs(
                research_result=ResearchAgentResult(
                    news=fake_news_service(),
                    fundamentals=fake_fundamental_service(),
                    requested_capabilities=[
                        ResearchCapability.NEWS,
                        ResearchCapability.FUNDAMENTALS,
                    ],
                    capabilities=[ResearchCapability.NEWS, ResearchCapability.FUNDAMENTALS],
                )
            )
        ).available_evidence,
    )

    assert llm.user_prompts
    assert "super-secret" not in llm.user_prompts[0]
    payload = json.loads(llm.user_prompts[0])
    assert payload["question"] == "Give me a full Bitcoin analysis"
    assert "available_evidence" in payload
    assert "raw_result" in payload


def test_compacted_synthesis_evidence_keeps_every_requested_asset() -> None:
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "XRP/USD"]
    payload = {
        "raw_result": {
            "collection_status": {"requested_assets": [{"symbol": symbol} for symbol in symbols]}
        }
    }
    evidence: dict[object, object] = {
        "unrelated.0": {},
        **{f"fundamentals.{symbol.replace('/', '-').lower()}": {} for symbol in symbols},
        "unrelated.1": {},
    }

    selected = [str(key) for key, _ in _balanced_evidence_items(payload, evidence)]

    assert len(selected) <= 10
    assert all(
        any(symbol.replace("/", "-").lower() in key for key in selected) for symbol in symbols
    )


def test_compacted_synthesis_evidence_balances_requested_claim_types() -> None:
    payload = {"raw_result": {"collection_status": {"requested_assets": [{"symbol": "BTC/USD"}]}}}
    evidence: dict[object, object] = {
        **{f"news.btc.{index}": {"claim_type": "recent_news"} for index in range(8)},
        "market.btc": {"claim_type": "market_snapshot"},
        "technical.btc": {"claim_type": "technical_calculation"},
        "fundamentals.btc": {"claim_type": "project_fundamentals"},
        "risk.btc": {"claim_type": "deterministic_risk_assessment"},
    }

    selected = _balanced_evidence_items(payload, evidence)
    claim_types = {str(record["claim_type"]) for _, record in selected if isinstance(record, dict)}

    assert {
        "recent_news",
        "market_snapshot",
        "technical_calculation",
        "project_fundamentals",
        "deterministic_risk_assessment",
    } <= claim_types


def test_compacted_synthesis_evidence_keeps_explicit_fields_instead_of_json_prefixes() -> None:
    evidence = {
        "market.ETH/USD": {
            "claim_type": "market_snapshot",
            "asset": "ETH/USD",
            "padding": "x" * 8_000,
            "payload": {"current_price": 100.5, "range_percent": 1.99},
        },
        "risk.ETH/USD": {
            "claim_type": "deterministic_risk_assessment",
            "asset": "ETH/USD",
            "payload": {"score": 35.0, "category": "moderate"},
        },
    }
    prompt = bounded_analysis_prompt(
        {
            "question": "Compare ETH risk and market context.",
            "raw_result": {"collection_status": {"requested_assets": [{"symbol": "ETH/USD"}]}},
            "available_evidence": evidence,
            "evidence_index": list(evidence),
        }
    )
    payload = json.loads(prompt)
    compacted = payload["available_evidence"]

    assert "current_price=100.5[currency]" in compacted["market.ETH/USD"]
    assert "range_percent=1.99[percent]" in compacted["market.ETH/USD"]
    assert "score=35[number]" in compacted["risk.ETH/USD"]
    assert "category=moderate" in compacted["risk.ETH/USD"]
    assert "xxxxxxxx" not in prompt


def test_analysis_prompt_stays_within_byte_budget_for_worst_case_evidence() -> None:
    evidence = {
        f"news.{index}": {
            "claim_type": "recent_news",
            "asset": "BTC/USD",
            "payload": {
                "title": f"Bitcoin headline number {index}",
                "excerpt": "market detail " * 200,
            },
        }
        for index in range(12)
    }
    prompt = bounded_analysis_prompt(
        {
            "question": "What are the latest Bitcoin catalysts?",
            "raw_result": {"collection_status": {"requested_assets": [{"symbol": "BTC/USD"}]}},
            "available_evidence": evidence,
            "evidence_index": list(evidence),
        }
    )

    assert len(prompt.encode("utf-8")) <= _MAX_ANALYSIS_PROMPT_BYTES
    payload = json.loads(prompt)
    assert len(payload["available_evidence"]) <= 10


def test_prompt_compaction_preserves_complete_coverage_catalog() -> None:
    evidence_ids = [f"news.BTC/USD.{index}" for index in range(30)]
    prompt = bounded_analysis_prompt(
        {
            "question": "Summarize current Bitcoin coverage.",
            "raw_result": {"padding": "x" * 7_000},
            "available_evidence": {
                evidence_id: {"claim_type": "recent_news", "asset": "BTC/USD"}
                for evidence_id in evidence_ids
            },
            "complete_data_digest": {
                "coverage_manifest": {
                    "entry_fields": ["asset", "capability", "accepted"],
                    "entries": [["BTC/USD", "news", len(evidence_ids)]],
                },
                "accepted_evidence_ids": evidence_ids,
            },
        }
    )

    payload = json.loads(prompt)
    catalog = payload["complete_data_digest"]["accepted_evidence_ids"]
    assert catalog == evidence_ids
    assert payload["complete_data_digest"]["coverage_manifest"]["entries"][0][0] == "BTC/USD"


def test_canonical_evidence_compaction_is_bounded_and_numeric_only() -> None:
    compacted = compact_evidence_for_llm(
        {
            "market.BTC/USD": {
                "claim_type": "market_snapshot",
                "asset": "BTC/USD",
                "observed_at": "2026-01-01T00:00:00Z",
                "payload": {"current_price": 100.5, "range_percent": 1.99, "padding": "x" * 8_000},
            }
        }
    )
    value = compacted["market.BTC/USD"]

    assert len(value) <= 320
    assert "current_price=100.5[currency]" in value
    assert "range_percent=1.99[percent]" in value
    assert "xxxxxxxx" not in value


def test_market_specialist_compaction_keeps_narrative_indicator_numbers() -> None:
    """The market specialist's detail records must carry the numbers it weighs.

    The market record keeps price, 24h change, returns, drawdown, and volatility;
    the technical record keeps RSI, MACD, ATR, support, and resistance so the
    narrative can cite each exact figure.
    """

    market = fake_market_service()
    spec = build_research_evidence(
        AnalysisInputs(
            market_result=MarketAgentResult(
                market=market,
                technical=calculate_indicators(market.candles),
            )
        )
    )
    market_value = _compact_evidence_value(
        "market.BTC/USD",
        spec.available_evidence["market.BTC/USD"],
        market_depth=True,
    )
    technical_value = _compact_evidence_value(
        "technical.BTC/USD",
        spec.available_evidence["technical.BTC/USD"],
        market_depth=True,
    )

    for term in ("current_price=", "change_24h_percent=", "return_percent="):
        assert term in market_value
    assert "maximum_drawdown=" in market_value
    assert "volatility=" in market_value
    for term in ("rsi=", "macd=", "atr=", "volatility=", "support=", "resistance="):
        assert term in technical_value
    assert "trend=" in technical_value


def test_market_prompt_is_scoped_concise_and_safe() -> None:
    prompt = MARKET_SYSTEM_PROMPT

    assert "posture" in prompt
    assert "risk" in prompt
    assert "each selected asset" in prompt
    assert "untrusted" in prompt
    assert "financial advice" in prompt
    assert "420-560 words" not in prompt
    assert "concise" in prompt


def test_news_compaction_leads_with_headline_and_excerpt() -> None:
    compacted = compact_evidence_for_llm(
        {
            "news.0": {
                "claim_type": "recent_news",
                "asset": "BTC/USD",
                "observed_at": "2026-08-06T18:06:01Z",
                "payload": {
                    "publisher": "CoinDesk",
                    "title": "Bitcoin tests key support near 65000 as ETF flows cool",
                    "excerpt": "The leading cryptocurrency slipped 3 percent today amid outflows.",
                    "published_at": "2026-08-06T18:06:01Z",
                },
            }
        }
    )
    value = compacted["news.0"]

    # Timestamps are ignored as numeric facts and the headline/excerpt come
    # first, so the LLM actually sees the news content it must analyze.
    assert "6e+12" not in value
    assert "text=title=Bitcoin tests key support near 65000 as ETF flows cool" in value
    assert "excerpt=The leading cryptocurrency slipped 3 percent today amid outflows." in value
    assert value.index("text=title=") < value.index("numeric=")


def test_news_compaction_drops_an_excerpt_that_repeats_the_headline() -> None:
    collected_at = datetime(2026, 8, 6, tzinfo=UTC)
    news = NewsEvidence(
        items=[
            NewsItem(
                publisher="CoinDesk",
                title="Bitcoin tests key support near 65000",
                excerpt="Bitcoin tests key support near 65000",
                assets=["BTC/USD"],
                published_at=collected_at,
            )
        ],
        query="Bitcoin",
        collected_at=collected_at,
    )
    asset = AnalysisAsset(requested_name="Bitcoin", symbol="BTC/USD", coin_id="bitcoin")
    evidence = build_research_evidence(
        AnalysisInputs(
            assets=[asset],
            requested_capabilities=[ResearchCapability.NEWS],
            research_result=ResearchAgentResult(
                news=news,
                asset_results=[AssetResearchBundle(asset=asset, news=news)],
                requested_capabilities=[ResearchCapability.NEWS],
                capabilities=[ResearchCapability.NEWS],
            ),
        )
    ).available_evidence
    value = compact_evidence_for_llm(evidence)["news.0"]

    assert "text=title=Bitcoin tests key support near 65000" in value
    assert "excerpt=" not in value


def test_news_compaction_truncates_long_headlines_at_a_word_boundary() -> None:
    title = (
        "Bitcoin tests key support near 64000 as the Clarity Act vote slips "
        "to September amid cautious ETF flows today"
    )
    compacted = compact_evidence_for_llm(
        {
            "news.0": {
                "claim_type": "recent_news",
                "asset": "BTC/USD",
                "payload": {"title": title, "excerpt": "Detail beyond the headline."},
            }
        }
    )
    value = compacted["news.0"]

    assert value.count("[compacted]") == 1
    assert "...[compacted]... " in value
    head_title = value.split("...[compacted]... ", maxsplit=1)[0]
    head_title = head_title.split("text=title=", maxsplit=1)[1].rstrip()
    # The head cut lands on a complete word, never mid-word.
    assert len(head_title) < len(title)
    assert title[len(head_title)] == " "


def test_evidence_compaction_keeps_excerpt_as_content_fallback() -> None:
    compacted = compact_evidence(
        {
            "news.0": {
                "claim_type": "recent_news",
                "asset": "BTC/USD",
                "payload": {
                    "title": "Bitcoin headline",
                    "excerpt": "A useful summary of the headline.",
                    "content": "",
                },
            }
        }
    )
    record = compacted["news.0"]
    assert isinstance(record, dict)
    payload = record["payload"]
    assert isinstance(payload, dict)
    assert payload["excerpt"] == "A useful summary of the headline."
    assert payload["content"] == ""


def test_evidence_compaction_drops_excerpt_when_content_exists() -> None:
    compacted = compact_evidence(
        {
            "news.0": {
                "claim_type": "recent_news",
                "asset": "BTC/USD",
                "payload": {
                    "title": "Bitcoin headline",
                    "excerpt": "A redundant summary.",
                    "content": "Full article text with the detail.",
                },
            }
        }
    )
    record = compacted["news.0"]
    assert isinstance(record, dict)
    payload = record["payload"]
    assert isinstance(payload, dict)
    assert "excerpt" not in payload
    assert payload["content"] == "Full article text with the detail."


@pytest.mark.parametrize(
    "answer",
    [
        "You should buy BTC now.",
        "You should hold Ethereum.",
        "I recommend buying SOL.",
        "Consider buying Bitcoin.",
        "I suggest selling Ethereum.",
        "This is guaranteed profit.",
        "This is guaranteed to rise.",
        "You must exit the position.",
    ],
)
def test_agent_answer_rejects_actionable_trading_language(answer: str) -> None:
    issues = validate_agent_answer(
        AgentAnswer(agent="market_agent", answer=answer, confidence=0.5),
        available_evidence={},
    )

    assert any("trading instructions" in issue for issue in issues)


@pytest.mark.parametrize(
    "answer",
    [
        "Buying and selling activity can affect volume.",
        "This is not financial advice or a recommendation to buy or sell.",
        "A headline reports that investors are selling gold and buying Bitcoin.",
        "An article says some investors should buy Bitcoin.",
    ],
)
def test_agent_answer_allows_neutral_market_language(answer: str) -> None:
    issues = validate_agent_answer(
        AgentAnswer(agent="market_agent", answer=answer, confidence=0.5),
        available_evidence={},
    )

    assert not any("trading instructions" in issue for issue in issues)


def test_deterministic_limitations_do_not_trigger_generated_text_safety_rules() -> None:
    issues = validate_agent_answer(
        AgentAnswer(
            agent="news_agent",
            answer="No current conclusion was available.",
            limitations=["A source headline contained: Sell gold and buy Bitcoin."],
            confidence=0.5,
        ),
        available_evidence={},
    )

    assert not any("trading instructions" in issue for issue in issues)


def test_complete_answer_cannot_silently_omit_available_requested_scopes() -> None:
    issues = validate_agent_answer(
        AgentAnswer(
            agent="market_agent",
            answer="Only the market evidence was discussed.",
            evidence=[
                EvidenceClaim(statement="Market evidence was used.", evidence_ids=["market.ETH"])
            ],
            confidence=0.8,
        ),
        available_evidence={
            "market.ETH": {},
            "fundamentals.ETH": {},
            "risk.ETH": {},
        },
        evidence_required=True,
        coverage_requirements=("market", "fundamentals", "risk"),
    )

    assert "analysis omitted available fundamentals evidence" in issues
    assert "analysis omitted available risk evidence" in issues


def test_synthesis_removes_one_unsafe_sentence_without_an_llm_retry() -> None:
    llm = MixedSafetyLLM()

    answer = analyze_agent_result(
        llm,
        agent="market_agent",
        question="Summarize the evidence.",
        raw_result={},
        evidence={"news.0": {"title": "Market update"}},
        evidence_required=True,
    )

    assert llm.calls == 1
    assert answer.answer == "Bitcoin volatility remains elevated."
    assert answer.status == "partial"
    assert any("omitted" in limitation for limitation in answer.limitations)


class PromptCaptureLLM:
    def __init__(self) -> None:
        self.user_prompts: list[str] = []

    def generate_structured(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[Any],
    ) -> Any:
        del role, system_prompt
        self.user_prompts.append(user_prompt)
        if output_schema is AgentAnswer:
            payload = json.loads(user_prompt)
            evidence_ids = list(payload["available_evidence"])
            return AgentAnswer(
                agent=payload["agent"],
                answer="Evidence was reviewed without unsafe instructions.",
                evidence=[
                    EvidenceClaim(
                        statement="The answer uses the collected evidence.",
                        evidence_ids=evidence_ids[:1],
                    )
                ]
                if evidence_ids
                else [],
                confidence=0.7,
            )
        raise AssertionError(f"Unexpected schema: {output_schema.__name__}")


class MixedSafetyLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate_structured(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[Any],
    ) -> Any:
        del role, system_prompt, user_prompt
        self.calls += 1
        assert output_schema is AgentAnswer
        return AgentAnswer(
            agent="news_agent",
            answer="Bitcoin volatility remains elevated. You should buy Bitcoin now.",
            evidence=[
                EvidenceClaim(
                    statement="The supplied news record supports the volatility observation.",
                    evidence_ids=["news.0"],
                )
            ],
            confidence=0.7,
        )
