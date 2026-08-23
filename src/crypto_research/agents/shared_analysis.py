"""Bounded, evidence-grounded analysis shared by all specialist agents."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, cast

from pydantic import ValidationError

from crypto_research.agents.base import AgentAnalyzer
from crypto_research.agents.fundamentals.fundamentals_analyzer import (
    FundamentalsLiveOutput,
)
from crypto_research.agents.guardrails import (
    AnswerRequirements,
    answer_requirement_issues,
    compact_evidence,
    compile_answer_requirements,
    coverage_issues,
    evidence_first_answer,
    live_unavailable_answer,
    no_evidence_message,
    numeric_repair_context,
    requirement_limitations,
    salvage_live_answer,
)
from crypto_research.agents.market.market_analyzer import (
    MarketLiveOutput,
)
from crypto_research.agents.news.news_analyzer import (
    NewsLiveOutput,
)
from crypto_research.agents.registry import analyzer_for
from crypto_research.domain.research import (
    AgentAnswer,
    AgentId,
    AnalysisInputs,
    ResearchCapability,
)
from crypto_research.llm.client import (
    DisabledLLMAdapter,
    LLMAdapter,
    LLMError,
    live_failure_category,
    public_failure_reason,
)
from crypto_research.llm.prompt_packing import (
    PromptEvidenceBundle,
    bounded_analysis_prompt,
    build_prompt_evidence_bundle,
    build_specialist_analysis_payload,
    contains_compaction_marker,
    jsonable_prompt_value,
)
from crypto_research.orchestration import evidence as evidence_services
from crypto_research.orchestration.evidence import EvidenceSpec

logger = logging.getLogger(__name__)


type _GuidedLiveOutput = MarketLiveOutput | FundamentalsLiveOutput | NewsLiveOutput

_UNSAFE_LANGUAGE = re.compile(
    r"(?i)(?:"
    r"\b(?:you|i)\s+(?:should|must|need to|ought to)\s+"
    r"(?:buy|sell|hold|short|invest|allocate|enter|exit)\b|"
    r"\b(?:i\s+)?(?:recommend|suggest)\s+(?:that\s+you\s+)?"
    r"(?:buy|buying|sell|selling|hold|holding|short|shorting|invest|allocate|enter|exit)\b|"
    r"\b(?:consider|try)\s+(?:buying|selling|holding|shorting|investing)\b|"
    r"\b(?:guaranteed|risk[- ]free|certain)\s+(?:profit|return|gain|win)\b|"
    r"\bguarantee(?:d|s)?\s+to\s+(?:rise|fall|increase|decrease)\b"
    r")"
)
_URL = re.compile(r"(?i)(?:https?://|javascript:|data:|file:)")

_GUIDED_SPECIALISTS = frozenset(
    {"market_agent", "news_agent", "fundamentals_agent", "onchain_agent"}
)


class AgentAnalysisError(RuntimeError):
    """Raised when a generated specialist answer cannot be made safe."""


def analyze_agent_result(
    llm: LLMAdapter,
    *,
    agent: AgentId,
    question: str,
    raw_result: object,
    evidence: Mapping[str, object],
    complete_data_digest: Mapping[str, object] | None = None,
    limitations: Sequence[str] = (),
    evidence_required: bool = False,
    coverage_requirements: Sequence[str] = (),
    no_evidence_message: str | None = None,
    enforce_live_requirements: bool = False,
    prompt_bundle: PromptEvidenceBundle | None = None,
    payload_budget_bytes: int | None = None,
) -> AgentAnswer:
    """Return one grounded specialist answer without a repair-model request.

    ``prompt_bundle`` is injectable so the evidence layer can pre-budget the
    specialist payload. Guided prompts receive compact details, while local
    deterministic claims retain the complete validated specialist ledger.
    """

    if agent not in _GUIDED_SPECIALISTS:
        raise ValueError(f"Specialist analysis does not support {agent!r}.")
    analyzer = analyzer_for(agent)
    if evidence_required and not evidence:
        return evidence_first_answer(
            agent=agent,
            question=question,
            message=no_evidence_message
            or "I could not find enough verified information to answer that reliably.",
            limitations=limitations,
            evidence={},
        )

    raw_value = jsonable_prompt_value(raw_result)
    raw_payload = raw_value if isinstance(raw_value, Mapping) else {}
    compacted_evidence = jsonable_prompt_value(compact_evidence(evidence))
    evidence_payload = compacted_evidence if isinstance(compacted_evidence, Mapping) else {}
    bundle = prompt_bundle or build_prompt_evidence_bundle(raw_payload, evidence_payload)
    selected_evidence = {
        evidence_id: evidence[evidence_id]
        for evidence_id in bundle.detailed_evidence_ids
        if evidence_id in evidence
    }
    grounding_evidence = dict(evidence) if bundle.specialist else selected_evidence
    requirements = compile_answer_requirements(
        question=question,
        raw_result=raw_payload,
        coverage_requirements=coverage_requirements,
        evidence=grounding_evidence,
    )
    prompt_limitations = list(
        dict.fromkeys(
            [
                *(str(item) for item in limitations if str(item).strip()),
                *requirement_limitations(requirements),
            ]
        )
    )
    processing_notes = (
        ["Long provider records were compacted before this specialist analysis."]
        if any(contains_compaction_marker(value) for value in (raw_payload, evidence_payload))
        else []
    )
    output_contract = {
        **analyzer.output_contract(coverage_requirements),
        "facts": "Use qualitative interpretation only; exact facts are inserted locally.",
        "safety": "No trading instructions, guarantees, URLs, or invented facts.",
    }
    payload = {
        "agent": agent,
        "question": _compact_question(question),
        "raw_result": raw_payload,
        "available_evidence": bundle.available_evidence,
        "evidence_index": list(bundle.detailed_evidence_ids),
        "detailed_evidence_ids": list(bundle.detailed_evidence_ids),
        "complete_data_digest": jsonable_prompt_value(
            bundle.analysis_data_digest or complete_data_digest or {}
        ),
        "evidence_required": evidence_required,
        "requested_coverage": list(dict.fromkeys(coverage_requirements)),
        "answer_requirements": requirements.prompt_value(),
        "known_limitations": prompt_limitations,
        "processing_notes": processing_notes,
        "output_contract": output_contract,
    }
    if bundle.specialist:
        # The three Guided specialists have a strict 3.5 KB request budget.
        # Do not let duplicated aliases, evidence-ID lists, or boilerplate
        # squeeze out a valid dense (for example 3 x 6) scope digest.
        payload = _specialist_prompt_payload(
            agent=agent,
            question=_compact_question(question),
            raw_result=raw_payload,
            requirements=requirements,
            limitations=prompt_limitations,
        )
    prompt = bounded_analysis_prompt(
        payload,
        budget_bytes=payload_budget_bytes or bundle.budget_bytes,
    )
    generated = _generate_agent_answer(
        llm,
        agent=agent,
        analyzer=analyzer,
        user_prompt=prompt,
        selected_evidence=grounding_evidence,
        prompt_limitations=prompt_limitations,
        requirements=requirements,
        use_specialist_schema=bundle.specialist and agent in _GUIDED_SPECIALISTS,
    )
    if getattr(llm, "last_call_used_fallback", False):
        return evidence_first_answer(
            agent=agent,
            question=question,
            message="A full specialist interpretation is currently unavailable.",
            limitations=prompt_limitations,
            evidence=grounding_evidence,
        )
    answer = AgentAnswer.model_validate(generated)
    if answer.status == "unavailable":
        return evidence_first_answer(
            agent=agent,
            question=question,
            message=(
                answer.limitations[0]
                if answer.limitations
                else "A reliable specialist interpretation could not be produced."
            ),
            limitations=[*answer.limitations[1:], *prompt_limitations],
            evidence=selected_evidence,
        )
    normalized = _normalize_answer(answer, agent=agent, limitations=prompt_limitations)
    issues = (
        _validate_composed_specialist_answer(
            normalized,
            available_evidence=grounding_evidence,
            evidence_required=evidence_required and bool(grounding_evidence),
        )
        if bundle.specialist and agent in _GUIDED_SPECIALISTS
        else validate_agent_answer(
            normalized,
            available_evidence=grounding_evidence,
            evidence_required=evidence_required and bool(grounding_evidence),
            coverage_requirements=coverage_requirements,
            answer_requirements=requirements if enforce_live_requirements else None,
        )
    )
    if not issues:
        return normalized
    salvaged = salvage_live_answer(normalized, grounding_evidence, issues, requirements)
    if salvaged is not None:
        return salvaged
    logger.info(
        "specialist_analysis agent=%s status=partial stage=answer_validation reasons=%s",
        agent,
        "; ".join(issues),
    )
    return evidence_first_answer(
        agent=agent,
        question=question,
        message="The generated interpretation did not pass evidence validation.",
        limitations=[*prompt_limitations, "Only validated records are shown below."],
        evidence=grounding_evidence,
    )


def _validate_composed_specialist_answer(
    answer: AgentAnswer,
    *,
    available_evidence: Mapping[str, object],
    evidence_required: bool,
) -> list[str]:
    """Validate only invariants that remain model-controlled after composition.

    Guided prose is composed locally: numeric facts and evidence claims come
    directly from validated records, while model text is forced qualitative.
    Re-running heuristic numeric/asset extraction over those local sentences
    produced false failures for dates and cross-asset news headlines.
    """

    issues: list[str] = []
    if not answer.answer.strip():
        issues.append("composed specialist answer is empty")
    if evidence_required and not answer.evidence:
        issues.append("composed specialist answer omitted evidence claims")
    allowed_ids = set(available_evidence)
    issues.extend(
        "unknown evidence identifiers: " + ", ".join(unknown)
        for claim in answer.evidence
        if (unknown := sorted(set(claim.evidence_ids) - allowed_ids))
    )
    if _UNSAFE_LANGUAGE.search(answer.answer) or _URL.search(answer.answer):
        issues.append("composed specialist answer crossed a safety boundary")
    return issues


def _generate_agent_answer(
    llm: LLMAdapter,
    *,
    agent: AgentId,
    analyzer: AgentAnalyzer,
    user_prompt: str,
    selected_evidence: Mapping[str, object],
    prompt_limitations: Sequence[str],
    requirements: AnswerRequirements,
    use_specialist_schema: bool,
) -> AgentAnswer:
    if use_specialist_schema:
        generated = cast(
            _GuidedLiveOutput,
            llm.generate_structured(
                role=analyzer.role,
                system_prompt=_specialist_structured_system_prompt(
                    analyzer.system_prompt,
                    instruction=analyzer.structured_instruction(requirements.scopes),
                ),
                user_prompt=user_prompt,
                output_schema=analyzer.output_schema,
            ),
        )
        if analyzer.compose is None:
            raise AgentAnalysisError(f"{agent} does not define a live-answer composer.")
        return analyzer.compose(
            generated,
            selected_evidence=selected_evidence,
            limitations=prompt_limitations,
            requirements=requirements,
        )
    return llm.generate_structured(
        role=analyzer.role,
        system_prompt=analyzer.system_prompt,
        user_prompt=user_prompt,
        output_schema=AgentAnswer,
    )


def _specialist_structured_system_prompt(
    system_prompt: str,
    *,
    instruction: str,
) -> str:
    return (
        system_prompt
        + "\n\nReturn the requested strict object with exactly one ordered entry per requested "
        "asset. Write one natural sentence in each text field. Do not repeat the verdict, use "
        "headings, or restate raw evidence. Do not include numeric values, dates, URLs, evidence "
        "IDs, approximate quantities written as words, markdown, trading instructions, or fields "
        "outside the schema. Exact facts are inserted locally. " + instruction
    )


def validate_agent_answer(
    answer: AgentAnswer,
    *,
    available_evidence: Mapping[str, object],
    evidence_required: bool = False,
    coverage_requirements: Sequence[str] = (),
    answer_requirements: AnswerRequirements | None = None,
) -> list[str]:
    """Validate specialist prose and claims against its exact prompt ledger."""

    issues: list[str] = []
    allowed_ids = set(available_evidence)
    text = " ".join(
        [
            answer.answer,
            answer.analysis,
            *(claim.statement for claim in answer.evidence),
            *answer.uncertainty,
            *answer.suggested_followups,
        ]
    )
    if _URL.search(text):
        issues.append("URLs must be supplied through validated source metadata, not generated text")
    if _UNSAFE_LANGUAGE.search(text):
        issues.append("analysis contains trading instructions or guaranteed-return language")
    for claim in answer.evidence:
        unknown = sorted(set(claim.evidence_ids) - allowed_ids)
        if unknown:
            issues.append("unknown evidence IDs: " + ", ".join(unknown))
    if evidence_required and not answer.evidence:
        issues.append("evidence-backed analysis requires at least one validated evidence claim")
    numeric_issues = numeric_repair_context(answer, available_evidence)
    if any(item.get("code") == "numeric_answer" for item in numeric_issues):
        issues.append("answer contains a numeric value not supported by a cited evidence claim")
    if any(item.get("code") == "numeric_claim" for item in numeric_issues):
        issues.append("evidence claim contains a numeric value not supported by its records")
    issues.extend(coverage_issues(answer, available_evidence, coverage_requirements))
    if answer_requirements is not None:
        issues.extend(
            answer_requirement_issues(
                answer,
                available_evidence=available_evidence,
                requirements=answer_requirements,
            )
        )
    return list(dict.fromkeys(issues))


def _normalize_answer(
    answer: AgentAnswer,
    *,
    agent: AgentId,
    limitations: Sequence[str],
) -> AgentAnswer:
    merged_limitations = list(dict.fromkeys([*answer.limitations, *limitations]))[:6]
    clean_answer = _safe_text(answer.answer)
    clean_analysis = _safe_text(answer.analysis)
    clean_claims = [
        claim
        for claim in answer.evidence
        if not _UNSAFE_LANGUAGE.search(claim.statement) and not _URL.search(claim.statement)
    ]
    if not clean_answer:
        raise AgentAnalysisError("all generated answer text failed safety validation")
    removed = (
        clean_answer != answer.answer
        or clean_analysis != answer.analysis
        or len(clean_claims) != len(answer.evidence)
    )
    if removed:
        merged_limitations = list(
            dict.fromkeys(
                [
                    "One generated statement was omitted because it failed safety validation.",
                    *merged_limitations,
                ]
            )
        )[:6]
    return AgentAnswer(
        agent=agent,
        answer=clean_answer,
        analysis=clean_analysis,
        structured_analysis=answer.structured_analysis,
        technical_terms=[
            term
            for term in answer.technical_terms
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", clean_answer, re.I)
        ],
        evidence=clean_claims,
        uncertainty=[
            item
            for item in answer.uncertainty
            if not _UNSAFE_LANGUAGE.search(item) and not _URL.search(item)
        ],
        limitations=merged_limitations,
        suggested_followups=[
            item
            for item in answer.suggested_followups
            if not _UNSAFE_LANGUAGE.search(item) and not _URL.search(item)
        ],
        confidence=answer.confidence,
        status="partial" if removed or answer.status == "partial" else "complete",
        analysis_state=answer.analysis_state,
        coverage_state="partial" if merged_limitations else answer.coverage_state,
    )


def _safe_text(value: str) -> str:
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", value.strip()):
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        clean = " ".join(
            item
            for item in sentences
            if item and not _UNSAFE_LANGUAGE.search(item) and not _URL.search(item)
        )
        if clean:
            paragraphs.append(clean)
    return "\n\n".join(paragraphs)


def _compact_question(question: str) -> str:
    """Reserve specialist-prompt space for grounded evidence and citations."""

    normalized = " ".join(question.split())
    if len(normalized) <= 700:
        return normalized
    return normalized[:680].rstrip() + " ...[question compacted]"


def _specialist_prompt_payload(
    *,
    agent: AgentId,
    question: str,
    raw_result: Mapping[str, object],
    requirements: AnswerRequirements,
    limitations: Sequence[str],
) -> dict[str, object]:
    """Build the compact envelope used by the three Guided specialists.

    The evidence bundle already budgets citation-ready details and scope
    aggregates.  This removes duplicated prompt metadata only; no selected
    scope row or citation ID is removed here.
    """

    return {
        "question": question,
        "analysis_briefs": _compact_specialist_raw_result(raw_result, agent),
        "requirements": _compact_specialist_requirements(requirements),
        # Scope-level provider gaps remain represented in the digest.  These
        # are only run-level notes, so they cannot displace selected scopes.
        "known_limitations": [str(item)[:100] for item in limitations[:2]],
    }


def _compact_specialist_raw_result(
    raw_result: Mapping[str, object],
    agent: AgentId,
) -> dict[str, object]:
    """Keep specialist scope metadata without duplicating provider summaries."""

    status = raw_result.get("collection_status")
    values = status if isinstance(status, Mapping) else {}
    raw_assets = values.get("requested_assets", [])
    assets = [
        {"symbol": str(item.get("symbol", "")).strip()}
        for item in raw_assets
        if isinstance(item, Mapping) and str(item.get("symbol", "")).strip()
    ]
    raw_capabilities = values.get("requested_capabilities", [])
    capabilities = [
        str(item) for item in raw_capabilities if isinstance(item, str) and item.strip()
    ]
    compact: dict[str, object] = {
        "specialist": agent,
        "collection_status": {
            "requested_assets": assets,
            "requested_capabilities": capabilities,
        },
    }
    analyzer = analyzer_for(agent)
    if analyzer.compact_briefs is not None:
        compact.update(analyzer.compact_briefs(raw_result))
    return compact


def _compact_specialist_requirements(
    requirements: AnswerRequirements,
) -> dict[str, object]:
    """Retain answer constraints without repeating every accepted alias."""

    return {
        "assets_in_required_order": [item.label for item in requirements.assets],
        "scopes": list(requirements.scopes),
        "comparison_required": requirements.comparison,
        "style": "One short natural paragraph per asset; qualitative interpretation only.",
    }


SpecialistAgentId = Literal["market_agent", "news_agent", "fundamentals_agent", "onchain_agent"]

_SPECIALIST_LLM_LOCK = threading.Lock()
_SPECIALIST_COOLDOWN_UNTIL = 0.0
_MAX_SPECIALIST_RETRY_AFTER_SECONDS = 30.0
_NEXT_SPECIALIST_TOKEN_RESERVE = 2_300


class SpecialistAnalysisRunner:
    """Create at most one grounded answer for each selected specialist.

    ``analyze_agent_result`` remains the common safety boundary.  Repair calls
    are explicitly disabled: an invalid live response is deterministically
    salvaged or replaced with a fact-first answer from the exact same evidence.
    """

    def __init__(
        self,
        *,
        llm: LLMAdapter | None = None,
        live_mode: bool = False,
    ) -> None:
        self._llm = llm or DisabledLLMAdapter()
        self._live_mode = live_mode

    def run(
        self,
        question: str,
        inputs: AnalysisInputs,
        *,
        agent: SpecialistAgentId,
        capabilities: Sequence[ResearchCapability],
    ) -> AgentAnswer:
        selected = _ordered_capabilities(capabilities)
        spec = _specialist_evidence(inputs, agent=agent, capabilities=selected)
        return self._run_with_spec(
            question,
            inputs,
            agent=agent,
            capabilities=selected,
            spec=spec,
        )

    def _run_with_spec(
        self,
        question: str,
        inputs: AnalysisInputs,
        *,
        agent: AgentId,
        capabilities: Sequence[ResearchCapability],
        spec: EvidenceSpec,
    ) -> AgentAnswer:
        evidence_required = bool(capabilities)
        limitation_values = tuple(dict.fromkeys(spec.limitations))
        raw_result = _raw_result(inputs, agent=agent, capabilities=capabilities)
        analyzer = analyzer_for(agent)
        asset_count = len(inputs.assets)
        prompt_budget = analyzer.prompt_budget(asset_count=asset_count)
        maximum, detail_chars, bundle_budget = analyzer.evidence_limits(asset_count)
        prompt_bundle = build_specialist_analysis_payload(
            raw_result,
            cast("Mapping[object, object]", spec.available_evidence),
            spec.analysis_data_digest,
            budget_bytes=prompt_budget,
            maximum=maximum,
            detail_char_limit=detail_chars,
            bundle_budget_bytes=bundle_budget,
        )
        try:
            return self._analyze_with_live_gate(
                agent=agent,
                call=lambda: analyze_agent_result(
                    self._llm,
                    agent=agent,
                    question=question,
                    raw_result=raw_result,
                    evidence=spec.available_evidence,
                    complete_data_digest=spec.analysis_data_digest,
                    limitations=limitation_values,
                    evidence_required=evidence_required,
                    coverage_requirements=tuple(capability.value for capability in capabilities),
                    no_evidence_message=no_evidence_message(inputs, agent=agent),
                    enforce_live_requirements=self._live_mode,
                    prompt_bundle=prompt_bundle,
                    payload_budget_bytes=prompt_budget,
                ),
            )
        except (AgentAnalysisError, LLMError, ValidationError) as exc:
            if spec.available_evidence:
                logger.warning(
                    "specialist_analysis agent=%s status=partial category=%s",
                    agent,
                    live_failure_category(exc),
                )
                return evidence_first_answer(
                    agent=agent,
                    question=question,
                    message=(
                        public_failure_reason(exc)
                        if isinstance(exc, LLMError)
                        else "A full specialist interpretation is temporarily unavailable."
                    ),
                    limitations=limitation_values,
                    evidence=spec.available_evidence,
                )
            logger.warning(
                "specialist_analysis agent=%s status=unavailable category=%s",
                agent,
                live_failure_category(exc),
            )
            return live_unavailable_answer(
                agent=agent,
                message=(
                    public_failure_reason(exc)
                    if isinstance(exc, LLMError)
                    else "No verified data were returned for this selected research scope."
                ),
                limitations=limitation_values,
                evidence_collected=False,
            )

    def _analyze_with_live_gate(
        self,
        *,
        agent: AgentId,
        call: Callable[[], AgentAnswer],
    ) -> AgentAnswer:
        """Serialize live specialist calls and honor provider retry-after hints."""

        if not self._live_mode:
            return call()
        with _SPECIALIST_LLM_LOCK:
            _wait_for_specialist_cooldown(agent)
            try:
                answer = call()
                _record_specialist_cooldown(self._llm, agent=agent)
                return answer
            except Exception:
                _record_specialist_cooldown(self._llm, agent=agent)
                raise


def _wait_for_specialist_cooldown(agent: AgentId) -> None:
    delay = _SPECIALIST_COOLDOWN_UNTIL - time.monotonic()
    if delay <= 0:
        return
    sleep_for = min(delay, _MAX_SPECIALIST_RETRY_AFTER_SECONDS)
    logger.info(
        "specialist_analysis agent=%s status=waiting retry_after_seconds=%.3f",
        agent,
        sleep_for,
    )
    time.sleep(sleep_for)


def _record_specialist_cooldown(llm: LLMAdapter, *, agent: AgentId) -> None:
    global _SPECIALIST_COOLDOWN_UNTIL
    telemetry = getattr(llm, "last_call_telemetry", ())
    values = telemetry if isinstance(telemetry, tuple) else (telemetry,)
    retry_after = max(
        (
            float(getattr(item, "retry_after_seconds", 0.0) or 0.0)
            for item in values
            if getattr(item, "status_category", "") == "rate_limited"
        ),
        default=0.0,
    )
    token_reset = max(
        (_low_token_reset(item) for item in values),
        default=0.0,
    )
    retry_after = max(retry_after, token_reset)
    if retry_after <= 0:
        return
    retry_after = min(retry_after, _MAX_SPECIALIST_RETRY_AFTER_SECONDS)
    _SPECIALIST_COOLDOWN_UNTIL = max(_SPECIALIST_COOLDOWN_UNTIL, time.monotonic() + retry_after)
    logger.warning(
        "specialist_analysis agent=%s status=rate_limit_cooldown retry_after_seconds=%.3f",
        agent,
        retry_after,
    )


def _low_token_reset(telemetry: object) -> float:
    remaining = getattr(telemetry, "remaining_tokens", None)
    if not isinstance(remaining, int) or remaining >= _NEXT_SPECIALIST_TOKEN_RESERVE:
        return 0.0
    return float(getattr(telemetry, "token_reset_seconds", 0.0) or 0.0)


def _specialist_evidence(
    inputs: AnalysisInputs,
    *,
    agent: SpecialistAgentId,
    capabilities: Sequence[ResearchCapability],
) -> EvidenceSpec:
    return evidence_services.build_specialist_evidence(
        inputs,
        agent=agent,
        capabilities=capabilities,
    )


def _ordered_capabilities(
    values: Sequence[ResearchCapability],
) -> tuple[ResearchCapability, ...]:
    selected = set(values)
    return tuple(capability for capability in ResearchCapability if capability in selected)


def _raw_result(
    inputs: AnalysisInputs,
    *,
    agent: AgentId,
    capabilities: Sequence[ResearchCapability],
) -> dict[str, object]:
    """Return bounded metadata prepared by the selected agent analyzer."""

    discovery = inputs.opportunity_result.candidates[:4] if inputs.opportunity_result else []
    result: dict[str, object] = {
        "collection_status": {
            "requested_capabilities": [item.value for item in capabilities],
            "requested_assets": [
                {
                    "requested_name": asset.requested_name,
                    "name": asset.name,
                    "symbol": asset.symbol,
                    "coin_id": asset.coin_id,
                }
                for asset in inputs.assets
            ]
            + [
                {"requested_name": candidate.asset, "symbol": candidate.symbol}
                for candidate in discovery
            ],
        },
        "specialist": agent,
    }
    summarize = analyzer_for(agent).summarize
    if summarize is not None:
        result.update(summarize(inputs))
    return result


__all__ = [
    "AgentAnalysisError",
    "SpecialistAgentId",
    "SpecialistAnalysisRunner",
    "analyze_agent_result",
    "validate_agent_answer",
]
