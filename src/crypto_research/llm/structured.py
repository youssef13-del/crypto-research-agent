"""Minimal provider wire contracts and deterministic public-model normalization."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from math import isfinite
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from crypto_research.domain.research import (
    AgentAnswer,
    AgentId,
    EvidenceClaim,
)
from crypto_research.llm.client import LLMRole, OutputT

_TECHNICAL_GLOSSARY = (
    "OHLCV",
    "RSI",
    "MACD",
    "ATR",
    "SMA",
    "EMA",
    "TVL",
    "DeFi",
    "market capitalization",
    "circulating supply",
    "volatility",
    "support",
    "resistance",
    "funding rate",
    "open interest",
    "liquidity",
    "governance",
    "smart contract",
    "proof of stake",
    "proof of work",
)


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _GeneratedEvidenceClaim(_WireModel):
    statement: str
    evidence_ids: list[str]
    claim_kind: Literal[
        "observed_fact",
        "calculation",
        "interpretation",
        "speculation",
        "risk",
    ]
    confidence: float


class _GeneratedAgentAnswer(_WireModel):
    answer: str
    analysis: str = ""
    evidence: list[_GeneratedEvidenceClaim]
    uncertainty: list[str]
    confidence: float
    suggested_followups: list[str] = []


def wire_schema(output_schema: type[OutputT]) -> type[BaseModel]:
    if output_schema is AgentAnswer:
        return _GeneratedAgentAnswer
    return output_schema


def unwrap_structured_result(value: object) -> tuple[object, object | None]:
    if not isinstance(value, Mapping) or not {"raw", "parsed", "parsing_error"} <= set(value):
        return value, None
    parsing_error = value.get("parsing_error")
    if isinstance(parsing_error, BaseException):
        raise parsing_error
    parsed = value.get("parsed")
    if parsed is None:
        raise ValueError("The provider returned no parsed structured output.")
    return parsed, value.get("raw")


def convert_output(
    value: object,
    *,
    output_schema: type[OutputT],
    role: LLMRole,
    user_prompt: str,
) -> OutputT:
    if output_schema is AgentAnswer:
        return cast(OutputT, _convert_agent_answer(value, role=role, user_prompt=user_prompt))
    try:
        return output_schema.model_validate(value)
    except (ValidationError, TypeError, ValueError) as exc:
        # If validation fails, try to recover by extracting the data from the raw value
        if isinstance(value, dict):
            # Try to find the actual data in the response
            if "parsed" in value:
                try:
                    return output_schema.model_validate(value["parsed"])
                except ValidationError, TypeError, ValueError:
                    pass
            if "raw" in value and isinstance(value["raw"], dict):
                # Try to extract from raw response
                raw_data = value["raw"]
                if hasattr(raw_data, "content"):
                    try:
                        import json

                        parsed = json.loads(raw_data.content)
                        return output_schema.model_validate(parsed)
                    except json.JSONDecodeError, TypeError, ValueError:
                        pass
        raise LLMResponseError(
            f"Failed to parse structured output for {output_schema.__name__}"
        ) from exc


def _convert_agent_answer(
    value: object,
    *,
    role: LLMRole,
    user_prompt: str,
) -> AgentAnswer:
    try:
        generated = _GeneratedAgentAnswer.model_validate(_select_agent_fields(value))
    except ValidationError as exc:
        # If the response doesn't match the expected schema, try to extract
        # the answer from the raw response
        if isinstance(value, dict):
            # Try to find answer in various fields
            for field in ["answer", "content", "text", "message"]:
                if field in value and value[field]:
                    answer_text = str(value[field])[:1200]
                    if answer_text.strip():
                        return AgentAnswer(
                            agent=_agent_id(value.get("agent"), role=role),
                            answer=answer_text,
                            analysis="",
                            evidence=[],
                            uncertainty=["The response structure was unexpected."],
                            limitations=["The LLM response had an unexpected format."],
                            confidence=0.3,
                            status="partial",
                        )
            # Also try to extract from nested structures
            if "response" in value and isinstance(value["response"], dict):
                for field in ["answer", "content", "text", "message"]:
                    if field in value["response"] and value["response"][field]:
                        answer_text = str(value["response"][field])[:1200]
                        if answer_text.strip():
                            return AgentAnswer(
                                agent=_agent_id(value.get("agent"), role=role),
                                answer=answer_text,
                                analysis="",
                                evidence=[],
                                uncertainty=["The response structure was unexpected."],
                                limitations=["The LLM response had an unexpected format."],
                                confidence=0.3,
                                status="partial",
                            )
        # If we can't recover, raise the original error
        raise LLMResponseError(f"Failed to parse agent answer: {exc}") from exc

    prompt = _json_mapping(user_prompt)
    answer = generated.answer.strip()[:1200]
    if not answer:
        raise ValueError("The generated answer is empty.")

    evidence = prompt.get("available_evidence")
    evidence_index = prompt.get("evidence_index")
    allowed_ids = {
        str(item)
        for item in (
            evidence_index
            if isinstance(evidence_index, list)
            else list(evidence)
            if isinstance(evidence, Mapping)
            else []
        )
    }
    claims: list[EvidenceClaim] = []
    for item in generated.evidence:
        statement = item.statement.strip()[:1200]
        evidence_ids = list(
            dict.fromkeys(
                evidence_id.strip()
                for evidence_id in item.evidence_ids
                if evidence_id.strip() in allowed_ids
            )
        )[:5]
        if not statement or not evidence_ids:
            continue
        claims.append(
            EvidenceClaim(
                statement=statement,
                evidence_ids=evidence_ids,
                claim_kind=item.claim_kind,
                confidence=_confidence(item.confidence),
            )
        )
        if len(claims) == 6:
            break

    limitations = _unique_text(prompt.get("known_limitations"), limit=6)
    uncertainty = _unique_text(generated.uncertainty, limit=6)
    confidence = _confidence(generated.confidence)
    evidence_required = prompt.get("evidence_required") is True
    status: Literal["complete", "partial"] = (
        "partial"
        if limitations or confidence <= 0 or (evidence_required and (not allowed_ids or not claims))
        else "complete"
    )
    return AgentAnswer(
        agent=_agent_id(prompt.get("agent"), role=role),
        answer=answer,
        analysis=_optional_text(generated.analysis, limit=500) or "",
        technical_terms=_technical_terms(answer),
        evidence=claims,
        uncertainty=uncertainty,
        limitations=limitations,
        suggested_followups=_unique_text(generated.suggested_followups, limit=3),
        confidence=confidence,
        status=status,
    )


def _select_agent_fields(value: object) -> dict[str, object]:
    payload = _object_mapping(value)
    return {
        key: payload[key]
        for key in (
            "answer",
            "analysis",
            "evidence",
            "uncertainty",
            "confidence",
            "suggested_followups",
        )
        if key in payload
    }


def _object_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    raise ValueError("Structured provider output must be an object.")


def _json_mapping(value: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _confidence(value: float) -> float:
    if not isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _unique_text(value: object, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized = (str(item).strip() for item in value)
    return list(dict.fromkeys(item for item in normalized if item))[:limit]


def _optional_text(value: str | None, *, limit: int) -> str | None:
    normalized = " ".join(value.split())[:limit] if value else ""
    return normalized or None


def _agent_id(value: object, *, role: LLMRole) -> AgentId:
    supported: set[AgentId] = {
        "market_agent",
        "news_agent",
        "fundamentals_agent",
        "onchain_agent",
        "forecast_agent",
    }
    if isinstance(value, str) and value in supported:
        return value
    by_role: dict[LLMRole, AgentId] = {
        LLMRole.MARKET: "market_agent",
        LLMRole.RESEARCH: "news_agent",
        LLMRole.FUNDAMENTALS: "fundamentals_agent",
        LLMRole.FORECAST: "forecast_agent",
    }
    return by_role[role]


def _technical_terms(answer: str) -> list[str]:
    selected = [
        term
        for term in _TECHNICAL_GLOSSARY
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
            answer,
            flags=re.IGNORECASE,
        )
    ]
    return selected[:12]


def strict_json_schema(output_schema: type[BaseModel]) -> dict[str, Any]:
    """Build a small Groq strict-mode schema with required, closed objects."""

    schema = output_schema.model_json_schema()
    _normalize_strict_schema(schema)
    return {
        "name": output_schema.__name__.lstrip("_"),
        "parameters": schema,
    }


def _normalize_strict_schema(value: object) -> None:
    if isinstance(value, dict):
        value.pop("default", None)
        value.pop("title", None)
        if "const" in value:
            value["enum"] = [value.pop("const")]
        for unsupported in (
            "format",
            "maxItems",
            "maxLength",
            "maximum",
            "minItems",
            "minLength",
            "minimum",
            "pattern",
        ):
            value.pop(unsupported, None)
        for item in list(value.values()):
            _normalize_strict_schema(item)
        properties = value.get("properties")
        if isinstance(properties, dict):
            value["required"] = list(properties)
            value["additionalProperties"] = False
    elif isinstance(value, list):
        for item in value:
            _normalize_strict_schema(item)


# Add missing import
from crypto_research.llm.client import LLMResponseError  # noqa: E402
