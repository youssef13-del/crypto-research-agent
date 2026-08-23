"""Provider-neutral LLM contracts and resilient execution helpers."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar, cast

import httpx
from pydantic import BaseModel, ValidationError

from crypto_research.domain.research import (
    UNAVAILABLE_ANSWER_MESSAGE,
    AgentAnswer,
)

logger = logging.getLogger(__name__)

OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class LLMCallTelemetry:
    """Sanitized metadata for one provider attempt."""

    model: str
    status_category: str
    request_id: str | None = None
    usage_tokens: int | None = None
    estimated_tokens: int | None = None
    retry_after_seconds: float = 0.0
    remaining_tokens: int | None = None
    token_reset_seconds: float = 0.0
    remaining_requests: int | None = None
    request_reset_seconds: float = 0.0


class LLMRole(StrEnum):
    MARKET = "market"
    RESEARCH = "research"
    FUNDAMENTALS = "fundamentals"
    FORECAST = "forecast"


class LLMError(RuntimeError):
    """Base error for controlled LLM failures."""


class LLMResponseError(LLMError):
    """Raised when the provider returns invalid structured output."""


class LLMPromptBudgetError(LLMError):
    """Raised before a request when the complete prompt exceeds its role budget."""


class LLMAdapter(Protocol):
    def generate_structured(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[OutputT],
    ) -> OutputT: ...


class SpecialistJSONAdapter(Protocol):
    def generate_specialist_json(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
    ) -> Mapping[str, object]: ...


def exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return tuple(chain)


def error_status_code(error: BaseException) -> int | None:
    for item in _failure_components(error):
        status = getattr(item, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(item, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
    return None


def provider_error_payloads(error: BaseException) -> tuple[Mapping[str, object], ...]:
    payloads: list[Mapping[str, object]] = []
    for item in _failure_components(error):
        body = getattr(item, "body", None)
        if not isinstance(body, Mapping):
            continue
        provider_error = body.get("error", body)
        if isinstance(provider_error, Mapping):
            payloads.append(provider_error)
    return tuple(payloads)


def is_provider_generation_failure(error: BaseException) -> bool:
    for payload in provider_error_payloads(error):
        if "failed_generation" in payload:
            return True
        message = str(payload.get("message", "")).casefold()
        if any(
            marker in message
            for marker in (
                "generated json",
                "invalid tool call",
                "tool call generated",
                "does not match the expected schema",
                "failed to validate json",
            )
        ):
            return True
    return False


def is_provider_schema_failure(error: BaseException) -> bool:
    """Identify request-schema defects without retaining provider response bodies."""

    for payload in provider_error_payloads(error):
        message = str(payload.get("message", "")).casefold()
        parameter = str(payload.get("param", "")).casefold()
        code = str(payload.get("code", "")).casefold()
        if parameter == "response_format" and "schema" in message:
            return True
        if "invalid json schema" in message or any(
            "discriminator_multiple_candidates" in value for value in (message, code)
        ):
            return True
    return False


class DisabledLLMAdapter:
    @property
    def last_call_used_fallback(self) -> bool:
        return True

    def generate_structured(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[OutputT],
    ) -> OutputT:
        del system_prompt
        if output_schema is AgentAnswer:
            return cast(OutputT, self._default_agent_answer(user_prompt))
        raise LLMResponseError(
            f"Disabled provider does not support structured schema {output_schema.__name__} "
            f"for role {role.value}."
        )

    def generate_specialist_json(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
    ) -> Mapping[str, object]:
        del role, system_prompt, user_prompt
        raise LLMResponseError("Disabled provider does not support specialist JSON generation.")

    @staticmethod
    def _default_agent_answer(user_prompt: str) -> AgentAnswer:
        try:
            payload = json.loads(user_prompt)
            agent = payload.get("agent", "market_agent")
            evidence = payload.get("available_evidence", {})
            question = str(payload.get("question", "the request")).strip()
            answer = (
                f"Offline mode could not produce a language-model analysis for: {question}."
                if not evidence
                else f"Offline mode found {len(evidence)} evidence items for this question, "
                "but deeper language-model analysis is unavailable."
            )
            return AgentAnswer(
                agent=agent,
                answer=answer,
                uncertainty=["The configured language-model provider is unavailable."],
                limitations=["Offline mode does not provide independent LLM analysis."],
                confidence=0.0,
                status="unavailable",
                analysis_state="unavailable",
                coverage_state="partial",
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):  # fmt: skip
            return AgentAnswer(
                agent="market_agent",
                answer="The configured language-model provider is unavailable.",
                uncertainty=["No language-model analysis was produced."],
                confidence=0.0,
                status="unavailable",
                analysis_state="unavailable",
                coverage_state="partial",
            )


class ResilientLLMAdapter:
    def __init__(
        self,
        primary: LLMAdapter,
        *,
        deterministic_fallback: bool = True,
    ) -> None:
        self._primary = primary
        self._fallback = DisabledLLMAdapter()
        self._deterministic_fallback = deterministic_fallback
        self._last_call_used_fallback = False
        self._last_failure_reason: str | None = None

    @property
    def last_call_used_fallback(self) -> bool:
        return self._last_call_used_fallback

    @property
    def last_failure_reason(self) -> str | None:
        return self._last_failure_reason

    @property
    def last_call_telemetry(self) -> tuple[LLMCallTelemetry, ...]:
        return _adapter_telemetry(self._primary)

    def generate_structured(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[OutputT],
    ) -> OutputT:
        self._last_call_used_fallback = False
        self._last_failure_reason = None

        try:
            result = self._primary.generate_structured(
                role=role,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=output_schema,
            )
            self._last_failure_reason = None
            _log_sanitized_event(
                role=role,
                adapter=self._primary,
                status_fallback="not_used",
            )
            return result
        except (ValidationError, TypeError, ValueError) as exc:
            # Validation errors are controlled LLM failures - log them and use fallback
            self._last_failure_reason = public_failure_reason(exc)
            _log_sanitized_event(
                role=role,
                adapter=self._primary,
                status_fallback=_fallback_status(
                    adapter=self._primary,
                    deterministic_fallback=self._deterministic_fallback,
                ),
                error=exc,
            )
            if not self._deterministic_fallback:
                raise
            self._last_call_used_fallback = True
            return self._fallback.generate_structured(
                role=role,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=output_schema,
            )
        except Exception as exc:
            if not _is_controlled_llm_failure(exc):
                raise
            self._last_failure_reason = public_failure_reason(exc)
            _log_sanitized_event(
                role=role,
                adapter=self._primary,
                status_fallback=_fallback_status(
                    adapter=self._primary,
                    deterministic_fallback=self._deterministic_fallback,
                ),
                error=exc,
            )
            if not self._deterministic_fallback:
                raise

        self._last_call_used_fallback = True
        return self._fallback.generate_structured(
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
        )

    def generate_specialist_json(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
    ) -> Mapping[str, object]:
        self._last_call_used_fallback = False
        self._last_failure_reason = None
        primary = _specialist_json_adapter(self._primary)
        if primary is None:
            raise LLMResponseError("Primary adapter does not support specialist JSON generation.")
        try:
            result = primary.generate_specialist_json(
                role=role,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            self._last_failure_reason = None
            _log_sanitized_event(
                role=role,
                adapter=self._primary,
                status_fallback="not_used",
            )
            return result
        except (ValidationError, TypeError, ValueError) as exc:
            self._last_failure_reason = public_failure_reason(exc)
            _log_sanitized_event(
                role=role,
                adapter=self._primary,
                status_fallback=_fallback_status(
                    adapter=self._primary,
                    deterministic_fallback=self._deterministic_fallback,
                ),
                error=exc,
            )
            raise LLMResponseError(
                f"Specialist JSON generation failed for role {role.value}."
            ) from exc
        except Exception as exc:
            if not _is_controlled_llm_failure(exc):
                raise
            self._last_failure_reason = public_failure_reason(exc)
            _log_sanitized_event(
                role=role,
                adapter=self._primary,
                status_fallback=_fallback_status(
                    adapter=self._primary,
                    deterministic_fallback=self._deterministic_fallback,
                ),
                error=exc,
            )
            raise LLMResponseError(
                f"Specialist JSON generation failed for role {role.value}."
            ) from exc


def _failure_components(error: BaseException) -> tuple[BaseException, ...]:
    return exception_chain(error)


def _is_controlled_llm_failure(error: Exception) -> bool:
    if isinstance(error, (LLMError, ValidationError)):
        return True
    if error_status_code(error) is not None:
        return True
    return any(
        isinstance(
            item,
            (ConnectionError, TimeoutError, httpx.NetworkError, httpx.TimeoutException),
        )
        or type(item).__name__
        in {
            "APIConnectionError",
            "APIResponseValidationError",
            "APITimeoutError",
            "ContentFilterFinishReasonError",
            "LengthFinishReasonError",
        }
        for item in exception_chain(error)
    )


def _adapter_telemetry(adapter: object | None) -> tuple[LLMCallTelemetry, ...]:
    if adapter is None:
        return ()
    value = getattr(adapter, "last_call_telemetry", ())
    if isinstance(value, LLMCallTelemetry):
        return (value,)
    if isinstance(value, tuple):
        return tuple(item for item in value if isinstance(item, LLMCallTelemetry))
    return ()


def _specialist_json_adapter(adapter: object) -> SpecialistJSONAdapter | None:
    method = getattr(adapter, "generate_specialist_json", None)
    return cast(SpecialistJSONAdapter, adapter) if callable(method) else None


def _fallback_status(
    *,
    adapter: object,
    deterministic_fallback: bool,
) -> str:
    if not deterministic_fallback:
        return "live_failed_unavailable"
    if len(_adapter_telemetry(adapter)) > 1:
        return "live_failed_then_deterministic"
    return "deterministic"


def _log_sanitized_event(
    *,
    role: LLMRole,
    adapter: object,
    status_fallback: str,
    error: Exception | None = None,
) -> None:
    attempts = _adapter_telemetry(adapter)
    models = ",".join(_log_field(item.model) for item in attempts) or "unavailable"
    request_ids = (
        ",".join(_log_field(item.request_id) for item in attempts if item.request_id) or "-"
    )
    usage = sum(item.usage_tokens or 0 for item in attempts)
    estimated = sum(item.estimated_tokens or 0 for item in attempts)
    retry_after = max((item.retry_after_seconds for item in attempts), default=0.0)
    status = (
        attempts[-1].status_category
        if attempts
        else _failure_status_category(error)
        if error is not None
        else "success"
    )
    log = logger.warning if error is not None else logger.info
    log(
        "llm_event role=%s model=%s status=%s request_id=%s usage_tokens=%s fallback=%s "
        "estimated_tokens=%s retry_after_seconds=%.3f",
        role.value,
        models,
        _log_field(status),
        request_ids,
        usage,
        status_fallback,
        estimated,
        retry_after,
    )


def _failure_status_category(error: Exception) -> str:
    if any(isinstance(item, LLMPromptBudgetError) for item in _failure_components(error)):
        return "prompt_budget"
    status = error_status_code(error)
    if status == 429:
        return "rate_limited"
    if status in {408, 498}:
        return "provider_capacity"
    if status is not None and status >= 500:
        return "provider_unavailable"
    if status == 400 and is_provider_schema_failure(error):
        return "schema_invalid"
    if status is not None:
        return f"http_{status}"
    if any(isinstance(item, ValidationError) for item in _failure_components(error)):
        return "invalid_output"
    return type(error).__name__


def _log_field(value: object) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._:/,-]+", "_", str(value))
    return normalized[:200] or "-"


def live_failure_category(error: Exception) -> str:
    """Classify a live synthesis failure into a sanitized log category."""

    if not isinstance(error, Exception):
        return type(error).__name__
    components = _failure_components(error)
    if any(isinstance(item, LLMPromptBudgetError) for item in components):
        return "prompt_budget"
    status = error_status_code(error)
    if status == 429:
        return "rate_limited"
    if status == 400 and is_provider_schema_failure(error):
        return "schema_invalid"
    if status == 400 and is_provider_generation_failure(error):
        return "invalid_generation"
    if status == 400:
        return "provider_rejected_request"
    if status in {408, 498} or status is not None and status >= 500:
        return "provider_unavailable"
    if any(isinstance(item, ValidationError) for item in components):
        return "answer_validation:invalid_output"
    message = str(error).casefold()
    if "strict specialist output" in message or "failed to parse structured output" in message:
        return "answer_validation:invalid_output"
    if "local groq rate-limit wait" in message:
        return "rate_limited_local"
    categories = [
        category
        for marker, category in (
            ("omitted", "coverage"),
            ("unavailable metric", "missing_data"),
            ("answer contains a numeric value", "numeric_answer"),
            ("evidence claim contains a numeric value", "numeric_claim"),
            ("unknown evidence", "evidence_id"),
            ("different requested asset", "asset_ownership"),
            ("duplicate", "duplication"),
            ("safety", "safety"),
            ("unavailable answer", "model_unavailable"),
        )
        if marker in message
    ]
    return "answer_validation:" + ",".join(dict.fromkeys(categories or ["composition_failure"]))


def public_failure_reason(error: Exception) -> str:
    components = _failure_components(error)
    if any(isinstance(item, LLMPromptBudgetError) for item in components):
        return "The live LLM prompt could not fit within the safe request budget."
    status_code = error_status_code(error)
    direct_reason = _PUBLIC_STATUS_REASONS.get(status_code) if status_code is not None else None
    if direct_reason is not None:
        return direct_reason
    if status_code == 400 and is_provider_schema_failure(error):
        return "The configured structured-output schema was rejected by the live provider."
    if status_code == 400 and is_provider_generation_failure(error):
        return "The live LLM could not produce a valid structured response."
    if status_code in {400, 409, 422, 424}:
        return f"The live Groq provider rejected the structured request (HTTP {status_code})."
    if status_code == 408:
        return "The live Groq provider timed out (HTTP 408)."
    if status_code is not None and status_code >= 500:
        return "The live Groq provider is temporarily unavailable."
    if status_code == 498:
        return "The live Groq provider has no capacity for this request (HTTP 498)."
    if status_code == 499:
        return "The live LLM request was cancelled (HTTP 499)."
    chain = components
    if any(
        isinstance(
            item,
            (ConnectionError, TimeoutError, httpx.NetworkError, httpx.TimeoutException),
        )
        or type(item).__name__ in {"APIConnectionError", "APITimeoutError"}
        for item in chain
    ):
        return "The live Groq provider could not be reached or timed out."
    if any(type(item).__name__ == "ValidationError" for item in chain):
        return "The live LLM returned an invalid structured response."
    return f"{UNAVAILABLE_ANSWER_MESSAGE} Please try again shortly."


_PUBLIC_STATUS_REASONS = {
    401: "The live Groq provider rejected the credentials (HTTP 401).",
    403: "The configured Groq model is not permitted for this account (HTTP 403).",
    404: "The configured Groq model or endpoint was not found (HTTP 404).",
    408: "The live Groq provider timed out (HTTP 408).",
    413: "The live LLM request exceeded the provider payload limit (HTTP 413).",
    429: "The live Groq provider rate limit was reached (HTTP 429).",
    498: "The live Groq provider has no capacity for this request (HTTP 498).",
    499: "The live LLM request was cancelled (HTTP 499).",
}


__all__ = [
    "DisabledLLMAdapter",
    "LLMAdapter",
    "LLMCallTelemetry",
    "LLMError",
    "LLMPromptBudgetError",
    "LLMResponseError",
    "LLMRole",
    "OutputT",
    "ResilientLLMAdapter",
    "SpecialistJSONAdapter",
    "error_status_code",
    "exception_chain",
    "is_provider_schema_failure",
    "live_failure_category",
    "public_failure_reason",
]
