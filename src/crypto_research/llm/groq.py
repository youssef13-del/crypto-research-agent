"""Direct Groq-compatible structured generation with bounded prompts."""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Mapping
from typing import Any, Protocol, TypedDict

import httpx
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from crypto_research.config import Settings
from crypto_research.llm.client import (
    LLMCallTelemetry,
    LLMPromptBudgetError,
    LLMResponseError,
    LLMRole,
    OutputT,
    error_status_code,
    exception_chain,
    is_provider_schema_failure,
)
from crypto_research.llm.structured import (
    convert_output,
    strict_json_schema,
    unwrap_structured_result,
    wire_schema,
)
from crypto_research.shared.text import estimate_tokens

_STRICT_OUTPUT_MODELS = frozenset(
    {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    }
)
_ROLE_MAX_INPUT_TOKENS = {
    LLMRole.MARKET: 1_700,
    LLMRole.RESEARCH: 1_700,
    LLMRole.FUNDAMENTALS: 1_700,
    LLMRole.FORECAST: 1_200,
}
_ROLE_MAX_OUTPUT_TOKENS = {
    LLMRole.MARKET: 600,
    LLMRole.RESEARCH: 600,
    LLMRole.FUNDAMENTALS: 600,
    LLMRole.FORECAST: 350,
}
_COMPLEX_ANALYSIS_MAX_OUTPUT_TOKENS = 1_000
_SPECIALIST_ROLES = frozenset(
    {LLMRole.MARKET, LLMRole.RESEARCH, LLMRole.FUNDAMENTALS, LLMRole.FORECAST}
)
_ROLE_TEMPERATURE = {
    LLMRole.MARKET: 0.3,
    LLMRole.RESEARCH: 0.1,
    LLMRole.FUNDAMENTALS: 0.1,
    LLMRole.FORECAST: 0.1,
}


def _request_options(role: LLMRole, output_tokens: int, model_name: str) -> dict[str, object]:
    options: dict[str, object] = {
        "max_tokens": output_tokens,
        "temperature": _ROLE_TEMPERATURE[role],
    }
    if "gpt-oss" in model_name.casefold():
        options["reasoning_effort"] = "low"
    return options


_PROVIDER_PACING_LOCK = threading.Lock()
_NEXT_REQUEST_AT: dict[str, float] = {}


class _RateLimitTelemetry(TypedDict):
    remaining_tokens: int | None
    token_reset_seconds: float
    remaining_requests: int | None
    request_reset_seconds: float


class StructuredClient(Protocol):
    def with_structured_output(
        self,
        schema: dict[str, Any] | type[Any],
        **_kwargs: Any,
    ) -> Any: ...


class GroqAdapter:
    def __init__(
        self,
        settings: Settings,
        *,
        model: str | None = None,
        client: StructuredClient | None = None,
    ) -> None:
        key = (
            settings.groq_api_key.get_secret_value() if settings.groq_api_key is not None else None
        )
        self._model_name = model or settings.groq_model
        if self._model_name not in _STRICT_OUTPUT_MODELS:
            supported = ", ".join(sorted(_STRICT_OUTPUT_MODELS))
            raise ValueError(
                f"Model {self._model_name!r} does not support this strict-output contract; "
                f"use one of: {supported}."
            )
        self._configured_max_tokens = settings.groq_max_tokens
        self._pace_live_requests = client is None
        self._minimum_request_interval = settings.groq_min_request_interval_seconds
        self._maximum_rate_wait = settings.groq_rate_limit_max_wait_seconds
        self._last_call_telemetry = LLMCallTelemetry(
            model=self._model_name,
            status_category="not_started",
        )
        self._client = client or ChatOpenAI(
            api_key=key,
            model=self._model_name,
            base_url=settings.groq_base_url,
            timeout=settings.groq_timeout_seconds,
            max_retries=settings.groq_max_retries,
            use_responses_api=False,
            include_response_headers=True,
        )

    @property
    def last_call_telemetry(self) -> LLMCallTelemetry:
        return self._last_call_telemetry

    def generate_structured(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[OutputT],
    ) -> OutputT:
        output_tokens = min(
            self._configured_max_tokens,
            _output_token_limit(role, user_prompt),
        )
        input_tokens = estimate_tokens(system_prompt, user_prompt)
        if input_tokens > _ROLE_MAX_INPUT_TOKENS[role]:
            self._last_call_telemetry = LLMCallTelemetry(
                model=self._model_name,
                status_category="prompt_budget",
            )
            raise LLMPromptBudgetError(
                f"The {role.value} prompt exceeds its safe provider input budget."
            )
        estimated_tokens = input_tokens + output_tokens
        if self._pace_live_requests:
            _pace_provider_request(
                self._model_name,
                minimum_interval=self._minimum_request_interval,
                maximum_wait=self._maximum_rate_wait,
            )

        provider_schema = strict_json_schema(wire_schema(output_schema))
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        options = _request_options(role, output_tokens, self._model_name)

        raw_message: object | None = None
        options_bound_to_client = False

        try:
            request_client: Any = self._client
            model_copy = getattr(request_client, "model_copy", None)
            if callable(model_copy):
                request_client = model_copy(update=options)
                options_bound_to_client = True
            else:
                options_bound_to_client = False

            # Try with strict schema first
            structured = request_client.with_structured_output(
                provider_schema,
                method="json_schema",
                strict=True,
                include_raw=True,
            )
            bind = getattr(structured, "bind", None)
            if not options_bound_to_client and callable(bind):
                structured = bind(**options)
            raw_result = structured.invoke(messages)
            if isinstance(raw_result, Mapping):
                raw_message = raw_result.get("raw")
            parsed, raw_message = unwrap_structured_result(raw_result)
            result = convert_output(
                parsed,
                output_schema=output_schema,
                role=role,
                user_prompt=user_prompt,
            )
        except (ValidationError, ValueError, TypeError) as strict_error:
            if role in _SPECIALIST_ROLES:
                actual_tokens = _usage_tokens(raw_message)
                self._last_call_telemetry = LLMCallTelemetry(
                    model=self._model_name,
                    status_category=_provider_failure_category(strict_error),
                    usage_tokens=actual_tokens,
                    estimated_tokens=estimated_tokens,
                )
                raise LLMResponseError(
                    f"Groq strict specialist output failed for role {role.value}."
                ) from strict_error
            # If strict schema validation fails, try with a more lenient approach
            # This handles cases where the LLM returns valid JSON but with extra fields
            # or slightly different structure than the strict schema expects
            try:
                request_client = self._client
                model_copy = getattr(request_client, "model_copy", None)
                if callable(model_copy):
                    request_client = model_copy(update=options)
                # Use non-strict schema as fallback
                loose_structured = request_client.with_structured_output(
                    provider_schema,
                    method="json_schema",
                    strict=False,
                    include_raw=True,
                )
                bind = getattr(loose_structured, "bind", None)
                if not options_bound_to_client and callable(bind):
                    loose_structured = bind(**options)
                raw_result = loose_structured.invoke(messages)
                if isinstance(raw_result, Mapping):
                    raw_message = raw_result.get("raw")
                parsed, raw_message = unwrap_structured_result(raw_result)
                result = convert_output(
                    parsed,
                    output_schema=output_schema,
                    role=role,
                    user_prompt=user_prompt,
                )
            except Exception as fallback_exc:
                actual_tokens = _usage_tokens(raw_message)
                self._last_call_telemetry = LLMCallTelemetry(
                    model=self._model_name,
                    status_category=_provider_failure_category(fallback_exc),
                    usage_tokens=actual_tokens,
                    estimated_tokens=estimated_tokens,
                )
                raise LLMResponseError(f"Groq failed for role {role.value}.") from fallback_exc
        except Exception as exc:
            actual_tokens = _usage_tokens(raw_message)
            if role in _SPECIALIST_ROLES and isinstance(exc, LLMResponseError):
                self._last_call_telemetry = LLMCallTelemetry(
                    model=self._model_name,
                    status_category="invalid_generation",
                    usage_tokens=actual_tokens,
                    estimated_tokens=estimated_tokens,
                )
                raise LLMResponseError(
                    f"Groq strict specialist output failed for role {role.value}."
                ) from exc
            if not _is_expected_provider_failure(exc):
                self._last_call_telemetry = LLMCallTelemetry(
                    model=self._model_name,
                    status_category="internal_error",
                    usage_tokens=actual_tokens,
                    estimated_tokens=estimated_tokens,
                )
                raise
            self._last_call_telemetry = LLMCallTelemetry(
                model=self._model_name,
                status_category=_provider_failure_category(exc),
                request_id=_error_request_id(exc),
                usage_tokens=actual_tokens,
                estimated_tokens=estimated_tokens,
                retry_after_seconds=_retry_after_seconds(exc) or 0.0,
                **_rate_limit_telemetry(_error_response_headers(exc)),
            )
            raise LLMResponseError(f"Groq failed for role {role.value}.") from exc

        usage = _usage_tokens(raw_message)
        headers = _response_headers(raw_message)
        self._last_call_telemetry = LLMCallTelemetry(
            model=self._model_name,
            status_category="success",
            request_id=_response_request_id(raw_message, headers),
            usage_tokens=usage,
            estimated_tokens=estimated_tokens,
            **_rate_limit_telemetry(headers),
        )
        return result

    def generate_specialist_json(
        self,
        *,
        role: LLMRole,
        system_prompt: str,
        user_prompt: str,
    ) -> Mapping[str, object]:
        """Generate a small JSON object without provider-side schema validation.

        Guided specialist cards use this path because Groq's strict structured
        response format can reject otherwise valid requests before the model
        has a chance to answer.  We still budget, log, parse, and validate
        locally; only the brittle provider schema layer is skipped.
        """

        output_tokens = min(
            self._configured_max_tokens,
            _output_token_limit(role, user_prompt),
        )
        input_tokens = estimate_tokens(system_prompt, user_prompt)
        if input_tokens > _ROLE_MAX_INPUT_TOKENS[role]:
            self._last_call_telemetry = LLMCallTelemetry(
                model=self._model_name,
                status_category="prompt_budget",
            )
            raise LLMPromptBudgetError(
                f"The {role.value} prompt exceeds its safe provider input budget."
            )
        estimated_tokens = input_tokens + output_tokens
        if self._pace_live_requests:
            _pace_provider_request(
                self._model_name,
                minimum_interval=self._minimum_request_interval,
                maximum_wait=self._maximum_rate_wait,
            )
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        options: dict[str, object] = {
            "max_tokens": output_tokens,
            "temperature": _ROLE_TEMPERATURE[role],
        }
        response_options: dict[str, object] = {"response_format": {"type": "json_object"}}
        if "gpt-oss" in self._model_name.casefold():
            options["reasoning_effort"] = "low"

        raw_message: object | None = None
        try:
            request_client: Any = self._client
            model_copy = getattr(request_client, "model_copy", None)
            if callable(model_copy):
                request_client = model_copy(update=options)
            else:
                bind = getattr(request_client, "bind", None)
                if callable(bind):
                    request_client = bind(**options)
            bind = getattr(request_client, "bind", None)
            if callable(bind):
                request_client = bind(**response_options)
            invoke = getattr(request_client, "invoke", None)
            if not callable(invoke):
                raise TypeError("The configured LLM client does not support direct invocation.")
            raw_message = invoke(messages)
            result = _json_object_from_message(raw_message)
        except Exception as exc:
            actual_tokens = _usage_tokens(raw_message)
            if not _is_expected_provider_failure(exc):
                self._last_call_telemetry = LLMCallTelemetry(
                    model=self._model_name,
                    status_category="internal_error",
                    usage_tokens=actual_tokens,
                    estimated_tokens=estimated_tokens,
                )
                raise
            self._last_call_telemetry = LLMCallTelemetry(
                model=self._model_name,
                status_category=_provider_failure_category(exc),
                request_id=_error_request_id(exc),
                usage_tokens=actual_tokens,
                estimated_tokens=estimated_tokens,
                retry_after_seconds=_retry_after_seconds(exc) or 0.0,
                **_rate_limit_telemetry(_error_response_headers(exc)),
            )
            raise LLMResponseError(f"Groq JSON generation failed for role {role.value}.") from exc

        usage = _usage_tokens(raw_message)
        headers = _response_headers(raw_message)
        self._last_call_telemetry = LLMCallTelemetry(
            model=self._model_name,
            status_category="success",
            request_id=_response_request_id(raw_message, headers),
            usage_tokens=usage,
            estimated_tokens=estimated_tokens,
            **_rate_limit_telemetry(headers),
        )
        return result


def _output_token_limit(role: LLMRole, user_prompt: str) -> int:
    base = _ROLE_MAX_OUTPUT_TOKENS[role]
    if role in _SPECIALIST_ROLES:
        return base
    return _COMPLEX_ANALYSIS_MAX_OUTPUT_TOKENS if _is_complex_prompt(user_prompt) else base


def _is_complex_prompt(user_prompt: str) -> bool:
    try:
        payload = json.loads(user_prompt)
    except json.JSONDecodeError, TypeError:
        return False
    if not isinstance(payload, Mapping):
        return False
    requirements = payload.get("answer_requirements")
    return (
        isinstance(requirements, Mapping) and requirements.get("complex") is True
    ) or "repair_request" in payload


def _json_object_from_message(raw_message: object) -> Mapping[str, object]:
    content = getattr(raw_message, "content", raw_message)
    if isinstance(content, Mapping):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text") or item.get("content")
                if text is not None:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        content = "\n".join(parts)
    text = str(content).strip()
    if text.startswith("```"):
        text = _strip_json_fence(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMResponseError("The specialist JSON response was not valid JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise LLMResponseError("The specialist JSON response was not an object.")
    return {str(key): value for key, value in parsed.items()}


def _strip_json_fence(value: str) -> str:
    lines = value.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _usage_tokens(raw_message: object | None) -> int | None:
    usage = getattr(raw_message, "usage_metadata", None)
    if isinstance(usage, Mapping):
        for key in ("total_tokens", "total_token_count"):
            value = usage.get(key)
            if isinstance(value, int):
                return value
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            return input_tokens + output_tokens
    metadata = getattr(raw_message, "response_metadata", None)
    token_usage = metadata.get("token_usage") if isinstance(metadata, Mapping) else None
    if isinstance(token_usage, Mapping):
        total = token_usage.get("total_tokens")
        if isinstance(total, int):
            return total
    return None


def _response_headers(raw_message: object | None) -> Mapping[str, object]:
    metadata = getattr(raw_message, "response_metadata", None)
    if not isinstance(metadata, Mapping):
        return {}
    headers = metadata.get("headers")
    return headers if isinstance(headers, Mapping) else {}


def _error_response_headers(error: Exception) -> Mapping[str, object]:
    for item in exception_chain(error):
        response = getattr(item, "response", None)
        headers = getattr(response, "headers", None)
        if isinstance(headers, Mapping):
            return headers
    return {}


def _rate_limit_telemetry(headers: Mapping[str, object]) -> _RateLimitTelemetry:
    return {
        "remaining_tokens": _header_int(headers, "x-ratelimit-remaining-tokens"),
        "token_reset_seconds": _duration_seconds(
            _header_value(headers, "x-ratelimit-reset-tokens")
        ),
        "remaining_requests": _header_int(headers, "x-ratelimit-remaining-requests"),
        "request_reset_seconds": _duration_seconds(
            _header_value(headers, "x-ratelimit-reset-requests")
        ),
    }


def _header_int(headers: Mapping[str, object], name: str) -> int | None:
    value = _header_value(headers, name)
    if value is None:
        return None
    try:
        return max(0, int(float(str(value))))
    except ValueError:
        return None


def _duration_seconds(value: object | None) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().casefold()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    total = 0.0
    for number, unit in re.findall(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)", text):
        scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3_600.0}[unit]
        total += float(number) * scale
    return total


def _pace_provider_request(
    model: str,
    *,
    minimum_interval: float,
    maximum_wait: float,
) -> None:
    if minimum_interval <= 0:
        return
    with _PROVIDER_PACING_LOCK:
        now = time.monotonic()
        wait_for = max(0.0, _NEXT_REQUEST_AT.get(model, 0.0) - now)
        if wait_for > maximum_wait:
            raise LLMResponseError("The local Groq rate-limit wait exceeded its safe bound.")
        if wait_for:
            time.sleep(wait_for)
        _NEXT_REQUEST_AT[model] = time.monotonic() + minimum_interval


def _retry_after_seconds(error: Exception) -> float | None:
    for item in exception_chain(error):
        response = getattr(item, "response", None)
        headers = getattr(response, "headers", None)
        if not isinstance(headers, Mapping):
            continue
        value = _header_value(headers, "retry-after")
        if value is None:
            continue
        try:
            return max(0.0, float(str(value)))
        except ValueError:
            return None
    return None


def _response_request_id(
    raw_message: object | None,
    headers: Mapping[str, object],
) -> str | None:
    header_id = _header_value(headers, "x-request-id")
    if header_id is not None:
        return str(header_id)
    metadata = getattr(raw_message, "response_metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("request_id", "id"):
            value = metadata.get(key)
            if value is not None:
                return str(value)
    value = getattr(raw_message, "id", None)
    return str(value) if value is not None else None


def _error_request_id(error: Exception) -> str | None:
    for item in exception_chain(error):
        response = getattr(item, "response", None)
        headers = getattr(response, "headers", None)
        if not isinstance(headers, Mapping):
            continue
        value = _header_value(headers, "x-request-id")
        if value is not None:
            return str(value)
    return None


def _header_value(headers: Mapping[str, object], name: str) -> object | None:
    for key, value in headers.items():
        if str(key).casefold() == name:
            return value
    return None


def _provider_failure_category(error: Exception) -> str:
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
    if any(
        isinstance(item, (ValidationError, TypeError, ValueError))
        for item in exception_chain(error)
    ):
        return "invalid_output"
    return "provider_error"


def _is_expected_provider_failure(error: Exception) -> bool:
    if isinstance(
        error,
        (
            ValidationError,
            TypeError,
            ValueError,
            LLMResponseError,
            httpx.HTTPError,
        ),
    ):
        return True
    if error_status_code(error) is not None:
        return True
    return any(
        isinstance(item, (ValidationError, TypeError, ValueError))
        or type(item).__name__
        in {
            "APIConnectionError",
            "APIResponseValidationError",
            "APITimeoutError",
            "ContentFilterFinishReasonError",
            "LengthFinishReasonError",
            "OutputParserException",
        }
        for item in exception_chain(error)
    )
