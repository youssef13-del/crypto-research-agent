"""Deterministic Guided forecasts with an isolated explanatory LLM pass."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from crypto_research.agents.base import AgentManifest
from crypto_research.domain.evidence import (
    AgentAnalysisSection,
    AgentAnswer,
    StructuredAgentAnalysis,
)
from crypto_research.domain.forecast import (
    ForecastAgentResult,
    ForecastFailure,
    ForecastRequest,
    ForecastRun,
)
from crypto_research.domain.research import AnalysisRequest, ResearchCapability
from crypto_research.llm.client import (
    DisabledLLMAdapter,
    LLMAdapter,
    LLMCallTelemetry,
    LLMResponseError,
    live_failure_category,
)

from .forecast_analyzer import ROLE, SYSTEM_PROMPT, normalize_live_output
from .forecast_collector import build_requests

LOGGER = logging.getLogger(__name__)
FORECAST_MANIFEST = AgentManifest(
    id="forecast_agent",
    label="Forecasting Agent",
    capabilities=frozenset({ResearchCapability.FORECAST}),
)


class ForecastServiceProtocol(Protocol):
    def run(self, request: ForecastRequest) -> ForecastRun | ForecastFailure: ...


class ForecastAgent:
    """Own forecasting computation and keep its interpretation boundary isolated."""

    def __init__(self, *, service: ForecastServiceProtocol, llm: LLMAdapter | None = None) -> None:
        self._service = service
        self._llm = llm or DisabledLLMAdapter()

    def run(self, request: AnalysisRequest) -> ForecastAgentResult:
        settings, jobs = build_requests(request)
        with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
            results = list(pool.map(self._service.run, jobs))
        return ForecastAgentResult(settings=settings, asset_results=results)

    def analyze(self, question: str, result: ForecastAgentResult) -> AgentAnswer:
        del question  # No sibling-agent context crosses this boundary.
        fallback = self._evidence_only_answer(result)
        successful = [item for item in result.asset_results if isinstance(item, ForecastRun)]
        if not successful:
            return fallback
        payload = {
            "task": "Summarize the deterministic forecast batch without changing its numbers.",
            "output": {
                "summary": "one complete qualitative sentence",
                "limitations": ["up to three short strings"],
                "confidence": "number from 0 to 1",
            },
            "rules": [
                "Return one JSON object with exactly summary, limitations, and confidence.",
                "Return one concise overall summary, not per-asset sections.",
                "Explain what the predicted prices imply relative to current prices.",
                "Use qualitative wording because exact values are inserted locally by the app.",
                "Discuss validation and uncertainty in plain prose; never give trading advice.",
                "Use only the forecast diagnostics supplied below.",
            ],
            "forecasts": [self._brief(item) for item in result.asset_results],
        }
        try:
            generate_json = getattr(self._llm, "generate_specialist_json", None)
            if not callable(generate_json):
                raise LLMResponseError("Forecast JSON generation is not supported.")
            raw = generate_json(
                role=ROLE,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=json.dumps(payload, separators=(",", ":")),
            )
            live = normalize_live_output(raw)
            prediction_summary = self._prediction_summary(result)
            structured = StructuredAgentAnalysis(
                verdict=live.summary,
                sections=[],
                comparison="",
            )
            answer = AgentAnswer(
                agent="forecast_agent",
                answer=f"{prediction_summary}\n\n**Model view**\n\n{live.summary}",
                structured_analysis=structured,
                uncertainty=live.limitations,
                limitations=[
                    "LLM interpretation is explanatory only; displayed model values "
                    "are deterministic."
                ],
                confidence=live.confidence,
                status="complete",
                analysis_state="live",
                coverage_state=(
                    "complete" if len(successful) == len(result.asset_results) else "partial"
                ),
            )
            self._log_summary_event(status="live")
            return answer
        except Exception as exc:
            self._log_summary_event(status="partial", category=live_failure_category(exc))
            return fallback

    def _log_summary_event(self, *, status: str, category: str = "success") -> None:
        telemetry = _latest_telemetry(self._llm)
        LOGGER.info(
            "specialist_analysis agent=forecast_agent status=%s category=%s "
            "request_id=%s usage_tokens=%s estimated_tokens=%s retry_after_seconds=%.3f",
            status,
            category,
            getattr(telemetry, "request_id", None),
            getattr(telemetry, "usage_tokens", None),
            getattr(telemetry, "estimated_tokens", None),
            getattr(telemetry, "retry_after_seconds", 0.0),
        )

    @staticmethod
    def _brief(item: ForecastRun | ForecastFailure) -> dict[str, object]:
        if isinstance(item, ForecastFailure):
            return {"asset": item.request.symbol, "status": "unavailable", "reason": item.code}
        return {
            "asset": item.request.symbol,
            "status": "validated" if item.quality.passed else "failed_quality_gates",
            "current_price": _price_text(item.market.current_price),
            "predicted_price": _price_text(item.model_output.predicted_price),
            "predicted_return": round(item.model_output.predicted_return, 6),
            "target_time": item.model_output.timestamp.isoformat(),
            "prediction_interval": [
                _price_text(item.model_output.lower_interval),
                _price_text(item.model_output.upper_interval),
            ],
            "mae": round(item.metrics.mae, 6),
            "baseline_mae": round(item.metrics.baseline_mae, 6),
            "mae_improvement": round(item.metrics.mae_improvement, 4),
            "directional_accuracy": round(item.metrics.directional_accuracy, 4),
            "failed_gates": [
                reason for reason in item.quality.reasons if reason.startswith("failed")
            ],
        }

    def _evidence_only_answer(self, result: ForecastAgentResult) -> AgentAnswer:
        sections: list[AgentAnalysisSection] = []
        successful = 0
        for item in result.asset_results:
            if isinstance(item, ForecastFailure):
                text = f"The model could not run: {item.message}"
            else:
                successful += 1
                text = (
                    "The deterministic model passed its validation gates."
                    if item.quality.passed
                    else (
                        "The model produced an estimate, but it failed one or more "
                        "validation gates."
                    )
                )
            sections.append(
                AgentAnalysisSection(asset=item.request.symbol, scope="forecast", text=text)
            )
        structured = StructuredAgentAnalysis(
            verdict="Forecast estimates are shown with their validation status and uncertainty.",
            sections=sections,
            comparison=(
                "Compare model direction only alongside each asset's validation quality."
                if len(sections) > 1
                else ""
            ),
        )
        return AgentAnswer(
            agent="forecast_agent",
            answer=(
                f"{self._prediction_summary(result)}\n\n"
                "Live model context is currently unavailable."
            ),
            structured_analysis=structured,
            limitations=[
                "Live forecast interpretation was unavailable; computed results remain intact."
            ],
            confidence=0.45 if successful else 0.0,
            status="partial" if successful else "unavailable",
            analysis_state="evidence_only" if successful else "unavailable",
            coverage_state="complete" if successful == len(sections) else "partial",
        )

    @staticmethod
    def _prediction_summary(result: ForecastAgentResult) -> str:
        entries: list[str] = []
        for item in result.asset_results:
            if isinstance(item, ForecastFailure):
                entries.append(
                    f"- **{item.request.symbol}**: model output unavailable [unavailable]"
                )
                continue
            point = item.model_output
            quality = "validation passed" if item.quality.passed else "not trusted"
            entries.append(
                f"- **{item.request.symbol}**: {_price_text(item.market.current_price)} → "
                f"{_price_text(point.predicted_price)} by "
                f"{point.timestamp.strftime('%d %b %Y %H:%M UTC')} "
                f"({point.predicted_return:+.2%}) [{quality}]"
            )
        return "**Predicted prices**\n\n" + "\n".join(entries)


def _latest_telemetry(adapter: object) -> LLMCallTelemetry | None:
    value = getattr(adapter, "last_call_telemetry", None)
    if isinstance(value, LLMCallTelemetry):
        return value
    if isinstance(value, tuple):
        return next((item for item in reversed(value) if isinstance(item, LLMCallTelemetry)), None)
    return None


def _price_text(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000:
        return f"${value:,.2f}"
    if magnitude >= 1:
        return f"${value:,.4f}"
    return f"${value:,.6f}"
