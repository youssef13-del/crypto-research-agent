"""Response schema and normalization for the Forecasting Agent."""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from crypto_research.agents.base import AgentAnalyzer, AgentEvidencePolicy
from crypto_research.llm.client import LLMRole
from crypto_research.shared.text import clean_generated_text

from .forecast_prompts import (
    SYSTEM_PROMPT,
    evidence_limits,
    output_contract,
    prompt_budget,
    structured_instruction,
)

ROLE = LLMRole.FORECAST
REQUIRED_SCOPES = ("forecast",)


class ForecastLiveOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=600)
    limitations: list[str] = Field(max_length=3)
    confidence: float = Field(ge=0, le=1)


def normalize_live_output(payload: object) -> ForecastLiveOutput:
    if not isinstance(payload, Mapping):
        raise ValueError("Forecast summary output must be a JSON object.")
    raw_summary = next(
        (
            value.strip()
            for key in ("summary", "answer", "verdict")
            if isinstance((value := payload.get(key)), str) and value.strip()
        ),
        "",
    )
    summary = clean_generated_text(
        raw_summary,
        max_chars=360,
        max_sentences=2,
        ensure_sentence=True,
    )
    if not summary:
        raise ValueError("Forecast summary output did not contain a usable summary.")
    raw_limitations = payload.get("limitations", [])
    limitations = (
        [
            clean_generated_text(
                item,
                max_chars=180,
                max_sentences=1,
                ensure_sentence=True,
            )
            for item in raw_limitations
            if isinstance(item, str) and item.strip()
        ]
        if isinstance(raw_limitations, list)
        else []
    )
    return ForecastLiveOutput(
        summary=summary,
        limitations=list(dict.fromkeys(item for item in limitations if item))[:3],
        confidence=_normalize_confidence(payload.get("confidence")),
    )


def _normalize_confidence(value: object) -> float:
    if isinstance(value, bool):
        return 0.5
    if isinstance(value, int | float):
        return min(1.0, max(0.0, float(value)))
    if isinstance(value, str):
        labels = {"low": 0.35, "medium": 0.6, "moderate": 0.6, "high": 0.8}
        normalized = value.strip().casefold()
        if normalized in labels:
            return labels[normalized]
        try:
            return min(1.0, max(0.0, float(normalized)))
        except ValueError:
            pass
    return 0.5


ANALYZER = AgentAnalyzer(
    id="forecast_agent",
    role=ROLE,
    system_prompt=SYSTEM_PROMPT,
    output_schema=ForecastLiveOutput,
    prompt_budget=prompt_budget,
    structured_instruction=structured_instruction,
    output_contract=output_contract,
    evidence_policy=AgentEvidencePolicy(
        allowed_kinds=frozenset(),
        limitations=lambda _inputs, _capabilities: [],
    ),
    evidence_limits=evidence_limits,
)

__all__ = [
    "ANALYZER",
    "ForecastLiveOutput",
    "REQUIRED_SCOPES",
    "ROLE",
    "SYSTEM_PROMPT",
    "normalize_live_output",
]
