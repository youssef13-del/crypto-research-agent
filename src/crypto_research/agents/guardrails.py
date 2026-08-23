"""Live-answer coverage requirements and evidence-grounding validation."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from crypto_research.agents.base import (
    AgentEvidencePolicy,
    NarrativeAssetAnalysis,
    NarrativeLiveOutput,
)
from crypto_research.domain.research import (
    UNAVAILABLE_ANSWER_MESSAGE,
    AgentAnalysisSection,
    AgentAnswer,
    AgentId,
    AnalysisInputs,
    EvidenceClaim,
    StructuredAgentAnalysis,
)
from crypto_research.shared.formatting import format_compact_number, format_money
from crypto_research.shared.numeric_grounding import (
    EVIDENCE_DROP_FIELDS,
    NumericFact,
    NumericToken,
    evidence_numeric_facts,
    numeric_fact_prompt_value,
    numeric_token_supported,
    numeric_token_value_matches,
    numeric_tokens,
    numeric_tokens_match,
)

_METRIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("current_price", re.compile(r"(?i)\b(?:current|latest)?\s*price\b|\btrading at\b")),
    ("change_24h", re.compile(r"(?i)\b(?:24\s*-?\s*hour|24h)\s+(?:change|return)\b")),
    ("support", re.compile(r"(?i)\bsupport\b")),
    ("resistance", re.compile(r"(?i)\bresistance\b")),
    ("rsi", re.compile(r"(?i)\brsi\b")),
    ("macd", re.compile(r"(?i)\bmacd\b")),
    ("volume", re.compile(r"(?i)\bvolume\b")),
    ("momentum", re.compile(r"(?i)\bmomentum\b|\bmarket structure\b")),
    ("news_sentiment", re.compile(r"(?i)\bsentiment\b")),
    ("catalysts", re.compile(r"(?i)\bcatalysts?\b")),
    ("market_cap", re.compile(r"(?i)\bmarket\s+cap(?:italization)?\b")),
    ("circulating_supply", re.compile(r"(?i)\bcirculating\s+supply\b")),
    ("total_supply", re.compile(r"(?i)\btotal\s+supply\b")),
    ("max_supply", re.compile(r"(?i)\bmax(?:imum)?\s+supply\b")),
    ("supply", re.compile(r"(?i)(?<!circulating\s)(?<!total\s)(?<!maximum\s)\bsupply\b")),
    ("tvl", re.compile(r"(?i)\btvl\b|\btotal value locked\b")),
    ("chains", re.compile(r"(?i)\b(?:supported\s+)?chains?\b|\bnetworks?\b")),
    ("funding_rate", re.compile(r"(?i)\bfunding\s+rate\b")),
    ("open_interest", re.compile(r"(?i)\bopen\s+interest\b")),
    ("liquidations", re.compile(r"(?i)\bliquidations?\b")),
    ("security_events", re.compile(r"(?i)\bexploits?\b|\baudits?\b|\bvulnerabilit")),
    ("risk", re.compile(r"(?i)\brisks?\b|\bsafe(?:ty)?\b")),
)
_METRIC_OUTPUT_TERMS: dict[str, tuple[str, ...]] = {
    "current_price": ("price", "trading", "$"),
    "change_24h": ("24h", "24-hour", "24 hour"),
    "support": ("support",),
    "resistance": ("resistance",),
    "rsi": ("rsi",),
    "macd": ("macd",),
    "volume": ("volume",),
    "momentum": ("momentum", "trend", "bullish", "bearish", "neutral"),
    "news_sentiment": ("sentiment", "tone", "bullish", "bearish", "mixed", "neutral"),
    "catalysts": ("catalyst", "driver", "development"),
    "market_cap": ("market cap", "market capitalization"),
    "circulating_supply": ("circulating supply",),
    "total_supply": ("total supply",),
    "max_supply": ("max supply", "maximum supply"),
    "supply": ("supply",),
    "tvl": ("tvl", "total value locked"),
    "chains": ("chain", "network"),
    "funding_rate": ("funding rate",),
    "open_interest": ("open interest",),
    "liquidations": ("liquidation",),
    "security_events": ("exploit", "audit", "security", "vulnerability"),
    "risk": ("risk", "safe", "safety"),
}
_METRIC_LABELS = {
    "current_price": "current price",
    "change_24h": "24-hour change",
    "news_sentiment": "news sentiment",
    "market_cap": "market capitalization",
    "circulating_supply": "circulating supply",
    "total_supply": "total supply",
    "max_supply": "maximum supply",
    "funding_rate": "funding rate",
    "open_interest": "open interest",
    "security_events": "security events",
}
_COMPARISON_LANGUAGE = re.compile(
    r"(?i)\b(?:compared|versus|vs\.?|higher|lower|stronger|weaker|more|less|while|whereas|"
    r"relative|respectively|outperform|underperform|both|same|similar|identical|equal|"
    r"different|differ|contrast)\b"
)
_FRESHNESS_LANGUAGE = re.compile(
    r"(?i)\b(?:as of|current|currently|latest|today|right now|recent|published|dated|"
    r"collection time|collected)\b"
)
_GUARANTEE_BOUNDARY = re.compile(
    r"(?i)\b(?:cannot|can't|can not|no one can)\b[^.!?]{0,50}\bguarantee\b|"
    r"\bnot guaranteed\b|\bno guarantee\b|\bnot risk[- ]free\b"
)
_PERSONAL_ADVICE_BOUNDARY = re.compile(
    r"(?i)\b(?:cannot|can't|can not|won't)\b[^.!?]{0,70}"
    r"\b(?:recommend(?:ed|ing)?|tell you|choose|allocate|invest|provide personalized)\b|"
    r"\bputting all (?:your|my) savings\b[^.!?]{0,40}\b(?:high|concentrated|serious) risk\b|"
    r"\b(?:allocating|putting) all (?:your|my)?\s*savings\b[^.!?]{0,120}"
    r"\bnot recommended\b"
)


@dataclass(frozen=True, slots=True)
class RequestedAssetRequirement:
    label: str
    aliases: tuple[str, ...]
    evidence_available: bool

    def prompt_value(self) -> dict[str, object]:
        return {
            "label": self.label,
            "accepted_names": list(self.aliases),
            "evidence_available": self.evidence_available,
        }


@dataclass(frozen=True, slots=True)
class AnswerRequirements:
    assets: tuple[RequestedAssetRequirement, ...]
    scopes: tuple[str, ...]
    requested_metrics: tuple[str, ...]
    available_metrics: tuple[str, ...]
    unavailable_metrics: tuple[str, ...]
    comparison: bool
    fresh: bool
    safety_boundaries: tuple[str, ...]
    complex: bool
    discovery: bool

    def prompt_value(self) -> dict[str, object]:
        return {
            "assets_in_required_order": [asset.prompt_value() for asset in self.assets],
            "scopes": list(self.scopes),
            "metrics_requested": list(self.requested_metrics),
            "metrics_with_evidence": list(self.available_metrics),
            "metrics_without_evidence": list(self.unavailable_metrics),
            "comparison_required": self.comparison,
            "freshness_required": self.fresh,
            "safety_boundaries": list(self.safety_boundaries),
            "complex": self.complex,
            "discovery_screen": self.discovery,
        }


@dataclass(frozen=True, slots=True)
class NumericGroundingIssue:
    code: str
    location: str
    token: NumericToken
    evidence_ids: tuple[str, ...]
    allowed_facts: tuple[NumericFact, ...]

    def prompt_value(self) -> dict[str, object]:
        matching_facts = [
            fact
            for fact in self.allowed_facts
            if fact.unit is self.token.unit or numeric_token_value_matches(self.token, fact)
        ]
        matching_facts.sort(key=lambda fact: _numeric_fact_repair_priority(fact, self.token))
        return {
            "code": self.code,
            "location": self.location,
            "unsupported_token": self.token.raw,
            "normalized_value": self.token.value,
            "unit": self.token.unit.value,
            "cited_evidence_ids": list(self.evidence_ids),
            "allowed_facts": [numeric_fact_prompt_value(fact) for fact in matching_facts[:4]],
        }


def _numeric_fact_repair_priority(
    fact: NumericFact,
    token: NumericToken,
) -> tuple[int, str, str]:
    path = fact.path.casefold()
    priority = (
        0 if numeric_token_value_matches(token, fact) else 1 if "current_price" in path else 2
    )
    return priority, fact.evidence_id, path


def compile_answer_requirements(
    *,
    question: str,
    raw_result: object,
    coverage_requirements: Sequence[str],
    evidence: Mapping[str, object],
) -> AnswerRequirements:
    assets: list[RequestedAssetRequirement] = []
    collection = raw_result.get("collection_status") if isinstance(raw_result, Mapping) else None
    requested_assets = (
        collection.get("requested_assets") if isinstance(collection, Mapping) else None
    )
    for raw_asset in requested_assets if isinstance(requested_assets, list) else []:
        if not isinstance(raw_asset, Mapping):
            continue
        values = [
            str(raw_asset.get(key, "")).strip()
            for key in ("symbol", "name", "requested_name", "coin_id")
        ]
        aliases = _asset_aliases(values)
        if not aliases:
            continue
        label = values[0] or aliases[0]
        assets.append(
            RequestedAssetRequirement(
                label=label,
                aliases=aliases,
                evidence_available=any(
                    _record_matches_asset(record, aliases) for record in evidence.values()
                ),
            )
        )

    metrics = tuple(metric for metric, pattern in _METRIC_PATTERNS if pattern.search(question))
    available_metrics = tuple(
        metric for metric in metrics if _metric_has_evidence(metric, evidence)
    )
    unavailable_metrics = tuple(metric for metric in metrics if metric not in available_metrics)
    lowered = question.casefold()
    safety: list[str] = []
    if any(
        phrase in lowered
        for phrase in ("guarantee", "cannot lose", "can't lose", "risk-free", "risk free")
    ):
        safety.append("reject_guarantee")
    if any(
        phrase in lowered
        for phrase in (
            "all my savings",
            "all your savings",
            "what to buy",
            "should i buy",
            "should i invest",
            "exactly what",
        )
    ):
        safety.append("reject_personalized_allocation")
    comparison = len(assets) >= 2 or bool(re.search(r"(?i)\b(?:compare|versus|vs\.?)\b", question))
    fresh = bool(
        re.search(r"(?i)\b(?:current|currently|latest|today|right now|recent)\b", question)
    )
    scopes = tuple(dict.fromkeys(coverage_requirements))
    complex_request = len(scopes) >= 3 or len(assets) >= 2 or len(metrics) >= 4
    return AnswerRequirements(
        assets=tuple(assets),
        scopes=scopes,
        requested_metrics=metrics,
        available_metrics=available_metrics,
        unavailable_metrics=unavailable_metrics,
        comparison=comparison,
        fresh=fresh,
        safety_boundaries=tuple(safety),
        complex=complex_request,
        discovery="discovery" in scopes,
    )


def requirement_limitations(requirements: AnswerRequirements) -> list[str]:
    limitations: list[str] = []
    if requirements.unavailable_metrics:
        labels = ", ".join(
            _METRIC_LABELS.get(metric, metric.replace("_", " "))
            for metric in requirements.unavailable_metrics
        )
        limitations.append(f"No verified current evidence was available for: {labels}.")
    missing_assets = [asset.label for asset in requirements.assets if not asset.evidence_available]
    if missing_assets:
        limitations.append(
            "No matching evidence was available for: " + ", ".join(missing_assets) + "."
        )
    return limitations


def answer_requirement_issues(
    answer: AgentAnswer,
    *,
    available_evidence: Mapping[str, object],
    requirements: AnswerRequirements,
) -> list[str]:
    text = " ".join(
        [
            answer.answer,
            answer.analysis,
            *(claim.statement for claim in answer.evidence),
            *answer.uncertainty,
            *answer.limitations,
            *answer.suggested_followups,
        ]
    )
    lowered = text.casefold()
    issues: list[str] = []
    issues.extend(
        f"analysis omitted requested asset {asset.label}"
        for asset in requirements.assets
        if asset.evidence_available
        and not any(_contains_alias(text, alias) for alias in asset.aliases)
    )
    issues.extend(
        "analysis omitted requested metric " + _METRIC_LABELS.get(metric, metric.replace("_", " "))
        for metric in requirements.available_metrics
        if not any(term in lowered for term in _METRIC_OUTPUT_TERMS[metric])
    )
    if requirements.comparison and len(requirements.assets) >= 2:
        available_assets = [asset for asset in requirements.assets if asset.evidence_available]
        if len(available_assets) >= 2 and not _COMPARISON_LANGUAGE.search(text):
            issues.append("comparison answer omitted a relative conclusion")
    if requirements.fresh and available_evidence and not _FRESHNESS_LANGUAGE.search(text):
        issues.append("fresh answer omitted its current-data context")
    if "reject_guarantee" in requirements.safety_boundaries and not _GUARANTEE_BOUNDARY.search(
        text
    ):
        issues.append("analysis omitted the required no-guarantee safety boundary")
    if (
        "reject_personalized_allocation" in requirements.safety_boundaries
        and not _PERSONAL_ADVICE_BOUNDARY.search(text)
    ):
        issues.append("analysis omitted the required personalized-advice boundary")
    issues.extend(_claim_asset_issues(answer, available_evidence, requirements.assets))
    normalized_claims = [" ".join(claim.statement.casefold().split()) for claim in answer.evidence]
    if len(normalized_claims) != len(set(normalized_claims)):
        issues.append("analysis contains duplicate evidence claims")
    if available_evidence:
        issues.extend(_numeric_grounding_issues(answer, available_evidence))
    return issues


_SALVAGEABLE_ISSUE_PREFIXES = (
    "analysis omitted available",
    "analysis omitted requested",
    "comparison answer omitted",
    "fresh answer omitted",
    "answer contains a numeric value",
    "evidence claim contains a numeric value",
)


def salvage_live_answer(
    answer: AgentAnswer,
    available_evidence: Mapping[str, object],
    issues: Sequence[str],
    requirements: AnswerRequirements,
) -> AgentAnswer | None:
    if not issues or not all(issue.startswith(_SALVAGEABLE_ISSUE_PREFIXES) for issue in issues):
        return None
    allowed_ids = set(available_evidence)
    claims = []
    seen: set[str] = set()
    for claim in answer.evidence:
        if not set(claim.evidence_ids) <= allowed_ids:
            continue
        cited = {evidence_id: available_evidence[evidence_id] for evidence_id in claim.evidence_ids}
        facts = evidence_numeric_facts(cited)
        if any(
            not numeric_token_supported(token, facts) for token in numeric_tokens(claim.statement)
        ):
            continue
        probe = AgentAnswer(
            agent=answer.agent,
            answer=claim.statement,
            evidence=[claim],
            confidence=answer.confidence,
            status="partial",
        )
        if _claim_asset_issues(probe, available_evidence, requirements.assets):
            continue
        normalized = " ".join(claim.statement.casefold().split())
        if normalized not in seen:
            seen.add(normalized)
            claims.append(claim)
    if not claims:
        return None
    prose = (
        claims[0].statement
        if any(issue.startswith("answer contains a numeric value") for issue in issues)
        else answer.answer
    )
    supplement = _omitted_asset_supplement(prose, available_evidence, requirements.assets)
    direct = f"{prose.rstrip()}\n\n{supplement}" if supplement else prose
    limitations = [
        "The live answer was limited to statements that passed evidence validation.",
        *answer.limitations,
        *(["Every requested asset with validated records appears above."] if supplement else []),
    ]
    salvaged = AgentAnswer(
        agent=answer.agent,
        answer=direct,
        technical_terms=[t for t in answer.technical_terms if t.casefold() in direct.casefold()],
        evidence=claims,
        uncertainty=answer.uncertainty,
        limitations=list(dict.fromkeys(limitations))[:6],
        confidence=answer.confidence,
        status="partial",
        analysis_state="live",
        coverage_state=answer.coverage_state,
    )
    grounded = salvaged.model_copy(update={"answer": prose})
    if _numeric_grounding_issues(grounded, available_evidence):
        return None
    return salvaged


def live_unavailable_answer(
    *,
    agent: AgentId,
    message: str,
    limitations: Sequence[str],
    evidence_collected: bool,
) -> AgentAnswer:
    answer = message.strip() or UNAVAILABLE_ANSWER_MESSAGE
    if evidence_collected and "evidence" not in answer.casefold():
        answer += " Collected evidence remains available below."
    return AgentAnswer(
        agent=agent,
        answer=answer,
        uncertainty=["No live model interpretation was produced."],
        limitations=list(dict.fromkeys(item for item in limitations if item.strip()))[:6],
        confidence=0.0,
        status="unavailable",
        analysis_state="unavailable",
        coverage_state="partial",
    )


def _asset_aliases(values: Sequence[str]) -> tuple[str, ...]:
    aliases: list[str] = []
    for value in values:
        normalized = " ".join(value.split()).strip()
        if not normalized:
            continue
        aliases.append(normalized)
        if "/" in normalized:
            aliases.append(normalized.split("/", maxsplit=1)[0])
    return tuple(dict.fromkeys(aliases))


def _record_matches_asset(record: object, aliases: Sequence[str]) -> bool:
    if not isinstance(record, Mapping):
        return False
    payload = record.get("payload")
    values = [record.get("asset")]
    if isinstance(payload, Mapping):
        values.extend(
            payload.get(key) for key in ("asset", "symbol", "name", "protocol", "slug", "assets")
        )
    text = " ".join(_flatten_text(value) for value in values)
    return any(_contains_alias(text, alias) for alias in aliases)


def _flatten_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value) if value is not None else ""


def _contains_alias(text: str, alias: str) -> bool:
    normalized = alias.strip()
    if not normalized:
        return False
    return (
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(normalized)}(?![A-Za-z0-9])",
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _omitted_asset_supplement(
    text: str,
    available_evidence: Mapping[str, object],
    assets: Sequence[RequestedAssetRequirement],
) -> str:
    statements: list[str] = []
    for asset in assets:
        if not asset.evidence_available:
            continue
        if any(_contains_alias(text, alias) for alias in asset.aliases):
            continue
        facts = deterministic_asset_statements(available_evidence, asset.label)
        if facts:
            statements.append(facts)
    return " ".join(statements)[:800]


def _metric_has_evidence(metric: str, evidence: Mapping[str, object]) -> bool:
    for value in evidence.values():
        if isinstance(value, Mapping) and _record_has_metric(metric, value):
            return True
    return False


def _record_has_metric(metric: str, record: Mapping[str, object]) -> bool:
    kind = str(record.get("claim_type", ""))
    payload = record.get("payload")
    data = payload if isinstance(payload, Mapping) else {}
    checks = {
        "market_snapshot": _market_metric_available,
        "technical_calculation": _technical_metric_available,
        "project_fundamentals": _fundamentals_metric_available,
        "defi_protocol_metrics": _defi_metric_available,
        "derivatives_positioning": _derivatives_metric_available,
    }
    check = checks.get(kind)
    if check is not None:
        return check(metric, data)
    return (
        kind == "recent_news"
        and metric in {"news_sentiment", "catalysts", "security_events"}
        or (kind == "deterministic_risk_assessment" and metric == "risk")
    )


def _market_metric_available(metric: str, payload: Mapping[str, object]) -> bool:
    snapshot = payload.get("snapshot")
    features = payload.get("ohlcv_features")
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    features = features if isinstance(features, Mapping) else {}
    return (
        (metric == "current_price" and _has_value(snapshot.get("current_price")))
        or (
            metric == "volume"
            and any(_has_value(features.get(key)) for key in ("base_volume", "average_volume"))
        )
        or (metric == "change_24h" and _available_return(features, "24h"))
    )


def _technical_metric_available(metric: str, payload: Mapping[str, object]) -> bool:
    return metric in {"support", "resistance", "rsi", "macd", "momentum"} and _has_value(
        payload.get("trend" if metric == "momentum" else metric)
    )


def _fundamentals_metric_available(metric: str, payload: Mapping[str, object]) -> bool:
    keys = {
        "market_cap": ("market_cap",),
        "circulating_supply": ("circulating_supply",),
        "total_supply": ("total_supply",),
        "max_supply": ("max_supply",),
        "supply": ("circulating_supply", "total_supply", "max_supply"),
    }.get(metric, ())
    return any(_has_value(payload.get(key)) for key in keys)


def _defi_metric_available(metric: str, payload: Mapping[str, object]) -> bool:
    return metric in {"tvl", "chains"} and _has_value(
        payload.get("tvl_usd" if metric == "tvl" else "chains")
    )


def _derivatives_metric_available(metric: str, payload: Mapping[str, object]) -> bool:
    return metric in {"funding_rate", "open_interest"} and _has_value(
        payload.get(
            "latest_funding_rate" if metric == "funding_rate" else "latest_open_interest_usd"
        )
    )


def _available_return(features: Mapping[object, object], label: str) -> bool:
    values = features.get("returns")
    if not isinstance(values, list):
        return False
    return any(
        isinstance(item, Mapping)
        and item.get("label") == label
        and item.get("status") == "available"
        and _has_value(item.get("return_percent"))
        for item in values
    )


def _has_value(value: object) -> bool:
    return value is not None and value != "" and value != []


def _claim_asset_issues(
    answer: AgentAnswer,
    available_evidence: Mapping[str, object],
    assets: Sequence[RequestedAssetRequirement],
) -> list[str]:
    issues: list[str] = []
    for claim in answer.evidence:
        mentioned = {
            index
            for index, asset in enumerate(assets)
            if any(_contains_alias(claim.statement, alias) for alias in asset.aliases)
        }
        cited = {
            index
            for evidence_id in claim.evidence_ids
            if evidence_id in available_evidence
            for index, asset in enumerate(assets)
            if _record_matches_asset(available_evidence[evidence_id], asset.aliases)
        }
        if mentioned and cited and not mentioned <= cited:
            issues.append("evidence claim cites a record owned by a different requested asset")
    return issues


def _numeric_grounding_issues(
    answer: AgentAnswer,
    available_evidence: Mapping[str, object],
) -> list[str]:
    codes = {issue.code for issue in _numeric_grounding_findings(answer, available_evidence)}
    messages: list[str] = []
    if "numeric_answer" in codes:
        messages.append("answer contains a numeric value not supported by a cited evidence claim")
    if "numeric_claim" in codes:
        messages.append("evidence claim contains a numeric value not supported by its records")
    return messages


def numeric_repair_context(
    answer: AgentAnswer,
    available_evidence: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return bounded, safe diagnostics for one strict live numeric repair."""

    return [
        issue.prompt_value()
        for issue in _numeric_grounding_findings(answer, available_evidence)[:12]
    ]


def _numeric_grounding_findings(
    answer: AgentAnswer,
    available_evidence: Mapping[str, object],
) -> list[NumericGroundingIssue]:
    findings: list[NumericGroundingIssue] = []
    supported_claim_tokens: list[NumericToken] = []
    all_facts = evidence_numeric_facts(available_evidence)
    for index, claim in enumerate(answer.evidence):
        cited = {
            evidence_id: available_evidence[evidence_id]
            for evidence_id in claim.evidence_ids
            if evidence_id in available_evidence
        }
        facts = evidence_numeric_facts(cited)
        for token in numeric_tokens(claim.statement):
            if numeric_token_supported(token, facts):
                supported_claim_tokens.append(token)
            else:
                findings.append(
                    NumericGroundingIssue(
                        code="numeric_claim",
                        location=f"evidence[{index}].statement",
                        token=token,
                        evidence_ids=tuple(claim.evidence_ids),
                        allowed_facts=facts,
                    )
                )
    for token in numeric_tokens(answer.answer):
        if any(numeric_tokens_match(token, supported) for supported in supported_claim_tokens):
            continue
        findings.append(
            NumericGroundingIssue(
                code="numeric_answer",
                location="answer",
                token=token,
                evidence_ids=(),
                allowed_facts=all_facts,
            )
        )
    return findings


_UNSAFE_GENERATED_LANGUAGE = re.compile(
    r"(?i)(?:\b(?:you|i)\s+(?:should|must|need to|ought to)\s+"
    r"(?:buy|sell|hold|short|invest|allocate|enter|exit)\b|"
    r"\b(?:i\s+)?(?:recommend|suggest)\s+(?:that\s+you\s+)?"
    r"(?:buy|buying|sell|selling|hold|holding|short|shorting|invest|allocate|enter|exit)\b|"
    r"\b(?:consider|try)\s+(?:buying|selling|holding|shorting|investing)\b|"
    r"\b(?:guaranteed|risk[- ]free|certain)\s+(?:profit|return|gain|win)\b|"
    r"\bguarantee(?:d|s)?\s+to\s+(?:rise|fall|increase|decrease)\b)"
)
_GENERATED_URL = re.compile(r"(?i)(?:https?://|javascript:|data:|file:)")
_WRITTEN_NUMBER = re.compile(
    r"(?i)\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|"
    r"trillion|percent)\b"
)
_UNSUPPORTED_NEWS_CAUSALITY = re.compile(
    r"(?i)(?:\b(?:caused|drove|triggered|sparked)\b.{0,45}"
    r"\b(?:price|rally|selloff|gain|drop|surge|decline)\b|"
    r"\bwill\b.{0,45}\b(?:price|rise|fall|rally|surge|decline)\b)"
)


def safe_generated_text(value: str) -> str:
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", value.strip()):
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        clean = " ".join(
            item
            for item in sentences
            if item
            and not _UNSAFE_GENERATED_LANGUAGE.search(item)
            and not _GENERATED_URL.search(item)
        )
        if clean:
            paragraphs.append(clean)
    return "\n\n".join(paragraphs)


def compact_prose(value: str, *, max_chars: int, max_sentences: int | None = None) -> str:
    clean = safe_generated_text(value)
    if not clean or max_chars <= 0:
        return ""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", clean.replace("\n\n", " "))
        if sentence.strip()
    ]
    if max_sentences is not None:
        sentences = sentences[:max_sentences]
    selected: list[str] = []
    for sentence in sentences:
        if len(" ".join([*selected, sentence])) <= max_chars:
            selected.append(sentence)
            continue
        if not selected:
            words: list[str] = []
            for word in sentence.rstrip(" .!?").split():
                if len(" ".join([*words, word]) + ".") > max_chars:
                    break
                words.append(word)
            if words:
                selected.append(" ".join(words).rstrip(",;:") + ".")
        break
    return " ".join(selected)


def join_paragraphs(values: Sequence[str], *, max_chars: int) -> str:
    paragraphs: list[str] = []
    for value in values:
        clean = value.strip()
        if not clean:
            continue
        if len("\n\n".join([*paragraphs, clean])) > max_chars:
            remaining = max_chars - len("\n\n".join(paragraphs)) - (2 if paragraphs else 0)
            clean = compact_prose(clean, max_chars=remaining)
            if not clean:
                break
        paragraphs.append(clean)
    return "\n\n".join(paragraphs)


def asset_key(value: str) -> str:
    return value.split("/", maxsplit=1)[0].strip().casefold()


def ordered_sections(
    values: Sequence[object],
    expected: Sequence[str],
    *,
    recovery_notes: list[str],
    label: str = "analysis",
) -> dict[str, object]:
    expected_keys = {asset_key(symbol): symbol for symbol in expected}
    sections: dict[str, object] = {}
    for value in values:
        key = asset_key(str(getattr(value, "symbol", "")))
        if key in expected_keys and key not in sections:
            sections[key] = value
    missing = [symbol for key, symbol in expected_keys.items() if key not in sections]
    if missing:
        recovery_notes.append(
            f"Live {label} was replaced with verified evidence for: {', '.join(missing)}."
        )
    return sections


def evidence_ids_for_asset(
    symbol: str,
    evidence: Mapping[str, object],
    *,
    kinds: set[str] | None = None,
) -> list[str]:
    key = asset_key(symbol)
    selected: list[str] = []
    for evidence_id, raw in evidence.items():
        record = raw if isinstance(raw, Mapping) else {}
        if kinds is not None and str(record.get("claim_type", "")) not in kinds:
            continue
        record_asset = asset_key(str(record.get("asset", "")))
        if record_asset == key or key in re.findall(r"[a-z0-9]+", evidence_id.casefold()):
            selected.append(evidence_id)
    return selected


def facts_for_asset(
    symbol: str,
    evidence: Mapping[str, object],
    *,
    kinds: Sequence[str],
    max_chars: int,
) -> str:
    statements: list[str] = []
    for kind in kinds:
        evidence_ids = evidence_ids_for_asset(symbol, evidence, kinds={kind})
        if kind == "recent_news":
            evidence_ids.sort(key=lambda item: news_evidence_rank(evidence[item]), reverse=True)
        statement = next(
            (
                rendered
                for evidence_id in evidence_ids
                if (rendered := evidence_statement(evidence[evidence_id])) is not None
            ),
            None,
        )
        if statement:
            statements.append(statement)
    clauses = [statement.rstrip(" .!?") for statement in dict.fromkeys(statements)]
    return compact_prose("; ".join(clauses) + "." if clauses else "", max_chars=max_chars)


def grounded_text(facts: str, interpretation: str, *, max_chars: int) -> str:
    return compact_prose(
        " ".join(value for value in (facts, interpretation) if value),
        max_chars=max_chars,
        max_sentences=3,
    )


def asset_note(symbol: str, facts: str, interpretation: str, *, max_chars: int) -> str:
    prefix = f"{symbol}: "
    pattern = rf"^{re.escape(symbol)}(?:\s+|\s*[:;,-]\s*)"
    clean_facts = re.sub(pattern, "", facts.strip(), count=1, flags=re.IGNORECASE)
    clean_facts = clean_facts[0].upper() + clean_facts[1:] if clean_facts else facts
    body = compact_prose(
        " ".join(part for part in (clean_facts, interpretation) if part),
        max_chars=max_chars - len(prefix),
        max_sentences=2,
    )
    return prefix + (body or "Verified evidence is limited for this asset.")


def merge_clauses(values: Iterable[str]) -> str:
    clauses = [value.rstrip(" .!?") for value in values if value]
    return "; ".join(dict.fromkeys(clauses)) + ("." if clauses else "")


def qualitative_text(value: str) -> str:
    clean = safe_generated_text(value)
    if not clean:
        return ""
    return " ".join(
        sentence
        for sentence in re.split(r"(?<=[.!?])\s+", clean)
        if sentence and not numeric_tokens(sentence) and not _WRITTEN_NUMBER.search(sentence)
    ).strip()


def complete_sentence(value: str) -> str:
    clean = value.strip().rstrip(" ,;:")
    if not clean:
        return ""
    return clean if clean.endswith((".", "!", "?")) else clean + "."


def useful_verdict(
    value: str,
    *,
    default: str,
    asset_count: int,
    reject_news_causality: bool = False,
) -> str:
    clean = qualitative_text(value)
    if reject_news_causality and _UNSUPPORTED_NEWS_CAUSALITY.search(clean):
        clean = ""
    if asset_count == 1 and re.search(
        r"(?i)\b(?:across|assets|comparison|compared|relative)\b", clean
    ):
        clean = ""
    if len(clean.split()) < 5:
        clean = default
    return complete_sentence(clean)


def useful_section(value: str, *, default: str, avoid: Sequence[str] = ()) -> str:
    clean = qualitative_text(value)
    if len(clean.split()) < 3 or any(
        _normalized_prose(clean) == _normalized_prose(item) for item in avoid
    ):
        clean = default
    return complete_sentence(clean)


def useful_comparison(
    value: str,
    *,
    asset_count: int,
    avoid: Sequence[str] = (),
    reject_news_causality: bool = False,
) -> str:
    if asset_count < 2:
        return ""
    clean = qualitative_text(value)
    if reject_news_causality and _UNSUPPORTED_NEWS_CAUSALITY.search(clean):
        clean = ""
    if len(clean.split()) < 5 or any(
        _normalized_prose(clean) == _normalized_prose(item) for item in avoid
    ):
        clean = "The comparison is limited to the verified differences shown for each asset."
    return complete_sentence(clean)


def _normalized_prose(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def deterministic_claims(
    expected: Sequence[str],
    evidence: Mapping[str, object],
    *,
    kind_groups: Sequence[set[str]],
    news: bool = False,
) -> list[EvidenceClaim]:
    claims: list[EvidenceClaim] = []
    for symbol in expected:
        for kinds in kind_groups:
            evidence_ids = evidence_ids_for_asset(symbol, evidence, kinds=kinds)
            if news:
                evidence_ids.sort(
                    key=lambda evidence_id: news_evidence_rank(evidence[evidence_id]),
                    reverse=True,
                )
            evidence_ids = evidence_ids[: 2 if news else 5]
            statements = [
                statement
                for evidence_id in evidence_ids
                if (statement := evidence_statement(evidence[evidence_id])) is not None
            ]
            if statements:
                claims.append(
                    EvidenceClaim(
                        statement=(f"{symbol}: " + " ".join(statements))[:1200],
                        evidence_ids=evidence_ids,
                        claim_kind=(
                            "risk" if "deterministic_risk_assessment" in kinds else "observed_fact"
                        ),
                        confidence=1.0,
                    )
                )
    return claims[:12]


def finish_composition(
    *,
    agent: AgentId,
    answer: str,
    analysis: str,
    verdict: str,
    sections: Sequence[AgentAnalysisSection],
    comparison: str,
    claims: Sequence[EvidenceClaim],
    limitations: Sequence[str],
    generated_limitations: Sequence[str],
    recovery_notes: Sequence[str],
    confidence: str,
    asset_count: int,
) -> AgentAnswer:
    clean_limitations = list(
        dict.fromkeys([*limitations, *generated_limitations, *recovery_notes])
    )[:6]
    return AgentAnswer(
        agent=agent,
        answer=answer,
        analysis=analysis,
        structured_analysis=StructuredAgentAnalysis(
            verdict=verdict,
            sections=list(sections),
            comparison=comparison if asset_count > 1 else "",
        ),
        evidence=list(claims),
        uncertainty=[],
        limitations=clean_limitations,
        confidence={"low": 0.35, "medium": 0.65, "high": 0.85}[confidence],
        status="partial" if recovery_notes else "complete",
        analysis_state="live",
        coverage_state="partial" if clean_limitations else "complete",
    )


def compose_narrative_answer(
    generated: NarrativeLiveOutput,
    *,
    agent: AgentId,
    scope: str,
    evidence_kind: str,
    selected_evidence: Mapping[str, object],
    limitations: Sequence[str],
    requirements: AnswerRequirements,
    default_verdict: str,
    default_interpretation: str,
    empty_evidence: str,
    include_facts_in_section: bool,
    reject_news_causality: bool = False,
) -> AgentAnswer:
    expected = [item.label for item in requirements.assets]
    recovery_notes: list[str] = []
    live_sections = ordered_sections(
        generated.assets,
        expected,
        recovery_notes=recovery_notes,
    )
    verdict = useful_verdict(
        generated.verdict,
        default=default_verdict,
        asset_count=len(expected),
        reject_news_causality=reject_news_causality,
    )
    structured: list[AgentAnalysisSection] = []
    paragraphs = [compact_prose(verdict, max_chars=180, max_sentences=1)]
    for symbol in expected:
        section = live_sections.get(asset_key(symbol))
        facts = facts_for_asset(
            symbol,
            selected_evidence,
            kinds=(evidence_kind,),
            max_chars=240 if include_facts_in_section else 220,
        )
        interpretation = useful_section(
            section.analysis if isinstance(section, NarrativeAssetAnalysis) else "",
            default=default_interpretation,
            avoid=(verdict,),
        )
        if reject_news_causality and _UNSUPPORTED_NEWS_CAUSALITY.search(interpretation):
            interpretation = default_interpretation
        section_text = (
            grounded_text(facts, interpretation, max_chars=480)
            if include_facts_in_section
            else interpretation
        )
        structured.append(AgentAnalysisSection(asset=symbol, scope=scope, text=section_text))
        paragraphs.append(
            asset_note(
                symbol,
                facts or empty_evidence,
                interpretation,
                max_chars=400,
            )
        )
    comparison = useful_comparison(
        generated.comparison,
        asset_count=len(expected),
        avoid=(verdict,),
        reject_news_causality=reject_news_causality,
    )
    if len(expected) > 1 and comparison:
        paragraphs.append(compact_prose(comparison, max_chars=220, max_sentences=1))
    return finish_composition(
        agent=agent,
        answer=join_paragraphs(paragraphs, max_chars=2_400),
        analysis="",
        verdict=verdict,
        sections=structured,
        comparison=comparison,
        claims=deterministic_claims(
            expected,
            selected_evidence,
            kind_groups=({evidence_kind},),
            news=reject_news_causality,
        ),
        limitations=limitations,
        generated_limitations=[
            value for item in generated.limitations if (value := safe_generated_text(str(item)))
        ],
        recovery_notes=recovery_notes,
        confidence=generated.confidence,
        asset_count=len(expected),
    )


_UNSAFE_SOURCE_TEXT = re.compile(
    r"(?i)(?:https?://|javascript:|\\b(?:buy|sell|short)\\s+[A-Z$0-9]|guaranteed?\\s+to)"
)
_COVERAGE_PREFIXES = {
    "discovery": ("opportunity.",),
    "market": ("market.", "technical."),
    "derivatives": ("derivatives.",),
    "news": ("news",),
    "fundamentals": ("fundamentals",),
    "defi": ("defi",),
    "risk": ("risk",),
    "onchain": ("onchain.",),
}


def _evidence_policy(agent: AgentId) -> AgentEvidencePolicy:
    from crypto_research.agents.registry import analyzer_for

    return analyzer_for(agent).evidence_policy


def compact_evidence(evidence: Mapping[str, object]) -> dict[str, object]:
    compacted: dict[str, object] = {}
    for evidence_id, value in evidence.items():
        if not isinstance(value, Mapping):
            compacted[evidence_id] = value
            continue
        record = _fallback_compact_mapping(value)
        observed_at = value.get("observed_at") or value.get("collected_at")
        if observed_at is not None:
            record["observed_at"] = observed_at
        compacted[evidence_id] = record
    return compacted


def evidence_first_answer(
    *,
    agent: AgentId,
    question: str,
    message: str,
    limitations: Sequence[str],
    evidence: Mapping[str, object],
) -> AgentAnswer:
    """Present collected facts directly when model-generated prose is unavailable."""

    if not evidence:
        return AgentAnswer(
            agent=agent,
            answer=message,
            uncertainty=["Verified information was insufficient."],
            limitations=_fallback_unique(limitations),
            confidence=0.0,
            status="unavailable",
            analysis_state="unavailable",
            coverage_state="partial",
        )

    policy = _evidence_policy(agent)
    claims = (
        _fallback_claims(policy, question, evidence)
        or _fallback_comparison_claims(question, evidence)
        or [
            EvidenceClaim(statement=statement, evidence_ids=[evidence_id], confidence=1.0)
            for evidence_id, record in _fallback_prioritized(question, evidence)
            if (statement := _fallback_statement(record)) is not None
        ][:6]
    )
    if not claims:
        evidence_id = next(iter(evidence))
        claims = [
            EvidenceClaim(
                statement="A verified record remains available for inspection.",
                evidence_ids=[evidence_id],
            )
        ]
    note = (
        "Verified records are shown directly because a full interpretation is temporarily "
        "unavailable."
    )
    answer_text = _fallback_prose(policy, claims)
    return AgentAnswer(
        agent=agent,
        answer=answer_text or " ".join(claim.statement for claim in claims)[:1200],
        structured_analysis=_fallback_structured_analysis(agent, evidence),
        evidence=claims,
        uncertainty=[message],
        limitations=_fallback_unique([*limitations, note]),
        confidence=0.7,
        status="partial",
        analysis_state="evidence_only",
        coverage_state="partial",
    )


def _fallback_structured_analysis(
    agent: AgentId,
    evidence: Mapping[str, object],
) -> StructuredAgentAnalysis | None:
    """Give degraded specialist output the same asset-first visual structure."""

    policy = _evidence_policy(agent)
    groups = policy.fallback_scopes
    kinds = tuple(kind for _, scope_kinds in groups for kind in scope_kinds)
    assets = _fallback_asset_order(evidence, kinds=kinds)
    if not assets:
        return None
    sections: list[AgentAnalysisSection] = []
    for asset in assets[:4]:
        for scope, scope_kinds in groups:
            statements = [
                statement
                for evidence_id, record in evidence.items()
                if _fallback_mapping(record).get("claim_type") in scope_kinds
                and _fallback_record_matches_asset(evidence_id, record, asset)
                and (statement := _fallback_statement(record)) is not None
            ]
            if statements:
                section_text = policy.structured_section_override or " ".join(statements)
                sections.append(
                    AgentAnalysisSection(
                        asset=asset,
                        scope=scope,
                        text=_fallback_bounded_section_text(section_text),
                    )
                )
    if not sections:
        return None
    return StructuredAgentAnalysis(
        verdict=policy.fallback_verdict,
        sections=sections,
        comparison=(
            "The selected assets differ in the validated metrics shown in each card."
            if len(assets) > 1
            else ""
        ),
    )


def _fallback_bounded_section_text(value: str) -> str:
    if len(value) <= 500:
        return value
    shortened = value[:497].rsplit(" ", maxsplit=1)[0].rstrip(" ,;:")
    return (shortened or value[:497]) + "..."


def no_evidence_message(inputs: AnalysisInputs, *, agent: AgentId | None = None) -> str:
    del inputs
    if agent is not None:
        return _evidence_policy(agent).no_evidence_message
    return "I could not verify enough evidence to answer that request confidently."


def coverage_issues(
    answer: AgentAnswer,
    evidence: Mapping[str, object],
    requirements: Sequence[str],
) -> list[str]:
    cited = {evidence_id for claim in answer.evidence for evidence_id in claim.evidence_ids}
    issues: list[str] = []
    for requirement in requirements:
        prefixes = _COVERAGE_PREFIXES.get(requirement, ())
        available = {
            evidence_id
            for evidence_id in evidence
            if any(evidence_id.startswith(prefix) for prefix in prefixes)
        }
        if available and not cited & available:
            issues.append(f"analysis omitted available {requirement} evidence")
    return issues


def _fallback_compact_mapping(value: Mapping[object, object]) -> dict[str, object]:
    drop = EVIDENCE_DROP_FIELDS
    if "content" in {str(key) for key in value} and not _fallback_non_empty_text(
        value.get("content")
    ):
        drop = drop - {"excerpt"}
    return {
        str(key): _fallback_compact_value(item)
        for key, item in value.items()
        if str(key) not in drop and item is not None
    }


def _fallback_non_empty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _fallback_compact_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _fallback_compact_mapping(value)
    if isinstance(value, list):
        return [_fallback_compact_value(item) for item in value[:5]]
    if isinstance(value, str):
        return " ".join(value.split())[:320]
    return value


def _fallback_prioritized(
    question: str, evidence: Mapping[str, object]
) -> list[tuple[str, object]]:
    lowered = question.casefold()
    wanted = [
        kind
        for terms, kind in (
            (("price", "market", "momentum", "rsi"), "market_snapshot"),
            (("news", "latest", "today", "catalyst"), "recent_news"),
            (("fundamental", "supply", "market cap", "tokenomics"), "project_fundamentals"),
            (("tvl", "defi", "yield", "apy"), "defi_protocol_metrics"),
            (("risk", "safe"), "deterministic_risk_assessment"),
            (("funding", "open interest", "derivatives", "futures"), "derivatives_positioning"),
        )
        if any(term in lowered for term in terms)
    ]
    ranks = {kind: index for index, kind in enumerate(wanted)}
    ranked = sorted(
        evidence.items(),
        key=lambda item: ranks.get(
            _fallback_text(_fallback_mapping(item[1]).get("claim_type")), len(ranks)
        ),
    )
    selected: list[tuple[str, object]] = []
    for kind in wanted:
        match = next(
            (
                item
                for item in ranked
                if item not in selected and _fallback_mapping(item[1]).get("claim_type") == kind
            ),
            None,
        )
        if match is not None:
            selected.append(match)
    return [*selected, *(item for item in ranked if item not in selected)]


def _fallback_statement(value: object) -> str | None:  # noqa: C901 - explicit claim-type formatting
    record = _fallback_mapping(value)
    payload = _fallback_mapping(record.get("payload"))
    kind, asset = (
        _fallback_text(record.get("claim_type")),
        _fallback_text(record.get("asset")) or "The asset",
    )
    if kind == "market_snapshot":
        snapshot, features = (
            _fallback_mapping(payload.get("snapshot")),
            _fallback_mapping(payload.get("ohlcv_features")),
        )
        price, change = (
            _fallback_number(snapshot.get("current_price")),
            _fallback_return(features, "24h"),
        )
        if price is not None:
            suffix = f", with a 24-hour change of {change:+.2f}%" if change is not None else ""
            return (
                f"{asset} was {format_money(price)} on "
                f"{_fallback_text(snapshot.get('exchange')).title()}{suffix}."
            )
    if kind == "technical_calculation":
        rsi = _fallback_number(payload.get("rsi"))
        return f"{asset} had a {_fallback_text(payload.get('trend')) or 'neutral'} trend" + (
            f" with RSI at {rsi:.1f}." if rsi is not None else "."
        )
    if kind == "market_screen":
        price = _fallback_number(payload.get("current_price"))
        score = _fallback_number(payload.get("score"))
        momentum = _fallback_number(payload.get("momentum_24h"))
        rank = _fallback_number(payload.get("rank"))
        symbol = _fallback_text(payload.get("symbol")) or asset
        if price is None or score is None or momentum is None:
            return None
        rank_label = f"Rank #{rank:.0f}: " if rank is not None else ""
        return (
            f"{rank_label}{symbol} was {_fallback_screen_price(price)}, scored {score:.1f}/100, "
            f"and had {momentum:+.2f}% 24-hour momentum."
        )
    if kind == "recent_news":
        title = _fallback_safe_text(payload.get("title"))
        if title:
            title = re.sub(r"\s+-\s+[^-]{2,60}$", "", title)
            source = _fallback_text(record.get("source"))
            source = "Recent coverage" if "google news" in source.casefold() else source
            published = _fallback_news_time_label(payload.get("published_at"))
            date_part = f" on {published}" if published else ""
            return f'{source or "A news source"} reported "{title}"{date_part}.'
        return None
    if kind == "project_fundamentals":
        cap = _fallback_number(payload.get("market_cap"))
        rank = _fallback_number(payload.get("rank"))
        circulating = _fallback_number(payload.get("circulating_supply"))
        maximum = _fallback_number(payload.get("max_supply"))
        total = _fallback_number(payload.get("total_supply"))
        developer = _fallback_mapping(payload.get("developer_activity"))
        commits = _fallback_number(developer.get("commits_4_weeks"))
        parts: list[str] = []
        if cap is not None:
            parts.append(f"market capitalization was {format_money(cap)}")
        if rank is not None:
            parts.append(f"provider rank was #{rank:,.0f}")
        if circulating is not None:
            parts.append(f"circulating supply was {format_compact_number(circulating)}")
        if maximum is not None:
            parts.append(f"maximum supply was {format_compact_number(maximum)}")
        elif total is not None:
            parts.append(f"total supply was {format_compact_number(total)}")
        if commits is not None:
            parts.append(f"CoinGecko reported {commits:,.0f} commits over four weeks")
        return (
            f"{_fallback_text(payload.get('name')) or asset} " + "; ".join(parts) + "."
            if parts
            else None
        )
    if kind == "defi_protocol_metrics":
        tvl = _fallback_number(payload.get("tvl_usd"))
        return (
            f"{_fallback_text(payload.get('protocol')) or asset} had TVL of {format_money(tvl)}."
            if tvl is not None
            else None
        )
    if kind == "deterministic_risk_assessment":
        score = _fallback_number(payload.get("score"))
        if score is None:
            return None
        confidence = _fallback_number(payload.get("evidence_confidence"))
        factors = payload.get("factors")
        gaps = payload.get("coverage_gaps")
        parts = [
            f"{asset} observed risk was "
            f"{_fallback_text(payload.get('band')).replace('_', ' ')} with score {score:.0f}"
        ]
        if confidence is not None:
            parts.append(f"with evidence confidence score {confidence:.0f}")
        if isinstance(factors, list) and factors:
            parts.append("leading factor: " + _fallback_brief_text(factors[0]))
        if isinstance(gaps, list) and gaps:
            parts.append("coverage gap: " + _fallback_brief_text(gaps[0]))
        return ", ".join(parts) + "."
    if kind == "derivatives_positioning":
        funding = _fallback_number(payload.get("latest_funding_rate"))
        open_interest = _fallback_number(payload.get("latest_open_interest_usd"))
        change = _fallback_number(payload.get("open_interest_change_24h_pct"))
        facts: list[str] = []
        if funding is not None:
            facts.append(f"latest funding was {funding:.4%}")
        if open_interest is not None:
            oi = f"open interest was {format_money(open_interest)}"
            if change is not None:
                oi += f" ({change:+.2f}% over 24 hours)"
            facts.append(oi)
        if not facts:
            return None
        venue = _fallback_text(payload.get("venue")) or _fallback_text(record.get("source"))
        return f"{asset} derivatives positioning on {venue}: " + "; ".join(facts) + "."
    if kind == "onchain_activity":
        latest = _fallback_number(payload.get("latest_value"))
        if latest is None:
            return None
        label = (
            _fallback_text(payload.get("label"))
            or _fallback_text(payload.get("metric"))
            or "Network activity"
        )
        rendered = (
            format_money(latest)
            if _fallback_text(payload.get("unit")) == "usd"
            else f"{latest:,.0f}"
        )
        change = _fallback_number(payload.get("seven_day_change_pct"))
        trend = f", {change:+.1f}% versus the prior seven days" if change is not None else ""
        return f"{asset} {label.casefold()} was {rendered}{trend}."
    if kind == "displayed_research_context" and (
        summary := _fallback_safe_text(payload.get("summary"), max_chars=260)
    ):
        return f"The displayed research summary says: {summary}."
    return None


def evidence_statement(value: object) -> str | None:
    """Render one validated evidence record as a deterministic factual sentence."""

    return _fallback_statement(value)


def news_evidence_rank(value: object) -> tuple[int, str]:
    record = value if isinstance(value, Mapping) else {}
    payload_value = record.get("payload")
    payload = payload_value if isinstance(payload_value, Mapping) else {}
    quality = {"high": 2, "medium": 1, "low": 0}.get(str(payload.get("source_quality")), 0)
    observed = record.get("observed_at") or payload.get("published_at") or ""
    return quality, str(observed)


def _fallback_news_time_label(value: object) -> str:
    if isinstance(value, str) and value.strip():
        try:
            value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return ""
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.tzinfo.utcoffset(value) is None
    ):
        return ""
    return value.astimezone(UTC).strftime("%d %b %Y, %H:%M UTC")


def _fallback_brief_text(value: object, *, max_chars: int = 70) -> str:
    text = _fallback_text(value)
    if len(text) <= max_chars:
        return text
    shortened = text[:max_chars].rsplit(" ", maxsplit=1)[0].rstrip(",;:")
    return shortened or text[:max_chars]


def _fallback_claims(
    policy: AgentEvidencePolicy,
    question: str,
    evidence: Mapping[str, object],
) -> list[EvidenceClaim]:
    if not policy.fallback_claim_kinds:
        return []
    return _fallback_grouped_asset_claims(
        evidence,
        kinds=policy.fallback_claim_kinds,
        include_comparison=(policy.compare_fallback_always or "compare" in question.casefold()),
    )


def _fallback_prose(
    policy: AgentEvidencePolicy,
    claims: Sequence[EvidenceClaim],
) -> str | None:
    if not claims:
        return None
    asset_claims = [claim for claim in claims if _fallback_claim_asset(claim)]
    selected = asset_claims or list(claims)
    intro = policy.fallback_intro
    if len(asset_claims) > 1 and policy.fallback_multi_intro:
        intro = policy.fallback_multi_intro
    paragraphs = [
        _fallback_clean_claim_statement(claim.statement)
        for claim in selected
        if not claim.statement.startswith("Automatic comparison:")
    ][:4]
    if len(paragraphs) > 1 and policy.fallback_comparison_note:
        paragraphs.append(policy.fallback_comparison_note)
    return "\n\n".join([intro, *paragraphs])[:1600]


def _fallback_claim_asset(claim: EvidenceClaim) -> str | None:
    match = re.match(r"([^:]{2,40}/[A-Z]{2,10}|[A-Z0-9]{2,15}):", claim.statement)
    return match.group(1) if match else None


def _fallback_clean_claim_statement(statement: str) -> str:
    # Grouped fallback claims already begin with an asset label.  Keep that
    # visible, but remove the double-echo created by lower-level fact statements.
    prefix, separator, rest = statement.partition(": ")
    if not separator:
        return statement
    cleaned = re.sub(rf"\b{re.escape(prefix)}:\s+", "", rest)
    return f"{prefix}: {cleaned}"


def _fallback_grouped_asset_claims(
    evidence: Mapping[str, object],
    *,
    kinds: Sequence[str],
    include_comparison: bool = False,
) -> list[EvidenceClaim]:
    assets = _fallback_asset_order(evidence, kinds=kinds)
    if not assets:
        return []
    claims: list[EvidenceClaim] = []
    for asset in assets[:4]:
        selected: list[tuple[str, object]] = []
        for kind in kinds:
            match = next(
                (
                    (evidence_id, record)
                    for evidence_id, record in evidence.items()
                    if _fallback_mapping(record).get("claim_type") == kind
                    and _fallback_record_matches_asset(evidence_id, record, asset)
                    and _fallback_statement(record)
                ),
                None,
            )
            if match is not None:
                selected.append(match)
        if not selected:
            continue
        statements = [
            statement
            for _, record in selected
            if (statement := _fallback_statement(record)) is not None
        ]
        claims.append(
            EvidenceClaim(
                statement=f"{asset}: " + " ".join(statements),
                evidence_ids=[evidence_id for evidence_id, _ in selected],
                confidence=1.0,
            )
        )
    if include_comparison and len(claims) > 1:
        claims.append(
            EvidenceClaim(
                statement=(
                    "Automatic comparison: the selected assets differ in the verified metrics "
                    "shown above; review the per-coin records before drawing a relative conclusion."
                ),
                evidence_ids=[
                    evidence_id for claim in claims for evidence_id in claim.evidence_ids
                ][:6],
                confidence=0.7,
            )
        )
    return claims[:6]


def _fallback_asset_order(evidence: Mapping[str, object], *, kinds: Sequence[str]) -> list[str]:
    assets: list[str] = []
    selected = set(kinds)
    for record in evidence.values():
        mapping = _fallback_mapping(record)
        if mapping.get("claim_type") not in selected:
            continue
        asset = _fallback_text(mapping.get("asset"))
        if asset and asset not in assets:
            assets.append(asset)
    return assets


def _fallback_comparison_claims(
    question: str, evidence: Mapping[str, object]
) -> list[EvidenceClaim]:
    if "compare" not in question.casefold():
        return []
    assets = list(
        dict.fromkeys(
            _fallback_text(_fallback_mapping(record).get("asset"))
            for record in evidence.values()
            if _fallback_mapping(record).get("claim_type") == "market_snapshot"
            and _fallback_text(_fallback_mapping(record).get("asset"))
        )
    )
    if len(assets) < 2:
        return []
    kinds = (
        "market_snapshot",
        "technical_calculation",
        "deterministic_risk_assessment",
        "recent_news",
        "project_fundamentals",
    )
    claims: list[EvidenceClaim] = []
    for asset in assets[:5]:
        records = [
            (evidence_id, record)
            for evidence_id, record in evidence.items()
            if _fallback_matches_asset(evidence_id, asset)
        ]
        matches = [
            next(
                (
                    item
                    for item in records
                    if _fallback_mapping(item[1]).get("claim_type") == kind
                    and _fallback_statement(item[1])
                ),
                None,
            )
            for kind in kinds
        ]
        selected = [item for item in matches if item is not None]
        if selected:
            claims.append(
                EvidenceClaim(
                    statement=" ".join(
                        statement
                        for _, record in selected
                        if (statement := _fallback_statement(record)) is not None
                    )[:1200],
                    evidence_ids=[evidence_id for evidence_id, _ in selected][:5],
                    confidence=1.0,
                )
            )
    return claims[:6]


def _fallback_matches_asset(evidence_id: str, asset: str) -> bool:
    base = asset.split("/", maxsplit=1)[0].casefold()
    return base in re.findall(r"[a-z0-9]+", evidence_id.casefold())


def _fallback_record_matches_asset(evidence_id: str, record: object, asset: str) -> bool:
    record_asset = _fallback_text(_fallback_mapping(record).get("asset"))
    return record_asset == asset or _fallback_matches_asset(evidence_id, asset)


def deterministic_asset_statements(evidence: Mapping[str, object], asset: str) -> str:
    records = [(eid, rec) for eid, rec in evidence.items() if _fallback_matches_asset(eid, asset)]
    parts: list[str] = []
    for kind in ("market_snapshot", "technical_calculation", "deterministic_risk_assessment"):
        match = next(
            (
                item
                for item in records
                if _fallback_mapping(item[1]).get("claim_type") == kind
                and _fallback_statement(item[1])
            ),
            None,
        )
        if match is not None and (statement := _fallback_statement(match[1])):
            parts.append(statement)
    return " ".join(parts)[:800]


def _fallback_mapping(value: object) -> Mapping[object, object]:
    return value if isinstance(value, Mapping) else {}


def _fallback_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _fallback_number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _fallback_return(features: Mapping[object, object], label: str) -> float | None:
    values = features.get("returns")
    if not isinstance(values, list):
        return None
    match = next((item for item in values if _fallback_mapping(item).get("label") == label), None)
    return _fallback_number(_fallback_mapping(match).get("return_percent"))


def _fallback_safe_text(value: object, *, max_chars: int = 180) -> str | None:
    text = " ".join(_fallback_text(value).replace("$", "").split())
    if len(text) > max_chars:
        text = text[: max_chars - 3].rsplit(" ", maxsplit=1)[0] + "..."
    return text if text and not _UNSAFE_SOURCE_TEXT.search(text) else None


def _fallback_screen_price(value: float) -> str:
    if abs(value) >= 1:
        return f"${value:,.2f}"
    if abs(value) >= 0.01:
        return f"${value:,.4f}"
    return f"${value:,.8f}"


def _fallback_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))[:6]


__all__ = [
    "AnswerRequirements",
    "compact_evidence",
    "coverage_issues",
    "deterministic_asset_statements",
    "evidence_first_answer",
    "evidence_statement",
    "news_evidence_rank",
    "no_evidence_message",
    "answer_requirement_issues",
    "compile_answer_requirements",
    "live_unavailable_answer",
    "numeric_repair_context",
    "requirement_limitations",
    "salvage_live_answer",
    "asset_key",
    "asset_note",
    "compact_prose",
    "compose_narrative_answer",
    "deterministic_claims",
    "evidence_ids_for_asset",
    "facts_for_asset",
    "finish_composition",
    "grounded_text",
    "join_paragraphs",
    "merge_clauses",
    "ordered_sections",
    "safe_generated_text",
    "useful_comparison",
    "useful_section",
    "useful_verdict",
]
