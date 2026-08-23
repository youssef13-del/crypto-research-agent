"""Numeric extraction, grounding, and prompt-safe evidence compaction."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

_NUMERIC_VALUE = re.compile(
    r"(?<![A-Za-z0-9])(?P<sign>[-+])?(?P<currency>\$)?"
    r"(?P<number>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s*(?P<scale>thousand|million|billion|trillion|k|m|bn|b|tn|t)"
    r"(?![A-Za-z0-9]))?(?P<percent>%)?",
    flags=re.IGNORECASE,
)
_NUMERIC_CONTEXT = re.compile(
    r"(?i)\b(?:(?:[a-z0-9]{2,15}/[a-z]{2,10})\s+(?:at|is)|trading at|price|change|"
    r"return|rsi|macd|score|rank(?:s|ed|ing)?|support|resistance|volume|supply|"
    r"quote volume|total volume|base volume|market cap|capitalization|tvl|funding|"
    r"open interest|liquidation|volatility|confidence|dominance)\b"
)
_ISO_DATE = re.compile(
    r"\b\d{4}[-\u2010-\u2015]\d{2}[-\u2010-\u2015]\d{2}"
    r"(?:[Tt]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:[Zz]|[+-]\d{2}:?\d{2})?)?\b"
)
_CURRENCY_CONTEXT = re.compile(
    r"(?i)\b(?:(?:[a-z0-9]{2,15}/[a-z]{2,10})\s+(?:at|is)|trading at|price|"
    r"market cap|capitalization|tvl|quote volume|total volume|support|resistance|atr|"
    r"macd|open interest|liquidation)\b"
)
_AMOUNT_CONTEXT = re.compile(r"(?i)\b(?:supply|base volume|token count)\b")
_DURATION_SUFFIX = re.compile(
    r"(?i)^\s*[-\u2010-\u2015]?\s*"
    r"(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|[mhdw])\b"
)
_PERCENT_PATH = re.compile(r"(?:percent|percentage|pct|change_[17]d)")
_CURRENCY_PATH = re.compile(
    r"(?:price(?:_usd)?|market_cap|tvl|(?:^|\.)[^.]*_usd|quote_volume|total_volume|"
    r"support|resistance|atr|macd|open_interest|liquidation|latest_completed_close|"
    r"range_absolute|(?:^|\.)(?:open|high|low|close)(?:\.|$))"
)
_AMOUNT_PATH = re.compile(r"(?:supply|base_volume)")
_NUMBER_PATH = re.compile(r"(?:^|\.)(?:[^.]*rank)(?:\.|$)")
_EVIDENCE_TEXT_FIELDS = frozenset(
    {
        "category",
        "excerpt",
        "name",
        "reason",
        "sentiment",
        "status",
        "summary",
        "symbol",
        "title",
        "trend",
    }
)
_EVIDENCE_IMPORTANT_NUMERIC = re.compile(
    r"(?:current_price|latest_price|range_percent|score|risk|market_cap|supply|tvl|rsi|sma|ema|"
    r"volatility|support|resistance|sentiment|return|momentum)"
)
_EVIDENCE_HOUSEKEEPING_NUMERIC = re.compile(r"(?:hours|count|delay|rank|timestamp)")

EVIDENCE_DROP_FIELDS = frozenset(
    {
        "evidence_id",
        "collected_at",
        "data_source",
        "coin_id",
        "latest_completed_close",
        "window_start",
        "window_end",
        "first_time",
        "last_time",
        "base_volume",
        "range_absolute",
        "contiguous",
        "hours",
        "reference_price",
        "latest_price",
        "change_absolute",
        "return_decimal",
        "period_start",
        "period_end",
        "homepage",
        "excerpt",
        "coverage_gaps",
    }
)


class NumericUnit(StrEnum):
    CURRENCY = "currency"
    PERCENT = "percent"
    AMOUNT = "amount"
    NUMBER = "number"


@dataclass(frozen=True, slots=True)
class NumericToken:
    raw: str
    value: float
    unit: NumericUnit
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class NumericFact:
    evidence_id: str
    path: str
    value: float
    unit: NumericUnit
    asset: str | None = None


def numeric_tokens(text: str) -> tuple[NumericToken, ...]:
    """Extract answer metrics while ignoring dates, durations, and incidental integers."""

    tokens: list[NumericToken] = []
    date_spans = tuple(match.span() for match in _ISO_DATE.finditer(text))
    for match in _NUMERIC_VALUE.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in date_spans):
            continue
        raw_number = match.group("number")
        explicit = bool(match.group("currency") or match.group("percent") or match.group("scale"))
        prefix = text[max(0, match.start() - 55) : match.start()]
        nearby = prefix + text[match.start() : match.end() + 55]
        plain = raw_number.replace(",", "")
        if not explicit and (
            _NUMERIC_CONTEXT.search(nearby) is None
            or _DURATION_SUFFIX.search(text[match.end() :]) is not None
            or plain.isdigit()
            and 1900 <= int(plain) <= 2100
        ):
            continue
        scale = _numeric_scale(match.group("scale"))
        sign = -1.0 if match.group("sign") == "-" else 1.0
        value = sign * float(plain) * scale
        tokens.append(
            NumericToken(
                raw=match.group(0),
                value=value,
                unit=_token_unit(match, prefix),
                start=match.start(),
                end=match.end(),
            )
        )
    return tuple(tokens)


def evidence_numeric_facts(evidence: Mapping[str, object]) -> tuple[NumericFact, ...]:
    """Create unit-aware numeric facts from allow-listed evidence records."""

    facts: list[NumericFact] = []
    for evidence_id, record in evidence.items():
        asset = (
            str(record.get("asset"))
            if isinstance(record, Mapping) and record.get("asset")
            else None
        )
        _collect_numeric_facts(record, (), evidence_id, asset, facts)
    return tuple(facts)


def compact_evidence_for_llm(
    evidence: Mapping[str, object],
    *,
    max_text_chars: int = 320,
    max_numeric_facts: int = 5,
    max_text_facts: int = 3,
    text_fact_chars: int = 90,
    numeric_priority: Sequence[str] | None = None,
) -> dict[str, str]:
    """Compact allow-listed evidence while preserving citation identifiers."""

    return {
        str(key): _compact_evidence_record(
            str(key),
            value,
            max_text_chars=max_text_chars,
            max_numeric_facts=max_numeric_facts,
            max_text_facts=max_text_facts,
            text_fact_chars=text_fact_chars,
            numeric_priority=numeric_priority,
        )
        for key, value in evidence.items()
    }


def _compact_evidence_record(
    evidence_id: str,
    value: object,
    *,
    max_text_chars: int,
    max_numeric_facts: int,
    max_text_facts: int,
    text_fact_chars: int,
    numeric_priority: Sequence[str] | None = None,
) -> str:
    record = value if isinstance(value, Mapping) else {}
    is_news = record.get("claim_type") == "recent_news"
    parts = [
        f"type={record.get('claim_type') or 'unknown'}",
        f"asset={record.get('asset') or 'global'}",
    ]
    numeric = _compact_numeric_facts(
        evidence_id,
        record,
        min(max_numeric_facts, 2) if is_news else max_numeric_facts,
        priority=numeric_priority,
    )
    text = _compact_text_facts(record.get("payload"), max_text_facts, text_fact_chars)
    if is_news:
        if text:
            parts.append("text=" + " | ".join(text))
        if numeric:
            parts.append("numeric=" + ",".join(numeric))
    else:
        if numeric:
            parts.append("numeric=" + ",".join(numeric))
        if text:
            parts.append("text=" + " | ".join(text))
    observed = record.get("observed_at") or record.get("collected_at")
    if observed:
        parts.append(f"observed_at={observed}")
    return _compact_prompt_text(";".join(parts), max_text_chars)


def _compact_numeric_facts(
    evidence_id: str,
    record: Mapping[object, object],
    limit: int,
    *,
    priority: Sequence[str] | None = None,
) -> list[str]:
    facts = sorted(
        evidence_numeric_facts({evidence_id: record}),
        key=lambda fact: _compact_numeric_priority(fact, priority=priority),
    )
    compact: list[str] = []
    seen: set[tuple[str, float, str]] = set()
    for fact in facts:
        field = fact.path.rsplit(".", maxsplit=1)[-1]
        key = (field, fact.value, fact.unit.value)
        if key in seen:
            continue
        seen.add(key)
        compact.append(f"{field}={fact.value:g}[{fact.unit.value}]")
        if len(compact) == limit:
            break
    return compact


def _compact_numeric_priority(
    fact: NumericFact,
    *,
    priority: Sequence[str] | None = None,
) -> tuple[int, int | str]:
    field = fact.path.rsplit(".", maxsplit=1)[-1]
    if priority is not None:
        for index, term in enumerate(priority):
            if field == term:
                return (0, index)
    important = _EVIDENCE_IMPORTANT_NUMERIC.search(field)
    housekeeping = _EVIDENCE_HOUSEKEEPING_NUMERIC.search(field)
    return (1 if important else 3 if housekeeping else 2), field


def _compact_text_facts(value: object, limit: int, text_fact_chars: int) -> list[str]:
    found: list[str] = []

    def visit(item: object, path: tuple[str, ...]) -> None:
        if len(found) >= limit:
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                visit(nested, (*path, str(key).casefold()))
        elif isinstance(item, list | tuple):
            for nested in item[:3]:
                visit(nested, path)
        elif isinstance(item, str) and path and path[-1] in _EVIDENCE_TEXT_FIELDS and item.strip():
            found.append(f"{path[-1]}={_compact_prompt_text(item.strip(), text_fact_chars)}")

    visit(value, ())
    return found


def _compact_prompt_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    head = max(1, limit * 2 // 3)
    tail = max(1, limit - head - 15)
    head_text = value[:head].rsplit(" ", maxsplit=1)[0]
    tail_text = value[-tail:].lstrip()
    if not head_text or not tail_text:
        return value[:head] + " ...[compacted]... " + value[-tail:]
    return head_text + " ...[compacted]... " + tail_text


def _collect_numeric_facts(
    value: object,
    path: tuple[str, ...],
    evidence_id: str,
    asset: str | None,
    facts: list[NumericFact],
) -> None:
    path_text = ".".join(path)
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        if not isfinite(value):
            return
        facts.append(
            NumericFact(
                evidence_id=evidence_id,
                path=path_text,
                value=float(value),
                unit=_path_unit(path_text),
                asset=asset,
            )
        )
    elif isinstance(value, str):
        facts.extend(
            NumericFact(evidence_id, path_text, token.value, token.unit, asset)
            for token in numeric_tokens(value)
        )
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            _collect_numeric_facts(
                nested,
                (*path, str(key).casefold()),
                evidence_id,
                asset,
                facts,
            )
    elif isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            _collect_numeric_facts(nested, (*path, str(index)), evidence_id, asset, facts)


def numeric_token_supported(token: NumericToken, facts: Iterable[NumericFact]) -> bool:
    """Match one displayed number to a same-unit fact with bounded rounding tolerance."""

    return any(
        token.unit is fact.unit and numeric_token_value_matches(token, fact) for fact in facts
    )


def numeric_token_value_matches(token: NumericToken, fact: NumericFact) -> bool:
    tolerance = _display_tolerance(token)
    return abs(token.value - fact.value) <= max(tolerance, abs(fact.value) * 0.005)


def numeric_tokens_match(left: NumericToken, right: NumericToken) -> bool:
    if left.unit is not right.unit:
        return False
    tolerance = max(_display_tolerance(left), _display_tolerance(right))
    return abs(left.value - right.value) <= max(tolerance, abs(right.value) * 0.005)


def numeric_fact_prompt_value(fact: NumericFact) -> dict[str, object]:
    return {
        "evidence_id": fact.evidence_id,
        "path": fact.path,
        "value": fact.value,
        "unit": fact.unit.value,
        "asset": fact.asset,
    }


def _token_unit(match: re.Match[str], prefix: str) -> NumericUnit:
    if match.group("currency"):
        return NumericUnit.CURRENCY
    if match.group("percent"):
        return NumericUnit.PERCENT
    metric_matches = list(_NUMERIC_CONTEXT.finditer(prefix))
    if metric_matches:
        nearest_metric = metric_matches[-1].group(0)
        if _CURRENCY_CONTEXT.fullmatch(nearest_metric):
            return NumericUnit.CURRENCY
        if _AMOUNT_CONTEXT.fullmatch(nearest_metric):
            return NumericUnit.AMOUNT
        return NumericUnit.NUMBER
    if match.group("scale"):
        return NumericUnit.AMOUNT
    return NumericUnit.NUMBER


def _path_unit(path: str) -> NumericUnit:
    lowered = path.casefold()
    if _NUMBER_PATH.search(lowered):
        return NumericUnit.NUMBER
    if _PERCENT_PATH.search(lowered):
        return NumericUnit.PERCENT
    if _CURRENCY_PATH.search(lowered):
        return NumericUnit.CURRENCY
    if _AMOUNT_PATH.search(lowered):
        return NumericUnit.AMOUNT
    return NumericUnit.NUMBER


def _numeric_scale(value: str | None) -> float:
    return {
        None: 1.0,
        "thousand": 1e3,
        "k": 1e3,
        "million": 1e6,
        "m": 1e6,
        "billion": 1e9,
        "b": 1e9,
        "bn": 1e9,
        "trillion": 1e12,
        "t": 1e12,
        "tn": 1e12,
    }[value.casefold() if value is not None else None]


def _display_tolerance(token: NumericToken) -> float:
    match = _NUMERIC_VALUE.fullmatch(token.raw.strip())
    if match is None:
        return max(0.02, abs(token.value) * 0.005)
    number = match.group("number").replace(",", "")
    decimals = len(number.rsplit(".", maxsplit=1)[1]) if "." in number else 0
    scale = _numeric_scale(match.group("scale"))
    shown_unit = scale * 10.0 ** (-decimals)
    if token.unit is NumericUnit.PERCENT:
        return max(0.15, shown_unit / 2)
    if scale != 1:
        return max(shown_unit / 2, abs(token.value) * 0.005)
    if token.unit is NumericUnit.CURRENCY and decimals == 0:
        zeros = len(number) - len(number.rstrip("0"))
        shown_unit = float(10**zeros)
    return max(0.02, shown_unit / 2, abs(token.value) * 0.005)
