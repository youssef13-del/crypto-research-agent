"""Deterministic evidence accounting for specialist analysis and UI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TypedDict

from crypto_research.domain.evidence import DerivativesEvidence
from crypto_research.domain.market import MarketEvidence
from crypto_research.domain.research import (
    AnalysisInputs,
    AssetResearchBundle,
    CapabilityCoverage,
    CollectionContext,
    EvidenceCoverageEntry,
    EvidenceCoverageSummary,
    NewsEvidence,
    ResearchCapability,
)
from crypto_research.shared.numeric_grounding import evidence_numeric_facts

CAPABILITY_LIMITATIONS: dict[ResearchCapability, str] = {
    ResearchCapability.MARKET: "Current market data were not available.",
    ResearchCapability.NEWS: "Recent matching news coverage was not available.",
    ResearchCapability.FUNDAMENTALS: "Verified project fundamentals were not available.",
    ResearchCapability.DEFI: "Current DeFi protocol metrics were not available.",
    ResearchCapability.RISK: "A supported risk assessment was not available.",
    ResearchCapability.DISCOVERY: "A current market-wide opportunity scan was not available.",
    ResearchCapability.FORECAST: "A deterministic forecast could not be produced.",
    ResearchCapability.ONCHAIN: "Current on-chain activity metrics were not available.",
    ResearchCapability.DERIVATIVES: "Current derivatives positioning data were not available.",
}

_CAPABILITY_BY_CLAIM = {
    "market_screen": ResearchCapability.DISCOVERY,
    "market_snapshot": ResearchCapability.MARKET,
    "technical_calculation": ResearchCapability.MARKET,
    "recent_news": ResearchCapability.NEWS,
    "project_fundamentals": ResearchCapability.FUNDAMENTALS,
    "defi_protocol_metrics": ResearchCapability.DEFI,
    "deterministic_risk_assessment": ResearchCapability.RISK,
    "onchain_activity": ResearchCapability.ONCHAIN,
    "derivatives_positioning": ResearchCapability.DERIVATIVES,
}
_CLAIMS_BY_CAPABILITY: dict[ResearchCapability, frozenset[str]] = {
    ResearchCapability.DISCOVERY: frozenset({"market_screen"}),
    ResearchCapability.MARKET: frozenset({"market_snapshot", "technical_calculation"}),
    ResearchCapability.NEWS: frozenset({"recent_news"}),
    ResearchCapability.FUNDAMENTALS: frozenset({"project_fundamentals"}),
    ResearchCapability.DEFI: frozenset({"defi_protocol_metrics"}),
    ResearchCapability.RISK: frozenset({"deterministic_risk_assessment", "recent_news"}),
    ResearchCapability.FORECAST: frozenset(),
    ResearchCapability.ONCHAIN: frozenset({"onchain_activity"}),
    ResearchCapability.DERIVATIVES: frozenset({"derivatives_positioning"}),
}


def build_capability_coverage(inputs: AnalysisInputs) -> list[CapabilityCoverage]:
    requested = _requested_capability_set(inputs)
    if not requested or not inputs.assets:
        return []
    research_bundles = {
        bundle.asset.key: bundle
        for bundle in (inputs.research_result.asset_results if inputs.research_result else [])
    }
    fundamentals_bundles = {
        bundle.asset.key: bundle
        for bundle in (inputs.research_result.asset_results if inputs.research_result else [])
    }
    fundamentals_bundles.update(
        {
            bundle.asset.key: bundle
            for bundle in (
                inputs.fundamentals_result.asset_results if inputs.fundamentals_result else []
            )
        }
    )
    onchain_bundles = {
        bundle.asset.key: bundle
        for bundle in (inputs.onchain_result.asset_results if inputs.onchain_result else [])
    }
    market_symbols = {
        item.market.symbol
        for item in (
            inputs.market_comparison_result.assets
            if inputs.market_comparison_result is not None
            else []
        )
        if is_current_market(item.market, context=inputs.collection_context)
    }
    if (
        inputs.market_result is not None
        and inputs.assets
        and is_current_market(inputs.market_result.market, context=inputs.collection_context)
    ):
        market_symbols.add(inputs.assets[0].symbol)
    derivatives_by_symbol = {}
    if inputs.market_result is not None and inputs.market_result.derivatives is not None:
        derivatives_by_symbol[inputs.market_result.market.symbol] = inputs.market_result.derivatives
    if inputs.market_comparison_result is not None:
        derivatives_by_symbol.update(
            {
                item.market.symbol: item.derivatives
                for item in inputs.market_comparison_result.assets
                if item.derivatives is not None
            }
        )
    risk_by_key = {
        item.asset.key: item.assessment
        for item in (inputs.risk_result.asset_results if inputs.risk_result is not None else [])
    }

    coverage: list[CapabilityCoverage] = []
    for asset in inputs.assets:
        for capability in ResearchCapability:
            if capability not in requested:
                continue
            bundle = (
                fundamentals_bundles.get(asset.key)
                if capability in {ResearchCapability.FUNDAMENTALS, ResearchCapability.DEFI}
                else onchain_bundles.get(asset.key)
                if capability is ResearchCapability.ONCHAIN
                else research_bundles.get(asset.key)
            )
            status, kinds, limitation = _coverage_status(
                capability,
                bundle=bundle,
                market_available=asset.symbol in market_symbols,
                discovery_available=bool(
                    inputs.opportunity_result is not None
                    and inputs.opportunity_result.candidates
                    and _is_current(
                        inputs.opportunity_result.collected_at, inputs.collection_context
                    )
                ),
                risk_available=(
                    risk_by_key.get(asset.key) is not None
                    and risk_by_key[asset.key].evidence_confidence > 0
                ),
                derivatives=derivatives_by_symbol.get(asset.symbol),
                context=inputs.collection_context,
            )
            coverage.append(
                CapabilityCoverage(
                    asset=asset,
                    capability=capability,
                    status=status,
                    evidence_kinds=kinds,
                    limitation=limitation,
                )
            )
    return coverage


def is_current_market(
    market: MarketEvidence,
    *,
    context: CollectionContext | None = None,
) -> bool:
    reference = _reference_time(context)
    return market.collected_at <= reference and market.last_time <= reference


def _coverage_status(
    capability: ResearchCapability,
    *,
    bundle: AssetResearchBundle | None,
    market_available: bool,
    discovery_available: bool,
    risk_available: bool,
    derivatives: DerivativesEvidence | None,
    context: CollectionContext | None,
) -> tuple[str, list[str], str | None]:
    if capability is ResearchCapability.MARKET:
        available, kinds = market_available, ["market_snapshot", "technical_calculation"]
    elif capability is ResearchCapability.DISCOVERY:
        available, kinds = discovery_available, ["market_screen"]
    elif capability is ResearchCapability.RISK:
        available, kinds = risk_available, ["deterministic_risk_assessment"]
    elif capability is ResearchCapability.NEWS:
        available, kinds = (
            _has_current_news(bundle.news if bundle is not None else None, context=context),
            ["recent_news"],
        )
    elif capability is ResearchCapability.FUNDAMENTALS:
        available, kinds = (
            bool(
                bundle is not None
                and bundle.fundamentals is not None
                and bundle.fundamentals.status == "available"
                and _is_current(bundle.fundamentals.collected_at, context)
            ),
            ["project_fundamentals"],
        )
    elif capability is ResearchCapability.DEFI:
        if (
            bundle is not None
            and bundle.defi is not None
            and bundle.defi.status == "not_applicable"
        ):
            return "not_applicable", [], "DeFi metrics are not applicable to this asset."
        available, kinds = (
            bool(
                bundle is not None
                and bundle.defi is not None
                and bundle.defi.status == "available"
                and _is_current(bundle.defi.collected_at, context)
            ),
            ["defi_protocol_metrics"],
        )
    elif capability is ResearchCapability.ONCHAIN:
        if (
            bundle is not None
            and bundle.onchain is not None
            and bundle.onchain.status == "not_applicable"
        ):
            return "not_applicable", [], "On-chain metrics are not mapped for this asset."
        available, kinds = (
            bool(
                bundle is not None
                and bundle.onchain is not None
                and bundle.onchain.metrics
                and _is_current(bundle.onchain.collected_at, context)
            ),
            ["onchain_activity"],
        )
    elif capability is ResearchCapability.DERIVATIVES:
        derivative_status = str(getattr(derivatives, "status", ""))
        if derivative_status == "not_applicable":
            return (
                "not_applicable",
                [],
                "No active Binance USD-M perpetual contract exists for this asset.",
            )
        available, kinds = (
            bool(
                derivatives is not None
                and derivative_status in {"complete", "partial"}
                and _is_current(derivatives.collected_at, context)
            ),
            ["derivatives_positioning"],
        )
    else:
        available, kinds = False, []
    return (
        "available" if available else "unavailable",
        kinds if available else [],
        None if available else CAPABILITY_LIMITATIONS[capability],
    )


def _has_current_news(
    news: NewsEvidence | None,
    *,
    context: CollectionContext | None,
) -> bool:
    if news is None:
        return False
    collected_at = news.collected_at
    reference = min(collected_at, _reference_time(context))
    return any(item.published_at <= reference for item in news.items)


def _reference_time(context: CollectionContext | None) -> datetime:
    return context.collected_at if context is not None else datetime.now(UTC)


def _is_current(value: datetime, context: CollectionContext | None) -> bool:
    return value <= _reference_time(context)


def _requested_capability_set(inputs: AnalysisInputs) -> set[ResearchCapability]:
    if inputs.requested_capabilities:
        return set(inputs.requested_capabilities)
    if inputs.research_result is not None:
        return set(inputs.research_result.requested_capabilities)
    if inputs.fundamentals_result is not None:
        return set(inputs.fundamentals_result.requested_capabilities)
    if inputs.onchain_result is not None:
        return set(inputs.onchain_result.requested_capabilities)
    return set()


class _CoverageValues(TypedDict):
    asset: str
    capability: ResearchCapability
    accepted: int
    excluded: int
    detailed: int
    providers: set[str]
    ids: list[str]
    times: list[datetime]
    timeframes: set[str]
    limitations: list[str]


def select_detailed_evidence(
    ledger: Mapping[str, object], *, maximum: int = 24
) -> dict[str, object]:
    """Choose bounded, balanced citations while every record stays in the digest."""

    groups: set[tuple[str, ResearchCapability]] = set()
    selected: list[str] = []
    for evidence_id, record in ledger.items():
        group = _record_group(record)
        if group not in groups:
            groups.add(group)
            selected.append(evidence_id)
    selected.extend(item for item in ledger if item not in selected)
    return {evidence_id: ledger[evidence_id] for evidence_id in selected[:maximum]}


def build_evidence_coverage_summary(
    inputs: AnalysisInputs,
    ledger: Mapping[str, object],
    detailed_ids: Sequence[str],
) -> EvidenceCoverageSummary:
    """Reconcile accepted records, quality exclusions, and prompt representation."""

    details = set(detailed_ids)
    grouped: dict[tuple[str, ResearchCapability], _CoverageValues] = {}
    for evidence_id, record in ledger.items():
        asset, capability = _record_group(record)
        values = grouped.setdefault((asset, capability), _new_values(asset, capability))
        values["accepted"] = int(values["accepted"]) + 1
        values["providers"].add(_record_text(record, "source"))
        values["ids"].append(evidence_id)
        if evidence_id in details:
            values["detailed"] = int(values["detailed"]) + 1
        _record_time(values, _record_text(record, "observed_at"))
        _record_time(values, _record_text(record, "collected_at"))
        _record_timeframe(values, record)

    requested = set(inputs.requested_capabilities)
    for asset, capability, excluded, limitation in _quality_exclusions(inputs):
        # A specialist's ``AnalysisInputs`` is deliberately capability-scoped.
        # Do not let whole-run market quality metadata create an unrelated
        # empty entry in, for example, a News-only evidence bundle.
        if requested and capability not in requested:
            continue
        values = grouped.setdefault((asset, capability), _new_values(asset, capability))
        values["excluded"] = int(values["excluded"]) + excluded
        if limitation:
            values["limitations"].append(limitation)

    for item in inputs.capability_coverage:
        if requested and item.capability not in requested:
            continue
        key = (item.asset.symbol, item.capability)
        if key not in grouped:
            values = grouped.setdefault(key, _new_values(*key))
            matching_ids = _matching_evidence_ids(ledger, item.asset.symbol, item.capability)
            values["accepted"] = len(matching_ids)
            values["detailed"] = sum(evidence_id in details for evidence_id in matching_ids)
            values["ids"].extend(matching_ids)
            values["providers"].update(
                _record_text(ledger[evidence_id], "source") for evidence_id in matching_ids
            )
            if item.limitation:
                values["limitations"].append(item.limitation)

    entries = [_entry(values) for _, values in sorted(grouped.items())]
    accepted = len(ledger)
    excluded = sum(item.excluded_records for item in entries)
    return EvidenceCoverageSummary(
        entries=entries,
        detailed_evidence_ids=[item for item in ledger if item in details],
        summarized_evidence_ids=[item for item in ledger if item not in details],
        total_collected_records=accepted + excluded,
        total_accepted_records=accepted,
        total_excluded_records=excluded,
    )


def build_complete_evidence_digest(
    ledger: Mapping[str, object], summary: EvidenceCoverageSummary
) -> dict[str, object]:
    """Compact all accepted evidence into deterministic, non-citable coverage facts."""

    entries = list(summary.entries)
    providers = sorted({provider for item in entries for provider in item.providers})
    provider_index = {provider: index for index, provider in enumerate(providers)}
    return {
        "coverage_manifest": {
            "entry_format": (
                "asset|capability|collected/accepted/excluded/detailed/summarized|"
                "timeframes|time-range|provider-indexes|limitations"
            ),
            "providers": providers,
            "entries": [_digest_entry(item, provider_index) for item in entries],
            "total_collected_records": summary.total_collected_records,
            "total_accepted_records": summary.total_accepted_records,
            "total_excluded_records": summary.total_excluded_records,
        },
        "accepted_evidence_catalog": _compact_evidence_id_catalog(list(ledger)),
    }


def build_analysis_evidence_digest(
    ledger: Mapping[str, object], summary: EvidenceCoverageSummary
) -> dict[str, object]:
    """Return the bounded, non-citable digest safe to send to an LLM.

    The full coverage digest deliberately retains its stable evidence-ID catalog for
    auditability.  That catalog is not useful context for analysis and can consume a
    surprising share of a specialist prompt, so this companion digest exposes only
    deterministic per-scope aggregates.  Every accepted record is represented in its
    scope count, provider/time coverage, and numeric aggregate where applicable.
    """

    records_by_scope: dict[tuple[str, ResearchCapability], dict[str, object]] = {}
    for evidence_id, record in ledger.items():
        records_by_scope.setdefault(_record_group(record), {})[evidence_id] = record
    return {
        "scope_digest": [
            _analysis_scope_digest(
                entry,
                records_by_scope.get((entry.asset, entry.capability), {}),
            )
            for entry in summary.entries
        ],
        "total_accepted_records": summary.total_accepted_records,
        "total_excluded_records": summary.total_excluded_records,
    }


def _new_values(asset: str, capability: ResearchCapability) -> _CoverageValues:
    return {
        "asset": asset,
        "capability": capability,
        "accepted": 0,
        "excluded": 0,
        "detailed": 0,
        "providers": set(),
        "ids": [],
        "times": [],
        "timeframes": set(),
        "limitations": [],
    }


def _record_group(record: object) -> tuple[str, ResearchCapability]:
    asset = _record_text(record, "asset") or "market-wide"
    scoped = _record_capability(record)
    if scoped is not None:
        return asset, scoped
    claim_type = _record_text(record, "claim_type")
    return asset, _CAPABILITY_BY_CLAIM.get(claim_type, ResearchCapability.NEWS)


def _record_capability(record: object) -> ResearchCapability | None:
    if not isinstance(record, Mapping):
        return None
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("research_capability")
    try:
        return ResearchCapability(str(value)) if value is not None else None
    except ValueError:
        return None


def _record_text(record: object, key: str) -> str:
    if not isinstance(record, Mapping):
        return ""
    value = record.get(key)
    return str(value) if value is not None else ""


def _digest_entry(item: EvidenceCoverageEntry, provider_index: Mapping[str, int]) -> str:
    counts = "/".join(
        str(value)
        for value in (
            item.collected_records,
            item.accepted_records,
            item.excluded_records,
            item.detailed_records,
            item.summarized_records,
        )
    )
    start = _compact_timestamp(item.earliest_observed_at)
    end = _compact_timestamp(item.latest_observed_at)
    time_range = start if start == end else f"{start}/{end}"
    return "|".join(
        (
            item.asset,
            item.capability.value,
            counts,
            ",".join(item.timeframes) or "-",
            time_range,
            ",".join(str(provider_index[provider]) for provider in item.providers) or "-",
            "; ".join(item.limitations) or "-",
        )
    )


def _compact_timestamp(value: datetime | None) -> str:
    if value is None:
        return "-"
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _compact_evidence_id_catalog(evidence_ids: Sequence[str]) -> list[str]:
    """Represent every stable evidence ID without inflating a bounded LLM payload."""

    grouped: dict[str, list[int]] = {}
    literal: list[str] = []
    for evidence_id in evidence_ids:
        prefix, separator, suffix = evidence_id.rpartition(".")
        if separator and suffix.isdecimal():
            grouped.setdefault(prefix, []).append(int(suffix))
        else:
            literal.append(evidence_id)
    ranges: list[str] = []
    for prefix, numbers in grouped.items():
        ordered = sorted(numbers)
        if len(ordered) > 1 and ordered[-1] - ordered[0] + 1 == len(ordered):
            ranges.append(f"{prefix}.[{ordered[0]}-{ordered[-1]}]")
        else:
            ranges.extend(f"{prefix}.{number}" for number in ordered)
    return [*literal, *ranges]


def _matching_evidence_ids(
    ledger: Mapping[str, object], asset: str, capability: ResearchCapability
) -> list[str]:
    allowed_claims = _CLAIMS_BY_CAPABILITY[capability]
    return [
        evidence_id
        for evidence_id, record in ledger.items()
        if (
            _record_capability(record) is capability
            if _record_capability(record) is not None
            else _record_text(record, "claim_type") in allowed_claims
        )
        and _assets_match(_record_text(record, "asset"), asset)
    ]


def _assets_match(record_asset: str, requested_asset: str) -> bool:
    return (
        record_asset.casefold().split("/", maxsplit=1)[0]
        == requested_asset.casefold().split("/", maxsplit=1)[0]
    )


def _record_time(values: _CoverageValues, value: str) -> None:
    if not value:
        return
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return
    values["times"].append(observed)


def _record_timeframe(values: _CoverageValues, record: object) -> None:
    if not isinstance(record, Mapping) or not isinstance(record.get("payload"), Mapping):
        return
    snapshot = record["payload"].get("snapshot")
    if isinstance(snapshot, Mapping) and isinstance(snapshot.get("timeframe"), str):
        values["timeframes"].add(snapshot["timeframe"])


def _entry(values: _CoverageValues) -> EvidenceCoverageEntry:
    accepted = values["accepted"]
    times = values["times"]
    return EvidenceCoverageEntry(
        asset=values["asset"],
        capability=values["capability"],
        collected_records=accepted + values["excluded"],
        accepted_records=accepted,
        excluded_records=values["excluded"],
        detailed_records=values["detailed"],
        summarized_records=accepted - values["detailed"],
        providers=sorted(item for item in values["providers"] if item),
        earliest_observed_at=min(times) if times else None,
        latest_observed_at=max(times) if times else None,
        timeframes=sorted(values["timeframes"]),
        limitations=list(dict.fromkeys(values["limitations"]))[:12],
    )


_AGGREGATE_FACT_PRIORITY = (
    "current_price",
    "momentum",
    "return",
    "score",
    "volatility",
    "market_cap",
    "supply",
    "tvl",
    "change",
    "risk",
    "rsi",
    "support",
    "resistance",
)


_NEWS_HEADLINES_CHAR_BUDGET = 1_100
_MARKET_POSTURE_CHAR_BUDGET = 1_100


def _news_headline_text(record: object) -> str:
    """Render one news record as a bounded ``Title (Publisher)`` headline."""
    if not isinstance(record, Mapping):
        return ""
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        return ""
    title = str(payload.get("title") or "").strip()
    if not title:
        return ""
    publisher = str(payload.get("publisher") or "").strip()
    if not publisher:
        publisher = str(record.get("source") or "").strip()
    if publisher and publisher.casefold() not in title.casefold():
        return f"{title} ({publisher})"
    return title


def _news_headlines_brief(records: Mapping[str, object]) -> str:
    """List every accepted headline so coverage breadth stays visible.

    Detailed citations stay limited to a few records per asset; the bounded
    headline brief lets the model acknowledge all accepted coverage without
    consuming the detail budget.  Pipes are removed so the line survives
    ``_scope_digest_fields`` field splitting.
    """

    briefs = [
        brief.replace("|", ",")
        for brief in (_news_headline_text(record) for record in records.values())
        if brief
    ]
    if not briefs:
        return ""
    return " // ".join(briefs)[:_NEWS_HEADLINES_CHAR_BUDGET]


def _market_posture_text(record: object) -> str:
    """Render one market record's posture as a bounded single-line brief.

    Pipes are removed so the line survives ``_scope_digest_fields`` splitting.
    """

    if not isinstance(record, Mapping):
        return ""
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        return ""
    posture = payload.get("market_posture")
    if not isinstance(posture, Mapping):
        return ""
    return "; ".join(
        part
        for part in (
            _posture_price_bits(posture),
            _posture_return_bits(posture),
            _posture_indicator_bits(posture),
            _posture_range_bits(posture),
            _posture_freshness_bits(posture),
        )
        if part
    )


def _posture_price_bits(posture: Mapping[object, object]) -> str:
    price = posture.get("price")
    change = posture.get("change_24h_percent")
    parts = [f"price {price:g}" if isinstance(price, int | float) else ""]
    if isinstance(change, int | float):
        parts.append(f"24h {change:+.1f}%")
    return " ".join(part for part in parts if part)


def _posture_return_bits(posture: Mapping[object, object]) -> str:
    returns = posture.get("window_returns")
    if not isinstance(returns, list | tuple):
        return ""
    return ", ".join(
        f"{label} {value:+.1f}%"
        for item in returns
        if isinstance(item, Mapping)
        and (label := item.get("label")) in {"4h", "7d"}
        and isinstance((value := item.get("return_percent")), int | float)
    )


def _posture_indicator_bits(posture: Mapping[object, object]) -> str:
    trend = posture.get("trend")
    rsi = posture.get("rsi")
    macd = posture.get("macd")
    atr = posture.get("atr")
    parts = [f"trend {trend}" if trend else ""]
    if isinstance(rsi, int | float):
        band = posture.get("rsi_band")
        parts.append(f"rsi {rsi:.1f}{f' ({band})' if band else ''}")
    if isinstance(macd, int | float):
        parts.append(f"macd {'positive' if macd > 0 else 'negative'}")
    if isinstance(atr, int | float):
        parts.append(f"atr {atr:g}")
    return " ".join(part for part in parts if part)


def _posture_range_bits(posture: Mapping[object, object]) -> str:
    specifications = (
        ("maximum_drawdown", "drawdown", 100, ".1f"),
        ("volatility", "volatility", 100, ".2f"),
        ("support", "support", 1, "g"),
        ("resistance", "resistance", 1, "g"),
    )
    return " ".join(
        f"{label} {value * scale:{precision}}%" if scale == 100 else f"{label} {value:{precision}}"
        for key, label, scale, precision in specifications
        if isinstance((value := posture.get(key)), int | float)
    )


def _posture_freshness_bits(posture: Mapping[object, object]) -> str:
    fresh = posture.get("fresh")
    parts = ["fresh" if fresh else "stale"] if isinstance(fresh, bool) else []
    if fresh is False and (delay := posture.get("data_delay_seconds")) is not None:
        parts.append(f"age {_market_data_age(delay)}")
    confirmation = posture.get("contextual_confirmation")
    if isinstance(confirmation, list | tuple) and confirmation:
        parts.append("higher " + ", ".join(str(item) for item in confirmation))
    return " ".join(parts)


def _market_data_age(seconds: object) -> str:
    """Render a bounded data-age label (for example ``45min`` or ``3h``)."""

    if not isinstance(seconds, int | float) or seconds <= 0:
        return "unknown"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"{minutes}min"
    return f"{minutes / 60:.1f}h"


def _market_posture_brief(records: Mapping[str, object]) -> str:
    """List every market record's posture so market breadth stays visible."""

    briefs = [
        brief.replace("|", ",")
        for brief in (_market_posture_text(record) for record in records.values())
        if brief
    ]
    if not briefs:
        return ""
    return " // ".join(briefs)[:_MARKET_POSTURE_CHAR_BUDGET]


def _analysis_scope_digest(
    entry: EvidenceCoverageEntry,
    records: Mapping[str, object],
) -> str:
    """Encode one scope's complete aggregate in a compact, deterministic line."""

    facts = _numeric_aggregate_facts(records)
    source_value = ",".join(entry.providers[:3]) or "-"
    if len(entry.providers) > 3:
        source_value += f"+{len(entry.providers) - 3}"
    start = _compact_timestamp(entry.earliest_observed_at)
    end = _compact_timestamp(entry.latest_observed_at)
    times = start if start == end else f"{start}/{end}"
    parts = [
        f"asset={entry.asset}",
        f"capability={entry.capability.value}",
        f"accepted={entry.accepted_records}",
        f"excluded={entry.excluded_records}",
        f"providers={source_value}",
        f"timeframes={','.join(entry.timeframes) or '-'}",
        f"observed={times}",
    ]
    if facts:
        parts.append("aggregate=" + ",".join(facts))
    if entry.capability == ResearchCapability.NEWS:
        headlines = _news_headlines_brief(records)
        if headlines:
            parts.append("headlines=" + headlines)
    elif entry.capability == ResearchCapability.MARKET:
        posture = _market_posture_brief(records)
        if posture:
            parts.append("market=" + posture)
    if entry.limitations:
        parts.append("limitations=" + "; ".join(entry.limitations[:2]))
    return "|".join(parts)


def _numeric_aggregate_facts(records: Mapping[str, object]) -> list[str]:
    values: dict[str, list[float]] = {}
    for fact in evidence_numeric_facts(records):
        field = fact.path.rsplit(".", maxsplit=1)[-1]
        values.setdefault(field, []).append(fact.value)
    ordered = sorted(
        values,
        key=lambda field: (
            next(
                (index for index, term in enumerate(_AGGREGATE_FACT_PRIORITY) if term in field),
                len(_AGGREGATE_FACT_PRIORITY),
            ),
            field,
        ),
    )
    facts: list[str] = []
    for field in ordered[:3]:
        numbers = values[field]
        low = min(numbers)
        high = max(numbers)
        value = f"{low:g}" if low == high else f"{low:g}..{high:g}"
        facts.append(f"{field}={value}")
    return facts


def _quality_exclusions(
    inputs: AnalysisInputs,
) -> list[tuple[str, ResearchCapability, int, str | None]]:
    values: list[tuple[str, ResearchCapability, int, str | None]] = []
    markets: list[MarketEvidence] = []
    if inputs.market_result is not None:
        markets.extend(
            [
                inputs.market_result.market,
                *(
                    item.market
                    for item in inputs.market_result.contextual_timeframes
                    if item.market
                ),
            ]
        )
        values.extend(
            (
                inputs.market_result.market.symbol,
                ResearchCapability.MARKET,
                0,
                f"{item.timeframe}: {item.limitation}",
            )
            for item in inputs.market_result.contextual_timeframes
            if item.limitation
        )
    if inputs.market_comparison_result is not None:
        for item in inputs.market_comparison_result.assets:
            markets.extend(
                [
                    item.market,
                    *(context.market for context in item.contextual_timeframes if context.market),
                ]
            )
            values.extend(
                (
                    item.market.symbol,
                    ResearchCapability.MARKET,
                    0,
                    f"{context.timeframe}: {context.limitation}",
                )
                for context in item.contextual_timeframes
                if context.limitation
            )
    for market in markets:
        quality = market.data_quality
        excluded = sum(
            (
                quality.excluded_future,
                quality.excluded_incomplete,
                quality.excluded_malformed,
                quality.excluded_misaligned,
                quality.excluded_duplicates,
                quality.excluded_noncontiguous_prefix,
            )
        )
        values.append(
            (
                market.symbol,
                ResearchCapability.MARKET,
                excluded,
                "; ".join(quality.warnings) or None,
            )
        )
    return values


__all__ = [
    "build_analysis_evidence_digest",
    "build_complete_evidence_digest",
    "build_evidence_coverage_summary",
    "select_detailed_evidence",
]
