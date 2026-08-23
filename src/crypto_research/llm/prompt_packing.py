"""System prompts for routing and independent agent analysis."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from crypto_research.shared.json import dumps_strict
from crypto_research.shared.numeric_grounding import compact_evidence_for_llm

_MAX_ANALYSIS_PROMPT_BYTES = 6_500
SPECIALIST_ANALYSIS_PROMPT_BYTES = 4_000
_DEFAULT_SPECIALIST_EVIDENCE_BUNDLE_BYTES = 3_800
_MAX_PROMPT_LIST_ITEMS = 6
_MAX_PROMPT_TEXT_CHARS = 500
_MAX_EVIDENCE_DETAIL_ITEMS = 10
_PROMPT_EVIDENCE_DETAIL_ITEMS = 8
# News citations are prose-driven: only the top few relevant sources deserve
# detail room in a bounded prompt.  The full ledger stays in the scope digest.
_MAX_NEWS_DETAIL_PER_ASSET = 3
# News records need enough room for a headline and excerpt; the generic
# specialist limit (120) would otherwise leave only truncated metadata.
_NEWS_EVIDENCE_DETAIL_CHARS = 220
# The market narrative weighs indicators, so market/technical detail records
# order their numeric facts by narrative importance rather than alphabetical.
_MARKET_EVIDENCE_DETAIL_CHARS = 400
_MARKET_NUMERIC_FACTS = 8
# Market text (asset symbol and one window status) is secondary to the numeric
# posture; the trend line lives in the technical_calculation record instead.
_MARKET_TEXT_FACTS = 2
_MARKET_NUMERIC_PRIORITY = (
    "current_price",
    "change_24h_percent",
    "return_percent",
    "maximum_drawdown",
    "volatility",
)
_MIN_COMPACT_SCOPE_CHARS = 64
_DEFAULT_EVIDENCE_CAPABILITIES = [
    "news",
    "market",
    "fundamentals",
    "defi",
    "risk",
    "discovery",
    "derivatives",
]
# One synthesis pass over already-validated specialist answers.  The evidence
# index is bounded so specialist narratives stay within the role input budget.
_SCOPE_DIGEST_FORMAT = (
    "a=asset|c=capability|n=accepted/excluded|p=providers|t=timeframes|"
    "o=observed|g=aggregate|l=limitations|h=headlines|m=market; ~ marks a value "
    "compacted for this prompt"
)
_EVIDENCE_KIND_ORDER = (
    "market_snapshot",
    "recent_news",
    "project_fundamentals",
    "defi_protocol_metrics",
    "deterministic_risk_assessment",
    "technical_calculation",
    "derivatives_positioning",
    "market_screen",
)


@dataclass(frozen=True, slots=True)
class PromptEvidenceBundle:
    """The exact citation-ready subset that is sent to one analysis call."""

    available_evidence: Mapping[str, str]
    detailed_evidence_ids: tuple[str, ...]
    omitted_evidence_catalog: tuple[dict[str, object], ...]
    analysis_data_digest: Mapping[str, object] = field(default_factory=dict)
    budget_bytes: int = _MAX_ANALYSIS_PROMPT_BYTES
    serialized_bytes: int = 0
    specialist: bool = False


def build_prompt_evidence_bundle(
    payload: Mapping[str, object],
    evidence: Mapping[object, object],
    *,
    maximum: int = _PROMPT_EVIDENCE_DETAIL_ITEMS,
    analysis_data_digest: Mapping[str, object] | None = None,
    budget_bytes: int = _MAX_ANALYSIS_PROMPT_BYTES,
    fair: bool = False,
    detail_char_limit: int = 260,
    reserved_detail_bytes: int = 0,
    specialist: bool = False,
    market_depth: bool = False,
) -> PromptEvidenceBundle:
    """Pack balanced, compact facts before prompt serialization changes any IDs.

    The resulting IDs are also the only IDs that answer validation permits.  Everything not
    selected remains represented by the deterministic complete-data digest supplied by the caller.
    """

    if maximum <= 0:
        raise ValueError("maximum must be positive.")
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive.")
    if detail_char_limit < 80:
        raise ValueError("detail_char_limit must be at least 80.")
    if reserved_detail_bytes < 0 or reserved_detail_bytes >= budget_bytes:
        raise ValueError("reserved_detail_bytes must leave room in the prompt budget.")
    digest = _fit_analysis_data_digest(
        _compact_analysis_data_digest(analysis_data_digest or {}),
        budget_bytes=budget_bytes - reserved_detail_bytes,
    )
    candidates = (
        _fair_evidence_items(payload, evidence)
        if fair
        else _balanced_evidence_items(payload, evidence)
    )
    selected: list[tuple[object, object]] = []
    compacted: dict[str, str] = {}
    # Keep all scope digest lines.  Detail records then fill the remaining
    # budget in fair round-robin order, so no asset/scope silently disappears.
    for key, value in candidates[:maximum]:
        candidate = dict(compacted)
        candidate[str(key)] = _compact_evidence_value(
            str(key),
            value,
            limit=detail_char_limit,
            market_depth=market_depth,
        )
        candidate_ids = tuple(candidate)
        candidate_bundle = _prompt_bundle_value(
            candidate,
            candidate_ids,
            total_records=len(evidence),
            digest=digest,
            specialist=specialist,
        )
        if _prompt_bytes(dumps_strict(candidate_bundle)) > budget_bytes:
            continue
        selected.append((key, value))
        compacted = candidate

    selected_ids = tuple(str(key) for key, _ in selected)
    bundle_value = _prompt_bundle_value(
        compacted,
        selected_ids,
        total_records=len(evidence),
        digest=digest,
        specialist=specialist,
    )
    return PromptEvidenceBundle(
        available_evidence=compacted,
        detailed_evidence_ids=selected_ids,
        omitted_evidence_catalog=tuple(
            cast("list[dict[str, object]]", bundle_value["omitted_evidence_catalog"])
        ),
        analysis_data_digest=digest,
        budget_bytes=budget_bytes,
        serialized_bytes=_prompt_bytes(dumps_strict(bundle_value)),
        specialist=specialist,
    )


def build_specialist_analysis_payload(
    raw_result: Mapping[str, object],
    evidence: Mapping[object, object],
    complete_data_digest: Mapping[str, object] | None,
    *,
    budget_bytes: int = SPECIALIST_ANALYSIS_PROMPT_BYTES,
    maximum: int = _PROMPT_EVIDENCE_DETAIL_ITEMS,
    detail_char_limit: int = 120,
    bundle_budget_bytes: int = _DEFAULT_SPECIALIST_EVIDENCE_BUNDLE_BYTES,
) -> PromptEvidenceBundle:
    """Pre-pack the bounded evidence section for one specialist LLM call.

    This function deliberately accepts no raw source catalog.  It returns the
    same citation IDs used by the caller's validation and deterministic fallback,
    while the compact digest accounts for every accepted record internally.
    """

    return build_prompt_evidence_bundle(
        raw_result,
        evidence,
        maximum=maximum,
        analysis_data_digest=complete_data_digest,
        budget_bytes=min(budget_bytes, bundle_budget_bytes),
        fair=True,
        # Reserve enough room to give each requested asset a citation-ready
        # first-pass record while retaining every selected scope digest.
        detail_char_limit=detail_char_limit,
        reserved_detail_bytes=450,
        specialist=True,
        # The market narrative must weigh indicator numbers, so the market and
        # technical detail records keep more numeric room here than in shared
        # shared synthesis compaction.
        market_depth=True,
    )


def _prompt_bundle_value(
    evidence: Mapping[str, str],
    detailed_ids: tuple[str, ...],
    *,
    total_records: int,
    digest: Mapping[str, object],
    specialist: bool = False,
) -> dict[str, object]:
    omitted = max(total_records - len(detailed_ids), 0)
    return {
        "available_evidence": dict(evidence),
        "detailed_evidence_ids": list(detailed_ids),
        "omitted_evidence_catalog": (
            [
                {
                    "count": omitted,
                    "notice": (
                        "Every omitted detailed record remains represented in "
                        "complete_data_digest.accepted_evidence_catalog."
                        if not specialist
                        else "Additional validated records are represented in the scope digest."
                    ),
                }
            ]
            if omitted
            else []
        ),
        "analysis_data_digest": dict(digest),
    }


def _compact_analysis_data_digest(value: Mapping[str, object]) -> dict[str, object]:
    """Remove internal catalogs while retaining every scope's aggregate line."""

    scope_values = value.get("scope_digest")
    if isinstance(scope_values, list | tuple):
        # Do not use the generic list/text compactor here.  It would truncate a
        # later scope before the prompt packer has a chance to retain its asset,
        # capability, aggregate, and limitation fields.
        scopes = [str(item) for item in scope_values]
        return {
            "scope_digest": scopes,
            "total_accepted_records": _safe_count(value.get("total_accepted_records")),
            "total_excluded_records": _safe_count(value.get("total_excluded_records")),
        }
    return {}


def _fit_analysis_data_digest(
    digest: Mapping[str, object],
    *,
    budget_bytes: int,
) -> dict[str, object]:
    """Fit complete scope aggregates without dropping a selected scope."""

    scope_values = digest.get("scope_digest")
    if not isinstance(scope_values, list):
        return dict(digest)
    fitted = dict(digest)
    baseline = _prompt_bundle_value({}, (), total_records=0, digest=fitted)
    if _prompt_bytes(dumps_strict(baseline)) <= budget_bytes:
        return fitted
    if not scope_values:
        raise ValueError("The analysis digest exceeds the configured prompt budget.")

    # Preserve every scope but reduce each line symmetrically.  The guided
    # four-asset cap applies to assets, not capabilities: Research may contain
    # many legitimate asset/capability rows.  A scope-aware encoding retains each
    # identity plus aggregate and limitation information without blindly
    # truncating tail rows.
    static_digest = dict(fitted)
    static_digest["scope_digest"] = []
    static_digest["scope_digest_format"] = _SCOPE_DIGEST_FORMAT
    static_size = _prompt_bytes(
        dumps_strict(_prompt_bundle_value({}, (), total_records=0, digest=static_digest))
    )
    per_scope = max(
        _MIN_COMPACT_SCOPE_CHARS,
        (budget_bytes - static_size - 24 * len(scope_values)) // len(scope_values),
    )
    fitted["scope_digest"] = [
        _compact_scope_digest_line(str(item), per_scope) for item in scope_values
    ]
    fitted["scope_digest_format"] = _SCOPE_DIGEST_FORMAT
    final_value = _prompt_bundle_value({}, (), total_records=0, digest=fitted)
    if _prompt_bytes(dumps_strict(final_value)) > budget_bytes:
        # Account for UTF-8 and JSON escaping exactly, then retry every entry
        # with a smaller bounded representation.  This never removes a scope.
        available = max(
            _MIN_COMPACT_SCOPE_CHARS,
            (budget_bytes - static_size - 24 * len(scope_values)) // len(scope_values) - 1,
        )
        while available >= _MIN_COMPACT_SCOPE_CHARS:
            fitted["scope_digest"] = [
                _compact_scope_digest_line(str(item), available) for item in scope_values
            ]
            final_value = _prompt_bundle_value({}, (), total_records=0, digest=fitted)
            if _prompt_bytes(dumps_strict(final_value)) <= budget_bytes:
                return fitted
            available -= 1
        raise ValueError("The complete selected-scope digest exceeds the configured prompt budget.")
    return fitted


def _compact_scope_digest_line(value: str, limit: int) -> str:
    """Compact one coverage row without losing its semantic scope fields.

    Normal scope rows are readable ``key=value`` records.  Under a dense
    specialist request we switch to a documented compact form rather than a
    head/tail slice: all rows still carry an asset, capability, record counts,
    aggregate representation, and limitation representation.  ``~`` makes any
    shortened value explicit to the model instead of silently dropping it.
    """

    if _prompt_bytes(value) <= limit:
        return value
    fields = _scope_digest_fields(value)
    if not fields:
        return _compact_prompt_text(value, limit)

    asset = fields.get("asset") or fields.get("a") or "-"
    capability = fields.get("capability") or fields.get("c") or "-"
    counts = fields.get("n")
    if counts and "/" in counts:
        accepted, excluded = counts.split("/", 1)
    else:
        accepted = fields.get("accepted", "0")
        excluded = fields.get("excluded", "0")
    values = {
        "p": fields.get("providers", "-"),
        "t": fields.get("timeframes", "-"),
        "o": fields.get("observed", "-"),
        "g": fields.get("aggregate", "-"),
        "l": fields.get("limitations", "-"),
        "h": fields.get("headlines", "-"),
        "m": fields.get("market", "-"),
    }

    # Curated Guided assets/capabilities are short.  Still compact unknown
    # integrations defensively so the guaranteed scope fields fit as well.
    core = f"a={asset}|c={capability}|n={accepted}/{excluded}"
    minimum = f"{core}|p=-|t=-|o=-|g=-|l=-|h=-|m=-"
    if _prompt_bytes(minimum) > limit:
        core_budget = max(16, limit - _prompt_bytes("a=|c=|n=0/0|p=-|t=-|o=-|g=-|l=-|h=-|m=-"))
        asset_budget = max(4, core_budget // 2)
        asset = _compact_prompt_text(asset, asset_budget)
        capability = _compact_prompt_text(capability, max(4, core_budget - asset_budget))
        core = f"a={asset}|c={capability}|n={accepted}/{excluded}"

    # Reserve one visible value character for each populated field, then
    # prioritize aggregate and limitation text.  Provider/time information is
    # represented too, but cannot displace the two deterministic fields that
    # explain the scope outcome.  News headline briefs and market posture
    # briefs get a bounded share each so coverage breadth remains visible
    # without crowding the deterministic fields.
    prefix_bytes = _prompt_bytes(f"{core}|p=|t=|o=|g=|l=|h=|m=")
    available = max(0, limit - prefix_bytes)
    populated = [name for name, item in values.items() if item != "-"]
    allocations = {name: 1 if name in populated else 1 for name in values}
    remaining = max(0, available - sum(allocations.values()))
    posture_allocation = max(1, remaining // 5)
    rest = max(0, remaining - posture_allocation)
    headline_allocation = max(1, rest // 4)
    rest = max(0, rest - headline_allocation)
    aggregate_allocation = (rest + 1) // 2
    limitation_allocation = rest - aggregate_allocation
    allocations["m"] += posture_allocation
    allocations["h"] += headline_allocation
    allocations["g"] += aggregate_allocation
    allocations["l"] += limitation_allocation
    rendered = {
        name: _compact_scope_value(item, allocations[name]) for name, item in values.items()
    }
    result = (
        f"{core}|p={rendered['p']}|t={rendered['t']}|o={rendered['o']}|"
        f"g={rendered['g']}|l={rendered['l']}|h={rendered['h']}|m={rendered['m']}"
    )
    return _truncate_utf8(result, limit)


def _scope_digest_fields(value: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in value.split("|"):
        key, separator, field_value = item.partition("=")
        if separator and key:
            fields[key] = field_value
    return fields


def _compact_scope_value(value: str, limit: int) -> str:
    if value == "-":
        return "-"
    if limit <= 1:
        return "~"
    if _prompt_bytes(value) <= limit:
        return value
    if limit <= 4:
        return _truncate_utf8(value, limit)
    marker = "~"
    remaining = limit - _prompt_bytes(marker)
    head = max(1, remaining * 2 // 3)
    tail = max(0, remaining - head)
    return _truncate_utf8(value, head) + marker + _tail_utf8(value, tail)


def _safe_count(value: object) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def jsonable_prompt_value(value: object, *, field_name: str = "") -> Any:
    """Convert runtime values to bounded, JSON-safe prompt data."""

    if hasattr(value, "model_dump"):
        return jsonable_prompt_value(value.model_dump(mode="json"), field_name=field_name)
    if isinstance(value, Mapping):
        return {
            str(key): jsonable_prompt_value(item, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        if field_name == "entries":
            return [
                [jsonable_prompt_value(item) for item in entry]
                if isinstance(entry, list | tuple)
                else jsonable_prompt_value(entry)
                for entry in value
            ]
        if field_name in {
            "accepted_evidence_catalog",
            "accepted_evidence_ids",
            "detailed_evidence_ids",
            "entry_fields",
            "evidence_ids",
            # Scope digests are already budgeted by the specialist evidence
            # packer.  Applying the generic six-item list compactor here would
            # silently remove valid asset/capability rows after they were
            # selected, notably 3 assets x 6 Research scopes.
            "scope_digest",
            "summarized_evidence_ids",
        }:
            return [jsonable_prompt_value(item) for item in value]
        if field_name == "candles" and value:
            return {
                "count": len(value),
                "first": jsonable_prompt_value(value[0]),
                "last": jsonable_prompt_value(value[-1]),
                "_compacted": True,
            }
        if len(value) > _MAX_PROMPT_LIST_ITEMS:
            head_count = _MAX_PROMPT_LIST_ITEMS // 2
            tail_count = _MAX_PROMPT_LIST_ITEMS - head_count
            return {
                "count": len(value),
                "items": [
                    *(jsonable_prompt_value(item) for item in value[:head_count]),
                    *(jsonable_prompt_value(item) for item in value[-tail_count:]),
                ],
                "_compacted": True,
            }
        return [jsonable_prompt_value(item) for item in value]
    if isinstance(value, str) and len(value) > _MAX_PROMPT_TEXT_CHARS:
        return value[:_MAX_PROMPT_TEXT_CHARS] + "... [compacted]"
    return value


def contains_compaction_marker(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value.get("_compacted")) or any(
            contains_compaction_marker(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(contains_compaction_marker(item) for item in value)
    return False


def bounded_analysis_prompt(
    payload: Mapping[str, object],
    *,
    budget_bytes: int = _MAX_ANALYSIS_PROMPT_BYTES,
) -> str:
    """Serialize an analysis payload while retaining preselected evidence identities."""

    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive.")

    prompt = dumps_strict(payload)
    if _prompt_bytes(prompt) <= budget_bytes:
        return prompt

    reduced = dict(payload)
    raw_result = reduced.get("raw_result")
    collection_status = (
        raw_result.get("collection_status", {}) if isinstance(raw_result, Mapping) else {}
    )
    reduced["raw_result"] = {
        "_compacted": True,
        "notice": "The raw result exceeded the LLM payload budget.",
        "collection_status": {
            "requested_assets": (
                collection_status.get("requested_assets", [])
                if isinstance(collection_status, Mapping)
                else []
            ),
            "requested_capabilities": (
                collection_status.get("requested_capabilities", [])
                if isinstance(collection_status, Mapping)
                else []
            ),
        },
    }
    _append_processing_note(
        reduced,
        "The raw result was reduced further to stay within the provider payload limit; "
        "complete data remains available in the research interface.",
    )
    prompt = dumps_strict(reduced)
    if _prompt_bytes(prompt) <= budget_bytes:
        return prompt

    evidence = reduced.get("available_evidence")
    if isinstance(evidence, Mapping):
        preselected_ids = reduced.get("detailed_evidence_ids")
        selected_evidence = (
            [
                (item, evidence[item])
                for item in preselected_ids
                if isinstance(item, str) and item in evidence
            ]
            if isinstance(preselected_ids, list)
            else _balanced_evidence_items(reduced, evidence)
        )
        selected_ids = {str(key) for key, _ in selected_evidence}
        reduced["available_evidence"] = {
            str(key): (
                value if isinstance(value, str) else _compact_evidence_value(str(key), value)
            )
            for key, value in selected_evidence
        }
        reduced["detailed_evidence_ids"] = [str(key) for key, _ in selected_evidence]
        omitted_count = sum(str(key) not in selected_ids for key in evidence)
        reduced["omitted_evidence_catalog"] = (
            [
                {
                    "count": omitted_count,
                    "notice": "Additional validated records are represented in the scope digest.",
                }
            ]
            if omitted_count
            else []
        )
    _append_processing_note(
        reduced,
        "Evidence details were compacted to satisfy the provider "
        "payload limit; visible source records remain authoritative.",
    )
    prompt = dumps_strict(reduced)
    if _prompt_bytes(prompt) <= budget_bytes:
        return prompt

    reduced["output_contract"] = {
        "evidence": (
            "Cite detailed evidence IDs for exact claims; use the digest only for coverage."
        ),
        "safety": "Never give trading instructions, guarantees, or fabricated facts.",
    }
    _append_processing_note(
        reduced,
        (
            "The repeated output-contract text was reduced; system safety and citation rules "
            "remain in force."
        ),
    )
    prompt = dumps_strict(reduced)
    if _prompt_bytes(prompt) <= budget_bytes:
        return prompt

    evidence = reduced.get("available_evidence")
    if isinstance(evidence, Mapping):
        reduced["available_evidence"] = {
            str(key): _compact_prompt_text(str(value), 180) for key, value in evidence.items()
        }
    _append_processing_note(
        reduced,
        "Citation-ready evidence was shortened but its identifiers and factual fields were kept.",
    )
    prompt = dumps_strict(reduced)
    if _prompt_bytes(prompt) <= budget_bytes:
        return prompt

    raw_limitations = reduced.get("known_limitations", [])
    limitation_values = raw_limitations if isinstance(raw_limitations, list) else []
    reduced["known_limitations"] = [
        _compact_prompt_text(str(item), 240) for item in limitation_values[:6]
    ]
    raw_notes = reduced.get("processing_notes", [])
    note_values = raw_notes if isinstance(raw_notes, list) else []
    reduced["processing_notes"] = [_compact_prompt_text(str(item), 200) for item in note_values[:3]]
    prompt = dumps_strict(reduced)
    if _prompt_bytes(prompt) <= budget_bytes:
        return prompt

    complete_digest = reduced.get("complete_data_digest")
    if isinstance(complete_digest, Mapping):
        # Re-fit the complete digest to its compact scope form.  Every selected
        # scope row keeps its asset, capability, aggregate, and limitation
        # fields; only the envelope shrinks.
        for digest_budget in range(2500, 1100, -100):
            try:
                fitted_digest = _fit_analysis_data_digest(
                    complete_digest,
                    budget_bytes=digest_budget,
                )
            except ValueError:
                continue
            refit = dict(reduced)
            refit["complete_data_digest"] = fitted_digest
            _append_processing_note(
                refit,
                "The complete data digest was compressed to its compact scope form; every "
                "selected asset/capability row and its aggregate totals are preserved.",
            )
            prompt = dumps_strict(refit)
            if _prompt_bytes(prompt) <= budget_bytes:
                return prompt
    raise ValueError(
        "The exact question and preselected evidence identifiers exceed the safe analysis "
        "prompt budget."
    )


def _prompt_bytes(value: str) -> int:
    return len(value.encode("utf-8"))


def _compact_prompt_text(value: str, limit: int) -> str:
    if _prompt_bytes(value) <= limit:
        return value
    if limit <= 4:
        return _truncate_utf8(value, limit)
    marker = " ...[compacted]... "
    remaining = max(1, limit - _prompt_bytes(marker))
    head = max(1, remaining * 2 // 3)
    tail = max(0, remaining - head)
    return _truncate_utf8(value, head) + marker + _tail_utf8(value, tail)


def _truncate_utf8(value: str, limit: int) -> str:
    """Return the largest valid UTF-8 prefix within ``limit`` bytes."""

    if limit <= 0:
        return ""
    return value.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def _tail_utf8(value: str, limit: int) -> str:
    """Return the largest valid UTF-8 suffix within ``limit`` bytes."""

    if limit <= 0:
        return ""
    encoded = value.encode("utf-8")
    return encoded[-limit:].decode("utf-8", errors="ignore")


def _append_processing_note(payload: dict[str, object], message: str) -> None:
    raw_notes = payload.get("processing_notes", [])
    notes = list(raw_notes) if isinstance(raw_notes, list) else []
    notes.append(message)
    payload["processing_notes"] = list(dict.fromkeys(str(item) for item in notes))


def _balanced_evidence_items(
    payload: Mapping[str, object],
    evidence: Mapping[object, object],
) -> list[tuple[object, object]]:
    """Return a stable evidence ordering for compact specialist prompts."""

    items = list(evidence.items())
    requested_assets = _requested_asset_symbols(payload)
    selected: list[tuple[object, object]] = []
    for asset in requested_assets:
        for claim_type in _EVIDENCE_KIND_ORDER:
            match = next(
                (
                    item
                    for item in items
                    if item not in selected
                    and _evidence_matches_asset(item, asset)
                    and isinstance(item[1], Mapping)
                    and item[1].get("claim_type") == claim_type
                ),
                None,
            )
            if match is not None:
                selected.append(match)
    for claim_type in _EVIDENCE_KIND_ORDER:
        match = next(
            (
                item
                for item in items
                if item not in selected
                and isinstance(item[1], Mapping)
                and item[1].get("claim_type") == claim_type
            ),
            None,
        )
        if match is not None:
            selected.append(match)
    selected.extend(item for item in items if item not in selected)
    return _cap_news_details(selected)[:_MAX_EVIDENCE_DETAIL_ITEMS]


def _cap_news_details(
    items: list[tuple[object, object]],
) -> list[tuple[object, object]]:
    """Keep only the top relevant news citations per asset in analysis prompts.

    News items are already ordered by provider quality and freshness before
    they reach the evidence ledger, so the first few per asset are the most
    relevant.  The complete news ledger is still accounted for in the digest.
    """

    counts: dict[str, int] = {}
    capped: list[tuple[object, object]] = []
    for item in items:
        if _evidence_capability(item) == "news":
            asset = _evidence_asset(item)
            if counts.get(asset, 0) >= _MAX_NEWS_DETAIL_PER_ASSET:
                continue
            counts[asset] = counts.get(asset, 0) + 1
        capped.append(item)
    return capped


def _fair_evidence_items(
    payload: Mapping[str, object],
    evidence: Mapping[object, object],
) -> list[tuple[object, object]]:
    items = list(evidence.items())
    requested_assets = _requested_asset_symbols(payload)
    requested_capabilities = _requested_capabilities(payload)
    grouped: dict[tuple[str, str], list[tuple[object, object]]] = {}

    # If requested_capabilities is empty, use the capabilities from the evidence
    capabilities = requested_capabilities or _DEFAULT_EVIDENCE_CAPABILITIES

    # Create groups for requested assets and capabilities
    for capability in capabilities:
        for asset in requested_assets or ("global",):
            grouped.setdefault((asset, capability), [])

    # Assign evidence items to groups
    for item in items:
        asset = _evidence_asset(item)
        capability = _evidence_capability(item)
        group = (asset, capability)
        grouped.setdefault(group, []).append(item)

    # Sort groups and select items in round-robin fashion
    for group in grouped:
        grouped[group].sort(key=_evidence_detail_sort_key)
        if group[1] == "news":
            grouped[group] = grouped[group][:_MAX_NEWS_DETAIL_PER_ASSET]

    selected: list[tuple[object, object]] = []
    maximum_group_size = max((len(values) for values in grouped.values()), default=0)
    for index in range(maximum_group_size):
        selected.extend(records[index] for records in grouped.values() if index < len(records))
    return selected[:_MAX_EVIDENCE_DETAIL_ITEMS]


def _requested_asset_symbols(payload: Mapping[str, object]) -> tuple[str, ...]:
    status = payload.get("collection_status")
    if not isinstance(status, Mapping):
        status = payload.get("raw_result")
        status = status.get("collection_status") if isinstance(status, Mapping) else None
    assets = status.get("requested_assets", []) if isinstance(status, Mapping) else []
    return tuple(
        str(asset.get("symbol", "")).casefold()
        for asset in assets
        if isinstance(asset, Mapping) and str(asset.get("symbol", "")).strip()
    )


def _requested_capabilities(payload: Mapping[str, object]) -> tuple[str, ...]:
    status = payload.get("collection_status")
    if not isinstance(status, Mapping):
        status = payload.get("raw_result")
        status = status.get("collection_status") if isinstance(status, Mapping) else None
    values = status.get("requested_capabilities", []) if isinstance(status, Mapping) else []
    return tuple(str(value).casefold() for value in values if str(value).strip())


def _evidence_asset(item: tuple[object, object]) -> str:
    record = item[1]
    asset = str(record.get("asset", "")) if isinstance(record, Mapping) else ""
    return asset.casefold() or "global"


def _evidence_capability(item: tuple[object, object]) -> str:
    record = item[1]
    if not isinstance(record, Mapping):
        return "unknown"
    payload = record.get("payload")
    if isinstance(payload, Mapping) and isinstance(payload.get("research_capability"), str):
        return str(payload["research_capability"]).casefold()
    claim_type = str(record.get("claim_type", "unknown"))
    return {
        "market_screen": "discovery",
        "market_snapshot": "market",
        "technical_calculation": "market",
        "derivatives_positioning": "derivatives",
        "recent_news": "news",
        "project_fundamentals": "fundamentals",
        "defi_protocol_metrics": "defi",
        "onchain_activity": "onchain",
        "deterministic_risk_assessment": "risk",
    }.get(claim_type, claim_type.casefold())


def _evidence_detail_sort_key(item: tuple[object, object]) -> tuple[int, int, int]:
    evidence_id, record = item
    if not isinstance(record, Mapping):
        return (3, 0, _evidence_index(evidence_id))
    claim_type = str(record.get("claim_type", ""))
    type_rank = (
        _EVIDENCE_KIND_ORDER.index(claim_type)
        if claim_type in _EVIDENCE_KIND_ORDER
        else len(_EVIDENCE_KIND_ORDER)
    )
    payload = record.get("payload")
    rank = payload.get("rank") if isinstance(payload, Mapping) else None
    candidate_rank = rank if isinstance(rank, int) and rank > 0 else 0
    return (type_rank, candidate_rank, _evidence_index(evidence_id))


def _evidence_index(evidence_id: object) -> int:
    prefix, separator, suffix = str(evidence_id).rpartition(".")
    if separator and suffix.isdecimal():
        return int(suffix)
    return 0


def _evidence_matches_asset(item: tuple[object, object], symbol: str) -> bool:
    key = re.sub(r"[^a-z0-9]+", "-", symbol).strip("-")
    record = item[1]
    asset = str(record.get("asset", "")).casefold() if isinstance(record, Mapping) else ""
    return (
        symbol in str(item[0]).casefold()
        or key in str(item[0]).casefold()
        or symbol == asset
        or key == re.sub(r"[^a-z0-9]+", "-", asset).strip("-")
    )


def _compact_evidence_value(
    evidence_id: str,
    value: object,
    *,
    limit: int = 320,
    market_depth: bool = False,
) -> str:
    record = value if isinstance(value, Mapping) else {}
    if record.get("claim_type") == "recent_news":
        # Headlines and excerpts are the analysis content for news; give them
        # enough room to survive the compact specialist envelope.
        return compact_evidence_for_llm(
            {evidence_id: value},
            max_text_chars=max(limit, _NEWS_EVIDENCE_DETAIL_CHARS),
            max_numeric_facts=2,
            max_text_facts=2,
            text_fact_chars=90,
        )[evidence_id]
    if market_depth and record.get("claim_type") in {
        "market_snapshot",
        "technical_calculation",
        "derivatives_positioning",
    }:
        # Market posture and indicator records carry the numbers the narrative
        # must weigh (drawdown, volatility, MACD, ATR, returns).  Give them
        # more numeric room than generic records while keeping the bounded
        # specialist envelope intact.
        return compact_evidence_for_llm(
            {evidence_id: value},
            max_text_chars=max(limit, _MARKET_EVIDENCE_DETAIL_CHARS),
            max_numeric_facts=_MARKET_NUMERIC_FACTS,
            max_text_facts=_MARKET_TEXT_FACTS,
            text_fact_chars=80,
            numeric_priority=_MARKET_NUMERIC_PRIORITY,
        )[evidence_id]
    return compact_evidence_for_llm(
        {evidence_id: value},
        max_text_chars=limit,
        max_numeric_facts=5,
        max_text_facts=3,
        text_fact_chars=80,
    )[evidence_id]


__all__ = [
    "SPECIALIST_ANALYSIS_PROMPT_BYTES",
    "bounded_analysis_prompt",
    "build_prompt_evidence_bundle",
    "build_specialist_analysis_payload",
    "contains_compaction_marker",
    "jsonable_prompt_value",
    "PromptEvidenceBundle",
]
