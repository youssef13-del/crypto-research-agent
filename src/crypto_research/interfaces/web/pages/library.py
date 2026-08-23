"""Durable research history, multi-run management, and deterministic exports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

import pandas as pd
import streamlit as st

from crypto_research.bootstrap import load_research_repository
from crypto_research.domain.history import (
    BulkDeleteResult,
    ResearchComparison,
    ResearchRunSummary,
    StoredResearchRun,
)
from crypto_research.domain.research import ResearchReport
from crypto_research.interfaces.web import runtime
from crypto_research.interfaces.web.components.layout import render_empty_panel, render_page_header
from crypto_research.interfaces.web.components.research import render_research_response
from crypto_research.interfaces.web.presentation import build_research_presentation
from crypto_research.orchestration.planning import capability_options

_TABLE_KEY = "research-library-table"
_RESET_SELECTION_KEY = "research-library-reset-selection"
_FLASH_KEY = "research-library-flash"
_PRUNED_AT_KEY = "research-library-pruned-at"
_TRACKED_FIELDS = frozenset(
    {
        "band",
        "change_1d",
        "change_7d",
        "current_price",
        "latest_value",
        "market_cap",
        "mae",
        "prediction",
        "predicted_price",
        "predicted_return",
        "rank",
        "score",
        "source_state",
        "status",
        "tvl_usd",
        "directional_accuracy",
    }
)


class ResearchLibraryRepository(Protocol):
    def list_runs(
        self,
        *,
        asset: str | None = None,
        capability: str | None = None,
        state: str | None = None,
        limit: int = 100,
    ) -> list[ResearchRunSummary]: ...

    def get_run(self, run_id: str) -> StoredResearchRun | None: ...

    def compare(self, run_id: str) -> ResearchComparison | None: ...

    def pin(self, run_id: str, *, pinned: bool) -> bool: ...

    def delete(self, run_id: str) -> bool: ...

    def delete_many(self, run_ids: Sequence[str]) -> BulkDeleteResult: ...

    def prune(self, *, retention_days: int = 365, now: datetime | None = None) -> int: ...

    def prune_cache(self, *, now: datetime | None = None) -> int: ...


def library_page() -> None:
    _consume_selection_reset()
    render_page_header(
        "Library",
        "Research Library",
        "Open, compare, export, pin, or delete reports saved to your account.",
    )
    _render_flash()
    try:
        owner_id = runtime.current_owner_id()
    except RuntimeError:
        st.info("Research Library is available after authenticated sign-in.")
        return
    settings = runtime.load_runtime_settings()
    base_repository = load_research_repository(
        settings.database_url,
        settings.research_retention_days,
    )
    if base_repository is None:
        st.error("Research history is unavailable. Try again after storage access is restored.")
        return
    repository: ResearchLibraryRepository = base_repository.for_owner(owner_id)
    _prune_once(repository, retention_days=settings.research_retention_days)
    summaries = _filtered_summaries(repository)
    if not summaries:
        render_empty_panel(
            "No matching saved research",
            "Completed Guided Research runs appear here automatically. Adjust the filters "
            "if your library already contains reports.",
            icon="R",
        )
        return

    _render_selection_controls(summaries)
    event = st.dataframe(
        _library_frame(summaries),
        key=_TABLE_KEY,
        width="stretch",
        height=min(475, 52 + len(summaries) * 35),
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        column_config={
            "Pinned": st.column_config.CheckboxColumn("Pinned", width="small"),
            "Assets": st.column_config.TextColumn("Assets", width="medium"),
            "Topics": st.column_config.TextColumn("Topics", width="large"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Exchange": st.column_config.TextColumn("Exchange", width="small"),
            "Timeframe": st.column_config.TextColumn("Timeframe", width="small"),
            "Evidence": st.column_config.NumberColumn("Evidence", width="small", format="%d"),
            "Completed": st.column_config.DatetimeColumn(
                "Completed",
                width="medium",
                format="D MMM YYYY, HH:mm",
                timezone="UTC",
            ),
        },
    )
    selected = selected_run_summaries(summaries, event["selection"]["rows"])
    st.caption(f"{len(selected)} of {len(summaries)} filtered reports selected.")
    if not selected:
        st.info("Select one report to reopen it, or select several for a combined PDF or deletion.")
        return
    if len(selected) == 1:
        _render_single_selection(repository, selected[0])
        return
    _render_multi_selection(repository, selected)


def _filtered_summaries(repository: ResearchLibraryRepository) -> list[ResearchRunSummary]:
    with st.container(border=True):
        st.markdown("#### Filters")
        filters = st.columns([1.1, 1, 1])
        asset = filters[0].text_input(
            "Asset",
            placeholder="BTC/USD",
            key="research-library-asset-filter",
            on_change=_request_selection_reset,
        )
        capability_labels = {"All topics": ""}
        capability_labels.update(
            {label: capability.value for capability, label, _ in capability_options()}
        )
        capability_label = filters[1].selectbox(
            "Topic",
            tuple(capability_labels),
            key="research-library-topic-filter",
            on_change=_request_selection_reset,
        )
        state_label = filters[2].selectbox(
            "Status",
            ("All", "Complete", "Partial", "Failed"),
            key="research-library-status-filter",
            on_change=_request_selection_reset,
        )
    return repository.list_runs(
        asset=asset or None,
        capability=capability_labels[capability_label] or None,
        state=None if state_label == "All" else state_label.casefold(),
        limit=100,
    )


def _render_selection_controls(summaries: Sequence[ResearchRunSummary]) -> None:
    controls = st.columns([1, 1, 4])
    if controls[0].button(
        "Select filtered",
        icon=":material/select_all:",
        width="stretch",
    ):
        st.session_state[_TABLE_KEY] = {"selection": {"rows": list(range(len(summaries)))}}
        st.rerun()
    if controls[1].button(
        "Clear selection",
        icon=":material/deselect:",
        width="stretch",
    ):
        st.session_state[_TABLE_KEY] = {"selection": {"rows": []}}
        st.rerun()
    controls[2].caption(
        "Pinned reports remain selectable for viewing and export, but are protected from "
        "bulk deletion."
    )


def _render_single_selection(
    repository: ResearchLibraryRepository,
    summary: ResearchRunSummary,
) -> None:
    stored = repository.get_run(summary.id)
    if stored is None:
        st.warning("This report could not be reopened because its stored payload is incompatible.")
        return
    st.markdown("### Selected report")
    _render_single_toolbar(repository, stored)
    _render_run_metadata(stored.summary)
    _render_comparison(repository.compare(summary.id))
    route = tuple(item.agent for item in stored.report.agent_statuses)
    presentation = build_research_presentation(stored.report, route=route)
    render_research_response("Saved research report.", presentation)


def _render_single_toolbar(
    repository: ResearchLibraryRepository,
    stored: StoredResearchRun,
) -> None:
    summary = stored.summary
    actions = st.columns([1, 1, 1, 1, 2.2])
    pin_label = "Unpin" if summary.pinned else "Pin"
    pin_icon = ":material/keep_off:" if summary.pinned else ":material/keep:"
    if actions[0].button(pin_label, icon=pin_icon, key=f"library-pin-{summary.id}"):
        repository.pin(summary.id, pinned=not summary.pinned)
        st.rerun()
    actions[1].download_button(
        "JSON",
        data=json.dumps(stored.report.model_dump(mode="json"), indent=2),
        file_name=f"chainscope-{summary.id}.json",
        mime="application/json",
        icon=":material/data_object:",
        key=f"library-json-{summary.id}",
    )
    _render_pdf_download(actions[2], (stored,), key=f"library-pdf-{summary.id}")
    confirm = actions[4].checkbox("Confirm deletion", key=f"library-confirm-{summary.id}")
    if actions[3].button(
        "Delete",
        icon=":material/delete:",
        disabled=not confirm,
        key=f"library-delete-{summary.id}",
    ):
        if repository.delete(summary.id):
            _reset_after_mutation("Deleted 1 saved research report.")
            st.rerun()
        st.error("The selected report could not be deleted.")


def _render_multi_selection(
    repository: ResearchLibraryRepository,
    selected: tuple[ResearchRunSummary, ...],
) -> None:
    st.markdown("### Selected reports")
    pinned_count = sum(item.pinned for item in selected)
    metrics = st.columns(3)
    metrics[0].metric("Selected", len(selected))
    metrics[1].metric("Pinned", pinned_count)
    metrics[2].metric("Evidence records", sum(item.evidence_count for item in selected))
    actions = st.columns([1.4, 1, 3])
    stored = tuple(
        report
        for summary in selected[:20]
        if (report := repository.get_run(summary.id)) is not None
    )
    if len(selected) <= 20 and len(stored) == len(selected):
        _render_pdf_download(actions[0], stored, key="library-combined-pdf")
    else:
        actions[0].button(
            "Combined PDF",
            icon=":material/picture_as_pdf:",
            disabled=True,
            width="stretch",
        )
        actions[0].caption("Select at most 20 compatible reports to create one combined PDF.")
    delete_disabled = pinned_count > 0
    if actions[1].button(
        "Delete selected",
        icon=":material/delete_sweep:",
        type="primary",
        disabled=delete_disabled,
        width="stretch",
    ):
        _confirm_bulk_delete(repository, selected)
    if delete_disabled:
        actions[2].warning(
            f"{pinned_count} selected report(s) are pinned. Unpin them before bulk deletion."
        )
    else:
        actions[2].caption(
            "Bulk deletion removes the selected reports and their evidence snapshots in one action."
        )


def _render_pdf_download(container: Any, runs: tuple[StoredResearchRun, ...], *, key: str) -> None:
    workspace = runtime.current_workspace()
    prepared_for = workspace.profile.effective_name if workspace is not None else None
    try:
        from crypto_research.interfaces.web.pdf_report import research_pdf_filename

        data = _cached_pdf(runs, prepared_for, datetime.now(UTC).date().isoformat())
        filename = research_pdf_filename(runs)
    except Exception:
        container.button(
            "PDF",
            icon=":material/picture_as_pdf:",
            disabled=True,
            width="stretch",
            key=f"{key}-unavailable",
        )
        container.caption("PDF generation is temporarily unavailable.")
        return
    container.download_button(
        "Combined PDF" if len(runs) > 1 else "PDF",
        data=data,
        file_name=filename,
        mime="application/pdf",
        icon=":material/picture_as_pdf:",
        width="stretch",
        key=key,
    )


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_pdf(
    runs: tuple[StoredResearchRun, ...],
    prepared_for: str | None,
    export_date: str,
) -> bytes:
    from crypto_research.interfaces.web.pdf_report import render_research_pdf

    del export_date
    return render_research_pdf(runs, prepared_for=prepared_for)


@st.dialog("Delete selected research", width="small")
def _confirm_bulk_delete(
    repository: ResearchLibraryRepository,
    selected: tuple[ResearchRunSummary, ...],
) -> None:
    count = len(selected)
    phrase = bulk_delete_confirmation(count)
    st.warning(
        f"This permanently deletes {count} saved reports and their evidence snapshots. "
        "This action cannot be undone."
    )
    st.caption("Selected: " + ", ".join(_short_run_label(item) for item in selected[:8]))
    if count > 8:
        st.caption(f"…and {count - 8} more reports.")
    signature = hashlib.sha256("|".join(item.id for item in selected).encode()).hexdigest()[:12]
    confirmation = st.text_input(
        f"Type {phrase} to confirm",
        key=f"library-bulk-confirmation-{signature}",
    )
    if not st.button(
        "Permanently delete reports",
        type="primary",
        icon=":material/delete_forever:",
        disabled=confirmation != phrase,
        width="stretch",
    ):
        return
    result = repository.delete_many(tuple(item.id for item in selected))
    if result.protected_count:
        st.error("Pinned reports were protected. Unpin them and try again.")
        return
    if result.deleted_count != count:
        st.warning(
            f"Deleted {result.deleted_count} of {count} selected reports. "
            "Some reports were no longer available."
        )
    _reset_after_mutation(f"Deleted {result.deleted_count} saved research report(s).")
    st.rerun()


def _render_run_metadata(summary: ResearchRunSummary) -> None:
    completed = summary.completed_at or summary.created_at
    values = st.columns(4)
    values[0].metric("Assets", ", ".join(summary.assets) or "Market discovery")
    values[1].metric("Topics", str(len(summary.capabilities)))
    values[2].metric("Evidence records", str(summary.evidence_count))
    values[3].metric("Completed", completed.strftime("%d %b %Y"))
    settings = " / ".join(item for item in (summary.exchange, summary.timeframe) if item)
    st.caption(
        f"{summary.state.title()} · {settings or 'Topic-only research'} · "
        f"{completed:%d %b %Y %H:%M UTC}"
    )


def _render_comparison(comparison: ResearchComparison | None) -> None:
    if comparison is None or comparison.previous is None:
        st.caption("No earlier run has the same assets, topics, exchange, and timeframe.")
        return
    changes = compare_reports(comparison.current.report, comparison.previous.report)
    with st.expander("Changes since the previous matching run", expanded=False):
        if not changes:
            st.caption("No tracked research fields changed.")
            return
        for label, previous, current in changes[:40]:
            st.markdown(f"**{label}:** `{previous}` → `{current}`")


def _library_frame(summaries: Sequence[ResearchRunSummary]) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "Pinned": summary.pinned,
                "Assets": ", ".join(summary.assets) or "Market discovery",
                "Topics": ", ".join(item.replace("_", " ").title() for item in summary.capabilities)
                or "General research",
                "Status": summary.state.title(),
                "Exchange": (summary.exchange or "—").title(),
                "Timeframe": summary.timeframe or "—",
                "Evidence": summary.evidence_count,
                "Completed": summary.completed_at or summary.created_at,
            }
            for summary in summaries
        ]
    )


def selected_run_summaries(
    summaries: Sequence[ResearchRunSummary],
    selected_rows: Sequence[int],
) -> tuple[ResearchRunSummary, ...]:
    """Map stable dataframe row positions back to stored summaries."""

    selected: list[ResearchRunSummary] = []
    seen: set[int] = set()
    for row in selected_rows:
        if row in seen or row < 0 or row >= len(summaries):
            continue
        seen.add(row)
        selected.append(summaries[row])
    return tuple(selected)


def bulk_delete_confirmation(count: int) -> str:
    if count < 2 or count > 100:
        raise ValueError("Bulk deletion requires between 2 and 100 reports.")
    return f"DELETE {count} REPORTS"


def compare_reports(
    current: ResearchReport,
    previous: ResearchReport,
) -> list[tuple[str, str, str]]:
    current_values = _tracked_values(current.model_dump(mode="json"))
    previous_values = _tracked_values(previous.model_dump(mode="json"))
    changes: list[tuple[str, str, str]] = []
    for path in sorted(current_values.keys() | previous_values.keys()):
        before = previous_values.get(path, "Unavailable")
        after = current_values.get(path, "Unavailable")
        if before != after:
            changes.append((_humanize_path(path), str(before), str(after)))
    return changes


def _tracked_values(value: object, *, path: str = "") -> dict[str, object]:
    values: dict[str, object] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in _TRACKED_FIELDS and isinstance(item, str | int | float | bool):
                values[child_path] = item
            if key in {"items", "sources"} and isinstance(item, list):
                titles = sorted(
                    str(entry["title"])
                    for entry in item
                    if isinstance(entry, Mapping) and isinstance(entry.get("title"), str)
                )
                values[f"{child_path}.titles"] = " | ".join(titles)
            values.update(_tracked_values(item, path=child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            values.update(_tracked_values(item, path=f"{path}[{index}]"))
    return values


def _humanize_path(path: str) -> str:
    return path.replace("_", " ").replace(".", " / ").title()


def _short_run_label(summary: ResearchRunSummary) -> str:
    assets = ", ".join(summary.assets) or "Market discovery"
    return f"{assets} ({summary.created_at:%d %b %Y})"


def _request_selection_reset() -> None:
    st.session_state[_RESET_SELECTION_KEY] = True


def _consume_selection_reset() -> None:
    if st.session_state.pop(_RESET_SELECTION_KEY, False):
        st.session_state.pop(_TABLE_KEY, None)


def _reset_after_mutation(message: str) -> None:
    st.session_state[_RESET_SELECTION_KEY] = True
    st.session_state[_FLASH_KEY] = message
    _cached_pdf.clear()


def _render_flash() -> None:
    message = st.session_state.pop(_FLASH_KEY, None)
    if isinstance(message, str) and message:
        st.success(message)


def _prune_once(repository: ResearchLibraryRepository, *, retention_days: int) -> None:
    today = datetime.now(UTC).date().isoformat()
    if st.session_state.get(_PRUNED_AT_KEY) == today:
        return
    repository.prune(retention_days=retention_days)
    repository.prune_cache()
    st.session_state[_PRUNED_AT_KEY] = today


__all__ = [
    "bulk_delete_confirmation",
    "compare_reports",
    "library_page",
    "selected_run_summaries",
]
