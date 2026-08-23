"""Authenticated identity, usage, and workspace controls."""

from __future__ import annotations

import html
from typing import Protocol, cast

import streamlit as st

from crypto_research.bootstrap import load_research_repository
from crypto_research.domain.account import UserWorkspace
from crypto_research.interfaces.web import runtime
from crypto_research.interfaces.web.auth import logout
from crypto_research.interfaces.web.components.layout import (
    render_page_header,
    render_section_header,
)

_DELETE_CONFIRMATION = "DELETE MY WORKSPACE"


class AccountRepositoryProtocol(Protocol):
    def get_workspace(self) -> UserWorkspace | None: ...

    def delete_workspace(self) -> bool: ...


def account_page() -> None:
    try:
        owner_id = runtime.current_owner_id()
    except RuntimeError:
        st.info("Account settings are available after authenticated sign-in.")
        return
    repository = _repository(owner_id)
    if repository is None:
        return
    fresh = repository.get_workspace()
    if fresh is None:
        st.error("Your workspace could not be loaded. Sign out, then sign in again.")
        return
    runtime.update_current_workspace(fresh)
    render_page_header(
        "Account",
        "Account settings",
        "Review your verified identity, research activity, and workspace controls.",
    )
    _render_identity(fresh)
    _render_danger_zone(repository)


def _render_identity(workspace: UserWorkspace) -> None:
    profile = workspace.profile
    columns = st.columns([1, 4])
    with columns[0]:
        if profile.avatar_url:
            st.image(profile.avatar_url, width=88)
        else:
            st.markdown(
                '<div class="cs-profile-avatar">'
                f"{html.escape(profile.effective_name[:1].upper())}</div>",
                unsafe_allow_html=True,
            )
    with columns[1]:
        st.markdown(f"### {html.escape(profile.effective_name)}")
        st.markdown(
            '<span class="cs-profile-verification">✓ Verified email</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"{profile.email} · {profile.provider.title()} identity")
        st.caption(f"Workspace created {profile.created_at:%d %b %Y}")
    stats = workspace.stats
    metrics = st.columns(4)
    metrics[0].metric("Saved research", stats.research_runs)
    metrics[1].metric("Pinned", stats.pinned_runs)
    metrics[2].metric("Evidence records", stats.evidence_records)
    metrics[3].metric(
        "Last research",
        stats.last_research_at.strftime("%d %b") if stats.last_research_at else "Not yet",
    )


def _render_danger_zone(repository: AccountRepositoryProtocol) -> None:
    render_section_header(
        "Workspace controls",
        "Signing out preserves your workspace. Deletion permanently removes ChainScope data.",
    )
    if st.button("Sign out", icon=":material/logout:"):
        logout()
    with st.expander("Delete ChainScope workspace", expanded=False):
        st.warning(
            "This permanently deletes your saved research, evidence snapshots, watchlist, "
            "preferences, and ChainScope profile. Your Auth0 identity is not deleted."
        )
        confirmation = st.text_input(
            f"Type {_DELETE_CONFIRMATION} to confirm",
            key="account-delete-confirmation",
        )
        if st.button(
            "Delete workspace data",
            type="primary",
            disabled=confirmation != _DELETE_CONFIRMATION,
            icon=":material/delete_forever:",
        ):
            if repository.delete_workspace():
                logout()
                return
            st.error("The workspace could not be deleted.")


def _repository(owner_id: str) -> AccountRepositoryProtocol | None:
    settings = runtime.load_runtime_settings()
    repository = load_research_repository(
        settings.database_url,
        settings.research_retention_days,
    )
    if repository is None:
        st.error("Account storage is unavailable. Try again after database access is restored.")
        return None
    return cast(AccountRepositoryProtocol, repository.for_owner(owner_id))


__all__ = ["account_page"]
