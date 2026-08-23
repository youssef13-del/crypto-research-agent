"""ChainScope Streamlit application shell and native page navigation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st
from pydantic import ValidationError

from crypto_research.bootstrap import load_research_repository
from crypto_research.config import Settings
from crypto_research.domain.account import UserIdentity
from crypto_research.interfaces.web import runtime
from crypto_research.interfaces.web.auth import authenticated_identity, login_page, signup_page
from crypto_research.interfaces.web.components.layout import (
    apply_theme,
)
from crypto_research.interfaces.web.components.shell import render_sidebar
from crypto_research.interfaces.web.pages.account import account_page
from crypto_research.interfaces.web.pages.dashboard import dashboard_page
from crypto_research.interfaces.web.pages.home import home_page
from crypto_research.interfaces.web.pages.library import library_page
from crypto_research.interfaces.web.pages.research import research_page
from crypto_research.interfaces.web.runtime import initialize_state
from crypto_research.interfaces.web.theme import initialize_theme_state

if TYPE_CHECKING:
    from streamlit.navigation.page import StreamlitPage

_AUTH_BINDING_STATE_KEY = "current_oidc_identity_binding"


def main() -> None:
    st.set_page_config(
        page_title="ChainScope | Crypto research workspace",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="auto",
    )
    initialize_state()
    initialize_theme_state()
    apply_theme()
    try:
        settings = runtime.load_runtime_settings()
    except ValidationError:
        st.error("The application configuration is invalid. Review the configured settings.")
        return
    identity = authenticated_identity()
    if identity is None:
        navigation = st.navigation(_signed_out_pages(), position="hidden")
        navigation.run()
        return
    if not _initialize_access(settings, identity):
        return
    render_sidebar(settings)

    pages = [
        st.Page(home_page, title="Home", icon=":material/home:", url_path="home"),
        st.Page(research_page, title="Research", icon=":material/science:", url_path="research"),
        st.Page(
            library_page,
            title="Research Library",
            icon=":material/history:",
            url_path="library",
        ),
        st.Page(
            dashboard_page,
            title="Market Dashboard",
            icon=":material/dashboard:",
            url_path="dashboard",
        ),
    ]
    if runtime.current_workspace() is not None:
        pages.append(
            st.Page(
                account_page,
                title="Account",
                icon=":material/account_circle:",
                url_path="account",
            )
        )
    navigation = st.navigation(pages, position="sidebar")
    navigation.run()


def _signed_out_pages() -> tuple[StreamlitPage, StreamlitPage]:
    sign_in_route: StreamlitPage
    sign_up_route: StreamlitPage

    def show_sign_in() -> None:
        login_page(open_sign_up=lambda: st.switch_page(sign_up_route))

    def show_sign_up() -> None:
        signup_page(open_sign_in=lambda: st.switch_page(sign_in_route))

    sign_in_route = st.Page(
        show_sign_in,
        title="Sign in",
        icon=":material/login:",
        url_path="login",
        default=True,
    )
    sign_up_route = st.Page(
        show_sign_up,
        title="Sign up",
        icon=":material/person_add:",
        url_path="signup",
    )
    return sign_in_route, sign_up_route


def _initialize_access(settings: Settings, identity: UserIdentity) -> bool:
    cached = runtime.current_workspace()
    binding = f"{identity.issuer}\n{identity.subject}"
    if cached is not None and st.session_state.get(_AUTH_BINDING_STATE_KEY) == binding:
        runtime.initialize_workspace(cached.profile.id, cached)
        return True
    repository = load_research_repository(
        settings.database_url,
        settings.research_retention_days,
    )
    if repository is None:
        st.error("Secure account storage is unavailable. Sign-in cannot continue safely.")
        return False
    try:
        workspace = repository.upsert_user(identity)
    except RuntimeError, ValueError:
        st.error("The verified account workspace could not be initialized.")
        return False
    runtime.initialize_workspace(workspace.profile.id, workspace)
    st.session_state[_AUTH_BINDING_STATE_KEY] = binding
    return True


if __name__ == "__main__":
    main()
