"""Repository-controlled views for authentication and configuration states."""

from __future__ import annotations

from typing import Literal

import streamlit as st

from crypto_research.interfaces.web.components.layout import render_brand

type AuthenticationAction = Literal["authenticate", "switch_view"]


def render_sign_in() -> AuthenticationAction | None:
    """Render the sign-in page and return the requested authentication action."""

    render_brand()
    st.markdown('<div class="cs-login-root"></div>', unsafe_allow_html=True)
    intro, access = st.columns([1.2, 0.8], gap="large", vertical_alignment="center")
    with intro:
        st.markdown(
            '<section class="cs-login-intro"><div class="cs-eyebrow">ChainScope terminal</div>'
            "<h1>Research crypto markets from one private workspace.</h1>"
            "<p>Run specialist research, compare market evidence, and keep saved reports "
            "separate from every other account.</p>"
            '<div class="cs-login-ledger">'
            "<div><b>&#10003;</b><span>Reports and preferences are scoped to your "
            "account</span></div>"
            "<div><b>&#10003;</b><span>Provider evidence retains source and observation "
            "time</span></div>"
            "<div><b>&#10003;</b><span>Passwords and identity tokens stay with Auth0</span></div>"
            "</div></section>",
            unsafe_allow_html=True,
        )
    with access, st.container(border=True):
        st.markdown(
            '<section class="cs-login-panel"><div class="cs-eyebrow">Secure access</div>'
            "<h2>Sign in to ChainScope</h2>"
            "<p>Continue to Auth0 to use Google or your configured account connection.</p>"
            "</section>",
            unsafe_allow_html=True,
        )
        authenticate = st.button(
            "Sign in with Auth0",
            type="primary",
            width="stretch",
            icon=":material/login:",
        )
        switch_view = st.button(
            "Create a new account",
            width="stretch",
            icon=":material/person_add:",
        )
        st.markdown(
            '<div class="cs-login-security">Authentication is handled by Auth0. '
            "ChainScope never stores passwords and keeps only the verified profile fields "
            "needed to identify your workspace.</div>",
            unsafe_allow_html=True,
        )
    if authenticate:
        return "authenticate"
    if switch_view:
        return "switch_view"
    return None


def render_sign_up() -> AuthenticationAction | None:
    """Render account creation and return the requested authentication action."""

    render_brand()
    st.markdown('<div class="cs-login-root"></div>', unsafe_allow_html=True)
    intro, access = st.columns([1.2, 0.8], gap="large", vertical_alignment="center")
    with intro:
        st.markdown(
            '<section class="cs-login-intro"><div class="cs-eyebrow">Create your workspace</div>'
            "<h1>Start a private crypto research library.</h1>"
            "<p>Create an Auth0 account to save reports, maintain a watchlist, and keep every "
            "research run isolated to your identity.</p>"
            '<div class="cs-login-ledger">'
            "<div><b>&#10003;</b><span>Your workspace is created after verified "
            "sign-up</span></div>"
            "<div><b>&#10003;</b><span>Google and configured Auth0 connections are "
            "supported</span></div>"
            "<div><b>&#10003;</b><span>ChainScope never receives or stores your "
            "password</span></div>"
            "</div></section>",
            unsafe_allow_html=True,
        )
    with access, st.container(border=True):
        st.markdown(
            '<section class="cs-login-panel"><div class="cs-eyebrow">New account</div>'
            "<h2>Sign up for ChainScope</h2>"
            "<p>Continue to Auth0 to create and verify your account.</p>"
            "</section>",
            unsafe_allow_html=True,
        )
        authenticate = st.button(
            "Create account with Auth0",
            type="primary",
            width="stretch",
            icon=":material/person_add:",
        )
        switch_view = st.button(
            "Already have an account? Sign in",
            width="stretch",
            icon=":material/login:",
        )
        st.markdown(
            '<div class="cs-login-security">Choose Sign up in Auth0 Universal Login to '
            "create your account. ChainScope stores only the verified "
            "profile fields needed to identify your private workspace.</div>",
            unsafe_allow_html=True,
        )
    if authenticate:
        return "authenticate"
    if switch_view:
        return "switch_view"
    return None


def render_invalid_identity() -> bool:
    render_brand()
    st.markdown(
        '<section class="cs-login-config"><div class="cs-eyebrow">Account check</div>'
        "<h1>We could not verify this account</h1>"
        "<p>Auth0 did not return a verified email and stable identity. Verify the account "
        "with Auth0, then sign in again.</p></section>",
        unsafe_allow_html=True,
    )
    return st.button("Sign out", icon=":material/logout:")


def render_workspace_loading() -> None:
    render_brand()
    st.markdown(
        '<section class="cs-login-config"><div class="cs-eyebrow">Session verified</div>'
        "<h1>Opening your workspace</h1>"
        "<p>Your account is verified. ChainScope is loading your saved workspace.</p>"
        "</section>",
        unsafe_allow_html=True,
    )


def render_auth_configuration_error() -> None:
    render_brand()
    st.markdown(
        '<section class="cs-login-config"><div class="cs-eyebrow">Configuration required</div>'
        "<h1>Auth0 is not ready</h1>"
        "<p>Copy <code>.streamlit/secrets.toml.example</code> to "
        "<code>.streamlit/secrets.toml</code>, then provide the callback URL, cookie secret, "
        "client ID, client secret, and metadata URL. Restart ChainScope after saving.</p>"
        "</section>",
        unsafe_allow_html=True,
    )


__all__ = [
    "render_auth_configuration_error",
    "render_invalid_identity",
    "render_sign_in",
    "render_sign_up",
    "render_workspace_loading",
]
