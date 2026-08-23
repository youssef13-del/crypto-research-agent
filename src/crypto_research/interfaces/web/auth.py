"""Streamlit OIDC gate and verified identity normalization."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.util import find_spec
from typing import Any

import streamlit as st
from pydantic import ValidationError

from crypto_research.domain.account import UserIdentity
from crypto_research.interfaces.web import runtime
from crypto_research.interfaces.web.components.login import (
    render_auth_configuration_error,
    render_invalid_identity,
    render_sign_in,
    render_sign_up,
    render_workspace_loading,
)
from crypto_research.shared.security import normalize_http_url

_AUTH_PROVIDER = "auth0"


def authenticated_identity() -> UserIdentity | None:
    """Return the current verified OIDC identity, if authentication is complete."""

    if not _auth_configuration_ready():
        return None
    if not bool(getattr(st.user, "is_logged_in", False)):
        return None
    try:
        return identity_from_claims(st.user.to_dict(), provider=_AUTH_PROVIDER)
    except AttributeError, TypeError, ValidationError, ValueError:
        return None


def login_page(*, open_sign_up: Callable[[], None]) -> None:
    """Render the signed-out login page."""

    if not _auth_configuration_ready():
        render_auth_configuration_error()
        return
    if _finish_authenticated_redirect():
        return
    action = render_sign_in()
    if action == "authenticate":
        st.login(_AUTH_PROVIDER)
    elif action == "switch_view":
        open_sign_up()


def signup_page(*, open_sign_in: Callable[[], None]) -> None:
    """Render account creation and initiate the configured Auth0 flow."""

    if not _auth_configuration_ready():
        render_auth_configuration_error()
        return
    if _finish_authenticated_redirect():
        return
    action = render_sign_up()
    if action == "authenticate":
        st.login(_AUTH_PROVIDER)
    elif action == "switch_view":
        open_sign_in()


def identity_from_claims(claims: Mapping[str, object], *, provider: str) -> UserIdentity:
    """Normalize the small verified claim set ChainScope is allowed to persist."""

    avatar = claims.get("picture")
    avatar_url = normalize_http_url(str(avatar)) if isinstance(avatar, str) else None
    verified = claims.get("email_verified") is True
    name = claims.get("name") or claims.get("preferred_username") or claims.get("email")
    return UserIdentity(
        provider=provider,
        issuer=str(claims.get("iss") or ""),
        subject=str(claims.get("sub") or ""),
        email=str(claims.get("email") or ""),
        email_verified=verified,
        provider_name=str(name or ""),
        avatar_url=avatar_url,
    )


def logout() -> None:
    """Clear private state before OIDC logout returns to signed-out navigation.

    The configured redirect URI must also be registered with the provider as an
    allowed post-logout redirect.
    """

    runtime.clear_user_session_state()
    st.logout()


def _finish_authenticated_redirect() -> bool:
    if not bool(getattr(st.user, "is_logged_in", False)):
        return False
    try:
        identity_from_claims(st.user.to_dict(), provider=_AUTH_PROVIDER)
    except AttributeError, TypeError, ValidationError, ValueError:
        if render_invalid_identity():
            logout()
        return True
    render_workspace_loading()
    st.rerun()
    return True


def _auth_configuration_ready() -> bool:
    if find_spec("authlib") is None:
        return False
    try:
        raw: Mapping[str, Any] = st.secrets.to_dict()
    except AttributeError, FileNotFoundError, OSError:
        return False
    auth = raw.get("auth")
    if not isinstance(auth, Mapping):
        return False
    return _provider_ready(auth, _AUTH_PROVIDER)


def _provider_ready(auth: Mapping[str, Any], provider_name: str) -> bool:
    provider = auth.get(provider_name)
    if not isinstance(provider, Mapping):
        return False
    shared_ready = all(auth.get(key) for key in ("redirect_uri", "cookie_secret"))
    provider_ready = all(
        provider.get(key) for key in ("client_id", "client_secret", "server_metadata_url")
    )
    return bool(shared_ready and provider_ready)


__all__ = [
    "authenticated_identity",
    "identity_from_claims",
    "login_page",
    "logout",
    "signup_page",
]
