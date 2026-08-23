from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from crypto_research.interfaces.web import auth
from crypto_research.interfaces.web import runtime as ui_runtime
from crypto_research.interfaces.web.auth import identity_from_claims


def test_oidc_claims_are_reduced_to_verified_identity_fields() -> None:
    identity = identity_from_claims(
        {
            "iss": "https://chainscope.example.auth0.com/",
            "sub": "google-oauth2|123",
            "email": "Researcher@Example.com",
            "email_verified": True,
            "name": "Researcher",
            "picture": "https://images.example.com/avatar.png",
            "access_token": "must-not-be-persisted",
            "id_token": "must-not-be-persisted",
        },
        provider="auth0",
    )

    payload = identity.model_dump()
    assert identity.email == "researcher@example.com"
    assert identity.subject == "google-oauth2|123"
    assert identity.avatar_url == "https://images.example.com/avatar.png"
    assert "access_token" not in payload
    assert "id_token" not in payload


def test_oidc_identity_requires_verified_email_and_stable_subject() -> None:
    with pytest.raises(ValidationError, match="verified"):
        identity_from_claims(
            {
                "iss": "https://chainscope.example.auth0.com/",
                "sub": "auth0|123",
                "email": "researcher@example.com",
                "email_verified": False,
                "name": "Researcher",
            },
            provider="auth0",
        )

    with pytest.raises(ValidationError):
        identity_from_claims(
            {
                "iss": "https://chainscope.example.auth0.com/",
                "sub": "",
                "email": "researcher@example.com",
                "email_verified": True,
                "name": "Researcher",
            },
            provider="auth0",
        )


def test_oidc_avatar_rejects_unsafe_urls() -> None:
    identity = identity_from_claims(
        {
            "iss": "https://chainscope.example.auth0.com/",
            "sub": "auth0|123",
            "email": "researcher@example.com",
            "email_verified": True,
            "name": "Researcher",
            "picture": "javascript:alert(1)",
        },
        provider="auth0",
    )

    assert identity.avatar_url is None


def test_signed_out_login_uses_the_named_auth0_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    invoked: list[str] = []
    streamlit = SimpleNamespace(
        user=SimpleNamespace(is_logged_in=False),
        login=invoked.append,
    )
    monkeypatch.setattr(auth, "st", streamlit)
    monkeypatch.setattr(auth, "render_sign_in", lambda: "authenticate")
    monkeypatch.setattr(auth, "_auth_configuration_ready", lambda: True)

    auth.login_page(
        open_sign_up=lambda: (_ for _ in ()).throw(AssertionError("must not navigate")),
    )

    assert invoked == ["auth0"]


def test_authentication_pages_switch_without_starting_oidc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    navigation: list[str] = []
    streamlit = SimpleNamespace(
        user=SimpleNamespace(is_logged_in=False),
        login=lambda provider: (_ for _ in ()).throw(AssertionError(provider)),
    )
    monkeypatch.setattr(auth, "st", streamlit)
    monkeypatch.setattr(auth, "_auth_configuration_ready", lambda: True)

    monkeypatch.setattr(auth, "render_sign_in", lambda: "switch_view")
    auth.login_page(open_sign_up=lambda: navigation.append("signup"))

    monkeypatch.setattr(
        auth,
        "render_sign_up",
        lambda: "switch_view",
    )
    auth.signup_page(open_sign_in=lambda: navigation.append("login"))

    assert navigation == ["signup", "login"]


def test_signup_uses_primary_auth0_universal_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoked: list[str] = []
    streamlit = SimpleNamespace(
        user=SimpleNamespace(is_logged_in=False),
        login=invoked.append,
    )
    monkeypatch.setattr(auth, "st", streamlit)
    monkeypatch.setattr(auth, "render_sign_up", lambda: "authenticate")
    monkeypatch.setattr(auth, "_auth_configuration_ready", lambda: True)

    auth.signup_page(
        open_sign_in=lambda: (_ for _ in ()).throw(AssertionError("must not navigate")),
    )

    assert invoked == ["auth0"]


def test_missing_oidc_secrets_fail_closed_with_actionable_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(auth, "render_auth_configuration_error", lambda: rendered.append("shown"))
    monkeypatch.setattr(auth, "_auth_configuration_ready", lambda: False)

    assert auth.authenticated_identity() is None
    auth.login_page(open_sign_up=lambda: None)

    assert rendered == ["shown"]


def test_authenticated_identity_rejects_unverified_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit = SimpleNamespace(
        user=SimpleNamespace(
            is_logged_in=True,
            to_dict=lambda: {
                "iss": "https://chainscope.example.auth0.com/",
                "sub": "auth0|123",
                "email": "researcher@example.com",
                "email_verified": False,
            },
        )
    )
    monkeypatch.setattr(auth, "st", streamlit)
    monkeypatch.setattr(auth, "_auth_configuration_ready", lambda: True)

    assert auth.authenticated_identity() is None


def test_logout_clears_workspace_state_before_streamlit_logout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        auth,
        "st",
        SimpleNamespace(logout=lambda: events.append("oidc_logout")),
    )
    monkeypatch.setattr(
        ui_runtime,
        "clear_user_session_state",
        lambda: events.append("clear_workspace"),
    )

    auth.logout()

    assert events == ["clear_workspace", "oidc_logout"]
