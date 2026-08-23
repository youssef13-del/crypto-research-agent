from streamlit.testing.v1 import AppTest


def test_sign_in_page_has_authentication_and_account_creation_actions() -> None:
    app = AppTest.from_string(
        """
from crypto_research.interfaces.web.components.layout import apply_theme
from crypto_research.interfaces.web.components.login import render_sign_in

apply_theme()
assert render_sign_in() is None
"""
    ).run()

    assert not app.exception
    assert [button.label for button in app.button] == [
        "Sign in with Auth0",
        "Create a new account",
    ]
    rendered = "\n".join(item.value for item in app.markdown)
    assert "Research crypto markets from one private workspace." in rendered
    assert "ChainScope never stores passwords" in rendered
    assert "Passwords and identity tokens stay with Auth0" in rendered


def test_sign_up_page_explains_auth0_ownership_and_sign_in_navigation() -> None:
    app = AppTest.from_string(
        """
from crypto_research.interfaces.web.components.layout import apply_theme
from crypto_research.interfaces.web.components.login import render_sign_up

apply_theme()
assert render_sign_up() is None
"""
    ).run()

    assert not app.exception
    assert [button.label for button in app.button] == [
        "Create account with Auth0",
        "Already have an account? Sign in",
    ]
    rendered = "\n".join(item.value for item in app.markdown)
    assert "Start a private crypto research library." in rendered
    assert "Choose Sign up in Auth0 Universal Login" in rendered
    assert "never receives or stores your password" in rendered


def test_missing_auth_configuration_is_actionable_without_exposing_secrets() -> None:
    app = AppTest.from_string(
        """
from crypto_research.interfaces.web.components.login import render_auth_configuration_error
render_auth_configuration_error()
"""
    ).run()

    assert not app.exception
    rendered = "\n".join(item.value for item in app.markdown)
    assert "Auth0 is not ready" in rendered
    assert ".streamlit/secrets.toml.example" in rendered
    assert "client secret" in rendered
    assert not app.button
