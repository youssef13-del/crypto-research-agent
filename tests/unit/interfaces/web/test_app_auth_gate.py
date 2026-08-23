from types import SimpleNamespace

import pytest

from crypto_research.config import Settings
from crypto_research.domain.account import UserIdentity
from crypto_research.interfaces.web import app
from crypto_research.interfaces.web import runtime as ui_runtime
from crypto_research.interfaces.web.auth import identity_from_claims


def _identity() -> UserIdentity:
    return identity_from_claims(
        {
            "iss": "https://chainscope.example.auth0.com/",
            "sub": "auth0|123",
            "email": "researcher@example.com",
            "email_verified": True,
            "name": "Researcher",
        },
        provider="auth0",
    )


def test_verified_session_reuses_its_initialized_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    workspace = SimpleNamespace(profile=SimpleNamespace(id="workspace-1"))
    binding = f"{identity.issuer}\n{identity.subject}"
    initialized: list[tuple[str, object]] = []
    monkeypatch.setattr(
        app, "st", SimpleNamespace(session_state={app._AUTH_BINDING_STATE_KEY: binding})
    )
    monkeypatch.setattr(ui_runtime, "current_workspace", lambda: workspace)
    monkeypatch.setattr(
        ui_runtime,
        "initialize_workspace",
        lambda owner_id, value: initialized.append((owner_id, value)),
    )
    monkeypatch.setattr(
        app,
        "load_research_repository",
        lambda *args: (_ for _ in ()).throw(AssertionError("repository must not reload")),
    )

    assert app._initialize_access(Settings(_env_file=None), identity)
    assert initialized == [("workspace-1", workspace)]


def test_verified_session_creates_workspace_and_records_identity_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    workspace = SimpleNamespace(profile=SimpleNamespace(id="workspace-1"))
    session_state: dict[str, object] = {}
    initialized: list[tuple[str, object]] = []
    repository = SimpleNamespace(upsert_user=lambda received: workspace)
    monkeypatch.setattr(app, "st", SimpleNamespace(session_state=session_state))
    monkeypatch.setattr(ui_runtime, "current_workspace", lambda: None)
    monkeypatch.setattr(
        ui_runtime,
        "initialize_workspace",
        lambda owner_id, value: initialized.append((owner_id, value)),
    )
    monkeypatch.setattr(app, "load_research_repository", lambda *args: repository)

    assert app._initialize_access(Settings(_env_file=None), identity)
    assert initialized == [("workspace-1", workspace)]
    assert session_state[app._AUTH_BINDING_STATE_KEY] == (
        "https://chainscope.example.auth0.com/\nauth0|123"
    )
