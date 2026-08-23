from streamlit.testing.v1 import AppTest


def test_settings_change_replaces_runtime() -> None:
    source = """
import streamlit as st
from datetime import UTC, datetime
from crypto_research.config import Settings
from crypto_research.domain.account import UserProfile, UserWorkspace
from crypto_research.interfaces.web import runtime

now = datetime.now(UTC)
workspace = UserWorkspace(profile=UserProfile(
    id="user-1", provider="auth0", email="user@example.com", provider_name="User",
    created_at=now, updated_at=now, last_login_at=now,
))
runtime.initialize_workspace(workspace.profile.id, workspace)
created = []
runtime.load_research_runtime = lambda settings, **_kwargs: created.append(settings) or object()
first = runtime.session_runtime(Settings(_env_file=None))
assert runtime.session_runtime(Settings(_env_file=None)) is first
second = runtime.session_runtime(Settings(groq_timeout_seconds=16, _env_file=None))
assert second is not first
assert len(created) == 2
"""
    app = AppTest.from_string(source)
    app.run()

    assert not app.exception


def test_changing_workspace_owner_clears_previous_user_session_state() -> None:
    source = """
import streamlit as st
from datetime import UTC, datetime
from crypto_research.domain.account import UserProfile, UserWorkspace
from crypto_research.interfaces.web import runtime

now = datetime.now(UTC)
first = UserWorkspace(profile=UserProfile(
    id="user-a", provider="auth0", email="a@example.com", provider_name="A",
    created_at=now, updated_at=now, last_login_at=now,
))
second = UserWorkspace(profile=UserProfile(
    id="user-b", provider="auth0", email="b@example.com", provider_name="B",
    created_at=now, updated_at=now, last_login_at=now,
))
runtime.initialize_workspace("user-a", first)
st.session_state[runtime.LATEST_RESEARCH_TURN_STATE_KEY] = "private research"
st.session_state["dashboard_last_result"] = "private dashboard"
runtime.initialize_workspace("user-b", second)
assert runtime.current_owner_id() == "user-b"
assert runtime.current_workspace() == second
assert st.session_state[runtime.LATEST_RESEARCH_TURN_STATE_KEY] is None
assert "dashboard_last_result" not in st.session_state
"""
    app = AppTest.from_string(source)
    app.run()

    assert not app.exception


def test_runtime_rejects_missing_or_mismatched_workspace_bindings() -> None:
    source = """
from datetime import UTC, datetime
import pytest
from crypto_research.domain.account import UserProfile, UserWorkspace
from crypto_research.interfaces.web import runtime

with pytest.raises(RuntimeError, match="authenticated workspace"):
    runtime.current_owner_id()
now = datetime.now(UTC)
workspace = UserWorkspace(profile=UserProfile(
    id="user-a", provider="auth0", email="a@example.com", provider_name="A",
    created_at=now, updated_at=now, last_login_at=now,
))
with pytest.raises(ValueError, match="binding"):
    runtime.initialize_workspace("user-b", workspace)
"""
    app = AppTest.from_string(source)
    app.run()

    assert not app.exception
