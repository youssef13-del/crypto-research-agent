from importlib import import_module
from pathlib import Path

from crypto_research.interfaces.web.runtime import public_agent_summary


def test_native_research_app_starts_with_navigation() -> None:
    module = import_module("crypto_research.interfaces.web.app")
    assert module.__file__ is not None
    app_source = Path(module.__file__).read_text(encoding="utf-8")

    assert callable(module.main)
    assert "st.navigation" in app_source
    assert "st.Page(research_page" in app_source
    assert 'url_path="research"' in app_source
    assert 'url_path="chat"' not in app_source
    assert "library_page" in app_source
    assert "account_page" in app_source
    assert "authenticated_identity" in app_source
    assert 'url_path="login"' in app_source
    assert 'url_path="signup"' in app_source


def test_research_page_is_importable_as_a_native_page() -> None:
    module = import_module("crypto_research.interfaces.web.pages.research")

    assert callable(module.research_page)


def test_public_agent_summary_lists_only_research_specialists() -> None:
    summary = public_agent_summary(("market_agent", "news_agent"))

    assert summary.startswith("Using live crypto research: ")
    assert "Market & Risk Agent" in summary
    assert "News Agent" in summary
    assert public_agent_summary(("unknown_agent",)) == ""
