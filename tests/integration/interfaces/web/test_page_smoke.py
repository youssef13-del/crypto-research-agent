from importlib import import_module

import pytest


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("crypto_research.interfaces.web.pages.home", "home_page"),
        ("crypto_research.interfaces.web.pages.research", "research_page"),
        ("crypto_research.interfaces.web.pages.dashboard", "dashboard_page"),
        ("crypto_research.interfaces.web.pages.library", "library_page"),
        ("crypto_research.interfaces.web.pages.account", "account_page"),
    ],
)
def test_active_page_is_importable(module_name: str, function_name: str) -> None:
    module = import_module(module_name)

    assert callable(getattr(module, function_name))


def test_removed_chat_page_has_no_legacy_alias() -> None:
    with pytest.raises(ModuleNotFoundError):
        import_module("crypto_research.interfaces.web.pages.chat")
