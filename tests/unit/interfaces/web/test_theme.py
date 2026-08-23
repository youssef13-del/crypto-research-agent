import re
import tomllib
from pathlib import Path

from pytest import MonkeyPatch
from tests.support.accessibility import contrast_ratio

from crypto_research.interfaces.web.components.layout import BASE_THEME_CSS, render_data_table
from crypto_research.interfaces.web.theme import (
    DARK_PALETTE,
    ThemePalette,
    theme_token_css,
)


def test_theme_tokens_cover_the_supported_mode() -> None:
    css = theme_token_css("dark")

    assert "--cs-canvas:" in css
    assert "--cs-surface:" in css
    assert "--cs-ink:" in css
    assert "--cs-accent:" in css
    assert "color-scheme: dark;" in css


def test_primary_and_muted_text_meet_wcag_aa(palette: ThemePalette = DARK_PALETTE) -> None:
    assert contrast_ratio(palette.ink, palette.canvas) >= 4.5
    assert contrast_ratio(palette.muted, palette.canvas) >= 4.5


def test_terminal_accents_are_legible_and_semantically_distinct() -> None:
    assert contrast_ratio("#031116", DARK_PALETTE.accent) >= 4.5
    assert len({DARK_PALETTE.positive, DARK_PALETTE.warning, DARK_PALETTE.danger}) == 3


def test_visual_system_includes_motion_and_responsive_rules() -> None:
    assert "prefers-reduced-motion: reduce" in BASE_THEME_CSS
    assert "@media (max-width: 900px)" in BASE_THEME_CSS
    assert "@media (max-width: 640px)" in BASE_THEME_CSS
    assert ":focus-visible" in BASE_THEME_CSS
    assert ":has(.cs-sticky-panel)" in BASE_THEME_CSS
    assert ":has(.cs-sticky-panel) { position: static" in BASE_THEME_CSS
    assert "max-width: min(86vw, 22rem)" in BASE_THEME_CSS
    assert '[data-baseweb="select"]' in BASE_THEME_CSS
    assert '[data-testid="stSegmentedControl"]' in BASE_THEME_CSS
    assert '[data-testid="stChatInput"]' not in BASE_THEME_CSS
    assert '[data-testid="stCheckbox"]' in BASE_THEME_CSS
    assert "--st-primary-color" in BASE_THEME_CSS
    assert ".cs-analysis-verdict-header" in BASE_THEME_CSS
    assert ".cs-analysis-confidence" in BASE_THEME_CSS
    assert ".cs-panel-section-heading" in BASE_THEME_CSS
    assert ".cs-resource-card" in BASE_THEME_CSS
    assert ".cs-claim-card" in BASE_THEME_CSS
    assert '[data-testid="stExpander"] [data-testid="stExpander"]' in BASE_THEME_CSS
    assert '[data-testid="stVegaLiteChart"]' in BASE_THEME_CSS
    assert ".cs-login-root" in BASE_THEME_CSS
    assert ".cs-login-panel" in BASE_THEME_CSS
    assert ".cs-sidebar-profile" in BASE_THEME_CSS
    assert ".cs-profile-avatar" in BASE_THEME_CSS
    assert ".st-key-home-actions" in BASE_THEME_CSS
    assert ":has(.cs-home-asset-card)" in BASE_THEME_CSS
    assert "flex: 1 1 calc(50% - .5rem)" in BASE_THEME_CSS
    assert ".cs-home-action { min-height: 10rem" in BASE_THEME_CSS
    assert ":has(.cs-home-action) { flex:" not in BASE_THEME_CSS


def test_terminal_surfaces_are_flat_and_do_not_use_decorative_motion() -> None:
    assert "linear-gradient" not in BASE_THEME_CSS
    assert "radial-gradient" not in BASE_THEME_CSS
    assert "translateY" not in BASE_THEME_CSS
    assert "translateX" not in BASE_THEME_CSS


def test_theme_does_not_cover_streamlit_dataframe_canvas_layers() -> None:
    assert '[data-testid="stDataFrame"]' in BASE_THEME_CSS
    assert ".glideDataEditor" not in BASE_THEME_CSS
    assert ".dvn-" not in BASE_THEME_CSS


def test_static_table_is_responsive_accessible_and_escapes_values(
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "crypto_research.interfaces.web.components.layout.st.markdown",
        lambda value, **kwargs: captured.update(value=value, kwargs=kwargs),
    )

    render_data_table(
        [{"Asset": "<BTC>", "Score": "82.4/100"}],
        label='Ranked "market" screen',
    )

    rendered = str(captured["value"])
    assert "cs-table-wrap" in rendered
    assert 'tabindex="0"' in rendered
    assert "&lt;BTC&gt;" in rendered
    assert 'aria-label="Ranked &quot;market&quot; screen"' in rendered
    assert captured["kwargs"] == {"unsafe_allow_html": True}


def test_every_variable_referenced_by_base_css_is_defined() -> None:
    referenced = set(
        re.findall(r"var\((--cs-[a-z0-9-]+)", BASE_THEME_CSS.replace("__THEME_TOKENS__", ""))
    )
    defined = set(re.findall(r"(--cs-[a-z0-9-]+):", theme_token_css("dark")))

    assert referenced
    assert referenced <= defined


def test_streamlit_native_theme_matches_application_palette() -> None:
    root = Path(__file__).parents[4]
    config = tomllib.loads((root / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    theme = config["theme"]
    dark = theme["dark"]

    expected = {
        "primaryColor": DARK_PALETTE.accent,
        "backgroundColor": DARK_PALETTE.canvas,
        "secondaryBackgroundColor": DARK_PALETTE.surface,
        "textColor": DARK_PALETTE.ink,
        "linkColor": DARK_PALETTE.blue,
        "redColor": DARK_PALETTE.danger,
        "orangeColor": DARK_PALETTE.orange,
        "blueColor": DARK_PALETTE.blue,
        "greenColor": DARK_PALETTE.positive,
        "yellowColor": DARK_PALETTE.warning,
        # Streamlit exposes a violet semantic slot. ChainScope maps that slot
        # to teal so native components cannot reintroduce the retired palette.
        "violetColor": DARK_PALETTE.teal,
        "grayColor": DARK_PALETTE.subtle,
        "redTextColor": DARK_PALETTE.danger,
        "orangeTextColor": DARK_PALETTE.orange,
        "yellowTextColor": DARK_PALETTE.warning,
        "blueTextColor": DARK_PALETTE.blue,
        "greenTextColor": DARK_PALETTE.positive,
        "violetTextColor": DARK_PALETTE.teal,
        "grayTextColor": DARK_PALETTE.muted,
        "codeTextColor": DARK_PALETTE.accent_strong,
        "codeBackgroundColor": DARK_PALETTE.surface_soft,
        "dataframeHeaderBackgroundColor": DARK_PALETTE.surface_soft,
    }
    for key, value in expected.items():
        assert theme[key] == value
        assert dark[key] == value

    assert theme["chartCategoricalColors"] == [
        DARK_PALETTE.accent,
        DARK_PALETTE.blue,
        DARK_PALETTE.positive,
        DARK_PALETTE.teal,
        DARK_PALETTE.warning,
        DARK_PALETTE.danger,
        DARK_PALETTE.orange,
        DARK_PALETTE.lime,
    ]


def test_streamlit_configuration_contains_no_legacy_violet_palette() -> None:
    root = Path(__file__).parents[4]
    source = (root / ".streamlit" / "config.toml").read_text(encoding="utf-8").lower()
    legacy_colors = {
        "#9d8ae8",
        "#131024",
        "#1c1731",
        "#2a2245",
        "#c39be6",
        "#241d3e",
    }

    assert all(color not in source for color in legacy_colors)


def test_user_visible_ui_sources_do_not_contain_common_mojibake() -> None:
    ui_root = Path(__file__).parents[3] / "src" / "crypto_research" / "ui"
    forbidden = ("Â", "â€", "â€¢", "ï¼", "�")

    for path in ui_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not any(marker in source for marker in forbidden), path
