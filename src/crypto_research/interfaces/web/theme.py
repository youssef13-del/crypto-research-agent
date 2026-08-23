# CSS selectors stay intact for easier browser inspection.
# ruff: noqa: E501

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

import streamlit as st


@dataclass(frozen=True, slots=True)
class ThemePalette:
    canvas: str
    surface: str
    surface_raised: str
    surface_soft: str
    ink: str
    muted: str
    subtle: str
    border: str
    accent: str
    accent_strong: str
    blue: str
    teal: str
    orange: str
    lime: str
    positive: str
    warning: str
    danger: str
    chart_grid: str
    shadow: str

    def css_declarations(self) -> str:
        aliases = {"warning": "amber"}
        return "\n".join(
            f"--cs-{aliases.get(item.name, item.name).replace('_', '-')}: "
            f"{getattr(self, item.name)};"
            for item in fields(self)
        )


DARK_PALETTE = ThemePalette(
    canvas="#071018",
    surface="#0c1720",
    surface_raised="#11212c",
    surface_soft="#09131b",
    ink="#e8f0f5",
    muted="#a6b6c2",
    subtle="#7f919e",
    border="rgba(159, 184, 200, 0.18)",
    accent="#22d3ee",
    accent_strong="#67e8f9",
    blue="#38bdf8",
    teal="#2dd4bf",
    orange="#fb923c",
    lime="#a3e635",
    positive="#34d399",
    warning="#fbbf24",
    danger="#fb7185",
    chart_grid="#1b2d39",
    shadow="0 18px 44px rgba(0, 0, 0, 0.24)",
)


TERMINAL_UI_CSS = """
/* Foundation */
.stApp { background: var(--cs-canvas); }
.block-container { max-width: 1320px; padding-top: 1.5rem; }
h1, h2, h3, h4, h5, h6 { letter-spacing: -.025em; }
h1 { font-size: clamp(1.8rem, 3.5vw, 2.65rem); }
p, li { line-height: 1.58; }
div.stButton > button, div.stFormSubmitButton > button, div.stPageLink > a { border-radius: 8px !important; box-shadow: none !important; }
div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] { background: var(--cs-accent); border-color: var(--cs-accent); color: #031116; }
div.stButton > button[kind="primary"] p, div.stFormSubmitButton > button[kind="primary"] p { color: #031116; }
[data-testid="stForm"], [data-testid="stMetric"], [data-testid="stExpander"], [data-testid="stVegaLiteChart"], .cs-card, .cs-home-action { background: var(--cs-surface); box-shadow: none; }
[data-testid="stForm"] { border-radius: 10px; }
[data-testid="stMetric"] { border-radius: 9px; padding: .8rem .9rem; }
[data-testid="stExpander"] { border-radius: 8px; }
[data-testid="stVegaLiteChart"] { border-radius: 9px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 10px; }

/* Application shell */
[data-testid="stSidebar"] { background: #08131b; }
[data-testid="stSidebarNav"] a { border-radius: 7px; }
[data-testid="stSidebarNav"] a:hover { transform: none; background: rgba(34, 211, 238, .07); }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: rgba(34, 211, 238, .10); color: var(--cs-accent-strong); border-left: 2px solid var(--cs-accent); }
.cs-brand { margin-bottom: .65rem; }
.cs-mark { width: 2rem; height: 2rem; border-radius: 6px; background: var(--cs-accent); color: #031116; box-shadow: none; }
.cs-brand-name { font-size: 1rem; }
.cs-brand-subtitle { letter-spacing: .08em; }
.cs-sidebar-profile { border-radius: 8px; background: transparent; padding: .58rem .15rem; border-width: 1px 0; }
.cs-sidebar-profile > span, .cs-profile-avatar { background: var(--cs-surface-raised); border: 1px solid var(--cs-border); color: var(--cs-accent-strong); box-shadow: none; }
.cs-status-dot, .cs-status-dot-muted, .cs-status-dot-error { box-shadow: none; }
.cs-research-activity-heading { display: flex; align-items: center; gap: .48rem; margin: .85rem 0 .28rem; padding-top: .72rem; border-top: 1px solid var(--cs-border); color: var(--cs-ink); font-size: .76rem; }
.cs-research-activity-dot { width: .48rem; height: .48rem; flex: 0 0 auto; border-radius: 50%; background: var(--cs-subtle); }
.cs-research-activity-dot.is-running { background: var(--cs-accent); }
.cs-research-activity-dot.is-complete { background: var(--cs-positive); }
.cs-research-activity-dot.is-error { background: var(--cs-danger); }
.cs-footer { color: var(--cs-subtle); }

/* Page hierarchy */
.cs-eyebrow { color: var(--cs-accent-strong); letter-spacing: .11em; margin-bottom: .42rem; }
.cs-page-header { margin-bottom: 1.15rem; }
.cs-page-header h1 { font-size: clamp(1.65rem, 3vw, 2.25rem); margin-bottom: .4rem; }
.cs-page-header p { max-width: 720px; font-size: .94rem; }
.cs-hero, .cs-home-hero { padding: 1.35rem 1.45rem; border-radius: 10px; background: var(--cs-surface); }
.cs-home-action { min-height: 7.8rem; border-radius: 9px; padding: .9rem 1rem; }
.cs-home-action-primary { border-color: rgba(34, 211, 238, .35); background: rgba(34, 211, 238, .035); }
.cs-section { margin: 1.35rem 0 .45rem; }
.cs-section-description { margin-bottom: .7rem; }
.cs-card { border-radius: 9px; }
.cs-disclaimer { border-radius: 7px; background: var(--cs-surface-soft); }

/* Authentication */
.cs-login-shell { min-height: calc(100vh - 7rem); display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(20rem, .8fr); gap: clamp(2rem, 7vw, 6rem); align-items: center; max-width: 1100px; margin: 0 auto; padding: 2rem 0; }
[data-testid="stMainBlockContainer"]:has(.cs-login-root) { max-width: 1100px; padding-top: 1.25rem; }
[data-testid="stMainBlockContainer"]:has(.cs-login-root) > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] { min-height: calc(100vh - 10rem); align-items: center; gap: clamp(2rem, 7vw, 6rem); }
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-login-panel) { border-radius: 10px; border-color: var(--cs-border); border-top: 2px solid var(--cs-accent); background: var(--cs-surface); }
.cs-login-intro { max-width: 620px; }
.cs-login-intro h1 { max-width: 590px; margin: 0 0 .9rem; font-size: clamp(2.25rem, 5vw, 4.1rem); line-height: 1.02; }
.cs-login-intro > p { max-width: 560px; margin: 0; color: var(--cs-muted); font-size: 1rem; }
.cs-login-ledger { display: grid; gap: 0; margin-top: 1.7rem; border-top: 1px solid var(--cs-border); }
.cs-login-ledger div { display: grid; grid-template-columns: 1.1rem 1fr; gap: .65rem; padding: .7rem 0; border-bottom: 1px solid var(--cs-border); color: var(--cs-muted); font-size: .82rem; }
.cs-login-ledger b { color: var(--cs-positive); font-weight: 700; }
.cs-login-panel { padding: .15rem 0 0; }
.cs-login-panel h2 { margin: .2rem 0 .45rem; font-size: 1.35rem; }
.cs-login-panel p { margin: 0 0 1rem; color: var(--cs-muted); font-size: .88rem; }
.cs-login-security { margin-top: .8rem; padding-top: .75rem; border-top: 1px solid var(--cs-border); color: var(--cs-subtle); font-size: .72rem; line-height: 1.5; }
.cs-login-config { max-width: 720px; margin: 10vh auto 0; padding: 1.25rem 1.35rem; border: 1px solid var(--cs-border); border-left: 3px solid var(--cs-amber); border-radius: 8px; background: var(--cs-surface); }
.cs-login-config h1 { margin: 0 0 .5rem; font-size: 1.55rem; }
.cs-login-config p { margin: 0; color: var(--cs-muted); }
.cs-login-config code { color: var(--cs-accent-strong); }

/* Research and data */
.cs-analysis-verdict, .cs-analysis-comparison, .cs-analysis-asset, .cs-resource-card, .cs-claim-card { border-radius: 8px; background: var(--cs-surface); }
.cs-analysis-asset::before { width: 2px; border-radius: 0; }
.cs-analysis-chip { border-radius: 5px; background: var(--cs-surface-soft); }
.cs-resource-card header { background: var(--cs-surface-soft); }
.cs-coverage-note { border-radius: 0 6px 6px 0; }
.cs-table-wrap, [data-testid="stDataFrame"] { border-radius: 8px; }

@media (max-width: 800px) {
  .cs-login-shell { min-height: auto; grid-template-columns: 1fr; gap: 1.4rem; padding: 1rem 0 3rem; }
  .cs-login-intro h1 { font-size: clamp(2rem, 10vw, 3rem); }
  .cs-login-ledger { margin-top: 1.1rem; }
  [data-testid="stMainBlockContainer"]:has(.cs-login-root) > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] { min-height: auto; }
}
@media (max-width: 640px) {
  .block-container { padding: 1rem .85rem 4rem; }
  .cs-login-shell { padding-top: .2rem; }
  .cs-login-intro > p { font-size: .92rem; }
  .cs-login-panel { padding: 1rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; animation: none !important; }
}
"""


ThemeMode = Literal["dark"]
THEME_STATE_KEY = "appearance_mode"
THEME_OPTIONS: tuple[ThemeMode, ...] = ("dark",)


def active_palette(mode: ThemeMode | None = None) -> ThemePalette:
    """Return the sole supported (dark) palette for every session."""

    del mode
    return DARK_PALETTE


def initialize_theme_state() -> None:
    """Pin the appearance to the only supported (dark) mode."""

    st.session_state[THEME_STATE_KEY] = "dark"


def theme_token_css(mode: ThemeMode = "dark") -> str:
    palette = active_palette(mode)
    return f":root {{\ncolor-scheme: {mode};\n{palette.css_declarations()}\n}}"


__all__ = [
    "DARK_PALETTE",
    "TERMINAL_UI_CSS",
    "THEME_OPTIONS",
    "THEME_STATE_KEY",
    "ThemeMode",
    "ThemePalette",
    "active_palette",
    "initialize_theme_state",
    "theme_token_css",
]
