"""Visual system and layout primitives shared by every Streamlit page."""

# Keep related CSS rules together so the visual system stays compact and scannable.
# ruff: noqa: E501

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence

import streamlit as st

from crypto_research.interfaces.web.theme import (
    TERMINAL_UI_CSS,
    initialize_theme_state,
    theme_token_css,
)

_BASE_THEME_CSS = """
<style>
__THEME_TOKENS__
:root { --primary-color: var(--cs-accent); --background-color: var(--cs-canvas); --secondary-background-color: var(--cs-surface); --text-color: var(--cs-ink); --sidebar-background-color: var(--cs-surface); --widget-border-color: var(--cs-border); --st-primary-color: var(--cs-accent); --st-background-color: var(--cs-canvas); --st-secondary-background-color: var(--cs-surface); --st-text-color: var(--cs-ink); --st-heading-color: var(--cs-ink); --st-border-color: var(--cs-border); --st-link-color: var(--cs-blue); }
.stApp { background: var(--cs-canvas); color: var(--cs-ink); }
[data-testid="stSidebar"] { background: var(--cs-surface); color: var(--cs-ink); border-right: 1px solid var(--cs-border); }
[data-testid="stHeader"] { background: color-mix(in srgb, var(--cs-canvas) 92%, transparent); }
[data-testid="stHeader"] svg, [data-testid="stHeader"] [data-testid="stIconMaterial"] { color: var(--cs-muted) !important; fill: currentColor !important; }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, label, [data-testid="stMarkdownContainer"] { color: var(--cs-ink); }
[data-testid="stCaptionContainer"], p, li { color: var(--cs-muted); }
a { color: var(--cs-blue); } hr { border-color: var(--cs-border) !important; }
[data-baseweb="input"] > div, [data-baseweb="select"] > div, textarea, input { background: var(--cs-surface) !important; color: var(--cs-ink) !important; border-color: var(--cs-border) !important; }
[data-baseweb="select"] span, [data-baseweb="input"] input, textarea { color: var(--cs-ink) !important; }
[data-baseweb="input"], [data-baseweb="textarea"], [role="combobox"], [role="combobox"] > div { background: var(--cs-surface) !important; border-color: var(--cs-border) !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within { border-color: color-mix(in srgb, var(--cs-accent) 56%, var(--cs-border)) !important; }
[data-baseweb="select"] svg, [role="combobox"] svg { fill: var(--cs-muted) !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stMultiSelect"] [data-baseweb="select"] > div { background: var(--cs-surface-soft) !important; border-color: var(--cs-border) !important; box-shadow: none !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within, [data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within { border-color: var(--cs-accent) !important; box-shadow: 0 0 0 1px var(--cs-accent) !important; }
[data-baseweb="tag"] { background: var(--cs-surface-raised) !important; border: 1px solid var(--cs-border) !important; color: var(--cs-accent-strong) !important; }
[data-testid="stSegmentedControl"] [data-baseweb="button-group"] { padding: 2px; border: 1px solid var(--cs-border); border-radius: 9px; background: var(--cs-surface-soft) !important; }
[data-testid="stSegmentedControl"] button { background: transparent !important; border-color: transparent !important; color: var(--cs-muted) !important; box-shadow: none !important; }
[data-testid="stSegmentedControl"] button:hover { background: color-mix(in srgb, var(--cs-accent) 6%, var(--cs-surface)) !important; color: var(--cs-ink) !important; }
[data-testid="stSegmentedControl"] button[aria-pressed="true"] { background: var(--cs-surface-raised) !important; color: var(--cs-accent-strong) !important; box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--cs-accent) 58%, var(--cs-border)) !important; }
div.stButton > button, div.stFormSubmitButton > button { background: var(--cs-surface-raised); color: var(--cs-ink); border-color: var(--cs-border); }
div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] { background: var(--cs-accent); color: #031116; border-color: var(--cs-accent); }
div.stButton > button:hover, div.stFormSubmitButton > button:hover { background: color-mix(in srgb, var(--cs-accent) 7%, var(--cs-surface-raised)); border-color: color-mix(in srgb, var(--cs-accent) 55%, var(--cs-border)); color: var(--cs-ink); }
div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button[kind="primary"]:hover { background: var(--cs-accent-strong); border-color: var(--cs-accent-strong); color: #031116; }
[data-testid="stDataFrame"] { border: 1px solid var(--cs-border); border-radius: 8px; overflow: hidden; background: var(--cs-surface); }
.cs-table-wrap { width: 100%; margin: .55rem 0 .8rem; overflow-x: auto; border: 1px solid var(--cs-border); border-radius: 8px; background: var(--cs-surface); }
.cs-table { width: 100%; min-width: 42rem; border-collapse: collapse; color: var(--cs-ink); font-size: .78rem; }
.cs-table th { padding: .66rem .72rem; background: var(--cs-surface-soft); color: var(--cs-muted); font-size: .68rem; font-weight: 720; letter-spacing: .035em; text-align: left; white-space: nowrap; }
.cs-table td { padding: .62rem .72rem; border-top: 1px solid var(--cs-border); color: var(--cs-ink); line-height: 1.4; white-space: nowrap; }
.cs-table tbody tr:hover { background: color-mix(in srgb, var(--cs-accent) 5%, transparent); }
[data-testid="stCheckbox"], [data-testid="stToggle"], [data-testid="stRadio"], [data-testid="stSlider"], [data-testid="stSelectSlider"] { color: var(--cs-ink) !important; }
[data-testid="stCheckbox"] [role="checkbox"], [data-testid="stRadio"] [role="radio"] { background: var(--cs-surface-raised) !important; border-color: var(--cs-border) !important; }
[data-testid="stCheckbox"] [role="checkbox"]:checked, [data-testid="stRadio"] [role="radio"]:checked, [data-testid="stToggle"] [role="switch"][aria-checked="true"] { background: var(--cs-accent) !important; border-color: var(--cs-accent) !important; }
[data-testid="stToggle"] [role="switch"] { background: var(--cs-surface-raised) !important; border-color: var(--cs-border) !important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .35rem; border-bottom-color: var(--cs-border); overflow-x: auto; scrollbar-width: thin; } [data-testid="stTabs"] button { color: var(--cs-muted); font-weight: 620; } [data-testid="stTabs"] button[aria-selected="true"] { color: var(--cs-accent-strong); }
[data-testid="stStatusWidget"] { background: var(--cs-surface); border: 1px solid var(--cs-border); border-radius: 8px; }
[data-baseweb="popover"], [data-baseweb="menu"], [data-baseweb="menu-list"], [data-testid="stPopover"] { background: var(--cs-surface) !important; color: var(--cs-ink) !important; border-color: var(--cs-border) !important; }
[role="option"], [data-baseweb="menu"] li, [data-baseweb="menu-list"] li { color: var(--cs-ink) !important; } [role="option"]:hover, [data-baseweb="menu"] li:hover, [data-baseweb="menu-list"] li:hover { background: var(--cs-surface-soft) !important; }
:focus-visible { outline: 3px solid var(--cs-accent) !important; outline-offset: 3px; box-shadow: 0 0 0 5px color-mix(in srgb, var(--cs-accent) 28%, transparent) !important; }
.stApp, [data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stMetric"], [data-testid="stExpander"], [data-testid="stForm"] { transition: background-color .25s ease, color .25s ease, border-color .25s ease; }
* { box-sizing: border-box; } html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; } html, body, .stApp { overflow-x: clip; }
.block-container { max-width: 1380px; padding: 1.8rem 2rem 4.5rem; } [data-testid="stSidebar"] > div:first-child { padding: 1rem 0.9rem 1.4rem; }
[data-testid="stSidebarNav"] { padding-top: .3rem; } [data-testid="stSidebarNav"] a { border-radius: 7px; margin: .08rem 0; min-height: 2.55rem; transition: background-color .18s ease, color .18s ease; }
[data-testid="stSidebarNav"] a:hover { background: color-mix(in srgb, var(--cs-accent) 9%, transparent); } [data-testid="stSidebarNav"] a[aria-current="page"] { background: color-mix(in srgb, var(--cs-accent) 10%, transparent); color: var(--cs-accent-strong); border-left: 2px solid var(--cs-accent); }
[data-testid="stSidebarNav"] span { font-size: .9rem; font-weight: 600; }
h1, h2, h3, h4, h5, h6 { letter-spacing: -.03em; } h1 { font-size: clamp(2rem, 4vw, 3.1rem); line-height: 1.04; } h2 { margin-top: 1.4rem; }
p, li, [data-testid="stCaptionContainer"] { overflow-wrap: anywhere; }
button, input, textarea { border-radius: 8px !important; }
div.stButton > button, div.stFormSubmitButton > button, div.stPageLink > a { min-height: 2.65rem; font-weight: 650; transition: background-color .16s ease, border-color .16s ease; }
[data-testid="stMetric"] { min-height: 100%; background: var(--cs-surface); border: 1px solid var(--cs-border); border-radius: 9px; padding: 0.9rem 1rem; box-shadow: none; }
[data-testid="stMetricLabel"] { color: var(--cs-muted); font-weight: 620; } [data-testid="stMetricValue"] { color: var(--cs-ink); letter-spacing: -.02em; }
[data-testid="stExpander"] { background: var(--cs-surface); border: 1px solid var(--cs-border); border-radius: 8px; overflow: hidden; } [data-testid="stExpander"] summary { color: var(--cs-ink); font-weight: 620; }
[data-testid="stVegaLiteChart"] { margin: .4rem 0 .75rem; padding: .7rem .75rem .35rem; border: 1px solid var(--cs-border); border-radius: 9px; background: var(--cs-surface); overflow: hidden; }
[data-testid="stVegaLiteChart"] canvas, [data-testid="stVegaLiteChart"] svg { border-radius: 9px; }
[data-testid="stForm"] { background: var(--cs-surface); border: 1px solid var(--cs-border); border-radius: 10px; padding: 1.1rem; box-shadow: none; } [data-testid="stTabs"] [role="tablist"] { gap: .35rem; overflow-x: auto; scrollbar-width: thin; }
[data-testid="stAlert"] { border-radius: 8px; border: 1px solid var(--cs-border); }
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-sticky-panel) { position: sticky; top: 1rem; align-self: flex-start; max-height: calc(100vh - 2rem); overflow-y: auto; border-radius: 10px; }
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"] { background: transparent; } * { scrollbar-color: var(--cs-border) transparent; }
[data-testid="stEmpty"] { color: var(--cs-muted) !important; }
.cs-brand { display: flex; align-items: center; gap: .7rem; margin: .1rem 0 .8rem; } .cs-mark { display: grid; place-items: center; width: 2.1rem; height: 2.1rem; border-radius: 6px; background: var(--cs-accent); color: #031116; box-shadow: none; font-weight: 800; }
.cs-brand-name { color: var(--cs-ink); font-size: 1.1rem; font-weight: 750; letter-spacing: -.025em; } .cs-brand-subtitle { color: var(--cs-subtle); font-size: .66rem; letter-spacing: .1em; text-transform: uppercase; }
.cs-eyebrow { color: var(--cs-accent-strong); font-size: .68rem; font-weight: 760; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .6rem; }
.cs-page-header { max-width: 920px; margin: .15rem 0 1.4rem; } .cs-page-header h1 { margin: 0 0 .55rem; max-width: 880px; } .cs-page-header p { color: var(--cs-muted); font-size: 1rem; max-width: 760px; line-height: 1.6; margin: 0; }
.cs-hero { position: relative; overflow: hidden; padding: clamp(1.25rem, 3vw, 1.65rem); border: 1px solid var(--cs-border); border-radius: 10px; background: var(--cs-surface); box-shadow: none; }
.cs-hero h1 { max-width: 820px; margin: 0 0 .8rem; font-size: clamp(1.7rem, 3.4vw, 2.4rem); } .cs-hero p { max-width: 760px; color: var(--cs-muted); font-size: 1rem; line-height: 1.6; }
.cs-chip { display: inline-flex; align-items: center; gap: .35rem; padding: .3rem .6rem; border-radius: 6px; background: color-mix(in srgb, var(--cs-accent) 8%, var(--cs-surface)); border: 1px solid color-mix(in srgb, var(--cs-accent) 16%, var(--cs-border)); color: var(--cs-accent-strong); font-size: .68rem; font-weight: 720; } .cs-chip-muted { background: var(--cs-surface-soft); border-color: var(--cs-border); color: var(--cs-muted); }
.cs-home-hero { margin-bottom: 1rem; background: var(--cs-surface); } .cs-home-hero-chips { display: flex; flex-wrap: wrap; gap: .45rem; margin-top: 1.15rem; }
.cs-home-section-label { margin: 1.45rem 0 .2rem; color: var(--cs-accent-strong); font-size: .66rem; font-weight: 760; letter-spacing: .13em; text-transform: uppercase; }
.st-key-home-actions [data-testid="stHorizontalBlock"] { align-items: stretch; gap: 1rem; } .st-key-home-actions [data-testid="stColumn"] { min-width: 0; } .cs-home-action { min-height: 10rem; padding: .95rem 1rem; overflow: hidden; border: 1px solid var(--cs-border); border-radius: 9px; background: var(--cs-surface); } .cs-home-action-primary { border-color: color-mix(in srgb, var(--cs-accent) 36%, var(--cs-border)); background: color-mix(in srgb, var(--cs-accent) 3%, var(--cs-surface)); } .cs-home-action > span { color: var(--cs-accent-strong); font-size: .62rem; font-weight: 760; letter-spacing: .1em; text-transform: uppercase; } .cs-home-action h3 { margin: .45rem 0 .35rem; font-size: 1.05rem; } .cs-home-action p { margin: 0; color: var(--cs-muted); font-size: .84rem; line-height: 1.5; overflow-wrap: anywhere; }
.cs-home-pulse-meta { display: flex; flex-wrap: wrap; align-items: center; gap: .4rem .8rem; margin: .2rem 0 .85rem; color: var(--cs-subtle); font-size: .72rem; } .cs-home-state { padding: .22rem .48rem; border: 1px solid var(--cs-border); border-radius: 999px; font-weight: 720; } .cs-home-state-live { color: var(--cs-positive); border-color: color-mix(in srgb, var(--cs-positive) 30%, var(--cs-border)); background: color-mix(in srgb, var(--cs-positive) 8%, var(--cs-surface)); } .cs-home-state-stale { color: var(--cs-amber); border-color: color-mix(in srgb, var(--cs-amber) 30%, var(--cs-border)); background: color-mix(in srgb, var(--cs-amber) 8%, var(--cs-surface)); }
.cs-home-mover { margin-top: .1rem; padding: .9rem 1rem; border: 1px solid var(--cs-border); border-radius: 13px; background: var(--cs-surface); } .cs-home-mover > span { display: block; color: var(--cs-subtle); font-size: .65rem; font-weight: 720; letter-spacing: .08em; text-transform: uppercase; } .cs-home-mover strong { display: block; margin-top: .25rem; color: var(--cs-ink); font-size: 1.15rem; } .cs-home-mover p { margin: .12rem 0 0; } .cs-home-mover-positive { border-left: 3px solid var(--cs-positive); } .cs-home-mover-negative { border-left: 3px solid var(--cs-amber); }
.cs-home-coverage { margin-top: .8rem; padding: .7rem .8rem; border-left: 3px solid var(--cs-amber); border-radius: 0 10px 10px 0; background: color-mix(in srgb, var(--cs-amber) 6%, var(--cs-surface)); color: var(--cs-muted); font-size: .78rem; line-height: 1.5; }
.cs-sidebar-profile { display: flex; align-items: center; gap: .65rem; margin: .65rem 0 .8rem; padding: .65rem .15rem; border: 1px solid var(--cs-border); border-width: 1px 0; background: transparent; min-width: 0; } .cs-sidebar-profile > span, .cs-profile-avatar { display: grid; place-items: center; flex: 0 0 auto; width: 2.1rem; height: 2.1rem; border-radius: 50%; background: var(--cs-surface-raised); border: 1px solid var(--cs-border); color: var(--cs-accent-strong); font-weight: 800; } .cs-sidebar-profile div { min-width: 0; } .cs-sidebar-profile strong, .cs-sidebar-profile small { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; } .cs-sidebar-profile strong { color: var(--cs-ink); font-size: .78rem; } .cs-sidebar-profile small { color: var(--cs-subtle); font-size: .64rem; }
.cs-profile-verification { display: inline-flex; align-items: center; margin: -.1rem 0 .35rem; padding: .22rem .48rem; border: 1px solid color-mix(in srgb, var(--cs-positive) 30%, var(--cs-border)); border-radius: 999px; background: color-mix(in srgb, var(--cs-positive) 8%, var(--cs-surface)); color: var(--cs-positive); font-size: .66rem; font-weight: 720; }
.cs-profile-avatar { width: 5.5rem; height: 5.5rem; font-size: 1.8rem; box-shadow: none; }
.cs-forecast-summary { margin: .7rem 0 .9rem; padding: .8rem .9rem; border: 1px solid var(--cs-border); border-radius: 12px; background: var(--cs-surface-soft); } .cs-forecast-summary-row { display: grid; grid-template-columns: minmax(5.5rem, .75fr) minmax(11rem, 1.4fr) minmax(4.5rem, .45fr) minmax(11rem, 1fr); align-items: center; gap: .55rem; padding: .5rem 0; color: var(--cs-ink); font-size: .78rem; } .cs-forecast-summary-row + .cs-forecast-summary-row { border-top: 1px solid var(--cs-border); } .cs-forecast-summary-row small { color: var(--cs-subtle); } .cs-forecast-price-path b { color: var(--cs-accent-strong); } .cs-forecast-summary-meta { display: flex; flex-direction: column; align-items: flex-start; gap: .2rem; } .cs-forecast-quality { font-weight: 700; } .cs-forecast-quality.is-trusted { color: var(--cs-positive); } .cs-forecast-quality.is-untrusted { color: var(--cs-amber); }
.cs-agent-header { display: flex; align-items: center; justify-content: space-between; gap: .7rem; margin-bottom: .35rem; color: var(--cs-ink); } .cs-agent-state { display: inline-flex; align-items: center; padding: .26rem .55rem; border-radius: 999px; border: 1px solid var(--cs-border); font-size: .68rem; font-weight: 720; white-space: nowrap; } .cs-agent-state-live { color: var(--cs-positive); background: color-mix(in srgb, var(--cs-positive) 10%, var(--cs-surface)); border-color: color-mix(in srgb, var(--cs-positive) 28%, var(--cs-border)); } .cs-agent-state-partial { color: var(--cs-amber); background: color-mix(in srgb, var(--cs-amber) 10%, var(--cs-surface)); border-color: color-mix(in srgb, var(--cs-amber) 28%, var(--cs-border)); }
.cs-agent-meta { display: flex; flex-wrap: wrap; align-items: baseline; gap: .28rem .55rem; margin: .1rem 0 .8rem; color: var(--cs-subtle); font-size: .7rem; } .cs-agent-meta > span { font-weight: 720; letter-spacing: .06em; text-transform: uppercase; } .cs-agent-meta > strong { color: var(--cs-muted); font-weight: 620; }
[data-testid="stExpander"] [data-testid="stExpander"] { margin-top: .9rem; } [data-testid="stExpander"] [data-testid="stExpander"] details { border-color: color-mix(in srgb, var(--cs-accent) 20%, var(--cs-border)); background: color-mix(in srgb, var(--cs-accent) 3%, var(--cs-surface)); }
.cs-coverage-note { margin: .65rem 0 .9rem; padding: .55rem .7rem; border-left: 2px solid var(--cs-amber); border-radius: 0 8px 8px 0; background: color-mix(in srgb, var(--cs-amber) 5%, transparent); color: var(--cs-muted); font-size: .76rem; line-height: 1.45; }
.cs-analysis-verdict, .cs-analysis-comparison { margin: .8rem 0; padding: .85rem .95rem; border: 1px solid color-mix(in srgb, var(--cs-accent) 22%, var(--cs-border)); border-radius: 12px; background: color-mix(in srgb, var(--cs-accent) 6%, var(--cs-surface)); }
.cs-analysis-verdict-header { display: flex; align-items: center; justify-content: space-between; gap: .7rem; margin-bottom: .35rem; } .cs-analysis-verdict-header .cs-analysis-kicker { margin-bottom: 0; } .cs-analysis-confidence { color: var(--cs-subtle); font-size: .66rem; font-weight: 700; white-space: nowrap; }
.cs-analysis-comparison { margin-top: .9rem; border-color: color-mix(in srgb, var(--cs-blue) 24%, var(--cs-border)); background: color-mix(in srgb, var(--cs-blue) 5%, var(--cs-surface)); }
.cs-analysis-kicker, .cs-analysis-block > span, .cs-analysis-development > span, .cs-analysis-empty > span { display: block; margin-bottom: .28rem; color: var(--cs-accent-strong); font-size: .64rem; font-weight: 760; letter-spacing: .1em; text-transform: uppercase; }
.cs-analysis-verdict p, .cs-analysis-comparison p, .cs-analysis-block p, .cs-analysis-development p, .cs-analysis-empty p { margin: 0; color: var(--cs-ink); line-height: 1.55; }
.cs-analysis-asset { position: relative; margin: .75rem 0; padding: .95rem 1rem; border: 1px solid var(--cs-border); border-radius: 13px; background: color-mix(in srgb, var(--cs-surface-raised) 54%, var(--cs-surface)); }
.cs-analysis-asset::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 3px; border-radius: 13px 0 0 13px; background: var(--cs-accent); opacity: .72; }
.cs-analysis-asset.cs-risk-moderate::before, .cs-analysis-asset.cs-risk-high::before, .cs-analysis-asset.cs-risk-very-high::before { background: var(--cs-amber); }
.cs-analysis-asset-header { display: flex; align-items: center; justify-content: space-between; gap: .6rem; margin-bottom: .65rem; } .cs-analysis-asset-header h5 { margin: 0; color: var(--cs-ink); font-size: .95rem; letter-spacing: -.01em; }
.cs-analysis-chips { display: flex; flex-wrap: wrap; gap: .38rem; margin: 0 0 .75rem; }
.cs-analysis-chip { display: inline-flex; align-items: baseline; gap: .32rem; padding: .3rem .48rem; border: 1px solid var(--cs-border); border-radius: 8px; background: var(--cs-surface-soft); color: var(--cs-ink); font-size: .72rem; }
.cs-analysis-chip-label { color: var(--cs-subtle); font-size: .63rem; font-weight: 680; text-transform: uppercase; letter-spacing: .04em; }
.cs-analysis-block { padding-top: .65rem; border-top: 1px solid color-mix(in srgb, var(--cs-border) 70%, transparent); } .cs-analysis-block + .cs-analysis-block { margin-top: .65rem; }
.cs-analysis-block-risk > span { color: var(--cs-blue); } .cs-analysis-development { margin-bottom: .7rem; padding: .65rem .75rem; border-radius: 9px; background: var(--cs-surface-soft); } .cs-analysis-development p { font-weight: 620; }
.cs-analysis-empty { margin-bottom: .7rem; padding: .65rem .75rem; border: 1px dashed var(--cs-border); border-radius: 9px; color: var(--cs-muted); }
.cs-analysis-gap { margin-top: .7rem; color: var(--cs-subtle); font-size: .7rem; line-height: 1.45; }
.cs-panel-section-heading { display: flex; flex-wrap: wrap; align-items: baseline; gap: .28rem .65rem; margin: 1.15rem 0 .65rem; padding-top: .9rem; border-top: 1px solid var(--cs-border); } .cs-panel-section-heading strong { color: var(--cs-ink); font-size: .86rem; } .cs-panel-section-heading span { color: var(--cs-subtle); font-size: .7rem; }
.cs-resource-card { overflow: hidden; margin: 0 0 .7rem; border: 1px solid var(--cs-border); border-radius: 8px; background: var(--cs-surface); } .cs-resource-card header { display: flex; align-items: center; justify-content: space-between; gap: .6rem; padding: .68rem .75rem; border-bottom: 1px solid var(--cs-border); background: var(--cs-surface-soft); color: var(--cs-ink); font-size: .78rem; } .cs-resource-state { flex: none; padding: .18rem .4rem; border: 1px solid var(--cs-border); border-radius: 999px; color: var(--cs-subtle); font-size: .6rem; font-weight: 720; text-transform: uppercase; } .cs-resource-state.is-complete { color: var(--cs-positive); border-color: color-mix(in srgb, var(--cs-positive) 28%, var(--cs-border)); } .cs-resource-state.is-limited { color: var(--cs-amber); border-color: color-mix(in srgb, var(--cs-amber) 28%, var(--cs-border)); }
.cs-resource-card dl { margin: 0; padding: .35rem .75rem .5rem; } .cs-resource-fact { display: grid; grid-template-columns: minmax(6.5rem, .85fr) minmax(0, 1.15fr); gap: .6rem; padding: .38rem 0; border-bottom: 1px solid color-mix(in srgb, var(--cs-border) 62%, transparent); } .cs-resource-fact:last-child { border-bottom: 0; } .cs-resource-fact dt { color: var(--cs-subtle); font-size: .68rem; } .cs-resource-fact dd { min-width: 0; margin: 0; color: var(--cs-ink); font-size: .72rem; font-weight: 620; overflow-wrap: anywhere; text-align: right; } .cs-resource-card-note, .cs-resource-note { display: flex; gap: .45rem; padding: .55rem .75rem; border-top: 1px solid color-mix(in srgb, var(--cs-amber) 22%, var(--cs-border)); background: color-mix(in srgb, var(--cs-amber) 5%, var(--cs-surface)); color: var(--cs-muted); font-size: .68rem; line-height: 1.45; } .cs-resource-card-note strong, .cs-resource-note strong { color: var(--cs-amber); } .cs-resource-note { margin-top: .7rem; border: 1px solid color-mix(in srgb, var(--cs-amber) 22%, var(--cs-border)); border-radius: 9px; }
.cs-claim-card { margin: 0 0 .55rem; padding: .7rem .75rem; border: 1px solid var(--cs-border); border-radius: 10px; background: color-mix(in srgb, var(--cs-surface-raised) 45%, var(--cs-surface)); } .cs-claim-card p { margin: 0 0 .55rem; color: var(--cs-ink); font-size: .76rem; line-height: 1.5; } .cs-evidence-ids { display: flex; flex-wrap: wrap; align-items: center; gap: .3rem; } .cs-evidence-ids > span:first-child { margin-right: .1rem; color: var(--cs-subtle); font-size: .6rem; font-weight: 720; letter-spacing: .06em; text-transform: uppercase; } .cs-evidence-id { padding: .18rem .35rem; border-radius: 6px; background: var(--cs-surface-soft); color: var(--cs-accent-strong); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .62rem; overflow-wrap: anywhere; } .cs-evidence-ids em { color: var(--cs-subtle); font-size: .65rem; }
.cs-card { height: 100%; padding: 1rem 1.1rem; border-radius: 9px; background: var(--cs-surface); border: 1px solid var(--cs-border); box-shadow: none; } .cs-card h3 { font-size: 1rem; margin: .7rem 0 .35rem; } .cs-card p { color: var(--cs-muted); line-height: 1.5; font-size: .9rem; } .cs-icon { color: var(--cs-accent-strong); font-size: 1.25rem; }
.cs-section { margin: 1.8rem 0 .6rem; } .cs-section h2 { margin: 0; font-size: 1.2rem; }
.cs-section-description { margin: -.1rem 0 .9rem; color: var(--cs-muted); max-width: 760px; } .cs-disclaimer { padding: .7rem .85rem; border: 1px solid var(--cs-border); border-radius: 12px; color: var(--cs-muted); font-size: .82rem; line-height: 1.5; }
.cs-disclaimer { border-left: 3px solid var(--cs-amber); background: color-mix(in srgb, var(--cs-amber) 6%, var(--cs-surface)); }
.cs-stepper { display: flex; gap: .35rem; flex-wrap: wrap; margin: .45rem 0 1.2rem; } .cs-step { display: inline-flex; align-items: center; gap: .35rem; padding: .34rem .6rem; border-radius: 999px; background: var(--cs-surface-soft); border: 1px solid var(--cs-border); color: var(--cs-muted); font-size: .72rem; }
.cs-step-active { background: color-mix(in srgb, var(--cs-accent) 9%, var(--cs-surface)); color: var(--cs-accent-strong); border-color: color-mix(in srgb, var(--cs-accent) 22%, var(--cs-border)); }
.cs-sidebar-label { margin: .85rem 0 .25rem; color: var(--cs-subtle); font-size: .64rem; font-weight: 760; letter-spacing: .12em; text-transform: uppercase; } .cs-sidebar-status { display: flex; align-items: center; gap: .4rem; color: var(--cs-muted); font-size: .74rem; margin: .55rem 0; }
.cs-status-dot { width: .42rem; height: .42rem; border-radius: 50%; background: var(--cs-positive); box-shadow: none; } .cs-status-dot-muted { background: var(--cs-amber); } .cs-status-dot-error { background: var(--cs-danger); }
.cs-footer { margin-top: 1.4rem; padding-top: .7rem; border-top: 1px solid var(--cs-border); color: var(--cs-subtle); font-size: .72rem; line-height: 1.45; }
@media (max-width: 900px) {
    .block-container { padding: 1.45rem 1rem 4.5rem; } .cs-hero { border-radius: 10px; padding: 1.35rem; }
    .cs-page-header { margin-bottom: 1.4rem; } [data-testid="stHorizontalBlock"] { gap: .75rem; }
    [data-testid="stHorizontalBlock"]:has(.cs-sticky-panel) { flex-wrap: wrap; }
    [data-testid="stHorizontalBlock"]:has(.cs-sticky-panel) > [data-testid="stColumn"] { flex: 1 1 100% !important; min-width: 100% !important; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.cs-sticky-panel) { position: static; width: 100%; max-height: none; overflow: visible; }
    .st-key-home-actions [data-testid="stHorizontalBlock"], [data-testid="stHorizontalBlock"]:has(.cs-home-asset-card) { flex-wrap: wrap; }
    .st-key-home-actions [data-testid="stHorizontalBlock"] > [data-testid="stColumn"], [data-testid="stHorizontalBlock"]:has(.cs-home-asset-card) > [data-testid="stColumn"] { flex: 1 1 calc(50% - .5rem) !important; min-width: min(17rem, 100%) !important; }
    [data-testid="stSidebar"] { max-width: min(86vw, 22rem); }
}
@media (max-width: 640px) {
    h1 { font-size: 2.15rem; } .cs-hero h1 { font-size: 2rem; } .cs-section { margin-top: 2rem; }
    [data-testid="stMetric"] { padding: .85rem .9rem; } [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    [data-testid="stColumn"] { flex: 1 1 100% !important; min-width: 100% !important; }
    [data-testid="stForm"] { padding: 1rem; border-radius: 10px; } div.stButton > button, div.stFormSubmitButton > button, div.stPageLink > a { width: 100%; }
    .cs-analysis-asset { padding: .85rem .8rem; } .cs-analysis-chip { flex: 1 1 auto; justify-content: space-between; } .cs-analysis-verdict, .cs-analysis-comparison { padding: .75rem .8rem; } .cs-home-action { min-height: auto; } .cs-home-pulse-meta { align-items: flex-start; flex-direction: column; gap: .35rem; } .cs-forecast-summary-row { grid-template-columns: 1fr; gap: .15rem; }
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; animation: none !important; } }
</style>
"""

BASE_THEME_CSS = _BASE_THEME_CSS.replace("</style>", TERMINAL_UI_CSS + "\n</style>")


def apply_theme() -> None:
    """Install the shared visual system once per Streamlit rerun.

    Colors follow the single supported (dark) palette so native widgets,
    custom CSS, and Vega charts use the same tokens.
    """

    initialize_theme_state()
    css = BASE_THEME_CSS.replace("__THEME_TOKENS__", theme_token_css())
    st.markdown(css, unsafe_allow_html=True)


def render_brand(*, compact: bool = False) -> None:
    subtitle = "Research workspace" if compact else "Crypto research workspace"
    st.markdown(
        f'<div class="cs-brand"><div class="cs-mark">C</div><div>'
        f'<div class="cs-brand-name">ChainScope</div>'
        f'<div class="cs-brand-subtitle">{subtitle}</div></div></div>',
        unsafe_allow_html=True,
    )


def render_page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="cs-page-header"><div class="cs-eyebrow">{html.escape(eyebrow)}</div>'
        f"<h1>{html.escape(title)}</h1><p>{html.escape(description)}</p></div>",
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str = "") -> None:
    st.markdown(
        f'<div class="cs-section"><h2>{html.escape(title)}</h2></div>',
        unsafe_allow_html=True,
    )
    if description:
        st.markdown(
            f'<p class="cs-section-description">{html.escape(description)}</p>',
            unsafe_allow_html=True,
        )


def render_disclaimer(text: str) -> None:
    st.markdown(f'<div class="cs-disclaimer">{html.escape(text)}</div>', unsafe_allow_html=True)


def render_data_table(
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str] | None = None,
    label: str = "Data table",
) -> None:
    """Render a theme-safe, responsive table without a canvas dependency."""

    if not rows:
        st.info("No table rows are available for this result.")
        return
    selected = tuple(columns or rows[0].keys())
    header = "".join(f'<th scope="col">{html.escape(str(column))}</th>' for column in selected)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in selected)
        + "</tr>"
        for row in rows
    )
    st.markdown(
        f'<div class="cs-table-wrap" role="region" aria-label="{html.escape(label)}" '
        f'tabindex="0"><table class="cs-table"><thead><tr>{header}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_empty_panel(title: str, body: str, *, icon: str = "—") -> None:
    st.markdown(
        f'<div class="cs-card"><div class="cs-icon">{html.escape(icon)}</div>'
        f"<h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></div>",
        unsafe_allow_html=True,
    )
