from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import as_file, files
from io import TextIOWrapper

import streamlit as st
from streamlit.web import cli as streamlit_cli

MIN_STREAMLIT_VERSION = (1, 59)


def _streamlit_version_tuple() -> tuple[int, ...]:
    try:
        raw = st.__version__
    except AttributeError:
        try:
            raw = version("streamlit")
        except PackageNotFoundError:
            return (0,)
    parts: list[int] = []
    for segment in raw.split("."):
        digits = ""
        for char in segment:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def streamlit_version_gap_message() -> str | None:
    """Return actionable guidance when Streamlit is too old, else ``None``."""
    installed = _streamlit_version_tuple()
    if installed >= MIN_STREAMLIT_VERSION:
        return None
    return (
        "ChainScope requires Streamlit 1.59 or newer, but this environment "
        f"has Streamlit {'.'.join(map(str, installed))}.\n"
        "Launch the UI from the project virtual environment instead, e.g.:\n"
        "  .\\.venv\\Scripts\\python.exe -m streamlit run "
        "src\\crypto_research\\interfaces\\web\\streamlit_app.py\n"
        'Or upgrade this environment with: pip install -e ".[dev]"'
    )


def main() -> None:
    """Launch the packaged Streamlit application."""

    gap_message = streamlit_version_gap_message()
    if gap_message is not None:
        raise SystemExit(gap_message)

    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    app_resource = files("crypto_research.interfaces.web").joinpath("streamlit_app.py")
    with as_file(app_resource) as app_path:
        sys.argv = [
            "streamlit",
            "run",
            str(app_path),
            *sys.argv[1:],
        ]
        raise SystemExit(streamlit_cli.main(prog_name="chainscope"))
