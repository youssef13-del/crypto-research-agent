"""Verified, LLM-assisted cryptocurrency research application."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("crypto-research-agent")
except PackageNotFoundError:
    __version__ = "0+unknown"
