"""Supported research capabilities, market catalogs, and asset parsing policy."""

import re
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from crypto_research.shared.time import timeframe_delta


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ResearchCapability(StrEnum):
    DISCOVERY = "discovery"
    MARKET = "market"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    DEFI = "defi"
    RISK = "risk"
    FORECAST = "forecast"
    ONCHAIN = "onchain"
    DERIVATIVES = "derivatives"


SupportedExchange = Literal["kraken", "coinbase", "binance"]
SupportedTimeframe = Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
SUPPORTED_EXCHANGES: tuple[SupportedExchange, ...] = ("kraken", "coinbase", "binance")
SUPPORTED_TIMEFRAMES: tuple[SupportedTimeframe, ...] = (
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
)

DEFAULT_QUOTE_BY_EXCHANGE: dict[SupportedExchange, str] = {
    "kraken": "USD",
    "coinbase": "USD",
    "binance": "USDT",
}

ASSET_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("bitcoin", "btc"),
    "ETH": ("ethereum", "ether", "eth"),
    "SOL": ("solana", "sol"),
    "XRP": ("ripple", "xrp"),
    "ADA": ("cardano", "ada"),
    "DOGE": ("dogecoin", "doge"),
    "DOT": ("polkadot", "dot"),
    "AVAX": ("avalanche", "avax"),
    "LINK": ("chainlink", "link"),
    "LTC": ("litecoin", "ltc"),
    "AAVE": ("aave",),
    "USDC": ("usd coin", "usdc"),
}

# These ticker symbols are ordinary English words when lower-cased. Require the
# upper-case ticker (or the unambiguous project name) to avoid accidental routes.
_CASE_SENSITIVE_TICKERS = {"DOT", "LINK"}

COIN_ID_BY_ASSET = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "LTC": "litecoin",
    "AAVE": "aave",
    "USDC": "usd-coin",
}

MAX_COMPARISON_ASSETS = 4

_NON_ASSET_TICKERS = {
    "API",
    "APR",
    "ATH",
    "DEX",
    "ETF",
    "MACD",
    "NFT",
    "OHLC",
    "RSI",
    "TVL",
    "USD",
    "USDT",
}


def extract_supported_assets(text: str) -> list[str]:
    lowered = text.lower()
    matches: list[tuple[int, str]] = []
    for asset, aliases in ASSET_ALIASES.items():
        offsets: list[int] = []
        for alias in aliases:
            if asset in _CASE_SENSITIVE_TICKERS and alias == asset.lower():
                match = re.search(rf"\b{re.escape(asset)}\b", text)
            else:
                match = re.search(rf"\b{re.escape(alias)}\b", lowered)
            if match is not None:
                offsets.append(match.start())
        if offsets:
            matches.append((min(offsets), asset))
    matches.sort(key=lambda item: item[0])
    resolved = [asset for _, asset in matches]
    explicit = _extract_explicit_tickers(text)
    return list(dict.fromkeys((*resolved, *explicit)))


def canonical_asset_symbol(asset: str) -> str | None:
    """Return a known symbol or the compact ticker, or None when neither exists."""

    known = extract_supported_assets(asset)
    if known:
        return known[0]
    compact = "".join(character for character in asset.upper() if character.isalnum())
    return compact if 2 <= len(compact) <= 15 else None


def build_market_symbol(asset: str, exchange: SupportedExchange) -> str:
    quote = DEFAULT_QUOTE_BY_EXCHANGE[exchange]
    return f"{asset}/{quote}"


def normalize_market_symbol(value: str) -> str:
    normalized = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,15}/[A-Z0-9]{2,15}", normalized):
        raise ValueError("Symbol must use BASE/QUOTE format, for example BTC/USD.")
    return normalized


def _extract_explicit_tickers(text: str) -> list[str]:
    """Recognize explicitly marked unknown tickers without treating acronyms as assets."""

    candidates: list[str] = []
    candidates.extend(
        match.upper() for match in re.findall(r"\$([A-Za-z][A-Za-z0-9]{1,9})\b", text)
    )
    candidates.extend(
        match.upper()
        for match in re.findall(r"\b([A-Z][A-Z0-9]{1,9})/(?:USD|USDT|BTC|ETH)\b", text)
    )
    candidates.extend(
        match.upper()
        for match in re.findall(
            r"\b([A-Z][A-Z0-9]{1,9})\s+(?:coin|token|cryptocurrency)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    return list(
        dict.fromkeys(
            ticker
            for ticker in candidates
            if ticker not in _NON_ASSET_TICKERS and not ticker.isdigit()
        )
    )


RiskBand = Literal["low", "moderate", "high", "very_high"]


def risk_band(score: float) -> RiskBand:
    if score <= 24:
        return "low"
    if score <= 49:
        return "moderate"
    if score <= 74:
        return "high"
    return "very_high"


class RiskAssessment(StrictModel):
    score: float = Field(ge=0, le=100, allow_inf_nan=False)
    band: RiskBand
    factors: list[str] = Field(default_factory=list)
    components: dict[str, float] = Field(default_factory=dict)
    evidence_confidence: float = Field(default=0, ge=0, le=100, allow_inf_nan=False)
    coverage_gaps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_components_and_band(self) -> Self:
        if self.band != risk_band(self.score):
            raise ValueError("Risk band must match the numeric score.")
        if any(value < 0 or not isfinite(value) for value in self.components.values()):
            raise ValueError("Risk components must be finite and non-negative.")
        return self


class OpportunityCandidate(StrictModel):
    rank: int = Field(ge=1)
    asset: str = Field(min_length=2, max_length=15)
    symbol: str = Field(min_length=5, max_length=31)
    current_price: float = Field(gt=0, allow_inf_nan=False)
    score: float = Field(ge=0, le=100, allow_inf_nan=False)
    momentum_24h: float = Field(allow_inf_nan=False)
    volatility_24h: float = Field(ge=0, allow_inf_nan=False)
    trend: Literal["bullish", "bearish", "neutral"]
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("asset")
    @classmethod
    def normalize_asset(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()


class OpportunityScanResult(StrictModel):
    exchange: SupportedExchange
    timeframe: SupportedTimeframe
    candidates: list[OpportunityCandidate] = Field(min_length=1)
    collected_at: datetime
    summary: str = Field(min_length=1, max_length=3000)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("exchange", mode="before")
    @classmethod
    def normalize_exchange(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("timeframe", mode="before")
    @classmethod
    def validate_timeframe(cls, value: str) -> str:
        normalized = value.strip().lower()
        timeframe_delta(normalized)
        return normalized

    @field_validator("collected_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("Opportunity collection timestamp must be timezone-aware.")
        return value.astimezone(UTC)
