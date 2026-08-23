from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from crypto_research.domain.core import StrictModel, SupportedExchange, SupportedTimeframe
from crypto_research.shared.time import timeframe_delta

MarketWindowLabel = Literal["1h", "4h", "24h", "7d"]


class FutureMarketDataError(ValueError): ...


class MarketDataQuality(StrictModel):
    """Auditable normalization outcomes for provider OHLCV observations."""

    accepted_candles: int = Field(default=0, ge=0)
    excluded_future: int = Field(default=0, ge=0)
    excluded_incomplete: int = Field(default=0, ge=0)
    excluded_malformed: int = Field(default=0, ge=0)
    excluded_misaligned: int = Field(default=0, ge=0)
    excluded_duplicates: int = Field(default=0, ge=0)
    excluded_noncontiguous_prefix: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list, max_length=12)


class Candle(StrictModel):
    timestamp: datetime
    open: float = Field(gt=0, allow_inf_nan=False)
    high: float = Field(gt=0, allow_inf_nan=False)
    low: float = Field(gt=0, allow_inf_nan=False)
    close: float = Field(gt=0, allow_inf_nan=False)
    volume: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("Candle timestamp must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_ohlc(self) -> Candle:
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("High/low must contain open and close.")
        return self


class MarketEvidence(StrictModel):
    exchange: SupportedExchange
    symbol: str = Field(min_length=3)
    timeframe: SupportedTimeframe
    candles: list[Candle] = Field(min_length=1)
    first_time: datetime
    last_time: datetime
    current_price: float = Field(gt=0, allow_inf_nan=False)
    collected_at: datetime
    coin_id: str | None = None
    data_source: str = Field(default="exchange OHLCV adapter", min_length=1, max_length=120)
    # A cache fallback copies the validated snapshot with this marker.  It is
    # intentionally default-safe for serialized evidence.
    source_state: Literal["live", "cached"] = "live"
    data_quality: MarketDataQuality = Field(default_factory=MarketDataQuality)

    @field_validator("exchange", mode="before")
    @classmethod
    def normalize_exchange(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("timeframe", mode="before")
    @classmethod
    def validate_timeframe(cls, value: str) -> str:
        normalized = value.strip().lower()
        timeframe_delta(normalized)
        return normalized

    @field_validator("first_time", "last_time", "collected_at")
    @classmethod
    def require_collected_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("Market collection timestamp must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.first_time != self.candles[0].timestamp:
            raise ValueError("first_time must match first candle.")
        if self.last_time != self.candles[-1].timestamp:
            raise ValueError("last_time must match last candle.")
        timestamps = [candle.timestamp for candle in self.candles]
        if any(
            current <= previous
            for previous, current in zip(timestamps, timestamps[1:], strict=False)
        ):
            raise ValueError("Candles must have unique, strictly increasing timestamps.")
        if self.current_price != self.candles[-1].close:
            raise ValueError("current_price must match the last candle close.")
        if self.collected_at < self.last_time:
            raise ValueError("collected_at cannot be earlier than the latest candle.")
        if self.data_quality.accepted_candles == 0:
            object.__setattr__(
                self,
                "data_quality",
                self.data_quality.model_copy(update={"accepted_candles": len(self.candles)}),
            )
        return self


class ComparisonMetrics(StrictModel):
    """Comparable metrics calculated from the same market window."""

    period_start: datetime
    period_end: datetime
    start_price: float = Field(gt=0, allow_inf_nan=False)
    end_price: float = Field(gt=0, allow_inf_nan=False)
    price_return: float = Field(allow_inf_nan=False)
    volatility: float = Field(ge=0, allow_inf_nan=False)
    high: float = Field(gt=0, allow_inf_nan=False)
    low: float = Field(gt=0, allow_inf_nan=False)
    total_volume: float = Field(ge=0, allow_inf_nan=False)
    observation_count: int = Field(gt=0)


class MarketWindowReturn(StrictModel):
    """Deterministic return over one exact, contiguous market window."""

    label: MarketWindowLabel
    hours: int = Field(gt=0)
    status: Literal["available", "unavailable", "not_applicable"]
    reference_price: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    latest_price: float = Field(gt=0, allow_inf_nan=False)
    change_absolute: float | None = Field(default=None, allow_inf_nan=False)
    return_decimal: float | None = Field(default=None, allow_inf_nan=False)
    return_percent: float | None = Field(default=None, allow_inf_nan=False)
    period_start: datetime | None = None
    period_end: datetime
    reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        calculated = (
            self.reference_price,
            self.change_absolute,
            self.return_decimal,
            self.return_percent,
            self.period_start,
        )
        if self.status == "available" and (
            any(value is None for value in calculated) or self.reason
        ):
            raise ValueError("Available market returns require complete values and no reason.")
        if self.status in {"unavailable", "not_applicable"} and (
            any(value is not None for value in calculated) or not (self.reason or "").strip()
        ):
            raise ValueError("Unavailable market returns require one reason and no values.")
        return self


class MarketFeatureSummary(StrictModel):
    """Compact deterministic OHLCV features suitable for an LLM evidence payload."""

    latest_completed_close: float = Field(gt=0, allow_inf_nan=False)
    observed_at: datetime
    window_start: datetime
    window_end: datetime
    candle_count: int = Field(gt=0)
    high: float = Field(gt=0, allow_inf_nan=False)
    low: float = Field(gt=0, allow_inf_nan=False)
    range_absolute: float = Field(ge=0, allow_inf_nan=False)
    range_percent: float = Field(ge=0, allow_inf_nan=False)
    base_volume: float = Field(ge=0, allow_inf_nan=False)
    quote_volume: float = Field(ge=0, allow_inf_nan=False)
    maximum_drawdown: float = Field(ge=0, le=1, allow_inf_nan=False)
    contiguous: bool
    fresh_at_collection: bool
    data_delay_seconds: float = Field(ge=0, allow_inf_nan=False)
    returns: list[MarketWindowReturn] = Field(min_length=4, max_length=4)


class MarketPostureSummary(StrictModel):
    """Deterministic market posture shared by the summary, ledger, and scope digest.

    A single source of truth built from validated OHLCV features, the technical
    snapshot, and any collected higher-timeframe confirmation.  The market-agent
    summary text, the evidence ledger, and the LLM scope digest all derive from
    it so the numbers never drift between consumers.
    """

    symbol: str = Field(min_length=1, max_length=40)
    exchange: str = Field(min_length=1, max_length=40)
    timeframe: str = Field(min_length=1, max_length=10)
    as_of: datetime
    collected_at: datetime
    price: float = Field(gt=0, allow_inf_nan=False)
    change_24h_percent: float | None = Field(default=None, allow_inf_nan=False)
    change_24h_absolute: float | None = Field(default=None, allow_inf_nan=False)
    window_returns: list[MarketWindowReturn] = Field(default_factory=list, max_length=4)
    high: float = Field(gt=0, allow_inf_nan=False)
    low: float = Field(gt=0, allow_inf_nan=False)
    range_percent: float = Field(ge=0, allow_inf_nan=False)
    quote_volume: float = Field(ge=0, allow_inf_nan=False)
    maximum_drawdown: float = Field(ge=0, le=1, allow_inf_nan=False)
    trend: Literal["bullish", "bearish", "neutral"]
    rsi: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    rsi_band: str = Field(default="neutral", max_length=20)
    macd: float | None = Field(default=None, allow_inf_nan=False)
    atr: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    volatility: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    support: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    resistance: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    fresh: bool
    data_delay_seconds: float = Field(ge=0, allow_inf_nan=False)
    contextual_confirmation: list[str] = Field(default_factory=list, max_length=3)
