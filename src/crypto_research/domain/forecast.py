import re
from datetime import datetime, timedelta
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from crypto_research.domain.core import StrictModel, SupportedExchange, normalize_market_symbol
from crypto_research.domain.market import MarketEvidence

ForecastModelId = Literal["gradient_boosting_huber", "ridge"]
ForecastStatus = Literal["complete", "suppressed"]
ForecastFailureCode = Literal[
    "DATA_UNAVAILABLE",
    "STALE_DATA",
    "INSUFFICIENT_DATA",
    "INVALID_DATA",
    "MODEL_ERROR",
]


class ForecastSettings(StrictModel):
    """Shared settings for a Guided Forecasting batch."""

    timeframe: Literal["1h", "4h"] = "1h"
    horizon_hours: Literal[4, 8, 12, 24, 48] = 24
    model_id: ForecastModelId = "gradient_boosting_huber"
    confidence_level: float = Field(default=0.8, ge=0.8, le=0.9)
    lookback_candles: int = Field(default=750, ge=400, le=2000)

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if value not in {0.8, 0.9}:
            raise ValueError("Confidence level must be 0.8 or 0.9.")
        return value


class ForecastRequest(StrictModel):
    """Validated parameters for one dedicated forecasting run."""

    asset: str = Field(min_length=1, max_length=100)
    coin_id: str | None = None
    exchange: SupportedExchange = "kraken"
    symbol: str = "BTC/USD"
    timeframe: Literal["1h", "4h"] = "1h"
    horizon_hours: Literal[4, 8, 12, 24, 48] = 24
    model_id: ForecastModelId = "gradient_boosting_huber"
    confidence_level: float = Field(default=0.8, ge=0.8, le=0.9)
    lookback_candles: int = Field(default=750, ge=400, le=2000)

    @field_validator("asset")
    @classmethod
    def normalize_asset(cls, value: str) -> str:
        return value.strip()

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return normalize_market_symbol(value)

    @field_validator("coin_id")
    @classmethod
    def normalize_coin_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized):
            raise ValueError("Coin ID must be a lowercase slug.")
        return normalized

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence_level(cls, value: float) -> float:
        if value not in {0.8, 0.9}:
            raise ValueError("Confidence level must be 0.8 or 0.9.")
        return value


class ForecastModelDetails(StrictModel):
    model_id: ForecastModelId
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    feature_columns: list[str] = Field(min_length=1)
    hyperparameters: dict[str, str | int | float | bool]
    training_samples: int = Field(ge=0)
    validation_samples: int = Field(ge=0)
    validation_folds: int = Field(ge=0)
    random_state: int | None = None


class ForecastMetrics(StrictModel):
    mae: float = Field(ge=0, allow_inf_nan=False)
    rmse: float = Field(ge=0, allow_inf_nan=False)
    directional_accuracy: float = Field(ge=0, le=1, allow_inf_nan=False)
    baseline_mae: float = Field(ge=0, allow_inf_nan=False)
    mae_improvement: float = Field(allow_inf_nan=False)
    validation_samples: int = Field(ge=0)
    validation_folds: int = Field(ge=0)


class ForecastPoint(StrictModel):
    timestamp: datetime
    predicted_price: float = Field(gt=0, allow_inf_nan=False)
    predicted_return: float = Field(allow_inf_nan=False)
    lower_interval: float = Field(gt=0, allow_inf_nan=False)
    upper_interval: float = Field(gt=0, allow_inf_nan=False)

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("Forecast timestamp must be timezone-aware.")
        return value

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if not self.lower_interval <= self.predicted_price <= self.upper_interval:
            raise ValueError("Forecast interval must contain the predicted price.")
        return self


class ForecastQuality(StrictModel):
    passed: bool
    reasons: list[str]
    prediction_suppressed: bool = False

    @model_validator(mode="after")
    def match_suppression(self) -> Self:
        if self.prediction_suppressed != (not self.passed):
            raise ValueError("Prediction suppression must match forecast quality.")
        return self


class ForecastRun(StrictModel):
    """Complete, user-facing result from one model run."""

    status: ForecastStatus
    request: ForecastRequest
    market: MarketEvidence
    model: ForecastModelDetails
    metrics: ForecastMetrics
    quality: ForecastQuality
    model_output: ForecastPoint
    prediction: ForecastPoint | None = None
    interval_method: str | None = None
    limitations: list[str] = Field(default_factory=list)
    disclaimer: str = "Experimental machine-learning prediction only. This is not financial advice."

    @model_validator(mode="after")
    def validate_prediction_state(self) -> Self:
        if self.status == "complete" and self.prediction is None:
            raise ValueError("Complete forecast runs require a prediction.")
        if self.status == "suppressed" and self.prediction is not None:
            raise ValueError("Suppressed forecast runs cannot expose a prediction.")
        if self.status == "complete" and not self.quality.passed:
            raise ValueError("Complete forecast runs require passed quality checks.")
        if self.status == "suppressed" and self.quality.passed:
            raise ValueError("Suppressed forecast runs require failed quality checks.")
        if (
            self.market.exchange != self.request.exchange
            or self.market.symbol != self.request.symbol
            or self.market.timeframe != self.request.timeframe
        ):
            raise ValueError("Forecast request and market evidence must describe the same market.")
        if self.prediction is not None and self.prediction.timestamp != (
            self.market.last_time + timedelta(hours=self.request.horizon_hours)
        ):
            raise ValueError("Forecast timestamp must match the requested horizon.")
        if self.model_output.timestamp != (
            self.market.last_time + timedelta(hours=self.request.horizon_hours)
        ):
            raise ValueError("Raw model output timestamp must match the requested horizon.")
        return self


class ForecastFailure(StrictModel):
    """Expected, sanitized failure returned when a forecast cannot be produced."""

    status: Literal["unavailable"] = "unavailable"
    request: ForecastRequest
    code: ForecastFailureCode
    message: str = Field(min_length=1, max_length=500)
    limitations: list[str] = Field(default_factory=list)


ForecastAssetResult = ForecastRun | ForecastFailure


class ForecastAgentResult(StrictModel):
    """Ordered deterministic outputs for one Guided Forecasting batch."""

    settings: ForecastSettings
    asset_results: list[ForecastAssetResult] = Field(min_length=1, max_length=4)
