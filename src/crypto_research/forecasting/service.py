"""Shared forecasting service with chronological validation and safe output gating."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore[import-untyped]
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]

from crypto_research.domain.forecast import (
    ForecastFailure,
    ForecastFailureCode,
    ForecastMetrics,
    ForecastModelDetails,
    ForecastPoint,
    ForecastQuality,
    ForecastRequest,
    ForecastRun,
)
from crypto_research.domain.market import Candle, MarketEvidence
from crypto_research.forecasting.models import FEATURE_COLUMNS, get_model_spec
from crypto_research.shared.time import timeframe_delta
from crypto_research.tools.market import fetch_market_evidence

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ForecastPolicy:
    minimum_training_samples: int = 200
    minimum_validation_samples: int = 50
    time_series_folds: int = 5
    minimum_mae_improvement: float = 0.02
    minimum_directional_accuracy: float = 0.52
    maximum_absolute_forecast_return: float = 0.20
    maximum_interval_width: float = 0.40


class ForecastService:
    """Fetch one verified market window and execute one registered model."""

    def __init__(
        self,
        *,
        policy: ForecastPolicy | None = None,
        market_fetcher: Callable[..., MarketEvidence] = fetch_market_evidence,
    ) -> None:
        self._policy = policy or ForecastPolicy()
        self._market_fetcher = market_fetcher

    @property
    def policy(self) -> ForecastPolicy:
        return self._policy

    def run(self, request: ForecastRequest) -> ForecastRun | ForecastFailure:
        try:
            market = self._market_fetcher(
                exchange_name=request.exchange,
                symbol=request.symbol,
                timeframe=request.timeframe,
                limit=request.lookback_candles,
                coin_id=request.coin_id,
                strict_contiguity=True,
            )
        except ValueError as exc:
            message = str(exc)
            code = _classify_data_failure(message)
            return ForecastFailure(
                request=request,
                code=code,
                message=message,
                limitations=["No point prediction was produced."],
            )
        except Exception as exc:
            return ForecastFailure(
                request=request,
                code="DATA_UNAVAILABLE",
                message="Historical market data could not be retrieved for this selection.",
                limitations=[f"The configured market provider returned {type(exc).__name__}."],
            )

        try:
            return _run_forecast(market=market, request=request, policy=self._policy)
        except ValueError as exc:
            message = str(exc)
            code = _classify_data_failure(message)
            return ForecastFailure(
                request=request,
                code=code,
                message=message,
                limitations=["No point prediction was produced."],
            )
        except Exception as exc:
            LOGGER.exception("Forecast model failed for %s", request.symbol)
            return ForecastFailure(
                request=request,
                code="MODEL_ERROR",
                message="The selected forecasting model could not complete.",
                limitations=[f"The model returned {type(exc).__name__}."],
            )


def _classify_data_failure(message: str) -> ForecastFailureCode:
    normalized = message.casefold()
    if "stale" in normalized:
        return "STALE_DATA"
    if "requires" in normalized or "too few" in normalized:
        return "INSUFFICIENT_DATA"
    return "INVALID_DATA"


def _run_forecast(
    *,
    market: MarketEvidence,
    request: ForecastRequest,
    policy: ForecastPolicy,
) -> ForecastRun:
    """Run one model against a contiguous, timestamped historical window."""

    _validate_request_market(request, market)
    candles = market.candles
    candle_delta = timeframe_delta(request.timeframe)
    candle_hours = int(candle_delta.total_seconds() // 3600)
    if request.horizon_hours % candle_hours != 0:
        raise ValueError("The selected forecast horizon must map to whole candles.")
    horizon = request.horizon_hours // candle_hours
    if not candles:
        raise ValueError("Forecast requires historical candles.")
    if any(
        current.timestamp - previous.timestamp != candle_delta
        for previous, current in zip(candles, candles[1:], strict=False)
    ):
        raise ValueError("Forecast candles must be contiguous at the selected timeframe.")

    frame = _build_feature_frame(candles, horizon=horizon)
    training_frame = frame.dropna(subset=[*FEATURE_COLUMNS, "target_return"])
    minimum_rows = (
        policy.minimum_training_samples + policy.minimum_validation_samples * 2 + horizon + 20
    )
    if len(training_frame) < minimum_rows:
        raise ValueError(
            "Forecast requires at least "
            f"{minimum_rows} complete rows; received {len(training_frame)}."
        )

    features = training_frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    target = training_frame["target_return"].to_numpy(dtype=np.float64)
    predictions, actual, folds = _chronological_predictions(
        features,
        target,
        folds=policy.time_series_folds,
        gap=horizon,
        minimum_training_samples=policy.minimum_training_samples,
        minimum_validation_samples=policy.minimum_validation_samples,
        model_factory=get_model_spec(request.model_id).factory,
    )
    metrics = _metrics(actual, predictions, validation_folds=folds)
    spec = get_model_spec(request.model_id)
    model = spec.factory()
    model.fit(features, target)
    latest = frame.loc[[frame.index[-1]], list(FEATURE_COLUMNS)]
    if latest.isna().any(axis=None):
        raise ValueError("The latest candle does not have a complete forecast feature row.")
    predicted_return = float(model.predict(latest.to_numpy(dtype=np.float64))[0])

    residuals = actual - predictions
    alpha = (1 - request.confidence_level) / 2
    lower_residual, upper_residual = np.percentile(residuals, [alpha * 100, (1 - alpha) * 100])
    current_price = candles[-1].close
    predicted_price = max(0.01, current_price * (1 + predicted_return))
    lower_price = max(0.01, current_price * (1 + predicted_return + float(lower_residual)))
    upper_price = max(0.01, current_price * (1 + predicted_return + float(upper_residual)))
    lower_price, upper_price = min(lower_price, predicted_price), max(upper_price, predicted_price)
    interval_width = (upper_price - lower_price) / current_price
    quality = _quality(metrics, predicted_return, interval_width, policy)
    model_details = ForecastModelDetails(
        model_id=request.model_id,
        display_name=spec.display_name,
        description=spec.description,
        feature_columns=spec.feature_columns,
        hyperparameters=spec.hyperparameters,
        training_samples=len(features),
        validation_samples=len(actual),
        validation_folds=folds,
        random_state=(
            int(spec.hyperparameters["random_state"])
            if isinstance(spec.hyperparameters.get("random_state"), int)
            else None
        ),
    )
    model_output = ForecastPoint(
        timestamp=candles[-1].timestamp + candle_delta * horizon,
        predicted_price=predicted_price,
        predicted_return=predicted_return,
        lower_interval=lower_price,
        upper_interval=upper_price,
    )
    prediction = None
    if quality.passed:
        prediction = model_output
    limitations = [
        "This is an experimental machine-learning output, not a guarantee or financial advice.",
        f"Intervals are empirical residual intervals at the {request.confidence_level:.0%} level.",
    ]
    if not quality.passed:
        limitations.append("The point prediction is withheld because quality checks did not pass.")
    return ForecastRun(
        status="complete" if quality.passed else "suppressed",
        request=request,
        market=market,
        model=model_details,
        metrics=metrics,
        quality=quality,
        model_output=model_output,
        prediction=prediction,
        interval_method="Empirical quantiles of purged walk-forward validation residuals",
        limitations=limitations,
    )


def _validate_request_market(request: ForecastRequest, market: MarketEvidence) -> None:
    mismatches: list[str] = []
    if request.exchange != market.exchange:
        mismatches.append("exchange")
    if request.symbol != market.symbol:
        mismatches.append("symbol")
    if request.timeframe != market.timeframe:
        mismatches.append("timeframe")
    if (
        request.coin_id is not None
        and market.coin_id is not None
        and request.coin_id != market.coin_id
    ):
        mismatches.append("asset identity")
    if mismatches:
        raise ValueError(
            "Forecast request does not match the supplied market evidence: "
            + ", ".join(mismatches)
            + "."
        )


def _build_feature_frame(candles: list[Candle], *, horizon: int) -> pd.DataFrame:
    close = pd.Series((candle.close for candle in candles), dtype=float)
    volume = pd.Series((candle.volume for candle in candles), dtype=float)
    result = pd.DataFrame(index=close.index)
    result["return_1"] = close.pct_change(1, fill_method=None)
    result["return_3"] = close.pct_change(3, fill_method=None)
    result["return_12"] = close.pct_change(12, fill_method=None)
    result["volatility_12"] = result["return_1"].rolling(12).std()
    result["sma_gap"] = close / close.rolling(20).mean() - 1
    result["volume_change"] = (
        volume.pct_change(1, fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    result["target_return"] = close.shift(-horizon) / close - 1
    return result.replace([np.inf, -np.inf], np.nan)


def _chronological_predictions(
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    folds: int,
    *,
    gap: int,
    minimum_training_samples: int,
    minimum_validation_samples: int,
    model_factory: Callable[[], Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    available_splits = (
        len(features) - gap - minimum_training_samples
    ) // minimum_validation_samples
    effective_splits = min(folds, available_splits)
    if effective_splits < 2:
        raise ValueError("Forecast has too few rows for purged time-series validation.")
    splitter = TimeSeriesSplit(
        n_splits=effective_splits,
        gap=gap,
        test_size=minimum_validation_samples,
    )
    predictions: list[float] = []
    actual: list[float] = []
    used_folds = 0
    for train_index, validation_index in splitter.split(features):
        if len(train_index) < minimum_training_samples:
            continue
        model = model_factory()
        model.fit(features[train_index], target[train_index])
        predictions.extend(float(value) for value in model.predict(features[validation_index]))
        actual.extend(float(value) for value in target[validation_index])
        used_folds += 1
    if len(actual) < minimum_validation_samples or used_folds < 2:
        raise ValueError("Forecast has too few eligible validation samples.")
    return np.array(predictions), np.array(actual), used_folds


def _metrics(
    actual: NDArray[np.float64],
    predictions: NDArray[np.float64],
    *,
    validation_folds: int,
) -> ForecastMetrics:
    baseline = np.zeros_like(actual)
    mae = float(mean_absolute_error(actual, predictions))
    baseline_mae = float(mean_absolute_error(actual, baseline))
    return ForecastMetrics(
        mae=mae,
        rmse=float(np.sqrt(mean_squared_error(actual, predictions))),
        directional_accuracy=float(np.mean(np.sign(actual) == np.sign(predictions))),
        baseline_mae=baseline_mae,
        mae_improvement=(baseline_mae - mae) / baseline_mae if baseline_mae else 0.0,
        validation_samples=len(actual),
        validation_folds=validation_folds,
    )


def _quality(
    metrics: ForecastMetrics,
    predicted_return: float,
    interval_width: float,
    policy: ForecastPolicy,
) -> ForecastQuality:
    checks = [
        (metrics.mae_improvement >= policy.minimum_mae_improvement, "MAE improvement threshold"),
        (
            metrics.directional_accuracy >= policy.minimum_directional_accuracy,
            "directional accuracy threshold",
        ),
        (
            abs(predicted_return) <= policy.maximum_absolute_forecast_return,
            "forecast return bound",
        ),
        (interval_width <= policy.maximum_interval_width, "interval width bound"),
    ]
    reasons = [f"{'passed' if ok else 'failed'}: {label}" for ok, label in checks]
    passed = all(ok for ok, _ in checks)
    return ForecastQuality(
        passed=passed,
        reasons=reasons,
        prediction_suppressed=not passed,
    )
