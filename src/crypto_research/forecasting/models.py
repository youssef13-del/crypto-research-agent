"""Allow-listed forecasting models and their user-facing metadata."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import GradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from crypto_research.domain.forecast import ForecastModelId

FEATURE_COLUMNS = (
    "return_1",
    "return_3",
    "return_12",
    "volatility_12",
    "sma_gap",
    "volume_change",
)


@dataclass(frozen=True, slots=True)
class ForecastModelSpec:
    model_id: ForecastModelId
    display_name: str
    description: str
    hyperparameters: dict[str, str | int | float | bool]
    factory: Callable[[], Any]

    @property
    def feature_columns(self) -> list[str]:
        return list(FEATURE_COLUMNS)


def _gradient_boosting() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="huber",
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=12,
        subsample=0.9,
        random_state=42,
    )


def _ridge() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


MODEL_REGISTRY: dict[ForecastModelId, ForecastModelSpec] = {
    "gradient_boosting_huber": ForecastModelSpec(
        model_id="gradient_boosting_huber",
        display_name="Gradient Boosting (Huber)",
        description="Robust nonlinear regressor that reduces the influence of large residuals.",
        hyperparameters={
            "loss": "huber",
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 2,
            "min_samples_leaf": 12,
            "subsample": 0.9,
            "random_state": 42,
        },
        factory=_gradient_boosting,
    ),
    "ridge": ForecastModelSpec(
        model_id="ridge",
        display_name="Ridge Regression",
        description="Standardized linear model providing a transparent low-complexity comparison.",
        hyperparameters={"scaler": "standard", "alpha": 1.0},
        factory=_ridge,
    ),
}


def get_model_spec(model_id: ForecastModelId) -> ForecastModelSpec:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported forecast model: {model_id}") from exc
