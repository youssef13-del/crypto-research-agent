"""Deterministic forecast request construction."""

from crypto_research.domain.forecast import ForecastRequest, ForecastSettings
from crypto_research.domain.research import AnalysisRequest


def build_requests(request: AnalysisRequest) -> tuple[ForecastSettings, list[ForecastRequest]]:
    settings = request.forecast_settings or ForecastSettings(timeframe="1h")
    return settings, [
        ForecastRequest(
            asset=asset.requested_name,
            coin_id=asset.coin_id,
            exchange=request.exchange,
            symbol=asset.symbol,
            timeframe=settings.timeframe,
            horizon_hours=settings.horizon_hours,
            model_id=settings.model_id,
            confidence_level=settings.confidence_level,
            lookback_candles=settings.lookback_candles,
        )
        for asset in request.ordered_assets()
    ]
