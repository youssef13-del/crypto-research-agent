from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from crypto_research.agents.forecast.forecast_agent import ForecastAgent
from crypto_research.agents.fundamentals.fundamentals_agent import FundamentalsAgent
from crypto_research.agents.market.market_agent import MarketAgent
from crypto_research.agents.news.news_agent import NewsAgent
from crypto_research.agents.onchain.onchain_agent import OnChainAgent
from crypto_research.agents.shared_analysis import SpecialistAnalysisRunner
from crypto_research.config import LLMProvider, Settings
from crypto_research.llm.client import (
    DisabledLLMAdapter,
    LLMAdapter,
    ResilientLLMAdapter,
)
from crypto_research.orchestration.runtime import ResearchRuntime
from crypto_research.tools.cache import configure_cache_backend
from crypto_research.tools.types import (
    FundamentalsServices,
    MarketServices,
    NewsServices,
    OnChainServices,
)

if TYPE_CHECKING:
    from crypto_research.domain.market import ComparisonMetrics, MarketEvidence
    from crypto_research.domain.research import (
        AnalysisRequest,
        OpportunityScanResult,
        TechnicalSnapshot,
    )
    from crypto_research.forecasting.service import ForecastService
    from crypto_research.storage.repository import ResearchRepository

logger = logging.getLogger(__name__)


def create_research_runtime(
    settings: Settings,
    *,
    owner_id: str,
    llm: LLMAdapter | None = None,
) -> ResearchRuntime:
    """Compose the concrete application dependencies for one runtime."""

    adapter = llm if llm is not None else create_llm_adapter(settings)
    history = load_research_repository(settings.database_url, settings.research_retention_days)
    configure_cache_backend(history)
    scoped_history = history.for_owner(owner_id) if history is not None else None
    from crypto_research.tools.derivatives import fetch_derivatives_evidence
    from crypto_research.tools.fundamentals import fetch_defi_evidence, fetch_fundamental_evidence
    from crypto_research.tools.market import (
        calculate_comparison_metrics,
        calculate_indicators,
        fetch_market_comparison,
        fetch_market_evidence,
        fetch_market_snapshots,
        scan_crypto_opportunities,
    )
    from crypto_research.tools.news import fetch_news_evidence, resolve_news_query
    from crypto_research.tools.onchain import fetch_onchain_evidence

    market_agent = MarketAgent(
        services=MarketServices(
            market_evidence=fetch_market_evidence,
            indicators=calculate_indicators,
            snapshots=fetch_market_snapshots,
            comparison=fetch_market_comparison,
            comparison_metrics=calculate_comparison_metrics,
            discovery=scan_crypto_opportunities,
            derivatives=fetch_derivatives_evidence,
        ),
        derivatives_base_url=settings.binance_futures_base_url,
    )
    news_agent = NewsAgent(
        services=NewsServices(collect=fetch_news_evidence, resolve_query=resolve_news_query),
    )
    fundamentals_agent = FundamentalsAgent(
        services=FundamentalsServices(
            fundamentals=fetch_fundamental_evidence,
            defi=fetch_defi_evidence,
        ),
        coingecko_api_key=settings.coingecko_api_key,
        defillama_base_url=settings.defillama_base_url,
    )
    return ResearchRuntime(
        market_agent=market_agent,
        news_agent=news_agent,
        fundamentals_agent=fundamentals_agent,
        onchain_agent=OnChainAgent(
            services=OnChainServices(collect=fetch_onchain_evidence),
            base_url=settings.coinmetrics_base_url,
        ),
        forecast_agent=ForecastAgent(
            service=create_forecast_service(settings),
            llm=adapter,
        ),
        specialist_analysis=SpecialistAnalysisRunner(
            llm=adapter,
            live_mode=settings.llm_provider is LLMProvider.GROQ,
        ),
        history=scoped_history,
    )


@lru_cache(maxsize=4)
def load_research_repository(
    database_url: str,
    retention_days: int = 365,
) -> ResearchRepository | None:
    """Create and migrate durable storage without making it runtime-critical."""

    try:
        from crypto_research.storage.repository import create_repository

        repository = create_repository(database_url)
        repository.recover_interrupted_runs()
        repository.prune(retention_days=retention_days)
        repository.prune_cache()
        return repository
    except Exception:
        logger.warning("Durable research storage is unavailable", exc_info=True)
        return None


def create_llm_adapter(settings: Settings) -> LLMAdapter:
    if settings.llm_provider is LLMProvider.DISABLED:
        return DisabledLLMAdapter()

    return ResilientLLMAdapter(
        _create_groq_adapter(settings, model=settings.groq_model),
        deterministic_fallback=False,
    )


def _create_groq_adapter(
    settings: Settings,
    *,
    model: str | None = None,
) -> LLMAdapter:
    from crypto_research.llm.groq import GroqAdapter

    return GroqAdapter(settings, model=model)


def create_forecast_service(settings: Settings) -> ForecastService:
    """Compose the dedicated forecasting service from validated application settings."""

    from crypto_research.forecasting.service import ForecastPolicy, ForecastService

    return ForecastService(
        policy=ForecastPolicy(
            minimum_training_samples=settings.minimum_training_samples,
            minimum_validation_samples=settings.minimum_validation_samples,
            time_series_folds=settings.time_series_folds,
            minimum_mae_improvement=settings.minimum_mae_improvement,
            minimum_directional_accuracy=settings.minimum_directional_accuracy,
            maximum_absolute_forecast_return=settings.maximum_absolute_forecast_return,
            maximum_interval_width=settings.maximum_interval_width,
        )
    )


def load_market_dashboard(
    request: AnalysisRequest,
) -> tuple[list[tuple[MarketEvidence, TechnicalSnapshot]], list[str]]:
    """Load the dashboard's shared market snapshots through the composition root."""

    from crypto_research.tools.market import fetch_market_comparison

    return fetch_market_comparison(request=request)


def load_home_market_overview() -> OpportunityScanResult:
    """Load the supported Kraken watchlist used by the Home market pulse."""

    from crypto_research.tools.market import scan_crypto_opportunities

    return scan_crypto_opportunities(
        exchange_name="kraken",
        quote="USD",
        timeframe="1h",
        limit=72,
    )


def calculate_dashboard_metrics(market: MarketEvidence) -> ComparisonMetrics:
    """Calculate the dashboard's standardized comparison metrics."""

    from crypto_research.tools.market import calculate_comparison_metrics

    return calculate_comparison_metrics(market)


def resolve_asset_request(request: AnalysisRequest, settings: Settings) -> AnalysisRequest:
    """Resolve a user-entered asset through the application composition root."""

    from crypto_research.tools.assets import resolve_analysis_request

    coingecko_key = (
        settings.coingecko_api_key.get_secret_value()
        if settings.coingecko_api_key is not None
        else None
    )
    return resolve_analysis_request(request, api_key=coingecko_key)
