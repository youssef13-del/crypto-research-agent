from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from crypto_research.domain.account import UserIdentity
from crypto_research.domain.evidence import EvidenceRecord
from crypto_research.domain.research import (
    AnalysisAsset,
    AnalysisRequest,
    FundamentalEvidence,
    FundamentalsAgentResult,
    ResearchCapability,
    ResearchReport,
)
from crypto_research.interfaces.web.pages.library import compare_reports
from crypto_research.storage.repository import ResearchRepository, create_repository


def _repository(tmp_path: Path) -> ResearchRepository:
    database = (tmp_path / "history.db").as_posix()
    return create_repository(f"sqlite+pysqlite:///{database}")


def _request() -> AnalysisRequest:
    return AnalysisRequest(user_intent="Research Bitcoin", asset_query="Bitcoin")


def _report(*, market_cap: float = 1_000_000) -> ResearchReport:
    return ResearchReport(
        request=_request(),
        fundamentals_result=FundamentalsAgentResult(
            fundamentals=FundamentalEvidence(
                name="Bitcoin",
                symbol="BTC",
                market_cap=market_cap,
            )
        ),
        status="complete",
    )


def _identity(subject: str, *, email: str = "researcher@example.com") -> UserIdentity:
    return UserIdentity(
        provider="auth0",
        issuer="https://chainscope.example.auth0.com/",
        subject=subject,
        email=email,
        email_verified=True,
        provider_name="Researcher",
    )


def test_repository_round_trips_report_and_normalized_evidence(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    request = _request()
    run_id = repository.create_run(
        request=request,
        capabilities=[ResearchCapability.FUNDAMENTALS],
        question=request.user_intent,
    )
    evidence = EvidenceRecord(
        evidence_id="fundamentals.btc-usd",
        claim_type="fundamental_snapshot",
        source="CoinGecko",
        source_tier="primary",
        collected_at=datetime.now(UTC),
        asset="BTC/USD",
        payload={"market_cap": 1_000_000},
    )

    report = _report()
    repository.complete_run(run_id, report, [evidence])

    stored = repository.get_run(run_id)
    assert stored is not None
    assert stored.report == report
    assert stored.summary.evidence_count == 1
    assert stored.summary.state == "complete"
    assert stored.summary.assets == ("BTC/USD",)


def test_repository_pins_and_prunes_only_expired_unpinned_runs(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    run_id = repository.create_run(
        request=_request(),
        capabilities=[ResearchCapability.FUNDAMENTALS],
        question="Research Bitcoin",
    )
    repository.complete_run(run_id, _report())
    assert repository.pin(run_id, pinned=True)

    future = datetime.now(UTC) + timedelta(days=366)
    assert repository.prune(retention_days=365, now=future) == 0
    assert repository.get_run(run_id) is not None

    assert repository.pin(run_id, pinned=False)
    assert repository.prune(retention_days=365, now=future) == 1
    assert repository.get_run(run_id) is None


def test_repository_finds_previous_matching_run_and_computes_deterministic_diff(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    first_id = repository.create_run(
        request=_request(),
        capabilities=[ResearchCapability.FUNDAMENTALS],
        question="Research Bitcoin",
    )
    repository.complete_run(first_id, _report(market_cap=1_000_000))
    second_id = repository.create_run(
        request=_request(),
        capabilities=[ResearchCapability.FUNDAMENTALS],
        question="Research Bitcoin again",
    )
    repository.complete_run(second_id, _report(market_cap=1_200_000))

    comparison = repository.compare(second_id)

    assert comparison is not None
    assert comparison.previous is not None
    changes = compare_reports(comparison.current.report, comparison.previous.report)
    assert any(
        "Market Cap" in label and previous == "1000000.0" and current == "1200000.0"
        for label, previous, current in changes
    )


def test_repository_persistent_cache_rejects_expired_entries(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.set_cache_entry(
        "news",
        "key",
        {"query": "btc"},
        fresh_seconds=1,
        stale_seconds=2,
    )

    assert repository.get_cache_entry("news", "key") is not None
    assert repository.prune_cache(now=datetime.now(UTC) + timedelta(seconds=3)) == 1
    assert repository.get_cache_entry("news", "key", allow_stale=True) is None


def test_repository_recovers_interrupted_runs(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    run_id = repository.create_run(
        request=_request(),
        capabilities=[ResearchCapability.MARKET],
        question="Research Bitcoin",
    )

    assert repository.recover_interrupted_runs() == 1

    summary = next(item for item in repository.list_runs() if item.id == run_id)
    assert summary.state == "failed"


def test_repository_filters_runs_before_applying_limit(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    ethereum_request = AnalysisRequest(
        user_intent="Research Ethereum",
        assets=[
            AnalysisAsset(
                requested_name="Ethereum",
                symbol="ETH/USD",
                coin_id="ethereum",
            )
        ],
    )
    ethereum_id = repository.create_run(
        request=ethereum_request,
        capabilities=[ResearchCapability.FUNDAMENTALS],
        question=ethereum_request.user_intent,
    )
    bitcoin_id = repository.create_run(
        request=_request(),
        capabilities=[ResearchCapability.MARKET],
        question="Research Bitcoin",
    )

    by_asset = repository.list_runs(asset="ETH/USD", limit=1)
    by_capability = repository.list_runs(capability="fundamentals", limit=1)

    assert [item.id for item in by_asset] == [ethereum_id]
    assert [item.id for item in by_capability] == [ethereum_id]
    assert bitcoin_id not in {item.id for item in (*by_asset, *by_capability)}


def test_user_workspace_upsert_preserves_default_preferences_and_watchlist(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    created = repository.upsert_user(_identity("auth0|first"))
    refreshed = repository.upsert_user(_identity("auth0|first"))

    assert created.watchlist == ("BTC", "ETH", "SOL", "XRP")
    assert created.preferences.default_exchange == "kraken"
    assert created.preferences.default_timeframe == "1h"
    assert refreshed.profile.id == created.profile.id
    assert refreshed.preferences == created.preferences
    assert refreshed.watchlist == created.watchlist


def test_same_email_with_distinct_oidc_subjects_creates_separate_workspaces(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)

    google = repository.upsert_user(_identity("google-oauth2|one"))
    password = repository.upsert_user(_identity("auth0|two"))

    assert google.profile.id != password.profile.id


def test_scoped_repository_prevents_cross_workspace_history_access(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    first = repository.upsert_user(_identity("auth0|first"))
    second = repository.upsert_user(_identity("auth0|second", email="second@example.com"))
    first_history = repository.for_owner(first.profile.id)
    second_history = repository.for_owner(second.profile.id)
    run_id = first_history.create_run(
        request=_request(),
        capabilities=[ResearchCapability.FUNDAMENTALS],
        question="Private Bitcoin research",
    )

    assert second_history.list_runs() == []
    assert second_history.get_run(run_id) is None
    assert second_history.compare(run_id) is None
    assert not second_history.pin(run_id, pinned=True)
    assert not second_history.delete(run_id)
    with pytest.raises(KeyError, match="Unknown research run"):
        second_history.complete_run(run_id, _report())

    first_history.complete_run(run_id, _report())
    assert first_history.get_run(run_id) is not None
    assert second_history.get_run(run_id) is None


def test_latest_run_uses_completion_time_not_library_pin_order(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    owner = repository.upsert_user(_identity("auth0|latest"))
    other = repository.upsert_user(_identity("auth0|other", email="other@example.com"))
    history = repository.for_owner(owner.profile.id)
    older = history.create_run(
        request=_request(),
        capabilities=[ResearchCapability.FUNDAMENTALS],
        question="Older report",
    )
    history.complete_run(older, _report(market_cap=1_000_000))
    assert history.pin(older, pinned=True)
    newest = history.create_run(
        request=_request(),
        capabilities=[ResearchCapability.FUNDAMENTALS],
        question="Newest report",
    )
    history.complete_run(newest, _report(market_cap=2_000_000))

    latest = history.latest_run()

    assert latest is not None
    assert latest.summary.id == newest
    assert repository.for_owner(other.profile.id).latest_run() is None


def test_bulk_delete_is_atomic_tenant_scoped_deduplicated_and_protects_pins(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    first = repository.upsert_user(_identity("auth0|first"))
    second = repository.upsert_user(_identity("auth0|second", email="second@example.com"))
    first_history = repository.for_owner(first.profile.id)
    second_history = repository.for_owner(second.profile.id)

    deletable = first_history.create_run(
        request=_request(),
        capabilities=[ResearchCapability.MARKET],
        question="Delete this report",
    )
    protected = first_history.create_run(
        request=_request(),
        capabilities=[ResearchCapability.FUNDAMENTALS],
        question="Keep this pinned report",
    )
    other_owner = second_history.create_run(
        request=_request(),
        capabilities=[ResearchCapability.MARKET],
        question="Private report in another workspace",
    )
    for scoped, run_id in (
        (first_history, deletable),
        (first_history, protected),
        (second_history, other_owner),
    ):
        scoped.complete_run(run_id, _report())
    assert first_history.pin(protected, pinned=True)

    result = first_history.delete_many((deletable, deletable, protected, other_owner))

    assert result.requested_count == 3
    assert result.deleted_count == 1
    assert result.protected_count == 1
    assert first_history.get_run(deletable) is None
    assert first_history.get_run(protected) is not None
    assert second_history.get_run(other_owner) is not None


def test_bulk_delete_limits_input_and_scoped_pruning_preserves_other_workspaces(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    first = repository.upsert_user(_identity("auth0|first"))
    second = repository.upsert_user(_identity("auth0|second", email="second@example.com"))
    first_history = repository.for_owner(first.profile.id)
    second_history = repository.for_owner(second.profile.id)
    first_run = first_history.create_run(
        request=_request(),
        capabilities=[ResearchCapability.MARKET],
        question="First workspace report",
    )
    second_run = second_history.create_run(
        request=_request(),
        capabilities=[ResearchCapability.MARKET],
        question="Second workspace report",
    )
    first_history.complete_run(first_run, _report())
    second_history.complete_run(second_run, _report())

    with pytest.raises(ValueError, match="limited to 100"):
        first_history.delete_many(tuple(f"run-{index}" for index in range(101)))

    future = datetime.now(UTC) + timedelta(days=366)
    assert first_history.prune(retention_days=365, now=future) == 1
    assert first_history.get_run(first_run) is None
    assert second_history.get_run(second_run) is not None


def test_workspace_deletion_cascades_owned_research_but_preserves_local_history(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    local_run = repository.create_run(
        request=_request(),
        capabilities=[ResearchCapability.MARKET],
        question="Legacy local research",
    )
    repository.complete_run(local_run, _report())
    workspace = repository.upsert_user(_identity("auth0|delete"))
    scoped = repository.for_owner(workspace.profile.id)
    owned_run = scoped.create_run(
        request=_request(),
        capabilities=[ResearchCapability.FUNDAMENTALS],
        question="Delete this workspace",
    )
    scoped.complete_run(owned_run, _report())

    assert repository.delete_workspace(workspace.profile.id)

    assert repository.get_workspace(workspace.profile.id) is None
    assert scoped.get_run(owned_run) is None
    assert repository.get_run(local_run) is not None
