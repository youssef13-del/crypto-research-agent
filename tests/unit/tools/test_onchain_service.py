from datetime import UTC, datetime, timedelta

import httpx

from crypto_research.domain.research import AnalysisAsset
from crypto_research.tools.onchain import fetch_onchain_evidence


def _asset(coin_id: str = "bitcoin") -> AnalysisAsset:
    return AnalysisAsset(requested_name="Bitcoin", symbol="BTC/USD", coin_id=coin_id)


def _client(
    rows: list[dict[str, object]],
    *,
    status_code: int = 200,
    available_metrics: tuple[str, ...] = ("AdrActCnt", "TxCnt", "FeeTotUSD"),
    catalog_max_time: str | None = None,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/catalog-v2/asset-metrics"):
            catalog = [
                {
                    "metric": metric,
                    "frequencies": [
                        {
                            "frequency": "1d",
                            "community": True,
                            **({"max_time": catalog_max_time} if catalog_max_time else {}),
                        }
                    ],
                }
                for metric in available_metrics
            ]
            return httpx.Response(
                status_code,
                json={"data": [{"asset": "btc", "metrics": catalog}]},
                request=request,
            )
        return httpx.Response(status_code, json={"data": rows}, request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_onchain_evidence_builds_daily_trends() -> None:
    collected = datetime(2026, 8, 11, 12, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for offset in range(14):
        observed = collected.replace(hour=0) - timedelta(days=13 - offset)
        value = 100 if offset < 7 else 200
        rows.append(
            {
                "time": observed.isoformat().replace("+00:00", "Z"),
                "AdrActCnt": str(value),
                "TxCnt": str(value * 10),
                "FeeTotUSD": str(value * 100),
            }
        )

    with _client(rows) as client:
        evidence = fetch_onchain_evidence(
            asset=_asset(),
            base_url="https://coinmetrics.test/v4/trends",
            collected_at=collected,
            client=client,
        )

    assert evidence.status == "complete"
    assert [metric.metric for metric in evidence.metrics] == ["AdrActCnt", "TxCnt", "FeeTotUSD"]
    assert all(metric.seven_day_change_pct == 100 for metric in evidence.metrics)
    assert evidence.metrics[0].latest_value == 200


def test_fetch_onchain_evidence_marks_one_metric_partial() -> None:
    collected = datetime(2026, 8, 11, 12, tzinfo=UTC)
    rows: list[dict[str, object]] = [{"time": "2026-08-11T00:00:00Z", "TxCnt": "42"}]

    with _client(rows, available_metrics=("TxCnt",)) as client:
        evidence = fetch_onchain_evidence(
            asset=_asset(),
            base_url="https://coinmetrics.test/v4/partial",
            collected_at=collected,
            client=client,
        )

    assert evidence.status == "partial"
    assert [metric.metric for metric in evidence.metrics] == ["TxCnt"]
    assert "AdrActCnt" in evidence.warnings[0]


def test_fetch_onchain_evidence_rejects_future_and_stale_rows() -> None:
    collected = datetime(2026, 8, 11, 12, tzinfo=UTC)
    rows: list[dict[str, object]] = [
        {"time": "2026-08-12T00:00:00Z", "TxCnt": "42"},
        {"time": "2026-05-01T00:00:00Z", "TxCnt": "20"},
    ]

    with _client(rows, available_metrics=("TxCnt",)) as client:
        evidence = fetch_onchain_evidence(
            asset=_asset(),
            base_url="https://coinmetrics.test/v4/quarantine",
            collected_at=collected,
            client=client,
        )

    assert evidence.status == "unavailable"
    assert evidence.metrics == []


def test_fetch_onchain_evidence_does_not_guess_unmapped_assets() -> None:
    evidence = fetch_onchain_evidence(
        asset=_asset("unknown-coin"),
        collected_at=datetime(2026, 8, 11, 12, tzinfo=UTC),
    )

    assert evidence.status == "not_applicable"
    assert evidence.provider_asset is None


def test_fetch_onchain_evidence_degrades_http_failures() -> None:
    with _client([], status_code=429) as client:
        evidence = fetch_onchain_evidence(
            asset=_asset(),
            base_url="https://coinmetrics.test/v4/failure",
            collected_at=datetime(2026, 8, 11, 12, tzinfo=UTC),
            client=client,
        )

    assert evidence.status == "unavailable"
    assert evidence.warnings == [
        "Coin Metrics Community rate limit was reached; try again shortly."
    ]


def test_fetch_onchain_evidence_derives_usd_fees_from_public_metrics() -> None:
    collected = datetime(2026, 8, 11, 12, tzinfo=UTC)
    rows: list[dict[str, object]] = [
        {
            "time": "2026-08-11T00:00:00Z",
            "AdrActCnt": "100",
            "TxCnt": "200",
            "FeeTotNtv": "0.5",
            "PriceUSD": "1000",
        }
    ]

    with _client(
        rows,
        available_metrics=("AdrActCnt", "TxCnt", "FeeTotNtv", "PriceUSD"),
    ) as client:
        evidence = fetch_onchain_evidence(
            asset=_asset(),
            base_url="https://coinmetrics.test/v4/derived-fees",
            collected_at=collected,
            client=client,
        )

    assert evidence.status == "complete"
    assert evidence.metrics[-1].metric == "FeeTotUSD"
    assert evidence.metrics[-1].latest_value == 500


def test_fetch_onchain_evidence_skips_assets_without_public_network_metrics() -> None:
    with _client([], available_metrics=()) as client:
        evidence = fetch_onchain_evidence(
            asset=_asset(),
            base_url="https://coinmetrics.test/v4/no-coverage",
            collected_at=datetime(2026, 8, 11, 12, tzinfo=UTC),
            client=client,
        )

    assert evidence.status == "unavailable"
    assert "does not provide" in evidence.warnings[0]


def test_fetch_onchain_evidence_skips_stale_catalog_coverage() -> None:
    with _client([], catalog_max_time="2022-06-03T00:00:00Z") as client:
        evidence = fetch_onchain_evidence(
            asset=_asset(),
            base_url="https://coinmetrics.test/v4/stale-coverage",
            collected_at=datetime(2026, 8, 11, 12, tzinfo=UTC),
            client=client,
        )

    assert evidence.status == "unavailable"
    assert "does not provide" in evidence.warnings[0]
