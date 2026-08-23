from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config

from crypto_research.storage import models


def test_lean_core_migration_prunes_retired_data_and_preserves_workspace(tmp_path: Path) -> None:
    database_path = tmp_path / "migration.db"
    config = _migration_config(database_path)
    command.upgrade(config, "0002_user_workspaces")
    _populate_revision_0002(database_path)

    command.upgrade(config, "head")

    with sqlite3.connect(database_path) as connection:
        preference_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(user_preferences)")
        }
        report_text, schema_version, owner_id, pinned = connection.execute(
            "SELECT report_payload, payload_schema_version, owner_id, pinned "
            "FROM research_runs WHERE id = 'run-1'"
        ).fetchone()
        report = json.loads(report_text)

        assert preference_columns == {"user_id", "default_exchange", "default_timeframe"}
        assert schema_version == 2
        assert owner_id == "user-1"
        assert pinned == 1
        assert report["market_result"]["derivatives"]["status"] == "complete"
        assert report["market_result"]["technical"]["maximum_drawdown"] == 0.12
        assert report["market_result"]["market"]["total_volume"] == 1234.0
        assert report["risk_result"]["assessment"]["score"] == 42.0
        assert report["agent_answers"][0]["answer"] == "Core analysis remains."
        serialized_report = json.dumps(report)
        assert "topic_research" not in serialized_report
        assert "explainer_research" not in serialized_report
        assert "chat_research" not in serialized_report
        assert "generated obsolete field" not in serialized_report

        assert connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
        assert connection.execute("SELECT asset FROM watchlist_assets").fetchone()[0] == "BTC"
        evidence_types = {
            row[0] for row in connection.execute("SELECT claim_type FROM evidence_snapshots")
        }
        assert evidence_types == {"market_snapshot"}
        cache_namespaces = {
            row[0] for row in connection.execute("SELECT namespace FROM cache_entries")
        }
        assert cache_namespaces == {"market"}

    command.downgrade(config, "0002_user_workspaces")
    with sqlite3.connect(database_path) as connection:
        restored = connection.execute(
            "SELECT default_research_goal FROM user_preferences WHERE user_id = 'user-1'"
        ).fetchone()
        assert restored == ("custom",)


def _migration_config(database_path: Path) -> Config:
    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(models.__file__).with_name("migrations")),
    )
    config.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{database_path.as_posix()}")
    return config


def _populate_revision_0002(database_path: Path) -> None:
    report = {
        "request": {"user_intent": "Research BTC", "topic_research": {"discard": True}},
        "market_result": {
            "market": {"symbol": "BTC/USD", "total_volume": 1234.0},
            "technical": {"maximum_drawdown": 0.12},
            "derivatives": {"status": "complete"},
        },
        "risk_result": {"assessment": {"score": 42.0}},
        "agent_answers": [
            {
                "agent": "market_agent",
                "answer": "Core analysis remains.",
                "development": "generated obsolete field",
            }
        ],
        "topic_research": {"discard": True},
        "explainer_research": {"discard": True},
        "chat_research": {"discard": True},
    }
    request = {"user_intent": "Research BTC", "symbol": "BTC/USD"}
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "user-1",
                "auth0",
                "https://issuer.example/",
                "subject-1",
                "user@example.com",
                1,
                "User",
                "Researcher",
                None,
                "2026-08-01T00:00:00+00:00",
                "2026-08-01T00:00:00+00:00",
                "2026-08-01T00:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO user_preferences VALUES (?, ?, ?, ?)",
            ("user-1", "kraken", "1h", "compare_assets"),
        )
        connection.execute(
            "INSERT INTO watchlist_assets VALUES (?, ?, ?, ?)",
            ("user-1", "BTC", 0, "2026-08-01T00:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO research_runs "
            "(id, owner_id, created_at, completed_at, state, question, mode, assets, "
            "capabilities, exchange, timeframe, scope_key, request_payload, report_payload, "
            "payload_schema_version, application_version, pinned, failure) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "run-1",
                "user-1",
                "2026-08-01T00:00:00+00:00",
                "2026-08-01T00:01:00+00:00",
                "complete",
                "Research BTC",
                "research",
                json.dumps(["BTC/USD"]),
                json.dumps(["market", "derivatives", "risk"]),
                "kraken",
                "1h",
                "scope-1",
                json.dumps(request),
                json.dumps(report),
                1,
                "0.1.0",
                1,
                None,
            ),
        )
        for evidence_id, claim_type in (
            ("market.btc", "market_snapshot"),
            ("topic.btc", "topic_research"),
        ):
            connection.execute(
                "INSERT INTO evidence_snapshots "
                "(run_id, evidence_id, claim_type, asset, source, collected_at, observed_at, "
                "payload_hash, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "run-1",
                    evidence_id,
                    claim_type,
                    "BTC/USD",
                    "provider",
                    "2026-08-01T00:00:00+00:00",
                    "2026-08-01T00:00:00+00:00",
                    "hash",
                    json.dumps({"kept": claim_type == "market_snapshot"}),
                ),
            )
        for namespace in ("market", "topic_research"):
            connection.execute(
                "INSERT INTO cache_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    namespace,
                    "key",
                    json.dumps({"namespace": namespace}),
                    "checksum",
                    1,
                    "2026-08-01T00:00:00+00:00",
                    "2026-08-01T01:00:00+00:00",
                    "2026-08-02T00:00:00+00:00",
                    0,
                ),
            )
