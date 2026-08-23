"""Remove retired preferences, research payloads, and cache data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "0003_lean_core"
down_revision: str | None = "0002_user_workspaces"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RETIRED_REPORT_FIELDS = frozenset(
    {
        "ath",
        "ath_change_pct",
        "catalyst_score",
        "chain_concentration",
        "chat_research",
        "circulating_supply_ratio",
        "derivatives_score",
        "development",
        "dex_volume_24h",
        "downside_deviation",
        "drawdown_duration",
        "drift_mae",
        "derived_metrics",
        "events",
        "evidence_gaps",
        "explainer_research",
        "fees_24h",
        "fully_diluted_valuation",
        "interval_coverage",
        "liquidity",
        "liquidity_score",
        "market_cap_fdv_ratio",
        "market_cap_tvl_ratio",
        "model_agreement",
        "model_stability",
        "momentum_score",
        "naive_mae",
        "provider_health",
        "recovery_attempts",
        "regime",
        "research_depth",
        "revenue_24h",
        "risk_penalty",
        "source_diversity",
        "topic_research",
        "topic_research_by_capability",
        "tvl_history",
        "volume_market_cap_ratio",
        "web_research",
    }
)


def upgrade() -> None:
    bind = op.get_bind()
    runs = sa.table(
        "research_runs",
        sa.column("id", sa.String()),
        sa.column("report_payload", sa.JSON()),
        sa.column("payload_schema_version", sa.Integer()),
    )
    rows = bind.execute(
        sa.select(runs.c.id, runs.c.report_payload).where(runs.c.report_payload.is_not(None))
    )
    for run_id, payload in rows:
        if not isinstance(payload, Mapping):
            continue
        bind.execute(
            sa.update(runs)
            .where(runs.c.id == run_id)
            .values(report_payload=_prune_payload(payload), payload_schema_version=2)
        )
    bind.execute(sa.update(runs).values(payload_schema_version=2))
    bind.execute(sa.text("DELETE FROM cache_entries WHERE namespace = 'topic_research'"))
    bind.execute(sa.text("DELETE FROM evidence_snapshots WHERE claim_type = 'topic_research'"))
    with op.batch_alter_table("user_preferences") as batch_op:
        batch_op.drop_column("default_research_goal")


def downgrade() -> None:
    with op.batch_alter_table("user_preferences") as batch_op:
        batch_op.add_column(
            sa.Column(
                "default_research_goal",
                sa.String(length=40),
                nullable=False,
                server_default="custom",
            )
        )


def _prune_payload(value: object) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _prune_payload(
                item[-30:] if key == "observations" and isinstance(item, list) else item
            )
            for key, item in value.items()
            if key not in _RETIRED_REPORT_FIELDS
        }
    if isinstance(value, list):
        return [_prune_payload(item) for item in value]
    return value
