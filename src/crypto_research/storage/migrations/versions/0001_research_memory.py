"""Create durable research history and provider cache."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_research_memory"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("state", sa.String(length=20), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(length=24), nullable=False),
        sa.Column("assets", sa.JSON(), nullable=False),
        sa.Column("capabilities", sa.JSON(), nullable=False),
        sa.Column("exchange", sa.String(length=40), nullable=True),
        sa.Column("timeframe", sa.String(length=20), nullable=True),
        sa.Column("scope_key", sa.String(length=64), nullable=False),
        sa.Column("request_payload", sa.JSON(), nullable=False),
        sa.Column("report_payload", sa.JSON(), nullable=True),
        sa.Column("payload_schema_version", sa.Integer(), nullable=False),
        sa.Column("application_version", sa.String(length=40), nullable=False),
        sa.Column("pinned", sa.Boolean(), nullable=False),
        sa.Column("failure", sa.String(length=500), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_research_runs_owner_id", "research_runs", ["owner_id"])
    op.create_index("ix_research_runs_created_at", "research_runs", ["created_at"])
    op.create_index("ix_research_runs_state", "research_runs", ["state"])
    op.create_index("ix_research_runs_scope_key", "research_runs", ["scope_key"])
    op.create_index("ix_research_runs_pinned", "research_runs", ["pinned"])
    op.create_index(
        "ix_research_runs_scope_completed",
        "research_runs",
        ["scope_key", "completed_at"],
    )
    op.create_table(
        "evidence_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("evidence_id", sa.String(length=240), nullable=False),
        sa.Column("claim_type", sa.String(length=80), nullable=False),
        sa.Column("asset", sa.String(length=80), nullable=True),
        sa.Column("source", sa.String(length=240), nullable=False),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_evidence_snapshots_run_id", "evidence_snapshots", ["run_id"])
    op.create_index("ix_evidence_snapshots_claim_type", "evidence_snapshots", ["claim_type"])
    op.create_index("ix_evidence_snapshots_asset", "evidence_snapshots", ["asset"])
    op.create_index(
        "ux_evidence_run_id", "evidence_snapshots", ["run_id", "evidence_id"], unique=True
    )
    op.create_table(
        "cache_entries",
        sa.Column("namespace", sa.String(length=80), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("checksum", sa.String(length=64), nullable=False),
        sa.Column("payload_schema_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fresh_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stale_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("negative", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("namespace", "cache_key"),
    )
    op.create_index("ix_cache_entries_fresh_until", "cache_entries", ["fresh_until"])
    op.create_index("ix_cache_entries_stale_until", "cache_entries", ["stale_until"])


def downgrade() -> None:
    op.drop_table("cache_entries")
    op.drop_table("evidence_snapshots")
    op.drop_table("research_runs")
