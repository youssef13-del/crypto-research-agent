"""Add authenticated user profiles, preferences, and watchlists."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_user_workspaces"
down_revision: str | None = "0001_research_memory"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=40), nullable=False),
        sa.Column("issuer", sa.String(length=500), nullable=False),
        sa.Column("subject", sa.String(length=500), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("email_verified", sa.Boolean(), nullable=False),
        sa.Column("provider_name", sa.String(length=120), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=True),
        sa.Column("avatar_url", sa.String(length=2048), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("issuer", "subject", name="ux_users_oidc_identity"),
    )
    op.create_index("ix_users_email", "users", ["email"])
    op.create_table(
        "user_preferences",
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("default_exchange", sa.String(length=40), nullable=False),
        sa.Column("default_timeframe", sa.String(length=20), nullable=False),
        sa.Column("default_research_goal", sa.String(length=40), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_table(
        "watchlist_assets",
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("asset", sa.String(length=16), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id", "asset"),
    )


def downgrade() -> None:
    op.drop_table("watchlist_assets")
    op.drop_table("user_preferences")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
