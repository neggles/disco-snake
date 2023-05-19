"""add discord users table

Revision ID: ad663d9bb5dc
Revises:
Create Date: 2023-05-19 15:22:16.016962

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

# revision identifiers, used by Alembic.
revision = "ad663d9bb5dc"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "discord_users",
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=True, start=42, cycle=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "discord_id",
            sa.BigInteger(),
            nullable=False,
            index=True,
            unique=True,
        ),
        sa.Column(
            "data",
            pg.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "first_seen",
            pg.TIMESTAMP(timezone=True, precision=2),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "last_seen",
            pg.TIMESTAMP(timezone=True, precision=2),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "username",
            sa.String(),
            sa.Computed("data->>'username'"),
            nullable=False,
        ),
        sa.Column(
            "avatar",
            sa.String(),
            sa.Computed("data->>'avatar'"),
            nullable=True,
        ),
        sa.Column(
            "banner",
            sa.String(),
            sa.Computed("data->>'banner'"),
            nullable=True,
        ),
        sa.Column(
            "discriminator",
            sa.Integer(),
            sa.Computed("COALESCE((data->>'discriminator')::integer, 0)"),
            nullable=False,
        ),
        sa.Column(
            "bot",
            sa.Boolean(),
            sa.Computed("(data->>'bot')::boolean"),
            nullable=True,
        ),
        sa.Column(
            "system",
            sa.Boolean(),
            sa.Computed("(data->>'system')::boolean"),
            nullable=True,
        ),
        sa.Column(
            "mfa_enabled",
            sa.Boolean(),
            sa.Computed("(data->>'mfa_enabled')::boolean"),
            nullable=True,
        ),
        sa.Column(
            "email",
            sa.String(),
            sa.Computed("data->>'email'"),
            nullable=True,
        ),
        sa.Column(
            "verified",
            sa.Boolean(),
            sa.Computed("(data->>'verified')::boolean"),
            nullable=True,
        ),
        sa.Column(
            "flags",
            sa.Integer(),
            sa.Computed("(data->>'flags')::integer"),
            nullable=True,
        ),
        sa.Column(
            "premium_type",
            sa.Integer(),
            sa.Computed("(data->>'premium_type')::integer"),
            nullable=True,
        ),
        sa.Column(
            "public_flags",
            sa.Integer(),
            sa.Computed("(data->>'public_flags')::integer"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_discord_user_discord_id"), table_name="discord_user")
    op.drop_table("discord_user")
