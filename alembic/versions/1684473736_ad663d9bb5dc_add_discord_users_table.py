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
        "users",
        sa.Column("id", sa.BigInteger(), nullable=False, primary_key=True),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("discriminator", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("global_name", sa.String(), nullable=True),
        sa.Column("avatar", sa.String(), nullable=True),
        sa.Column("bot", sa.Boolean(), nullable=False),
        sa.Column("system", sa.Boolean(), nullable=False),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("verified", sa.Boolean(), nullable=True),
        sa.Column("flags", sa.Integer(), nullable=True),
        sa.Column("premium_type", sa.Integer(), nullable=True),
        sa.Column("public_flags", sa.Integer(), nullable=True),
        sa.Column(
            "first_seen",
            pg.TIMESTAMP(timezone=True, precision=2),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.Column(
            "last_updated",
            pg.TIMESTAMP(timezone=True, precision=2),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
    )
    op.create_table(
        "username_history",
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "timestamp",
            pg.TIMESTAMP(timezone=True, precision=2),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.Column("username", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id", "timestamp"),
    )


def downgrade() -> None:
    op.drop_table("username_history")
    op.drop_table("users")
