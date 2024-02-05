"""add blacklist

Revision ID: d42f44c7e7c7
Revises: fd4919fc1b2a
Create Date: 2023-07-01 11:59:36.842199+00:00

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "d42f44c7e7c7"
down_revision = "fd4919fc1b2a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_blacklist",
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("reason", sa.String(length=256), nullable=False),
        sa.Column(
            "timestamp",
            postgresql.TIMESTAMP(timezone=True, precision=2),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_index(op.f("ix_user_blacklist_timestamp"), "user_blacklist", ["timestamp"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_user_blacklist_timestamp"), table_name="user_blacklist")
    op.drop_table("user_blacklist")
