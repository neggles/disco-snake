"""update user schema

Revision ID: c32963d028b7
Revises: 6d79d7e5a2c7
Create Date: 2023-06-07 02:48:42.139940+00:00

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c32963d028b7"
down_revision = "6d79d7e5a2c7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint("users_id_key", "users", ["id"])
    op.drop_constraint("users_global_name_key", "users", type_="unique")
    op.drop_column("users", "global_name")
    op.alter_column("users", column_name="tos_accepted_at", new_column_name="tos_timestamp")
    op.add_column(
        "users",
        sa.Column("tos_rejected", sa.Boolean(), server_default=sa.false(), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("users", "tos_rejected")
    op.alter_column("users", column_name="tos_timestamp", new_column_name="tos_accepted_at")
    op.add_column(
        "users",
        sa.Column("global_name", sa.VARCHAR(), nullable=True),
    )
    op.create_unique_constraint("users_global_name_key", "users", ["global_name"])
    op.drop_constraint("users_id_key", "users", type_="unique")
