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
    op.add_column(
        "users",
        sa.Column("log_disable", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.add_column(
        "users",
        sa.Column("log_anonymous", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.execute("UPDATE users SET log_disable = false, log_anonymous = false")


def downgrade() -> None:
    op.drop_column("users", "log_anonymous")
    op.drop_column("users", "log_disable")
    op.add_column(
        "users",
        sa.Column("global_name", sa.VARCHAR(), nullable=True),
    )
    op.create_unique_constraint("users_global_name_key", "users", ["global_name"])
    op.drop_constraint("users_id_key", "users", type_="unique")
