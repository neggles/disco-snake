"""add TOS status indexes to users table

Revision ID: f95735356898
Revises: afef72766d6f
Create Date: 2023-06-27 12:19:45.698355+00:00

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "f95735356898"
down_revision = "afef72766d6f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(op.f("ix_users_tos_accepted"), "users", ["tos_accepted"])
    op.create_index(op.f("ix_users_tos_rejected"), "users", ["tos_rejected"])


def downgrade() -> None:
    op.drop_index(op.f("ix_users_tos_rejected"), table_name="users")
    op.drop_index(op.f("ix_users_tos_accepted"), table_name="users")
