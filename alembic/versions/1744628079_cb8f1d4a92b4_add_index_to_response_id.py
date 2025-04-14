"""add index to response_id

Revision ID: cb8f1d4a92b4
Revises: e11b4e55cc83
Create Date: 2025-04-14 10:54:39.335450+00:00

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "cb8f1d4a92b4"
down_revision = "e11b4e55cc83"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(op.f("ix_ai_message_logs_response_id"), "ai_message_logs", ["response_id"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_ai_message_logs_response_id"), table_name="ai_message_logs")
