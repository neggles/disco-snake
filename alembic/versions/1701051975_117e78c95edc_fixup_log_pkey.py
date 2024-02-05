"""fixup log pkey

Revision ID: 117e78c95edc
Revises: d42f44c7e7c7
Create Date: 2023-11-27 02:26:15.742620+00:00

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "117e78c95edc"
down_revision = "d42f44c7e7c7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_ai_message_logs_app_id", table_name="ai_message_logs")
    op.drop_constraint("ai_message_logs_pkey", "ai_message_logs", type_="primary")
    op.create_primary_key("ai_message_logs_pkey", "ai_message_logs", ["id", "app_id"])


def downgrade() -> None:
    op.drop_constraint("ai_message_logs_pkey", "ai_message_logs", type_="primary")
    op.create_primary_key("ai_message_logs_pkey", "ai_message_logs", ["id"])
    op.create_index("ix_ai_message_logs_app_id", "ai_message_logs", ["app_id"], unique=False)
