"""add new log keys

Revision ID: be17fbcd2057
Revises: 117e78c95edc
Create Date: 2024-02-26 23:28:24.793353+00:00

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "be17fbcd2057"
down_revision = "117e78c95edc"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ai_message_logs", sa.Column("n_prompt_tokens", sa.Integer(), nullable=True))
    op.add_column("ai_message_logs", sa.Column("n_context_tokens", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("ai_message_logs", "n_context_tokens")
    op.drop_column("ai_message_logs", "n_prompt_tokens")
