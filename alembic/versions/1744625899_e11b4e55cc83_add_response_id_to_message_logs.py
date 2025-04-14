"""add response_id to message logs

Revision ID: e11b4e55cc83
Revises: 1de669bdebb8
Create Date: 2025-04-14 10:18:19.075522+00:00

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "e11b4e55cc83"
down_revision = "1de669bdebb8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ai_message_logs", sa.Column("response_id", sa.BigInteger(), nullable=True))


def downgrade() -> None:
    op.drop_column("ai_message_logs", "response_id")
