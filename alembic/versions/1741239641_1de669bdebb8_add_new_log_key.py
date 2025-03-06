"""add new log key

Revision ID: 1de669bdebb8
Revises: be17fbcd2057
Create Date: 2025-03-06 05:40:41.774013+00:00

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "1de669bdebb8"
down_revision = "be17fbcd2057"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ai_message_logs", sa.Column("thoughts", postgresql.ARRAY(sa.String()), nullable=True))


def downgrade() -> None:
    op.drop_column("ai_message_logs", "thoughts")
