"""add app log table

Revision ID: 62efb1742d19
Revises: c32963d028b7
Create Date: 2023-06-18 01:51:49.860494+00:00

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "62efb1742d19"
down_revision = "c32963d028b7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "logs",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False, start=1, cycle=True), nullable=False),
        sa.Column("app_id", sa.BigInteger(), nullable=False),
        sa.Column("instance", sa.String(), nullable=False),
        sa.Column(
            "timestamp",
            postgresql.TIMESTAMP(timezone=True, precision=2),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.Column("logger", sa.String(), nullable=False),
        sa.Column(
            "level",
            sa.Enum(
                "NOTSET", "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL", name="loglevel"
            ),
            nullable=False,
        ),
        sa.Column("message", sa.String(), nullable=False),
        sa.Column("trace", sa.String(), nullable=True),
        sa.Column("record", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_logs_app_id"), "logs", ["app_id"], unique=False)
    op.create_index(op.f("ix_logs_timestamp"), "logs", ["timestamp"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_logs_timestamp"), table_name="logs")
    op.drop_index(op.f("ix_logs_app_id"), table_name="logs")
    op.drop_table("logs")
    op.execute("DROP TYPE loglevel")
