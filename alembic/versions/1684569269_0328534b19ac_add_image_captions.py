"""add image captions

Revision ID: 0328534b19ac
Revises: ad663d9bb5dc
Create Date: 2023-05-20 17:54:29.523081

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0328534b19ac"
down_revision = "ad663d9bb5dc"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "image_captions",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("description", sa.String(length=1024), nullable=True),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("url", sa.String(), nullable=False),
        sa.Column("proxy_url", sa.String(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("caption", sa.String(length=512), nullable=False),
        sa.Column(
            "captioned_at",
            postgresql.TIMESTAMP(timezone=True, precision=2),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.Column("captioned_with", sa.String(length=256), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("image_captions")
