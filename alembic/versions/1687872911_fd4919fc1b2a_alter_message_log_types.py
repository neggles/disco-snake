"""alter message log types

Revision ID: fd4919fc1b2a
Revises: f95735356898
Create Date: 2023-06-27 13:35:11.429397+00:00

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "fd4919fc1b2a"
down_revision = "f95735356898"
branch_labels = None
depends_on = None

# statement to create a function for casting a JSONB array of strings to a text array
create_jsonb_to_array_func = sa.text(
    "CREATE FUNCTION jsonb_to_text_array(_js jsonb) RETURNS text[]\n"
    + "LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT\n"
    + "AS 'SELECT ARRAY(SELECT jsonb_array_elements_text(_js))';\n"
)
del_jsonb_to_array_func = sa.text(
    "DROP FUNCTION IF EXISTS jsonb_to_text_array(_js jsonb);",
)

# create a cast to text array using the above function
create_jsonb_to_array_cast = sa.text(
    "CREATE CAST (jsonb AS text[]) WITH FUNCTION jsonb_to_text_array(_js jsonb) AS IMPLICIT;"
)
del_jsonb_to_array_cast = sa.text(
    "DROP CAST IF EXISTS (jsonb AS text[]);",
)


def upgrade() -> None:
    # create the conversion function and cast
    op.execute(create_jsonb_to_array_func)
    op.execute(create_jsonb_to_array_cast)
    # cast the columns to text arrays
    op.alter_column(
        "ai_message_logs",
        "conversation",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=postgresql.ARRAY(sa.String()),
        existing_nullable=True,
        postgresql_using="conversation::text[]",
    )
    op.alter_column(
        "ai_message_logs",
        "context",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=postgresql.ARRAY(sa.String()),
        existing_nullable=True,
        postgresql_using="context::text[]",
    )


def downgrade() -> None:
    op.alter_column(
        "ai_message_logs",
        "context",
        existing_type=postgresql.ARRAY(sa.String()),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
        postgresql_using="to_jsonb(context)",
    )
    op.alter_column(
        "ai_message_logs",
        "conversation",
        existing_type=postgresql.ARRAY(sa.String()),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
        postgresql_using="to_jsonb(conversation)",
    )
    # delete the conversion function and cast
    op.execute(del_jsonb_to_array_cast)
    op.execute(del_jsonb_to_array_func)
