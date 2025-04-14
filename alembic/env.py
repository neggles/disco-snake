import asyncio
from logging.config import fileConfig

from alembic import context
from alembic.environment import EnvironmentContext
from alembic.script import ScriptDirectory
from pydantic import PostgresDsn
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from db.base import Base
from disco_snake.settings import BotSettings, get_settings

# acquire alembic.ini value dict
config = context.config

# Set up `logging` module with settings from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Retrieve the SQLAlchemy ORM declarative base object
target_metadata = Base.metadata

# Try to load the bot config object from the application code
try:
    bot_settings: BotSettings | None = get_settings()
except Exception as e:
    bot_settings = None

# retrieve -x arguments from the command line
x_args = context.get_x_argument(as_dictionary=True)

# pull db_uri from bot config file, if available
if bot_settings is not None:
    pg_dsn: PostgresDsn = bot_settings.db_uri
    pg_uri: str = pg_dsn.unicode_string()

    # optional port override from command line
    x_port = x_args.get("port", None)
    if x_port is not None:
        print(f"DB port overriden: -x port={x_port}")
        pg_uri = pg_uri.replace(pg_dsn.hosts()[0].port, x_port)

    config.set_main_option("sqlalchemy.url", pg_uri)
    print(f"Using db_uri from bot config: {pg_uri}")
else:
    print(f"Using db_uri from alembic.ini: {config.get_main_option('sqlalchemy.url')}")


def run_migrations_offline() -> None:
    """Run migrations in offline mode (no database connection).

    This assumes an empty database and generates SQL statements for the entire migration chain.
    """

    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        process_revision_directives=process_revision_directives,
        compare_server_default=True,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Actually do the migrations, finally."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_server_default=True,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations asynchronously in online mode."""

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations synchronously in online mode (used by the alembic CLI)"""
    asyncio.run(run_async_migrations())


def process_revision_directives(context: EnvironmentContext, revision, directives):
    # extract Migration
    migration_script = directives[0]
    # extract current head revision
    head_revision = ScriptDirectory.from_config(context.config).get_current_head()

    if head_revision is None:
        # edge case with first migration
        new_rev_id = 1
    else:
        # default branch with incrementation
        last_rev_id = int(head_revision)
        new_rev_id = last_rev_id + 1
    # fill zeros up to 3 digits: 1 -> 001
    migration_script.rev_id = "{0:03}".format(new_rev_id)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
