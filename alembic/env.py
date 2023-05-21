import asyncio
from logging.config import fileConfig

from alembic import context
from alembic.environment import EnvironmentContext
from alembic.script import ScriptDirectory
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from db.base import Base
from disco_snake.settings import Settings, get_settings

## Custom: Load bot config object
try:
    bot_settings: Settings = get_settings()
except Exception as e:
    bot_settings = None

# this is the Alembic Config object, which provides access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging. This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

## Custom: If bot config object is loaded, use it to set the db_uri, otherwise use the .ini
if bot_settings is not None:
    pg_uri = bot_settings.db_uri
    config.set_main_option("sqlalchemy.url", pg_uri)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        process_revision_directives=process_revision_directives,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    In this scenario we need to create an Engine and associate a connection with the context.

    """

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

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
