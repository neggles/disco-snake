import os
from functools import lru_cache

from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import create_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from db.base import Base

from .settings import RattlesnakeSettings, get_settings

# Define type aliases for sessionmakers
type SessionType = async_sessionmaker[AsyncSession]
type SyncSessionType = sessionmaker[DbSession]


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"missing required env var: {name}")
    return val


@lru_cache(maxsize=1)
def get_sync_engine() -> Engine:
    settings: RattlesnakeSettings = get_settings()
    return create_engine(
        url=settings.db_uri.unicode_string(),
        echo=settings.debug,
    )


SyncSession: SyncSessionType = sessionmaker(get_sync_engine(), expire_on_commit=False)


@lru_cache(maxsize=1)
def get_async_engine() -> AsyncEngine:
    settings: RattlesnakeSettings = get_settings()
    return create_async_engine(
        url=settings.db_uri.unicode_string(),
        echo=settings.debug,
    )


# using the sqlmodel subclass of AsyncEngine makes pyright upset, so we ignore the type check here
get_session: SessionType = async_sessionmaker(get_async_engine(), class_=AsyncSession, expire_on_commit=False)  # pyright: ignore[reportAssignmentType]


async def insert_if_not_exists(
    session: AsyncSession,
    instance: Base,
) -> bool:
    """Insert the given instance into the database if a record with the same unique attributes does not already exist.

    Args:
        session: The database session to use for the operation.
        instance: The instance to insert.

    Returns:
        True if the instance was inserted, False if one already existed with the same unique attributes.
    """
    result = await session.exec(
        pg.insert(instance.__class__)
        .values(**instance.model_dump(exclude_unset=True))
        .on_conflict_do_nothing()
    )
    return result.rowcount > 0
