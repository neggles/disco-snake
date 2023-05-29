from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from disco_snake.settings import Settings, get_settings


@lru_cache(maxsize=1)
def get_sync_engine() -> Engine:
    settings: Settings = get_settings()
    return create_engine(
        url=settings.db_uri,
        echo=settings.debug,
    )


def get_engine() -> AsyncEngine:
    settings: Settings = get_settings()
    return create_async_engine(
        url=settings.db_uri,
        echo=settings.debug,
    )


Session = async_sessionmaker(get_engine(), expire_on_commit=False)

SyncSession = sessionmaker(get_sync_engine(), expire_on_commit=False)
