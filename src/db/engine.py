from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from disco_snake.settings import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def get_sync_engine() -> Engine:
    return create_engine(
        url=settings.db_uri,
        echo=settings.debug,
    )


def get_engine() -> AsyncEngine:
    return create_async_engine(
        url=settings.db_uri,
        echo=settings.debug,
    )


engine = get_engine()
Session = async_sessionmaker(engine)

sync_engine = get_sync_engine()
SyncSession = sessionmaker(sync_engine)
