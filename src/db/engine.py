import json
from functools import lru_cache
from typing import Any

from pydantic_core import to_jsonable_python
from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from disco_snake.settings import BotSettings, get_settings

# Define type aliases for session makers
type SessionType = async_sessionmaker[AsyncSession]
type SyncSessionType = sessionmaker[DbSession]


def json_serializer(obj: Any, *args, **kwargs) -> str:
    _, _ = kwargs.pop("default", None), kwargs.pop("ensure_ascii", None)
    return json.dumps(obj, *args, default=to_jsonable_python, ensure_ascii=False, **kwargs)


# this is a kinda hacky way to make an effective module-level singleton, but it works
# TODO: refactor to proper singleton classes
@lru_cache(maxsize=1)
def get_sync_engine() -> Engine:
    settings: BotSettings = get_settings()
    return create_engine(
        url=settings.db_uri.unicode_string(),
        echo=settings.debug,
        json_serializer=json_serializer,
    )


SyncSession: SyncSessionType = sessionmaker(get_sync_engine(), expire_on_commit=False)


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    settings: BotSettings = get_settings()
    return create_async_engine(
        url=settings.db_uri.unicode_string(),
        echo=settings.debug,
        json_serializer=json_serializer,
    )


Session: SessionType = async_sessionmaker(get_engine(), expire_on_commit=False)
