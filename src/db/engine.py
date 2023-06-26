import json
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, parse_obj_as
from pydantic.json import pydantic_encoder
from sqlalchemy import JSON, Engine, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import TypeDecorator

from disco_snake.settings import Settings, get_settings


class PydanticType(TypeDecorator):
    """Pydantic type.
    SAVING:
    - Uses SQLAlchemy JSON type under the hood.
    - Acceps the pydantic model and converts it to a dict on save.
    - SQLAlchemy engine JSON-encodes the dict to a string.
    RETRIEVING:
    - Pulls the string from the database.
    - SQLAlchemy engine JSON-decodes the string to a dict.
    - Uses the dict to create a pydantic model.
    """

    impl = JSONB  # change to sa.types.JSON for non-PostgreSQL databases

    def __init__(self, pydantic_type: BaseModel):
        super().__init__()
        self.pydantic_type = pydantic_type

    def load_dialect_impl(self, dialect):
        # Use JSONB for PostgreSQL and JSON for other databases.
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: BaseModel, dialect):
        return value.dict() if value else None

    def process_result_value(self, value, dialect):
        return parse_obj_as(self.pydantic_type, value) if value else None


def json_serializer(obj: Any, *args, **kwargs) -> str:
    _, _ = kwargs.pop("default", None), kwargs.pop("ensure_ascii", None)
    return json.dumps(obj, *args, default=pydantic_encoder, ensure_ascii=False, **kwargs)


@lru_cache(maxsize=1)
def get_sync_engine() -> Engine:
    settings: Settings = get_settings()
    return create_engine(
        url=settings.db_uri,
        echo=settings.debug,
        json_serializer=json_serializer,
    )


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    settings: Settings = get_settings()
    return create_async_engine(
        url=settings.db_uri,
        echo=settings.debug,
        json_serializer=json_serializer,
    )


Session: async_sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)

SyncSession: sessionmaker = sessionmaker(get_sync_engine(), expire_on_commit=False)
