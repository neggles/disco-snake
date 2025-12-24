from datetime import datetime
from typing import Any

import sqlalchemy as sa
from pydantic import BaseModel, Field
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, BigIntPK, CreateTimestamp


class AiMessageInfo(BaseModel):
    id: int = Field(...)
    timestamp: datetime = Field(datetime.now())
    guild_id: int = Field(...)
    guild: str | dict[str, Any] = Field(...)
    author_id: int | None = None
    author: str | dict[str, Any] | None = None
    channel_id: int | None = None
    channel: str | dict[str, Any] | None = None
    author_name: str | None = None
    trigger: str = Field(...)
    content: str = Field(...)


class AiResponseLog(Base):
    __tablename__ = "ai_message_logs"
    __mapper_args__ = {
        "eager_defaults": True,
        "primary_key": ["id", "app_id"],
    }

    id: Mapped[BigIntPK]
    app_id: Mapped[int] = mapped_column(sa.BigInteger, nullable=False)
    instance: Mapped[str] = mapped_column(sa.String, nullable=False)
    timestamp: Mapped[CreateTimestamp]
    message: Mapped[AiMessageInfo | None] = mapped_column(pg.JSONB, nullable=True)
    parameters: Mapped[dict[str, Any] | None] = mapped_column(pg.JSONB, nullable=True)
    conversation: Mapped[list[str] | None] = mapped_column(pg.ARRAY(sa.String), nullable=True)
    context: Mapped[list[str] | None] = mapped_column(pg.ARRAY(sa.String), nullable=True)
    response: Mapped[str | None] = mapped_column(sa.String, nullable=True)
    response_raw: Mapped[str | None] = mapped_column(sa.String, nullable=True)
    response_id: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True, index=True, unique=True)
    thoughts: Mapped[list[str] | None] = mapped_column(pg.ARRAY(sa.String), nullable=True)
    n_prompt_tokens: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    n_context_tokens: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
