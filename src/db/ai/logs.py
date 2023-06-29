from datetime import datetime
from typing import Any, Optional

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
    author_id: int = Field(...)
    author: str | dict[str, Any] = Field(...)
    channel_id: int = Field(...)
    channel: str | dict[str, Any] = Field(...)
    author_name: str = Field(...)
    trigger: str = Field(...)
    content: str = Field(...)


class AiResponseLog(Base):
    __tablename__ = "ai_message_logs"
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[BigIntPK]
    app_id: Mapped[int] = mapped_column(sa.BigInteger, nullable=False, index=True)
    instance: Mapped[str] = mapped_column(sa.String, nullable=False)
    timestamp: Mapped[CreateTimestamp]
    message: Mapped[Optional[AiMessageInfo]] = mapped_column(pg.JSONB, nullable=True)
    parameters: Mapped[Optional[dict[str, Any]]] = mapped_column(pg.JSONB, nullable=True)
    conversation: Mapped[Optional[list[str]]] = mapped_column(pg.ARRAY(sa.String), nullable=True)
    context: Mapped[Optional[list[str]]] = mapped_column(pg.ARRAY(sa.String), nullable=True)
    response: Mapped[Optional[str]] = mapped_column(sa.String, nullable=True)
    response_raw: Mapped[Optional[str]] = mapped_column(sa.String, nullable=True)

    @classmethod
    def from_old_format(cls, log_obj: dict) -> "AiResponseLog":
        return cls(
            app_id=log_obj["app_id"],
            instance=log_obj["app"],
            timestamp=datetime.fromisoformat(log_obj["timestamp"]),
            message=AiMessageInfo.parse_obj(log_obj["message"]),
            conversation=log_obj["conversation"],
            parameters=log_obj["gensettings"],
            context=log_obj["context"],
            response=log_obj["response"],
            response_raw=log_obj["response_raw"],
        )
