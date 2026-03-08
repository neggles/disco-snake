from datetime import datetime
from typing import Annotated, Any

import sqlalchemy as sa
from pydantic import BaseModel, Field
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, BigIntPK, CreateTimestamp, RequiredTimestamp, UpdateTimestamp
from db.discord.types import DiscordSnowflake


class HandledMessage(Base):
    __tablename__ = "handled_messages"
    __table_args__ = (
        sa.Index("idx_handled_messages_channel_message", "message_id", "channel_id", unique=True),
        {"schema": "rattlesnake"},
    )
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[BigIntPK]
    bot_id: Mapped[str] = mapped_column(sa.String, nullable=False)
    message_id: Mapped[DiscordSnowflake] = mapped_column(sa.BigInteger)
    channel_id: Mapped[DiscordSnowflake] = mapped_column(sa.BigInteger)
    channel_type: Mapped[str] = mapped_column(sa.String)
    reply_message_id: Mapped[DiscordSnowflake | None] = mapped_column(sa.BigInteger, nullable=True)

    message_ts: Mapped[RequiredTimestamp]
    handled_at: Mapped[CreateTimestamp]


class ReplyRequest(BaseModel):
    channel_id: DiscordSnowflake
    message_id: DiscordSnowflake
    bot_id: str
    reply_message_id: DiscordSnowflake


class ReplyResponse(BaseModel):
    ok: bool
    reason: str | None = None
    handled_by: str | None = None
    reply_message_id: DiscordSnowflake | None = None


class MessageDecision(Base):
    __tablename__ = "message_decisions"
    __table_args__ = (
        sa.Index("idx_message_decisions_channel_message", "message_id", "channel_id", unique=True),
        {"schema": "rattlesnake"},
    )
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[BigIntPK]

    message_id: Mapped[DiscordSnowflake] = mapped_column(sa.BigInteger)
    channel_id: Mapped[DiscordSnowflake] = mapped_column(sa.BigInteger)

    decided_at: Mapped[UpdateTimestamp]
    # list of candidate bot ids, ordered by score
    candidates: Mapped[dict[str, Any]] = mapped_column(pg.JSONB, nullable=True)
    # Optional detailed score info (no schema)
    scores: Mapped[dict[str, Any] | None] = mapped_column(pg.JSONB, nullable=True)
    # Optional heuristic/rule explanations, feature flags, etc. (no schema)
    reason: Mapped[dict[str, Any] | None] = mapped_column(pg.JSONB, nullable=True)


class DecisionRequest(BaseModel):
    bot_id: str
    channel_id: DiscordSnowflake
    message_id: DiscordSnowflake
    author_id: DiscordSnowflake
    content: str
    reply_to_message_id: DiscordSnowflake | None = None
    mentions: Annotated[list[str], Field(default_factory=list)]
    timestamp: datetime


class DecisionResponse(BaseModel):
    channel_id: DiscordSnowflake
    message_id: DiscordSnowflake
    decision: dict[str, Any]
