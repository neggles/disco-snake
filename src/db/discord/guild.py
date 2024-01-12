from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.orm.collections import attribute_keyed_dict

from db.base import Base, Timestamp, UpdateTimestamp
from db.discord.types import DiscordSnowflake

if TYPE_CHECKING:
    from db.discord.user import DiscordUser


class NSFWLevel(int, Enum):
    DEFAULT = 0
    EXPLICIT = 1
    SAFE = 2
    AGE_RESTRICTED = 3


if False:

    class DiscordGuild(Base):
        __tablename__ = "guilds"

        id: Mapped[DiscordSnowflake] = mapped_column(primary_key=True)
        name: Mapped[str] = mapped_column(sa.String(length=100), nullable=False)
        member_count: Mapped[int] = mapped_column(sa.Integer, nullable=False)
        description: Mapped[Optional[str]] = mapped_column(sa.String(length=1000))
        created_at: Mapped[Timestamp]
        nsfw_level: Mapped[NSFWLevel] = mapped_column(sa.Enum(NSFWLevel), nullable=False)

        guild_member_associations: Mapped[list[GuildMemberAssociation]] = relationship()

        guild_member_associations: Mapped[dict[int, GuildMemberAssociation]] = relationship(
            back_populates="guild",
            collection_class=attribute_keyed_dict("user_id"),
            cascade="all, delete-orphan",
        )

        def __init__(self, id: int):
            self.id = id

    class GuildMemberAssociation(Base):
        __tablename__ = "guild_members"
        __table_args__ = (
            sa.PrimaryKeyConstraint("guild_id"),
            sa.UniqueConstraint("guild_id", "user_id"),
        )

        guild_id: Mapped[DiscordSnowflake] = mapped_column(sa.ForeignKey("guilds.id"), primary_key=True)
        user_id: Mapped[DiscordSnowflake] = mapped_column(sa.ForeignKey("users.id"))

        guild: Mapped[DiscordGuild] = relationship(back_populates="guild_member_associations")
        member: Mapped[DiscordUser] = relationship(back_populates="guild_associations")

        display_name: Mapped[str] = mapped_column(sa.String(length=100), nullable=False)
        display_avatar: Mapped[str] = mapped_column(sa.String(length=100), nullable=False)
        joined_at: Mapped[Timestamp] = mapped_column(sa.DateTime, nullable=False)
        last_updated: Mapped[UpdateTimestamp]
