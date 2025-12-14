from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import disnake
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, column_property, mapped_column, relationship

from db.base import Base, CreateTimestamp, Timestamp, UpdateTimestamp
from db.discord import DiscordName, DiscordSnowflake

if TYPE_CHECKING:
    from disnake.ext import commands

    # from db.discord.guild import DiscordGuild, GuildMemberAssociation


class BlacklistEntry(Base):
    __tablename__ = "user_blacklist"
    __table_args__ = (sa.PrimaryKeyConstraint("user_id"),)

    user_id: Mapped[int] = mapped_column(sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    user: Mapped[DiscordUser] = relationship()
    reason: Mapped[str] = mapped_column(sa.String(length=256), nullable=False)
    timestamp: Mapped[CreateTimestamp]


class DiscordUser(Base):
    __tablename__ = "users"
    __table_args__ = (
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username", "discriminator"),
    )

    id: Mapped[DiscordSnowflake] = mapped_column(primary_key=True)
    username: Mapped[DiscordName]
    discriminator: Mapped[int] = mapped_column(sa.Integer, server_default="0")

    avatar: Mapped[str | None] = mapped_column(sa.String(length=512))
    bot: Mapped[bool] = mapped_column(sa.Boolean)
    system: Mapped[bool] = mapped_column(sa.Boolean, server_default=sa.false())

    email: Mapped[str | None] = mapped_column(sa.String(length=256), default=None)
    verified: Mapped[bool | None] = mapped_column(sa.Boolean, default=None)

    flags: Mapped[int | None] = mapped_column(sa.Integer, default=None)
    premium_type: Mapped[int | None] = mapped_column(sa.Integer, default=None)
    public_flags: Mapped[int | None] = mapped_column(sa.Integer, default=None)

    first_seen: Mapped[CreateTimestamp]
    last_updated: Mapped[UpdateTimestamp]

    tos_accepted: Mapped[bool] = mapped_column(sa.Boolean, server_default=sa.false(), index=True)
    tos_rejected: Mapped[bool] = mapped_column(sa.Boolean, server_default=sa.false(), index=True)
    tos_timestamp: Mapped[Timestamp]

    blacklisted: Mapped[bool] = column_property(
        sa.select(BlacklistEntry.user_id).where(BlacklistEntry.user_id == id).exists()
    )

    # guild_member_associations: Mapped[dict[int, GuildMemberAssociation]] = relationship(
    #     back_populates="member",
    #     collection_class=attribute_keyed_dict("guild_id"),
    #     cascade="all, delete-orphan",
    # )
    # guilds: AssociationProxy[dict[int, DiscordGuild]] = association_proxy(
    #     "guild_member_associations",
    #     "guild",
    #     creator=lambda guild: GuildMemberAssociation(guild=guild),
    # )

    @hybrid_property
    def name(self) -> str:
        if self.discriminator == 0:
            return self.username.rstrip("#0")
        return f"{self.username}#{self.discriminator}"

    @name.setter
    def name(self, name: str, discriminator: int = 0) -> None:
        """Set the user's name, optionally with discriminator.

        Discriminator can be passed as a separate argument, or as part of the name string.
        Pass 0 as the discriminator to set the global name.

        :param name: Discord user name with optional discriminator (e.g. "username#1234")
        :type name: str
        :param discriminator: Discord user discriminator
        :type discriminator: int
        :param name: str:
        :param discriminator: int:  (Default value = 0)

        """
        if "#" in name:
            username, discriminator = name.split("#")
            discriminator = int(discriminator)
        self.username = username
        self.discriminator = discriminator

    @classmethod
    def from_discord(cls, user: disnake.User | disnake.Member) -> DiscordUser:
        """Convert a disnake User or Member object to a DiscordUser object.

        :param user: :class:`disnake.User` | :class:`disnake.Member`
        :return: the DiscordUser object
        :rtype: :class:`DiscordUser`

        """
        if isinstance(user, disnake.Member):
            return cls(
                id=user.id,
                username=user.name,
                discriminator=user.discriminator,
                avatar=user.avatar.key if user.avatar else None,
                bot=user.bot,
                system=user.system,
                flags=user.flags.value,
                public_flags=user.public_flags.value,
            )
        return cls(
            id=user.id,
            username=user.name,
            discriminator=user.discriminator,
            avatar=user.avatar.key if user.avatar else None,
            bot=user.bot,
            system=user.system,
            public_flags=user.public_flags.value,
        )

    async def to_discord(self, bot: commands.Bot) -> disnake.User:
        """Convert the user to a disnake.User object.

        This will fetch the user from Discord, and raise a :class:`ValueError`
        if the user is not found.

        **This does *not* use the values from the database, or update them!**

        :param bot: The bot instance to use for fetching the user
        :type bot: commands.Bot
        :raises ValueError: If the user is not found on Discord
        :return: The user object
        :rtype: disnake.User
        """
        user = await bot.fetch_user(self.id)
        if user is None:
            raise ValueError(f"User {self.id} not found on Discord")
        return user


## coming back to these later
if False:

    class UsernameHistory(Base):
        __tablename__ = "username_history"
        __table_args__ = (sa.PrimaryKeyConstraint("user_id", "timestamp"),)

        user_id: Mapped[int] = mapped_column(sa.BigInteger, sa.ForeignKey("users.id", ondelete="CASCADE"))
        user: Mapped[DiscordUser] = relationship(backref="username_history")
        timestamp: Mapped[datetime] = mapped_column(
            pg.TIMESTAMP(timezone=True, precision=2),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        )
        username: Mapped[DiscordName] = mapped_column(nullable=False)

    class NicknameHistory(Base):
        __tablename__ = "nickname_history"
        __table_args__ = (sa.PrimaryKeyConstraint("user_id", "timestamp"),)

        user_id: Mapped[int] = mapped_column(sa.BigInteger, sa.ForeignKey("users.id", ondelete="CASCADE"))
        guild_id: Mapped[int] = mapped_column(sa.BigInteger, sa.ForeignKey("guilds.id", ondelete="CASCADE"))
        timestamp: Mapped[datetime] = mapped_column(
            pg.TIMESTAMP(timezone=True, precision=2),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        )
        username: Mapped[str] = mapped_column(sa.String, nullable=False)
