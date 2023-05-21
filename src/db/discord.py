from datetime import datetime
from typing import Annotated, Optional

import disnake
import sqlalchemy as sa
from disnake.ext import commands
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship, synonym_for

from db.base import Base, BigIntPK, CreateTimestamp, Timestamp, UpdateTimestamp

## Annotated types for Discord objects
DiscordSnowflake = Annotated[
    int,  # Discord "snowflake" (object ID)
    mapped_column(sa.BigInteger, unique=True),
]

DiscordName = Annotated[
    str,  # Discord Username string
    mapped_column(
        sa.String(length=64),  # Discord caps at 32, but we'll allow for 64 just in case
    ),
]


class DiscordUser(Base):
    __tablename__ = "users"
    __table_args__ = (
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username", "discriminator"),
    )

    id: Mapped[DiscordSnowflake] = mapped_column(primary_key=True)
    username: Mapped[DiscordName]
    discriminator: Mapped[int] = mapped_column(sa.Integer, server_default="0")

    avatar: Mapped[Optional[str]] = mapped_column(sa.String(length=512))
    bot: Mapped[bool] = mapped_column(sa.Boolean)
    system: Mapped[bool] = mapped_column(sa.Boolean, server_default=sa.false())

    email: Mapped[Optional[str]] = mapped_column(sa.String(length=256))
    verified: Mapped[Optional[bool]] = mapped_column(sa.Boolean)

    flags: Mapped[Optional[int]] = mapped_column(sa.Integer)
    premium_type: Mapped[Optional[int]] = mapped_column(sa.Integer)
    public_flags: Mapped[Optional[int]] = mapped_column(sa.Integer)

    first_seen: Mapped[CreateTimestamp]
    last_updated: Mapped[UpdateTimestamp]

    tos_accepted: Mapped[bool] = mapped_column(sa.Boolean, server_default=sa.false())
    tos_accepted_at: Mapped[Timestamp]
    # tos_accept_msg: Mapped[]

    @hybrid_property
    def name(self) -> str:
        if self.discriminator == 0:
            return self.username
        return f"{self.username}#{self.discriminator}"

    @name.setter
    def name(self, name: str, discriminator: int = 0) -> None:
        """Set the user's name, optionally with discriminator.
            Discriminator can be passed as a separate argument, or
            as part of the name string.
            Pass 0 as the discriminator to set the global name.


        Args:
            name (str): Discord user name with optional discriminator (e.g. "username#1234")
            discriminator (int, optional, default=0): Discord user discriminator
        """
        if "#" in name:
            username, discriminator = name.split("#")
            discriminator = int(discriminator)
        self.username = username
        self.discriminator = discriminator

    @classmethod
    def from_discord(cls, user: disnake.User | disnake.Member) -> "DiscordUser":
        if isinstance(user, disnake.Member):
            return cls(
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
        user = await bot.fetch_user(self.id)
        if user is None:
            raise ValueError(f"User {self.id} not found on Discord")


## coming back to these later
# class UsernameHistory(Base):
#     __tablename__ = "username_history"
#     __table_args__ = (sa.PrimaryKeyConstraint("user_id", "timestamp"),)

#     user_id: Mapped[int] = mapped_column(sa.BigInteger, sa.ForeignKey("users.id", ondelete="CASCADE"))
#     user: Mapped[DiscordUser] = relationship(backref="username_history")
#     timestamp: Mapped[datetime] = mapped_column(
#         pg.TIMESTAMP(timezone=True, precision=2),
#         server_default=sa.func.current_timestamp(),
#         nullable=False,
#     )
#     username: Mapped[DiscordName] = mapped_column(nullable=False)


# class NicknameHistory(Base):
#     __tablename__ = "nickname_history"
#     __table_args__ = (sa.PrimaryKeyConstraint("user_id", "timestamp"),)

#     user_id: Mapped[int] = mapped_column(sa.BigInteger, sa.ForeignKey("users.id", ondelete="CASCADE"))
#     guild_id: Mapped[int] = mapped_column(sa.BigInteger, sa.ForeignKey("guilds.id", ondelete="CASCADE"))
#     timestamp: Mapped[datetime] = mapped_column(
#         pg.TIMESTAMP(timezone=True, precision=2),
#         server_default=sa.func.current_timestamp(),
#         nullable=False,
#     )
#     username: Mapped[str] = mapped_column(sa.String, nullable=False)
