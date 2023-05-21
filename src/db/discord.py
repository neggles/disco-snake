from datetime import datetime
from typing import Annotated, Optional

import disnake
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.orm import Mapped, mapped_column, relationship

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
        sa.UniqueConstraint("global_name"),
    )

    id: Mapped[DiscordSnowflake] = mapped_column(primary_key=True)
    username: Mapped[DiscordName]
    discriminator: Mapped[int] = mapped_column(sa.Integer, server_default="0")
    global_name: Mapped[Optional[str]] = mapped_column(sa.String)

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

    @classmethod
    def from_discord(cls, user: disnake.User | disnake.Member) -> "DiscordUser":
        return cls(
            id=user.id,
            username=user.name,
            discriminator=user.discriminator,
            global_name=getattr(user, "global_name", None),
            avatar=user.avatar.key if user.avatar else None,
            bot=user.bot,
            system=user.system,
            flags=user.flags.value if hasattr(user, "flags") else None,
            public_flags=user.public_flags.value,
        )


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
