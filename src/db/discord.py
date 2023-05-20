from datetime import datetime
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class DiscordUser(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True)
    username: Mapped[str] = mapped_column(sa.String)
    discriminator: Mapped[int] = mapped_column(sa.Integer, server_default="0")
    global_name: Mapped[Optional[str]] = mapped_column(sa.String)

    avatar: Mapped[Optional[str]] = mapped_column(sa.String)
    bot: Mapped[bool] = mapped_column(sa.Boolean)
    system: Mapped[bool] = mapped_column(sa.Boolean, server_default=sa.false())

    email: Mapped[Optional[str]] = mapped_column(sa.String)
    verified: Mapped[Optional[bool]] = mapped_column(sa.Boolean)

    flags: Mapped[Optional[int]] = mapped_column(sa.Integer)
    premium_type: Mapped[Optional[int]] = mapped_column(sa.Integer)
    public_flags: Mapped[Optional[int]] = mapped_column(sa.Integer)

    first_seen: Mapped[datetime] = mapped_column(
        pg.TIMESTAMP(timezone=True, precision=2),
        server_default=sa.func.current_timestamp(),
        nullable=False,
    )
    last_updated: Mapped[datetime] = mapped_column(
        pg.TIMESTAMP(timezone=True, precision=2),
        server_default=sa.func.current_timestamp(),
        server_onupdate=sa.func.current_timestamp(),
        nullable=False,
    )


class UsernameHistory(Base):
    __tablename__ = "username_history"
    __table_args__ = (sa.PrimaryKeyConstraint("user_id", "timestamp"),)

    user_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        pg.TIMESTAMP(timezone=True, precision=2),
        server_default=sa.func.current_timestamp(),
        nullable=False,
    )
    username: Mapped[str] = mapped_column(sa.String, nullable=False)
