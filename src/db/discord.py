from datetime import datetime
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class DiscordUser(Base):
    __tablename__ = "discord_users"

    id: Mapped[int] = mapped_column(
        sa.Integer,
        sa.Identity(start=42, cycle=True, always=True),
        nullable=False,
        primary_key=True,
    )
    discord_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        nullable=False,
        unique=True,
        index=True,
    )
    data: Mapped[dict] = mapped_column(
        pg.JSONB,
        nullable=False,
        server_default="{}",
    )
    first_seen: Mapped[datetime] = mapped_column(
        pg.TIMESTAMP(timezone=True, precision=2),
        nullable=False,
        server_default=sa.func.current_timestamp(),
    )
    last_seen: Mapped[datetime] = mapped_column(
        pg.TIMESTAMP(timezone=True, precision=2),
        nullable=False,
        server_default=sa.func.current_timestamp(),
        server_onupdate=sa.func.current_timestamp(),
    )

    username: Mapped[str] = mapped_column(
        sa.String,
        sa.Computed("data->>'username'"),
        nullable=False,
    )
    avatar: Mapped[Optional[str]] = mapped_column(
        sa.String,
        sa.Computed("data->>'avatar'"),
    )
    banner: Mapped[Optional[str]] = mapped_column(
        sa.String,
        sa.Computed("data->>'banner'"),
    )
    discriminator: Mapped[int] = mapped_column(
        sa.Integer,
        sa.Computed("COALESCE((data->>'discriminator')::integer, 0)"),
    )

    bot: Mapped[Optional[bool]] = mapped_column(
        sa.Boolean,
        sa.Computed("(data->>'bot')::boolean"),
    )
    system: Mapped[Optional[bool]] = mapped_column(
        sa.Boolean,
        sa.Computed("(data->>'system')::boolean"),
    )
    mfa_enabled: Mapped[Optional[bool]] = mapped_column(
        sa.Boolean,
        sa.Computed("(data->>'mfa_enabled')::boolean"),
    )

    email: Mapped[Optional[str]] = mapped_column(
        sa.String,
        sa.Computed("data->>'email'"),
    )
    verified: Mapped[Optional[bool]] = mapped_column(
        sa.Boolean,
        sa.Computed("(data->>'verified')::boolean"),
    )

    flags: Mapped[Optional[int]] = mapped_column(
        sa.Integer,
        sa.Computed("(data->>'flags')::integer"),
    )
    premium_type: Mapped[Optional[int]] = mapped_column(
        sa.Integer,
        sa.Computed("(data->>'premium_type')::integer"),
    )
    public_flags: Mapped[Optional[int]] = mapped_column(
        sa.Integer,
        sa.Computed("(data->>'public_flags')::integer"),
    )
