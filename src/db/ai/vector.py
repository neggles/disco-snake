from typing import Optional

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, BigIntPK, CreateTimestamp
from db.discord import DiscordSnowflake


class Memory(Base):
    id: Mapped[BigIntPK]
    app_id: Mapped[int] = mapped_column(sa.BigInteger, nullable=False, index=True)
    timestamp: Mapped[CreateTimestamp]
    context_id: Mapped[DiscordSnowflake]
