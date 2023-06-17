from datetime import datetime
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, BigIntPK, CreateTimestamp, Timestamp, UpdateTimestamp


class AiResponseLog(Base):
    __tablename__ = "ai_message_logs"
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[BigIntPK]
    app_id: Mapped[int] = mapped_column(sa.BigInteger, nullable=False, index=True)
    app: Mapped[str] = mapped_column(sa.String, nullable=False)
    timestamp: Mapped[CreateTimestamp]
    message: Mapped[Optional[dict]] = mapped_column(pg.JSONB, nullable=True)
    parameters: Mapped[Optional[dict]] = mapped_column(pg.JSONB, nullable=True)
    context: Mapped[Optional[dict]] = mapped_column(pg.JSONB, nullable=True)
