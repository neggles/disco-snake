from enum import IntEnum

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, BigIntPK, CreateTimestamp


class LogLevel(IntEnum):
    NOTSET = 0
    TRACE = 5
    DEBUG = 10
    INFO = 20
    SUCCESS = 25
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogMessage(Base):
    __tablename__ = "logs"
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[BigIntPK]
    app_id: Mapped[int] = mapped_column(sa.BigInteger, nullable=False, index=True)
    instance: Mapped[str] = mapped_column(sa.String, nullable=False)
    timestamp: Mapped[CreateTimestamp]
    logger: Mapped[str] = mapped_column(sa.String, nullable=False)
    level: Mapped[LogLevel] = mapped_column(sa.Enum(LogLevel), nullable=False)
    message: Mapped[str] = mapped_column(sa.String, nullable=False)
    trace: Mapped[str | None] = mapped_column(sa.String, nullable=True)
    record: Mapped[dict | None] = mapped_column(pg.JSONB, nullable=True)
