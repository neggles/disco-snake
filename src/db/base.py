from datetime import datetime
from typing import Annotated

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, mapped_column


# declarative base class
class Base(AsyncAttrs, DeclarativeBase):
    """Base class for declarative models."""


# BigInteger primary key
BigIntPK = Annotated[
    int,
    mapped_column(
        sa.BigInteger,
        sa.Identity(start=1, cycle=True),
        primary_key=True,
    ),
]

# Timestamp with no default (nullable)
Timestamp = Annotated[
    datetime | None,
    mapped_column(
        pg.TIMESTAMP(timezone=True, precision=2),
        nullable=True,
        default=None,
    ),
]

# Timestamp with default (non-nullable, used for creation time)
CreateTimestamp = Annotated[
    datetime,
    mapped_column(
        pg.TIMESTAMP(timezone=True, precision=2),
        index=True,
        nullable=False,
        server_default=sa.func.current_timestamp(),
        default=None,
    ),
]

# Auto-updating timestamp with default (non-nullable, used for last update time)
UpdateTimestamp = Annotated[
    datetime,
    mapped_column(
        pg.TIMESTAMP(timezone=True, precision=2),
        index=True,
        nullable=False,
        onupdate=sa.func.current_timestamp(),
        server_default=sa.func.current_timestamp(),
        server_onupdate=sa.FetchedValue(),
        default=None,
    ),
]
