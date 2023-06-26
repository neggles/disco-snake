from datetime import datetime
from typing import Annotated, Optional

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass, mapped_column


# declarative base class
class Base(AsyncAttrs, DeclarativeBase, MappedAsDataclass):
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
    Optional[datetime],
    mapped_column(
        pg.TIMESTAMP(timezone=True, precision=2),
        nullable=True,
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
    ),
]
