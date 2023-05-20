from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import AsyncAttrs


# declarative base class
class Base(AsyncAttrs, DeclarativeBase):
    pass
