from datetime import datetime
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class ImageCaption(Base):
    __tablename__ = "image_captions"

    id: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True)
    filename: Mapped[str] = mapped_column(sa.String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(sa.String(length=1024))
    size: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    url: Mapped[str] = mapped_column(sa.String, nullable=False)
    height: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    width: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(pg.TIMESTAMP(timezone=True, precision=2))

    thumbnail: Mapped[Optional[bytes]] = mapped_column(sa.LargeBinary)
    caption: Mapped[str] = mapped_column(sa.String(length=512))
    captioned_at: Mapped[datetime] = mapped_column(
        pg.TIMESTAMP(timezone=True, precision=2),
        nullable=False,
        server_default=sa.func.current_timestamp(),
    )
    captioned_with: Mapped[str] = mapped_column(sa.String(length=256))
