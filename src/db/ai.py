import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class ImageCaption(Base):
    __tablename__ = "image_captions"
