from functools import lru_cache

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.orm import sessionmaker

from disco_snake.settings import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def get_engine() -> sa.Engine:
    return sa.create_engine(
        url=settings.db_uri,
        echo=settings.debug,
    )


engine = get_engine()
Session = sessionmaker(engine)
