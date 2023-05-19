import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg


def mock_dump(sql, *multiparams, **params):
    print(sql.compile(dialect=mock_engine.dialect))


mock_engine = sa.create_mock_engine("postgresql+psycopg://", mock_dump)
