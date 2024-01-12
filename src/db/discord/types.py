from typing import Annotated

import sqlalchemy as sa
from sqlalchemy.orm import mapped_column

## Annotated types for Discord objects
DiscordSnowflake = Annotated[
    int,  # Discord "snowflake" (object ID)
    mapped_column(sa.BigInteger, unique=True),
]

DiscordName = Annotated[
    str,  # Discord Username string
    mapped_column(
        sa.String(length=64),  # Discord caps at 32, but we'll allow for 64 just in case
    ),
]
