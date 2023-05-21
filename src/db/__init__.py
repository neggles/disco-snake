import logging

import logsnake
from db.ai import ImageCaption
from db.base import Base
from db.discord import DiscordUser
from db.engine import Session, get_engine
from disco_snake import LOG_FORMAT, LOGDIR_PATH

__all__ = [
    "Base",
    "DiscordUser",
    "ImageCaption",
    "Session",
    "get_engine",
]


# setup DB logger
logger = logsnake.setup_logger(
    name=__name__,
    level=logging.DEBUG,
    isRootLogger=False,
    formatter=logsnake.LogFormatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{__name__}.log"),
    fileLoglevel=logging.INFO,
    maxBytes=1 * (2**20),
    backupCount=2,
)
