import logging

import logsnake
from ai.core import Ai, setup
from disco_snake import LOG_FORMAT, LOGDIR_PATH

logger = logsnake.setup_logger(
    name=__name__,
    level=logging.DEBUG,
    isRootLogger=False,
    formatter=logsnake.LogFormatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{__name__}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=1 * (1**20),
    backupCount=3,
    propagate=True,
)

__all__ = ["Ai", "setup", "logger"]
