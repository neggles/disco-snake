import logging
from datetime import datetime
from pathlib import Path
from random import uniform as rand_float
from time import perf_counter

from shimeji.memorystore_provider import MemoryStoreProvider, PostgreSQLMemoryStore
from shimeji.model_provider import (
    ModelGenArgs,
    ModelGenRequest,
    ModelLogitBiasArgs,
    ModelPhraseBiasArgs,
    ModelProvider,
    ModelSampleArgs,
    SukimaModel,
)
from transformers.utils import logging as t2logging

import logsnake
from disco_snake import DATADIR_PATH, LOG_FORMAT, LOGDIR_PATH
from disco_snake.bot import DiscoSnake
from helpers import checks

MODULE_UID = "aiengine"

logger = logsnake.setup_logger(
    level=logging.DEBUG,
    isRootLogger=False,
    name=MODULE_UID,
    formatter=logsnake.LogFormatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{MODULE_UID}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=2 * (2**20),
    backupCount=2,
)
