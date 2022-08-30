import logging
import queue  # imported for using queue.Empty exception
import time
from multiprocessing import Lock, Process, Queue, current_process
from pathlib import Path
from traceback import format_exc
from typing import List

import disnake
import logsnake
from aitextgen import aitextgen
from disco_snake.cli import DATADIR_PATH, LOGDIR_PATH
from disnake.ext import commands, tasks
from helpers.misc import parse_log_level

MODELS_ROOT = DATADIR_PATH.joinpath("models")
NEWLINE = "\n"

EMOJI = {
    "huggingface": "🤗",
    "disco": "💃",
    "snake": "🐍",
    "disco_snake": "💃🐍",
    "robot": "🤖",
    "confus": "😕",
    "thonk": "🤔",
}
