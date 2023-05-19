import json
import logging
import re
from asyncio import Lock
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import choice as random_choice
from traceback import format_exc
from typing import Any, Dict, List, Optional, Tuple, Union

from dacite import from_dict
from disnake import (
    DMChannel,
    Embed,
    File,
    GroupChannel,
    Message,
    MessageInteraction,
    TextChannel,
    Thread,
    User,
)

import logsnake
from ai.config import ChatbotConfig, MemoryStoreConfig, ModelProviderConfig
from ai.model import get_enma_model, get_ooba_model
from ai.types import MessageChannel
from disco_snake import DATADIR_PATH, LOG_FORMAT, LOGDIR_PATH
from disco_snake.bot import DiscoSnake


logger = logging.getLogger(__name__)
