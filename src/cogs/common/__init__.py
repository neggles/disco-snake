from typing import Union

from disnake import DMChannel, GroupChannel, TextChannel, Thread

from .upscaler import Upscaler

from . import utils

MessageChannel = Union[TextChannel, Thread, DMChannel, GroupChannel]

__all__ = "Upscaler", "MessageChannel", "utils"
