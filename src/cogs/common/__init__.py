from typing import Union

from disnake import DMChannel, GroupChannel, TextChannel, Thread

from .upscaler import Upscaler

MessageChannel = Union[TextChannel, Thread, DMChannel, GroupChannel]


def setup(*args, **kwargs):
    """
    Stub function for setup so the cog loader doesn't complain
    """
    pass


__all__ = "Upscaler", "MessageChannel", "utils", "setup"
