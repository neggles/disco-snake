import json
from typing import Callable, TypeVar

from disnake.ext import commands

from disco_snake.cli import CONFIG_PATH, DATADIR_PATH
from exceptions import UserBlacklisted, UserNotOwner

T = TypeVar("T")


def is_owner() -> Callable[[T], T]:
    """
    This is a custom check to see if the user executing the command is an owner of the bot.
    """

    async def predicate(context: commands.Context) -> bool:
        data = json.loads(CONFIG_PATH.read_bytes())
        if context.author.id not in context.bot.owner_ids:
            raise UserNotOwner
        return True

    return commands.check(predicate)


def not_blacklisted() -> Callable[[T], T]:
    """
    This is a custom check to see if the user executing the command is blacklisted.
    """

    async def predicate(context: commands.Context) -> bool:
        data = json.loads(DATADIR_PATH.joinpath("blacklist.json").read_bytes())
        if context.author.id in data["ids"]:
            raise UserBlacklisted
        return True

    return commands.check(predicate)
