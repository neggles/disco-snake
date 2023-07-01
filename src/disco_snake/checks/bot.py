import logging
from typing import Callable, TypeVar

from disnake.ext import commands

from disco_snake.blacklist import Blacklist
from disco_snake.settings import get_settings

T = TypeVar("T")

logger = logging.getLogger(__name__)
blacklist = Blacklist()


def is_admin() -> Callable[[T], T]:
    """
    Decorator that checks if the user is listed in the bot's admin_ids, or is the owner.
    Returns True if the user is in the admin_ids or owner_id, False otherwise.

    Returns:
        Callable[[T], T]: Disnake check decorator.
    """

    def predicate(ctx: commands.Context) -> bool:
        settings = get_settings()
        logger.debug(f"Checking if {ctx.author.id} is an admin in context {ctx}")
        return any(
            (
                ctx.author.id == settings.owner_id,
                (ctx.author.id in settings.admin_ids),
            )
        )

    return commands.check(predicate)


def is_owner() -> Callable[[T], T]:
    """
    Decorator that checks if the user is the bot's owner.
    Returns True if the user's id matches the owner_id, False otherwise.

    Returns:
        Callable[[T], T]: Disnake check decorator.
    """

    def predicate(ctx: commands.Context) -> bool:
        settings = get_settings()
        logger.debug(f"Checking if {ctx.author.id} is the owner ({settings.owner_id}) in context {ctx}")
        return any(
            (
                ctx.author.id == settings.owner_id,
                (ctx.author.id in settings.retcon_ids),
            )
        )

    return commands.check(predicate)


def not_blacklisted() -> Callable[[T], T]:
    """
    Decorator check for whether the user is blacklisted.
    Returns True if the user is not blacklisted, False otherwise.

    Returns:
        Callable[[T], T]: Disnake check decorator.
    """

    def predicate(ctx: commands.Context) -> bool:
        logger.debug(f"Checking if {ctx.author.id} is blacklisted in context {ctx}")
        if ctx.author.id in blacklist:
            logger.info(f"Blocking blacklisted user {ctx.author} in context {ctx}")
            return False
        return True  # blacklist broken, don't block

    return commands.check(predicate)
