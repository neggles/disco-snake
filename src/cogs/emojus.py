import logging
from asyncio import sleep
from typing import Any, Coroutine, Union

import disnake
from disnake import Emoji, Guild
from disnake.ext import commands, tasks

import logsnake
from disco_snake import DATADIR_PATH, LOG_FORMAT, LOGDIR_PATH
from disco_snake.bot import DiscoSnake

COG_UID = "emojus"


logger = logsnake.setup_logger(
    name=COG_UID,
    level=logging.DEBUG,
    isRootLogger=False,
    formatter=logsnake.LogFormatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{COG_UID}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=1 * (2**20),
    backupCount=1,
    propagate=True,
)


class Emojus(commands.Cog, name=COG_UID):
    def __init__(self, bot: DiscoSnake):
        self.bot: DiscoSnake = bot
        self.save_dir = DATADIR_PATH.joinpath(COG_UID)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    async def cog_load(self) -> None:
        logger.info("idk seems pretty sus")
        self.acquire_emoji_task.start()
        return await super().cog_load()

    def cog_unload(self) -> None:
        logger.info("now somewhat less sus")
        self.acquire_emoji_task.cancel()
        return super().cog_unload()

    async def on_ready(self) -> None:
        """
        The code in this even is executed when the bot is ready
        """
        logger.info("emojus is ready to be sus")
        if self.acquire_emoji_task.is_running() is not True:
            logger.info("starting emoji acquisition task")
            self.acquire_emoji_task.start()
            logger.info(f"task will next run at: {self.acquire_emoji_task.next_iteration}")
        return await super().on_ready()

    @tasks.loop(seconds=900.0, count=1)
    async def acquire_emoji_task(self) -> None:
        guild: Guild
        async for guild in self.bot.fetch_guilds():
            try:
                logger.info(f"acquiring emoji from {guild.name}")
                guild_dir = self.save_dir.joinpath(f"{guild.id}")
                guild_dir.mkdir(parents=True, exist_ok=True)

                emoji: Emoji
                acquired = 0
                for emoji in await guild.fetch_emojis():
                    emoji_extn = "gif" if emoji.animated else "png"
                    emoji_path = guild_dir.joinpath(f"{emoji.name}.{emoji.id}.{emoji_extn}")
                    if not emoji_path.exists():
                        logger.debug(f"emoji {emoji.name} has not been saved, acquiring")
                        await emoji.save(emoji_path)
                        acquired += 1
            except Exception:
                logger.exception(f"failed to acquire emoji from {guild.name}")
        logger.debug("done acquiring emoji, sleeping for 1 hour")

    @acquire_emoji_task.before_loop
    async def before_status_task(self) -> None:
        logger.info("waiting for ready... just to be sure")
        await self.bot.wait_until_ready()
        logger.info("ready!")


def setup(bot):
    bot.add_cog(Emojus(bot))
