import asyncio
import logging
from typing import List, Optional

from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.redis import RedisStorage2
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.types import ContentType, Message
from aiogram.utils.executor import Executor
from disnake.ext import commands

import logsnake
from ai.core import (
    COG_UID as AI_COG_UID,
    Ai,
)
from cogs.telegram.settings import get_tg_settings
from disco_snake import DATADIR_PATH, LOG_FORMAT, LOGDIR_PATH

COG_UID = "telegram"

logger = logsnake.setup_logger(
    name=__name__,
    level=logging.DEBUG,
    isRootLogger=False,
    formatter=logsnake.LogFormatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{COG_UID}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=1 * (2**20),
    backupCount=3,
    propagate=True,
)

# redirect aiogram logger to cog logger
aiogram_logger = logging.getLogger("aiogram")
aiogram_logger.propagate = False
for handler in logger.handlers:
    aiogram_logger.addHandler(handler)


TG_DATA_DIR = DATADIR_PATH.joinpath("telegram")


class TelegramCog(commands.Cog, name="telegram"):
    def __init__(self, discord_bot: commands.Bot):
        TG_DATA_DIR.mkdir(exist_ok=True, parents=True)
        self.discord: commands.Bot = discord_bot
        self.ai: Optional[Ai] = None

        self.settings = get_tg_settings()
        self.bot = Bot(token=self.settings.token)
        self.storage = RedisStorage2(**self.settings.redis.dict())
        self.dp = Dispatcher(bot=self.bot, storage=self.storage)
        self.dp.middleware.setup(LoggingMiddleware())

        self.exec: Executor
        self.task: Optional[asyncio.Task] = None

    async def cog_load(self) -> None:
        """Runs when the cog is first loaded"""
        logger.info("Telegram bot engine loading...")
        self.exec = Executor(dispatcher=self.dp, skip_updates=True, loop=asyncio.get_event_loop())
        self.dp.register_message_handler(self.tg_welcome, commands=["start", "help"])
        self.dp.register_message_handler(self.tg_message, state="*", content_types=ContentType.ANY)
        pass

    def cog_unload(self) -> None:
        """Runs when the cog is unloaded, does cleanup. Cannot be async."""
        if self.task is not None:
            logger.info("Shutting down telegram task:")
            loop = asyncio.get_event_loop()
            self.task.cancel()
            self.task = None
            loop.run_until_complete(self.exec._shutdown_polling())
        return super().cog_unload()

    @commands.Cog.listener("on_ready")
    async def on_ready(self):
        """Run when the cog is ready (may run multiple times)"""
        logger.info("Discord bot is ready, starting Telegram one")
        # retrieve the AI cog
        self.ai = self.discord.cogs.get(AI_COG_UID, None)
        if self.ai is None:
            logger.error("Could not find AI cog :(")
            self.discord.remove_cog(self.__cog_name__)
            raise ValueError("Could not find AI cog :(")
        if self.task is None:
            await self.start_polling()
        logger.info("TG bot is running in theory???")

    async def start_polling(
        self,
        reset_webhook=None,
        timeout=20,
        relax=0.1,
        fast=True,
        allowed_updates: Optional[List[str]] = None,
    ):
        """Start the Telegram polling loop"""
        loop = asyncio.get_event_loop()
        logger.info("Starting polling...")
        await self.exec._startup_polling()
        self.task = loop.create_task(
            self.dp.start_polling(
                reset_webhook=reset_webhook,
                timeout=timeout,
                relax=relax,
                fast=fast,
                allowed_updates=allowed_updates,
            )
        )
        logger.info("Poller is running")

    ### Telegram shit
    async def tg_welcome(self, message: Message):
        """
        This handler will be called when user sends `/start` or `/help` command
        """
        logger.debug(f"Got message: {message.as_json()}")
        await message.reply("ohayou~")

    async def tg_message(self, message: Message):
        if message.from_user.is_bot:
            return
        logger.debug(f"Got message: {message.as_json()}")
        if message.chat.id not in self.settings.enabled_chats:
            return

        message_content = message.text
        conversation = [message.chat.get_current()]

        pass


def setup(bot):
    bot.add_cog(TelegramCog(bot))
