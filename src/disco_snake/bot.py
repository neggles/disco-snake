import json
import logging
import os
import platform
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import partial as partial_func
from pathlib import Path
from traceback import print_exception
from zoneinfo import ZoneInfo

from disnake import (
    Activity,
    ActivityType,
    ApplicationCommandInteraction,
    Embed,
    Guild,
    Intents,
    InteractionResponseType,
    Message,
    __version__ as DISNAKE_VERSION,
)
from disnake.ext import commands, tasks
from humanize import naturaldelta as fuzzydelta

import exceptions
from disco_snake import COGDIR_PATH, DATADIR_PATH, USERDATA_PATH
from disco_snake.embeds import CooldownEmbed, MissingPermissionsEmbed, MissingRequiredArgumentEmbed
from disco_snake.settings import get_settings
from helpers.misc import filename_filter, get_package_root

PACKAGE_ROOT = get_package_root()

BOT_INTENTS = Intents.all()
BOT_INTENTS.typing = False
BOT_INTENTS.presences = False
BOT_INTENTS.members = True
BOT_INTENTS.message_content = True

logger = logging.getLogger(__package__)


class DiscoSnake(commands.Bot):
    def __init__(self, *args, **kwargs):
        intents = kwargs.pop("intents", BOT_INTENTS)

        super().__init__(*args, command_prefix=None, intents=intents, **kwargs)

        self.config = get_settings()

        self.datadir_path: Path = DATADIR_PATH
        self.userdata_path: Path = USERDATA_PATH
        self.userdata: dict = None
        self.cogdir_path: Path = COGDIR_PATH
        self.start_time: datetime = datetime.now(tz=ZoneInfo("UTC"))
        self.home_guild: Guild = None  # set in on_ready

        # thread pool for blocking code
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="bot")

        # single thread worker for blocking gpu code
        self.gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bot_gpu")

    @property
    def timezone(self) -> ZoneInfo:
        return self.config.timezone

    @property
    def uptime(self) -> timedelta:
        return datetime.now(tz=ZoneInfo("UTC")) - self.start_time

    @property
    def fuzzyuptime(self) -> str:
        return fuzzydelta(self.uptime)

    async def do(self, func, *args, **kwargs):
        funcname = getattr(func, "__name__", None)
        if funcname is None:
            funcname = getattr(func.__class__, "__name__", "unknown")
        logger.info(f"Running {funcname} in background thread...")
        return await self.loop.run_in_executor(self.executor, partial_func(func, *args, **kwargs))

    async def do_gpu(self, func, *args, **kwargs):
        funcname = getattr(func, "__name__", None)
        if funcname is None:
            funcname = getattr(func.__class__, "__name__", "unknown")
        logger.info(f"Running {funcname} on GPU...")
        res = await self.loop.run_in_executor(self.gpu_executor, partial_func(func, *args, **kwargs))
        return res

    def save_userdata(self):
        if self.userdata is not None and self.userdata_path.is_file():
            with self.userdata_path.open("w") as f:
                json.dump(self.userdata, f, skipkeys=True, indent=2)

    def load_userdata(self):
        if self.userdata_path.is_file():
            with self.userdata_path.open("r") as f:
                self.userdata = json.load(f)

    def save_guild_metadata(self, guild_id: int):
        guild = self.get_guild(guild_id)
        memberlist = []
        for member in guild.members:
            memberlist.append(
                {
                    "id": member.id,
                    "name": member.name,
                    "display_name": member.display_name,
                    "avatar": str(member.avatar.url) if member.avatar else None,
                    "is_bot": member.bot,
                    "is_system": member.system,
                }
            )
        with (self.datadir_path / "guilds" / f"{guild_id}-members.json").open("w") as f:
            json.dump(memberlist, f, skipkeys=True, indent=2)

    def available_cogs(self):
        cogs = [
            p.stem
            for p in self.cogdir_path.iterdir()
            if (p.is_dir() or p.suffix == ".py") and not p.name.startswith("_")
        ]

        if len(self.config.disable_cogs) > 0:
            cogs = [x for x in cogs if x not in self.config.disable_cogs]
        return cogs

    def load_cogs(self):
        cogs = self.available_cogs()
        if cogs:
            for cog in cogs:
                try:
                    self.load_extension(f"cogs.{cog}")
                    logger.info(f"Loaded cog '{cog}'")
                except Exception as e:
                    etype, exc, tb = sys.exc_info()
                    exception = f"{etype}: {exc}"
                    logger.error(f"Failed to load cog {cog}:\n{exception}")
                    print_exception(etype, exc, tb)
        else:
            logger.info("No cogs found")

    @tasks.loop(minutes=1.0)
    async def status_task(self) -> None:
        """
        Set up the bot's status task
        """
        activity_type = getattr(ActivityType, self.config.status_type, ActivityType.playing)
        activity = Activity(name=random.choice(self.config.statuses), type=activity_type)
        await self.change_presence(activity=activity)

    @status_task.before_loop
    async def before_status_task(self):
        print("waiting...")
        await self.wait_until_ready()

    @tasks.loop(minutes=3.0)
    async def userdata_task(self) -> None:
        """
        Background task to flush user state to disk
        """
        self.save_userdata()

    async def on_ready(self) -> None:
        """
        The code in this even is executed when the bot is ready
        """
        logger.info(f"Logged in as {self.user.name}")
        logger.info(f"disnake API version: {DISNAKE_VERSION}")
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"Running on: {platform.system()} {platform.release()} ({os.name})")
        logger.info("-------------------")
        if self.home_guild is None:
            logger.info("Saving home guild metadata to disk")
            self.home_guild = self.get_guild(self.config.home_guild)
            self.save_guild_metadata(self.home_guild.id)
        if not self.status_task.is_running():
            logger.info("Starting status update task")
            self.status_task.start()
        if not self.userdata_task.is_running():
            logger.info("Starting userdata flush task")
            self.userdata_task.start()

    async def on_message(self, message: Message) -> None:
        """
        The code in this event is executed every time someone sends a message, with or without the prefix
        :param message: The message that was sent.
        """
        if message.author == self.user or message.author.bot:
            return

        await self.process_commands(message)

    async def on_slash_command(self, ctx: ApplicationCommandInteraction) -> None:
        """
        The code in this event is executed every time a slash command has been *successfully* executed
        :param ctx: The slash command that has been executed.
        """
        logger.info(
            f"Executed {ctx.data.name} command in {ctx.guild.name} (ID: {ctx.guild.id}) by {ctx.author} (ID: {ctx.author.id})"
        )

    async def on_slash_command_error(self, ctx: ApplicationCommandInteraction, error) -> None:
        if isinstance(error, commands.CommandOnCooldown):
            logger.info(
                f"User {ctx.author} attempted to use {ctx.application_command.qualified_name} on cooldown."
            )
            embed = CooldownEmbed(error.retry_after + 1, ctx.author)
            return await ctx.send(embed=embed, ephemeral=True)

        elif isinstance(error, exceptions.UserBlacklisted):
            logger.info(
                f"User {ctx.author} attempted to use {ctx.application_command.qualified_name}, but is blacklisted."
            )
            embed = Embed(
                title="Error!",
                description="You have been blacklisted and cannot use this bot. If you think this is a mistake, please contact the bot owner.",
                color=0xE02B2B,
            )
            return await ctx.send(embed=embed, ephemeral=True)

        elif isinstance(error, exceptions.UserNotOwner):
            embed = Embed(
                title="Error!",
                description="This command requires admin permissions. soz bb xoxo <3",
                color=0xE02B2B,
            )
            logger.warn(
                f"User {ctx.author} attempted to execute {ctx.application_command.qualified_name} without admin permissions."
            )
            return await ctx.send(embed=embed, ephemeral=True)

        elif isinstance(error, commands.MissingPermissions):
            logger.warn(
                f"User {ctx.author} attempted to execute {ctx.application_command.qualified_name} without authorization."
            )
            embed = MissingPermissionsEmbed(ctx.author, error.missing_permissions)
            return await ctx.send(embed=embed, ephemeral=True)

        # that covers all the usual errors, so let's catch the rest
        # first work out if we've deferred the response so we can send an ephemeral message if we need to
        ctx_rtype = getattr(ctx.response, "_response_type", None)
        ctx_ephemeral = (
            True
            if (ctx_rtype == InteractionResponseType.deferred_channel_message)
            or (ctx_rtype == InteractionResponseType.deferred_message_update)
            else False
        )

        embed = Embed(
            title="Error!",
            description="An unknown error occurred while executing this command. Please try again later or contact the bot owner if the problem persists.",
            color=0xE02B2B,
        )
        await ctx.send(embed=embed, ephemeral=ctx_ephemeral)

        logger.warn(f"Unhandled error in slash command {ctx}: {error}")
        raise error

    async def on_command_completion(self, ctx: commands.Context) -> None:
        """
        The code in this event is executed every time a normal command has been *successfully* executed
        :param ctx: The ctx of the command that has been executed.
        """
        full_command_name = ctx.command.qualified_name
        split = full_command_name.split(" ")
        executed_command = str(split[0])
        logger.info(
            f"Executed {executed_command} command in {ctx.guild.name} (ID: {ctx.message.guild.id}) by {ctx.message.author} (ID: {ctx.message.author.id})"
        )

    async def on_command_error(self, ctx: commands.Context, error) -> None:
        """
        The code in this event is executed every time a normal valid command catches an error
        :param ctx: The normal command that failed executing.
        :param error: The error that has been faced.
        """
        if isinstance(error, commands.CommandOnCooldown):
            logger.info(f"User {ctx.author} attempted to use {ctx.command.qualified_name} on cooldown.")
            embed = CooldownEmbed(error.retry_after + 1, ctx.author)
            return await ctx.send(embed=embed, ephemeral=True)

        elif isinstance(error, exceptions.UserBlacklisted):
            logger.info(
                f"User {ctx.author} attempted to use {ctx.command.qualified_name}, but is blacklisted."
            )
            embed = Embed(
                title="Error!",
                description="You have been blacklisted and cannot use this bot. If you think this is a mistake, please contact the bot owner.",
                color=0xE02B2B,
            )
            return await ctx.send(embed=embed, ephemeral=True)

        elif isinstance(error, exceptions.UserNotOwner):
            embed = Embed(
                title="Error!",
                description="This command requires admin permissions. soz bb xoxo <3",
                color=0xE02B2B,
            )
            logger.warn(
                f"User {ctx.author} attempted to execute {ctx.command.qualified_name} without authorization."
            )
            return await ctx.send(embed=embed, ephemeral=True)

        elif isinstance(error, commands.MissingPermissions):
            logger.warn(
                f"User {ctx.author} attempted to execute {ctx.command.qualified_name} without authorization."
            )
            embed = MissingPermissionsEmbed(ctx.author, error.missing_permissions)
            return await ctx.send(embed=embed, ephemeral=True)
        elif isinstance(error, commands.MissingRequiredArgument):
            logger.info(
                f"User {ctx.author} attempted to execute {ctx.command.qualified_name} without the required arguments"
            )
            embed = MissingRequiredArgumentEmbed(ctx.author, error.param.name)
            return await ctx.send(embed=embed)
        elif isinstance(error, commands.CommandNotFound):
            # This is actually fine so lets just pretend everything is okay.
            logger.info(
                f"User {ctx.author} attempted to execute a non-existent command: {ctx.message.content}"
            )
            return
        logger.warn(f"Unhandled error in command {ctx}: {error}")
        raise error
