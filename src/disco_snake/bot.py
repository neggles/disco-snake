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
from typing import Optional
from zoneinfo import ZoneInfo

from disnake import (
    Activity,
    ActivityType,
    ApplicationCommandInteraction,
    Embed,
    Guild,
    Intents,
    InteractionResponseType,
    Invite,
    Member,
    Message,
    TextChannel,
    __version__ as DISNAKE_VERSION,
)
from disnake.ext import commands, tasks
from humanize import naturaldelta as fuzzydelta

import exceptions
from db import DiscordUser, Session
from disco_snake import COGDIR_PATH, DATADIR_PATH
from disco_snake.embeds import (
    CooldownEmbed,
    MissingPermissionsEmbed,
    MissingRequiredArgumentEmbed,
    NotAdminEmbed,
)
from disco_snake.settings import get_settings
from helpers import get_package_root

PACKAGE_ROOT = get_package_root()

BOT_INTENTS = Intents.default()
BOT_INTENTS.presences = False
BOT_INTENTS.members = True
BOT_INTENTS.message_content = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = True


class DiscoSnake(commands.Bot):
    def __init__(self, config_path: Path, *args, **kwargs):
        intents = kwargs.pop("intents", BOT_INTENTS)
        super().__init__(*args, command_prefix=None, intents=intents, **kwargs)

        self.config = get_settings(config_path)
        self.datadir_path: Path = DATADIR_PATH
        self.cogdir_path: Path = COGDIR_PATH
        self.start_time: datetime = datetime.now(tz=ZoneInfo("UTC"))

        # thread pool for blocking code
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bot")

    @property
    def timezone(self) -> ZoneInfo:
        return self.config.timezone

    @property
    def uptime(self) -> timedelta:
        return datetime.now(tz=ZoneInfo("UTC")) - self.start_time

    @property
    def fuzzyuptime(self) -> str:
        return fuzzydelta(self.uptime)

    @property
    def owner_link(self):
        return f"[{self.owner}](https://discordapp.com/users/{self.owner_id})"

    @property
    def support_guild(self) -> Guild:
        return self.get_guild(self.config.support_guild)  # type: ignore

    @property
    def home_guild(self) -> Guild:
        return self.get_guild(self.config.home_guild)  # type: ignore

    @property
    def support_channel(self) -> TextChannel:
        channel = self.get_channel(self.config.support_channel)
        if channel is None:
            channel = self.support_guild.rules_channel or self.support_guild.text_channels[0]
        return channel  # type: ignore

    async def support_invite(self) -> Optional[Invite]:
        if self.support_channel is None:
            return None
        try:
            invite = await self.support_channel.create_invite(
                reason="Support Invite", max_uses=1, max_age=1800, unique=True
            )
        except Exception:
            logger.error("Failed to create support invite", exc_info=True)
            invite = None
        return invite

    async def do(self, func, *args, **kwargs):
        funcname = getattr(func, "__name__", None)
        if funcname is None:
            funcname = getattr(func.__class__, "__name__", "unknown")
        logger.info(f"Running {funcname} in background thread...")
        return await self.loop.run_in_executor(self.executor, partial_func(func, *args, **kwargs))

    def save_guild_metadata(self, guild_id: int):
        # get guild metadata (members, channels, etc.)
        guild = self.get_guild(guild_id)
        if guild is None:
            logger.warning(f"Guild {guild_id} not found")
            return
        guild_data = {
            "id": guild.id,
            "name": guild.name,
            "member_count": guild.member_count,
            "description": guild.description,
            "created_at": guild.created_at.isoformat(),
            "nsfw_level": guild.nsfw_level.name,
            "members": [
                {
                    "id": member.id,
                    "name": member.name,
                    "discriminator": member.discriminator,
                    "display_name": member.display_name,
                    "avatar": str(member.avatar.url) if member.avatar else None,
                    "bot": member.bot,
                    "system": member.system,
                    "slots": {
                        slot: getattr(member, slot, None)
                        for slot in member.__slots__
                        if not slot.startswith("_")
                    },
                }
                for member in guild.members
            ],
            "channels": [
                {
                    "id": channel.id,
                    "name": channel.name,
                    "category": (
                        {"id": channel.category_id, "name": channel.category.name} if channel.category else {}
                    ),
                    "position": channel.position,
                    "slots": {
                        slot: getattr(channel, slot, None)
                        for slot in channel.__slots__
                        if not slot.startswith("_")
                    },
                }
                for channel in guild.channels
            ],
            "slots": {
                slot: getattr(guild, slot, None) for slot in guild.__slots__ if not slot.startswith("_")
            },
        }

        # save member_data
        guild_data_path = self.datadir_path.joinpath("guilds", f"{guild_id}-meta.json")
        guild_data_path.parent.mkdir(exist_ok=True, parents=True)
        with guild_data_path.open("w", encoding="utf-8") as f:
            json.dump(guild_data, f, skipkeys=True, indent=2, default=str)

    def available_cogs(self) -> list[str]:
        cogs = [
            p.stem
            for p in self.cogdir_path.iterdir()
            if (p.is_dir() or p.suffix == ".py") and not p.name.startswith("_")
        ]

        if len(self.config.disable_cogs) > 0:
            cogs = [x for x in cogs if x not in self.config.disable_cogs]
        return cogs

    def load_cogs(self) -> None:
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

    async def on_ready(self) -> None:
        """
        The code in this even is executed when the bot is ready
        """
        logger.info(f"Logged in as {self.user.name}")
        logger.info(f"disnake API version: {DISNAKE_VERSION}")
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"Running on: {platform.system()} {platform.release()} ({os.name})")
        logger.info("-------------------")
        if not self.status_task.is_running():
            logger.info("Starting status update task")
            self.status_task.start()
        if not self.user_save_task.is_running():
            logger.info("Starting user save task")
            self.user_save_task.start()
        if not self.guild_save_task.is_running():
            logger.info("Starting guild save task")
            self.guild_save_task.start()

    async def on_message(self, message: Message) -> None:
        await self.process_commands(message)

    async def on_slash_command(self, ctx: ApplicationCommandInteraction) -> None:
        logger.info(f"Executing {ctx.data.name} command in {ctx} by {ctx.author} (ID: {ctx.author.id})")

    async def on_slash_command_error(self, interaction: ApplicationCommandInteraction, exception) -> None:
        if isinstance(exception, commands.CommandOnCooldown):
            logger.info(
                f"User {interaction.author} attempted to use {interaction.application_command.qualified_name} on cooldown."
            )
            embed = CooldownEmbed(int(exception.retry_after + 1), interaction.author)
            await interaction.send(embed=embed, ephemeral=True)

        elif isinstance(exception, exceptions.UserBlacklisted):
            logger.info(
                f"User {interaction.author} attempted to use {interaction.application_command.qualified_name}, but is blacklisted."
            )
            embed = Embed(title="Error!", description=exception.message, color=0xE02B2B)
            await interaction.send(embed=embed, ephemeral=True)

        elif isinstance(exception, commands.MissingPermissions):
            logger.warning(
                f"User {interaction.author} attempted to execute {interaction.application_command.qualified_name} without authorization."
            )
            embed = MissingPermissionsEmbed(interaction.author, exception)
            await interaction.send(embed=embed, ephemeral=True)

        elif isinstance(exception, exceptions.UserNotAdmin):
            logger.warning(
                f"User {interaction.author} attempted to execute {interaction.application_command.qualified_name}, but is not an admin."
            )
            embed = NotAdminEmbed(interaction.author, exception.message)
            await interaction.send(embed=embed, delete_after=30.0)

        else:
            # that covers all the usual errors, so let's catch the rest
            # first work out if we've deferred the response so we can send an ephemeral message if we need to
            ctx_rtype = getattr(interaction.response, "_response_type", None)
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
            await interaction.send(
                embed=embed,
                ephemeral=ctx_ephemeral,
                delete_after=30.0 if not ctx_ephemeral else None,  # type: ignore
            )
            logger.warning(f"Unhandled error in slash command {interaction}: {exception}")
            raise exception

    async def on_command_completion(self, ctx: commands.Context) -> None:
        """
        The code in this event is executed every time a normal command has been *successfully* executed
        :param ctx: The ctx of the command that has been executed.
        """
        full_command_name = ctx.command.qualified_name if ctx.command else ctx.invoked_with
        split = full_command_name.split(" ") if full_command_name else ["Unknown"]
        executed_command = str(split[0])
        logger.info(
            f"Executed {executed_command} command in {ctx} by {ctx.message.author} (ID: {ctx.message.author.id})"
        )

    async def on_command_error(self, context: commands.Context, exception) -> None:
        """
        The code in this event is executed every time a normal valid command catches an error
        :param interaction: The normal command that failed executing.
        :param exception: The exception that has been faced.
        """
        if context.command is None:
            logger.warning(f"Command not found for {context}: {exception}")
            return

        embed = None
        match exception:
            case commands.CommandOnCooldown():
                logger.info(f"User {context.author} attempted to use {context.command.name} on cooldown.")
                embed = CooldownEmbed(int(exception.retry_after + 1), context.author)
            case exceptions.UserBlacklisted():
                logger.info(
                    f"User {context.author} attempted to use {context.command.qualified_name}, but is blacklisted."
                )
                embed = Embed(
                    title="Error!",
                    description="You have been blacklisted and cannot use this bot. If you think this is a mistake, please contact the bot owner.",
                    color=0xE02B2B,
                )
            case exceptions.UserNotOwner():
                embed = Embed(
                    title="Error!",
                    description="This command requires admin permissions. soz bb xoxo <3",
                    color=0xE02B2B,
                )
                logger.warning(
                    f"User {context.author} attempted to execute {context.command.qualified_name} without authorization."
                )
            case commands.MissingPermissions():
                logger.warning(
                    f"User {context.author} attempted to execute {context.command.qualified_name} without authorization."
                )
                embed = MissingPermissionsEmbed(context.author, exception)
            case commands.MissingRequiredArgument():
                logger.info(
                    f"User {context.author} attempted to execute {context.command.qualified_name} without the required arguments"
                )
                embed = MissingRequiredArgumentEmbed(context.author, exception)
            case commands.CommandNotFound():
                logger.info(
                    f"User {context.author} attempted to execute a non-existent command: {context.message.content}"
                )
            case _:
                logger.warning(f"Unhandled exception in command {context}: {exception}")
                raise exception

        if embed is not None:
            await context.send(embed=embed)
        return

    @tasks.loop(minutes=1.5)
    async def status_task(self) -> None:
        """
        Set up the bot's status task
        """
        activity_type = getattr(ActivityType, self.config.status_type, ActivityType.playing)
        activity = Activity(name=random.choice(self.config.statuses), type=activity_type)
        await self.change_presence(activity=activity)

    @status_task.before_loop
    async def before_status_task(self) -> None:
        logger.info("waiting for ready... just to be sure")
        await self.wait_until_ready()

    @tasks.loop(minutes=3.0)
    async def user_save_task(self) -> None:
        async with Session() as session:
            async with session.begin():
                users: list[DiscordUser] = []

                user: Member
                for user in self.get_all_members():
                    if user.id in [x.id for x in users]:
                        continue
                    user_obj = DiscordUser.from_discord(user)
                    await session.merge(user_obj)
                await session.commit()

    @user_save_task.before_loop
    async def before_user_save_task(self) -> None:
        logger.info("waiting for ready... just to be sure")
        await self.wait_until_ready()
        logger.debug("ready!")

    @tasks.loop(minutes=3.0, count=1)
    async def guild_save_task(self) -> None:
        for guild in self.guilds:
            self.save_guild_metadata(guild.id)

    @guild_save_task.before_loop
    async def before_guild_save_task(self) -> None:
        logger.info("waiting for ready... just to be sure")
        await self.wait_until_ready()
        logger.debug("ready!")
