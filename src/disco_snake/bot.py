import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from traceback import print_exception
from zoneinfo import ZoneInfo

import disnake
from disnake import ApplicationCommandInteraction
from disnake.ext import commands, tasks

import exceptions
from helpers.misc import get_package_root

PACKAGE_ROOT = get_package_root()


intents = disnake.Intents.default()
intents.members = True
intents.presences = True
intents.message_content = True
intents.typing = False

logger = logging.getLogger(__package__)


class DiscoSnake(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # attributes set up in cli.py. this is a dumb way to do this but it works
        self.config: dict = None
        self.timezone: ZoneInfo = None
        self.datadir_path: Path = None
        self.userstate_path: Path = None
        self.userstate: dict = None
        self.cogdir_path: Path = None
        self.extdir_path: Path = None

    def save_userstate(self):
        if self.userstate is not None and self.userstate_path.is_file():
            with self.userstate_path.open("w") as f:
                json.dump(self.userstate, f, skipkeys=True, indent=2)
            logger.debug("Flushed user states to disk")

    def load_userstate(self):
        if self.userstate_path.is_file():
            with self.userstate_path.open("r") as f:
                self.userstate = json.load(f)
            logger.debug("Loaded user states from disk")

    def available_cogs(self):
        return [f.stem for f in self.cogdir_path.glob("*.py") if f.stem != "template"]

    def load_cogs(self, override: bool = False):
        cogs = self.available_cogs()
        if cogs:
            for cog in cogs:
                try:
                    self.load_extension(f"cogs.{cog}")
                    logger.info(f"Loaded cog '{cog}'")
                except Exception as e:
                    etype, exc, tb = sys.exc_info()
                    exception = f"{etype}: {exc}"
                    logger.error(f"Failed to load extension {cog}:\n{exception}")
                    print_exception(etype, exc, tb)
        else:
            logger.info("No cogs found")

    def available_extensions(self):
        return [f.stem for f in self.extdir_path.glob("*.py") if f.stem != "template"]

    def load_extensions(self):
        extensions = self.available_extensions()
        if extensions:
            for ext in extensions:
                try:
                    self.load_extension(f"extensions.{ext}")
                    logger.info(f"Loaded extension '{ext}'")
                except Exception as e:
                    etype, exc, tb = sys.exc_info()
                    exception = f"{etype}: {exc}"
                    logger.error(f"Failed to load extension {ext}:\n{exception}")
                    print_exception(etype, exc, tb)
        else:
            logger.info("No extensions found")


bot = DiscoSnake(command_prefix=commands.when_mentioned, intents=intents, help_command=None)


@bot.event
async def on_ready() -> None:
    """
    The code in this even is executed when the bot is ready
    """
    logger.info(f"Logged in as {bot.user.name}")
    logger.info(f"disnake API version: {disnake.__version__}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Running on: {platform.system()} {platform.release()} ({os.name})")
    logger.info("-------------------")
    if not status_task.is_running():
        status_task.start()
    if not userstate_task.is_running():
        userstate_task.start()


@tasks.loop(minutes=1.0)
async def status_task() -> None:
    """
    Set up the bot's status task
    """
    statuses = bot.config["statuses"]
    activity = disnake.Activity(name=random.choice(statuses), type=disnake.ActivityType.listening)
    await bot.change_presence(activity=activity)


@tasks.loop(minutes=3.0)
async def userstate_task() -> None:
    """
    Background task to flush user state to disk
    """
    bot.save_userstate()
    logger.debug("Flushed userstates to disk")


@bot.event
async def on_message(message: disnake.Message) -> None:
    """
    The code in this event is executed every time someone sends a message, with or without the prefix
    :param message: The message that was sent.
    """
    if message.author == bot.user or message.author.bot:
        return

    await bot.process_commands(message)


@bot.event
async def on_slash_command(interaction: ApplicationCommandInteraction) -> None:
    """
    The code in this event is executed every time a slash command has been *successfully* executed
    :param interaction: The slash command that has been executed.
    """
    logger.info(
        f"Executed {interaction.data.name} command in {interaction.guild.name} (ID: {interaction.guild.id}) by {interaction.author} (ID: {interaction.author.id})"
    )


@bot.event
async def on_slash_command_error(interaction: ApplicationCommandInteraction, error: Exception) -> None:
    """
    The code in this event is executed every time a valid slash command catches an error
    :param interaction: The slash command that failed executing.
    :param error: The error that has been faced.
    """
    if isinstance(error, exceptions.UserBlacklisted):
        """
        The code here will only execute if the error is an instance of 'UserBlacklisted', which can occur when using
        the @checks.is_owner() check in your command, or you can raise the error by yourself.

        'hidden=True' will make so that only the user who execute the command can see the message
        """
        embed = disnake.Embed(
            title="Error!", description="You are blacklisted from using the bot.", color=0xE02B2B
        )
        logger.info("A blacklisted user tried to execute a command.")
        return await interaction.send(embed=embed, ephemeral=True)
    elif isinstance(error, commands.errors.MissingPermissions):
        embed = disnake.Embed(
            title="Error!",
            description="You are missing the permission(s) `"
            + ", ".join(error.missing_permissions)
            + "` to execute this command!",
            color=0xE02B2B,
        )
        logger.info("A blacklisted user tried to execute a command.")
        return await interaction.send(embed=embed, ephemeral=True)
    raise error


@bot.event
async def on_command_completion(context: commands.Context) -> None:
    """
    The code in this event is executed every time a normal command has been *successfully* executed
    :param context: The context of the command that has been executed.
    """
    full_command_name = context.command.qualified_name
    split = full_command_name.split(" ")
    executed_command = str(split[0])
    logger.info(
        f"Executed {executed_command} command in {context.guild.name} (ID: {context.message.guild.id}) by {context.message.author} (ID: {context.message.author.id})"
    )


@bot.event
async def on_command_error(context: commands.Context, error) -> None:
    """
    The code in this event is executed every time a normal valid command catches an error
    :param context: The normal command that failed executing.
    :param error: The error that has been faced.
    """
    if isinstance(error, commands.CommandOnCooldown):
        minutes, seconds = divmod(error.retry_after, 60)
        hours, minutes = divmod(minutes, 60)
        hours = hours % 24
        embed = disnake.Embed(
            title="Hey, please slow down!",
            description=f"You can use this command again in {f'{round(hours)} hours' if round(hours) > 0 else ''} {f'{round(minutes)} minutes' if round(minutes) > 0 else ''} {f'{round(seconds)} seconds' if round(seconds) > 0 else ''}.",
            color=0xE02B2B,
        )
        await context.send(embed=embed)
    elif isinstance(error, commands.MissingPermissions):
        embed = disnake.Embed(
            title="Error!",
            description="You are missing the permission(s) `"
            + ", ".join(error.missing_permissions)
            + "` to execute this command!",
            color=0xE02B2B,
        )
        await context.send(embed=embed)
    elif isinstance(error, commands.MissingRequiredArgument):
        embed = disnake.Embed(
            title="Error!",
            # We need to capitalize because the command arguments have no capital letter in the code.
            description=str(error).capitalize(),
            color=0xE02B2B,
        )
        await context.send(embed=embed)
    elif isinstance(error, commands.CommandNotFound):
        # This is actually fine so lets just pretend everything is okay.
        return
    else:
        raise error
