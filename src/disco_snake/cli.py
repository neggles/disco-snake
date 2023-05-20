import asyncio
import json
import logging
import sys

import click
import daemonocle
import uvloop
from daemonocle.cli import DaemonCLI
from rich.pretty import install as install_pretty
from rich.traceback import install as install_traceback

import logsnake
from disco_snake import DATADIR_PATH, LOG_FORMAT, LOGDIR_PATH, PACKAGE
from disco_snake.bot import DiscoSnake
from helpers.misc import parse_log_level

MBYTE = 2**20

logfmt = logsnake.LogFormatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
# setup root logger
logging.root = logsnake.setup_logger(
    level=logging.DEBUG,
    isRootLogger=True,
    formatter=logfmt,
    logfile=LOGDIR_PATH.joinpath(f"{PACKAGE}-debug.log"),
    fileLoglevel=logging.INFO,
    maxBytes=1 * MBYTE,
    backupCount=3,
)
# setup package logger
logger = logsnake.setup_logger(
    level=logging.DEBUG,
    isRootLogger=False,
    name=__package__,
    formatter=logfmt,
    logfile=LOGDIR_PATH.joinpath(f"{PACKAGE}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=1 * MBYTE,
    backupCount=3,
)

# install rich traceback handler
install_pretty()
install_traceback(show_locals=True)

bot: DiscoSnake = None  # type: ignore


def cb_shutdown(message: str, code: int):
    logger.warning(f"Daemon is stopping: {code}")
    if bot is not None:
        bot.save_userdata()
        loop = bot.loop
    else:
        loop = asyncio.get_event_loop()

    logger.info(message)
    loop.stop()
    logger.info("All tasks completed, shutting down...")

    return code


class BotDaemon(daemonocle.Daemon):
    @daemonocle.expose_action
    def reload(self) -> None:
        """Reload the bot."""
        pass


@click.command(
    cls=DaemonCLI,
    daemon_class=BotDaemon,
    daemon_params={
        "name": "disco-snake",
        "pid_file": f"{DATADIR_PATH}/disco-snake.pid",
        "shutdown_callback": cb_shutdown,
        "stop_timeout": 60,
    },
)
@click.version_option(package_name="disco-snake")
@click.pass_context
def cli(ctx: click.Context):
    """
    Main entrypoint for your application.
    """
    return start_bot(ctx)


def start_bot(ctx: click.Context = None):
    global bot
    bot = DiscoSnake()
    if ctx is not None:
        ctx.obj: DiscoSnake = bot

    # have to use a different method on python 3.11 and up because of a change to how asyncio works
    # not sure how to implement that with disnake, so for now, no uvloop on python 3.11 and up
    if sys.version_info < (3, 11):
        logger.info("installing uvloop...")
        uvloop.install()

    logger.info("Starting disco-snake")

    config_log_level = parse_log_level(bot.config.log_level)
    logger.setLevel(config_log_level)
    logger.info(f"Effective log level: {logging.getLevelName(logger.getEffectiveLevel())}")

    logging.getLogger("disnake.gateway").setLevel(logging.INFO)
    logging.getLogger("disnake.http").setLevel(config_log_level)

    # create log and data directories if they don't exist
    if not DATADIR_PATH.exists():
        DATADIR_PATH.mkdir(parents=True)
    if not LOGDIR_PATH.exists():
        LOGDIR_PATH.mkdir(parents=True)

    cfg_dict = bot.config.dict()
    for key in cfg_dict:
        logger.debug(f"    {key}: {json.dumps(cfg_dict[key], default=str)}")

    bot.load_cogs()
    return bot.run(token=bot.config.bot_token, reconnect=True)
