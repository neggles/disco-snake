import asyncio
import json
import logging
import sys
from zoneinfo import ZoneInfo

import click
import daemonocle
import uvloop
from daemonocle.cli import DaemonCLI
from rich.pretty import install as install_pretty
from rich.traceback import install as install_traceback

import logsnake
from disco_snake import CONFIG_PATH, DATADIR_PATH, LOG_FORMAT, LOGDIR_PATH, PACKAGE, USERDATA_PATH
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

    pending = asyncio.all_tasks()
    logger.info(f"Waiting for {len(pending)} remaining tasks to complete...")
    loop.run_until_complete(asyncio.gather(*pending))
    logger.info("All tasks completed, shutting down...")

    return code


class BotDaemon(daemonocle.Daemon):
    @daemonocle.expose_action
    def reload(self):
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
    # Load config
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_bytes())
    else:
        raise FileNotFoundError(f"Config file '{CONFIG_PATH}' not found!")

    config_log_level = parse_log_level(config["log_level"])
    logger.setLevel(config_log_level)
    logger.info(f"Effective log level: {logging.getLevelName(logger.getEffectiveLevel())}")

    logging.getLogger("disnake.gateway").setLevel(config_log_level)
    logging.getLogger("disnake.http").setLevel(config_log_level)

    # create log and data directories if they don't exist
    if not DATADIR_PATH.exists():
        DATADIR_PATH.mkdir(parents=True)
    if not LOGDIR_PATH.exists():
        LOGDIR_PATH.mkdir(parents=True)

    # load userdata
    if USERDATA_PATH.is_file():
        userdata = json.loads(USERDATA_PATH.read_bytes())
    else:
        logger.info(f"User state file does not exist, creating empty one at {USERDATA_PATH}")
        userdata = {}
        USERDATA_PATH.write_text(json.dumps(userdata, indent=4))

    logger.info(f"Loaded configuration from {CONFIG_PATH}")

    for key in config.keys():
        logger.debug(f"    {key}: {json.dumps(config[key], default=str)}")

    bot.config = config
    bot.timezone = ZoneInfo(config["timezone"])
    bot.userdata = userdata
    bot.reload = config["reload"]

    bot.load_cogs()
    bot.run(config["token"])

    cb_shutdown("Normal shutdown", 0)
