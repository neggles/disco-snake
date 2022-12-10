import json
import logging
import sys
from zoneinfo import ZoneInfo

import click
import daemonocle
import uvloop
from daemonocle.cli import DaemonCLI

import logsnake
from disco_snake import LOGDIR_PATH, CONFIG_PATH, DATADIR_PATH, USERDATA_PATH, PACKAGE
from disco_snake.bot import DiscoSnake
from helpers.misc import parse_log_level

MBYTE = 2**20

logfmt = logsnake.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S")
# setup root logger
logging.root = logsnake.setup_logger(
    level=logging.INFO,
    isRootLogger=True,
    formatter=logfmt,
    logfile=LOGDIR_PATH.joinpath(f"{PACKAGE}-debug.log"),
    fileLoglevel=logging.INFO,
    maxBytes=2 * MBYTE,
    backupCount=2,
)
# setup package logger
logger = logsnake.setup_logger(
    level=logging.DEBUG,
    isRootLogger=False,
    name=__package__,
    formatter=logfmt,
    logfile=LOGDIR_PATH.joinpath(f"{PACKAGE}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=2 * MBYTE,
    backupCount=2,
)

bot: DiscoSnake = DiscoSnake()


def cb_shutdown(message: str, code: int):
    logger.warning(f"Daemon is stopping: {code}")
    bot.save_userstate()
    logger.info(message)
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
    },
)
@click.version_option(package_name="disco-snake")
@click.pass_context
def cli(ctx: click.Context):
    """
    Main entrypoint for your application.
    """
    ctx.obj: DiscoSnake = bot

    # have to use a different method on python 3.11 and up because of a change to how asyncio works
    # not sure how to implement that with disnake, so for now, no uvloop on python 3.11 and up
    if sys.version_info < (3, 11):
        uvloop.install()

    logger.info("Starting disco-snake")
    # Load config
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_bytes())
    else:
        raise FileNotFoundError(f"Config file '{CONFIG_PATH}' not found!")

    logger.setLevel(parse_log_level(config["log_level"]))
    logger.info(f"Effective log level: {logging.getLevelName(logger.getEffectiveLevel())}")

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
    logger.debug(f"    {json.dumps(config, indent=4)}")

    bot.config = config
    bot.timezone = ZoneInfo(config["timezone"])
    bot.userdata = userdata
    bot.reload = config["reload"]

    bot.load_extensions()
    bot.load_cogs()

    bot.run(config["token"])

    cb_shutdown("Normal shutdown", 0)
