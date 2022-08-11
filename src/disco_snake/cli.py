import json
import logging
import sys
from traceback import print_exception
from zoneinfo import ZoneInfo
from pathlib import Path

import click
import logsnake
from daemonocle.cli import DaemonCLI

from helpers.misc import get_package_root, parse_log_level

from disco_snake.bot import bot

logfmt = logsnake.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__package__)

LOGDIR_PATH = Path.cwd().joinpath("logs")

DATADIR_PATH = Path.cwd().joinpath("data")
CONFIG_PATH = DATADIR_PATH.joinpath("config.json")
USERSTATE_PATH = DATADIR_PATH.joinpath("userstate.json")

PACKAGE_ROOT = get_package_root()

COGDIR_PATH = PACKAGE_ROOT.joinpath("cogs")
EXTDIR_PATH = PACKAGE_ROOT.joinpath("extensions")

MBYTE = 2**20


def cb_shutdown(message: str, code: int):
    logger.warning(f"Daemon is stopping: {code}")
    bot.save_userstate()
    logger.info(message)
    return code


@click.command(
    cls=DaemonCLI,
    daemon_params={
        "name": "disco-snake",
        "pid_file": "./data/disco-snake.pid",
        "shutdown_callback": cb_shutdown,
    },
)
@click.version_option(package_name="disco-snake")
@click.pass_context
def cli(ctx: click.Context):
    """
    Main entrypoint for your application.
    """
    global bot
    ctx.obj = bot

    # clamp log level to DEBUG
    logging.root = logsnake.setup_logger(
        level=logging.INFO,
        isRootLogger=True,
        formatter=logfmt,
        logfile=LOGDIR_PATH.joinpath("disco-snake.log"),
        fileLoglevel=logging.INFO,
        maxBytes=5 * MBYTE,
        backupCount=5,
    )

    logger.setLevel(logging.DEBUG)

    logger.info("Starting disco-snake")
    # Load config
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_bytes())
    else:
        raise FileNotFoundError(f"Config file '{CONFIG_PATH}' not found!")

    logger.setLevel(parse_log_level(config["log_level"]))
    logger.info(f"Effective log level: {logging.getLevelName(logger.getEffectiveLevel())}")

    # same for logs
    if not LOGDIR_PATH.exists():
        LOGDIR_PATH.mkdir(parents=True)

    # load userdata
    if USERSTATE_PATH.is_file():
        userstate = json.loads(USERSTATE_PATH.read_bytes())
    else:
        logger.info(f"User state file does not exist, creating empty one at {USERSTATE_PATH}")
        userstate = {}
        USERSTATE_PATH.write_text(json.dumps(userstate, indent=4))

    logger.info(f"Loaded configuration from {CONFIG_PATH}")
    logger.debug(f"    {json.dumps(config, indent=4)}")

    bot.config = config
    bot.timezone = ZoneInfo(config["timezone"])
    bot.datadir_path = DATADIR_PATH
    bot.userstate_path = USERSTATE_PATH
    bot.cogdir_path = COGDIR_PATH
    bot.extdir_path = EXTDIR_PATH
    bot.userstate = userstate
    bot.reload = config["reload"]

    bot.load_extensions()
    bot.load_cogs()

    bot.run(config["token"])

    cb_shutdown("Normal shutdown", 0)
