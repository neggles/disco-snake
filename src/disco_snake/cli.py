import json
import logging
import sys
import time
from pathlib import Path

import click
import logzero
from daemonocle.cli import DaemonCLI

from .bot import bot
from helpers.misc import get_package_root

logfmt = logzero.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__package__)

PACKAGE_ROOT = get_package_root()


DATADIR_PATH = (
    Path.cwd().joinpath("data")
    if Path.cwd().joinpath("data").exists() and Path.cwd().joinpath("data").is_dir()
    else Path.cwd().parent.joinpath("data")
)
USERSTATE_PATH = DATADIR_PATH.joinpath("userstate.json")
CONFIG_PATH = None


def load_commands() -> None:
    for file in PACKAGE_ROOT.joinpath("cogs").iterdir():
        if file.suffix == ".py" and file.stem != "template":
            extension = file.stem
            try:
                bot.load_extension(f"cogs.{extension}")
                logger.info(f"Loaded extension '{extension}'")
            except Exception as e:
                exception = f"{type(e).__name__}: {e}"
                logger.error(f"Failed to load extension {extension}\n{exception}")


def cb_shutdown(message: str, code: int):
    logger.warning(f"Daemon is stopping: {code}")
    if bot.userstate is not None and USERSTATE_PATH.is_file():
        with USERSTATE_PATH.open("w") as f:
            json.dump(bot.userstate, f, skipkeys=True, indent=4)
        logger.info("Flushed user states to disk")
    logger.info(message)
    sys.exit(code)


@click.command(
    cls=DaemonCLI,
    daemon_params={
        "name": "disco-snake",
        "pid_file": "./disco-snake.pid",
        "shutdown_callback": cb_shutdown,
    },
)
@click.version_option(package_name="disco-snake")
@click.pass_context
def cli(ctx: click.Context):
    """
    Main entrypoint for your application.
    """
    verbose = 2
    config_path = DATADIR_PATH.joinpath("config.json")
    global bot
    # clamp log level to DEBUG
    loglevel = max(logging.WARNING - (verbose * 10), 10)
    logging.root = logzero.setup_logger(level=loglevel, isRootLogger=True, formatter=logfmt)

    logger.info("Starting disco-snake")
    logger.debug("Commandline options:")
    logger.debug(f"    verbose = {verbose}")
    logger.debug(f"    config_path = {config_path}")
    logger.debug(f"Effective log level: {logging.getLevelName(logger.getEffectiveLevel())}")

    # Load config
    if config_path.exists():
        global CONFIG_PATH
        CONFIG_PATH = config_path
        config = json.loads(config_path.read_bytes())
    else:
        raise FileNotFoundError(f"Config file '{config_path}' not found!")

    # create data directory if it doesn't exist
    if not DATADIR_PATH.exists():
        DATADIR_PATH.mkdir(parents=True)

    # load userdata
    if USERSTATE_PATH.exists() and USERSTATE_PATH.is_file():
        userstate = json.loads(USERSTATE_PATH.read_bytes())
    else:
        logger.info(f"User state file does not exist, creating empty one at {USERSTATE_PATH}")
        userstate = {}
        USERSTATE_PATH.write_text(json.dumps(userstate, indent=4))

    logger.info(f"Loaded configuration from {config_path}")
    logger.debug(f"    {json.dumps(config, indent=4)}")

    load_commands()

    bot.config = config
    bot.datadir_path = DATADIR_PATH
    bot.userstate_path = USERSTATE_PATH
    bot.userstate = userstate
    bot.run(config["token"])

    cb_shutdown("Normal shutdown", 0)
    sys.exit(0)
